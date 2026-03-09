#include<cuda_runtime.h>
#include<torch/extension.h>
#include<math.h>
#include<stdio.h>
#define D 64 // head dim
#define Br 16 // query tile size
#define Bc 16 // key/value tile size



// local window global stride.
void build_sparse_mask(
int* mask, // seq_len*seq_len matrix
int seq_len,
int window,
int stride
)
{
for (int i=0; i<seq_len; i++)
{
    for( int j=0; j< seq_len; j++)
    {
        bool local = abs(i-j) <= window;
        bool global = (j%stride == 0);
        mask[i*seq_len+j] = local || global ? 1 : 0 ;
    }
}
// say N= 64 tokens. split them into blocks of Br =16. 
// e.g. block 0 handles queries [0..  15]
// blockIdx.x gives which block
// threadIdx.x which row in the tile [0... 15]
// threadIdx.y which column in the tile [0... 63]
}

__global__ void flash_attention_v2(    
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int* mask
)
{
    __shared__ float Qs[Br][D];
    __shared__ float Ks[Bc][D];
    __shared__ float Vs[Bc][D];

    int tile_row_start = blockIdx.x * Br;
    int global_row = tile_row_start + threadIdx.x;

    // load Q tile cooperatively
    if (global_row < seq_len && threadIdx.y < D)
        Qs[threadIdx.x][threadIdx.y] = Q[global_row * D + threadIdx.y];
    __syncthreads();

    // ── online softmax accumulators (replaces S[1024]) ──
    float m = -1e9f;      // running max
    float l = 0.0f;       // running sum of exp
    float acc =0.0f;    
    int j= threadIdx.y;     // running output
    // for (int k = 0; k < D; k++) acc[k] = 0.0f;

    for (int tile = 0; tile < seq_len / Bc; tile++)
    {
        int tile_col_start = tile * Bc;

        // sparsity check
        bool any_active = false;
        for (int i = tile_row_start; i < tile_row_start + Br && !any_active; i++)
            for (int jj = tile_col_start; jj < tile_col_start + Bc && !any_active; jj++)
                if (i < seq_len && jj < seq_len && mask[i * seq_len + jj])
                    any_active = true;
        if (!any_active) continue;

        // load K/V tile
        int global_kv_row = tile_col_start + threadIdx.x;
        if (global_kv_row < seq_len && threadIdx.y < D) {
            Ks[threadIdx.x][threadIdx.y] = K[global_kv_row * D + threadIdx.y];
            Vs[threadIdx.x][threadIdx.y] = V[global_kv_row * D + threadIdx.y];
        }
        __syncthreads();

        if (global_row < seq_len) {
            // compute scores for this tile only (Bc floats, not N)
            float scores[Bc];
            for (int t = 0; t < Bc; t++) {
                int key_idx = tile_col_start + t;
                if (key_idx >= seq_len || !mask[global_row * seq_len + key_idx]) {
                    scores[t] = -1e9f;  // mask out — will vanish after exp
                    continue;
                }
                float score = 0.0f;
                for (int k = 0; k < D; k++)
                    score += Qs[threadIdx.x][k] * Ks[t][k];
                scores[t] = score / sqrtf((float)D);
            }

            // find tile max
            float m_tile = -1e9f;
            for (int t = 0; t < Bc; t++)
                m_tile = max(m_tile, scores[t]);

            // update running max
            float m_new = max(m, m_tile);

            // rescale old accumulator
            float scale = expf(m - m_new);
            // for (int k = 0; k < D; k++)
            //     acc[k] *= scale;
            acc*=scale;

            // accumulate new tile into output
            float l_tile = 0.0f;
            for (int t = 0; t < Bc; t++) {
                float a = expf(scores[t] - m_new);
                l_tile += a;
                // for (int k = 0; k < D; k++)
                acc += a * Vs[t][j];
            }

            // update running sum
            l = l * scale + l_tile;
            m = m_new;
        }

        __syncthreads();
    }

    // final output — divide by normalizer
    if (global_row < seq_len) {
        // int j = threadIdx.y;
        if (j < D)
            O[global_row * D + j] = acc / l;
    }
}
__global__ void flash_attention(    
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int * mask
)
{
    __shared__ float Qs[Br][D];
    __shared__ float Ks[Bc][D];
    __shared__ float Vs[Bc][D];
    int tile_row_start = blockIdx.x*Br; // 
    int global_row = tile_row_start + threadIdx.x;
    if (global_row <seq_len && threadIdx.y <D)
        Qs[threadIdx.x][threadIdx.y] = Q[global_row*D+ threadIdx.y];
    __syncthreads();

    float  S[1024] = {0.0f};
    for (int tile =0; tile < seq_len/Bc; tile++)
    {
        int tile_col_start = tile*Bc;

    
            bool any_active = false;
        for (int i= tile_row_start; i < tile_row_start+Br ; i++)
        {
            for(int j= tile_col_start; j< tile_col_start+Bc;j++)
            {
                if(mask[i*seq_len+j]){
                    any_active = true;
                    break;
                }
            }
        }
        if(!any_active) continue; 
    int global_kv_row = tile_col_start + threadIdx.x;
    if (global_kv_row <seq_len && threadIdx.y <D) {
        Ks[threadIdx.x][threadIdx.y] = K[global_kv_row*D + threadIdx.y];
        Vs[threadIdx.x][threadIdx.y] =  V[global_kv_row*D + threadIdx.y];
    }
    __syncthreads();
    // compute partial scores for this tile.
    for (int t =0; t< Bc; t++)
    {
        int key_idx = tile_col_start + t;
        if(global_row >= seq_len || key_idx  >= seq_len) continue;
        if(!mask[global_row*seq_len+ key_idx]) continue;
        float score = 0.0f;
        for(int k=0; k< D; k++)
        {
            score+= Qs[threadIdx.x][k]*Ks[t][k];

        }
        S[key_idx] = score / sqrtf((float)D);

    }
    __syncthreads();
    
    
    }

    if(global_row <seq_len) {
        float max_val = -1e9f;
        for(int t=0;t<seq_len; t++)
        {
            if(max_val < S[t]) max_val= S[t];
        }
        float sum_exp =0.0f;
        for(int t =0;t<seq_len; t++)
        {
            S[t]= expf(S[t]- max_val);
            sum_exp+= S[t];
            
        }
        for(int t=0; t<seq_len; t++)
        {
            S[t]/= sum_exp;


        }
        int j= threadIdx.y;
        if (j<D)
        {
            float output = 0.0f;
            for(int t=0;t<seq_len;t++)
        {
            output+=S[t]*V[t*D+j];

        }
        O[global_row*D+j] = output;
        }
    }

}

__global__ void naive_attention(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int head_dim
)
{
    // each thread should handle one output element O[i, j]
    int i = blockIdx.x*blockDim.x + threadIdx.x; // each row handles the full sequence for a single head. so we have to move along the row
    int j = blockIdx.y*blockDim.y + threadIdx.y; // each columns handles the full head dimension for a single token. so we have to move along the column
    if( i>= seq_len || j>=head_dim) return;

    // S[i,:]= Q[i, :] @ K.T;
    // S= S/sqrt(head_dim);
    // A= softmax(S, dim=-1);
    // O = A @ V;
    float  S[1024]; // assuming seq_len <= 1024
    // we need S[i,t]= dot (Q[i,:], K[t,:])
    for (int t=0 ; t<seq_len; t++)
    {
        float dot = 0.0f;
        for(int s=0 ;s < head_dim; s++)
        {
            
            dot+= Q[i*head_dim+s]*K[t*head_dim+s];
        }
        S[t]= dot/sqrt((float)head_dim);
    }
    // for numerical stability, we should calculate max of S[i,:]
    float max_val = -1e9f;
    for (int t=0;t <seq_len;t++)
    {
        if(S[t]>max_val)max_val = S[t];
    }
    float sum_exp = 0.0f;
    for(int t=0;t<seq_len;t++)
    { S[t]= expf(S[t]-max_val); // to prevent overflowing
        sum_exp+= S[t];

    }
    for(int t=0;t<seq_len;t++)
    {
        S[t]=S[t]/sum_exp;
    }
    float output =0.0f;
    for (int t=0;t<seq_len;t++)
    {
        output+= S[t]*V[t*head_dim+j];
    }
    O[i*head_dim+ j]= output;
}
torch::Tensor sparse_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor mask
)
{
    int N = Q.size(0);
    auto O = torch::zeros_like(Q);
    dim3 block(Br, D);
    dim3 grid(N/Br);
    flash_attention<<< grid, block>>> (Q.data_ptr<float>(),
    K.data_ptr<float>(),
    V.data_ptr<float>(),
    O.data_ptr<float>(),
    N,
    mask.data_ptr<int>()
);
return O;
}
torch::Tensor sparse_attention_v2_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor mask
) {
    int N = Q.size(0);
    auto O = torch::zeros_like(Q);
    dim3 block(Br, D);
    dim3 grid(N / Br);
    flash_attention_v2<<<grid, block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N,
        mask.data_ptr<int>()
    );
    return O;
}
// int main()
// {
//     const int N =64;
//     const int window =2;
//     const int stride =4;
//     // allocate host memory
//     float *hQ, *hK, *hV, *hO;
//     int *hMask;
//     hQ = new float[N*D]; // malloc also works
//     hK = new float[N*D];
//     hV = new float[N*D];
//     hO = new float[N*D];
//     hMask = new int[N*N];
//     for (int i=0;i<N*D;i++)
//     {
//         hQ[i] = (float)rand()/ RAND_MAX;
//         hK[i] = (float) rand()/ RAND_MAX;
//         hV[i] = (float) rand()/ RAND_MAX;

//     }
//     build_sparse_mask(hMask, N, window, stride);
//     // allocate device memory
//     float *dQ, *dK, *dV, *dO;
//     int *dMask;

//     cudaMalloc (&dQ, N*D*sizeof(float));
//     cudaMalloc (&dK, N*D*sizeof(float));
//     cudaMalloc (&dV, N*D*sizeof(float));
//     cudaMalloc (&dO, N*D*sizeof(float));
//     cudaMalloc (&dMask, N*N*sizeof(int));

//     cudaMemcpy (dQ, hQ, N*D*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy (dK, hK, N*D*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy (dV, hV, N*D*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy (dMask, hMask, N*N*sizeof(int), cudaMemcpyHostToDevice);

//     dim3 block(Br , D);
//     dim3 grid(N/Br);
//     sparse_attention <<< grid, block>>> (dQ, dK, dV, dO, dMask, N);

//     cudaMemcpy (hO, dO, N*D*sizeof(float), cudaMemcpyDeviceToHost);

//     printf("O[0][0] = %f\n", hO[0]);
//     delete[] hQ ;
//     delete[] hK ;
//     delete[] hV ;
//     delete[] hO ;
//     delete[] hMask;
//     cudaFree(dQ);
//     cudaFree(dK);
//     cudaFree(dV);
//     cudaFree(dO);
//     cudaFree(dMask);
//     return 0;



// }