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
    for (int t=0 ; t<=seq_len; t++)
    {
        float dot = 0.0f;
        for(int s=0 ;s <= head_dim; s++)
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
        out+= S[t]*V[t*head_dim+j];
    }
    out[j]= out;
}

__global__ void tiled_attention(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int head_dim
)
{   const int Br= 16; // row, query tile size
    const int Bc=16; // column key value tile size
    // __shared__ float Qs[Br][head_dim];
    // __shared__ float Ks[Bc][head_dim];
    // __shared__ float Vs[Bc][head_dim];
    // cuda doesn't allow dynamic values in __shared__ array dimensions.
    __shared__ float Qs[Br][64];
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];
    
    int tile_row = threadIdx.x;
    int tile_col = threadIdx.y;

    int global_row = blockIdx.x*Br + tile_row;
    Qs[tile_row][tile_col]= Q[global_row*head_dim+ tile_col]

    for(int tile =0; tile <N/Bc;tile++)
    {
        //cooperatively load K and V tiles
        int global_row = tile*Bc+ threadIdx.x;
        Ks[threadIdx.x][threadIdx.y]= K[threadIdx.x*head_dim+ threadIdx.y];
        Vs[threadIdx.x][threadIdx.y] = V[threadIdx.x*head_dim+ threadIdx.y];
        __syncthreads()
        for(int t=0;t<Bc;t++)
        {
            float score = 0.0f;
            for(int k=0;k<head_dim;k++)
            {
                score+= Qs[threadIdx.x][k]*Ks[t][k];
            }
            score/= sqrt((float)head_dim);
            S[tile*Bc+ t] = score;
        }
        __syncthreads();
    }

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
    for (int t=0 ; t<=seq_len; t++)
    {
        float dot = 0.0f;
        for(int s=0 ;s <= head_dim; s++)
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
        out+= S[t]*V[t*head_dim+j];
    }
    out[j]= out;
}