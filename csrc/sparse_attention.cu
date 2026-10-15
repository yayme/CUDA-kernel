#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>

#define D 64            // head dim
#define Br 16           // query tile size
#define Bc 16           // key/value tile size
#define MAX_SEQ_LEN 1024 // naive and v1 keep a full score row per thread, so they cap N

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Step 0 (naive): one thread per output element O[i][j], dense attention, no mask.
// Every thread rebuilds the full score row from global memory.
__global__ void naive_attention(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int head_dim
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= seq_len || j >= head_dim) return;

    float S[MAX_SEQ_LEN];
    for (int t = 0; t < seq_len; t++)
    {
        float dot = 0.0f;
        for (int s = 0; s < head_dim; s++)
            dot += Q[i * head_dim + s] * K[t * head_dim + s];
        S[t] = dot / sqrtf((float)head_dim);
    }

    float max_val = -1e9f;
    for (int t = 0; t < seq_len; t++)
        if (S[t] > max_val) max_val = S[t];

    float sum_exp = 0.0f;
    for (int t = 0; t < seq_len; t++)
    {
        S[t] = expf(S[t] - max_val);
        sum_exp += S[t];
    }

    float output = 0.0f;
    for (int t = 0; t < seq_len; t++)
        output += (S[t] / sum_exp) * V[t * head_dim + j];
    O[i * head_dim + j] = output;
}

// Step 1 (v1): Q/K/V tiles in shared memory, skip K/V tiles with no active pair.
// Still keeps the full score row S[N] per thread, which spills to local memory.
__global__ void flash_attention(
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
    if (global_row < seq_len && threadIdx.y < D)
        Qs[threadIdx.x][threadIdx.y] = Q[global_row * D + threadIdx.y];
    __syncthreads();

    // masked and skipped keys keep a very negative score so they vanish after exp
    float S[MAX_SEQ_LEN];
    for (int t = 0; t < seq_len; t++) S[t] = -1e9f;

    int num_tiles = ceil_div(seq_len, Bc);
    for (int tile = 0; tile < num_tiles; tile++)
    {
        int tile_col_start = tile * Bc;

        // skip the tile if no (query, key) pair in it is active; same answer for every thread
        bool any_active = false;
        for (int i = tile_row_start; i < min(tile_row_start + Br, seq_len) && !any_active; i++)
            for (int jj = tile_col_start; jj < min(tile_col_start + Bc, seq_len) && !any_active; jj++)
                if (mask[i * seq_len + jj]) any_active = true;
        if (!any_active) continue;

        int global_kv_row = tile_col_start + threadIdx.x;
        if (global_kv_row < seq_len && threadIdx.y < D) {
            Ks[threadIdx.x][threadIdx.y] = K[global_kv_row * D + threadIdx.y];
            Vs[threadIdx.x][threadIdx.y] = V[global_kv_row * D + threadIdx.y];
        }
        __syncthreads();

        for (int t = 0; t < Bc; t++)
        {
            int key_idx = tile_col_start + t;
            if (global_row >= seq_len || key_idx >= seq_len) continue;
            if (!mask[global_row * seq_len + key_idx]) continue;
            float score = 0.0f;
            for (int k = 0; k < D; k++)
                score += Qs[threadIdx.x][k] * Ks[t][k];
            S[key_idx] = score / sqrtf((float)D);
        }
        __syncthreads();
    }

    // full-row softmax, then dot with V straight from global memory
    if (global_row < seq_len) {
        float max_val = -1e9f;
        for (int t = 0; t < seq_len; t++)
            if (max_val < S[t]) max_val = S[t];

        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            S[t] = expf(S[t] - max_val);
            sum_exp += S[t];
        }

        int j = threadIdx.y;
        float output = 0.0f;
        for (int t = 0; t < seq_len; t++)
            output += (S[t] / sum_exp) * V[t * D + j];
        O[global_row * D + j] = output;
    }
}

// Step 2 (v2): same tiling and skipping, but online softmax.
// Running max m, running denominator l and running output acc replace S[N].
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
    if (global_row < seq_len && threadIdx.y < D)
        Qs[threadIdx.x][threadIdx.y] = Q[global_row * D + threadIdx.y];
    __syncthreads();

    float m = -1e9f;
    float l = 0.0f;
    float acc = 0.0f;
    int j = threadIdx.y;

    int num_tiles = ceil_div(seq_len, Bc);
    for (int tile = 0; tile < num_tiles; tile++)
    {
        int tile_col_start = tile * Bc;

        bool any_active = false;
        for (int i = tile_row_start; i < min(tile_row_start + Br, seq_len) && !any_active; i++)
            for (int jj = tile_col_start; jj < min(tile_col_start + Bc, seq_len) && !any_active; jj++)
                if (mask[i * seq_len + jj]) any_active = true;
        if (!any_active) continue;

        int global_kv_row = tile_col_start + threadIdx.x;
        if (global_kv_row < seq_len && threadIdx.y < D) {
            Ks[threadIdx.x][threadIdx.y] = K[global_kv_row * D + threadIdx.y];
            Vs[threadIdx.x][threadIdx.y] = V[global_kv_row * D + threadIdx.y];
        }
        __syncthreads();

        if (global_row < seq_len) {
            // scores for this tile only (Bc floats, not N)
            float scores[Bc];
            for (int t = 0; t < Bc; t++) {
                int key_idx = tile_col_start + t;
                if (key_idx >= seq_len || !mask[global_row * seq_len + key_idx]) {
                    scores[t] = -1e9f;
                    continue;
                }
                float score = 0.0f;
                for (int k = 0; k < D; k++)
                    score += Qs[threadIdx.x][k] * Ks[t][k];
                scores[t] = score / sqrtf((float)D);
            }

            float m_tile = -1e9f;
            for (int t = 0; t < Bc; t++)
                m_tile = max(m_tile, scores[t]);

            // rescale what was accumulated under the old max, then add this tile
            float m_new = max(m, m_tile);
            float scale = expf(m - m_new);
            acc *= scale;

            float l_tile = 0.0f;
            for (int t = 0; t < Bc; t++) {
                float a = expf(scores[t] - m_new);
                l_tile += a;
                acc += a * Vs[t][j];
            }

            l = l * scale + l_tile;
            m = m_new;
        }
        __syncthreads();
    }

    if (global_row < seq_len && j < D)
        O[global_row * D + j] = acc / l;
}

// ---------------------------------------------------------------------------
// Launchers: validate inputs, launch on PyTorch's current stream, check the launch.
// ---------------------------------------------------------------------------

#define CHECK_INPUT(t, dtype) \
    TORCH_CHECK((t).is_cuda() && (t).is_contiguous() && (t).scalar_type() == (dtype), \
                #t " must be a contiguous CUDA tensor of type " #dtype)

static void check_qkv(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V)
{
    CHECK_INPUT(Q, torch::kFloat32);
    CHECK_INPUT(K, torch::kFloat32);
    CHECK_INPUT(V, torch::kFloat32);
    TORCH_CHECK(Q.dim() == 2 && Q.size(1) == D, "Q must have shape [N, 64]");
    TORCH_CHECK(K.sizes() == Q.sizes() && V.sizes() == Q.sizes(), "Q, K, V must have the same shape");
}

static void check_mask(const torch::Tensor& mask, int N)
{
    CHECK_INPUT(mask, torch::kInt32);
    TORCH_CHECK(mask.dim() == 2 && mask.size(0) == N && mask.size(1) == N, "mask must have shape [N, N]");
}

torch::Tensor naive_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    check_qkv(Q, K, V);
    int N = Q.size(0);
    TORCH_CHECK(N <= MAX_SEQ_LEN, "naive kernel supports N <= 1024");

    auto O = torch::zeros_like(Q);
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 16), ceil_div(D, 16));
    naive_attention<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), N, D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}

torch::Tensor sparse_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask)
{
    check_qkv(Q, K, V);
    int N = Q.size(0);
    check_mask(mask, N);
    TORCH_CHECK(N <= MAX_SEQ_LEN, "v1 kernel supports N <= 1024");

    auto O = torch::zeros_like(Q);
    dim3 block(Br, D);
    dim3 grid(ceil_div(N, Br));
    flash_attention<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), N,
        mask.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}

torch::Tensor sparse_attention_v2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor mask)
{
    check_qkv(Q, K, V);
    int N = Q.size(0);
    check_mask(mask, N);

    auto O = torch::zeros_like(Q);
    dim3 block(Br, D);
    dim3 grid(ceil_div(N, Br));
    flash_attention_v2<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), N,
        mask.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return O;
}
