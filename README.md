# Sparse attention

This is an exploratory project to implement a CUDA kernel for sparse attention and benchmark its performance against PyTorch's dense attention. Built this to understand what actually happens inside Flash Attention , not just use it as a black box.

The core idea: instead of computing attention between every pair of tokens (O(n²)), define a sparsity pattern and skip tiles that have no active token pairs entirely. The pattern used here is a local window of `w` nearby tokens plus global tokens every `s` steps , same idea as Longformer.

---

## Files

| File | What it is |
|------|------------|
| `sparse_attention.cu` | The CUDA kernels — naive baseline, v1 (tiled shared memory), v2 (online softmax) |
| `sparse_attention_ext.cpp` | pybind11 wrapper that exposes the kernels as a PyTorch extension |
| `setup.py` | Compiles everything via `CUDAExtension` |
| `benchmark.ipynb` | Colab notebook — builds the extension, runs correctness checks, plots benchmarks |
| `log.md` | Dev log of every iteration, what broke, and why |

---

## Build

```bash
pip install torch
python setup.py build_ext --inplace
```

Tested on T4 GPU (CUDA 12.8, Python 3.12).

---

## Usage

```python
import torch
import sparse_attention as sa

N, D = 512, 64
Q = torch.randn(N, D, device='cuda')
K = torch.randn(N, D, device='cuda')
V = torch.randn(N, D, device='cuda')

# build sparse mask: local window=2, global stride=4
mask = torch.zeros(N, N, dtype=torch.int32)
for i in range(N):
    for j in range(N):
        if abs(i - j) <= 2 or j % 4 == 0:
            mask[i, j] = 1
mask = mask.cuda()

O = sa.sparse_attention_v2(Q, K, V, mask)  # v2 — online softmax, faster
# O = sa.sparse_attention(Q, K, V, mask)   # v1 — tiled, S[N] spill version
```

---

## How it works

### Sparsity pattern
Each query token `i` attends only to:
- **Local window**: tokens within distance `w` → `abs(i-j) <= w`
- **Global stride**: every `s`-th token → `j % s == 0`

Everything else is masked out. The kernel checks each K/V tile before loading it — if no token pair in that tile is active, the tile is skipped entirely.

### v1 — tiled shared memory
Loads Q/K/V tiles into shared memory (SRAM) cooperatively so threads don't each fetch their own copy from global memory. Stores the full score array `S[N]` per thread, softmaxes at the end. Works, but `S[N]` spills to global memory as N grows — O(N) register usage per thread, which is why it gets slow fast.

### v2 — online softmax
Replaces `S[N]` with four scalars per thread: running max `m`, running denominator `l`, running output `acc`, and the thread's output column `j`. Updates these incrementally as each tile arrives — never needs to store the full score array. O(1) memory per thread regardless of sequence length. This is the core idea from the Flash Attention paper.

---

## Results

### v1 vs Dense (PyTorch)
![v1 vs Dense](assets/benchmark_v1_vs_dense.png)

### v1 vs v2 vs Dense
![v1 vs v2 vs Dense](assets/benchmark_v1_v2_vs_dense.png)

| N | Dense (PyTorch) | v1 | v2 | v2 speedup over v1 |
|---|---|---|---|---|
| 64 | 0.058 ms | 1.026 ms | 0.717 ms | 1.43× |
| 128 | 0.057 ms | 0.794 ms | 0.547 ms | 1.45× |
| 256 | 0.113 ms | 1.638 ms | 0.894 ms | 1.83× |
| 512 | 0.088 ms | 5.145 ms | 3.371 ms | 1.53× |

Both custom kernels are slower than PyTorch dense at these sequence lengths. That's expected — PyTorch uses cuBLAS which is optimized at the assembly level. The sparsity advantage should show up at N ≥ 2048 where skipping O(n²) tiles actually saves meaningful work vs the kernel launch overhead.

The more interesting result is v2 vs v1 — the online softmax rewrite gives a consistent 1.4–1.8× speedup just by eliminating the `S[N]` array. Same algorithm, same sparsity pattern, just better memory behavior.

---

## Key things learned

**Flat indexing** Python slice is easy to use. It took me a while to get used to flat indexing and be able to mentally calculate them.

**`__syncthreads()` has two jobs, not one.** One call protects reads (don't read shared memory before everyone's written to it), another protects writes (don't overwrite the tile before everyone's done reading it). 

**Shared memory size must be known at compile time.** You can't do `__shared__ float Qs[Br][d]` if `d` is a runtime variable. Either hardcode it, use a template parameter, or use dynamic shared memory with `extern __shared__`.

**Register pressure is a real constraint.** `float S[1024]` looks innocent but puts 1024 floats per thread into local memory, which spills to global memory when registers run out. With 1024 threads per block that's enough to hit `cudaErrorLaunchOutOfResources` before the kernel even starts. The fix **online softmax** isn't just algorithmically cleaner, it's what makes the kernel actually launchable at scale.

**Doing extra work to avoid work can backfire.** The `any_active` sparsity check inside the kernel reads global memory for every tile to decide whether to skip it. At short sequences this overhead dominates any savings from skipping. Precomputing the active tile list on the CPU and passing it in would remove this entirely.


---

## What's next

- **Precompute active tile list on CPU** — right now the `any_active` sparsity check runs inside the GPU kernel, reading global memory for every tile. This is doing extra work to avoid work. The fix is to precompute a list of `(row_tile, col_tile)` pairs on the CPU and pass it in, so the GPU just iterates the list and skips blindly.

- **Test at N = 2048, 4096** — this is where sparse attention should actually win. At short sequences the kernel launch overhead dominates. At long sequences, skipping most of the O(n²) tiles is the whole point.

- **Implement BLAST (Block Sparse Attention)** — planning to read the BLAST paper and implement their block sparsity pattern. More structured than the local+stride pattern here, better hardware utilization.

- **Wrap in `nn.Module`** — make it drop-in compatible with `nn.MultiheadAttention` so it can slot into an existing transformer without rewriting the model.

- **Multi-head support** — current kernel handles one head at a time. Real transformers run 8–96 heads in parallel. Need to add a batch/head dimension and launch accordingly.
