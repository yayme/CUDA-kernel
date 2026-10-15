# Upgrade plan: one big idea per phase

Each phase ships a new kernel version **next to** the old ones (v3, v4, …). Don't overwrite earlier versions. Benchmarked side by side, they show the progression, and that progression is what you talk about in an interview.

Every phase follows the same loop:

1. **Idea:** the one principle this phase is about.
2. **Change:** the code you write.
3. **Prove it's correct:** tests pass against the PyTorch reference.
4. **Prove it helped:** benchmark plus one profiler metric that should move.
5. **One line for the log:** what you learned, written into `log.md`.

Target hardware stays the Colab T4 (sm_75, 16 GB, fp16 tensor cores, **no** `cp.async`, **no** bf16). Write the code locally and run it on Colab.

---

## Bugs and hidden costs in the current code (fix or address along the way)

Reading `sparse_attention.cu` turned up a few things that matter more than they look:

| Issue | Where | Consequence | Fixed in |
|---|---|---|---|
| `grid(N/Br)` and `tile < seq_len/Bc` use integer division | launchers, both kernels | When N isn't a multiple of 16, the last rows are silently never computed | Phase 0 |
| No dtype/contiguity/device checks, no `cudaGetLastError()` | launchers | Bad input or a failed launch returns garbage silently | Phase 0 |
| `block(Br, D)`: `threadIdx.x` (the row) varies fastest within a warp | K/V/Q loads | Consecutive threads read addresses 64 floats apart, so loads are **uncoalesced** | Phase 2 |
| All 64 threads on a query row compute the **same** 16 dot products | `scores[t]` loop in v2 | 64× redundant arithmetic. This is probably the biggest cost in v2 | Phase 2 |
| 1024 threads per block | launch config | Only one block is resident per SM, so nothing hides stalls | Phase 2 |
| Every thread scans the N×N mask for each tile | `any_active` loop | Redundant global reads, plus an O(N²) mask (256 MB at N=8192) | Phase 1 |

---

## Phase 0: Measurement foundation
**Idea:** you can't claim a speedup you can't measure, and you can't trust a measurement from a wrong kernel.

**Change**
- Restructure the repo: `csrc/` (kernels and bindings), `tests/`, `bench/`. Keep `benchmark.ipynb` as a thin Colab driver that calls these files.
- `tests/test_correctness.py` (pytest): compare every kernel against a reference, `softmax((QKᵀ/√d).masked_fill(~mask, -inf)) @ V` in fp64. Sweep N ∈ {16, 17, 63, 64, 100, 512, 2048} (include sizes that aren't multiples of 16), window and stride values, and random seeds. Use `atol=1e-4` for fp32 kernels.
- Fix the ceil-div bugs (`(N + Br - 1) / Br`) and add `TORCH_CHECK`s for `is_cuda`, `is_contiguous`, `dtype`, and `D == 64`. Add `C10_CUDA_KERNEL_LAUNCH_CHECK()` after each launch.
- `bench/bench.py`: time with CUDA events, 10 warmup runs, median of 50, N up to 8192. Use three baselines:
  - PyTorch dense, unmasked (the speed ceiling)
  - `F.scaled_dot_product_attention` with the boolean mask (the fair comparison)
  - each of your kernels
- Write results to CSV and plot both ms vs N **and** effective throughput. For throughput, count only the FLOPs in active pairs, i.e. useful work.
- Add `extra_compile_args={'nvcc': ['-Xptxas', '-v', '-lineinfo']}` to `setup.py` so each build prints registers per thread and spill bytes.
- Run `ncu --set full` on one N=4096 run of v2 and save the report. That becomes the baseline profile.

**Done when:** all tests pass (including odd N), the benchmark CSV covers N up to 8192, and you have a baseline ncu report for v2.

**Interview line:** "Before optimizing I built a correctness harness against an fp64 reference, and it immediately caught that non-multiple-of-16 lengths were silently dropped."

---

## Phase 1: Move the sparsity decision off the GPU (v3)
**Idea:** don't make the hot path compute something it can be told. Also, sparsity only pays off at tile granularity.

**Change**
- On the host, build a **block-sparse layout in CSR form over tiles**:
  - `row_ptr[num_q_tiles + 1]` and `col_idx[nnz_tiles]`: the K/V tiles that each query tile visits
  - `tile_kind[nnz_tiles]`: `FULL` (every pair active, so no per-element mask) or `PARTIAL`
- Drop the N×N mask. For `PARTIAL` tiles, compute the element mask analytically in the kernel (`abs(i-j) <= w || j % s == 0`). Memory goes from O(N²) to O(N·w / tile).
- The kernel loop becomes `for (p = row_ptr[b]; p < row_ptr[b+1]; ++p)`. The loop bound is uniform across the block, so `__syncthreads()` inside it stays safe.
- Keep the layout builder general: it should accept any pattern (local+global, or later BigBird/BLAST random blocks). This is where the "Implement BLAST" goal plugs in for free.

**Measure:** runtime vs N should now be roughly **linear** (it tracks `nnz_tiles`, not N²). Plot `nnz_tiles` next to runtime. In ncu, DRAM bytes read should drop sharply.

**Interview line:** "The kernel no longer decides anything. It walks a precomputed tile list, and the mask stopped being O(N²)."

---

## Phase 2: Map threads to work properly (v4)
**Idea:** work partitioning, coalescing and occupancy. This is the FlashAttention-2 lesson.

**Change**
- **Stop the 64× redundancy.** Compute the score tile `S[Br][Bc]` once, cooperatively: each thread computes a few `(row, key)` dot products and writes them into a shared `Ss[Br][Bc]`. Do the online softmax row statistics with **warp shuffles** (`__shfl_xor_sync`) rather than serial loops. Then each thread updates the output columns it owns.
- **Coalesce loads.** Make the fastest-varying thread index walk along `D`, either by swapping the block dimensions or by using a flat `threadIdx.x` with a computed (row, col). Consecutive lanes then read consecutive addresses. Pad shared arrays (`[D+1]`) wherever the access pattern would cause bank conflicts.
- **Shrink the block.** Use 128 or 256 threads, with each thread owning several output elements (register tiling). Bigger `Br` (e.g. 64) gives more reuse per K/V load.
- Template on `D` (`template<int HEAD_DIM>`) so the shared memory size is still compile-time but not hard-wired to 64.

**Measure:** ncu achieved occupancy, `l1tex__t_sectors_per_request` (coalescing), shared bank conflicts, and instructions executed (should fall by roughly the redundancy factor). Expect the largest single speedup of the whole project here.

**Interview line:** "Every thread in a row was recomputing the same dot products, 64× redundant work. Repartitioning so each score is computed once, with warp shuffles for the row max and sum, was the biggest win."

---

## Phase 3: Double buffering (v5)
**Idea:** latency you can't remove, you hide.

**Change**
- Two shared buffers for K/V. While computing on tile `p`, prefetch tile `p+1`.
- The T4 (sm_75) has **no `cp.async`**, so use register-staged prefetch: issue the global loads for `p+1` into registers at the top of the iteration, compute on `p`, then store the registers into the other shared buffer, with one `__syncthreads()`. Note in the README that on sm_80+ this would be `cp.async` with `__pipeline_commit` (a natural follow-up if you get A100 time).
- Watch the shared memory budget: two K/V buffers must still leave room for 2+ resident blocks per SM.

**Measure:** ncu "stall long scoreboard" should fall, and the timeline should show memory and compute overlapping. Gains are modest when you're already compute-bound. Report whatever you actually get.

**Interview line:** "On sm_75 there's no async copy, so I staged the next tile through registers. It hides latency, but it costs registers, so I checked occupancy afterward."

---

## Phase 4: Tensor cores (v6)
**Idea:** use the matmul unit. It's the main reason a handwritten kernel loses to cuBLAS.

**Change**
- fp16 Q/K/V inputs with **fp32 accumulation and fp32 softmax statistics**.
- Use WMMA (`nvcuda::wmma`, 16×16×16 fragments) for both `S = Q Kᵀ` and `O += P V`. Your 16-wide tiles already line up with the fragment shape.
- The softmax step needs element access to the accumulator fragment. The simplest correct route is to store the `S` fragment to shared memory, do the online softmax there, convert `P` to fp16, and load it back as a fragment. (Operating on fragments in registers directly is the advanced version; do it only if profiling says the round-trip matters.)
- Loosen the test tolerance (~`1e-2` relative) and compare against fp16 `scaled_dot_product_attention`.

**Measure:** ncu tensor pipe utilization. Compare effective TFLOP/s against the T4's ~65 TFLOP/s fp16 peak and its ~8 TFLOP/s fp32 peak. This is where you finally have a real shot at beating masked SDPA at long N. Report it either way.

**Interview line:** "The inner product is a 16×16 dense matmul, so moving it to tensor cores with fp32 accumulation was the step change, and I kept the softmax stats in fp32 for stability."

---

## Phase 5: Batch and multi-head (v7)
**Idea:** free parallelism, and the shape real models use.

**Change**
- Accept `[B, H, N, D]` and launch `grid(num_q_tiles, H, B)`. The tile layout is shared across heads and batch, so it's built once.
- At short N with B·H large, this fills the GPU properly. It's the fix for "slower at small N" being partly an occupancy artifact.
- Optional: causal + local pattern (just a different layout builder).

**Measure:** benchmark at realistic shapes, e.g. B=4, H=12, D=64, N ∈ {1k…8k}.

---

## Phase 6: Backward pass with recomputation (v8)
**Idea:** trade arithmetic for memory. This is the half of FlashAttention you haven't implemented yet.

**Change**
- In the forward pass, also save the per-row **logsumexp** `L = m + log(l)` (one float per row). Don't store `P`.
- Backward kernel(s), recomputing `P = exp(S − L)` tile by tile:
  - `Δ = rowsum(dO ∘ O)` (small pre-kernel)
  - `dV += Pᵀ dO`, `dP = dO Vᵀ`, `dS = P ∘ (dP − Δ)`
  - `dQ += dS K / √d`, `dK += dSᵀ Q / √d`
- Parallelize over **K/V tiles**, the FA2 approach: each block owns a K/V tile, loops over the query tiles that attend to it, and accumulates dK/dV in registers. This needs the **transposed** CSR layout (col → rows), which the Phase 1 builder can emit. dQ needs `atomicAdd` or a separate pass. Start with a separate pass for simplicity and determinism.
- Wrap it in `torch.autograd.Function`. Test against PyTorch autograd on the reference implementation at small sizes.

**Measure:** peak memory vs N for your op compared with autograd through the dense reference (yours linear, theirs quadratic), plus backward time.

**Interview line:** "Backward stores only the output and one logsumexp per row and recomputes the scores, so memory is linear. Same tile list, transposed."

---

## Phase 7: Packaging and an honest write-up
- Make `SparseAttention(nn.Module)` a drop-in for the attention core of `nn.MultiheadAttention` (projections stay in PyTorch). Add a small test that swaps it into a tiny transformer and checks the loss decreases.
- Rewrite the README around the version ladder: one table with all versions vs both baselines across N, and a roofline plot that places each version.
- Keep the honesty rule from your rehearsal notes: state exactly which baseline each speedup is measured against.

---

## Ordering rationale
- **0 before everything:** each later phase depends on its tests and benchmarks.
- **1 before 2:** Phase 1 rewrites the loop structure. Optimizing thread mapping on a loop you're about to replace is wasted work.
- **2 before 3 and 4:** double buffering and tensor cores only pay off once redundant compute is gone. Otherwise they'd hide the real bottleneck.
- **5 is cheap** and could move earlier if you want realistic shapes sooner. It's mostly indexing.
- **6 is the largest** piece of work. It's last because it reuses everything before it: the tile layout, partitioning, and WMMA.

Stretch ideas, after the plan: split-KV / sequence parallelism for long N with few heads, `cp.async` and sm_80 variants, and BLAST/BigBird random-block layouts through the Phase 1 builder.
