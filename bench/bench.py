"""Benchmark every kernel against PyTorch baselines across sequence lengths.

Run from the repo root:   python -m bench.bench
Writes bench/results/results.csv; plot it with   python -m bench.plot
"""
import argparse
import csv
import os
import statistics

import torch
import torch.nn.functional as F

import sparse_attention as sa
from reference import build_mask

D = 64
RESULTS = os.path.join(os.path.dirname(__file__), "results", "results.csv")


def time_once(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def time_ms(fn, budget_ms=1000, max_reps=50):
    """Median runtime in ms. Rep count adapts so slow kernels at large N stay within the budget."""
    first = time_once(fn)
    reps = max(5, min(max_reps, int(budget_ms / max(first, 1e-3))))
    for _ in range(min(10, reps)):
        fn()
    return statistics.median(time_once(fn) for _ in range(reps))


def methods_for(n, q, k, v, mask_bool, mask_int):
    """name -> (callable, is_sparse). Kernels that can't run at this N are left out."""
    q4, k4, v4 = (t.view(1, 1, n, D) for t in (q, k, v))
    methods = {
        "dense_sdpa": (lambda: F.scaled_dot_product_attention(q4, k4, v4), False),
        "masked_sdpa": (lambda: F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask_bool), True),
        "v2": (lambda: sa.sparse_attention_v2(q, k, v, mask_int), True),
    }
    if n <= 1024:
        methods["naive"] = (lambda: sa.naive_attention(q, k, v), False)
        methods["v1"] = (lambda: sa.sparse_attention(q, k, v, mask_int), True)
    return methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", default="64,128,256,512,1024,2048,4096,8192")
    # window:stride pairs. 2:4 is the original pattern; 32:512 is a genuinely sparse one
    parser.add_argument("--patterns", default="2:4,32:512")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    rows = []
    for pattern in args.patterns.split(","):
        window, stride = map(int, pattern.split(":"))
        for n in map(int, args.ns.split(",")):
            q, k, v = (torch.randn(n, D, device="cuda") for _ in range(3))
            mask_bool = build_mask(n, window, stride)
            mask_int = mask_bool.int().contiguous()
            active_pairs = int(mask_bool.sum())

            for name, (fn, is_sparse) in methods_for(n, q, k, v, mask_bool, mask_int).items():
                try:
                    ms = time_ms(fn)
                except torch.cuda.OutOfMemoryError:
                    ms = float("nan")
                # useful work: QK^T and PV, 2 FLOPs per multiply-add, over the pairs that matter
                pairs = active_pairs if is_sparse else n * n
                gflops = 4 * pairs * D / (ms * 1e6)
                rows.append(dict(method=name, n=n, window=window, stride=stride, ms=ms,
                                 active_pairs=active_pairs, density=active_pairs / n**2, useful_gflops=gflops))
                print(f"w={window:<3} s={stride:<4} N={n:<5} {name:<12} {ms:9.3f} ms  {gflops:8.1f} GFLOP/s")

    with open(RESULTS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
