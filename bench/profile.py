"""Run one kernel once at one size, so Nsight Compute has exactly one launch to profile.

Run from the repo root:
  ncu --set full -k regex:flash_attention -c 1 -o bench/results/v2_n4096 \
      python -m bench.profile --kernel v2 --n 4096
"""
import argparse

import torch

import sparse_attention as sa
from reference import build_mask

KERNELS = {"v1": sa.sparse_attention, "v2": sa.sparse_attention_v2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=KERNELS, default="v2")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    q, k, v = (torch.randn(args.n, 64, device="cuda") for _ in range(3))
    mask = build_mask(args.n, args.window, args.stride).int().contiguous()
    KERNELS[args.kernel](q, k, v, mask)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
