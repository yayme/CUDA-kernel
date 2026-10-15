"""Shared helpers for tests and benchmarks: the sparsity mask and an fp64 reference."""
import torch


def build_mask(n, window, stride, device="cuda"):
    """Bool [n, n] mask: query i attends to key j if |i - j| <= window or j % stride == 0."""
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    return ((i - j).abs() <= window) | (j % stride == 0)


def reference_attention(q, k, v, mask=None):
    """softmax(QK^T / sqrt(d), masked) @ V computed in fp64."""
    q, k, v = q.double(), k.double(), v.double()
    scores = q @ k.T / q.shape[-1] ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
