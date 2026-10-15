"""Every kernel against the fp64 reference, including lengths that are not multiples of the tile size."""
import pytest
import torch

from reference import build_mask, reference_attention

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)
sa = pytest.importorskip("sparse_attention")

D = 64
SEQ_LENS = [16, 17, 63, 64, 100, 512, 2048]
# (window, stride): the original dense-ish pattern, a genuinely sparse one, and diagonal + token 0 only
PATTERNS = [(2, 4), (8, 64), (0, 10**9)]

# name -> (fn, max supported N)
SPARSE_KERNELS = {
    "v1": (lambda q, k, v, m: sa.sparse_attention(q, k, v, m), 1024),
    "v2": (lambda q, k, v, m: sa.sparse_attention_v2(q, k, v, m), None),
}


def random_qkv(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(n, D, device="cuda", generator=g) for _ in range(3)]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("window,stride", PATTERNS)
@pytest.mark.parametrize("n", SEQ_LENS)
@pytest.mark.parametrize("kernel", SPARSE_KERNELS)
def test_sparse_kernel_matches_reference(kernel, n, window, stride, seed):
    fn, max_n = SPARSE_KERNELS[kernel]
    if max_n is not None and n > max_n:
        pytest.skip(f"{kernel} supports N <= {max_n}")

    q, k, v = random_qkv(n, seed)
    mask = build_mask(n, window, stride)
    out = fn(q, k, v, mask.int().contiguous())
    torch.testing.assert_close(out.double(), reference_attention(q, k, v, mask), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n", [16, 17, 100, 1024])
def test_naive_matches_dense_reference(n):
    q, k, v = random_qkv(n, seed=0)
    out = sa.naive_attention(q, k, v)
    torch.testing.assert_close(out.double(), reference_attention(q, k, v), atol=1e-4, rtol=1e-4)


def test_rejects_bad_inputs():
    q, k, v = random_qkv(64, seed=0)
    mask = build_mask(64, 2, 4).int()
    with pytest.raises(RuntimeError):
        sa.sparse_attention_v2(q.double(), k, v, mask)          # wrong dtype
    with pytest.raises(RuntimeError):
        sa.sparse_attention_v2(q.cpu(), k, v, mask)             # wrong device
    with pytest.raises(RuntimeError):
        sa.sparse_attention_v2(q[:, :32].contiguous(), k, v, mask)  # wrong head dim
    with pytest.raises(RuntimeError):
        sa.sparse_attention_v2(q, k, v, mask.bool())            # mask must be int32
    with pytest.raises(RuntimeError):
        sa.sparse_attention(*random_qkv(2048, 0), build_mask(2048, 2, 4).int())  # v1 caps N at 1024
