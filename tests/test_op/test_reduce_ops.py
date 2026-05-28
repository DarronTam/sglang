"""
Tests for Zeus reduce operators: sum, mean, norm.

Dispatched via REGISTER_PRIVATEUSE1_DISPATCH (sum_stub, mean_stub, norm_stub)
→ ZENL kernel (zenlReduce).
"""

import torch
import pytest
from conftest import DEVICE, to_zeus, assert_close


# ============================================================================
# torch.sum
# ============================================================================

class TestSum:
    def test_full_reduce(self):
        x = torch.randn(4, 5)
        assert_close(to_zeus(x).sum(), x.sum())

    def test_single_dim(self):
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).sum(dim=1), x.sum(dim=1))

    def test_keepdim(self):
        x = torch.randn(3, 4, 5)
        out = to_zeus(x).sum(dim=1, keepdim=True)
        ref = x.sum(dim=1, keepdim=True)
        assert_close(out, ref)
        assert out.shape == ref.shape

    def test_multi_dim(self):
        x = torch.randn(2, 3, 4, 5)
        assert_close(to_zeus(x).sum(dim=(1, 3)), x.sum(dim=(1, 3)))

    def test_last_dim(self):
        """Contiguous reduction → fast-path kernel."""
        x = torch.randn(8, 16)
        assert_close(to_zeus(x).sum(dim=-1), x.sum(dim=-1))

    def test_first_dim(self):
        """Prefix reduction → mid-dim kernel."""
        x = torch.randn(8, 16)
        assert_close(to_zeus(x).sum(dim=0), x.sum(dim=0))

    def test_1d(self):
        x = torch.randn(100)
        assert_close(to_zeus(x).sum(), x.sum())

    def test_scalar(self):
        x = torch.tensor(3.14)
        assert_close(to_zeus(x).sum(), x.sum())

    def test_mid_dim_3d(self):
        """[B, M, N] reduce middle dim → mid kernel."""
        x = torch.randn(4, 8, 16)
        out = to_zeus(x).sum(dim=1)
        assert_close(out, x.sum(dim=1))
        assert out.shape == (4, 16)

    def test_mid_dims_4d(self):
        """[2, 3, 4, 5] reduce dims (1,2) → B=2, M=12, N=5."""
        x = torch.randn(2, 3, 4, 5)
        assert_close(to_zeus(x).sum(dim=(1, 2)), x.sum(dim=(1, 2)))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_dtypes(self, dtype):
        x = torch.randn(4, 8, dtype=dtype)
        assert_close(to_zeus(x).sum(dim=1), x.sum(dim=1))

    def test_large(self):
        x = torch.randn(64, 128, 32)
        assert_close(to_zeus(x).sum(dim=1), x.sum(dim=1),
                     rtol=1e-4, atol=1e-4)

    def test_non_contiguous(self):
        """Transposed tensor (non-contiguous)."""
        x = torch.randn(4, 8).t()
        assert not x.is_contiguous()
        assert_close(to_zeus(x).sum(dim=1), x.sum(dim=1))

    # --- decomposed reduce (non-contiguous dims) ----------------------------

    def test_scattered_dims_3d(self):
        """Reduce dims {0, 2} in 3D → decomposed path."""
        x = torch.randn(2, 3, 4)
        assert_close(to_zeus(x).sum(dim=(0, 2)), x.sum(dim=(0, 2)))

    def test_scattered_dims_4d(self):
        """Reduce dims {0, 3} in 4D → decomposed path."""
        x = torch.randn(2, 3, 2, 4)
        assert_close(to_zeus(x).sum(dim=(0, 3)), x.sum(dim=(0, 3)))

    def test_scattered_dims_5d_three_blocks(self):
        """Reduce dims {0, 2, 4} in 5D → 3-block decomposition."""
        x = torch.randn(2, 3, 2, 3, 4)
        assert_close(to_zeus(x).sum(dim=(0, 2, 4)), x.sum(dim=(0, 2, 4)),
                     rtol=1e-4, atol=1e-4)

    def test_scattered_dims_bf16(self):
        """BF16 decomposed path."""
        x = torch.randn(3, 4, 5, dtype=torch.bfloat16)
        assert_close(to_zeus(x).sum(dim=(0, 2)), x.sum(dim=(0, 2)))

    def test_scattered_dims_keepdim(self):
        """Decomposed path with keepdim=True."""
        x = torch.randn(2, 3, 4)
        out = to_zeus(x).sum(dim=(0, 2), keepdim=True)
        ref = x.sum(dim=(0, 2), keepdim=True)
        assert_close(out, ref)
        assert out.shape == ref.shape


# ============================================================================
# torch.mean
# ============================================================================

class TestMean:
    def test_full_reduce(self):
        x = torch.randn(4, 5)
        assert_close(to_zeus(x).mean(), x.mean())

    def test_single_dim(self):
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).mean(dim=2), x.mean(dim=2))

    def test_keepdim(self):
        x = torch.randn(3, 4, 5)
        out = to_zeus(x).mean(dim=1, keepdim=True)
        ref = x.mean(dim=1, keepdim=True)
        assert_close(out, ref)
        assert out.shape == ref.shape

    def test_multi_dim(self):
        x = torch.randn(2, 3, 4, 5)
        assert_close(to_zeus(x).mean(dim=(0, 2)), x.mean(dim=(0, 2)))


# ============================================================================
# torch.norm
# ============================================================================

class TestNorm:
    def test_l1(self):
        x = torch.randn(3, 4, 5)
        assert_close(torch.norm(to_zeus(x), p=1, dim=1),
                     torch.norm(x, p=1, dim=1))

    def test_l2(self):
        x = torch.randn(3, 4, 5)
        assert_close(torch.norm(to_zeus(x), p=2, dim=1),
                     torch.norm(x, p=2, dim=1))

    def test_l2_keepdim(self):
        x = torch.randn(3, 4, 5)
        out = torch.norm(to_zeus(x), p=2, dim=1, keepdim=True)
        ref = torch.norm(x, p=2, dim=1, keepdim=True)
        assert_close(out, ref)
        assert out.shape == ref.shape

    def test_frobenius(self):
        x = torch.randn(4, 5)
        assert_close(torch.norm(to_zeus(x)), torch.norm(x))
