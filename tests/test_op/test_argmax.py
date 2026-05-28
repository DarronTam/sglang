"""
Tests for Zeus argmax / argmin operators.

Dispatched via REGISTER_PRIVATEUSE1_DISPATCH (argmax_stub, argmin_stub)
-> ZENL kernel (zenlReduce with REDUCE_MAX/MIN + ONLY_INDICES).
"""

import torch
import pytest
from conftest import DEVICE, to_zeus, assert_close


# ============================================================================
# torch.argmax
# ============================================================================

class TestArgmax:
    def test_single_dim_last(self):
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).argmax(dim=2), x.argmax(dim=2))

    def test_single_dim_first(self):
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).argmax(dim=0), x.argmax(dim=0))

    def test_single_dim_mid(self):
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).argmax(dim=1), x.argmax(dim=1))

    def test_keepdim(self):
        x = torch.randn(3, 4, 5)
        out = to_zeus(x).argmax(dim=1, keepdim=True)
        ref = x.argmax(dim=1, keepdim=True)
        assert_close(out, ref)
        assert out.shape == ref.shape

    def test_2d(self):
        x = torch.randn(8, 16)
        assert_close(to_zeus(x).argmax(dim=-1), x.argmax(dim=-1))

    def test_2d_dim0(self):
        x = torch.randn(8, 16)
        assert_close(to_zeus(x).argmax(dim=0), x.argmax(dim=0))

    def test_1d(self):
        x = torch.randn(100)
        assert_close(to_zeus(x).argmax(dim=0), x.argmax(dim=0))

    def test_full_reduce(self):
        """argmax with no dim -> reduce all dimensions."""
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).argmax(), x.argmax())

    def test_full_reduce_2d(self):
        x = torch.randn(8, 16)
        assert_close(to_zeus(x).argmax(), x.argmax())

    def test_negative_dim(self):
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).argmax(dim=-2), x.argmax(dim=-2))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_dtypes(self, dtype):
        x = torch.randn(4, 8, dtype=dtype)
        assert_close(to_zeus(x).argmax(dim=1), x.argmax(dim=1))

    def test_large(self):
        x = torch.randn(64, 128, 32)
        assert_close(to_zeus(x).argmax(dim=1), x.argmax(dim=1))

    def test_single_element_dim(self):
        """Dimension with size 1 -> argmax is always 0."""
        x = torch.randn(3, 1, 5)
        assert_close(to_zeus(x).argmax(dim=1), x.argmax(dim=1))

    def test_4d(self):
        x = torch.randn(2, 3, 4, 5)
        assert_close(to_zeus(x).argmax(dim=2), x.argmax(dim=2))

    def test_output_dtype_is_long(self):
        """argmax output should always be int64."""
        x = torch.randn(3, 4, 5)
        out = to_zeus(x).argmax(dim=1)
        assert out.dtype == torch.long


# ============================================================================
# torch.argmin
# ============================================================================

class TestArgmin:
    def test_single_dim_last(self):
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).argmin(dim=2), x.argmin(dim=2))

    def test_single_dim_first(self):
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).argmin(dim=0), x.argmin(dim=0))

    def test_single_dim_mid(self):
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).argmin(dim=1), x.argmin(dim=1))

    def test_keepdim(self):
        x = torch.randn(3, 4, 5)
        out = to_zeus(x).argmin(dim=1, keepdim=True)
        ref = x.argmin(dim=1, keepdim=True)
        assert_close(out, ref)
        assert out.shape == ref.shape

    def test_2d(self):
        x = torch.randn(8, 16)
        assert_close(to_zeus(x).argmin(dim=-1), x.argmin(dim=-1))

    def test_full_reduce(self):
        """argmin with no dim -> reduce all dimensions."""
        x = torch.randn(3, 4, 5)
        assert_close(to_zeus(x).argmin(), x.argmin())

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_dtypes(self, dtype):
        x = torch.randn(4, 8, dtype=dtype)
        assert_close(to_zeus(x).argmin(dim=1), x.argmin(dim=1))

    def test_large(self):
        x = torch.randn(64, 128, 32)
        assert_close(to_zeus(x).argmin(dim=1), x.argmin(dim=1))

    def test_output_dtype_is_long(self):
        """argmin output should always be int64."""
        x = torch.randn(3, 4, 5)
        out = to_zeus(x).argmin(dim=1)
        assert out.dtype == torch.long
