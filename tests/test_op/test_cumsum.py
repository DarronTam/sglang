"""
Tests for Zeus cumsum operator.
"""

import pytest
import torch
from conftest import DEVICE, to_zeus, assert_close


class TestCumsum:
    def test_1d_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert_close(to_zeus(x).cumsum(dim=0), x.cumsum(dim=0))

    def test_dim0(self):
        x = torch.randn(4, 5)
        assert_close(to_zeus(x).cumsum(dim=0), x.cumsum(dim=0))

    def test_dim1(self):
        x = torch.randn(4, 5)
        assert_close(to_zeus(x).cumsum(dim=1), x.cumsum(dim=1))

    def test_negative_dim(self):
        x = torch.randn(2, 3, 4)
        assert_close(to_zeus(x).cumsum(dim=-1), x.cumsum(dim=-1))

    def test_non_contiguous(self):
        x = torch.randn(4, 8).t()
        assert not x.is_contiguous()
        assert_close(to_zeus(x).cumsum(dim=1), x.cumsum(dim=1))

    def test_scalar(self):
        x = torch.tensor(5.0)
        assert_close(to_zeus(x).cumsum(dim=0), x.cumsum(dim=0))

    def test_empty(self):
        x = torch.empty(0, 5)
        out = to_zeus(x).cumsum(dim=0)
        ref = x.cumsum(dim=0)
        assert_close(out, ref)
        assert out.shape == ref.shape

    def test_out_variant(self):
        x = torch.randn(3, 4)
        out = torch.empty_like(x, device=DEVICE)
        ret = torch.cumsum(to_zeus(x), dim=1, out=out)
        assert ret is out
        assert_close(out, torch.cumsum(x, dim=1))

    def test_out_variant_uses_out_dtype(self):
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        out = torch.empty(2, 2, dtype=torch.float32, device=DEVICE)
        torch.cumsum(to_zeus(x), dim=1, out=out)
        assert out.dtype == torch.float32
        assert_close(out, torch.cumsum(x, dim=1, out=torch.empty(2, 2, dtype=torch.float32)))

    def test_dtype_override(self):
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        out = torch.cumsum(to_zeus(x), dim=1, dtype=torch.int64)
        ref = torch.cumsum(x, dim=1, dtype=torch.int64)
        assert out.dtype == torch.int64
        assert_close(out, ref)

    def test_int64_input(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        assert_close(to_zeus(x).cumsum(dim=0), x.cumsum(dim=0))

    def test_bfloat16_input(self):
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        assert_close(to_zeus(x).cumsum(dim=1), x.cumsum(dim=1))

    def test_explicit_dtype_mismatch_raises(self):
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        out = torch.empty(2, 2, dtype=torch.int32, device=DEVICE)
        with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
            torch.cumsum(to_zeus(x), dim=1, dtype=torch.int64, out=out)
