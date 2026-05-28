"""
Tests for neg operators.

Dispatched via direct TORCH_LIBRARY_IMPL registration -> ZENL neg kernel
(zenlNeg).
"""

import pytest
import torch

from conftest import assert_close, to_zeus


class TestNeg:
    def test_neg_basic(self):
        x = torch.tensor([-2.0, -0.5, 0.5, 3.0])
        assert_close(torch.neg(to_zeus(x)), torch.neg(x))

    def test_neg_method(self):
        x = torch.randn(128)
        assert_close(to_zeus(x).neg(), x.neg())

    def test_neg_inplace(self):
        x = torch.randn(256)
        z = to_zeus(x.clone())
        ret = z.neg_()
        assert ret is z
        assert_close(z, x.neg())

    def test_neg_out(self):
        x = torch.randn(128)
        out = torch.empty_like(to_zeus(x))
        ret = torch.neg(to_zeus(x), out=out)
        assert ret is out
        assert_close(out, torch.neg(x))

    def test_neg_out_resizes_output(self):
        x = torch.randn(8, 16)
        out = torch.empty(1, device="zeus")
        ret = torch.neg(to_zeus(x), out=out)
        assert ret is out
        assert out.shape == x.shape
        assert_close(out, torch.neg(x))

    def test_neg_out_non_contiguous(self):
        x = torch.randn(3, 2)
        base = torch.empty(2, 3, device="zeus")
        out = base.t()
        ret = torch.neg(to_zeus(x), out=out)
        assert ret is out
        assert not out.is_contiguous()
        assert_close(out, torch.neg(x))

    def test_neg_out_partial_overlap_input(self):
        base_cpu = torch.tensor([1.0, -2.0, 3.0, -4.0, 99.0])
        self_cpu = base_cpu[:-1]
        base = to_zeus(base_cpu)
        self = base[:-1]
        out = base[1:]
        ret = torch.neg(self, out=out)
        expected = torch.neg(self_cpu)
        assert ret is out
        assert_close(out, expected)

    def test_neg_non_contiguous_inplace(self):
        x = torch.randn(4, 8)
        z = to_zeus(x.clone()).t()
        ret = z.neg_()
        assert ret is z
        assert not z.is_contiguous()
        assert_close(z, x.t().neg())

    def test_neg_scalar_tensor(self):
        x = torch.tensor(3.14)
        result = torch.neg(to_zeus(x))
        expected = torch.neg(x)
        assert result.dim() == 0
        assert_close(result, expected)

    def test_neg_bfloat16(self):
        x = torch.randn(256, dtype=torch.bfloat16)
        assert_close(torch.neg(to_zeus(x)), torch.neg(x))

    def test_neg_int8(self):
        x = torch.tensor([-5, 0, 10, 127], dtype=torch.int8)
        result = torch.neg(to_zeus(x)).cpu()
        expected = torch.neg(x)
        assert torch.equal(result, expected)

    def test_neg_int8_min_value_wrap(self):
        x = torch.tensor([-128], dtype=torch.int8)
        result = torch.neg(to_zeus(x)).cpu()
        expected = torch.neg(x)
        assert torch.equal(result, expected)
        assert int(result.item()) == -128

    def test_neg_fp8e4m3(self):
        x = torch.tensor([-2.0, -0.5, 0.5, 3.0]).to(torch.float8_e4m3fn)
        result = torch.neg(to_zeus(x)).float().cpu()
        expected = torch.tensor([2.0, 0.5, -0.5, -3.0])
        torch.testing.assert_close(result, expected, atol=0.125, rtol=0)

    def test_neg_empty(self):
        x = torch.empty(0)
        result = torch.neg(to_zeus(x))
        assert result.numel() == 0

    def test_neg_large(self):
        x = torch.randn(8192)
        assert_close(torch.neg(to_zeus(x)), torch.neg(x))

    @pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float64])
    def test_neg_unsupported_dtypes_raise(self, dtype):
        x = torch.ones(4, dtype=dtype)
        with pytest.raises(RuntimeError):
            torch.neg(to_zeus(x))

    def test_neg_int32(self):
        x = torch.tensor([-5, 0, 10, 127], dtype=torch.int32)
        result = torch.neg(to_zeus(x)).cpu()
        expected = torch.neg(x)
        assert torch.equal(result, expected)

    def test_neg_int32_min_wrap(self):
        x = torch.tensor([-2147483648], dtype=torch.int32)
        result = torch.neg(to_zeus(x)).cpu()
        expected = torch.neg(x)  # INT32_MIN wraps to itself
        assert torch.equal(result, expected)
