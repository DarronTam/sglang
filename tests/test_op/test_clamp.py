"""
Tests for clamp operators.

Dispatched via direct TORCH_LIBRARY_IMPL registration -> ZENL clamp kernel
(zenlClamp).

First-stage ATen coverage focuses on scalar-bound clamp for float32 /
bfloat16 / int8 / fp8e4m3.
"""

import pytest
import torch

from conftest import assert_close, to_zeus


class TestClamp:
    def test_clamp_basic(self):
        x = torch.tensor([-2.0, -0.5, 0.5, 3.0])
        assert_close(
            torch.clamp(to_zeus(x), min=-1.0, max=1.0),
            torch.clamp(x, min=-1.0, max=1.0),
        )

    def test_clamp_method(self):
        x = torch.randn(128)
        assert_close(
            to_zeus(x).clamp(min=-0.25, max=0.25),
            x.clamp(min=-0.25, max=0.25),
        )

    def test_clamp_scalar_tensor(self):
        x = torch.tensor(3.14)
        result = torch.clamp(to_zeus(x), min=-1.0, max=1.0)
        expected = torch.clamp(x, min=-1.0, max=1.0)
        assert_close(result, expected)
        assert result.dim() == 0

    def test_clamp_2d(self):
        x = torch.randn(16, 32)
        assert_close(
            torch.clamp(to_zeus(x), min=-0.5, max=0.5),
            torch.clamp(x, min=-0.5, max=0.5),
        )

    def test_clamp_non_contiguous(self):
        x = torch.randn(8, 16).t()
        assert not x.is_contiguous()
        assert_close(
            torch.clamp(to_zeus(x), min=-0.5, max=0.5),
            torch.clamp(x, min=-0.5, max=0.5),
        )

    def test_clamp_min_only(self):
        x = torch.tensor([-3.0, -1.0, 0.5, 2.0])
        assert_close(torch.clamp(to_zeus(x), min=-0.5), torch.clamp(x, min=-0.5))

    def test_clamp_max_only(self):
        x = torch.tensor([-3.0, -1.0, 0.5, 2.0])
        assert_close(torch.clamp(to_zeus(x), max=0.75), torch.clamp(x, max=0.75))

    def test_clamp_equal_bounds(self):
        x = torch.randn(64)
        assert_close(
            torch.clamp(to_zeus(x), min=0.5, max=0.5),
            torch.clamp(x, min=0.5, max=0.5),
        )

    def test_clamp_min_greater_than_max(self):
        x = torch.tensor([-3.0, -1.0, 0.5, 2.0])
        assert_close(
            torch.clamp(to_zeus(x), min=2.0, max=1.0),
            torch.clamp(x, min=2.0, max=1.0),
        )

    def test_clamp_out(self):
        x = torch.randn(128)
        out = torch.empty_like(to_zeus(x))
        ret = torch.clamp(to_zeus(x), min=-1.0, max=1.0, out=out)
        assert ret is out
        assert_close(out, torch.clamp(x, min=-1.0, max=1.0))

    def test_clamp_out_resizes_output(self):
        x = torch.randn(8, 16)
        out = torch.empty(1, device="zeus")
        ret = torch.clamp(to_zeus(x), min=-0.25, max=0.25, out=out)
        assert ret is out
        assert out.shape == x.shape
        assert_close(out, torch.clamp(x, min=-0.25, max=0.25))

    def test_clamp_out_non_contiguous(self):
        x = torch.randn(3, 2)
        base = torch.empty(2, 3, device="zeus")
        out = base.t()
        ret = torch.clamp(to_zeus(x), min=-0.5, max=0.5, out=out)
        assert ret is out
        assert not out.is_contiguous()
        assert_close(out, torch.clamp(x, min=-0.5, max=0.5))

    def test_clamp_out_aliases_input(self):
        x = torch.randn(64)
        z = to_zeus(x.clone())
        ret = torch.clamp(z, min=-0.5, max=0.5, out=z)
        assert ret is z
        assert_close(z, torch.clamp(x, min=-0.5, max=0.5))

    def test_clamp_out_partial_overlap_input(self):
        base_cpu = torch.tensor([-2.0, -0.25, 0.75, 2.0, 99.0])
        self_cpu = base_cpu[:-1]
        base = to_zeus(base_cpu)
        self = base[:-1]
        out = base[1:]
        ret = torch.clamp(self, min=-0.5, max=0.5, out=out)
        expected = torch.clamp(self_cpu, min=-0.5, max=0.5)
        assert ret is out
        assert_close(out, expected)

    def test_clamp_inplace(self):
        x = torch.randn(256)
        z = to_zeus(x.clone())
        ret = z.clamp_(min=-0.75, max=0.5)
        assert ret is z
        assert_close(z, x.clamp(min=-0.75, max=0.5))

    def test_clamp_non_contiguous_inplace(self):
        x = torch.randn(4, 8)
        z = to_zeus(x.clone()).t()
        ret = z.clamp_(min=-0.75, max=0.5)
        assert ret is z
        assert not z.is_contiguous()
        assert_close(z, x.t().clamp(min=-0.75, max=0.5))

    def test_clamp_bfloat16(self):
        x = torch.randn(256, dtype=torch.bfloat16)
        assert_close(
            torch.clamp(to_zeus(x), min=-0.5, max=0.5),
            torch.clamp(x, min=-0.5, max=0.5),
        )

    def test_clamp_int8(self):
        x = torch.tensor([-120, -5, 10, 120], dtype=torch.int8)
        result = torch.clamp(to_zeus(x), min=-10, max=20).cpu()
        expected = torch.tensor([-10, -5, 10, 20], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_clamp_int8_float_bounds_promote_to_float(self):
        x = torch.tensor([-120, -5, 10, 120], dtype=torch.int8)
        result = torch.clamp(to_zeus(x), min=-10.5, max=20.25)
        expected = torch.clamp(x, min=-10.5, max=20.25)
        assert result.dtype == expected.dtype == torch.float32
        assert_close(result, expected)

    def test_clamp_int8_out_of_range_integer_bound(self):
        x = to_zeus(torch.tensor([-120, -5, 10, 120], dtype=torch.int8))
        with pytest.raises(RuntimeError, match="overflow"):
            torch.clamp(x, min=-200, max=20)

    def test_clamp_fp8e4m3(self):
        x = torch.tensor([-2.0, -0.5, 0.5, 3.0]).to(torch.float8_e4m3fn)
        result = torch.clamp(to_zeus(x), min=-1.0, max=1.0).float().cpu()
        expected = torch.tensor([-1.0, -0.5, 0.5, 1.0])
        torch.testing.assert_close(result, expected, atol=0.125, rtol=0)

    def test_clamp_empty(self):
        x = torch.empty(0)
        result = torch.clamp(to_zeus(x), min=-1.0, max=1.0)
        assert result.numel() == 0

    def test_clamp_large(self):
        x = torch.randn(8192)
        assert_close(
            torch.clamp(to_zeus(x), min=-1.0, max=1.0),
            torch.clamp(x, min=-1.0, max=1.0),
        )

    def test_clamp_without_bounds_raises(self):
        x = to_zeus(torch.randn(4))
        with pytest.raises(RuntimeError):
            torch.clamp(x)

    @pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float64])
    def test_clamp_unsupported_dtypes_raise(self, dtype):
        x = torch.ones(4, dtype=dtype)
        with pytest.raises(RuntimeError):
            torch.clamp(to_zeus(x), min=0, max=1)

    def test_clamp_out_wrong_dtype_raises(self):
        x = torch.randn(4)
        out = torch.empty(4, dtype=torch.bfloat16, device="zeus")
        with pytest.raises(RuntimeError):
            torch.clamp(to_zeus(x), min=-1.0, max=1.0, out=out)

    def test_clamp_int32(self):
        x = torch.tensor([-200, -5, 0, 10, 200], dtype=torch.int32)
        result = torch.clamp(to_zeus(x), min=-10, max=20).cpu()
        expected = torch.clamp(x, min=-10, max=20)
        assert torch.equal(result, expected)

    def test_clamp_int32_min_only(self):
        x = torch.tensor([-5, 0, 10], dtype=torch.int32)
        result = torch.clamp(to_zeus(x), min=0).cpu()
        expected = torch.clamp(x, min=0)
        assert torch.equal(result, expected)
