"""
Tests for arange operators.

Dispatched via TORCH_LIBRARY_IMPL(aten, PrivateUse1)
-> zenl_arange_internal -> zenlArange kernel.
"""

import pytest
import torch

from conftest import DEVICE, assert_close


def assert_fp8_close(actual, expected):
    torch.testing.assert_close(actual.float().cpu(), expected.float(), atol=0.125, rtol=0)


class TestArange:
    def test_default_integer_dtype_raises(self):
        with pytest.raises(RuntimeError):
            torch.arange(5, device=DEVICE)

    def test_start_end(self):
        result = torch.arange(2, 9, dtype=torch.int8, device=DEVICE)
        expected = torch.arange(2, 9, dtype=torch.int8)
        assert_close(result, expected)

    def test_start_end_step(self):
        result = torch.arange(2, 9, 2, dtype=torch.int8, device=DEVICE)
        expected = torch.arange(2, 9, 2, dtype=torch.int8)
        assert_close(result, expected)

    def test_negative_step(self):
        result = torch.arange(9, 2, -2, dtype=torch.int8, device=DEVICE)
        expected = torch.arange(9, 2, -2, dtype=torch.int8)
        assert_close(result, expected)

    @pytest.mark.parametrize(
        "args",
        [
            (5, 5),
            (5, 5, -1),
            (5, 5, 1),
        ],
    )
    def test_empty_ranges(self, args):
        result = torch.arange(*args, dtype=torch.int8, device=DEVICE)
        expected = torch.arange(*args, dtype=torch.int8)
        assert result.numel() == 0
        assert_close(result, expected)

    @pytest.mark.parametrize(
        "args",
        [
            (5, 1),
            (1, 5, -1),
        ],
    )
    def test_inconsistent_bounds_raise(self, args):
        with pytest.raises(RuntimeError):
            torch.arange(*args, dtype=torch.int8, device=DEVICE)

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.bfloat16, torch.int8, torch.float8_e4m3fn],
    )
    def test_dtypes(self, dtype):
        result = torch.arange(0, 8, dtype=dtype, device=DEVICE)
        assert result.dtype == dtype
        if dtype == torch.float8_e4m3fn:
            expected = torch.arange(0, 8, dtype=torch.float32).to(dtype)
            assert_fp8_close(result, expected)
        else:
            expected = torch.arange(0, 8, dtype=dtype)
            assert_close(result, expected)

    def test_float_step(self):
        result = torch.arange(0, 2, 0.25, dtype=torch.float32, device=DEVICE)
        expected = torch.arange(0, 2, 0.25, dtype=torch.float32)
        assert_close(result, expected)

    def test_bfloat16_float_step(self):
        result = torch.arange(0, 2, 0.5, dtype=torch.bfloat16, device=DEVICE)
        expected = torch.arange(0, 2, 0.5, dtype=torch.bfloat16)
        assert_close(result, expected)

    def test_default_dtype_for_float_bounds(self):
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            result = torch.arange(0.0, 1.0, 0.25, device=DEVICE)
            expected = torch.arange(0.0, 1.0, 0.25)
            assert result.dtype == expected.dtype == torch.float32
            assert_close(result, expected)
        finally:
            torch.set_default_dtype(old_dtype)

    @pytest.mark.parametrize(
        "args",
        [
            (0.0, 1.0, 0.2),
            (0.1, 0.8, 0.2),
            (1.0, -0.1, -0.3),
        ],
    )
    def test_float_boundary_sizes(self, args):
        result = torch.arange(*args, dtype=torch.float32, device=DEVICE)
        expected = torch.arange(*args, dtype=torch.float32)
        assert result.shape == expected.shape
        assert_close(result, expected)

    @pytest.mark.parametrize("dtype", [torch.int8])
    def test_integer_dtype_float_bounds(self, dtype):
        result = torch.arange(1.5, 5.5, 1.5, dtype=dtype, device=DEVICE)
        expected = torch.arange(1.5, 5.5, 1.5, dtype=dtype)
        assert_close(result, expected)

    @pytest.mark.parametrize("dtype", [torch.int8])
    def test_integer_dtype_float_step_casts_to_zero_raises(self, dtype):
        with pytest.raises(RuntimeError):
            torch.arange(1.5, 3.5, 0.5, dtype=dtype, device=DEVICE)

    def test_int8_wrapping(self):
        result = torch.arange(126, 130, dtype=torch.int8, device=DEVICE)
        expected = torch.arange(126, 130, dtype=torch.int8)
        assert torch.equal(result.cpu(), expected)

    def test_bool_bounds_follow_pytorch(self):
        result = torch.arange(True, 5, dtype=torch.int8, device=DEVICE)
        expected = torch.arange(True, 5, dtype=torch.int8)
        assert_close(result, expected)

    def test_out(self):
        out = torch.empty(1, dtype=torch.int8, device=DEVICE)
        ret = torch.arange(0, 10, 2, out=out)
        expected = torch.arange(0, 10, 2, dtype=torch.int8)
        assert ret is out
        assert out.shape == expected.shape
        assert_close(out, expected)

    def test_out_non_contiguous(self):
        base = torch.empty(10, dtype=torch.int8, device=DEVICE)
        out = base[::2]
        ret = torch.arange(0, 10, 2, out=out)
        expected = torch.arange(0, 10, 2, dtype=torch.int8)
        assert ret is out
        assert not out.is_contiguous()
        assert_close(out, expected)

    def test_out_float32(self):
        out = torch.empty(0, dtype=torch.float32, device=DEVICE)
        ret = torch.arange(1.5, 3.5, 0.5, out=out)
        expected = torch.arange(1.5, 3.5, 0.5)
        assert ret is out
        assert out.dtype == torch.float32
        assert_close(out, expected)

    @pytest.mark.parametrize("dtype", [torch.int8])
    def test_out_integer_dtype_float_bounds(self, dtype):
        out = torch.empty(0, dtype=dtype, device=DEVICE)
        ret = torch.arange(1.5, 5.5, 1.5, out=out)
        expected = torch.arange(1.5, 5.5, 1.5, dtype=dtype)
        assert ret is out
        assert out.dtype == dtype
        assert_close(out, expected)

    def test_large(self):
        result = torch.arange(8192, dtype=torch.float32, device=DEVICE)
        expected = torch.arange(8192, dtype=torch.float32)
        assert_close(result, expected)

    def test_step_zero_raises(self):
        with pytest.raises(RuntimeError):
            torch.arange(0, 4, 0, device=DEVICE)

    @pytest.mark.parametrize(
        "dtype",
        [torch.int64, torch.uint8, torch.int16, torch.float64],
    )
    def test_unsupported_dtypes(self, dtype):
        with pytest.raises(RuntimeError):
            torch.arange(0, 4, dtype=dtype, device=DEVICE)

    def test_int32_dtype(self):
        result = torch.arange(0, 8, dtype=torch.int32, device=DEVICE)
        expected = torch.arange(0, 8, dtype=torch.int32)
        assert torch.equal(result.cpu(), expected)

    def test_pin_memory_raises(self):
        with pytest.raises(RuntimeError):
            torch.arange(4, device=DEVICE, pin_memory=True)
