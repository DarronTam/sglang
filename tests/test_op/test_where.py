"""
Tests for where operators.

Dispatched via TORCH_LIBRARY_IMPL(aten, PrivateUse1)
-> zenl_where_internal -> zenlWhere kernel.
"""

import pytest
import torch

from conftest import DEVICE, assert_close, to_zeus


class TestWhere:
    def test_basic_float32(self):
        cond = to_zeus(torch.tensor([[True, False], [False, True]]))
        self = to_zeus(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        other = to_zeus(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
        result = torch.where(cond, self, other)
        expected = torch.where(cond.cpu(), self.cpu(), other.cpu())
        assert result.device.type == DEVICE
        assert result.dtype == torch.float32
        assert_close(result, expected)

    def test_out(self):
        cond_cpu = torch.tensor([True, False, True, False])
        self_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        other_cpu = torch.tensor([5.0, 6.0, 7.0, 8.0])
        out = torch.empty(1, dtype=torch.float32, device=DEVICE)
        ret = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu), out=out)
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert ret is out
        assert out.shape == expected.shape
        assert_close(out, expected)

    def test_tensor_method(self):
        cond_cpu = torch.tensor([False, True, False])
        self_cpu = torch.tensor([1.0, 2.0, 3.0])
        other_cpu = torch.tensor([4.0, 5.0, 6.0])
        result = to_zeus(self_cpu).where(to_zeus(cond_cpu), to_zeus(other_cpu))
        expected = self_cpu.where(cond_cpu, other_cpu)
        assert_close(result, expected)

    def test_condition_broadcast(self):
        cond_cpu = torch.tensor([[True], [False]])
        self_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        other_cpu = torch.full((2, 3), -1.0)
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert_close(result, expected)

    def test_value_broadcast(self):
        cond_cpu = torch.tensor([[True, False, True], [False, True, False]])
        self_cpu = torch.tensor([[1.0], [2.0]])
        other_cpu = torch.tensor([10.0, 20.0, 30.0])
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert_close(result, expected)

    @pytest.mark.parametrize(
        "self_cpu, other_cpu",
        [
            (
                torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
                torch.tensor([4.0, 5.0, 6.0], dtype=torch.bfloat16),
            ),
            (
                torch.tensor([1, 2, 3], dtype=torch.int8),
                torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32),
            ),
        ],
    )
    def test_promotion(self, self_cpu, other_cpu):
        cond_cpu = torch.tensor([True, False, True])
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert result.dtype == expected.dtype
        assert_close(result, expected)

    @pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16])
    def test_supported_dtypes(self, dtype):
        cond_cpu = torch.tensor([True, False, True, False])
        self_cpu = torch.tensor([1, 2, 3, 4], dtype=dtype)
        other_cpu = torch.tensor([5, 6, 7, 8], dtype=dtype)
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert result.dtype == dtype
        assert_close(result, expected)

    def test_supported_fp8e4m3(self):
        cond_cpu = torch.tensor([True, False, True, False])
        self_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(torch.float8_e4m3fn)
        other_cpu = torch.tensor([5.0, 6.0, 7.0, 8.0]).to(torch.float8_e4m3fn)
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.tensor([1.0, 6.0, 3.0, 8.0])
        assert result.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(result.float().cpu(), expected, atol=0.125, rtol=0)

    def test_zero_dim_tensor(self):
        cond_cpu = torch.tensor([True, False, True])
        self_cpu = torch.tensor(2.5)
        other_cpu = torch.tensor([1.0, 2.0, 3.0])
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert_close(result, expected)

    def test_zero_dim_condition(self):
        cond_cpu = torch.tensor(True)
        self_cpu = torch.tensor([1.0, 2.0, 3.0])
        other_cpu = torch.tensor([4.0, 5.0, 6.0])
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert_close(result, expected)

    def test_empty(self):
        cond_cpu = torch.empty((0, 3), dtype=torch.bool)
        self_cpu = torch.empty((0, 3), dtype=torch.float32)
        other_cpu = torch.empty((0, 3), dtype=torch.float32)
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert result.shape == expected.shape
        assert_close(result, expected)

    def test_empty_broadcast(self):
        cond_cpu = torch.empty((0, 1), dtype=torch.bool)
        self_cpu = torch.empty((0, 3), dtype=torch.float32)
        other_cpu = torch.empty((1, 3), dtype=torch.float32)
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert result.shape == expected.shape == (0, 3)
        assert_close(result, expected)

    def test_non_contiguous_input(self):
        cond_cpu = torch.tensor([[True, False], [False, True], [True, False]]).t()
        self_cpu = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()
        other_cpu = torch.full((3, 2), -1.0).t()
        result = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu))
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert not cond_cpu.is_contiguous()
        assert_close(result, expected)

    def test_non_contiguous_out(self):
        cond_cpu = torch.tensor([[True, False], [False, True], [True, False]])
        self_cpu = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        other_cpu = torch.full((3, 2), -1.0)
        base = torch.empty((2, 3), dtype=torch.float32, device=DEVICE)
        out = base.t()
        ret = torch.where(
            to_zeus(cond_cpu), to_zeus(self_cpu), to_zeus(other_cpu), out=out)
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert ret is out
        assert not out.is_contiguous()
        assert_close(out, expected)

    def test_out_alias_input(self):
        cond_cpu = torch.tensor([False, True, False, True])
        self_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0])
        other_cpu = torch.tensor([5.0, 6.0, 7.0, 8.0])
        self = to_zeus(self_cpu)
        ret = torch.where(to_zeus(cond_cpu), self, to_zeus(other_cpu), out=self)
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert ret is self
        assert_close(self, expected)

    def test_out_partial_overlap_input(self):
        cond_cpu = torch.tensor([False, True, False, True])
        base_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 99.0])
        self_cpu = base_cpu[:-1]
        other_cpu = torch.tensor([5.0, 6.0, 7.0, 8.0])
        base = to_zeus(base_cpu)
        self = base[:-1]
        out = base[1:]
        ret = torch.where(to_zeus(cond_cpu), self, to_zeus(other_cpu), out=out)
        expected = torch.where(cond_cpu, self_cpu, other_cpu)
        assert ret is out
        assert_close(out, expected)

    def test_condition_non_bool_raises(self):
        with pytest.raises(RuntimeError):
            torch.where(
                to_zeus(torch.tensor([1, 0], dtype=torch.int8)),
                to_zeus(torch.tensor([1.0, 2.0])),
                to_zeus(torch.tensor([3.0, 4.0])),
            )

    def test_broadcast_mismatch_raises(self):
        cond = to_zeus(torch.tensor([True, False]))
        self = to_zeus(torch.randn(2, 3))
        other = to_zeus(torch.randn(4, 3))
        with pytest.raises(RuntimeError):
            torch.where(cond, self, other)

    @pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int64])
    def test_unsupported_value_dtypes_raise(self, dtype):
        cond = to_zeus(torch.tensor([True, False]))
        self = to_zeus(torch.tensor([1, 2], dtype=dtype))
        other = to_zeus(torch.tensor([3, 4], dtype=dtype))
        with pytest.raises(RuntimeError):
            torch.where(cond, self, other)

    def test_out_wrong_dtype_raises(self):
        cond = to_zeus(torch.tensor([True, False]))
        self = to_zeus(torch.tensor([1.0, 2.0]))
        other = to_zeus(torch.tensor([3.0, 4.0]))
        out = torch.empty(2, dtype=torch.bfloat16, device=DEVICE)
        with pytest.raises(RuntimeError):
            torch.where(cond, self, other, out=out)

    def test_where_int32(self):
        cond = to_zeus(torch.tensor([True, False, True, False]))
        x = to_zeus(torch.tensor([100, 200, 300, 400], dtype=torch.int32))
        y = to_zeus(torch.tensor([-1, -2, -3, -4], dtype=torch.int32))
        result = torch.where(cond, x, y).cpu()
        expected = torch.where(
            torch.tensor([True, False, True, False]),
            torch.tensor([100, 200, 300, 400], dtype=torch.int32),
            torch.tensor([-1, -2, -3, -4], dtype=torch.int32),
        )
        assert torch.equal(result, expected)
