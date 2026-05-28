"""
Tests for Zeus tensor creation operators.

Dispatched via TORCH_LIBRARY_IMPL(aten, PrivateUse1):
  empty.memory_format, empty_strided, zeros, ones, full
"""

import torch
import pytest
from conftest import DEVICE


class TestZeros:
    def test_basic(self):
        t = torch.zeros(4, 8, device=DEVICE)
        assert t.shape == (4, 8)
        assert (t.cpu() == 0).all()

    def test_1d(self):
        t = torch.zeros(16, device=DEVICE)
        assert t.shape == (16,)
        assert (t.cpu() == 0).all()

    def test_dtype(self):
        t = torch.zeros(4, device=DEVICE, dtype=torch.bfloat16)
        assert t.dtype == torch.bfloat16
        assert (t.cpu() == 0).all()


class TestOnes:
    def test_basic(self):
        t = torch.ones(4, 8, device=DEVICE)
        assert t.shape == (4, 8)
        assert (t.cpu() == 1).all()

    def test_1d(self):
        t = torch.ones(16, device=DEVICE)
        assert (t.cpu() == 1).all()

    def test_dtype(self):
        t = torch.ones(4, device=DEVICE, dtype=torch.bfloat16)
        assert t.dtype == torch.bfloat16
        assert (t.cpu() == 1).all()


class TestFull:
    def test_basic(self):
        t = torch.full((3, 5), 3.14, device=DEVICE)
        assert t.shape == (3, 5)
        assert torch.allclose(t.cpu(), torch.full((3, 5), 3.14))

    def test_int8(self):
        t = torch.full((4,), 42, device=DEVICE, dtype=torch.int8)
        assert (t.cpu() == 42).all()


class TestEmpty:
    def test_shape(self):
        t = torch.empty(4, 8, device=DEVICE)
        assert t.shape == (4, 8)
        assert t.device.type == "zeus"

    def test_strided(self):
        t = torch.empty_strided((4, 8), (8, 1), device=DEVICE)
        assert t.shape == (4, 8)
        assert t.stride() == (8, 1)

    @pytest.mark.parametrize("dtype", [
        torch.float32, torch.bfloat16, torch.float16, torch.int32, torch.int8,
    ])
    def test_dtypes(self, dtype):
        t = torch.empty(4, device=DEVICE, dtype=dtype)
        assert t.dtype == dtype
