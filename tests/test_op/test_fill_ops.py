"""
Tests for fill_.Scalar and zero_ operators.

Dispatched via TORCH_LIBRARY_IMPL(aten, PrivateUse1)
→ zenl_fill_internal → zenlFill kernel.

All values (including 0) go through the ZENL fill kernel.
"""

import torch
import pytest
from conftest import DEVICE, to_zeus, assert_close


# ============================================================================
# Tensor.fill_
# ============================================================================

class TestFillScalar:
    # --- basic dtypes -------------------------------------------------------

    def test_fill_f32_basic(self):
        t = torch.empty(4, device=DEVICE)
        t.fill_(3.14)
        expected = torch.full((4,), 3.14)
        assert_close(t, expected)

    def test_fill_f32_zero(self):
        t = torch.ones(8, device=DEVICE)
        t.fill_(0.0)
        assert_close(t, torch.zeros(8))

    def test_fill_f32_one(self):
        t = torch.zeros(8, device=DEVICE)
        t.fill_(1.0)
        assert_close(t, torch.ones(8))

    def test_fill_f32_negative(self):
        t = torch.empty(6, device=DEVICE)
        t.fill_(-1.5)
        assert_close(t, torch.full((6,), -1.5))

    def test_fill_bfloat16(self):
        t = torch.empty(16, dtype=torch.bfloat16, device=DEVICE)
        t.fill_(2.0)
        expected = torch.full((16,), 2.0, dtype=torch.bfloat16)
        assert_close(t, expected)

    def test_fill_bfloat16_zero(self):
        t = torch.ones(16, dtype=torch.bfloat16, device=DEVICE)
        t.fill_(0.0)
        assert_close(t, torch.zeros(16, dtype=torch.bfloat16))

    def test_fill_int8(self):
        t = torch.empty(8, dtype=torch.int8, device=DEVICE)
        t.fill_(42)
        expected = torch.full((8,), 42, dtype=torch.int8)
        assert_close(t, expected)

    def test_fill_int8_zero(self):
        t = torch.ones(8, dtype=torch.int8, device=DEVICE)
        t.fill_(0)
        assert_close(t, torch.zeros(8, dtype=torch.int8))

    def test_fill_int8_negative(self):
        t = torch.empty(4, dtype=torch.int8, device=DEVICE)
        t.fill_(-10)
        expected = torch.full((4,), -10, dtype=torch.int8)
        assert_close(t, expected)

    def test_fill_fp8e4m3(self):
        """fp8e4m3 works via TORCH_LIBRARY_IMPL (bypasses TensorIterator)."""
        t = torch.empty(8, dtype=torch.float8_e4m3fn, device=DEVICE)
        t.fill_(2.0)
        result = t.float().cpu()
        expected = torch.full((8,), 2.0)
        torch.testing.assert_close(result, expected, atol=0.125, rtol=0)

    def test_fill_fp8e4m3_zero(self):
        t = torch.ones(8, device=DEVICE).to(torch.float8_e4m3fn)
        t.fill_(0.0)
        result = t.float().cpu()
        assert_close(result, torch.zeros(8))

    def test_fill_fp8e4m3_negative(self):
        t = torch.empty(4, dtype=torch.float8_e4m3fn, device=DEVICE)
        t.fill_(-1.5)
        result = t.float().cpu()
        expected = torch.full((4,), -1.5)
        torch.testing.assert_close(result, expected, atol=0.125, rtol=0)

    # --- shapes -------------------------------------------------------------

    def test_fill_2d(self):
        t = torch.empty(8, 16, device=DEVICE)
        t.fill_(5.0)
        assert_close(t, torch.full((8, 16), 5.0))

    def test_fill_3d(self):
        t = torch.empty(4, 8, 16, device=DEVICE)
        t.fill_(7.0)
        assert_close(t, torch.full((4, 8, 16), 7.0))

    def test_fill_scalar_tensor(self):
        """Single-element tensor."""
        t = torch.empty(1, device=DEVICE)
        t.fill_(99.0)
        assert_close(t, torch.tensor([99.0]))

    def test_fill_empty_tensor(self):
        """numel==0: no-op, should not crash."""
        t = torch.empty(0, device=DEVICE)
        t.fill_(1.0)
        assert t.numel() == 0

    # --- large (multicore path) --------------------------------------------

    def test_fill_large(self):
        """numel >= 4096 triggers the 2-core kernel path."""
        t = torch.empty(8192, device=DEVICE)
        t.fill_(6.28)
        assert_close(t, torch.full((8192,), 6.28))

    def test_fill_large_zero(self):
        t = torch.ones(8192, device=DEVICE)
        t.fill_(0.0)
        assert_close(t, torch.zeros(8192))

    # --- return value -------------------------------------------------------

    def test_fill_returns_self(self):
        """fill_ must return the same tensor (in-place)."""
        t = torch.empty(4, device=DEVICE)
        ret = t.fill_(1.0)
        assert ret is t


# ============================================================================
# Tensor.zero_  (delegates to fill_(0))
# ============================================================================

class TestZero:
    def test_zero_f32(self):
        t = torch.ones(16, device=DEVICE)
        t.zero_()
        assert_close(t, torch.zeros(16))

    def test_zero_bfloat16(self):
        t = torch.ones(16, dtype=torch.bfloat16, device=DEVICE)
        t.zero_()
        assert_close(t, torch.zeros(16, dtype=torch.bfloat16))

    def test_zero_int8(self):
        t = torch.full((8,), 5, dtype=torch.int8, device=DEVICE)
        t.zero_()
        assert_close(t, torch.zeros(8, dtype=torch.int8))

    def test_zero_fp8e4m3(self):
        t = torch.empty(8, dtype=torch.float8_e4m3fn, device=DEVICE)
        t.fill_(1.0)
        t.zero_()
        result = t.float().cpu()
        assert_close(result, torch.zeros(8))

    def test_zero_2d(self):
        t = torch.ones(16, 32, device=DEVICE)
        t.zero_()
        assert_close(t, torch.zeros(16, 32))

    def test_zero_large(self):
        t = torch.ones(8192, device=DEVICE)
        t.zero_()
        assert_close(t, torch.zeros(8192))

    def test_zero_returns_self(self):
        t = torch.ones(4, device=DEVICE)
        ret = t.zero_()
        assert ret is t


# ============================================================================
# torch.zeros / torch.ones / torch.full  (factory ops that call fill_)
# ============================================================================

class TestFactoryOps:
    def test_zeros(self):
        t = torch.zeros(16, device=DEVICE)
        assert_close(t, torch.zeros(16))

    def test_ones(self):
        t = torch.ones(16, device=DEVICE)
        assert_close(t, torch.ones(16))

    def test_full(self):
        t = torch.full((8,), 3.14, device=DEVICE)
        assert_close(t, torch.full((8,), 3.14))

    def test_full_2d(self):
        t = torch.full((4, 8), -1.0, device=DEVICE)
        assert_close(t, torch.full((4, 8), -1.0))

    def test_full_bfloat16(self):
        t = torch.full((16,), 2.0, dtype=torch.bfloat16, device=DEVICE)
        expected = torch.full((16,), 2.0, dtype=torch.bfloat16)
        assert_close(t, expected)
