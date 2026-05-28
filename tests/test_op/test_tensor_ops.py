"""
Tests for Zeus tensor manipulation operators.

Dispatched via TORCH_LIBRARY_IMPL(aten, PrivateUse1):
  fill_.Scalar, zero_, copy_, _to_copy, clone, cat
"""

import torch
import pytest
from conftest import DEVICE, to_zeus, assert_close


# ============================================================================
# fill_ / zero_
# ============================================================================

class TestFill:
    def test_fill_scalar(self):
        t = torch.empty(4, 8, device=DEVICE)
        t.fill_(3.14)
        assert torch.allclose(t.cpu(), torch.full((4, 8), 3.14))

    def test_fill_int8(self):
        t = torch.empty(8, device=DEVICE, dtype=torch.int8)
        t.fill_(7)
        assert (t.cpu() == 7).all()

    def test_zero(self):
        t = torch.randn(4, 8).to(DEVICE)
        t.zero_()
        assert (t.cpu() == 0).all()


# ============================================================================
# copy_ / _to_copy
# ============================================================================

class TestCopy:
    def test_copy_zeus_to_zeus(self):
        src = torch.randn(4, 8).to(DEVICE)
        dst = torch.empty(4, 8, device=DEVICE)
        dst.copy_(src)
        assert_close(dst, src.cpu())

    def test_copy_cpu_to_zeus(self):
        src = torch.randn(4, 8)
        dst = torch.empty(4, 8, device=DEVICE)
        dst.copy_(src)
        assert_close(dst, src)

    def test_copy_zeus_to_cpu(self):
        src_cpu = torch.randn(4, 8)
        src = src_cpu.to(DEVICE)
        dst = torch.empty(4, 8)
        dst.copy_(src)
        torch.testing.assert_close(dst, src_cpu)

    def test_to_device(self):
        """tensor.to(device) uses _to_copy internally."""
        x = torch.randn(4, 8)
        xz = x.to(DEVICE)
        assert_close(xz, x)

    def test_to_cpu(self):
        x = torch.randn(4, 8)
        xz = x.to(DEVICE)
        xc = xz.to("cpu")
        torch.testing.assert_close(xc, x)

    def test_to_dtype(self):
        """dtype conversion via _to_copy."""
        x = torch.randn(4, 8, device=DEVICE)
        xbf = x.to(torch.bfloat16)
        assert xbf.dtype == torch.bfloat16
        xf = xbf.to(torch.float32)
        assert xf.dtype == torch.float32


# ============================================================================
# clone
# ============================================================================

class TestClone:
    def test_basic(self):
        x = torch.randn(4, 8).to(DEVICE)
        y = x.clone()
        assert_close(y, x.cpu())
        # Verify it's a different allocation
        assert y.data_ptr() != x.data_ptr()

    def test_preserves_dtype(self):
        x = torch.randn(4, device=DEVICE, dtype=torch.bfloat16)
        y = x.clone()
        assert y.dtype == torch.bfloat16
        assert_close(y, x.cpu())


# ============================================================================
# cat
# ============================================================================

class TestCat:
    def test_1d(self):
        a, b = torch.randn(4), torch.randn(6)
        ref = torch.cat([a, b])
        out = torch.cat([to_zeus(a), to_zeus(b)])
        assert_close(out, ref)

    def test_2d_dim0(self):
        a, b = torch.randn(2, 4), torch.randn(3, 4)
        ref = torch.cat([a, b], dim=0)
        out = torch.cat([to_zeus(a), to_zeus(b)], dim=0)
        assert_close(out, ref)

    def test_2d_dim1(self):
        a, b = torch.randn(4, 2), torch.randn(4, 3)
        ref = torch.cat([a, b], dim=1)
        out = torch.cat([to_zeus(a), to_zeus(b)], dim=1)
        assert_close(out, ref)

    def test_three_tensors(self):
        ts = [torch.randn(4, 8) for _ in range(3)]
        ref = torch.cat(ts)
        out = torch.cat([to_zeus(t) for t in ts])
        assert_close(out, ref)
