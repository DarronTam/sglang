"""
Tests for Zeus view / reshape operators.

Dispatched via TORCH_LIBRARY_IMPL(aten, PrivateUse1):
  view, _reshape_alias, as_strided, as_strided_
"""

import torch
from conftest import DEVICE, to_zeus, assert_close


class TestView:
    def test_basic(self):
        x = torch.randn(4, 8).to(DEVICE)
        y = x.view(2, 16)
        assert y.shape == (2, 16)
        assert_close(y, x.cpu().view(2, 16))

    def test_flatten(self):
        x = torch.randn(2, 3, 4).to(DEVICE)
        y = x.view(-1)
        assert y.shape == (24,)

    def test_expand_dim(self):
        x = torch.randn(24).to(DEVICE)
        y = x.view(2, 3, 4)
        assert y.shape == (2, 3, 4)
        assert_close(y, x.cpu().view(2, 3, 4))


class TestReshape:
    def test_basic(self):
        x = torch.randn(4, 8).to(DEVICE)
        y = x.reshape(2, 16)
        assert y.shape == (2, 16)
        assert_close(y, x.cpu().reshape(2, 16))

    def test_infer(self):
        x = torch.randn(2, 3, 4).to(DEVICE)
        y = x.reshape(-1, 4)
        assert y.shape == (6, 4)

    def test_contiguous(self):
        """reshape on contiguous tensor should return a view."""
        x = torch.randn(4, 8).to(DEVICE)
        y = x.reshape(32)
        assert y.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()


class TestAsStrided:
    def test_basic(self):
        x = torch.randn(4, 8).to(DEVICE)
        y = x.as_strided((2, 4), (8, 1))
        assert y.shape == (2, 4)

    def test_overlapping(self):
        """as_strided can create overlapping views."""
        x = torch.arange(10, dtype=torch.float32).to(DEVICE)
        y = x.as_strided((3, 4), (1, 1))
        assert y.shape == (3, 4)
        # y[0] starts at 0, y[1] starts at 1, etc.
        assert_close(y[0], torch.arange(4, dtype=torch.float32))
