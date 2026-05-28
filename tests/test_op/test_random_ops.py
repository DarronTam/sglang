"""
Tests for Zeus random operators.

Dispatched via TORCH_LIBRARY_IMPL(aten, PrivateUse1):
  random_, random_.to, random_.from, uniform_, normal_,
  bernoulli_.float, bernoulli_.Tensor, exponential_
"""

import torch
import pytest
from conftest import DEVICE


class TestUniform:
    def test_range(self):
        t = torch.empty(1000, device=DEVICE)
        t.uniform_(0.0, 1.0)
        tc = t.cpu()
        assert tc.min() >= 0.0
        assert tc.max() <= 1.0

    def test_custom_range(self):
        t = torch.empty(1000, device=DEVICE)
        t.uniform_(-5.0, 5.0)
        tc = t.cpu()
        assert tc.min() >= -5.0
        assert tc.max() <= 5.0

    def test_mean_approx(self):
        """Mean of uniform[0,1] should be ~0.5."""
        t = torch.empty(10000, device=DEVICE)
        t.uniform_(0.0, 1.0)
        assert abs(t.cpu().mean().item() - 0.5) < 0.05


class TestNormal:
    def test_basic(self):
        t = torch.empty(10000, device=DEVICE)
        t.normal_(0.0, 1.0)
        tc = t.cpu()
        assert abs(tc.mean().item()) < 0.1
        assert abs(tc.std().item() - 1.0) < 0.1

    def test_custom_params(self):
        t = torch.empty(10000, device=DEVICE)
        t.normal_(5.0, 2.0)
        tc = t.cpu()
        assert abs(tc.mean().item() - 5.0) < 0.2
        assert abs(tc.std().item() - 2.0) < 0.2


class TestRandom:
    def test_range(self):
        t = torch.empty(1000, device=DEVICE, dtype=torch.int64)
        t.random_(0, 10)
        tc = t.cpu()
        assert tc.min() >= 0
        assert tc.max() < 10

    def test_full_range(self):
        t = torch.empty(1000, device=DEVICE, dtype=torch.float32)
        t.random_()
        # Should produce non-zero values
        assert t.cpu().abs().sum() > 0


class TestBernoulli:
    def test_half(self):
        t = torch.empty(10000, device=DEVICE)
        t.bernoulli_(0.5)
        tc = t.cpu()
        assert set(tc.unique().tolist()).issubset({0.0, 1.0})
        assert abs(tc.mean().item() - 0.5) < 0.05

    def test_zero(self):
        t = torch.empty(100, device=DEVICE)
        t.bernoulli_(0.0)
        assert (t.cpu() == 0).all()

    def test_one(self):
        t = torch.empty(100, device=DEVICE)
        t.bernoulli_(1.0)
        assert (t.cpu() == 1).all()

    def test_tensor_p(self):
        p = torch.full((1000,), 0.3, device=DEVICE)
        t = torch.empty(1000, device=DEVICE)
        t.bernoulli_(p)
        tc = t.cpu()
        assert set(tc.unique().tolist()).issubset({0.0, 1.0})
        assert abs(tc.mean().item() - 0.3) < 0.05


class TestExponential:
    def test_basic(self):
        t = torch.empty(10000, device=DEVICE)
        t.exponential_(1.0)
        tc = t.cpu()
        assert tc.min() >= 0.0
        # Mean of exponential(lamda=1) = 1.0
        assert abs(tc.mean().item() - 1.0) < 0.1
