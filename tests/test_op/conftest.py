"""Shared fixtures and helpers for Zeus operator tests."""

import torch
import torch_zeus as _torch_zeus  # noqa: F401 — registers the backend
_ = _torch_zeus  # suppress "not accessed" warning
import pytest


DEVICE = "zeus"


@pytest.fixture(autouse=True)
def skip_if_unavailable():
    """Skip all tests in test_op if Zeus device is not available."""
    if not torch.zeus.is_available():
        pytest.skip("Zeus device not available")


def to_zeus(t: torch.Tensor) -> torch.Tensor:
    """Move a CPU tensor to Zeus device."""
    return t.to(DEVICE)


def assert_close(actual_zeus, expected_cpu, **kwargs):
    """Compare Zeus result (on CPU) with expected CPU tensor."""
    torch.testing.assert_close(actual_zeus.cpu(), expected_cpu, **kwargs)
