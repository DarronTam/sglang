"""
Tests for Zeus embedding operator.

Dispatched via TORCH_LIBRARY_IMPL(aten, PrivateUse1) → zenlEmbedding.
"""

import torch
import torch.nn.functional as F
import pytest
from conftest import DEVICE, assert_close


class TestEmbedding:
    def test_basic(self):
        weight = torch.randn(10, 4)
        indices = torch.tensor([1, 2, 4, 5])
        ref = F.embedding(indices, weight)
        out = F.embedding(indices.to(DEVICE), weight.to(DEVICE))
        assert_close(out, ref)

    def test_module(self):
        emb = torch.nn.Embedding(100, 32)
        indices = torch.randint(0, 100, (8,))
        ref = emb(indices)
        out = emb.to(DEVICE)(indices.to(DEVICE))
        assert_close(out, ref)

    def test_2d_indices(self):
        weight = torch.randn(50, 16)
        indices = torch.randint(0, 50, (4, 8))
        ref = F.embedding(indices, weight)
        out = F.embedding(indices.to(DEVICE), weight.to(DEVICE))
        assert_close(out, ref)

    def test_3d_indices(self):
        weight = torch.randn(50, 16)
        indices = torch.randint(0, 50, (2, 4, 8))
        ref = F.embedding(indices, weight)
        out = F.embedding(indices.to(DEVICE), weight.to(DEVICE))
        assert_close(out, ref)

    def test_single_index(self):
        weight = torch.randn(10, 4)
        indices = torch.tensor([0])
        ref = F.embedding(indices, weight)
        out = F.embedding(indices.to(DEVICE), weight.to(DEVICE))
        assert_close(out, ref)

    def test_padding_idx(self):
        emb = torch.nn.Embedding(10, 4, padding_idx=0)
        indices = torch.tensor([0, 1, 2, 0, 3])
        ref = emb(indices)
        out = emb.to(DEVICE)(indices.to(DEVICE))
        assert_close(out, ref)
        assert (out[0].cpu() == 0).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtypes(self, dtype):
        weight = torch.randn(20, 8, dtype=dtype)
        indices = torch.randint(0, 20, (4,))
        ref = F.embedding(indices, weight)
        out = F.embedding(indices.to(DEVICE), weight.to(DEVICE))
        assert_close(out, ref)

    def test_large_vocab(self):
        """LLM-scale: 32K tokens, seq_len=256."""
        weight = torch.randn(32000, 128)
        indices = torch.randint(0, 32000, (1, 256))
        ref = F.embedding(indices, weight)
        out = F.embedding(indices.to(DEVICE), weight.to(DEVICE))
        assert_close(out, ref)
