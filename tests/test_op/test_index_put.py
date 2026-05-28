"""
Tests for index_put_ / index_put operators.

Dispatched via TORCH_LIBRARY_IMPL(aten, PrivateUse1)
-> zenl_index_put_internal -> zenlIndexPut kernel.

Current limitations (vs MLU CNNL):
  - accumulate=True not supported
  - Bool mask indexing not supported
  - AdvancedIndexing (None dims, subspace transpose) not supported
  - self must be contiguous
"""

import torch
import pytest
from conftest import DEVICE, assert_close


# ============================================================================
# index_put_ (in-place)
# ============================================================================

class TestIndexPutInplace:
    # --- basic 1D indexing ---------------------------------------------------

    def test_1d_basic(self):
        """self[indices] = values on 1D tensor."""
        cpu = torch.zeros(10)
        zeus = cpu.clone().to(DEVICE)
        indices = (torch.tensor([1, 3, 5, 7]),)
        values = torch.tensor([10., 30., 50., 70.])

        cpu.index_put_(indices, values)
        zeus.index_put_([i.to(DEVICE) for i in indices], values.to(DEVICE))
        assert_close(zeus, cpu)

    def test_1d_overwrite(self):
        """Overwrite existing values."""
        cpu = torch.arange(8, dtype=torch.float32)
        zeus = cpu.clone().to(DEVICE)
        indices = (torch.tensor([0, 2, 4]),)
        values = torch.tensor([100., 200., 300.])

        cpu.index_put_(indices, values)
        zeus.index_put_([i.to(DEVICE) for i in indices], values.to(DEVICE))
        assert_close(zeus, cpu)

    def test_1d_single_element(self):
        cpu = torch.zeros(5)
        zeus = cpu.clone().to(DEVICE)
        indices = (torch.tensor([3]),)
        values = torch.tensor([99.])

        cpu.index_put_(indices, values)
        zeus.index_put_([i.to(DEVICE) for i in indices], values.to(DEVICE))
        assert_close(zeus, cpu)

    def test_1d_scalar_value(self):
        """Broadcast a single scalar to all indexed positions."""
        cpu = torch.zeros(6)
        zeus = cpu.clone().to(DEVICE)
        indices = (torch.tensor([0, 2, 4]),)
        values = torch.tensor(7.)

        cpu.index_put_(indices, values)
        zeus.index_put_([i.to(DEVICE) for i in indices], values.to(DEVICE))
        assert_close(zeus, cpu)

    # --- 2D indexing --------------------------------------------------------

    @pytest.mark.xfail(reason="single-dim index selecting whole rows not yet supported (values numel > num_indices)")
    def test_2d_single_dim(self):
        """Index first dimension of a 2D tensor."""
        cpu = torch.zeros(4, 3)
        zeus = cpu.clone().to(DEVICE)
        indices = (torch.tensor([0, 2]),)
        values = torch.tensor([[1., 2., 3.], [7., 8., 9.]])

        cpu.index_put_(indices, values)
        zeus.index_put_([i.to(DEVICE) for i in indices], values.to(DEVICE))
        assert_close(zeus, cpu)

    def test_2d_both_dims(self):
        """Index both dimensions → element-level scatter."""
        cpu = torch.zeros(4, 4)
        zeus = cpu.clone().to(DEVICE)
        row = torch.tensor([0, 1, 2, 3])
        col = torch.tensor([3, 2, 1, 0])
        indices = (row, col)
        values = torch.tensor([10., 20., 30., 40.])

        cpu.index_put_(indices, values)
        zeus.index_put_([i.to(DEVICE) for i in indices], values.to(DEVICE))
        assert_close(zeus, cpu)

    def test_2d_repeated_row(self):
        """Multiple writes to the same row, different columns."""
        cpu = torch.zeros(3, 4)
        zeus = cpu.clone().to(DEVICE)
        row = torch.tensor([0, 0, 0])
        col = torch.tensor([0, 1, 2])
        values = torch.tensor([1., 2., 3.])

        cpu.index_put_((row, col), values)
        zeus.index_put_((row.to(DEVICE), col.to(DEVICE)), values.to(DEVICE))
        assert_close(zeus, cpu)

    # --- dtypes -------------------------------------------------------------

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_dtypes(self, dtype):
        cpu = torch.zeros(8, dtype=dtype)
        zeus = cpu.clone().to(DEVICE)
        indices = (torch.tensor([1, 3, 5]),)
        values = torch.tensor([1., 2., 3.], dtype=dtype)

        cpu.index_put_(indices, values)
        zeus.index_put_([i.to(DEVICE) for i in indices], values.to(DEVICE))
        assert_close(zeus, cpu)

    def test_int_indices(self):
        """int32 indices (not just int64)."""
        cpu = torch.zeros(8)
        zeus = cpu.clone().to(DEVICE)
        indices = (torch.tensor([1, 4, 6], dtype=torch.int32),)
        values = torch.tensor([10., 40., 60.])

        cpu.index_put_(indices, values)
        zeus.index_put_([i.to(DEVICE) for i in indices], values.to(DEVICE))
        assert_close(zeus, cpu)

    # --- returns self -------------------------------------------------------

    def test_returns_self(self):
        """index_put_ must return the same tensor (in-place)."""
        t = torch.zeros(4, device=DEVICE)
        ret = t.index_put_((torch.tensor([0], device=DEVICE),),
                           torch.tensor([1.], device=DEVICE))
        assert ret is t

    # --- large (multicore) --------------------------------------------------

    def test_large(self):
        """Large scatter to exercise the 2-core kernel path."""
        N = 4096
        cpu = torch.zeros(N * 2)
        zeus = cpu.clone().to(DEVICE)
        indices = (torch.arange(0, N * 2, 2),)  # even positions
        values = torch.arange(1, N + 1, dtype=torch.float32)

        cpu.index_put_(indices, values)
        zeus.index_put_([i.to(DEVICE) for i in indices], values.to(DEVICE))
        assert_close(zeus, cpu)


# ============================================================================
# index_put (out-of-place)
# ============================================================================

class TestIndexPut:
    def test_out_of_place(self):
        """index_put (non-inplace) returns a new tensor."""
        src_cpu = torch.zeros(6)
        src_zeus = src_cpu.clone().to(DEVICE)
        indices = (torch.tensor([1, 3]),)
        values = torch.tensor([10., 30.])

        out_cpu = src_cpu.index_put(indices, values)
        out_zeus = src_zeus.index_put(
            [i.to(DEVICE) for i in indices], values.to(DEVICE))

        # Original unchanged
        assert_close(src_zeus, src_cpu)
        # Result matches
        assert_close(out_zeus, out_cpu)

    def test_out_of_place_2d(self):
        src_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        src_zeus = src_cpu.clone().to(DEVICE)
        row = torch.tensor([0, 2])
        col = torch.tensor([1, 3])
        values = torch.tensor([99., 88.])

        out_cpu = src_cpu.index_put((row, col), values)
        out_zeus = src_zeus.index_put(
            (row.to(DEVICE), col.to(DEVICE)), values.to(DEVICE))
        assert_close(out_zeus, out_cpu)


# ============================================================================
# Bracket assignment sugar (t[idx] = val)
# ============================================================================

class TestBracketAssignment:
    def test_bracket_1d(self):
        cpu = torch.zeros(10)
        zeus = cpu.clone().to(DEVICE)
        idx = torch.tensor([2, 5, 8])

        cpu[idx] = 1.0
        zeus[idx.to(DEVICE)] = 1.0
        assert_close(zeus, cpu)

    @pytest.mark.xfail(reason="single-dim index selecting whole rows not yet supported (values numel > num_indices)")
    def test_bracket_2d_row(self):
        cpu = torch.zeros(4, 3)
        zeus = cpu.clone().to(DEVICE)
        idx = torch.tensor([1, 3])

        cpu[idx] = torch.ones(2, 3)
        zeus[idx.to(DEVICE)] = torch.ones(2, 3, device=DEVICE)
        assert_close(zeus, cpu)


# ============================================================================
# Accumulate (expect error)
# ============================================================================

class TestAccumulateUnsupported:
    def test_accumulate_raises(self):
        t = torch.zeros(4, device=DEVICE)
        with pytest.raises(RuntimeError, match="accumulate"):
            t.index_put_(
                (torch.tensor([0, 1], device=DEVICE),),
                torch.tensor([1., 2.], device=DEVICE),
                accumulate=True)
