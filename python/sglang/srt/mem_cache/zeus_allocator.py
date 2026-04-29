"""
Zeus NPU paged allocator.

All internal bookkeeping (free_pages, release_pages) kept on CPU.
Only the final output indices are moved to Zeus device.
This avoids Zeus CPU fallback for allocator arithmetic ops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.utils import get_bool_env_var, get_num_new_pages

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


def _alloc_extend_naive(
    prefix_lens,
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    page_size,
):
    """Pure PyTorch page-aligned extend allocation on CPU."""
    extend_lens = seq_lens - prefix_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens
    num_new_pages = (seq_lens + page_size - 1) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    num_full_new_pages = seq_lens // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    need_page = num_new_pages - num_full_new_pages
    end_new_pages = torch.cumsum(num_new_pages, 0)
    start_new_pages = end_new_pages - num_new_pages
    pos_in_page = torch.arange(page_size, dtype=torch.int32)

    for i in range(len(prefix_lens)):
        num1 = (
            min(
                seq_lens[i],
                (prefix_lens[i] + page_size - 1) // page_size * page_size,
            )
            - prefix_lens[i]
        )
        if num1:
            out_indices[start_pos[i] : start_pos[i] + num1] = (
                last_loc[i] + 1 + pos_in_page[:num1].view(-1)
            )

        num2 = (
            seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size
        ) * page_size
        if num2:
            pages = (
                free_pages[start_new_pages[i] : end_new_pages[i] - need_page[i]]
                * page_size
            )
            out_indices[start_pos[i] + num1 : start_pos[i] + num1 + num2] = (
                pages.view(-1, 1) + pos_in_page.view(1, -1)
            ).view(-1)

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size
        if num3:
            out_indices[end_pos[i] - num3 : end_pos[i]] = (
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)


def _to_cpu_tensor(x):
    """Convert list or tensor to CPU int64 tensor."""
    if isinstance(x, torch.Tensor):
        return x.cpu() if x.device.type != "cpu" else x
    return torch.tensor(x, dtype=torch.int64)


class ZeusPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    """Page-aligned allocator for Zeus NPU.

    All internal state (free_pages, release_pages) is on CPU.
    Output indices are moved to Zeus device on return.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self._zeus_device = device
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        # Move internal state to CPU after base class init
        self.free_pages = self.free_pages.cpu()
        self.release_pages = self.release_pages.cpu()

    def clear(self):
        self.free_pages = torch.arange(0, self.num_pages, dtype=torch.int64)
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64)

    def merge_and_sort_free(self):
        self.free_pages = torch.sort(
            torch.cat([self.free_pages] + self.free_group + [self.release_pages])
        ).values
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64)
        self.is_not_in_free_group = True

    def alloc(self, need_size: int):
        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size)
        ).reshape(-1)
        # Zeus chip rule: no int64 device-side. KV-cache slot index trivially
        # fits in int32 (max_total_num_tokens << 2^31). CPU-side dtype cast
        # before H2D keeps the device-side copy_ on the same-dtype fast path.
        return out_indices.to(torch.int32).to(self._zeus_device)

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu,
        seq_lens: torch.Tensor,
        seq_lens_cpu,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        # All arithmetic on CPU
        pl = _to_cpu_tensor(prefix_lens_cpu)
        sl = _to_cpu_tensor(seq_lens_cpu)
        ll = last_loc.cpu()

        roundup = self.page_size - 1
        num_new_pages = (
            (sl + roundup) // self.page_size - (pl + roundup) // self.page_size
        ).sum().item()

        if self.need_sort and num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        out_indices = torch.empty(extend_num_tokens, dtype=torch.int64)
        _alloc_extend_naive(pl, sl, ll, self.free_pages, out_indices, self.page_size)

        self.free_pages = self.free_pages[num_new_pages:]
        # Zeus int32 collapse — see alloc() for rationale.
        return out_indices.to(torch.int32).to(self._zeus_device)

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu,
        last_loc: torch.Tensor,
    ):
        sl = _to_cpu_tensor(seq_lens_cpu)
        ll = last_loc.cpu()

        num_new_pages = get_num_new_pages(
            seq_lens=sl, page_size=self.page_size, decode=True
        )

        if num_new_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_new_pages > len(self.free_pages):
            return None

        need_new_pages = (sl % self.page_size == 1).int()
        end_new_pages = torch.cumsum(need_new_pages, 0)
        start_new_pages = end_new_pages - need_new_pages

        if num_new_pages == 0:
            out_indices = ll + 1
        else:
            out_indices = (ll + 1) * (1 - need_new_pages) + self.free_pages[
                start_new_pages
            ] * self.page_size * need_new_pages

        self.free_pages = self.free_pages[num_new_pages:]
        # Zeus int32 collapse — see alloc() for rationale.
        return out_indices.to(torch.int32).to(self._zeus_device)

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        free_index_cpu = free_index.cpu()
        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index_cpu // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat(
                    (free_page_indices, self.release_pages)
                )
            else:
                self.free_pages = torch.cat(
                    (free_page_indices, self.free_pages)
                )
        else:
            self.free_group.append(free_index_cpu)
