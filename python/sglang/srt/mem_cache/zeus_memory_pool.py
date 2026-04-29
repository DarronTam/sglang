"""
Zeus NPU paged KV cache pool.

Uses sgl-kernel-zeus store_kv_cache to write KV data in Zeus tiled memory layout
(K: block-tiled row-major, V: 16-byte column-group interleaved).

Buffer shape: [num_pages, num_kv_heads, page_size, head_dim] per layer
(unlike MHA's flat [max_tokens, num_kv_heads, head_dim]).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


# ---------------------------------------------------------------------------
# Stable per-(layer, role, shape) contiguous buffers for set_kv_buffer.
#
# qkv_proj produces a single output tensor that is split with `torch.split`
# along the last dim into (q, k, v); the resulting k / v are *strided views*
# (non-contiguous along the last dim). store_kv_cache requires contiguous
# input, so we used to call `.contiguous()` inside set_kv_buffer — but under
# graph capture that allocation lands in the graph private pool, gets baked
# into the captured kernel's args, then released back to the pool when
# set_kv_buffer returns. The next captured kernel reuses that memory and
# overwrites the K/V data; on replay the captured store_kv_cache reads the
# overwritten transient and writes garbage into the persistent KV cache —
# attention then always sees its capture-time (zeroed/warmup) state, decode
# logits become stable, and output collapses to a single repeated token.
#
# Fix mirrors the rotary / embedding stable-buffer pattern: cache one
# process-lifetime contiguous buffer per (layer_id, "k"/"v", tuple(shape)),
# and copy the strided source into it. The captured store_kv_cache kernel
# reads from the stable address; replay updates that buffer in-place via
# the captured copy_, so the K/V data flowing into KV cache is always fresh.
# ---------------------------------------------------------------------------
_KV_CONTIG_BUFS: dict = {}


def _stable_contiguous(src: torch.Tensor, key_tag) -> torch.Tensor:
    """Return a process-lifetime contiguous mirror of ``src``.

    No-op fast path when ``src`` is already contiguous (avoids the wasted
    copy and keeps eager mode at zero overhead). When a copy is needed, the
    destination buffer is reused across calls keyed by
    ``(layer_id, "k"/"v", shape, dtype, device)`` so its data_ptr is stable
    across capture and replay. The buffer is refreshed via stride-aware
    `add_` (captured into the graph through TensorIteratorBridge dispatch),
    NOT via `copy_` — Zeus's `copy_(non_contig_src)` falls into a CPU-bounce
    path that is not recorded into the capture, so a copy_-based mirror
    would hold capture-time stale K/V data at replay (decode logits collapse,
    output sticks on a single repeated token).
    """
    if src.is_contiguous():
        return src
    cache_key = (key_tag, tuple(src.shape), src.dtype, src.device)
    buf = _KV_CONTIG_BUFS.get(cache_key)
    if buf is None:
        # First-time alloc lives outside graph capture (lazy-init at first
        # forward). Use zeros so the first add_ produces correct content.
        buf = torch.zeros(src.shape, dtype=src.dtype, device=src.device)
        _KV_CONTIG_BUFS[cache_key] = buf
    # `buf.zero_(); buf.add_(src)` — both go through Zeus's TensorIterator
    # dispatch and are captured into the graph, so replay refreshes buf from
    # the *current* strided src content rather than reading capture-time stale
    # data. zero_ on bf16 is a normal kernel (the int/bool fill_ workaround
    # does not apply to bf16).
    buf.zero_()
    buf.add_(src, alpha=1.0)
    return buf


class ZeusTokenToKVPool(MHATokenToKVPool):
    """KV cache pool for Zeus NPU with paged tiled memory layout.

    Data is written via sgl-kernel-zeus store_kv_cache which handles
    the Zeus-specific tiled memory layout. The buffers should NOT be
    accessed through PyTorch indexing — only via sgl-kernel-zeus kernels.
    """

    def _create_buffers(self):
        total_tokens = self.size + self.page_size  # Match MHA padding
        num_pages = (total_tokens + self.page_size - 1) // self.page_size
        self._num_pages = num_pages

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [num_pages, num_kv_heads, page_size, head_dim] per layer
            self.k_buffer = [
                torch.zeros(
                    (num_pages, self.head_num, self.page_size, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (num_pages, self.head_num, self.page_size, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]

        # data_ptrs / data_strides for compatibility with base class logging.
        # Build on CPU to avoid Zeus ATen fallback for tensor/cat, then move.
        k_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer], dtype=torch.uint64
        )
        v_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_buffer], dtype=torch.uint64
        )
        self.k_data_ptrs = k_ptrs.to(self.device)
        self.v_data_ptrs = v_ptrs.to(self.device)
        self.data_ptrs = torch.cat([k_ptrs, v_ptrs], dim=0).to(self.device)
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
        ).to(self.device)

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        from sgl_kernel_zeus import store_kv_cache

        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        cache_k = _stable_contiguous(cache_k, (layer_id, "k"))
        cache_v = _stable_contiguous(cache_v, (layer_id, "v"))

        store_kv_cache(
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
            loc,
            cache_k,
            cache_v,
            self.page_size,
        )

    def _get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)
