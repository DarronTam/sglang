"""
Zeus NPU attention backend.

Uses sgl-kernel-zeus extend_attention / decode_attention kernels with
paged KV cache in Zeus tiled memory layout.

Triton is not supported (Zeus tensors are not accessible from Triton kernels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class ZeusAttnMetadata:
    """Pre-computed metadata for Zeus attention kernels."""

    kv_indptr: torch.Tensor  # [batch_size + 1] int32 — CSR offsets into kv_indices
    kv_indices: torch.Tensor  # [total_kv_tokens] int32 — absolute token positions
    # Extend-only fields:
    qo_indptr: Optional[torch.Tensor] = None  # [batch_size + 1] int32
    prefix_lens: Optional[torch.Tensor] = None  # [batch_size] int32


class ZeusAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata: Optional[ZeusAttnMetadata] = None
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.max_context_len = model_runner.model_config.context_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.req_to_token_pool = model_runner.req_to_token_pool

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Pre-allocate fixed-size tensors for graph capture/replay."""
        self.cuda_graph_kv_indptr = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        max_total_kv = max_bs * self.max_context_len
        self.cuda_graph_kv_indices = torch.zeros(
            max_total_kv, dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info=None,
    ):
        """Fill metadata for graph capture (decode only)."""
        self._fill_decode_metadata_for_graph(bs, req_pool_indices, seq_lens)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info=None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        """Update metadata in-place for graph replay (decode only)."""
        self._fill_decode_metadata_for_graph(bs, req_pool_indices, seq_lens)

    def _fill_decode_metadata_for_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """Build CSR kv_indptr / kv_indices and set forward_metadata.

        This runs *outside* graph capture (before capture or before replay).
        Prefix sums and the ragged req_to_token gather run on Zeus when the
        build_kv_indices op is available.
        """
        # kv_indptr — compute on Zeus now that torch_zeus supports cumsum.
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        kv_indptr[1:] = torch.cumsum(seq_lens[:bs], dim=0, dtype=torch.int32)
        self.cuda_graph_kv_indptr[: bs + 1].copy_(kv_indptr)

        # kv_indices — prefer Zeus device ragged gather, with CPU mirror
        # fallback for environments that have not rebuilt sgl-kernel-zeus yet.
        try:
            self._build_kv_indices_device(
                req_pool_indices[:bs],
                seq_lens[:bs],
                kv_indptr,
                self.cuda_graph_kv_indices,
            )
        except (ImportError, RuntimeError, AttributeError):
            kv_indices_cpu = self._build_kv_indices_cpu(
                req_pool_indices[:bs],
                seq_lens[:bs],
            )
            total_kv = kv_indices_cpu.numel()
            if total_kv > 0:
                self.cuda_graph_kv_indices[:total_kv].copy_(kv_indices_cpu)

        # Use the full pre-allocated kv_indices buffer (not a slice).
        # Graph capture records tensor shape; the kernel uses kv_indptr to
        # determine valid access ranges, so the full buffer is safe and
        # consistent across different batch sizes and replay seq_lens.
        self.forward_metadata = ZeusAttnMetadata(
            kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
            kv_indices=self.cuda_graph_kv_indices,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Build kv_indptr, kv_indices (and qo_indptr for extend) from forward_batch.

        Prefix sums and req_to_token -> kv_indices ragged gather run on Zeus
        when sgl_kernel_zeus.build_kv_indices is available. A CPU mirror
        fallback is kept for partially rebuilt development environments.
        """
        seq_lens = forward_batch.seq_lens
        batch_size = seq_lens.shape[0]
        req_pool_indices = forward_batch.req_pool_indices

        # kv_indptr: [batch_size + 1], CSR prefix sum of seq_lens.
        kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)

        # kv_indices: concatenated absolute token positions for all sequences.
        try:
            kv_indices = torch.empty(
                forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
            )
            self._build_kv_indices_device(
                req_pool_indices,
                seq_lens,
                kv_indptr,
                kv_indices,
            )
        except (ImportError, RuntimeError, AttributeError):
            kv_indices = self._build_kv_indices_cpu(req_pool_indices, seq_lens).to(
                self.device
            )

        # Extend-specific metadata
        qo_indptr = None
        prefix_lens = None
        if forward_batch.extend_seq_lens is not None:
            qo_indptr = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            qo_indptr[1:] = torch.cumsum(
                forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
            )
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)

        # Move final tensors to Zeus device
        self.forward_metadata = ZeusAttnMetadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=qo_indptr,
            prefix_lens=prefix_lens,
        )

    def _build_kv_indices_device(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
    ):
        from sgl_kernel_zeus import build_kv_indices

        build_kv_indices(
            self.req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            kv_indices,
        )

    def _build_kv_indices_cpu(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        seq_lens_cpu = seq_lens.cpu()
        req_pool_indices_cpu = req_pool_indices.cpu()
        req_to_token_cpu = self.req_to_token_pool.req_to_token_cpu

        total_kv = int(seq_lens_cpu.sum().item())
        kv_indices = torch.empty(total_kv, dtype=torch.int32)
        offset = 0
        for b in range(seq_lens_cpu.shape[0]):
            sl = int(seq_lens_cpu[b].item())
            if sl > 0:
                req_idx = int(req_pool_indices_cpu[b].item())
                kv_indices[offset : offset + sl] = req_to_token_cpu[
                    req_idx, :sl
                ].to(torch.int32)
                offset += sl
        return kv_indices

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        from sgl_kernel_zeus import extend_attention

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        md = self.forward_metadata
        extend_attention(
            q_,
            o_,
            k_cache,
            v_cache,
            md.qo_indptr,
            md.kv_indptr,
            md.kv_indices,
            md.prefix_lens,
            num_q_heads=layer.tp_q_head_num,
            num_kv_heads=layer.tp_k_head_num,
            head_dim=layer.qk_head_dim,
            page_size=self.page_size,
            sm_scale=layer.scaling,
            is_causal=causal,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        from sgl_kernel_zeus import decode_attention

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        md = self.forward_metadata
        decode_attention(
            q_,
            o_,
            k_cache,
            v_cache,
            md.kv_indptr,
            md.kv_indices,
            num_q_heads=layer.tp_q_head_num,
            num_kv_heads=layer.tp_k_head_num,
            head_dim=layer.qk_head_dim,
            page_size=self.page_size,
            sm_scale=layer.scaling,
        )
        return o

    def support_triton(self):
        return False
