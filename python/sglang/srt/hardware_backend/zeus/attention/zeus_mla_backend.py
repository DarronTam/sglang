"""
Zeus NPU MLA + NSA attention backend.

Two-stage sparse attention on Zeus:
  1. Indexer.forward_zeus produces ``topk_indices`` of shape
     ``[total_tokens, index_topk]`` — absolute kv-cache positions.
  2. ``forward_extend`` / ``forward_decode`` gather the corresponding pages and
     run sparse MLA via ``sgl_kernel_zeus.sparse_mla_paged_zeus``.

Only the ``forward_absorb`` MLA path is supported (latent KV stays in latent
form, ``q_rope`` is passed in separately). The ``forward_normal`` path
(decompressed KV) is not implemented for Zeus — upstream MLA must run
``forward_absorb_prepare`` before reaching this backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.nsa_backend import (
    NSAIndexerMetadata,
    NSAMetadata,
    TopkTransformMethod,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


class ZeusMLABackend(AttentionBackend):
    def __init__(self, model_runner: "ModelRunner", skip_prefill: bool = False):
        super().__init__()
        self.device = model_runner.device
        self.real_page_size = model_runner.page_size
        self.max_context_len = model_runner.model_config.context_len
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads
            // model_runner.tp_size
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.skip_prefill = skip_prefill

        assert model_runner.req_to_token_pool is not None
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.req_to_token_pool = model_runner.req_to_token_pool

        # CUDA-graph (decode-only) preallocated buffers, filled by the
        # capture/replay hooks. Sized in ``init_cuda_graph_state``.
        self._g_cache_seqlens: Optional[torch.Tensor] = None
        self._g_cu_seqlens_q: Optional[torch.Tensor] = None
        self._g_cu_seqlens_k: Optional[torch.Tensor] = None
        self._g_page_table: Optional[torch.Tensor] = None

        self.forward_metadata: Optional[NSAMetadata] = None

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        bs = forward_batch.batch_size
        cache_seqlens = forward_batch.seq_lens.to(torch.int32)
        cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_k[1:] = torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
        assert forward_batch.seq_lens_cpu is not None
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
            cu_seqlens_q = torch.arange(
                bs + 1, dtype=torch.int32, device=self.device
            )
            max_seqlen_q: Literal[1] = 1
        else:
            ext = forward_batch.extend_seq_lens.to(torch.int32)
            cu_seqlens_q = torch.zeros(
                bs + 1, dtype=torch.int32, device=self.device
            )
            cu_seqlens_q[1:] = torch.cumsum(ext, dim=0, dtype=torch.int32)
            assert forward_batch.extend_seq_lens_cpu is not None
            max_seqlen_q = int(max(forward_batch.extend_seq_lens_cpu))

        page_table_1 = self.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]
        real_page_table = self._transform_table_1_to_real(page_table_1)

        self.forward_metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            real_page_table=real_page_table,
            # NSA expanded fields — unused on Zeus single-path (sparse MLA
            # consumes ``topk_indices`` directly), so we feed safe defaults.
            nsa_cache_seqlens_int32=cache_seqlens,
            nsa_cu_seqlens_q=cu_seqlens_q,
            nsa_cu_seqlens_k=cu_seqlens_k,
            nsa_extend_seq_lens_list=[],
            nsa_seqlens_expanded=cache_seqlens,
            nsa_max_seqlen_q=max_seqlen_q if max_seqlen_q == 1 else 1,
        )

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided = torch.arange(
            0, max_seqlen_k, page_size,
            device=page_table.device, dtype=torch.int32,
        )
        return page_table[:, strided] // page_size

    # ------------------------------------------------------------------ #
    # CUDA-graph hooks (decode-only)
    # ------------------------------------------------------------------ #
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self._g_cache_seqlens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._g_cu_seqlens_q = torch.arange(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._g_cu_seqlens_k = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self._g_page_table = torch.zeros(
            (max_bs, self.max_context_len),
            dtype=torch.int32, device=self.device,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"] = None,
    ):
        self._fill_decode_graph_metadata(bs, req_pool_indices, seq_lens)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"] = None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        self._fill_decode_graph_metadata(
            bs, req_pool_indices, seq_lens, seq_lens_cpu
        )

    def _fill_decode_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        assert self._g_cache_seqlens is not None, "init_cuda_graph_state not called"
        cache_seqlens = seq_lens[:bs].to(torch.int32)
        self._g_cache_seqlens[:bs].copy_(cache_seqlens)
        self._g_cu_seqlens_k[: bs + 1].zero_()
        self._g_cu_seqlens_k[1 : bs + 1].copy_(
            torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
        )
        max_seqlen_k = self.max_context_len
        self._g_page_table[:bs, :max_seqlen_k].copy_(
            self.req_to_token[req_pool_indices[:bs], :max_seqlen_k]
        )
        page_table_1 = self._g_page_table[:bs, :max_seqlen_k]
        real_page_table = self._transform_table_1_to_real(page_table_1)

        # max_seq_len_k on host: prefer the scheduler's existing CPU copy to
        # avoid a per-replay device .max() (D->H sync). Falls back to a one-off
        # .cpu() during capture, where seq_lens_cpu is not provided.
        if seq_lens_cpu is not None:
            max_seq_len_k = int(seq_lens_cpu[:bs].max().item())
        else:
            max_seq_len_k = int(seq_lens[:bs].cpu().max().item())

        self.forward_metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=self._g_cache_seqlens[:bs],
            max_seq_len_q=1,
            max_seq_len_k=max_seq_len_k,
            cu_seqlens_q=self._g_cu_seqlens_q[: bs + 1],
            cu_seqlens_k=self._g_cu_seqlens_k[: bs + 1],
            page_table_1=page_table_1,
            real_page_table=real_page_table,
            nsa_cache_seqlens_int32=self._g_cache_seqlens[:bs],
            nsa_cu_seqlens_q=self._g_cu_seqlens_q[: bs + 1],
            nsa_cu_seqlens_k=self._g_cu_seqlens_k[: bs + 1],
            nsa_extend_seq_lens_list=[],
            nsa_seqlens_expanded=self._g_cache_seqlens[:bs],
            nsa_max_seqlen_q=1,
        )

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    def support_triton(self) -> bool:
        return False

    # ------------------------------------------------------------------ #
    # Indexer plumbing
    # ------------------------------------------------------------------ #
    def get_indexer_metadata(
        self, layer_id: int, forward_batch: "ForwardBatch",
    ) -> NSAIndexerMetadata:
        assert self.forward_metadata is not None, "init_forward_metadata not called"
        return NSAIndexerMetadata(
            attn_metadata=self.forward_metadata,
            topk_transform_method=TopkTransformMethod.PAGED,
            paged_mqa_schedule_metadata=None,
            force_unfused_topk=False,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward_sparse_mla(
            q=q,
            k=k,
            k_rope=k_rope,
            q_rope=q_rope,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            topk_indices=topk_indices,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._forward_sparse_mla(
            q=q,
            k=k,
            k_rope=k_rope,
            q_rope=q_rope,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            topk_indices=topk_indices,
        )

    def _forward_sparse_mla(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        k_rope: Optional[torch.Tensor],
        q_rope: Optional[torch.Tensor],
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool,
        topk_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert q_rope is not None, (
            "ZeusMLABackend only supports the forward_absorb MLA path; "
            "q_rope must be provided."
        )
        assert topk_indices is not None, (
            "ZeusMLABackend requires sparse topk_indices from Indexer.forward_zeus"
        )

        # 1) write latent KV (nope + rope halves) into the paged pool.
        if k is not None and save_kv_cache:
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, cache_loc, k, k_rope,
            )

        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        # 2) reshape q halves to (tokens, heads, dim).
        q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        q_rope = q_rope.view(
            -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim,
        )

        # 3) align topk_indices with q (TP + partial DP attention can pad q).
        if topk_indices.shape[0] != q_nope.shape[0]:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        # 4) sparse paged MLA kernel — single Zeus impl.
        md = self.forward_metadata
        assert md is not None
        from sgl_kernel_zeus import sparse_mla_paged_zeus

        return sparse_mla_paged_zeus(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_cache=kv_cache,
            page_table=topk_indices,
            cache_seqlens=md.cache_seqlens_int32,
            cu_seqlens_q=md.cu_seqlens_q,
            cu_seqlens_k=md.cu_seqlens_k,
            max_seqlen_q=md.max_seq_len_q,
            sm_scale=layer.scaling,
            v_head_dim=layer.v_head_dim,
            kv_lora_rank=self.kv_lora_rank,
        )

    @staticmethod
    def _pad_topk_indices(
        topk_indices: torch.Tensor, target_tokens: int,
    ) -> torch.Tensor:
        cur = topk_indices.shape[0]
        if cur == target_tokens:
            return topk_indices
        if cur > target_tokens:
            return topk_indices[:target_tokens]
        pad = torch.full(
            (target_tokens - cur, topk_indices.shape[1]),
            -1,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        return torch.cat([topk_indices, pad], dim=0)
