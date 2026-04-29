from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
    compute_local_num_token_non_padded,
)


def _requires_zeus_fill_workaround(
    *, device: torch.device | str | None, dtype: torch.dtype
) -> bool:
    if device is None:
        return False
    device_type = torch.device(device).type
    return device_type == "zeus" and dtype in (
        torch.bool,
        torch.int16,
        torch.int32,
        torch.int64,
    )


def create_filled_tensor(
    shape,
    fill_value: int | bool,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Allocate a filled tensor while avoiding Zeus int fill_ limitations."""
    normalized_shape = (shape,) if isinstance(shape, int) else shape

    if not _requires_zeus_fill_workaround(device=device, dtype=dtype):
        with torch.device(device):
            return torch.full(normalized_shape, fill_value, dtype=dtype)

    with torch.device(device):
        tensor = torch.zeros(normalized_shape, dtype=dtype)

    if fill_value != 0:
        tensor.copy_(
            torch.full(normalized_shape, fill_value, dtype=dtype, device="cpu")
        )

    return tensor


def fill_tensor_(tensor: torch.Tensor, fill_value: int | bool) -> torch.Tensor:
    """In-place fill that works around Zeus int fill_ limitations."""
    if not _requires_zeus_fill_workaround(device=tensor.device, dtype=tensor.dtype):
        return tensor.fill_(fill_value)

    if fill_value == 0:
        src = torch.zeros(tensor.shape, dtype=tensor.dtype, device="cpu")
    else:
        src = torch.full(tensor.shape, fill_value, dtype=tensor.dtype, device="cpu")

    return tensor.copy_(src)


@dataclass
class GraphInputBuffers:
    input_ids: torch.Tensor
    input_embeds: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    out_cache_loc: torch.Tensor
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    num_token_non_padded: torch.Tensor
    custom_mask: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    global_num_tokens_gpu: torch.Tensor
    global_num_tokens_for_logprob_gpu: torch.Tensor
    encoder_lens: Optional[torch.Tensor]
    pp_proxy_tensors: Optional[Dict[str, torch.Tensor]]

    @classmethod
    def create(
        cls,
        *,
        device: torch.device,
        max_bs: int,
        max_num_token: int,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        dp_size: int,
        pp_size: int,
        is_encoder_decoder: bool,
        require_mlp_tp_gather: bool,
        seq_len_fill_value: int,
        encoder_len_fill_value: int,
        num_tokens_per_bs: int,
        cache_loc_dtype: torch.dtype,
    ) -> "GraphInputBuffers":
        # Zeus chip cannot operate on int64 / fp64 device-side. The
        # graph-mode rotary/embedding bindings rely on int32 indices on the
        # fast path; an int64→int32 `copy_` inside graph capture goes through
        # synchronous CPU bounce on Zeus (TensorOps.cpp:copy_) which is *not*
        # recorded into the graph, so replay would read stale capture-time
        # values and decode silently produces garbage. Allocate index buffers
        # as int32 here so the per-replay populate happens *outside* the
        # captured graph and the bindings see int32 with no cast.
        indices_dtype = (
            torch.int32 if torch.device(device).type == "zeus" else torch.int64
        )

        with torch.device(device):
            input_ids = torch.zeros((max_num_token,), dtype=indices_dtype)
            input_embeds = torch.zeros((max_num_token, hidden_size), dtype=dtype)
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int32)
            seq_lens = create_filled_tensor(
                (max_bs,),
                seq_len_fill_value,
                dtype=torch.int32,
                device=device,
            )
            out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
            positions = torch.zeros((max_num_token,), dtype=indices_dtype)
            mrope_positions = torch.zeros((3, max_num_token), dtype=indices_dtype)
            num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            custom_mask = create_filled_tensor(
                (max_bs * seq_len_fill_value + max_num_token) * num_tokens_per_bs,
                True,
                dtype=torch.bool,
                device=device,
            )
            next_token_logits_buffer = torch.zeros(
                (max_num_token, vocab_size),
                # Zeus chip rule: a `copy_` writing logits (bf16, from
                # lm_head) into a float32 buffer crosses dtypes and falls
                # into the synchronous CPU-bounce path, which is *not*
                # recorded into the captured graph. The buffer would then
                # hold capture-time logits at every replay, decode logits
                # collapse to a fixed token. Match the model dtype on Zeus
                # so the copy stays on the dtype-matching D2D fast path
                # (zenlMemcpy, captured). Other backends keep float32.
                dtype=dtype if torch.device(device).type == "zeus" else torch.float,
            )

            if pp_size > 1:
                pp_proxy_tensors = {
                    "hidden_states": torch.zeros((max_bs, hidden_size), dtype=dtype),
                    "residual": torch.zeros((max_bs, hidden_size), dtype=dtype),
                }
            else:
                pp_proxy_tensors = None

            if is_encoder_decoder:
                encoder_lens = create_filled_tensor(
                    (max_bs,),
                    encoder_len_fill_value,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                encoder_lens = None

            if require_mlp_tp_gather:
                global_num_tokens_gpu = torch.zeros((dp_size,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros(
                    (dp_size,), dtype=torch.int32
                )
            else:
                global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros((1,), dtype=torch.int32)

        # Keep seq_lens_cpu as a true CPU tensor, like the old implementation.
        seq_lens_cpu = torch.full(
            (max_bs,),
            seq_len_fill_value,
            dtype=torch.int32,
            device="cpu",
        )

        return cls(
            input_ids=input_ids,
            input_embeds=input_embeds,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            positions=positions,
            mrope_positions=mrope_positions,
            num_token_non_padded=num_token_non_padded,
            custom_mask=custom_mask,
            next_token_logits_buffer=next_token_logits_buffer,
            encoder_lens=encoder_lens,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def populate_from_forward_batch(
        self,
        *,
        forward_batch: ForwardBatch,
        raw_bs: int,
        raw_num_token: int,
        bs: int,
        seq_len_fill_value: int,
        require_gathered_buffer: bool,
        num_tokens_per_bs: int,
        nsa_enable_prefill_cp: bool,
        enable_num_token_non_padded_flag: bool,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Optional[torch.Tensor]:
        if bs != raw_bs:
            fill_tensor_(self.seq_lens, seq_len_fill_value)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)

        seq_lens_cpu: Optional[torch.Tensor] = None
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
            seq_lens_cpu = self.seq_lens_cpu[:bs]

        if self.encoder_lens is not None and forward_batch.encoder_lens is not None:
            self.encoder_lens[:raw_bs].copy_(forward_batch.encoder_lens)

        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_num_token].copy_(forward_batch.mrope_positions)

        if require_gathered_buffer:
            fill_tensor_(self.global_num_tokens_gpu, bs * num_tokens_per_bs)
            fill_tensor_(
                self.global_num_tokens_for_logprob_gpu,
                bs * num_tokens_per_bs,
            )

        if enable_num_token_non_padded_flag:
            if require_gathered_buffer and not nsa_enable_prefill_cp:
                num_tokens_per_dp = bs * num_tokens_per_bs
                local = compute_local_num_token_non_padded(
                    global_num_token_non_padded=forward_batch.num_token_non_padded,
                    num_tokens_per_dp=num_tokens_per_dp,
                )
                self.num_token_non_padded.copy_(local)
            else:
                self.num_token_non_padded.copy_(forward_batch.num_token_non_padded)

        # Pipeline-parallel proxy tensors.
        if pp_proxy_tensors is not None and self.pp_proxy_tensors is not None:
            for key, buf in self.pp_proxy_tensors.items():
                src = pp_proxy_tensors.tensors[key]
                dim = src.shape[0]
                buf[:dim].copy_(src)

        return seq_lens_cpu
