"""
Shared V3 helpers for GLM-MoE-DSA dev milestone tests.

V3 changes (see GlmMoeDsa_dev_V3.md):
  * decode + prefill BOTH go through MLA absorb path (handle_attention_nsa
    returns AttnForwardMethod.MLA by default; MHA_ONE_SHOT only for prefill
    short sequences).
  * sparse attention happens in latent space [Nh, Rkv+Dro=576], outputs
    [Nh, Rkv=512]; not [Nh, Dv=128] like V1 incorrectly drew.
  * absorption matrices w_kc / w_vc are derived OFFLINE from kv_b_proj.weight
    (see deepseek_common/deepseek_weight_loader.py:565-610).
  * indexer prep notes Hadamard (orthogonal rotation; identity at bf16 REF
    level, only matters for FP8 quantization to spread outliers).
  * main MLA RoPE and Indexer RoPE may differ in interleave style; this proxy
    uses one simplified RoPE for both — production must split.

This module reuses small primitives from `dev_glm_moe_dsa_common` and adds
V3-specific absorb / latent-space helpers.
"""

from __future__ import annotations

import torch

from dev_glm_moe_dsa_common import (
    ProxyDsaConfig,
    _apply_rope,
    _linear,
    _rmsnorm,
    ref_dense_causal_attention,
)


# =====================================================================
# Step 0: offline absorb matrix split (SG-W0)
# =====================================================================

def derive_w_kc_w_vc(weights, cfg: ProxyDsaConfig):
    """Split kv_b_proj.weight into K-absorb and V-absorb matrices.

    Matches sglang `deepseek_common/deepseek_weight_loader.py:565-610` —
    `kv_b_proj.weight` is stored as `[Nh*(Dnope+Dv), Rkv]`. After unflatten +
    split:
      * w_kc: `[Nh, Dnope, Rkv]`  (kept as-is in production for `bmm w_kc`)
      * w_vc: `[Nh, Rkv, Dv]`     (production transposes (1,2) post-load)

    These are the shapes consumed by `forward_absorb_prepare/core` directly.
    """
    w = weights["kv_b_proj"].view(
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim + cfg.v_head_dim,
        cfg.kv_lora_rank,
    )
    w_kc = w[:, : cfg.qk_nope_head_dim, :].contiguous()                  # [Nh, Dnope, Rkv]
    w_vc = w[:, cfg.qk_nope_head_dim :, :].transpose(1, 2).contiguous()  # [Nh, Rkv, Dv]
    return w_kc, w_vc


# =====================================================================
# D0 / P0: fused qkv-a projection + RMS norms
# =====================================================================

def ref_qkv_a_proj_norm(hidden_states, weights, cfg: ProxyDsaConfig):
    """`hidden [T,H]` -> (q_lora_norm [T,Rq], kv_lora_norm [T,Rkv], k_pe [T,Dro]).

    Production fuses `q_a_proj + kv_a_proj_with_mqa` into a single GEMM
    `fused_qkv_a_proj_with_mqa [Rq+Rkv+Dro, H]`. Proxy keeps them split for
    clarity.
    """
    q_lora = _linear(hidden_states, weights["q_a_proj"])
    q_lora_norm = _rmsnorm(q_lora, weights["q_a_norm"], cfg.rms_norm_eps)

    latent = _linear(hidden_states, weights["kv_a_proj"])
    kv_lora, k_pe = latent.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1)
    kv_lora_norm = _rmsnorm(kv_lora, weights["kv_a_norm"], cfg.rms_norm_eps)
    return q_lora_norm, kv_lora_norm, k_pe


# =====================================================================
# D1 / P1: q_b_proj + split + RoPE + K absorb
# =====================================================================

def ref_q_b_proj_split_rope_absorb(q_lora_norm, k_pe, positions, weights, cfg: ProxyDsaConfig, w_kc):
    """
    Returns:
        q_nope_out [T, Nh, Rkv]   -- K-absorbed Q nope  ★ V3 added vs V1 △
        q_pe       [T, Nh, Dro]   -- RoPE'd Q pe
        k_pe_rope  [T, 1, Dro]    -- RoPE'd K pe (single latent head)
    """
    q = _linear(q_lora_norm, weights["q_b_proj"])
    q = q.view(-1, cfg.num_attention_heads, cfg.qk_head_dim)
    q_nope, q_pe_raw = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)

    # main MLA RoPE — interleave by default per deepseek_v2.py:1244;
    # the proxy uses simplified RoPE for both main and indexer (see module doc).
    # _apply_rope upcasts to float32 internally; cast back to input dtype.
    q_pe = _apply_rope(q_pe_raw, positions, cfg.rope_theta).to(q_pe_raw.dtype)
    k_pe_rope = _apply_rope(k_pe, positions, cfg.rope_theta).to(k_pe.dtype).unsqueeze(1)

    # bmm(q_nope.transpose(0,1) [Nh,T,Dnope], w_kc [Nh,Dnope,Rkv]) -> [Nh,T,Rkv]
    q_nope_out = (
        torch.bmm(q_nope.float().transpose(0, 1), w_kc.float())
        .transpose(0, 1)
        .to(q_nope.dtype)
    )
    return q_nope_out, q_pe, k_pe_rope


def assemble_q_latent(q_nope_out, q_pe):
    """[T, Nh, Rkv] + [T, Nh, Dro] -> [T, Nh, Rkv+Dro]  (latent-space Q for D4/P4)."""
    return torch.cat([q_nope_out, q_pe], dim=-1)


# =====================================================================
# Latent KV cache layout: [pool, 1, Rkv+Dro]
# (NSATokenToKVPool stores [num_pages, page_size, 1, Rkv+Dro]; we flatten the
# pages * page_size dimension into a single 'pool' axis for the proxy.)
# =====================================================================

def build_latent_kv_row(kv_lora_norm, k_pe_rope):
    """`[T, Rkv]` + `[T, 1, Dro]` -> `[T, 1, Rkv+Dro]`."""
    return torch.cat([kv_lora_norm.unsqueeze(1), k_pe_rope], dim=-1)


def store_latent_kv_cache(cache, slots, latent_kv):
    """cache: [pool, 1, Rkv+Dro]; side effect: cache[slots] = latent_kv."""
    cache[slots.to(torch.long)] = latent_kv
    return cache


# =====================================================================
# D2 / P2: Hadamard (REF-only identity; required for FP8 path in production)
# =====================================================================

def hadamard_identity_rotate(x):
    """REF-only Hadamard placeholder.

    The real op (`sglang.jit_kernel.hadamard.hadamard_transform`) is an
    orthogonal rotation. For bf16 dot products it is a no-op (preserved norms
    and inner products). It only matters before FP8 group-quant to spread
    outliers — there it materially changes numerics.

    V1 doc / slides missed this step entirely (SG-H1 in V3 §6).
    """
    return x


# =====================================================================
# D4: decode sparse MQA in latent space (single decode token)
# =====================================================================

def ref_decode_sparse_mqa_latent(q_latent_row, kv_cache_latent_slice, ids, cfg: ProxyDsaConfig):
    """One decode step for one request.

    Args:
        q_latent_row          : [Nh, Rkv+Dro]  -- already absorbed + concat
        kv_cache_latent_slice : [S, 1, Rkv+Dro] -- cache rows visible to this query
        ids                   : [Ktop] int local indices into the slice (-1 = pad)

    Returns:
        out_latent : [Nh, Rkv]
    """
    valid = ids >= 0
    ids_safe = ids.to(torch.long).clamp(min=0)
    kv = kv_cache_latent_slice[ids_safe].squeeze(1)  # [K, Rkv+Dro]
    # Zero out invalid (-1) padding positions defensively. The cache may be
    # `torch.empty` and contain NaN in unread slots; without this, prob=0 at
    # padded positions would still propagate NaN through `0 * NaN`.
    kv = torch.where(valid.unsqueeze(-1), kv, torch.zeros_like(kv))

    # latent dim is both K (full Rkv+Dro for scoring) and pre-absorb V (front Rkv).
    k_part = kv
    v_part = kv[:, : cfg.kv_lora_rank]

    # ★ scale uses ORIGINAL qk_head_dim (= Dnope+Dro), NOT Rkv+Dro.
    # Mathematically equivalent to scoring in decompressed space (see V3 §4).
    scale = cfg.qk_head_dim ** -0.5
    score = torch.einsum("hd,kd->hk", q_latent_row.float(), k_part.float()) * scale
    score = score.masked_fill(~valid.unsqueeze(0), float("-inf"))
    prob = torch.softmax(score, dim=-1)
    return torch.einsum("hk,kd->hd", prob, v_part.float()).to(q_latent_row.dtype)


# =====================================================================
# P4: prefill sparse MQA in latent space (ragged; one query at a time for clarity)
# =====================================================================

def ref_prefill_sparse_mqa_latent(q_latent, kv_cache_latent, topk_indices, cfg: ProxyDsaConfig):
    """
    Args:
        q_latent        : [T, Nh, Rkv+Dro]
        kv_cache_latent : [pool, 1, Rkv+Dro]
        topk_indices    : [T, Ktop]  -- physical cache slot ids (-1 = pad)

    Returns:
        out_latent : [T, Nh, Rkv]
    """
    T, Nh, _ = q_latent.shape
    out = torch.empty(T, Nh, cfg.kv_lora_rank, dtype=q_latent.dtype)
    for t in range(T):
        out[t] = ref_decode_sparse_mqa_latent(
            q_latent[t], kv_cache_latent, topk_indices[t], cfg
        )
    return out


# =====================================================================
# D5 / P5: V absorb
# =====================================================================

def ref_v_absorb(attn_out_latent, w_vc):
    """`[T, Nh, Rkv]` @ `w_vc [Nh, Rkv, Dv]` -> `[T, Nh, Dv]`."""
    attn = attn_out_latent.float().transpose(0, 1)        # [Nh, T, Rkv]
    out = torch.bmm(attn, w_vc.float())                   # [Nh, T, Dv]
    return out.transpose(0, 1).to(attn_out_latent.dtype)  # [T, Nh, Dv]


# =====================================================================
# P4-alt: dense MHA_ONE_SHOT fallback (kv_b_proj decompress + standard MHA)
# =====================================================================

def ref_dense_mha_one_shot(q_full, kv_lora_norm, k_pe_rope, weights, cfg: ProxyDsaConfig, request_lens):
    """Short-sequence dense fallback (P4-alt).

    Triggered when `max_kv_len <= SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`
    (default = `index_topk` = 2048 for GLM5) and not on Blackwell.

    Args:
        q_full       : [T, Nh, Dqk=Dnope+Dro]  -- NOT absorbed
        kv_lora_norm : [T, Rkv]
        k_pe_rope    : [T, 1, Dro]
        request_lens : list[int]  -- ragged metadata for causal varlen

    Returns:
        attn_out_dense : [T, Nh, Dv]   ★ skips V absorb, directly into o_proj
    """
    kv = _linear(kv_lora_norm, weights["kv_b_proj"])
    kv = kv.view(-1, cfg.num_attention_heads, cfg.qk_nope_head_dim + cfg.v_head_dim)
    k_nope, v = kv.split([cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1)
    k_pe = k_pe_rope.expand(-1, cfg.num_attention_heads, -1)
    k_full = torch.cat([k_nope, k_pe], dim=-1)
    return ref_dense_causal_attention(q_full, k_full, v, request_lens=request_lens)


# =====================================================================
# Cross-check: V1 decompress path vs V3 absorb path should be numerically equivalent
# =====================================================================

def assemble_q_full(q_lora_norm, positions, weights, cfg: ProxyDsaConfig):
    """Reproduces V1 'decompressed Q' `[T, Nh, Dqk]` for cross-validation."""
    q = _linear(q_lora_norm, weights["q_b_proj"])
    q = q.view(-1, cfg.num_attention_heads, cfg.qk_head_dim)
    q_nope, q_pe_raw = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
    q_pe = _apply_rope(q_pe_raw, positions, cfg.rope_theta).to(q_pe_raw.dtype)
    return torch.cat([q_nope, q_pe], dim=-1)


def ref_full_kv(kv_lora_norm, k_pe_rope, weights, cfg: ProxyDsaConfig):
    """Reproduces V1 'decompressed K/V' for cross-validation.

    k_full: [T, Nh, Dqk]   v_full: [T, Nh, Dv]
    """
    kv = _linear(kv_lora_norm, weights["kv_b_proj"])
    kv = kv.view(-1, cfg.num_attention_heads, cfg.qk_nope_head_dim + cfg.v_head_dim)
    k_nope, v_full = kv.split([cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1)
    k_pe = k_pe_rope.expand(-1, cfg.num_attention_heads, -1)
    k_full = torch.cat([k_nope, k_pe], dim=-1)
    return k_full, v_full
