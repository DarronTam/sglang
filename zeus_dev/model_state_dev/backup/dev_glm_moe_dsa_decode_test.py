"""
Decode-path milestone tests for GLM-MoE-DSA.

Stages are aligned with GlmMoeDsa_dev.md Decode Path:
  D0 decode_q_proj_fused
  D1 decode_kv_proj_cache_store_fused
  D2 decode_indexer_prep_store_fused
  D3 decode_indexer_topk_fused
  D4 decode_sparse_attn_fused
  D5 decode_o_proj

In --mode zeus, only landed kernels run. Unsupported DSA kernels return SKIP
instead of silently falling back to REF math.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dev_glm_moe_dsa_common import (
    DEFAULT_CONFIG_PATH,
    ProxyDsaConfig,
    compare_tensors,
    fail_stage,
    make_inputs,
    pass_stage,
    ref_causal_topk_single,
    ref_decode_sparse_attention,
    ref_indexer_prep,
    ref_mla_projection_rope,
    ref_o_proj,
    run_stage_table,
    run_zeus_kv_proj_cache_store,
    run_zeus_q_proj,
    skip_stage,
    stage_header,
    store_index_k_cache,
    store_main_kv_cache,
    test_config_summary,
)


def _decode_case(args, *, index_topk=None):
    cfg = ProxyDsaConfig(index_topk=index_topk or args.index_topk)
    hidden, positions, weights = make_inputs(args.history_tokens, cfg, args.seed)
    q_lora, q, k, v = ref_mla_projection_rope(hidden, positions, weights, cfg)
    q_idx, k_idx, gate = ref_indexer_prep(hidden, q_lora, positions, weights, cfg)
    return cfg, hidden, positions, weights, q_lora, q, k, v, q_idx, k_idx, gate


def test_decode_q_proj_fused(args):
    stage_header("D0 decode_q_proj_fused")
    cfg = ProxyDsaConfig(index_topk=args.index_topk)
    hidden, positions, weights = make_inputs(args.batch, cfg, args.seed)
    ref_q_lora, ref_q = ref_mla_projection_rope(hidden, positions, weights, cfg)[:2]
    print(f"  hidden      : {tuple(hidden.shape)} {hidden.dtype}")
    print(f"  q_lora_norm : {tuple(ref_q_lora.shape)} {ref_q_lora.dtype}")
    print(f"  q_full      : {tuple(ref_q.shape)} {ref_q.dtype}")

    if args.mode == "ref":
        return pass_stage((ref_q_lora, ref_q))

    try:
        got_q_lora, got_q = run_zeus_q_proj(hidden, positions, weights, cfg)
    except ImportError as exc:
        return skip_stage(f"Zeus runtime unavailable for D0: {exc}")
    ok = True
    ok &= compare_tensors(
        "D0/q_lora_norm", ref_q_lora, got_q_lora.cpu(), atol=2e-2, rtol=1e-2
    )
    ok &= compare_tensors(
        "D0/q_full", ref_q, got_q.cpu(), atol=2e-2, rtol=1e-2
    )
    return pass_stage((got_q_lora, got_q)) if ok else fail_stage()


def test_decode_kv_proj_cache_store_fused(args):
    stage_header("D1 decode_kv_proj_cache_store_fused")
    cfg, hidden, positions, weights, _, _, k, v, _, _, _ = _decode_case(args)
    t = args.history_tokens - 1
    slots = torch.tensor([args.cache_offset + t], dtype=torch.long)
    pool_size = args.cache_offset + args.history_tokens + 4
    print(f"  new_slot: {int(slots[0])}")
    print(f"  hidden_t : {tuple(hidden[t : t + 1].shape)}")
    print(f"  k_full_t : {tuple(k[t : t + 1].shape)}  v_t: {tuple(v[t : t + 1].shape)}")

    if args.mode == "ref":
        k_cache = torch.full(
            (pool_size, cfg.num_attention_heads, cfg.qk_head_dim),
            float("nan"), dtype=k.dtype,
        )
        v_cache = torch.full(
            (pool_size, cfg.num_attention_heads, cfg.v_head_dim),
            float("nan"), dtype=v.dtype,
        )
        store_main_kv_cache(k_cache, v_cache, slots, k[t : t + 1], v[t : t + 1])
        ok = True
        ok &= compare_tensors("D1/k_cache_new_slot", k[t : t + 1], k_cache[slots])
        ok &= compare_tensors("D1/v_cache_new_slot", v[t : t + 1], v_cache[slots])
        return pass_stage((k_cache, v_cache, slots)) if ok else fail_stage()

    k_cache = torch.full(
        (pool_size, cfg.num_attention_heads, cfg.qk_head_dim),
        float("nan"), dtype=torch.bfloat16,
    )
    v_cache = torch.full(
        (pool_size, cfg.num_attention_heads, cfg.v_head_dim),
        float("nan"), dtype=torch.bfloat16,
    )
    try:
        got_k, got_v = run_zeus_kv_proj_cache_store(
            hidden[t : t + 1], positions[t : t + 1], slots, weights, cfg,
            k_cache, v_cache,
        )
    except ImportError as exc:
        return skip_stage(f"Zeus runtime unavailable for D1: {exc}")
    ok = True
    ok &= compare_tensors(
        "D1/k_cache_new_slot", k[t : t + 1], got_k.cpu()[slots], atol=2e-2, rtol=1e-2
    )
    ok &= compare_tensors(
        "D1/v_cache_new_slot", v[t : t + 1], got_v.cpu()[slots], atol=2e-2, rtol=1e-2
    )
    return pass_stage((got_k, got_v, slots)) if ok else fail_stage()


def test_decode_indexer_prep_store_fused(args):
    stage_header("D2 decode_indexer_prep_store_fused")
    if args.mode == "zeus":
        return skip_stage("D2 fused indexer prep + index K cache store is not landed")
    cfg, _, _, _, _, _, _, _, q_idx, k_idx, gate = _decode_case(args)
    t = args.history_tokens - 1
    slots = torch.tensor([args.cache_offset + t], dtype=torch.long)
    pool_size = args.cache_offset + args.history_tokens + 4
    index_cache = torch.full(
        (pool_size, cfg.index_head_dim), float("nan"), dtype=k_idx.dtype
    )
    store_index_k_cache(index_cache, slots, k_idx[t : t + 1])
    ok = True
    ok &= q_idx.shape == (
        args.history_tokens,
        cfg.index_n_heads,
        cfg.index_head_dim,
    )
    ok &= gate.shape == (args.history_tokens, cfg.index_n_heads)
    ok &= compare_tensors("D2/index_cache_new_slot", k_idx[t : t + 1], index_cache[slots])
    print(f"  q_idx       : {tuple(q_idx.shape)}")
    print(f"  gate        : {tuple(gate.shape)}")
    print(f"  index slot  : {int(slots[0])}")
    return pass_stage((q_idx, index_cache, gate)) if ok else fail_stage()


def test_decode_indexer_topk_fused(args):
    stage_header("D3 decode_indexer_topk_fused")
    if args.mode == "zeus":
        return skip_stage("D3 decode indexer logits + topk transform is not landed")
    cfg, _, _, _, _, _, _, _, q_idx, k_idx, gate = _decode_case(args)
    t = args.history_tokens - 1
    _, topk = ref_causal_topk_single(q_idx[: t + 1], k_idx[: t + 1], gate[: t + 1], cfg)
    row = topk[t]
    valid = row[row >= 0]
    ok = bool(valid.numel() and int(valid.max()) <= t)
    print(f"  decode_t   : {t}")
    print(f"  topk_slots : {row.tolist()}")
    print(f"  [causal] {'PASS' if ok else 'DIFF'}")
    return pass_stage(row) if ok else fail_stage()


def test_decode_sparse_attn_fused(args):
    stage_header("D4 decode_sparse_attn_fused")
    if args.mode == "zeus":
        return skip_stage("D4 decode sparse FlashMLA-style DSA kernel is not landed")
    cfg, _, _, _, _, q, k, v, q_idx, k_idx, gate = _decode_case(args)
    t = args.history_tokens - 1
    slots = torch.arange(args.cache_offset, args.cache_offset + args.history_tokens)
    pool_size = args.cache_offset + args.history_tokens + 4
    k_cache = torch.empty(
        pool_size, cfg.num_attention_heads, cfg.qk_head_dim, dtype=k.dtype
    )
    v_cache = torch.empty(
        pool_size, cfg.num_attention_heads, cfg.v_head_dim, dtype=v.dtype
    )
    store_main_kv_cache(k_cache, v_cache, slots, k, v)
    _, full_topk = ref_causal_topk_single(q_idx[: t + 1], k_idx[: t + 1], gate[: t + 1], cfg)
    local_ids = full_topk[t]
    cached_out = ref_decode_sparse_attention(
        q[t], k_cache[slots[: t + 1]], v_cache[slots[: t + 1]], local_ids
    )
    print(f"  q_t        : {tuple(q[t].shape)}")
    print(f"  local_ids  : {local_ids.tolist()}")
    print(f"  attn_out   : {tuple(cached_out.shape)}")
    return pass_stage(cached_out)


def test_decode_o_proj(args):
    stage_header("D5 decode_o_proj")
    if args.mode == "zeus":
        return skip_stage("D5 is ordinary GEMM; no DSA-specific Zeus stage is wired here")
    cfg, _, _, weights, _, q, k, v, q_idx, k_idx, gate = _decode_case(args)
    t = args.history_tokens - 1
    _, full_topk = ref_causal_topk_single(q_idx[: t + 1], k_idx[: t + 1], gate[: t + 1], cfg)
    attn = ref_decode_sparse_attention(q[t], k[: t + 1], v[: t + 1], full_topk[t])
    out = ref_o_proj(attn.unsqueeze(0), weights)
    print(f"  attn_out: {tuple(attn.shape)}")
    print(f"  o_proj  : {tuple(out.shape)}")
    ok = out.shape == (1, cfg.hidden_size)
    return pass_stage(out) if ok else fail_stage()


def test_decode_full_path(args):
    stage_header("decode_full_path")
    if args.mode == "zeus":
        return skip_stage("decode full path waits for D1-D5 Zeus kernels")
    cfg, _, _, weights, _, q, k, v, q_idx, k_idx, gate = _decode_case(args)
    t = args.history_tokens - 1
    _, topk = ref_causal_topk_single(q_idx[: t + 1], k_idx[: t + 1], gate[: t + 1], cfg)
    attn = ref_decode_sparse_attention(q[t], k[: t + 1], v[: t + 1], topk[t])
    out = ref_o_proj(attn.unsqueeze(0), weights)
    print(f"  final out: {tuple(out.shape)}")
    return pass_stage(out)


STAGES = {
    "config_summary": test_config_summary,
    "decode_q_proj_fused": test_decode_q_proj_fused,
    "decode_kv_proj_cache_store_fused": test_decode_kv_proj_cache_store_fused,
    "decode_indexer_prep_store_fused": test_decode_indexer_prep_store_fused,
    "decode_indexer_topk_fused": test_decode_indexer_topk_fused,
    "decode_sparse_attn_fused": test_decode_sparse_attn_fused,
    "decode_o_proj": test_decode_o_proj,
    "decode_full_path": test_decode_full_path,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", *STAGES.keys()], default="all")
    parser.add_argument("--mode", choices=["ref", "zeus"], default="zeus")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--history-tokens", type=int, default=12)
    parser.add_argument("--index-topk", type=int, default=8)
    parser.add_argument("--cache-offset", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    run_stage_table(STAGES, args)


if __name__ == "__main__":
    main()
