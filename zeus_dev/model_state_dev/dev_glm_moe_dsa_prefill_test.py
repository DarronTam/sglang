"""
Prefill/extend-path milestone tests for GLM-MoE-DSA.

Stages are aligned with GlmMoeDsa_dev.md Prefill / Extend Path:
  P0 prefill_q_proj_fused
  P1 prefill_kv_proj_cache_store_fused
  P2 prefill_indexer_prep_store_fused
  P3 prefill_ragged_indexer_topk_fused
  P4 prefill_sparse_attn_fused
  P5 prefill_o_proj
  P-policy dense_fallback_policy

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
    parse_request_lens,
    pass_stage,
    ref_dense_causal_attention,
    ref_indexer_prep,
    ref_mla_projection_rope,
    ref_o_proj,
    ref_prefill_ragged_topk,
    ref_sparse_attention,
    run_stage_table,
    run_zeus_q_proj,
    skip_stage,
    stage_header,
    store_index_k_cache,
    store_main_kv_cache,
    test_config_summary,
)


def _prefill_case(args, *, index_topk=None):
    cfg = ProxyDsaConfig(index_topk=index_topk or args.index_topk)
    request_lens = parse_request_lens(args.request_lens, args.tokens)
    hidden, positions, weights = make_inputs(args.tokens, cfg, args.seed)
    q_lora, q, k, v = ref_mla_projection_rope(hidden, positions, weights, cfg)
    q_idx, k_idx, gate = ref_indexer_prep(hidden, q_lora, positions, weights, cfg)
    return cfg, request_lens, hidden, positions, weights, q_lora, q, k, v, q_idx, k_idx, gate


def _out_cache_loc(args):
    return torch.arange(args.cache_offset, args.cache_offset + args.tokens, dtype=torch.long)


def test_prefill_q_proj_fused(args):
    stage_header("P0 prefill_q_proj_fused")
    cfg = ProxyDsaConfig(index_topk=args.index_topk)
    hidden, positions, weights = make_inputs(args.tokens, cfg, args.seed)
    ref_q_lora, ref_q = ref_mla_projection_rope(hidden, positions, weights, cfg)[:2]
    print(f"  hidden      : {tuple(hidden.shape)} {hidden.dtype}")
    print(f"  q_lora_norm : {tuple(ref_q_lora.shape)} {ref_q_lora.dtype}")
    print(f"  q_full      : {tuple(ref_q.shape)} {ref_q.dtype}")

    if args.mode == "ref":
        return pass_stage((ref_q_lora, ref_q))

    try:
        got_q_lora, got_q = run_zeus_q_proj(hidden, positions, weights, cfg)
    except ImportError as exc:
        return skip_stage(f"Zeus runtime unavailable for P0: {exc}")
    ok = True
    ok &= compare_tensors(
        "P0/q_lora_norm", ref_q_lora, got_q_lora.cpu(), atol=2e-2, rtol=1e-2
    )
    ok &= compare_tensors(
        "P0/q_full", ref_q, got_q.cpu(), atol=2e-2, rtol=1e-2
    )
    return pass_stage((got_q_lora, got_q)) if ok else fail_stage()


def test_prefill_kv_proj_cache_store_fused(args):
    stage_header("P1 prefill_kv_proj_cache_store_fused")
    if args.mode == "zeus":
        return skip_stage("P1 fused KV projection + main KV cache store is not landed")
    cfg, _, _, _, _, _, _, k, v, _, _, _ = _prefill_case(args)
    slots = _out_cache_loc(args)
    pool_size = args.cache_offset + args.tokens + 4
    k_cache = torch.full(
        (pool_size, cfg.num_attention_heads, cfg.qk_head_dim),
        float("nan"),
        dtype=k.dtype,
    )
    v_cache = torch.full(
        (pool_size, cfg.num_attention_heads, cfg.v_head_dim),
        float("nan"),
        dtype=v.dtype,
    )
    store_main_kv_cache(k_cache, v_cache, slots, k, v)
    ok = True
    ok &= compare_tensors("P1/k_cache_readback", k, k_cache[slots])
    ok &= compare_tensors("P1/v_cache_readback", v, v_cache[slots])
    print(f"  out_cache_loc: {slots.tolist()}")
    return pass_stage((k_cache, v_cache, slots)) if ok else fail_stage()


def test_prefill_indexer_prep_store_fused(args):
    stage_header("P2 prefill_indexer_prep_store_fused")
    if args.mode == "zeus":
        return skip_stage("P2 fused indexer prep + index K cache store is not landed")
    cfg, _, _, _, _, _, _, _, _, q_idx, k_idx, gate = _prefill_case(args)
    slots = _out_cache_loc(args)
    pool_size = args.cache_offset + args.tokens + 4
    index_cache = torch.full(
        (pool_size, cfg.index_head_dim), float("nan"), dtype=k_idx.dtype
    )
    store_index_k_cache(index_cache, slots, k_idx)
    ok = True
    ok &= compare_tensors("P2/index_cache_readback", k_idx, index_cache[slots])
    ok &= q_idx.shape == (args.tokens, cfg.index_n_heads, cfg.index_head_dim)
    ok &= gate.shape == (args.tokens, cfg.index_n_heads)
    print(f"  q_idx        : {tuple(q_idx.shape)}")
    print(f"  gate         : {tuple(gate.shape)}")
    print(f"  out_cache_loc: {slots.tolist()}")
    return pass_stage((q_idx, index_cache, gate)) if ok else fail_stage()


def test_prefill_ragged_indexer_topk_fused(args):
    stage_header("P3 prefill_ragged_indexer_topk_fused")
    if args.mode == "zeus":
        return skip_stage("P3 ragged indexer logits + topk transform is not landed")
    cfg, request_lens, _, _, _, _, _, _, _, q_idx, k_idx, gate = _prefill_case(args)
    logits, topk = ref_prefill_ragged_topk(q_idx, k_idx, gate, request_lens, cfg)
    print(f"  request_lens: {request_lens}")
    print(f"  logits      : {tuple(logits.shape)}")
    print(f"  topk        : {tuple(topk.shape)}")
    print(f"  topk[0]     : {topk[0].tolist()}")

    ok = True
    offset = 0
    for req_len in request_lens:
        for local_q in range(req_len):
            row = offset + local_q
            valid = topk[row][topk[row] >= 0]
            if valid.numel():
                ok &= bool(valid.min() >= offset)
                ok &= bool(valid.max() <= row)
        offset += req_len
    print(f"  [ragged_causal] {'PASS' if ok else 'DIFF'}")
    return pass_stage((logits, topk)) if ok else fail_stage()


def test_prefill_sparse_attn_fused(args):
    stage_header("P4 prefill_sparse_attn_fused")
    if args.mode == "zeus":
        return skip_stage("P4 prefill sparse FlashMLA-style DSA kernel is not landed")
    cfg, request_lens, _, _, _, _, q, k, v, q_idx, k_idx, gate = _prefill_case(args)
    _, topk = ref_prefill_ragged_topk(q_idx, k_idx, gate, request_lens, cfg)
    sparse = ref_sparse_attention(q, k, v, topk)
    print(f"  request_lens: {request_lens}")
    print(f"  sparse_out  : {tuple(sparse.shape)}")
    return pass_stage(sparse)


def test_prefill_o_proj(args):
    stage_header("P5 prefill_o_proj")
    if args.mode == "zeus":
        return skip_stage("P5 is ordinary GEMM; no DSA-specific Zeus stage is wired here")
    cfg, request_lens, _, _, weights, _, q, k, v, q_idx, k_idx, gate = _prefill_case(args)
    _, topk = ref_prefill_ragged_topk(q_idx, k_idx, gate, request_lens, cfg)
    attn = ref_sparse_attention(q, k, v, topk)
    out = ref_o_proj(attn, weights)
    print(f"  attn_out: {tuple(attn.shape)}")
    print(f"  o_proj  : {tuple(out.shape)}")
    ok = out.shape == (args.tokens, cfg.hidden_size)
    return pass_stage(out) if ok else fail_stage()


def test_dense_fallback_policy(args):
    stage_header("P-policy dense_fallback_policy")
    if args.mode == "zeus":
        return skip_stage("P-policy dense/MHA dispatch is not wired in Zeus DSA tests")
    index_topk = max(args.tokens, args.index_topk)
    cfg, request_lens, _, _, _, _, q, k, v, q_idx, k_idx, gate = _prefill_case(
        args, index_topk=index_topk
    )
    _, topk = ref_prefill_ragged_topk(q_idx, k_idx, gate, request_lens, cfg)
    sparse = ref_sparse_attention(q, k, v, topk)
    dense = ref_dense_causal_attention(q, k, v, request_lens=request_lens)
    max_kv_len = max(request_lens)
    should_dense = max_kv_len <= cfg.index_topk
    print(f"  request_lens: {request_lens}")
    print(f"  index_topk  : {cfg.index_topk}")
    print(f"  should_dense: {should_dense}")
    ok = compare_tensors(
        "P-policy/sparse_all_vs_dense", dense, sparse, atol=1e-3, rtol=1e-3
    )
    return pass_stage((sparse, dense)) if ok else fail_stage()


def test_prefill_full_path(args):
    stage_header("prefill_full_path")
    if args.mode == "zeus":
        return skip_stage("prefill full path waits for P1-P5 Zeus kernels")
    cfg, request_lens, _, _, weights, _, q, k, v, q_idx, k_idx, gate = _prefill_case(args)
    _, topk = ref_prefill_ragged_topk(q_idx, k_idx, gate, request_lens, cfg)
    attn = ref_sparse_attention(q, k, v, topk)
    out = ref_o_proj(attn, weights)
    print(f"  final out: {tuple(out.shape)}")
    return pass_stage(out)


STAGES = {
    "config_summary": test_config_summary,
    "prefill_q_proj_fused": test_prefill_q_proj_fused,
    "prefill_kv_proj_cache_store_fused": test_prefill_kv_proj_cache_store_fused,
    "prefill_indexer_prep_store_fused": test_prefill_indexer_prep_store_fused,
    "prefill_ragged_indexer_topk_fused": test_prefill_ragged_indexer_topk_fused,
    "prefill_sparse_attn_fused": test_prefill_sparse_attn_fused,
    "prefill_o_proj": test_prefill_o_proj,
    "dense_fallback_policy": test_dense_fallback_policy,
    "prefill_full_path": test_prefill_full_path,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", *STAGES.keys()], default="all")
    parser.add_argument("--mode", choices=["ref", "zeus"], default="zeus")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tokens", type=int, default=12)
    parser.add_argument("--request-lens", default="5,7")
    parser.add_argument("--index-topk", type=int, default=8)
    parser.add_argument("--cache-offset", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    run_stage_table(STAGES, args)


if __name__ == "__main__":
    main()
