"""
Prefill/extend-path milestone tests for GLM-MoE-DSA — V3.

Stages aligned with GlmMoeDsa_dev_V3.md §8 (Prefill / Extend Path), 7 steps +
fallback branch:
  P0 prefill_qkv_a_proj_norm_fused       (new vs V1)
  P1 prefill_q_proj_fused                (V3: + bmm w_kc absorb)
  P1-side prefill_kv_cache_store         (latent KV cache batch write [T,1,Rkv+Dro])
  P2 prefill_indexer_prep_store_fused    (includes Hadamard note)
  P3 prefill_ragged_indexer_topk_fused
  P4 prefill_sparse_mqa_fused            (V3 RENAMED — output [T,Nh,Rkv=512])
  P4-alt dense_fallback_policy           (MHA_ONE_SHOT for max_kv_len <= index_topk)
  P5 prefill_v_absorb                    (new vs V1)
  P6 prefill_o_proj
  bonus prefill_absorb_equiv             (V1 decompress == V3 absorb numerically)
  prefill_full_path                      (P0–P6 end-to-end, absorb branch)
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
    ref_indexer_prep,
    ref_o_proj,
    ref_prefill_ragged_topk,
    ref_sparse_attention,
    run_stage_table,
    skip_stage,
    stage_header,
    store_index_k_cache,
    test_config_summary,
)
from dev_glm_moe_dsa_common_v3 import (
    assemble_q_full,
    assemble_q_latent,
    build_latent_kv_row,
    derive_w_kc_w_vc,
    hadamard_identity_rotate,
    ref_dense_mha_one_shot,
    ref_full_kv,
    ref_prefill_sparse_mqa_latent,
    ref_q_b_proj_split_rope_absorb,
    ref_qkv_a_proj_norm,
    ref_v_absorb,
    store_latent_kv_cache,
)


# ---------------------------------------------------------------------
# Shared scaffold
# ---------------------------------------------------------------------

def _prefill_case(args, *, index_topk=None):
    cfg = ProxyDsaConfig(index_topk=index_topk or args.index_topk)
    request_lens = parse_request_lens(args.request_lens, args.tokens)
    hidden, positions, weights = make_inputs(args.tokens, cfg, args.seed)
    w_kc, w_vc = derive_w_kc_w_vc(weights, cfg)

    q_lora_norm, kv_lora_norm, k_pe = ref_qkv_a_proj_norm(hidden, weights, cfg)
    q_nope_out, q_pe, k_pe_rope = ref_q_b_proj_split_rope_absorb(
        q_lora_norm, k_pe, positions, weights, cfg, w_kc
    )
    q_idx, k_idx, gate = ref_indexer_prep(hidden, q_lora_norm, positions, weights, cfg)
    q_idx = hadamard_identity_rotate(q_idx)
    k_idx = hadamard_identity_rotate(k_idx)

    return {
        "cfg": cfg,
        "request_lens": request_lens,
        "hidden": hidden,
        "positions": positions,
        "weights": weights,
        "w_kc": w_kc,
        "w_vc": w_vc,
        "q_lora_norm": q_lora_norm,
        "kv_lora_norm": kv_lora_norm,
        "k_pe": k_pe,
        "q_nope_out": q_nope_out,
        "q_pe": q_pe,
        "k_pe_rope": k_pe_rope,
        "q_idx": q_idx,
        "k_idx": k_idx,
        "gate": gate,
    }


def _out_cache_loc(args):
    return torch.arange(
        args.cache_offset, args.cache_offset + args.tokens, dtype=torch.long
    )


def _pool_size(args):
    return args.cache_offset + args.tokens + 4


# ---------------------------------------------------------------------
# P0 — fused qkv-a projection + norms
# ---------------------------------------------------------------------

def test_prefill_qkv_a_proj_norm_fused(args):
    stage_header("P0 prefill_qkv_a_proj_norm_fused (V3 new)")
    cfg = ProxyDsaConfig(index_topk=args.index_topk)
    hidden, _, weights = make_inputs(args.tokens, cfg, args.seed)
    q_lora_norm, kv_lora_norm, k_pe = ref_qkv_a_proj_norm(hidden, weights, cfg)
    print(f"  hidden       : {tuple(hidden.shape)} {hidden.dtype}")
    print(f"  q_lora_norm  : {tuple(q_lora_norm.shape)}")
    print(f"  kv_lora_norm : {tuple(kv_lora_norm.shape)}")
    print(f"  k_pe         : {tuple(k_pe.shape)}")
    if args.mode == "zeus":
        return skip_stage("P0 fused qkv-a projection + norms not landed in Zeus")
    ok = q_lora_norm.shape == (args.tokens, cfg.q_lora_rank)
    ok &= kv_lora_norm.shape == (args.tokens, cfg.kv_lora_rank)
    ok &= k_pe.shape == (args.tokens, cfg.qk_rope_head_dim)
    return pass_stage((q_lora_norm, kv_lora_norm, k_pe)) if ok else fail_stage()


# ---------------------------------------------------------------------
# P1 — q_b_proj + split + RoPE + bmm w_kc  (V1 ✓ -> V3 △)
# ---------------------------------------------------------------------

def test_prefill_q_proj_fused(args):
    stage_header("P1 prefill_q_proj_fused (V3: + absorb-K via bmm w_kc)")
    cfg = ProxyDsaConfig(index_topk=args.index_topk)
    hidden, positions, weights = make_inputs(args.tokens, cfg, args.seed)
    w_kc, _ = derive_w_kc_w_vc(weights, cfg)

    q_lora_norm, _, k_pe = ref_qkv_a_proj_norm(hidden, weights, cfg)
    ref_q_nope_out, ref_q_pe, ref_k_pe_rope = ref_q_b_proj_split_rope_absorb(
        q_lora_norm, k_pe, positions, weights, cfg, w_kc
    )
    print(f"  q_nope_out : {tuple(ref_q_nope_out.shape)}  -- [T, Nh, Rkv]")
    print(f"  q_pe       : {tuple(ref_q_pe.shape)}")
    print(f"  k_pe_rope  : {tuple(ref_k_pe_rope.shape)}")

    if args.mode == "ref":
        ok = ref_q_nope_out.shape == (args.tokens, cfg.num_attention_heads, cfg.kv_lora_rank)
        ok &= ref_q_pe.shape == (args.tokens, cfg.num_attention_heads, cfg.qk_rope_head_dim)
        ok &= ref_k_pe_rope.shape == (args.tokens, 1, cfg.qk_rope_head_dim)
        return pass_stage((ref_q_nope_out, ref_q_pe, ref_k_pe_rope)) if ok else fail_stage()

    # V3 P1 contract = V1 work + bmm(q_nope, w_kc). Existing Zeus
    # dsa_q_proj_fused only covers V1 work; bmm w_kc absorb is not in the
    # kernel. SKIP per "未支持就 SKIP" convention.
    return skip_stage(
        "P1 V3 contract requires bmm(q_nope, w_kc); existing Zeus q_proj only "
        "covers V1 work (q_full without absorb). V1 ✓ -> V3 △."
    )


# ---------------------------------------------------------------------
# P1-side — latent KV cache batch write
# ---------------------------------------------------------------------

def test_prefill_kv_cache_store(args):
    stage_header("P1-side prefill_kv_cache_store (latent [T, 1, Rkv+Dro])")
    case = _prefill_case(args)
    cfg = case["cfg"]
    slots = _out_cache_loc(args)
    pool_size = _pool_size(args)

    latent_rows = build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    cache = torch.full(
        (pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim),
        float("nan"),
        dtype=latent_rows.dtype,
    )

    if args.mode == "ref":
        store_latent_kv_cache(cache, slots, latent_rows)
        ok = compare_tensors("P1-side/latent_kv_readback", latent_rows, cache[slots])
        print(f"  out_cache_loc: {slots.tolist()}")
        print(f"  latent_rows  : {tuple(latent_rows.shape)}")
        return pass_stage((cache, slots)) if ok else fail_stage()

    # Zeus mode: invoke V3 dsa_kv_proj_cache_store_fused kernel for all T tokens,
    # verify cache[slots] matches REF latent_rows.
    try:
        import sgl_kernel_zeus  # noqa: F401
        import torch_zeus  # noqa: F401
    except ImportError as exc:
        return skip_stage(f"Zeus runtime unavailable for P1-side: {exc}")

    cache_z = cache.clone().to("zeus")
    got = sgl_kernel_zeus.dsa_kv_proj_cache_store_fused(
        case["hidden"].contiguous().to("zeus"),
        case["positions"].to(torch.int32).contiguous().to("zeus"),
        slots.to(torch.int32).contiguous().to("zeus"),
        case["weights"]["kv_a_proj"].contiguous().to("zeus"),
        case["weights"]["kv_a_norm"].contiguous().to("zeus"),
        cache_z,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
    )
    ok = compare_tensors(
        "P1-side/latent_kv_readback (zeus)",
        latent_rows, got.cpu()[slots],
        atol=2e-2, rtol=1e-2,
    )
    print(f"  out_cache_loc: {slots.tolist()}")
    print(f"  latent_rows  : {tuple(latent_rows.shape)}")
    return pass_stage((got, slots)) if ok else fail_stage()


# ---------------------------------------------------------------------
# P2 — indexer prep + cache store
# ---------------------------------------------------------------------

def test_prefill_indexer_prep_store_fused(args):
    stage_header("P2 prefill_indexer_prep_store_fused (V3: + Hadamard SG-H1)")
    if args.mode == "zeus":
        return skip_stage("P2 fused indexer prep + index K cache store not landed")
    case = _prefill_case(args)
    cfg = case["cfg"]
    slots = _out_cache_loc(args)
    pool_size = _pool_size(args)
    index_cache = torch.full(
        (pool_size, cfg.index_head_dim), float("nan"), dtype=case["k_idx"].dtype
    )
    store_index_k_cache(index_cache, slots, case["k_idx"])
    ok = True
    ok &= compare_tensors("P2/index_cache_readback", case["k_idx"], index_cache[slots])
    ok &= case["q_idx"].shape == (args.tokens, cfg.index_n_heads, cfg.index_head_dim)
    ok &= case["gate"].shape == (args.tokens, cfg.index_n_heads)
    print(f"  q_idx        : {tuple(case['q_idx'].shape)}")
    print(f"  gate         : {tuple(case['gate'].shape)}")
    print(f"  out_cache_loc: {slots.tolist()}")
    print("  [note] Hadamard rotation kept as identity for bf16 REF;")
    print("         production must apply rotate_activation before FP8 quant (SG-H1).")
    return pass_stage((case["q_idx"], index_cache, case["gate"])) if ok else fail_stage()


# ---------------------------------------------------------------------
# P3 — ragged causal top-k
# ---------------------------------------------------------------------

def test_prefill_ragged_indexer_topk_fused(args):
    stage_header("P3 prefill_ragged_indexer_topk_fused")
    if args.mode == "zeus":
        return skip_stage("P3 ragged indexer logits + topk transform not landed")
    case = _prefill_case(args)
    cfg = case["cfg"]
    logits, topk = ref_prefill_ragged_topk(
        case["q_idx"], case["k_idx"], case["gate"], case["request_lens"], cfg
    )
    print(f"  request_lens: {case['request_lens']}")
    print(f"  logits      : {tuple(logits.shape)}")
    print(f"  topk        : {tuple(topk.shape)}")
    print(f"  topk[0]     : {topk[0].tolist()}")

    ok = True
    offset = 0
    for req_len in case["request_lens"]:
        for local_q in range(req_len):
            row = offset + local_q
            valid = topk[row][topk[row] >= 0]
            if valid.numel():
                ok &= bool(valid.min() >= offset)
                ok &= bool(valid.max() <= row)
        offset += req_len
    print(f"  [ragged_causal] {'PASS' if ok else 'DIFF'}")
    return pass_stage((logits, topk)) if ok else fail_stage()


# ---------------------------------------------------------------------
# P4 — sparse MQA in latent space (V3 RENAMED + DIM CHANGE)
# ---------------------------------------------------------------------

def test_prefill_sparse_mqa_fused(args):
    stage_header("P4 prefill_sparse_mqa_fused (V3: output [T,Nh,Rkv], NOT Dv)")
    if args.mode == "zeus":
        return skip_stage("P4 prefill sparse FlashMLA-style latent kernel not landed")
    case = _prefill_case(args)
    cfg = case["cfg"]
    slots = _out_cache_loc(args)
    pool_size = _pool_size(args)

    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )

    q_latent = assemble_q_latent(case["q_nope_out"], case["q_pe"])  # [T, Nh, Rkv+Dro]

    # Topk is over PHYSICAL cache slots (P3 produces local indices in [0, T);
    # we map them through `slots` to the cache).
    _, topk_local = ref_prefill_ragged_topk(
        case["q_idx"], case["k_idx"], case["gate"], case["request_lens"], cfg
    )
    topk_phys = topk_local.clone()
    pos = topk_phys >= 0
    topk_phys[pos] = slots[topk_phys[pos].to(torch.long)].to(topk_phys.dtype)

    out_latent = ref_prefill_sparse_mqa_latent(q_latent, cache, topk_phys, cfg)
    print(f"  q_latent    : {tuple(q_latent.shape)}")
    print(f"  topk_phys[0]: {topk_phys[0].tolist()}")
    print(f"  out_latent  : {tuple(out_latent.shape)}  -- [T, Nh, Rkv]")
    ok = out_latent.shape == (args.tokens, cfg.num_attention_heads, cfg.kv_lora_rank)
    return pass_stage(out_latent) if ok else fail_stage()


# ---------------------------------------------------------------------
# P4-alt — dense fallback (MHA_ONE_SHOT)
# ---------------------------------------------------------------------

def test_dense_fallback_policy(args):
    stage_header("P4-alt dense_fallback_policy (MHA_ONE_SHOT; skip V absorb)")
    if args.mode == "zeus":
        return skip_stage("P4-alt dense fallback dispatch not wired in Zeus DSA tests")

    # Use an inflated index_topk so threshold >= max_kv_len -> fallback should trigger.
    index_topk = max(args.tokens, args.index_topk)
    case = _prefill_case(args, index_topk=index_topk)
    cfg = case["cfg"]

    q_full = assemble_q_full(case["q_lora_norm"], case["positions"], case["weights"], cfg)
    dense_attn = ref_dense_mha_one_shot(
        q_full,
        case["kv_lora_norm"],
        case["k_pe_rope"],
        case["weights"],
        cfg,
        case["request_lens"],
    )

    # Cross-check: when fallback would fire, dense MHA should match the absorb path
    # taken over ALL visible tokens (i.e., topk picks everything). We simulate that.
    slots = _out_cache_loc(args)
    pool_size = _pool_size(args)
    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )
    q_latent = assemble_q_latent(case["q_nope_out"], case["q_pe"])

    # Build "see everything visible" topk per row (causal + same-request)
    T = args.tokens
    Ktop_pad = cfg.index_topk
    full_topk = torch.full((T, Ktop_pad), -1, dtype=torch.int32)
    offset = 0
    for req_len in case["request_lens"]:
        for local_q in range(req_len):
            row = offset + local_q
            visible = torch.arange(offset, offset + local_q + 1)
            phys = slots[visible]
            full_topk[row, : phys.numel()] = phys.to(torch.int32)
        offset += req_len
    out_latent_full = ref_prefill_sparse_mqa_latent(q_latent, cache, full_topk, cfg)
    absorb_attn = ref_v_absorb(out_latent_full, case["w_vc"])

    max_kv_len = max(case["request_lens"])
    should_dense = max_kv_len <= cfg.index_topk
    print(f"  request_lens : {case['request_lens']}")
    print(f"  index_topk   : {cfg.index_topk}")
    print(f"  max_kv_len   : {max_kv_len}")
    print(f"  should_dense : {should_dense}")
    print(f"  dense_attn   : {tuple(dense_attn.shape)}")
    print(f"  absorb_attn  : {tuple(absorb_attn.shape)}")

    # dense and absorb-all paths must agree numerically
    ok = compare_tensors(
        "P4-alt/dense_vs_absorb_all", dense_attn, absorb_attn, atol=5e-3, rtol=5e-3
    )
    return pass_stage((dense_attn, absorb_attn, should_dense)) if ok else fail_stage()


# ---------------------------------------------------------------------
# P5 — V absorb
# ---------------------------------------------------------------------

def test_prefill_v_absorb(args):
    stage_header("P5 prefill_v_absorb (V3 new: bmm(out_latent, w_vc))")
    if args.mode == "zeus":
        return skip_stage("P5 V absorb (bmm w_vc) not landed in Zeus")
    case = _prefill_case(args)
    cfg = case["cfg"]
    slots = _out_cache_loc(args)
    pool_size = _pool_size(args)

    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )
    q_latent = assemble_q_latent(case["q_nope_out"], case["q_pe"])
    _, topk_local = ref_prefill_ragged_topk(
        case["q_idx"], case["k_idx"], case["gate"], case["request_lens"], cfg
    )
    topk_phys = topk_local.clone()
    pos = topk_phys >= 0
    topk_phys[pos] = slots[topk_phys[pos].to(torch.long)].to(topk_phys.dtype)

    out_latent = ref_prefill_sparse_mqa_latent(q_latent, cache, topk_phys, cfg)
    attn_out = ref_v_absorb(out_latent, case["w_vc"])
    print(f"  out_latent : {tuple(out_latent.shape)}")
    print(f"  attn_out   : {tuple(attn_out.shape)}  -- [T, Nh, Dv]")
    ok = attn_out.shape == (args.tokens, cfg.num_attention_heads, cfg.v_head_dim)
    return pass_stage(attn_out) if ok else fail_stage()


# ---------------------------------------------------------------------
# P6 — output projection
# ---------------------------------------------------------------------

def test_prefill_o_proj(args):
    stage_header("P6 prefill_o_proj")
    if args.mode == "zeus":
        return skip_stage("P6 ordinary GEMM; no DSA-specific Zeus kernel here")
    case = _prefill_case(args)
    cfg = case["cfg"]
    slots = _out_cache_loc(args)
    pool_size = _pool_size(args)

    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )
    q_latent = assemble_q_latent(case["q_nope_out"], case["q_pe"])
    _, topk_local = ref_prefill_ragged_topk(
        case["q_idx"], case["k_idx"], case["gate"], case["request_lens"], cfg
    )
    topk_phys = topk_local.clone()
    pos = topk_phys >= 0
    topk_phys[pos] = slots[topk_phys[pos].to(torch.long)].to(topk_phys.dtype)

    out_latent = ref_prefill_sparse_mqa_latent(q_latent, cache, topk_phys, cfg)
    attn_out = ref_v_absorb(out_latent, case["w_vc"])
    out = ref_o_proj(attn_out, case["weights"])
    print(f"  attn_out : {tuple(attn_out.shape)}")
    print(f"  out      : {tuple(out.shape)}")
    ok = out.shape == (args.tokens, cfg.hidden_size)
    return pass_stage(out) if ok else fail_stage()


# ---------------------------------------------------------------------
# Bonus — V1 decompress path == V3 absorb path
# ---------------------------------------------------------------------

def test_prefill_absorb_equiv(args):
    stage_header("bonus prefill_absorb_equiv (V1 decompress vs V3 absorb)")
    if args.mode == "zeus":
        return skip_stage("equivalence check is REF-only")
    case = _prefill_case(args)
    cfg = case["cfg"]

    # Common topk (ragged causal)
    _, topk_local = ref_prefill_ragged_topk(
        case["q_idx"], case["k_idx"], case["gate"], case["request_lens"], cfg
    )

    # ---- V3 absorb path ----
    slots = _out_cache_loc(args)
    pool_size = _pool_size(args)
    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )
    q_latent = assemble_q_latent(case["q_nope_out"], case["q_pe"])
    topk_phys = topk_local.clone()
    pos = topk_phys >= 0
    topk_phys[pos] = slots[topk_phys[pos].to(torch.long)].to(topk_phys.dtype)

    out_latent = ref_prefill_sparse_mqa_latent(q_latent, cache, topk_phys, cfg)
    v3_attn = ref_v_absorb(out_latent, case["w_vc"])

    # ---- V1 decompress path (full Q + full K/V + standard sparse attention) ----
    q_full = assemble_q_full(case["q_lora_norm"], case["positions"], case["weights"], cfg)
    k_full, v_full = ref_full_kv(
        case["kv_lora_norm"], case["k_pe_rope"], case["weights"], cfg
    )
    v1_attn = ref_sparse_attention(q_full, k_full, v_full, topk_local)

    ok = compare_tensors("equiv/attn_out", v1_attn, v3_attn, atol=5e-3, rtol=5e-3)
    return pass_stage((v1_attn, v3_attn)) if ok else fail_stage()


# ---------------------------------------------------------------------
# Full path P0–P6
# ---------------------------------------------------------------------

def test_prefill_full_path(args):
    stage_header("prefill_full_path (P0–P6 V3; absorb branch)")
    if args.mode == "zeus":
        return skip_stage("prefill full path needs P0–P6 Zeus kernels (currently SKIP)")
    case = _prefill_case(args)
    cfg = case["cfg"]
    slots = _out_cache_loc(args)
    pool_size = _pool_size(args)

    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )
    q_latent = assemble_q_latent(case["q_nope_out"], case["q_pe"])
    _, topk_local = ref_prefill_ragged_topk(
        case["q_idx"], case["k_idx"], case["gate"], case["request_lens"], cfg
    )
    topk_phys = topk_local.clone()
    pos = topk_phys >= 0
    topk_phys[pos] = slots[topk_phys[pos].to(torch.long)].to(topk_phys.dtype)

    out_latent = ref_prefill_sparse_mqa_latent(q_latent, cache, topk_phys, cfg)
    attn_out = ref_v_absorb(out_latent, case["w_vc"])
    out = ref_o_proj(attn_out, case["weights"])
    print(f"  q_latent   : {tuple(q_latent.shape)}")
    print(f"  out_latent : {tuple(out_latent.shape)}")
    print(f"  attn_out   : {tuple(attn_out.shape)}")
    print(f"  final out  : {tuple(out.shape)}")
    return pass_stage(out)


STAGES = {
    "config_summary": test_config_summary,
    "prefill_qkv_a_proj_norm_fused": test_prefill_qkv_a_proj_norm_fused,
    "prefill_q_proj_fused": test_prefill_q_proj_fused,
    "prefill_kv_cache_store": test_prefill_kv_cache_store,
    "prefill_indexer_prep_store_fused": test_prefill_indexer_prep_store_fused,
    "prefill_ragged_indexer_topk_fused": test_prefill_ragged_indexer_topk_fused,
    "prefill_sparse_mqa_fused": test_prefill_sparse_mqa_fused,
    "dense_fallback_policy": test_dense_fallback_policy,
    "prefill_v_absorb": test_prefill_v_absorb,
    "prefill_o_proj": test_prefill_o_proj,
    "prefill_absorb_equiv": test_prefill_absorb_equiv,
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
