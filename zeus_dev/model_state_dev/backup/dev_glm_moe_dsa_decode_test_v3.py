"""
Decode-path milestone tests for GLM-MoE-DSA — V3.

Stages aligned with GlmMoeDsa_dev_V3.md §7 (Decode Path), 7 steps:
  D0 decode_qkv_a_proj_norm_fused        (new vs V1)
  D1 decode_q_proj_fused                 (V3: extends V1 ✓ to △ — adds bmm w_kc absorb)
  D1-side decode_kv_cache_store          (latent KV cache row [1, Rkv+Dro])
  D2 decode_indexer_prep_store_fused     (includes Hadamard note)
  D3 decode_indexer_topk_fused
  D4 decode_sparse_mqa_fused             (V3 RENAMED — output is [B,Nh,Rkv=512], not Dv)
  D5 decode_v_absorb                     (new vs V1)
  D6 decode_o_proj
  bonus decode_absorb_equiv              (V1 decompress == V3 absorb numerically)
  decode_full_path                       (D0–D6 end-to-end)

In --mode zeus, only landed kernels run. Most V3 stages currently SKIP because
the existing zeus `dsa_q_proj_fused` does not yet emit q_nope_out (it returns
q_full); D1 is tested by composing the existing zeus q_proj with a Python-side
absorb so we at least cover the projection half.
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
    ref_indexer_prep,
    ref_o_proj,
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
    ref_decode_sparse_mqa_latent,
    ref_full_kv,
    ref_q_b_proj_split_rope_absorb,
    ref_qkv_a_proj_norm,
    ref_v_absorb,
    store_latent_kv_cache,
)


# ---------------------------------------------------------------------
# Shared scaffold for all decode stages
# ---------------------------------------------------------------------

def _decode_case(args, *, index_topk=None):
    cfg = ProxyDsaConfig(index_topk=index_topk or args.index_topk)
    hidden, positions, weights = make_inputs(args.history_tokens, cfg, args.seed)

    # Offline absorb-matrix split (SG-W0)
    w_kc, w_vc = derive_w_kc_w_vc(weights, cfg)

    # D0: hidden -> (q_lora_norm, kv_lora_norm, k_pe)
    q_lora_norm, kv_lora_norm, k_pe = ref_qkv_a_proj_norm(hidden, weights, cfg)

    # D1: q_b_proj + split + RoPE + bmm w_kc; emit also k_pe_rope for cache row
    q_nope_out, q_pe, k_pe_rope = ref_q_b_proj_split_rope_absorb(
        q_lora_norm, k_pe, positions, weights, cfg, w_kc
    )

    # D2: indexer prep (Hadamard kept as identity in REF; bf16 invariant)
    q_idx, k_idx, gate = ref_indexer_prep(hidden, q_lora_norm, positions, weights, cfg)
    q_idx = hadamard_identity_rotate(q_idx)
    k_idx = hadamard_identity_rotate(k_idx)

    return {
        "cfg": cfg,
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


def _pool_size(args):
    return args.cache_offset + args.history_tokens + 4


def _history_slots(args):
    """Physical cache slots used by the history (and the new decode token).

    The LAST entry corresponds to the new decode token at position
    `args.history_tokens - 1`.
    """
    return torch.arange(
        args.cache_offset,
        args.cache_offset + args.history_tokens,
        dtype=torch.long,
    )


# ---------------------------------------------------------------------
# D0 — fused qkv-a projection + norms
# ---------------------------------------------------------------------

def test_decode_qkv_a_proj_norm_fused(args):
    stage_header("D0 decode_qkv_a_proj_norm_fused (V3 new)")
    cfg = ProxyDsaConfig(index_topk=args.index_topk)
    hidden, _, weights = make_inputs(args.batch, cfg, args.seed)
    q_lora_norm, kv_lora_norm, k_pe = ref_qkv_a_proj_norm(hidden, weights, cfg)
    print(f"  hidden       : {tuple(hidden.shape)} {hidden.dtype}")
    print(f"  q_lora_norm  : {tuple(q_lora_norm.shape)} {q_lora_norm.dtype}")
    print(f"  kv_lora_norm : {tuple(kv_lora_norm.shape)} {kv_lora_norm.dtype}")
    print(f"  k_pe         : {tuple(k_pe.shape)} {k_pe.dtype}")

    if args.mode == "zeus":
        return skip_stage("D0 fused qkv-a projection + norms not landed in Zeus")

    ok = True
    ok &= q_lora_norm.shape == (args.batch, cfg.q_lora_rank)
    ok &= kv_lora_norm.shape == (args.batch, cfg.kv_lora_rank)
    ok &= k_pe.shape == (args.batch, cfg.qk_rope_head_dim)
    return pass_stage((q_lora_norm, kv_lora_norm, k_pe)) if ok else fail_stage()


# ---------------------------------------------------------------------
# D1 — q_b_proj + split + RoPE + bmm w_kc  (V1 ✓ -> V3 △)
# ---------------------------------------------------------------------

def test_decode_q_proj_fused(args):
    stage_header("D1 decode_q_proj_fused (V3: + absorb-K via bmm w_kc)")
    cfg = ProxyDsaConfig(index_topk=args.index_topk)
    hidden, positions, weights = make_inputs(args.batch, cfg, args.seed)
    w_kc, _ = derive_w_kc_w_vc(weights, cfg)

    q_lora_norm, _, k_pe = ref_qkv_a_proj_norm(hidden, weights, cfg)
    ref_q_nope_out, ref_q_pe, ref_k_pe_rope = ref_q_b_proj_split_rope_absorb(
        q_lora_norm, k_pe, positions, weights, cfg, w_kc
    )
    print(f"  q_nope_out : {tuple(ref_q_nope_out.shape)}  -- [B, Nh, Rkv]")
    print(f"  q_pe       : {tuple(ref_q_pe.shape)}  -- [B, Nh, Dro]")
    print(f"  k_pe_rope  : {tuple(ref_k_pe_rope.shape)}  -- [B, 1, Dro]")

    if args.mode == "ref":
        ok = ref_q_nope_out.shape == (args.batch, cfg.num_attention_heads, cfg.kv_lora_rank)
        ok &= ref_q_pe.shape == (args.batch, cfg.num_attention_heads, cfg.qk_rope_head_dim)
        ok &= ref_k_pe_rope.shape == (args.batch, 1, cfg.qk_rope_head_dim)
        return pass_stage((ref_q_nope_out, ref_q_pe, ref_k_pe_rope)) if ok else fail_stage()

    # V3 D1 contract = V1 work (q_b_proj + split + RoPE) + bmm(q_nope, w_kc).
    # Existing Zeus dsa_q_proj_fused only covers the V1 part (returns q_full
    # [B,Nh,Dqk] without absorb-K). Per V3 doc §7.3 the original V1 ✓ is
    # downgraded to △; until the Zeus kernel emits q_nope_out, this stage
    # SKIPs in zeus mode.
    return skip_stage(
        "D1 V3 contract requires bmm(q_nope, w_kc); existing Zeus q_proj only "
        "covers V1 work (q_full without absorb). V1 ✓ -> V3 △."
    )


# ---------------------------------------------------------------------
# D1-side — latent KV cache store (single latent head, Rkv+Dro)
# ---------------------------------------------------------------------

def test_decode_kv_cache_store(args):
    stage_header("D1-side decode_kv_cache_store (latent [1, Rkv+Dro])")
    case = _decode_case(args)
    cfg = case["cfg"]
    t = args.history_tokens - 1
    slot = torch.tensor([args.cache_offset + t], dtype=torch.long)

    latent_row = build_latent_kv_row(
        case["kv_lora_norm"][t : t + 1], case["k_pe_rope"][t : t + 1]
    )  # [1, 1, Rkv+Dro]
    pool_size = _pool_size(args)
    cache = torch.full(
        (pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim),
        float("nan"),
        dtype=latent_row.dtype,
    )

    if args.mode == "ref":
        store_latent_kv_cache(cache, slot, latent_row)
        ok = compare_tensors(
            "D1-side/latent_kv_readback", latent_row, cache[slot]
        )
        print(f"  new_slot      : {int(slot[0])}")
        print(f"  latent_row    : {tuple(latent_row.shape)}")
        return pass_stage((cache, slot)) if ok else fail_stage()

    # Zeus mode: invoke V3 dsa_kv_proj_cache_store_fused kernel for token t,
    # verify the latent row at slot matches the REF.
    try:
        import sgl_kernel_zeus  # noqa: F401
        import torch_zeus  # noqa: F401
    except ImportError as exc:
        return skip_stage(f"Zeus runtime unavailable for D1-side: {exc}")

    cache_z = cache.clone().to("zeus")
    got = sgl_kernel_zeus.dsa_kv_proj_cache_store_fused(
        case["hidden"][t : t + 1].contiguous().to("zeus"),
        case["positions"][t : t + 1].to(torch.int32).contiguous().to("zeus"),
        slot.to(torch.int32).contiguous().to("zeus"),
        case["weights"]["kv_a_proj"].contiguous().to("zeus"),
        case["weights"]["kv_a_norm"].contiguous().to("zeus"),
        cache_z,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
    )
    ok = compare_tensors(
        "D1-side/latent_kv_readback (zeus)",
        latent_row, got.cpu()[slot],
        atol=2e-2, rtol=1e-2,
    )
    print(f"  new_slot      : {int(slot[0])}")
    print(f"  latent_row    : {tuple(latent_row.shape)}")
    return pass_stage((got, slot)) if ok else fail_stage()


# ---------------------------------------------------------------------
# D2 — indexer prep (+ Hadamard) + index K cache store
# ---------------------------------------------------------------------

def test_decode_indexer_prep_store_fused(args):
    stage_header("D2 decode_indexer_prep_store_fused (V3: + Hadamard SG-H1)")
    if args.mode == "zeus":
        return skip_stage("D2 fused indexer prep + index K cache store not landed")
    case = _decode_case(args)
    cfg = case["cfg"]
    t = args.history_tokens - 1
    slot = torch.tensor([args.cache_offset + t], dtype=torch.long)
    pool_size = _pool_size(args)
    index_cache = torch.full(
        (pool_size, cfg.index_head_dim), float("nan"), dtype=case["k_idx"].dtype
    )
    store_index_k_cache(index_cache, slot, case["k_idx"][t : t + 1])

    ok = True
    ok &= case["q_idx"].shape == (args.history_tokens, cfg.index_n_heads, cfg.index_head_dim)
    ok &= case["gate"].shape == (args.history_tokens, cfg.index_n_heads)
    ok &= compare_tensors(
        "D2/index_cache_new_slot", case["k_idx"][t : t + 1], index_cache[slot]
    )
    print(f"  q_idx       : {tuple(case['q_idx'].shape)}")
    print(f"  gate        : {tuple(case['gate'].shape)}")
    print(f"  index slot  : {int(slot[0])}")
    print("  [note] Hadamard rotation kept as identity for bf16 REF;")
    print("         production must apply rotate_activation before FP8 quant (SG-H1).")
    return pass_stage((case["q_idx"], index_cache, case["gate"])) if ok else fail_stage()


# ---------------------------------------------------------------------
# D3 — decode top-k from paged index K cache
# ---------------------------------------------------------------------

def test_decode_indexer_topk_fused(args):
    stage_header("D3 decode_indexer_topk_fused")
    if args.mode == "zeus":
        return skip_stage("D3 decode indexer logits + topk transform not landed")
    case = _decode_case(args)
    cfg = case["cfg"]
    t = args.history_tokens - 1
    _, topk = ref_causal_topk_single(
        case["q_idx"][: t + 1], case["k_idx"][: t + 1], case["gate"][: t + 1], cfg
    )
    row = topk[t]
    valid = row[row >= 0]
    ok = bool(valid.numel() and int(valid.max()) <= t)
    print(f"  decode_t   : {t}")
    print(f"  topk_slots : {row.tolist()}")
    print(f"  [causal] {'PASS' if ok else 'DIFF'}")
    return pass_stage(row) if ok else fail_stage()


# ---------------------------------------------------------------------
# D4 — sparse MQA in latent space (V3 RENAMED + DIM CHANGE)
# ---------------------------------------------------------------------

def test_decode_sparse_mqa_fused(args):
    stage_header("D4 decode_sparse_mqa_fused (V3: output [B,Nh,Rkv=Rkv], NOT Dv)")
    if args.mode == "zeus":
        return skip_stage("D4 decode sparse FlashMLA-style latent kernel not landed")
    case = _decode_case(args)
    cfg = case["cfg"]
    t = args.history_tokens - 1
    slots = _history_slots(args)
    pool_size = _pool_size(args)

    # Populate latent cache for the entire history
    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    latent_rows = build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    store_latent_kv_cache(cache, slots, latent_rows)

    # Decode-Q in latent space
    q_latent_t = assemble_q_latent(case["q_nope_out"], case["q_pe"])[t]  # [Nh, Rkv+Dro]

    # Topk from indexer (local ids over [0, t])
    _, topk = ref_causal_topk_single(
        case["q_idx"][: t + 1], case["k_idx"][: t + 1], case["gate"][: t + 1], cfg
    )
    local_ids = topk[t]

    out_latent = ref_decode_sparse_mqa_latent(
        q_latent_t, cache[slots[: t + 1]], local_ids, cfg
    )
    print(f"  q_latent    : {tuple(q_latent_t.shape)}")
    print(f"  local_ids   : {local_ids.tolist()}")
    print(f"  out_latent  : {tuple(out_latent.shape)}  -- [Nh, Rkv]")
    ok = out_latent.shape == (cfg.num_attention_heads, cfg.kv_lora_rank)
    return pass_stage(out_latent) if ok else fail_stage()


# ---------------------------------------------------------------------
# D5 — V absorb (new vs V1)
# ---------------------------------------------------------------------

def test_decode_v_absorb(args):
    stage_header("D5 decode_v_absorb (V3 new: bmm(out_latent, w_vc))")
    if args.mode == "zeus":
        return skip_stage("D5 V absorb (bmm w_vc) not landed in Zeus")
    case = _decode_case(args)
    cfg = case["cfg"]
    t = args.history_tokens - 1
    slots = _history_slots(args)
    pool_size = _pool_size(args)

    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )

    q_latent_t = assemble_q_latent(case["q_nope_out"], case["q_pe"])[t]
    _, topk = ref_causal_topk_single(
        case["q_idx"][: t + 1], case["k_idx"][: t + 1], case["gate"][: t + 1], cfg
    )
    out_latent = ref_decode_sparse_mqa_latent(
        q_latent_t, cache[slots[: t + 1]], topk[t], cfg
    )
    attn_out = ref_v_absorb(out_latent.unsqueeze(0), case["w_vc"])  # [1, Nh, Dv]
    print(f"  out_latent : {tuple(out_latent.shape)}")
    print(f"  attn_out   : {tuple(attn_out.shape)}  -- [1, Nh, Dv]")
    ok = attn_out.shape == (1, cfg.num_attention_heads, cfg.v_head_dim)
    return pass_stage(attn_out) if ok else fail_stage()


# ---------------------------------------------------------------------
# D6 — output projection
# ---------------------------------------------------------------------

def test_decode_o_proj(args):
    stage_header("D6 decode_o_proj")
    if args.mode == "zeus":
        return skip_stage("D6 ordinary GEMM; no DSA-specific Zeus kernel here")
    case = _decode_case(args)
    cfg = case["cfg"]
    t = args.history_tokens - 1
    slots = _history_slots(args)
    pool_size = _pool_size(args)

    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )
    q_latent_t = assemble_q_latent(case["q_nope_out"], case["q_pe"])[t]
    _, topk = ref_causal_topk_single(
        case["q_idx"][: t + 1], case["k_idx"][: t + 1], case["gate"][: t + 1], cfg
    )
    out_latent = ref_decode_sparse_mqa_latent(
        q_latent_t, cache[slots[: t + 1]], topk[t], cfg
    )
    attn_out = ref_v_absorb(out_latent.unsqueeze(0), case["w_vc"])
    out = ref_o_proj(attn_out, case["weights"])
    print(f"  attn_out : {tuple(attn_out.shape)}")
    print(f"  out      : {tuple(out.shape)}")
    ok = out.shape == (1, cfg.hidden_size)
    return pass_stage(out) if ok else fail_stage()


# ---------------------------------------------------------------------
# Bonus — V1 decompress path == V3 absorb path (numerical equivalence)
# ---------------------------------------------------------------------

def test_decode_absorb_equiv(args):
    stage_header("bonus decode_absorb_equiv (V1 decompress vs V3 absorb)")
    if args.mode == "zeus":
        return skip_stage("equivalence check is REF-only")
    case = _decode_case(args)
    cfg = case["cfg"]
    t = args.history_tokens - 1

    # Common topk
    _, topk = ref_causal_topk_single(
        case["q_idx"][: t + 1], case["k_idx"][: t + 1], case["gate"][: t + 1], cfg
    )
    ids = topk[t]
    valid = ids >= 0
    ids_safe = ids.to(torch.long).clamp(min=0)

    # ---- V3 absorb path ----
    slots = _history_slots(args)
    pool_size = _pool_size(args)
    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )
    q_latent_t = assemble_q_latent(case["q_nope_out"], case["q_pe"])[t]
    out_latent = ref_decode_sparse_mqa_latent(
        q_latent_t, cache[slots[: t + 1]], ids, cfg
    )
    v3_attn = ref_v_absorb(out_latent.unsqueeze(0), case["w_vc"]).squeeze(0)  # [Nh, Dv]

    # ---- V1 decompress path ----
    q_full = assemble_q_full(case["q_lora_norm"], case["positions"], case["weights"], cfg)
    k_full, v_full = ref_full_kv(
        case["kv_lora_norm"], case["k_pe_rope"], case["weights"], cfg
    )
    # standard sparse attention over q_full[t]
    kg = k_full[: t + 1][ids_safe]   # [K, Nh, Dqk]
    vg = v_full[: t + 1][ids_safe]   # [K, Nh, Dv]
    scale = cfg.qk_head_dim ** -0.5
    score = torch.einsum("hd,khd->hk", q_full[t].float(), kg.float()) * scale
    score = score.masked_fill(~valid.unsqueeze(0), float("-inf"))
    prob = torch.softmax(score, dim=-1)
    v1_attn = torch.einsum("hk,khd->hd", prob, vg.float()).to(torch.bfloat16)

    ok = compare_tensors("equiv/attn_out", v1_attn, v3_attn, atol=5e-3, rtol=5e-3)
    return pass_stage((v1_attn, v3_attn)) if ok else fail_stage()


# ---------------------------------------------------------------------
# Full path D0–D6
# ---------------------------------------------------------------------

def test_decode_full_path(args):
    stage_header("decode_full_path (D0–D6 V3)")
    if args.mode == "zeus":
        return skip_stage("decode full path needs D0–D6 Zeus kernels (currently SKIP)")
    case = _decode_case(args)
    cfg = case["cfg"]
    t = args.history_tokens - 1
    slots = _history_slots(args)
    pool_size = _pool_size(args)

    cache = torch.empty(
        pool_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, dtype=case["k_pe"].dtype
    )
    store_latent_kv_cache(
        cache, slots, build_latent_kv_row(case["kv_lora_norm"], case["k_pe_rope"])
    )

    q_latent_t = assemble_q_latent(case["q_nope_out"], case["q_pe"])[t]
    _, topk = ref_causal_topk_single(
        case["q_idx"][: t + 1], case["k_idx"][: t + 1], case["gate"][: t + 1], cfg
    )
    out_latent = ref_decode_sparse_mqa_latent(
        q_latent_t, cache[slots[: t + 1]], topk[t], cfg
    )
    attn_out = ref_v_absorb(out_latent.unsqueeze(0), case["w_vc"])
    out = ref_o_proj(attn_out, case["weights"])
    print(f"  q_latent_t : {tuple(q_latent_t.shape)}")
    print(f"  out_latent : {tuple(out_latent.shape)}")
    print(f"  attn_out   : {tuple(attn_out.shape)}")
    print(f"  final out  : {tuple(out.shape)}")
    return pass_stage(out)


STAGES = {
    "config_summary": test_config_summary,
    "decode_qkv_a_proj_norm_fused": test_decode_qkv_a_proj_norm_fused,
    "decode_q_proj_fused": test_decode_q_proj_fused,
    "decode_kv_cache_store": test_decode_kv_cache_store,
    "decode_indexer_prep_store_fused": test_decode_indexer_prep_store_fused,
    "decode_indexer_topk_fused": test_decode_indexer_topk_fused,
    "decode_sparse_mqa_fused": test_decode_sparse_mqa_fused,
    "decode_v_absorb": test_decode_v_absorb,
    "decode_o_proj": test_decode_o_proj,
    "decode_absorb_equiv": test_decode_absorb_equiv,
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
