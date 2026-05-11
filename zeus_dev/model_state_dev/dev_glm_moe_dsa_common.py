"""
Shared proxy math for GLM-MoE-DSA dev milestone tests.

The two entry points are:
  - dev_glm_moe_dsa_decode_test.py
  - dev_glm_moe_dsa_prefill_test.py

This file intentionally keeps all math in small pure-torch proxy shapes so that
REF mode is cheap and deterministic. Zeus mode is only used by stage scripts for
kernels that have actually landed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


DEFAULT_CONFIG_PATH = Path(
    "/datau38020T/Application/tanzh/hf_cache/hub/16b_hf/config.json"
)


@dataclass
class ProxyDsaConfig:
    hidden_size: int = 128
    num_attention_heads: int = 4
    q_lora_rank: int = 48
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 8
    v_head_dim: int = 16
    index_n_heads: int = 2
    index_head_dim: int = 16
    index_topk: int = 8
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass
class StageResult:
    status: str
    payload: object = None

    @property
    def ok(self) -> bool:
        return self.status == "PASS" or self.status.startswith("SKIP")


def pass_stage(payload=None) -> StageResult:
    return StageResult("PASS", payload)


def fail_stage(payload=None) -> StageResult:
    return StageResult("FAIL", payload)


def skip_stage(reason: str) -> StageResult:
    print(f"  [skip] {reason}")
    return StageResult("SKIP (REF-only; Zeus TODO)", None)


def stage_header(name: str) -> None:
    print()
    print("=" * 72)
    print(f"Stage: {name}")
    print("=" * 72)


def load_glm5_next_config(path: Path) -> Dict:
    return json.loads(path.read_text())


def test_config_summary(args) -> StageResult:
    stage_header("config_summary")
    cfg_json = load_glm5_next_config(args.config)
    fields = [
        "architectures",
        "model_type",
        "hidden_size",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "max_position_embeddings",
    ]
    for key in fields:
        print(f"  {key:24s}: {cfg_json.get(key)}")
    lac = cfg_json.get("linear_attn_config", {})
    print(f"  full_attn_layers        : {lac.get('full_attn_layers')}")
    print(f"  kda_layers              : {lac.get('kda_layers')}")
    return pass_stage()


def compare_tensors(name, ref_out, got_out, atol=5e-3, rtol=5e-3) -> bool:
    a = ref_out.detach().float().cpu()
    b = got_out.detach().float().cpu()
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: ref={a.shape} got={b.shape}")
        return False
    abs_diff = (a - b).abs()
    close = torch.allclose(a, b, atol=atol, rtol=rtol)
    status = "PASS" if close else "DIFF"
    print(
        f"  [{name}] {status} | max_diff={abs_diff.max().item():.6e} "
        f"mean_diff={abs_diff.mean().item():.6e} shape={list(a.shape)}"
    )
    return close


def compare_ids(name, ref_ids, got_ids, allow_permutation=False) -> bool:
    a = ref_ids.detach().cpu().to(torch.int64)
    b = got_ids.detach().cpu().to(torch.int64)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: ref={a.shape} got={b.shape}")
        return False
    if allow_permutation:
        ok = torch.equal(a.sort(dim=-1).values, b.sort(dim=-1).values)
    else:
        ok = torch.equal(a, b)
    status = "PASS" if ok else "DIFF"
    mismatch = (a != b).sum().item()
    print(
        f"  [{name}] {status} | mismatch_cells={mismatch}/{a.numel()} "
        f"shape={list(a.shape)}"
    )
    return ok


def _linear(x, weight):
    return x.float().matmul(weight.float().t()).to(x.dtype)


def _rmsnorm(x, weight=None, eps=1e-5):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_cache(positions, dim, theta=10000.0):
    assert dim % 2 == 0
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=positions.device).float() / dim)
    )
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _apply_rope(x, positions, theta=10000.0):
    cos, sin = _rope_cache(positions, x.shape[-1], theta)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def make_proxy_weights(cfg: ProxyDsaConfig, seed=0):
    torch.manual_seed(seed)

    def randn(*shape, scale=0.02):
        return torch.randn(*shape, dtype=torch.bfloat16) * scale

    return {
        "q_a_proj": randn(cfg.q_lora_rank, cfg.hidden_size),
        "q_a_norm": torch.ones(cfg.q_lora_rank, dtype=torch.float32),
        "q_b_proj": randn(
            cfg.num_attention_heads * cfg.qk_head_dim, cfg.q_lora_rank
        ),
        "kv_a_proj": randn(
            cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size
        ),
        "kv_a_norm": torch.ones(cfg.kv_lora_rank, dtype=torch.float32),
        "kv_b_proj": randn(
            cfg.num_attention_heads
            * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            cfg.kv_lora_rank,
        ),
        "o_proj": randn(cfg.hidden_size, cfg.num_attention_heads * cfg.v_head_dim),
        "index_wq_b": randn(
            cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank
        ),
        "index_wk": randn(cfg.index_head_dim, cfg.hidden_size),
        "index_k_norm": torch.ones(cfg.index_head_dim, dtype=torch.float32),
        "index_weights_proj": randn(
            cfg.index_n_heads, cfg.hidden_size, scale=0.01
        ).float(),
    }


def make_inputs(num_tokens: int, cfg: ProxyDsaConfig, seed: int):
    torch.manual_seed(seed)
    hidden_states = torch.randn(num_tokens, cfg.hidden_size, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, dtype=torch.long)
    weights = make_proxy_weights(cfg, seed=seed + 1)
    return hidden_states, positions, weights


def ref_dsa_q_proj(hidden_states, positions, weights, cfg):
    q_lora = _linear(hidden_states, weights["q_a_proj"])
    q_lora_norm = _rmsnorm(q_lora, weights["q_a_norm"], cfg.rms_norm_eps)
    q = _linear(q_lora_norm, weights["q_b_proj"])
    q = q.view(-1, cfg.num_attention_heads, cfg.qk_head_dim)
    q_nope, q_pe = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
    q_pe = _apply_rope(q_pe, positions, cfg.rope_theta)
    return q_lora_norm, torch.cat([q_nope, q_pe], dim=-1).to(torch.bfloat16)


def ref_mla_projection_rope(hidden_states, positions, weights, cfg):
    q_lora_norm, q_full = ref_dsa_q_proj(hidden_states, positions, weights, cfg)

    latent = _linear(hidden_states, weights["kv_a_proj"])
    kv_lora, k_pe = latent.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1)
    kv_lora_norm = _rmsnorm(kv_lora, weights["kv_a_norm"], cfg.rms_norm_eps)
    kv = _linear(kv_lora_norm, weights["kv_b_proj"])
    kv = kv.view(
        -1,
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim + cfg.v_head_dim,
    )
    k_nope, v = kv.split([cfg.qk_nope_head_dim, cfg.v_head_dim], dim=-1)

    k_pe = _apply_rope(k_pe, positions, cfg.rope_theta).unsqueeze(1)
    k_pe = k_pe.expand(-1, cfg.num_attention_heads, -1)
    k_full = torch.cat([k_nope, k_pe], dim=-1)
    return q_lora_norm, q_full, k_full, v


def ref_indexer_prep(hidden_states, q_lora_norm, positions, weights, cfg):
    q_idx = _linear(q_lora_norm, weights["index_wq_b"])
    q_idx = q_idx.view(-1, cfg.index_n_heads, cfg.index_head_dim)
    q_rope, q_nope = q_idx.split(
        [cfg.qk_rope_head_dim, cfg.index_head_dim - cfg.qk_rope_head_dim],
        dim=-1,
    )
    q_rope = _apply_rope(q_rope, positions, cfg.rope_theta)
    q_idx = torch.cat([q_rope, q_nope], dim=-1)

    k_idx = _linear(hidden_states, weights["index_wk"])
    k_idx = _rmsnorm(k_idx, weights["index_k_norm"], cfg.rms_norm_eps)
    k_rope, k_nope = k_idx.split(
        [cfg.qk_rope_head_dim, cfg.index_head_dim - cfg.qk_rope_head_dim],
        dim=-1,
    )
    k_rope = _apply_rope(k_rope, positions, cfg.rope_theta)
    k_idx = torch.cat([k_rope, k_nope], dim=-1)

    head_weights = hidden_states.float().matmul(weights["index_weights_proj"].t())
    head_weights = head_weights * (cfg.index_n_heads**-0.5)
    head_weights = head_weights * (cfg.index_head_dim**-0.5)
    return q_idx, k_idx, head_weights


def store_index_k_cache(cache, slots, k_idx):
    cache[slots.to(torch.long)] = k_idx
    return cache


def store_main_kv_cache(k_cache, v_cache, slots, k, v):
    slots = slots.to(torch.long)
    k_cache[slots] = k
    v_cache[slots] = v
    return k_cache, v_cache


def ref_causal_topk_single(q_idx, k_idx, head_weights, cfg):
    raw = torch.einsum("qhd,kd->qhk", q_idx.float(), k_idx.float())
    logits = (raw * head_weights.float().unsqueeze(-1)).sum(dim=1)
    T = logits.shape[0]
    causal = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=logits.device), diagonal=1
    )
    logits = logits.masked_fill(causal, float("-inf"))

    topk = torch.full((T, cfg.index_topk), -1, dtype=torch.int32, device=logits.device)
    for t in range(T):
        row_k = min(cfg.index_topk, t + 1)
        topk[t, :row_k] = torch.topk(logits[t, : t + 1], k=row_k).indices.to(
            torch.int32
        )
    return logits, topk


def ref_prefill_ragged_topk(q_idx, k_idx, head_weights, request_lens, cfg):
    T = q_idx.shape[0]
    logits = torch.full((T, T), float("-inf"), dtype=torch.float32, device=q_idx.device)
    topk = torch.full((T, cfg.index_topk), -1, dtype=torch.int32, device=q_idx.device)

    offset = 0
    for req_len in request_lens:
        for local_q in range(req_len):
            q_row = offset + local_q
            valid = torch.arange(offset, offset + local_q + 1, device=q_idx.device)
            raw = torch.einsum(
                "hd,kd->hk", q_idx[q_row].float(), k_idx[valid].float()
            )
            score = (raw * head_weights[q_row].float().unsqueeze(-1)).sum(dim=0)
            logits[q_row, valid] = score
            row_k = min(cfg.index_topk, valid.numel())
            top_local = torch.topk(score, k=row_k).indices
            topk[q_row, :row_k] = valid[top_local].to(torch.int32)
        offset += req_len
    return logits, topk


def ref_sparse_attention(q, k, v, topk_indices):
    T, Hh, D = q.shape
    out = torch.empty(T, Hh, v.shape[-1], dtype=v.dtype, device=v.device)
    for t in range(T):
        ids = topk_indices[t].to(torch.long)
        valid = ids >= 0
        ids_safe = ids.clamp(min=0)
        kg = k[ids_safe]
        vg = v[ids_safe]
        score = torch.einsum("hd,khd->hk", q[t].float(), kg.float()) * (D**-0.5)
        score = score.masked_fill(~valid.unsqueeze(0), float("-inf"))
        prob = torch.softmax(score, dim=-1)
        out[t] = torch.einsum("hk,khd->hd", prob, vg.float()).to(v.dtype)
    return out


def ref_decode_sparse_attention(q_t, k_cache, v_cache, ids):
    valid = ids >= 0
    ids_safe = ids.to(torch.long).clamp(min=0)
    kg = k_cache[ids_safe]
    vg = v_cache[ids_safe]
    score = torch.einsum("hd,khd->hk", q_t.float(), kg.float()) * (
        q_t.shape[-1] ** -0.5
    )
    score = score.masked_fill(~valid.unsqueeze(0), float("-inf"))
    prob = torch.softmax(score, dim=-1)
    return torch.einsum("hk,khv->hv", prob, vg.float()).to(v_cache.dtype)


def ref_dense_causal_attention(q, k, v, request_lens=None):
    if request_lens is None:
        request_lens = [q.shape[0]]
    out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], dtype=v.dtype)
    offset = 0
    scale = q.shape[-1] ** -0.5
    for req_len in request_lens:
        q_req = q[offset : offset + req_len]
        k_req = k[offset : offset + req_len]
        v_req = v[offset : offset + req_len]
        scores = torch.einsum("thd,shd->hts", q_req.float(), k_req.float()) * scale
        causal = torch.triu(
            torch.ones(req_len, req_len, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[offset : offset + req_len] = torch.einsum(
            "hts,shv->thv", probs, v_req.float()
        ).to(v.dtype)
        offset += req_len
    return out


def ref_o_proj(attn_out, weights):
    attn_flat = attn_out.reshape(attn_out.shape[0], -1)
    return _linear(attn_flat, weights["o_proj"])


def run_zeus_q_proj(hidden_states, positions, weights, cfg):
    import torch_zeus  # noqa: F401
    import sgl_kernel_zeus

    return sgl_kernel_zeus.dsa_q_proj_fused(
        hidden_states.contiguous().to("zeus"),
        positions.to(torch.int32).contiguous().to("zeus"),
        weights["q_a_proj"].contiguous().to("zeus"),
        weights["q_a_norm"].contiguous().to("zeus"),
        weights["q_b_proj"].contiguous().to("zeus"),
        num_heads=cfg.num_attention_heads,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
    )


def parse_request_lens(text: str | None, total_tokens: int) -> List[int]:
    if not text:
        return [total_tokens]
    lens = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not lens or any(x <= 0 for x in lens):
        raise ValueError("--request-lens must contain positive integers")
    if sum(lens) != total_tokens:
        raise ValueError(
            f"sum(--request-lens)={sum(lens)} must equal --tokens={total_tokens}"
        )
    return lens


def run_stage_table(stage_fns: Dict[str, object], args) -> None:
    selected = list(stage_fns.keys()) if args.stage == "all" else [args.stage]
    results = []
    for name in selected:
        result = stage_fns[name](args)
        results.append((name, result.status))

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    for name, status in results:
        print(f"  {name:38s}: {status}")

    if any(status == "FAIL" for _, status in results):
        raise SystemExit(1)
