"""
GLM5-Next DSA Decode 段逐算子 REF-vs-Zeus 对齐脚本

范围（见 glm5next_dsa_decode_dev.md）：
  - 起点：DSA attention 模块收到的 hidden_state [B, H]
  - 终点：self_attn.o_proj 输出 out [B, H]
  - 单 DSA 模块、decode only、TP=1；CP 用 in-process ranks 模拟

Stage:
  q_a_proj_norm          #0.Q  hidden -> q_a_proj -> RMSNorm -> q_lora_out
  kv_a_proj_norm_store   #0.KV hidden -> kv_a_proj -> RMSNorm -> latent_kv_cache[slot]
  q_main             #1  q_b_proj + absorb bmm(w_kc)，Dro=0，无 split/RoPE
  idx_q_weights      #2.Q  Indexer Q: wq_b + Hadamard + FP8-proxy quant + weights=gate*q_scale
  idx_k_prep_store   #2.K  Indexer K: wk + full LayerNorm + Hadamard + FP8-proxy quant + cache store
  idx_logits         #3  local Index GEMM -> gate reduce -> k_scale
  local_topk         #4  S_local<=Ktop 直接返回；否则 local topK
  cp_topk_merge      #5/#6  local-topK all-gather + global merge-topK
  latent_gather      #7  topK owner filter; owned latent K Gmem -> Lmem.
                          6-output: K_local × 2 cores × 2 layouts + mask × 2
                          cores. c0/c1 broadcast (identical content) so each
                          core in #8 can read its own bank with no cross-core
                          traffic. With CORE_NUM=1 the *_c1 buffers are
                          reservations (allocated but content undefined).
  sparse_mqa_partial #8  sparse MQA partial. Takes the 6 per-core inputs from
                          #7. Score GEMM reads K_local_T_c{core_id} (no
                          tl.trans), O update reads K_local_c{core_id}.
                          输出 partial_out + partial_lse
  post_o_proj_nocp   #10 CP=1 no-reduce post，V absorb + o_proj（REF-vs-Zeus 真对拍）
  post_o_proj_cp     #11 CP>1 FA-reduce + V absorb + o_proj
  decode_full_nocp   #0..#10 端到端 no-CP decode
  decode_full_cp     #0..#11 端到端 CP decode，校验 cp=1 vs cp=k

用法:
  python zeus_dev/model_state_dev/dev_glm5next_dsa_decode_test.py
  python zeus_dev/model_state_dev/dev_glm5next_dsa_decode_test.py --stage decode_full_cp --cp 4
  python zeus_dev/model_state_dev/dev_glm5next_dsa_decode_test.py --mode ref
  python zeus_dev/model_state_dev/dev_glm5next_dsa_decode_test.py --mode zeus

约定：
  - 默认 mode=both：先跑 REF，再尝试 Zeus。
  - 当前 DSA fused Zeus kernels 尚未落地；Zeus 侧统一打印 SKIP/TODO，Summary 显示
    "SKIP (REF PASS; Zeus TODO)"。
  - 不允许 silent fallback 到 torch/native 伪装 Zeus kernel。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

try:
    import torch_zeus  # noqa: F401 - registers zeus backend when available
    import sgl_kernel_zeus  # noqa: F401

    ZEUS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import availability is environment-specific
    torch_zeus = None
    sgl_kernel_zeus = None
    ZEUS_IMPORT_ERROR = exc


REF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Glm5NextDsaConfig:
    name: str
    H: int
    Nh: int
    Rq: int
    Rkv: int
    Dnope: int
    Dro: int
    Dqk: int
    Dv: int
    I: int
    Di: int
    Ktop: int
    full_attn_layers: Tuple[int, ...]
    rms_norm_eps: float = 1e-5

    @property
    def scaling(self) -> float:
        return self.Dqk ** -0.5


@dataclass
class GlobalHistory:
    latent_kv: torch.Tensor      # [B, S, Rkv] bf16
    index_body: torch.Tensor     # [B, S, Di] bf16, FP8 integer proxy in bf16 container
    index_scale: torch.Tensor    # [B, S] fp32
    positions: torch.Tensor      # [S] int64


@dataclass
class RankCache:
    rank: int
    cp_size: int
    positions: torch.Tensor      # [S_local] int64
    latent_kv: torch.Tensor      # [B, S_local, Rkv] bf16
    index_body: torch.Tensor     # [B, S_local, Di] bf16
    index_scale: torch.Tensor    # [B, S_local] fp32


@dataclass
class DevContext:
    cfg: Glm5NextDsaConfig
    weights: Dict[str, torch.Tensor]
    history: GlobalHistory
    hidden: torch.Tensor         # [B, H] bf16
    new_pos: int
    block_span: int


def make_config_16b() -> Glm5NextDsaConfig:
    return Glm5NextDsaConfig(
        name="GLM5-Next-16B",
        H=2048,
        Nh=32,
        Rq=768,
        Rkv=512,
        Dnope=128,
        Dro=0,
        Dqk=128,
        Dv=128,
        I=8,
        Di=128,
        Ktop=2048,
        full_attn_layers=(3, 7, 11, 15, 19, 23),
    )


def make_config_next() -> Glm5NextDsaConfig:
    # config.json writes qk_rope_head_dim=64, but the deployed target for this
    # dev slice uses Dro=0. Keep the effective values here.
    return Glm5NextDsaConfig(
        name="GLM5-Next",
        H=4096,
        Nh=64,
        Rq=1536,
        Rkv=512,
        Dnope=192,
        Dro=0,
        Dqk=192,
        Dv=256,
        I=8,
        Di=128,
        Ktop=2048,
        full_attn_layers=(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43),
    )


def select_config(name: str) -> Glm5NextDsaConfig:
    if name == "16b":
        return make_config_16b()
    if name == "next":
        return make_config_next()
    raise ValueError(f"unknown config: {name}")


def owner_for_pos(pos: int, cp_size: int, block_span: int) -> int:
    return (pos // block_span) % cp_size


def init_weights(cfg: Glm5NextDsaConfig, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rn(*shape, scale=0.02, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype)

    w: Dict[str, torch.Tensor] = {}
    w["fused_qkv_a"] = rn(cfg.Rq + cfg.Rkv, cfg.H)
    w["q_a_norm"] = torch.ones(cfg.Rq, dtype=torch.float32)
    w["kv_a_norm"] = torch.ones(cfg.Rkv, dtype=torch.float32)
    w["q_b_proj"] = rn(cfg.Nh * cfg.Dqk, cfg.Rq)
    w["w_kc"] = rn(cfg.Nh, cfg.Dnope, cfg.Rkv)
    w["w_vc"] = rn(cfg.Nh, cfg.Rkv, cfg.Dv)
    w["o_proj"] = rn(cfg.H, cfg.Nh * cfg.Dv)
    w["wq_b"] = rn(cfg.I * cfg.Di, cfg.Rq)
    w["wk_idx"] = rn(cfg.Di, cfg.H)
    w["k_norm_weight"] = torch.ones(cfg.Di, dtype=torch.float32)
    w["k_norm_bias"] = torch.zeros(cfg.Di, dtype=torch.float32)
    w["weights_proj"] = rn(cfg.I, cfg.H)
    # Sylvester ±1 Hadamard matrix for Indexer rotate_activation along Di axis.
    # bf16 ±1 is exact; the 1/sqrt(N) scale is applied at rotate time.
    w["hadamard_Di"] = hadamard_matrix(cfg.Di).to(torch.bfloat16)
    return w


def init_history(cfg: Glm5NextDsaConfig, batch: int, seqlen: int, seed: int) -> GlobalHistory:
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(batch, seqlen, cfg.Rkv, generator=g, dtype=torch.bfloat16) * 0.05
    idx_body = torch.randint(
        low=-32,
        high=33,
        size=(batch, seqlen, cfg.Di),
        generator=g,
        dtype=torch.int16,
    ).to(torch.bfloat16)
    idx_scale = torch.rand(batch, seqlen, generator=g, dtype=torch.float32) * 0.02 + 0.01
    positions = torch.arange(seqlen, dtype=torch.long)
    return GlobalHistory(latent, idx_body, idx_scale, positions)


def partition_history(
    history: GlobalHistory,
    cp_size: int,
    block_span: int,
) -> List[RankCache]:
    ranks: List[RankCache] = []
    for rank in range(cp_size):
        mask = torch.tensor(
            [owner_for_pos(int(p.item()), cp_size, block_span) == rank for p in history.positions],
            dtype=torch.bool,
        )
        ranks.append(
            RankCache(
                rank=rank,
                cp_size=cp_size,
                positions=history.positions[mask].clone(),
                latent_kv=history.latent_kv[:, mask].clone(),
                index_body=history.index_body[:, mask].clone(),
                index_scale=history.index_scale[:, mask].clone(),
            )
        )
    return ranks


def build_context(args) -> DevContext:
    cfg = select_config(args.config)
    torch.manual_seed(args.seed)
    weights = init_weights(cfg, args.seed)
    history = init_history(cfg, args.batch, args.seqlen, args.seed + 1)
    hidden = torch.randn(args.batch, cfg.H, dtype=torch.bfloat16) * 0.05
    return DevContext(
        cfg=cfg,
        weights=weights,
        history=history,
        hidden=hidden,
        new_pos=args.seqlen,
        block_span=args.block_span,
    )


def linear_bf16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x.float(), w.float()).to(torch.bfloat16)


def linear_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x.float(), w.float())


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()
    return out.to(torch.bfloat16)


def layernorm_full(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    out = (xf - mean) * torch.rsqrt(var + eps)
    out = out * weight.float() + bias.float()
    return out.to(torch.bfloat16)


def fake_fp8_row_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp8 e4m3 row-quant proxy (hardware RNE + saturate at ±448).

    Mirrors the sgl-kernel-zeus indexer Q/K quant: per-row amax / 448 fp32 scale,
    body cast to torch.float8_e4m3fn. The act_quant(..., fmt='ue8m0') prod path
    uses the same fp8 dtype with a power-of-two scale; we keep fp32 scale here.
    """
    xf = x.float()
    scale = xf.abs().amax(dim=-1).clamp_min(1e-6) / 448.0
    body = (xf / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return body, scale.float()


def hadamard_matrix(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sylvester-construction Walsh-Hadamard matrix of size n × n (n power of 2).

    Entries are ±1. Normalized H/sqrt(n) is orthogonal. Used by the Indexer's
    `rotate_activation = hadamard_transform(x, n**-0.5)`.
    """
    if n & (n - 1) != 0:
        raise ValueError(f"hadamard_matrix: n must be a power of 2, got {n}")
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H


def hadamard_rotate(x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """y = (x @ H) / sqrt(N)  — matches sglang `rotate_activation` semantics.

    x: [..., N] bf16; H: [N, N] (±1, fp32 or bf16). Output: bf16.
    """
    N = H.shape[0]
    y = x.float().matmul(H.float()) * (N ** -0.5)
    return y.to(torch.bfloat16)


def ref_pre_store(hidden: torch.Tensor, w: Dict[str, torch.Tensor], cfg: Glm5NextDsaConfig):
    qkv = linear_bf16(hidden, w["fused_qkv_a"])
    q_raw, kv_raw = qkv.split([cfg.Rq, cfg.Rkv], dim=-1)
    q_lora = rmsnorm(q_raw, w["q_a_norm"], cfg.rms_norm_eps)
    k_new = rmsnorm(kv_raw, w["kv_a_norm"], cfg.rms_norm_eps).unsqueeze(1)
    return q_lora, k_new


def ref_q_main(q_lora: torch.Tensor, w: Dict[str, torch.Tensor], cfg: Glm5NextDsaConfig):
    B = q_lora.shape[0]
    q = linear_bf16(q_lora, w["q_b_proj"]).view(B, cfg.Nh, cfg.Dqk)
    q_out = torch.bmm(q.transpose(0, 1).float(), w["w_kc"].float()).transpose(0, 1)
    return q_out.to(torch.bfloat16)


def ref_idx_q_weights(
    hidden: torch.Tensor,
    q_lora: torch.Tensor,
    w: Dict[str, torch.Tensor],
    cfg: Glm5NextDsaConfig,
):
    """#2.Q: q_lora + hidden -> q_body, q_scale, weights.

    Real Sylvester ±1 Hadamard rotation along Di — no longer identity proxy.
    """
    B = hidden.shape[0]
    q_idx = linear_bf16(q_lora, w["wq_b"]).view(B, cfg.I, cfg.Di)
    q_idx_rot = hadamard_rotate(q_idx, w["hadamard_Di"])
    q_body, q_scale = fake_fp8_row_quant(q_idx_rot)
    gate = linear_fp32(hidden, w["weights_proj"]) * (cfg.I ** -0.5)
    weights = gate * q_scale * (cfg.Di ** -0.5)
    return q_body, q_scale, weights


def ref_idx_k_prep_store(
    hidden: torch.Tensor,
    w: Dict[str, torch.Tensor],
    cfg: Glm5NextDsaConfig,
):
    """#2.K: hidden -> k_body, k_scale (then scatter to index_k cache).

    Full LayerNorm (mean + var + weight + bias) on the per-token k_idx vector,
    then Sylvester ±1 Hadamard rotation along Di.
    """
    k_idx = linear_bf16(hidden, w["wk_idx"])
    k_idx = layernorm_full(k_idx, w["k_norm_weight"], w["k_norm_bias"], cfg.rms_norm_eps)
    k_idx_rot = hadamard_rotate(k_idx, w["hadamard_Di"])
    k_body, k_scale = fake_fp8_row_quant(k_idx_rot)
    return k_body, k_scale


def append_current(
    ranks: List[RankCache],
    new_pos: int,
    block_span: int,
    k_new: torch.Tensor,
    k_idx_body: torch.Tensor,
    k_idx_scale: torch.Tensor,
):
    owner = owner_for_pos(new_pos, len(ranks), block_span)
    rank = ranks[owner]
    rank.positions = torch.cat([rank.positions, torch.tensor([new_pos], dtype=torch.long)])
    rank.latent_kv = torch.cat([rank.latent_kv, k_new.to(torch.bfloat16)], dim=1)
    rank.index_body = torch.cat([rank.index_body, k_idx_body.unsqueeze(1).to(torch.bfloat16)], dim=1)
    rank.index_scale = torch.cat([rank.index_scale, k_idx_scale.unsqueeze(1).float()], dim=1)


def ref_idx_logits_local(
    q_body: torch.Tensor,
    weights: torch.Tensor,
    rank: RankCache,
) -> torch.Tensor:
    if rank.index_body.shape[1] == 0:
        return torch.empty(q_body.shape[0], 0, dtype=torch.float32)
    raw = torch.einsum("bid,bnd->bin", q_body.float(), rank.index_body.float())
    logits = (raw * weights.unsqueeze(-1).float()).sum(dim=1)
    logits = logits * rank.index_scale.float()
    return logits


def ref_local_topk(
    logits: torch.Tensor,
    positions: torch.Tensor,
    ktop: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N = logits.shape
    out_logits = torch.full((B, ktop), float("-inf"), dtype=torch.float32)
    out_pos = torch.full((B, ktop), -1, dtype=torch.long)
    if N == 0:
        return out_logits, out_pos

    if N <= ktop:
        out_logits[:, :N] = logits
        out_pos[:, :N] = positions.view(1, N).expand(B, N)
        return out_logits, out_pos

    vals, idx = torch.topk(logits, k=ktop, dim=-1)
    out_logits[:] = vals
    out_pos[:] = positions[idx]
    return out_logits, out_pos


def ref_merge_topk(
    local_logits: List[torch.Tensor],
    local_pos: List[torch.Tensor],
    ktop: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cand_logits = torch.cat(local_logits, dim=1)
    cand_pos = torch.cat(local_pos, dim=1)
    B, N = cand_logits.shape
    out_logits = torch.full((B, ktop), float("-inf"), dtype=torch.float32)
    out_pos = torch.full((B, ktop), -1, dtype=torch.long)

    for b in range(B):
        valid = cand_pos[b] >= 0
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue
        vals = cand_logits[b, valid]
        pos = cand_pos[b, valid]
        if n_valid <= ktop:
            out_logits[b, :n_valid] = vals
            out_pos[b, :n_valid] = pos
        else:
            top_vals, top_idx = torch.topk(vals, k=ktop)
            out_logits[b] = top_vals
            out_pos[b] = pos[top_idx]
    return out_logits, out_pos


def ref_sparse_mqa_partial(
    q_new: torch.Tensor,
    topk_pos: torch.Tensor,
    rank: RankCache,
    cfg: Glm5NextDsaConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, Nh, _ = q_new.shape
    partial_out = torch.zeros(B, Nh, cfg.Rkv, dtype=torch.float32)
    partial_lse = torch.full((B, Nh), float("-inf"), dtype=torch.float32)
    pos_to_col = {int(p.item()): i for i, p in enumerate(rank.positions)}

    for b in range(B):
        cols = [pos_to_col[int(p.item())] for p in topk_pos[b] if int(p.item()) in pos_to_col]
        if not cols:
            continue
        kv = rank.latent_kv[b, torch.tensor(cols, dtype=torch.long)].float()
        q = q_new[b].float()
        logits = torch.einsum("hd,nd->hn", q, kv) * cfg.scaling
        partial_lse[b] = torch.logsumexp(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        partial_out[b] = torch.einsum("hn,nd->hd", probs, kv)
    return partial_out, partial_lse


def ref_lse_merge(partial_outs: List[torch.Tensor], partial_lses: List[torch.Tensor]) -> torch.Tensor:
    outs = torch.stack(partial_outs, dim=0).float()
    lses = torch.stack(partial_lses, dim=0).float()
    m = lses.max(dim=0).values
    all_empty = torch.isinf(m)
    safe_m = torch.where(all_empty, torch.zeros_like(m), m)
    weights = torch.exp(lses - safe_m.unsqueeze(0))
    weights = torch.where(torch.isinf(lses), torch.zeros_like(weights), weights)
    z = weights.sum(dim=0).clamp_min(1e-20)
    merged = (weights.unsqueeze(-1) * outs).sum(dim=0) / z.unsqueeze(-1)
    return torch.where(all_empty.unsqueeze(-1), torch.zeros_like(merged), merged)


def ref_post_o_proj(attn_latent: torch.Tensor, w: Dict[str, torch.Tensor], cfg: Glm5NextDsaConfig):
    B = attn_latent.shape[0]
    attn = torch.bmm(attn_latent.transpose(0, 1).float(), w["w_vc"].float()).transpose(0, 1)
    attn = attn.reshape(B, cfg.Nh * cfg.Dv).to(torch.bfloat16)
    return linear_bf16(attn, w["o_proj"])


def run_ref_decode(ctx: DevContext, cp_size: int):
    cfg = ctx.cfg
    ranks = partition_history(ctx.history, cp_size, ctx.block_span)
    q_lora, k_new = ref_pre_store(ctx.hidden, ctx.weights, cfg)
    q_new = ref_q_main(q_lora, ctx.weights, cfg)
    q_body, _q_scale, weights = ref_idx_q_weights(ctx.hidden, q_lora, ctx.weights, cfg)
    k_body, k_scale = ref_idx_k_prep_store(ctx.hidden, ctx.weights, cfg)
    append_current(ranks, ctx.new_pos, ctx.block_span, k_new, k_body, k_scale)

    raw_logits: List[torch.Tensor] = []
    local_topk_logits: List[torch.Tensor] = []
    local_pos: List[torch.Tensor] = []
    for rank in ranks:
        logits = ref_idx_logits_local(q_body, weights, rank)
        vals, pos = ref_local_topk(logits, rank.positions, cfg.Ktop)
        raw_logits.append(logits)
        local_topk_logits.append(vals)
        local_pos.append(pos)
    _, global_pos = ref_merge_topk(local_topk_logits, local_pos, cfg.Ktop)

    partial_outs: List[torch.Tensor] = []
    partial_lses: List[torch.Tensor] = []
    for rank in ranks:
        po, pl = ref_sparse_mqa_partial(q_new, global_pos, rank, cfg)
        partial_outs.append(po)
        partial_lses.append(pl)

    if cp_size == 1:
        attn_latent = partial_outs[0].to(torch.bfloat16)
    else:
        attn_latent = ref_lse_merge(partial_outs, partial_lses).to(torch.bfloat16)
    out = ref_post_o_proj(attn_latent, ctx.weights, cfg)
    return {
        "out": out,
        "attn_latent": attn_latent,
        "global_topk_pos": global_pos,
        "q_lora": q_lora,
        "q_new": q_new,
        "q_body": q_body,
        "weights": weights,
        "ranks": ranks,
        "raw_logits": raw_logits,
        "local_topk_logits": local_topk_logits,
        "local_pos": local_pos,
        "partial_outs": partial_outs,
        "partial_lses": partial_lses,
    }


def compare_tensors(name: str, ref: torch.Tensor, got: torch.Tensor, atol=5e-3, rtol=5e-3) -> bool:
    a = ref.detach().float().cpu()
    b = got.detach().float().cpu()
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: ref={tuple(a.shape)} got={tuple(b.shape)}")
        return False
    diff = (a - b).abs()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print(
        f"  [{name}] {'PASS' if ok else 'DIFF'} | "
        f"max_diff={diff.max().item():.6e} mean_diff={diff.mean().item():.6e} "
        f"shape={list(a.shape)}"
    )
    return ok


def compare_topk_sets(name: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    ok = True
    for row in range(a.shape[0]):
        sa = {int(x.item()) for x in a[row] if int(x.item()) >= 0}
        sb = {int(x.item()) for x in b[row] if int(x.item()) >= 0}
        if sa != sb:
            ok = False
            print(
                f"  [{name}] DIFF row={row} | "
                f"|a|={len(sa)} |b|={len(sb)} |a&b|={len(sa & sb)}"
            )
            break
    if ok:
        print(f"  [{name}] PASS | rows={a.shape[0]}")
    return ok


def zeus_skip(kernel_name: str, anchor: str) -> None:
    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (torch_zeus/sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return
    if not hasattr(sgl_kernel_zeus, kernel_name):
        print(f"  ZEUS: SKIP (sgl_kernel_zeus.{kernel_name} not implemented)")
        print(f"        TODO anchor: {anchor}")
        return
    print(f"  ZEUS: SKIP (sgl_kernel_zeus.{kernel_name} exists, but this stage wrapper is not wired yet)")
    print(f"        TODO anchor: {anchor}")


def finish_stage(args, ref_ok: bool, kernel_name: str, anchor: str) -> Optional[bool]:
    if not ref_ok:
        return False
    if args.mode == "ref":
        return True
    zeus_skip(kernel_name, anchor)
    return None


def skip_stage(reason: str) -> Optional[bool]:
    print(f"  SKIP: {reason}")
    return None


def stage_q_a_proj_norm(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: q_a_proj_norm (#0.Q)")
    print("=" * 60)
    cfg = ctx.cfg
    q_lora_ref, _ = ref_pre_store(ctx.hidden, ctx.weights, cfg)
    ref_ok = q_lora_ref.shape == (args.batch, cfg.Rq)
    print(f"  hidden: {tuple(ctx.hidden.shape)}  q_lora_norm: {tuple(q_lora_ref.shape)}")
    print("  path:   hidden -> q_a_proj -> RMSNorm -> q_lora_out")

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_q_a_proj_norm"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_q_a_proj_norm not yet built)")
        return None

    # The fused QKV-A weight is [Rq + Rkv, H]; row-slice [:Rq] is the Q-A
    # block and remains contiguous bf16. q_a_weight lives in Lmem (the host
    # wrapper detects LocalMem via isLocalMem and materializes for the sim).
    q_a_weight = ctx.weights["fused_qkv_a"][: cfg.Rq].contiguous()
    q_a_norm = ctx.weights["q_a_norm"]
    q_a_weight_lmem = torch.zeus.local_memory.from_tensor(
        q_a_weight.to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    got = sgl_kernel_zeus.dsa_q_a_proj_norm(
        ctx.hidden.to("zeus"),
        q_a_weight_lmem,
        q_a_norm.to("zeus"),
        eps=cfg.rms_norm_eps,
    )
    return compare_tensors(
        "q_lora_norm REF vs Zeus",
        q_lora_ref, got.cpu(),
        atol=2e-2, rtol=1e-2,
    )


def stage_kv_a_proj_norm_store(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: kv_a_proj_norm_store (#0.KV)")
    print("=" * 60)
    cfg = ctx.cfg
    ranks = partition_history(ctx.history, args.cp, ctx.block_span)
    _, k_new_ref = ref_pre_store(ctx.hidden, ctx.weights, cfg)
    dummy_body = torch.zeros(args.batch, cfg.Di, dtype=torch.bfloat16)
    dummy_scale = torch.ones(args.batch, dtype=torch.float32)
    before = [r.latent_kv.shape[1] for r in ranks]
    append_current(ranks, ctx.new_pos, ctx.block_span, k_new_ref, dummy_body, dummy_scale)
    after = [r.latent_kv.shape[1] for r in ranks]
    owner = owner_for_pos(ctx.new_pos, args.cp, ctx.block_span)
    ref_ok = after[owner] == before[owner] + 1 and all(
        after[i] == before[i] for i in range(args.cp) if i != owner
    )
    print(f"  k_new: {tuple(k_new_ref.shape)}  Rkv={cfg.Rkv}")
    print(f"  owner rank for pos={ctx.new_pos}: {owner}")
    print(f"  latent rows before={before} after={after}")
    print("  path:   hidden -> kv_a_proj -> RMSNorm -> latent_kv_cache[slot] in-place")

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_kv_a_proj_norm_store"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_kv_a_proj_norm_store not yet built)")
        return None

    # KV-A weight = fused_qkv_a[Rq : Rq + Rkv]. kv_a_weight is a Lmem weight;
    # latent_kv_cache must stay in Gmem (downstream #7 dsa_latent_k_gather
    # reads it as a Gmem activation pool).
    kv_a_weight = ctx.weights["fused_qkv_a"][cfg.Rq:].contiguous()
    kv_a_norm = ctx.weights["kv_a_norm"]
    # Each decode token writes to its own slot; cache is a fresh flat pool so
    # we can read back deterministically per slot.
    num_slots = args.batch
    slot_mapping = torch.arange(args.batch, dtype=torch.int32)
    cache_z = torch.zeros((num_slots, cfg.Rkv), dtype=torch.bfloat16).to("zeus")
    kv_a_weight_lmem = torch.zeus.local_memory.from_tensor(
        kv_a_weight.to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    sgl_kernel_zeus.dsa_kv_a_proj_norm_store(
        ctx.hidden.to("zeus"),
        kv_a_weight_lmem,
        kv_a_norm.to("zeus"),
        slot_mapping.to("zeus"),
        cache_z,
        eps=cfg.rms_norm_eps,
    )
    got = cache_z.cpu()
    # k_new_ref has shape [batch, 1, Rkv]; collapse middle dim for compare.
    return compare_tensors(
        "latent_kv_cache[slot] REF vs Zeus",
        k_new_ref.squeeze(1), got,
        atol=2e-2, rtol=1e-2,
    )


def stage_q_main(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: q_main (#1)")
    print("=" * 60)
    cfg = ctx.cfg
    q_lora, _ = ref_pre_store(ctx.hidden, ctx.weights, cfg)
    q_new_ref = ref_q_main(q_lora, ctx.weights, cfg)
    ref_ok = q_new_ref.shape == (args.batch, cfg.Nh, cfg.Rkv)
    print(f"  q_new: {tuple(q_new_ref.shape)}  scaling={cfg.scaling:.6f}")
    print("  Dro=0: no q_pe split, no RoPE, q_new is q_nope_out")

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_q_main_absorb"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_q_main_absorb not yet built)")
        return None

    # q_b_weight and w_kc both live in Lmem (weight DRAM).
    q_b_weight = ctx.weights["q_b_proj"].contiguous()
    w_kc = ctx.weights["w_kc"].contiguous()
    q_b_weight_lmem = torch.zeus.local_memory.from_tensor(
        q_b_weight.to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    w_kc_lmem = torch.zeus.local_memory.from_tensor(
        w_kc.to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    got = sgl_kernel_zeus.dsa_q_main_absorb(
        q_lora.to("zeus"),
        q_b_weight_lmem,
        w_kc_lmem,
    )
    return compare_tensors(
        "q_new REF vs Zeus",
        q_new_ref, got.cpu(),
        atol=3e-2, rtol=2e-2,
    )


def stage_idx_q_weights(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: idx_q_weights (#2.Q)")
    print("=" * 60)
    cfg = ctx.cfg
    q_lora, _ = ref_pre_store(ctx.hidden, ctx.weights, cfg)
    q_body, q_scale, weights = ref_idx_q_weights(ctx.hidden, q_lora, ctx.weights, cfg)
    ref_ok = (
        q_body.shape == (args.batch, cfg.I, cfg.Di)
        and q_scale.shape == (args.batch, cfg.I)
        and weights.shape == (args.batch, cfg.I)
    )
    print(f"  q_body:   {tuple(q_body.shape)}  q_scale: {tuple(q_scale.shape)}")
    print(f"  weights:  {tuple(weights.shape)}")
    print("  path:     wq_b -> reshape -> Hadamard(±1)/√Di -> fp8 e4m3 quant; weights = gate · I^-½ · q_scale · Di^-½")

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_indexer_q_weights"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_indexer_q_weights not yet built)")
        return None

    wq_b = ctx.weights["wq_b"].contiguous()
    weights_proj = ctx.weights["weights_proj"].contiguous()
    H_Di = ctx.weights["hadamard_Di"].contiguous()
    q_body_got, q_scale_got, weights_got = sgl_kernel_zeus.dsa_indexer_q_weights(
        q_lora.to("zeus"),
        ctx.hidden.to("zeus"),
        wq_b.to("zeus"),
        H_Di.to("zeus"),
        weights_proj.to("zeus"),
        num_index_heads=cfg.I,
        index_head_dim=cfg.Di,
    )
    # body is fp8 e4m3. PyTorch matmul vs. sim.c naive accumulation can drift
    # <1 ulp in fp32 and occasionally cross one e4m3 rounding boundary;
    # allow one e4m3 ulp (rtol=0.13 ≈ 1/2^3).
    ok_body = compare_tensors(
        "q_body  REF vs Zeus", q_body, q_body_got.cpu(), atol=0.0, rtol=0.13,
    )
    ok_scale = compare_tensors(
        "q_scale REF vs Zeus", q_scale, q_scale_got.cpu(), atol=2e-3, rtol=2e-3,
    )
    ok_weights = compare_tensors(
        "weights REF vs Zeus", weights, weights_got.cpu(), atol=3e-2, rtol=3e-2,
    )
    return ok_body and ok_scale and ok_weights


def stage_idx_k_prep_store(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: idx_k_prep_store (#2.K)")
    print("=" * 60)
    cfg = ctx.cfg
    k_body_ref, k_scale_ref = ref_idx_k_prep_store(ctx.hidden, ctx.weights, cfg)
    ref_ok = (
        k_body_ref.shape == (args.batch, cfg.Di)
        and k_scale_ref.shape == (args.batch,)
    )
    print(f"  k_body:  {tuple(k_body_ref.shape)}  k_scale: {tuple(k_scale_ref.shape)}")
    print("  path:    wk -> full LayerNorm(weight, bias) -> Hadamard(±1)/√Di -> fp8 e4m3 quant -> scatter cache[slot]")

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_indexer_k_prep_store"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_indexer_k_prep_store not yet built)")
        return None

    wk = ctx.weights["wk_idx"].contiguous()
    k_norm_w = ctx.weights["k_norm_weight"]
    k_norm_b = ctx.weights["k_norm_bias"]
    H_Di = ctx.weights["hadamard_Di"].contiguous()
    num_slots = args.batch
    slot_mapping = torch.arange(args.batch, dtype=torch.int32)
    # wk / H_Di / body_cache live in Lmem (Lmem doesn't support fp32, so
    # scale_cache stays on regular Gmem).
    wk_lmem = torch.zeus.local_memory.from_tensor(
        wk.to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    H_Di_lmem = torch.zeus.local_memory.from_tensor(
        H_Di.to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    body_cache_init = torch.zeros(
        (num_slots, cfg.Di), dtype=torch.float8_e4m3fn,
    )
    body_cache = torch.zeus.local_memory.from_tensor(
        body_cache_init.to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    scale_cache = torch.zeros((num_slots,), dtype=torch.float32).to("zeus")
    sgl_kernel_zeus.dsa_indexer_k_prep_store(
        ctx.hidden.to("zeus"),
        wk_lmem,
        k_norm_w.to("zeus"),
        k_norm_b.to("zeus"),
        H_Di_lmem,
        slot_mapping.to("zeus"),
        body_cache,
        scale_cache,
        eps=cfg.rms_norm_eps,
    )
    ok_body = compare_tensors(
        "k_body_cache  REF vs Zeus", k_body_ref, body_cache.cpu(),
        atol=0.0, rtol=0.0,
    )
    ok_scale = compare_tensors(
        "k_scale_cache REF vs Zeus", k_scale_ref, scale_cache.cpu(),
        atol=2e-3, rtol=2e-3,
    )
    return ok_body and ok_scale


def stage_idx_logits(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: idx_logits (#3)")
    print("=" * 60)
    cfg = ctx.cfg
    decoded = run_ref_decode(ctx, args.cp)
    ranks = decoded["ranks"]
    raw_logits = decoded["raw_logits"]
    q_body_ref = decoded["q_body"]
    weights_ref = decoded["weights"]
    ref_ok = True
    for rank, logits in zip(ranks, raw_logits):
        print(f"  rank {rank.rank}: logits {tuple(logits.shape)} positions={rank.positions.numel()}")
        ref_ok &= logits.shape == (args.batch, rank.positions.numel())
    print("  order: Index GEMM -> gate reduce -> k_scale")

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_index_logits"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_index_logits not yet built)")
        return None

    ok = True
    for rank, logits_ref in zip(ranks, raw_logits):
        n_local = rank.positions.numel()
        if n_local == 0:
            # Empty rank: the kernel doesn't run; REF returns [B, 0] tensor.
            continue
        # k_body in RankCache is bf16 integer-valued (history seed); the actual
        # cache produced by #2.K is fp8 e4m3. Convert here so the Zeus kernel
        # gets the same dtype as in prod. k_body lives in Lmem (index_k_body
        # cache populated by kv transfer + #2.K stores); k_scale stays in Gmem.
        k_body_fp8 = rank.index_body.to(torch.float8_e4m3fn).contiguous()
        k_scale_f32 = rank.index_scale.contiguous()
        k_body_lmem = torch.zeus.local_memory.from_tensor(
            k_body_fp8.to("zeus"), kind="weight", Tr=1, Tc=1,
        )
        logits_got = sgl_kernel_zeus.dsa_index_logits(
            q_body_ref.to("zeus"),
            weights_ref.to("zeus"),
            k_body_lmem,
            k_scale_f32.to("zeus"),
        )
        # REF used bf16 k_body; Zeus path uses fp8 k_body. Recompute REF with
        # the same fp8 cast so the comparison is dtype-aligned.
        raw = torch.einsum(
            "bid,bnd->bin",
            q_body_ref.float(),
            k_body_fp8.float(),
        )
        gate = (raw * weights_ref.unsqueeze(-1).float()).sum(dim=1)
        logits_dtype_aligned_ref = gate * k_scale_f32.float()
        ok &= compare_tensors(
            f"rank {rank.rank} logits REF vs Zeus",
            logits_dtype_aligned_ref, logits_got.cpu(),
            atol=2e-3, rtol=2e-3,
        )
    return ok


def stage_local_topk(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: local_topk (#4)")
    print("=" * 60)
    decoded = run_ref_decode(ctx, args.cp)
    ref_ok = True
    for rank, logits, pos in zip(decoded["ranks"], decoded["raw_logits"], decoded["local_pos"]):
        valid = int((pos >= 0).sum(dim=1).max().item()) if pos.numel() else 0
        branch = "direct" if logits.shape[1] <= ctx.cfg.Ktop else "topk"
        print(
            f"  rank {rank.rank}: local_N={logits.shape[1]} branch={branch} "
            f"out={tuple(pos.shape)} max_valid={valid}"
        )
        ref_ok &= pos.shape == (args.batch, ctx.cfg.Ktop)

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_local_topk_radix"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_local_topk_radix not yet built)")
        return None

    cfg = ctx.cfg
    Ktop = cfg.Ktop
    ok = True
    for rank, raw_logits_r, ref_pos_r in zip(
        decoded["ranks"], decoded["raw_logits"], decoded["local_pos"]
    ):
        s_local = rank.positions.numel()
        if s_local == 0:
            # Empty rank: REF puts all -1 in ref_pos_r. Kernel does the same;
            # skip the launch since positions tensor has 0 elements.
            print(f"  rank {rank.rank}: empty (S_local=0) — skipping Zeus call")
            continue

        # Zeus expects fp32 logits + i32 positions. REF logits already fp32;
        # rank.positions is int64 by construction so binding does stable-buffer
        # cast to i32 internally.
        logits_z = raw_logits_r.to("zeus")
        positions_z = rank.positions.to(torch.int32).to("zeus")

        top_lg_got, top_pos_got = sgl_kernel_zeus.dsa_local_topk_radix(
            logits_z, positions_z, Ktop=Ktop,
        )
        top_pos_got_cpu = top_pos_got.cpu().to(torch.int64)

        # REF positions are int64; downstream uses set comparison
        # (compare_topk_sets) which drops -1 padding. Reuse it for parity.
        ok &= compare_topk_sets(
            f"rank {rank.rank} local_topk_pos REF vs Zeus",
            ref_pos_r, top_pos_got_cpu,
        )
    return ok


def stage_cp_topk_merge(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: cp_topk_merge (#5/#6)")
    print("=" * 60)
    cp_decoded = run_ref_decode(ctx, args.cp)
    base_decoded = run_ref_decode(ctx, 1)
    ref_ok = compare_topk_sets(
        "global_topk_pos cp=1 vs cp=k",
        base_decoded["global_topk_pos"],
        cp_decoded["global_topk_pos"],
    )
    print(f"  gathered local candidates: cp={args.cp}, Ktop={ctx.cfg.Ktop}")
    return finish_stage(args, ref_ok, "dsa_cp_merge_topk", "glm5next_dsa_decode_dev.md #5/#6")


def stage_latent_gather(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: latent_gather (#7)")
    print("=" * 60)
    decoded = run_ref_decode(ctx, args.cp)
    topk = decoded["global_topk_pos"]
    ref_ok = True
    for rank in decoded["ranks"]:
        owned = set(int(p.item()) for p in rank.positions)
        counts = []
        for b in range(args.batch):
            counts.append(sum(1 for p in topk[b].tolist() if p in owned))
        print(
            f"  rank {rank.rank}: owned latent rows={rank.latent_kv.shape[1]} "
            f"gather_to_lmem_rows_per_req={counts}"
        )
        ref_ok &= rank.latent_kv.shape[-1] == ctx.cfg.Rkv
    print("  semantics: only topK positions owned by this device are gathered from local Gmem to Lmem; non-owner positions stay invalid")

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_latent_k_gather"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_latent_k_gather not yet built)")
        return None

    cfg = ctx.cfg
    Ktop = cfg.Ktop
    ok = True
    for rank in decoded["ranks"]:
        if rank.positions.numel() == 0:
            # Empty rank: gather is trivially all-zeros; skip the kernel
            # call (zero-extent cache makes the binding contract awkward).
            continue
        # Position-to-slot map for this rank (positions are shared across
        # batches in this dev setup; only the row contents differ per batch).
        pos_to_slot = {int(p.item()): i for i, p in enumerate(rank.positions)}
        for b in range(args.batch):
            slot_list = [pos_to_slot.get(int(p.item()), -1) for p in topk[b]]
            slot_indices_b = torch.tensor(
                slot_list, dtype=torch.int32
            ).unsqueeze(0)  # [1, Ktop]
            cache_b = rank.latent_kv[b].contiguous()  # [S_local, Rkv]

            # REF: K_local zeros at invalid positions; mask = 1 valid / 0 invalid.
            # K_local_T is the pre-transposed view: K_local_T[b,:,k] == K_local[b,k,:].
            K_ref = torch.zeros((1, Ktop, cfg.Rkv), dtype=torch.bfloat16)
            mask_ref = torch.zeros((1, Ktop), dtype=torch.bfloat16)
            valid = slot_indices_b[0] >= 0
            if valid.any():
                K_ref[0, valid] = cache_b[slot_indices_b[0, valid].long()]
                mask_ref[0, valid] = 1.0
            K_T_ref = K_ref.transpose(1, 2).contiguous()  # [1, Rkv, Ktop]

            # Zeus: skip-invalid kernel produces 6 outputs (per-core × 3).
            # Invalid rows of K_local and invalid columns of K_local_T are
            # garbage on both cores; mask drives the comparison. Broadcast
            # invariant: c0 and c1 hold bit-identical valid content.
            (K_c0, K_c1, K_T_c0, K_T_c1, m_c0, m_c1) = \
                sgl_kernel_zeus.dsa_latent_k_gather(
                    slot_indices_b.to("zeus"),
                    cache_b.to("zeus"),
                )
            # Mask: both cores' masks should equal REF exactly.
            ok &= compare_tensors(
                f"rank {rank.rank} b={b} mask_c0 REF vs Zeus",
                mask_ref, m_c0.cpu(), atol=0.0, rtol=0.0,
            )
            ok &= compare_tensors(
                f"rank {rank.rank} b={b} mask_c1 REF vs Zeus",
                mask_ref, m_c1.cpu(), atol=0.0, rtol=0.0,
            )

            def _norm_K(K_t, m_t):
                valid_3d = m_t.cpu().unsqueeze(-1).bool().expand_as(K_t.cpu())
                return torch.where(
                    valid_3d, K_t.cpu(), torch.zeros_like(K_t.cpu())
                )

            def _norm_KT(KT_t, m_t):
                valid_3d_T = m_t.cpu().unsqueeze(1).bool().expand_as(KT_t.cpu())
                return torch.where(
                    valid_3d_T, KT_t.cpu(), torch.zeros_like(KT_t.cpu())
                )

            K_c0_norm   = _norm_K (K_c0,   m_c0)
            K_c1_norm   = _norm_K (K_c1,   m_c1)
            K_T_c0_norm = _norm_KT(K_T_c0, m_c0)
            K_T_c1_norm = _norm_KT(K_T_c1, m_c1)

            # Per-core REF check.
            ok &= compare_tensors(
                f"rank {rank.rank} b={b} K_local_c0 REF vs Zeus",
                K_ref, K_c0_norm, atol=0.0, rtol=0.0,
            )
            ok &= compare_tensors(
                f"rank {rank.rank} b={b} K_local_c1 REF vs Zeus",
                K_ref, K_c1_norm, atol=0.0, rtol=0.0,
            )
            ok &= compare_tensors(
                f"rank {rank.rank} b={b} K_local_T_c0 REF vs Zeus",
                K_T_ref, K_T_c0_norm, atol=0.0, rtol=0.0,
            )
            ok &= compare_tensors(
                f"rank {rank.rank} b={b} K_local_T_c1 REF vs Zeus",
                K_T_ref, K_T_c1_norm, atol=0.0, rtol=0.0,
            )
            # Cross-check the dual-output invariant on each core.
            ok &= compare_tensors(
                f"rank {rank.rank} b={b} K_local_T_c0 == K_c0.transpose() (valid)",
                K_c0_norm.transpose(1, 2).contiguous(), K_T_c0_norm,
                atol=0.0, rtol=0.0,
            )
            # Cross-check the broadcast invariant (c0 == c1 at valid positions).
            ok &= compare_tensors(
                f"rank {rank.rank} b={b} K_local broadcast (c0 == c1, valid)",
                K_c0_norm, K_c1_norm, atol=0.0, rtol=0.0,
            )
    return ok


def stage_sparse_mqa_partial(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: sparse_mqa_partial (#8)")
    print("=" * 60)
    decoded = run_ref_decode(ctx, args.cp)
    ref_ok = True
    for rank, po, pl in zip(decoded["ranks"], decoded["partial_outs"], decoded["partial_lses"]):
        empty_heads = int(torch.isinf(pl).sum().item())
        print(
            f"  rank {rank.rank}: partial_out={tuple(po.shape)} "
            f"partial_lse={tuple(pl.shape)} empty_heads={empty_heads}"
        )
        ref_ok &= po.shape == (args.batch, ctx.cfg.Nh, ctx.cfg.Rkv)
        ref_ok &= pl.shape == (args.batch, ctx.cfg.Nh)

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_sparse_mqa_partial"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_sparse_mqa_partial not yet built)")
        return None

    cfg = ctx.cfg
    topk = decoded["global_topk_pos"]
    q_new = decoded["q_new"]            # [B, Nh, Rkv] bf16
    scaling = cfg.scaling
    ok = True
    for rank, po_ref, pl_ref in zip(
        decoded["ranks"], decoded["partial_outs"], decoded["partial_lses"]
    ):
        # Build K_local + mask for this rank: lookup each topK position in
        # rank.positions, gather from rank.latent_kv; mask=1 if owned else 0.
        pos_to_slot = {int(p.item()): i for i, p in enumerate(rank.positions)}
        K_local = torch.zeros(
            (args.batch, cfg.Ktop, cfg.Rkv), dtype=torch.bfloat16
        )
        mask = torch.zeros((args.batch, cfg.Ktop), dtype=torch.bfloat16)
        for b in range(args.batch):
            for k, p_tensor in enumerate(topk[b]):
                p = int(p_tensor.item())
                slot = pos_to_slot.get(p, -1)
                if slot >= 0:
                    K_local[b, k] = rank.latent_kv[b, slot]
                    mask[b, k] = 1.0
        # Sanity: rank with zero owned positions across both batches → empty.
        if rank.positions.numel() == 0 and bool(mask.any()):
            print(f"  rank {rank.rank}: WARN mask non-zero on empty cache")

        # Build the pre-transposed K_local_T view that #8 expects as a
        # second input — same data, transposed in Lmem so the score GEMM
        # `Q @ K^T` doesn't need a runtime tl.trans.
        K_local_T = K_local.transpose(1, 2).contiguous()  # [B, Rkv, Ktop]

        # #8 now expects 6 per-core Lmem inputs (k_local × 2 cores ×
        # 2 layouts + mask × 2 cores). Broadcast the same data to both
        # banks — that mirrors what #7 dsa_latent_k_gather emits.
        def _lmem(t):
            return torch.zeus.local_memory.from_tensor(
                t.to("zeus"), kind="weight", Tr=1, Tc=1,
            )

        po_got, pl_got = sgl_kernel_zeus.dsa_sparse_mqa_partial(
            q_new.to("zeus"),
            _lmem(K_local),   _lmem(K_local),       # c0, c1: natural
            _lmem(K_local_T), _lmem(K_local_T),     # c0, c1: transposed
            _lmem(mask),      _lmem(mask),          # c0, c1: mask
            scaling=scaling,
        )
        # partial_out: bf16 tolerance ~8e-3
        ok &= compare_tensors(
            f"rank {rank.rank} partial_out REF vs Zeus",
            po_ref.to(torch.bfloat16),
            po_got.cpu(),
            atol=1e-2, rtol=1e-2,
        )
        # partial_lse: handle -inf rows separately (bit-exact for empty heads).
        ref_finite = ~torch.isinf(pl_ref)
        got_finite = ~torch.isinf(pl_got.cpu())
        if not torch.equal(ref_finite, got_finite):
            print(
                f"  rank {rank.rank} lse: empty-head mask MISMATCH "
                f"ref={int(ref_finite.sum())} got={int(got_finite.sum())}"
            )
            ok = False
            continue
        # Compare only finite entries.
        ref_f = pl_ref[ref_finite]
        got_f = pl_got.cpu()[ref_finite]
        ok &= compare_tensors(
            f"rank {rank.rank} partial_lse finite REF vs Zeus",
            ref_f, got_f,
            atol=5e-3, rtol=5e-3,
        )
    return ok


def stage_post_o_proj_nocp(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: post_o_proj_nocp (#10)")
    print("=" * 60)
    cfg = ctx.cfg
    decoded = run_ref_decode(ctx, 1)
    out_ref = decoded["out"]
    attn_latent = decoded["attn_latent"]    # [B, Nh, Rkv] bf16
    ref_ok = out_ref.shape == (args.batch, cfg.H)
    print(f"  attn_latent: {tuple(attn_latent.shape)}")
    print(f"  out:         {tuple(out_ref.shape)}")
    print("  path:        no CP, no FA-reduce — V absorb (w_vc) + o_proj")

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    if not hasattr(sgl_kernel_zeus, "dsa_post_o_proj_no_cp"):
        print("  ZEUS: SKIP (sgl_kernel_zeus.dsa_post_o_proj_no_cp not yet built)")
        return None

    w_vc = ctx.weights["w_vc"].contiguous()
    o_proj_w = ctx.weights["o_proj"].contiguous()
    got = sgl_kernel_zeus.dsa_post_o_proj_no_cp(
        attn_latent.to("zeus"),
        w_vc.to("zeus"),
        o_proj_w.to("zeus"),
    )
    # bf16 round between V absorb and o_proj plus a 4096-wide o_proj accum
    # ⇒ tolerance ~3e-2 (matches tests/test_dsa_post_o_proj_no_cp.py).
    return compare_tensors(
        "out REF vs Zeus",
        out_ref, got.cpu(),
        atol=3e-2, rtol=2e-2,
    )


def stage_post_o_proj_cp(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: post_o_proj_cp (#11)")
    print("=" * 60)
    if args.cp <= 1:
        return skip_stage("post_o_proj_cp requires --cp > 1")

    decoded = run_ref_decode(ctx, args.cp)
    out = decoded["out"]
    ref_ok = out.shape == (args.batch, ctx.cfg.H)
    print(f"  cp:          {args.cp}")
    print(f"  attn_latent: {tuple(decoded['attn_latent'].shape)}")
    print(f"  out:         {tuple(out.shape)}")
    print("  path:        partial all-gather + FA-reduce + V absorb + o_proj")
    return finish_stage(args, ref_ok, "dsa_fa_reduce_o_proj_cp", "glm5next_dsa_decode_dev.md #11")


def stage_decode_full_nocp(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: decode_full_nocp (#0..#10)")
    print("=" * 60)
    cfg = ctx.cfg
    decoded = run_ref_decode(ctx, 1)
    out_ref = decoded["out"]
    ref_ok = (
        out_ref.shape == (args.batch, cfg.H)
        and decoded["attn_latent"].shape == (args.batch, cfg.Nh, cfg.Rkv)
        and decoded["global_topk_pos"].shape == (args.batch, cfg.Ktop)
    )
    print(f"  topk_pos:    {tuple(decoded['global_topk_pos'].shape)}")
    print(f"  attn_latent: {tuple(decoded['attn_latent'].shape)}")
    print(f"  out:         {tuple(out_ref.shape)}")
    print("  path:        CP=1 local topK + sparse MQA + no-CP post")

    if not ref_ok:
        return False
    if args.mode == "ref":
        return True

    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
        return None
    needed = (
        "dsa_q_a_proj_norm", "dsa_kv_a_proj_norm_store", "dsa_q_main_absorb",
        "dsa_indexer_q_weights", "dsa_indexer_k_prep_store", "dsa_index_logits",
        "dsa_local_topk_radix", "dsa_latent_k_gather", "dsa_sparse_mqa_partial",
        "dsa_post_o_proj_no_cp",
    )
    missing = [n for n in needed if not hasattr(sgl_kernel_zeus, n)]
    if missing:
        print(f"  ZEUS: SKIP (missing kernels: {missing})")
        return None

    def lmem(t: torch.Tensor):
        return torch.zeus.local_memory.from_tensor(
            t.to("zeus"), kind="weight", Tr=1, Tc=1,
        )

    q_a_w = ctx.weights["fused_qkv_a"][: cfg.Rq].contiguous()
    kv_a_w = ctx.weights["fused_qkv_a"][cfg.Rq:].contiguous()
    q_b_w = ctx.weights["q_b_proj"].contiguous()
    w_kc = ctx.weights["w_kc"].contiguous()
    w_vc = ctx.weights["w_vc"].contiguous()
    o_proj_w = ctx.weights["o_proj"].contiguous()
    wq_b = ctx.weights["wq_b"].contiguous()
    wk_idx = ctx.weights["wk_idx"].contiguous()
    H_Di = ctx.weights["hadamard_Di"].contiguous()
    weights_proj = ctx.weights["weights_proj"].contiguous()
    hidden_z = ctx.hidden.to("zeus")
    slot_mapping = torch.arange(args.batch, dtype=torch.int32).to("zeus")

    # #0.Q: hidden -> q_a_proj -> RMSNorm -> q_lora
    q_lora_z = sgl_kernel_zeus.dsa_q_a_proj_norm(
        hidden_z, lmem(q_a_w), ctx.weights["q_a_norm"].to("zeus"),
        eps=cfg.rms_norm_eps,
    )

    # #0.KV: fresh slot pool holds only this step's K; historical rows live
    # in ctx.history.latent_kv and get concatenated below.
    kv_new_cache = torch.zeros((args.batch, cfg.Rkv), dtype=torch.bfloat16).to("zeus")
    sgl_kernel_zeus.dsa_kv_a_proj_norm_store(
        hidden_z, lmem(kv_a_w), ctx.weights["kv_a_norm"].to("zeus"),
        slot_mapping, kv_new_cache, eps=cfg.rms_norm_eps,
    )

    # #1: q_b_proj + absorb bmm(w_kc) -> q_new
    q_new_z = sgl_kernel_zeus.dsa_q_main_absorb(q_lora_z, lmem(q_b_w), lmem(w_kc))

    # #2.Q: q_body (fp8) / q_scale / weights
    q_body_z, _q_scale_z, weights_z = sgl_kernel_zeus.dsa_indexer_q_weights(
        q_lora_z, hidden_z, wq_b.to("zeus"), H_Di.to("zeus"),
        weights_proj.to("zeus"),
        num_index_heads=cfg.I, index_head_dim=cfg.Di,
    )

    # #2.K: fresh slot pool for the new step's k_idx body (fp8) + scale (fp32)
    body_cache = lmem(torch.zeros((args.batch, cfg.Di), dtype=torch.float8_e4m3fn))
    scale_cache = torch.zeros((args.batch,), dtype=torch.float32).to("zeus")
    sgl_kernel_zeus.dsa_indexer_k_prep_store(
        hidden_z, lmem(wk_idx),
        ctx.weights["k_norm_weight"].to("zeus"),
        ctx.weights["k_norm_bias"].to("zeus"),
        lmem(H_Di), slot_mapping, body_cache, scale_cache,
        eps=cfg.rms_norm_eps,
    )

    # Combine history rows + new-token row into the cp=1 rank's caches. The
    # historical bf16 index_body is cast to fp8 to match the prod indexer
    # cache dtype (same convention as stage_idx_logits).
    new_k = kv_new_cache.cpu().unsqueeze(1)
    new_body = body_cache.cpu().unsqueeze(1)
    new_scale = scale_cache.cpu().unsqueeze(1)
    full_latent = torch.cat([ctx.history.latent_kv, new_k], dim=1)
    full_body = torch.cat(
        [ctx.history.index_body.to(torch.float8_e4m3fn), new_body], dim=1,
    )
    full_scale = torch.cat([ctx.history.index_scale, new_scale], dim=1)
    S_full = full_latent.shape[1]

    # #3: index GEMM -> gate reduce -> k_scale
    logits_z = sgl_kernel_zeus.dsa_index_logits(
        q_body_z, weights_z, lmem(full_body), full_scale.to("zeus"),
    )

    # #4: local top-K. cp=1 -> this IS the global topK (no #5/#6 merge step).
    positions_z = torch.arange(S_full, dtype=torch.int32).to("zeus")
    _top_lg_z, top_pos_z = sgl_kernel_zeus.dsa_local_topk_radix(
        logits_z, positions_z, Ktop=cfg.Ktop,
    )
    top_pos = top_pos_z.cpu().to(torch.int64)

    # #7: gather K_local + K_local_T + mask per batch into per-core Lmem
    # banks. position == slot for cp=1, so we feed top_pos straight in;
    # -1 sentinels are treated as invalid. The kernel now produces 6
    # outputs (× 2 cores broadcast). For this end-to-end chain we collapse
    # the per-core outputs into one canonical view (c0 == c1 valid content
    # by the gather contract) and re-broadcast that into the 6-input #8
    # call below.
    #
    # 2026-05-25 pool 契约：K_local / K_local_T / mask buffer 走 `torch.zeros`
    # 初始化（layer-pool 模拟），gather Python API 也已经把默认 _alloc 改成
    # `torch.zeros`（见 `dsa_latent_k_gather` docstring "Pool contract"），
    # 所以这里的 per-batch gather 输出 invalid 位置必为 0（finite）。下游 #8
    # sparse_mqa_partial 的算术 mask `score*0 + 1*NEG_LARGE` 干净退到 NEG_LARGE，
    # 无需 host 端 `torch.where(mask, K, zeros)` 再清理一次。
    K_local_full   = torch.zeros(args.batch, cfg.Ktop, cfg.Rkv, dtype=torch.bfloat16)
    K_local_T_full = torch.zeros(args.batch, cfg.Rkv, cfg.Ktop, dtype=torch.bfloat16)
    mask_full      = torch.zeros(args.batch, cfg.Ktop,           dtype=torch.bfloat16)
    for b in range(args.batch):
        slot_idx_b = top_pos[b].to(torch.int32).unsqueeze(0)
        (K_c0_b, _K_c1_b, K_T_c0_b, _K_T_c1_b, m_c0_b, _m_c1_b) = \
            sgl_kernel_zeus.dsa_latent_k_gather(
                slot_idx_b.to("zeus"), full_latent[b].contiguous().to("zeus"),
            )
        # Read core-0 banks (c1 is bit-identical content by the broadcast
        # invariant; we just need one canonical copy here for downstream
        # re-broadcast). Gather Python API auto-allocs with `torch.zeros` so
        # invalid rows/columns are 0.0 bf16 directly — no host cleanup needed.
        K_local_full[b]   = K_c0_b.cpu()[0]
        K_local_T_full[b] = K_T_c0_b.cpu()[0]
        mask_full[b]      = m_c0_b.cpu()[0]

    # #8: sparse MQA partial. cp=1 -> partial_out IS attn_latent (no LSE merge).
    # Each core needs its own Lmem copy of (K_local, K_local_T, mask); we
    # re-broadcast the canonical view into both banks.
    po_z, _pl_z = sgl_kernel_zeus.dsa_sparse_mqa_partial(
        q_new_z,
        lmem(K_local_full),   lmem(K_local_full),
        lmem(K_local_T_full), lmem(K_local_T_full),
        lmem(mask_full),      lmem(mask_full),
        scaling=cfg.scaling,
    )
    # partial_out 已经是 bf16（见 dsa_sparse_mqa_partial host wrapper），
    # 不再做 D→H→D round-trip 的 dtype cast。
    attn_latent_z = po_z

    # #10: V absorb + o_proj
    out_z = sgl_kernel_zeus.dsa_post_o_proj_no_cp(
        attn_latent_z, w_vc.to("zeus"), o_proj_w.to("zeus"),
    )
    return compare_tensors(
        "out REF vs Zeus (decode_full_nocp)",
        out_ref, out_z.cpu(),
        atol=5e-2, rtol=3e-2,
    )


def stage_decode_full_cp(args, ctx: DevContext) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: decode_full_cp (#0..#11)")
    print("=" * 60)
    if args.cp <= 1:
        return skip_stage("decode_full_cp requires --cp > 1")

    base = run_ref_decode(ctx, 1)
    cp = run_ref_decode(ctx, args.cp)
    ok_lat = compare_tensors(
        "attn_latent cp=1 vs cp=k",
        base["attn_latent"],
        cp["attn_latent"],
        atol=3e-3,
        rtol=3e-3,
    )
    ok_out = compare_tensors(
        "out cp=1 vs cp=k",
        base["out"],
        cp["out"],
        atol=8e-3,
        rtol=8e-3,
    )
    ok_topk = compare_topk_sets(
        "topk set cp=1 vs cp=k",
        base["global_topk_pos"],
        cp["global_topk_pos"],
    )
    return finish_stage(args, ok_lat and ok_out and ok_topk, "dsa_decode_full_cp", "glm5next_dsa_decode_dev.md #0..#11")


STAGES = {
    "q_a_proj_norm": stage_q_a_proj_norm,
    "kv_a_proj_norm_store": stage_kv_a_proj_norm_store,
    "q_main": stage_q_main,
    "idx_q_weights": stage_idx_q_weights,
    "idx_k_prep_store": stage_idx_k_prep_store,
    "idx_logits": stage_idx_logits,
    "local_topk": stage_local_topk,
    "cp_topk_merge": stage_cp_topk_merge,
    "latent_gather": stage_latent_gather,
    "sparse_mqa_partial": stage_sparse_mqa_partial,
    "post_o_proj_nocp": stage_post_o_proj_nocp,
    "post_o_proj_cp": stage_post_o_proj_cp,
    "decode_full_nocp": stage_decode_full_nocp,
    "decode_full_cp": stage_decode_full_cp,
}


def main():
    parser = argparse.ArgumentParser(description="GLM5-Next DSA Decode REF-vs-Zeus dev script")
    parser.add_argument("--stage", default="all", choices=["all"] + list(STAGES.keys()))
    parser.add_argument("--mode", default="both", choices=["ref", "zeus", "both"])
    parser.add_argument("--config", default="16b", choices=["16b", "next"])
    parser.add_argument("--cp", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument(
        "--block-span",
        type=int,
        default=1,
        help="number of consecutive token positions assigned before round-robin owner changes",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.cp < 1:
        raise ValueError("--cp must be >= 1")
    if args.block_span < 1:
        raise ValueError("--block-span must be >= 1")
    if args.seqlen < 1:
        raise ValueError("--seqlen must be >= 1")

    ctx = build_context(args)
    print(f"Config: {ctx.cfg.name}  H={ctx.cfg.H} Nh={ctx.cfg.Nh} Rq={ctx.cfg.Rq}")
    print(f"        Rkv={ctx.cfg.Rkv} Dqk={ctx.cfg.Dqk} Dv={ctx.cfg.Dv} Dro={ctx.cfg.Dro}")
    print(f"        I={ctx.cfg.I} Di={ctx.cfg.Di} Ktop={ctx.cfg.Ktop} layers={len(ctx.cfg.full_attn_layers)}")
    print(f"Run:    B={args.batch} seqlen={args.seqlen} cp={args.cp} block_span={args.block_span}")
    print(f"Mode:   {args.mode}  REF_DEVICE={REF_DEVICE}")
    if ZEUS_IMPORT_ERROR is not None:
        print(f"Zeus:   unavailable ({ZEUS_IMPORT_ERROR})")
    else:
        print("Zeus:   import ok; unsupported DSA kernels will be skipped")

    stage_names = list(STAGES) if args.stage == "all" else [args.stage]
    results: Dict[str, Optional[bool]] = {}
    for name in stage_names:
        try:
            results[name] = STAGES[name](args, ctx)
        except Exception as exc:
            print(f"  [{name}] EXCEPTION: {exc}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        if ok is True:
            status = "PASS"
        elif ok is False:
            status = "FAIL"
        else:
            status = "SKIP (REF PASS; Zeus TODO)"
        print(f"  {name:22s} : {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
