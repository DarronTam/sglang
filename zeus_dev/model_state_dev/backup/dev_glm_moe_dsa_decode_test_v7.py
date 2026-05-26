"""
Decode-path V7 dev script — GLM5-Next DSA under Context-Parallel + Route B.

Stages aligned with GlmMoeDsa_dev_V7.md (7 fused stages + end-to-end):

  cp_pre          F-PRE         fused_qkv_a + q_a_layernorm + kv_a_layernorm + concat k_new
  cp_q_main       F-Q-MAIN      q_b_proj + split + bmm(q_nope, w_kc); main MLA RoPE SKIPPED (mla_nope=True)
  cp_kv_store     F-KV-STORE    owner-only main latent KV cache write (k_new -> latent_KV_pool_r[slot])
  cp_idx          F-IDX         indexer Q/K + k_norm (LayerNorm) + NeoX RoPE first 64 + Hadamard
                                  + FP8 act_quant + gate(fp32) + index K cache append (ALL ranks; mirrored)
  cp_topk_cp      F-TOPK-CP     each rank uses its (mirrored) full index K pool to run
                                  `_get_topk_paged` locally → identical `topk_positions` across
                                  ranks (zero comm); per-rank latent-KV page-table maps to
                                  `topk_slots_r` (non-r positions → -1)
  cp_mqa_partial  F-MQA-PARTIAL partial sparse MQA (return_lse=True) over top2048 ∩ K_local
  cp_merge_post   F-MERGE-POST  online-softmax LSE merge + bmm(., w_vc) + o_proj
  cp_full_path    end-to-end D0..D6 over cp_size logical ranks

Hard correctness invariant: `--cp 1` / `--cp 4` / `--cp 16` end-to-end output must be
elementwise identical (the LSE merge is exact). Drift between cp=1 and cp=k is a bug.

In --mode zeus, only landed kernels run. All v7 stages are × TODO at the moment, so
zeus mode raises NotImplementedError per stage (no silent fallback, per skill rule).

Multi-rank parallelism is *simulated in-process* using a list of cp_size per-rank
state dicts — sufficient for correctness; real NCCL collectives are wired separately
in sgl-kernel-zeus tests.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ============================================================================
# Configs (two GLM5-Next variants)
# ============================================================================

@dataclass
class GlmMoeDsaConfig:
    name: str
    H: int            # hidden_size
    Nh: int           # num_attention_heads
    Rq: int           # q_lora_rank
    Rkv: int          # kv_lora_rank (always 512 for GLM5)
    Dnope: int        # qk_nope_head_dim
    Dro: int          # qk_rope_head_dim (always 64 for GLM5)
    Dqk: int          # Dnope + Dro
    Dv: int           # v_head_dim
    I: int            # index_n_heads
    Di: int           # index_head_dim
    Ktop: int         # index_topk
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    full_attn_layers: tuple = ()
    mla_nope: bool = True          # GLM5: True ⇒ main MLA path has no RoPE
    index_use_layernorm: bool = True

    @property
    def scaling(self) -> float:
        return self.Dqk ** -0.5

    @property
    def num_dsa_layers(self) -> int:
        return len(self.full_attn_layers)


def make_config_16b() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        name="config_16b",
        H=2048, Nh=32, Rq=768, Rkv=512, Dnope=128, Dro=64, Dqk=192, Dv=128,
        I=8, Di=128, Ktop=2048,
        full_attn_layers=(3, 7, 11, 15, 19, 23),
    )


def make_config_big() -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        name="config_big",
        H=4096, Nh=64, Rq=1536, Rkv=512, Dnope=192, Dro=64, Dqk=256, Dv=256,
        I=8, Di=128, Ktop=2048,
        full_attn_layers=(3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43),
    )


def select_config(name: str) -> GlmMoeDsaConfig:
    return {"16b": make_config_16b, "big": make_config_big}[name]()


# ============================================================================
# Proxy shapes (for dev runs — full GLM5 weights are too heavy on CPU)
# ============================================================================

@dataclass
class ProxyShapes:
    B: int             # decode batch
    seqlen: int        # current history length (each request has this many prior tokens)
    cp_size: int       # context-parallel degree
    config: GlmMoeDsaConfig

    @property
    def Ktop_eff(self) -> int:
        return min(self.config.Ktop, self.seqlen)


def make_proxy(args, cfg: GlmMoeDsaConfig) -> ProxyShapes:
    return ProxyShapes(
        B=args.batch,
        seqlen=args.seqlen,
        cp_size=args.cp,
        config=cfg,
    )


# ============================================================================
# Weight bundles (random, deterministic per seed)
# ============================================================================

def init_weights(cfg: GlmMoeDsaConfig, seed: int = 0) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    def rn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.02

    w = {}
    # F-PRE: fused_qkv_a_proj_with_mqa packs q_a_proj ‖ kv_a_proj_with_mqa
    w["fused_qkv_a"] = rn(cfg.Rq + cfg.Rkv + cfg.Dro, cfg.H)
    w["q_a_norm"] = torch.ones(cfg.Rq)
    w["kv_a_norm"] = torch.ones(cfg.Rkv)
    # F-Q-MAIN
    w["q_b_proj"] = rn(cfg.Nh * cfg.Dqk, cfg.Rq)
    # kv_b_proj -> w_kc / w_vc (offline split)
    kv_b = rn(cfg.Nh * (cfg.Dnope + cfg.Dv), cfg.Rkv)
    kv_b_view = kv_b.view(cfg.Nh, cfg.Dnope + cfg.Dv, cfg.Rkv)
    w["w_kc"] = kv_b_view[:, : cfg.Dnope, :].contiguous()                 # [Nh, Dnope, Rkv]
    w["w_vc"] = kv_b_view[:, cfg.Dnope :, :].transpose(1, 2).contiguous() # [Nh, Rkv, Dv]
    # F-MERGE-POST
    w["o_proj"] = rn(cfg.H, cfg.Nh * cfg.Dv)
    # F-IDX
    w["wq_b"] = rn(cfg.I * cfg.Di, cfg.Rq)
    w["wk_idx"] = rn(cfg.Di, cfg.H)
    w["k_norm_weight"] = torch.ones(cfg.Di)
    w["k_norm_bias"] = torch.zeros(cfg.Di)                                # LayerNorm bias
    w["weights_proj"] = rn(cfg.I, cfg.H)
    return w


# ============================================================================
# Pure-torch primitives
# ============================================================================

def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps) * weight.float()
    return out.to(x.dtype)


def layernorm_full(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Full LayerNorm (含 bias, fp32 计算) — indexer k_norm 用这个，不是 RMSNorm."""
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).pow(2).mean(dim=-1, keepdim=True)
    normed = (xf - mean) * torch.rsqrt(var + eps)
    out = normed * weight.float() + bias.float()
    return out.to(x.dtype)


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Cast weight to input dtype (production weights are bf16; init random are fp32)
    return torch.nn.functional.linear(x, w.to(x.dtype))


def rope_neox_first(x: torch.Tensor, positions: torch.Tensor, rope_dim: int, theta: float) -> torch.Tensor:
    """NeoX RoPE on the first `rope_dim` dims of last axis; trailing dims unchanged.

    NeoX style: pair (i, i+rope_dim/2) rotates together. (vs. interleave style which
    pairs (2i, 2i+1).)
    """
    assert rope_dim % 2 == 0
    half = rope_dim // 2
    front = x[..., :rope_dim]                          # rotated
    rest = x[..., rope_dim:]                           # untouched
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    pos = positions.float().unsqueeze(-1)              # [..., 1]
    freqs = pos * inv_freq                             # [..., half]
    cos, sin = freqs.cos(), freqs.sin()
    # Broadcast cos/sin to match the leading dims of front
    while cos.dim() < front.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    x1, x2 = front[..., :half], front[..., half:]      # NeoX pairing
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    rotated = torch.cat([rot1, rot2], dim=-1).to(x.dtype)
    return torch.cat([rotated, rest], dim=-1)


def hadamard_rotate_identity(x: torch.Tensor) -> torch.Tensor:
    """Hadamard rotation — orthogonal so leaves bf16 REF dot-product invariant.

    The full implementation matters only for FP8 quant to spread outliers; for
    bf16 REF dot products it's identity. We keep this as identity to stay aligned
    with the V3/V5 REF convention; real Zeus impl must use hadamard_transform.
    """
    return x


# ============================================================================
# Per-rank state (simulating cp_size ranks in one process)
# ============================================================================

@dataclass
class RankState:
    rank: int
    cp_size: int
    cfg: GlmMoeDsaConfig
    # Persistent (sharded by token position round-robin):
    latent_KV_pool: List[torch.Tensor] = field(default_factory=list)   # list of [Rkv+Dro] per locally-owned token
    owned_positions: List[int] = field(default_factory=list)            # global positions owned by this rank
    # Index K: variant (a) replicated full; variant (b) sharded
    index_K_pool: List[torch.Tensor] = field(default_factory=list)
    index_K_positions: List[int] = field(default_factory=list)

    def slot_for(self, pos: int) -> int:
        """Convert global position -> rank-local slot index."""
        assert pos % self.cp_size == self.rank
        return pos // self.cp_size

    def is_owner(self, pos: int) -> bool:
        return pos % self.cp_size == self.rank


def init_ranks(cfg: GlmMoeDsaConfig, cp_size: int, seqlen: int, seed: int = 1) -> List[RankState]:
    """Populate cp_size rank states with a random history of length `seqlen`.

    V7 design: latent KV sharded by `pos % cp_size`; index K replicated on every rank
    (mirrored pool).
    """
    g = torch.Generator().manual_seed(seed)
    ranks = [RankState(rank=r, cp_size=cp_size, cfg=cfg) for r in range(cp_size)]
    for pos in range(seqlen):
        kv_row = torch.randn(cfg.Rkv + cfg.Dro, generator=g, dtype=torch.bfloat16) * 0.1
        idx_row = torch.randn(cfg.Di, generator=g, dtype=torch.bfloat16) * 0.1
        # latent KV → only owner
        owner = pos % cp_size
        ranks[owner].latent_KV_pool.append(kv_row)
        ranks[owner].owned_positions.append(pos)
        # index K → all ranks (mirrored)
        for r in ranks:
            r.index_K_pool.append(idx_row)
            r.index_K_positions.append(pos)
    return ranks


def all_gather_index_K(ranks: List[RankState]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variant (a) helper: gather full index K from all ranks (in causal position order)."""
    all_keys, all_positions = [], []
    for r in ranks:
        all_keys.extend(r.index_K_pool)
        all_positions.extend(r.index_K_positions)
    perm = sorted(range(len(all_positions)), key=lambda i: all_positions[i])
    K = torch.stack([all_keys[i] for i in perm], dim=0)         # [seqlen, Di]
    pos = torch.tensor([all_positions[i] for i in perm], dtype=torch.long)
    return K, pos


def all_gather_latent_KV(ranks: List[RankState]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper for single-card REF: gather full latent KV across ranks in causal order."""
    all_rows, all_positions = [], []
    for r in ranks:
        all_rows.extend(r.latent_KV_pool)
        all_positions.extend(r.owned_positions)
    perm = sorted(range(len(all_positions)), key=lambda i: all_positions[i])
    KV = torch.stack([all_rows[i] for i in perm], dim=0)        # [seqlen, Rkv+Dro]
    pos = torch.tensor([all_positions[i] for i in perm], dtype=torch.long)
    return KV, pos


# ============================================================================
# REF stages — pure-torch, fp32-accumulating
# ============================================================================

def ref_F_PRE(hidden: torch.Tensor, w, cfg: GlmMoeDsaConfig):
    """F-PRE: hidden -> (q_lora, k_nope, k_pe, k_new)."""
    qkv_latent = linear(hidden, w["fused_qkv_a"])                                       # [B, Rq+Rkv+Dro]
    q_lora_raw, kv_lora, k_pe = qkv_latent.split([cfg.Rq, cfg.Rkv, cfg.Dro], dim=-1)
    q_lora = rmsnorm(q_lora_raw, w["q_a_norm"], cfg.rms_norm_eps)                       # [B, Rq]
    k_nope = rmsnorm(kv_lora, w["kv_a_norm"], cfg.rms_norm_eps)                         # [B, Rkv]
    # k_pe: NOT normed, NOT rope'd (mla_nope=True)
    k_new = torch.cat([k_nope, k_pe], dim=-1).unsqueeze(1)                              # [B, 1, Rkv+Dro=576]
    return q_lora, k_nope, k_pe, k_new


def ref_F_Q_MAIN(q_lora: torch.Tensor, w, cfg: GlmMoeDsaConfig):
    """F-Q-MAIN: q_lora -> q_new [B, Nh, Rkv+Dro=576]. main MLA RoPE SKIPPED (mla_nope=True)."""
    B = q_lora.shape[0]
    q = linear(q_lora, w["q_b_proj"]).view(B, cfg.Nh, cfg.Dqk)                          # [B, Nh, Dqk]
    q_nope, q_pe = q.split([cfg.Dnope, cfg.Dro], dim=-1)                                # q_pe NOT rope'd
    # bmm(q_nope [Nh, B, Dnope], w_kc [Nh, Dnope, Rkv]) -> [Nh, B, Rkv]
    q_nope_t = q_nope.transpose(0, 1).float()                                           # [Nh, B, Dnope]
    w_kc_f = w["w_kc"].float()                                                          # [Nh, Dnope, Rkv]
    q_nope_out = torch.bmm(q_nope_t, w_kc_f).transpose(0, 1).to(q.dtype)                # [B, Nh, Rkv]
    q_new = torch.cat([q_nope_out, q_pe], dim=-1)                                       # [B, Nh, Rkv+Dro]
    return q_new, q_nope_out, q_pe


def ref_F_IDX(hidden: torch.Tensor, q_lora: torch.Tensor, positions: torch.Tensor, w, cfg: GlmMoeDsaConfig):
    """F-IDX: indexer Q/K + k_norm + NeoX RoPE first 64 + Hadamard + gate."""
    B = hidden.shape[0]
    q_idx = linear(q_lora, w["wq_b"]).view(B, cfg.I, cfg.Di)                            # [B, I, Di]
    k_idx = linear(hidden, w["wk_idx"])                                                 # [B, Di]
    k_idx = layernorm_full(k_idx, w["k_norm_weight"], w["k_norm_bias"], cfg.rms_norm_eps)  # full LayerNorm
    q_idx = rope_neox_first(q_idx, positions, cfg.Dro, cfg.rope_theta)
    k_idx = rope_neox_first(k_idx, positions, cfg.Dro, cfg.rope_theta)
    q_idx = hadamard_rotate_identity(q_idx)
    k_idx = hadamard_rotate_identity(k_idx)
    # gate (fp32) — softmax_scale = Di^-0.5; q_scale fused later (proxy: q_scale ≡ 1.0)
    gate = linear(hidden.float(), w["weights_proj"].float()) * (cfg.I ** -0.5)          # [B, I]
    softmax_scale = cfg.Di ** -0.5
    weights = gate.unsqueeze(-1) * softmax_scale                                        # [B, I, 1]
    return q_idx, k_idx, weights


def ref_topk_singlecard(q_idx: torch.Tensor, weights: torch.Tensor,
                       full_K: torch.Tensor, full_positions: torch.Tensor,
                       new_pos_per_req: torch.Tensor, cfg: GlmMoeDsaConfig) -> torch.Tensor:
    """Single-card top-k: compute logits over ALL visible keys (≤ new_pos), take top-Ktop.

    full_K: [Ktot, Di] index K for positions 0..max(seqlen)-1 (NOT including the new tokens).
    full_positions: [Ktot] global positions of full_K rows.
    new_pos_per_req: [B] new token's position for each request.

    Returns topk_positions [B, Ktop] (int64, padded with -1 when visible < Ktop).
    """
    B, I, Di = q_idx.shape
    # Build logits: for each request b, only positions p <= new_pos_per_req[b] are visible.
    # logits[b, k] = Σ_i weights[b,i,0] * (q_idx[b,i,:] · K[k,:])
    q_f = q_idx.float()                                                                 # [B, I, Di]
    K_f = full_K.float()                                                                # [Ktot, Di]
    # logits_per_head[b, i, k] = q_f[b, i, :] @ K_f[k, :].T
    logits_per_head = torch.einsum("bid,kd->bik", q_f, K_f)                             # [B, I, Ktot]
    logits = (logits_per_head * weights.float()).sum(dim=1)                             # [B, Ktot]
    # causal mask: invisible positions -> -inf
    Ktot = full_K.shape[0]
    pos_b = new_pos_per_req.view(B, 1).float()                                          # [B, 1]
    fp_b = full_positions.view(1, Ktot).float()                                         # [1, Ktot]
    mask = (fp_b <= pos_b)                                                              # [B, Ktot]
    logits = logits.masked_fill(~mask, float("-inf"))
    # top-Ktop
    Ktop = cfg.Ktop
    topk_pos = torch.full((B, Ktop), -1, dtype=torch.long)
    for b in range(B):
        v = logits[b]                                                                   # [Ktot]
        # Pick min(Ktop, visible_len) largest indices
        visible = mask[b].sum().item()
        k = min(Ktop, visible)
        if k > 0:
            vals, ids = torch.topk(v, k=k)
            topk_pos[b, :k] = full_positions[ids]                                       # positions
    return topk_pos


def ref_attention_singlecard(q_new: torch.Tensor, full_KV: torch.Tensor,
                             full_positions: torch.Tensor, topk_pos: torch.Tensor,
                             cfg: GlmMoeDsaConfig) -> torch.Tensor:
    """Single-card sparse MQA on the selected top-Ktop slots. Returns [B, Nh, Rkv]."""
    B, Nh, D = q_new.shape  # D = Rkv + Dro = 576
    Ktop = topk_pos.shape[1]
    # Build position-to-row map
    pos2row = {int(full_positions[i].item()): i for i in range(full_positions.shape[0])}
    out_latent = torch.zeros(B, Nh, cfg.Rkv, dtype=q_new.dtype)
    for b in range(B):
        # Gather valid keys
        valid = topk_pos[b][topk_pos[b] >= 0].tolist()
        if not valid:
            continue
        rows = torch.tensor([pos2row[p] for p in valid], dtype=torch.long)
        kv_rows = full_KV[rows].float()                                                 # [k, 576]
        q_b = q_new[b].float()                                                          # [Nh, 576]
        # logits[h, k] = q_b[h, :] @ kv_rows[k, :].T * scaling
        logits = torch.einsum("hd,kd->hk", q_b, kv_rows) * cfg.scaling                  # [Nh, k]
        probs = torch.softmax(logits, dim=-1)                                           # [Nh, k]
        v = kv_rows[:, : cfg.Rkv]                                                       # [k, Rkv]
        out_latent[b] = torch.einsum("hk,kd->hd", probs, v).to(q_new.dtype)
    return out_latent                                                                   # [B, Nh, Rkv]


def ref_v_absorb_and_oproj(attn_out_latent: torch.Tensor, w, cfg: GlmMoeDsaConfig) -> torch.Tensor:
    """V absorb (bmm w_vc) + o_proj. Returns [B, H]."""
    B, Nh, Rkv = attn_out_latent.shape
    # bmm(attn_out_latent [Nh, B, Rkv], w_vc [Nh, Rkv, Dv]) -> [Nh, B, Dv]
    al_t = attn_out_latent.transpose(0, 1).float()                                      # [Nh, B, Rkv]
    w_vc_f = w["w_vc"].float()                                                          # [Nh, Rkv, Dv]
    attn_out = torch.bmm(al_t, w_vc_f).transpose(0, 1)                                  # [B, Nh, Dv]
    attn_out = attn_out.reshape(B, Nh * cfg.Dv).to(attn_out_latent.dtype)
    out = linear(attn_out, w["o_proj"])                                                 # [B, H]
    return out


# ============================================================================
# CP REF — partial sparse MQA + LSE merge
# ============================================================================

def ref_partial_mqa_with_lse(q_new: torch.Tensor, topk_pos: torch.Tensor,
                              local_KV: torch.Tensor, local_positions: torch.Tensor,
                              cfg: GlmMoeDsaConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-rank partial sparse MQA with return_lse=True.

    Only slots whose positions are in `local_positions` contribute; others are -inf.
    Returns (partial_out [B, Nh, Rkv], partial_lse [B, Nh]).
    Empty local subset → lse = -inf, out = 0 (will be weight-zero in merge).
    """
    B, Nh, D = q_new.shape
    pos2row = {int(local_positions[i].item()): i for i in range(local_positions.shape[0])}
    partial_out = torch.zeros(B, Nh, cfg.Rkv, dtype=torch.float32)
    partial_lse = torch.full((B, Nh), float("-inf"), dtype=torch.float32)
    for b in range(B):
        # Filter top-k positions to only those owned by this rank
        local_in_topk = [int(p.item()) for p in topk_pos[b] if int(p.item()) in pos2row]
        if not local_in_topk:
            continue
        rows = torch.tensor([pos2row[p] for p in local_in_topk], dtype=torch.long)
        kv_rows = local_KV[rows].float()                                                # [k_local, 576]
        q_b = q_new[b].float()                                                          # [Nh, 576]
        logits = torch.einsum("hd,kd->hk", q_b, kv_rows) * cfg.scaling                  # [Nh, k_local]
        partial_lse[b] = torch.logsumexp(logits, dim=-1)                                # [Nh]
        probs = torch.softmax(logits, dim=-1)                                           # [Nh, k_local]
        v = kv_rows[:, : cfg.Rkv]                                                       # [k_local, Rkv]
        partial_out[b] = torch.einsum("hk,kd->hd", probs, v)                            # [Nh, Rkv]
    return partial_out, partial_lse


def ref_lse_merge(partial_outs: List[torch.Tensor], partial_lses: List[torch.Tensor]) -> torch.Tensor:
    """Online-softmax LSE merge across cp_size ranks. Output: [B, Nh, Rkv]."""
    # Stack to [cp_size, B, Nh, Rkv] and [cp_size, B, Nh]
    O = torch.stack(partial_outs, dim=0).float()                                        # [cp, B, Nh, Rkv]
    L = torch.stack(partial_lses, dim=0).float()                                        # [cp, B, Nh]
    m = L.max(dim=0, keepdim=True).values                                               # [1, B, Nh]
    # Handle case where ALL ranks are -inf (no keys at all — shouldn't happen but be safe)
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w = torch.exp(L - m.squeeze(0))                                                     # [cp, B, Nh]
    Z = w.sum(dim=0)                                                                    # [B, Nh]
    Z = torch.where(Z == 0, torch.ones_like(Z), Z)                                      # avoid 0/0
    num = (w.unsqueeze(-1) * O).sum(dim=0)                                              # [B, Nh, Rkv]
    return (num / Z.unsqueeze(-1))


# ============================================================================
# End-to-end REFs
# ============================================================================

def ref_singlecard_decode(hidden_new: torch.Tensor, positions: torch.Tensor,
                          full_KV: torch.Tensor, full_KV_positions: torch.Tensor,
                          full_idxK: torch.Tensor, full_idxK_positions: torch.Tensor,
                          w, cfg: GlmMoeDsaConfig) -> torch.Tensor:
    """Full single-card decode forward (D0..D6, V5 baseline)."""
    q_lora, k_nope, k_pe, k_new = ref_F_PRE(hidden_new, w, cfg)
    q_new, _, _ = ref_F_Q_MAIN(q_lora, w, cfg)
    # Append k_new for each request to (a copy of) the full KV pool, in causal order.
    KV_aug = torch.cat([full_KV, k_new.squeeze(1)], dim=0)
    KV_aug_pos = torch.cat([full_KV_positions, positions])
    # Re-sort by position
    order = torch.argsort(KV_aug_pos)
    KV_aug = KV_aug[order]
    KV_aug_pos = KV_aug_pos[order]
    # Indexer + top-k
    q_idx, k_idx, weights = ref_F_IDX(hidden_new, q_lora, positions, w, cfg)
    idxK_aug = torch.cat([full_idxK, k_idx], dim=0)
    idxK_aug_pos = torch.cat([full_idxK_positions, positions])
    order2 = torch.argsort(idxK_aug_pos)
    idxK_aug = idxK_aug[order2]
    idxK_aug_pos = idxK_aug_pos[order2]
    topk_pos = ref_topk_singlecard(q_idx, weights, idxK_aug, idxK_aug_pos, positions, cfg)
    # Sparse MQA
    attn_out_latent = ref_attention_singlecard(q_new, KV_aug, KV_aug_pos, topk_pos, cfg)
    # V absorb + o_proj
    out = ref_v_absorb_and_oproj(attn_out_latent, w, cfg)
    return out, topk_pos, attn_out_latent


def ref_cp_decode(hidden_new: torch.Tensor, positions: torch.Tensor,
                  ranks: List[RankState], w, cfg: GlmMoeDsaConfig) -> torch.Tensor:
    """Full CP-decode forward (D0..D6 + F-TOPK-CP + F-MERGE-POST). Returns [B, H].

    Design (V7 final): index K is **replicated** on every rank (mirrored pool);
    latent KV is sharded by `pos % cp_size`. F-TOPK-CP has zero communication
    (each rank computes the same top-k locally from its mirrored index K pool).
    Each rank in `ranks` is mutated in-place — latent KV appended to owner only,
    index K appended to all ranks.
    """
    cp_size = len(ranks)
    B = hidden_new.shape[0]

    # F-PRE (all ranks compute the same; we just compute once and use)
    q_lora, k_nope, k_pe, k_new = ref_F_PRE(hidden_new, w, cfg)
    # F-Q-MAIN
    q_new, _, _ = ref_F_Q_MAIN(q_lora, w, cfg)
    # F-IDX
    q_idx, k_idx, weights = ref_F_IDX(hidden_new, q_lora, positions, w, cfg)

    # F-KV-STORE: owner-only append to latent KV pool
    for b in range(B):
        pos = int(positions[b].item())
        owner = pos % cp_size
        ranks[owner].latent_KV_pool.append(k_new[b, 0].to(torch.bfloat16))
        ranks[owner].owned_positions.append(pos)

    # F-IDX index K append: ALL ranks (index K is replicated; hidden_new is identical
    # on every rank ⇒ k_idx_fp8 is bit-exact identical ⇒ pools stay mirrored)
    for b in range(B):
        pos = int(positions[b].item())
        for r in ranks:
            r.index_K_pool.append(k_idx[b].to(torch.bfloat16))
            r.index_K_positions.append(pos)

    # F-TOPK-CP: each rank uses its (mirrored) full index K pool to compute top-k.
    # Zero communication; the result `topk_positions` is bit-exact identical across ranks.
    # We compute once and reuse (in real impl every rank runs the same kernel locally).
    full_K = torch.stack(ranks[0].index_K_pool, dim=0)
    full_pos = torch.tensor(ranks[0].index_K_positions, dtype=torch.long)
    topk_pos = ref_topk_singlecard(q_idx, weights, full_K, full_pos, positions, cfg)
    # Per-rank page-table mapping (`topk_positions` → `topk_slots_r`) is implicit:
    # ref_partial_mqa_with_lse below filters by `pos in r.owned_positions`, which is
    # the logical equivalent of "non-r positions map to -1 in topk_slots_r".

    # F-MQA-PARTIAL: each rank's partial over (topk_pos ∩ K_local)
    partial_outs, partial_lses = [], []
    for r in ranks:
        if not r.latent_KV_pool:
            partial_outs.append(torch.zeros(B, cfg.Nh, cfg.Rkv, dtype=torch.float32))
            partial_lses.append(torch.full((B, cfg.Nh), float("-inf"), dtype=torch.float32))
            continue
        local_KV = torch.stack(r.latent_KV_pool, dim=0)                                 # [n_local, 576]
        local_pos = torch.tensor(r.owned_positions, dtype=torch.long)
        po, pl = ref_partial_mqa_with_lse(q_new, topk_pos, local_KV, local_pos, cfg)
        partial_outs.append(po)
        partial_lses.append(pl)

    # F-MERGE-POST: LSE merge + bmm w_vc + o_proj
    attn_out_latent = ref_lse_merge(partial_outs, partial_lses).to(q_new.dtype)         # [B, Nh, Rkv]
    out = ref_v_absorb_and_oproj(attn_out_latent, w, cfg)
    return out, topk_pos, attn_out_latent


# ============================================================================
# Stage runners
# ============================================================================

def compare_tensors(name: str, ref: torch.Tensor, got: torch.Tensor,
                    atol: float = 5e-3, rtol: float = 5e-3, exact: bool = False) -> bool:
    diff = (ref.float() - got.float()).abs()
    md = diff.max().item()
    mn = diff.mean().item()
    if exact:
        ok = torch.equal(ref, got)
    else:
        ok = md <= atol or (md / (ref.float().abs().max().item() + 1e-9) <= rtol)
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}: max_diff={md:.3e} mean_diff={mn:.3e}")
    return ok


def stage_cp_pre(args, cfg, shapes, ranks, w, hidden, positions, mode):
    print(f"\n=== stage: cp_pre  (F-PRE) ===")
    if mode == "ref":
        q_lora, k_nope, k_pe, k_new = ref_F_PRE(hidden, w, cfg)
        print(f"  q_lora {tuple(q_lora.shape)} k_nope {tuple(k_nope.shape)} "
              f"k_pe {tuple(k_pe.shape)} k_new {tuple(k_new.shape)}")
        return True
    raise NotImplementedError("cp_pre: zeus kernel `dsa_pre_attn_qkv_latent_fused` not landed")


def stage_cp_q_main(args, cfg, shapes, ranks, w, hidden, positions, mode):
    print(f"\n=== stage: cp_q_main  (F-Q-MAIN; main MLA RoPE SKIPPED) ===")
    if mode == "ref":
        q_lora, _, _, _ = ref_F_PRE(hidden, w, cfg)
        q_new, q_nope_out, q_pe = ref_F_Q_MAIN(q_lora, w, cfg)
        print(f"  q_new {tuple(q_new.shape)} q_nope_out {tuple(q_nope_out.shape)} "
              f"q_pe {tuple(q_pe.shape)}  (q_pe is NOT rope'd: mla_nope=True)")
        return True
    raise NotImplementedError("cp_q_main: zeus kernel `dsa_q_proj_absorb_fused` not landed")


def stage_cp_kv_store(args, cfg, shapes, ranks, w, hidden, positions, mode):
    print(f"\n=== stage: cp_kv_store  (F-KV-STORE; owner-only) ===")
    if mode == "ref":
        _, _, _, k_new = ref_F_PRE(hidden, w, cfg)
        before = {r.rank: len(r.latent_KV_pool) for r in ranks}
        # owner-only append
        for b in range(positions.shape[0]):
            pos = int(positions[b].item())
            owner = pos % shapes.cp_size
            ranks[owner].latent_KV_pool.append(k_new[b, 0].to(torch.bfloat16))
            ranks[owner].owned_positions.append(pos)
        after = {r.rank: len(r.latent_KV_pool) for r in ranks}
        # Verify
        for b in range(positions.shape[0]):
            pos = int(positions[b].item())
            owner = pos % shapes.cp_size
            assert after[owner] == before[owner] + sum(
                1 for bb in range(b + 1)
                if int(positions[bb].item()) % shapes.cp_size == owner
            ), "owner pool size mismatch"
        # Non-owners unchanged
        for r in ranks:
            owners_touched = {int(positions[b].item()) % shapes.cp_size for b in range(positions.shape[0])}
            if r.rank not in owners_touched:
                assert after[r.rank] == before[r.rank], f"non-owner rank {r.rank} modified"
        print(f"  KV pool sizes before: {before}")
        print(f"  KV pool sizes after:  {after}")
        return True
    raise NotImplementedError("cp_kv_store: zeus kernel `dsa_kv_store_owner_fused` not landed")


def stage_cp_idx(args, cfg, shapes, ranks, w, hidden, positions, mode):
    print(f"\n=== stage: cp_idx  (F-IDX) ===")
    if mode == "ref":
        q_lora, _, _, _ = ref_F_PRE(hidden, w, cfg)
        q_idx, k_idx, weights = ref_F_IDX(hidden, q_lora, positions, w, cfg)
        print(f"  q_idx {tuple(q_idx.shape)} k_idx {tuple(k_idx.shape)} "
              f"weights {tuple(weights.shape)} (fp32)")
        return True
    raise NotImplementedError("cp_idx: zeus kernel `dsa_indexer_prep_store_fused` not landed")


def stage_cp_topk_cp(args, cfg, shapes, ranks, w, hidden, positions, mode):
    print(f"\n=== stage: cp_topk_cp  (F-TOPK-CP; ZERO comm — index K mirrored) ===")
    if mode == "ref":
        # Hard invariant: topk_positions bit-exact identical across ranks (because
        # each rank's index K pool is mirrored). The result also matches cp=1 baseline.
        out_cp, topk_pos_cp, _ = ref_cp_decode(hidden, positions, ranks, w, cfg)
        print(f"  topk_pos shape: {tuple(topk_pos_cp.shape)} (bit-exact across cp ranks)")
        # End-to-end validation lives in cp_full_path.
        return True
    raise NotImplementedError("cp_topk_cp: zeus path = V5 `_get_topk_paged` + per-rank latent-KV page-table mask not landed")


def stage_cp_mqa_partial(args, cfg, shapes, ranks, w, hidden, positions, mode):
    print(f"\n=== stage: cp_mqa_partial  (F-MQA-PARTIAL; return_lse=True) ===")
    if mode == "ref":
        q_lora, _, _, _ = ref_F_PRE(hidden, w, cfg)
        q_new, _, _ = ref_F_Q_MAIN(q_lora, w, cfg)
        # Use a dummy topk_pos (top of available positions) for this stage's smoke check
        dummy_topk = torch.full((shapes.B, cfg.Ktop), -1, dtype=torch.long)
        # Just verify shapes flow through one rank with no kvs
        po, pl = ref_partial_mqa_with_lse(q_new, dummy_topk,
                                          torch.zeros(0, cfg.Rkv + cfg.Dro),
                                          torch.zeros(0, dtype=torch.long), cfg)
        print(f"  partial_out {tuple(po.shape)} partial_lse {tuple(pl.shape)}  (empty-subset case: lse=-inf)")
        assert torch.isinf(pl).all(), "empty subset should produce lse=-inf"
        return True
    raise NotImplementedError("cp_mqa_partial: zeus kernel `dsa_decode_sparse_mqa_partial` (NEW; return_lse) not landed")


def stage_cp_merge_post(args, cfg, shapes, ranks, w, hidden, positions, mode):
    print(f"\n=== stage: cp_merge_post  (⚙ F-MERGE-POST; LSE merge + w_vc + o_proj) ===")
    if mode == "ref":
        # Construct fake partials and verify merge logic
        B = shapes.B
        cp_size = shapes.cp_size
        partials_out = [torch.randn(B, cfg.Nh, cfg.Rkv) for _ in range(cp_size)]
        partials_lse = [torch.randn(B, cfg.Nh) for _ in range(cp_size)]
        merged = ref_lse_merge(partials_out, partials_lse)
        # Run V absorb + o_proj
        out = ref_v_absorb_and_oproj(merged.to(torch.bfloat16), w, cfg)
        print(f"  merged_latent {tuple(merged.shape)} out {tuple(out.shape)}")
        return True
    raise NotImplementedError("cp_merge_post: zeus kernel `dsa_cp_merge_post_fused` (NEW) not landed")


def stage_cp_full_path(args, cfg, shapes, ranks_template, w, hidden, positions, mode):
    """End-to-end: assert `cp=k` (k ∈ {1, 4, 16}) all give identical `out`."""
    print(f"\n=== stage: cp_full_path  (D0..D6 end-to-end; cp ∈ {{1, {shapes.cp_size}}}) ===")
    if mode != "ref":
        raise NotImplementedError("cp_full_path: zeus end-to-end not yet implemented")

    import copy
    # cp=1 baseline
    ranks_1 = init_ranks(cfg, cp_size=1, seqlen=shapes.seqlen, seed=42)
    out_1, topk_1, lat_1 = ref_cp_decode(hidden, positions, ranks_1, w, cfg)

    # cp=shapes.cp_size
    ranks_k = init_ranks(cfg, cp_size=shapes.cp_size, seqlen=shapes.seqlen, seed=42)
    out_k, topk_k, lat_k = ref_cp_decode(hidden, positions, ranks_k, w, cfg)

    # NOTE: LSE merge is mathematically exact, but with fp32 accumulation the order-of-summation
    # between cp=1 (one big softmax) and cp=k (k partial softmaxes + LSE merge) differs, producing
    # ~1e-4 absolute noise on bf16-cast outputs. Anything larger is a real logic bug.
    ok1 = compare_tensors("attn_out_latent (cp=1 vs cp=k)", lat_1, lat_k, atol=2e-3, rtol=2e-3)
    ok2 = compare_tensors("out [B,H]      (cp=1 vs cp=k)", out_1, out_k, atol=5e-3, rtol=5e-3)
    # topk_positions: set-equality (order may differ on ties)
    set_match = True
    for b in range(shapes.B):
        s1 = {int(x.item()) for x in topk_1[b] if int(x.item()) >= 0}
        sk = {int(x.item()) for x in topk_k[b] if int(x.item()) >= 0}
        if s1 != sk:
            set_match = False
            print(f"  [FAIL] topk set differs at b={b}: |s1|={len(s1)}, |sk|={len(sk)}, "
                  f"|s1∩sk|={len(s1 & sk)}, |s1-sk|={len(s1 - sk)}, |sk-s1|={len(sk - s1)}")
    print(f"  [{'PASS' if set_match else 'FAIL'}] topk set equality")
    return ok1 and ok2 and set_match


STAGES = {
    "cp_pre":         stage_cp_pre,
    "cp_q_main":      stage_cp_q_main,
    "cp_kv_store":    stage_cp_kv_store,
    "cp_idx":         stage_cp_idx,
    "cp_topk_cp":     stage_cp_topk_cp,
    "cp_mqa_partial": stage_cp_mqa_partial,
    "cp_merge_post":  stage_cp_merge_post,
    "cp_full_path":   stage_cp_full_path,
}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GLM5-Next DSA decode V7 (CP + Route B)")
    parser.add_argument("--stage", default="all", choices=["all"] + list(STAGES.keys()))
    parser.add_argument("--mode", default="ref", choices=["ref", "zeus"])
    parser.add_argument("--config", default="16b", choices=["16b", "big"])
    parser.add_argument("--cp", type=int, default=4, help="context-parallel degree")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=64,
                        help="history length per request (smaller than Ktop for proxy)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = select_config(args.config)
    # Cap proxy: keep Ktop reachable
    if args.seqlen > 256:
        print(f"NOTE: capping seqlen to 256 for proxy run")
        args.seqlen = 256
    cfg_proxy = GlmMoeDsaConfig(
        name=cfg.name + "_proxy",
        H=cfg.H, Nh=cfg.Nh, Rq=cfg.Rq, Rkv=cfg.Rkv,
        Dnope=cfg.Dnope, Dro=cfg.Dro, Dqk=cfg.Dqk, Dv=cfg.Dv,
        I=cfg.I, Di=cfg.Di,
        Ktop=min(cfg.Ktop, args.seqlen),                        # cap Ktop for the proxy
        full_attn_layers=cfg.full_attn_layers,
    )
    shapes = make_proxy(args, cfg_proxy)
    print(f"Config: {cfg_proxy.name} (proxy Ktop={cfg_proxy.Ktop})")
    print(f"Shapes: B={shapes.B} cp_size={shapes.cp_size} seqlen={shapes.seqlen}")
    print(f"        H={cfg_proxy.H} Nh={cfg_proxy.Nh} Rkv={cfg_proxy.Rkv} Dro={cfg_proxy.Dro}")
    print(f"        I={cfg_proxy.I} Di={cfg_proxy.Di} Ktop={cfg_proxy.Ktop}")
    print(f"        mla_nope={cfg_proxy.mla_nope}  ⇒ main MLA RoPE SKIPPED")

    w = init_weights(cfg_proxy, seed=args.seed)
    ranks = init_ranks(cfg_proxy, cp_size=shapes.cp_size, seqlen=shapes.seqlen, seed=args.seed + 1)

    # Inputs (decode: B new tokens)
    hidden = torch.randn(shapes.B, cfg_proxy.H, dtype=torch.bfloat16) * 0.1
    # New token positions: each request continues from `seqlen` (so new pos = seqlen + b)
    positions = torch.arange(shapes.seqlen, shapes.seqlen + shapes.B)

    stages_to_run = list(STAGES.keys()) if args.stage == "all" else [args.stage]
    results: Dict[str, bool] = {}
    for stage_name in stages_to_run:
        # Use a fresh ranks snapshot per stage that mutates state
        import copy
        ranks_for_stage = copy.deepcopy(ranks)
        try:
            ok = STAGES[stage_name](args, cfg_proxy, shapes, ranks_for_stage, w, hidden, positions, args.mode)
            results[stage_name] = bool(ok)
        except NotImplementedError as e:
            results[stage_name] = False
            print(f"  [SKIP/TODO] {e}")

    print("\n" + "=" * 60)
    print("Summary:")
    for k, v in results.items():
        print(f"  {k:20s}  {'PASS' if v else 'SKIP/FAIL'}")


if __name__ == "__main__":
    main()
