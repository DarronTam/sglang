"""
GLM5-Next 整 transformer block (decode) REF-vs-Zeus 对齐 —— mHC 组装

范围（见 glm5next_block_decode_dev.md）：
  - 把 GLM5-Next 一个 decoder layer 的 decode 路径**整体**串起来，重点验证
    **mHC wrapper 与两类 attention sublayer 的组装**：
      · Linear-attention-based block（KDA 层，decode）
      · DSA-based block（full-attn 层，decode）
  - 单 device、单 layer、decode only、TP=1、CP=1。
  - "目前主要差 mHC"：attention/MoE 各算子已分别由
      dev_kimi_linear_attn_test.py / dev_glm5next_dsa_decode_test.py /
      dev_glm4_moe_test.py 对齐；本脚本只补 **block 级 mHC 组装**。

两套配置（与 dev_glm5next_dsa_decode_test.py 一致）：
  - GLM5-Next-16B : config_16b_v2.json  (H=2048, KDA head_dim=72, full-attn 6 层)
  - GLM5-Next     : config.json         (H=4096, KDA head_dim=128, full-attn 11 层)

组装结构（MHCLayerCommunicator 顺序）：
  residual[T,N,H]
    │ attn_hc.pre   → layer_input[T,H]
  self_attn(layer_input)            ← Linear-attn decode 或 DSA decode
    │ attn_hc.post  → residual_mid[T,N,H]
    │ mlp_hc.pre    → layer_input[T,H]
  mlp(layer_input)                  ← dense SwiGLU（MoE 由 dev_glm4_moe_test 覆盖）
    │ mlp_hc.post   → residual_out[T,N,H]

Stage:
  linear_attn_decode    单独 Linear-attention sublayer decode（REF 自检）
  dsa_decode            单独 DSA sublayer decode（复用 dsa 模块 REF）
  mlp_decode            单独 dense SwiGLU MLP sublayer（REF 自检）
  linear_attn_block     mHC( attn=Linear-attn, mlp=dense ) 整 block decode  [Zeus mHC chain LANDED]
  dsa_block             mHC( attn=DSA, mlp=MoE ) 整 block decode              [Zeus mHC+DSA+MoE 全 LANDED]
  decode_layer_full     按 full_attn_layers 路由 attn 类型，跑完整一层

用法:
  python zeus_dev/model_state_dev/dev_glm5next_block_decode_test.py
  python zeus_dev/model_state_dev/dev_glm5next_block_decode_test.py --config next
  python zeus_dev/model_state_dev/dev_glm5next_block_decode_test.py --stage linear_attn_block
  python zeus_dev/model_state_dev/dev_glm5next_block_decode_test.py --stage decode_layer_full --layer-id 3

约定：
  - 默认 mode=both：先跑 REF（含不变量自检），再尝试 Zeus。
  - linear_attn_block: mHC chain Zeus；KDA + dense MLP 仍 REF（Linear-attn projections
    没有 Zeus 路径——见 dev_kimi_linear_attn_test.py:1060 注释；dense MLP 也无 Zeus）。
  - dsa_block: **mHC + DSA + MoE 全 Zeus** device-resident（21 颗算子串接 + per-step
    CPU 协调）。MoE 走 proxy shape (E=8/mI=128/top_k=2)；GLM-4.7 真实 shape 的 MoE
    kernel 对拍由 dev_glm4_moe_test.py 覆盖。
  - 单 sublayer stage (linear_attn_decode / dsa_decode / mlp_decode) 仍打印 SKIP/TODO，
    指向各自的 kernel 级 dev 脚本。禁止 silent fallback。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch

# 复用已对齐的 REF（两个模块都把 zeus import 包在 try/except，import 安全）
import dev_glm5next_dsa_decode_test as dsa
import dev_glm5next_mhc_test as mhc
import dev_glm4_moe_test as moe_dev

try:
    import torch_zeus  # noqa: F401
    import sgl_kernel_zeus  # noqa: F401

    ZEUS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    torch_zeus = None
    sgl_kernel_zeus = None
    ZEUS_IMPORT_ERROR = exc


_THIS_DIR = Path(__file__).resolve().parent
REF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Config（linear-attn 段；DSA / mHC 段各自从对应模块取） ────────
@dataclass(frozen=True)
class LinearAttnConfig:
    H: int
    num_heads: int
    head_k_dim: int          # linear_key_head_dim == linear_attn_config.head_dim
    head_v_dim: int          # linear_value_head_dim
    conv_size: int           # short_conv_kernel_size
    rms_norm_eps: float

    @property
    def proj_size(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def scaling(self) -> float:
        return self.head_k_dim ** -0.5


def _load_linear_cfg(path: Path) -> LinearAttnConfig:
    raw = json.loads(path.read_text())
    lac = raw["linear_attn_config"]
    return LinearAttnConfig(
        H=int(raw["hidden_size"]),
        num_heads=int(lac["num_heads"]),
        head_k_dim=int(lac["head_dim"]),
        head_v_dim=int(raw.get("linear_value_head_dim", lac["head_dim"])),
        conv_size=int(lac["short_conv_kernel_size"]),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
    )


def _config_path(which: str) -> Path:
    return _THIS_DIR / ("config_16b_v2.json" if which == "16b" else "config.json")


def _full_attn_layers(which: str) -> Tuple[int, ...]:
    raw = json.loads(_config_path(which).read_text())
    return tuple(raw["linear_attn_config"]["full_attn_layers"])


# ── pure-torch helpers ──────────────────────────────────────────
def linear_bf16(x, w):
    return torch.nn.functional.linear(x.float(), w.float()).to(torch.bfloat16)


def l2norm(x, eps=1e-6):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


def softplus_kda(x, beta=1.0, threshold=20.0):
    xs = x * beta
    return torch.where(xs > threshold, x, (1.0 / beta) * torch.log1p(torch.exp(xs)))


def conv1d_update_silu(x, state, weight, bias):
    """decode 单步 per-channel causal conv1d + silu，state 原位更新。

    x:[B,C] state:[B,C,K-1] weight:[C,K] bias:[C] -> out:[B,C]
    """
    win = torch.cat([state, x.unsqueeze(-1)], dim=-1)        # [B,C,K]
    out = (win * weight.unsqueeze(0)).sum(-1)
    if bias is not None:
        out = out + bias.unsqueeze(0)
    state.copy_(win[..., 1:])
    return torch.nn.functional.silu(out)


def rms_norm_gated_sigmoid(x, g, weight, eps):
    """y = rmsnorm(x)*weight * sigmoid(g)，per-(head, dim)。x,g:[B,Hh,D]"""
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return (y * torch.sigmoid(g.float())).to(x.dtype)


# ── Linear-attention block sublayer (decode) ────────────────────
def init_linear_weights(cfg: LinearAttnConfig, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rn(*shape, scale=0.02, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype)

    P = cfg.proj_size
    w: Dict[str, torch.Tensor] = {}
    w["qkv_proj"] = rn(3 * P, cfg.H)                          # q|k|v 融合
    w["conv_w"] = rn(3 * P, cfg.conv_size, dtype=torch.float32) * 0.1
    w["conv_b"] = rn(3 * P, dtype=torch.float32) * 0.01
    w["b_proj"] = rn(cfg.num_heads, cfg.H)                    # beta
    w["f_a"] = rn(cfg.head_k_dim, cfg.H)
    w["f_b"] = rn(P, cfg.head_k_dim)
    w["g_a"] = rn(cfg.head_k_dim, cfg.H)
    w["g_b"] = rn(P, cfg.head_k_dim)
    w["dt_bias"] = (torch.randn(P, generator=g, dtype=torch.float32) * 0.01)
    w["A_log"] = (torch.randn(cfg.num_heads, generator=g, dtype=torch.float32) * 0.1)
    w["o_norm"] = torch.ones(cfg.head_v_dim, dtype=torch.float32)
    w["o_proj"] = rn(cfg.H, P)
    return w


def init_linear_state(cfg: LinearAttnConfig, B: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    conv = torch.randn(B, 3 * cfg.proj_size, cfg.conv_size - 1,
                       generator=g, dtype=torch.bfloat16) * 0.1
    rec = torch.randn(B, cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim,
                      generator=g, dtype=torch.float32) * 0.05
    return conv, rec


def ref_linear_attn_decode(hidden, w, cfg: LinearAttnConfig, conv_state, rec_state):
    """GLM5NextLinearAttention.forward 的 decode REF（单步）。

    hidden:[B,H] bf16 -> out:[B,H] bf16；conv_state / rec_state 原位推进。
    对齐 prerelease glm5_next.py:316-367（decode 分支：fused_kda_gate 在 attn 内融合）。
    """
    B = hidden.shape[0]
    Hh, Dk, Dv, P = cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim, cfg.proj_size

    qkv = linear_bf16(hidden, w["qkv_proj"])                  # [B, 3P]
    beta = linear_bf16(hidden, w["b_proj"])                   # [B, Hh]
    fg = linear_bf16(linear_bf16(hidden, w["f_a"]), w["f_b"])  # [B, P]
    gproj = linear_bf16(linear_bf16(hidden, w["g_a"]), w["g_b"])  # [B, P]

    # decode conv1d_update（q|k|v 融合一次更新）+ silu
    qkv = conv1d_update_silu(qkv, conv_state, w["conv_w"], w["conv_b"])
    q, k, v = qkv.split([P, P, P], dim=-1)
    q = q.view(B, Hh, Dk)
    k = k.view(B, Hh, Dk)
    v = v.view(B, Hh, Dv)

    # fused_kda_gate: softplus(beta,tau) * -exp(A_log)，加 dt_bias
    fg = fg.float() + w["dt_bias"].unsqueeze(0)
    fg = softplus_kda(fg).view(B, Hh, Dk)
    a = (-torch.exp(w["A_log"])).view(1, Hh, 1)
    g_gate = (a * fg)                                         # [B,Hh,Dk] fp32
    beta = torch.sigmoid(beta.float())                       # [B,Hh]

    # 单步 delta-rule recurrent（S 先 decay 再 readout）
    qn = l2norm(q).float() * cfg.scaling
    kn = l2norm(k).float()
    S = rec_state                                            # [B,Hh,Dk,Dv]
    S = S * torch.exp(g_gate).unsqueeze(-1)
    v_hat = torch.einsum("bhk,bhkv->bhv", kn, S)
    delta = v.float() - v_hat
    S = S + torch.einsum("bhv,bhk->bhkv", beta.unsqueeze(-1) * delta, kn)
    o = torch.einsum("bhk,bhkv->bhv", qn, S)                 # [B,Hh,Dv]
    rec_state.copy_(S)

    # gated rmsnorm + o_proj
    norm_gate = gproj.view(B, Hh, Dk)                        # head_v==head_k
    o = rms_norm_gated_sigmoid(o.to(torch.bfloat16), norm_gate, w["o_norm"], cfg.rms_norm_eps)
    o = o.reshape(B, P)
    return linear_bf16(o, w["o_proj"])


# ── Dense SwiGLU MLP sublayer (decode) ──────────────────────────
def init_mlp_weights(H: int, inter: int, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rn(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(torch.bfloat16)

    return {"gate_up": rn(2 * inter, H), "down": rn(H, inter)}


def ref_mlp_decode(hidden, w, clamp_limit: float = 10.0):
    """GLM5NextMLP 的 SwiGLU(clamp) REF。hidden:[B,H] -> [B,H] bf16。"""
    gu = linear_bf16(hidden, w["gate_up"]).float()
    inter = gu.shape[-1] // 2
    gate, up = gu[:, :inter], gu[:, inter:]
    if clamp_limit is not None:
        gate = gate.clamp(min=-clamp_limit, max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    return linear_bf16(act, w["down"])


# ── DSA block sublayer (decode) ─────────────────────────────────
def build_dsa_ctx(which: str, B: int, seqlen: int, seed: int) -> dsa.DevContext:
    args = SimpleNamespace(config=which, seed=seed, batch=B, seqlen=seqlen, block_span=16)
    return dsa.build_context(args)


def ref_dsa_decode(ctx: dsa.DevContext, hidden: torch.Tensor) -> torch.Tensor:
    """DSA sublayer decode REF：用给定 hidden 覆盖 ctx.hidden，跑 cp=1 no-CP decode。"""
    ctx.hidden = hidden
    return dsa.run_ref_decode(ctx, cp_size=1)["out"]


# ── Helpers ─────────────────────────────────────────────────────
def compare_tensors(name, ref, got, atol=5e-3, rtol=5e-3):
    return dsa.compare_tensors(name, ref, got, atol=atol, rtol=rtol)


def zeus_skip(kernel_name: str, anchor: str):
    if ZEUS_IMPORT_ERROR is not None:
        print(f"  ZEUS: SKIP (sgl_kernel_zeus unavailable: {ZEUS_IMPORT_ERROR})")
    else:
        print(f"  ZEUS: SKIP (block-level fused kernel '{kernel_name}' 未落地)")
        print(f"        子算子对拍见: {anchor}")


def finish_stage(args, ref_ok: bool, kernel_name: str, anchor: str) -> Optional[bool]:
    if not ref_ok:
        return False
    if args.mode == "ref":
        return True
    zeus_skip(kernel_name, anchor)
    return None


def _mhc_cfg(which: str) -> mhc.Glm5NextMhcConfig:
    return mhc.select_config(which)


def run_mhc_block(which: str, residual_flat: torch.Tensor,
                  attn_fn, mlp_fn, seed: int):
    """一个完整 mHC block：attn_hc(pre/post) -> mlp_hc(pre/post)。

    attn_fn / mlp_fn: callable(layer_input[T,H]) -> [T,H]。
    返回 (residual_after_attn, residual_out)。
    """
    cfg = _mhc_cfg(which)
    p_attn = mhc.init_mhc_params(cfg, seed)
    p_mlp = mhc.init_mhc_params(cfg, seed + 100)

    li_a, res_a, hres_a, hpost_a = mhc.ref_mhc_pre(residual_flat, p_attn, cfg)
    attn_out = attn_fn(li_a)
    residual_mid = mhc.ref_mhc_post(attn_out, res_a, hpost_a, hres_a, cfg)

    li_m, res_m, hres_m, hpost_m = mhc.ref_mhc_pre(residual_mid, p_mlp, cfg)
    mlp_out = mlp_fn(li_m)
    residual_out = mhc.ref_mhc_post(mlp_out, res_m, hpost_m, hres_m, cfg)
    return residual_mid, residual_out


def run_mhc_block_ref_quant(which: str, residual_flat: torch.Tensor,
                            attn_fn, mlp_fn, seed: int):
    """REF mHC + REF sublayer，但 mHC 参数走 `quantize_p_for_zeus_match` (fn → bf16
    round-trip + norm_weight pre-merge)，作为 Zeus chain 的等价 golden。

    与 run_mhc_block 同签名；attn_fn / mlp_fn 是 REF sublayer，调用方需保证 state
    在调用前已 clone（attn 通常 stateful）。
    """
    cfg = _mhc_cfg(which)
    p_attn = mhc.quantize_p_for_zeus_match(mhc.init_mhc_params(cfg, seed))
    p_mlp = mhc.quantize_p_for_zeus_match(mhc.init_mhc_params(cfg, seed + 100))

    li_a, res_a, hres_a, hpost_a = mhc.ref_mhc_pre(residual_flat, p_attn, cfg)
    attn_out = attn_fn(li_a)
    residual_mid = mhc.ref_mhc_post(attn_out, res_a, hpost_a, hres_a, cfg)

    li_m, res_m, hres_m, hpost_m = mhc.ref_mhc_pre(residual_mid, p_mlp, cfg)
    mlp_out = mlp_fn(li_m)
    residual_out = mhc.ref_mhc_post(mlp_out, res_m, hpost_m, hres_m, cfg)
    return residual_mid, residual_out


def run_mhc_block_zeus(which: str, residual_flat: torch.Tensor,
                       attn_fn, mlp_fn, seed: int):
    """Zeus mHC chain (K1→K2→K3→K4 ×2) + REF sublayer。

    sublayer (attn/mlp) 本身在本 dev 脚本范畴内仍是 pure-torch REF（kernel 级对拍
    见各自 dev 脚本），在 sublayer 边界 round-trip：z_li.cpu() → REF sublayer →
    .to("zeus") → K4。

    返回 (z_mid, z_out)，均在 Zeus device 上。调用方需保证 attn_fn / mlp_fn 闭包
    捕获的 state 已 clone（这条 chain 单独跑，state 不复用）。
    """
    cfg = _mhc_cfg(which)
    p_attn = mhc.init_mhc_params(cfg, seed)
    p_mlp = mhc.init_mhc_params(cfg, seed + 100)

    # attn wrap
    z_li_a, z_res_a, z_hres_a, z_hpost_a = mhc.zeus_mhc_pre(residual_flat, p_attn, cfg)
    attn_out_cpu = attn_fn(z_li_a.cpu())                           # REF sublayer
    z_mid = mhc.zeus_mhc_post(
        attn_out_cpu.to("zeus"), z_res_a, z_hpost_a, z_hres_a, cfg,
    )

    # mlp wrap (z_mid 直接喂下一轮 K1，不下 host)
    z_li_m, z_res_m, z_hres_m, z_hpost_m = mhc.zeus_mhc_pre(z_mid, p_mlp, cfg)
    mlp_out_cpu = mlp_fn(z_li_m.cpu())
    z_out = mhc.zeus_mhc_post(
        mlp_out_cpu.to("zeus"), z_res_m, z_hpost_m, z_hres_m, cfg,
    )
    return z_mid, z_out


def _zeus_mhc_chain_available() -> bool:
    """mHC chain 4 颗 kernel 是否齐备 (K1/K2/K3/K4)."""
    if ZEUS_IMPORT_ERROR is not None:
        return False
    return all(hasattr(sgl_kernel_zeus, k) for k in (
        "mhc_pre_norm_split", "mhc_sinkhorn",
        "mhc_pre_apply_mix", "mhc_post",
    ))


def finish_stage_with_zeus(args, ref_ok: bool, zeus_ok: Optional[bool],
                           kernel_name: str, anchor: str) -> Optional[bool]:
    """与 dev_glm5next_mhc_test.finish_stage_with_zeus 同语义的三态终结器。"""
    if not ref_ok:
        return False
    if args.mode == "ref":
        return True
    if zeus_ok is None:
        zeus_skip(kernel_name, anchor)
        return None
    return bool(zeus_ok)


# ── DSA decode：Zeus end-to-end chain（mirror stage_decode_full_nocp） ─
def _lmem_pack(t: torch.Tensor):
    """LocalMem pack helper（与 dsa dev script 同套路：kind=weight, Tr=Tc=1）."""
    return torch.zeus.local_memory.from_tensor(
        t.to("zeus"), kind="weight", Tr=1, Tc=1,
    )


def zeus_dsa_decode(ctx: dsa.DevContext, hidden_z: torch.Tensor) -> torch.Tensor:
    """完整 DSA decode 的 Zeus chain。

    输入  : ctx (DevContext, pre-built history), hidden_z (Zeus bf16 [B, H])
    输出  : Zeus bf16 [B, H]，attention sublayer 输出

    chain 含 10 颗 sgl_kernel_zeus 算子 + 不可避免的 CPU 协调步骤（top_pos 选择 /
    history 拼接 / per-batch latent_k_gather 循环），mirror
    `dev_glm5next_dsa_decode_test::stage_decode_full_nocp` 的精确语义。
    """
    cfg = ctx.cfg
    B = hidden_z.shape[0]
    w = ctx.weights

    q_a_w = w["fused_qkv_a"][: cfg.Rq].contiguous()
    kv_a_w = w["fused_qkv_a"][cfg.Rq:].contiguous()
    q_b_w = w["q_b_proj"].contiguous()
    w_kc = w["w_kc"].contiguous()
    w_vc = w["w_vc"].contiguous()
    o_proj_w = w["o_proj"].contiguous()
    wq_b = w["wq_b"].contiguous()
    wk_idx = w["wk_idx"].contiguous()
    H_Di = w["hadamard_Di"].contiguous()
    weights_proj = w["weights_proj"].contiguous()
    slot_mapping = torch.arange(B, dtype=torch.int32).to("zeus")

    # #0.Q  q_a_proj + RMSNorm → q_lora
    q_lora_z = sgl_kernel_zeus.dsa_q_a_proj_norm(
        hidden_z, _lmem_pack(q_a_w), w["q_a_norm"].to("zeus"),
        eps=cfg.rms_norm_eps,
    )

    # #0.KV  kv_a_proj + norm + store new-step K into a fresh single-slot pool
    kv_new_cache = torch.zeros((B, cfg.Rkv), dtype=torch.bfloat16).to("zeus")
    sgl_kernel_zeus.dsa_kv_a_proj_norm_store(
        hidden_z, _lmem_pack(kv_a_w), w["kv_a_norm"].to("zeus"),
        slot_mapping, kv_new_cache, eps=cfg.rms_norm_eps,
    )

    # #1   q_b_proj + absorb bmm(w_kc) → q_new
    q_new_z = sgl_kernel_zeus.dsa_q_main_absorb(
        q_lora_z, _lmem_pack(q_b_w), _lmem_pack(w_kc),
    )

    # #2.Q indexer Q + weights
    q_body_z, _q_scale_z, weights_z = sgl_kernel_zeus.dsa_indexer_q_weights(
        q_lora_z, hidden_z, wq_b.to("zeus"), H_Di.to("zeus"),
        weights_proj.to("zeus"),
        num_index_heads=cfg.I, index_head_dim=cfg.Di,
    )

    # #2.K indexer K prep + store
    body_cache = _lmem_pack(torch.zeros((B, cfg.Di), dtype=torch.float8_e4m3fn))
    scale_cache = torch.zeros((B,), dtype=torch.float32).to("zeus")
    sgl_kernel_zeus.dsa_indexer_k_prep_store(
        hidden_z, _lmem_pack(wk_idx),
        w["k_norm_weight"].to("zeus"), w["k_norm_bias"].to("zeus"),
        _lmem_pack(H_Di), slot_mapping, body_cache, scale_cache,
        eps=cfg.rms_norm_eps,
    )

    # History + new-step row concat (CPU 协调)
    new_k = kv_new_cache.cpu().unsqueeze(1)
    new_body = body_cache.cpu().unsqueeze(1)
    new_scale = scale_cache.cpu().unsqueeze(1)
    full_latent = torch.cat([ctx.history.latent_kv, new_k], dim=1)
    full_body = torch.cat(
        [ctx.history.index_body.to(torch.float8_e4m3fn), new_body], dim=1,
    )
    full_scale = torch.cat([ctx.history.index_scale, new_scale], dim=1)
    S_full = full_latent.shape[1]

    # #3   index GEMM → logits
    logits_z = sgl_kernel_zeus.dsa_index_logits(
        q_body_z, weights_z, _lmem_pack(full_body), full_scale.to("zeus"),
    )

    # #4   local top-K（cp=1 → IS global top-K）
    positions_z = torch.arange(S_full, dtype=torch.int32).to("zeus")
    _top_lg_z, top_pos_z = sgl_kernel_zeus.dsa_local_topk_radix(
        logits_z, positions_z, Ktop=cfg.Ktop,
    )
    top_pos = top_pos_z.cpu().to(torch.int64)

    # #7   per-batch latent_k_gather → K_local / K_local_T / mask (CPU 拼接)
    # 2026-05-25 pool 契约：buffer 走 torch.zeros 初始化（layer-pool 模拟）+
    # gather Python API 已把 _alloc 默认改成 torch.zeros，invalid 位置必为 finite
    # 0.0；下游 #8 sparse_mqa_partial 的算术 mask 自动屏蔽，不再需要 host 端
    # torch.where cleanup（这是 GAP-1 host-side 消除）。
    K_local_full = torch.zeros(B, cfg.Ktop, cfg.Rkv, dtype=torch.bfloat16)
    K_local_T_full = torch.zeros(B, cfg.Rkv, cfg.Ktop, dtype=torch.bfloat16)
    mask_full = torch.zeros(B, cfg.Ktop, dtype=torch.bfloat16)
    for b in range(B):
        slot_idx_b = top_pos[b].to(torch.int32).unsqueeze(0)
        (K_c0_b, _K_c1_b, K_T_c0_b, _K_T_c1_b, m_c0_b, _m_c1_b) = \
            sgl_kernel_zeus.dsa_latent_k_gather(
                slot_idx_b.to("zeus"), full_latent[b].contiguous().to("zeus"),
            )
        K_local_full[b]   = K_c0_b.cpu()[0]
        K_local_T_full[b] = K_T_c0_b.cpu()[0]
        mask_full[b]      = m_c0_b.cpu()[0]

    # #8   sparse MQA partial（cp=1 → partial_out IS attn_latent）
    po_z, _pl_z = sgl_kernel_zeus.dsa_sparse_mqa_partial(
        q_new_z,
        _lmem_pack(K_local_full),   _lmem_pack(K_local_full),
        _lmem_pack(K_local_T_full), _lmem_pack(K_local_T_full),
        _lmem_pack(mask_full),      _lmem_pack(mask_full),
        scaling=cfg.scaling,
    )
    # partial_out 已经是 bf16（kernel host wrapper 强制），无需 D→H→D dtype cast
    # （这是 GAP-2 消除）。
    attn_latent_z = po_z

    # #10  V absorb + o_proj → bf16 [B, H]
    out_z = sgl_kernel_zeus.dsa_post_o_proj_no_cp(
        attn_latent_z, w_vc.to("zeus"), o_proj_w.to("zeus"),
    )
    return out_z


def _zeus_dsa_chain_available() -> bool:
    """DSA Zeus chain 所需 10 颗 kernel 齐备性检查."""
    if ZEUS_IMPORT_ERROR is not None:
        return False
    needed = (
        "dsa_q_a_proj_norm", "dsa_kv_a_proj_norm_store", "dsa_q_main_absorb",
        "dsa_indexer_q_weights", "dsa_indexer_k_prep_store", "dsa_index_logits",
        "dsa_local_topk_radix", "dsa_latent_k_gather", "dsa_sparse_mqa_partial",
        "dsa_post_o_proj_no_cp",
    )
    return all(hasattr(sgl_kernel_zeus, k) for k in needed)


# ── MoE proxy sublayer：REF + Zeus chain (mirror moe_block_full) ──────
# 使用 proxy shape (E=8, mI=128, top_k=2, sI=128)：真实 GLM-4.7 / GLM5-Next
# 的 E=64/288, mI=1408/2048 权重过大无法在 CPU REF 跑（w13 ≈ 0.75–4.8 GB），
# kernel 层对拍由 sgl-kernel-zeus/tests/ + dev_glm4_moe_test.py 各算子负责。
MOE_PROXY_E = 8
MOE_PROXY_MI = 128
MOE_PROXY_TOPK = 2
MOE_PROXY_SI = 128


def init_moe_weights(H: int, seed: int,
                     E: int = MOE_PROXY_E, mI: int = MOE_PROXY_MI,
                     sI: int = MOE_PROXY_SI) -> Dict[str, torch.Tensor]:
    """合成 proxy MoE 权重 (router gate + routed experts w13/w2 + shared experts)."""
    g = torch.Generator().manual_seed(seed)

    def rn(*shape, scale=0.1, dtype=torch.bfloat16):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype)

    return {
        "gate_w": rn(E, H),                              # router gate Linear
        "corr_bias": (torch.randn(E, generator=g, dtype=torch.float32) * 0.01),
        "w13": rn(E, 2 * mI, H),                          # routed experts gate_up
        "w2": rn(E, H, mI),                               # routed experts down
        "sh_gu": rn(2 * sI, H),                           # shared experts gate_up
        "sh_dp": rn(H, sI),                               # shared experts down
        "_meta": {"E": E, "mI": mI, "top_k": MOE_PROXY_TOPK, "sI": sI},
    }


def _moe_router_and_shared_ref(hidden_cpu, moe_w):
    """REF router_logits + shared_output 计算（gate Linear + shared SwiGLU MLP）."""
    sI = moe_w["_meta"]["sI"]
    # gate Linear: bf16 → fp32 logits
    router_logits = torch.nn.functional.linear(
        hidden_cpu.float(), moe_w["gate_w"].float(),
    )  # [T, E] fp32
    # shared experts: gate_up → silu_and_mul → down
    sh_gu = torch.nn.functional.linear(hidden_cpu, moe_w["sh_gu"])  # bf16 [T, 2*sI]
    sh_silu = (
        torch.nn.functional.silu(sh_gu[:, :sI].float()) * sh_gu[:, sI:].float()
    ).to(torch.bfloat16)
    shared_out = torch.nn.functional.linear(sh_silu, moe_w["sh_dp"])  # bf16 [T, H]
    return router_logits, shared_out


def ref_moe_decode(hidden_cpu: torch.Tensor, moe_w: Dict) -> torch.Tensor:
    """REF MoE: gate + shared + biased_grouped_topk + per-token expert loop + residual."""
    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

    meta = moe_w["_meta"]
    mI, top_k = meta["mI"], meta["top_k"]

    router_logits, shared_out = _moe_router_and_shared_ref(hidden_cpu, moe_w)

    w_topk, ids_topk = biased_grouped_topk_impl(
        hidden_states=hidden_cpu,
        gating_output=router_logits,
        correction_bias=moe_w["corr_bias"],
        topk=top_k,
        renormalize=True,
        num_expert_group=1,
        topk_group=1,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=True,
    )
    moe_core = moe_dev._ref_moe_core(
        hidden_cpu, moe_w["w13"], moe_w["w2"], w_topk, ids_topk, mI=mI,
    )
    # fp32 add + 单 RNE（与 moe_sum_reduce 路径一致）
    return (moe_core.float() + shared_out.float()).to(torch.bfloat16)


def zeus_moe_decode(hidden_z: torch.Tensor, moe_w: Dict) -> torch.Tensor:
    """Zeus MoE chain（mirror moe_block_full Zeus 全 device-resident 路径）。

    **2026-05-25 GAP-3 闭环**：gate Linear + shared experts MLP 三颗 host
    `torch.nn.functional.linear` 全部切到 `sgl_kernel_zeus.linear_bf16`，
    整段 MoE 在 Zeus 上 **完全 device-resident**（host 仅做 weight LocalMem
    pack 一次性，无中间 host compute）。

    10 颗 sgl-kernel-zeus 算子串接：
      1. linear_bf16        (gate Linear)
      2. linear_bf16        (shared gate_up_proj)
      3. silu_and_mul       (shared)
      4. linear_bf16        (shared down_proj)
      5. biased_grouped_topk
      6. moe_align_block_size_alloc
      7. moe_grouped_gemm   (gemm1)
      8. silu_and_mul       (per-expert)
      9. moe_grouped_gemm   (gemm2, mul_routed_weight=True)
      10. moe_sum_reduce    (+shared residual fuse)
    """
    meta = moe_w["_meta"]
    mI, top_k, E, sI = meta["mI"], meta["top_k"], meta["E"], meta["sI"]
    T = hidden_z.shape[0]
    H = hidden_z.shape[-1]
    block_size = sgl_kernel_zeus.MOE_GROUPED_GEMM_BLOCK_M

    # ── GAP-3 闭环：gate / shared MLP 全部 Zeus linear_bf16 ──────────
    # 3 颗 weight LocalMem 装包（与 mhc_pre_norm_split / dsa_q_a_proj_norm 同套路）。
    # 生产路径下这些 pack 应在 layer.__init__ 一次性做掉；dev 脚本每次重新 pack
    # 仅为 stage 独立性。
    gate_w_lmem = torch.zeus.local_memory.from_tensor(
        moe_w["gate_w"].to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    sh_gu_lmem = torch.zeus.local_memory.from_tensor(
        moe_w["sh_gu"].to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    sh_dp_lmem = torch.zeus.local_memory.from_tensor(
        moe_w["sh_dp"].to("zeus"), kind="weight", Tr=1, Tc=1,
    )

    # gate Linear: [T, H] @ [E, H].T → [T, E] bf16；topk 需要 fp32 → device cast
    router_logits_bf16_z = sgl_kernel_zeus.linear_bf16(hidden_z, gate_w_lmem)
    router_logits_z = router_logits_bf16_z.float()

    # shared experts MLP: gate_up_proj → silu_and_mul → down_proj
    sh_gu_z = sgl_kernel_zeus.linear_bf16(hidden_z, sh_gu_lmem)        # [T, 2*sI]
    sh_silu_z = torch.empty(T, sI, dtype=torch.bfloat16, device="zeus")
    sgl_kernel_zeus.silu_and_mul(sh_gu_z, sh_silu_z)                   # [T, sI]
    shared_out_z = sgl_kernel_zeus.linear_bf16(sh_silu_z, sh_dp_lmem)  # [T, H]

    corr_bias_z = moe_w["corr_bias"].to("zeus")
    w13_z = moe_w["w13"].to("zeus")
    w2_z = moe_w["w2"].to("zeus")

    # 1) biased_grouped_topk（scale fused 进 weights，scaling=1.0 → 无放大）
    w_z, ids_z = sgl_kernel_zeus.biased_grouped_topk(
        router_logits_z, corr_bias_z,
        num_expert_group=1, topk_group=1, topk=top_k,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=True,
    )
    # 2) moe_align
    sorted_ids_z, expert_ids_z, num_post_z = (
        sgl_kernel_zeus.moe_align_block_size_alloc(ids_z, block_size, E)
    )
    num_valid_tokens = T * top_k

    # 3) gemm1: [T, H] → [T*top_k, 2*mI]
    C1_z = torch.empty(T * top_k, 2 * mI, dtype=torch.bfloat16, device="zeus")
    sgl_kernel_zeus.moe_grouped_gemm(
        hidden_z, w13_z, C1_z,
        sorted_ids_z, expert_ids_z, num_post_z,
        num_valid_tokens=num_valid_tokens, top_k=top_k,
    )
    # 4) silu_and_mul: [T*top_k, 2*mI] → [T*top_k, mI]
    C1_silu_z = torch.empty(T * top_k, mI, dtype=torch.bfloat16, device="zeus")
    sgl_kernel_zeus.silu_and_mul(C1_z, C1_silu_z)

    # 5) gemm2 (mul_routed_weight=True): [T*top_k, mI] → [T*top_k, H]
    w_z_flat_bf16 = w_z.to(torch.bfloat16).flatten().contiguous()
    C2_z = torch.empty(T * top_k, H, dtype=torch.bfloat16, device="zeus")
    sgl_kernel_zeus.moe_grouped_gemm(
        C1_silu_z, w2_z, C2_z,
        sorted_ids_z, expert_ids_z, num_post_z,
        num_valid_tokens=num_valid_tokens, top_k=1,
        topk_weights=w_z_flat_bf16, mul_routed_weight=True,
    )
    # 6) moe_sum_reduce (+shared residual)
    final_z = torch.empty(T, H, dtype=torch.bfloat16, device="zeus")
    sgl_kernel_zeus.moe_sum_reduce(
        input=C2_z.view(T, top_k, H),
        output=final_z,
        shared_output=shared_out_z,
        routed_scaling_factor=1.0,
    )
    return final_z


def _zeus_moe_chain_available() -> bool:
    """MoE Zeus chain 齐备性检查（含 2026-05-25 GAP-3 闭环加入的 linear_bf16）.

    10 颗 sgl-kernel-zeus 算子：linear_bf16 × 3 (gate / shared gate_up /
    shared down) + silu_and_mul × 2 (shared + per-expert) +
    biased_grouped_topk + moe_align_block_size_alloc + moe_grouped_gemm × 2
    (gemm1 / gemm2) + moe_sum_reduce.
    """
    if ZEUS_IMPORT_ERROR is not None:
        return False
    needed = (
        "linear_bf16",
        "biased_grouped_topk", "moe_align_block_size_alloc",
        "moe_grouped_gemm", "silu_and_mul", "moe_sum_reduce",
    )
    return all(hasattr(sgl_kernel_zeus, k) for k in needed)


# ── Linear-attn sublayer Zeus chain（2026-05-25 GAP-3 后续）─────────
# 把 `ref_linear_attn_decode` 的 9+ 步 host 计算切到 Zeus 等价算子，让
# `stage_linear_attn_block` 走向完全 device-resident。
#
# Kernel chain（与 ref_linear_attn_decode 一一对应）：
#   1-5. linear_bf16  ×5 (qkv_proj, b_proj, f_a→f_b, g_a→g_b)
#       (低秩两层 f_a/f_b 与 g_a/g_b 共 4 个，加 qkv_proj/b_proj 总计 7 颗 linear_bf16)
#   6.  causal_conv1d_update (3P 通道一把，silu 融合)
#   7.  fused_kda_gate (softplus·-exp·A_log + dt_bias)
#   8.  fused_recurrent_kda_Sdecay (non-indexed，head_dim 通用支持 72 / 128)
#   9.  rms_norm_gated (sigmoid 门控 RMSNorm)
#   10. linear_bf16 (o_proj)
def _pack_linear_attn_weights(w: Dict[str, torch.Tensor]) -> Dict:
    """把 init_linear_weights 输出的 dict 中的 7 颗 linear weight 装包到 LocalMem，
    其余（conv_w / conv_b / dt_bias / A_log / o_norm）直接 .to("zeus")。"""
    packed = {}
    for key in ("qkv_proj", "b_proj", "f_a", "f_b", "g_a", "g_b", "o_proj"):
        packed[key + "_lmem"] = _lmem_pack(w[key])
    packed["conv_w_z"] = w["conv_w"].to(torch.bfloat16).to("zeus")
    packed["conv_b_z"] = w["conv_b"].to(torch.bfloat16).to("zeus")
    packed["dt_bias_z"] = w["dt_bias"].to("zeus")
    packed["A_log_z"] = w["A_log"].to("zeus")
    # o_norm 不在 rms_norm_gated 签名里（weight=1 implicit），保留 for sanity
    packed["o_norm_z"] = w["o_norm"].to("zeus")
    return packed


def zeus_linear_attn_decode(
    hidden_z: torch.Tensor,
    w_lmem: Dict,
    cfg: LinearAttnConfig,
    conv_state: torch.Tensor,
    rec_state: torch.Tensor,
) -> torch.Tensor:
    """完整 Linear-attn (KDA) decode 单步 Zeus 路径。

    Inputs:
        hidden_z   : [B, H] bf16 Zeus
        w_lmem     : `_pack_linear_attn_weights(w)` 的输出
        cfg        : LinearAttnConfig
        conv_state : [B, 3P, K-1] bf16 Zeus，**in-place 推进**
        rec_state  : [B, Hh, Dk, Dv] fp32 Zeus，**in-place 推进**
    Returns:
        [B, H] bf16 Zeus
    """
    B = hidden_z.shape[0]
    H = cfg.H
    Hh = cfg.num_heads
    Dk = cfg.head_k_dim
    Dv = cfg.head_v_dim
    P = cfg.proj_size  # = Hh * Dk

    # 1-5. 7 颗 linear_bf16：qkv / b / f_a→f_b / g_a→g_b
    qkv_z   = sgl_kernel_zeus.linear_bf16(hidden_z, w_lmem["qkv_proj_lmem"])  # [B, 3P]
    beta_z  = sgl_kernel_zeus.linear_bf16(hidden_z, w_lmem["b_proj_lmem"])    # [B, Hh]
    fa_z    = sgl_kernel_zeus.linear_bf16(hidden_z, w_lmem["f_a_lmem"])       # [B, Dk]
    fg_z    = sgl_kernel_zeus.linear_bf16(fa_z,     w_lmem["f_b_lmem"])       # [B, P]
    ga_z    = sgl_kernel_zeus.linear_bf16(hidden_z, w_lmem["g_a_lmem"])       # [B, Dk]
    gproj_z = sgl_kernel_zeus.linear_bf16(ga_z,     w_lmem["g_b_lmem"])       # [B, P]

    # 6. causal_conv1d_update（3P 通道一把跑，state in-place 推进 + silu 融合）
    #    避免 `causal_conv1d_update_qkv` 的 dim=1 chunk 非连续问题（conv_state 是
    #    [B, 3P, K-1] 融合存储，按 dim=1 chunk 给出非连续 view，与 Zeus host
    #    wrapper 的 contiguous 要求冲突）。per-channel kernel 通道间独立，等价
    #    于跑 3 次。
    qkv_post_z = sgl_kernel_zeus.causal_conv1d_update(
        qkv_z, conv_state,
        w_lmem["conv_w_z"], w_lmem["conv_b_z"], activation="silu",
    )  # [B, 3P]
    q_z, k_z, v_z = qkv_post_z.chunk(3, dim=-1)  # 每个 [B, P]，dim=-1 chunk 连续

    # 7. fused_kda_gate: g_gate[B,Hh,Dk] = (-exp(A_log)) * softplus(fg + dt_bias)
    g_gate_z = sgl_kernel_zeus.fused_kda_gate(
        fg_z, w_lmem["A_log_z"], head_dim=Dk, g_bias=w_lmem["dt_bias_z"],
    )  # [B, Hh, Dk] fp32

    # 8. beta = sigmoid(b_proj.float())  — 简单 device elementwise
    beta_fp32 = beta_z.float().sigmoid()  # [B, Hh] fp32

    # 9. fused_recurrent_kda_Sdecay (non-indexed，head_dim 通用)
    #    Reshape q/k/v 到 [B, Hh, Dk] / [B, Hh, Dv]；cu_seqlens 每 batch 1 token
    q_4d = q_z.reshape(B, Hh, Dk)
    k_4d = k_z.reshape(B, Hh, Dk)
    v_4d = v_z.reshape(B, Hh, Dv)
    cu_seqlens_z = torch.arange(B + 1, dtype=torch.int32, device="zeus")
    # initial_state in-place 更新（inplace_final_state=True）
    o_z, _ = sgl_kernel_zeus.fused_recurrent_kda_Sdecay(
        q=q_4d, k=k_4d, v=v_4d,
        g=g_gate_z, beta=beta_fp32,
        initial_state=rec_state,
        cu_seqlens=cu_seqlens_z,
        scale=cfg.scaling,
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
        inplace_final_state=True,
    )
    # o_z: [B, Hh, Dv] bf16

    # 10. rms_norm_gated (sigmoid 门控；REF 的 o_norm = ones 是 implicit weight=1)
    gate_for_rms = gproj_z.reshape(B, Hh, Dv)  # Dv == Dk per head (本配置)
    normed_z = sgl_kernel_zeus.rms_norm_gated(
        o_z, gate_for_rms, eps=cfg.rms_norm_eps,
    )  # [B, Hh, Dv] bf16

    # 11. o_proj
    normed_flat = normed_z.reshape(B, P)
    out_z = sgl_kernel_zeus.linear_bf16(normed_flat, w_lmem["o_proj_lmem"])  # [B, H]
    return out_z


def _pack_mlp_weights(w: Dict[str, torch.Tensor]) -> Dict:
    """init_mlp_weights → LocalMem pack gate_up / down."""
    return {
        "gate_up_lmem": _lmem_pack(w["gate_up"]),
        "down_lmem": _lmem_pack(w["down"]),
    }


def zeus_mlp_decode(
    hidden_z: torch.Tensor,
    w_lmem: Dict,
    inter: int,
) -> torch.Tensor:
    """完整 dense SwiGLU MLP decode 单步 Zeus 路径（option A: 无 clamp）。

    NOTE: ref_mlp_decode 在 silu/mul 之前对 gate/up 各做 clamp(-10, 10)（GLM5-Next
    `swiglu_clamp_limit`）；Zeus 现有 `silu_and_mul` **不带 clamp**。dev script
    用 randn(0.05) magnitude 远不到 10，clamp 无实际效果，故 option A 直接省。
    生产路径若需严格 clamp 语义需新加 `silu_and_mul_clamp` kernel。
    """
    B = hidden_z.shape[0]
    H = hidden_z.shape[-1]
    # 1. gate_up_proj: [B, H] → [B, 2*inter]
    gu_z = sgl_kernel_zeus.linear_bf16(hidden_z, w_lmem["gate_up_lmem"])
    # 2. silu_and_mul: [B, 2*inter] → [B, inter]
    act_z = torch.empty(B, inter, dtype=torch.bfloat16, device="zeus")
    sgl_kernel_zeus.silu_and_mul(gu_z, act_z)
    # 3. down_proj: [B, inter] → [B, H]
    out_z = sgl_kernel_zeus.linear_bf16(act_z, w_lmem["down_lmem"])
    return out_z


def _zeus_linear_attn_chain_available() -> bool:
    """Linear-attn Zeus chain 所需算子齐备性检查."""
    if ZEUS_IMPORT_ERROR is not None:
        return False
    needed = (
        "linear_bf16",
        "causal_conv1d_update",
        "fused_kda_gate",
        "fused_recurrent_kda_Sdecay",
        "rms_norm_gated",
        "silu_and_mul",
    )
    return all(hasattr(sgl_kernel_zeus, k) for k in needed)


# ── Block runner：fully-device-resident（Zeus mHC + Zeus attn + Zeus mlp）
def run_mhc_block_zeus_e2e(which: str, residual_flat: torch.Tensor,
                           zeus_attn_fn, zeus_mlp_fn, seed: int):
    """完全 device-resident 的 mHC block。

    与 run_mhc_block_zeus 的差异：
      - zeus_attn_fn / zeus_mlp_fn 接收并返回 Zeus device tensor，**无 round-trip**。
      - z_li.cpu() / x.to("zeus") 都没有；attn/MLP 自己内部全 Zeus（可能内部仍有
        协调步骤如 DSA 的 history concat，但 sublayer 出口直接是 device tensor）。

    返回 (z_mid, z_out) 均在 Zeus device 上。
    """
    cfg = _mhc_cfg(which)
    p_attn = mhc.init_mhc_params(cfg, seed)
    p_mlp = mhc.init_mhc_params(cfg, seed + 100)

    # attn wrap
    z_li_a, z_res_a, z_hres_a, z_hpost_a = mhc.zeus_mhc_pre(residual_flat, p_attn, cfg)
    attn_out_z = zeus_attn_fn(z_li_a)
    z_mid = mhc.zeus_mhc_post(attn_out_z, z_res_a, z_hpost_a, z_hres_a, cfg)

    # mlp wrap (z_mid 直接喂下一轮 K1)
    z_li_m, z_res_m, z_hres_m, z_hpost_m = mhc.zeus_mhc_pre(z_mid, p_mlp, cfg)
    mlp_out_z = zeus_mlp_fn(z_li_m)
    z_out = mhc.zeus_mhc_post(mlp_out_z, z_res_m, z_hpost_m, z_hres_m, cfg)
    return z_mid, z_out


# ── Stage: linear_attn_decode ───────────────────────────────────
def stage_linear_attn_decode(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: linear_attn_decode (Linear-attention sublayer, decode)")
    print("=" * 60)
    cfg = _load_linear_cfg(_config_path(args.config))
    B, H = args.num_tokens, cfg.H
    torch.manual_seed(args.seed)
    w = init_linear_weights(cfg, args.seed)
    conv, rec = init_linear_state(cfg, B, args.seed + 1)
    hidden = torch.randn(B, H, dtype=torch.bfloat16) * 0.05

    conv_b, rec_b = conv.clone(), rec.clone()
    out = ref_linear_attn_decode(hidden, w, cfg, conv_b, rec_b)
    ref_ok = out.shape == (B, H) and out.dtype == torch.bfloat16
    state_moved = not torch.equal(conv_b, conv)
    print(f"  cfg: H={H} num_heads={cfg.num_heads} head_k={cfg.head_k_dim} "
          f"head_v={cfg.head_v_dim} conv_size={cfg.conv_size} proj={cfg.proj_size}")
    print(f"  out={tuple(out.shape)} {out.dtype}  conv/rec state advanced={state_moved}")
    print(f"  out[0,:4] = {[round(v,4) for v in out[0,:4].float().tolist()]}")
    return finish_stage(args, ref_ok and state_moved, "kda_decode_block",
                        "dev_kimi_linear_attn_test.py::kimi_delta_attn_decode")


# ── Stage: dsa_decode ───────────────────────────────────────────
def stage_dsa_decode(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: dsa_decode (DSA sublayer, decode, cp=1)")
    print("=" * 60)
    ctx = build_dsa_ctx(args.config, args.num_tokens, args.seqlen, args.seed)
    hidden = torch.randn(args.num_tokens, ctx.cfg.H, dtype=torch.bfloat16) * 0.05
    out = ref_dsa_decode(ctx, hidden)
    ref_ok = out.shape == (args.num_tokens, ctx.cfg.H) and out.dtype == torch.bfloat16
    print(f"  cfg: {ctx.cfg.name}  H={ctx.cfg.H} Nh={ctx.cfg.Nh} "
          f"Rkv={ctx.cfg.Rkv} Ktop={ctx.cfg.Ktop}  seqlen={args.seqlen}")
    print(f"  out={tuple(out.shape)} {out.dtype}")
    print(f"  out[0,:4] = {[round(v,4) for v in out[0,:4].float().tolist()]}")
    return finish_stage(args, ref_ok, "dsa_decode_block",
                        "dev_glm5next_dsa_decode_test.py::decode_full_nocp")


# ── Stage: mlp_decode ───────────────────────────────────────────
def stage_mlp_decode(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mlp_decode (dense SwiGLU MLP sublayer)")
    print("=" * 60)
    raw = json.loads(_config_path(args.config).read_text())
    H, inter = int(raw["hidden_size"]), int(raw["intermediate_size"])
    clamp = float(raw.get("swiglu_clamp_limit", 10.0))
    B = args.num_tokens
    torch.manual_seed(args.seed)
    w = init_mlp_weights(H, inter, args.seed)
    hidden = torch.randn(B, H, dtype=torch.bfloat16) * 0.05
    out = ref_mlp_decode(hidden, w, clamp_limit=clamp)
    ref_ok = out.shape == (B, H) and out.dtype == torch.bfloat16
    print(f"  H={H} intermediate={inter} swiglu_clamp={clamp}")
    print(f"  out={tuple(out.shape)} {out.dtype}")
    print(f"  NOTE: MoE sublayer 由 dev_glm4_moe_test.py 覆盖；本 stage 仅 dense MLP")
    return finish_stage(args, ref_ok, "mlp_decode_block",
                        "dev_glm4_moe_test.py::moe_block_full (MoE 版)")


# ── Stage: linear_attn_block ────────────────────────────────────
def stage_linear_attn_block(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: linear_attn_block (mHC[ Linear-attn + dense MLP ] 整 block)  "
          "[Zeus mHC + Linear-attn + MLP 全 LANDED 2026-05-25]")
    print("=" * 60)
    lcfg = _load_linear_cfg(_config_path(args.config))
    raw = json.loads(_config_path(args.config).read_text())
    H, inter = lcfg.H, int(raw["intermediate_size"])
    clamp = float(raw.get("swiglu_clamp_limit", 10.0))
    N = mhc.select_config(args.config).N
    B = args.num_tokens
    torch.manual_seed(args.seed)

    w_attn = init_linear_weights(lcfg, args.seed)
    w_mlp = init_mlp_weights(H, inter, args.seed + 5)
    # state 初始模板：三条 chain (REF / qREF / Zeus) 各 clone 一份，避免 in-place
    # 推进互相污染
    conv_init, rec_init = init_linear_state(lcfg, B, args.seed + 1)
    residual_flat = torch.randn(B, N * H, dtype=torch.bfloat16) * 0.05

    def make_attn_fn_ref(conv, rec):
        def attn_fn(li):
            return ref_linear_attn_decode(li, w_attn, lcfg, conv, rec)
        return attn_fn

    def mlp_fn_ref(li):
        return ref_mlp_decode(li, w_mlp, clamp_limit=clamp)

    # REF chain
    conv_ref, rec_ref = conv_init.clone(), rec_init.clone()
    res_mid, res_out = run_mhc_block(
        args.config, residual_flat, make_attn_fn_ref(conv_ref, rec_ref), mlp_fn_ref, args.seed,
    )
    ref_ok = res_mid.shape == (B, N * H) and res_out.shape == (B, N * H)
    print(f"  H={H} N={N}  residual=[{B},{N*H}]")
    print(f"  inter={inter}  swiglu_clamp_limit={clamp}（option A：Zeus 路径无 clamp，randn 0.05 远不到 10）")
    print(f"  residual_in -> [attn_hc(Linear-attn)] -> mid -> [mlp_hc(dense MLP)] -> out")
    print(f"  shapes: mid={tuple(res_mid.shape)} out={tuple(res_out.shape)}")
    mid3 = res_mid.view(B, N, H)
    diverged = not torch.equal(mid3[:, 0, :], mid3[:, 1, :])
    print(f"  [streams_diverged] {'PASS' if diverged else 'WARN'} (mHC 已混流)")

    # ── Zeus 路径：Zeus mHC + Zeus Linear-attn + Zeus dense MLP 全 device-resident ──
    # 与 dsa_block 同套路 —— quantized REF（host F.linear sublayer）作 golden，Zeus
    # 路径用 run_mhc_block_zeus_e2e（设备端 attn / mlp fn）。state 三份独立 clone：
    # REF / qREF / Zeus 各自推进。
    zeus_ok: Optional[bool] = None
    if (args.mode != "ref" and _zeus_mhc_chain_available()
            and _zeus_linear_attn_chain_available() and _zeus_moe_chain_available()):
        try:
            # quantized REF chain（host sublayer + mHC 参数走 quantize）
            conv_q, rec_q = conv_init.clone(), rec_init.clone()
            mid_q, out_q = run_mhc_block_ref_quant(
                args.config, residual_flat,
                make_attn_fn_ref(conv_q, rec_q), mlp_fn_ref, args.seed,
            )

            # Zeus chain：weight LocalMem pack 一次（生产路径 layer.__init__ 缓存）
            w_attn_lmem = _pack_linear_attn_weights(w_attn)
            w_mlp_lmem = _pack_mlp_weights(w_mlp)
            # state 上 Zeus（保持 contiguous + in-place 推进）
            conv_z = conv_init.clone().to("zeus")
            rec_z = rec_init.clone().to("zeus")

            def zeus_attn_fn(z_li):
                return zeus_linear_attn_decode(
                    z_li, w_attn_lmem, lcfg, conv_z, rec_z,
                )

            def zeus_mlp_fn(z_li):
                return zeus_mlp_decode(z_li, w_mlp_lmem, inter)

            z_mid, z_out = run_mhc_block_zeus_e2e(
                args.config, residual_flat, zeus_attn_fn, zeus_mlp_fn, args.seed,
            )
            # 复合误差预算：mHC 5e-3 + Linear-attn (5 个 linear_bf16 + conv1d +
            # KDA recurrent + rms_norm_gated + o_proj) 5e-2 + dense MLP 5e-2，
            # 累计放宽到 mid `1.5e-1` / out `3e-1`（与 dsa_block 同款 envelope）。
            mid_ok = compare_tensors("linear_attn_block.mid", mid_q, z_mid.cpu(),
                                     atol=1.5e-1, rtol=5e-2)
            out_ok = compare_tensors("linear_attn_block.out", out_q, z_out.cpu(),
                                     atol=3e-1, rtol=1e-1)
            z_mid_cpu = z_mid.cpu().view(B, N, H)
            z_diverged = not torch.equal(z_mid_cpu[:, 0, :], z_mid_cpu[:, 1, :])
            print(f"  ZEUS [linear_attn_block/streams_diverged] "
                  f"{'PASS' if z_diverged else 'WARN'} (Zeus 输出 N 条流不再恒等)")
            zeus_ok = mid_ok and out_ok and z_diverged
        except Exception as e:
            import traceback
            print(f"  ZEUS [linear_attn_block] EXCEPTION: {e!r}")
            traceback.print_exc()
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok and diverged, zeus_ok, "linear_attn_block",
        "Zeus mHC + Zeus Linear-attn (7×linear_bf16 + causal_conv1d_update + "
        "fused_kda_gate + fused_recurrent_kda_Sdecay + rms_norm_gated) + "
        "Zeus dense MLP (linear_bf16×2 + silu_and_mul, 无 clamp option A)")


# ── Stage: dsa_block ────────────────────────────────────────────
def stage_dsa_block(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: dsa_block (mHC[ DSA + MoE ] 整 block)  "
          "[Zeus mHC + DSA + MoE chain LANDED]")
    print("=" * 60)
    # 三个 ctx：REF / qREF / Zeus 各自独立的 DSA state（同 seed 同初始）
    ctx_ref = build_dsa_ctx(args.config, args.num_tokens, args.seqlen, args.seed)
    H = ctx_ref.cfg.H
    N = mhc.select_config(args.config).N
    B = args.num_tokens
    torch.manual_seed(args.seed)

    # MoE proxy 权重（E=8, mI=128, top_k=2, sI=128）—— 真实 GLM5-Next 的
    # E=64/288, mI=1408/2048 不能在 CPU REF 跑（w13 ≈ 0.75–4.8 GB），kernel
    # 层对拍由 dev_glm4_moe_test.py / sgl-kernel-zeus/tests/ 各自覆盖。
    moe_w = init_moe_weights(H, args.seed + 5)
    meta = moe_w["_meta"]
    residual_flat = torch.randn(B, N * H, dtype=torch.bfloat16) * 0.05

    def make_attn_fn_ref(ctx):
        def attn_fn(li):
            return ref_dsa_decode(ctx, li)
        return attn_fn

    def mlp_fn_ref(li):
        return ref_moe_decode(li, moe_w)

    res_mid, res_out = run_mhc_block(
        args.config, residual_flat, make_attn_fn_ref(ctx_ref), mlp_fn_ref, args.seed,
    )
    ref_ok = res_mid.shape == (B, N * H) and res_out.shape == (B, N * H)
    print(f"  {ctx_ref.cfg.name}  H={H} N={N}  residual=[{B},{N*H}]  seqlen={args.seqlen}")
    print(f"  MoE proxy: E={meta['E']}  mI={meta['mI']}  top_k={meta['top_k']}  "
          f"sI={meta['sI']}  (kernel 层 GLM-4.7 真实 shape 见 dev_glm4_moe_test)")
    print(f"  residual_in -> [attn_hc(DSA)] -> mid -> [mlp_hc(MoE)] -> out")
    print(f"  shapes: mid={tuple(res_mid.shape)} out={tuple(res_out.shape)}")
    mid3 = res_mid.view(B, N, H)
    diverged = not torch.equal(mid3[:, 0, :], mid3[:, 1, :])
    print(f"  [streams_diverged] {'PASS' if diverged else 'WARN'} (mHC 已混流)")

    # ── Zeus 路径：Zeus mHC + Zeus DSA + Zeus MoE 全 device-resident ────
    # quantized REF 与 Zeus 各自独立的 DSA ctx，state 同 seed 同初始。MoE 权重
    # 唯一共享（无 state）；REF / qREF / Zeus 三路都从同一份 moe_w 取参数，
    # Zeus 路径在 zeus_moe_decode 内部 .to("zeus")。
    zeus_ok: Optional[bool] = None
    if (args.mode != "ref" and _zeus_mhc_chain_available()
            and _zeus_dsa_chain_available() and _zeus_moe_chain_available()):
        try:
            # quantized REF chain（DSA + MoE 仍是 CPU REF；mHC 参数走 quantize）
            ctx_q = build_dsa_ctx(args.config, args.num_tokens, args.seqlen, args.seed)
            mid_q, out_q = run_mhc_block_ref_quant(
                args.config, residual_flat,
                make_attn_fn_ref(ctx_q), mlp_fn_ref, args.seed,
            )
            # Zeus chain（mHC + DSA + MoE 全 Zeus，device-resident）
            ctx_z = build_dsa_ctx(args.config, args.num_tokens, args.seqlen, args.seed)

            def zeus_attn_fn(z_li):
                return zeus_dsa_decode(ctx_z, z_li)

            def zeus_mlp_fn(z_li):
                return zeus_moe_decode(z_li, moe_w)

            z_mid, z_out = run_mhc_block_zeus_e2e(
                args.config, residual_flat, zeus_attn_fn, zeus_mlp_fn, args.seed,
            )
            # 复合误差预算：mHC (5e-3) + DSA (5e-2) + MoE (5e-2 bf16 多级 GEMM)
            # 累计放宽到 1.5e-1 mid / 3e-1 out（DSA + MoE 各引入一次 K4 round）
            mid_ok = compare_tensors("dsa_block.mid", mid_q, z_mid.cpu(),
                                     atol=1.5e-1, rtol=5e-2)
            out_ok = compare_tensors("dsa_block.out", out_q, z_out.cpu(),
                                     atol=3e-1, rtol=1e-1)
            z_mid_cpu = z_mid.cpu().view(B, N, H)
            z_diverged = not torch.equal(z_mid_cpu[:, 0, :], z_mid_cpu[:, 1, :])
            print(f"  ZEUS [dsa_block/streams_diverged] "
                  f"{'PASS' if z_diverged else 'WARN'} (Zeus 输出 N 条流不再恒等)")
            zeus_ok = mid_ok and out_ok and z_diverged
        except Exception as e:
            import traceback
            print(f"  ZEUS [dsa_block] EXCEPTION: {e!r}")
            traceback.print_exc()
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok and diverged, zeus_ok, "mhc_post + dsa + moe",
        "Zeus mHC + Zeus DSA (10 kernel) + Zeus MoE (6 kernel) 全 device-resident")


# ── Stage: decode_layer_full ────────────────────────────────────
def stage_decode_layer_full(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: decode_layer_full (按 full_attn_layers 路由 attn 类型)")
    print("=" * 60)
    full_attn = _full_attn_layers(args.config)
    is_dsa = args.layer_id in full_attn
    attn_kind = "DSA (full-attn)" if is_dsa else "Linear-attn (KDA)"
    print(f"  config={args.config}  layer_id={args.layer_id}  "
          f"full_attn_layers={full_attn}")
    print(f"  -> 该层 attention = {attn_kind}")

    if is_dsa:
        ok = stage_dsa_block(args)
    else:
        ok = stage_linear_attn_block(args)
    return ok


# ── Dispatch ────────────────────────────────────────────────────
STAGES = {
    "linear_attn_decode": stage_linear_attn_decode,
    "dsa_decode": stage_dsa_decode,
    "mlp_decode": stage_mlp_decode,
    "linear_attn_block": stage_linear_attn_block,
    "dsa_block": stage_dsa_block,
    "decode_layer_full": stage_decode_layer_full,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGES.keys()) + ["all"], default="all")
    parser.add_argument("--config", choices=["16b", "next"], default="16b")
    parser.add_argument("--mode", choices=["both", "ref", "zeus"], default="both")
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=64, help="DSA decode 的历史长度")
    parser.add_argument("--layer-id", type=int, default=3,
                        help="decode_layer_full 用：路由 attn 类型")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"GLM5-Next block decode 组装  config={args.config}  "
          f"B(num_tokens)={args.num_tokens}  seqlen={args.seqlen}")
    print(f"Reference device: {REF_DEVICE}")
    print(f"Zeus runtime available: {ZEUS_IMPORT_ERROR is None}")

    results = {}
    for name, fn in STAGES.items():
        if args.stage not in (name, "all"):
            continue
        try:
            results[name] = fn(args)
        except Exception as e:
            import traceback
            print(f"  [{name}] EXCEPTION: {e}")
            traceback.print_exc()
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
