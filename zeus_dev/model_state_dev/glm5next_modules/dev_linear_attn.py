"""
GLM5-Next Linear Attention (KDA) sublayer 独立模块 + REF↔Zeus 对拍.

目标:
  - 把 GLM5-Next decoder layer 的 Linear-attention (KDA) decode sublayer 抽成
    独立模块 ``Glm5NextLinearAttn``，暴露 ``__init__`` + ``init_state`` +
    ``forward`` (REF, host bf16) + ``forward_zeus`` (Zeus device-resident).
  - 设计与 ``dev_moe.py`` 同构（都来自同一类 module dev test scaffolding，
    公共部分见 ``_common.py``).
  - 与 dev_kimi_linear_attn_test.py 的 kernel 层对拍互补：本脚本对拍 **整段
    Linear-attn sublayer 的输入+state→输出** 路径.

形状 (从 linear_attn_config 子段读取):
  - 16b:   head_dim=72,  num_heads 由 config 决定; H=2048
  - next:  head_dim=128, num_heads 由 config 决定; H=4096

Chain (与 dev_glm5next_block_decode_test.zeus_linear_attn_decode 一致):
  1-7.  linear_bf16 × 6  (qkv_proj, f_a→f_b, g_a→g_b, o_proj) 全 LocalMem
   8.   **linear_bf16_outfp32** (b_proj) → fp32 直接出, 消除 host `.float()` cast
   9.   causal_conv1d_update (qkv 3P 通道一起跑, silu 融合, conv_state 原位更新)
   10.  fused_kda_gate         (softplus·-exp·A_log + dt_bias → g_gate fp32)
   11.  fused_recurrent_kda_Sdecay  (KDA recurrent, rec_state 原位更新)
   12.  rms_norm_gated         (sigmoid 门控 RMSNorm)

State:
  - conv_state: bf16 [B, 3*P, K-1]  (短卷积窗口，原位推进)
  - rec_state:  fp32 [B, Hh, Dk, Dv]  (KDA recurrent S 矩阵, 原位推进)

用法:
  python glm5next_modules/dev_linear_attn.py                # 16b / both
  python glm5next_modules/dev_linear_attn.py --config next
  python glm5next_modules/dev_linear_attn.py --mode zeus
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# 公共脚手架
import _common
from _common import (
    ZEUS_IMPORT_ERROR, sgl_kernel_zeus,
    config_path, compare_tensors, zeus_chain_available,
    make_argparser, print_header, print_summary,
)


# ── Config ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Glm5NextLinearAttnConfig:
    """Linear-attention sublayer config (单 device, TP=1, CP=1)."""
    H: int
    num_heads: int
    head_k_dim: int          # linear_attn_config.head_dim (key/query head dim)
    head_v_dim: int          # linear_value_head_dim (== head_k_dim in current configs)
    conv_size: int           # short_conv_kernel_size
    rms_norm_eps: float
    name: str = "16b"

    @property
    def proj_size(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def scaling(self) -> float:
        return self.head_k_dim ** -0.5

    @classmethod
    def from_json(cls, path: Path, name: Optional[str] = None) -> "Glm5NextLinearAttnConfig":
        raw = json.loads(Path(path).read_text())
        lac = raw["linear_attn_config"]
        return cls(
            H=int(raw["hidden_size"]),
            num_heads=int(lac["num_heads"]),
            head_k_dim=int(lac["head_dim"]),
            head_v_dim=int(raw.get("linear_value_head_dim", lac["head_dim"])),
            conv_size=int(lac["short_conv_kernel_size"]),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
            name=name or Path(path).stem,
        )


def load_cfg(which: str) -> Glm5NextLinearAttnConfig:
    return Glm5NextLinearAttnConfig.from_json(config_path(which), name=which)


# ── REF math helpers (KDA-specific, host-side) ──────────────────
def _linear_bf16_host(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 F.linear → bf16 RNE store (镜像 Zeus linear_bf16 单 RNE 语义)."""
    return torch.nn.functional.linear(x.float(), w.float()).to(torch.bfloat16)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


def _softplus_kda(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    xs = x * beta
    return torch.where(xs > threshold, x, (1.0 / beta) * torch.log1p(torch.exp(xs)))


def _conv1d_update_silu(x: torch.Tensor, state: torch.Tensor,
                        weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """decode 单步 per-channel causal conv1d + silu, state 原位更新.

    ``x: [B, C]  state: [B, C, K-1]  weight: [C, K]  bias: [C]  ->  [B, C]``
    """
    win = torch.cat([state, x.unsqueeze(-1)], dim=-1)            # [B, C, K]
    out = (win * weight.unsqueeze(0)).sum(-1)
    if bias is not None:
        out = out + bias.unsqueeze(0)
    state.copy_(win[..., 1:])
    return torch.nn.functional.silu(out)


def _rms_norm_gated_sigmoid(x: torch.Tensor, g: torch.Tensor,
                            weight: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    """``y = rmsnorm(x) * weight * sigmoid(g)``, per-(head, dim).  x, g: [B, Hh, D]"""
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return (y * torch.sigmoid(g.float())).to(x.dtype)


# ── Module ──────────────────────────────────────────────────────
class Glm5NextLinearAttn:
    """GLM5-Next Linear-attention (KDA) sublayer.

    使用模式::

        attn = Glm5NextLinearAttn(cfg, seed=42)
        conv, rec = attn.init_state(B=batch, seed=...)
        # REF
        out = attn.forward(hidden, conv, rec)         # host bf16, state 原位推进
        # Zeus
        conv_z = conv.clone().to("zeus"); rec_z = rec.clone().to("zeus")
        out_z = attn.forward_zeus(hidden.to("zeus"), conv_z, rec_z)

    State 由调用方持有 & 显式传入 —— 这样 REF / Zeus 各自 clone 一份独立推进，
    互不污染.
    """

    def __init__(self, cfg: Glm5NextLinearAttnConfig, seed: int = 0):
        self.cfg = cfg
        # gate/o reshape (REF forward L241, forward_zeus gproj_z.reshape) 假设
        # head_v_dim == head_k_dim: gproj/g_b 实际按 Dk 切, 却 reshape 成 [.., Hh, Dv].
        # 异构 head dim 会静默 mis-shape, 此处一次性硬校验.
        assert cfg.head_v_dim == cfg.head_k_dim, (
            f"gate/o reshape 假设 head_v_dim == head_k_dim, got "
            f"Dv={cfg.head_v_dim} Dk={cfg.head_k_dim}; 异构 head dim 需重写 gate reshape"
        )
        g = torch.Generator().manual_seed(seed)

        def rn(*shape, scale: float = 0.02, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
            return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(dtype)

        P = cfg.proj_size
        # bf16 projection weights (走 Zeus LocalMem)
        self.qkv_proj = rn(3 * P, cfg.H)                                    # [3P, H]
        self.b_proj   = rn(cfg.num_heads, cfg.H)                            # [Hh, H]
        self.f_a      = rn(cfg.head_k_dim, cfg.H)                           # [Dk, H]
        self.f_b      = rn(P, cfg.head_k_dim)                               # [P,  Dk]
        self.g_a      = rn(cfg.head_k_dim, cfg.H)                           # [Dk, H]
        self.g_b      = rn(P, cfg.head_k_dim)                               # [P,  Dk]
        self.o_proj   = rn(cfg.H, P)                                        # [H,  P]
        # conv / gate auxiliary (fp32 host, Zeus 端用 bf16/fp32)
        self.conv_w   = (rn(3 * P, cfg.conv_size, dtype=torch.float32) * 0.1)
        self.conv_b   = (rn(3 * P, dtype=torch.float32) * 0.01)
        self.dt_bias  = (torch.randn(P, generator=g, dtype=torch.float32) * 0.01)
        self.A_log    = (torch.randn(cfg.num_heads, generator=g, dtype=torch.float32) * 0.1)
        self.o_norm   = torch.ones(cfg.head_v_dim, dtype=torch.float32)     # implicit weight=1

        # Zeus device-resident state — lazy
        self._zeus_packed = False
        self._w_lmem: Dict[str, torch.Tensor] = {}
        # Per-B cu_seqlens cache: arange(B+1) 只取决于 B, 不依赖输入数据 / 权重 /
        # state. 缓存进 init-time-equivalent 池, 每个 B 值首次 forward_zeus 时 alloc,
        # 之后复用. (生产路径下应在 layer.__init__ 按 max-B 预分配, dev 脚本里用
        # lazy-by-B 简化.)
        self._cu_seqlens_cache: Dict[int, torch.Tensor] = {}

    # ── State init ──────────────────────────────────────────────
    def init_state(self, B: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """构造初始 (conv_state, rec_state) host tensors.

        - conv_state: bf16 [B, 3P, K-1]    (short-conv 历史窗口)
        - rec_state:  fp32 [B, Hh, Dk, Dv] (KDA recurrent S)
        """
        cfg = self.cfg
        g = torch.Generator().manual_seed(seed)
        conv = (torch.randn(B, 3 * cfg.proj_size, cfg.conv_size - 1,
                            generator=g, dtype=torch.bfloat16) * 0.1)
        rec = (torch.randn(B, cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim,
                           generator=g, dtype=torch.float32) * 0.05)
        return conv, rec

    # ── REF forward ─────────────────────────────────────────────
    def forward(self, hidden: torch.Tensor,
                conv_state: torch.Tensor,
                rec_state: torch.Tensor) -> torch.Tensor:
        """GLM5NextLinearAttention.forward decode REF (单步, state 原位推进).

        ``hidden: [B, H] bf16  ->  [B, H] bf16``

        对齐 prerelease/glm5_next.py:316-367 的 decode 分支 (fused_kda_gate 在
        attn 内融合).
        """
        cfg = self.cfg
        B = hidden.shape[0]
        Hh, Dk, Dv, P = cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim, cfg.proj_size

        qkv   = _linear_bf16_host(hidden, self.qkv_proj)                    # [B, 3P]
        beta  = _linear_bf16_host(hidden, self.b_proj)                      # [B, Hh]
        fg    = _linear_bf16_host(_linear_bf16_host(hidden, self.f_a), self.f_b)  # [B, P]
        gproj = _linear_bf16_host(_linear_bf16_host(hidden, self.g_a), self.g_b)  # [B, P]

        # decode conv1d_update (q|k|v 融合一次更新) + silu
        qkv = _conv1d_update_silu(qkv, conv_state, self.conv_w, self.conv_b)
        q, k, v = qkv.split([P, P, P], dim=-1)
        q = q.view(B, Hh, Dk); k = k.view(B, Hh, Dk); v = v.view(B, Hh, Dv)

        # fused_kda_gate: softplus(beta, tau) * -exp(A_log), 加 dt_bias
        fg = fg.float() + self.dt_bias.unsqueeze(0)
        fg = _softplus_kda(fg).view(B, Hh, Dk)
        a = (-torch.exp(self.A_log)).view(1, Hh, 1)
        g_gate = (a * fg)                                                   # [B, Hh, Dk] fp32
        beta = torch.sigmoid(beta.float())                                  # [B, Hh] fp32

        # 单步 delta-rule recurrent (S 先 decay 再 readout)
        qn = _l2norm(q).float() * cfg.scaling
        kn = _l2norm(k).float()
        S = rec_state
        S = S * torch.exp(g_gate).unsqueeze(-1)
        v_hat = torch.einsum("bhk,bhkv->bhv", kn, S)
        delta = v.float() - v_hat
        S = S + torch.einsum("bhv,bhk->bhkv", beta.unsqueeze(-1) * delta, kn)
        o = torch.einsum("bhk,bhkv->bhv", qn, S)                            # [B, Hh, Dv]
        rec_state.copy_(S)

        # gated rmsnorm + o_proj
        norm_gate = gproj.view(B, Hh, Dk)                                   # head_v == head_k
        o = _rms_norm_gated_sigmoid(o.to(torch.bfloat16), norm_gate, self.o_norm, cfg.rms_norm_eps)
        o = o.reshape(B, P)
        return _linear_bf16_host(o, self.o_proj)

    # ── Zeus pack (lazy) ────────────────────────────────────────
    def _pack_zeus(self) -> None:
        """首次 forward_zeus 时把 7 个 linear weight 装包到 LocalMem，
        其余 (conv_w / conv_b / dt_bias / A_log / o_norm) 直接 .to("zeus").
        生产路径下这部分逻辑应放在 layer.__init__；dev 脚本里 lazy 是为了让
        REF-only 调用方不强制依赖 Zeus runtime.
        """
        if ZEUS_IMPORT_ERROR is not None:
            raise RuntimeError(f"Zeus runtime unavailable: {ZEUS_IMPORT_ERROR}")

        self._w_lmem = {
            "qkv_proj": sgl_kernel_zeus.linear_bf16.pack(self.qkv_proj),
            "b_proj":   sgl_kernel_zeus.linear_bf16_outfp32_sigmoid.pack(self.b_proj),
            "f_a":      sgl_kernel_zeus.linear_bf16.pack(self.f_a),
            "f_b":      sgl_kernel_zeus.linear_bf16.pack(self.f_b),
            "g_a":      sgl_kernel_zeus.linear_bf16.pack(self.g_a),
            "g_b":      sgl_kernel_zeus.linear_bf16.pack(self.g_b),
            "o_proj":   sgl_kernel_zeus.linear_bf16.pack(self.o_proj),
            # plain Zeus tensors (非 weight 矩阵)
            "conv_w_z":  self.conv_w.to(torch.bfloat16).to("zeus"),
            "conv_b_z":  self.conv_b.to(torch.bfloat16).to("zeus"),
            "dt_bias_z": self.dt_bias.to("zeus"),
            "A_log_z":   self.A_log.to("zeus"),
            "o_norm_z":  self.o_norm.to("zeus"),
        }
        self._zeus_packed = True

    def _get_cu_seqlens_z(self, B: int) -> torch.Tensor:
        """Lazy per-B cached `cu_seqlens = [0, 1, ..., B]` on Zeus device.

        `torch.arange(B+1, device='zeus')` 本身是一次 device alloc + 填充, 但
        值仅依赖 B —— 因此池化掉, 同一 B 多次 forward 时复用.
        """
        cu = self._cu_seqlens_cache.get(B)
        if cu is None:
            cu = torch.arange(B + 1, dtype=torch.int32, device="zeus")
            self._cu_seqlens_cache[B] = cu
        return cu

    # ── Zeus forward ────────────────────────────────────────────
    def forward_zeus(self, hidden_z: torch.Tensor,
                     conv_state: torch.Tensor,
                     rec_state: torch.Tensor) -> torch.Tensor:
        """Zeus Linear-attn decode 单步 (state 原位推进).

        ``hidden_z: [B, H] bf16 (zeus)  ->  [B, H] bf16 (zeus)``

        - ``conv_state``: bf16 [B, 3P, K-1] (zeus)  原位更新
        - ``rec_state``:  fp32 [B, Hh, Dk, Dv] (zeus)  原位更新
        """
        if not self._zeus_packed:
            self._pack_zeus()

        cfg = self.cfg
        w = self._w_lmem
        B = hidden_z.shape[0]
        H, Hh, Dk, Dv, P = cfg.H, cfg.num_heads, cfg.head_k_dim, cfg.head_v_dim, cfg.proj_size

        # ── host/setup 准备 (集中在 device chain 之前, 对齐 DSA 链路标准) ──
        # cu_seqlens = [0,1,..,B] 只依赖 B, pooled per-B; hoist 到链外, 让下面
        # #1→#13 是一串纯 kernel 调用, 中途无 device alloc.
        cu_seqlens_z = self._get_cu_seqlens_z(B)

        # 1-7. 6 颗 linear_bf16 + 1 颗 linear_bf16_outfp32_sigmoid (b_proj
        #      走 fp32+sigmoid 融合直出，消除原本 `linear_bf16(b_proj) +
        #      .float().sigmoid()` 中的 cast + sigmoid host fallback)
        qkv_z   = sgl_kernel_zeus.linear_bf16(hidden_z, w["qkv_proj"])       # [B, 3P]
        beta_fp32 = sgl_kernel_zeus.linear_bf16_outfp32_sigmoid(
            hidden_z, w["b_proj"],
        )                                                                     # [B, Hh] fp32 sigmoid
        fa_z    = sgl_kernel_zeus.linear_bf16(hidden_z, w["f_a"])            # [B, Dk]
        fg_z    = sgl_kernel_zeus.linear_bf16(fa_z,     w["f_b"])            # [B, P]
        ga_z    = sgl_kernel_zeus.linear_bf16(hidden_z, w["g_a"])            # [B, Dk]
        gproj_z = sgl_kernel_zeus.linear_bf16(ga_z,     w["g_b"])            # [B, P]

        # 8. causal_conv1d_update_split — 单 [B, 3P] qkv 输入 + 共享 conv_state,
        #    **直接输出 3 个独立 contig [B, P] tensor** (q_z / k_z / v_z), 不再
        #    需要 chunk + reshape 那条非 contig 链 (原 caller `.contiguous()`
        #    hidden copy 已消除).
        q_z, k_z, v_z = sgl_kernel_zeus.causal_conv1d_update_split(
            qkv_z, conv_state, w["conv_w_z"], w["conv_b_z"], activation="silu",
        )

        # 9. fused_kda_gate: g_gate[B,Hh,Dk] = (-exp(A_log)) * softplus(fg + dt_bias)
        g_gate_z = sgl_kernel_zeus.fused_kda_gate(
            fg_z, w["A_log_z"], head_dim=Dk, g_bias=w["dt_bias_z"],
        )

        # 10. beta_fp32 已在 step 1-7 通过 linear_bf16_outfp32_sigmoid 一并出，
        #     无需再做 cast 或 sigmoid.

        # 11. fused_recurrent_kda_Sdecay (non-indexed, head_dim 通用)
        #     q/k/v 直接 .view 到 [B, Hh, Dk/Dv] (contig parent → contig view,
        #     零 device 操作; downstream wrapper 不再触发 .contiguous() copy).
        q_4d = q_z.view(B, Hh, Dk)
        k_4d = k_z.view(B, Hh, Dk)
        v_4d = v_z.view(B, Hh, Dv)
        o_z, _ = sgl_kernel_zeus.fused_recurrent_kda_Sdecay(
            q=q_4d, k=k_4d, v=v_4d,
            g=g_gate_z, beta=beta_fp32,
            initial_state=rec_state,
            cu_seqlens=cu_seqlens_z,
            scale=cfg.scaling,
            use_qk_l2norm_in_kernel=True,
            output_final_state=True,
            inplace_final_state=True,
        )  # [B, Hh, Dv] bf16

        # 12. rms_norm_gated (sigmoid 门控；o_norm=ones 是 implicit weight=1)
        # gproj_z[B,P]→[B,Hh,Dv] 拆末尾维 (P=Hh*Dv); normed_z[B,Hh,Dv]→[B,P] 合末尾维.
        # 均在 contiguous kernel 输出上零拷贝, 用 .view 让零拷贝不变量 load-bearing
        # (非 contiguous 时 .view 报错而非静默拷贝).
        gate_for_rms = gproj_z.view(B, Hh, Dv)  # Dv == Dk per head (本配置)
        normed_z = sgl_kernel_zeus.rms_norm_gated(o_z, gate_for_rms, eps=cfg.rms_norm_eps)
        normed_flat = normed_z.view(B, P)

        # 13. o_proj
        return sgl_kernel_zeus.linear_bf16(normed_flat, w["o_proj"])         # [B, H]


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = (
    "linear_bf16", "linear_bf16_outfp32_sigmoid",
    "causal_conv1d_update_split", "fused_kda_gate",
    "fused_recurrent_kda_Sdecay", "rms_norm_gated",
)


def _run_stage(args) -> Optional[bool]:
    """单 stage：从 config 加载真实 shape, 按 mode 跑 REF / Zeus / 对拍."""
    print("\n" + "=" * 60)
    print(f"Stage: {args.config} (real shape)")
    print("=" * 60)
    cfg = load_cfg(args.config)
    print(f"  cfg: H={cfg.H}  num_heads={cfg.num_heads}  "
          f"head_k={cfg.head_k_dim}  head_v={cfg.head_v_dim}  "
          f"conv_size={cfg.conv_size}  proj={cfg.proj_size}")

    B = args.num_tokens
    torch.manual_seed(args.seed)
    attn = Glm5NextLinearAttn(cfg, seed=args.seed)
    hidden = (torch.randn(B, cfg.H, dtype=torch.float32) * 0.05).to(torch.bfloat16)

    # 三份 state 模板：REF / Zeus 各自 clone 推进, 互不污染
    conv_init, rec_init = attn.init_state(B, seed=args.seed + 1)

    # ── REF ───────────────────────────────────────────────────
    ref_out: Optional[torch.Tensor] = None
    if args.mode in ("ref", "both"):
        conv_ref = conv_init.clone()
        rec_ref = rec_init.clone()
        ref_out = attn.forward(hidden, conv_ref, rec_ref)
        state_advanced = (
            not torch.equal(conv_ref, conv_init) and not torch.equal(rec_ref, rec_init)
        )
        ok = (ref_out.shape == (B, cfg.H) and ref_out.dtype == torch.bfloat16
              and state_advanced)
        print(f"  REF out shape={tuple(ref_out.shape)} dtype={ref_out.dtype}  "
              f"state_advanced={state_advanced}")
        print(f"  REF out[0,:4] = "
              f"{[round(v,4) for v in ref_out[0,:4].float().tolist()]}")
        if not ok:
            return False

    # ── Zeus ──────────────────────────────────────────────────
    zeus_ok: Optional[bool] = None
    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"  ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
        else:
            try:
                conv_z = conv_init.clone().to("zeus")
                rec_z = rec_init.clone().to("zeus")
                z_out = attn.forward_zeus(hidden.to("zeus"), conv_z, rec_z)
                z_out_cpu = z_out.cpu()
                finite = torch.isfinite(z_out_cpu).all().item()
                shape_ok = (z_out_cpu.shape == (B, cfg.H)
                            and z_out_cpu.dtype == torch.bfloat16)
                print(f"  ZEUS out shape={tuple(z_out_cpu.shape)} dtype={z_out_cpu.dtype}  "
                      f"finite={finite}")
                print(f"  ZEUS out[0,:4] = "
                      f"{[round(v,4) for v in z_out_cpu[0,:4].float().tolist()]}")
                if ref_out is not None:
                    # 累计误差预算：7 linear_bf16 + conv1d + KDA recurrent +
                    # rms_norm_gated + o_proj，统一放宽到 5e-2.
                    zeus_ok = compare_tensors(
                        f"linear_attn.{args.config}.out", ref_out, z_out_cpu,
                        atol=5e-2, rtol=5e-2,
                    ) and finite and shape_ok
                else:
                    zeus_ok = finite and shape_ok
            except Exception as e:
                import traceback
                print(f"  ZEUS EXCEPTION: {e!r}")
                traceback.print_exc()
                zeus_ok = False

    # ── status 汇总 ──────────────────────────────────────────
    if args.mode == "ref":
        return True
    if args.mode == "zeus":
        return zeus_ok
    # both
    return True if zeus_ok is None else zeus_ok


def main():
    parser = make_argparser("dev_linear_attn",
                            description="GLM5-Next Linear-attention (KDA) sublayer dev test")
    args = parser.parse_args()
    print_header("GLM5-Next Linear-attention (KDA) sublayer", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_linear_attn ({args.config})", ok)


if __name__ == "__main__":
    main()
