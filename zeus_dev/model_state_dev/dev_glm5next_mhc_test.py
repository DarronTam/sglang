"""
GLM5-Next mHC (Manifold-Constrained Hyper-Connections) 逐算子 REF-vs-Zeus 对齐

范围（见 glm5next_mhc_dev.md）：
  - 起点：transformer block 入口的 **多流 residual** [T, N, H]（N=mhc_num_residual_streams=4），
          以 flatten 形态 [T, N*H] 在 block 之间流动。
  - 终点：一个 sublayer（attn 或 mlp）处理完之后回写的新多流 residual [T, N*H]。
  - 单 device、单 layer；attn-side 与 mlp-side 各自独立包一次 mHC（HyperConnection）。
  - 不含训练反向；N 硬锁 4；bf16 residual/x + fp32 mHC 参数（fn/scale/base/norm_weight）。

对齐目标：`/root/project/sglang-feat-v0.5.10-prerelease-glm` 的
  `sglang/srt/layers/mhc/hyper_connection.py::HyperConnection`
  → `functional.py::hc_pre / hc_post`
  → `_mhc_pre_dispatch / _mhc_post_dispatch`（SGLANG_OPT_USE_TORCH_MHC 分支即 pure-torch REF）。

本脚本 REF 侧 **内联** 复刻 `functional.py::_mhc_pre_torch / _mhc_post_torch`，
不依赖 ref_tile_kernels_torch，使 GLM5-Next 部署路径成为唯一 golden。

两套配置（与 dev_glm5next_dsa_decode_test.py 一致）：
  - GLM5-Next-16B : config_16b_v2.json  (H=2048, mhc_tau=1.0,  no_norm_weight=true, post_mult=2)
  - GLM5-Next     : config.json         (H=4096, mhc_tau=0.05, hres_vwnstyle=true)

Stage（★ = 一颗 Zeus kernel 边界；无★的两个 norm_fn/split_mixes 仅作 debug 子步）:
  mhc_expand            K0★  embedding [T,H] -> 多流 residual [T,N*H]（broadcast，sequence 入口）  [Zeus K0 LANDED]
  mhc_pre_norm_fn        —   [融合 K1 子步 a] RMSNorm(flatten N*H) + fp32 Linear(fn) -> mixes [T,24] fp32
  mhc_pre_split_mixes    —   [融合 K1 子步 b] mixes*scale+base -> sigmoid -> (pre, post, comb_logits)
  mhc_pre_norm_split    K1★  融合 #1+#2：BLOCK_T 1-pass，residual -> (pre,post,comb_logits)，mixes 不落 DRAM  [Zeus K1 LANDED]
  mhc_sinkhorn          K2★  comb logits -> doubly-stochastic via Sinkhorn-Knopp  [Zeus K2 LANDED]
  mhc_pre_apply_mix     K3★  (residual × pre).sum(streams) -> layer_input [T,H] bf16  [Zeus K3 LANDED]
  mhc_pre                    K1..K3 组合 = HyperConnection.pre_forward 一次  [Zeus chain LANDED]
  mhc_post              K4★  x·post + einsum(comb, residual) -> [T,N*H] bf16（fp32 accum + 单 RNE）  [Zeus K4 LANDED]
  mhc_sublayer_wrap          端到端一轮 wrap：pre -> identity sublayer -> post（attn 或 mlp side 同构）  [Zeus chain LANDED]
  mhc_layer_full             一整个 decoder layer 的双 wrap：attn_hc(pre/post) -> mlp_hc(pre/post)  [Zeus chain LANDED]

用法:
  python zeus_dev/model_state_dev/dev_glm5next_mhc_test.py
  python zeus_dev/model_state_dev/dev_glm5next_mhc_test.py --config next
  python zeus_dev/model_state_dev/dev_glm5next_mhc_test.py --stage mhc_sinkhorn
  python zeus_dev/model_state_dev/dev_glm5next_mhc_test.py --stage mhc_layer_full --mode ref

约定：
  - 默认 mode=both：先跑 REF（含不变量自检），再尝试 Zeus。
  - 已落地的 Zeus kernel（K0..K4 全套 5 颗）+ 三个组合 stage（mhc_pre / mhc_sublayer_wrap /
    mhc_layer_full）的 host-side 串接路径会被真实调用并对照 REF（用 bf16-quantized fn 跑的
    REF）校验；Summary 显示 "PASS"。
  - 禁止 silent fallback 到 torch 伪装 Zeus kernel。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    import torch_zeus  # noqa: F401 — registers zeus backend when available
    import sgl_kernel_zeus  # noqa: F401

    ZEUS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover — environment-specific
    torch_zeus = None
    sgl_kernel_zeus = None
    ZEUS_IMPORT_ERROR = exc


_THIS_DIR = Path(__file__).resolve().parent
REF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Config ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class Glm5NextMhcConfig:
    name: str
    H: int
    N: int                       # mhc_num_residual_streams (==4 硬锁)
    sinkhorn_iters: int
    sinkhorn_eps: float          # hc_eps
    pre_eps: float               # hc_eps（pre sigmoid 后 + 的 eps）
    post_mult_value: float
    no_norm_weight: bool         # mhc_no_norm_weight；True 则 RMSNorm weight 旁路
    rms_norm_eps: float
    tau: float                   # mhc_tau（GLM 扩展，悬空：当前 REF 不使用）
    hres_vwnstyle: bool          # GLM 扩展（悬空）

    @property
    def mix_hc(self) -> int:
        return self.N * (2 + self.N)   # 4*6 = 24

    @property
    def d_model(self) -> int:
        return self.N * self.H


def _load_json_cfg(path: Path, name: str) -> Glm5NextMhcConfig:
    raw = json.loads(path.read_text())
    N = int(raw.get("mhc_num_residual_streams", 4))
    return Glm5NextMhcConfig(
        name=name,
        H=int(raw["hidden_size"]),
        N=N,
        # HyperConnection.__init__ 默认 sinkhorn_iterations=20；config 显式覆盖优先
        sinkhorn_iters=int(raw.get("mhc_sinkhorn_iterations", 20)),
        sinkhorn_eps=float(raw.get("hc_eps", 1e-6)),
        pre_eps=float(raw.get("hc_eps", 1e-6)),
        # HyperConnection.__init__ 默认 post_mult_value=2.0
        post_mult_value=float(raw.get("mhc_post_mult_value", 2.0)),
        # 缺省按 HyperConnection.__init__ 默认 False（即使用 RMSNorm weight）
        no_norm_weight=bool(raw.get("mhc_no_norm_weight", False)),
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-5)),
        tau=float(raw.get("mhc_tau", 1.0)),
        hres_vwnstyle=bool(raw.get("hres_vwnstyle", False)),
    )


def select_config(which: str) -> Glm5NextMhcConfig:
    if which == "16b":
        return _load_json_cfg(_THIS_DIR / "config_16b_v2.json", "GLM5-Next-16B")
    if which == "next":
        return _load_json_cfg(_THIS_DIR / "config.json", "GLM5-Next")
    raise ValueError(f"unknown config: {which}")


# ── mHC 参数（一个 HyperConnection 的可学习权重） ──────────────────
@dataclass
class MhcParams:
    fn: torch.Tensor             # [mix_hc, N*H] fp32  (mapping_proj.weight)
    scale: torch.Tensor          # [3] fp32
    base: torch.Tensor           # [mix_hc] fp32  (bias，布局 [pre(N)|post(N)|comb(N*N)])
    norm_weight: Optional[torch.Tensor]  # [N*H] fp32 或 None（no_norm_weight 时 None）


def init_mhc_params(cfg: Glm5NextMhcConfig, seed: int) -> MhcParams:
    g = torch.Generator().manual_seed(seed)

    def rn(*shape, scale):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * scale

    norm_weight = None if cfg.no_norm_weight else (1.0 + rn(cfg.d_model, scale=0.02))
    return MhcParams(
        fn=rn(cfg.mix_hc, cfg.d_model, scale=0.02),
        scale=rn(3, scale=0.5),
        base=rn(cfg.mix_hc, scale=0.1),
        norm_weight=norm_weight,
    )


# ── pure-torch REF (mirror of prerelease functional.py) ─────────
def ref_mhc_expand(hidden: torch.Tensor, n: int) -> torch.Tensor:
    """[T, H] -> [T, N*H]，by replication（hc_expand: x.repeat(1, n)）。"""
    return hidden.repeat(1, n)


def ref_sinkhorn(comb: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """comb logits [..., N, N] -> doubly-stochastic。

    与 functional.py::_mhc_pre_torch 内联段一致：首轮 softmax(-1)+eps 然后列归一，
    之后 (repeat-1) 轮交替 行/列 归一。共 2*repeat-1 次 reduce。
    """
    x = comb.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def ref_mhc_pre_norm_fn(residual_3d: torch.Tensor, p: MhcParams,
                        cfg: Glm5NextMhcConfig) -> torch.Tensor:
    """RMSNorm(flatten N*H) + fp32 Linear -> mixes [T, mix_hc] fp32.

    residual_3d: [T, N, H] bf16
    返回 mixes: [T, mix_hc] fp32
    """
    s, n, h = residual_3d.shape
    fn = p.fn if p.norm_weight is None else p.fn * p.norm_weight
    x_flat = residual_3d.reshape(s, n * h).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    mixes = torch.nn.functional.linear(x_flat, fn) * rsqrt
    return mixes


def ref_mhc_pre_split_mixes(mixes: torch.Tensor, p: MhcParams,
                            cfg: Glm5NextMhcConfig
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mixes [T, mix_hc] fp32 -> (pre[T,N,1], post[T,N,1], comb_logits[T,N,N]) fp32."""
    s = mixes.shape[0]
    n = cfg.N
    pre_raw = mixes[:, :n]
    post_raw = mixes[:, n:2 * n]
    comb_raw = mixes[:, 2 * n:].view(s, n, n)
    pre_base = p.base[:n]
    post_base = p.base[n:2 * n]
    comb_base = p.base[2 * n:].view(n, n)

    pre = torch.sigmoid(pre_raw * p.scale[0] + pre_base) + cfg.pre_eps
    post = cfg.post_mult_value * torch.sigmoid(post_raw * p.scale[1] + post_base)
    comb_logits = comb_raw * p.scale[2] + comb_base
    return pre.unsqueeze(-1), post.unsqueeze(-1), comb_logits


def ref_mhc_pre_norm_split(residual_3d: torch.Tensor, p: MhcParams,
                           cfg: Glm5NextMhcConfig
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """融合算子 K1 = (#1 mhc_pre_norm_fn) + (#2 mhc_pre_split_mixes) 一趟完成。

    对应 Zeus 单 kernel 的 **BLOCK_T 1-pass**：每个 T-tile 读入 [BLOCK_T, N*H] residual，
    在寄存器/shared 内一气呵成 RMSNorm → fp32 GEMM(fn) → 24 维 mixes →
    scale/base/sigmoid/split，直接吐出 (pre, post, comb_logits)；中间 mixes [T,24]
    **不落 DRAM**（省去 #1→#2 之间的一次 [T,24] 读写）。

    RMSNorm 沿 N*H 轴是 per-token 归约，T 轴各 token 完全独立，故 BLOCK_T 可任意取值
    而结果 bit-exact（见 stage_mhc_pre_norm_split 的 tile-invariance 自检）。

    residual_3d: [T, N, H] bf16
    返回 (pre[T,N,1], post[T,N,1], comb_logits[T,N,N]) fp32
    """
    mixes = ref_mhc_pre_norm_fn(residual_3d, p, cfg)
    return ref_mhc_pre_split_mixes(mixes, p, cfg)


def ref_mhc_pre_apply_mix(residual_3d: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """(residual × pre).sum(streams) -> layer_input [T, H] bf16."""
    dtype = residual_3d.dtype
    return (pre * residual_3d.float()).sum(dim=1).to(dtype)


def ref_mhc_pre(residual_flat: torch.Tensor, p: MhcParams, cfg: Glm5NextMhcConfig):
    """HyperConnection.pre_forward 的 pure-torch REF。

    residual_flat: [T, N*H] bf16
    返回：layer_input [T,H] bf16, residual_flat（passthrough）, h_res [T,N*N] fp32, h_post [T,N] fp32
    （comb 已经过 Sinkhorn）
    """
    s, total = residual_flat.shape
    n = cfg.N
    h = total // n
    residual_3d = residual_flat.view(s, n, h)
    # K1 融合：norm_fn + split_mixes 合并为一趟（mixes 不落 DRAM）
    pre, post, comb_logits = ref_mhc_pre_norm_split(residual_3d, p, cfg)
    comb = ref_sinkhorn(comb_logits, cfg.sinkhorn_iters, cfg.sinkhorn_eps)
    layer_input = ref_mhc_pre_apply_mix(residual_3d, pre)
    h_res = comb.reshape(s, n * n)
    h_post = post.reshape(s, n)
    return layer_input, residual_flat, h_res, h_post


def ref_mhc_post(x: torch.Tensor, residual_flat: torch.Tensor,
                 h_post: torch.Tensor, h_res: torch.Tensor,
                 cfg: Glm5NextMhcConfig) -> torch.Tensor:
    """HyperConnection.post_forward 的 pure-torch REF。

    x:            [T, H]    bf16  (sublayer 输出)
    residual_flat:[T, N*H]  bf16
    h_post:       [T, N]    fp32
    h_res:        [T, N*N]  fp32
    返回：next residual [T, N*H] bf16（fp32 accum + 单次 RNE）
    """
    s, h = x.shape
    n = cfg.N
    residual_3d = residual_flat.view(s, n, h)
    post = h_post.view(s, n, 1)
    comb = h_res.view(s, n, n)
    # out[s,n,h] = post[s,n,1]*x[s,1,h] + sum_m comb[s,m,n]*residual[s,m,h]
    out = post * x.unsqueeze(1) + (
        comb.unsqueeze(-1) * residual_3d.unsqueeze(2)
    ).sum(dim=1)
    return out.view(s, n * h).to(x.dtype)


# ── Zeus path helpers (compound stages: mhc_pre / sublayer_wrap / layer_full) ──
def quantize_p_for_zeus_match(p: MhcParams) -> MhcParams:
    """构造与 Zeus K1 实际使用的权重等价的 MhcParams.

    流程: fn → host-side pre-merge norm_weight → .to(bf16) 量化 → .float() round-trip.
    把这套参数喂给 ref_mhc_pre / ref_mhc_post 跑 REF，会把 bf16 fn 量化误差吸收到 REF 侧，
    使 Zeus-vs-REF 对照的差异只剩 BLAS 累加重排噪声（与 stage_mhc_pre_norm_split 同套路）。
    """
    fn_eff_fp32 = p.fn if p.norm_weight is None else p.fn * p.norm_weight
    fn_eff_bf16 = fn_eff_fp32.to(torch.bfloat16)
    return MhcParams(
        fn=fn_eff_bf16.float(),
        scale=p.scale,
        base=p.base,
        norm_weight=None,
    )


def zeus_mhc_pre(residual_flat: torch.Tensor, p: MhcParams,
                 cfg: Glm5NextMhcConfig):
    """K1 → K2 → K3 串接（HyperConnection.pre_forward 的 Zeus 路径）。

    residual_flat: [T, N*H] bf16，CPU 或已在 zeus device 上均可（.to("zeus") 自适应）。
    返回 (layer_input[T,H] bf16, residual_z 直通, h_res[T,N*N] fp32, h_post[T,N] fp32)
    —— 所有输出均在 Zeus device 上。
    """
    s, total = residual_flat.shape
    n = cfg.N
    fn_eff_fp32 = p.fn if p.norm_weight is None else p.fn * p.norm_weight
    fn_eff_bf16 = fn_eff_fp32.to(torch.bfloat16)
    fn_lmem = torch.zeus.local_memory.from_tensor(
        fn_eff_bf16.to("zeus"), kind="weight", Tr=1, Tc=1,
    )
    residual_z = residual_flat.to("zeus")
    pre, post, comb_logits = sgl_kernel_zeus.mhc_pre_norm_split(
        residual_z,
        fn_lmem,
        p.scale,                                # CPU OK
        p.base.to("zeus"),
        n=n,
        rms_norm_eps=cfg.rms_norm_eps,
        pre_eps=cfg.pre_eps,
        post_mult_value=cfg.post_mult_value,
    )
    comb = sgl_kernel_zeus.mhc_sinkhorn(
        comb_logits, repeat=cfg.sinkhorn_iters, eps=cfg.sinkhorn_eps,
    )
    layer_input = sgl_kernel_zeus.mhc_pre_apply_mix(residual_z, pre, n=n)
    h_res = comb.reshape(s, n * n)
    h_post = post.reshape(s, n)
    return layer_input, residual_z, h_res, h_post


def zeus_mhc_post(x: torch.Tensor, residual: torch.Tensor,
                  h_post: torch.Tensor, h_res: torch.Tensor,
                  cfg: Glm5NextMhcConfig) -> torch.Tensor:
    """K4 mhc_post 的 Zeus 路径直通调用。所有输入都应在 Zeus device 上。"""
    return sgl_kernel_zeus.mhc_post(x, residual, h_post, h_res, n=cfg.N)


def _zeus_has(*kernels: str) -> bool:
    """检查 sgl_kernel_zeus 是否同时具备一组 kernel 名（组合 stage 用)."""
    if ZEUS_IMPORT_ERROR is not None:
        return False
    return all(hasattr(sgl_kernel_zeus, k) for k in kernels)


# ── Helpers ────────────────────────────────────────────────────
def compare_tensors(name: str, ref: torch.Tensor, got: torch.Tensor,
                    atol=5e-3, rtol=5e-3) -> bool:
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


def assert_doubly_stochastic(name: str, comb: torch.Tensor, tol: float) -> bool:
    row_err = (comb.sum(-1) - 1.0).abs().max().item()
    col_err = (comb.sum(-2) - 1.0).abs().max().item()
    ok = row_err < tol and col_err < tol
    print(f"  [{name}/doubly_stochastic] {'PASS' if ok else 'FAIL'} | "
          f"max|row_sum-1|={row_err:.3e} max|col_sum-1|={col_err:.3e} tol={tol:.1e}")
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


def zeus_compare(name: str, ref: torch.Tensor, got: torch.Tensor,
                 *, atol: float = 0.0, rtol: float = 0.0,
                 exact: bool = False) -> bool:
    """Compare a Zeus device tensor against a CPU REF tensor.

    For pure-memcpy / bf16-only kernels (K0 mhc_expand), set ``exact=True`` to
    require逐 bit 一致；K2..K4 走 fp32 中间累加，应使用 (atol, rtol)。
    """
    a = ref.detach().float().cpu()
    b = got.detach().float().cpu()
    if a.shape != b.shape:
        print(f"  ZEUS [{name}] SHAPE MISMATCH: ref={tuple(a.shape)} got={tuple(b.shape)}")
        return False
    if exact:
        # bf16 round-trip is byte-exact for memcpy kernels; check on the original dtype.
        ok = torch.equal(ref.detach().cpu(), got.detach().cpu())
        diff = (a - b).abs()
        print(f"  ZEUS [{name}] {'PASS' if ok else 'FAIL'} (bit-exact) "
              f"| max_diff={diff.max().item():.3e} shape={list(a.shape)}")
        return ok
    diff = (a - b).abs()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print(f"  ZEUS [{name}] {'PASS' if ok else 'DIFF'} "
          f"| max_diff={diff.max().item():.6e} mean_diff={diff.mean().item():.6e} "
          f"atol={atol:.1e} rtol={rtol:.1e} shape={list(a.shape)}")
    return ok


def finish_stage(args, ref_ok: bool, kernel_name: str, anchor: str) -> Optional[bool]:
    if not ref_ok:
        return False
    if args.mode == "ref":
        return True
    zeus_skip(kernel_name, anchor)
    return None


def finish_stage_with_zeus(args, ref_ok: bool, zeus_ok: Optional[bool],
                           kernel_name: str, anchor: str) -> Optional[bool]:
    """Variant of finish_stage for stages whose Zeus kernel is **landed**.

    Return contract（与 main 的 Summary 三态对齐）：
      - True   : REF PASS + Zeus PASS（或 mode=ref 时仅 REF PASS）
      - False  : 任一 REF / Zeus 失败
      - None   : Zeus 不可用（mode!=ref 时 zeus_ok 为 None；按 SKIP 显示）
    """
    if not ref_ok:
        return False
    if args.mode == "ref":
        return True
    if zeus_ok is None:
        # zeus 路径未启用（环境不可用 / 未找到 op），打印 SKIP anchor 兜底
        zeus_skip(kernel_name, anchor)
        return None
    return bool(zeus_ok)


# ── Stage: mhc_expand (#0) ──────────────────────────────────────
def stage_mhc_expand(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_expand (#0 embedding -> multi-stream residual)  [Zeus K0 LANDED]")
    print("=" * 60)
    T, H, N = args.num_tokens, cfg.H, cfg.N
    torch.manual_seed(args.seed)
    hidden = torch.randn(T, H, dtype=torch.bfloat16)

    out = ref_mhc_expand(hidden, N)
    ref_ok = out.shape == (T, N * H)
    # 不变量：N 条流初始内容完全相同（纯 broadcast）
    out_3d = out.view(T, N, H)
    same = all(torch.equal(out_3d[:, k, :], hidden) for k in range(N))
    print(f"  shape: T={T} H={H} N={N}  out={tuple(out.shape)}")
    print(f"  [mhc_expand/broadcast] {'PASS' if same else 'FAIL'} "
          f"(all {N} streams == source hidden)")

    # ── Zeus K0 真实路径 ─────────────────────────────────────────────
    # 算子: sgl_kernel_zeus.mhc_expand(hidden, n=N) -> [T, N*H] bf16
    # 实现: csrc/glm5next_mhc/{mhc_expand_kernel.py, sgl_mhc_expand_sim.c, mhc_expand_zeus.cpp}
    # 文档: sgl-kernel-zeus/docs/mhc_expand.md
    # 期望: bf16 纯 memcpy → 与 REF (x.repeat(1, N)) 逐 bit 一致 (exact=True)
    zeus_ok: Optional[bool] = None
    if args.mode != "ref" and ZEUS_IMPORT_ERROR is None \
            and hasattr(sgl_kernel_zeus, "mhc_expand"):
        try:
            got = sgl_kernel_zeus.mhc_expand(hidden.to("zeus"), n=N).cpu()
            zeus_ok = zeus_compare("mhc_expand", out, got, exact=True)
            # 与 REF 一致的 stream-identity 不变量
            got_3d = got.view(T, N, H)
            zeus_streams_ok = all(torch.equal(got_3d[:, k, :], hidden) for k in range(N))
            print(f"  ZEUS [mhc_expand/streams_identical] "
                  f"{'PASS' if zeus_streams_ok else 'FAIL'} (Zeus 输出 N 条流 == hidden)")
            zeus_ok = zeus_ok and zeus_streams_ok
        except Exception as e:
            print(f"  ZEUS [mhc_expand] EXCEPTION: {e!r}")
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok and same, zeus_ok, "mhc_expand",
        "sgl-kernel-zeus mhc_expand_kernel "
        "(BLOCK_T×BLOCK_H tile + N=4 静态展开扇出)")


# ── Stage: mhc_pre_norm_fn (#1) ─────────────────────────────────
def stage_mhc_pre_norm_fn(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_pre_norm_fn  [融合 K1 子步 a — debug 粒度, RMSNorm + fp32 Linear]")
    print("=" * 60)
    T, H, N = args.num_tokens, cfg.H, cfg.N
    torch.manual_seed(args.seed)
    residual = torch.randn(T, N, H, dtype=torch.bfloat16) * 0.1

    mixes = ref_mhc_pre_norm_fn(residual, p, cfg)
    ref_ok = mixes.shape == (T, cfg.mix_hc) and mixes.dtype == torch.float32
    print(f"  shape: residual=[{T},{N},{H}] bf16  fn=[{cfg.mix_hc},{cfg.d_model}] fp32")
    print(f"  norm_weight: {'None (no_norm_weight)' if p.norm_weight is None else 'fp32 RMSNorm weight'}")
    print(f"  mixes: {tuple(mixes.shape)} {mixes.dtype}  "
          f"|mean|={mixes.abs().mean().item():.3e} |max|={mixes.abs().max().item():.3e}")
    return finish_stage(args, ref_ok, "mhc_pre_norm_split",
                        "已并入融合 K1 mhc_pre_norm_split；本 stage 仅 debug 子步 a (RMSNorm + fp32 split-K GEMM)")


# ── Stage: mhc_pre_split_mixes (#2) ─────────────────────────────
def stage_mhc_pre_split_mixes(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_pre_split_mixes  [融合 K1 子步 b — debug 粒度, sigmoid / scale / base split]")
    print("=" * 60)
    T, N = args.num_tokens, cfg.N
    torch.manual_seed(args.seed)
    mixes = torch.randn(T, cfg.mix_hc, dtype=torch.float32)

    pre, post, comb = ref_mhc_pre_split_mixes(mixes, p, cfg)
    ref_ok = (pre.shape == (T, N, 1) and post.shape == (T, N, 1)
              and comb.shape == (T, N, N))
    print(f"  mix_hc={cfg.mix_hc}  post_mult_value={cfg.post_mult_value}  pre_eps={cfg.pre_eps}")
    print(f"  pre : {tuple(pre.shape)} range=[{pre.min().item():.3e},{pre.max().item():.3e}] "
          f"(sigmoid + pre_eps)")
    print(f"  post: {tuple(post.shape)} range=[{post.min().item():.3e},{post.max().item():.3e}] "
          f"(sigmoid * {cfg.post_mult_value})")
    print(f"  comb: {tuple(comb.shape)} |mean|={comb.abs().mean().item():.3e} (raw logits)")
    # 不变量：pre ∈ (eps, 1+eps)，post ∈ (0, post_mult_value)
    inv = (bool((pre >= cfg.pre_eps - 1e-6).all()) and bool((pre <= 1 + cfg.pre_eps + 1e-4).all())
           and bool((post >= 0).all()) and bool((post <= cfg.post_mult_value + 1e-4).all()))
    print(f"  [mhc_pre_split_mixes/range_invariant] {'PASS' if inv else 'FAIL'}")
    return finish_stage(args, ref_ok and inv, "mhc_pre_norm_split",
                        "已并入融合 K1 mhc_pre_norm_split；本 stage 仅 debug 子步 b (纯 elementwise split)")


# ── Stage: mhc_pre_norm_split (融合 K1 = #1 + #2) ────────────────
def stage_mhc_pre_norm_split(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_pre_norm_split (融合 K1 = #1 mhc_pre_norm_fn + #2 mhc_pre_split_mixes, BLOCK_T 1-pass)  [Zeus K1 LANDED]")
    print("=" * 60)
    T, H, N = args.num_tokens, cfg.H, cfg.N
    torch.manual_seed(args.seed)
    residual = torch.randn(T, N, H, dtype=torch.bfloat16) * 0.1

    pre, post, comb = ref_mhc_pre_norm_split(residual, p, cfg)
    ref_ok = (pre.shape == (T, N, 1) and post.shape == (T, N, 1)
              and comb.shape == (T, N, N)
              and pre.dtype == post.dtype == comb.dtype == torch.float32)
    print(f"  residual=[{T},{N},{H}] bf16  fn=[{cfg.mix_hc},{cfg.d_model}] fp32  BLOCK_T={args.block_t}")
    print(f"  -> pre={tuple(pre.shape)} post={tuple(post.shape)} comb_logits={tuple(comb.shape)} (all fp32)")

    # 等价性：融合 == 顺序两步（#1 then #2），应当逐 bit 一致
    mixes = ref_mhc_pre_norm_fn(residual, p, cfg)
    pre2, post2, comb2 = ref_mhc_pre_split_mixes(mixes, p, cfg)
    fuse_eq = (torch.equal(pre, pre2) and torch.equal(post, post2)
               and torch.equal(comb, comb2))
    print(f"  [fused == split(norm_fn)] {'PASS' if fuse_eq else 'FAIL'} (融合与分两步逐 bit 一致)")

    # tile-invariance：按 BLOCK_T 分块逐 tile 跑 == 整批跑（证明 T 轴可任意切分）。
    # 算法上 per-token RMSNorm 令各 token 独立，故 Zeus kernel（每 token 定序 dot）会逐 bit
    # 一致；但 REF 的 BLAS F.linear 对 [bt,N*H] 子批与整批用不同 fp32 累加顺序，会带 ~1e-6
    # 重排噪声，故这里用紧致 tol 而非 torch.equal。
    bt = max(1, args.block_t)
    pre_t, post_t, comb_t = [], [], []
    for i in range(0, T, bt):
        pr, po, co = ref_mhc_pre_norm_split(residual[i:i + bt], p, cfg)
        pre_t.append(pr); post_t.append(po); comb_t.append(co)
    pre_t, post_t, comb_t = torch.cat(pre_t, 0), torch.cat(post_t, 0), torch.cat(comb_t, 0)
    tile_diff = max((pre - pre_t).abs().max().item(),
                    (post - post_t).abs().max().item(),
                    (comb - comb_t).abs().max().item())
    tile_ok = tile_diff < 1e-4
    print(f"  [tile_invariant BLOCK_T={bt}] {'PASS' if tile_ok else 'FAIL'} "
          f"max_diff={tile_diff:.3e} (分块逐 tile == 整批；per-token RMSNorm 保证 T 轴可切，"
          f"残差为 BLAS 重排噪声)")

    # ── Zeus K1 真实路径 ─────────────────────────────────────────────
    # 算子: sgl_kernel_zeus.mhc_pre_norm_split(residual, fn_lmem_bf16, scale, base, ...)
    # 实现: csrc/glm5next_mhc/{mhc_pre_norm_split_kernel.py, sgl_mhc_pre_norm_split_sim.c, mhc_pre_norm_split_zeus.cpp}
    # 文档: sgl-kernel-zeus/docs/mhc_pre_norm_split.md
    # 权重路径: fp32 fn → host pre-merge norm_weight → .to(bf16) 量化 → LocalMem pack
    #           (kind="weight", Tr=1, Tc=1)；kernel 走 `memory_type='weight'` 加载。
    #           Lmem 不支持 fp32，所以 K1 GEMM 必须走 bf16 weight。
    # 期望: bf16 weight 量化引入 ~5e-4 单元素误差；sigmoid 平滑后 pre ≤ 5e-3，
    #       post (×2) ≤ 1e-2，comb (raw logits) ≤ 1e-3；用 atol=rtol=2e-2 容忍。
    zeus_ok: Optional[bool] = None
    if args.mode != "ref" and ZEUS_IMPORT_ERROR is None \
            and hasattr(sgl_kernel_zeus, "mhc_pre_norm_split"):
        try:
            # 调用方在 host 侧 pre-merge norm_weight 进 fn（REF 与 Zeus 同源），
            # 再 bf16 量化 + LocalMem pack。
            fn_eff_fp32 = p.fn if p.norm_weight is None else p.fn * p.norm_weight
            fn_eff_bf16 = fn_eff_fp32.to(torch.bfloat16)
            fn_lmem = torch.zeus.local_memory.from_tensor(
                fn_eff_bf16.to("zeus"), kind="weight", Tr=1, Tc=1,
            )

            # bf16 量化后的 REF（吸收量化误差到 REF 侧，对照差异只剩 BLAS 累加噪声）
            mixes_bf16 = ref_mhc_pre_norm_fn(residual,
                                             MhcParams(fn=fn_eff_bf16.float(),
                                                       scale=p.scale,
                                                       base=p.base,
                                                       norm_weight=None),
                                             cfg)
            pre_bf16ref, post_bf16ref, comb_bf16ref = ref_mhc_pre_split_mixes(
                mixes_bf16, p, cfg)

            z_pre, z_post, z_comb = sgl_kernel_zeus.mhc_pre_norm_split(
                residual.to("zeus"),
                fn_lmem,
                p.scale,                         # CPU OK — wrapper 会 .cpu() 抽 3 个 scalar
                p.base.to("zeus"),
                n=N,
                rms_norm_eps=cfg.rms_norm_eps,
                pre_eps=cfg.pre_eps,
                post_mult_value=cfg.post_mult_value,
            )
            z_pre_ok  = zeus_compare("mhc_pre_norm_split.pre",
                                     pre_bf16ref,  z_pre,  atol=2e-2, rtol=1e-2)
            z_post_ok = zeus_compare("mhc_pre_norm_split.post",
                                     post_bf16ref, z_post, atol=2e-2, rtol=1e-2)
            z_comb_ok = zeus_compare("mhc_pre_norm_split.comb_logits",
                                     comb_bf16ref, z_comb, atol=2e-2, rtol=1e-2)
            zeus_ok = z_pre_ok and z_post_ok and z_comb_ok
        except Exception as e:
            print(f"  ZEUS [mhc_pre_norm_split] EXCEPTION: {e!r}")
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok and fuse_eq and tile_ok, zeus_ok, "mhc_pre_norm_split",
        "sgl-kernel-zeus pre_norm_split_kernel "
        "(BLOCK_T 1-pass: RMSNorm + bf16 GEMM(LocalMem) + scale/base/sigmoid/split, mixes 不落 DRAM)")


# ── Stage: mhc_sinkhorn (#3) ────────────────────────────────────
def stage_mhc_sinkhorn(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_sinkhorn (#3 Birkhoff-polytope projection)  [Zeus K2 LANDED]")
    print("=" * 60)
    T, N = args.num_tokens, cfg.N
    repeat, eps = cfg.sinkhorn_iters, cfg.sinkhorn_eps
    torch.manual_seed(args.seed)
    comb_logits = torch.randn(T, N, N, dtype=torch.float32) * 0.5

    comb = ref_sinkhorn(comb_logits, repeat, eps)
    ref_ok = comb.shape == (T, N, N) and comb.dtype == torch.float32
    print(f"  shape: T={T} N={N}  repeat={repeat}  eps={eps}")
    ds_tol = 1e-4 if repeat >= 15 else 5e-3
    ok_ds = assert_doubly_stochastic("mhc_sinkhorn", comb, tol=ds_tol)
    neg = int((comb < 0).sum().item())
    ok_nn = neg == 0
    print(f"  [mhc_sinkhorn/non_negative] {'PASS' if ok_nn else 'FAIL'} "
          f"negative_entries={neg}/{comb.numel()}")

    # ── Zeus K2 真实路径 ─────────────────────────────────────────────
    # 算子: sgl_kernel_zeus.mhc_sinkhorn(comb_logits, repeat, eps) -> [T, N, N] fp32
    # 实现: csrc/glm5next_mhc/{mhc_sinkhorn_kernel.py, sgl_mhc_sinkhorn_sim.c, mhc_sinkhorn_zeus.cpp}
    # 文档: sgl-kernel-zeus/docs/mhc_sinkhorn.md
    # 切核策略: 沿 T 轴 (N 是 reduce 轴不可切)；T 不整除时 cdiv + boundary 兜底
    # 期望: fp32 全程；Sinkhorn 是 contractive iteration → ~ULP 量级噪声，紧 tol 即可
    zeus_ok: Optional[bool] = None
    if args.mode != "ref" and ZEUS_IMPORT_ERROR is None \
            and hasattr(sgl_kernel_zeus, "mhc_sinkhorn"):
        try:
            z_comb = sgl_kernel_zeus.mhc_sinkhorn(
                comb_logits.to("zeus"), repeat=repeat, eps=eps,
            )
            zeus_ok = zeus_compare("mhc_sinkhorn", comb, z_comb,
                                   atol=5e-5, rtol=5e-5)
            # Doubly-stochastic 不变量在 Zeus 输出上重测一遍
            z_cpu = z_comb.cpu()
            z_row = (z_cpu.sum(-1) - 1.0).abs().max().item()
            z_col = (z_cpu.sum(-2) - 1.0).abs().max().item()
            z_ds_ok = z_row < ds_tol and z_col < ds_tol
            print(f"  ZEUS [mhc_sinkhorn/doubly_stochastic] "
                  f"{'PASS' if z_ds_ok else 'FAIL'} | "
                  f"max|row_sum-1|={z_row:.3e} max|col_sum-1|={z_col:.3e}")
            z_neg = int((z_cpu < 0).sum().item())
            z_nn_ok = z_neg == 0
            print(f"  ZEUS [mhc_sinkhorn/non_negative] "
                  f"{'PASS' if z_nn_ok else 'FAIL'} "
                  f"negative_entries={z_neg}/{z_cpu.numel()}")
            zeus_ok = zeus_ok and z_ds_ok and z_nn_ok
        except Exception as e:
            print(f"  ZEUS [mhc_sinkhorn] EXCEPTION: {e!r}")
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok and ok_ds and ok_nn, zeus_ok, "mhc_sinkhorn",
        "sgl-kernel-zeus sinkhorn_kernel (沿 T 轴切核，2*repeat-1 reduce per token，fp32)")


# ── Stage: mhc_pre_apply_mix (#4) ───────────────────────────────
def stage_mhc_pre_apply_mix(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_pre_apply_mix (#4 weighted sum over streams)  [Zeus K3 LANDED]")
    print("=" * 60)
    T, H, N = args.num_tokens, cfg.H, cfg.N
    torch.manual_seed(args.seed)
    residual = torch.randn(T, N, H, dtype=torch.bfloat16) * 0.1
    pre = torch.rand(T, N, 1, dtype=torch.float32) * 0.5 + 0.05

    out = ref_mhc_pre_apply_mix(residual, pre)
    ref_ok = out.shape == (T, H) and out.dtype == torch.bfloat16
    print(f"  residual=[{T},{N},{H}] bf16  pre=[{T},{N},1] fp32  out={tuple(out.shape)}")
    # sanity：pre=1/N 且各流相同 -> out ≈ residual[:,0,:]
    uni_pre = torch.full((T, N, 1), 1.0 / N, dtype=torch.float32)
    uni_res = residual[:, 0:1, :].expand(T, N, H).contiguous()
    sanity = ref_mhc_pre_apply_mix(uni_res, uni_pre)
    d = (sanity.float() - residual[:, 0, :].float()).abs().max().item()
    print(f"  [mhc_pre_apply_mix/uniform_sanity] {'PASS' if d < 5e-3 else 'DIFF'} "
          f"max_diff={d:.3e} (expect ≈ 0)")

    # ── Zeus K3 真实路径 ─────────────────────────────────────────────
    # 算子: sgl_kernel_zeus.mhc_pre_apply_mix(residual, pre, n=N) -> [T, H] bf16
    # 实现: csrc/glm5next_mhc/{mhc_pre_apply_mix_kernel.py, sgl_mhc_pre_apply_mix_sim.c, mhc_pre_apply_mix_zeus.cpp}
    # 文档: sgl-kernel-zeus/docs/mhc_pre_apply_mix.md
    # 切核策略: 沿 H 轴 CORE_NUM=2（每核负责 disjoint 的 H//2 列），无 cross-core 通信
    # 期望: fp32 acc + 单 bf16 RNE → atol=rtol=1e-2 容忍 bf16 输出 round noise
    zeus_ok: Optional[bool] = None
    if args.mode != "ref" and ZEUS_IMPORT_ERROR is None \
            and hasattr(sgl_kernel_zeus, "mhc_pre_apply_mix"):
        try:
            z_out = sgl_kernel_zeus.mhc_pre_apply_mix(
                residual.to("zeus"),
                pre.to("zeus"),
                n=N,
            )
            zeus_ok = zeus_compare("mhc_pre_apply_mix", out, z_out,
                                   atol=1e-2, rtol=1e-2)
        except Exception as e:
            print(f"  ZEUS [mhc_pre_apply_mix] EXCEPTION: {e!r}")
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok and d < 5e-3, zeus_ok, "mhc_pre_apply_mix",
        "sgl-kernel-zeus pre_apply_mix_kernel (沿 H 轴切核，4 路 multiply-add + 单 RNE)")


# ── Stage: mhc_pre (#5 组合) ────────────────────────────────────
def stage_mhc_pre(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_pre (#5 HyperConnection.pre_forward 组合 = K1·K2·K3)  [Zeus chain LANDED]")
    print("=" * 60)
    T, H, N = args.num_tokens, cfg.H, cfg.N
    torch.manual_seed(args.seed)
    residual_flat = (torch.randn(T, N * H, dtype=torch.bfloat16) * 0.1)

    layer_input, res_pass, h_res, h_post = ref_mhc_pre(residual_flat, p, cfg)
    ref_ok = (layer_input.shape == (T, H) and h_res.shape == (T, N * N)
              and h_post.shape == (T, N) and torch.equal(res_pass, residual_flat))
    print(f"  in residual=[{T},{N*H}] -> layer_input={tuple(layer_input.shape)} "
          f"h_res={tuple(h_res.shape)} h_post={tuple(h_post.shape)}")
    comb = h_res.view(T, N, N)
    ds_tol = 1e-4 if cfg.sinkhorn_iters >= 15 else 5e-3
    ok_ds = assert_doubly_stochastic("mhc_pre/comb", comb, tol=ds_tol)

    # ── Zeus 路径：K1 → K2 → K3 串接 ────────────────────────────────
    # REF 用 bf16-quantized fn 吸收 K1 GEMM 量化噪声，对照只剩 BLAS 重排噪声。
    zeus_ok: Optional[bool] = None
    if args.mode != "ref" and _zeus_has(
            "mhc_pre_norm_split", "mhc_sinkhorn", "mhc_pre_apply_mix"):
        try:
            p_quant = quantize_p_for_zeus_match(p)
            li_q, _, hres_q, hpost_q = ref_mhc_pre(residual_flat, p_quant, cfg)

            z_li, _, z_hres, z_hpost = zeus_mhc_pre(residual_flat, p, cfg)
            z_li_ok    = zeus_compare("mhc_pre.layer_input", li_q,    z_li,
                                       atol=2e-2, rtol=1e-2)
            z_hres_ok  = zeus_compare("mhc_pre.h_res",       hres_q,  z_hres,
                                       atol=2e-2, rtol=1e-2)
            z_hpost_ok = zeus_compare("mhc_pre.h_post",      hpost_q, z_hpost,
                                       atol=2e-2, rtol=1e-2)
            # Zeus 输出的 doubly-stochastic 不变量
            z_comb_cpu = z_hres.view(T, N, N).cpu()
            z_row = (z_comb_cpu.sum(-1) - 1.0).abs().max().item()
            z_col = (z_comb_cpu.sum(-2) - 1.0).abs().max().item()
            z_ds_ok = z_row < ds_tol and z_col < ds_tol
            print(f"  ZEUS [mhc_pre/doubly_stochastic] "
                  f"{'PASS' if z_ds_ok else 'FAIL'} | "
                  f"max|row_sum-1|={z_row:.3e} max|col_sum-1|={z_col:.3e} tol={ds_tol:.1e}")
            zeus_ok = z_li_ok and z_hres_ok and z_hpost_ok and z_ds_ok
        except Exception as e:
            print(f"  ZEUS [mhc_pre] EXCEPTION: {e!r}")
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok and ok_ds, zeus_ok, "mhc_pre_norm_split",
        "K1 mhc_pre_norm_split + K2 mhc_sinkhorn + K3 mhc_pre_apply_mix 串接 "
        "(host 侧 pre-merge norm_weight + bf16 LocalMem fn pack)")


# ── Stage: mhc_post (#6) ────────────────────────────────────────
def stage_mhc_post(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_post (#6 scatter back to streams + residual mix)  [Zeus K4 LANDED]")
    print("=" * 60)
    T, H, N = args.num_tokens, cfg.H, cfg.N
    torch.manual_seed(args.seed)
    x = torch.randn(T, H, dtype=torch.bfloat16) * 0.1
    residual_flat = torch.randn(T, N * H, dtype=torch.bfloat16) * 0.1
    h_post = torch.rand(T, N, dtype=torch.float32) * 0.3
    comb_logits = torch.randn(T, N, N, dtype=torch.float32) * 0.5
    comb = ref_sinkhorn(comb_logits, cfg.sinkhorn_iters, cfg.sinkhorn_eps)
    h_res = comb.reshape(T, N * N)

    out = ref_mhc_post(x, residual_flat, h_post, h_res, cfg)
    ref_ok = out.shape == (T, N * H) and out.dtype == torch.bfloat16
    print(f"  x=[{T},{H}] residual=[{T},{N*H}] post=[{T},{N}] comb=[{T},{N},{N}] "
          f"-> out={tuple(out.shape)}")
    # sanity：post=0, comb=I -> out == residual（residual 原样 propagate 回 streams）
    id_comb = torch.eye(N).expand(T, N, N).reshape(T, N * N).contiguous()
    zero_post = torch.zeros(T, N, dtype=torch.float32)
    sanity = ref_mhc_post(x, residual_flat, zero_post, id_comb, cfg)
    d = (sanity.float() - residual_flat.float()).abs().max().item()
    print(f"  [mhc_post/identity_sanity] {'PASS' if d < 5e-3 else 'DIFF'} "
          f"max_diff={d:.3e} (post=0, comb=I -> out≈residual)")

    # ── Zeus K4 真实路径 ─────────────────────────────────────────────
    # 算子: sgl_kernel_zeus.mhc_post(x, residual, post, comb, n=N) -> [T, N*H] bf16
    # 实现: csrc/glm5next_mhc/{mhc_post_kernel.py, sgl_mhc_post_sim.c, mhc_post_zeus.cpp}
    # 文档: sgl-kernel-zeus/docs/mhc_post.md
    # 切核策略: 沿 H 轴 CORE_NUM=2（H 是 per-output 维度，与 K3 同思路）
    # 期望: 5-way fp32 acc + 单 bf16 RNE → atol=rtol=1e-2 (bf16 输出 round noise 主导)
    zeus_ok: Optional[bool] = None
    if args.mode != "ref" and ZEUS_IMPORT_ERROR is None \
            and hasattr(sgl_kernel_zeus, "mhc_post"):
        try:
            z_out = sgl_kernel_zeus.mhc_post(
                x.to("zeus"),
                residual_flat.to("zeus"),
                h_post.to("zeus"),
                h_res.to("zeus"),
                n=N,
            )
            zeus_ok = zeus_compare("mhc_post", out, z_out, atol=1e-2, rtol=1e-2)
        except Exception as e:
            print(f"  ZEUS [mhc_post] EXCEPTION: {e!r}")
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok and d < 5e-3, zeus_ok, "mhc_post",
        "sgl-kernel-zeus post_kernel (沿 H 轴切核，4 输出 × 5-way fp32 acc + 单 RNE)")


# ── Stage: mhc_sublayer_wrap (#7) ───────────────────────────────
def stage_mhc_sublayer_wrap(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_sublayer_wrap (#7 pre -> identity sublayer -> post)  [Zeus chain LANDED]")
    print("=" * 60)
    T, H, N = args.num_tokens, cfg.H, cfg.N
    torch.manual_seed(args.seed)
    residual_flat = torch.randn(T, N * H, dtype=torch.bfloat16) * 0.1

    layer_input, res_pass, h_res, h_post = ref_mhc_pre(residual_flat, p, cfg)
    # identity sublayer：x_sub = layer_input（真实 sublayer 由 attn/mlp 模块替换，
    # 见 dev_glm5next_block_decode_test.py）
    x_sub = layer_input
    out = ref_mhc_post(x_sub, res_pass, h_post, h_res, cfg)
    ref_ok = out.shape == (T, N * H) and out.dtype == torch.bfloat16
    print(f"  residual=[{T},{N*H}] -> pre -> identity -> post -> out={tuple(out.shape)}")
    print(f"  out[0,:4] = {[round(v,4) for v in out[0,:4].float().tolist()]}")

    # ── Zeus 路径：K1→K2→K3 → identity → K4 端到端 ───────────────────
    # REF 用 quantized fn 跑同一条 wrap，吸收 K1 量化噪声；最终 out 容忍 bf16 输出 RNE。
    zeus_ok: Optional[bool] = None
    if args.mode != "ref" and _zeus_has(
            "mhc_pre_norm_split", "mhc_sinkhorn",
            "mhc_pre_apply_mix", "mhc_post"):
        try:
            p_quant = quantize_p_for_zeus_match(p)
            li_q, res_q, hres_q, hpost_q = ref_mhc_pre(residual_flat, p_quant, cfg)
            out_ref_q = ref_mhc_post(li_q, res_q, hpost_q, hres_q, cfg)

            z_li, z_res, z_hres, z_hpost = zeus_mhc_pre(residual_flat, p, cfg)
            # identity sublayer：x_sub_z = z_li（device-resident，无需 round-trip）
            z_out = zeus_mhc_post(z_li, z_res, z_hpost, z_hres, cfg)
            zeus_ok = zeus_compare("mhc_sublayer_wrap.out", out_ref_q, z_out,
                                   atol=2e-2, rtol=1e-2)
        except Exception as e:
            print(f"  ZEUS [mhc_sublayer_wrap] EXCEPTION: {e!r}")
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok, zeus_ok, "mhc_post",
        "K1+K2+K3 → identity sublayer → K4 端到端 wrap (一个 HyperConnection 完整往返)")


# ── Stage: mhc_layer_full (#8) ──────────────────────────────────
def stage_mhc_layer_full(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    print("\n" + "=" * 60)
    print("Stage: mhc_layer_full (#8 attn_hc + mlp_hc 双 wrap)  [Zeus chain LANDED]")
    print("=" * 60)
    T, H, N = args.num_tokens, cfg.H, cfg.N
    torch.manual_seed(args.seed)
    # 两个 HyperConnection（attn-side / mlp-side），权重独立
    p_attn = init_mhc_params(cfg, args.seed)
    p_mlp = init_mhc_params(cfg, args.seed + 1)
    residual_flat = torch.randn(T, N * H, dtype=torch.bfloat16) * 0.1

    # attn-side wrap：pre -> (identity attn) -> post
    li_a, res_a, hres_a, hpost_a = ref_mhc_pre(residual_flat, p_attn, cfg)
    attn_out = li_a                                       # identity attn sublayer
    residual_after_attn = ref_mhc_post(attn_out, res_a, hpost_a, hres_a, cfg)

    # mlp-side wrap：pre -> (identity mlp) -> post
    li_m, res_m, hres_m, hpost_m = ref_mhc_pre(residual_after_attn, p_mlp, cfg)
    mlp_out = li_m                                        # identity mlp sublayer
    residual_next = ref_mhc_post(mlp_out, res_m, hpost_m, hres_m, cfg)

    ref_ok = (residual_after_attn.shape == (T, N * H)
              and residual_next.shape == (T, N * H))
    print(f"  layer chain: residual_in -> [attn_hc] -> residual_mid -> [mlp_hc] -> residual_out")
    print(f"  shapes: in={tuple(residual_flat.shape)} mid={tuple(residual_after_attn.shape)} "
          f"out={tuple(residual_next.shape)}")
    # 一致性：mid 与 out 各 stream 应当不再完全相同（mHC 已混流）
    mid3 = residual_after_attn.view(T, N, H)
    diverged = not torch.equal(mid3[:, 0, :], mid3[:, 1, :])
    print(f"  [mhc_layer_full/streams_diverged] {'PASS' if diverged else 'WARN'} "
          f"(mHC 应当令 N 条流不再恒等)")

    # ── Zeus 路径：attn wrap + mlp wrap 串接（K1→K2→K3→K4 共 8 颗 kernel 调用）─
    # REF 用 quantized fn 跑同一条 chain（两套权重各自量化），吸收 K1 量化噪声。
    # mid 比 out 噪声更小（只一层 K4 round），分别用紧 / 略松的 tol 评估。
    zeus_ok: Optional[bool] = None
    if args.mode != "ref" and _zeus_has(
            "mhc_pre_norm_split", "mhc_sinkhorn",
            "mhc_pre_apply_mix", "mhc_post"):
        try:
            p_attn_q = quantize_p_for_zeus_match(p_attn)
            p_mlp_q = quantize_p_for_zeus_match(p_mlp)

            # REF chain（quantized）
            li_aq, res_aq, hres_aq, hpost_aq = ref_mhc_pre(residual_flat, p_attn_q, cfg)
            mid_ref_q = ref_mhc_post(li_aq, res_aq, hpost_aq, hres_aq, cfg)
            li_mq, res_mq, hres_mq, hpost_mq = ref_mhc_pre(mid_ref_q, p_mlp_q, cfg)
            out_ref_q = ref_mhc_post(li_mq, res_mq, hpost_mq, hres_mq, cfg)

            # Zeus chain（device-resident 全程，z_mid 直接喂下一轮 K1）
            z_li_a, z_res_a, z_hres_a, z_hpost_a = zeus_mhc_pre(residual_flat, p_attn, cfg)
            z_mid = zeus_mhc_post(z_li_a, z_res_a, z_hpost_a, z_hres_a, cfg)
            z_li_m, z_res_m, z_hres_m, z_hpost_m = zeus_mhc_pre(z_mid, p_mlp, cfg)
            z_out = zeus_mhc_post(z_li_m, z_res_m, z_hpost_m, z_hres_m, cfg)

            z_mid_ok = zeus_compare("mhc_layer_full.mid", residual_after_attn, z_mid,
                                    atol=2e-2, rtol=1e-2)
            # out 端误差由两轮 K4 + 两轮 K1 GEMM 复合，放宽到 atol=5e-2
            z_out_ok = zeus_compare("mhc_layer_full.out", out_ref_q, z_out,
                                    atol=5e-2, rtol=2e-2)
            # Zeus 流分歧不变量（与 REF 同步）
            z_mid_cpu = z_mid.cpu().view(T, N, H)
            z_diverged = not torch.equal(z_mid_cpu[:, 0, :], z_mid_cpu[:, 1, :])
            print(f"  ZEUS [mhc_layer_full/streams_diverged] "
                  f"{'PASS' if z_diverged else 'WARN'} (Zeus 输出 N 条流不再恒等)")
            zeus_ok = z_mid_ok and z_out_ok and z_diverged
        except Exception as e:
            print(f"  ZEUS [mhc_layer_full] EXCEPTION: {e!r}")
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok and diverged, zeus_ok, "mhc_post",
        "完整 decoder layer 的 attn_hc + mlp_hc 双 wrap (K1+K2+K3+K4 × 2)")


# ── Stage: linear_bf16 (配套：GAP-3 解决 MoE router / shared / Linear-attn projections) ──
def stage_linear_bf16(args, cfg: Glm5NextMhcConfig, p: MhcParams) -> Optional[bool]:
    """linear_bf16 packed-weight dense GEMM kernel 配套自检（2026-05-25）。

    本 stage 不属于 mHC 本体路径，只是借 mHC dev script 的 cfg / Zeus runtime
    脚手架快速验证 `sgl_kernel_zeus.linear_bf16` 在 GLM5-Next-16B 实战 shape 下
    的数值正确性。它针对 `glm5next_dsa_block_zeus_flow.md` 的 GAP-3 / MoE
    `linear_zeus` 算子缺口。

    覆盖三类典型 shape (per GLM5-Next 16B, H=2048)：
      - shared experts gate_up_proj : [T, 2048] @ [2*1408=2816, 2048].T → [T, 2816]
      - shared experts down_proj    : [T, 1408] @ [2048, 1408].T        → [T, 2048]
      - dense Linear (general)      : [T, 1024] @ [512, 1024].T         → [T, 512]
    """
    import torch.nn.functional as F

    print("\n" + "=" * 60)
    print("Stage: linear_bf16 (packed-weight dense GEMM — GAP-3 配套)  [LANDED 2026-05-25]")
    print("=" * 60)

    T = args.num_tokens
    H = cfg.H
    # GLM5-Next-16B shared experts 形态（mI=1408 是 16B 的 moe_intermediate_size）
    # GLM5-Next 形态 (H=4096) 也复用同一组 shape，weight 实际大小翻倍。
    shapes = [
        (T,  H * 2 + 2 * 128, H,   "shared gate_up（与 GLM-4.7 同形）"),  # gate_up: N = 2*sI
        (T,  H,                H // 2 + 128, "shared down"),
        (T,  512,              1024, "general small Linear"),
    ]

    torch.manual_seed(args.seed)
    ref_ok = True
    zeus_ok: Optional[bool] = None

    if args.mode != "ref" and ZEUS_IMPORT_ERROR is None and hasattr(
            sgl_kernel_zeus, "linear_bf16"):
        zeus_ok = True

    for (M, N, K, label) in shapes:
        # N 必须满足 v1 约束（CORE_NUM × BLOCK_N = 256 倍数）；选 shape 时已保证。
        if N % 256 != 0:
            print(f"  [{label}] SKIP shape ({M}, {N}, {K}) (N % 256 != 0)")
            continue

        input_t = (torch.randn(M, K, dtype=torch.bfloat16) * 0.05).contiguous()
        weight  = (torch.randn(N, K, dtype=torch.bfloat16) * 0.05).contiguous()

        ref = F.linear(input_t.float(), weight.float()).to(torch.bfloat16)
        print(f"  [{label}]  ({M}, {N}, {K})  weight={N*K*2/1024:.1f} KB")

        if zeus_ok is None:
            continue

        try:
            weight_lmem = torch.zeus.local_memory.from_tensor(
                weight.to("zeus"), kind="weight", Tr=1, Tc=1,
            )
            got = sgl_kernel_zeus.linear_bf16(input_t.to("zeus"), weight_lmem)
            diff = (got.cpu().float() - ref.float()).abs()
            this_ok = diff.max().item() < 5e-2
            print(f"    ZEUS [linear_bf16] {'PASS' if this_ok else 'DIFF'} | "
                  f"max_diff={diff.max().item():.4e} mean_diff={diff.mean().item():.4e}")
            zeus_ok = zeus_ok and this_ok
        except Exception as e:
            print(f"    ZEUS [linear_bf16] EXCEPTION: {e!r}")
            zeus_ok = False

    return finish_stage_with_zeus(
        args, ref_ok, zeus_ok, "linear_bf16",
        "sgl_kernel_zeus.linear_bf16(input, weight_lmem) → [M, N] bf16；"
        "fp32 acc + 单 RNE；N-axis 2-core split (v1 sim grid=1)")


# ── Dispatch ───────────────────────────────────────────────────
STAGES = {
    "mhc_expand": stage_mhc_expand,
    "mhc_pre_norm_fn": stage_mhc_pre_norm_fn,
    "mhc_pre_split_mixes": stage_mhc_pre_split_mixes,
    "mhc_pre_norm_split": stage_mhc_pre_norm_split,
    "mhc_sinkhorn": stage_mhc_sinkhorn,
    "mhc_pre_apply_mix": stage_mhc_pre_apply_mix,
    "mhc_pre": stage_mhc_pre,
    "mhc_post": stage_mhc_post,
    "mhc_sublayer_wrap": stage_mhc_sublayer_wrap,
    "mhc_layer_full": stage_mhc_layer_full,
    "linear_bf16": stage_linear_bf16,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=list(STAGES.keys()) + ["all"], default="all")
    parser.add_argument("--config", choices=["16b", "next"], default="16b")
    parser.add_argument("--mode", choices=["both", "ref", "zeus"], default="both")
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--block-t", type=int, default=4,
                        help="BLOCK_T tile size for融合 K1 mhc_pre_norm_split 的 tile-invariance 自检")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = select_config(args.config)
    p = init_mhc_params(cfg, args.seed)
    print(f"Config: {cfg.name}  H={cfg.H}  N={cfg.N}  mix_hc={cfg.mix_hc}  d_model={cfg.d_model}")
    print(f"  sinkhorn_iters={cfg.sinkhorn_iters}  hc_eps={cfg.sinkhorn_eps}  "
          f"post_mult_value={cfg.post_mult_value}  no_norm_weight={cfg.no_norm_weight}")
    print(f"  GLM-extended (悬空): mhc_tau={cfg.tau}  hres_vwnstyle={cfg.hres_vwnstyle}")
    print(f"Reference device: {REF_DEVICE}")
    print(f"Zeus runtime available: {ZEUS_IMPORT_ERROR is None}")

    results = {}
    for name, fn in STAGES.items():
        if args.stage not in (name, "all"):
            continue
        try:
            results[name] = fn(args, cfg, p)
        except Exception as e:
            print(f"  [{name}] EXCEPTION: {e}")
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
        print(f"  {name:24s} : {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
