"""
GLM5-Next mHC (HyperConnection) wrapper 独立模块 + REF↔Zeus 对拍.

设计与 ``dev_linear_attn.py`` / ``dev_moe.py`` 同构 —— 单独把 mHC pre/post
装成一个 module, 在 main 里用 **identity sublayer** 单元测试, 隔离观察 mHC
本身是否引入奇怪的 host↔device cast / 非 contig view / fallback / 反复
LocalMem repack 等隐藏操作.

mHC chain (与 dev_glm5next_block_decode_test / dev_linear_attn_block 一致):
  residual[T, N*H]
    │ pre  (K1 mhc_pre_norm_split + K2 mhc_sinkhorn + K3 mhc_pre_apply_mix)
    │    → layer_input[T, H] bf16, h_res[T, N*N] fp32, h_post[T, N] fp32
  sublayer(layer_input)                              ← 本脚本用 identity (passthrough)
    │ post (K4 mhc_post)
    │    → residual_out[T, N*H] bf16

与 dev_glm5next_mhc_test 中 ``zeus_mhc_pre`` 的差别:
  - 本 wrapper 把 ``fn`` LocalMem pack + ``base.to("zeus")`` 等 host→device 准备
    工作搬到 ``_pack_zeus()`` 一次性做掉, ``forward_pre_zeus`` 不再每次 repack
    weight. 与 Glm5NextLinearAttn / Glm5NextMoE 的 lazy-pack 模式一致.

用法:
  python glm5next_modules/dev_mhc.py                # 16b / both
  python glm5next_modules/dev_mhc.py --config next
  python glm5next_modules/dev_mhc.py --mode zeus
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# 公共脚手架
import _common
from _common import (
    ZEUS_IMPORT_ERROR, sgl_kernel_zeus,
    compare_tensors, zeus_chain_available,
    make_argparser, print_header, print_summary,
)

# mHC 底层 API (位于上一级 model_state_dev/)
import dev_glm5next_mhc_test as mhc


# ── Module ──────────────────────────────────────────────────────
class Glm5NextMhc:
    """mHC HyperConnection wrapper (pre + post 两段, sublayer 由 caller 提供).

    使用模式::

        block = Glm5NextMhc(which="16b", seed=42)
        # REF
        li, res, hres, hpost = block.forward_pre(residual_flat)
        sub_out = my_sublayer(li)
        residual_out = block.forward_post(sub_out, res, hpost, hres)
        # Zeus
        z_li, z_res, z_hres, z_hpost = block.forward_pre_zeus(residual_flat.to("zeus"))
        z_sub_out = my_sublayer_zeus(z_li)
        z_residual_out = block.forward_post_zeus(z_sub_out, z_res, z_hpost, z_hres)
    """

    def __init__(self, which: str, seed: int = 0):
        self.cfg = mhc.select_config(which)
        self.params = mhc.init_mhc_params(self.cfg, seed)
        # quantized 版用于与 Zeus 对拍 (mHC params 在 K1 内会做 bf16 截断 +
        # norm_weight pre-merge; 让 REF 走 quantize 版后 K1 input 与 Zeus 等价).
        self.params_q = mhc.quantize_p_for_zeus_match(self.params)

        # Zeus device-resident state — lazy
        self._zeus_packed = False
        self._fn_lmem: Optional[torch.Tensor] = None
        self._base_z: Optional[torch.Tensor] = None

    # ── REF forward ─────────────────────────────────────────────
    def forward_pre(self, residual_flat: torch.Tensor, *, quantize: bool = False
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """REF mHC pre.  residual_flat: [T, N*H] bf16
        → (layer_input[T, H] bf16, residual (passthrough), h_res[T, N*N] fp32,
          h_post[T, N] fp32). ``comb`` 已经过 Sinkhorn (doubly-stochastic)."""
        p = self.params_q if quantize else self.params
        return mhc.ref_mhc_pre(residual_flat, p, self.cfg)

    def forward_post(self, x: torch.Tensor, residual: torch.Tensor,
                     h_post: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
        """REF mHC post. x[T,H] bf16 + residual[T,N*H] bf16 + h_post[T,N] fp32
        + h_res[T,N*N] fp32 → residual_out[T, N*H] bf16."""
        return mhc.ref_mhc_post(x, residual, h_post, h_res, self.cfg)

    # ── Zeus pack (lazy, 一次性) ────────────────────────────────
    def _pack_zeus(self) -> None:
        """把 ``fn`` 与 ``norm_weight`` host-side pre-merge 后 bf16 量化, 一次性
        装包到 LocalMem; ``base`` 也搬上 Zeus device 缓存.

        与 ``dev_glm5next_mhc_test.zeus_mhc_pre`` 的关键差别: 那个函数每次调用
        都重新 ``.to(bf16).to("zeus")`` + ``from_tensor`` LocalMem pack, host 端
        反复跑同样的 norm_weight merge + 量化. 我们这里做成 layer-init 一次性,
        forward 复用 (与 ``Glm5NextLinearAttn._pack_zeus`` /
        ``Glm5NextMoE._pack_zeus`` 同套路).
        """
        if ZEUS_IMPORT_ERROR is not None:
            raise RuntimeError(f"Zeus runtime unavailable: {ZEUS_IMPORT_ERROR}")
        p = self.params
        # fn 与 norm_weight 在 host 端先 merge, 然后 bf16 RNE 量化, 然后 LocalMem
        # tiled pack. K1 (mhc_pre_norm_split) 期望 weight=fn_effective(bf16) LocalMem.
        fn_eff_fp32 = p.fn if p.norm_weight is None else p.fn * p.norm_weight
        fn_eff_bf16 = fn_eff_fp32.to(torch.bfloat16)
        self._fn_lmem = sgl_kernel_zeus.mhc_pre_norm_split.pack(fn_eff_bf16)
        self._base_z = p.base.to("zeus")
        # NOTE: p.scale (shape [3] fp32) — K1 kernel 直接接 CPU fp32 tensor, 无需
        # 搬到 device (见 dev_glm5next_mhc_test.zeus_mhc_pre 的实现).
        self._zeus_packed = True

    # ── Zeus forward ────────────────────────────────────────────
    def forward_pre_zeus(self, residual_flat_z: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Zeus mHC pre (K1 → K2 → K3 device-resident).

        residual_flat_z: [T, N*H] bf16 Zeus → (layer_input[T, H] bf16 Zeus,
        residual_z passthrough, h_res[T, N*N] fp32 Zeus, h_post[T, N] fp32 Zeus).
        """
        if not self._zeus_packed:
            self._pack_zeus()
        cfg = self.cfg
        s = residual_flat_z.shape[0]
        n = cfg.N
        # K1: fused (RMSNorm + fp32 Linear(fn) + split + scale/base/sigmoid)
        pre, post, comb_logits = sgl_kernel_zeus.mhc_pre_norm_split(
            residual_flat_z,
            self._fn_lmem,
            self.params.scale,          # CPU fp32 — kernel 直接接受
            self._base_z,
            n=n,
            rms_norm_eps=cfg.rms_norm_eps,
            pre_eps=cfg.pre_eps,
            post_mult_value=cfg.post_mult_value,
        )
        # K2: Sinkhorn (comb_logits 双随机化)
        comb = sgl_kernel_zeus.mhc_sinkhorn(
            comb_logits, repeat=cfg.sinkhorn_iters, eps=cfg.sinkhorn_eps,
        )
        # K3: residual × pre → layer_input[T, H]
        layer_input = sgl_kernel_zeus.mhc_pre_apply_mix(
            residual_flat_z, pre, n=n,
        )
        # comb[s,n,n]→[s,n*n] 合并末尾连续维; post[s,n,1]→[s,n] 去末尾 size-1 维.
        # 二者在 contiguous kernel 输出上恒为零拷贝, 用 .view (而非 .reshape) 让这条
        # 不变量 load-bearing —— 将来 kernel 输出若变非 contiguous 直接报错, 不静默拷贝.
        h_res = comb.view(s, n * n)
        h_post = post.view(s, n)
        return layer_input, residual_flat_z, h_res, h_post

    def forward_post_zeus(self, x_z: torch.Tensor, residual_z: torch.Tensor,
                          h_post_z: torch.Tensor, h_res_z: torch.Tensor) -> torch.Tensor:
        """Zeus mHC post (K4) device-resident."""
        return sgl_kernel_zeus.mhc_post(
            x_z, residual_z, h_post_z, h_res_z, n=self.cfg.N,
        )


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = (
    "mhc_pre_norm_split", "mhc_sinkhorn", "mhc_pre_apply_mix", "mhc_post",
)


def _run_stage(args) -> Optional[bool]:
    """单 stage: identity-sublayer 单元测试隔离 mHC 行为."""
    print("\n" + "=" * 60)
    print(f"Stage: mHC wrapper ({args.config})  [sublayer = identity]")
    print("=" * 60)

    torch.manual_seed(args.seed)
    block = Glm5NextMhc(args.config, seed=args.seed)
    cfg = block.cfg
    B = args.num_tokens
    H, N = cfg.H, cfg.N
    print(f"  cfg: H={H}  N={N}  mix_hc={cfg.mix_hc}  "
          f"sinkhorn_iters={cfg.sinkhorn_iters}  "
          f"no_norm_weight={cfg.no_norm_weight}")

    residual_flat = (
        torch.randn(B, N * H, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)

    # ── REF (quantize=True 让 mHC 参数与 Zeus 等价) ─────────────
    li_ref: Optional[torch.Tensor] = None
    hres_ref: Optional[torch.Tensor] = None
    hpost_ref: Optional[torch.Tensor] = None
    out_ref: Optional[torch.Tensor] = None
    if args.mode in ("ref", "both"):
        li_ref, res_ref, hres_ref, hpost_ref = block.forward_pre(
            residual_flat, quantize=True,
        )
        # identity sublayer (passthrough)
        sub_ref = li_ref
        out_ref = block.forward_post(sub_ref, res_ref, hpost_ref, hres_ref)

        # 不变量: Sinkhorn 输出 doubly-stochastic, mHC 必须把 N 条 stream 混在一起
        comb_ref = hres_ref.view(B, N, N)
        ds_ok = mhc.assert_doubly_stochastic("ref.comb", comb_ref, tol=1e-2)
        # streams_diverged: identity sublayer 下 out 的 N 条流应不再相等 (post 块
        # 用 comb 把 residual N 条流 mix 进每条 out, 即便 sub_out 单条也会触发分散)
        out3 = out_ref.view(B, N, H)
        diverged = not torch.equal(out3[:, 0, :], out3[:, 1, :])
        print(f"  REF li={tuple(li_ref.shape)} {li_ref.dtype}  "
              f"hres={tuple(hres_ref.shape)} {hres_ref.dtype}  "
              f"hpost={tuple(hpost_ref.shape)} {hpost_ref.dtype}  "
              f"out={tuple(out_ref.shape)} {out_ref.dtype}")
        print(f"  REF streams_diverged={diverged}  doubly_stochastic={ds_ok}")
        if not (ds_ok and diverged):
            return False

    # ── Zeus ──────────────────────────────────────────────────
    zeus_ok: Optional[bool] = None
    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"  ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
        else:
            try:
                z_li, z_res, z_hres, z_hpost = block.forward_pre_zeus(
                    residual_flat.to("zeus"),
                )
                # identity sublayer (device-resident, passthrough)
                z_sub = z_li
                z_out = block.forward_post_zeus(z_sub, z_res, z_hpost, z_hres)

                z_li_cpu = z_li.cpu()
                z_hres_cpu = z_hres.cpu()
                z_hpost_cpu = z_hpost.cpu()
                z_out_cpu = z_out.cpu()
                finite = torch.isfinite(z_out_cpu).all().item()
                print(f"  ZEUS li={tuple(z_li.shape)} {z_li.dtype}  "
                      f"hres={tuple(z_hres.shape)} {z_hres.dtype}  "
                      f"hpost={tuple(z_hpost.shape)} {z_hpost.dtype}  "
                      f"out={tuple(z_out.shape)} {z_out.dtype}  finite={finite}")
                # 不变量
                z_comb = z_hres_cpu.view(B, N, N)
                z_ds_ok = mhc.assert_doubly_stochastic("zeus.comb", z_comb, tol=1e-2)
                z_out3 = z_out_cpu.view(B, N, H)
                z_diverged = not torch.equal(z_out3[:, 0, :], z_out3[:, 1, :])
                print(f"  ZEUS streams_diverged={z_diverged}")

                if li_ref is not None:
                    # 4 路对拍: pre 三个输出 + post 输出. mHC 内部 fp32 acc,
                    # 单 bf16 RNE store —— 与 REF (quantize 版) 应该 5e-3 envelope.
                    li_ok    = compare_tensors("mhc.pre.li",    li_ref,    z_li_cpu,    atol=5e-3, rtol=5e-3)
                    hres_ok  = compare_tensors("mhc.pre.hres",  hres_ref,  z_hres_cpu,  atol=5e-3, rtol=5e-3)
                    hpost_ok = compare_tensors("mhc.pre.hpost", hpost_ref, z_hpost_cpu, atol=5e-3, rtol=5e-3)
                    out_ok   = compare_tensors("mhc.out",       out_ref,   z_out_cpu,   atol=5e-3, rtol=5e-3)
                    zeus_ok = (li_ok and hres_ok and hpost_ok and out_ok
                               and finite and z_ds_ok and z_diverged)
                else:
                    zeus_ok = finite and z_ds_ok and z_diverged
            except Exception as e:
                import traceback
                print(f"  ZEUS EXCEPTION: {e!r}")
                traceback.print_exc()
                zeus_ok = False

    if args.mode == "ref":
        return True
    if args.mode == "zeus":
        return zeus_ok
    return True if zeus_ok is None else zeus_ok


def main():
    parser = make_argparser(
        "dev_mhc",
        description="GLM5-Next mHC (HyperConnection) wrapper dev test",
    )
    args = parser.parse_args()
    print_header("GLM5-Next mHC (HyperConnection) wrapper", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_mhc ({args.config})", ok)


if __name__ == "__main__":
    main()
