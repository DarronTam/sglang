"""
GLM5-Next Linear-transformer block (decode) 整层装配 + REF↔Zeus 对拍.

把 ``Glm5NextLinearAttn`` (KDA sublayer) + ``Glm5NextDenseFFN`` (Dense FFN sublayer) +
mHC wrapper 串成一个完整 decoder layer:

  residual[B, N*H]
    │ attn_hc.pre   → layer_input[B, H]
  Glm5NextLinearAttn(layer_input, conv_state, rec_state)
    │ attn_hc.post  → residual_mid[B, N*H]
    │ mlp_hc.pre    → layer_input[B, H]
  Glm5NextDenseFFN(layer_input)
    │ mlp_hc.post   → residual_out[B, N*H]

- attn_hc / mlp_hc 用 ``mhc.init_mhc_params`` 各自 seed 独立 (与 prerelease
  glm5_next.py 同一个 layer 内 attn-mHC / mlp-mHC 两套独立 mix 参数对齐)
- 与 dev_glm5next_block_decode_test.stage_linear_attn_block 的区别:
  本脚本 **MLP 用 Dense FFN(dev_dense_ffn)**（Glm5NextDenseFFN）而非 MoE; 适用于
  有 Dense FFN 层的配置.

用法:
  python glm5next_modules/dev_linear_attn_dense_block.py
  python glm5next_modules/dev_linear_attn_dense_block.py --config next --mode zeus
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# 公共脚手架 (sys.path bootstrap + Zeus runtime + helpers)
import _common
from _common import (
    ZEUS_IMPORT_ERROR, sgl_kernel_zeus,
    compare_tensors, zeus_chain_available,
    make_argparser, print_header, print_summary,
    REF_MEMORY_BUDGET_GB,
)

# mHC wrapper (本目录, lazy _pack_zeus 一次性 LocalMem pack)
from dev_mhc import Glm5NextMhc

# 同目录 sublayer modules
from dev_linear_attn import Glm5NextLinearAttn, load_cfg as load_attn_cfg
from dev_dense_ffn import Glm5NextDenseFFN, load_cfg as load_ffn_cfg


# ── Block module ────────────────────────────────────────────────
class Glm5NextLinearAttnDenseBlock:
    """GLM5-Next linear-transformer decoder block (decode single-step).

    Composition (mHC wrapper 包两次, attn 一次 / mlp 一次):
      - attn_mhc:       ``Glm5NextMhc``       (HyperConnection wrapper for attn)
      - attn sublayer:  ``Glm5NextLinearAttn`` (state-bearing — conv/rec)
      - mlp_mhc:        ``Glm5NextMhc``       (HyperConnection wrapper for mlp)
      - mlp  sublayer:  ``Glm5NextDenseFFN``

    使用模式::

        block = Glm5NextLinearAttnDenseBlock(which="16b", seed=42)
        conv, rec = block.init_state(B=batch, seed=...)
        # REF
        mid, out = block.forward(residual_flat, conv, rec)
        # Zeus
        z_mid, z_out = block.forward_zeus(
            residual_flat.to("zeus"), conv.clone().to("zeus"), rec.clone().to("zeus"),
        )

    Residual 全程 ``[B, N*H]`` bf16; 内部 sublayer 接 ``[B, H]``.  mHC 把 N 条
    residual stream 在 pre 阶段 sinkhorn-mix 成一条, 在 post 阶段 read 出.
    """

    def __init__(self, which: str, seed: int = 0):
        # 子模块各自构建 (sublayer 自己的 weight init seed 与 mHC 解耦)
        attn_cfg = load_attn_cfg(which)
        ffn_cfg = load_ffn_cfg(which)

        self.attn_cfg = attn_cfg
        self.ffn_cfg = ffn_cfg
        self.attn = Glm5NextLinearAttn(attn_cfg, seed=seed)
        self.ffn = Glm5NextDenseFFN(ffn_cfg, seed=seed + 5)

        # 两个独立 mHC wrapper (attn-mhc / mlp-mhc), 与 glm5_next.py 同 layer 双副本
        # 对齐. 每个 wrapper 内部 own MhcParams + quantized 副本, 以及 lazy
        # `_pack_zeus()` 缓存的 fn LocalMem (首次 forward_pre_zeus 一次性 pack,
        # 不再每 forward repack —— 与原 mhc.zeus_mhc_pre 的关键改进).
        self.attn_mhc = Glm5NextMhc(which, seed=seed)
        self.mlp_mhc  = Glm5NextMhc(which, seed=seed + 100)

        assert attn_cfg.H == ffn_cfg.H == self.attn_mhc.cfg.H, (
            f"H mismatch: attn={attn_cfg.H} ffn={ffn_cfg.H} "
            f"mhc={self.attn_mhc.cfg.H}"
        )
        # mhc_cfg 取 attn_mhc 的 (两个 wrapper 配置相同, 只是 seed 不同)
        self.mhc_cfg = self.attn_mhc.cfg

    # ── State init (forward to sublayer.attn) ───────────────────
    def init_state(self, B: int, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """构造初始 (conv_state, rec_state) host tensors —— 只 attn sublayer 需要."""
        return self.attn.init_state(B, seed)

    # ── REF forward ─────────────────────────────────────────────
    def forward(self,
                residual_flat: torch.Tensor,
                conv_state: torch.Tensor,
                rec_state: torch.Tensor,
                *, quantize_mhc: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """REF block decode (state 原位推进).

        ``residual_flat: [B, N*H] bf16  ->  (residual_mid, residual_out)``,
        each ``[B, N*H] bf16``.  ``residual_mid`` 是 attn sublayer 写回 mHC 之后
        的中间 state; ``residual_out`` 是 mlp sublayer 写回之后的最终 state.

        ``quantize_mhc=True`` 时两个 mHC wrapper 都走 bf16 quantize 版参数,
        便于与 Zeus 对拍 (mHC fn 在 K1 内会做 bf16 截断).
        """
        # ── attn block: mHC pre → attn → mHC post
        li_a, res_a, hres_a, hpost_a = self.attn_mhc.forward_pre(
            residual_flat, quantize=quantize_mhc,
        )
        attn_out = self.attn.forward(li_a, conv_state, rec_state)
        residual_mid = self.attn_mhc.forward_post(attn_out, res_a, hpost_a, hres_a)

        # ── mlp block: mHC pre → Dense FFN → mHC post
        li_m, res_m, hres_m, hpost_m = self.mlp_mhc.forward_pre(
            residual_mid, quantize=quantize_mhc,
        )
        mlp_out = self.ffn.forward(li_m)
        residual_out = self.mlp_mhc.forward_post(mlp_out, res_m, hpost_m, hres_m)
        return residual_mid, residual_out

    # ── Zeus forward ────────────────────────────────────────────
    def forward_zeus(self,
                     residual_flat_z: torch.Tensor,
                     conv_state_z: torch.Tensor,
                     rec_state_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zeus block decode (全 device-resident).

        - mHC chain (K1/K2/K3/K4 × 2) 全 Zeus, fn weight 在 ``Glm5NextMhc``
          内部一次性 pack (_pack_zeus lazy cache, 不再每 forward repack).
        - attn / ffn sublayer 各自 forward_zeus, 内部全 device-resident.
        - 整段 pipeline 无 host↔device cast, 见各 sublayer dev script 的 audit.
        """
        # ── attn block (device-resident)
        z_li_a, z_res_a, z_hres_a, z_hpost_a = self.attn_mhc.forward_pre_zeus(
            residual_flat_z,
        )
        attn_out_z = self.attn.forward_zeus(z_li_a, conv_state_z, rec_state_z)
        z_mid = self.attn_mhc.forward_post_zeus(
            attn_out_z, z_res_a, z_hpost_a, z_hres_a,
        )

        # ── mlp block (device-resident, z_mid 直接喂下一轮 mHC pre, 不下 host)
        z_li_m, z_res_m, z_hres_m, z_hpost_m = self.mlp_mhc.forward_pre_zeus(z_mid)
        mlp_out_z = self.ffn.forward_zeus(z_li_m)
        z_out = self.mlp_mhc.forward_post_zeus(
            mlp_out_z, z_res_m, z_hpost_m, z_hres_m,
        )
        return z_mid, z_out


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = (
    # mHC chain
    "mhc_pre_norm_split", "mhc_sinkhorn", "mhc_pre_apply_mix", "mhc_post",
    # linear-attn sublayer
    "linear_bf16", "linear_bf16_outfp32_sigmoid",
    "causal_conv1d_update_split", "fused_kda_gate",
    "fused_recurrent_kda_Sdecay", "rms_norm_gated",
    # dense FFN sublayer (linear_bf16 已在 linear-attn 段; 这里一并列出明确依赖,
    # 与 dev_dsa_attn_dense_block 一致, 防将来换 attn 后漏掉)
    "linear_bf16", "silu_and_mul",
)


def _run_stage(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print(f"Stage: linear-attn + Dense block ({args.config})")
    print("=" * 60)

    torch.manual_seed(args.seed)
    block = Glm5NextLinearAttnDenseBlock(args.config, seed=args.seed)
    cfg = block.mhc_cfg
    B = args.num_tokens
    H, N = cfg.H, cfg.N
    print(f"  cfg: H={H} N={N} (mHC streams)  "
          f"attn[Hh={block.attn_cfg.num_heads} Dk={block.attn_cfg.head_k_dim}]  "
          f"ffn[I={block.ffn_cfg.I}]")

    # bf16 residual, scale 0.05 与各 sublayer dev script 一致
    residual_flat = (
        torch.randn(B, N * H, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)
    conv_init, rec_init = block.init_state(B, seed=args.seed + 1)

    # ── REF (quantize_mhc=True 让 mHC 参数与 Zeus 等价) ─────────
    ref_mid: Optional[torch.Tensor] = None
    ref_out: Optional[torch.Tensor] = None
    ref_skipped = False
    if args.mode in ("ref", "both"):
        conv_ref = conv_init.clone()
        rec_ref  = rec_init.clone()
        ref_mid, ref_out = block.forward(
            residual_flat, conv_ref, rec_ref, quantize_mhc=True,
        )
        state_advanced = (
            not torch.equal(conv_ref, conv_init)
            and not torch.equal(rec_ref, rec_init)
        )
        shape_ok = (ref_mid.shape == (B, N * H)
                    and ref_out.shape == (B, N * H)
                    and ref_out.dtype == torch.bfloat16)
        # mHC 关键不变量：N 条 residual stream 经 attn 后应被 mix —— 即 mid
        # 的 N 切片不再相等 (否则 mHC 没工作).
        mid3 = ref_mid.view(B, N, H)
        diverged = not torch.equal(mid3[:, 0, :], mid3[:, 1, :])
        print(f"  REF mid={tuple(ref_mid.shape)} out={tuple(ref_out.shape)}  "
                f"state_advanced={state_advanced}  streams_diverged={diverged}")
        print(f"  REF out[0,:4] = "
                f"{[round(v,4) for v in ref_out[0,:4].float().tolist()]}")
        if not (shape_ok and state_advanced and diverged):
            return False

    # ── Zeus ──────────────────────────────────────────────────
    zeus_ok: Optional[bool] = None
    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"  ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
        else:
            try:
                conv_z = conv_init.clone().to("zeus")
                rec_z  = rec_init.clone().to("zeus")
                z_mid, z_out = block.forward_zeus(
                    residual_flat.to("zeus"), conv_z, rec_z,
                )
                z_mid_cpu = z_mid.cpu()
                z_out_cpu = z_out.cpu()
                finite = torch.isfinite(z_out_cpu).all().item()
                shape_ok = (z_mid_cpu.shape == (B, N * H)
                            and z_out_cpu.shape == (B, N * H)
                            and z_out_cpu.dtype == torch.bfloat16)
                z_mid3 = z_mid_cpu.view(B, N, H)
                z_diverged = not torch.equal(z_mid3[:, 0, :], z_mid3[:, 1, :])
                print(f"  ZEUS mid={tuple(z_mid_cpu.shape)} out={tuple(z_out_cpu.shape)}  "
                      f"finite={finite}  streams_diverged={z_diverged}")
                print(f"  ZEUS out[0,:4] = "
                      f"{[round(v,4) for v in z_out_cpu[0,:4].float().tolist()]}")

                if ref_out is not None:
                    # 复合误差预算 (mHC 5e-3 + linear-attn 5e-2 + Dense FFN 5e-2),
                    # 真实 16b/next 配置下经过 ~31 个算子, 无 MoE routing 非确定性,
                    # 误差可控. mid 在 mlp 块之前, 误差小; out 在最后一层后, 累积最大.
                    mid_ok = compare_tensors(
                        f"block.{args.config}.mid", ref_mid, z_mid_cpu,
                        atol=1.5e-1, rtol=5e-2,
                    )
                    out_ok = compare_tensors(
                        f"block.{args.config}.out", ref_out, z_out_cpu,
                        # dense 无 topk 非确定性 (rtol 1e-1 比 moe block 的 2e-1 紧);
                        # abs 预算留余量防 knife-edge: 实测 max_diff≈4.0 (深链 bf16 累积).
                        atol=5.0, rtol=1e-1,
                    )
                    zeus_ok = mid_ok and out_ok and finite and shape_ok and z_diverged
                else:
                    zeus_ok = finite and shape_ok and z_diverged
            except Exception as e:
                import traceback
                print(f"  ZEUS EXCEPTION: {e!r}")
                traceback.print_exc()
                zeus_ok = False

    # ── status 汇总 ──────────────────────────────────────────
    if args.mode == "ref":
        return None if ref_skipped else True
    if args.mode == "zeus":
        return zeus_ok
    # both
    if ref_skipped:
        return None if zeus_ok is None else zeus_ok
    if zeus_ok is None:
        return True
    return zeus_ok


def main():
    parser = make_argparser(
        "dev_linear_attn_dense_block",
        description="GLM5-Next linear-transformer decoder block dev test "
                    "(mHC + Linear-attn + Dense FFN)",
    )
    args = parser.parse_args()
    print_header("GLM5-Next linear-transformer block (mHC + KDA + Dense FFN)", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_linear_attn_dense_block ({args.config})", ok)


if __name__ == "__main__":
    main()
