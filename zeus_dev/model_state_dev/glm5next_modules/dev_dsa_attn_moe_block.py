"""
GLM5-Next DSA-transformer block (decode) 整层装配 + REF↔Zeus 对拍.

与 ``dev_linear_attn_moe_block.py`` 同构, 但 attn sublayer 换成 DSA (paged-attention):

  residual[B, N*H]
    │ attn_hc.pre   → layer_input[B, H]
  Glm5NextDsaAttn(layer_input, history / paged_state)
    │ attn_hc.post  → residual_mid[B, N*H]
    │ mlp_hc.pre    → layer_input[B, H]
  Glm5NextMoE(layer_input)
    │ mlp_hc.post   → residual_out[B, N*H]

- attn_hc / mlp_hc 用 ``mhc.init_mhc_params`` 各自 seed 独立 (与 prerelease
  glm5_next.py 同一个 layer 内 attn-mHC / mlp-mHC 两套独立 mix 参数对齐)
- DSA attn sublayer 走 paged-attention chain: REF 用 ``dsa.GlobalHistory`` host
  tensors; Zeus 用 :meth:`Glm5NextDsaAttn.init_paged_state` 构造的 paged_state
  (latent/body/scale 共享池 + block_table + seq_lens), init 时即一次性搬上
  device, 之后跨 step 复用, 全链路 device-resident.
- MLP 与 ``dev_linear_attn_moe_block.py`` 一致, 用 ``Glm5NextMoE``.

用法:
  python glm5next_modules/dev_dsa_attn_moe_block.py
  python glm5next_modules/dev_dsa_attn_moe_block.py --config next --mode zeus --seqlen 64
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
from dev_dsa_attn import Glm5NextDsaAttn
from dev_moe import Glm5NextMoE, load_cfg as load_moe_cfg

# DSA 底层 API (用于 history 类型注解)
import dev_glm5next_dsa_decode_test as dsa


# ── Block module ────────────────────────────────────────────────
class Glm5NextDsaAttnMoeBlock:
    """GLM5-Next DSA-transformer decoder block (decode single-step).

    Composition (mHC wrapper 包两次, attn 一次 / mlp 一次):
      - attn_mhc:       ``Glm5NextMhc``        (HyperConnection wrapper for attn)
      - attn sublayer:  ``Glm5NextDsaAttn``    (paged-attention, state-bearing)
      - mlp_mhc:        ``Glm5NextMhc``        (HyperConnection wrapper for mlp)
      - mlp  sublayer:  ``Glm5NextMoE``

    使用模式::

        block = Glm5NextDsaAttnMoeBlock(which="16b", seed=42)
        history, block_span = block.init_state(B=batch, seqlen=ctx_len, seed=...)
        # REF
        mid, out = block.forward(residual_flat, history, new_pos=ctx_len)
        # Zeus (dual-core paged)
        paged_state = block.init_paged_state(
            history, page_size=512, num_physical_pages=4,
        )
        block.prepare_decode_step(paged_state)   # host: slot_mapping + advance seq_lens
        z_mid, z_out = block.forward_zeus(residual_flat.to("zeus"), paged_state)

    Residual 全程 ``[B, N*H]`` bf16; 内部 sublayer 接 ``[B, H]``. mHC 把 N 条
    residual stream 在 pre 阶段 sinkhorn-mix 成一条, 在 post 阶段 read 出.
    """

    def __init__(self, which: str, seed: int = 0):
        # DSA attn sublayer (用 which 字符串构造, 内部走 dsa.select_config)
        self.attn = Glm5NextDsaAttn(which, seed=seed)

        # MoE sublayer
        moe_cfg = load_moe_cfg(which)
        self.moe_cfg = moe_cfg
        self.moe = Glm5NextMoE(moe_cfg, seed=seed + 5)

        # 两个独立 mHC wrapper, 与 dev_linear_attn_moe_block 同方案
        self.attn_mhc = Glm5NextMhc(which, seed=seed)
        self.mlp_mhc  = Glm5NextMhc(which, seed=seed + 100)

        # H 必须一致 (attn / moe / mhc 都看同一条 residual stream)
        attn_H = self.attn.cfg.H
        assert attn_H == moe_cfg.H == self.attn_mhc.cfg.H, (
            f"H mismatch: attn={attn_H} moe={moe_cfg.H} "
            f"mhc={self.attn_mhc.cfg.H}"
        )
        self.mhc_cfg = self.attn_mhc.cfg
        self.attn_cfg = self.attn.cfg

    # ── State init ──────────────────────────────────────────────
    def init_state(self, B: int, seqlen: int, seed: int = 0,
                   block_span: int = 16
                   ) -> Tuple[dsa.GlobalHistory, int]:
        """构造初始 KV history (REF 用) + block_span 常量.

        Zeus 路径需要再调 :meth:`init_paged_state(history, page_size=...,
        num_physical_pages=...)` 把 history 摊到 paged pool. 两套 state 共用
        同一份 host history, 保证 REF / Zeus 起点一致.
        """
        return self.attn.init_state(B, seqlen, seed=seed, block_span=block_span)

    def init_paged_state(self, history: dsa.GlobalHistory, *,
                         page_size: int, num_physical_pages: int) -> dict:
        """从 history 构造 dual-core paged_state (Zeus 用)."""
        return self.attn.init_paged_state(
            history, page_size=page_size, num_physical_pages=num_physical_pages,
        )

    def prepare_decode_step(self, paged_state: dict) -> None:
        """Host per-step prepare (算 slot_mapping + advance seq_lens), 委托给 attn。
        每个 decode step 在 :meth:`forward_zeus` 前调用一次 (对齐上游 prepare_for_decode)。"""
        self.attn.prepare_decode_step(paged_state)

    # ── REF forward ─────────────────────────────────────────────
    def forward(self,
                residual_flat: torch.Tensor,
                history: dsa.GlobalHistory,
                new_pos: int,
                *, block_span: int = 16,
                quantize_mhc: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """REF block decode.

        ``residual_flat: [B, N*H] bf16  ->  (residual_mid, residual_out)``,
        each ``[B, N*H] bf16``. ``residual_mid`` 是 attn sublayer 写回 mHC 之后
        的中间 state; ``residual_out`` 是 mlp sublayer 写回之后的最终 state.

        ``quantize_mhc=True`` 时两个 mHC wrapper 都走 bf16 quantize 版参数,
        便于与 Zeus 对拍 (mHC fn 在 K1 内会做 bf16 截断).
        """
        # ── attn block: mHC pre → DSA → mHC post
        li_a, res_a, hres_a, hpost_a = self.attn_mhc.forward_pre(
            residual_flat, quantize=quantize_mhc,
        )
        attn_out = self.attn.forward(li_a, history, new_pos=new_pos,
                                     block_span=block_span)
        residual_mid = self.attn_mhc.forward_post(attn_out, res_a, hpost_a, hres_a)

        # ── mlp block: mHC pre → MoE → mHC post
        li_m, res_m, hres_m, hpost_m = self.mlp_mhc.forward_pre(
            residual_mid, quantize=quantize_mhc,
        )
        mlp_out = self.moe.forward(li_m)
        residual_out = self.mlp_mhc.forward_post(mlp_out, res_m, hpost_m, hres_m)
        return residual_mid, residual_out

    # ── Zeus forward ────────────────────────────────────────────
    def forward_zeus(self,
                     residual_flat_z: torch.Tensor,
                     paged_state: dict
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zeus block decode (全 device-resident).

        - mHC chain (K1/K2/K3/K4 × 2) 全 Zeus, fn weight 在 ``Glm5NextMhc``
          内部一次性 pack (_pack_zeus lazy cache, 不再每 forward repack).
        - DSA attn sublayer 走 paged chain; ``paged_state`` 跨 forward 复用,
          首次进入时一次性 .to("zeus") + LocalMem pack, 之后纯 device-resident.
        - MoE sublayer 与 dev_linear_attn_moe_block 同步, 全 device-resident.
        - 整段 pipeline 无 host↔device cast, 见各 sublayer dev script 的 audit.
        """
        # ── attn block (device-resident)
        z_li_a, z_res_a, z_hres_a, z_hpost_a = self.attn_mhc.forward_pre_zeus(
            residual_flat_z,
        )
        attn_out_z = self.attn.forward_zeus(z_li_a, paged_state)
        z_mid = self.attn_mhc.forward_post_zeus(
            attn_out_z, z_res_a, z_hpost_a, z_hres_a,
        )

        # ── mlp block (device-resident, z_mid 直接喂下一轮 mHC pre, 不下 host)
        z_li_m, z_res_m, z_hres_m, z_hpost_m = self.mlp_mhc.forward_pre_zeus(z_mid)
        mlp_out_z = self.moe.forward_zeus(z_li_m)
        z_out = self.mlp_mhc.forward_post_zeus(
            mlp_out_z, z_res_m, z_hpost_m, z_hres_m,
        )
        return z_mid, z_out


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = (
    # mHC chain
    "mhc_pre_norm_split", "mhc_sinkhorn", "mhc_pre_apply_mix", "mhc_post",
    # DSA attn sublayer (dual-core paged)
    "dsa_q_a_proj_norm", "dsa_kv_a_proj_norm_store",
    "dsa_q_main_absorb", "dsa_indexer_q_weights",
    "dsa_indexer_k_prep_store_dual_core",
    "dsa_index_logits_lmem_addr_table_dual_core",
    "dsa_local_topk_radix", "dsa_translate_topk_positions",
    "dsa_latent_k_gather_paged", "dsa_sparse_mqa_partial",
    "dsa_post_o_proj_no_cp",
    # MoE sublayer
    "linear_bf16_outfp32", "biased_grouped_topk",
    "moe_align_block_size_alloc", "moe_grouped_gemm",
    "silu_and_mul", "moe_sum_reduce",
)


def _run_stage(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print(f"Stage: DSA-attn + MoE block ({args.config})  seqlen={args.seqlen}")
    print("=" * 60)

    torch.manual_seed(args.seed)
    block = Glm5NextDsaAttnMoeBlock(args.config, seed=args.seed)
    cfg = block.mhc_cfg
    B = args.num_tokens
    H, N = cfg.H, cfg.N
    attn_cfg = block.attn_cfg
    print(f"  cfg: H={H} N={N} (mHC streams)  "
          f"attn[Nh={attn_cfg.Nh} Rq={attn_cfg.Rq} Rkv={attn_cfg.Rkv} "
          f"I={attn_cfg.I} Di={attn_cfg.Di} Ktop={attn_cfg.Ktop}]  "
          f"moe[E={block.moe_cfg.E} mI={block.moe_cfg.mI} top_k={block.moe_cfg.top_k}]")

    # bf16 residual, scale 0.05 与各 sublayer dev script 一致
    residual_flat = (
        torch.randn(B, N * H, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)
    history, block_span = block.init_state(B, args.seqlen, seed=args.seed + 1)

    # ── REF (quantize_mhc=True 让 mHC 参数与 Zeus 等价) ─────────
    ref_mid: Optional[torch.Tensor] = None
    ref_out: Optional[torch.Tensor] = None
    ref_skipped = False
    if args.mode in ("ref", "both"):
        ref_mid, ref_out = block.forward(
            residual_flat, history, new_pos=args.seqlen,
            block_span=block_span, quantize_mhc=True,
        )
        shape_ok = (ref_mid.shape == (B, N * H)
                    and ref_out.shape == (B, N * H)
                    and ref_out.dtype == torch.bfloat16)
        # mHC 关键不变量：N 条 residual stream 经 attn 后应被 mix —— 即 mid
        # 的 N 切片不再相等 (否则 mHC 没工作).
        mid3 = ref_mid.view(B, N, H)
        diverged = not torch.equal(mid3[:, 0, :], mid3[:, 1, :])
        print(f"  REF mid={tuple(ref_mid.shape)} out={tuple(ref_out.shape)}  "
                f"streams_diverged={diverged}")
        print(f"  REF out[0,:4] = "
                f"{[round(v,4) for v in ref_out[0,:4].float().tolist()]}")
        if not (shape_ok and diverged):
            return False

    # ── Zeus ──────────────────────────────────────────────────
    zeus_ok: Optional[bool] = None
    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"  ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
        else:
            try:
                # paged_state 从同一份 host history 构造 —— REF / Zeus 起点一致.
                # 退化几何: page_size=S_hist+1 → 每 seq 单 logical page,
                # num_physical_pages=2*B 满足 dual-core 逐核容量.
                paged_state = block.init_paged_state(
                    history, page_size=args.seqlen + 1,
                    num_physical_pages=2 * B,
                )
                # host prepare_for_decode 等价步: 算 slot_mapping + advance seq_lens
                block.prepare_decode_step(paged_state)
                z_mid, z_out = block.forward_zeus(
                    residual_flat.to("zeus"), paged_state,
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
                    # 复合误差预算 (mHC 5e-3 + DSA 5e-2 + MoE 5e-2). DSA chain
                    # 比 linear-attn 长 (11 颗算子), 且 top-K 选择对 fp8 量化误差
                    # 敏感 (排名翻转可能让下游选到不同 slot); MoE routing 同样可能
                    # 跨 topk boundary. mid 在 mlp 块之前, 误差小; out 在最终累积最大.
                    mid_ok = compare_tensors(
                        f"block.{args.config}.mid", ref_mid, z_mid_cpu,
                        atol=1.5e-1, rtol=5e-2,
                    )
                    out_ok = compare_tensors(
                        f"block.{args.config}.out", ref_out, z_out_cpu,
                        atol=5.0, rtol=2e-1,
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
        "dev_dsa_attn_moe_block",
        description="GLM5-Next DSA-transformer decoder block dev test "
                    "(mHC + DSA-attn + MoE)",
    )
    parser.add_argument("--seqlen", type=int, default=64,
                        help="DSA decode 的历史长度 (KV cache 长度, 不含 new step)")
    args = parser.parse_args()
    print_header("GLM5-Next DSA-transformer block (mHC + DSA + MoE)", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_dsa_attn_moe_block ({args.config})", ok)


if __name__ == "__main__":
    main()
