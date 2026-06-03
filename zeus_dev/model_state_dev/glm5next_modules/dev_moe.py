"""
GLM5-Next MoE sublayer 独立模块 + REF↔Zeus 对拍.

目标:
  - 把 GLM5-Next decoder layer 的 MoE sublayer 抽成一个独立模块
    ``Glm5NextMoE``，暴露 ``__init__`` + ``forward`` (REF, host bf16) +
    ``forward_zeus`` (Zeus device-resident)，方便后续直接给 full-block 使用.
  - 与 dev_glm4_moe_test.py 的 kernel 层对拍互补：那个脚本逐 kernel 对拍
    (topk / align / grouped_gemm / sum_reduce)，本脚本对拍 **整段 MoE chain
    的输入→输出**.

形状 (均为真实 size, 不使用 proxy):
  - 16b:   config_16b_v2.json  (H=2048, E=64,  mI=1408, n_shared=2, top_k=6)
           w13 ≈ 0.7 GB bf16, REF 在 CPU 上可跑.
  - next:  config.json         (H=4096, E=288, mI=2048, n_shared=1, top_k=7)
           w13 ≈ 19 GB bf16, REF 在 CPU 上不可行；mode=both 时 REF 自动 SKIP,
           Zeus-only 自检.

Chain (与 dev_glm5next_block_decode_test.zeus_moe_decode 一致):
  1. linear_bf16_outfp32  gate Linear        [T, H] → [T, E] fp32  (直接出 fp32)
  2. linear_bf16          shared gate_up     [T, H] → [T, 2*sI]
  3. silu_and_mul         shared             [T, 2*sI] → [T, sI]
  4. linear_bf16          shared down        [T, sI] → [T, H]
  5. biased_grouped_topk  router 选 expert
  6. moe_align_block_size_alloc
  7. moe_grouped_gemm     gemm1 (routed)     [T, H] → [T*top_k, 2*mI]
  8. silu_and_mul         per-expert         [T*top_k, 2*mI] → [T*top_k, mI]
  9. moe_grouped_gemm     gemm2 (mul_routed_weight=True)
  10. moe_sum_reduce      +shared residual fuse → [T, H]

用法:
  python glm5next_modules/dev_moe.py                    # 16b / both
  python glm5next_modules/dev_moe.py --config next      # next / both (REF SKIP)
  python glm5next_modules/dev_moe.py --mode zeus
  python glm5next_modules/dev_moe.py --triton_forward   # REF / Zeus-simC / Zeus-Triton 三路对拍

三路对拍 (``--triton_forward``):
  - forward             REF host bf16 MoE chain                         (golden A)
  - forward_zeus        Zeus C++ sim-C MoE chain (sgl_kernel_zeus)       (golden B)
  - forward_zeus_triton Zeus Triton-JIT MoE chain (sgl_kernel_zeus_triton)
  三者使用同一份权重；next 配置 REF 内存过大时三路对拍自动 SKIP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

# 公共脚手架：sys.path bootstrap、Zeus runtime、config / compare 等
import _common
from _common import (
    ZEUS_IMPORT_ERROR, sgl_kernel_zeus,
    config_path, compare_tensors, zeus_chain_available,
    make_argparser, print_header, print_summary,
    REF_MEMORY_BUDGET_GB,
)

# REF MoE core 复用 (上一级 dev_glm4_moe_test)
import dev_glm4_moe_test as moe_dev


# ── sgl-kernel-zeus-triton (sibling Triton-JIT package) ─────────
# Editable-installed alongside sgl_kernel_zeus (`pip install -e
# sgl-kernel-zeus-triton`), so it imports directly — no sys.path bootstrap.
# Guarded so the basic REF/Zeus test still runs when it isn't installed.
try:
    import sgl_kernel_zeus_triton
except Exception:  # pragma: no cover
    sgl_kernel_zeus_triton = None


# ── Config ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Glm5NextMoEConfig:
    """MoE sublayer config (单 device, TP=1).

    ``sI = n_shared_experts * moe_intermediate_size`` —— shared experts 在本
    脚本范畴下被坍缩成一个等价大 MLP（与 prerelease/glm5_next.py 中 shared
    experts 通过 hidden_size 复制 + n_shared_experts 倍 inter 的语义一致；
    kernel 层无 per-shared-expert 区分）.
    """
    H: int
    E: int
    mI: int
    sI: int
    top_k: int
    num_expert_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    name: str = "proxy"

    @classmethod
    def from_json(cls, path: Path, name: Optional[str] = None) -> "Glm5NextMoEConfig":
        import json
        raw = json.loads(Path(path).read_text())
        mI = int(raw["moe_intermediate_size"])
        n_shared = int(raw.get("n_shared_experts", 1))
        return cls(
            H=int(raw["hidden_size"]),
            E=int(raw["n_routed_experts"]),
            mI=mI,
            sI=n_shared * mI,
            top_k=int(raw["num_experts_per_tok"]),
            num_expert_group=1,
            topk_group=int(raw.get("topk_group", 1)),
            routed_scaling_factor=float(raw.get("routed_scaling_factor", 1.0)),
            name=name or Path(path).stem,
        )


def load_cfg(which: str) -> Glm5NextMoEConfig:
    return Glm5NextMoEConfig.from_json(config_path(which), name=which)


def estimate_ref_memory_gb(cfg: Glm5NextMoEConfig) -> float:
    """REF 路径峰值显存估算 (GB).

    ``_ref_moe_core`` 内部把 w13 / w2 cast 到 fp32 (×2), 再加 bf16 原副本 (×1)，
    总占用 ≈ (w13 + w2) * 6 字节/entry.
    """
    elements = cfg.E * (2 * cfg.mI + cfg.mI) * cfg.H  # w13 + w2 entries
    return elements * 3 * 2 / (1024 ** 3)


# ── Module ──────────────────────────────────────────────────────
class Glm5NextMoE:
    """GLM5-Next MoE sublayer（router + routed experts + shared experts 整段）.

    使用模式::

        moe = Glm5NextMoE(cfg, seed=42)
        out = moe.forward(hidden)              # REF (host bf16)
        out_z = moe.forward_zeus(hidden_z)     # Zeus device-resident

    REF / Zeus 两条路径共享同一份权重 (Zeus LocalMem pack 首次调用时 lazy 建立
    并缓存，等价生产 layer.__init__ 一次性 pack).
    """

    def __init__(self, cfg: Glm5NextMoEConfig, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)

        def rn(*shape, scale: float = 0.1) -> torch.Tensor:
            return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
                torch.bfloat16
            )

        H, E, mI, sI = cfg.H, cfg.E, cfg.mI, cfg.sI
        self.gate_w = rn(E, H)                               # router gate
        self.corr_bias = torch.randn(E, generator=g, dtype=torch.float32) * 0.01
        self.w13 = rn(E, 2 * mI, H)                          # routed experts gate_up
        self.w2 = rn(E, H, mI)                               # routed experts down
        self.sh_gu = rn(2 * sI, H)                           # shared experts gate_up
        self.sh_dp = rn(H, sI)                               # shared experts down

        # Zeus device-resident state — lazy
        self._zeus_packed = False
        self._gate_w_lmem = None
        self._sh_gu_lmem = None
        self._sh_dp_lmem = None
        self._corr_bias_z = None
        self._w13_z = None
        self._w2_z = None
        self._triton_weights_packed = False
        self._gate_w_triton_z = None
        self._sh_gu_triton_z = None
        self._sh_dp_triton_z = None
        self._w13_triton_z = None
        self._w2_triton_z = None
        # Per-T scratch buffer cache (lazy alloc per batch-size, reused across
        # forwards with the same T). 5 个中间 buffer 都仅由 T 决定 (top_k / sI /
        # mI / H 都来自 config), 因此可以池化避免每次 forward 重新 alloc.
        self._scratch_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    # 兼容旧 callers (dev_glm5next_block_decode_test dict-style 用法)
    def as_weight_dict(self) -> Dict:
        return {
            "gate_w": self.gate_w,
            "corr_bias": self.corr_bias,
            "w13": self.w13,
            "w2": self.w2,
            "sh_gu": self.sh_gu,
            "sh_dp": self.sh_dp,
            "_meta": {"E": self.cfg.E, "mI": self.cfg.mI,
                      "top_k": self.cfg.top_k, "sI": self.cfg.sI},
        }

    # ── REF forward ─────────────────────────────────────────────
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """REF MoE: gate + shared + biased_grouped_topk + per-token expert loop + residual.

        ``hidden: [T, H] bf16 (host)  ->  [T, H] bf16``
        """
        from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

        cfg = self.cfg
        sI = cfg.sI

        # gate Linear (fp32 logits)
        router_logits = torch.nn.functional.linear(
            hidden.float(), self.gate_w.float(),
        )

        # shared experts MLP
        sh_gu = torch.nn.functional.linear(hidden, self.sh_gu)
        sh_silu = (
            torch.nn.functional.silu(sh_gu[:, :sI].float()) * sh_gu[:, sI:].float()
        ).to(torch.bfloat16)
        shared_out = torch.nn.functional.linear(sh_silu, self.sh_dp)

        w_topk, ids_topk = biased_grouped_topk_impl(
            hidden_states=hidden,
            gating_output=router_logits,
            correction_bias=self.corr_bias,
            topk=cfg.top_k,
            renormalize=True,
            num_expert_group=cfg.num_expert_group,
            topk_group=cfg.topk_group,
            num_fused_shared_experts=0,
            routed_scaling_factor=cfg.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=True,
        )
        moe_core = moe_dev._ref_moe_core(
            hidden, self.w13, self.w2, w_topk, ids_topk, mI=cfg.mI,
        )
        return (moe_core.float() + shared_out.float()).to(torch.bfloat16)

    # ── Zeus pack (lazy) ────────────────────────────────────────
    def _pack_zeus(self) -> None:
        if ZEUS_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"Zeus runtime unavailable: {ZEUS_IMPORT_ERROR}"
            )
        self._gate_w_lmem = torch.zeus.local_memory.from_tensor(
            self.gate_w.to("zeus"), kind="weight", Tr=1, Tc=1,
        )
        self._sh_gu_lmem = torch.zeus.local_memory.from_tensor(
            self.sh_gu.to("zeus"), kind="weight", Tr=1, Tc=1,
        )
        self._sh_dp_lmem = torch.zeus.local_memory.from_tensor(
            self.sh_dp.to("zeus"), kind="weight", Tr=1, Tc=1,
        )
        self._corr_bias_z = self.corr_bias.to("zeus")
        # w13 / w2 走 LocalMem multi-matrix (num_matrices=E)：每个 expert 是
        # 一个独立 2D tiled slice；moe_grouped_gemm sim 端按
        # `expert*slice_bytes + lm_weight_offset_bytes(n, k, ...)` 寻址.
        self._w13_z = torch.zeus.local_memory.from_tensor(
            self.w13.to("zeus"), kind="weight", Tr=1, Tc=1,
        )
        self._w2_z = torch.zeus.local_memory.from_tensor(
            self.w2.to("zeus"), kind="weight", Tr=1, Tc=1,
        )
        self._zeus_packed = True

    def _pack_zeus_triton_weights(self) -> None:
        """Plain Zeus contiguous weights consumed by Triton-JIT wrappers.

        Lmem initialization for these tensors is owned by the
        ``sgl_kernel_zeus_triton`` Python wrappers and cached there.
        """
        if ZEUS_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"Zeus runtime unavailable: {ZEUS_IMPORT_ERROR}"
            )
        if self._gate_w_triton_z is None:
            self._gate_w_triton_z = self.gate_w.to("zeus").contiguous()
        if self._sh_gu_triton_z is None:
            self._sh_gu_triton_z = self.sh_gu.to("zeus").contiguous()
        if self._sh_dp_triton_z is None:
            self._sh_dp_triton_z = self.sh_dp.to("zeus").contiguous()
        if self._w13_triton_z is None:
            self._w13_triton_z = self.w13.to("zeus").contiguous()
        if self._w2_triton_z is None:
            self._w2_triton_z = self.w2.to("zeus").contiguous()
        if self._corr_bias_z is None:
            self._corr_bias_z = self.corr_bias.to("zeus")
        self._triton_weights_packed = True

    def _get_scratch(self, T: int) -> Dict[str, torch.Tensor]:
        """Per-T scratch buffer pool. Allocates once per batch-size T, reuses on
        subsequent forwards. 生产路径下 layer 持有的 buffer 也是按 max-T 预分配,
        这里 lazy-by-T 是 dev 脚本的简化版."""
        s = self._scratch_cache.get(T)
        if s is None:
            cfg = self.cfg
            top_k = cfg.top_k
            num_valid_tokens = T * top_k
            s = {
                "router_logits": torch.empty(T, cfg.E, dtype=torch.float32, device="zeus"),
                "sh_gu":    torch.empty(T, 2 * cfg.sI,
                                        dtype=torch.bfloat16, device="zeus"),
                "sh_silu": torch.empty(T, cfg.sI, dtype=torch.bfloat16, device="zeus"),
                "shared_out": torch.empty(T, cfg.H,
                                          dtype=torch.bfloat16, device="zeus"),
                "C1":      torch.empty(num_valid_tokens, 2 * cfg.mI,
                                       dtype=torch.bfloat16, device="zeus"),
                "C1_silu": torch.empty(num_valid_tokens, cfg.mI,
                                       dtype=torch.bfloat16, device="zeus"),
                "C2":      torch.empty(num_valid_tokens, cfg.H,
                                       dtype=torch.bfloat16, device="zeus"),
                "final":   torch.empty(T, cfg.H, dtype=torch.bfloat16, device="zeus"),
            }
            self._scratch_cache[T] = s
        return s

    # ── Zeus forward ────────────────────────────────────────────
    def forward_zeus(self, hidden_z: torch.Tensor) -> torch.Tensor:
        """Zeus MoE chain (10 颗算子), 与 dev_glm5next_block_decode_test 同语义.

        ``hidden_z: [T, H] bf16 (zeus)  ->  [T, H] bf16 (zeus)``
        """
        if not self._zeus_packed:
            self._pack_zeus()

        cfg = self.cfg
        T = hidden_z.shape[0]
        H = hidden_z.shape[-1]
        E, mI, top_k = cfg.E, cfg.mI, cfg.top_k
        block_size = sgl_kernel_zeus.MOE_GROUPED_GEMM_BLOCK_M
        num_valid_tokens = T * top_k
        scratch = self._get_scratch(T)

        # gate Linear (fp32 logits, device-resident — linear_bf16_outfp32
        # 直接出 fp32，无 host `.float()` cast round-trip)
        router_logits_z = sgl_kernel_zeus.linear_bf16_outfp32(
            hidden_z, self._gate_w_lmem,
        )

        # shared experts MLP
        sh_gu_z = sgl_kernel_zeus.linear_bf16(hidden_z, self._sh_gu_lmem)
        sh_silu_z = scratch["sh_silu"]
        sgl_kernel_zeus.silu_and_mul(sh_gu_z, sh_silu_z)
        shared_out_z = sgl_kernel_zeus.linear_bf16(sh_silu_z, self._sh_dp_lmem)

        # router topk
        w_z, ids_z = sgl_kernel_zeus.biased_grouped_topk(
            router_logits_z, self._corr_bias_z,
            num_expert_group=cfg.num_expert_group, topk_group=cfg.topk_group,
            topk=top_k,
            num_fused_shared_experts=0,
            routed_scaling_factor=cfg.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=True,
        )
        sorted_ids_z, expert_ids_z, num_post_z = (
            sgl_kernel_zeus.moe_align_block_size_alloc(ids_z, block_size, E)
        )

        # gemm1
        C1_z = scratch["C1"]
        sgl_kernel_zeus.moe_grouped_gemm(
            hidden_z, self._w13_z, C1_z,
            sorted_ids_z, expert_ids_z, num_post_z,
            num_valid_tokens=num_valid_tokens, top_k=top_k,
        )
        # silu_and_mul
        C1_silu_z = scratch["C1_silu"]
        sgl_kernel_zeus.silu_and_mul(C1_z, C1_silu_z)

        # gemm2 (mul_routed_weight=True) — biased_grouped_topk 已直出 bf16,
        # 此处仅 flatten (metadata-only view, 零 device 操作).
        w_z_flat_bf16 = w_z.flatten()
        C2_z = scratch["C2"]
        sgl_kernel_zeus.moe_grouped_gemm(
            C1_silu_z, self._w2_z, C2_z,
            sorted_ids_z, expert_ids_z, num_post_z,
            num_valid_tokens=num_valid_tokens, top_k=1,
            topk_weights=w_z_flat_bf16, mul_routed_weight=True,
        )
        # sum_reduce + shared residual fuse
        final_z = scratch["final"]
        sgl_kernel_zeus.moe_sum_reduce(
            input=C2_z.view(T, top_k, H),
            output=final_z,
            shared_output=shared_out_z,
            routed_scaling_factor=cfg.routed_scaling_factor,
        )
        return final_z

    # ── Zeus Triton-JIT forward ─────────────────────────────────
    def forward_zeus_triton(self, hidden_z: torch.Tensor) -> torch.Tensor:
        """Zeus MoE chain via the Triton-JIT kernels — SAME weights as
        :meth:`forward_zeus`.

        ``hidden_z: [T, H] bf16 (zeus)  ->  [T, H] bf16 (zeus)``
        """
        if sgl_kernel_zeus_triton is None:
            raise RuntimeError("sgl_kernel_zeus_triton is not installed")
        if not self._triton_weights_packed:
            self._pack_zeus_triton_weights()

        cfg = self.cfg
        T = hidden_z.shape[0]
        H = hidden_z.shape[-1]
        E, top_k = cfg.E, cfg.top_k
        block_size = sgl_kernel_zeus.MOE_GROUPED_GEMM_BLOCK_M
        num_valid_tokens = T * top_k
        scratch = self._get_scratch(T)

        router_logits_z = scratch["router_logits"]
        sgl_kernel_zeus_triton.linear_bf16_outfp32(
            hidden_z,
            self._gate_w_triton_z,
            out=router_logits_z,
        )

        sh_gu_z = scratch["sh_gu"]
        sgl_kernel_zeus_triton.linear_bf16(
            hidden_z,
            self._sh_gu_triton_z,
            out=sh_gu_z,
        )
        sh_silu_z = scratch["sh_silu"]
        sgl_kernel_zeus_triton.silu_and_mul(sh_gu_z, out=sh_silu_z)
        shared_out_z = scratch["shared_out"]
        sgl_kernel_zeus_triton.linear_bf16(
            sh_silu_z,
            self._sh_dp_triton_z,
            out=shared_out_z,
        )

        w_z, ids_z = sgl_kernel_zeus_triton.biased_grouped_topk(
            router_logits_z,
            self._corr_bias_z,
            num_expert_group=cfg.num_expert_group,
            topk_group=cfg.topk_group,
            topk=top_k,
            num_fused_shared_experts=0,
            routed_scaling_factor=cfg.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=True,
        )
        sorted_ids_z, expert_ids_z, num_post_z = sgl_kernel_zeus_triton.moe_align_block_size_alloc(
            ids_z,
            block_size,
            E,
        )

        C1_z = scratch["C1"]
        sgl_kernel_zeus_triton.moe_grouped_gemm(
            hidden_z,
            self._w13_triton_z,
            sorted_ids_z,
            expert_ids_z,
            num_post_z,
            num_valid_tokens=num_valid_tokens,
            top_k=top_k,
            out=C1_z,
        )

        C1_silu_z = scratch["C1_silu"]
        sgl_kernel_zeus_triton.silu_and_mul(C1_z, out=C1_silu_z)

        C2_z = scratch["C2"]
        sgl_kernel_zeus_triton.moe_grouped_gemm(
            C1_silu_z,
            self._w2_triton_z,
            sorted_ids_z,
            expert_ids_z,
            num_post_z,
            topk_weights=w_z.flatten(),
            num_valid_tokens=num_valid_tokens,
            top_k=1,
            mul_routed_weight=True,
            out=C2_z,
        )

        final_z = scratch["final"]
        sgl_kernel_zeus_triton.moe_sum_reduce(
            C2_z.view(T, top_k, H),
            output=final_z,
            shared_output=shared_out_z,
            routed_scaling_factor=cfg.routed_scaling_factor,
        )
        return final_z


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = (
    "linear_bf16", "linear_bf16_outfp32",
    "biased_grouped_topk", "moe_align_block_size_alloc",
    "moe_grouped_gemm", "silu_and_mul", "moe_sum_reduce",
)


def _run_triton_three_way(moe, hidden, cfg, args) -> bool:
    """REF / Zeus-simC / Zeus-Triton 三路对拍 (full MoE chain).

    三路:
      - REF          forward             host bf16 MoE chain
      - Zeus-simC    forward_zeus        C++ sim-C kernels
      - Zeus-Triton  forward_zeus_triton Triton-JIT kernels
    """
    hidden_z = hidden.to("zeus")
    ref = moe.forward(hidden)                    # golden A (host)
    simc = moe.forward_zeus(hidden_z).cpu()      # golden B (C++ sim-C)
    tri = moe.forward_zeus_triton(hidden_z).cpu()  # under test (Triton)

    print(f"  ZEUS-TRITON out shape={tuple(tri.shape)} dtype={tri.dtype}")
    print(f"  ZEUS-TRITON out[0,:4] = "
          f"{[round(v,4) for v in tri[0,:4].float().tolist()]}")
    finite_t = torch.isfinite(tri).all().item()
    shape_ok_t = (tri.shape == (args.num_tokens, cfg.H)
                  and tri.dtype == torch.bfloat16)
    print(f"  ZEUS-TRITON finite={finite_t} shape_ok={shape_ok_t}")

    c_tri_ref = compare_tensors(
        f"moe.{args.config}.triton-vs-REF", ref, tri, atol=5e-2, rtol=5e-2)
    c_tri_simc = compare_tensors(
        f"moe.{args.config}.triton-vs-simC", simc, tri, atol=5e-2, rtol=5e-2)
    # golden 自洽: REF vs simC 也应一致
    c_ref_simc = compare_tensors(
        f"moe.{args.config}.REF-vs-simC", ref, simc, atol=5e-2, rtol=5e-2)

    return bool(finite_t and shape_ok_t and c_tri_ref and c_tri_simc and c_ref_simc)


def _run_stage(args) -> Optional[bool]:
    """单 stage：从 config 加载真实 shape, 按 mode 跑 REF / Zeus / 对拍."""
    print("\n" + "=" * 60)
    print(f"Stage: {args.config} (real shape)")
    print("=" * 60)
    cfg = load_cfg(args.config)
    ref_mem = estimate_ref_memory_gb(cfg)
    print(f"  cfg: H={cfg.H} E={cfg.E} mI={cfg.mI} sI={cfg.sI} top_k={cfg.top_k}  "
          f"groups={cfg.num_expert_group}/{cfg.topk_group}  "
          f"scale={cfg.routed_scaling_factor}")
    print(f"  REF memory estimate: {ref_mem:.2f} GB "
          f"(budget {REF_MEMORY_BUDGET_GB:.1f} GB)")

    torch.manual_seed(args.seed)
    moe = Glm5NextMoE(cfg, seed=args.seed)
    hidden = (torch.randn(args.num_tokens, cfg.H, dtype=torch.float32) * 0.05).to(
        torch.bfloat16
    )

    # ── REF ───────────────────────────────────────────────────
    ref_out: Optional[torch.Tensor] = None
    ref_skipped = False
    if args.mode in ("ref", "both"):
        if ref_mem > REF_MEMORY_BUDGET_GB:
            print(f"  REF: SKIP ({ref_mem:.1f} GB > budget {REF_MEMORY_BUDGET_GB:.1f} GB; "
                  f"kernel 层对拍见 dev_glm4_moe_test.py)")
            ref_skipped = True
        else:
            ref_out = moe.forward(hidden)
            ok = ref_out.shape == (args.num_tokens, cfg.H) and ref_out.dtype == torch.bfloat16
            print(f"  REF out shape={tuple(ref_out.shape)} dtype={ref_out.dtype}")
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
                z_out = moe.forward_zeus(hidden.to("zeus"))
                z_out_cpu = z_out.cpu()
                print(f"  ZEUS out shape={tuple(z_out_cpu.shape)} dtype={z_out_cpu.dtype}")
                print(f"  ZEUS out[0,:4] = "
                      f"{[round(v,4) for v in z_out_cpu[0,:4].float().tolist()]}")
                finite = torch.isfinite(z_out_cpu).all().item()
                print(f"  ZEUS finite={finite}")
                shape_ok = (z_out_cpu.shape == (args.num_tokens, cfg.H)
                            and z_out_cpu.dtype == torch.bfloat16)
                if ref_out is not None:
                    zeus_ok = compare_tensors(
                        f"moe.{args.config}.out", ref_out, z_out_cpu,
                        atol=5e-2, rtol=5e-2,
                    ) and finite and shape_ok
                else:
                    zeus_ok = finite and shape_ok
            except Exception as e:
                import traceback
                print(f"  ZEUS EXCEPTION: {e!r}")
                traceback.print_exc()
                zeus_ok = False

    # ── Zeus Triton-JIT (三路对拍) ────────────────────────────
    # 仅当 --triton_forward 时跑第三路, 与 REF (golden A) / Zeus-simC (golden B)
    # 做三路比较.
    triton_ok: Optional[bool] = None
    if getattr(args, "triton_forward", False):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"  ZEUS-TRITON: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
        elif sgl_kernel_zeus_triton is None:
            print("  ZEUS-TRITON: SKIP (sgl_kernel_zeus_triton not installed: "
                  "`pip install -e sgl-kernel-zeus-triton`)")
        elif ref_mem > REF_MEMORY_BUDGET_GB:
            print(f"  ZEUS-TRITON: SKIP three-way ({ref_mem:.1f} GB > budget "
                  f"{REF_MEMORY_BUDGET_GB:.1f} GB; REF unavailable)")
        else:
            try:
                triton_ok = _run_triton_three_way(moe, hidden, cfg, args)
            except Exception as e:
                import traceback
                print(f"  ZEUS-TRITON EXCEPTION: {e!r}")
                traceback.print_exc()
                triton_ok = False

    # ── status 汇总 ──────────────────────────────────────────
    def _fold(base: Optional[bool]) -> Optional[bool]:
        """把三路对拍结果叠加进 base 状态: base False 恒 False, 否则取 triton_ok."""
        if triton_ok is None:
            return base
        if base is False:
            return False
        return triton_ok

    if args.mode == "ref":
        return _fold(None if ref_skipped else True)
    if args.mode == "zeus":
        return _fold(zeus_ok)
    # both
    if ref_skipped:
        return _fold(None if zeus_ok is None else zeus_ok)
    if zeus_ok is None:
        return _fold(True)
    return _fold(zeus_ok)


def main():
    parser = make_argparser("dev_moe", description="GLM5-Next MoE sublayer dev test")
    parser.add_argument(
        "--triton_forward",
        action="store_true",
        help="额外跑 forward_zeus_triton 并做 REF / Zeus-simC / Zeus-Triton 三路对拍",
    )
    args = parser.parse_args()
    print_header("GLM5-Next MoE sublayer", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_moe ({args.config})", ok)


if __name__ == "__main__":
    main()
