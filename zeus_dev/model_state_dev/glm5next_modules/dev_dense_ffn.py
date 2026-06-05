"""
GLM5-Next Dense FFN sublayer 独立模块 + REF↔Zeus 对拍.

目标:
  - 把 GLM5-Next dense 层 (前 first_k_dense_replace 层) 的 SwiGLU MLP 抽成
    独立模块 ``Glm5NextDenseFFN``,与 ``Glm5NextMoE`` 同接口 (forward /
    forward_zeus / _pack_zeus),便于在 block 里与 MoE 一行互换.
  - 维度用真 dense 层 ``intermediate_size`` (glm5_next.py:515
    ``Glm5NextMLP(intermediate_size=config.intermediate_size)``),不是 MoE
    shared experts 的 ``sI``.  算子链 (linear→silu_and_mul→linear) 照搬
    dev_moe 的 shared-expert path.

形状 (真实 size):
  - 16b:  config_16b_v2.json  H=2048, I=10944
  - next: config.json         H=4096, I=12288

Chain:
  1. linear_bf16   gate_up   [T, H] → [T, 2*I]
  2. silu_and_mul            [T, 2*I] → [T, I]
  3. linear_bf16   down      [T, I] → [T, H]

用法:
  python glm5next_modules/dev_dense_ffn.py                 # 16b / both
  python glm5next_modules/dev_dense_ffn.py --config next   # next / both
  python glm5next_modules/dev_dense_ffn.py --mode zeus
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

import _common
from _common import (
    ZEUS_IMPORT_ERROR, sgl_kernel_zeus,
    config_path, compare_tensors, zeus_chain_available,
    make_argparser, print_header, print_summary,
)


@dataclass(frozen=True)
class Glm5NextDenseFFNConfig:
    """Dense FFN sublayer config (单 device, TP=1).

    ``I = intermediate_size`` —— 真 dense 层维度 (glm5_next.py:515),区别于
    MoE shared experts 的 ``sI = n_shared * moe_intermediate_size``.
    """
    H: int
    I: int
    name: str = "proxy"

    @classmethod
    def from_json(cls, path: Path, name: Optional[str] = None) -> "Glm5NextDenseFFNConfig":
        import json
        raw = json.loads(Path(path).read_text())
        return cls(
            H=int(raw["hidden_size"]),
            I=int(raw["intermediate_size"]),
            name=name or Path(path).stem,
        )


def load_cfg(which: str) -> Glm5NextDenseFFNConfig:
    return Glm5NextDenseFFNConfig.from_json(config_path(which), name=which)


class Glm5NextDenseFFN:
    """GLM5-Next dense-layer MLP (SwiGLU: gate_up → silu_and_mul → down).

    与 ``Glm5NextMoE`` 同接口,故在 block 里可与 MoE 一行互换.
    """

    def __init__(self, cfg: Glm5NextDenseFFNConfig, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)

        def rn(*shape, scale: float = 0.1) -> torch.Tensor:
            return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
                torch.bfloat16
            )

        H, I = cfg.H, cfg.I
        self.gate_up = rn(2 * I, H)      # fused SwiGLU gate_up
        self.down = rn(H, I)             # down proj

        self._zeus_packed = False
        self._gate_up_lmem = None
        self._down_lmem = None
        self._scratch_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    # ── REF forward ─────────────────────────────────────────────
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """REF dense SwiGLU.  ``hidden: [T, H] bf16 (host) -> [T, H] bf16``."""
        I = self.cfg.I
        gu = torch.nn.functional.linear(hidden, self.gate_up)
        silu = (
            torch.nn.functional.silu(gu[:, :I].float()) * gu[:, I:].float()
        ).to(torch.bfloat16)
        return torch.nn.functional.linear(silu, self.down)

    # ── Zeus pack (lazy) ────────────────────────────────────────
    def _pack_zeus(self) -> None:
        """LocalMem-pack via 各 op 自带 ``.pack`` (与 dev_moe shared path 同范式)."""
        if ZEUS_IMPORT_ERROR is not None:
            raise RuntimeError(f"Zeus runtime unavailable: {ZEUS_IMPORT_ERROR}")
        self._gate_up_lmem = sgl_kernel_zeus.linear_bf16.pack(self.gate_up)
        self._down_lmem = sgl_kernel_zeus.linear_bf16.pack(self.down)
        self._zeus_packed = True

    def _get_scratch(self, T: int) -> Dict[str, torch.Tensor]:
        """Per-T scratch buffer pool. 按 batch-size T 首次出现时 lazy alloc 一次,
        之后复用 (与 dev_moe._get_scratch 同范式, buffer-sink)."""
        s = self._scratch_cache.get(T)
        if s is None:
            s = {
                "silu": torch.empty(T, self.cfg.I, dtype=torch.bfloat16, device="zeus"),
            }
            self._scratch_cache[T] = s
        return s

    # ── Zeus forward ────────────────────────────────────────────
    def forward_zeus(self, hidden_z: torch.Tensor) -> torch.Tensor:
        """Zeus dense SwiGLU (3 算子).  ``[T, H] bf16 (zeus) -> [T, H] bf16``."""
        if not self._zeus_packed:
            self._pack_zeus()
        scratch = self._get_scratch(hidden_z.shape[0])
        gu_z = sgl_kernel_zeus.linear_bf16(hidden_z, self._gate_up_lmem)
        silu_z = scratch["silu"]
        sgl_kernel_zeus.silu_and_mul(gu_z, silu_z)
        return sgl_kernel_zeus.linear_bf16(silu_z, self._down_lmem)


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = ("linear_bf16", "silu_and_mul")


def _run_stage(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print(f"Stage: dense-ffn ({args.config})")
    print("=" * 60)
    cfg = load_cfg(args.config)
    print(f"  cfg: H={cfg.H} I={cfg.I}")

    torch.manual_seed(args.seed)
    ffn = Glm5NextDenseFFN(cfg, seed=args.seed)
    hidden = (torch.randn(args.num_tokens, cfg.H, dtype=torch.float32) * 0.05).to(
        torch.bfloat16
    )

    ref_out: Optional[torch.Tensor] = None
    if args.mode in ("ref", "both"):
        ref_out = ffn.forward(hidden)
        ok = ref_out.shape == (args.num_tokens, cfg.H) and ref_out.dtype == torch.bfloat16
        print(f"  REF out shape={tuple(ref_out.shape)} dtype={ref_out.dtype}")
        print(f"  REF out[0,:4] = "
              f"{[round(v,4) for v in ref_out[0,:4].float().tolist()]}")
        if not ok:
            return False

    zeus_ok: Optional[bool] = None
    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"  ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
        else:
            try:
                z_out = ffn.forward_zeus(hidden.to("zeus"))
                z_out_cpu = z_out.cpu()
                print(f"  ZEUS out shape={tuple(z_out_cpu.shape)} dtype={z_out_cpu.dtype}")
                print(f"  ZEUS out[0,:4] = "
                      f"{[round(v,4) for v in z_out_cpu[0,:4].float().tolist()]}")
                finite = torch.isfinite(z_out_cpu).all().item()
                print(f"  ZEUS finite={finite}")
                shape_ok = (z_out_cpu.shape == (args.num_tokens, cfg.H)
                            and z_out_cpu.dtype == torch.bfloat16)
                if ref_out is not None:
                    # next config (H=4096, I=12288) 的 bf16 累积误差略大 (max_diff≈1.56e-2),
                    # 宽松至 atol/rtol=2e-2 (刚好覆盖实测值,不过度放宽).
                    tol = 2e-2 if args.config == "next" else 5e-3
                    zeus_ok = compare_tensors(
                        f"dense_ffn.{args.config}.out", ref_out, z_out_cpu,
                        atol=tol, rtol=tol,
                    ) and finite and shape_ok
                else:
                    zeus_ok = finite and shape_ok
            except Exception as e:
                import traceback
                print(f"  ZEUS EXCEPTION: {e!r}")
                traceback.print_exc()
                zeus_ok = False

    if args.mode == "ref":
        return True
    if args.mode == "zeus":
        return zeus_ok
    if zeus_ok is None:
        return True
    return zeus_ok


def main():
    parser = make_argparser("dev_dense_ffn",
                            description="GLM5-Next dense FFN sublayer dev test")
    args = parser.parse_args()
    print_header("GLM5-Next dense FFN sublayer", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_dense_ffn ({args.config})", ok)


if __name__ == "__main__":
    main()
