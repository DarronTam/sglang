"""
GLM5-Next 尾巴 (mHC contract + final RMSNorm + lm_head) 独立模块 + REF↔Zeus 对拍.

把 ``Glm5NextModel`` 出口完整 MHC-aware 尾巴抽成一个独立 module ``Glm5NextLmHead``:

  last block 的 residual_out  [T, N*H] bf16
        │
        ├─ ① hc_contract:  reshape→mean over N streams   → [T, H] bf16
        │       (sgl_kernel_zeus.mhc_contract, prerelease 等价 hc_contract)
        │
        ├─ ② RMSNorm(H):  standalone (residual=None 分支; MHC 模式下 model 出口
        │                 的 residual 永远是 None, 参 communicator_mhc.py
        │                 postprocess_layer 在 is_last_layer=True 时返回
        │                 (hidden[T,H], None))
        │
        └─ ③ ParallelLMHead:  linear_bf16_outfp32 → [T, V] fp32 logits

形状 (从 config json 读真实 vocab/hidden/N):
  - 16b:   V=154880, H=2048, N=4   → lm_head_w ≈ 0.61 GB bf16, residual in [T, 8192]
  - next:  V=154880, H=4096, N=4   → lm_head_w ≈ 1.21 GB bf16, residual in [T, 16384]

参考代码:
  - reference glm5_next.py:921-925  (model 出口 norm, MHC 模式 residual=None 分支)
  - communicator_mhc.py:218-222     (last layer postprocess: mlp_combine + contract_output)
  - mhc/functional.py:270-272       (hc_contract = x.unflatten(-1,(n,-1)).mean(-2))
  - reference glm5_next.py:1054-1057 (logits_processor: decode mode 下 last-token
    slicing 是 identity, 这里直接产 logits, 不包 wrapper)

不做的事:
  - 非 MHC 路径 (GLM5-Next 配置永远 mhc=true, 不支持 mhc=false)
  - LogitsProcessor 完整 wrapper (penalty / log_softmax 留给 dev_sampler.py)
  - tie_word_embeddings 共享 embed_w (config 显式 tie_word_embeddings=false)
  - TP / DP head 切分

用法:
  python glm5next_modules/dev_lm_head.py                       # 16b / both
  python glm5next_modules/dev_lm_head.py --config next         # next / both
  python glm5next_modules/dev_lm_head.py --mode zeus
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# 公共脚手架
import _common
from _common import (
    ZEUS_IMPORT_ERROR, sgl_kernel_zeus,
    config_path, compare_tensors, zeus_chain_available,
    make_argparser, print_header, print_summary,
    REF_MEMORY_BUDGET_GB,
)


# ── Config ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Glm5NextLmHeadConfig:
    """Final tail (mHC contract + norm + lm_head) config (单 device, TP=1)."""
    V: int
    H: int
    N: int = 4
    eps: float = 1e-5
    name: str = "proxy"

    @classmethod
    def from_json(cls, path: Path, name: Optional[str] = None) -> "Glm5NextLmHeadConfig":
        import json
        raw = json.loads(Path(path).read_text())
        return cls(
            V=int(raw["vocab_size"]),
            H=int(raw["hidden_size"]),
            N=int(raw.get("mhc_num_residual_streams", 4)),
            eps=float(raw.get("rms_norm_eps", 1e-5)),
            name=name or Path(path).stem,
        )


def load_cfg(which: str) -> Glm5NextLmHeadConfig:
    return Glm5NextLmHeadConfig.from_json(config_path(which), name=which)


def estimate_ref_memory_gb(cfg: Glm5NextLmHeadConfig) -> float:
    """REF 权重显存估算 (GB).

    主体是 ``lm_head_w[V, H] bf16``; 再保留一份给 Zeus pack, 总占用 ~2x.
    norm_w[H] 与 hidden buffer 相对忽略.
    """
    return cfg.V * cfg.H * 2 * 2 / (1024 ** 3)


# ── REF math helper ─────────────────────────────────────────────
def _ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """REF RMSNorm: bf16 in, bf16 out, fp32 中间累加."""
    x32 = x.float()
    rms = x32.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x32 / rms * w.float()).to(torch.bfloat16)


def _ref_hc_contract(residual: torch.Tensor, N: int) -> torch.Tensor:
    """REF hc_contract: [T, N*H] → [T, H] by averaging N streams.

    与 prerelease ``mhc/functional.py::hc_contract`` 完全一致:
        residual.unflatten(-1, (N, -1)).mean(dim=-2)
    bf16 mean 内部 fp32 accumulate / ÷N / cast bf16, 与 sgl_kernel_zeus.mhc_contract
    同精度模型, 应当 bit-exact.
    """
    return residual.unflatten(-1, (N, -1)).mean(dim=-2)


# ── Module ──────────────────────────────────────────────────────
class Glm5NextLmHead:
    """GLM5-Next 尾巴: mHC contract + final RMSNorm + LM head linear.

    使用模式::

        head = Glm5NextLmHead(cfg, seed=42)
        # REF (host bf16 / fp32)
        logits = head.forward(residual_flat)                    # [T, V] fp32
        # Zeus (device-resident)
        logits_z = head.forward_zeus(residual_flat_z)           # [T, V] fp32

    Input shape ``[T, N*H]`` bf16 ── 直接拼自 last block 的 residual_out (MHC 模式
    下 residual stream 全程 [T, N*H]). 内部先 mhc_contract 收口 → [T, H], 再过
    standalone RMSNorm + lm_head.
    """

    def __init__(self, cfg: Glm5NextLmHeadConfig, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)

        def rn(*shape, scale: float = 0.1) -> torch.Tensor:
            return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
                torch.bfloat16
            )

        # final RMSNorm weight (生产路径下接近 1.0, 这里 0.01 std 模拟)
        self.norm_w = (
            torch.randn(cfg.H, generator=g, dtype=torch.float32) * 0.01 + 1.0
        ).to(torch.bfloat16)
        # lm_head linear weight
        scale = 1.0 / (cfg.H ** 0.5)
        self.lm_head_w = rn(cfg.V, cfg.H, scale=scale)

        # Zeus device-resident state — lazy
        self._zeus_packed = False
        self._norm_w_z: Optional[torch.Tensor] = None
        self._lm_head_lmem: Optional[torch.Tensor] = None

    # ── REF forward ─────────────────────────────────────────────
    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        """REF: mHC contract + RMSNorm + lm_head Linear (fp32 logits).

        ``residual: [T, N*H] bf16  ->  logits[T, V] fp32``
        """
        # ① hc_contract: [T, N*H] → [T, H]
        hidden = _ref_hc_contract(residual, self.cfg.N)
        # ② RMSNorm (standalone, residual=None branch)
        h_norm = _ref_rmsnorm(hidden, self.norm_w, self.cfg.eps)
        # ③ lm_head linear (fp32 output)
        logits = torch.nn.functional.linear(h_norm.float(), self.lm_head_w.float())
        return logits

    # ── Zeus pack (lazy) ────────────────────────────────────────
    def _pack_zeus(self) -> None:
        if ZEUS_IMPORT_ERROR is not None:
            raise RuntimeError(f"Zeus runtime unavailable: {ZEUS_IMPORT_ERROR}")
        self._norm_w_z = self.norm_w.to("zeus")
        self._lm_head_lmem = sgl_kernel_zeus.linear_bf16_outfp32.pack(self.lm_head_w)
        self._zeus_packed = True

    # ── Zeus forward ────────────────────────────────────────────
    def forward_zeus(self, residual_z: torch.Tensor) -> torch.Tensor:
        """Zeus: 全链路 device-resident 三步, 无零碎 PyTorch op.

        ``residual_z: [T, N*H] bf16 (zeus)  ->  logits[T, V] fp32 (zeus)``

        三步全是 sgl_kernel_zeus kernel:
          K1: mhc_contract       — 多流收口
          K2: rmsnorm            — final norm
          K3: linear_bf16_outfp32 — lm_head GEMM (fp32 logits 直出)
        """
        if not self._zeus_packed:
            self._pack_zeus()

        # ① mhc_contract: [T, N*H] → [T, H] bf16
        hidden_z = sgl_kernel_zeus.mhc_contract(residual_z, n=self.cfg.N)
        # ② RMSNorm standalone
        h_norm_z = sgl_kernel_zeus.rmsnorm(hidden_z, self._norm_w_z, self.cfg.eps)
        # ③ lm_head (fp32 logits)
        logits_z = sgl_kernel_zeus.linear_bf16_outfp32(h_norm_z, self._lm_head_lmem)
        return logits_z


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = ("mhc_contract", "rmsnorm", "linear_bf16_outfp32")


def _run_stage(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print(f"Stage: {args.config}")
    print("=" * 60)
    cfg = load_cfg(args.config)
    ref_mem = estimate_ref_memory_gb(cfg)
    print(f"  cfg: V={cfg.V} H={cfg.H} N={cfg.N} eps={cfg.eps}")
    print(f"  REF memory estimate: {ref_mem:.2f} GB "
          f"(budget {REF_MEMORY_BUDGET_GB:.1f} GB)")

    torch.manual_seed(args.seed)
    head = Glm5NextLmHead(cfg, seed=args.seed)
    T = args.num_tokens
    g_in = torch.Generator().manual_seed(args.seed + 1)
    # 模拟 last block 的 residual_out: [T, N*H] bf16
    residual = (
        torch.randn(T, cfg.N * cfg.H, generator=g_in, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)

    # ── REF ───────────────────────────────────────────────────
    ref_out: Optional[torch.Tensor] = None
    ref_skipped = False
    if args.mode in ("ref", "both"):
        if ref_mem > REF_MEMORY_BUDGET_GB:
            print(f"  REF: SKIP ({ref_mem:.1f} GB > budget "
                  f"{REF_MEMORY_BUDGET_GB:.1f} GB)")
            ref_skipped = True
        else:
            ref_out = head.forward(residual)
            ok = (ref_out.shape == (T, cfg.V) and ref_out.dtype == torch.float32)
            print(f"  REF logits shape={tuple(ref_out.shape)} dtype={ref_out.dtype}")
            print(f"  REF logits[0,:4] = "
                  f"{[round(v,4) for v in ref_out[0,:4].tolist()]}")
            if not ok:
                return False

    # ── Zeus ──────────────────────────────────────────────────
    zeus_ok: Optional[bool] = None
    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"  ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
        else:
            try:
                z_logits = head.forward_zeus(residual.to("zeus"))
                z_logits_cpu = z_logits.cpu()
                print(f"  ZEUS logits shape={tuple(z_logits_cpu.shape)} "
                      f"dtype={z_logits_cpu.dtype}")
                print(f"  ZEUS logits[0,:4] = "
                      f"{[round(v,4) for v in z_logits_cpu[0,:4].tolist()]}")
                finite = torch.isfinite(z_logits_cpu).all().item()
                print(f"  ZEUS finite={finite}")
                shape_ok = (z_logits_cpu.shape == (T, cfg.V)
                            and z_logits_cpu.dtype == torch.float32)
                if ref_out is not None:
                    # mhc_contract 是 bit-exact (与 torch bf16.mean 同精度模型),
                    # rmsnorm 引入 fp32 中间累加误差, lm_head bf16 GEMM 累加误差
                    # ~ atol 5e-2 量级 (与 dev_moe 同档)
                    zeus_ok = compare_tensors(
                        f"lm_head.{args.config}.logits", ref_out, z_logits_cpu,
                        atol=5e-2, rtol=5e-2,
                    ) and finite and shape_ok
                else:
                    zeus_ok = finite and shape_ok
            except Exception as e:
                import traceback
                print(f"  ZEUS EXCEPTION: {e!r}")
                traceback.print_exc()
                zeus_ok = False

    if args.mode == "ref":
        return None if ref_skipped else True
    if args.mode == "zeus":
        return zeus_ok
    if ref_skipped:
        return None if zeus_ok is None else zeus_ok
    if zeus_ok is None:
        return True
    return zeus_ok


def main():
    parser = make_argparser(
        "dev_lm_head",
        description="GLM5-Next tail (mHC contract + final norm + lm_head) dev test",
    )
    args = parser.parse_args()
    print_header("GLM5-Next lm_head (mHC contract + norm + linear)", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_lm_head ({args.config})", ok)


if __name__ == "__main__":
    main()
