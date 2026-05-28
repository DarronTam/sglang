"""
GLM5-Next embedding (入口) 独立模块 + REF↔Zeus 对拍.

把 ``Glm5NextModel.embed_tokens`` (``VocabParallelEmbedding``) 抽成一个独立
module ``Glm5NextEmbed``, 暴露 ``__init__`` + ``forward`` (REF host bf16) +
``forward_zeus`` (Zeus device-resident), 供 multi-layer 装配复用.

形状 (从 config json 读真实 vocab/hidden):
  - 16b:   V=154880, H=2048  → embed_w ≈ 0.61 GB bf16  (REF 可跑)
  - next:  V=154880, H=4096  → embed_w ≈ 1.21 GB bf16  (REF 可跑)

Chain (decode-only, 单 device, TP=1):
  - input_ids[T] int64       (host, 由调度器给出)
  - embed_w[V, H] bf16       (LocalMem 之外: Zeus op `embedding` 直接吃 plain
                              device tensor, 不走 LocalMem pack)
  - REF:   out = embed_w[ids]           (torch index_select)
  - Zeus:  out = sgl_kernel_zeus.embedding(ids_z, embed_w_z)   → [T, H] bf16

不做的事:
  - TP shard / DP sharding (单 device, vocab 整张表常驻 Zeus)
  - input_scattered padded-row zero scrub (decode path 没有 extend_num_tokens,
    所有 row 都是真 token; reference glm5_next.py:836-841 那段 zero_() 不触发)
  - tie_word_embeddings 共享权重 (在 ``dev_lm_head.py`` 那边处理)

用法:
  python glm5next_modules/dev_embed.py                  # 16b / both
  python glm5next_modules/dev_embed.py --config next    # next / both
  python glm5next_modules/dev_embed.py --mode zeus
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
class Glm5NextEmbedConfig:
    """Embedding sublayer config (单 device, TP=1).

    Vocab 直接来自 config.json (``vocab_size`` / ``padded_vocab_size`` 一致).
    """
    V: int
    H: int
    name: str = "proxy"

    @classmethod
    def from_json(cls, path: Path, name: Optional[str] = None) -> "Glm5NextEmbedConfig":
        import json
        raw = json.loads(Path(path).read_text())
        return cls(
            V=int(raw["vocab_size"]),
            H=int(raw["hidden_size"]),
            name=name or Path(path).stem,
        )


def load_cfg(which: str) -> Glm5NextEmbedConfig:
    return Glm5NextEmbedConfig.from_json(config_path(which), name=which)


def estimate_ref_memory_gb(cfg: Glm5NextEmbedConfig) -> float:
    """REF 权重显存估算 (GB).

    单张 embed table ``[V, H] bf16`` = V*H*2 字节; 再保留一份 host copy 用于
    Zeus pack 验证, 总占用 ~2x.
    """
    return cfg.V * cfg.H * 2 * 2 / (1024 ** 3)


# ── Module ──────────────────────────────────────────────────────
class Glm5NextEmbed:
    """GLM5-Next input embedding (token_ids → hidden_states).

    使用模式::

        embed = Glm5NextEmbed(cfg, seed=42)
        h = embed.forward(input_ids)              # REF (host bf16)
        h_z = embed.forward_zeus(input_ids_z)     # Zeus device-resident

    REF / Zeus 共享同一份权重; Zeus 端 lazy ``_pack_zeus`` 把 embed_w 一次性
    搬上 device 并缓存 (等价生产 layer.__init__ 一次性 to('zeus')).
    """

    def __init__(self, cfg: Glm5NextEmbedConfig, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        # 与 VocabParallelEmbedding 默认初始化 (normal(0, 1/sqrt(H))) 对齐量级
        scale = 1.0 / (cfg.H ** 0.5)
        self.embed_w = (
            torch.randn(cfg.V, cfg.H, generator=g, dtype=torch.float32) * scale
        ).to(torch.bfloat16)

        # Zeus device-resident state — lazy
        self._zeus_packed = False
        self._embed_w_z: Optional[torch.Tensor] = None

    # ── REF forward ─────────────────────────────────────────────
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """REF embedding lookup.

        ``input_ids: [T] int64 (host)  ->  [T, H] bf16 (host)``
        """
        return self.embed_w[input_ids.long()]

    # ── Zeus pack (lazy) ────────────────────────────────────────
    def _pack_zeus(self) -> None:
        if ZEUS_IMPORT_ERROR is not None:
            raise RuntimeError(f"Zeus runtime unavailable: {ZEUS_IMPORT_ERROR}")
        # `embedding` op 吃 plain device tensor, 不需要 LocalMem pack
        self._embed_w_z = self.embed_w.to("zeus")
        self._zeus_packed = True

    # ── Zeus forward ────────────────────────────────────────────
    def forward_zeus(self, input_ids_z: torch.Tensor) -> torch.Tensor:
        """Zeus embedding lookup.

        ``input_ids_z: [T] int64 (zeus)  ->  [T, H] bf16 (zeus)``
        """
        if not self._zeus_packed:
            self._pack_zeus()
        return sgl_kernel_zeus.embedding(input_ids_z, self._embed_w_z)


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = ("embedding",)


def _run_stage(args) -> Optional[bool]:
    """单 stage: 按 config 加载真实 V/H, 按 mode 跑 REF / Zeus / 对拍."""
    print("\n" + "=" * 60)
    print(f"Stage: {args.config} (real shape)")
    print("=" * 60)
    cfg = load_cfg(args.config)
    ref_mem = estimate_ref_memory_gb(cfg)
    print(f"  cfg: V={cfg.V} H={cfg.H}")
    print(f"  REF memory estimate: {ref_mem:.2f} GB "
          f"(budget {REF_MEMORY_BUDGET_GB:.1f} GB)")

    torch.manual_seed(args.seed)
    embed = Glm5NextEmbed(cfg, seed=args.seed)
    # decode-only path: T 即 batch (一步 decode 一个 token / batch token)
    input_ids = torch.randint(
        0, cfg.V, (args.num_tokens,),
        generator=torch.Generator().manual_seed(args.seed + 1),
        dtype=torch.int64,
    )

    # ── REF ───────────────────────────────────────────────────
    ref_out: Optional[torch.Tensor] = None
    ref_skipped = False
    if args.mode in ("ref", "both"):
        if ref_mem > REF_MEMORY_BUDGET_GB:
            print(f"  REF: SKIP ({ref_mem:.1f} GB > budget "
                  f"{REF_MEMORY_BUDGET_GB:.1f} GB)")
            ref_skipped = True
        else:
            ref_out = embed.forward(input_ids)
            ok = (ref_out.shape == (args.num_tokens, cfg.H)
                  and ref_out.dtype == torch.bfloat16)
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
                z_out = embed.forward_zeus(input_ids.to("zeus"))
                z_out_cpu = z_out.cpu()
                print(f"  ZEUS out shape={tuple(z_out_cpu.shape)} "
                      f"dtype={z_out_cpu.dtype}")
                print(f"  ZEUS out[0,:4] = "
                      f"{[round(v,4) for v in z_out_cpu[0,:4].float().tolist()]}")
                finite = torch.isfinite(z_out_cpu).all().item()
                print(f"  ZEUS finite={finite}")
                shape_ok = (z_out_cpu.shape == (args.num_tokens, cfg.H)
                            and z_out_cpu.dtype == torch.bfloat16)
                if ref_out is not None:
                    # Embedding 是 pure index_select, REF / Zeus 应当 bit-exact
                    zeus_ok = compare_tensors(
                        f"embed.{args.config}.out", ref_out, z_out_cpu,
                        atol=0.0, rtol=0.0,
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
    parser = make_argparser("dev_embed", description="GLM5-Next embedding dev test")
    args = parser.parse_args()
    print_header("GLM5-Next embedding", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_embed ({args.config})", ok)


if __name__ == "__main__":
    main()
