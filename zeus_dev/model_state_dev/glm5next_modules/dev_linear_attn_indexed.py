"""
GLM5-Next linear-attn conv1d indexed parity test.

This script validates the production decode conv-state shape:

  causal_conv1d_update_indexed(...).split([P, P, P])
      == causal_conv1d_update_split_indexed(...)

Both paths read/write a cache pool state [N_pool, 3P, K-1] through
cache_indices [B]. The split-indexed op should return contiguous q/k/v outputs
while producing the same pool update as the existing indexed op.

Usage:
  cd /workspace/sglang/zeus_dev/model_state_dev
  python glm5next_modules/dev_linear_attn_indexed.py
  python glm5next_modules/dev_linear_attn_indexed.py --config next --num-tokens 4 --steps 3
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

import _common
from _common import (
    ZEUS_IMPORT_ERROR,
    compare_tensors,
    config_path,
    make_argparser,
    print_header,
    print_summary,
    sgl_kernel_zeus,
    zeus_chain_available,
)


@dataclass(frozen=True)
class LinearAttnShape:
    name: str
    num_heads: int
    head_dim: int
    conv_size: int

    @property
    def proj_size(self) -> int:
        return self.num_heads * self.head_dim


def load_shape(which: str) -> LinearAttnShape:
    import json

    raw = json.loads(Path(config_path(which)).read_text())
    lac = raw["linear_attn_config"]
    return LinearAttnShape(
        name=which,
        num_heads=int(lac["num_heads"]),
        head_dim=int(lac["head_dim"]),
        conv_size=int(lac["short_conv_kernel_size"]),
    )


def _make_cache_indices(batch: int, pool_size: int, seed: int) -> torch.Tensor:
    if pool_size < batch:
        raise ValueError(f"pool_size ({pool_size}) must be >= batch ({batch})")
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(pool_size, generator=g, dtype=torch.int64)[:batch].to(torch.int32)


def _make_step_inputs(
    batch: int,
    channels: int,
    kernel_width: int,
    pool_size: int,
    *,
    has_bias: bool,
    seed: int,
):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(batch, channels, generator=g, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    weight = (torch.randn(channels, kernel_width, generator=g, dtype=torch.float32) * 0.03).to(torch.bfloat16)
    bias = None
    if has_bias:
        bias = (torch.randn(channels, generator=g, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    pool = (torch.randn(pool_size, channels, kernel_width - 1, generator=g, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    return x, weight, bias, pool


def _compare_exact_or_report(name: str, ref: torch.Tensor, got: torch.Tensor) -> bool:
    if torch.equal(ref, got):
        print(f"  [OK] {name}: bit-exact")
        return True
    return compare_tensors(name, ref, got, atol=0.0, rtol=0.0)


def run_once(args) -> Optional[bool]:
    required = (
        "causal_conv1d_update_indexed",
        "causal_conv1d_update_split_indexed",
    )
    if not zeus_chain_available(*required):
        print(f"  ZEUS: SKIP (required ops unavailable: {ZEUS_IMPORT_ERROR})")
        return None

    shape = load_shape(args.config)
    batch = args.num_tokens
    pool_size = args.pool_size or max(16, batch * 4)
    p = shape.proj_size
    channels = 3 * p
    kernel_width = shape.conv_size
    has_bias = not args.no_bias

    print("\n" + "=" * 60)
    print(f"Stage: split_indexed parity  config={shape.name}")
    print("=" * 60)
    print(
        f"  B={batch}  N_pool={pool_size}  P={p}  C=3P={channels}  "
        f"K={kernel_width}  steps={args.steps}  bias={has_bias}"
    )

    x0, weight, bias, pool0 = _make_step_inputs(
        batch,
        channels,
        kernel_width,
        pool_size,
        has_bias=has_bias,
        seed=args.seed,
    )
    cache_indices = _make_cache_indices(batch, pool_size, seed=args.seed + 97)
    untouched = sorted(set(range(pool_size)) - set(cache_indices.tolist()))
    print(f"  cache_indices={cache_indices.tolist()}")

    pool_indexed = pool0.clone().to("zeus")
    pool_split_indexed = pool0.clone().to("zeus")
    weight_z = weight.to("zeus")
    bias_z = bias.to("zeus") if bias is not None else None
    cache_indices_z = cache_indices.to("zeus")

    ok = True
    for step in range(args.steps):
        if step == 0:
            x = x0
        else:
            g = torch.Generator().manual_seed(args.seed + 1000 + step)
            x = (torch.randn(batch, channels, generator=g, dtype=torch.float32) * 0.05).to(torch.bfloat16)
        x_z = x.to("zeus")

        qkv = sgl_kernel_zeus.causal_conv1d_update_indexed(
            x_z,
            pool_indexed,
            cache_indices_z,
            weight_z,
            bias_z,
            activation="silu",
        )
        q_ref, k_ref, v_ref = qkv.split([p, p, p], dim=-1)

        q, k, v = sgl_kernel_zeus.causal_conv1d_update_split_indexed(
            x_z,
            pool_split_indexed,
            cache_indices_z,
            weight_z,
            bias_z,
            activation="silu",
        )

        print(f"\n  step={step}")
        ok &= _compare_exact_or_report("q", q_ref.cpu(), q.cpu())
        ok &= _compare_exact_or_report("k", k_ref.cpu(), k.cpu())
        ok &= _compare_exact_or_report("v", v_ref.cpu(), v.cpu())
        ok &= _compare_exact_or_report("pool", pool_indexed.cpu(), pool_split_indexed.cpu())
        print(
            f"  output_contiguous: q={q.is_contiguous()} "
            f"k={k.is_contiguous()} v={v.is_contiguous()}"
        )
        ok &= q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

    pool_final = pool_split_indexed.cpu()
    for slot in untouched:
        if not torch.equal(pool_final[slot], pool0[slot]):
            print(f"  [FAIL] untouched pool slot modified: {slot}")
            ok = False
            break
    if untouched:
        print(f"\n  untouched slots checked: {len(untouched)}")

    return ok


def main(argv=None) -> int:
    def extra(parser):
        parser.add_argument("--steps", type=int, default=3)
        parser.add_argument("--pool-size", type=int, default=None)
        parser.add_argument("--no-bias", action="store_true")

    parser = make_argparser(
        "dev_linear_attn_indexed.py",
        "Validate causal_conv1d_update_split_indexed against indexed(...).split(...).",
        extra_setup=extra,
    )
    args = parser.parse_args(argv)
    if args.mode not in ("both", "zeus"):
        print("This script is Zeus-only; use --mode both or --mode zeus.")
        return 2

    print_header("GLM5Next linear-attn indexed conv1d", args)
    ok = run_once(args)
    print_summary("causal_conv1d_update_split_indexed", ok)
    return 0 if ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
