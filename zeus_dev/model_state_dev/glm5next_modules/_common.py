"""
glm5next_modules._common — GLM5-Next 子层 dev 脚本的公共脚手架

子目录脚本（dev_moe.py / dev_linear_attn.py）共享：
  - sys.path bootstrap：让脚本以 `python glm5next_modules/dev_xxx.py` 跑时，
    sibling 包（dev_glm4_moe_test）与 config_*.json 仍然可达
  - Zeus runtime 探测（一次 import，三个错误处理点共享）
  - config json 路径解析（"16b" / "next" → 真实 path）
  - compare_tensors 包装（统一对 dev_glm4_moe_test 复用）
  - zeus_chain_available 齐备性检查
  - CLI scaffold：标准 `--config / --mode / --num-tokens / --seed` argparser
    + 统一 header / summary printers

不放进来的:
  - 各模块 Config dataclass / 权重 init / forward 实现 —— 这些每个模块自己来
  - REF / Zeus chain 拼装顺序 —— 同上
  - 数学 helper (l2norm / softplus / silu) —— 每个模块自带
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch

# ── Path bootstrap ───────────────────────────────────────────────
# 子目录脚本被 `python glm5next_modules/dev_*.py` 直接跑时，把上一级
# (model_state_dev/) 加进 sys.path，让 `dev_glm4_moe_test` 与 config_*.json
# 这些 sibling 可见。
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))


# ── Zeus runtime detection ──────────────────────────────────────
try:
    import torch_zeus  # noqa: F401
    import sgl_kernel_zeus  # noqa: F401

    ZEUS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    torch_zeus = None
    sgl_kernel_zeus = None
    ZEUS_IMPORT_ERROR = exc


# ── Config path resolution ──────────────────────────────────────
def config_path(which: str) -> Path:
    """解析 config JSON 路径 (位于上一级 model_state_dev/)."""
    name = "config_16b_v2.json" if which == "16b" else "config.json"
    return _PARENT_DIR / name


def load_raw_config(which: str) -> dict:
    return json.loads(config_path(which).read_text())


# ── Compare ─────────────────────────────────────────────────────
def compare_tensors(name, ref, got, atol: float = 5e-3, rtol: float = 5e-3) -> bool:
    """统一的对拍打印 (与 dev_glm4_moe_test.compare_tensors 保持一致).

    早期版本通过 lazy `import dev_glm4_moe_test` 复用,但该模块顶层会 import
    `sglang.srt.server_args`,进而触发 fla/utils 在 import 期探测 triton driver
    并打 "Triton is not supported" warning. compare_tensors 自身只有几行
    `torch.allclose` + 打印, 没必要把 sglang 拖进来 — 直接内联.
    """
    a = ref.detach().float().cpu()
    b = got.detach().float().cpu()
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: ref={a.shape} got={b.shape}")
        return False
    abs_diff = (a - b).abs()
    close = torch.allclose(a, b, atol=atol, rtol=rtol)
    status = "PASS" if close else "DIFF"
    print(
        f"  [{name}] {status} | max_diff={abs_diff.max().item():.6e} "
        f"mean_diff={abs_diff.mean().item():.6e} shape={list(a.shape)}"
    )
    return close


# ── Zeus chain availability ────────────────────────────────────
def zeus_chain_available(*op_names: str) -> bool:
    """检查 sgl_kernel_zeus.* 里指定 op 是否齐备."""
    if ZEUS_IMPORT_ERROR is not None:
        return False
    return all(hasattr(sgl_kernel_zeus, n) for n in op_names)


# ── 默认 REF 显存阈值 (每个模块可覆盖) ───────────────────────
REF_MEMORY_BUDGET_GB = 8.0


# ── CLI scaffold ────────────────────────────────────────────────
def make_argparser(
    prog: str,
    description: str = "",
    extra_setup=None,
) -> argparse.ArgumentParser:
    """构造统一 dev script argparser.

    标准参数：--config / --mode / --num-tokens / --seed.
    模块如需要额外参数，传 ``extra_setup(parser)`` 回调，或在返回的 parser
    上直接 ``add_argument``.
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--config", choices=["16b", "next"], default="16b")
    parser.add_argument("--mode", choices=["both", "ref", "zeus"], default="both")
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    if extra_setup is not None:
        extra_setup(parser)
    return parser


def print_header(label: str, args: argparse.Namespace) -> None:
    """统一的 stage header (模块名 + config + mode + 其它参数 + Zeus 可达性)."""
    extras = "  ".join(
        f"{k}={v}" for k, v in vars(args).items()
        if k not in ("config", "mode")
    )
    print(f"{label}  config={args.config}  mode={args.mode}  {extras}")
    print(f"Zeus runtime available: {ZEUS_IMPORT_ERROR is None}")


def print_summary(name: str, ok: Optional[bool]) -> None:
    """统一的 PASS / FAIL / SKIP 总结块."""
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if ok is True:
        status = "PASS"
    elif ok is False:
        status = "FAIL"
    else:
        status = "SKIP"
    print(f"  {name} : {status}")
    print("=" * 60)


# ── Public re-exports (方便子脚本 `from _common import sgl_kernel_zeus`) ─
__all__ = [
    "ZEUS_IMPORT_ERROR",
    "torch_zeus",
    "sgl_kernel_zeus",
    "config_path",
    "load_raw_config",
    "compare_tensors",
    "zeus_chain_available",
    "make_argparser",
    "print_header",
    "print_summary",
    "REF_MEMORY_BUDGET_GB",
]
