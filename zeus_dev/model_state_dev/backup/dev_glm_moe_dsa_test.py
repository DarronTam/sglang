"""
Compatibility wrapper for the split GLM-MoE-DSA milestone tests.

Use the path-specific scripts directly when developing kernels:
  python dev_glm_moe_dsa_decode_test.py --stage decode_q_proj_fused
  python dev_glm_moe_dsa_prefill_test.py --stage prefill_q_proj_fused

The split scripts default to --mode zeus, matching dev_kimi_linear_attn_test.py:
implemented kernels run, unsupported Zeus stages are summarized as SKIP.
Pass --mode ref when you want the full pure-torch golden path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def _run(script: str, args: list[str]) -> int:
    cmd = [sys.executable, str(THIS_DIR / script), *args]
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        choices=["decode", "prefill", "both"],
        default="both",
        help="Which split DSA test path to run.",
    )
    args, rest = parser.parse_known_args()

    rc = 0
    if args.path in {"decode", "both"}:
        rc = _run("dev_glm_moe_dsa_decode_test.py", rest)
        if rc:
            raise SystemExit(rc)
    if args.path in {"prefill", "both"}:
        rc = _run("dev_glm_moe_dsa_prefill_test.py", rest)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
