#!/usr/bin/env bash
# Pure-Zeus install setup for sglang.
#
# Mirrors the AMD / Ascend-NPU install flow documented in
#   docs/platforms/amd_gpu.md
#   docs/platforms/ascend_npu.md
# — swap the CUDA `python/pyproject.toml` out of the way so pip resolves
# the `[srt_zeus]` extra against the multi-backend `pyproject_other.toml`
# without pulling cuda build dependencies.
#
# After this script:
#   - python/pyproject.toml      = the multi-backend definition (was pyproject_other.toml)
#   - python/pyproject_other.toml = no longer present in working tree
#   - sglang installed editable with [srt_zeus] extra
#
# To return to upstream-clean state (e.g. before `git pull`):
#   git checkout python/pyproject.toml python/pyproject_other.toml
#
# Prerequisites:
#   - python>=3.13 venv active (see zeus_dev/zeus_install.md §0 for box layout)
#   - torch_zeus + sgl_kernel_zeus already installed from their source trees
#     (see zeus_dev/zeus_install.md §1)
#   - rust toolchain (only required if outlines_core==0.1.26 wheel is missing
#     for your interpreter)

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ ! -f python/pyproject_other.toml ]]; then
  echo "error: python/pyproject_other.toml not found." >&2
  echo "  Either you've already run this script (git checkout to restore)," >&2
  echo "  or you're running outside the sglang repo." >&2
  exit 1
fi

echo "[zeus-setup] swapping pyproject.toml -> pyproject_other.toml (mirror upstream amd/npu flow)"
rm -rf python/pyproject.toml
mv python/pyproject_other.toml python/pyproject.toml

echo "[zeus-setup] pip install -e \"python[srt_zeus]\" --no-build-isolation"
pip install -e "python[srt_zeus]" --no-build-isolation

echo
echo "[zeus-setup] done. To restore upstream-clean state before pulling, run:"
echo "    git checkout python/pyproject.toml python/pyproject_other.toml"
