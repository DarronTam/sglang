#!/usr/bin/env bash
# Pure-Zeus development install helper for SGLang.
#
# This is a developer convenience wrapper, not the formal platform install
# path. The public docs/platforms/zeus.md flow is the source of truth:
# install the PyTorch CPU stack, install torch_zeus / sgl_kernel_zeus, copy
# python/pyproject_zeus.toml to python/pyproject.toml, then install SGLang.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ ! -f python/pyproject_zeus.toml ]]; then
  echo "error: python/pyproject_zeus.toml not found." >&2
  echo "  Restore it from git or run this from the sglang repo." >&2
  exit 1
fi

if [[ ! -f zeus_dev/check_zeus_env.py ]]; then
  echo "error: zeus_dev/check_zeus_env.py not found." >&2
  echo "  Restore it from git or run this from the sglang repo." >&2
  exit 1
fi

echo "[zeus-setup] checking active Python environment"
python zeus_dev/check_zeus_env.py

echo "[zeus-setup] using python/pyproject_zeus.toml as python/pyproject.toml"
cp python/pyproject_zeus.toml python/pyproject.toml

echo "[zeus-setup] pip install -e python --no-build-isolation"
python -m pip install -e python --no-build-isolation

echo
echo "[zeus-setup] done. python/pyproject.toml now contains the Zeus-specific pyproject."
