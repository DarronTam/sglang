#!/usr/bin/env bash
# Configure the local torch_zeus runtime environment and run add implementation probe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export ZEUSV3_SIMULATOR_DIR="${ZEUSV3_SIMULATOR_DIR:-$REPO_ROOT/../Zeus3FunctionalSimulator}"
export ZENL_SIM_LOADER_DEBUG="${ZENL_SIM_LOADER_DEBUG:-1}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$REPO_ROOT/zenl/build:$REPO_ROOT/runtime/build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Configured environment:"
echo "  ZEUSV3_SIMULATOR_DIR=$ZEUSV3_SIMULATOR_DIR"
echo "  ZENL_SIM_LOADER_DEBUG=$ZENL_SIM_LOADER_DEBUG"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  PYTHONPATH=$PYTHONPATH"
echo

exec python "$SCRIPT_DIR/probe_add_impl.py"
