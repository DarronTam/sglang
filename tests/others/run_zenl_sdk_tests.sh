#!/usr/bin/env bash
# Validate the installed ZENL SDK shape with ABI, CMake, pkg-config, and a
# small out-of-tree executable that calls a real ZENL operator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PREFIX="${PREFIX:-/tmp/zenl_sdk_check}"
BUILD_DIR="${BUILD_DIR:-/tmp/zenl_sdk_example_build}"
DEFAULT_ZEUSV3_SIMULATOR_DIR="$REPO_ROOT/../Zeus3FunctionalSimulator"

if [[ -z "${ZEUSV3_SIMULATOR_DIR:-}" && -d "$DEFAULT_ZEUSV3_SIMULATOR_DIR" ]]; then
  export ZEUSV3_SIMULATOR_DIR="$DEFAULT_ZEUSV3_SIMULATOR_DIR"
fi

echo "ZENL SDK validation"
echo "  PREFIX:    $PREFIX"
echo "  BUILD_DIR: $BUILD_DIR"
if [[ -n "${ZEUSV3_SIMULATOR_DIR:-}" ]]; then
  echo "  ZEUSV3_SIMULATOR_DIR: $ZEUSV3_SIMULATOR_DIR"
fi

echo ""
echo "==> make -C zenl test"
make -C zenl test

echo ""
echo "==> make -C zenl abi-check"
make -C zenl abi-check

echo ""
echo "==> make -C zenl install"
make -C zenl install PREFIX="$PREFIX"

for required_file in \
  "$PREFIX/include/zenl/zenl.h" \
  "$PREFIX/include/zenl/zenl_api.h" \
  "$PREFIX/include/zenl/zenl_export.h" \
  "$PREFIX/include/zenl/zenl_types.h" \
  "$PREFIX/include/zenl/zenl_version.h" \
  "$PREFIX/lib/cmake/ZENL/ZENLConfig.cmake" \
  "$PREFIX/lib/cmake/ZENL/ZENLTargets.cmake" \
  "$PREFIX/lib/cmake/ZENL/ZENLConfigVersion.cmake" \
  "$PREFIX/lib/pkgconfig/zenl.pc"; do
  if [[ ! -f "$required_file" ]]; then
    echo "missing installed SDK file: $required_file" >&2
    exit 1
  fi
done

echo ""
echo "==> cmake configure"
cmake -S "$SCRIPT_DIR/zenl_sdk_smoke" \
  -B "$BUILD_DIR" \
  -DCMAKE_PREFIX_PATH="$PREFIX"

echo ""
echo "==> cmake build"
cmake --build "$BUILD_DIR"

echo ""
echo "==> run out-of-tree smoke"
"$BUILD_DIR/zenl_sdk_smoke"

echo ""
echo "==> pkg-config"
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" pkg-config --cflags --libs zenl

echo ""
echo "ZENL SDK validation passed"
