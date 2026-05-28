#!/usr/bin/env bash
# Build and test selected zenl kernels through the Triton AOT + V3 simulator path.
#
# Defaults intentionally exclude GEMM and index_put experimental kernels.
#
# Examples:
#   ./tests/others/run_triton_aot_tests.sh
#   ./tests/others/run_triton_aot_tests.sh --preset smoke
#   ./tests/others/run_triton_aot_tests.sh --preset elementwise
#   ./tests/others/run_triton_aot_tests.sh --kernels "memcpy add_f32 add_bf16"
#   ./tests/others/run_triton_aot_tests.sh --no-runtime --timeout 300

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_TRITON_BUILD_DIR="$REPO_ROOT/../triton_qmnpu/build/cmake.linux-x86_64-cpython-3.13"
DEFAULT_ZEUSV3_SIMULATOR_DIR="$REPO_ROOT/../Zeus3FunctionalSimulator"
DEFAULT_KERNEL_CONFIG_DIR="$REPO_ROOT/zenl/configs/zenl_triton_kernels"

TRITON_BUILD_DIR="${TRITON_BUILD_DIR:-$DEFAULT_TRITON_BUILD_DIR}"
ZEUSV3_SIMULATOR_DIR="${ZEUSV3_SIMULATOR_DIR:-$DEFAULT_ZEUSV3_SIMULATOR_DIR}"
PRESET="simple"
KERNELS=""
KERNELS_CONFIG="${TRITON_KERNELS_CONFIG:-}"
BUILD_RUNTIME=1
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
LOG_DIR="${LOG_DIR:-}"
KEEP_GOING=0

usage() {
    cat <<EOF
Usage: ./tests/others/run_triton_aot_tests.sh [options]

Options:
  --preset NAME       Kernel preset: smoke, elementwise, reduce, simple.
                      Default: simple.
  --kernels LIST      Explicit TRITON_KERNELS string. Overrides --preset.
  --config FILE       Kernel list file. Overrides --preset and is overridden
                      by --kernels.
  --triton-build DIR  Triton build dir containing zeus-compiler.
                      Default: $DEFAULT_TRITON_BUILD_DIR
  --simulator-dir DIR Zeus V3 simulator dir.
                      Default: $DEFAULT_ZEUSV3_SIMULATOR_DIR
  --no-runtime        Skip 'make runtime'.
  --timeout SEC       Run zenl retest under timeout. 0 disables timeout.
  --log-dir DIR       Write command output to DIR/triton_aot_<preset>.log.
  --keep-going        Pass -k to make where supported by make.
  -h, --help          Show this help.

Environment overrides:
  TRITON_BUILD_DIR, TRITON_KERNELS_CONFIG, ZEUSV3_SIMULATOR_DIR,
  TIMEOUT_SECONDS, LOG_DIR

Examples:
  ./tests/others/run_triton_aot_tests.sh --preset smoke
  ./tests/others/run_triton_aot_tests.sh --preset elementwise
  ./tests/others/run_triton_aot_tests.sh --config zenl/configs/zenl_triton_kernels/simple.list
  ./tests/others/run_triton_aot_tests.sh --kernels "memcpy add_f32 add_bf16"
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

normalize_ws() {
    local value="$*"
    echo "$value" | tr '\n' ' ' | xargs
}

read_kernel_config() {
    local path="$1"
    [[ -f "$path" ]] || die "kernel config not found: $path"
    sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$path" | xargs
}

dedupe_words() {
    local value="$*"
    echo "$value" | tr ' ' '\n' | awk 'NF && !seen[$0]++' | tr '\n' ' ' | xargs
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            [[ $# -ge 2 ]] || die "--preset requires a value"
            PRESET="$2"
            shift 2
            ;;
        --kernels)
            [[ $# -ge 2 ]] || die "--kernels requires a value"
            KERNELS="$2"
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || die "--config requires a value"
            KERNELS_CONFIG="$2"
            shift 2
            ;;
        --triton-build)
            [[ $# -ge 2 ]] || die "--triton-build requires a value"
            TRITON_BUILD_DIR="$2"
            shift 2
            ;;
        --simulator-dir)
            [[ $# -ge 2 ]] || die "--simulator-dir requires a value"
            ZEUSV3_SIMULATOR_DIR="$2"
            shift 2
            ;;
        --no-runtime)
            BUILD_RUNTIME=0
            shift
            ;;
        --timeout)
            [[ $# -ge 2 ]] || die "--timeout requires a value"
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || die "--log-dir requires a value"
            LOG_DIR="$2"
            shift 2
            ;;
        --keep-going)
            KEEP_GOING=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [[ -z "$KERNELS" ]]; then
    if [[ -z "$KERNELS_CONFIG" ]]; then
        KERNELS_CONFIG="$DEFAULT_KERNEL_CONFIG_DIR/$PRESET.list"
    fi
    KERNELS="$(read_kernel_config "$KERNELS_CONFIG")"
fi

KERNELS="$(dedupe_words "$(normalize_ws "$KERNELS")")"

[[ -d "$TRITON_BUILD_DIR" ]] || die "TRITON_BUILD_DIR not found: $TRITON_BUILD_DIR"
[[ -x "$TRITON_BUILD_DIR/third_party/triton_shared/tools/zeus-compiler/zeus-compiler" ]] || \
    die "zeus-compiler not found under TRITON_BUILD_DIR: $TRITON_BUILD_DIR"
[[ -d "$ZEUSV3_SIMULATOR_DIR" ]] || die "ZEUSV3_SIMULATOR_DIR not found: $ZEUSV3_SIMULATOR_DIR"

echo "Triton AOT V3 test"
echo "  preset:             $PRESET"
if [[ -n "$KERNELS_CONFIG" ]]; then
    echo "  kernel config:      $KERNELS_CONFIG"
fi
echo "  kernels:            $KERNELS"
echo "  TRITON_BUILD_DIR:   $TRITON_BUILD_DIR"
echo "  ZEUSV3_SIMULATOR_DIR: $ZEUSV3_SIMULATOR_DIR"
echo "  timeout seconds:    $TIMEOUT_SECONDS"

if [[ "$BUILD_RUNTIME" -eq 1 ]]; then
    echo ""
    echo "==> make runtime"
    make runtime
fi

MAKE_ARGS=()
if [[ "$KEEP_GOING" -eq 1 ]]; then
    MAKE_ARGS+=("-k")
fi

RETEST_CMD=(
    make
    "${MAKE_ARGS[@]}"
    -C zenl
    retest
    USE_TRITON=1
    "TRITON_KERNELS=$KERNELS"
    "TRITON_KERNELS_CONFIG="
    "TRITON_BUILD_DIR=$TRITON_BUILD_DIR"
)

echo ""
echo "==> ${RETEST_CMD[*]}"

if [[ -n "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/triton_aot_${PRESET}.log"
    echo "  log: $LOG_FILE"
    if [[ "$TIMEOUT_SECONDS" != "0" ]]; then
        timeout "$TIMEOUT_SECONDS" env ZEUSV3_SIMULATOR_DIR="$ZEUSV3_SIMULATOR_DIR" "${RETEST_CMD[@]}" 2>&1 | tee "$LOG_FILE"
    else
        ZEUSV3_SIMULATOR_DIR="$ZEUSV3_SIMULATOR_DIR" "${RETEST_CMD[@]}" 2>&1 | tee "$LOG_FILE"
    fi
else
    if [[ "$TIMEOUT_SECONDS" != "0" ]]; then
        timeout "$TIMEOUT_SECONDS" env ZEUSV3_SIMULATOR_DIR="$ZEUSV3_SIMULATOR_DIR" "${RETEST_CMD[@]}"
    else
        ZEUSV3_SIMULATOR_DIR="$ZEUSV3_SIMULATOR_DIR" "${RETEST_CMD[@]}"
    fi
fi
