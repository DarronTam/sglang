#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage: ./run_test.sh [target]

Targets:
  all     Run all related test scripts (default)
  smoke   Run lightweight graph capture tests
  s1      Run test_zeus_graph_s1.py
  e2e     Run test_zeus_graph_e2e.py
  glm4    Run dev_glm4_moe_test.py
  kimi    Run dev_kimi_linear_attn_test.py
  list    Show available targets
  help    Show this help

Examples:
  ./run_test.sh
  ./run_test.sh smoke
  ./run_test.sh e2e
EOF
}

run_py() {
  local name="$1"
  local script="$2"

  echo ""
  echo "========== [START] ${name} =========="
  "$PYTHON_BIN" "$script"
  echo "========== [ PASS] ${name} =========="
}

run_smoke() {
  run_py "S1 Graph Capture" "test_zeus_graph_s1.py"
}

run_all() {
  run_py "S1 Graph Capture" "test_zeus_graph_s1.py"
  run_py "E2E Graph Validation" "test_zeus_graph_e2e.py"
  run_py "GLM4 MoE Stage Align" "dev_glm4_moe_test.py"
  run_py "Kimi Linear Attn Stage Align" "dev_kimi_linear_attn_test.py"
}

TARGET="${1:-all}"

case "$TARGET" in
  all)
    run_all
    ;;
  smoke)
    run_smoke
    ;;
  s1)
    run_py "S1 Graph Capture" "test_zeus_graph_s1.py"
    ;;
  e2e)
    run_py "E2E Graph Validation" "test_zeus_graph_e2e.py"
    ;;
  glm4)
    run_py "GLM4 MoE Stage Align" "dev_glm4_moe_test.py"
    ;;
  kimi)
    run_py "Kimi Linear Attn Stage Align" "dev_kimi_linear_attn_test.py"
    ;;
  list)
    echo "Available targets: all smoke s1 e2e glm4 kimi"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown target: $TARGET"
    echo ""
    usage
    exit 1
    ;;
esac

echo ""
echo "All requested tests finished."
