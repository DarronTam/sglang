#!/bin/sh

set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON:-python}

cd "$SCRIPT_DIR" || exit 1

usage() {
  cat <<'EOF'
Usage: ./run_test.sh [target]

Targets:
  all                Run all Python scripts under the related test directories (default)
  model_state_dev    Run Python scripts under model_state_dev recursively
  model_state_zhipu  Run Python scripts under model_state_zhipu recursively
  smoke_test         Run Python scripts under smoke_test recursively
  list               Show scripts that would be run
  help               Show this help

Examples:
  sh run_test.sh
  sh run_test.sh model_state_dev
  sh run_test.sh smoke_test
  PYTHON=python3 sh run_test.sh all
EOF
}

find_py_scripts() {
  dir=$1

  if [ ! -d "$dir" ]; then
    echo "error: directory not found: $dir" >&2
    exit 1
  fi

  find "$dir" -type d -name backup -prune -o -type f -name "*.py" -print | sort
}

print_scripts() {
  find_py_scripts "$1"
}

record_pass() {
  total_count=$((total_count + 1))
  pass_count=$((pass_count + 1))
}

record_fail() {
  script=$1
  total_count=$((total_count + 1))
  fail_count=$((fail_count + 1))
  failed_scripts=${failed_scripts}${script}'
'
}

run_script() {
  script=$1

  echo ""
  echo "========== [START] ${script} =========="
  if "$PYTHON_BIN" "$script"; then
    echo "========== [ PASS] ${script} =========="
    record_pass
  else
    rc=$?
    echo "========== [ FAIL] ${script} (exit=${rc}) ==========" >&2
    record_fail "$script"
  fi
}

run_dir() {
  dir=$1
  scripts_file=$(mktemp)
  find_py_scripts "$dir" > "$scripts_file"

  if [ ! -s "$scripts_file" ]; then
    echo "warning: no Python scripts found under $dir" >&2
    rm -f "$scripts_file"
    return 0
  fi

  while IFS= read -r script; do
    run_script "$script"
  done < "$scripts_file"

  rm -f "$scripts_file"
}

list_all() {
  print_scripts "model_state_dev"
  print_scripts "model_state_zhipu"
  print_scripts "smoke_test"
}

run_all() {
  run_dir "model_state_dev"
  run_dir "model_state_zhipu"
  run_dir "smoke_test"
}

print_summary() {
  echo ""
  echo "========== [SUMMARY] =========="
  echo "Total:  ${total_count}"
  echo "Passed: ${pass_count}"
  echo "Failed: ${fail_count}"

  if [ "$fail_count" -gt 0 ]; then
    echo ""
    echo "Failed scripts:"
    printf "%s" "$failed_scripts" | sed '/^$/d; s/^/  - /'
  fi
}

TARGET=${1:-all}
total_count=0
pass_count=0
fail_count=0
failed_scripts=

case "$TARGET" in
  all)
    run_all
    ;;
  model_state_dev|model_state_zhipu|smoke_test)
    run_dir "$TARGET"
    ;;
  list)
    list_all
    exit 0
    ;;
  help|-h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    echo "" >&2
    usage >&2
    exit 1
    ;;
esac

print_summary

if [ "$fail_count" -gt 0 ]; then
  exit 1
fi

exit 0
