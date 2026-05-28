#!/usr/bin/env bash

set -u

TEST_ROOT="${TEST_ROOT:-/workspace/sglang/zeus_dev}"
REPO_ROOT="${REPO_ROOT:-/workspace/sglang}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"

passed=()
failed=()

mapfile -d '' py_files < <(
  find "${TEST_ROOT}" \
    -type d -name "__pycache__" -prune -o \
    -type f -name "*.py" -print0 | sort -z
)

total="${#py_files[@]}"

if [[ "${total}" -eq 0 ]]; then
  echo "No Python scripts found under ${TEST_ROOT}"
  exit 0
fi

echo "Found ${total} Python scripts under ${TEST_ROOT}"
echo "Using Python: ${PYTHON_BIN}"
echo

for script in "${py_files[@]}"; do
  rel_path="${script#"${TEST_ROOT}/"}"

  echo
  echo "*** Running script: ${rel_path} ***"
  echo

  (
    cd "${TEST_ROOT}" || exit 1
    "${PYTHON_BIN}" "${script}"
  )
  status=$?

  if [[ "${status}" -eq 0 ]]; then
    echo "[PASS] ${rel_path}"
    passed+=("${rel_path}")
  else
    echo "[FAIL] ${rel_path} (exit code: ${status})"
    failed+=("${rel_path} (exit code: ${status})")
  fi

  echo
done

echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Total:  ${total}"
echo "Passed: ${#passed[@]}"
echo "Failed: ${#failed[@]}"

if [[ "${#failed[@]}" -eq 0 ]]; then
  echo
  echo "All Python scripts passed."
  exit 0
fi

echo
echo "Failed scripts:"
for item in "${failed[@]}"; do
  echo "  - ${item}"
done

exit 1
