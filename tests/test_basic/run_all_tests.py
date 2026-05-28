#!/usr/bin/env python3
"""
Run all Zeus tests and generate a summary report

Usage:
    python run_all_tests.py              # Run all tests with summary
    python run_all_tests.py -v           # Verbose mode (show test output)
    python run_all_tests.py --sgl-kernel-mode jit
    python run_all_tests.py -h           # Show help
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored(text, color):
    """Add color to text"""
    return f"{color}{text}{Colors.ENDC}"


def print_header(text):
    """Print a header"""
    print("\n" + "=" * 80)
    print(colored(text, Colors.HEADER + Colors.BOLD))
    print("=" * 80)


def print_section(text):
    """Print a section header"""
    print("\n" + colored(text, Colors.OKBLUE + Colors.BOLD))
    print("-" * 80)


def format_time(seconds):
    """Format time in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"


class TestResult:
    """Store test result information"""
    def __init__(self, name, passed, duration, output="", error=""):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.output = output
        self.error = error


def run_test(test_file, verbose=False, env=None):
    """Run a single test file and return the result"""
    test_name = test_file.stem

    if verbose:
        print(f"\n{colored('▶', Colors.OKBLUE)} Running {test_name}...")
    else:
        print(f"{colored('▶', Colors.OKBLUE)} Running {test_name}...", end=" ", flush=True)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60  # 60 seconds timeout per test
        )

        duration = time.time() - start_time
        passed = result.returncode == 0

        if verbose:
            print(result.stdout)
            if result.stderr:
                print(colored("STDERR:", Colors.WARNING))
                print(result.stderr)

        if passed:
            status = colored("✓ PASS", Colors.OKGREEN)
        else:
            status = colored("✗ FAIL", Colors.FAIL)

        if not verbose:
            print(f"{status} ({format_time(duration)})")
        else:
            print(f"{status} in {format_time(duration)}")

        return TestResult(
            name=test_name,
            passed=passed,
            duration=duration,
            output=result.stdout,
            error=result.stderr
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        status = colored("✗ TIMEOUT", Colors.FAIL)
        print(f"{status} ({format_time(duration)})")

        return TestResult(
            name=test_name,
            passed=False,
            duration=duration,
            error="Test timed out after 60 seconds"
        )

    except Exception as e:
        duration = time.time() - start_time
        status = colored("✗ ERROR", Colors.FAIL)
        print(f"{status} ({format_time(duration)})")

        return TestResult(
            name=test_name,
            passed=False,
            duration=duration,
            error=str(e)
        )


def find_test_files(test_dir):
    """Find all test files in the directory"""
    test_files = sorted(test_dir.glob("test_*.py"))
    # Exclude this script itself
    test_files = [f for f in test_files if f.name != "run_all_tests.py"]
    return test_files


def default_simulator_dir():
    """Return the project-local functional simulator path."""
    return Path(__file__).resolve().parents[3] / "Zeus3FunctionalSimulator"


def build_test_env(args):
    """Build child-process environment for the selected sgl-kernel-zeus mode."""
    env = os.environ.copy()
    mode = args.sgl_kernel_mode

    sim_dir = args.zeusv3_simulator_dir
    if sim_dir is None:
        sim_dir = env.get("ZEUSV3_SIMULATOR_DIR") or str(default_simulator_dir())
    env["ZEUSV3_SIMULATOR_DIR"] = sim_dir

    if mode == "aot":
        env["ZECC_JIT_FORCE_AOT"] = "1"
        for key in (
            "ZECC_JIT",
            "ZECC_JIT_STRICT",
            "ZECC_JIT_LOG",
            "ZECC_JIT_DUMP_DIR",
        ):
            env.pop(key, None)
    elif mode == "jit":
        env.pop("ZECC_JIT_FORCE_AOT", None)
        env["ZECC_JIT"] = "1"
        env["ZECC_JIT_LOG"] = "1"
        if args.sgl_kernel_jit_strict:
            env["ZECC_JIT_STRICT"] = "1"
        else:
            env.pop("ZECC_JIT_STRICT", None)
    else:
        raise ValueError(f"Unsupported sgl-kernel-zeus mode: {mode}")

    return env


def print_sgl_kernel_mode(args, env):
    """Print the selected sgl-kernel-zeus dispatch mode."""
    print(f"sgl-kernel-zeus mode: {args.sgl_kernel_mode}")
    print(f"ZEUSV3_SIMULATOR_DIR: {env.get('ZEUSV3_SIMULATOR_DIR', '<unset>')}")
    if args.sgl_kernel_mode == "aot":
        print("  ZECC_JIT_FORCE_AOT=1")
        print("  ZECC_JIT unset")
    else:
        strict = env.get("ZECC_JIT_STRICT", "0")
        print("  ZECC_JIT=1")
        print(f"  ZECC_JIT_STRICT={strict}")
        if strict != "1":
            print("  note: graph-capture tests may fall back to sim.c/AOT by design")
        else:
            print("  note: test_graph_e2e.py allows sim.c/AOT fallback because runtime JIT is not graph-capture-safe yet")


def env_for_test_file(base_env, args, test_file):
    """Return per-file env, relaxing strict JIT for graph-capture coverage."""
    if (
        args.sgl_kernel_mode == "jit"
        and args.sgl_kernel_jit_strict
        and test_file.name == "test_graph_e2e.py"
    ):
        env = base_env.copy()
        env["ZECC_JIT_STRICT"] = "0"
        return env
    return base_env


def print_summary(results, total_time):
    """Print a summary of all test results"""
    print_header("TEST SUMMARY")

    passed_tests = [r for r in results if r.passed]
    failed_tests = [r for r in results if not r.passed]

    total_tests = len(results)
    num_passed = len(passed_tests)
    num_failed = len(failed_tests)

    # Overall statistics
    print(f"\nTotal tests:  {total_tests}")
    print(f"Passed:       {colored(str(num_passed), Colors.OKGREEN)}")
    print(f"Failed:       {colored(str(num_failed), Colors.FAIL if num_failed > 0 else Colors.OKGREEN)}")
    print(f"Pass rate:    {colored(f'{num_passed/total_tests*100:.1f}%', Colors.OKGREEN if num_failed == 0 else Colors.WARNING)}")
    print(f"Total time:   {format_time(total_time)}")

    # Passed tests
    if passed_tests:
        print_section("Passed Tests")
        for result in passed_tests:
            print(f"  {colored('✓', Colors.OKGREEN)} {result.name:<30} ({format_time(result.duration)})")

    # Failed tests
    if failed_tests:
        print_section("Failed Tests")
        for result in failed_tests:
            print(f"  {colored('✗', Colors.FAIL)} {result.name:<30} ({format_time(result.duration)})")
            if result.error:
                # Print first line of error
                error_lines = result.error.strip().split('\n')
                print(f"      Error: {error_lines[0][:70]}")

    # Final verdict
    print("\n" + "=" * 80)
    if num_failed == 0:
        print(colored("✓ ALL TESTS PASSED!", Colors.OKGREEN + Colors.BOLD))
    else:
        print(colored(f"✗ {num_failed} TEST{'S' if num_failed > 1 else ''} FAILED!", Colors.FAIL + Colors.BOLD))
    print("=" * 80 + "\n")

    return num_failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all Zeus tests and generate a summary report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tests.py        Run all tests with summary
  python run_all_tests.py -v     Run with verbose output
  python run_all_tests.py --sgl-kernel-mode aot
  python run_all_tests.py --sgl-kernel-mode jit
  python run_all_tests.py --sgl-kernel-mode jit --sgl-kernel-jit-strict
        """
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output from each test'
    )
    parser.add_argument(
        '--sgl-kernel-mode',
        choices=('aot', 'jit'),
        default='aot',
        help='sgl-kernel-zeus dispatch mode for child tests (default: aot)'
    )
    parser.add_argument(
        '--sgl-kernel-jit-strict',
        action='store_true',
        help=(
            'In --sgl-kernel-mode jit, set ZECC_JIT_STRICT=1. '
            'Do not use this with graph-capture tests unless the op is graph-safe.'
        )
    )
    parser.add_argument(
        '--zeusv3-simulator-dir',
        default=None,
        help='Override ZEUSV3_SIMULATOR_DIR for child tests'
    )

    args = parser.parse_args()
    test_env = build_test_env(args)

    # Get test directory (same directory as this script)
    test_dir = Path(__file__).parent

    print_header("ZEUS TEST SUITE")
    print(f"Test directory: {test_dir}")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print_sgl_kernel_mode(args, test_env)

    # Find all test files
    test_files = find_test_files(test_dir)

    if not test_files:
        print(colored("\n✗ No test files found!", Colors.FAIL))
        return 1

    print(f"\nFound {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  • {test_file.name}")

    print_section(
        f"Running Tests (verbose={'ON' if args.verbose else 'OFF'}, "
        f"sgl-kernel-zeus={args.sgl_kernel_mode})"
    )

    # Run all tests
    results = []
    start_time = time.time()

    for test_file in test_files:
        result = run_test(
            test_file,
            verbose=args.verbose,
            env=env_for_test_file(test_env, args, test_file),
        )
        results.append(result)

    total_time = time.time() - start_time

    # Print summary
    all_passed = print_summary(results, total_time)

    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{colored('✗ Tests interrupted by user', Colors.WARNING)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{colored(f'✗ Unexpected error: {e}', Colors.FAIL)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
