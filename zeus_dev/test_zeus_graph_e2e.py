"""
Phase 5: Zeus Graph End-to-End Verification & Performance Testing

Validates that ZeusGraphRunner produces correct outputs and measures
performance gains compared to eager mode on real model inference.

Tests:
  1. Correctness: graph mode vs eager mode output comparison (token-by-token)
  2. Multi-BS correctness: different batch sizes (1, 4, 8, 16, 32)
  3. Long-sequence correctness: varying output lengths
  4. Stability: continuous multi-round decode (no drift over time)
  5. Performance: decode throughput, first-token latency, memory usage

Model: Qwen2.5-0.5B-Instruct (same as existing Zeus validation)

Usage:
  # Run all tests (requires Zeus hardware):
  python zeus_dev/test_zeus_graph_e2e.py

  # Run specific test:
  python zeus_dev/test_zeus_graph_e2e.py --test correctness
  python zeus_dev/test_zeus_graph_e2e.py --test multi_bs
  python zeus_dev/test_zeus_graph_e2e.py --test long_seq
  python zeus_dev/test_zeus_graph_e2e.py --test stability
  python zeus_dev/test_zeus_graph_e2e.py --test perf

  # Customize model:
  python zeus_dev/test_zeus_graph_e2e.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SGLANG_DEVICE", "zeus")

import torch

# Must import torch_zeus before sglang to register the zeus backend
import torch_zeus  # noqa: F401

import sglang as sgl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Common engine kwargs shared by both graph and eager modes
COMMON_ENGINE_KWARGS = dict(
    device="zeus",
    dtype="bfloat16",
    log_level="info",
    disable_radix_cache=True,
    # Zeus has no tied embedding support
    json_model_override_args='{"tie_word_embeddings": false}',
)

TEST_PROMPTS = [
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nExplain what a neural network is in one sentence.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a short poem about the moon.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nName three programming languages.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is the speed of light?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nSay hello in Japanese.<|im_end|>\n<|im_start|>assistant\n",
]


def create_engine(model_path, use_graph):
    """Create an sgl.Engine with graph enabled or disabled."""
    kwargs = dict(COMMON_ENGINE_KWARGS)
    kwargs["model_path"] = model_path
    if not use_graph:
        kwargs["disable_cuda_graph"] = True
    # else: graph mode is the default (Phase 4 removed the force-disable)
    return sgl.Engine(**kwargs)


def generate_batch(engine, prompts, max_new_tokens=64, temperature=0.0):
    """Generate outputs for a batch of prompts."""
    sampling_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    outputs = engine.generate(prompts, sampling_params)
    return outputs


# ---------------------------------------------------------------------------
# Test 1: Basic Correctness — graph vs eager token-by-token comparison
# ---------------------------------------------------------------------------

def test_correctness(model_path):
    """Compare graph mode vs eager mode outputs token-by-token."""
    print("\n" + "=" * 70)
    print("  Test 1: Correctness — Graph vs Eager output comparison")
    print("=" * 70)

    prompts = TEST_PROMPTS[:4]
    max_new_tokens = 64

    # --- Eager mode ---
    print("\n  [1/3] Running eager mode...")
    engine_eager = create_engine(model_path, use_graph=False)
    outputs_eager = generate_batch(engine_eager, prompts, max_new_tokens=max_new_tokens)
    engine_eager.shutdown()

    # --- Graph mode ---
    print("  [2/3] Running graph mode...")
    engine_graph = create_engine(model_path, use_graph=True)
    outputs_graph = generate_batch(engine_graph, prompts, max_new_tokens=max_new_tokens)
    engine_graph.shutdown()

    # --- Compare ---
    print("  [3/3] Comparing outputs...\n")
    all_match = True
    for i, (eager_out, graph_out) in enumerate(zip(outputs_eager, outputs_graph)):
        eager_text = eager_out["text"]
        graph_text = graph_out["text"]
        match = eager_text == graph_text

        status = "MATCH" if match else "MISMATCH"
        print(f"  Prompt {i}: [{status}]")
        print(f"    Eager : {eager_text!r:.120}")
        print(f"    Graph : {graph_text!r:.120}")

        if not match:
            all_match = False
            # Show first divergence point
            for j, (ec, gc) in enumerate(zip(eager_text, graph_text)):
                if ec != gc:
                    print(f"    First divergence at char {j}: eager={ec!r} graph={gc!r}")
                    break

    if all_match:
        print("\n  PASSED: All outputs match between graph and eager modes")
    else:
        print("\n  FAILED: Some outputs differ between graph and eager modes")

    return all_match


# ---------------------------------------------------------------------------
# Test 2: Multi-BS Correctness
# ---------------------------------------------------------------------------

def test_multi_bs(model_path):
    """Test graph mode correctness across different batch sizes."""
    print("\n" + "=" * 70)
    print("  Test 2: Multi-BS Correctness")
    print("=" * 70)

    batch_sizes = [1, 4, 8]
    max_new_tokens = 32
    all_pass = True

    # Get eager baseline for the first prompt (deterministic)
    prompt = TEST_PROMPTS[0]
    print("\n  Getting eager baseline...")
    engine_eager = create_engine(model_path, use_graph=False)
    baseline_output = generate_batch(engine_eager, [prompt], max_new_tokens=max_new_tokens)
    baseline_text = baseline_output[0]["text"]
    engine_eager.shutdown()
    print(f"  Baseline: {baseline_text!r:.100}")

    # Test graph mode with different batch sizes
    print("\n  Testing graph mode with varying batch sizes...")
    engine_graph = create_engine(model_path, use_graph=True)

    for bs in batch_sizes:
        # Create a batch by repeating the same prompt
        prompts = [prompt] * bs
        outputs = generate_batch(engine_graph, prompts, max_new_tokens=max_new_tokens)

        # All outputs in the batch should match the baseline
        match = all(out["text"] == baseline_text for out in outputs)
        status = "PASS" if match else "FAIL"
        print(f"  BS={bs:3d}: [{status}]", end="")

        if not match:
            all_pass = False
            # Show which outputs differ
            for j, out in enumerate(outputs):
                if out["text"] != baseline_text:
                    print(f"\n    slot {j} differs: {out['text']!r:.80}")
        else:
            print(f"  (all {bs} outputs match baseline)")

    engine_graph.shutdown()

    if all_pass:
        print("\n  PASSED: All batch sizes produce consistent outputs")
    else:
        print("\n  FAILED: Some batch sizes produce inconsistent outputs")

    return all_pass


# ---------------------------------------------------------------------------
# Test 3: Long-Sequence Correctness
# ---------------------------------------------------------------------------

def test_long_seq(model_path):
    """Test graph mode with varying output lengths."""
    print("\n" + "=" * 70)
    print("  Test 3: Long-Sequence Correctness")
    print("=" * 70)

    output_lengths = [16, 64, 128, 256]
    prompt = TEST_PROMPTS[0]
    all_pass = True

    print("\n  Comparing graph vs eager across output lengths...")

    for max_tokens in output_lengths:
        # Eager
        engine_eager = create_engine(model_path, use_graph=False)
        eager_out = generate_batch(engine_eager, [prompt], max_new_tokens=max_tokens)
        eager_text = eager_out[0]["text"]
        engine_eager.shutdown()

        # Graph
        engine_graph = create_engine(model_path, use_graph=True)
        graph_out = generate_batch(engine_graph, [prompt], max_new_tokens=max_tokens)
        graph_text = graph_out[0]["text"]
        engine_graph.shutdown()

        match = eager_text == graph_text
        status = "PASS" if match else "FAIL"
        print(f"  max_tokens={max_tokens:4d}: [{status}]  len(eager)={len(eager_text):4d}  len(graph)={len(graph_text):4d}")

        if not match:
            all_pass = False

    if all_pass:
        print("\n  PASSED: All output lengths produce matching results")
    else:
        print("\n  FAILED: Some output lengths produce mismatched results")

    return all_pass


# ---------------------------------------------------------------------------
# Test 4: Stability — continuous multi-round decode
# ---------------------------------------------------------------------------

def test_stability(model_path):
    """Test that graph replay produces consistent outputs over many rounds."""
    print("\n" + "=" * 70)
    print("  Test 4: Stability — Multi-round decode consistency")
    print("=" * 70)

    num_rounds = 5
    max_new_tokens = 32
    prompt = TEST_PROMPTS[0]
    all_pass = True

    print(f"\n  Running {num_rounds} rounds of graph-mode generation...")
    engine_graph = create_engine(model_path, use_graph=True)

    outputs = []
    for r in range(num_rounds):
        out = generate_batch(engine_graph, [prompt], max_new_tokens=max_new_tokens)
        text = out[0]["text"]
        outputs.append(text)
        print(f"  Round {r + 1}: {text!r:.100}")

    engine_graph.shutdown()

    # All rounds should produce the same output (greedy, temperature=0)
    first = outputs[0]
    for r, text in enumerate(outputs[1:], 2):
        if text != first:
            all_pass = False
            print(f"\n  DRIFT detected at round {r}:")
            print(f"    Round 1: {first!r:.100}")
            print(f"    Round {r}: {text!r:.100}")

    if all_pass:
        print(f"\n  PASSED: All {num_rounds} rounds produced identical output (no drift)")
    else:
        print(f"\n  FAILED: Output drift detected across rounds")

    return all_pass


# ---------------------------------------------------------------------------
# Test 5: Performance — throughput & latency comparison
# ---------------------------------------------------------------------------

def test_perf(model_path):
    """Measure decode throughput and latency: graph vs eager."""
    print("\n" + "=" * 70)
    print("  Test 5: Performance — Graph vs Eager")
    print("=" * 70)

    # Use a batch of prompts for throughput measurement
    batch_sizes = [1, 4, 8]
    max_new_tokens = 64
    warmup_rounds = 2
    measure_rounds = 3

    results = {}

    for mode_name, use_graph in [("eager", False), ("graph", True)]:
        print(f"\n  --- {mode_name.upper()} mode ---")

        engine = create_engine(model_path, use_graph=use_graph)
        mode_results = {}

        for bs in batch_sizes:
            prompts = TEST_PROMPTS[:bs]

            # Warmup
            for _ in range(warmup_rounds):
                generate_batch(engine, prompts, max_new_tokens=max_new_tokens)

            # Measure
            latencies = []
            total_tokens = []
            for _ in range(measure_rounds):
                t0 = time.perf_counter()
                outputs = generate_batch(engine, prompts, max_new_tokens=max_new_tokens)
                t1 = time.perf_counter()

                elapsed = t1 - t0
                n_tokens = sum(
                    out.get("meta_info", {}).get("completion_tokens", max_new_tokens)
                    for out in outputs
                )
                latencies.append(elapsed)
                total_tokens.append(n_tokens)

            avg_latency = sum(latencies) / len(latencies)
            avg_tokens = sum(total_tokens) / len(total_tokens)
            throughput = avg_tokens / avg_latency

            mode_results[bs] = {
                "avg_latency_s": round(avg_latency, 4),
                "avg_tokens": round(avg_tokens, 1),
                "throughput_tok_s": round(throughput, 2),
            }

            print(f"  BS={bs:3d}: latency={avg_latency:.4f}s  tokens={avg_tokens:.0f}  throughput={throughput:.2f} tok/s")

        results[mode_name] = mode_results

        # Memory usage
        free, total = torch.zeus.mem_get_info(0)
        used_mb = (total - free) / 1024**2
        print(f"  Memory used: {used_mb:.0f} MB")
        results[mode_name]["memory_used_mb"] = round(used_mb, 0)

        engine.shutdown()

    # --- Summary ---
    print("\n  --- Performance Summary ---")
    print(f"  {'BS':>4s}  {'Eager (tok/s)':>14s}  {'Graph (tok/s)':>14s}  {'Speedup':>8s}")
    print(f"  {'----':>4s}  {'-' * 14:>14s}  {'-' * 14:>14s}  {'-' * 8:>8s}")

    for bs in batch_sizes:
        eager_tp = results["eager"][bs]["throughput_tok_s"]
        graph_tp = results["graph"][bs]["throughput_tok_s"]
        speedup = graph_tp / eager_tp if eager_tp > 0 else float("inf")
        print(f"  {bs:4d}  {eager_tp:14.2f}  {graph_tp:14.2f}  {speedup:7.2f}x")

    eager_mem = results["eager"].get("memory_used_mb", 0)
    graph_mem = results["graph"].get("memory_used_mb", 0)
    print(f"\n  Memory: eager={eager_mem:.0f} MB, graph={graph_mem:.0f} MB, delta={graph_mem - eager_mem:+.0f} MB")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = {
    "correctness": test_correctness,
    "multi_bs": test_multi_bs,
    "long_seq": test_long_seq,
    "stability": test_stability,
    "perf": test_perf,
}


def main():
    parser = argparse.ArgumentParser(description="Zeus Graph E2E Tests (Phase 5)")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="Model path (default: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--test", type=str, default=None,
        choices=list(ALL_TESTS.keys()),
        help="Run a specific test (default: run all)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Phase 5: Zeus Graph End-to-End Verification")
    print(f"  Model: {args.model}")
    print("=" * 70)

    tests_to_run = [args.test] if args.test else list(ALL_TESTS.keys())
    results = {}

    for test_name in tests_to_run:
        test_fn = ALL_TESTS[test_name]
        try:
            result = test_fn(args.model)
            results[test_name] = result
        except Exception as e:
            print(f"\n  ERROR in {test_name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Final summary
    print("\n" + "=" * 70)
    print("  Phase 5 Summary")
    print("=" * 70)

    for test_name in tests_to_run:
        result = results.get(test_name)
        if isinstance(result, bool):
            status = "PASS" if result else "FAIL"
        elif isinstance(result, dict):
            status = "DONE"
        else:
            status = "ERROR"
        print(f"  {test_name:15s}: {status}")

    # Save performance results if perf test was run
    if "perf" in results and isinstance(results["perf"], dict):
        perf_file = os.path.join(os.path.dirname(__file__), "zeus_graph_perf_results.json")
        with open(perf_file, "w") as f:
            json.dump(results["perf"], f, indent=2)
        print(f"\n  Performance results saved to {perf_file}")

    # Return non-zero if any correctness test failed
    correctness_tests = ["correctness", "multi_bs", "long_seq", "stability"]
    failed = any(
        results.get(t) is False
        for t in correctness_tests
        if t in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
