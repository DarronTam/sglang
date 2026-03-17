"""
SGLang LLM benchmark demo on CUDA

Test 1: 3 chat prompts with TTFT / TPOT / TPS metrics
Test 2: Specified input/output length latency test
"""

import time

import sglang as sgl


def bench_prompts(llm, prompts, max_new_tokens=64):
    """Benchmark a list of prompts one by one with streaming to measure TTFT."""
    sampling_params = {"max_new_tokens": max_new_tokens, "temperature": 0}

    results = []
    for prompt in prompts:
        t_start = time.perf_counter()
        first_token_time = None

        chunks = llm.generate(prompt, sampling_params, stream=True)
        final = None
        for chunk in chunks:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            final = chunk
        t_end = time.perf_counter()

        meta = final["meta_info"]
        prompt_tokens = meta["prompt_tokens"]
        completion_tokens = meta["completion_tokens"]
        ttft = first_token_time - t_start
        decode_time = t_end - first_token_time
        # TPOT: exclude first token (already counted in TTFT)
        tpot = decode_time / max(completion_tokens - 1, 1)
        tps = completion_tokens / max(t_end - t_start, 1e-9)

        results.append({
            "prompt": prompt,
            "text": final["text"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_ms": ttft * 1000,
            "tpot_ms": tpot * 1000,
            "tps": tps,
            "e2e_ms": (t_end - t_start) * 1000,
        })

    return results


def bench_fixed_length(llm, input_len=128, output_len=128):
    """Benchmark with a prompt padded to a specific input token length."""
    # Build a prompt that tokenizes to roughly `input_len` tokens
    # "hello " is ~1-2 tokens; repeat to fill
    base = "hello " * (input_len // 1)  # overshoot, then trim via tokenizer
    sampling_params = {"max_new_tokens": output_len, "temperature": 0}

    t_start = time.perf_counter()
    first_token_time = None

    chunks = llm.generate(base, sampling_params, stream=True)
    final = None
    for chunk in chunks:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        final = chunk
    t_end = time.perf_counter()

    meta = final["meta_info"]
    prompt_tokens = meta["prompt_tokens"]
    completion_tokens = meta["completion_tokens"]
    ttft = first_token_time - t_start
    decode_time = t_end - first_token_time
    tpot = decode_time / max(completion_tokens - 1, 1)
    tps = completion_tokens / max(t_end - t_start, 1e-9)

    return {
        "input_len_target": input_len,
        "output_len_target": output_len,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_ms": ttft * 1000,
        "tpot_ms": tpot * 1000,
        "tps": tps,
        "e2e_ms": (t_end - t_start) * 1000,
    }


def print_table(rows, title):
    """Print a list of dicts as a formatted table."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    header = f"{'Prompt Tok':>10} {'Compl Tok':>10} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'TPS':>8} {'E2E(ms)':>10}"
    print(header)
    print("-" * 80)
    for r in rows:
        print(
            f"{r['prompt_tokens']:>10} "
            f"{r['completion_tokens']:>10} "
            f"{r['ttft_ms']:>10.2f} "
            f"{r['tpot_ms']:>10.2f} "
            f"{r['tps']:>8.1f} "
            f"{r['e2e_ms']:>10.2f}"
        )


if __name__ == "__main__":
    llm = sgl.Engine(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="float16",
        log_level="warning",
    )

    # ── Test 1: Chat prompts ──────────────────────────────────
    prompts = [
        "Hello, who are you?",
        "What is 1 + 1?",
        "Say 'hi' in French.",
    ]
    results = bench_prompts(llm, prompts, max_new_tokens=32)

    print_table(results, "Test 1: Chat Prompts (max_new_tokens=32)")
    for r in results:
        text_preview = r["text"].strip().replace("\n", " ")[:60]
        print(f"  >> {text_preview}...")

    # ── Test 2: Fixed input/output length ─────────────────────
    configs = [
        (128, 128),
        (256, 256),
        (8192, 8192),
    ]
    fixed_results = []
    for in_len, out_len in configs:
        res = bench_fixed_length(llm, input_len=in_len, output_len=out_len)
        fixed_results.append(res)

    print_table(fixed_results, "Test 2: Fixed Input/Output Length")

    llm.shutdown()
