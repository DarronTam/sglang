"""
CUDA LLM benchmark demo for the current Torch/SGLang environment.

This script intentionally does not use sglang.Engine. The current prerelease
SGLang import path eagerly imports quantization modules, which can require
legacy sgl_kernel symbols even when the demo itself does not use quantization.
For a plain CUDA inference sanity check, use Transformers + torch.cuda directly.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class BenchResult:
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    tpot_ms: float
    tps: float
    e2e_ms: float
    text: str = ""


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"auto", "float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model(model_path: str, dtype: str, local_files_only: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This demo is for CUDA inference.")

    torch_dtype = get_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.eval()
    model.to("cuda")
    return tokenizer, model


def format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def generate_once(tokenizer, model, prompt: str, max_new_tokens: int) -> BenchResult:
    formatted = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    sync_cuda()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    sync_cuda()
    t1 = time.perf_counter()

    generated_ids = output_ids[0, prompt_tokens:]
    completion_tokens = int(generated_ids.numel())
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    e2e = t1 - t0
    # Transformers generate() is not streaming here, so exact TTFT is unavailable.
    # Use full-request latency as TTFT upper bound and report TPOT over all output tokens.
    ttft = e2e
    tpot = e2e / max(completion_tokens, 1)
    tps = completion_tokens / max(e2e, 1e-9)

    return BenchResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft * 1000,
        tpot_ms=tpot * 1000,
        tps=tps,
        e2e_ms=e2e * 1000,
        text=text,
    )


def make_fixed_prompt(tokenizer, target_tokens: int) -> str:
    words = []
    text = ""
    while True:
        words.append("hello")
        text = " ".join(words)
        token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
        if token_count >= target_tokens:
            return text


def bench_prompts(tokenizer, model, prompts: Iterable[str], max_new_tokens: int) -> List[BenchResult]:
    return [generate_once(tokenizer, model, prompt, max_new_tokens) for prompt in prompts]


def bench_fixed_length(tokenizer, model, input_len: int, output_len: int) -> BenchResult:
    prompt = make_fixed_prompt(tokenizer, input_len)
    return generate_once(tokenizer, model, prompt, output_len)


def print_table(rows: List[BenchResult], title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print(
        f"{'Prompt Tok':>10} {'Compl Tok':>10} {'TTFT*(ms)':>10} "
        f"{'TPOT(ms)':>10} {'TPS':>8} {'E2E(ms)':>10}"
    )
    print("-" * 80)
    for r in rows:
        print(
            f"{r.prompt_tokens:>10} "
            f"{r.completion_tokens:>10} "
            f"{r.ttft_ms:>10.2f} "
            f"{r.tpot_ms:>10.2f} "
            f"{r.tps:>8.1f} "
            f"{r.e2e_ms:>10.2f}"
        )
    print("* TTFT is an upper bound because this direct Transformers path is non-streaming.")


def parse_args():
    parser = argparse.ArgumentParser(description="CUDA LLM inference demo using torch + transformers.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32", "auto"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--fixed", action="store_true", help="Run fixed input/output length benchmarks.")
    parser.add_argument("--fixed-config", action="append", default=[], help="Format: input_len,output_len. Can be passed multiple times.")
    parser.add_argument("--local-files-only", action="store_true", help="Only load model files from local HuggingFace cache or local model path.")
    parser.add_argument("--hf-endpoint", default=None, help="Optional HuggingFace endpoint or mirror, for example https://hf-mirror.com.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("CUDA LLM demo")
    print("=" * 80)
    print(f"torch          : {torch.__version__}")
    print(f"torch cuda     : {torch.version.cuda}")
    print(f"cuda available : {torch.cuda.is_available()}")
    print(f"device         : {torch.cuda.get_device_name(0)}")
    print(f"model          : {args.model_path}")
    print(f"dtype          : {args.dtype}")
    print(f"local only     : {args.local_files_only}")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print(f"hf endpoint    : {args.hf_endpoint}")

    try:
        tokenizer, model = load_model(args.model_path, args.dtype, args.local_files_only)
    except Exception as exc:
        print("\nModel load failed.")
        print(f"  error: {type(exc).__name__}: {exc}")
        print("\nHints:")
        print("  1. Use a local model path: --model-path /path/to/model")
        print("  2. Use local cache only: --local-files-only")
        print("  3. Use a reachable mirror: --hf-endpoint https://hf-mirror.com")
        print("  4. Check proxy variables: HTTP_PROXY / HTTPS_PROXY")
        raise

    prompts = [
        "Hello, who are you?",
        "What is 1 + 1?",
        "Say 'hi' in French.",
    ]
    rows = bench_prompts(tokenizer, model, prompts, args.max_new_tokens)
    print_table(rows, f"Test 1: Chat Prompts (max_new_tokens={args.max_new_tokens})")
    for row in rows:
        preview = row.text.strip().replace("\n", " ")[:80]
        print(f"  >> {preview}...")

    if args.fixed:
        configs = []
        if args.fixed_config:
            for item in args.fixed_config:
                in_len, out_len = item.split(",", 1)
                configs.append((int(in_len), int(out_len)))
        else:
            configs = [(128, 128), (256, 256)]

        fixed_rows = [bench_fixed_length(tokenizer, model, i, o) for i, o in configs]
        print_table(fixed_rows, "Test 2: Fixed Input/Output Length")


if __name__ == "__main__":
    main()
