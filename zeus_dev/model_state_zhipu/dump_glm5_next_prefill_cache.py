"""
Dump GLM5-next 16B prefill KV / KDA state to disk so it can later be
injected into a decode-only Zeus Engine (see test_zeus_decode_only_llm.py).

Run on a CUDA host with enough memory to host the full 16B model:

    python zeus_dev/dump_glm5_next_prefill_cache.py

Produces ``/tmp/glm5_next_16b_prefill_dump.pt``. Copy that file to the Zeus
host before running ``test_zeus_decode_only_llm.py``.

NOTE: We deliberately do not set SGLANG_DEVICE=zeus and do not import
torch_zeus. This is the CUDA path.
"""

import os

DUMP_PATH = "/workspace/maoxuecheng/glm5_next_16b_prefill_dump.pt"
MODEL_PATH = "/infra/Linear/16b_hf/"
PROMPT = "中国的首都是"

# Only snapshot KV / KDA state for layers in ``[0, NUM_LAYERS)``. Set to 0 to
# dump all layers. Keeping this small (e.g. 2-8) makes dev iteration fast —
# the full 16B has 92 layers and a full dump is huge / slow. The decode-only
# test script MUST set the same NUM_LAYERS, otherwise the fingerprint
# assertion in the inject hook fails.
NUM_LAYERS = 2

# Hand the dump path + layer cap to the scheduler subprocess via env vars.
os.environ["ZEUS_DECODE_DUMP_PATH"] = DUMP_PATH
os.environ["ZEUS_DECODE_NUM_LAYERS"] = str(NUM_LAYERS)


def _run_scheduler_with_dump_hook(*args, **kwargs):
    """Subprocess entry: install the dump hook, then start the scheduler."""
    # The subprocess is spawned (not forked), so sys.path here doesn't see
    # zeus_dev/. zeus_dev/ also isn't a Python package — no __init__.py —
    # so we put zeus_dev/ itself on sys.path and import the helper by name.
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)

    import _decode_only_hooks as h
    from sglang.srt.managers.scheduler import run_scheduler_process

    h.install_dump_hook(os.environ["ZEUS_DECODE_DUMP_PATH"])
    run_scheduler_process(*args, **kwargs)


def main():
    # Import the real Engine class (sglang.Engine is a LazyImport wrapper
    # that can't be subclassed).
    from sglang.srt.entrypoints.engine import Engine

    print("=" * 60)
    print("GLM5-next 16B prefill cache dump (CUDA)")
    print("=" * 60)
    print(f"  MODEL_PATH  = {MODEL_PATH}")
    print(f"  PROMPT      = {PROMPT!r}")
    print(f"  DUMP_PATH   = {DUMP_PATH}")
    print(f"  NUM_LAYERS  = {NUM_LAYERS} (0 = all)")
    print()

    if not os.path.exists(MODEL_PATH):
        print(f"[SKIP] MODEL_PATH not found: {MODEL_PATH}")
        print("[SKIP] This CUDA dump script requires the 16B model path.")
        return

    class _DumpEngine(Engine):
        run_scheduler_process_func = staticmethod(_run_scheduler_with_dump_hook)

    llm = _DumpEngine(
        model_path=MODEL_PATH,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        # NSA models (GLM5-next is one — Glm5NextForCausalLM is on the
        # NSA arch whitelist in model_config.is_deepseek_nsa) hard-require
        # page_size == 64 in NSATokenToKVPool (memory_pool.py:1970).
        # Zeus side has the same constraint (hardware_backend/zeus/utils.py).
        page_size=64,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        # Disable the shared-experts fusion optimization: glm5_next.py:1146-1148
        # asserts n_shared_experts == 1 when fusion is enabled, which the 16B
        # config doesn't satisfy. We don't need the perf in a dev test.
        disable_shared_experts_fusion=True,
        # GLM5-next is an NSA model. Leaving attention_backend unset lets
        # server_args._handle_attention_backend_* auto-pick "nsa" (CUDA
        # path; server_args.py:1567-1569) — matches what the decode-side
        # Zeus engine auto-picks ("zeus_mla"). Forcing "triton" here
        # would bypass NSA's sparse-attn metadata setup and break.
        # attention_backend left unset.
        mem_fraction_static=0.5,
        max_running_requests=1,
        log_level="info",
    )

    out = llm.generate(
        [PROMPT],
        {"max_new_tokens": 1, "temperature": 0.0},
    )
    print()
    print("[DUMP] prefill output:", out)

    llm.shutdown()

    if os.path.exists(DUMP_PATH):
        size_mb = os.path.getsize(DUMP_PATH) / (1 << 20)
        print(f"[DUMP] dump saved to {DUMP_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"[DUMP] WARNING: {DUMP_PATH} not found — dump hook may not have fired")


if __name__ == "__main__":
    main()
