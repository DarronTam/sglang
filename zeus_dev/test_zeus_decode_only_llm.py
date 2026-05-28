"""
Zeus decode-only Engine test for GLM5-next 16B.

Simulates the DECODE side of a PD-disaggregated deployment:

    1. Engine init with disaggregation_mode="decode" +
       disaggregation_transfer_backend="fake".
    2. The scheduler subprocess installs a monkey-patch on
       ``FakeKVReceiver.send_metadata`` that loads the dump file produced by
       ``dump_glm5_next_prefill_cache.py`` and writes the prefill KV / KDA
       state into the live pool via the production set_kv_buffer path
       (``ZeusTokenToKVPool.set_kv_buffer`` → ``sgl_kernel_zeus.store_kv_cache``).
    3. The scheduler treats the request as PREBUILT (skips prefill forward)
       and runs only decode.
    4. We call ``llm.generate(prompt)`` and print the generated tokens.

Prerequisite: ``/tmp/glm5_next_16b_prefill_dump.pt`` must exist on the host.
Generate it with ``zeus_dev/dump_glm5_next_prefill_cache.py`` on a CUDA host
and copy over.

NOTE: SGLANG_DEVICE=zeus and ``import torch_zeus`` must happen BEFORE any
``sglang`` import, to ensure ``is_zeus()`` resolves True everywhere (see
``demo_zeus_llm.py`` header).
"""

import os

# Must be set before importing sglang.
os.environ["SGLANG_DEVICE"] = "zeus"

import torch_zeus  # noqa: F401  — registers zeus backend

DUMP_PATH = "/workspace/maoxuecheng/glm5_next_16b_prefill_dump.pt"
MODEL_PATH = "/infra/Linear/16b_hf/"
PROMPT = "中国的首都是"  # MUST match dump_glm5_next_prefill_cache.py

# MUST match NUM_LAYERS in dump_glm5_next_prefill_cache.py. Only KV / KDA
# state for layers ``[0, NUM_LAYERS)`` will be injected; the rest stays
# zero-initialized in the pool. 0 = all layers (matches a full dump).
NUM_LAYERS = 2

os.environ["ZEUS_DECODE_DUMP_PATH"] = DUMP_PATH
os.environ["ZEUS_DECODE_NUM_LAYERS"] = str(NUM_LAYERS)

assert os.path.exists(DUMP_PATH), (
    f"先在 CUDA 上跑 dump_glm5_next_prefill_cache.py 生成 {DUMP_PATH}，"
    f"再把文件拷到 Zeus host 同样路径"
)


def _run_scheduler_with_inject_hook(*args, **kwargs):
    """Subprocess entry: install the inject hook, then start the scheduler."""
    # Spawned subprocess: zeus_dev/ isn't a Python package, so put it on
    # sys.path here and import the helper by bare module name.
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)

    import _decode_only_hooks as h
    from sglang.srt.managers.scheduler import run_scheduler_process

    h.install_inject_hook(os.environ["ZEUS_DECODE_DUMP_PATH"])
    run_scheduler_process(*args, **kwargs)


def main():
    # Import the real Engine class (sglang.Engine is a LazyImport wrapper
    # that can't be subclassed).
    from sglang.srt.entrypoints.engine import Engine

    print("=" * 60)
    print("Zeus decode-only test (GLM5-next 16B + fake KV transfer)")
    print("=" * 60)
    print(f"  MODEL_PATH  = {MODEL_PATH}")
    print(f"  PROMPT      = {PROMPT!r}")
    print(f"  DUMP_PATH   = {DUMP_PATH}")
    print(f"  NUM_LAYERS  = {NUM_LAYERS} (0 = all; must match dump)")
    print()

    class _DecodeOnlyEngine(Engine):
        run_scheduler_process_func = staticmethod(
            _run_scheduler_with_inject_hook
        )

    llm = _DecodeOnlyEngine(
        model_path=MODEL_PATH,
        device="zeus",
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        # NSA model → NSATokenToKVPool requires page_size == 64 (both
        # sides). Zeus hardware_backend/zeus/utils.py also forces this
        # when is_nsa is True.
        page_size=64,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        # Same as dump side: avoid glm5_next.py:1146-1148 assert on
        # n_shared_experts == 1.
        disable_shared_experts_fusion=True,
        # Let Zeus auto-select: NSA models -> "zeus_mla" (will trigger
        # missing sparse_mla_paged_zeus kernel ImportError, which is the
        # known TODO we want to surface). Forcing "zeus" here would
        # conflict with page_size=64 (server_args asserts % 128 == 0 for
        # the non-NSA Zeus path).
        # attention_backend left unset (None) — _handle_zeus_backends picks.
        attention_backend="zeus",
        mem_fraction_static=0.5,
        max_running_requests=1,
        disaggregation_mode="decode",
        disaggregation_transfer_backend="fake",
        log_level="info",
    )

    out = llm.generate(
        [PROMPT],
        {"max_new_tokens": 16, "temperature": 0.0},
    )
    print()
    print("[DECODE-ONLY] output:", out)

    llm.shutdown()
    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
