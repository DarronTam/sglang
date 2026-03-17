"""
SGLang + Zeus device: Engine init milestone demo

Milestones:
  M1: DeviceConfig recognizes zeus           [PASS]
  M2: ServerArgs accepts zeus, auto-detect   [PASS]
  M3: Engine process spawning                [PASS]
  M4: init_torch_distributed()               [PASS]
  M5: Pre-weight-loading checks pass         [PASS]
  M6: Weight loading (needs localmem loader)  [PASS - CPU fallback]
"""

import torch

# Must import torch_zeus BEFORE sglang to register the zeus backend
import torch_zeus


def main():
    print("=" * 60)
    print("Zeus device info")
    print("=" * 60)
    print(f"  torch_zeus version : {torch_zeus.__version__}")
    print(f"  C++ ext loaded     : {torch_zeus._has_cpp_ext}")
    print(f"  device_count       : {torch.zeus.device_count()}")
    print(f"  is_available       : {torch.zeus.is_available()}")
    if torch.zeus.is_available():
        print(f"  device name        : {torch.zeus.get_device_name(0)}")
        free, total = torch.zeus.mem_get_info(0)
        print(f"  memory             : {free // 1024**2} MB free / {total // 1024**2} MB total")

    print()
    print("=" * 60)
    print("SGLang integration checks")
    print("=" * 60)

    # M1: DeviceConfig
    from sglang.srt.configs.device_config import DeviceConfig

    cfg = DeviceConfig(device="zeus")
    print(f"  M1 DeviceConfig    : device_type={cfg.device_type}")

    # M2: Utility functions
    from sglang.srt.utils import (
        get_available_gpu_memory,
        get_device_memory_capacity,
        is_zeus,
    )

    print(f"  M2 is_zeus()       : {is_zeus()}")
    mem_cap = get_device_memory_capacity("zeus")
    print(f"  M2 memory capacity : {mem_cap} MB")
    avail = get_available_gpu_memory("zeus", 0)
    print(f"  M2 avail memory    : {avail:.2f} GB")

    # M3-M5: Engine init
    print()
    print("=" * 60)
    print("Attempting Engine init (device=zeus)...")
    print("=" * 60)

    import sglang as sgl

    try:
        llm = sgl.Engine(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            device="zeus",
            dtype="bfloat16",
            log_level="info",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            # Zeus has no tied embedding support; use separate lm_head weight
            # so embed_tokens stays in GDG (fast gather) and lm_head goes to
            # LocalMem (fast GEMM).
            json_model_override_args='{"tie_word_embeddings": false}',
        )
        print("  Engine init succeeded!")

        # M7: generate() — requires attention backend + sampling
        print()
        print("=" * 60)
        print("Attempting generate()...")
        print("=" * 60)
        try:
            # Qwen Instruct models respond better to explicit prompt formatting.
            # Using an unformatted string like "Hello, my name is" can confuse the model
            # or cause it to output unstructured text indefinitely until it hits a max_new_tokens limit.
            # To ensure the model knows it is answering a user query and should output a concise response,
            # we format it using ChatML or instruct format. We also set a hard stop limit.
            
            prompts = [
                "<|im_start|>user\nWhat is the capital of France? Please answer in one word.<|im_end|>\n<|im_start|>assistant\n"
            ]
            
            sampling_params = {
                "max_new_tokens": 32, # limit the max length just in case
                "temperature": 0.0,   # greedy decoding for determinism
            }
            
            print(f"  Input prompt: {prompts[0]!r}")
            out = llm.generate(prompts, sampling_params)
            print(f"  generate() succeeded!")
            print(f"  output: {out}")
        except Exception as e:
            print(f"  generate() failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        llm.shutdown()
    except Exception as e:
        print(f"  Engine init stopped at: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
