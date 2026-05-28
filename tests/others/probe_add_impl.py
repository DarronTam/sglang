#!/usr/bin/env python3
"""Probe which backend implementation is used by Zeus tensor addition.

Run through ./run_probe_add_impl.sh so LD_LIBRARY_PATH and runtime debug
environment variables are set before torch_zeus is imported.
"""

import os
import sys

import torch


def to_zeus(t: torch.Tensor) -> torch.Tensor:
    return t.to("zeus")


def main() -> int:
    print("=== environment ===")
    for name in (
        "LD_LIBRARY_PATH",
        "ZEUSV3_SIMULATOR_DIR",
        "ZENL_SIM_LOADER_DEBUG",
    ):
        print(f"{name}={os.environ.get(name, '')}")

    print("\n=== import ===")
    import torch_zeus  # noqa: F401
    import torch_zeus._C as torch_zeus_c

    print(f"torch: {torch.__version__}")
    print(f"torch_zeus._C: {torch_zeus_c.__file__}")
    print(f"zeus available: {torch.zeus.is_available()}")

    if not torch.zeus.is_available():
        print("Zeus backend is not available; cannot run add probe.", file=sys.stderr)
        return 2

    print("\n=== add probe ===")
    a = torch.tensor([1.0, 2.0, 3.0, 4.5], dtype=torch.bfloat16)
    b = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.bfloat16)
    expected = a + b

    print("Moving inputs to zeus...")
    za = to_zeus(a)
    zb = to_zeus(b)

    print("Running: to_zeus(a) + to_zeus(b)")
    out = za + zb
    got = out.cpu()

    print(f"cpu expected: {expected}")
    print(f"zeus result:  {got}")
    torch.testing.assert_close(got, expected)
    print("assert_close: PASS")

    print("\n=== how to read native logs ===")
    print("- If you see '[V3 SIM] launch_npu_simulator', this add ran via Triton V3 zbin.")
    print("- If you see '[sim loader] dlopen+dlsym(zenl_add_f32_kernel_sim)', this add ran via sim.c V1 ELF.")
    print("- If neither appears, check that ZENL_SIM_LOADER_DEBUG=1 was set before import.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
