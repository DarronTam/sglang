#!/usr/bin/env python3
"""One-file smoke test for Zeus Triton JIT silu_and_mul.

Run from the repo root:
    python tests/others/silu_and_mul_triton_jit_smoke.py
"""

from __future__ import annotations

from sgl_kernel_zeus import jit
import triton
import triton.language as tl


@triton.jit
def sgl_silu_and_mul_kernel_bf16(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    num_tokens,
    dim,
    CORE_NUM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    core_id = tl.program_id(axis=0)

    total_row_blocks = tl.cdiv(num_tokens, BLOCK_TOKENS)
    base_blocks_per_core = total_row_blocks // CORE_NUM
    remainder = total_row_blocks % CORE_NUM
    blocks_to_process = base_blocks_per_core + tl.where(core_id < remainder, 1, 0)
    start_block = core_id * base_blocks_per_core + tl.minimum(core_id, remainder)

    x_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(num_tokens, 2 * dim),
        strides=(2 * dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_TOKENS, BLOCK_DIM),
        order=(1, 0),
    )
    y_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(num_tokens, 2 * dim),
        strides=(2 * dim, 1),
        offsets=(0, dim),
        block_shape=(BLOCK_TOKENS, BLOCK_DIM),
        order=(1, 0),
    )
    out_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(num_tokens, dim),
        strides=(dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_TOKENS, BLOCK_DIM),
        order=(1, 0),
    )

    total_col_blocks = tl.cdiv(dim, BLOCK_DIM)
    for i in range(blocks_to_process):
        row = (start_block + i) * BLOCK_TOKENS
        for j in range(total_col_blocks):
            col = j * BLOCK_DIM
            x = tl.load(
                tl.advance(x_ptr, (row, col)),
                boundary_check=(0, 1),
                padding_option="zero",
            ).to(tl.float32)
            y = tl.load(
                tl.advance(y_ptr, (row, col)),
                boundary_check=(0, 1),
                padding_option="zero",
            ).to(tl.float32)
            out = (x * tl.zeus.sigmoid(x) * y).to(tl.bfloat16)
            tl.store(
                tl.advance(out_ptr, (row, col)),
                out,
                boundary_check=(0, 1),
            )


def reference_silu_and_mul(x):
    import torch

    x_fp32 = x.cpu().float()
    dim = x_fp32.shape[-1] // 2
    return (torch.nn.functional.silu(x_fp32[:, :dim]) * x_fp32[:, dim:]).to(x.dtype)


def silu_and_mul_jit(input, out=None, *, enable_jit=True):
    import torch

    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        assert input.ndim == out.ndim, f"{input.ndim} != {out.ndim}"
        assert input.shape[:-1] == out.shape[:-1], f"{input.shape[:-1]} != {out.shape[:-1]}"
        assert input.shape[-1] == 2 * out.shape[-1], f"{input.shape[-1]} != {2 * out.shape[-1]}"
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )

    num_tokens = input.numel() // input.shape[-1]
    dim = out.shape[-1]
    sgl_silu_and_mul_kernel_bf16[(2,)](
        out,
        input,
        num_tokens,
        dim,
        CORE_NUM=2,
        BLOCK_TOKENS=64,
        BLOCK_DIM=64 if dim <= 64 else 128,
        zeus_enable_jit=enable_jit,
        zeus_strict=True,
        zeus_log=True,
    )
    return out


def run_silu_and_mul_check(*, enable_jit=True) -> None:
    import torch
    import torch_zeus  # noqa: F401

    torch.manual_seed(20260505)
    x = torch.randn(2, 256, dtype=torch.bfloat16, device="privateuseone")
    out = silu_and_mul_jit(x, enable_jit=enable_jit)
    ref = reference_silu_and_mul(x)
    torch.testing.assert_close(out.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=2e-2)


def main() -> int:
    run_silu_and_mul_check(enable_jit=True)
    print("PASS: silu_and_mul Python-level Zeus JIT launcher smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
