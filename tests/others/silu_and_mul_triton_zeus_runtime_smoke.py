#!/usr/bin/env python3
"""Smoke test for CUDA-Triton-style Zeus silu_and_mul launch syntax.

Run from the repo root:
    python tests/others/silu_and_mul_triton_zeus_runtime_smoke.py
"""

from __future__ import annotations

import torch
import torch_zeus  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def sgl_silu_and_mul_kernel_bf16(
    output_ptr,
    input_ptr,
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

    x_base = tl.make_block_ptr(
        base=input_ptr,
        shape=(num_tokens, 2 * dim),
        strides=(2 * dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_TOKENS, BLOCK_DIM),
        order=(1, 0),
    )
    y_base = tl.make_block_ptr(
        base=input_ptr,
        shape=(num_tokens, 2 * dim),
        strides=(2 * dim, 1),
        offsets=(0, dim),
        block_shape=(BLOCK_TOKENS, BLOCK_DIM),
        order=(1, 0),
    )
    out_base = tl.make_block_ptr(
        base=output_ptr,
        shape=(num_tokens, dim),
        strides=(dim, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_TOKENS, BLOCK_DIM),
        order=(1, 0),
    )

    total_col_blocks = tl.cdiv(dim, BLOCK_DIM)
    for row_block_idx in range(blocks_to_process):
        t_start = (start_block + row_block_idx) * BLOCK_TOKENS
        for col_block_idx in range(total_col_blocks):
            d_start = col_block_idx * BLOCK_DIM

            x = tl.load(
                tl.advance(x_base, (t_start, d_start)),
                boundary_check=(0, 1),
                padding_option="zero",
            ).to(tl.float32)
            y = tl.load(
                tl.advance(y_base, (t_start, d_start)),
                boundary_check=(0, 1),
                padding_option="zero",
            ).to(tl.float32)
            out = (x * tl.zeus.sigmoid(x) * y).to(tl.bfloat16)
            tl.store(
                tl.advance(out_base, (t_start, d_start)),
                out,
                boundary_check=(0, 1),
            )


def silu_and_mul(input, out=None):
    if input.ndim < 2:
        raise ValueError("silu_and_mul expects at least 2 dimensions")
    if input.shape[-1] % 2 != 0:
        raise ValueError("silu_and_mul expects an even last dimension")
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")

    dim = input.shape[-1] // 2
    if out is None:
        out = torch.empty(input.shape[:-1] + (dim,), dtype=input.dtype, device=input.device)
    else:
        assert out.shape == input.shape[:-1] + (dim,), (out.shape, input.shape)
        assert out.dtype == input.dtype, (out.dtype, input.dtype)
        assert out.device == input.device, (out.device, input.device)

    num_tokens = input.numel() // input.shape[-1]
    block_dim = 64 if dim <= 64 else 128
    sgl_silu_and_mul_kernel_bf16[(2,)](
        out,
        input,
        num_tokens,
        dim,
        CORE_NUM=2,
        BLOCK_TOKENS=64,
        BLOCK_DIM=block_dim,
    )
    return out


def reference_silu_and_mul(input):
    input_f32 = input.cpu().float()
    dim = input_f32.shape[-1] // 2
    ref = torch.nn.functional.silu(input_f32[..., :dim]) * input_f32[..., dim:]
    return ref.to(torch.bfloat16)


def main() -> int:
    torch.manual_seed(20260505)
    x = torch.randn(48, 5120, dtype=torch.bfloat16, device="zeus")
    out = silu_and_mul(x)
    ref = reference_silu_and_mul(x)

    torch.testing.assert_close(out.cpu().float(), ref.float(), rtol=1e-2, atol=2e-2)

    print("[silu_and_mul]\nSim: ", out.cpu(), "\nRef: ", ref)
    print("PASS: Zeus Triton silu_and_mul runtime smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
