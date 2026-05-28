#!/usr/bin/env python3
"""Smoke test for copy_uint8 Zeus Triton kernel.

Run from the repo root:
    python tests/others/copy_uint8_smoke.py
"""

from __future__ import annotations

import torch
import torch_zeus  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def copy_uint8(
    dst_ptr,         # destination pointer, *uint8, [nbytes]
    src_ptr,         # source pointer, *uint8, [nbytes]
    nbytes,          # total bytes, i32
    BLOCK: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    core_id = tl.program_id(0)
    total_blocks = tl.cdiv(nbytes, BLOCK)

    base_blocks_per_core = total_blocks // CORE_NUM
    remainder = total_blocks % CORE_NUM
    blocks_to_process = base_blocks_per_core + tl.where(core_id < remainder, 1, 0)
    start_block = core_id * base_blocks_per_core + tl.minimum(core_id, remainder)

    src_ptr_template = tl.make_block_ptr(
        base=src_ptr, shape=(nbytes,), strides=(1,), offsets=(0,),
        block_shape=(BLOCK,), order=(0,)
    )
    dst_ptr_template = tl.make_block_ptr(
        base=dst_ptr, shape=(nbytes,), strides=(1,), offsets=(0,),
        block_shape=(BLOCK,), order=(0,)
    )

    for i in range(blocks_to_process):
        block_id = start_block + i
        offset = block_id * BLOCK
        data = tl.load(tl.advance(src_ptr_template, (offset,)),
                       boundary_check=(0,), padding_option="zero")
        tl.store(tl.advance(dst_ptr_template, (offset,)), data, boundary_check=(0,))


BLOCK = 1024
CORE_NUM = 2


def copy(src: torch.Tensor) -> torch.Tensor:
    src_u8 = src.view(torch.uint8)
    dst_u8 = torch.empty_like(src_u8)
    nbytes = src_u8.numel()
    copy_uint8[(CORE_NUM,)](
        dst_u8,
        src_u8,
        nbytes,
        BLOCK=BLOCK,
        CORE_NUM=CORE_NUM,
    )
    return dst_u8.view(src.dtype)


def main() -> int:
    torch.manual_seed(20260505)

    # bfloat16: 2048 elements = 4096 bytes, 4 blocks of 1024
    x_bf16 = torch.randn(2048, dtype=torch.bfloat16, device="zeus")
    out_bf16 = copy(x_bf16)
    torch.testing.assert_close(out_bf16.cpu(), x_bf16.cpu(), rtol=0, atol=0)
    print("[Copy bfloat16]\nSrc:", x_bf16.cpu(), "\nDst:", out_bf16.cpu())

    # float32: 1024 elements = 4096 bytes, 4 blocks of 1024
    x_f32 = torch.randn(1024, dtype=torch.float32, device="zeus")
    out_f32 = copy(x_f32)
    torch.testing.assert_close(out_f32.cpu(), x_f32.cpu(), rtol=0, atol=0)
    print("[Copy float32]\nSrc:", x_f32.cpu(), "\nDst:", out_f32.cpu())

    # uint8 native: 2048 bytes, 2 blocks of 1024
    x_u8 = torch.randint(0, 256, (2048,), dtype=torch.uint8, device="zeus")
    dst_u8 = torch.empty_like(x_u8)
    copy_uint8[(CORE_NUM,)](dst_u8, x_u8, x_u8.numel(), BLOCK=BLOCK, CORE_NUM=CORE_NUM)
    torch.testing.assert_close(dst_u8.cpu(), x_u8.cpu(), rtol=0, atol=0)
    print("[Copy uint8]\nSrc:", x_u8.cpu(), "\nDst:", dst_u8.cpu())

    print("PASS: copy_uint8 Zeus Triton smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
