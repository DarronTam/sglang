#!/usr/bin/env python3
"""Smoke test for CUDA-Triton-style Zeus launch syntax.

Run from the repo root:
    python tests/others/add_triton_zeus_runtime_smoke.py
"""

from __future__ import annotations

import torch
import torch_zeus  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    core_id = tl.program_id(0)
    per_core = tl.cdiv(n_elements, CORE_NUM)
    start = core_id * per_core
    x_base = tl.make_block_ptr(
        base=x_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(start,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    y_base = tl.make_block_ptr(
        base=y_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(start,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    out_base = tl.make_block_ptr(
        base=out_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(start,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    for block_idx in tl.range(tl.cdiv(per_core, BLOCK)):
        offset = block_idx * BLOCK
        x = tl.load(tl.advance(x_base, (offset,)), boundary_check=(0,))
        y = tl.load(tl.advance(y_base, (offset,)), boundary_check=(0,))
        out = (x.to(tl.float32) + y.to(tl.float32)).to(tl.bfloat16)
        tl.store(tl.advance(out_base, (offset,)), out, boundary_check=(0,))


def add(x, y):
    out = torch.empty_like(x)
    add_kernel[(2,)](
        x,
        y,
        out,
        out.numel(),
        BLOCK=1024,
        CORE_NUM=2,
    )
    return out


@triton.jit
def inplace_add_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    core_id = tl.program_id(0)
    per_core = tl.cdiv(n_elements, CORE_NUM)
    start = core_id * per_core
    x_base = tl.make_block_ptr(
        base=x_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(start,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    y_base = tl.make_block_ptr(
        base=y_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(start,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    for block_idx in tl.range(tl.cdiv(per_core, BLOCK)):
        offset = block_idx * BLOCK
        x = tl.load(tl.advance(x_base, (offset,)), boundary_check=(0,))
        y = tl.load(tl.advance(y_base, (offset,)), boundary_check=(0,))
        out = (x.to(tl.float32) + y.to(tl.float32)).to(tl.bfloat16)
        tl.store(tl.advance(x_base, (offset,)), out, boundary_check=(0,))


def add_inplace(x, y):
    inplace_add_kernel[(2,)](
        x,
        y,
        x.numel(),
        BLOCK=1024,
        CORE_NUM=2,
    )
    return x


def main() -> int:
    torch.manual_seed(20260505)
    x = torch.randn(2048, dtype=torch.bfloat16, device="zeus")
    y = torch.randn(2048, dtype=torch.bfloat16, device="zeus")
    out = add(x, y)
    ref = (x.cpu().float() + y.cpu().float()).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
    print("[Normal Add]\nSim: ", out.cpu(), "\nRef: ", ref)

    x_inplace = torch.randn(2048, dtype=torch.bfloat16, device="zeus")
    y_inplace = torch.randn(2048, dtype=torch.bfloat16, device="zeus")
    inplace_ref = (x_inplace.cpu().float() + y_inplace.cpu().float()).to(torch.bfloat16)
    add_inplace(x_inplace, y_inplace)
    torch.testing.assert_close(x_inplace.cpu(), inplace_ref, rtol=0, atol=0)
    print("[Normal Add]\nSim: ", x_inplace.cpu(), "\nRef: ", inplace_ref)
    print("PASS: Zeus Triton runtime smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
