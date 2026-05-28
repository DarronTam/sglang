#!/usr/bin/env python3
"""Smoke test for Zeus Triton JIT bf16 GEMM with LocalMem weight.

Weight is created as a plain [N, K] tensor and packed into LocalMem via
torch.zeus.local_memory.from_tensor() which returns a ZeusLocalMemTensor.

The kernel follows the same pattern as gemm_bf16_dense_kernel in zenl:
  mat2_ptr: *i64  (weight descriptor; zecc_jit emits localmem_weight_ptr)
  weight_addr = tl.load(mat2_ptr + core_id)   # per-core weight base (i64)
  weight_ptr  = weight_addr.to(*bf16)
  b_base      = tl.make_block_ptr(weight_ptr, ...)
  b_nk        = tl.load(tl.advance(b_base, ...), memory_type='weight')

CPU reference uses the original (pre-pack) weight and torch.nn.functional.linear.

Run from the repo root (torch10_312 env):
    conda run -n torch10_312 python tests/others/gemm_bf16_triton_jit_smoke.py
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def gemm_bf16_jit_smoke_kernel(
    output_ptr,              # *bf16   output [M, N]
    mat1_ptr,                # *bf16   mat1   [M, K]
    mat2_ptr,                # *i64    per-core LocalMem weight descriptor
    M,                       # i32
    N,                       # i32
    K,                       # i32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # bf16 LocalMem block-column width = 64
    CORE_NUM: tl.constexpr,
):
    core_id    = tl.program_id(0)
    n_per_core = N // CORE_NUM

    # Load the hardware weight base address for this core, then cast to *bf16.
    # This mirrors gemm_bf16_dense_kernel exactly so Zeus compiler can assign
    # the three hardware buffer registers (act, wgt, dst) for tl.dot.
    weight_addr = tl.load(mat2_ptr + core_id)
    weight_ptr  = weight_addr.to(tl.pointer_type(tl.bfloat16))

    a_base = tl.make_block_ptr(
        base=mat1_ptr, shape=(M, K), strides=(K, 1),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    b_base = tl.make_block_ptr(
        base=weight_ptr, shape=(n_per_core, K), strides=(K, 1),
        offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0),
    )
    out_base = tl.make_block_ptr(
        base=output_ptr, shape=(M, N), strides=(N, 1),
        offsets=(0, n_per_core * core_id), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )

    for m_blk in tl.range(tl.cdiv(M, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        for n_blk in tl.range(tl.cdiv(n_per_core, BLOCK_N)):
            n_start = n_blk * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_blk in tl.range(tl.cdiv(K, BLOCK_K)):
                k_start = k_blk * BLOCK_K
                a = tl.load(
                    tl.advance(a_base, (m_start, k_start)),
                    boundary_check=(0, 1), padding_option="zero",
                )
                b_nk = tl.load(
                    tl.advance(b_base, (n_start, k_start)),
                    memory_type='weight',
                    boundary_check=(0, 1), padding_option="zero",
                )
                acc += tl.dot(a, tl.trans(b_nk), out_dtype=tl.float32)
            tl.store(
                tl.advance(out_base, (m_start, n_start)),
                acc.to(tl.bfloat16),
                boundary_check=(0, 1),
            )


def run_gemm_jit_check() -> None:
    import torch
    import torch.nn.functional as F
    import torch_zeus

    torch.manual_seed(20260505)
    M, N, K = 1200, 1200, 1200  # K=64 = one bf16 LocalMem block-column width

    mat1    = torch.randn(M, K, dtype=torch.bfloat16, device='zeus')
    out     = torch.empty(M, N, dtype=torch.bfloat16, device='zeus')
    weight_nk = torch.randn(N, K, dtype=torch.bfloat16, device='zeus')  # [N, K]

    # Pack [N, K] weight into LocalMem NK-tiled format.
    # from_tensor handles the layout internally; the JIT launcher memcpy's the
    # resulting bytes directly into the simulator weight buffer.
    weight_packed = torch.zeus.local_memory.from_tensor(weight_nk, kind='weight', Tr=1, Tc=1)

    # CPU reference: F.linear(mat1, weight_nk) = mat1 @ weight_nk.T
    ref = F.linear(mat1.cpu().float(), weight_nk.cpu().float()).to(torch.bfloat16)

    # zecc_jit detects ZeusLocalMemTensor → emits *i64 signature + localmem_weight_ptr;
    # jit_launcher memcpy's the NK-tiled LocalMem bytes as the simulator weight.
    gemm_bf16_jit_smoke_kernel[(1,)](
        out, mat1, weight_packed,
        M, N, K,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=64,
        zeus_enable_jit=True,
        zeus_strict=True,
        zeus_log=True,
    )

    torch.testing.assert_close(
        out.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=2e-2
    )


def main() -> int:
    run_gemm_jit_check()
    print("PASS: gemm_bf16 Python-level Zeus JIT launcher smoke (LocalMem weight)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
