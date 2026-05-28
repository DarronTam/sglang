#!/usr/bin/env python3
"""Python-level Zeus JIT smoke for the v3 block MoE grouped GEMM.

Reference kernel:
  /datau38020T/Application/caof/workspace/zeus514/triton_shared/python/
  zeus_examples/GEMM/
  sgl_moe_grouped_gemm_kernel_v3_bf16xbf16_bf16dst_f32acc_8x128x128_2core.py

This variant models the gemm1 path: A is [T, K], top_k is the router top-k,
MUL_ROUTED_WEIGHT is 0, and BLOCK_M is 8 for the block GEMM staging kernel.
"""

from __future__ import annotations

from sgl_kernel_zeus import jit as _jit  # noqa: F401 - installs Zeus JIT hook
import triton
import triton.language as tl


CORE_NUM = 2
BLOCK_M = 8
BLOCK_N = 128
BLOCK_K = 128


@triton.jit
def sgl_moe_grouped_gemm_kernel_v3_bf16xbf16_bf16dst_f32acc_8x128x128_2core(
    a_ptr,                     # *bf16 [T, K]
    w_ptr_list,                # *i64  LocalMem weight descriptor
    c_ptr,                     # *bf16 [T*topk, N]
    sorted_token_ids_ptr,      # *fp32 [EM]
    expert_ids_ptr,            # *fp32 [EM / BLOCK_M]
    num_tokens_post_pad_ptr,   # *i32  [1]
    topk_weights_ptr,          # *bf16 [T*topk], unused when MUL_ROUTED_WEIGHT=0
    N,                         # i32
    K,                         # i32
    num_valid_tokens,          # i32 (= T * topk)
    top_k,                     # i32
    CORE_NUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
):
    core_id = tl.program_id(axis=0)
    n_per_core = N // CORE_NUM

    w_addr = tl.load(w_ptr_list + core_id)
    current_w_ptr = w_addr.to(tl.pointer_type(tl.bfloat16))

    top_k_f = top_k.to(tl.float32)

    num_tokens_post_padded = tl.load(num_tokens_post_pad_ptr)
    total_m_blocks = tl.cdiv(num_tokens_post_padded, BLOCK_M)
    total_n_blocks = tl.cdiv(n_per_core, BLOCK_N)
    total_k_blocks = tl.cdiv(K, BLOCK_K)

    for m_block in range(total_m_blocks):
        m_start = m_block * BLOCK_M

        offs_token_f = tl.load(
            sorted_token_ids_ptr + m_start + tl.arange(0, BLOCK_M)
        )
        a_row_ids = (offs_token_f / top_k_f).to(tl.int32)
        offs_token = offs_token_f.to(tl.int32)
        valid_row = offs_token < num_valid_tokens

        off_expert_f = tl.load(expert_ids_ptr + m_block)
        off_expert = off_expert_f.to(tl.int32)

        for n_block in range(total_n_blocks):
            n_start_local = n_block * BLOCK_N
            n_start_global = core_id * n_per_core + n_start_local

            offs_cn = n_start_global + tl.arange(0, BLOCK_N)

            if off_expert == -1:
                zeros_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
                c_ptrs = c_ptr + offs_token[:, None] * N + offs_cn[None, :]
                tl.store(c_ptrs, zeros_tile, mask=valid_row[:, None])
            else:
                b_template = tl.make_block_ptr(
                    base=current_w_ptr + off_expert * n_per_core * K,
                    shape=(n_per_core, K),
                    strides=(K, 1),
                    offsets=(n_start_local, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(1, 0),
                )

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k_block in range(total_k_blocks):
                    k_start = k_block * BLOCK_K

                    col_offsets = k_start + tl.arange(0, BLOCK_K)
                    a_ptrs = a_ptr + a_row_ids[:, None] * K + col_offsets[None, :]
                    a_tile = tl.load(
                        a_ptrs,
                        mask=valid_row[:, None],
                        other=0.0,
                    )

                    b_blk = tl.advance(b_template, (0, k_start))
                    b_tile = tl.load(
                        b_blk,
                        memory_type="weight",
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )

                    acc += tl.dot(a_tile, tl.trans(b_tile), out_dtype=tl.float32)

                c_ptrs = c_ptr + offs_token[:, None] * N + offs_cn[None, :]
                tl.store(c_ptrs, acc.to(tl.bfloat16), mask=valid_row[:, None])


def _build_align_cpu(topk_ids, num_experts: int):
    import torch

    flat = topk_ids.flatten().to(torch.int64).cpu()
    sentinel = int(flat.numel())
    sorted_ids = []
    expert_ids = []

    for expert in range(num_experts):
        token_ids = torch.nonzero(flat == expert, as_tuple=False).flatten().tolist()
        if not token_ids:
            continue
        pad = (-len(token_ids)) % BLOCK_M
        sorted_ids.extend(token_ids)
        sorted_ids.extend([sentinel] * pad)
        expert_ids.extend([expert] * ((len(token_ids) + pad) // BLOCK_M))

    if not sorted_ids:
        sorted_ids = [sentinel] * BLOCK_M
        expert_ids = [-1]

    return (
        torch.tensor(sorted_ids, dtype=torch.float32),
        torch.tensor(expert_ids, dtype=torch.float32),
        torch.tensor([len(sorted_ids)], dtype=torch.int32),
    )


def _ref_moe_grouped_gemm_v3(
    A,
    B,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad: int,
    num_valid_tokens: int,
    top_k: int,
):
    import torch

    N = B.shape[1]
    C = torch.zeros(num_valid_tokens, N, dtype=torch.bfloat16)
    A_f = A.cpu().float()
    B_f = B.cpu().float()
    sorted_ids = sorted_token_ids.cpu().to(torch.int64)
    experts = expert_ids.cpu().to(torch.int64)

    m_blocks = (num_tokens_post_pad + BLOCK_M - 1) // BLOCK_M
    for m_block in range(m_blocks):
        block_ids = sorted_ids[m_block * BLOCK_M : (m_block + 1) * BLOCK_M]
        mask = block_ids < num_valid_tokens
        off_expert = int(experts[m_block].item())

        if off_expert == -1:
            C[block_ids[mask]] = 0
            continue

        a_rows = torch.where(mask, block_ids // top_k, torch.zeros_like(block_ids))
        a_tile = A_f[a_rows]
        a_tile[~mask] = 0
        acc = a_tile @ B_f[off_expert].T
        C[block_ids[mask]] = acc[mask].to(torch.bfloat16)

    return C


def run_moe_grouped_gemm_v3_jit_check() -> None:
    import torch
    import torch_zeus  # noqa: F401

    torch.manual_seed(20260520)
    T, E, topk = 8, 4, 2
    N, K = 256, 256
    assert N % (CORE_NUM * BLOCK_N) == 0

    topk_ids = torch.argsort(torch.rand(T, E), dim=1)[:, :topk].to(torch.int32)
    topk_weights = torch.rand(T, topk, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    topk_weights = topk_weights.flatten().to(torch.bfloat16).contiguous()

    A = torch.randn(T, K, dtype=torch.bfloat16) * 0.05
    B = torch.randn(E, N, K, dtype=torch.bfloat16) * 0.05

    n_per_core = N // CORE_NUM
    B_c0 = B[:, :n_per_core, :].reshape(E * n_per_core, K)
    B_c1 = B[:, n_per_core:, :].reshape(E * n_per_core, K)
    B_flat = torch.cat([B_c0, B_c1], dim=0).to("zeus").contiguous()
    weight_packed = torch.zeus.local_memory.to_local_mem(
        B_flat, kind="weight", Tr=1, Tc=1, num_cores=CORE_NUM, partition="row"
    )

    sorted_ids, expert_ids, ntpp = _build_align_cpu(topk_ids, E)
    num_valid_tokens = T * topk
    C = torch.empty(num_valid_tokens, N, dtype=torch.bfloat16, device="zeus")

    sgl_moe_grouped_gemm_kernel_v3_bf16xbf16_bf16dst_f32acc_8x128x128_2core[
        (CORE_NUM,)
    ](
        A.to("zeus").contiguous(),
        weight_packed,
        C,
        sorted_ids.to("zeus"),
        expert_ids.to("zeus"),
        ntpp.to("zeus"),
        topk_weights.to("zeus"),
        N,
        K,
        num_valid_tokens,
        topk,
        CORE_NUM=CORE_NUM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        MUL_ROUTED_WEIGHT=0,
        zeus_enable_jit=True,
        zeus_strict=True,
        zeus_log=True,
    )

    ref = _ref_moe_grouped_gemm_v3(
        A,
        B,
        sorted_ids,
        expert_ids,
        int(ntpp.item()),
        num_valid_tokens,
        topk,
    )
    torch.testing.assert_close(C.cpu().float(), ref.float(), rtol=1e-2, atol=2e-2)


def main() -> int:
    run_moe_grouped_gemm_v3_jit_check()
    print("PASS: 2-core v3 block sgl_moe_grouped_gemm Python-level Zeus JIT smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
