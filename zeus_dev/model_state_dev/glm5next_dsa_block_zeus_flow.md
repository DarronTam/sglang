# GLM5-Next `dsa_block` 一步 decode 的 sgl-kernel-zeus 调用流程

> 对应实现：`dev_glm5next_block_decode_test.py::stage_dsa_block`（mHC + DSA + MoE
> 端到端 Zeus path）。本文档完整列出 host / device 上的每一步操作，逐项标注分类，
> 用来 verify "中间是否还有其它非 sgl-kernel-zeus 操作"。

## 1. 范围

- **单 device、单 layer**（full-attn 层，如 16B layer 3/7/.../23、Next layer
  3/7/.../43）
- **decode 单步**：`B = num_tokens`、`seqlen = history length`
- **路径**：`mHC pre → DSA decode → mHC post → mHC pre → MoE decode → mHC post`
- **MoE 走 proxy shape**（E=8 / mI=128 / top_k=2 / sI=128），kernel 层真实 shape
  对拍由 `dev_glm4_moe_test.py` 覆盖
- **DSA 走 cp=1（无 CP merge）**，full_attn_layers 的 decode 路径

## 2. 用户预批准的"允许非 Zeus 例外"

| # | 类别 | 示例 | 备注 |
|---|---|---|---|
| 1 | **初始化权重 + LocalMem pack** | layer.__init__ 一次性把 fp32/bf16 权重 `.to("zeus")` 并 `local_memory.from_tensor(kind="weight", Tr=Tc=1)` | 不在热路径，每个 layer 一次 |
| 2 | **Latent K / Index-K transfer（PD-disagg 必须）** | DSA 写出 `kv_new_cache`（latent K）+ `body_cache` / `scale_cache`（Index-K）；history `latent_kv` / `index_body` / `index_scale` 在 device↔host 间拼接 | prefill 集群 → decode 集群必须能传 KV traffic，history concat 是其语义副作用 |

下面所有 **不属于这两类、又不是 sgl-kernel-zeus 算子调用** 的步骤，都视为 **GAP** ⚠️。

## 3. 完整流程图（一步 decode，attn_hc + mlp_hc 两条 wrap）

```
┌─────────────────────────────────────────────────────────────────────┐
│ 入口  residual_flat  [B, N*H] bf16  (host)                          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ attn_hc.pre               │
              │  mhc.zeus_mhc_pre         │
              │  ─────────────────────    │
init / weight │  fn = p.fn * p.norm_w     │  ← host fp32 multiply (CPU)
   pack       │  fn.bf16 → LocalMem       │  ← weight init/pack
              │  p.base → Zeus            │  ← weight init
              │  residual → Zeus          │  ← input transfer
              │                           │
   ▶ Zeus K1  │  mhc_pre_norm_split       │  RMSNorm + bf16 GEMM + sigmoid
   ▶ Zeus K2  │  mhc_sinkhorn             │  Birkhoff projection
   ▶ Zeus K3  │  mhc_pre_apply_mix        │  N-stream weighted reduce
              │  → z_li_a [B, H] (Zeus)   │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼──────────────────────────┐
              │ attn = zeus_dsa_decode                 │
              │  ──────────────────────────────────    │
weight init   │  q_a_w / kv_a_w / q_b_w / w_kc /       │  ← all LocalMem pack
              │   wq_b / wk_idx / weights_proj /       │
              │   o_proj / w_vc + norms                │
control flow  │  slot_mapping = arange(B) → Zeus       │
              │                                        │
   ▶ DSA #0Q  │  dsa_q_a_proj_norm                     │  Q LoRA + RMSNorm
              │  → q_lora_z                            │
              │                                        │
              │  alloc kv_new_cache (latent K buffer)  │  ← Latent K output
   ▶ DSA #0KV │  dsa_kv_a_proj_norm_store              │  KV LoRA + norm + store
              │  → kv_new_cache [B, Rkv]               │
              │                                        │
   ▶ DSA #1   │  dsa_q_main_absorb                     │  q_b_proj + absorb bmm
              │  → q_new_z [B, Nh, Rkv]                │
              │                                        │
   ▶ DSA #2Q  │  dsa_indexer_q_weights                 │  Indexer Q + weights
              │  → q_body_z, weights_z                 │
              │                                        │
              │  alloc body_cache (Index-K body, fp8)  │  ← Index-K output
              │  alloc scale_cache (Index-K scale)     │
   ▶ DSA #2K  │  dsa_indexer_k_prep_store              │  Indexer K prep + store
              │  → body_cache + scale_cache            │
              │                                        │
PD-disagg     │  kv_new_cache.cpu() / body_cache.cpu() │  ← Latent/Index-K D→H
PD-disagg     │  scale_cache.cpu()                     │  ← Index-K D→H
PD-disagg     │  full_latent = cat([history, new_k])   │  ← Latent K history concat
PD-disagg     │  full_body = cat([history, new_body])  │  ← Index-K history concat
PD-disagg     │  full_scale = cat([...])               │
PD-disagg     │  _lmem_pack(full_body) → Zeus          │  ← Index-K H→D
PD-disagg     │  full_scale → Zeus                     │
              │                                        │
   ▶ DSA #3   │  dsa_index_logits                      │  Index GEMM
              │  → logits_z [B, S_full]                │
              │                                        │
control flow  │  positions = arange(S_full) → Zeus     │
   ▶ DSA #4   │  dsa_local_topk_radix                  │  → top_pos [B, Ktop]
control flow  │  top_pos.cpu().int64()  (sparse index) │  ← needed for per-b loop
              │                                        │
              │  alloc K_local_full / K_local_T_full / │  ← per-batch gather buffers
              │   mask_full (CPU)                      │
              │                                        │
              │  for b in range(B):                    │
PD-disagg     │    full_latent[b].to("zeus")           │  ← Latent K transfer
   ▶ DSA #7   │    dsa_latent_k_gather                 │  Latent K gather
              │                                        │  (pool 契约：auto-alloc
              │                                        │   走 torch.zeros，invalid
              │                                        │   位置 finite 0)
              │    K_local_full[b] = K_c0_b.cpu()[0]   │  ← gather 输出直接搬，
              │    (mask 由下游 #8 算术 blend 屏蔽)    │   无 torch.where cleanup
              │                                        │
              │  _lmem_pack(K_local_full / _T / mask)  │  ← latent transfer back
   ▶ DSA #8   │  dsa_sparse_mqa_partial                │  Sparse MQA partial
              │  → po_z (kernel 已强制 bf16 输出，     │
              │          无 D→H→D dtype cast)          │
              │                                        │
weight init   │  w_vc / o_proj_w → Zeus                │
   ▶ DSA #10  │  dsa_post_o_proj_no_cp                 │  V absorb + o_proj
              │  → attn_out_z [B, H] (Zeus bf16)       │
              └─────────────┬──────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ attn_hc.post              │
   ▶ Zeus K4  │  mhc.zeus_mhc_post        │  fp32 acc + 单 bf16 RNE
              │  → z_mid [B, N*H]         │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ mlp_hc.pre (与 attn 同套路) │
init / weight │  fn_mlp_lmem + p_mlp.base │
              │                           │
   ▶ Zeus K1' │  mhc_pre_norm_split       │
   ▶ Zeus K2' │  mhc_sinkhorn             │
   ▶ Zeus K3' │  mhc_pre_apply_mix        │
              │  → z_li_m [B, H] (Zeus)   │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────────────────┐
              │ mlp = zeus_moe_decode                 │
              │  ─────────────────────────────────    │
   ▶ Linear   │  linear_bf16(x, gate_w_lmem)          │  gate Linear (GAP-3 fix)
              │     → router_logits_bf16              │
              │  router_logits.float() (device cast)  │
   ▶ Linear   │  linear_bf16(x, sh_gu_lmem)           │  shared gate_up (GAP-3 fix)
              │     → sh_gu_bf16 [T, 2*sI]            │
   ▶ silu_mul │  silu_and_mul(sh_gu) → sh_silu        │  shared activation (GAP-3 fix)
   ▶ Linear   │  linear_bf16(sh_silu, sh_dp_lmem)     │  shared down (GAP-3 fix)
              │     → shared_out_z [T, H]             │
weight init   │  corr_bias / w13 / w2 → Zeus          │
              │                                       │
   ▶ MoE #1   │  biased_grouped_topk                  │  → w_z, ids_z
   ▶ MoE #2   │  moe_align_block_size_alloc           │  → sorted/expert/num_post
              │  alloc C1_z [T*K, 2*mI]               │
   ▶ MoE #3   │  moe_grouped_gemm (gemm1)             │
              │  alloc C1_silu_z [T*K, mI]            │
   ▶ MoE #4   │  silu_and_mul                         │
              │  w_z.bf16().flatten() (device view)   │
              │  alloc C2_z [T*K, H]                  │
   ▶ MoE #5   │  moe_grouped_gemm (gemm2,             │
              │                     mul_routed_weight)│
              │  alloc final_z [T, H]                 │
   ▶ MoE #6   │  moe_sum_reduce (+shared residual)    │  fp32 acc + bf16 RNE
              │  → mlp_out_z [B, H] (Zeus bf16)       │
              └─────────────┬─────────────────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ mlp_hc.post               │
   ▶ Zeus K4' │  mhc.zeus_mhc_post        │
              │  → z_out [B, N*H] (Zeus)  │
              └───────────────────────────┘
```

## 4. Zeus kernel 计数（一步 decode）

| 段 | kernel | 次数 |
|---|---|---:|
| **mHC attn pre** | `mhc_pre_norm_split` | 1 |
| | `mhc_sinkhorn` | 1 |
| | `mhc_pre_apply_mix` | 1 |
| **DSA decode** | `dsa_q_a_proj_norm` | 1 |
| | `dsa_kv_a_proj_norm_store` | 1 |
| | `dsa_q_main_absorb` | 1 |
| | `dsa_indexer_q_weights` | 1 |
| | `dsa_indexer_k_prep_store` | 1 |
| | `dsa_index_logits` | 1 |
| | `dsa_local_topk_radix` | 1 |
| | `dsa_latent_k_gather` | **B** |
| | `dsa_sparse_mqa_partial` | 1 |
| | `dsa_post_o_proj_no_cp` | 1 |
| **mHC attn post** | `mhc_post` | 1 |
| **mHC mlp pre** | `mhc_pre_norm_split` | 1 |
| | `mhc_sinkhorn` | 1 |
| | `mhc_pre_apply_mix` | 1 |
| **MoE decode** | `linear_bf16` (gate / shared gate_up / shared down) | 3 |
| | `silu_and_mul` (shared experts) | 1 |
| | `biased_grouped_topk` | 1 |
| | `moe_align_block_size_alloc` | 1 |
| | `moe_grouped_gemm` (gemm1 / gemm2) | 2 |
| | `silu_and_mul` (per-expert) | 1 |
| | `moe_sum_reduce` | 1 |
| **mHC mlp post** | `mhc_post` | 1 |
| **合计** | | **28 + B** 颗 |

B=4 时 = **27 颗 sgl-kernel-zeus 调用**。

## 5. 非 Zeus 操作逐项分类

| 位置 | 操作 | 类别 | 状态 |
|---|---|---|---|
| mHC pre 入口 (×2) | `fn = p.fn * p.norm_weight`（host fp32 multiply） | 权重初始化 | ✅ 允许 |
| mHC pre 入口 (×2) | `fn.to(bfloat16)` + LocalMem pack | 权重初始化 | ✅ 允许 |
| mHC pre 入口 (×2) | `p.base.to("zeus")`、`residual.to("zeus")` | 权重 / 输入 transfer | ✅ 允许 |
| DSA 入口 | 9 颗 weight (`q_a_w` etc.) + 2 颗 norm `.to("zeus")` + LocalMem pack | 权重初始化 | ✅ 允许 |
| DSA 入口 | `slot_mapping = arange(B) → Zeus` | 控制流（小 int32） | ✅ 允许 |
| DSA #0.KV 后 | `kv_new_cache.cpu()` | **Latent K D→H** | ✅ 允许 (PD-disagg) |
| DSA #2.K 后 | `body_cache.cpu()` / `scale_cache.cpu()` | **Index-K D→H** | ✅ 允许 (PD-disagg) |
| DSA #2.K 后 | `torch.cat([history, new_k/body/scale])` (CPU) | **history concat** | ✅ 允许 (PD-disagg coordination) |
| DSA #3 前 | `_lmem_pack(full_body)` + `full_scale.to("zeus")` | **Index-K H→D** | ✅ 允许 (PD-disagg) |
| DSA #4 后 | `positions = arange(S_full) → Zeus` | 控制流 | ✅ 允许 |
| DSA #4 后 | `top_pos.cpu()` | 控制流（sparse index 必需） | ✅ 允许（algorithm 要求 per-b 拆 latent_k_gather） |
| DSA #7 入口 (×B) | `full_latent[b].to("zeus")` | **Latent K H→D** | ✅ 允许 (PD-disagg) |
| **DSA #7 出口 (×B)** | ~~`torch.where(mask, K, zeros)` 屏蔽无效行~~（2026-05-25 已删，pool 契约消除）| 非 Zeus compute | ✅ **已修复（GAP-1）** |
| DSA #7 后 | `_lmem_pack(K_local_full / _T / mask)` | Latent K 输入 transfer | ✅ 允许 |
| **DSA #8 → #10 之间** | ~~`po_z.cpu().to(bf16).to("zeus")` D→H→D~~（2026-05-25 已删，kernel 本就 bf16 输出）| dead round-trip | ✅ **已修复（GAP-2）** |
| DSA #10 前 | `w_vc.to("zeus")` / `o_proj_w.to("zeus")` | 权重初始化 | ✅ 允许 |
| **MoE 入口** | ~~`z_li_m.cpu()` D→H 进 router/shared~~（2026-05-25 已删，linear_bf16 切 Zeus）| 非 Zeus compute | ✅ **已修复（GAP-3）** |
| **MoE 前段** | ~~`router_logits = nn.functional.linear(x, gate_w)` (CPU)~~（2026-05-25 切 `sgl_kernel_zeus.linear_bf16`）| gate Linear 已上 Zeus | ✅ **已修复（GAP-3）** |
| **MoE 前段** | ~~`shared_out = SwiGLU MLP` on CPU~~（2026-05-25 切 `linear_bf16 × 2 + silu_and_mul`）| shared experts 已上 Zeus | ✅ **已修复（GAP-3）** |
| MoE 中段 | `router_logits / shared_out / corr_bias / w13 / w2 → Zeus` | 权重 + 前处理结果 transfer | ✅ 允许 |
| MoE `w_z.bf16().flatten()` | device 端 dtype cast + view | view-only | ✅ 允许 |
| 各处 `.reshape()` | view operations | view-only | ✅ 允许 |

## 6. 总结：GAP 清单（按严重程度）

### ✅ GAP-3（2026-05-25 已解决）：MoE router gate + shared experts MLP 已切 Zeus `linear_bf16`

原状态（修复前）：
```
z_li_m.cpu() ──▶ nn.Linear(x, gate_w) ──▶ router_logits (CPU fp32)
              └▶ nn.Linear(x, sh_gu) ──▶ silu_and_mul ──▶ nn.Linear(., sh_dp) ──▶ shared_out (CPU bf16)
```

**根因诊断**：sgl-kernel-zeus 历史上只有 `moe_grouped_gemm`（grouped + sorted_ids
寻址），**没有 generic dense Linear** —— `glm4_moe_ffn_dev.md` 表里 #1 / #3 / #4
之前标 ✅ `linear_zeus` 是文档误标（实际未注册）。同款限制还体现在
`dev_kimi_linear_attn_test.py:1060` 的明文 TODO（Linear-attn 5 个 projection）。

**修复**（2026-05-25 一日落地）：

1. **新增 `sgl_kernel_zeus.linear_bf16`** packed-weight dense bf16 GEMM kernel
   （五件套 + docs + slides 完整交付，见 `sgl-kernel-zeus/docs/linear_bf16.md`）。
   bf16 × bf16 → fp32 acc → 单 RNE，N-axis 2-core split (与 `moe_grouped_gemm`
   同拓扑)，weight 走 LocalMem (`kind="weight"`)。**`tests/test_linear_bf16.py`
   14/14 PASS**（含 GLM5-Next-16B 实战 shape）。
2. **`dev_glm4_moe_test.py::test_moe_block_full`**：3 颗 host `F.linear` (gate /
   shared gate_up / shared down) 全切 Zeus `linear_bf16`。验收 PASS，
   `final max_diff=4.88e-4`（与切之前同量级，无回归）。
3. **`dev_glm5next_block_decode_test.py::zeus_moe_decode`**：同款切换。
   验收 PASS（16B mid=4.88e-4 / out=1.46e-3；Next mid=4.88e-4 / out=7.81e-3，
   全部与切之前同量级，无回归）。

**`zeus_moe_decode` 现在是 10 颗 sgl-kernel-zeus 算子的完整 device-resident chain**：
```
1.  linear_bf16        (gate Linear)            ← GAP-3 fix
2.  linear_bf16        (shared gate_up_proj)    ← GAP-3 fix
3.  silu_and_mul       (shared experts)         ← GAP-3 fix
4.  linear_bf16        (shared down_proj)       ← GAP-3 fix
5.  biased_grouped_topk
6.  moe_align_block_size_alloc
7.  moe_grouped_gemm   (gemm1)
8.  silu_and_mul       (per-expert)
9.  moe_grouped_gemm   (gemm2, mul_routed_weight=True)
10. moe_sum_reduce     (+shared residual fuse)
```

中间无任何 host CPU compute、无 D→H→D round-trip（host 仅做一次性 weight
LocalMem pack）。

### ✅ GAP-2（2026-05-25 已解决）：DSA `po_z` D→H→D dtype cast round-trip

**根因诊断**：实际上 `dsa_sparse_mqa_partial` 的 host wrapper 已经强制
`partial_out.scalar_type() == at::kBFloat16`（参 `dsa_sparse_mqa_partial_zeus.cpp:195-197`），
Python API auto-alloc 也默认 bf16。**dev script 里的 `po_z.cpu().to(torch.bfloat16).to("zeus")`
是 100% dead code**（bf16 → CPU → bf16 no-op cast → Zeus），kernel 侧不需要任何改动。

**修复**：删除 `dev_glm5next_dsa_decode_test.py::stage_decode_full_nocp` 与
`dev_glm5next_block_decode_test.py::zeus_dsa_decode` 里这一行 dead round-trip，
直接 `attn_latent_z = po_z` 喂下游 `dsa_post_o_proj_no_cp`。

### ✅ GAP-1（2026-05-25 已解决）：DSA #7 后 `torch.where` 屏蔽无效行

**根因诊断**：`dsa_latent_k_gather` kernel 的 "skip = save bandwidth" 设计——invalid
行 / 列不写，由 caller 的 output buffer 初始内容决定 invalid 位置的 bit pattern。
之前 Python API 默认用 `torch.empty` 分配（uninitialized），残留可能包含 NaN/Inf，
下游 `dsa_sparse_mqa_partial` 的算术 mask `score*0 + (1-mask)*NEG_LARGE` 在
`NaN*0 = NaN` 下会污染整 head 输出，因此 dev script 用 `torch.where(mask, K, 0)`
在 host 端 cleanup 兜底。

**修复（采用 pool 契约方案，无 kernel 改动）**：
1. **`sgl_kernel_zeus.dsa_latent_k_gather` Python API**：`_alloc` 默认从
   `torch.empty` 改为 `torch.zeros`（见 `glm5next_dsa.py::dsa_latent_k_gather`
   docstring "Pool contract" 段），首次 auto-alloc 即 finite 0.0 bf16；docstring
   推荐 production 路径用 layer.__init__ 持久 pool（一次 zero-init，跨 step 复用，
   保留 "skip = save bandwidth" 全部好处，stale 残留也必为 finite）。
2. **`dev_glm5next_dsa_decode_test.py` / `dev_glm5next_block_decode_test.py`**：
   删除 host 端 `torch.where(mask, K, zeros)` cleanup，直接把 gather 输出搬到
   K_local_full[b]。
3. **测试**：新增 `test_dsa_latent_k_gather_default_alloc_is_zero_init`（验证
   default alloc 不再是 garbage）+ `test_dsa_sparse_mqa_partial_pool_reuse_finite_residual`
   （模拟生产 pool reuse 下 invalid 位置含 stale finite valid-K 残留，验证
   sparse_mqa_partial 仍正确）。

为什么不在 `dsa_sparse_mqa_partial` 里加 `tl.where` 防 NaN：pool 契约从源头消除 NaN
可能性，kernel 无需任何防御代码，保留原有"算术 mask blend"的简洁性。同时 kernel
契约从"任意 garbage 都能 handle"收窄到"caller 必须 finite 初始化"，前提清晰更易
推理。

### 🟢 history concat 是 PD-disagg 的 CPU 实现（用户预批准）

- 当前在 CPU 上 `torch.cat`，**dev script 简化**
- 生产路径理想是：prefill cluster 直接吐出 device 上的 latent_kv 拼到 decode KV
  cache slot（zero-copy），但这是 dev script 设计的边界，不算 GAP

## 7. 结论

**`dsa_block` 一步 decode 总共 33+B 颗 sgl-kernel-zeus 调用**（mHC 8 + DSA 10+B
+ MoE 10 + mHC 4 + mHC 4 = 36+B），全部 device-resident，**中间无任何 host
CPU compute**。GAP-1 / GAP-2 / GAP-3 已全部在 2026-05-25 消除。

| 例外类别 | 是否在用户预批准内 |
|---|---|
| 初始化 weight + LocalMem pack（含 MoE gate / shared / w13 / w2）| ✅ 在 |
| Latent K / Index-K transfer + history concat | ✅ 在 |
| 控制流小张量（slot_mapping / positions / top_pos）| ✅ 视为算法必需 |
| ~~DSA latent_k_gather mask 后处理 (CPU torch.where × B)~~ | ✅ 已修复（GAP-1，2026-05-25 pool 契约） |
| ~~DSA po_z bf16 cast (D→H→D)~~ | ✅ 已修复（GAP-2，2026-05-25 删 dead round-trip） |
| ~~MoE router gate + shared experts MLP（CPU 上 3 Linear + 1 SwiGLU）~~ | ✅ 已修复（GAP-3，2026-05-25 `linear_bf16` 落地 + 切换） |

**完全在 Zeus 上的状态已达成**。剩余值得做的工作（不算 GAP）：

- **`linear_attn_block` 全 Zeus 化**：Linear-attn 5 个 projection（qkv / b /
  f_a/f_b / g_a/g_b / o_proj）以及 `kimi_delta_attn_decode` 的剩余 host `F.linear`
  也可以切到 `linear_bf16`；之后 dense MLP layer 0~2 (`ref_mlp_decode`) 的
  gate_up / down 同款切换。这条 chain 走通后 `linear_attn_block` 也能进入完全
  device-resident。
- **真实 GLM5-Next 配置**：当前 MoE 用 proxy shape (E=8, mI=128, sI=128)；真实
  GLM5-Next-16B `E=64, mI=1408` / GLM5-Next `E=288, mI=2048` 的 kernel 层对拍
  由各 sub-kernel 自己的 sgl-kernel-zeus tests 覆盖（`test_moe_grouped_gemm.py`
  / `test_linear_bf16.py` 等）。dev 集成测试维持 proxy size 避免 CPU REF OOM。
- **生产部署**：layer.__init__ 一次性 LocalMem pack 缓存（dev 脚本每次重新 pack
  是 stage 独立性的简化）。
