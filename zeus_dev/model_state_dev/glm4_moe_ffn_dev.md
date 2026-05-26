# GLM-4.7 MoE-FFN Zeus 适配开发追踪

> 对齐目标：GLM-4.5 / 4.6 / 4.7 共享同一 `Glm4MoeForCausalLM` 结构（见
> `python/sglang/srt/models/glm4_moe.py`），GLM-4.7 只是权重更新。本文档记录
> 把它的 MoE-FFN 段在纯 Zeus（无 CUDA）环境逐算子跑通的工作。

## 范围与方法

- **起点**：`self_attn.o_proj` 的输出 `[T, H]`（即 attention 子层结束后、进入
  post-attention rmsnorm 的张量）。
- **终点**：MoE Block 输出 `[T, H]`，可直接喂给下一层的 input_layernorm。
- **切片**：单 device、单 layer、MoE-FFN only。先不碰 TP/EP/DeepEP/NextN。
- **对齐方式**：沿用 `demo_zeus_layer_compare.py` 的 "REF vs Zeus" 模式 ——
  同一份权重、同一份输入，REF（CUDA 若有，否则 CPU/torchnative）产生 golden，
  Zeus 侧跑完后逐算子 `compare_tensors`。**Zeus 路径禁止落到 torchnative
  fallback**，必须对齐到 `sgl_kernel_zeus` 的 kernel 调用。
- **dev 脚本**：`zeus_dev/dev_glm4_moe_test.py`，stage 化，按 `--stage` 单独跑。

## GLM-4.7 相关配置（来自 `zeus_dev/config.json`）

| 字段 | 值 |
|---|---|
| `hidden_size` (H) | 5120 |
| `moe_intermediate_size` (mI) | 1536 |
| `intermediate_size` | 12288（只在 dense layer 用，MoE layer 不用） |
| `n_routed_experts` (E) | 160 |
| `num_experts_per_tok` (top_k) | 8 |
| `n_group` | 1 |
| `topk_group` | 1 |
| `n_shared_experts` | 1 |
| `first_k_dense_replace` | 3（前 3 层是 dense，其余是 MoE） |
| `routed_scaling_factor` | 2.5 |
| `norm_topk_prob` | true |
| `num_hidden_layers` | 92 |

关键观察：`n_group == topk_group == 1`，所以 grouped-topk 退化成"先按 sigmoid+bias
选 top-k"；correction bias 在 `Glm4MoeGate.e_score_correction_bias` 上，fp32。

## MoE-FFN 算子依赖表（单卡、非量化路径）

> 源：`Glm4MoeSparseMoeBlock.forward_normal`（`python/sglang/srt/models/glm4_moe.py:518`）
> → `FusedMoE.forward`（`python/sglang/srt/layers/moe/fused_moe_triton/layer.py:883`）
> → `fused_experts_impl`（`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:369`）

| # | 子步骤 | shape | CUDA sgl-kernel | Zeus 现状 |
|---|---|---|---|---|
| 0 | fused add+rmsnorm（post_attention） | `[T,H]` | `sgl_kernel.fused_add_rmsnorm` | ✅ `sgl_kernel_zeus.fused_add_rmsnorm` |
| 1 | shared_experts gate_up_proj | `[T,H]→[T,2·mI]` | GEMM | ✅ `sgl_kernel_zeus.linear_bf16`（2026-05-25 LANDED，见 `linear_bf16.md`）|
| 2 | shared_experts SiluAndMul | `[T,2·mI]→[T,mI]` | `sgl_kernel.silu_and_mul` | ✅ `sgl_kernel_zeus.silu_and_mul` |
| 3 | shared_experts down_proj | `[T,mI]→[T,H]` | GEMM | ✅ `sgl_kernel_zeus.linear_bf16`（2026-05-25 LANDED，见 `linear_bf16.md`）|
| 4 | gate Linear（router_logits） | `[T,H]→[T,E=160]` | GEMM | ✅ `sgl_kernel_zeus.linear_bf16`（2026-05-25 LANDED；E=160 不整除 256 时由 caller host pad，sim path 无 N % 256 硬约束）|
| 5 | biased_grouped_topk | `[T,E]→(topk_w[T,k], topk_ids[T,k])` | `sgl_kernel.moe_fused_gate` | ✅ `sgl_kernel_zeus.biased_grouped_topk`（v1 通用，fp32，`num_fused_shared_experts=0`）；✅ `sgl_kernel_zeus.glm4_biased_grouped_topk`（GLM-4.7 特化快路径，E=160/K=8/G=1/Gk=1，bf16 IO，scale 已 fuse） |
| 6 | moe_align_block_size | `(topk_ids)→(sorted_ids, expert_ids, num_post)` | `sgl_kernel.moe_align_block_size` | ✅ `sgl_kernel_zeus.moe_align_block_size`（+ `moe_align_block_size_alloc` 分配 wrapper；Zeus 约束：`sorted_ids/expert_ids/cumsum` 改 fp32 bit-exact，`num_post` 保留 i32） |
| 7 | fused_moe GEMM-1（w13/gate_up） | 分块 `[T,H]→[T·k,2·mI]` | Triton `invoke_fused_moe_kernel` | ✅ `sgl_kernel_zeus.moe_grouped_gemm`（`top_k=原始 topk`、`mul_routed_weight=False`；v1：bf16 A/B/C、fp32 accum、CORE_NUM=2 N-split、BLOCK_M/N/K=64/128/128） |
| 8 | per-expert SiluAndMul | `[T·k,2·mI]→[T·k,mI]` | `sgl_kernel.silu_and_mul` | ✅ 复用 `sgl_kernel_zeus.silu_and_mul` |
| 9 | fused_moe GEMM-2（w2/down） | 分块 `[T·k,mI]→[T·k,H]` | Triton `invoke_fused_moe_kernel` | ✅ `sgl_kernel_zeus.moe_grouped_gemm`（`top_k=1`、`mul_routed_weight=True`，路由权重在 accumulator 上融合；B 走 weight DRAM） |
| 10 | moe_sum_reduce | `[T,k,H]→[T,H]` | `sgl_kernel.moe_sum_reduce` | ✅ `sgl_kernel_zeus.moe_sum_reduce`（v1：bf16 I/O + fp32 accum，CORE_NUM=2 H-split，BLOCK_T/K/H=4/2/128；**Zeus 扩展**：可选 `shared_output` residual fuse——省一次 DRAM 往返，精度更高） |
| 11 | `*routed_scaling_factor` + shared_output 相加 | `[T,H]` | elementwise | ✅ `*routed_scaling_factor` 已经被 `glm4_biased_grouped_topk` (`apply_scale_on_output=True`) + gemm2 (`mul_routed_weight=True`) 在上游 fuse；shared residual 被 `moe_sum_reduce` 的 `shared_output` 口子吸收 |
| 12 | all-reduce | — | 单卡跳过 | — |

> ⚠️ **2026-05-24 误标修正**：上表里 #1 / #3 / #4 之前标 ✅ `linear_zeus`，但
> `sgl_kernel_zeus` 实际**从未注册过** `linear_zeus` 这颗算子（`docs/glm4_moe_ffn_slides.html`
> 也是同样误标）。整段 GLM-4.7 MoE 端到端 dev stage 里这 3 处 Linear 实际是在
> host 端跑 `torch.nn.functional.linear`，详见 `dev_glm4_moe_test.py::test_moe_block_full`
> 第 800-817 行明文："`gate Linear` 与 `shared_experts MLP` 的 GEMM 不是本 stage
> 的测试对象，我们在 CPU/CUDA 上用纯 torch **预计算**"。这种简化让 MoE block
> dev script 跑得通，但**实际部署到 GLM5-Next decode 路径时整段 shared / router
> 仍在 host 上**——见 `glm5next_dsa_block_zeus_flow.md` 的 GAP-3。

## GAP-3 / `linear_bf16` 计划

为了把 #1 / #3 / #4 真的 Zeus 化，需要补一颗 **generic dense bf16 GEMM**。设计如下：

| 新算子 | 签名 | 用途 |
|---|---|---|
| **`linear_bf16(input, weight, *, out=None)`** | `[M, K] bf16 × [N, K] bf16 → [M, N] bf16`，fp32 accumulator + RNE store | 替换 #1 / #3 / #4 三处 host Linear |

设计要点：
- 直接对标 `sgl-kernel-zeus/docs/gemm_normal_dense_fp8xfp8_bf16dst_bf16acc_128x512x512_2core.py`
  的 dense slab GEMM blueprint，把 fp8 → bf16，accumulator 仍 fp32
- 复用 `moe_grouped_gemm` 的 N-split 2-core 拓扑：每核拿 `[N/2, K]` weight slab
- 砍掉 `sorted_ids / expert_ids / mul_routed_weight` 自由度，纯 dense
- weight 走 LocalMem pack（`kind="weight", Tr=Tc=1`，与 `mhc_pre_norm_split` /
  `dsa_q_a_proj_norm` 一致）
- **shape 约束**：`N % (CORE_NUM × BLOCK_N) == 0`；GLM-4.7 的 E=160 不是 256 的
  倍数，gate Linear 需要 host 端 pad 或 kernel 内 N-tail boundary check
- **scope v1 不做**：fp8 / int8 量化、per-channel scale、bias fuse、transposed weight

### 这颗 kernel 的连带收益（一颗解多个 GAP）

| 用途 | 当前在哪 | 影响 |
|---|---|---|
| MoE router gate Linear | `dev_glm4_moe_test::moe_block_full` host 端 | 本文档 #4 |
| MoE shared experts gate_up / down | `dev_glm4_moe_test::moe_block_full` host 端 | 本文档 #1 / #3 |
| **Linear-attn 5 个 projection**（qkv_proj / b_proj / f_a/f_b / g_a/g_b / o_proj） | `dev_kimi_linear_attn_test.py:1060` 明文 TODO | 让 `kimi_delta_attn_decode` 端到端真正全 Zeus |
| 任意 dense Linear（dense MLP layer 0~2） | 各 dev 脚本 | 进一步消除 host Linear |

也就是说，这一颗 `linear_bf16` 是 **GLM5-Next 走向"纯 Zeus 端到端"的必要条件**，
不只服务 MoE。

### 备选方案：进一步加一颗 `dense_ffn_swiglu` 融合算子

如果性能 profiling 发现 shared experts 路径的 `[T, 2·sI]` 中间产物 DRAM 流量是
瓶颈，可以再加一颗：

| 备选算子 | 签名 | 用途 |
|---|---|---|
| `dense_ffn_swiglu(x, gate_up_w, down_w, *, out=None)` | `[T,H] →(gate_up)→ [T,2·sI] →(silu_and_mul)→ [T,sI] →(down)→ [T,H]` | 把 gate_up + silu_and_mul + down 融合成一颗 kernel，中间张量留 Lmem 不落 DRAM |

但**先做 `linear_bf16`，确认基础闭环之后再视情况加 `dense_ffn_swiglu`**。理由：
- `linear_bf16` 已经能让 dev script 全 Zeus 跑通
- `dense_ffn_swiglu` 只服务 dense MLP / shared experts，复用度低
- 与 `moe_grouped_gemm` / `moe_sum_reduce` 的演进路径一致：先 split 再 fuse

### 为什么不能把 router + shared FFN 塞进一颗 kernel？

考虑过把 router gate Linear 与 shared experts dense FFN 合成一颗 kernel，但
**不可行**：

1. **输出张量形态不一**：router → `[T, E=160]`（消费者：`biased_grouped_topk`）；
   shared FFN → `[T, H]`（消费者：`moe_sum_reduce` residual）。两个 output
   shape、dtype、下游消费者都不同。
2. **权重 shape 不一**：`gate_w [E, H]` vs `gate_up_w [2·sI, H]` vs `down_w [H, sI]`，
   三个 N 维度都不一样，没法做单一 GEMM。
3. **数据依赖链不同**：shared FFN 是 `gate_up → silu_and_mul → down` 串联；
   router 是单层 GEMM。塞一颗 kernel 里只能"并行执行两个独立子图"，与"host
   并发投递两颗 kernel"等价。

故最佳方案就是 `linear_bf16`（A）+ 视性能加 `dense_ffn_swiglu`（B）的两阶段路线。

## MoE-FFN 计算流（算子组合 + 中间变量传递）

下图给出单卡、非 fused-shared-experts、非 a2a 下的完整 MoE-FFN 计算流。
符号约定：`T`=token 数，`H`=5120，`mI`=1536，`E`=160，`K`=8。方框里标注算子；
箭头上标注**中间变量（名字 + shape + dtype）**。

```
                          residual_in [T,H] bf16        hidden_in [T,H] bf16
                                    │                          │
                                    └────────────┬─────────────┘
                                                 ▼
                                ┌────────────────────────────────┐
                           (0)  │ fused_add_rmsnorm              │
                                │ (sgl_kernel_zeus)              │
                                └──────────────┬─────────────────┘
                                               │ x [T,H] bf16              ← 同时也是 residual_out
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │ (shared experts path)    │ (routed experts path)    │
                    ▼                          ▼                          │
          ┌──────────────────┐       ┌──────────────────────┐             │
     (1)  │ linear_zeus      │  (4)  │ linear_zeus (gate)   │             │
          │ gate_up_proj     │       │ x · W_gate^T         │             │
          └────────┬─────────┘       └──────────┬───────────┘             │
                   │                            │                         │
          gu_s [T,2·mI] bf16           router_logits [T,E] bf16           │
                   │                            │                         │
                   ▼                            ▼                         │
          ┌──────────────────┐       ┌───────────────────────────────┐    │
     (2)  │ silu_and_mul     │  (5)  │ glm4_biased_grouped_topk      │    │
          │ (sgl_kernel_zeus)│       │ apply_scale_on_output=True    │    │
          └────────┬─────────┘       │ (scale=2.5 fused into weights)│    │
                   │                 └──────┬───────────────┬────────┘    │
          act_s [T,mI] bf16                 │               │             │
                   │                  topk_weights      topk_ids          │
                   ▼                  [T,K] bf16        [T,K] int32       │
          ┌──────────────────┐               │               │            │
     (3)  │ linear_zeus      │               │               ▼            │
          │ down_proj        │               │   ┌───────────────────────────┐
          └────────┬─────────┘               │   │ (6) moe_align_block_size  │
                   │                         │   │     _alloc (block_size=64)│
       shared_out [T,H] bf16                 │   └──┬───────────┬───────┬───┘
                   │                         │      │           │       │
                   │                         │   sorted_ids  expert_ids num_post
                   │                         │   [M'] fp32   [B] int32  [1] int32
                   │                         │   (M'=T·K+pad,B=ceil(M'/block))
                   │                         │      │           │       │
                   │                         │      ▼           ▼       ▼
                   │                         │  ┌────────────────────────────┐
                   │                         │  │ (7) moe_grouped_gemm       │
                   │                         │  │   A = x [T,H] bf16         │ ◀─── 来自 (0)
                   │                         │  │   B = w13 [E,2·mI,H] bf16  │
                   │                         │  │   top_k=K, mul_w=False     │
                   │                         │  └─────────────┬──────────────┘
                   │                         │                │
                   │                         │        C1 [T·K, 2·mI] bf16
                   │                         │                │
                   │                         │                ▼
                   │                         │  ┌────────────────────────────┐
                   │                         │  │ (8) silu_and_mul           │
                   │                         │  │ (sgl_kernel_zeus, 复用)    │
                   │                         │  └─────────────┬──────────────┘
                   │                         │                │
                   │                         │      C1_silu [T·K, mI] bf16
                   │                         │                │
                   │                         │                ▼
                   │                         │  ┌────────────────────────────┐
                   │                         │  │ (9) moe_grouped_gemm       │
                   │                         │  │   A = C1_silu [T·K,mI] bf16│
                   │                         │  │   B = w2 [E,H,mI] bf16     │
                   │                         │  │   top_k=1, mul_w=True      │ ◀── topk_weights 乘入 accum
                   │                         │  │   (routing weight on accum)│
                   │                         │  └─────────────┬──────────────┘
                   │                         │                │
                   │                         │        C2 [T·K, H] bf16
                   │                         │                │
                   │                         │         view   ▼
                   │                         │        C2v [T, K, H] bf16
                   │                         │                │
                   └─────── shared_out ──────┴────────────────┤
                            [T,H] bf16                        │
                                                              ▼
                                            ┌─────────────────────────────────┐
                                       (10) │ moe_sum_reduce                  │
                                            │   input = C2v [T,K,H]           │
                                            │   shared_output = shared_out    │
                                            │   routed_scaling_factor = 1.0   │ ← scale 已上游 fuse
                                            │   (fp32 accum, 单次 RNE→bf16)   │
                                            └───────────────┬─────────────────┘
                                                            │
                                                   out [T, H] bf16   ← MoE block 最终输出
                                                            │
                                                            ▼
                                             (下一层 input_layernorm)
```

**关键数据流说明**：

- **(0) → 两路共享同一 `x [T,H]`**：rmsnorm 的输出既喂 `shared_experts.gate_up_proj`，
  也喂 `gate` 和 `moe_grouped_gemm(gemm1)` 的 A。两路**并行**，最终在 (10) 合流。
- **(5) 的 `topk_weights` 已 fuse `routed_scaling_factor=2.5`**：`glm4_biased_grouped_topk`
  的 `apply_routed_scaling_factor_on_output=True`，所以 `topk_weights` 行和 = 2.5 而非 1.0。
- **(7) 与 (9) 复用同一 `moe_grouped_gemm` kernel**，区别只在参数：
  - gemm1：A 是 `[T,H]`（未按路由展开），`top_k=K`，`mul_routed_weight=False`；
    kernel 用 `a_row = offs_token // top_k` 反演回原 token 行。
  - gemm2：A 是 `[T·K, mI]`（已按路由展开、过了 silu_and_mul），`top_k=1`，
    `mul_routed_weight=True`；`topk_weights` 在 accumulator 上乘入。
- **(6) 的三个输出只进 (7) 和 (9)**：`sorted_ids` 决定每个 block 消费哪些 token 行，
  `expert_ids` 决定这个 block 用哪个 expert 的 weights，`num_post` 决定 block 总数。
  gemm1 与 gemm2 共享同一份 `(sorted_ids, expert_ids, num_post)`。
- **(10) 承担两件事**：①沿 K 轴把 C2v 加回 `[T,H]`；②把 shared 路径的
  `shared_out` 作为 residual 在 fp32 累加器上直接加进来（CUDA 版需要 kernel 外
  再 add 一次，Zeus 版把它 fuse 进同一 kernel，省一次 DRAM 往返、少一次舍入）。
- **routed_scaling_factor 归属链**：
  `(5) apply_scale_on_output=True` → `topk_weights` 带 2.5 → `(9) mul_routed_weight=True`
  把 2.5 乘进 accum → `(10) scale=1.0` 只做纯 sum。整条流水线 scaling 只在
  (5) 注入一次，不重复也不遗漏。
- **精度路径**：(7)(9)(10) 内部全程 fp32 accumulator，仅在 kernel 边界 RNE→bf16；
  (10) 的 shared 累加也在 fp32 域完成，整条流水线端到端与 REF（fp32 per-token 循环）
  对拍 `max_diff ≈ 4.9e-4`（proxy shape，见 `moe_block_full` stage）。

## Dev 脚本 Stage 顺序

1. `shared_mlp` —— 复用 Qwen 已验证算子，走 E=160 之外的 shape 做一次 sanity。
2. `gate_linear` —— `[T,H=5120] → [T,E=160]` GEMM sanity。
3. `biased_grouped_topk` —— ✅ 已对齐（2026-04-17）。
4. `moe_align_block_size` —— ✅ 已对齐（2026-04-18）。
5. `moe_grouped_gemm`（gemm1 + gemm2 两条路径）—— ✅ 已对齐（2026-04-22）。
   dev 脚本内用 proxy shape（T=16, H=256, mI=128, E=8, top_k=4）两秒跑完；
   GLM-4.7 真实 shape 的端到端 align 由 `sgl-kernel-zeus/tests/test_moe_grouped_gemm.py::test_glm4_shape` 承担。
6. `moe_sum_reduce` —— ✅ 已对齐（2026-04-22）。GEMM-2 输出 `[T, topk, H]` 按 topk
   聚合回 `[T, H]`；Zeus 扩展把 `shared_experts` 的 `[T, H]` 作为 residual 直接
   fuse 进同一个 kernel（fp32 accum + 单次 RNE）。
7. `moe_block_full` —— ✅ 已对齐（2026-04-22）。proxy shape（T=16, H=256, mI=128,
   E=8, top_k=4）下端到端 `max_diff ≈ 4.9e-4`，`mean_diff ≈ 5.7e-5`。

## 开发日志

### 2026-04-17 · 起点
- 创建本文档、`dev_glm4_moe_test.py` 骨架。
- 确认 `sgl_kernel_zeus` 当前已暴露算子：`fused_add_rmsnorm`, `rmsnorm`,
  `silu_and_mul`, `rotary_embedding`, `store_kv_cache`, `extend_attention`,
  `decode_attention` 等；**未暴露** `moe_fused_gate` / `biased_grouped_topk` /
  `moe_align_block_size` / `moe_sum_reduce` / fused-MoE GEMM。
- 今日目标：stage `biased_grouped_topk` 的 REF 对齐跑通（REF 侧先用
  `biased_grouped_topk_impl` 作为 golden，不依赖 `moe_fused_gate`）；
  Zeus 侧暂以 torch ops 拼装作为临时实现（flag 出来以便后续替换为 kernel）。

### 2026-04-17 · biased_grouped_topk 对齐完成

**Kernel 落地**（在 `/root/project/torch_zeus/sgl-kernel-zeus/` 中）：
- `csrc/moe/sgl_moe_fused_gate_sim.c` —— CPU sim，argmax-loop top-k，镜像
  `biased_grouped_topk_impl` 语义（renormalize=True, `num_fused_shared_experts=0`）。
- `csrc/moe/moe_fused_gate_zeus.cpp` —— host wrapper，`TORCH_CHECK` 收窄 scope
  到 fp32 input/bias、`num_fused_shared_experts == 0`，`int64 → int32` 收敛在
  ATen 边界。
- `csrc/moe/moe_fused_gate_kernel.py` —— Zeus Triton 参考（skill §3.5 要求，
  一个 program 一个 token 行，`tl.static_range` 显式展开 G / Gk / K 维）。
- `python/sgl_kernel_zeus/moe.py` —— Python API，返回 `(topk_weights, topk_ids)`
  tuple；bf16 input auto-cast 到 fp32。
- 注册链：`sgl_kernel_zeus_ops.h` + `common_extension.cpp` + `__init__.py`
  + `setup.py` 的 SIM / HOST sources。
- `tests/test_moe_fused_gate.py` —— 17 个 case 全通过：GLM-4.7 shape (E=160)、
  deepseek v3 shape (E=256, G=8)、scaling、renorm 不变量、bf16 auto-cast、
  `num_fused_shared_experts>0` rejected。
- Scope 对齐：E 不要求是 2 的幂（GLM-4.7 E=160 需要），`num_fused_shared_experts`
  暂不支持。

**Dev 脚本 REF vs Zeus 对齐**（`zeus_dev/dev_glm4_moe_test.py` stage
`biased_grouped_topk`）：
- 两侧输入均 fp32（router_logits bf16 → `.float()`，对齐 topk.py:745-754 的
  CUDA call site 行为）。
- 结果：
  - `ids`：集合逐位相等（sorted），REF 未排序 / Zeus 降序，`allow_permutation=True`
    通过。
  - `weights`：按 id 对齐后 `max_diff ~1.5e-8`、`mean_diff ~4.7e-9`，fp32 round-off
    级别完全吻合。
- 命令：`python zeus_dev/dev_glm4_moe_test.py --stage biased_grouped_topk`
  → `Summary: biased_grouped_topk : PASS`。

**下一步焦点**：`moe_align_block_size` —— 这是 FusedMoE 执行阶段的第一道
index 计算（`sgl_kernel.moe_align_block_size`），纯整型，shape 检查简单。

### 2026-04-17 · glm4_moe_fused_gate 特化快路径对齐完成

在通用 `moe_fused_gate` 之外再做一个 GLM-4.7 特化版本，砍掉配置自由度换取
更短的 sim 路径与更直白的 kernel 侧语义。等价关系（自 `sgl-kernel-zeus`
Python API 的 docstring）：

    glm4_moe_fused_gate(input, bias, scale)
  ≡ moe_fused_gate(input, bias, G=1, Gk=1, K=8, num_fused_shared_experts=0,
                   routed_scaling_factor=scale,
                   apply_routed_scaling_factor_on_output=True)

**Kernel 侧新增**（在 `/root/project/torch_zeus/sgl-kernel-zeus/` 中）：
- `csrc/moe/sgl_glm4_moe_fused_gate_sim.c` —— 硬编码 E=160/K=8、无 group
  分支；结构上省掉通用版的 per-group top2.sum 与 expert_keep 掩码，直接
  `sigmoid + bias → argmax×8 → renorm → *scale`。
- `csrc/moe/glm4_moe_fused_gate_zeus.cpp` —— 签名收窄到
  `(input, bias, topk_weights, topk_ids, routed_scaling_factor)`；`TORCH_CHECK`
  强制 `input.shape == [T, 160]`、`bias.shape == [160]`、fp32。
- Python API `sgl_kernel_zeus.glm4_moe_fused_gate(input, bias, routed_scaling_factor=2.5)`。
- 注册链更新：`sgl_kernel_zeus_ops.h` + `common_extension.cpp`（`m.def/m.impl`）
  + `__init__.py`（导出 + `__all__`） + `setup.py`（`SIM_SOURCES` /
  `HOST_SOURCES` 各加一条）。

**Dev 脚本**（`zeus_dev/dev_glm4_moe_test.py`）：
- 新增 stage `glm4_moe_fused_gate`（保留 `biased_grouped_topk` 以持续验证
  通用版）。
- REF 侧 `apply_routed_scaling_factor_on_output=True` —— 这是对齐两侧
  weights 的关键，否则 weights 会差一个 `scale` 倍。
- 结果：
  - `ids`：集合逐位相等，`allow_permutation=True` 通过。
  - `weights`：按 id 对齐后 `max_diff ~6e-8`、`mean_diff ~1.2e-8`。
  - 行和不变量：`w[row].sum() == routed_scaling_factor`（=2.5）与 REF 完全
    一致，确认 scaling 已正确 fuse 进 weights。
- 命令：`python zeus_dev/dev_glm4_moe_test.py --stage glm4_moe_fused_gate`
  → `Summary: glm4_moe_fused_gate : PASS`；`--stage all` 两个 stage 同时
  PASS。

**两个 stage 的语义差异**（记在这里避免以后 debug 对半天）：

| | `biased_grouped_topk`（通用） | `glm4_moe_fused_gate`（特化） |
|---|---|---|
| Zeus kernel | `moe_fused_gate` | `glm4_moe_fused_gate` |
| REF `apply_scaling_on_output` | `False`（MoE 外 `*scale`） | `True`（fuse 进 weights） |
| shape 约束 | E/G/K 灵活 | 硬编码 E=160 K=8 G=1 Gk=1 |
| 每行 weights sum | 1.0 | `routed_scaling_factor` |
| 上游调用点 | 通用 MoE / 其它模型 | 仅 GLM-4.7 路径 |

### 2026-04-17 · glm4_moe_fused_gate 切换到 bf16 IO 后的 dev 适配

kernel 侧把 `glm4_moe_fused_gate` 的 IO 从 fp32 改成 bf16（`input` /
`bias` / `topk_weights` 都是 bf16，`topk_ids` 仍 int32）。sim.c 内部走
**混合精度**：`scores = bf16_roundtrip(sigmoid(fp32(x)))` → `choice_scores =
fp32(scores) + fp32(bias)`（top-k compare 在 fp32 域）→ gather raw weights
（bf16 域）→ renormalize + `*scale` 在 fp32，最后 bf16 round-to-store。

**Dev 脚本 stage 的 3 处关键改动**（见 `dev_glm4_moe_test.py` 的
`test_glm4_moe_fused_gate`）：

1. **REF bias 故意保持 fp32**：两侧 `router_logits` 都是 bf16，但喂给
   `biased_grouped_topk_impl` 的 bias 是 fp32（Zeus 侧 API 合约要 bf16，
   单独准备一份 bf16 副本喂 kernel）。原因：Zeus sim 的 top-k compare
   domain 是 `fp32(scores) + fp32(bias)`；REF 里 `scores(bf16) +
   bias(fp32)` 自动提升到 fp32，这样两边排名顺序一致。若给 REF 传 bf16
   bias，比较留在 bf16，会因低精度产生大量 tie-break 翻转，sorted-set 都
   对不上。
2. **weights tolerance 放宽到 `atol=rtol=5e-3`**：bf16 ~2-3 位小数精度，
   加上 `*scale=2.5` 约 0.2% 相对误差，实测 `max_diff ~2e-3`、`mean_diff
   ~6e-4`。
3. **row sum 不再 bit-exact**：REF 2.5039 / Zeus 2.5020，都在 `scale=2.5`
   的 bf16 量化范围内，确认 scaling 已 fuse。

**结果**：`glm4_moe_fused_gate` 单 stage PASS，`--stage all` 两个 stage
同时 PASS（`biased_grouped_topk` 未受影响，仍走 fp32 路径）。

**教训**：bf16 对齐最容易踩的坑是"比较域 vs 存储域"错位。写 dev 对齐脚
本前，一定先读 sim.c 的 dataflow 注释（那份 kernel header 明写了
"topk compare path 用 fp32"），再决定 REF 哪些张量保持 fp32。

### 2026-04-18 · moe_align_block_size 对齐完成

FusedMoE 执行阶段的第一道 index 簿记：把 `[T, K]` 的 topk_ids 展平后按 expert
bin 排序，给每个 block_size 大小的块标一个 expert id，便于 fused-MoE GEMM 分片。
完全对齐 `sgl_kernel.moe_align_block_size`（`sgl-kernel/csrc/moe/moe_align_kernel.cu`）。

**Kernel 落地**（在 `/root/project/torch_zeus/sgl-kernel-zeus/` 中）：
- `csrc/moe/sgl_moe_align_block_size_sim.c` —— CPU sim，顺序实现：pre-fill
  → count → pad-to-block → exclusive prefix-sum → binary search expert_ids
  → per-bin cursor scatter。输入 int32 topk_ids，输出全 int32。
- `csrc/moe/moe_align_block_size_zeus.cpp` —— host wrapper，`TORCH_CHECK` 覆盖
  所有 5 个张量的 contiguous / int32 约束；scalar 参数 `int64 → int32` 收敛
  在 ATen 边界；对 `cumsum_buffer / expert_ids / sorted_token_ids` 的 size
  也做下限检查。
- `csrc/moe/moe_align_block_size_kernel.py` —— Zeus Triton 参考（skill §3.5，
  单 program id + 显式 `tl.static_range` / `tl.range` 展开 expert/block 维度）。
- `python/sgl_kernel_zeus/moe.py` —— 两个 Python API：
  - `moe_align_block_size(...)`：一比一镜像 `sgl_kernel.moe_align_block_size`，
    caller 自行分配所有 buffer。
  - `moe_align_block_size_alloc(topk_ids, block_size, num_experts)`：按
    sglang Python wrapper 的分配策略（`max_num_tokens_padded = numel +
    (num_experts + 1) * (block_size - 1)`，`cumsum_buffer = num_experts + 2`）
    分配并直接返回 `(sorted_ids, expert_ids, num_tokens_post_pad)`，用于
    `Glm4MoeSparseMoeBlock` 组装时的一次性调用。
- 注册链：`sgl_kernel_zeus_ops.h` + `common_extension.cpp`（`m.def/m.impl`）
  + `__init__.py`（导出 + `__all__`） + `setup.py`（SIM_SOURCES / HOST_SOURCES
  各加一条）。
- `tests/test_moe_align_block_size.py` —— 36 个 case 全通过：GLM-4.7 shape
  (real_num_experts=160, K=8) × {block_size=32/64/128} × {pad True/False} +
  各类小 / 边界 shape (1/33/128 tokens, 1/2/6/8 topk, 8/64/160 experts) +
  alloc wrapper 与 low-level 一致性 + int64 topk_ids 被 reject。

**Scope 对齐**：
- `topk_ids` dtype 收窄到 int32（CUDA 版原本支持所有整型；我们下游唯一调用
  点是 `sgl_kernel_zeus.moe_fused_gate / glm4_moe_fused_gate` 的 int32 输出，
  v1 不做 int64 dispatch）。
- `num_experts` 语义与 sglang Python wrapper 一致：caller 传入 `real + 1`
  （bin 0 = EP-filtered / padding）。
- `pad_sorted_token_ids` 支持 True / False。True 时 kernel 自己 pre-fill；
  False 时 caller 负责（测试里由于 Zeus `fill_` 不支持 int32，改为 CPU
  full + `.to("zeus")` 绕开）。

**Dev 脚本**（`zeus_dev/dev_glm4_moe_test.py`）：
- 新增 stage `moe_align_block_size`，REF 侧用纯 torch 复现 count/pad/prefix/
  scatter，Zeus 侧走 `moe_align_block_size_alloc`。
- 对齐 4 项不变量：
  - `num_tokens_post_pad`：标量完全相等。
  - `expert_ids[0..num_blocks)`：逐位相等。
  - `sorted_token_ids` 按 bin 分段比集合（同 bin 内 token 顺序不保证，
    与 CUDA atomic 语义一致；Zeus sim 是 i-order 写入，REF 也是 i-order，
    实测集合 + 顺序都相等）。
  - 尾部 sentinel：`sorted_token_ids[num_tokens_post_pad:] == numel`。
- 结果：
  - `num_tokens_post_pad`：zeus=5824 ref=5824。
  - `expert_ids`：91 blocks，逐位 PASS。
  - `sorted_token_ids`：per-bin 集合 PASS。
  - sentinel tail 长度 4447，全部 = numel=128。
- 命令：`python zeus_dev/dev_glm4_moe_test.py --stage moe_align_block_size`
  → `Summary: moe_align_block_size : PASS`；`--stage all` 三个 stage 同时
  PASS。

**下一步焦点**：`fused_moe_experts` —— gemm1(w13/gate_up) + per-expert
SiluAndMul + gemm2(w2/down) 组合，这是 MoE-FFN 段真正的算力热点，需要实现
grouped / masked GEMM 的 Zeus 版本。

### 2026-04-22 · moe_grouped_gemm（fused-experts GEMM driver）对齐完成

把 `sgl_kernel.fused_moe_kernel` 的"block 级 grouped GEMM"搬到 Zeus —— 这是
MoE-FFN 段的算力热点。gemm1（w13 / gate_up）与 gemm2（w2 / down）复用同一份
kernel，通过 `top_k` 和 `mul_routed_weight` 两个参数切路径。

**Kernel 落地**（在 `/root/project/torch_zeus/sgl-kernel-zeus/` 中）：

- `csrc/moe/sgl_moe_grouped_gemm_sim.c` —— CPU sim，双核 N-split + 显式三
  重循环（`m_block` / `n_block` / `k_block`）。fp32 acc + bf16 store + RNE；
  `offs_token`（= `flat_idx ∈ [0, T*topk)`）从 fp32 bit-exact sorted_ids
  cast，`a_row = offs_token // top_k` 反演回 A 的行；`expert_id == -1`
  写零 tile，保留 `token_mask`；可选 routed-weight 融合在 accumulator 上。
- `csrc/moe/moe_grouped_gemm_kernel.py` —— Zeus Triton 参考，`make_block_ptr`
  + `tl.advance` 的"N-K slab"形态（对齐
  `docs/gemm_normal_dense_fp8xfp8_bf16dst_bf16acc_128x512x512_2core.py`），
  B 以 `memory_type='weight'` 走 weight DRAM，A 以 pointer-vector gather
  （因为 `sorted_token_ids` 重排过）。
- `csrc/moe/moe_grouped_gemm_zeus.cpp` —— host wrapper，`TORCH_CHECK` 覆盖
  contiguous / dtype（bf16 A/B/C、fp32 sorted_ids/expert_ids、i32 num_post、
  bf16 topk_weights）/ `N % (CORE_NUM * BLOCK_N=256) == 0` / `top_k>=1` /
  `mul_routed_weight=True` 时 `topk_weights` 必给。
- `python/sgl_kernel_zeus/moe.py` —— `moe_grouped_gemm(A, B, C, sorted_ids,
  expert_ids, num_post, num_valid_tokens, top_k, topk_weights=None,
  mul_routed_weight=False)`；暴露 `MOE_GROUPED_GEMM_BLOCK_M/N/CORE_NUM = 64/128/2`
  常量便于 caller 对齐 `block_size`。caller 自带 `C` —— 因为 gemm1 的 N=2·mI，
  gemm2 的 N=H，kernel 不猜。
- 注册链：`sgl_kernel_zeus_ops.h` + `common_extension.cpp` + `__init__.py`
  + `setup.py`。
- `tests/test_moe_grouped_gemm.py` —— 9 个 case 全通过：4 个小 shape（T/H/mI/E/K
  矩阵）+ 2 个 GLM-4.7 真实 shape（`T=32`，gemm1：`H=5120, N=3072, topk=8`；
  gemm2：`mI=1536, N=5120, topk=1`）+ 3 个 reject case（N misalign / 非 bf16
  / `mul_routed_weight` 缺权重）。bf16 精度下小 shape `atol=5e-2`，GLM-4.7
  K=5120 `atol=5e-2`。

**Scope v1 显式不做**：fp8/int8/int4 量化、per-expert bias、block-wise / per-channel
scaling、`c_sorted=True`（TMA 下行）、`filter_expert=False`、kernel 内部 chunk。

**语义对齐（v1 严格复刻 CUDA `fused_moe_kernel` 的 `c_sorted=False` 分支）**：
- `offs_token < num_valid_tokens` 是**唯一** token mask（sentinel / pad gap /
  EP 过滤都由此 cover）。
- `a_row = offs_token // top_k`：gemm1 传原 topk（A=[T, H]），gemm2 传 1
  （A=[T·topk, mI]，因为 silu_and_mul 之后已按路由展开）。
- `filter_expert=True`：`expert_id == -1` 写零 tile，**保留 `token_mask`**（pad
  行不写），这样下游 `moe_sum_reduce` 累加 0 不污染。
- `MUL_ROUTED_WEIGHT` 在 accumulator 上融合（不是 input 上），按 `offs_token`
  直接索引扁平的 `topk_weights[T*topk]`。
- C 以 `c_ptr + offs_token[:, None] * N + offs_cn[None, :]` scatter 回 `[T·topk, N]`，
  下游按 `flat_idx // topk` 聚合。

**Dev 脚本**（`zeus_dev/dev_glm4_moe_test.py`）：
- 新增 stage `moe_grouped_gemm`，gemm1 + gemm2 两条 sub-test 串跑。
- **proxy shape**：T=16, H=256, mI=128, E=8, top_k=4（约束：`N≥CORE_NUM·BLOCK_N=256`，
  所以 H/2·mI 最小 256）。真实 GLM-4.7 权重 w13≈4.8 GB、w2≈2.4 GB，超过 CPU
  ref 可接受的内存预算，因此真 shape 对齐由 `sgl-kernel-zeus/tests/test_moe_grouped_gemm.py::test_glm4_shape`
  承担（已通过）。
- REF 侧：`_ref_moe_grouped_gemm`（dev 脚本内）—— 块级循环，镜像 kernel 的
  `sorted_ids` 消费顺序和 `offs_token // top_k` / filter-expert / mul_routed_weight
  分支；fp32 accum，bf16 store。
- 结果：
  - `gemm1`（w13/gate_up）：`max_diff ≈ 9.77e-4`，`mean_diff ≈ 6e-8`
  - `gemm2`（w2/down，mul_routed_weight=True）：`max_diff = 0`（本随机种子下
    bf16 量化恰好 bit-exact 相等）
- 命令：`python zeus_dev/dev_glm4_moe_test.py --stage moe_grouped_gemm`
  → `Summary: moe_grouped_gemm : PASS`；`--stage all` 四 stage 同时 PASS。

**关键设计抉择（porting 到芯片时参考 `sgl-kernel-zeus/docs/moe_grouped_gemm.md`）**：
1. **2-core N-split**：每核拿 `[E, N/CORE_NUM, K]` 的权重子块，B 一律走 weight
   DRAM，两核共享 `sorted_ids / expert_ids` 索引 buffer。相比切 M 会让两核读
   不相交的 expert 权重（DRAM 利用率差），切 N 对 MoE 场景（E 多、N 适中）最合适。
2. **BLOCK_M 的"GPU-friendly vs Zeus-friendly"取舍**：当前 `BLOCK_M=64` 对齐
   CUDA 习惯，便于 Triton 参考编过；**porting 到芯片时建议改成 4 或 8**（Zeus
   是高带宽 / 低计算密度的设计，且 A gather 与 C scatter 的离散 DMA 事务数 ∝ BLOCK_M），
   届时 `moe_align_block_size` 的 `block_size` 要同步改。
3. **"gather / scatter 事实上是多条 line DMA"**：Zeus 无原生 gather / scatter，
   编译器会把 `a_tile = tl.load(a_row_ptrs)` / `tl.store(c_row_ptrs)` / `tl.load(topk_weights_ptr + offs_token)`
   lower 成 `BLOCK_M` 条独立 load / store（每行内部仍 contiguous）。因 BLOCK_M
   小，代价可控；但是这也解释了为什么"不切 N 而切 M"在 Zeus 上更亏。

**下一步焦点**：`moe_sum_reduce` —— gemm2 输出 `[T·k, H]` 按 topk 聚合回 `[T, H]`
（gemm2 的 `MUL_ROUTED_WEIGHT=True` 已经把权重融入 accumulator，所以 sum-reduce
纯粹是按 token 聚合 topk 条路由的 `[H]` 向量）。之后拼 `moe_block_full` 对齐端到端。

### 2026-04-22 · moe_sum_reduce（+ 可选 shared-expert fuse）对齐完成

fused-MoE FFN 流水线的**收尾**节点：把 gemm2 输出的 `[T, topk, H]`（每 token 的
topk 份 expert 结果散落在 topk 行上）沿 topk 轴求和，回到 `[T, H]`。对齐
`sgl_kernel.moe_sum_reduce`（`sgl-kernel/csrc/moe/moe_sum_reduce.cu`），并**额外**
把 GLM-4.7 里两路并行 FFN（routed experts + shared experts）合流的 elementwise
add **吸收**进这同一个 kernel —— 这是 Zeus 侧相对 CUDA 签名的扩展。

**Kernel 落地**（在 `/root/project/torch_zeus/sgl-kernel-zeus/` 中）：

- `csrc/moe/sgl_moe_sum_reduce_sim.c` —— CPU sim，2 核 H-split。signature struct
  含 `shared_output_ptr` + `has_shared` flag；语义：
  `output[t, h] = scale * sum_{k} input[t, k, h] + (shared[t, h] if has_shared else 0)`。
  fp32 accumulator 累加 topk 载入，`acc *= scale`，若 `has_shared` 则 fp32 add
  shared（**不乘 scale**），最后整体 RNE 到 bf16 存回。
- `csrc/moe/moe_sum_reduce_kernel.py` —— Zeus-friendly Triton 参考，全量使用
  `tl.make_block_ptr` + `tl.advance`：input 用 3D block_ptr `[T, topk, H]` +
  tile `[BLOCK_T, BLOCK_K, BLOCK_H]`（让 compiler 能把 topk 相邻 slot 合并成
  一次更大带宽的搬运），K 轴归约用 `tl.sum(axis=1)` 在 fp32 域完成；shared /
  output 用 2D block_ptr `[T, H]` + tile `[BLOCK_T, BLOCK_H]`。`HAS_SHARED`
  为 `tl.constexpr`，host 根据 `shared_output.has_value()` 设置，分支在编译期
  消除。
- `csrc/moe/moe_sum_reduce_zeus.cpp` —— host wrapper，打包
  `SglMoeSumReduceArgs`（含 `shared_output_ptr` + `has_shared`）。`TORCH_CHECK`
  覆盖：contiguous、bf16、3D input / 2D output、shape 匹配、`H % CORE_NUM == 0`；
  提供 `shared_output` 时额外校验 bf16 / 2D / shape = output.shape。
- `python/sgl_kernel_zeus/moe.py` —— `moe_sum_reduce(input, output,
  shared_output=None, routed_scaling_factor=1.0)`；暴露
  `MOE_SUM_REDUCE_CORE_NUM=2` / `MOE_SUM_REDUCE_BLOCK_H=128`。
- 注册链：`sgl_kernel_zeus_ops.h` + `common_extension.cpp`（schema
  `moe_sum_reduce(Tensor input, Tensor! output, Tensor? shared_output, float routed_scaling_factor) -> ()`）
  + `__init__.py` + `setup.py`。
- `tests/test_moe_sum_reduce.py` —— 17 case 全通过：5 个 plain-sum（topk
  2/4/8/9, scale 1.0/2.5）+ 1 个 GLM-4.7 真实 shape（T=32, topk=8, H=5120）+
  4 个 shared-fuse 小 shape + 1 个 GLM-4.7 shape + shared + 6 个 reject case
  （非 bf16 / 非 contiguous / H 不整除 CORE_NUM / shape mismatch /
  shared_output shape / shared_output dtype）。
- `docs/moe_sum_reduce.md` —— 完整设计文档，含 porting 注意事项。

**Scope v1**：bf16 I/O + fp32 accumulator；`H % CORE_NUM == 0`；`CORE_NUM=2`。
CUDA 版有 fp16/bf16/fp32 dispatch + 多条 fast path（warp-per-token、bf16
vec-16），Zeus 先做最小闭环 + shared-expert fuse。

**routed_scaling_factor 的归属**（整条 MoE-FFN 一致性）：GLM-4.7 Zeus 路径下
`glm4_biased_grouped_topk` 以 `apply_routed_scaling_factor_on_output=True` 把
scaling 折进 `topk_weights`；`moe_grouped_gemm` 的 gemm2 再以
`mul_routed_weight=True` 把该权重乘进 accumulator。所以到 `moe_sum_reduce`
**典型调用是 `scale=1.0`**，它只负责求和。`routed_scaling_factor` 形参
保留是为了匹配 CUDA 签名，也便于 dev stage 校验签名可用性。

**shared-expert fuse 的动机 & 精度论证**（关键设计抉择，porting 时保留）：

GLM-4.7 的 FFN 是两路并行 —— routed experts（MoE 基建）+ shared experts
（独立 dense MLP, `n_shared_experts=1`，**不**走 MoE 基建，因为配置
`num_fused_shared_experts=0`）。两路最终要做 `final = routed_sum + shared_out`
合流。把这个 elementwise add **吸收**进 `moe_sum_reduce` 有两个好处：

1. 省一次 `[T, H]` 的 DRAM 往返 + 一次 kernel launch。
2. **更高精度**：独立 add 是 `bf16 + bf16 → bf16`，有一次中间舍入；fuse 版本
   shared 直接在 fp32 累加器上加，再整体 RNE，只有一次舍入。

传 `shared_output=None`（默认）即为纯 sum 行为，签名向后兼容 CUDA 版。

**设计参数**：
- `CORE_NUM=2`：H 轴切分因子。
- `BLOCK_T=4`：T 轴分块，每 tile 覆盖 4 个 token，K-loop 内对 `[BLOCK_T, BLOCK_H]`
  做矩阵向量-加（而不是逐 token）。
- `BLOCK_K=2`：topk 轴分块。topk=8/9 时 K-loop = 4/5 iters；3D tile 让 DMA
  尽可能合并 topk 相邻 slot。
- `BLOCK_H=128`：H 轴分块粒度。
- `HAS_SHARED`：`tl.constexpr`，host 根据 `shared_output.has_value()` 设，编
  译期消除分支。

**Dev 脚本**（`zeus_dev/dev_glm4_moe_test.py`）：
- 新增 stage `moe_sum_reduce`，内部 4 条 sub-test（plain × {1.0, 2.5} +
  shared × {1.0, 2.5}），使用 GLM-4.7 对齐形态（T=`num_tokens`, topk=8,
  H=5120）。
- REF 侧 `_ref_moe_sum_reduce`：`x.float().sum(dim=1) * scale + (shared.float()
  if shared else 0) → bf16 RNE`，严格镜像 kernel 的精度路径。
- 结果（`T=16, topk=8, H=5120`）：4 条 sub-test 均 PASS，bf16 量化级别
  `max_diff ≈ 1-3e-2` / `mean_diff ≈ 几个 1e-4`，落在 `atol=rtol=2e-2` 范围内。
- 命令：`python zeus_dev/dev_glm4_moe_test.py --stage moe_sum_reduce`
  → `Summary: moe_sum_reduce : PASS`；`--stage all` 五 stage 同时 PASS。

**Kernel 层自测**：`sgl-kernel-zeus/tests/test_moe_sum_reduce.py` 17/17 PASS；
MoE 全量回归 `tests/` 70/70 PASS（150s），确认新 kernel 没有撞到旧路径。

**porting 到芯片时的注意事项**（完整版见 `docs/moe_sum_reduce.md` §7）：
1. Tile 形状 `BLOCK_T=4/BLOCK_K=2/BLOCK_H=128` 是保守默认，sum-reduce 对算力
   要求极低、瓶颈在带宽，按 DMA 对齐 & 向量宽度微调。
2. `scale=1.0` 常态 —— 可以把 `scale` 作为 constexpr 分支出去（scale==1 ⇒
   省一次乘法）。
3. 精度约定：routed sum / scale / shared residual 全程 fp32，**最后一次**
   RNE 到 bf16 —— porting 时务必保持这个顺序，独立 `bf16+bf16` add 会偏离
   参考 1e-3 ~ 1e-2 量级。
4. `HAS_SHARED` constexpr 消除运行时分支；`shared_output` 只参与每 tile
   一次连续 load，不进 K-loop，是纯连续 DMA。

**下一步焦点**：`moe_block_full` —— 组装
`glm4_biased_grouped_topk → moe_align → moe_grouped_gemm(gemm1) →
silu_and_mul → moe_grouped_gemm(gemm2) → moe_sum_reduce(+shared)`，
对齐 `Glm4MoeSparseMoeBlock.forward_normal` 单卡、不含 a2a / 不含
fused_shared_experts 的端到端输出。

### 2026-04-22 · moe_block_full 端到端对齐完成

MoE-FFN 流水线的收官 stage：把前 5 个已对齐的 kernel 按
`Glm4MoeSparseMoeBlock.forward_normal` 的顺序串起来，端到端与纯 torch REF
对拍。

**组装顺序**（单卡、非 fused_shared_experts、非 a2a）：
1. `biased_grouped_topk(scale fused on output=True)`
   → `(topk_weights[T,K] fp32, topk_ids[T,K] int32)`
2. `moe_align_block_size_alloc(topk_ids, block_size=64, E)`
   → `(sorted_ids, expert_ids, num_tokens_post_pad)`
3. `moe_grouped_gemm(gemm1)`: `x[T,H] × w13[E,2·mI,H] → C1[T·K, 2·mI]`，
   `top_k=原始 K`，`mul_routed_weight=False`
4. `silu_and_mul(C1) → C1_silu[T·K, mI]`
5. `moe_grouped_gemm(gemm2)`: `C1_silu × w2[E,H,mI] → C2[T·K, H]`，
   `top_k=1`，`mul_routed_weight=True`（routing weights 在 accumulator 上融进去）
6. `moe_sum_reduce(input=C2.view(T,K,H), output=[T,H], shared_output=…,
   routed_scaling_factor=1.0)`：topk 求和 + shared residual fuse

**routed_scaling_factor 的归属**（整条流水线一致性）：
- 上游 `biased_grouped_topk(apply_routed_scaling_factor_on_output=True)`
  把 `scale` 融进 `topk_weights`。
- gemm2 的 `mul_routed_weight=True` 把这个 weights 乘进 accumulator。
- 到 `moe_sum_reduce` 传 `scale=1.0` —— 它只做 topk sum + shared residual。

REF 侧走 **一致语义**：`biased_grouped_topk_impl(apply_scale_on_output=True)`
→ per-token per-expert 纯 torch 循环 `final[t] += w[t,k] * mlp_expert(x[t], e)`
→ `final += shared_output`，不再显式 `*scale`。

**Dev 脚本**（`zeus_dev/dev_glm4_moe_test.py`）：

- 新增 stage `moe_block_full`，proxy shape 与 `moe_grouped_gemm` 保持一致
  （T=16, H=256, mI=128, E=8, top_k=4）—— 真实 GLM-4.7 权重（w13≈4.8GB）
  无法在 CPU REF 跑。
- **范围边界**：`gate` Linear 与 `shared_experts` MLP 的 GEMM 不是本 stage
  的测试对象（已在 qwen demo 的 transformer_block / 早期 MoE stage 验证过），
  所以 `router_logits` 与 `shared_output` 在 `REF_DEVICE` 上用纯 torch
  **预计算一次**，两条路径共享同一份输入，专心比较 MoE 核心 6-kernel
  pipeline 的组装。
- REF 侧辅助函数 `_ref_moe_core(x, w13, w2, topk_weights, topk_ids, mI)`：
  fp32 per-token per-expert 循环，镜像 `FusedMoE.forward` 在
  `should_fuse_routed_scaling_factor_in_topk=True` 下的语义。
- 结果：
  - `topk_ids`：集合 PASS（与 `biased_grouped_topk` stage 一致，排列不定）。
  - `final`：`max_diff ≈ 4.9e-4`、`mean_diff ≈ 5.7e-5`，远小于 `atol=rtol=5e-2`
    的容忍阈值。头 6 个元素前 4 位小数与 REF 字面一致。
- 命令：`python zeus_dev/dev_glm4_moe_test.py --stage moe_block_full`
  → `Summary: moe_block_full : PASS`；`--stage all` 六 stage 同时 PASS。

**为什么端到端精度这么高（~5e-4）而不是"bf16 多级累加"级别（~5e-2）**：
- 整条 Zeus 流水线的每个 GEMM / reduce 都用 fp32 accumulator，只在 kernel
  边界存回 bf16（`moe_grouped_gemm` / `moe_sum_reduce` 均如此），与 REF
  的 fp32 per-token 循环语义非常接近。
- `moe_sum_reduce` 的 shared-fuse 路径在 fp32 累加器上加 shared，再整体
  RNE —— 与 REF 的 `(routed_fp32 + shared_fp32).to(bf16)` 完全同一路径。
- 主要误差来自 bf16 gemm1 / gemm2 的输入 bf16 化（weights / activations
  都先 cast 到 bf16 再参与 GEMM），随机 0.1 缩放下累加量级本就不大。

**下一步**：MoE-FFN 段本身已闭环。上层集成（把 Zeus MoE 核心接回
`Glm4MoeSparseMoeBlock` 或等价 runner 路径）留给模型装配环节。

## 注意事项

- **top_k 语义**：代码里 `self.top_k = config.num_experts_per_tok + num_fused_shared_experts`，
  在单卡非 fused-shared-experts 情况下 `num_fused_shared_experts=0`，
  所以 top_k=8。
- **dtype**：`router_logits` 是 bf16（来自 `F.linear(bf16, bf16)`），
  `correction_bias` 是 fp32；`moe_fused_gate` 要求 `gating_output` cast 成 fp32。
  输出 `topk_weights` fp32，`topk_ids` int32。
- **renormalize**：GLM-4.7 `norm_topk_prob=true`，且 CUDA kernel 把
  `routed_scaling_factor` 也 fuse 进了 topk（见 `topk.py:745`）。我们的
  REF / Zeus 两侧要保持一致的 "要/不要 fuse scaling" 选择，避免对齐时被这项
  差异带偏。

### 2026-05-25 · GAP-3 闭环：`moe_block_full` 切到 Zeus `linear_bf16`
- **背景**：算子表 #1 / #3 / #4 的 `shared gate_up_proj` / `shared down_proj` /
  `gate Linear` 历史上一直在 host `torch.nn.functional.linear` 上算
  （`dev_glm4_moe_test.py::test_moe_block_full` 第 800-817 行明文承认），违反
  `glm5next_dsa_block_zeus_flow.md` GAP-3 的"sublayer 全 Zeus device-resident"
  目标。本日切到 `sgl_kernel_zeus.linear_bf16`（2026-05-25 同日落地）。
- **改动**（`dev_glm4_moe_test.py::test_moe_block_full`）：
  - 3 颗 host Linear 全部改成 Zeus `linear_bf16`：
    - `gate Linear`: `linear_bf16(x_z, gate_w_lmem) → router_logits_bf16_z` → `.float()` 喂 topk
    - `shared gate_up`: `linear_bf16(x_z, sh_gu_lmem) → sh_gu_z`
    - `shared down`:   `linear_bf16(sh_silu_z, sh_dp_lmem) → shared_out_z`
  - 3 颗 weight 全部 LocalMem 装包：`torch.zeus.local_memory.from_tensor(...,
    kind="weight", Tr=1, Tc=1)`，与 `mhc_pre_norm_split` / `dsa_q_a_proj_norm`
    同套路
  - 中间 `silu_and_mul` 复用现有 `sgl_kernel_zeus.silu_and_mul`
  - **测试方法学**：REF 与 Zeus **共享同一份 Zeus-computed `router_logits` 和
    `shared_output`**（Zeus 算出后 `.cpu()` 传入 REF 路径），避免 `linear_bf16`
    (fp32 acc + bf16 RNE) vs `F.linear` (bf16 acc) 的量化噪声造成 top-k 路由
    翻转把测试变成"linear_bf16 测试"。`linear_bf16` 的独立正确性由
    `sgl-kernel-zeus/tests/test_linear_bf16.py` 14/14 PASS 保证。
- **验收**（torch10_312 env）：
  - `dev_glm4_moe_test.py --stage moe_block_full`: **PASS**
    - topk_ids：permutation OK (mismatch_cells=34/64 是 ids 排序差异，集合一致)
    - **final `max_diff = 4.88e-4`**（与 GAP-3 闭环前 ~5e-4 同量级，无回归）
    - mean_diff = 5.69e-5
    - ZEUS final[0, :6] 与 REF final[0, :6] **逐位完全一致**（bf16 精度内）
  - 全 6 stage 跑全：biased_grouped_topk / glm4_biased_grouped_topk /
    moe_align_block_size / moe_grouped_gemm / moe_sum_reduce / moe_block_full
    均 PASS
- **GAP-3 在 MoE 路径上完全消除**：`test_moe_block_full` 的 Zeus 路径现在是
  9 颗 sgl-kernel-zeus 算子串接 device-resident：
  1. `linear_bf16` (gate)
  2. `linear_bf16` (shared gate_up)
  3. `silu_and_mul` (shared)
  4. `linear_bf16` (shared down)
  5. `biased_grouped_topk`
  6. `moe_align_block_size_alloc`
  7. `moe_grouped_gemm` (gemm1)
  8. `silu_and_mul` (per-expert)
  9. `moe_grouped_gemm` (gemm2, mul_routed_weight)
  10. `moe_sum_reduce` (+shared residual fuse)

  整段 MoE-FFN 在 Zeus 上**完全 device-resident**（host 仅做 weight LocalMem
  pack 一次性初始化 + 输入 residual 的 H→D transfer，无中间 host compute）。
- **下一步焦点**：把 `dev_glm5next_block_decode_test.py::zeus_moe_decode` 里
  host 端的 `_moe_router_and_shared_ref` 也切到 `linear_bf16`，让 GLM5-Next
  整 block decode 真正闭环 GAP-3；之后再用同一颗 kernel 把 Linear-attn 5 个
  projection 切到 Zeus，让 `linear_attn_block` 也走向完全 device-resident。
