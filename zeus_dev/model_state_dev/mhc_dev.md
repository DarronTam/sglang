# mHC (Manifold-Constrained Hyper-Connections) Zeus 适配开发追踪

> 对齐目标：把 **mHC**（多残差流 + Birkhoff-polytope 约束的 hyper-connection）在纯
> Zeus（无 CUDA）环境逐算子跑通。DeepSeek-V4 (`hc_*` 命名) 和 GLM-5 Next
> (`mhc_*` 命名) 共享同一机制，本文档用 `mhc_*` 命名。文档与 `glm4_moe_ffn_dev.md`
> **互相独立**——后者覆盖 MoE-FFN 段本身，本文档覆盖**包在 MoE-FFN 外面的
> mHC wrapper**；两份工作在 `dev_mhc_test.py::mhc_block_ffn_full` stage 汇合。

## 1. 范围与方法

- **起点**：transformer block 入口的 **多流 residual** `[T, mhc, H]`（从 embedding
  用 `expand_to_mhc` 或上一层 block 的输出接入）。
- **终点**：sublayer 处理完之后回写的多流 residual `[T, mhc, H]`。
- **切片**：单 device、单 layer；attn-side 与 FFN-side 各自独立包一次 mHC。
  本文档主聚焦 **FFN-side**（因为 FFN sublayer 刚好是 `glm4_moe_ffn_dev.md`
  已对齐的 MoE block）；attn-side 只给一段"照抄 FFN-side"的注记，留给
  `dev_kimi_linear_attn_test.py` 那边集成。
- **不含**：TP/EP/DeepEP/NextN；mHC 的训练反向（仅前向对齐）；多层堆叠
  （仅验证单次 pre/post 往返）；LM head 的 `mhc_head_compute_mix`（不在
  MoE-FFN 热路径）。
- **对齐方式**：
  - REF 侧直接 import 官方 `TileKernels/tile_kernels/torch/mhc.py` 的纯 torch
    golden（`ref_tile_kernels_torch/mhc.py`，85 行）。
  - Zeus 侧逐算子落地 `sgl_kernel_zeus.mhc_*`，禁止落到 torchnative fallback。

## 2. 参考代码来源（已下载到 `zeus_dev/`）

| 路径 | 职责 | 用途 |
|---|---|---|
| `ref_deepseek_v4_model.py` | DeepSeek-V4 `Block.forward` 顶层组装（`hc_pre` / `hc_post` 方法） | 理解 block-level 包裹顺序 |
| `ref_deepseek_v4_kernel.py` | `hc_split_sinkhorn` 等的原生 CUDA kernel 引用（带 TileLang/Triton） | 了解生产侧 fused 路径 |
| `ref_tile_kernels_mhc/*.py` | 10 个 TileLang kernel 文件 | 后续 Zeus kernel port 的 shape/tile 参考 |
| `ref_tile_kernels_modeling_mhc/functional.py` | 顶层 API `mhc_pre` / `mhc_head`，包含 inference 大融合分支 | 顶层调用顺序权威来源 |
| `ref_tile_kernels_modeling_mhc/ops/*.py` | 每个 kernel 的 `torch.autograd.Function` 包装 | 单算子前后向签名参考 |
| **`ref_tile_kernels_torch/mhc.py`** | **纯 torch 的 REF** — 7 个金函数覆盖全 pipeline | **本 stage test REF 直接来源** |

## 3. Config 字段三方对照

| DeepSeek-V4 `hc_*` | GLM-5 Next `mhc_*` | TileKernels API | 含义 |
|---|---|---|---|
| `hc_mult=4` | `mhc_num_residual_streams=4` | `mhc_mult=4` | N = 残差流条数。**当前 kernel 硬编码 N=4**（"currently only 4 is guaranteed to work"，见 `functional.py:22`） |
| `hc_sinkhorn_iters=20` | *(缺省 → 20)* | `sinkhorn_repeat=10` (默认) | Sinkhorn-Knopp 迭代数。**以 config 为准**（DeepSeek-V4 prod = 20，TileKernels API 默认 10） |
| `hc_eps=1e-6` | *(缺省)* | `sinkhorn_eps=1e-6` | Sinkhorn 数值稳定 eps |
| *(无)* | `mhc_tau=0.05` | **不对应** | **GLM 内部扩展**，可能作用于 Sinkhorn 前 softmax 温度，语义待 HF `modeling_glm5_next.py` 确认 |
| *(无)* | `hres_vwnstyle=true` | **不对应** | **GLM 内部扩展**，语义悬空 |
| *(无)* | *(缺省)* | `pre_eps=1e-6`, `post_mult_value=1.0`, `norm_eps=1e-6` | sigmoid bias、post 乘子、RMSNorm eps |

**对齐策略**：Stage-1 阶段**只按 DeepSeek-V4 / TileKernels 公开语义**写 REF；
`mhc_tau` 和 `hres_vwnstyle` 留 hook（函数签名占位 + 文档 TODO），等 HF 端
建模源码发布再补分支。

## 4. 数据流图（FFN-side 一轮 mHC wrap）

```
              x_in: residual_streams [T, mhc=4, H] bf16
                              │
     ┌────────────────────────┴────────────────────────┐
     │                                                 │
     │                                                 │  residual (for post)
     │                                                 │
     ▼                                                 │
 ┌──────────────────────────┐                          │
 │ (1) mhc_pre_norm_fn      │                          │
 │  RMSNorm(flatten hc·H) + │                          │
 │  F.linear(fn fp32)       │                          │
 └──────────┬───────────────┘                          │
   mixes [T, mix_hc=24] fp32                           │
            │                                          │
            ▼                                          │
 ┌──────────────────────────┐                          │
 │ (2) mhc_pre_split_mixes  │                          │
 │  x*scale + base → sigmoid│                          │
 │  split: pre / post / comb│                          │
 └──────────┬───────────────┘                          │
   ┌────────┼─────────────────┐                        │
   ▼        ▼                 ▼                        │
 pre     post           comb_logits                    │
 [T,4,1] [T,4,1]        [T,4,4]                        │
   │        │                 │                        │
   │        │                 ▼                        │
   │        │       ┌──────────────────────────┐       │
   │        │       │ (3) sinkhorn_normalize   │       │
   │        │       │  softmax → iter(row,col) │       │
   │        │       │  repeat = 20             │       │
   │        │       └──────────┬───────────────┘       │
   │        │          comb [T,4,4] (doubly-stochastic)│
   │        │                  │                       │
   │        ▼                  │                       │
   │      (saved for post)     │                       │
   │                           │                       │
   ▼                           │                       │
 ┌──────────────────────────┐  │                       │
 │ (4) mhc_pre_apply_mix    │  │                       │
 │  (residual × pre).sum(-2)│◀─┼───────────────────────┤  residual
 │  → bf16                  │  │                       │
 └──────────┬───────────────┘  │                       │
   layer_input [T, H] bf16     │                       │
            │                  │                       │
            ▼                  │                       │
 ┌──────────────────────────┐  │                       │
 │    SUBLAYER (FFN)        │  │                       │
 │    = ffn_norm(x) → MoE   │  │                       │
 │    （调用已对齐的        │  │                       │
 │    moe_block_full 管线） │  │                       │
 └──────────┬───────────────┘  │                       │
    x_sub [T, H] bf16          │                       │
            │                  │                       │
            ▼                  ▼                       ▼
        ┌─────────────────────────────────────────────────┐
   (5)  │ mhc_post                                        │
        │   x_sub·post  +  einsum('abmn,abmc->abnc',      │
        │                          comb, residual.float())│
        │   → bf16                                        │
        └──────────────────┬──────────────────────────────┘
                           │
                  x_out [T, mhc=4, H] bf16
                           │
                           ▼
                    下一个 block 或 attn-side 再包一次
```

**关键数据流说明**：

- **(1) `pre_norm_fn` 的 fp32 约束**：residual（bf16）在 `.flatten(2,3).float()` 之后
  进入；`fn` 本身存 fp32；GEMM 在 fp32 域完成。这是为了 Sinkhorn 稳定性——
  24 维 logits 进 softmax + 20 次 Sinkhorn 迭代对精度非常敏感。
- **(2) `pre_split_mixes` 的 24 维布局**：`mix_hc = mhc·(2+mhc) = 24`。
  - `mixes[:, :, :4]`  → sigmoid → `+pre_eps` → **pre mix** `[T,4]`（后面 unsqueeze(-1)）。
  - `mixes[:, :, 4:8]` → sigmoid → `*post_mult_value` → **post mix** `[T,4]`。
  - `mixes[:, :, 8:24]` → 直接 view 成 `[T,4,4]` → **comb logits**（给 Sinkhorn）。
  - `scale[3]` 按 `[4, 4, 16]` 分组广播（即 `scale[0]` 复制给 pre, `scale[1]` 给 post,
    `scale[2]` 给 comb），`base[24]` 直接按位加。
- **(3) `sinkhorn_normalize` 的精确形式**（TileKernels `torch/mhc.py:8-14`）：
  ```python
  x = x.softmax(-1) + eps                  # 首轮 softmax + 正化（而非 exp+iter）
  x = x / (x.sum(-2, keepdim=True) + eps)  # 首次列归一
  for _ in range(repeat - 1):              # 之后交替 行 / 列 归一
      x = x / (x.sum(-1, keepdim=True) + eps)
      x = x / (x.sum(-2, keepdim=True) + eps)
  ```
  **不变量**：收敛后 `comb[t, :, :]` 是 doubly-stochastic —— 行和 ≈ 1、列和 ≈ 1
  （N×N Birkhoff polytope 的内点）。Stage-3 的 REF 自检会 assert 这一点。
- **(4) `pre_apply_mix` 的收尾精度**：`(x*mix).sum(-2).bfloat16()` —— sum 在
  x 自身 dtype（bf16）上做，最后只是一次 dtype 转换。和 CUDA 侧一致。
- **(5) `mhc_post` 的 fp32 累加 + 单次 RNE**（`torch/mhc.py:62-63`）：
  ```python
  term2 = einsum('abmn,abmc->abnc', comb, residual.float())   # fp32
  return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
  ```
  —— 完全对齐我们 `moe_sum_reduce` 的精度路径（fp32 acc → 单次 bf16 RNE）。

**和 `glm4_moe_ffn_dev.md` 的 payload 衔接点**：(4) 的输出 `layer_input[T,H]` 就是
`Glm4MoeSparseMoeBlock.forward_normal` 的入口（原先由 `post_attn_layernorm`
产生的那份），(5) 收到的 `x_sub[T,H]` 就是 `moe_block_full` 的输出（原先用于
"`*routed_scaling_factor` + shared"的那份，但这次 scaling 已在 MoE 管线里
fuse 完，shared 也在 `moe_sum_reduce` 里 fuse 完了，所以 `x_sub` 已经是干净的
`[T,H]`，可以直接喂给 `mhc_post`）。

## 5. 算子表（Zeus kernel 规划）

> 源：`ref_tile_kernels_torch/mhc.py`（REF 纯 torch）+
> `ref_tile_kernels_modeling_mhc/ops/*.py`（每算子的 autograd.Function 包装）+
> `ref_tile_kernels_mhc/*.py`（TileLang kernel）。

| # | 算子 | shape | TileKernels kernel file | sgl-kernel-zeus 现状 |
|---|---|---|---|---|
| 0 | `expand_to_mhc` | `[T,H] → [T, mhc, H]` | `expand_kernel.py` (60) | ⏳ TODO（仅用于 embedding 初始化，每个 sequence 一次，非热路径；可先 fallback `.unsqueeze(-2).expand(...).contiguous()`） |
| 1 | `mhc_pre_norm_fn` | `[T,mhc,H] bf16 → [T, mhc·(2+mhc)] fp32` | `norm_fn_kernel.py` (289) | ⏳ TODO。融合 RMSNorm + fp32 GEMM，split-K。**最复杂**的一环。 |
| 2 | `mhc_pre_split_mixes` | `[T, 24] fp32 → (pre[T,4,1], post[T,4,1], comb[T,4,4])` | `pre_split_mixes_kernel.py` (163) | ⏳ TODO。纯 elementwise + split，最简单。 |
| 3 | `sinkhorn_normalize` | `[T, 4, 4] → [T, 4, 4]` | `sinkhorn_kernel.py` (165) | ⏳ TODO。20 次迭代，每次 2 个 reduce。**必须 per-token 收敛**。 |
| 4 | `mhc_pre_apply_mix` | `[T, mhc, H] × [T, mhc, 1] → [T, H] bf16` | `pre_apply_mix_kernel.py` (110) | ⏳ TODO。按 mhc 轴 weighted reduce。 |
| 5 | `mhc_post` | `[T,H] + [T,mhc,H] + [T,mhc,1] + [T,mhc,mhc] → [T,mhc,H]` | `post_kernel.py` (221) | ⏳ TODO。einsum + scale + add，fp32 accum + 单次 RNE。 |
| — | `mhc_pre` (fused) | 上面 (1)(2)(3)(4) 的推理大融合核 | `pre_big_fuse_kernel.py` (131) | 📌 long-term：先分 4 个子 kernel 对齐，再看是否 fuse |

**v1 Scope 显式不做**：
- Training backward（所有 `autograd.Function.backward`）。
- LM head 侧的 `mhc_head_compute_mix` / `mhc_head`（不在 MoE-FFN 热路径）。
- `mhc_pre_big_fuse` 推理大融合（等 4 个子 kernel 对齐之后再考虑）。
- `multilayer_recompute`（训练内存优化）。
- N != 4（kernel 层硬约束）。
- fp8 / fp4 量化路径（DeepSeek-V4 prod 走 fp8，本 stage 先 bf16）。

## 6. Stage 顺序（dev 脚本 `dev_mhc_test.py`）

Stage-by-stage 独立对齐。REF 直接从 `ref_tile_kernels_torch/mhc.py` import；
Zeus 侧 kernel 未就绪前用 `NotImplementedError("kernel pending: <op>")` 占位，
但每个 stage 仍会**跑 REF + 校验不变量**（形状、doubly-stochastic、fp32 dtype
留存等），让脚本在 kernel 缺位的情况下仍有 diagnostic value。

1. `mhc_expand` —— REF sanity（`[T,H] → [T,mhc,H]`，内存 contiguous 检查）。
2. `mhc_pre_norm_fn` —— REF 输出 shape = `[T, 24]`，dtype = fp32；assert
   `fn` 权重 fp32、residual bf16 的约束。
3. `mhc_pre_split_mixes` —— REF 三路输出 shape 检查；`post` 乘子不变量
   （`post_mult_value=1.0` 时 `post = sigmoid(...).unsqueeze(-1)`，范围 (0,1)）。
4. `sinkhorn_normalize` —— REF 输出 **doubly-stochastic 自检**：
   `comb.sum(-1)` 每行 ≈ 1，`comb.sum(-2)` 每列 ≈ 1（容忍 `eps + 1e-5`）。
5. `mhc_pre_apply_mix` —— REF 输出 `[T, H] bf16`；sanity：`pre` 全为 1/mhc
   且 residual 各流相同时，输出 ≈ residual。
6. `mhc_post` —— REF 输出 `[T, mhc, H] bf16`；sanity：`comb=I/mhc` 时
   residual 应被均匀 propagate（此项作为数值自检）。
7. `mhc_block_ffn_full` —— 端到端组装：(1)→(2)→(3)→(4) → **trivial sublayer**
   （初版用 identity，后续替换为 `moe_block_full` 的 Zeus 结果）→ (5)。
   当前 stage 只保证 wrap 逻辑通，MoE 融合留给后续 follow-up。

## 7. 注意事项

- **mhc_mult 硬锁 4**：当前 kernel（以及 config）只支持 N=4，所有 shape 常量
  都围绕这个展开。v2 再扩展到任意 N 时需要同步改 sim.c 的 `mix_hc`
  常量以及 Sinkhorn 的 tile shape。
- **fp32 参数全程 fp32**：`fn / scale / base / norm_weight` 永远 fp32，
  哪怕 residual/x 是 bf16。`with set_dtype(torch.float32)` 是 DeepSeek-V4
  `Block.__init__` 的硬约束。
- **dynamic per-token**：`pre / post / comb` 是 token-dependent（由 residual 的
  RMSNorm+Linear 投影产生），**不是** static per-layer。这决定了 Sinkhorn
  必须 per-token 跑 20 iter，不能复用跨 token 的结果。
- **Sinkhorn 首轮不同**：第一次是 `softmax(-1) + eps` 然后 col-normalize，
  后续才是 row↔col 交替；共计 `2*repeat - 1` 次 reduce（不是 `2*repeat`）。
  REF 照抄 `torch/mhc.py:8-14`，Zeus 侧 port 时必须对齐这个 off-by-one。
- **`sinkhorn_repeat` 以 config 为准**：DeepSeek-V4 prod = 20，TileKernels
  API 默认 10，**实测 20 和 10 的 doubly-stochastic 偏差差一个量级**
  （20 iter 下 `|row_sum - 1| ≈ 1e-6`，10 iter 下 ≈ 1e-3），config 要走谁
  就用谁。
- **`hres_vwnstyle` 与 `mhc_tau`**：公开 TileKernels 与 DeepSeek-V4 inference
  代码均**无对应分支**，GLM-5 Next 上游 HF 建模代码发布后再补实现。
  当前 REF 与 Zeus 侧的函数签名里**留 `tau` 与 `vwnstyle` 参数槽**，
  默认值 `None / False` 对齐 TileKernels 公开行为。

## 8. 开发日志

### 2026-04-24 · 起点
- 创建本文档 + `dev_mhc_test.py` 骨架。
- 下载到 `zeus_dev/ref_deepseek_v4_model.py` / `ref_deepseek_v4_kernel.py` /
  `ref_tile_kernels_mhc/` / `ref_tile_kernels_modeling_mhc/` /
  `ref_tile_kernels_torch/` 作参考源码。
- REF 侧定下：直接引用 `ref_tile_kernels_torch/mhc.py` 的 7 个纯 torch
  函数作 golden；Zeus 侧 kernel 全部 TODO。
- Config 字段对照表、数据流图、6-stage 算子规划表就位。
- 今日目标：确认 REF 跑通，7 个函数 import 成功，每 stage 打印自洽 invariants。
- 下一步焦点：选一个最简单的 kernel（大概率是 `pre_split_mixes` —— 纯
  elementwise）作为 Zeus 侧落地的第一颗，把注册链 / sim.c / host wrapper /
  Python API 跑通，再依次攻 `mhc_post` / `sinkhorn_normalize` /
  `mhc_pre_apply_mix` / `mhc_pre_norm_fn`。
