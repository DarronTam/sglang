# GLM5-Next mHC (Manifold-Constrained Hyper-Connections) Zeus 适配开发追踪

> 对齐目标：`/root/project/sglang-feat-v0.5.10-prerelease-glm` 中
> `Glm5NextDecoderLayer` 用到的 **mHC**（多残差流 + Birkhoff-polytope 约束的
> hyper-connection）。每个 decoder layer 包两个 `HyperConnection` 实例
> （attn-side + mlp-side），把单流 sublayer 包成多流残差。
>
> 本文同时覆盖两套配置：
> - **GLM5-Next-16B**：`zeus_dev/model_state_dev/config_16b_v2.json`
> - **GLM5-Next**：`zeus_dev/model_state_dev/config.json`
>
> 文档/脚本格式与 `glm5next_dsa_decode_dev.md` / `kimi_linear_attn_dev.md` /
> `glm4_moe_ffn_dev.md` 对齐：先范围与方法，再配置表、数据流图、算子表、
> dev stage 顺序，最后注意事项与开发日志。
>
> 与既有 `mhc_dev.md` 的关系：那份用 DeepSeek-V4 `hc_*` / TileKernels 公开语义、
> REF 来自 `ref_tile_kernels_torch/mhc.py`；**本份专注 GLM5-Next 部署路径**，
> REF 内联复刻 prerelease repo 的 `functional.py::_mhc_pre_torch / _mhc_post_torch`
> （即 `HyperConnection` 在 `SGLANG_OPT_USE_TORCH_MHC=True` 下的真实 fallback），
> 不依赖外部 ref 包，让生产路径成为唯一 golden。

## 范围与方法

- **起点**：transformer block 入口的 **多流 residual** `[T, N, H]`（N=4），
  在 block 间以 flatten 形态 `[T, N*H]` 流动。序列入口由 `hc_expand`（broadcast）
  从 embedding `[T, H]` 生成。
- **终点**：一个 sublayer（attn 或 mlp）处理完之后，`hc_post` 回写的新多流 residual
  `[T, N*H]`。
- **切片**：单 device、单 layer、decode 视角（mHC 本身与 decode/extend 无关，
  逐 token 独立）。attn-side 与 mlp-side 各包一次，结构完全同构。
- **对齐方式**：沿用 stage 化 "REF vs Zeus"。REF 内联 pure-torch（mirror
  `functional.py`）；Zeus 侧逐算子落地 `sgl_kernel_zeus.mhc_*`，**禁止 silent
  fallback** 到 torchnative 伪装 Zeus kernel；未落地 kernel 显式 SKIP/TODO。
- **dev 脚本**：`zeus_dev/model_state_dev/dev_glm5next_mhc_test.py`，按 `--stage`
  / `--config {16b,next}` / `--mode {both,ref,zeus}` 单独跑。

不在本文 scope：
- 训练反向（所有 `autograd.Function.backward`）。
- LM head 侧的 `mhc_head_compute_mix` / `mhc_head`（不在 layer 热路径）。
- `mhc_pre_big_fuse` 推理大融合（先分子 kernel 对齐，再考虑 fuse）。
- TP/EP/DP-attention 跨进程 collective（`MHCLayerCommunicator` 的通信编排）。
- `N != 4`（kernel 层硬约束）。
- fp8 / fp4 量化路径（首版 bf16 residual + fp32 参数）。
- `mhc_tau` 与 `hres_vwnstyle`（GLM 扩展，语义悬空，见注意事项）。

## GLM5-Next mHC 相关配置

| 字段 | **GLM5-Next-16B**<br>`config_16b_v2.json` | **GLM5-Next**<br>`config.json` | 含义 |
|---|---:|---:|---|
| `mhc` | true | true | 启用 mHC |
| `hidden_size` (H) | 2048 | 4096 | 单流宽度 |
| `mhc_num_residual_streams` (N) | 4 | 4 | 残差流条数（kernel 硬锁 4） |
| `mix_hc` = N·(2+N) | 24 | 24 | mapping 输出维度 = pre(4)+post(4)+comb(16) |
| `d_model` = N·H | 8192 | 16384 | mapping 输入维度（flatten 多流） |
| `mhc_sinkhorn_iterations` | 20 | *(缺省→20)* | Sinkhorn-Knopp 迭代数 |
| `hc_eps` | 1e-6 | *(缺省→1e-6)* | Sinkhorn / pre sigmoid eps |
| `mhc_post_mult_value` | 2 | *(缺省→2.0)* | post mix 的乘子 |
| `mhc_no_norm_weight` | true | *(缺省→False)* | True 则 RMSNorm weight 旁路 |
| `rms_norm_eps` | 1e-5 | 1e-5 | pre RMSNorm eps |
| `mhc_tau` | 1.0 | 0.05 | **GLM 扩展，悬空**（当前 REF 不使用） |
| `hres_vwnstyle` | false | true | **GLM 扩展，悬空** |

> **缺省值来源**：`config.json` 未显式写 `mhc_sinkhorn_iterations` /
> `mhc_post_mult_value` / `mhc_no_norm_weight`，dev 脚本按 `HyperConnection.__init__`
> 的默认值（20 / 2.0 / False）回填。注意 **16B 旁路 RMSNorm weight（no_norm_weight=true）、
> Next 使用 weight**，这是两套配置在 mHC 上唯一的结构性差异（其余只是 H 不同）。

## HyperConnection 在 layer 中的位置

`Glm5NextDecoderLayer.__init__`（prerelease repo glm5_next.py:480-503）为每层建两个
`HyperConnection`：

```python
if self.config.mhc:
    self.self_attention_hyper_connection = HyperConnection(H, N, ...)
    self.mlp_hyper_connection           = HyperConnection(H, N, ...)
```

`MHCLayerCommunicator`（communicator_mhc.py::MHCState）按下列顺序调度一层：

```
residual_in [T,N*H]
   │  attn_hc.pre_forward  ── layer_input[T,H], (h_res,h_post 暂存)
   ▼
 self_attn(layer_input)      ── DSA decode 或 Linear-attn decode
   │  attn_hc.post_forward ── residual_mid [T,N*H]
   ▼
   │  mlp_hc.pre_forward   ── layer_input[T,H], (h_res,h_post 暂存)
   ▼
 mlp(layer_input)            ── MoE 或 dense MLP
   │  mlp_hc.post_forward  ── residual_out [T,N*H]
   ▼
residual_out -> 下一层
```

即 mHC 是 **包在 sublayer 外面的多流残差容器**：sublayer 永远只看到单流
`[T,H]`（`layer_input`），多流 mixing 全在 `pre/post` 里完成。

## 数据流图（一个 HyperConnection 的 pre/post 往返）

```
              residual: [T, N=4, H] bf16  (flatten [T, N*H])
                              │
     ┌────────────────────────┴───────────────────────────┐
     │                                                     │ residual (留给 post)
     ▼                                                     │
 ┌─────────────────────────────────────┐                   │
 │ (K1) mhc_pre_norm_split  ★融合 #1+#2 │  fn:[24,N*H] fp32 │
 │  BLOCK_T 1-pass，逐 T-tile：          │  scale[3],base[24]│
 │   RMSNorm(flatten N·H, fp32)          │ (*norm_weight 旁路)│
 │   + F.linear(fn) → mixes[T,24]        │                   │
 │   + x*scale+base → sigmoid → split    │  ← mixes 不落 DRAM│
 └──────────┬────────────────────────────┘                   │
   ┌────────┼──────────────────────┐                       │
   ▼        ▼                       ▼                       │
 pre[T,4,1] post[T,4,1]      comb_logits[T,4,4]            │
   │        │(留给 post)            │                       │
   │        │                       ▼                       │
   │        │           ┌──────────────────────────┐       │
   │        │           │ (K2) sinkhorn_normalize  │       │
   │        │           │  softmax(-1)+eps → 列归一 │       │
   │        │           │  (repeat-1)×(行→列)       │       │
   │        │           └──────────┬───────────────┘       │
   │        │          comb[T,4,4] doubly-stochastic       │
   │        │(留给 post)           │(留给 post)            │
   ▼        │                      │                       │
 ┌──────────────────────────────┐  │                       │
 │ (K3) mhc_pre_apply_mix       │  │                       │
 │  (residual × pre).sum(stream)│◀─┼───────────────────────┤ residual
 │  → bf16                      │  │                       │
 └──────────┬───────────────────┘  │                       │
   layer_input [T, H] bf16         │                       │
            │                      │                       │
            ▼                      │                       │
 ┌──────────────────────────────┐  │                       │
 │   SUBLAYER (单流)            │  │                       │
 │   attn: DSA / Linear-attn    │  │                       │
 │   mlp : MoE / dense MLP      │  │                       │
 └──────────┬───────────────────┘  │                       │
    x_sub [T, H] bf16              │ comb (h_res)          │ residual
            │              post (h_post)                   │
            ▼                      ▼                       ▼
        ┌─────────────────────────────────────────────────┐
  (K4)  │ mhc_post                                        │
        │   x_sub·post + einsum('tmn,tmh->tnh',          │
        │                        comb, residual.float())  │
        │   fp32 accum → 单次 bf16 RNE                    │
        └──────────────────┬──────────────────────────────┘
                           │
                  residual_out [T, N=4, H] bf16
```

## 算子表（Zeus kernel 规划）

> 源：prerelease repo `functional.py`（部署 REF）+ `dsv4_mhc.py`（TileLang fused）+
> TileKernels `*_kernel.py`（tile shape 参考）。
>
> **2026-05-23 算子拆分调整**：原 #1 `mhc_pre_norm_fn` 与 #2 `mhc_pre_split_mixes`
> 合并为一颗 **K1 `mhc_pre_norm_split`**（BLOCK_T 1-pass）。两者都是 per-token 运算，
> #1 算出 24 维 `mixes`、#2 是对这 24 维的纯 elementwise（scale/base/sigmoid/split），
> 融合后 `mixes [T,24]` 留在寄存器/shared、**不落 DRAM**，省一次 [T,24] 读写。
> 5 颗 kernel（K0..K4），比原来少一颗。

| K | 算子 | shape | 现状 | 备注 |
|---|---|---|---|---|
| K0 | `mhc_expand` | `[T,H] → [T,N*H]` | ✅ LANDED (2026-05-23) | sequence 入口一次，非热路径；sgl-kernel-zeus 5 件套 + docs/slides 已就位 |
| K1 | `mhc_pre_norm_split`<br>★融合 #1+#2 | `[T,N,H] bf16 → pre[T,N,1]/post[T,N,1]/comb_logits[T,N,N] fp32` | ✅ LANDED (2026-05-23) | **BLOCK_T 1-pass**：逐 T-tile 做 RMSNorm(flatten N·H, fp32) + **bf16×bf16→fp32 GEMM(fn LocalMem)** → 24 维 mixes → scale/base/sigmoid/split；mixes 不落 DRAM。**最复杂**。fn 量化到 bf16 + `from_tensor(kind="weight", Tr=1, Tc=1)` 打包到 LocalMem（Lmem 不支持 fp32 weight）。当前 sim 单核，porting 切 12/12（dev doc §"K1 双核并行布局"）。 |
| K2 | `mhc_sinkhorn` | `[T,4,4] → [T,4,4]` | ✅ LANDED (2026-05-23) | 沿 T 轴切核（CORE_NUM=2；N 是 reduce 轴不能切；T 不整除 CORE_NUM 时 cdiv + boundary_check 兜底，无 host 硬约束）；2D block_ptr `[BT, NN]` + reshape 到 3D `[BT, N, N]` 做 axis-1 col norm + axis-2 row norm；fp32 全程，stable softmax；sgl-kernel-zeus 5 件套 + docs/slides 已就位。 |
| K3 | `mhc_pre_apply_mix` | `[T,N,H]×[T,N,1] → [T,H] bf16` | ✅ LANDED (2026-05-23) | 沿 H 轴切核（CORE_NUM=2，disjoint 输出列，无 cross-core 通信，是 reduce 轴之外的维度）；N=4 静态展开 4 路 multiply-add；fp32 acc + 单 bf16 RNE round。sgl-kernel-zeus 5 件套 + docs/slides 已就位。 |
| K4 | `mhc_post` | `[T,H]+[T,N*H]+[T,N]+[T,N*N] → [T,N*H] bf16` | ✅ LANDED (2026-05-24) | 沿 H 轴切核（CORE_NUM=2，H 是 per-output 维度，与 K3 同思路）；T 循环外 20-tile preload (4 post + 16 comb)，跨 H tile 复用；4 输出 stream × 5-way (1 + N=4) multiply-add 双重静态展开；fp32 acc + 单 bf16 RNE round。sgl-kernel-zeus 5 件套 + docs/slides 已就位。**mHC 5 颗 kernel 全部 LANDED**。 |
| — | `mhc_pre` (fused) | K1·K2·K3 推理大融合 | 📌 long-term | 先分子 kernel 对齐，再 fuse |

> **dev 子步保留**：dev 脚本仍单列 `mhc_pre_norm_fn` / `mhc_pre_split_mixes` 两个
> stage 作为 K1 的 debug 子步（定位数值失配用），它们不再各自映射独立 Zeus kernel。
> K1 的 stage `mhc_pre_norm_split` 额外做两项自检：① 融合输出 == 顺序两步（bit-exact）；
> ② **tile-invariance**——按 `--block-t` 分块逐 tile 跑 ≈ 整批跑（per-token RMSNorm 令 T 轴
> 各 token 独立，故 Zeus kernel 定序 dot 会逐 bit 一致；REF 因 BLAS 跨批 fp32 重排带
> ~1e-6 噪声，用紧 tol 校验）。

**v1 Scope 显式不做**：training backward / LM-head mHC / `mhc_pre_big_fuse` /
`multilayer_recompute` / `N≠4` / fp8 量化。

## K1 双核并行：device 内 2-core 推荐 split-K 12/12

> 单 device 两个 core 怎么并行 K1？结论：**沿 GEMM 输出维（24 行）切 12/12**，
> 不切 T。下表是带宽与均衡度的逐项推导。

**为什么 decode 路径切 K 不切 T**（以 16B 为例，fn 是 fp32）：

| 数据 | 大小 | 是否随 T 增长 | 说明 |
|---|---|---|---|
| `fn[24, N·H]` | 24 × 8192 × 4 ≈ **0.75 MB** | 否（每层静态） | 每次调用 DRAM 读一次 |
| `residual[T, N·H]` | T × 8192 × 2 ≈ **16 KB · T** | 是 | per-token 读一次 |
| 输出 `pre / post / comb` | T × 24 × 4 ≈ 96 B · T | 是 | 可忽略 |

decode 单步 T 通常 ≤ 16，`fn` 流量 (~0.75–1.5 MB) **远大于** `residual`。
切 K 把每核 fn 流量减半（核 0/核 1 各只读 12 行 = 384 KB / 768 KB）；切 T 反而让两核
**都拉全 fn**，浪费一半带宽。临界 T ≈ 48（residual 流量追上 fn）—— 这在 decode 不会发生，
留给 prefill / 大 batch 路径再单独议。

**12 / 12 是唯一均衡分法**：

| 切法 | core0 / core1 GEMM 行 | 平衡 | 备注 |
|---|---|---|---|
| **12 / 12** ★ | 12 / 12 | **1 : 1** | 推荐；post-process 异构但 GEMM 主导 → < 5% 不平衡 |
| 8 / 16（按 pre+post / comb） | 8 / 16 | 1 : 2 | GEMM 严重失衡 |
| 4 / 20（按 pre / 其它） | 4 / 20 | 1 : 5 | 更差 |

**布局**（每核跑同一组 BLOCK_T tile，K 切片不同）：

```
core 0  ── 12 行 GEMM (dims 0..11) ──── 异构 post-process ──── 写 disjoint ──
  load  residual[t, :]
  rsqrt = rsqrt(mean(x²)+eps)              # 与 core1 冗余，无 barrier
  mixes_local[0..11] = fn[0..11, :] · x · rsqrt
    dims 0..3  → pre  (σ + eps)                    → write pre[t, :]
    dims 4..7  → post (σ · post_mult)              → write post[t, :]
    dims 8..11 → comb_logits row 0 (scale+base)    → write comb_logits[t, 0, :]

core 1  ── 12 行 GEMM (dims 12..23) ─── 同构 post-process ──── 写 disjoint ──
  load  residual[t, :]
  rsqrt = rsqrt(mean(x²)+eps)              # 与 core0 冗余，无 barrier
  mixes_local[12..23] = fn[12..23, :] · x · rsqrt
    dims 12..23 → comb_logits rows 1..3 (scale+base 全 16 维同处理)
                                                   → write comb_logits[t, 1:4, :]
```

**核间同步只剩一个共享标量 `rsqrt`** —— 两种处理：

| 方案 | 同步代价 | 冗余代价 | 推荐 |
|---|---|---|---|
| **两核各自算一遍 rsqrt** | 0（无 barrier） | +1 次 N·H reduce / 核 ≈ 总 reduce 的 4% | ★ 首选，wall-clock 不增 |
| 一核算 + barrier 广播 | 1 次 intra-device barrier | 0 | 仅当 Zeus barrier 接近免费时 |

**正交性**：
- BLOCK_T 内循环不变 —— 两核各跑同一 T-tile 序列，互不阻塞；
- tile-invariance 保留 —— per-token 独立 + 每核内 dot-product 定序 → Zeus 输出对 BLOCK_T 取值依然 bit-exact；
- 输出无写冲突 —— core0 写 pre/post/comb_logits[t,0,:]，core1 写 comb_logits[t,1:4,:]，DRAM 区段 disjoint；
- 后续 K2 sinkhorn 读 comb_logits[t,:,:] 是完整 4×4，无须特殊改造（K1 结束后两核输出已拼齐）。

**一句话**：切 K=12/12，每核读全 residual 但只读一半 fn、只写 disjoint 输出、共享标量 rsqrt 冗余算两遍——零 barrier、GEMM 完全均衡、fn 带宽减半。

## Dev 脚本 stage 顺序（`dev_glm5next_mhc_test.py`）

| stage | kernel | 内容 | REF 不变量自检 |
|---|---|---|---|
| `mhc_expand` | K0 ✅ | embedding → 多流（broadcast） | N 条流初始恒等 + Zeus 逐 bit 一致 |
| `mhc_pre_norm_fn` | *(K1 子步 a)* | RMSNorm + fp32 Linear | shape `[T,24]` / dtype fp32 |
| `mhc_pre_split_mixes` | *(K1 子步 b)* | scale+base+sigmoid 三路拆分 | pre∈(eps,1+eps), post∈(0,post_mult) |
| `mhc_pre_norm_split` | **K1** | ★融合 #1+#2，BLOCK_T 1-pass | 融合==顺序两步(bit) + tile-invariance |
| `mhc_sinkhorn` | K2 | Birkhoff projection | doubly-stochastic + 非负 |
| `mhc_pre_apply_mix` | K3 | stream 轴 weighted sum | uniform pre 时 ≈ 单流 |
| `mhc_pre` | K1·K2·K3 | 组合 = `pre_forward` | comb doubly-stochastic |
| `mhc_post` | K4 | scatter 回 streams | post=0,comb=I 时 out≈residual |
| `mhc_sublayer_wrap` | — | pre → identity → post 端到端 | shape/dtype 闭合 |
| `mhc_layer_full` | — | attn_hc + mlp_hc 双 wrap | 输出后 N 条流不再恒等 |

`mhc_sublayer_wrap` / `mhc_layer_full` 当前用 **identity sublayer** 把 wrap
逻辑跑通；真实 sublayer（DSA decode / Linear-attn decode / MoE）的接入见
`glm5next_block_decode_dev.md` + `dev_glm5next_block_decode_test.py`。

## 注意事项

- **N 硬锁 4**：`mix_hc=24`、`d_model=N·H`、Sinkhorn 的 `4×4` 都围绕 N=4 展开。
  扩展任意 N 需同步改 sim.c 常量与 Sinkhorn tile shape。
- **fp32 参数全程 fp32**：`fn / scale / base / norm_weight` 永远 fp32，哪怕
  residual/x 是 bf16。`mapping_proj = nn.Linear(..., dtype=torch.float32)`
  是 `HyperConnection.__init__` 的硬约束。这是为 Sinkhorn 的 24 维 logits +
  20 迭代提供数值稳定性。
- **dynamic per-token**：`pre / post / comb` 由 residual 的 RMSNorm+Linear 投影
  产生，**token-dependent**，不是 static per-layer。Sinkhorn 必须 per-token
  跑满迭代，不能跨 token 复用。
- **Sinkhorn 首轮不同**：第一步是 `softmax(-1)+eps` 然后列归一，之后才是
  行↔列交替；共 `2·repeat−1` 次 reduce（不是 `2·repeat`）。Zeus port 必须
  对齐这个 off-by-one。
- **`mhc_no_norm_weight` 两套配置不同**：16B 旁路 RMSNorm weight（`fn` 直接用），
  Next 使用 weight（`fn = fn * norm_weight`）。dev 脚本据 config 自动切换。
- **`post_mult_value` 默认 2**：16B 显式写 2；Next 缺省，按 HyperConnection
  默认 2.0。注意 `mhc_dev.md`（DeepSeek-V4 那份）里曾用 1.0，本份按 GLM
  生产默认 2.0，两份不要混。
- **`mhc_tau` / `hres_vwnstyle` 悬空**：prerelease repo `functional.py` /
  `dsv4_mhc.py` 均无对应分支，GLM5-Next 上游 HF 建模代码发布后再补。当前
  REF 与 Zeus 签名留参数槽，默认 `tau=1.0 / vwnstyle=False`，行为对齐
  `_mhc_pre_torch`。
- **mhc_post 的 fp32 accum + 单 RNE**：`(x.float()*post + einsum(comb, residual.float())).bfloat16()`
  —— 与 `moe_sum_reduce` 的精度路径一致（fp32 累加 → 一次 bf16 RNE），
  比独立 bf16 加法少一次舍入。

## 开发日志

### 2026-05-22 · 起点
- 创建 `glm5next_mhc_dev.md` + `glm5next_mhc_slides.html` + `dev_glm5next_mhc_test.py`。
- REF 侧定下：**内联** pure-torch 复刻 prerelease `functional.py::_mhc_pre_torch /
  _mhc_post_torch`（即 HyperConnection 的 torch fallback），不依赖外部 ref 包，
  让 GLM5-Next 部署路径成为唯一 golden。
- 配置驱动：`config_16b_v2.json` / `config.json` 直接读 `mhc_*` / `hc_eps` /
  `hidden_size`；缺省字段按 `HyperConnection.__init__` 默认回填。
- 9 stage（expand / pre_norm_fn / pre_split_mixes / sinkhorn / pre_apply_mix /
  pre / post / sublayer_wrap / layer_full）REF 全 PASS（16B + Next 两套配置）。
- Zeus 侧 6 个 `mhc_*` kernel 全部 TODO；`--mode both` 打印 SKIP/TODO 锚点。
- 下一步焦点：从 `mhc_expand` 起手把 5 件套接通走通流程（**已于同日落地，见下条**），
  随后依次攻 `mhc_post` / `mhc_sinkhorn` / `mhc_pre_apply_mix` / `mhc_pre_norm_split`。

### 2026-05-23 · 算子拆分调整：#1 + #2 → 融合 K1 `mhc_pre_norm_split`
- 把原 #1 `mhc_pre_norm_fn`（RMSNorm + fp32 GEMM → 24 维 mixes）与 #2
  `mhc_pre_split_mixes`（对 24 维做 scale/base/sigmoid/split）合并为一颗
  **K1 `mhc_pre_norm_split`**，用 **BLOCK_T 的 T 循环作为 1-pass**：每个 T-tile 把
  residual 读入后在寄存器/shared 内一气呵成 norm→GEMM→split，中间 `mixes [T,24]`
  **不落 DRAM**。Zeus kernel 数：6 → **5**（K0..K4）。
- dev 脚本：新增 stage `mhc_pre_norm_split`；保留 `mhc_pre_norm_fn` /
  `mhc_pre_split_mixes` 作为 K1 的 debug 子步（不再各自映射独立 kernel）。
  `ref_mhc_pre` 改走 `ref_mhc_pre_norm_split`，单一 golden 源。
- K1 stage 两项自检全 PASS（16B + Next）：① 融合输出 == 顺序两步逐 bit 一致；
  ② tile-invariance：`--block-t 4` 分块逐 tile ≈ 整批，max_diff≈9.5e-7（per-token
  RMSNorm 令 T 轴独立，残差仅 BLAS 跨批 fp32 重排噪声；Zeus kernel 每 token 定序 dot
  会逐 bit 一致，故首版即可任意取 BLOCK_T）。
- 全 10 stage REF PASS（16B + Next）；Zeus 侧 K0..K4 全 SKIP/TODO。新落地顺序：
  `mhc_post` / `mhc_sinkhorn` / `mhc_pre_apply_mix` → `mhc_pre_norm_split`（含 GEMM，最重）。

### 2026-05-23 · K0 `mhc_expand` 落地（sgl-kernel-zeus 首个 mHC 算子）
- **5 件套 + 文档全套交付**（路径 `sgl-kernel-zeus/csrc/glm5next_mhc/` & `sgl-kernel-zeus/docs/`）：
  - Triton blueprint：`mhc_expand_kernel.py`（`grid=(CORE_NUM=2,)`，T 轴跨核切分 + 余数派发；
    核内 T×H 双向 tile `[BLOCK_T=16, BLOCK_H=128]`；**单 load + `tl.static_range(N=4)` 扇出 4
    次 store**，输出 N 路共享一个 block_ptr 模板，仅 advance 列偏移 `k*H + h_start` 不同）。
  - CPU sim：`sgl_mhc_expand_sim.c`（scalar `for t / for k` + `memcpy(row_out + k*H, row_in, H*2)`；
    bf16 → bf16 纯字节拷贝，无需 fp32 中转）。
  - Host wrapper：`mhc_expand_zeus.cpp`（dtype/contiguous/`n==4` 硬锁/shape 强校验，
    `zertLaunchKernel(grid=1)`；porting 真 triton zbin 时切 `grid = CORE_NUM`）。
  - Python API：`glm5next_mhc.py::mhc_expand(hidden, *, n=4, out=None)`，可选预分配 out。
  - Tests：`tests/test_mhc_expand.py`（多 shape：含 GLM5-Next-16B `H=2048` / GLM5-Next `H=4096`
    实战 shape + 4 个拒绝用例）。
  - Docs：`docs/mhc_expand.md`（7 节，对照 `silu_and_mul.md` 骨架）+ `docs/mhc_expand_slides.html`
    （8 张暗色幻灯片，含 mini 示意 `T=4,H=8` 的 N=4 段拼接布局图）。
- **dev 脚本接通**：`stage_mhc_expand` 不再走 `zeus_skip` SKIP/TODO 分支，直接调用
  `sgl_kernel_zeus.mhc_expand(hidden.to("zeus"), n=N)`，与 REF (`x.repeat(1, N)`) **逐 bit 比较**
  （`exact=True`，max_diff = 0.000e+00）。同时校验 Zeus 输出的 N 条流恒等不变量。
  新增 `finish_stage_with_zeus` 三态返回（True=REF+Zeus PASS / False=失败 / None=Zeus 不可用→SKIP）。
- **验收**：16B (T=16, H=2048 → out=[16, 8192]) + Next (T=16, H=4096 → out=[16, 16384])
  两套配置均逐 bit PASS；Summary 现在 K0 显示 "PASS"，K1..K4 仍 "SKIP (REF PASS; Zeus TODO)"。
- **下一步**：按 2026-05-22 计划继续 K4 `mhc_post` / K2 `mhc_sinkhorn` / K3 `mhc_pre_apply_mix`
  → K1 `mhc_pre_norm_split`（最重，含 fp32 GEMM）。dev 脚本的 `zeus_compare` / `finish_stage_with_zeus`
  helper 可直接给后续 stage 复用，K4 落地时只需在 stage 里加一段 `if hasattr(sgl_kernel_zeus, ...)`
  分支即可。

### 2026-05-23 · K1 双核并行布局：split-K 12/12
- 单 device 两个 core 怎么并行 K1，定下来 **沿 GEMM 输出维（24 行）切 12/12，不切 T**。
  推导：decode 单步 T 通常 ≤ 16，`fn[24,N·H] fp32` (~0.75–1.5 MB) 流量远大于
  `residual[T,N·H] bf16` (~16–32 KB · T)，**fn 是首要 DRAM 带宽瓶颈**，切 K 直接把每核
  fn 流量减半（每核读 12 行）；切 T 则两核都拉全 fn，浪费一半带宽。临界 T ≈ 48（residual
  追上 fn）—— 只在 prefill / 大 batch 才反转为切 T 更优。
- 24 行天然 12/12 均分；穷举 8/16 / 4/20 均严重失衡（1:2 / 1:5）。post-process 异构
  （core0 三分支 = pre+post+comb_row[0]，core1 同构 = comb rows[1..3]），但 GEMM 主导，
  不平衡 < 5%，可忽略。
- 核间同步只剩 per-token 标量 `rsqrt`。决策：**两核冗余各算一遍 rsqrt**（+4% reduce / 核，
  无 barrier，wall-clock 不增），优于"一核算+barrier 广播"——除非 Zeus intra-device sync
  近乎免费。
- 输出写盘 disjoint：core0 → pre[t,:] + post[t,:] + comb_logits[t,0,:]；core1 →
  comb_logits[t,1:4,:]。K2 sinkhorn 读完整 4×4 comb_logits 无须改造。
- BLOCK_T 内循环与 tile-invariance 完全正交：两核各跑相同 T-tile 序列、各自独立，
  Zeus 输出对 BLOCK_T 取值依然 bit-exact。
- 文档：md 新增 "## K1 双核并行" 一节；slides 插入第 7 张 "K1 2-core"，原 7..10 顺延为 8..11。

### 2026-05-23 · K1 `mhc_pre_norm_split` 落地（5 件套 + docs/slides + dev 脚本接通）
- **5 件套 + 文档全套交付**（路径 `sgl-kernel-zeus/csrc/glm5next_mhc/` & `sgl-kernel-zeus/docs/`）：
  - Triton blueprint：`mhc_pre_norm_split_kernel.py`。**Single-pass BLOCK_T 循环 + 3 颗子-GEMM**：
    一次读 residual → bf16→fp32 upcast → 算 sum_sq + inv_rms (per-row) → fn 三段切片（`(0,0)` /
    `(N,0)` / `(2N,0)` block_ptr，全部 `memory_type='weight'`）→ pre/post/comb 三颗独立 `tl.dot
    (fp32×fp32→fp32)` → scale/base/sigmoid/+pre_eps/×post_mult → 直接 store。**`mixes [T,24]`
    寄存器 tile 从不构造**——三颗子-GEMM 各自直接吐 [BT, 4]/[BT, 4]/[BT, 16]，比 dev doc 描述的
    "mixes 寄存器内 split" 还更激进。CORE_NUM=1 单核，注释保留 2-core split-K 12/12 作 porting 目标。
  - CPU sim：`sgl_mhc_pre_norm_split_sim.c`（scalar `for t / for i / for j`：bf16→float upcast 整行 →
    `double` 累加 `sum_sq`（避免 NH=16384 项 fp32 累加漂移）→ `inv_rms` → 24 次 fp32×fp32 dot 算
    mixes → 三段 post-process。**stable sigmoid**（正负号分支避免 expf 溢出）。
  - Host wrapper：`mhc_pre_norm_split_zeus.cpp`。Residual 支持 `[T, N, H]` 或 `[T, N*H]` 两种 view，
    host 统一 flat 处理；outputs 允许 `[T, N]` / `[T, N, 1]` 等 view（按 numel 校验）。
    `scale[3]` 由 Python API 抽 3 个 fp32 scalar 传入（layer-static，sync 仅 12 B，不参与 graph capture）。
    全套 TORCH_CHECK：dtype / contiguous / `n==4` / shape / numel。
  - Python API：`glm5next_mhc.py::mhc_pre_norm_split(residual, fn, scale, base, *, n=4, rms_norm_eps,
    pre_eps, post_mult_value, pre_out, post_out, comb_logits_out)`，三件套输出可预分配，未给则自动按
    `[T, N, 1]` / `[T, N, 1]` / `[T, N, N]` 分配。**norm_weight 由调用方在 host 侧 pre-merge 进 fn**
    （与 REF 同源），kernel 不感知差异。
  - Tests：`tests/test_mhc_pre_norm_split.py`（**14 用例全 PASS**）：6 个 shape (含 16B `H=2048` / Next
    `H=4096` 实战 shape) 数值对照 REF（`atol=rtol=5e-3`）；预分配 outputs；range 不变量；2D vs 3D
    residual 一致；4 个拒绝用例（非 bf16 / 非 fp32 fn / `n!=4` / fn shape / 非 contiguous）。
  - Docs：`docs/mhc_pre_norm_split.md`（7 节，对照 `silu_and_mul.md` / `mhc_expand.md` 骨架，含 fp32
    weight GEMM porting 注意事项）+ `docs/mhc_pre_norm_split_slides.html`（9 张暗色幻灯片，
    fn 三段切片布局图 + single-pass 数据流 + GLM5-Next-16B 带宽预算 + 2-core split-K 12/12 porting 示意）。
- **dev 脚本接通**：`stage_mhc_pre_norm_split` 不再走 SKIP/TODO，调用
  `sgl_kernel_zeus.mhc_pre_norm_split(residual.to("zeus"), fn_eff.to("zeus"), p.scale, p.base.to("zeus"),
   ...)`，其中 `fn_eff = p.fn if p.norm_weight is None else p.fn * p.norm_weight` 在 host 侧 pre-merge。
  三件套 (pre/post/comb_logits) 各自与 REF 对比。
- **验收**（torch10_312 env）：
  - 16B (T=16, H=2048 → fn=[24,8192] fp32): `pre` max_diff=5.96e-8 / `post` max_diff=5.96e-7 /
    `comb_logits` max_diff=3.58e-7 — 远低于 5e-3 tol，**PASS**。
  - Next (T=16, H=4096 → fn=[24,16384] fp32): `pre` max_diff=1.19e-7 / `post` 1.19e-7 /
    `comb_logits` 3.58e-7 — **PASS**。
  - 全 10 stage Summary：K0 + K1 显示 "PASS"，K2..K4 + 组合 stage 仍 "SKIP (REF PASS; Zeus TODO)"。
- **下一步**：继续 K4 `mhc_post` (einsum + fp32 accum + 单 RNE) / K2 `mhc_sinkhorn` (per-token 2·repeat−1
  reduce 迭代) / K3 `mhc_pre_apply_mix` (stream 轴 weighted reduce)。helper `zeus_compare /
  finish_stage_with_zeus` 已可复用，每个后续 stage 只需在 zeus_skip 之上加 `if hasattr(sgl_kernel_zeus,
  ...)` 真实分支。

### 2026-05-23 · K1 GEMM 切到 bf16 weight + LocalMem（修正硬件路径）
- **背景**：Zeus weight DRAM / LocalMem **不支持 fp32 weight**。dev doc 早期 §"注意事项 fp32
  参数全程 fp32" 是对 Sinkhorn 20 迭代数值稳定性的考虑，但落地时与 Zeus chip 实际能力冲突。
  实测发现：bf16 量化的 fn 经过 fp32 累加 GEMM、fp32 inv_rms、fp32 sigmoid 后，pre / post
  / comb 输出与 fp32-fn REF 的差异：单元素 ~5e-4 (fn quant) → 24-维 dot 累加 ~1e-3 → sigmoid
  平滑后 pre ≤ 5e-3 / post (×2) ≤ 1e-2 / comb 直接 ~1e-3。Sinkhorn 对 ~1e-3 量级输入扰动鲁棒
  （softmax + 列归一对该量级输入扰动不放大），不影响 K2 输出的 doubly-stochastic 性质。
- **决策**：K1 GEMM 切到 **bf16 × bf16 → fp32 acc**，与 `dsa_q_a_proj_norm` /
  `gemm_bf16_jit_smoke_kernel` 同硬件路径。fn 走 **LocalMem (`kind="weight"`)** 装包。
- **改动**（5 件套 + tests + docs 一致更新）：
  - **Triton kernel** (`mhc_pre_norm_split_kernel.py`)：fn_ptr 类型注释 fp32 → bf16；
    `tl.dot(x_bf16, tl.trans(fn_pre_bf16), out_dtype=fp32)` (×3，pre/post/comb)；x_fp32 仅给
    sum_sq 用，GEMM 路径走 x_bf16；三个 fn block_ptr 仍带 `memory_type='weight'` (从 LocalMem 载入)。
  - **sim.c**：`fn` 指针 `const float*` → `const uint16_t*`；GEMM 内层 `bf16_to_float(&fn[i*NH+j])`
    再 double 累加，与 triton 路径数学等价。
  - **Host wrapper**：`TORCH_CHECK(fn.scalar_type() == at::kBFloat16, ...)` 替换原 fp32 检查；
    加 `torch_zeus::LocalMemAllocator::isLocalMem(fn.data_ptr())` 分支，LocalMem 时校验
    `handle->{batch_size==1, inner_rows==MIX_HC, inner_cols==NH}` 并 `copyFromLocalMem`
    物化到 row-major bf16 `fn_for_sim` 给 sim 用；non-LocalMem 时要求 contiguous bf16。
    与 `dsa_q_a_proj_norm_zeus.cpp` 完全同路径。
  - **Python API** (`glm5next_mhc.py`)：docstring 明示 fn 为 LocalMem-packed bf16，加示例：
    `fn_eff_fp32 = fn * norm_weight` → `.to(bf16)` → `torch.zeus.local_memory.from_tensor(
     ..., kind="weight", Tr=1, Tc=1)`。函数签名不变。
  - **Tests** (`test_mhc_pre_norm_split.py`)：所有正向用例都通过 `_pack_fn_localmem` helper
    走 LocalMem 路径；REF 内部 `.to(bf16).float()` round-trip 把 fn 量化误差吸收到 REF 侧，
    对照差异只剩 BLAS 累加噪声；tolerance `atol=2e-2, rtol=1e-2`（与 `test_dsa_q_a_proj_norm.py`
    一致）。`test_rejects_non_fp32_fn` 更名为 `test_rejects_non_bf16_fn`（fp32 fn 现在被拒绝）。
  - **Docs** (`docs/mhc_pre_norm_split.md` + slides)：全文 fp32 weight → bf16 LocalMem；§7
    porting 注意事项新增 "LocalMem from_tensor Tr/Tc 选择"；带宽预算更新：fn 流量 768 KB → 384 KB
    （省一半）。
- **dev 脚本** (`dev_glm5next_mhc_test.py::stage_mhc_pre_norm_split`)：调用前 host 侧
  `fn_eff_fp32 → .to(bf16) → from_tensor(kind="weight", Tr=1, Tc=1) → LocalMem`；REF 内部
  也用 bf16-quantized fn 算 (`MhcParams(fn=fn_eff_bf16.float(), ...)`) 让对照消除 fn 量化噪声，
  tolerance `atol=2e-2, rtol=1e-2`。
- **验收**：
  - sgl-kernel-zeus pytest：14 / 14 PASS（atol=2e-2, rtol=1e-2 envelope）。
  - dev 脚本 16B（T=16/H=2048, fn=[24,8192] bf16 LocalMem）：pre max_diff=5.96e-8 /
    post 2.38e-7 / comb 1.79e-7 — bf16 量化噪声已吸收到 REF 侧后只剩 BLAS 累加重排噪声，PASS。
  - dev 脚本 Next（T=16/H=4096, fn=[24,16384] bf16 LocalMem）：pre 5.96e-8 / post 1.19e-7 /
    comb 1.79e-7 — PASS。
- **下一步**：K2 mhc_sinkhorn / K3 mhc_pre_apply_mix / K4 mhc_post。K2 / K3 / K4 无 weight GEMM，
  不受 LocalMem fp32 限制影响；K4 的 einsum 也可以走 bf16 × bf16 → fp32 路径，但 K4 的 right
  matrix 是 residual（激活），不打包 LocalMem。

### 2026-05-23 · K3 `mhc_pre_apply_mix` 落地（5 件套 + docs/slides + dev 脚本接通）
- **5 件套 + 文档全套交付**（路径 `sgl-kernel-zeus/csrc/glm5next_mhc/` & `sgl-kernel-zeus/docs/`）：
  - Triton blueprint：`mhc_pre_apply_mix_kernel.py`。**沿 H 轴切核 CORE_NUM=2**（用户明确：H 比 N
    更独立，因为 H 是 reduce 之外的维度——两核各自完整跑 N 维 reduce、写 disjoint 输出，**无
    barrier / 无 atomic add**；沿 N 切则需要跨核 reduce）。T 循环外 4 路 `pre [BT, 1]` fp32
    load 常驻寄存器（跨 H tile 复用，64 B / token-batch）；H tile 内 N=4 静态展开（hardcoded
    `pre_0..pre_3` + `x0..x3`，避免 `pre[:, k]` 整数下标的前端 lowering 问题）：每路
    bf16→fp32 upcast + multiply-add 进 fp32 acc，4 路完成后单次 `to(bf16)` RNE round。
  - CPU sim：`sgl_mhc_pre_apply_mix_sim.c`（scalar `for t / for h / for k`：每 (t, h) 累加
    N 项 `pre[t,k] * bf16_to_float(residual[t, k*H + h])`，最后 `float_to_bf16` 单 RNE round。
    与 PyTorch 路径 `(pre * residual.float()).sum(dim=1).to(bf16)` 数学等价）。
  - Host wrapper：`mhc_pre_apply_mix_zeus.cpp`（dtype / contiguous / `n==4` / shape / numel
    强校验；residual 支持 `[T, N, H]` / `[T, N*H]` 两种 view；pre 支持 `[T, N, 1]` / `[T, N]`
    （按 numel 校验，同 storage）；**`H % CORE_NUM == 0` 整除校验**（CORE_NUM=2 硬编码 in host，
    GLM5-Next 实战 H ∈ {2048, 4096} 均整除）；`zertLaunchKernel(grid=1)` 投递）。
  - Python API：`glm5next_mhc.py::mhc_pre_apply_mix(residual, pre, *, n=4, out=None)`，未给
    out 时按 `(T, H)` bf16 自动分配；residual 2D / 3D 由 shape 推 H。
  - Tests：`tests/test_mhc_pre_apply_mix.py`（**16 用例全 PASS**）：6 个 shape (含 16B `H=2048`
    / Next `H=4096` 实战 shape，`atol=1e-2 rtol=1e-2`)；2D vs 3D residual + pre view 一致；
    **uniform-pre sanity**（pre=1/N + N 条流恒等 → out ≈ residual[:, 0, :]）；**one-hot pre
    sanity**（pre=e_k 时 out 逐 bit 等于 residual[:, k, :]，4 个 k 全测）；预分配 out；5 个
    拒绝用例（非 bf16 residual / 非 fp32 pre / `n!=4` / `H%2!=0` / 非 contiguous）。
  - Docs：`docs/mhc_pre_apply_mix.md`（7 节，对照 `silu_and_mul.md` / `mhc_expand.md` 骨架）
    + `docs/mhc_pre_apply_mix_slides.html`（8 张暗色幻灯片，mini 示意 `T=2, H=4, N=4` 的
    4 路加权 reduce 表格 + 沿 H 切核归属图 + GLM5-Next-16B 带宽预算）。
- **dev 脚本接通**：`stage_mhc_pre_apply_mix` 不再走 SKIP/TODO，调用
  `sgl_kernel_zeus.mhc_pre_apply_mix(residual.to("zeus"), pre.to("zeus"), n=N)`，与 REF
  对比 `atol=1e-2, rtol=1e-2`。
- **验收**（torch10_312 env）：
  - sgl-kernel-zeus pytest：**16/16 PASS**。
  - dev 脚本 16B (T=16, H=2048): max_diff=0.000e+00 — **PASS**。
  - dev 脚本 Next (T=16, H=4096): max_diff=0.000e+00 — **PASS**。
  - max_diff=0 因为 sim.c 用 `float`-acc + RNE round 与 PyTorch 的
    `(pre * residual.float()).sum(dim=1).to(bf16)` 数学路径完全一致（fp32 累加顺序也一致：
    对每 (t, h) 都是按 k=0..3 顺序累加）。
- **切核策略选择**：用户明确建议"考虑按 H 更独立"——选 H 是因为它是 reduce 之外的维度，
  两核 disjoint 写、无 barrier 是最优。N 切则要么 barrier 同步 + shared mem，要么 atomic
  add，两者都比 disjoint 写贵。pre 跨核冗余读（256 B / call，可忽略）是切 H 的唯一代价。
- **下一步**：K2 mhc_sinkhorn / K4 mhc_post。K3 落地后 dev script summary 显示 K0/K1/K3 三个
  PASS，仅剩 K2 + K4 + 组合 stage 是 SKIP/TODO。

### 2026-05-23 · K2 `mhc_sinkhorn` 落地（5 件套 + docs/slides + dev 脚本接通）
- **5 件套 + 文档全套交付**（路径 `sgl-kernel-zeus/csrc/glm5next_mhc/` & `sgl-kernel-zeus/docs/`）：
  - Triton blueprint：`mhc_sinkhorn_kernel.py`。**沿 T 轴切核 CORE_NUM=2**（K2 没有 H 轴
    可切；N 是 reduce 轴不能切，softmax + row/col norm 都需要完整 N=4 元素；T 是 reduce 之外
    的 disjoint 维度，每 token 完全独立——选 T 与"H 更独立"的设计 spirit 一致）。**2D
    block_ptr `[T, N*N]` + `tl.reshape` 到 3D `[BLOCK_T, N, N]`**（Zeus V3 对 3D
    `make_block_ptr` 支持偏弱，2D + reshape 是首选下降路径，参 SKILL.md §"1D 数据包成
    [1, N] 2D 视图"）。每 BLOCK_T tile 跑：stable softmax (axis=-1，减 max 防溢出) + 首轮
    col norm (axis=-2) + (repeat-1) 轮交替 row+col norm，共 `2*repeat-1` reduce / token。
    **`T % CORE_NUM` 不要求整除**：cdiv + boundary_check 兜底。
  - CPU sim：`sgl_mhc_sinkhorn_sim.c`（scalar per-token：4 嵌套 for + stable expf；栈分配
    `float x[64]` 暂存 4×4 矩阵）。
  - Host wrapper：`mhc_sinkhorn_zeus.cpp`（全套 dtype / contiguous / N==4 / 方阵 /
    `repeat >= 1` 校验；T 任意取值，无 T % CORE_NUM 硬约束）。
  - Python API：`glm5next_mhc.py::mhc_sinkhorn(comb_logits, *, repeat=20, eps=1e-6, out=None)`，
    未给 out 时按 `torch.empty_like(comb_logits)` 自动分配。
  - Tests：`tests/test_mhc_sinkhorn.py`（**27 用例全 PASS**）：6 个 T × 3 个 repeat = 18 个
    shape 组合（含 T=17 测 cdiv 兜底）；doubly-stochastic + non-negative 不变量；repeat=1
    边界；预分配 out；5 个拒绝用例（非 fp32 / N!=4 / 非方阵 / repeat=0 / 非 contiguous）。
    Tolerance `atol=5e-5, rtol=5e-5`（Sinkhorn contractive iteration 衰减 fp32 累加噪声）。
  - Docs：`docs/mhc_sinkhorn.md`（7 节，对照 `silu_and_mul.md` / `mhc_pre_apply_mix.md`
    骨架）+ `docs/mhc_sinkhorn_slides.html`（8 张暗色幻灯片，含 mini 示意 T=2 时 4×4 矩阵
    `[token0, token1]` 的 doubly-stochastic 投影对照图 + Sinkhorn 迭代步骤图）。
- **dev 脚本接通**：`stage_mhc_sinkhorn` 不再走 SKIP/TODO，调用
  `sgl_kernel_zeus.mhc_sinkhorn(comb_logits.to("zeus"), repeat=20, eps=1e-6)`，与 REF
  对比 `atol=5e-5, rtol=5e-5`；额外校验 Zeus 输出的 doubly-stochastic + non-negative
  不变量（与 REF 同步）。
- **验收**（torch10_312 env）：
  - sgl-kernel-zeus pytest：**27/27 PASS**。
  - dev 脚本 16B (T=16, N=4, repeat=20): max_diff=5.96e-8 / row_sum_err=1.07e-6 /
    col_sum_err=1.07e-6 / 0 negative entries — **PASS**。
  - dev 脚本 Next (同 shape，sinkhorn 不依赖 H): max_diff=5.96e-8 — **PASS**。
- **切核策略选择 vs 用户原建议 "H 或 N 倾向 H 更独立"**：K2 **没有 H 轴**——用户的建议从
  K3 语境复制过来。K2 实际可切轴只有 T（N 是 reduce 轴不能切）；而 T 是 reduce 之外的
  disjoint 维度，与"H 更独立"的 spirit 完全一致。文档 §3.1 + slides §6 都明确解释了
  这个选择推理。
- **下一步**：K4 mhc_post（最后一颗）。落地后整套 mHC 5 颗 kernel（K0..K4）就齐了，
  dev script 的组合 stage (mhc_pre / mhc_sublayer_wrap / mhc_layer_full) 可以接通 Zeus
  端到端路径。

### 2026-05-24 · K4 `mhc_post` 落地（5 件套 + docs/slides + dev 脚本接通）— mHC 5 颗全套完成 🎉
- **5 件套 + 文档全套交付**（路径 `sgl-kernel-zeus/csrc/glm5next_mhc/` & `sgl-kernel-zeus/docs/`）：
  - Triton blueprint：`mhc_post_kernel.py`。**沿 H 轴切核 CORE_NUM=2**（H 是 per-output
    维度——`out[t, n, h]` 只依赖该 h，不跨 h 累加；与 K3 mhc_pre_apply_mix 切核策略一致，
    用户原建议 "H 更独立"）。**T 循环外 20-tile preload**（4 个 `post [BT, 1]` fp32
    + 16 个 `comb [BT, 1]` fp32，共 ~320 B 寄存器驻留，跨 8 个 H tile 复用，不重复
    DRAM 读）。H tile 内 4 个输出 stream × 5-way (`1 post*x + 4 comb*res`) multiply-add
    双重静态展开（hardcoded `acc_0..acc_3` + `c_00..c_33` + `post_0..post_3` + `res_0..res_3`，
    全 fp32 acc，每输出元素 5 multiply-add）。**输出 4 个 disjoint streams 共享同一 out
    block_ptr 模板**，仅 advance 列偏移 `n * H + h_start` 不同（与 K0 mhc_expand N-路
    fanout 同 pattern）。
  - CPU sim：`sgl_mhc_post_sim.c`（scalar 4-嵌套 for: `t / n / h / m`：每 (t, n, h) 累加
    1 + N = 5 fp32 项，单 RNE 到 bf16；数学等价 PyTorch
    `post * x.unsqueeze(1) + (comb.unsqueeze(-1) * res.unsqueeze(2)).sum(dim=1)`）。
  - Host wrapper：`mhc_post_zeus.cpp`（5 个 tensor 校验：x [T,H] bf16 / residual 2D或3D
    bf16 / post 2D或3D fp32 / comb 2D或3D fp32 / out 多 view，按 numel 统一）；
    `H % CORE_NUM == 0` 整除校验；`n==4` 硬锁；`zertLaunchKernel(grid=1)` 投递。
  - Python API：`glm5next_mhc.py::mhc_post(x, residual, post, comb, *, n=4, out=None)`，
    未给 out 时按 `residual` view 形态自动分配（3D `[T, N, H]` 或 flat `[T, N*H]`），
    便于链式调用 K0..K4 全套。
  - Tests：`tests/test_mhc_post.py`（**16 用例全 PASS**）：6 个 shape 组合（含 16B
    `H=2048` / Next `H=4096` 实战 shape，`atol=1e-2, rtol=1e-2`）；2D vs 3D residual /
    post / comb view 一致性 (3 个测试)；**identity_sanity (post=0, comb=I → out=residual)**
    + **x_only_pickup (residual=0, comb=0, post=1 → 4 stream 全 = x)** 两个 sanity；
    预分配 out；5 个拒绝用例（非 bf16 x / 非 fp32 post / `n != 4` / `H%2 != 0` /
    非 contiguous）。
  - Docs：`docs/mhc_post.md`（7 节）+ `docs/mhc_post_slides.html`（8 张暗色幻灯片，含
    完整数据流图 + N=4 双重静态展开示例 + 沿 H 切核归属图 + GLM5-Next-16B 带宽预算）。
- **dev 脚本接通**：`stage_mhc_post` 不再走 SKIP/TODO，调用
  `sgl_kernel_zeus.mhc_post(x.to("zeus"), residual.to("zeus"), h_post.to("zeus"),
   h_res.to("zeus"), n=N)`，与 REF 对比 `atol=1e-2, rtol=1e-2`。
- **验收**（torch10_312 env）：
  - sgl-kernel-zeus pytest：**16/16 PASS**。
  - dev 脚本 16B (T=16, H=2048): max_diff=6.10e-5 — **PASS**。
  - dev 脚本 Next (T=16, H=4096): max_diff=2.44e-4 — **PASS**。
  - 差异主要来自 bf16 输出 RNE round noise（~4e-3 abs 上限，远低于 atol=1e-2）。
- **切核策略**：沿 H 轴（与 K3 同思路）— H 是 per-output 维度，两核 disjoint 写不同
  H 区段无 cross-core 通信。N (输出 stream n) 也是 disjoint 维度，但只 4 路太粗（核 0/1
  各 2 stream），H 更细（H/2 = 1024 列）。m (residual reduce 维) 不能切。
- **mHC 整套 5 颗 kernel (K0..K4) 全部 LANDED**：
  - K0 mhc_expand: bf16 memcpy fanout
  - K1 mhc_pre_norm_split: RMSNorm + bf16 GEMM(LocalMem) + scale/base/sigmoid/split
  - K2 mhc_sinkhorn: Birkhoff-polytope projection
  - K3 mhc_pre_apply_mix: stream-axis weighted reduce
  - K4 mhc_post: post-scatter + comb-weighted residual mix
- **下一步**：dev 脚本的组合 stage（mhc_pre / mhc_sublayer_wrap / mhc_layer_full）当前
  仍走 REF-only 路径；它们由 K0..K4 子算子组合而成，落地需要在 dev 脚本里把 Zeus 路径
  串起来（pre 三件套 → sinkhorn → apply_mix → identity sublayer → post）。这是端到端
  集成验证，可作为下一阶段任务。

### 2026-05-24 · 三个组合 stage 接通 Zeus 路径（mhc_pre / mhc_sublayer_wrap / mhc_layer_full）
- **背景**：K0..K4 5 颗 kernel 全部 LANDED 之后，dev 脚本里的三个组合 stage 仅做 REF
  自检，未真正调用 Zeus chain。本次在 dev 脚本里加 host-side 编排，把 K1→K2→K3 (→K4)
  串起来跑端到端 Zeus 路径，让组合 stage 也产出 PASS。
- **新增 helpers**（`dev_glm5next_mhc_test.py`，插在 `ref_mhc_post` 后）：
  - `quantize_p_for_zeus_match(p)`：构造与 Zeus K1 实际使用的等价 MhcParams，把
    norm_weight pre-merge 进 fn 再做 bf16→fp32 round-trip。喂给 `ref_mhc_pre /
    ref_mhc_post` 跑 REF 时会把 bf16 fn 量化误差吸收到 REF 侧，对照差异只剩 BLAS
    重排噪声（与 K1 stage 既有套路一致）。
  - `zeus_mhc_pre(residual_flat, p, cfg)`：K1 mhc_pre_norm_split → K2 mhc_sinkhorn
    → K3 mhc_pre_apply_mix 串接，返回 `(layer_input, residual_z, h_res, h_post)` 全部
    device-resident。fn_lmem 在 helper 内构造（fp32→bf16→LocalMem pack）；
    `residual_flat.to("zeus")` 对已在 zeus 上的 tensor 是 no-op，所以同一 helper
    既能接 CPU 入口也能接前一轮 K4 输出。
  - `zeus_mhc_post(x, residual, h_post, h_res, cfg)`：K4 直通调用。
  - `_zeus_has(*kernels)`：组合 stage 用的多 kernel availability 检查。
- **stage_mhc_pre**：加 Zeus 分支。REF 用 `quantize_p_for_zeus_match(p)` 跑一遍获得
  quantized golden，与 Zeus chain 对照三件套 (layer_input / h_res / h_post)
  `atol=2e-2 rtol=1e-2`；再校验 Zeus comb 的 doubly-stochastic 不变量。
- **stage_mhc_sublayer_wrap**：加 Zeus 分支。`zeus_mhc_pre → identity sublayer
  (x_sub_z = z_li) → zeus_mhc_post` 一条 K1+K2+K3+K4 链。与 quantized REF 的端到端
  `out` 对照 `atol=2e-2 rtol=1e-2`。
- **stage_mhc_layer_full**：加 Zeus 分支。两套独立 `p_attn / p_mlp` 各自 quantize；
  Zeus 路径 `attn_hc(K1..K4) → mlp_hc(K1..K4)` 共 8 颗 kernel 调用，z_mid 不下 host
  直接喂下一轮 K1。对照 `mid` (atol=2e-2 rtol=1e-2) + `out` (atol=5e-2 rtol=2e-2，
  双轮 K4 round + 双轮 K1 GEMM 复合) + Zeus 输出的流分歧不变量。
- **验收**（torch10_312 env）：
  - **16B** (T=16, H=2048)：
    - `mhc_pre`: layer_input max_diff=1.91e-6 / h_res 1.04e-7 / h_post 2.38e-7 — **PASS**
    - `mhc_sublayer_wrap`: out max_diff=9.77e-4 — **PASS**
    - `mhc_layer_full`: mid 3.91e-3 / out 7.81e-3 — **PASS**
  - **Next** (T=16, H=4096)：
    - `mhc_pre`: layer_input max_diff=4.88e-4 / h_res 8.94e-8 / h_post 1.19e-7 — **PASS**
    - `mhc_sublayer_wrap`: out max_diff=9.77e-4 — **PASS**
    - `mhc_layer_full`: mid 3.91e-3 / out 7.81e-3 — **PASS**
  - 两套配置 Summary 现在 K0/K1/K2/K3/K4 + mhc_pre / sublayer_wrap / layer_full
    全部 "PASS"；只剩 `mhc_pre_norm_fn` / `mhc_pre_split_mixes` 是 K1 的 debug 子步
    （已并入 `mhc_pre_norm_split`，不映射独立 Zeus kernel），仍正常 SKIP。
- **结论**：mHC 在 dev 脚本层面 **端到端 Zeus 路径全部接通**；下一步可由
  `dev_glm5next_block_decode_test.py` 把真实 attn/MLP sublayer 替换 identity，进入
  block 级端到端集成。

### 2026-05-25 · `linear_bf16` packed-weight GEMM 配套落地（解 GAP-3 / 不在 mHC 本体路径）
- **背景**：`glm5next_dsa_block_zeus_flow.md` 的 **GAP-3** 指出 GLM5-Next decode
  路径上 MoE router gate Linear、shared experts gate_up / down 这 3 颗 Linear
  以及 Linear-attn 的 5 个 projection 都还在 host CPU `torch.nn.functional.linear`
  上算。`sgl-kernel-zeus` 历史上只有 `moe_grouped_gemm`（grouped + sorted_ids
  寻址），没有 generic dense Linear。本日落地一颗 **`sgl_kernel_zeus.linear_bf16`**
  packed-weight bf16 GEMM kernel，一锤多治这一类 host Linear。
- **五件套 + 文档全套交付**（路径 `sgl-kernel-zeus/csrc/moe/` & `sgl-kernel-zeus/docs/`）：
  - Triton blueprint：`linear_bf16_kernel.py`。**`grid = (CORE_NUM = 2,)`，沿
    N 轴 split**（与 `moe_grouped_gemm` 拓扑一致；两核 disjoint 写 N 列段，无
    cross-core 通信）。每核 M × n_per_core × K 三重 for；每 (m, n_block) tile
    `[BLOCK_M=16, BLOCK_N=128]` 的 fp32 acc 经 K-loop 累加，最后单次 RNE 到
    bf16 store。weight 走 `memory_type='weight'`（LocalMem），与
    `mhc_pre_norm_split` / `dsa_q_a_proj_norm` / `moe_grouped_gemm` 同硬件路径。
    `BLOCK_M=16` 是为 decode T ≤ 16 优化（与 mHC 同 envelope）；`BLOCK_N=128` /
    `BLOCK_K=128` 与 `moe_grouped_gemm` 一致。
  - CPU sim：`sgl_linear_bf16_sim.c`。scalar 三重 `for m / for n / for k`：
    每个 (m, n) 累加 K 项 `bf16_to_float(input[m,k]) * bf16_to_float(weight[n,k])`，
    最后 `float_to_bf16(out[m,n])` 单 RNE round。与 PyTorch
    `F.linear(input.float(), weight.float()).to(bf16)` 数学等价。
  - Host wrapper：`linear_bf16_zeus.cpp`。全套 `TORCH_CHECK`：dtype (input /
    weight / output 都 bf16) / contiguous（input + output）/ 2D / shape /
    `weight.shape[1] == K, weight.shape[0] == N`。**LocalMem 物化路径**与
    `mhc_pre_norm_split_zeus.cpp` 完全一致：`isLocalMem` 检测 → 校验 handle
    shape `(batch=1, inner_rows=N, inner_cols=K)` → `copyFromLocalMem` 物化回
    row-major `[N, K]` bf16 给 sim 用；non-LocalMem 时 contiguous 校验。
    `zertLaunchKernel(grid=1)` (sim path)；porting 真 zbin 时切 grid=CORE_NUM=2。
  - Python API：`sgl_kernel_zeus.linear_bf16(input, weight, *, out=None)`。
    docstring 写明 weight 推荐用 `torch.zeus.local_memory.from_tensor(weight,
    kind="weight", Tr=1, Tc=1)` LocalMem 装包（production path），plain bf16
    contiguous tensor 也接受（host 自动检测 fallback）。
  - Tests：`tests/test_linear_bf16.py`（**14/14 用例全 PASS**）：6 个 shape（toy
    + GLM5-Next-16B 实战 `[16, 2816, 2048]` shared gate_up / `[16, 2048, 1408]`
    shared down + dense 风格 `[32, 256, 2048]`）；contiguous weight fallback；
    preallocated out；`F.linear` 语义 sanity；5 个 reject case（非 bf16 input /
    非 bf16 weight / shape 不匹配 / 3D input / 非 contiguous）。`atol=2e-2,
    rtol=1e-2`（与 `mhc_pre_norm_split` 同 envelope）。
  - Docs：`docs/linear_bf16.md`（7 节，对照 `silu_and_mul.md` / `mhc_post.md`
    骨架，含 GLM5-Next-16B shared gate_up 完整带宽预算 + porting checklist）+
    `docs/linear_bf16_slides.html`（8 张暗色幻灯片：title / GAP-3 motivation /
    F.linear semantics / N-split 切核 / GLM5-Next-16B 例子 / 单 tile K-loop /
    mini matrix walkthrough / 带宽 + porting）。
- **注册链**：`include/sgl_kernel_zeus_ops.h` 加 `linear_bf16_cxx` 声明；
  `csrc/common_extension.cpp` 加 `m.def("linear_bf16(Tensor input, Tensor
  weight, Tensor! output) -> ()")` + `m.impl`；`setup.py` 把 sim.c 和 .cpp
  加进 SIM_SOURCES / HOST_SOURCES；`python/sgl_kernel_zeus/__init__.py`
  re-export + `__all__`。
- **dev 脚本新 stage**：`dev_glm5next_mhc_test.py::stage_linear_bf16`（在
  `mhc_layer_full` 之后），不属于 mHC 本体路径，借 mHC dev 脚手架快速验证
  Zeus runtime + GLM5-Next-16B / Next 两套实战 shape。覆盖三类典型：
  shared gate_up、shared down、general small Linear。
- **验收**（torch10_312 env）：
  - sgl-kernel-zeus pytest：**14/14 PASS**（含 GLM5-Next-16B `[16, 2816, 2048]`
    / `[16, 2048, 1408]` 实战 shape）
  - dev 脚本 **16B** (T=16, H=2048)：
    - shared gate_up (M=16, N=4352, K=2048): max_diff=9.77e-4 — **PASS**
    - shared down    (M=16, N=2048, K=1152): max_diff=1.22e-4 — **PASS**
    - general small  (M=16, N=512,  K=1024): max_diff=0.000   — **PASS** (bit-exact)
  - dev 脚本 **Next** (T=16, H=4096)：
    - shared gate_up (M=16, N=8448, K=4096): max_diff=1.95e-3 — **PASS**
    - shared down    (M=16, N=4096, K=2176): max_diff=1.22e-4 — **PASS**
    - general small  (M=16, N=512,  K=1024): max_diff=1.49e-8 — **PASS**
- **GAP-3 解锁**：有了 `linear_bf16` 之后：
  - `dev_glm4_moe_test.py::moe_block_full` 里 host 端的 `gate Linear` 与
    `shared experts MLP` 可切到 Zeus（router / shared gate_up / shared down 共 3 处）
  - `dev_kimi_linear_attn_test.py::kimi_delta_attn_decode` 里
    `dev_kimi_linear_attn_test.py:1060` 明文 TODO 的 `o_proj` 以及其余 4 个
    projection 可切到 Zeus
  - `dev_glm5next_block_decode_test.py::zeus_moe_decode` 的 MoE router /
    shared experts CPU 段可切到 Zeus（彻底解 GAP-3，整条 block 真正 device-resident）
- **v1 scope 显式不做**：bias / activation fuse（gemm_bf16_dense 有 alpha/beta/
  bias 蓝本可参考）；fp8 / int8 / int4 量化；per-channel scale；transposed
  weight（`input @ weight`）；N-tail boundary_check 路径（GLM-4.7 router E=160
  不整除 256，v1 让 caller host pad）。
- **下一步**：dev_glm4_moe_test::moe_block_full 把 host 端 router + shared MLP
  切到 Zeus，彻底闭环 GAP-3；之后用同一颗 kernel 把 Linear-attn 5 个 projection
  也切到 Zeus，让 `dev_glm5next_block_decode_test.py::linear_attn_block` 也走向
  完全 device-resident。
