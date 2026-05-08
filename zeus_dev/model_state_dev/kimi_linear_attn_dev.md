# Kimi Linear Attention Zeus 适配开发追踪

> 对齐目标：`KimiDeltaAttention`（`python/sglang/srt/models/kimi_linear.py:161`）
> —— Kimi Linear 的 KDA（Kimi Delta Attention）子层，在纯 Zeus（无 CUDA）
> 环境下逐算子跑通。本文档记录这条路径的 porting 工作。

## 范围与方法

- **起点**：layer 输入的 `hidden_states [T, H]`（即 `input_layernorm` 之后、
  进入 `self_attn` 的张量）。
- **终点**：`o_proj` 的输出 `[T, H]`，可直接喂给 `post_attention_layernorm`。
- **切片**：单 device、单 layer、KDA layer only（`config.is_kda_layer(i)==True`）。
  先不碰 TP / PP / speculative / MLA layer。
- **关键模式切分**：KDA 在推理阶段有两条**完全不同**的数据流：
  - **decode**：`forward_batch.forward_mode.is_decode_or_idle()` —— 每 token
    单步，conv1d 走 **state-update** 路径，core attention 走 **fused recurrent**
    路径。
  - **prefill / extend**：`forward_batch.forward_mode.is_extend()` —— 多 token
    一次性前向，conv1d 走 **varlen causal conv**，core attention 走
    **chunk-wise**（O(T·BT) with `chunk_size=64`）路径。
  两条路径在 projection / gate / output-norm 上共享；但 conv1d + KDA 核心
  kernel 形态不同，Zeus 也要分别 porting 与测试。
- **对齐方式**：沿用 `demo_zeus_layer_compare.py` 与
  `dev_glm4_moe_test.py` 的 "REF vs Zeus" 模式 —— 同一份权重、同一份输入，
  REF（pure-torch，必要时落 CUDA Triton）产生 golden，Zeus 侧跑完后逐算子
  `compare_tensors`。**Zeus 路径禁止落到 torchnative fallback**，必须对齐到
  `sgl_kernel_zeus` 的 kernel 调用。
- **dev 脚本**：`zeus_dev/dev_kimi_linear_attn_test.py`，stage 化，按
  `--stage` 单独跑。未实现的 Zeus stage 显式 raise `NotImplementedError`
  并标 TODO，不要 silent fallback。

## Kimi Linear 相关配置

Kimi-Linear 的权威配置来自 HuggingFace checkpoint（`Kimi-Linear-48B-A3B`
等），`sglang/srt/configs/kimi_linear.py:KimiLinearConfig` 持有它。本 dev 文档
下表为**典型值**，真实跑 model 时从 checkpoint 的 `config.json` 读取：

| 字段 | 典型值 | 含义 |
|---|---|---|
| `hidden_size` (H) | 4096 / 5120 | 主干 residual 维度 |
| `linear_attn_config.num_heads` | 16 / 32 | KDA head 数 |
| `linear_attn_config.head_dim` | 128 | KDA head 维度 |
| `linear_attn_config.short_conv_kernel_size` | 4 | 因果卷积核 |
| `num_hidden_layers` | 32–64 | 总层数 |
| `linear_attn_config.kda_layers` | 稀疏索引集 | 哪些 layer 是 KDA（1-indexed） |
| `linear_attn_config.full_attn_layers` | 其余 | 走 MLA 路径（本文档不覆盖） |
| `rms_norm_eps` | 1e-5 / 1e-6 | `o_norm` 的 eps |

关键派生：
- `projection_size = num_heads * head_dim`。q/k/v/f_b/g_b 的输出维度都是这个
  大小，q/k/v 的 conv1d 也沿这维走。
- `local_num_heads = num_heads / tp_size`（单卡即 `num_heads`）。
- `A_log` 的 shape 为 `[1, 1, local_num_heads, 1]`，fp32。每层 fixed 参数。
- `dt_bias` 的 shape 为 `[projection_size / tp_size]`，fp32。
- `use_qk_l2norm_in_kernel=True` —— q/k 在进入 KDA core 之前做 L2 normalization，
  kernel 内部 fused 进 `fused_recurrent_kda` / `chunk_kda`。

## 算子依赖表（单卡、非量化、KDA layer）

> 源头：`KimiDeltaAttention.forward`（`python/sglang/srt/models/kimi_linear.py:296`）
> → `ForwardBatch.attn_backend.forward(**kwargs)` →
> `KimiLinearAttnBackend.forward_decode / forward_extend`
> （`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:307,401`）。

表里的 "Zeus 现状" 一栏都是**空白**（当前无任何 KDA 相关 kernel 落地），逐 stage
攻下后回填，与 `glm4_moe_ffn_dev.md` 同款更新节奏。

| # | 子步骤 | 形态 | shape | CUDA 实现 | Zeus 现状 |
|---|---|---|---|---|---|
| 0 | `input_layernorm`（fused add + rmsnorm） | 两路共享 | `[T,H]` | `sgl_kernel.fused_add_rmsnorm` | ✅ `sgl_kernel_zeus.fused_add_rmsnorm` |
| 1 | q / k / v / b / f_a / f_b / g_a / g_b Linear | 两路共享 | 各 GEMM | GEMM | ✅ `linear_zeus`（复用 qwen 已验证） |
| 2a-single | q / k / v causal conv1d — decode（单 projection，三次调用） | single-step | `[N, C]` + conv_state `[N, C, K-1]`，C=`H_qkv` | `causal_conv1d_update`（`mamba/causal_conv1d_triton.py:973`） | ✅ `sgl_kernel_zeus.causal_conv1d_update`（紧凑 state，26 tests PASS）+ ✅ **`causal_conv1d_update_indexed`**（直接对全局 `[N_pool, C, K-1]` 池按 `cache_indices [N]` slot 寻址，省 host gather/scatter；38 tests PASS，含 strict parity 与 untouched-slots bit-exact 验证；`csrc/mamba/causal_conv1d_update_indexed_{kernel.py,zeus.cpp,sim.c}`） |
| 2a-fused | q / k / v causal conv1d — decode（Q/K/V 一次 launch 融合） | single-step | 同 2a-single，三份独立 weight/bias/state | — （CUDA 侧三次独立 `causal_conv1d_update`） | ✅ `sgl_kernel_zeus.causal_conv1d_update_qkv`（`csrc/mamba/causal_conv1d_update_qkv_{kernel.py,zeus.cpp}`，要求 C%CORE_NUM=2==0；省 2 次 launch + concat 拷贝） |
| 2b | q / k / v causal conv1d — extend | varlen full | `[H_qkv, T]` + cu_seqlens | `causal_conv1d_fn`（`mamba/causal_conv1d_triton.py:378`） | ❌ TODO: `sgl_kernel_zeus.causal_conv1d_fn` |
| 3 | `beta = sigmoid(b_proj(x).float())` | 两路共享 | `[T, num_heads]` | GEMM + sigmoid | ✅ 复用 `linear_zeus` + `sigmoid`（elementwise） |
| 4 | `fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)` | 两路共享 | `[T, H_qkv]` → `[T, num_heads, head_dim]` | Triton `kda_gate_fwd_kernel`（`fla/kda.py:1244`） | ✅ `sgl_kernel_zeus.fused_kda_gate`（`csrc/mamba/fused_kda_gate_{kernel.py,zeus.cpp}`，19 tests PASS；stable softplus = max(z,0)+log(1+exp(−\|z\|))，无 tl.where） |
| 5 | q / k L2 norm（fused in kernel） | 两路共享 | `[1, T, num_heads, head_dim]` | `l2norm_fwd`（`fla/l2norm.py`）；`use_qk_l2norm_in_kernel=True` 时融进下一步 | ✅ `sgl_kernel_zeus.l2norm`（`csrc/mamba/l2norm_{kernel.py,zeus.cpp}`，21 tests PASS；CORE_NUM 沿 N 维 ceiling 分 core；单 tile per-row reduce BLOCK_D=128；无 N%2 约束；接受任意 `[*,D]` shape） |
| 6 | **decode core**：`fused_recurrent_kda` | recurrent | `q/k/v/g: [1,T,H,K]`, `beta: [1,T,H]`, `initial_state: [N,H,K,V]` | Triton `fused_recurrent_gated_delta_rule_fwd_kernel`（`fla/fused_recurrent.py`，`IS_KDA=True`） | ✅ **生产默认**：`sgl_kernel_zeus.fused_recurrent_kda_Sdecay`（21 tests PASS；S 先 decay 再 readout，与 CUDA fla 同形态、ops 更少）+ ✅ **生产融合默认**：`fused_recurrent_kda_Sdecay_indexed`（直接对全局 `[N_pool,H,K,V]` 池按 `cache_indices [N]` slot 寻址，省 host gather/scatter；17 tests PASS，含 strict parity；`csrc/mamba/fused_recurrent_kda_Sdecay_indexed_{kernel.py,zeus.cpp,sim.c}`）+ ✅ **Tensor Core 实验**：`fused_recurrent_kda`（18 tests PASS；decay 折进 k → `tl.dot` 走矩阵引擎、weight/share 双 memory tier 调度）。三版数学严格等价（S_new、o[t] 一致）。v1 要求 K==V==128。|
| 7 | **extend core**：`chunk_kda` | chunked O(T·BT) | 同 6；chunk_size=64 | 多 triton kernel 串联（`fla/kda.py:1141 chunk_kda_fwd`：cumsum → scaled_dot_kkt → solve_tril → recompute_w_u → chunk_delta_h → chunk_gla_o_gk） | ❌ TODO: `sgl_kernel_zeus.chunk_kda`（7 个子 kernel 组装） |
| 8 | `o_norm` —— `FusedRMSNormGated(out, g_gate, activation="sigmoid")` | 两路共享 | `[T, H_qkv]`, g=sigmoid | Triton `layer_norm_gated_fwd_kernel`（`fla/kda.py:159,239`） | ✅ `sgl_kernel_zeus.rms_norm_gated`（`csrc/mamba/rms_norm_gated_{kernel.py,zeus.cpp}`，24 tests PASS；两遍多 tile reduce 适配 H_qkv=2048；mean(x²) 而非 sum；activation=sigmoid；weight=1 固定）|
| 9 | `o_proj` | 两路共享 | `[T, H_qkv] → [T, H]` | GEMM | ✅ `linear_zeus` |
| 10 | cache update（conv_state / ssm_states） | 两路共享 | in-place | pointer scatter | ✅ `sgl_kernel_zeus.cache_index_gather` / `cache_index_scatter`（独立 row-level DMA，48 tests PASS；行内 contiguous + 行间 indirected）+ **生产路径已不再需要**：conv_state pool 由 `causal_conv1d_update_indexed` 内嵌寻址；ssm_state pool 由 `fused_recurrent_kda_Sdecay_indexed` 内嵌寻址。两个独立算子保留作为 conv1d_update_qkv / chunk_kda 等待融合 kernel 的 fallback + 通用工具。 |

**不在本文档 scope**：
- full-attn layer（走 `DeepseekV2AttentionMLA` / MLA 路径，是另一条完全不同的
  ported 链）。
- Speculative decoding 的 `target_verify` / `EagleVerifyInput` 分支。
- Cuda-graph capture 下的 `get_is_capture_mode()` 子图（`alt_stream` 双流
  shared-experts 并发）。
- TP 切分（目前所有 dev 对齐假设 `tp_size=1`）。

## KDA 计算流（算子组合 + 中间变量传递）

KDA 推理路径在 conv1d 与 core attention 两段上完全不同，其余算子（projection、gate
计算、output-norm、o_proj）两路共享。下面分 **Decode 路径**与 **Extend 路径**分别给
出单卡、非 TP 下的完整算子串联。符号约定：`T`=当前批次 token 总数，`H`=hidden_size，
`Hp`=projection_size=`Nh×D`，`Nh`=num_heads，`D`=head_dim，`K`=short_conv_kernel_size
（=4），`B`=batch_size（decode 下 `T=B`）。括号内序号对应算子依赖表的 `#` 列。

### Decode 路径（`is_decode_or_idle()`，每请求 1 token，T = B）

```
         residual_in [T,H] bf16       hidden_states [T,H] bf16
                  │                           │
                  └────────────┬──────────────┘
                               ▼
                  ┌────────────────────────────┐
             (0)  │  fused_add_rmsnorm          │
                  │  (sgl_kernel_zeus)          │
                  └────────────┬───────────────┘
                               │  x [T,H] bf16                 ← residual_out
     ┌───────────┬─────────────┼───────────────────┬────────────────────────┐
     │           │             │                   │                        │
     ▼           ▼             ▼                   ▼                        ▼
┌─────────┐ ┌─────────┐ ┌─────────┐  ┌──────────────────────┐  ┌───────────────────────┐
│(1)q_proj│ │(1)k_proj│ │(1)v_proj│  │ (1) b_proj           │  │ (1) f_a_proj          │
│ lin_z   │ │ lin_z   │ │ lin_z   │  │     linear_zeus      │  │     linear_zeus       │
└────┬────┘ └────┬────┘ └────┬────┘  └───────────┬──────────┘  └───────────┬───────────┘
     │           │           │    b_out [T,Nh] bf16             f_a_out [T,D] bf16
q/k/v_proj_states│           │                   │                          │
  [T,Hp] bf16    │           │                (3)▼ sigmoid(.float())    (1) ▼ f_b_proj lin_z
     │           │           │      beta [T,Nh] fp32            f_b_out [T,Hp] bf16
     │           │           │                                             │
     │           │           │                                         (4) ▼ fused_kda_gate
     │           │           │                                     (sgl_kernel_zeus)
     │           │           │                               −exp(A_log) × softplus(·+dt_bias)
     │           │           │                                             │
     │           │           │                                      g [T,Nh,D] fp32
     │           │           │
     │ ← conv_state_q/k/v_pool [N_pool,Hp,K-1] bf16  (full pool；Zeus indexed
     │   kernel 内部按 cache_indices[n] 寻址、in-place 更新对应 slot)
     │           │           │
     └───────────┴───────────┘
                 │
     ┌───────────────────────────────────────────────────────────────────┐
(2a) │  causal_conv1d_update_indexed × 3（q / k / v 各调一次）          │
     │  (sgl_kernel_zeus)                                                │
     │  对每 batch 元素 n： slot = cache_indices[n]                     │
     │  silu(Σ_k [pool[slot,:,k]; x_c][k] * w[c,k] + bias[c])           │
     │  pool[slot] 原位左移 + 追加 x_c；其它 slot 保持不变             │
     └───────────────────────────────────────────────────────────────────┘
                 │
  q [T,Hp] bf16    k [T,Hp] bf16    v [T,Hp] bf16
                 │
          rearrange  "n (h d) → 1 n h d"
                 │
  q / k / v   [1,T,Nh,D] bf16
  g           [1,T,Nh,D] fp32    ← unsqueeze(0)
  beta        [1,T,Nh]   fp32    ← unsqueeze(0)
  ssm_state_pool [N_pool,Nh,D,D] fp32  ← 直接传全局池（不做 gather）
  cache_indices  [B] i32                ← 散点 slot id
                 │
                 ▼
     ┌───────────────────────────────────────────────────────────────────┐
(6)  │  fused_recurrent_kda_Sdecay_indexed  (sgl_kernel_zeus)           │
     │  k̂_t = l2norm(k_t),  q̂_t = l2norm(q_t)       ← (5) fused      │
     │  per request n: slot = cache_indices[n]                          │
     │     S = pool[slot, :]    （load 一次）                            │
     │     S ← exp(g_t) · S                                             │
     │           + β_t · (v_t − S k̂_t) ⊗ k̂_t                          │
     │     o_t = S q̂_t                                                  │
     │     pool[slot, :] = S    （store 一次）                           │
     └───────────────────────────┬───────────────────────────────────────┘
                                 │  core_attn_out [1,T,Nh,D] bf16
                                 │  pool[cache_indices] 已被原地更新 (10)
                                 │
  x →(1)g_a_proj→[T,D]→(1)g_b_proj→[T,Hp]→rearrange→ g_gate [T,Nh,D] bf16
                                 │                                      │
                                 └──────────────────────────────────────┘
                                 │ (core_attn_out + g_gate 同时传入 o_norm)
                                 ▼
     ┌───────────────────────────────────────────────────────────────────┐
(8)  │  rms_norm_gated  (sgl_kernel_zeus)                               │
     │  y = rmsnorm(core_attn_out) * sigmoid(g_gate)                    │
     └───────────────────────────┬───────────────────────────────────────┘
                                 │  rearrange  "1 n h d → n (h d)"
                                 │  normed_out [T,Hp] bf16
                                 ▼
     ┌───────────────────────────────────────────────────────────────────┐
(9)  │  o_proj  (linear_zeus)                                           │
     └───────────────────────────┬───────────────────────────────────────┘
                                 │  output [T,H] bf16
                                 ▼
                     (post_attention_layernorm)
```

### Extend 路径（`is_extend()`，多 token 展平，cu_seqlens 分割序列边界）

```
         residual_in [T,H] bf16       hidden_states [T,H] bf16
                  │                           │
                  └────────────┬──────────────┘
                               ▼
                  ┌────────────────────────────┐
             (0)  │  fused_add_rmsnorm          │
                  │  (sgl_kernel_zeus)          │
                  └────────────┬───────────────┘
                               │  x [T,H] bf16                 ← residual_out
     ┌───────────┬─────────────┼───────────────────┬────────────────────────┐
     │           │             │                   │                        │
     ▼           ▼             ▼                   ▼                        ▼
(1) q/k/v_proj  （同 Decode）      b_proj + sigmoid            f_a_proj → f_b_proj + fused_kda_gate
     │                                             │                        │
  q/k/v_proj_states [T,Hp] bf16    beta [T,Nh] fp32             g [T,Nh,D] fp32
     │
  .transpose(0,1) → [Hp,T] bf16       ← causal_conv1d_fn 要求 channels-first 输入
     │
  conv_state_q/k/v [B,Hp,K-1] bf16   ← from cache（Zeus 不需转置）
  has_initial_state [B] bool          ← extend_prefix_lens > 0
  cu_seqlens / query_start_loc [B+1] int32
     │
     ┌───────────────────────────────────────────────────────────────────┐
(2b) │  causal_conv1d_fn × 3（q / k / v 各调一次）                      │
     │  (sgl_kernel_zeus)                                                │
     │  input [Hp,T] bf16；varlen：cu_seqlens 分割多序列边界            │
     │  has_initial_state=True 时从 conv_state 取初始历史帧             │
     │  conv_state 原位更新为每 seq 末尾 K-1 帧；output [Hp,T] bf16     │
     └───────────────────────────────────────────────────────────────────┘
     │
  .transpose(0,1) → [T,Hp] bf16
  rearrange  "n (h d) → 1 n h d"
     │
  q / k / v  [1,T,Nh,D] bf16
  g           [1,T,Nh,D] fp32    ← unsqueeze(0)
  beta        [1,T,Nh]   fp32    ← unsqueeze(0)
  initial_state [B,Nh,D,D] fp32  ← ssm_states[cache_indices]
  cu_seqlens    [B+1] int32
     │
     ▼
     ┌───────────────────────────────────────────────────────────────────┐
(7)  │  chunk_kda  (sgl_kernel_zeus)   chunk_size BT=64                 │
     │  k̂=l2norm(k), q̂=l2norm(q)              ← (5) fused             │
     │  ┌── 每 chunk 内（BT tokens）:                                   │
     │  │   cumsum_g        → intra-chunk gate 前缀积                   │
     │  │   scaled_dot_kkt  → A[t,s] = β_t · k̂_t · k̂_s · exp(Σg(s→t))│
     │  │   solve_tril      → delta rule 已见 k̂ 校正（三角方程组）      │
     │  │   recompute_w_u   → 修正后的 weighted key w 与 value u        │
     │  └── 跨 chunk:                                                   │
     │      chunk_delta_h   → inter-chunk S 递推（同 fused_recurrent 规则）│
     │      chunk_gla_o_gk  → intra + inter 合并得 o                   │
     └───────────────────────────┬───────────────────────────────────────┘
                                 │  core_attn_out [1,T,Nh,D] bf16
                                 │  final_state   [B,Nh,D,D] fp32
                                 │     └──→ ssm_states[cache_indices]      (10)
                                 │
  x →(1)g_a_proj→[T,D]→(1)g_b_proj→[T,Hp]→rearrange→ g_gate [T,Nh,D] bf16
                                 │                                      │
                                 └──────────────────────────────────────┘
                                 │ (core_attn_out + g_gate 同时传入 o_norm)
                                 ▼
     ┌───────────────────────────────────────────────────────────────────┐
(8)  │  rms_norm_gated  (sgl_kernel_zeus)                               │
     │  y = rmsnorm(core_attn_out) * sigmoid(g_gate)                    │
     └───────────────────────────┬───────────────────────────────────────┘
                                 │  rearrange  "1 n h d → n (h d)"
                                 │  normed_out [T,Hp] bf16
                                 ▼
     ┌───────────────────────────────────────────────────────────────────┐
(9)  │  o_proj  (linear_zeus)                                           │
     └───────────────────────────┬───────────────────────────────────────┘
                                 │  output [T,H] bf16
                                 ▼
                     (post_attention_layernorm)
```

**关键数据流说明**：

- **(1) 三组投影的执行时序不同**：`q/k/v_proj` 在 `KimiDeltaAttention.forward()` 里
  调用 backend **之前**完成；`b_proj / f_a_proj / f_b_proj` 以 callable 传入，在
  backend 内部执行（`forward_decode/forward_extend` 里直接调用）；`g_a_proj /
  g_b_proj` 在 backend 返回之后才运行——因此 g_gate 路径逻辑上可与 backend 并行，
  但目前 Python 实现是顺序执行。

- **(2a) vs (2b) 的 layout 差异**：Decode 的 `causal_conv1d_update` 直接接受
  `[T,Hp]` 输入；Extend 的 `causal_conv1d_fn` 要求 channels-first `[Hp,T]`，
  backend 在调用前后各做一次 `.transpose(0,1)` 适配。Zeus porting 必须保持这个
  约定，或在 `causal_conv1d_fn` 内部处理两种 layout。

- **conv_state layout 与 CUDA 不同**：CUDA backend 在调用前对 conv_state 做
  `.transpose(-1,-2)` 以满足 triton kernel 的 stride 要求；Zeus 直接接受自然
  C-contiguous `[B,Hp,K-1]` layout，**不需要转置**。两侧 layout 对齐是 porting
  `causal_conv1d_update/fn` 时最容易遗漏的坑。

- **(5) L2 norm fused 进 core kernel**：`use_qk_l2norm_in_kernel=True` 是 Kimi 的
  生产配置，q/k 的 L2 归一化在 `fused_recurrent_kda`（步骤 6）或 `chunk_kda`
  （步骤 7）kernel 内部完成，不单独 launch。stage 5 的独立 `l2norm` 测试只验证
  精度路径，不代表生产调用链会插入额外 kernel。

- **精度路径**：所有 projection 输入/输出为 bf16；beta、g 在 fp32 域计算；
  ssm_states 保持 fp32（防止矩阵累加 underflow）；conv_state 为 bf16（存储效率）；
  core kernel（fused_recurrent_kda / chunk_kda）以 fp32 递推 S_t，输出 o_t 存回
  bf16；rms_norm_gated 的归一化中间值为 fp32。

- **(7) chunk_kda 的 7-子-kernel 流水线**：子 kernel 之间共享 `[B,Nh,D,D]` 的
  inter-chunk state；每个 chunk 处理 BT=64 tokens；`cu_seqlens` 保证多序列边界
  对齐，每 seq 有独立的初始 S_0（`initial_state[b]`）。Zeus porting 按
  `cumsum_g → scaled_dot_kkt → solve_tril → recompute_w_u → chunk_delta_h →
  chunk_gla_o_gk` 顺序逐子 kernel 对齐，再组装整条 driver。

- **(8) rms_norm_gated 的门控激活**：`FusedRMSNormGated` 初始化时传
  `activation="sigmoid"`，是 `y = rmsnorm(x) * sigmoid(g)`，**不是**
  swish（`g * sigmoid(g)`）。对齐脚本中 REF 必须使用 `sigmoid`，否则
  权重维度对上但数值对不上。

## Dev 脚本 Stage 顺序

按 "从小到大、每步独立可验证" 的原则排布，与 MoE dev 脚本同款。8 条
projection（q/k/v/b/f_a/f_b/g_a/g_b）复用 `linear_zeus`（qwen demo 已验证），
不单独开 stage；所有 stage 都假设 projection 结果已在手。

1. `fused_kda_gate` —— softplus(β=1, threshold=20) * (−exp(A_log)) 的 per-head
   点乘 kernel。**先跑这个**，因为它完全独立于 conv1d / core attn，
   shape / dtype 最简单（`A_log` fp32，`g` bf16/fp32）。
2. `causal_conv1d_update` —— decode 单步 conv1d state-update。`kernel_size=4`，
   输入 `[T=bs, H_qkv]`，state `[H_qkv, K-1]` 就地更新。
3. `causal_conv1d_fn` —— extend varlen conv1d。输入 `[H_qkv, total_tokens]`，
   `has_initial_state[bs]` + `query_start_loc[bs+1]`；state 同样就地更新。
4. `l2norm` —— q/k 的 per-head-dim L2 normalize。小 kernel，用来独立验证
   精度路径（fp32 accum + bf16 I/O）。**注**：如果 Zeus 侧决定把 L2 norm 融进
   core kernel（`USE_QK_L2NORM_IN_KERNEL=True`），这 stage 就退化成 REF-only
   的等价性检查。
5. `fused_recurrent_kda` —— **decode core**。recurrent 形态，单步/多步
   （cu_seqlens 展平 batch）；输出 `[1, T, num_heads, head_dim]` + 更新后
   `final_state`。Zeus 侧是 v1 最小闭环，不追求 autotune。**已落地**两版：
   - **生产默认** (`fused_recurrent_kda_Sdecay`)：S 先 decay → 用未折叠 k 做 v̂
     readout → outer-update。**与 CUDA fla 参考同形态**（`fla/fused_recurrent.py`
     的 `fused_recurrent_gated_delta_rule_fwd_kernel` 即此顺序），少一次
     `k_eff = k · decay` 的 K 维向量乘，寄存器路径直观。生产 decode 链路与
     端到端 stage（`kimi_delta_attn_decode`）默认调用此版本。
   - **Tensor Core GEMV 实验** (`fused_recurrent_kda`)：decay 折进 k → v̂ readout
     → outer-update → S 末尾衰减；v̂ 与 o_t 都用 `tl.dot([1,K] bf16, [K,V] bf16)
     → [1,V] fp32` 走 Zeus 矩阵引擎，state 经 `memory_type='weight'` 落 weight 内存层、
     9b 之前的 element-wise S·decay 显式再 load 一份到 share 层。作真核
     Tensor Core 路径的精度 / 调度验证，**不是生产入口**。
   - 两版**数学严格等价**：`Σ_k (S[k,:]·decay[k]) · k̂[k] ≡ Σ_k S[k,:] · (k̂[k]·decay[k])`，
     S_new 与 o[t] 完全一致；当前两版 sim.c 算法体一致，dev parity max_diff = 0；
     真核 porting 后预期 parity 落入 ~1 ULP（baseline 的 `k_eff` 多 1 次 bf16 RNE）。
     详见 `sgl-kernel-zeus/docs/fused_recurrent_kda_Sdecay.md`。
6. `chunk_kda` —— **extend core**。7-sub-kernel 流水线，先拼对 "单个短
   seq" 再试 cu_seqlens varlen。建议 Zeus 侧按
   `cumsum → scaled_dot_kkt → solve_tril → recompute_w_u → chunk_delta_h →
   chunk_gla_o_gk` 的顺序逐子 kernel 对齐，再组装整条 driver。
7. `rms_norm_gated` —— `o_norm(core_attn_out, g_from_g_b_proj)` 带 sigmoid 门
   控（KDA 固定 `activation="sigmoid"`）。shape `[T, H_qkv]`，与标准 rmsnorm
   的差别在于 `y = rmsnorm(x) * sigmoid(g)`。
8. `kimi_delta_attn_decode` —— 端到端 **decode 路径**：projection → conv1d_update
   → kda_gate → fused_recurrent_kda → rms_norm_gated → o_proj。完全对齐
   `KimiLinearAttnBackend.forward_decode`。
9. `kimi_delta_attn_extend` —— 端到端 **extend 路径**：projection →
   conv1d_fn → kda_gate → chunk_kda → rms_norm_gated → o_proj。完全对齐
   `KimiLinearAttnBackend.forward_extend`。

stage 8/9 是**分别独立**的，和 MoE 端到端 `moe_block_full` 的关系类似 ——
它们不共享 REF 实现（REF 侧 decode 用 recurrent torch loop，extend 用
chunk-wise torch loop），需要各自单独拼装。

### 2026-04-24 · l2norm 落地

- 交付：五件套全部完成 + docs/l2norm.md + docs/l2norm_slides.html，21 个 pytest PASS。
- Zeus 侧接口：`sgl_kernel_zeus.l2norm(x, eps=1e-6, scale=None, out=None)`。
  - 输入接受任意 `[*, D]` bf16，Python API 内部 `reshape(-1, D)`，kernel 看到 `[N, D]`，返回前 reshape 回原 shape。
  - CORE_NUM=2 沿 N（行）维 **ceiling division**：`N_per_core = ceil(N / CORE_NUM)`，无 `N % 2 == 0` 约束，边界行由 `boundary_check` 自动掩码。
  - per-row reduce 在单个 `[BLOCK_T, BLOCK_D]` tile 内完成：`tl.sum(x*x, axis=1)` + `tl.sqrt` + `/ norm[:, None]`；BLOCK_D=128 覆盖 KDA head_dim=128。
  - OOB D 列补零不影响 sum_sq（0²=0）；OOB 行的 store 被 boundary_check 掩码，不写入输出。
  - `scale=None` 通过 `HAS_SCALE=False` constexpr 分支在编译期消除乘法（与 `scale=1.0` 不同，后者仍执行一次向量乘）。

## 开发日志


### 2026-04-23 · 起点

- 创建本文档、`dev_kimi_linear_attn_test.py` 骨架。
- 盘点 `sgl_kernel_zeus` 当前暴露算子（`python/sgl_kernel_zeus/__init__.py:72`）：
  - ✅ 可直接复用：`fused_add_rmsnorm`, `rmsnorm`, `silu_and_mul`,
    `rotary_embedding`, `store_kv_cache`, `extend_attention`, `decode_attention`,
    `embedding`，以及 MoE 套件。
  - ❌ **未暴露**（本 porting 的目标清单）：`causal_conv1d_update`,
    `causal_conv1d_fn`, `fused_kda_gate`, `l2norm`, `fused_recurrent_kda`,
    `chunk_kda`（以及它的 7 个子 kernel），`rms_norm_gated`。
- 两条 CUDA 参考路径的代码锚点：
  - decode：`hybrid_linear_attn_backend.py:307 KimiLinearAttnBackend.forward_decode`
  - extend：`hybrid_linear_attn_backend.py:401 KimiLinearAttnBackend.forward_extend`
  - core kernels：`python/sglang/srt/layers/attention/fla/`（`kda.py`,
    `fused_recurrent.py`, `chunk.py`, `chunk_delta_h.py`, `l2norm.py`,
    `cumsum.py`, `solve_tril.py`, `wy_fast.py`, `chunk_scaled_dot_kkt.py`,
    `chunk_o.py`）
  - conv1d kernels：`python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`
- 今日目标：把 stage 1 / 2（`qkv_linear_sanity`, `fused_kda_gate`）REF 侧跑
  通；Zeus 侧 stage 2 暂标 TODO 等待 kernel 落地。conv1d / KDA 核心先按
  pure-torch REF 打基线，做到 "REF 自洽、Zeus 侧报 NotImplementedError 并提示
  TODO 锚点"。

### 2026-04-23 · causal_conv1d_update 落地

- 交付：五件套全部完成，26 个 pytest PASS，dev stage `causal_conv1d_update` PASS。
- 新增 `sgl-kernel-zeus/csrc/mamba/` 目录（首个 mamba-family 算子）。
- Zeus 侧接口：`sgl_kernel_zeus.causal_conv1d_update(x, conv_state, weight, bias, activation)`。
  - `conv_state [N, C, K-1]` 原位更新，无需在 host 侧做 transpose（与 CUDA
    backend 不同，Zeus 直接接受自然 C-contiguous layout）。
  - CORE_NUM=2 整除校验已在 host wrapper 中加入，porting 到真核时不需要额外改动校验逻辑。
- `dev_kimi_linear_attn_test.py`：注意 Zeus 侧需先把 `conv_state` 显式移到
  `"privateuseone"` 设备（`.to("privateuseone")` 会创建新张量），以便 in-place
  update 被 `state_zeus_dev` 捕获；不要直接用 CPU 张量的 `.to()` 临时返回值做比较。

### 2026-04-24 · fused_kda_gate 落地

- 交付：五件套全部完成 + docs/fused_kda_gate.md + docs/fused_kda_gate_slides.html，19 个 pytest PASS。
- Zeus 侧接口：`sgl_kernel_zeus.fused_kda_gate(x, A_log, head_dim, g_bias=None, out=None)`。
  - 输入 `x [T, Hp] bf16`，A_log 接受 `[1,1,H,1]` 或 `[H]`（API 层统一 reshape），输出 `[T, H, D] fp32`。
  - **softplus 稳定化**：CUDA 用 `tl.where(z>20, z, log(1+exp(z)))` 的阈值分支；Zeus
    改用 `max(z,0) + log(1+exp(-|z|))` 等价公式，无 `tl.where`，无 exp 溢出，
    大正/大负 z 均数值稳定（tests 验证 z=+25 和 z=-30 的极端情况）。
  - A_log 以 `[1]` block_ptr 加载后 `[None, :]` reshape 为 `[1,1]`，广播乘
    `[BLOCK_T, BLOCK_D]` 无需额外 kernel launch。
  - g_bias 在 head 循环外加载一次（DRAM 访问 per-head 而非 per-tile）。
  - H % CORE_NUM == 0 约束已在 host wrapper 校验。

## KDA 推理过程详解

> 本节补充「整条推理链的完整机制」，作为算子开发的背景知识。
> 目标读者：对 Linear Attention / Mamba 原理不熟，需要从头理解 KDA 如何工作。

---

### 一、KDA 是什么

**Kimi Delta Attention（KDA）** 是一种线性复杂度的序列建模机制，属于
Linear Attention 家族。它的核心思想是用一个 **递推隐状态矩阵** `S_t ∈ ℝ^{K×V}`
替代 Softmax Attention 中的全 Token KV 缓存：

```
经典 KV Cache Attention（O(T) 内存 per layer）：
  attn_out_t = softmax(q_t @ K_{1:t}^T) @ V_{1:t}

KDA Linear Attention（O(K²) 固定内存 per layer）：
  S_t = S_{t-1} * exp(g_t) + beta_t * (v_t - S_{t-1} k̂_t) ⊗ k̂_t   ← delta 规则
  o_t = q̂_t @ S_t
```

`S_t` 是 **K×V 的矩阵**（在实现里 head_dim×head_dim），与 Mamba 的 SSM 状态
类比：无论序列多长，存储量固定。代价是序列内精度不如 Softmax（信息有损压缩），
Kimi 通过稀疏分布 KDA/MLA 混合层来弥补这个差异。

**符号定义**（单个 head，时刻 t）：

| 符号 | 来源 | 含义 |
|---|---|---|
| `q̂_t, k̂_t` | `q_proj → conv → L2norm → scale` | query / key（L2 归一化后） |
| `v_t` | `v_proj → conv` | value |
| `g_t ∈ ℝ^K` | `fused_kda_gate` 输出 | per-dim 衰减（对数空间为 A_log + softplus） |
| `beta_t ∈ ℝ` | `b_proj → sigmoid` | per-head 遗忘系数（0~1） |
| `S_t ∈ ℝ^{K×V}` | `ssm_states` 维护 | 递推状态矩阵（= 滚动 KV 关联矩阵） |

递推更新（展开写）：

```
delta_t  = v_t - S_{t-1} k̂_t          # 当前 v 与状态预测的 v̂ 之差
S_t      = diag(exp(g_t)) S_{t-1}      # 衰减旧状态（沿 K 维逐元素乘）
         + beta_t * (delta_t ⊗ k̂_t)  # 写入新信息
o_t      = S_t q̂_t                    # 读出当前 token 的输出
```

---

### 二、完整 Forward 流程（单 KDA Layer）

两条路径（decode / extend）在投影、gate、归一化上**完全相同**，仅
conv1d 和 core attention 核形态不同。

#### 2.1 共有步骤（两路路径均执行）

**输入**：`hidden_states [T, H]`（经 `input_layernorm` 之后的 residual stream，
T = 当前批次 token 总数，H = hidden_size）。

```
Step A: 8 个线性投影（并行）
  q_proj_states = q_proj(hidden_states)   → [T, Hp]
  k_proj_states = k_proj(hidden_states)   → [T, Hp]
  v_proj_states = v_proj(hidden_states)   → [T, Hp]
  b_proj_states = b_proj(hidden_states)   → [T, N_h]   （N_h = num_heads）
  f_a_out       = f_a_proj(hidden_states) → [T, D]      （D = head_dim）
  f_b_out       = f_b_proj(f_a_out)       → [T, Hp]
  g_a_out       = g_a_proj(hidden_states) → [T, D]
  g_b_out       = g_b_proj(g_a_out)       → [T, Hp]

  Hp = projection_size = num_heads * head_dim

Step B: beta 与 gate
  beta = sigmoid(b_proj_states.float())   → [T, N_h]   float32
  g    = fused_kda_gate(f_b_out, A_log, head_dim, g_bias=dt_bias)
       → [T, N_h, D]  float32
       # 公式：g[t, h, d] = (-exp(A_log[h])) * softplus(f_b_out[t, h*D+d] + dt_bias[h*D+d])
       # softplus 阈值 threshold=20 防溢出

Step G: output gate（为后面 o_norm 准备）
  g_gate = rearrange(g_b_out, "T (h d) -> T h d", d=head_dim)
         → [T, N_h, D]   # g_b_out 已包含 g_a_proj 的残差信息
```

---

#### 2.2 Decode 路径（`is_decode_or_idle() == True`）

**场景**：自回归生成，每次为每个请求生成 1 个新 token。
批次里 `T == batch_size`（每个请求恰好 1 个 token）。

```
Step 1: 因果 conv1d 单步更新（per q/k/v 各一次）
  从 cache 取 conv_state:
    q_conv_state [B, Hp, K-1]   # K = short_conv_kernel_size = 4
    k_conv_state [B, Hp, K-1]
    v_conv_state [B, Hp, K-1]

  q = causal_conv1d_update(q_proj_states, q_conv_state, q_conv_weight, q_conv_bias, "silu")
    → [B, Hp]
    # 公式（per channel c）:
    #   window = [state[0], state[1], ..., state[K-2], x[c]]
    #   out[c] = silu(Σ_k window[k] * weight[c,k] + bias[c])
    #   conv_state[c, 0..K-2] = window[1..K-1]   ← 原位左移
  # 同理处理 k, v

Step 2: reshape 为 head 维度
  q = rearrange(q, "B (h d) -> 1 B h d", d=head_dim)  → [1, B, N_h, D]
  k 同理；v 同理

Step 3: 从状态池取 SSM 初始状态
  initial_state = ssm_states[cache_indices]             → [B, N_h, D, D]
  # cache_indices: 每个请求在全局 state pool 里的槽位编号

Step 4: fused_recurrent_kda（线性 RNN 单步或多步递推）
  cu_seqlens = [0, 1, 2, ..., B]   # decode 下每 seq 恰好 1 token

  (core_attn_out, final_state) = fused_recurrent_kda(
      q=q,               # [1, B, N_h, D]
      k=k,               # [1, B, N_h, D]
      v=v,               # [1, B, N_h, D]
      g=g.unsqueeze(0),  # [1, B, N_h, D]
      beta=beta.unsqueeze(0),        # [1, B, N_h]
      initial_state=initial_state,   # [B, N_h, D, D]
      use_qk_l2norm_in_kernel=True,
      cu_seqlens=cu_seqlens,
  )
  # core_attn_out: [1, B, N_h, D]
  # final_state:   [B, N_h, D, D]   ← 更新后的 S_t

  # kernel 内部每个 token t（每个 head h）执行：
  #   k̂_t = l2norm(k_t) * scale
  #   q̂_t = l2norm(q_t) * scale
  #   S_t = diag(exp(g_t)) S_{t-1}  +  beta_t * (v_t - S_{t-1}k̂_t) ⊗ k̂_t
  #   o_t = S_t q̂_t

Step 5: 写回状态
  ssm_states[cache_indices] = final_state   ← 原位更新 cache slot
  # conv_state 在 Step 1 已原位更新

Step 6: output 归一化（回到 KimiDeltaAttention.forward）
  core_attn_out = o_norm(core_attn_out, g_gate)
  # FusedRMSNormGated：y = rmsnorm(x) * sigmoid(g_gate)
  # x:      [1, B, N_h, D]
  # g_gate: [B, N_h, D]

Step 7: 线性输出投影
  core_attn_out = rearrange(core_attn_out, "1 B h d -> B (h d)")
  output = o_proj(core_attn_out)  → [B, H]
```

---

#### 2.3 Extend / Prefill 路径（`is_extend() == True`）

**场景**：初次 prefill 或 extend（带 prefix）。批次里可以有多条序列，每条序列
长度不同。所有 token 展平成 `[T_total, H]`，用 `cu_seqlens = [0, T_1, T_1+T_2, ...]`
标记边界。

```
Step 1: varlen 因果 conv1d（per q/k/v 各一次）
  # 先转置：[T_total, Hp] → [Hp, T_total]
  q_proj_states_t = q_proj_states.transpose(0, 1)

  q = causal_conv1d_fn(
      x                = q_proj_states_t,   # [Hp, T_total]
      weight           = q_conv_weight,      # [Hp, K]
      bias             = q_conv_bias,        # [Hp]
      activation       = "silu",
      conv_states      = q_conv_state,       # [B, Hp, K-1] in-place
      has_initial_state= has_initial_state,  # [B] bool: 是否有 prefix 状态
      cache_indices    = cache_indices,
      query_start_loc  = query_start_loc,    # [B+1] cumulative offsets
  ).transpose(0, 1)    → [T_total, Hp]

  # 每个 seq b（长度 T_b）：
  #   如果 has_initial_state[b]：state 初始化为 conv_state[b]
  #   否则：全零 state
  #   对 T_b 个 token 逐步做 causal conv1d（每步左移 state）
  #   最终 conv_state[b] 更新为最后 K-1 帧

Step 2: reshape 同 decode
  q/k/v: [T_total, Hp] → [1, T_total, N_h, D]

Step 3: 取 SSM 初始状态（同 decode）

Step 4: chunk_kda（分块 O(T·BT) 算法，BT=64）
  cu_seqlens 如 [0, T_1, T_1+T_2] 表示多序列边界

  (core_attn_out, final_state) = chunk_kda(
      q=q, k=k, v=v,
      g=g.unsqueeze(0),
      beta=beta.unsqueeze(0),
      initial_state=initial_state,
      output_final_state=True,
      use_qk_l2norm_in_kernel=True,
      cu_seqlens=cu_seqlens,
  )
  # core_attn_out: [1, T_total, N_h, D]
  # final_state:   [B, N_h, D, D]

  # chunk_kda 内部流水线（以单序列为例，chunk_size=64）：
  #   设共 ceil(T/64) 个 chunk，每 chunk 内 BT=64 tokens
  #   ┌── 每 chunk 内：
  #   │   cumsum_g  → 前缀累加 g（用于 intra-chunk gate 因子）
  #   │   scaled_dot_kkt → 计算 intra-chunk attention 矩阵 A[t,s]
  #   │              A[t,s] = beta[t] * k[t]·k[s] * exp(Σg(s..t))
  #   │   solve_tril → 三角线性方程组（处理 delta rule 的 "已看见 k" 校正）
  #   │   recompute_w_u → 计算修正后的 w（weighted key）和 u（weighted value）
  #   └── chunk 间：
  #       chunk_delta_h → inter-chunk 递推更新 S（与 fused_recurrent 相同的 delta 规则）
  #       chunk_gla_o_gk → 把 intra-chunk attention + inter-chunk state 合并出 o

Step 5/6/7 同 decode。
```

---

### 三、两路路径对比总结

| 维度 | Decode | Extend/Prefill |
|---|---|---|
| 触发条件 | `is_decode_or_idle()` | `is_extend()` |
| token 数 T | = batch_size（每请求 1 token） | = Σ seq_len（多 token 展平） |
| conv1d 算子 | `causal_conv1d_update`（单步） | `causal_conv1d_fn`（varlen） |
| core attention | `fused_recurrent_kda`（O(T) 纯递推） | `chunk_kda`（O(T·64) 分块） |
| cu_seqlens | `[0, 1, 2, ..., B]` | `[0, T_1, T_1+T_2, ...]` |
| has_initial_state | 始终 True（状态从 cache 来） | False 对全新 seq；True 对 extend |
| 性能特征 | 极低延迟（每步 O(1) 计算） | 高吞吐（长序列并行分块） |

---

### 四、状态管理：两级缓存

```
请求 req_id  ←→  cache slot（由 mamba_cache_indices 映射）

Layer 视角（一层一个 Layer 的 State）：

  conv_state  [num_slots, Hp, K-1]  bf16
  ├── 语义：q/k/v 各 3 份，存近 K-1 帧的投影特征
  ├── 更新：decode → causal_conv1d_update 原位写
  │         extend → causal_conv1d_fn 原位写最后 K-1 帧
  └── 初始化：新请求时清零

  ssm_states  [num_slots, N_h, D, D]  fp32
  ├── 语义：每 head 的 K×V 关联矩阵（= 压缩的 "记忆"）
  ├── 更新：kernel 返回 final_state 后 scatter 写回对应 slot
  └── 初始化：新请求时清零；有 prefix 时从 checkpoint 载入
```

**为什么 conv_state layout 是 [N, Hp, K-1] 而不是 [N, K-1, Hp]**：
Zeus 直接接受自然 C-contiguous 布局（最后一维 K-1 是 time 轴）。
CUDA 侧的 triton kernel 恰好相反（要求 dim 轴 stride=1），所以 CUDA
backend 会在传参前做一次 `.transpose(-1, -2)`。**Zeus porting 不需要这个转置**。

---

### 五、cu_seqlens 语义图解

```
设 batch = 3 个请求，prefill 阶段长度为 [5, 3, 7]：

展平后 hidden_states: [15, H]
  token indices:  0 1 2 3 4 | 5 6 7 | 8 9 10 11 12 13 14
                  ←seq 0──→   ←─1──→  ←────seq 2───────→

cu_seqlens = [0, 5, 8, 15]   (int32, shape [4])

含义：
  seq b 的 token 范围 = [cu_seqlens[b], cu_seqlens[b+1])
  fused_recurrent_kda / chunk_kda 用它在展平 batch 中
  定位各 seq 的边界，每个 seq 有独立的初始状态 initial_state[b]

Decode 阶段（每 req 1 token）：
  cu_seqlens = [0, 1, 2, 3]   即 arange(0, B+1)
```

---

### 六、数值精度路径

| 位置 | dtype | 原因 |
|---|---|---|
| hidden_states, q/k/v proj | bf16 | 网络主干精度 |
| conv_state | bf16 | 存储效率 |
| ssm_states（S_t） | fp32 | 累加精度，防止矩阵元素 underflow |
| beta, g（A_log 输出） | fp32 | gate 运算精度敏感 |
| conv1d 累加器 | fp32（sim.c）/ fp32（triton） | 避免 bf16 mul-add 误差 |
| o_norm 中间值 | fp32 | rmsnorm 归一化需要 |
| kernel 输出 o | bf16 | 写回 residual stream |

---

## 注意事项（踩坑记录 / 语义差异）

- **Linear vs Full-attn 双路径共存**：Kimi 的每层 `self_attn` 要么是 KDA 要么
  是 MLA，二选一（`config.is_kda_layer(layer_idx)`）。本 dev **仅**覆盖 KDA
  子路径；MLA 子路径的 Zeus porting 另起一篇 doc。
- **Decode vs Extend 不是对称的**：虽然 projection / gate / output-norm 共享，
  **conv1d + core attention 的 kernel 完全不同**（状态 update 型 vs varlen chunk
  型）。Zeus 侧必须分别 porting，对齐脚本里每条路径也得独立 REF 实现。不要
  试图用同一个 Zeus kernel "cover 两路"。
- **`cu_seqlens` 语义**：`fused_recurrent_kda` 与 `chunk_kda` 在
  `cu_seqlens != None` 时要求 `q.shape[0] == 1`（即整个 batch 必须**先展平**
  成一个 flat sequence，再用 cu_seqlens 切回原始序列边界）。对齐脚本里 REF
  必须按这个 convention 喂数据，否则维度对不上。
- **`ssm_states` / `conv_state` 原位更新**：decode 和 extend 都是**原位写回**
  `req_to_token_pool.mamba2_layer_cache(layer_id)` 的 buffer。dev 脚本里要
  保留两份 state 副本（REF 用一份、Zeus 用一份），跑完分别比，避免互相污染。
- **`use_qk_l2norm_in_kernel=True` 是生产配置**：Kimi 的 KDA 始终把 q/k 的 L2
  normalize 融进 core kernel。Zeus v1 建议保持一致（避免额外 kernel launch），
  stage 5 独立 `l2norm` 测试只是为了**数值路径**能独立验证，不代表生产路径
  会单独调。
- **`fused_kda_gate` 的 softplus 数值路径**：kernel 内 `softplus(x, β=1) =
  log(1 + exp(x))`，当 `x > threshold=20` 时切到线性近似 `x`，避免 `exp` 溢出。
  Zeus sim 必须 **完整镜像**这条分支，否则大 `f_b_proj` 输出下精度会偏。
- **`rms_norm_gated` 的激活是 `sigmoid`（而非 swish）**：KDA 的 `FusedRMSNormGated`
  初始化时传的是 `activation="sigmoid"`（`kimi_linear.py:287`）。是 `y =
  rmsnorm(x) * sigmoid(g)`，**不乘 `g` 本身** —— 对齐时别把 swish（`g * sigmoid(g)`）
  误当成 sigmoid。
- **Conv1d state 的 layout 转置**：backend 里 `conv_state = conv_state.transpose(-1, -2)`
  是为了适配 triton kernel 的内存排布（`[H_qkv, K-1]` vs `[K-1, H_qkv]`）。
  Zeus 侧 porting 时要先决定内部 layout，再在 host wrapper 中处理转置/视图，
  **不要**把转置代价下沉进 kernel 热路径。

## 开发日志

### 2026-04-24 — stage #5 l2norm 完成

- **交付**：`sgl_kernel_zeus.l2norm`，五件套 + docs/l2norm.md + docs/l2norm_slides.html。
- **关键决策**：CORE_NUM ceiling division over N 行，无 N%2 约束；单 tile per-row reduce（BLOCK_D≥D，适用于 KDA head_dim=128）；OOB 零填充不影响 sum_sq（0²=0）；HAS_SCALE 消除 scale=None 时的额外乘法。
- **测试**：21 tests PASS（多种形状 + 3D/4D multi-dim + unit-norm 验证 + scale=None/1.0 等价 + preallocated out + 4 rejection tests）。

### 2026-04-24 — stage #8 rms_norm_gated 完成

- **交付**：`sgl_kernel_zeus.rms_norm_gated`，五件套 + docs/rms_norm_gated.md + docs/rms_norm_gated_slides.html。
- **关键决策**：两遍多 tile reduce（Phase 1 累加 sum_sq，Phase 2 normalize+gate）适配 H_qkv=2048（D_blocks=16）；mean(x²) 而非 sum(x²)；activation=sigmoid（非 swish）；weight=1 固定（无可学习 γ）；CORE_NUM ceiling division over N，无整除约束。
- **sim.c**：两遍 for 循环，fp32 累加，`1/(1+expf(-gv))` sigmoid，RNE round 写 bf16。
- **测试**：24 tests PASS（11 种形状 + KDA 生产 shape [4,2048] + 3D/4D + preallocated + large_g/zero_g 语义验证 + 7 rejection tests）；原有 94 tests（causal_conv1d×26 + qkv + fused_kda_gate×19 + l2norm×21）全部 regression 通过。

### 2026-04-27 — stage #6 fused_recurrent_kda 两版落地（生产默认 = Sdecay）

- **角色分配**：
  - **生产默认** = `sgl_kernel_zeus.fused_recurrent_kda_Sdecay`（21 tests PASS）。S 先 decay 再 readout，**与 CUDA fla 参考严格同形态**，operations 比 baseline 少一次 K 维向量乘（`k_eff = k · decay`）；寄存器路径与 sim.c / 数学公式一一对应。生产 decode 链路、端到端 stage `kimi_delta_attn_decode`、未来集成到 `KimiLinearAttnBackend.forward_decode` 时**默认调用此版本**。
  - **Tensor Core GEMV 实验** = `sgl_kernel_zeus.fused_recurrent_kda`（18 tests PASS）。decay 折进 k 形态，v̂ 与 o_t 都用 `tl.dot([1,K] bf16, [K,V] bf16) → [1,V] fp32` 走 Zeus 矩阵引擎；state 经 `memory_type='weight'` 落 weight 内存层、9b 之前显式再 load 一份到 share 层。作真核 Tensor Core 路径的精度 / 调度验证用，**不是生产入口**。
- **共 39 tests PASS**（生产默认 21 + TC 实验 18），dev stage `fused_recurrent_kda_Sdecay` / `fused_recurrent_kda` 均 PASS（vs pure-torch REF + 互比 parity）。
- **Zeus 侧接口**（两 op 入参完全相同）：
  - `sgl_kernel_zeus.fused_recurrent_kda_Sdecay(q, k, v, g, beta, initial_state, cu_seqlens, scale=None, eps=1e-6, use_qk_l2norm_in_kernel=True, output_final_state=True, inplace_final_state=True, o=None)` ← **默认调这个**
  - `sgl_kernel_zeus.fused_recurrent_kda(...)` ← TC 实验，签名同上
  - 两者都 in-place 更新 `initial_state` 为 final_state；返回 `(o, final_state_alias)`。
- **数学等价依据**：`Σ_k (S[k,:]·decay[k]) · k̂[k] ≡ Σ_k S[k,:] · (k̂[k]·decay[k])`（K 维分配律展开同表达式）。两版的 S_new 与 o[t] 完全一致；当前 sim 路径互比 max_diff = 0，真核 porting 后预期 parity 落入 ~1 ULP（来自 baseline 的 `k_eff` 多 1 次 bf16 RNE）。
- **关键决策**（TC 实验版 baseline kernel）：
  - 三层循环（request→head→token），CORE_NUM=2 沿 head 切 ceiling division。
  - state 经 2D `make_block_ptr` `[N*H*BLOCK_K, BLOCK_V]` 暴露，load 直出 `[BLOCK_K, BLOCK_V]` 不 reshape；v1 host enforce K==V==128。
  - S 在 token loop 内全程 **bf16 register working copy**；fp32 算术在操作数处 `.to(fp32)` cast，编译器折叠到张量计算流水。
  - **矩阵引擎路径**：步骤 7 v̂ 与步骤 10 o_t 都用 `tl.dot([1,K] bf16, [K,V] bf16) → [1,V] fp32`，对齐 Zeus 的 PE 原生 bf16×bf16→fp32 路径。
  - **双 memory tier 调度**：
    - `memory_type='weight'`：初始 S load、10pre 的 store→load round-trip、final state store —— feed 矩阵引擎。
    - **9b 之前显式再 load** 一次 S 到 share-mem（不带 memory_type），feed 向量引擎做 element-wise decay 乘法。
    - 同一份 DRAM 内容两 tier 同步，避免向量引擎跨网络读 weight tier。
  - **每 token 1 次 bf16 RNE**：发生在 10pre 的 `S_new_fp32.to(bf16) → store(weight-mem)` 处；store 同时把 in-loop final state 写回 DRAM，pair 末尾的 final store 主要兜底零 token 边界。
- **生产默认（Sdecay）的关键决策**：保持 broadcast-multiply + `tl.sum` 形态（VP 路径），S 全程 fp32 register working；与 simple kernel 同形态，跨 token 无寄存器层 RNE，靠每 pair 末尾一次 fp32→bf16 RNE 写回 DRAM。
- **sim.c**：fp32 标量循环（两个 sim 文件算法体一致，仅入口符号不同），与 dev_doc 公式一一对应；现在两版 sim 一致让 dev 互比 parity 完全为 0。真核 porting 后预期 parity 落入 ~1 ULP 范围。
- **dev script**：新增 `_make_kda_inputs` 生成函数让两 stage 共用同一份输入；`fused_recurrent_kda_Sdecay` stage 同时验证 vs REF 与 vs baseline parity（atol/rtol=1e-2）。两个 stage 都覆盖 cfg.head_dim 用 v1 生产值 D=128。STAGES dispatch 把 Sdecay 放在 baseline 前面，`--stage all` 优先跑生产默认。

### 2026-04-28 — stage #2a causal_conv1d_update_indexed 落地（融合 conv_state pool 寻址）

- **交付**：五件套 + docs/causal_conv1d_update_indexed.md，38 tests PASS（28 vs REF 含 has_bias × use_silu 全组合 + 4 strict parity vs `gather → update → scatter` + 1 sequential decode + 1 preallocated + 4 rejection）；全量 mamba regression **293 PASS**。
- **融合定位**：`causal_conv1d_update` 的 indexed-pool 变体——直接吃全局 `conv_state [N_pool, C, K-1]` bf16 + `cache_indices [N] i32`，省掉调用者侧 `cache_index_gather → update → cache_index_scatter` 的 2 次额外 launch + 1 次 `B×C×(K-1)×2B` DRAM 来回（KDA 典型: B=8, C=4096, K=4 → 192 KB / launch；q/k/v 三份 → **576 KB / layer / step**）。conv1d_update 的外层 batch 循环天然就是"进新 batch 元素 → load state → 跑 → store state"，gather/scatter 的紧凑中间布局对 kernel 内部毫无价值。
- **与非 indexed 版的全部差异**：仅外层 batch 循环开头多 1 行 `slot = cache_indices[n_block]` + conv_state advance 首维从 `n_start` 改为 `slot`。中间 7 步（state cols accumulate + fused left-shift + x·w_last + activation + store out + store last col）一字未改。sim.c 与非 indexed 同构，只是 state 行偏移用 `slot * C * Ks + ...` 替原 `n * C * Ks + ...`。
- **BLOCK_N 强制为 1**：indexed 路径下 `[BLOCK_N, *]` tile 内的 n 们各落不同 slot，无法用 contiguous block_ptr advance；BLOCK_N=1 让每个 outer 迭代处理 1 个 batch 元素，slot 在 C 循环内为常量。N 通常 ≤ 32，外层串行不是瓶颈，C 维 tile 才是真正并行轴。
- **Zeus 接口**：`sgl_kernel_zeus.causal_conv1d_update_indexed(x, conv_state, cache_indices, weight, bias=None, activation=None, out=None)`；返回 `out`，pool 已原地更新。
- **Strict parity 验收**：同输入分别走 `gather → update → scatter` 三步链 vs `_indexed` 一步直达，output 与最终 pool **bit-exact 相等**（atol=rtol=0），4 组随机 shape × scattered indices 全 PASS。
- **正确性验收**：7 组 (N, C, K, indices, N_pool) 组合 × `(has_bias, use_silu)` 4 组 = 28 vs pure-torch REF；同时验证 pool 中"未被命中"的 slot **bit-exact 不变**（in-place 副作用边界）。
- **与 CUDA 形态对齐**：CUDA `_causal_conv1d_update_kernel` 本来就有 `IS_CONTINUOUS_BATCHING` + `conv_state_indices_ptr` 分支；本变体 = Zeus 上的形态对齐。
- **后续可融**：`causal_conv1d_update_qkv` 的 indexed 形态（三份 conv_state 共享同一组 cache_indices）；`causal_conv1d_fn` (extend) 也走同款融合，CUDA 已是这种形态。
- 详见 `sgl-kernel-zeus/docs/causal_conv1d_update_indexed.md`。

### 2026-04-28 — stage #6 fused_recurrent_kda_Sdecay_indexed 落地（融合 ssm pool 寻址）

- **交付**：五件套 + docs/fused_recurrent_kda_Sdecay_indexed.md，17 tests PASS（8 vs REF + 4 strict parity vs `gather → Sdecay → scatter` + 1 preallocated + 4 rejection）；全量 mamba regression 255 PASS。
- **融合定位**：`fused_recurrent_kda_Sdecay` 的 indexed-pool 变体——直接吃全局 `ssm_states [N_pool, H, K, V]` fp32 + `cache_indices [N] i32`，省掉调用者侧 `cache_index_gather → Sdecay → cache_index_scatter` 的 2 次额外 launch + 1 次 `B×H×K×V×4B` DRAM 来回（`B=8, H=32, K=V=128` → 16 MB / layer / step）。recurrent kernel 的请求外循环天然就是 "进新 request → load S → 跑 → store S"，gather/scatter 提供的紧凑中间布局对 kernel 内部没有任何价值，indexed advance 一处即可消掉两次 IO。
- **与 _Sdecay 的全部差异**：仅请求循环开头多 1 行 `slot = cache_indices[n_idx]` + pair_idx 公式从 `n_idx*H+h` 改为 `slot*H+h`。token loop 内部所有 12 步（decay → S·decay → v_hat → δ → outer-update → o readout → store）一字未改。sim.c 与 _Sdecay 同款，只是 state 行偏移 `(slot * H + h) * K * V` 替原 `(n * H + h) * K * V`。
- **Zeus 接口**：`sgl_kernel_zeus.fused_recurrent_kda_Sdecay_indexed(q, k, v, g, beta, state_pool, cache_indices, cu_seqlens, ...)`；不返回 final_state（pool 已经原地更新，调用者按 slot 自取）。
- **Strict parity 验收**：同一份输入分别走 `gather → Sdecay → scatter` 三步链 vs `Sdecay_indexed` 一步直达，`o` 与最终 pool **bit-exact 相等**（atol=rtol=0）—— 同 sim 算法 + 同标量算术顺序 → 必然位级一致；4 组随机 shape × scattered indices 全 PASS。
- **正确性验收**：8 组 (seq_lens, H, indices, N_pool) 组合 vs pure-torch REF（atol/rtol=2e-2）；同时验证 pool 中"未被命中"的 slot **bit-exact 不变**（in-place 副作用边界）。
- **与 CUDA 形态对齐**：CUDA fla `fused_recurrent_gated_delta_rule_fwd_kernel` 本来就是直接对全局池做 indexed 寻址（stride trick），不会先 gather；本变体 = Zeus 上的形态对齐。
- **后续可融**：把 indices 加进 `chunk_kda` (extend core, 待 porting) 时一开始就支持 indexed 形态；conv1d 类 kernel 加 `conv_state_indices` 是更进一步方向（性价比看 v2 perf profiling）。
- 详见 `sgl-kernel-zeus/docs/fused_recurrent_kda_Sdecay_indexed.md`。

### 2026-04-28 — stage #10 cache_index_gather / cache_index_scatter 落地

- **交付**：双算子五件套全部完成 + docs/cache_index_gather_scatter.md，48 个 pytest PASS（24 gather + 24 scatter）；既有 90 个 mamba 测试 regression 通过。
- **算子定位**：解决 KDA backend 与全局 state pool 之间的 row-level gather / scatter。Linear / Mamba 状态 cache 与传统 KV cache 的关键差异在于"不连续维度只在 slot 维（pool 首维）"——每请求占 1 个固定大小 slot，但 continuous batching 让活跃 batch 的 B 个 slot id 散点（如 `[3, 17, 42, 5]`）。pool 行内是 contiguous 的，故算子主体是 B 次 row-level DMA + 行间按 indices 跳转，**不是** element-level scatter。
- **Zeus 侧接口（v1 bf16 only）**：
  - `sgl_kernel_zeus.cache_index_gather(pool, indices, out=None)`：返回 `[B, *trailing]` bf16
  - `sgl_kernel_zeus.cache_index_scatter(pool, src, indices)`：pool in-place，无返回值
  - 任意 trailing 维由 Python API flatten 成 R；kernel 看到 `[N_pool, R]` / `[B, R]` 二维。indices 固定 int32。
- **关键决策**：
  - **CORE_NUM=2 沿 R 维 ceiling division**（不约束 B，与 l2norm / rms_norm_gated 同款）；R % CORE_NUM 不要求整除，OOB 列由 boundary_check 掩码。
  - **三组 block_ptr 模板**（pool / src-or-out / indices）：indices 也走 `[1, B]` 2-D block_ptr 形态再 reshape 到 scalar，避免 1-D / raw-pointer load 路径下降弱。
  - 双层 for：外层 `R_blocks` 列 tile，内层 `B` 次 indirected row DMA。每次 row 内是大块 contiguous DMA（默认 BLOCK_R=2048 bf16 = 4KB / DMA），落在 Zeus 片上 DMA 引擎舒适区。
  - **纯数据搬运、无算术**：bf16 进 bf16 出，不升 fp32 / 不 RNE，与 l2norm 等带 reduce 的算子有明显差异；测试用 `atol=rtol=0` 严格逐 bit 比对。
  - host 不做 indices 越界 elementwise 检查（生产路径每请求独占 slot，不会越界；查界开销留给上层）。
- **sim.c**：单线程逐 b 行 `memcpy(R * sizeof(uint16_t))`；bf16 数据透传，与 triton 输出位级一致。
- **v2 follow-up**：fp32 变体（用于 ssm_states，CUDA 上 fp32）；可选地把 indexed indirection 内嵌进 `causal_conv1d_update` / `fused_recurrent_kda` 以省一次 launch + 一次 pool 行的 DRAM 来回（CUDA 形态对齐）。
- 详见 `sgl-kernel-zeus/docs/cache_index_gather_scatter.md`。

### 2026-04-28 — stage #8 kimi_delta_attn_decode 端到端跑通（indexed 链路）

- **交付**：`zeus_dev/dev_kimi_linear_attn_test.py::test_kimi_delta_attn_decode` 由"REF-only 骨架 + zeus_todo"升级为完整的 REF-vs-Zeus 对照；用 **indexed 版本** 的 `causal_conv1d_update_indexed` + `fused_recurrent_kda_Sdecay_indexed` 直接对全局 state 池做 slot 寻址，**无需** host 侧 `cache_index_gather`/`cache_index_scatter`。
- **形状（dev proxy）**：N=2 decode batch、N_pool=16、cache_indices=[3, 5]（散点）、H=512 / Hh=8 / Hd=128（v1 强制 K=V=128）/ Hq=1024 / K=4。
- **比对锚点（17 个）全 PASS**：
  - 中间产物：conv_q/k/v_out（max_diff ≈ 1e-3）、g (kda_gate, max_diff ≈ 2e-7)、core_attn_out (max_diff ≈ 1.5e-5)、normed_out (max_diff ≈ 4e-3)、output (o_proj, max_diff ≈ 7.8e-3，e2e 容差 1e-2)
  - touched pool slots（`indices=[3, 5]`）：conv_q/k/v_pool、ssm_pool 共 8 项与 REF 一致（conv pool max_diff=0；ssm pool max_diff ≈ 5e-4，fp32 累积层级）
  - **untouched pool slots：14 slots × 4 pools = 56 项 bit-exact 不变**（`torch.equal` 严格相等，验证 indexed kernel 的 in-place 副作用边界）
- **REF 路径**：CPU 上手动 `pool[indices]` gather → 非 indexed REF kernel chain → `pool[indices] = ...` scatter（与 CUDA backend `forward_decode` 形态一致）。
- **Zeus 路径**：indexed kernel 一步直达——所有 conv_state / ssm_state pool 直接全量传入 kernel，kernel 内部按 `cache_indices[n]` 寻址 → 一次 launch 同时完成 read state + 计算 + 写回 state。
- **跳过 projection**：与各 sub-stage 同款策略，直接从 post-projection 张量起步（`q/k/v_proj_states`、`b_proj_states`、`f_b_states`、`g_b_states`），让两条路径面对的 bf16 bits 完全相同，对比聚焦在 indexed kernel 链路本身。
- **o_proj 收尾在 CPU 上做**：Zeus matmul 当前要求 packed weight，本 stage 不做 packing；统一在 CPU 上跑最后这一步线性变换，让 REF / Zeus 看到相同 matmul 实现。
- **下一步**：把同款融合策略推到 `causal_conv1d_update_qkv_indexed`（一次 launch 同时更新 q/k/v 三份 conv_state，indices 共享）；待 `causal_conv1d_fn` + `chunk_kda` 落地后做 `kimi_delta_attn_extend` 端到端对照。
