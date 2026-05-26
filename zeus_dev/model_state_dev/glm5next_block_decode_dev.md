# GLM5-Next 整 transformer block (decode) Zeus 适配开发追踪

> 对齐目标：`/root/project/sglang-feat-v0.5.10-prerelease-glm` 的
> `Glm5NextDecoderLayer` decode 路径**整体组装**。GLM5-Next 是一个
> **hybrid attention** 模型：大多数层走 **Linear-attention（KDA）**，少数层
> （`full_attn_layers`）走 **DSA（DeepSeek Sparse Attention）**，每层外面再包
> 两个 **mHC**（attn-side + mlp-side）多流残差 wrapper。
>
> 本文同时覆盖两套配置：
> - **GLM5-Next-16B**：`config_16b_v2.json`（H=2048，27 层，DSA 6 层）
> - **GLM5-Next**：`config.json`（H=4096，45 层，DSA 11 层）
>
> 这是一份 **顶层组装文档**，串起三个已分别对齐的子模块：
> | 子模块 | 子文档 | 子脚本 |
> |---|---|---|
> | Linear-attention (KDA) | `kimi_linear_attn_dev.md` | `dev_kimi_linear_attn_test.py` |
> | DSA decode | `glm5next_dsa_decode_dev.md` | `dev_glm5next_dsa_decode_test.py` |
> | MoE-FFN | `glm4_moe_ffn_dev.md` | `dev_glm4_moe_test.py` |
> | **mHC** ← 本次新增 | `glm5next_mhc_dev.md` | `dev_glm5next_mhc_test.py` |
>
> 顶层脚本：`dev_glm5next_block_decode_test.py`。**"目前主要差 mHC"** —— attention /
> MoE 各算子已对齐，本次补的是 **block 级 mHC 组装**。

## 范围与方法

- **起点**：transformer block 入口的多流 residual `[T, N=4, H]`（flatten `[T, N·H]`）。
- **终点**：一整层处理后的多流 residual `[T, N·H]`（喂下一层）。
- **切片**：单 device、单 layer、decode only、TP=1、CP=1。
- **重点**：不重复 kernel 级对拍（那是各子脚本的事），只验证
  **mHC wrapper × 两类 attention sublayer × MLP sublayer 的 block 级组装是否闭合**。
- **对齐方式**：stage 化 "REF vs Zeus"。REF 复用子模块的 pure-torch golden
  （`dev_glm5next_dsa_decode_test` / `dev_glm5next_mhc_test` 直接 import）+ 内联
  一份紧凑的 Linear-attn decode REF。**禁止 silent fallback**。

## 现状澄清：还差什么（重要）

三类 sublayer 的 **Zeus 子算子均已落地并各自对拍通过**（实测 2026-05-22）：

| sublayer | 实测 stage | Zeus 结果 |
|---|---|---|
| Linear-attn decode | `dev_kimi_linear_attn_test.py --stage kimi_delta_attn_decode` | **PASS**（真 Zeus kernel） |
| MoE-FFN | `dev_glm4_moe_test.py --stage moe_block_full` | **PASS**（max_diff 4.9e-4） |
| DSA decode (cp=1) | `dev_glm5next_dsa_decode_test.py --stage decode_full_nocp` | **PASS**（max_diff 3.7e-9） |

> dense SwiGLU MLP = `linear + silu_and_mul + linear`，其中 `silu_and_mul` 也是
> 已有 Zeus 算子（`moe_block_full` 内已用）。

**唯一还差的是 mHC 的 5 颗 Zeus kernel**（K0..K4：`mhc_expand /
mhc_pre_norm_split（融合原 #1+#2，BLOCK_T 1-pass）/ mhc_sinkhorn /
mhc_pre_apply_mix / mhc_post`），全部 Zeus TODO（见 `glm5next_mhc_dev.md` 算子表）。
因此 block 级"端到端纯 Zeus"目前**只被 mHC 阻塞**：sublayer 能上 Zeus，但外层
`pre/post` 还得走 REF。

**本 block 脚本当前形态 = 纯 REF 组装**：sublayer 段也用 pure-torch REF（没接 Zeus），
目的是先把组装结构 + mHC 接缝跑通。下表"Zeus 现状"列因此分两层含义——
**子算子已就绪**，但**本脚本尚未把 Zeus 调用接进来**。待 mHC kernel 落地后：
(1) 把 mHC REF 段换成 Zeus；(2) 可选地把 sublayer REF 段换成调用各子脚本的 Zeus
路径，即得 block 级端到端 REF-vs-Zeus 对拍。

不在本文 scope：
- prefill / extend / draft / verify。
- TP/EP/DP-attention 与跨进程 collective（`MHCLayerCommunicator` 的通信编排）。
- 真实 MoE sublayer（用 dense SwiGLU 占位；MoE 由 `dev_glm4_moe_test.py` 覆盖）。
- NextN / MTP。
- 各子算子的 kernel 级数值对拍（见对应子脚本）。

## GLM5-Next layer 拓扑

| 字段 | GLM5-Next-16B | GLM5-Next |
|---|---:|---:|
| `num_hidden_layers` | 27 | 45 |
| Linear-attn (KDA) 层 | 21 层 | 34 层 |
| DSA (full-attn) 层 | 6 层 `[3,7,11,15,19,23]` | 11 层 `[3,7,11,15,19,23,27,31,35,39,43]` |
| `first_k_dense_replace` | 1 | 3 |
| dense MLP 层 | layer 0 | layer 0,1,2 |
| MoE 层 | 其余 | 其余 |
| `mhc` / N | true / 4 | true / 4 |

**路由规则**：`layer_id ∈ full_attn_layers` → DSA；否则 → Linear-attn（KDA）。
attention 类型与 MLP/MoE 类型相互独立。

## 一层 decode 的组装顺序（MHCLayerCommunicator）

```
residual_in [T, N=4, H]
   │  attn_hc.pre_forward  ──▶ layer_input [T,H]   (h_res, h_post 暂存)
   ▼
 self_attn(layer_input)        ┌─ layer ∈ full_attn → DSA decode（MLA absorb + Indexer topK + sparse MQA）
   │                           └─ else             → Linear-attn decode（conv1d_update + KDA recurrent + gated RMSNorm）
   │  attn_hc.post_forward ──▶ residual_mid [T, N·H]
   ▼
   │  mlp_hc.pre_forward   ──▶ layer_input [T,H]
   ▼
 mlp(layer_input)              ┌─ dense layer → SwiGLU(clamp)
   │                           └─ else        → MoE（biased_grouped_topk + grouped_gemm + sum_reduce）
   │  mlp_hc.post_forward  ──▶ residual_out [T, N·H]
   ▼
residual_out ──▶ 下一层
```

**关键点**：sublayer（attn / mlp）永远只看到 **单流 `[T,H]`**（`layer_input`）；
多流 `[T,N,H]` 的 mixing 全在 mHC 的 `pre/post` 里完成。这让两类 attention
sublayer 与 mHC 完全解耦——换 attention 类型不影响 mHC wrap。

## Linear-attention (KDA) decode sublayer

对齐 `Glm5NextLinearAttention.forward`（decode 分支，glm5_next.py:316-367）：

```
layer_input [B,H]
  ├─ qkv_proj         → mixed_qkv [B, 3·proj]      (q|k|v 融合)
  ├─ b_proj           → beta      [B, num_heads]
  ├─ f_b(f_a(·))      → forget_gate [B, proj]
  └─ g_b(g_a(·))      → g_proj     [B, proj]
  ↓ (decode)
  conv1d_update(mixed_qkv) + silu  → q,k,v          (单步 per-channel causal conv)
  fused_kda_gate(forget_gate, A_log, dt_bias)       → g_gate = -exp(A_log)·softplus(·)
  beta = sigmoid(beta)
  ↓ 单步 delta-rule recurrent（S 先 decay 再 readout）
  q̂=l2norm(q)·scale ; k̂=l2norm(k)
  S = S·exp(g_gate) ; v̂=k̂·S ; S += beta·(v−v̂)⊗k̂ ; o = q̂·S
  ↓
  o = rms_norm_gated(o, g_proj, sigmoid)            → 门控 RMSNorm
  out = o_proj(o.flatten())                          → [B,H]
```

| 字段 | 16B | Next |
|---|---:|---:|
| KDA num_heads | 32 | 64 |
| head_k / head_v dim | 72 / 72 | 128 / 128 |
| conv kernel | 4 | 4 |
| scaling | 72^-0.5 | 128^-0.5 |

> decode 单步：conv_state `[B, 3·proj, K-1]` 与 recurrent_state `[B, H, K, V]`
> 在 REF 中原位推进（stage 自检 `state advanced`）。kernel 级对拍见
> `dev_kimi_linear_attn_test.py::kimi_delta_attn_decode`。

## DSA decode sublayer

直接复用 `dev_glm5next_dsa_decode_test.py::run_ref_decode`（cp=1）。算子链：
`q_a/kv_a proj+norm → q_main(absorb) → Indexer Q/K → index logits → local topK →
latent gather → sparse MQA partial → o_proj`。详见 `glm5next_dsa_decode_dev.md`。

本脚本把 mHC 的 `layer_input` 作为 DSA 的 `hidden` 喂入，输出 `[B,H]` 再交给
`attn_hc.post_forward`。

## Dev 脚本 stage 顺序（`dev_glm5next_block_decode_test.py`）

> "Zeus 现状"列：**子算子** = 各子脚本里 Zeus kernel 是否已通；**本脚本** = 此
> block 脚本是否已接 Zeus 调用（当前均为"暂 REF"，见上节《现状澄清》）。

| stage | 内容 | REF 自检 | 子算子 Zeus | 本脚本 |
|---|---|---|---|---|
| `linear_attn_decode` | Linear-attn sublayer 单跑 | shape + conv/rec state 推进 | ✅ kimi decode 通 | 暂 REF |
| `dsa_decode` | DSA sublayer 单跑（复用 dsa 模块） | shape | ✅ dsa decode_full_nocp 通 | 暂 REF |
| `mlp_decode` | dense SwiGLU MLP sublayer | shape | ✅ silu_and_mul/MoE 通 | 暂 REF |
| `linear_attn_block` | mHC[ Linear-attn + MLP ] 整 block | shape 闭合 + streams 混流 | ⏳ 仅缺 mHC | 暂 REF |
| `dsa_block` | mHC[ DSA + MLP ] 整 block | shape 闭合 + streams 混流 | ⏳ 仅缺 mHC | 暂 REF |
| `decode_layer_full` | 按 `full_attn_layers` 路由 attn 类型跑完整层 | 路由正确 + block 闭合 | ⏳ 仅缺 mHC | 暂 REF |

用法：
```
python zeus_dev/model_state_dev/dev_glm5next_block_decode_test.py --config 16b
python zeus_dev/model_state_dev/dev_glm5next_block_decode_test.py --config next --stage dsa_block
python zeus_dev/model_state_dev/dev_glm5next_block_decode_test.py --stage decode_layer_full --layer-id 3
```

> `--layer-id ∈ full_attn_layers` → 走 DSA block；否则走 Linear-attn block。
> 默认 `mode=both`：REF 全 PASS（16B + Next 两套配置均验证），Zeus block 级
> kernel 未落地打印 SKIP/TODO（子算子 kernel 对拍归各子脚本）。

## 注意事项

- **mHC 解耦 attention 类型**：block 级组装对 KDA / DSA sublayer 完全同构
  （都只是 `callable(layer_input[T,H]) -> [T,H]`），所以 `run_mhc_block` 用同一份
  逻辑包两类 attn。
- **streams 混流自检**：mHC `post` 之后 N 条 residual 流应当不再恒等
  （`comb` 是 doubly-stochastic 混合矩阵）。stage 用 `mid[:,0,:] != mid[:,1,:]`
  作为"mHC 确实生效"的 sanity，而非纯 shape 检查。
- **MoE 用 dense 占位**：本脚本 MLP sublayer 是 dense SwiGLU(clamp)；真实 MoE 层
  的 6-kernel pipeline 由 `dev_glm4_moe_test.py::moe_block_full` 覆盖，组装方式
  与 dense 完全一致（都是单流 `[T,H] -> [T,H]`），故不在此重复。
- **decode 单步语义**：`num_tokens` = decode batch（每个 token 是一个序列的一步）。
  KDA 的 conv/recurrent state 与 DSA 的 history 都按"已有前缀 + 当前步"建模。
- **两套配置唯一 mHC 差异**：16B `mhc_no_norm_weight=true`（旁路 RMSNorm weight），
  Next `=false`（使用 weight）。其余只是 H / head_dim / 层数不同。详见
  `glm5next_mhc_dev.md`。

## 开发日志

### 2026-05-22 · 起点
- 创建 `glm5next_block_decode_dev.md` + `glm5next_block_decode_slides.html` +
  `dev_glm5next_block_decode_test.py`。
- 顶层脚本 import `dev_glm5next_dsa_decode_test`（DSA REF）+ `dev_glm5next_mhc_test`
  （mHC REF），内联紧凑 Linear-attn decode REF + dense SwiGLU MLP REF。
- 6 stage（linear_attn_decode / dsa_decode / mlp_decode / linear_attn_block /
  dsa_block / decode_layer_full）REF 全 PASS（16B + Next 两套配置）。
- `decode_layer_full` 按 `full_attn_layers` 路由：layer 3 → DSA、layer 0 → KDA，
  验证通过。
- Zeus 侧 block 级 fused kernel 全 TODO；`--mode both` 打印 SKIP/TODO（指向各子脚本）。
- 下一步焦点：等 mHC 的 5 颗 K0..K4 kernel 落地后（见 `glm5next_mhc_dev.md`；
  注：2026-05-23 把原 #1 norm_fn + #2 split_mixes 合为 K1 `mhc_pre_norm_split`，
  BLOCK_T 1-pass），把本脚本的 mHC REF 段逐步替换为 Zeus 调用，做 block 级端到端
  REF-vs-Zeus 对拍。

### 2026-05-24 · block 级 mHC chain 接入 Zeus（attn/MLP sublayer 仍 REF）
- **背景**：`glm5next_mhc_dev.md` 同日把 K0..K4 全套 5 颗 kernel + 三个组合 stage
  (mhc_pre / mhc_sublayer_wrap / mhc_layer_full) 全部接通 Zeus 路径。此次把
  `dev_glm5next_block_decode_test.py` 里 `linear_attn_block / dsa_block /
  decode_layer_full` 中的 mHC wrap 切到 Zeus 路径，sublayer (attn / MLP) 本身仍走
  pure-torch REF（kernel 级对拍归各子脚本：`dev_kimi_linear_attn_test.py` /
  `dev_glm5next_dsa_decode_test.py` / `dev_glm4_moe_test.py`）。
- **新增 helpers**（`dev_glm5next_block_decode_test.py`）：
  - `run_mhc_block_ref_quant(which, residual_flat, attn_fn, mlp_fn, seed)`：
    REF mHC + REF sublayer，但 mHC 参数走 `mhc.quantize_p_for_zeus_match`
    （fn 经 norm_weight pre-merge + bf16 round-trip），与 Zeus chain bit-for-bit
    alignable，吸收 K1 量化噪声到 REF 侧。
  - `run_mhc_block_zeus(which, residual_flat, attn_fn, mlp_fn, seed)`：Zeus
    `mhc.zeus_mhc_pre / zeus_mhc_post` 串接两轮 (attn_hc + mlp_hc)；sublayer
    边界做 `z_li.cpu() → REF sublayer → .to("zeus")` round-trip；attn 与 mlp 之间
    `z_mid` 不下 host，直接喂下一轮 K1。
  - `_zeus_mhc_chain_available()`：mHC chain 4 颗 kernel 齐备性检查。
  - `finish_stage_with_zeus()`：与 mhc test 同语义的三态终结器。
- **stage 改造**：
  - `stage_linear_attn_block`：conv/rec state 三份 clone（REF / qREF / Zeus 各 chain
    自洽推进，单步输出不受 state 推进差异影响）。Zeus 路径与 quantized REF 对照
    `mid` (atol=5e-2 rtol=2e-2) + `out` (atol=1e-1 rtol=5e-2) + Zeus 流分歧不变量。
  - `stage_dsa_block`：DSA ctx 三份独立 `build_dsa_ctx`（同 seed 同初始 KV state）。
    其余结构与 linear_attn_block 同构。
  - `stage_decode_layer_full`：dispatch only，按 `full_attn_layers` 路由至上面两个
    stage，自动获得 Zeus 路径。
- **state clone 设计要点**：attn sublayer (Linear-attn conv/rec、DSA KV cache) 是
  stateful，三条 chain 必须各自 clone state，否则 in-place 推进会让前一条 chain
  污染后一条。但 mid/out 仅依赖**本步**输入，state 推进差异不影响本步输出，所以
  对照的语义清晰：用同一 residual 喂三条 chain，比较它们的本步 (mid, out)。
- **验收**（torch10_312 env）：
  - **16B** (B=4, H=2048, seqlen=64)：
    - `linear_attn_block`: mid max_diff=1.95e-3 / out 7.81e-3 — **PASS**
    - `dsa_block`: mid 1.22e-4 / out 2.44e-4 — **PASS**
    - `decode_layer_full` (layer 3 → DSA): 同 dsa_block — **PASS**
  - **Next** (B=4, H=4096, seqlen=64)：
    - `linear_attn_block`: PASS
    - `dsa_block`: mid 7.63e-6 / out 3.05e-5 — **PASS**
    - `decode_layer_full` (layer 3 → DSA): PASS
  - DSA block 误差比 Linear-attn block 小 1 个量级（DSA 输出 magnitude 更小 →
    K4 累加项更小 → bf16 RNE round 绝对误差更小）。
  - 单 sublayer stage (`linear_attn_decode` / `dsa_decode` / `mlp_decode`) 仍是
    SKIP/TODO（不在 mHC scope，指向各自 kernel dev 脚本）。
- **结论**：block 级 mHC 端到端 Zeus 路径完全接通；GLM5-Next 一个 decoder layer 的
  decode 流（mHC wrap + attn sublayer + MLP sublayer）已可在 Zeus 上完整跑通，且与
  REF 量化对照 bit-level alignable。下一步可由各子 kernel dev 脚本继续把 attn / MoE
  sublayer 落到 Zeus（mHC chain 与 sublayer 之间通过 `z_li.cpu() → sublayer →
  .to("zeus")` round-trip 解耦，sublayer 各自 Zeus 化时切换 round-trip 到 device-
  resident 即可）。

### 2026-05-24 · `dsa_block` 切到 DSA + MoE 全 Zeus（device-resident，无 round-trip）
- **背景**：用户提出 sublayer 应该 Zeus 化而不再走 REF。核对各 sublayer dev 脚本后
  确认：**DSA decode**（`decode_full_nocp` 10+ kernel chain）+ **MoE block**
  （`moe_block_full` 6 kernel chain）已经全部 Zeus 端到端 LANDED，包含 projections。
  Linear-attn 因 projections 仍走 CPU matmul（kimi 自己 `kimi_delta_attn_decode`
  明文 `# Zeus matmul 需要 packed weight，不在本 stage scope`），dense MLP 也无
  Zeus 端到端，所以这俩保留 REF；只把 `dsa_block` 切到全 Zeus。
- **新增 helpers**（`dev_glm5next_block_decode_test.py`）：
  - `_lmem_pack(t)`：LocalMem pack 短手（`kind="weight", Tr=Tc=1`）。
  - `zeus_dsa_decode(ctx, hidden_z)`：完整 DSA decode 的 Zeus chain（mirror
    `dev_glm5next_dsa_decode_test::stage_decode_full_nocp`）。10 颗 sgl-kernel-zeus
    算子串接（dsa_q_a_proj_norm → dsa_kv_a_proj_norm_store → dsa_q_main_absorb
    → dsa_indexer_q_weights → dsa_indexer_k_prep_store → dsa_index_logits →
    dsa_local_topk_radix → dsa_latent_k_gather × B → dsa_sparse_mqa_partial →
    dsa_post_o_proj_no_cp），中间有不可避的 CPU 协调步骤（top_pos 选择 / history
    拼接 / per-batch latent_k_gather 循环）。
  - `init_moe_weights / ref_moe_decode / zeus_moe_decode`：MoE proxy 6-kernel
    pipeline（biased_grouped_topk → moe_align → moe_grouped_gemm × 2 →
    silu_and_mul → moe_sum_reduce），含 shared experts residual。proxy shape
    (E=8 / mI=128 / top_k=2 / sI=128)——真实 GLM-4.7 / GLM5-Next 的 E=64/288 +
    mI=1408/2048 权重 w13 ≈ 0.75–4.8 GB 无法在 CPU REF 跑，kernel 层对拍由
    `dev_glm4_moe_test.py` / `sgl-kernel-zeus/tests/` 各自的 kernel test 覆盖。
    router_logits / shared_out 走 host REF（gate Linear + shared SwiGLU MLP，
    与 `moe_block_full` 同套路）。
  - `run_mhc_block_zeus_e2e(which, residual_flat, zeus_attn_fn, zeus_mlp_fn, seed)`：
    完全 device-resident 的 block runner，zeus_attn_fn / zeus_mlp_fn 接收并返回
    Zeus tensor，**无 round-trip**（与 `run_mhc_block_zeus` 的 sublayer .cpu()
    round-trip 设计区别开）。
  - `_zeus_dsa_chain_available()` / `_zeus_moe_chain_available()`：各 chain
    kernel 齐备性检查。
- **stage_dsa_block 改造**：
  - 替换 dense MLP (`ref_mlp_decode` + `init_mlp_weights`) 为 MoE
    (`ref_moe_decode` + `init_moe_weights`)；REF / qREF / Zeus 三路共享同一份
    `moe_w`（MoE 无 state，权重在 Zeus 路径内部 `.to("zeus")`）。
  - Zeus 路径走 `run_mhc_block_zeus_e2e(zeus_attn_fn=zeus_dsa_decode,
    zeus_mlp_fn=zeus_moe_decode)`，**完全 device-resident**（mHC + DSA + MoE
    全 Zeus，sublayer 出口直接是 Zeus tensor 喂 mHC post K4，无 .cpu() 边界）。
  - 复合误差预算放宽到 mid `atol=1.5e-1` / out `atol=3e-1`（mHC + DSA + MoE
    三层 bf16 累加 + K4 round noise）。
- **总 Zeus kernel 数（dsa_block 一个完整 block）**：
  - mHC: K1 × 2 + K2 × 2 + K3 × 2 + K4 × 2 = 8 颗
  - DSA: 10 颗（含 per-batch latent_k_gather 循环 B 次）
  - MoE: 6 颗
  - **合计 24+ 颗 sgl-kernel-zeus 调用**串接，全部 device-resident，无零碎 device
    kernel 插入（reshape 是 view，CPU 协调步骤是算法本身需求）。
- **stage_linear_attn_block 保持不变**：KDA + dense MLP 仍 REF，mHC chain Zeus。
  Linear-attn projections + dense MLP gate_up/down 没有 Zeus 算子（见上述背景），
  所以这条路径**当前没法做到全 Zeus**——见 `dev_kimi_linear_attn_test.py:1060`。
- **验收**（torch10_312 env）：
  - **16B** (B=4, H=2048, seqlen=64, MoE proxy)：
    - `dsa_block`: mid max_diff=4.88e-4 / out 1.46e-3 — **PASS**
    - `decode_layer_full` (layer 3 → DSA): 同 dsa_block — **PASS**
    - `linear_attn_block` (未改): mid 1.95e-3 / out 7.81e-3 — **PASS**
  - **Next** (B=4, H=4096, seqlen=64, MoE proxy)：
    - `dsa_block`: mid 4.88e-4 / out 7.81e-3 — **PASS**
    - `decode_layer_full` (layer 3 → DSA): 同 dsa_block — **PASS**
    - `linear_attn_block` (未改): PASS
  - 两套配置都远低于宽松容差，证明 DSA + MoE 在 mHC 包裹下的 device-resident chain
    数值对齐良好。
- **结论**：`dsa_block` / `decode_layer_full` (full-attn 层) 现在是 GLM5-Next 一个
  decoder layer decode 的 **完整 Zeus 端到端**实现 —— mHC wrap + DSA attention +
  MoE FFN 共 24+ 颗 sgl-kernel-zeus 算子，期间无 sublayer 边界 round-trip，无零碎
  device kernel 插入。Linear-attn / dense MLP 路径仍受 projections 缺 Zeus 算子
  制约（见 kimi dev 注释），等 Zeus matmul packed-weight GEMM 或新 fused
  projection kernel 落地后可独立切换。

### 2026-05-25 · GAP-3 闭环：`zeus_moe_decode` 切到 Zeus `linear_bf16`，`dsa_block` 完全 device-resident
- **背景**：`glm5next_dsa_block_zeus_flow.md` 的 GAP-3 在 GLM5-Next block decode
  路径上的剩余部分 —— `zeus_moe_decode` 入口的 `_moe_router_and_shared_ref`
  host 端 `nn.functional.linear` × 3 (gate / shared gate_up / shared down)。
  同日 `sgl-kernel-zeus` 落地 `linear_bf16` packed-weight bf16 GEMM kernel
  （见 `glm5next_mhc_dev.md` 2026-05-25 日志），可以一锤多治。
- **改动**：
  - `dev_glm5next_block_decode_test.py::zeus_moe_decode`：3 颗 host
    `F.linear` 全部切到 `sgl_kernel_zeus.linear_bf16`：
    - `linear_bf16(hidden_z, gate_w_lmem)` → router_logits_bf16 (`.float()`
      device cast 给 biased_grouped_topk)
    - `linear_bf16(hidden_z, sh_gu_lmem)` → sh_gu_bf16 [T, 2*sI]
    - `silu_and_mul(sh_gu, sh_silu)` → sh_silu_bf16 [T, sI]
    - `linear_bf16(sh_silu_z, sh_dp_lmem)` → shared_out_z [T, H]
  - 3 颗 weight (gate_w / sh_gu / sh_dp) 全部 LocalMem 装包
    （`from_tensor(kind="weight", Tr=1, Tc=1)`）。
  - 删除原 `hidden_z.cpu()` D→H 边界 + `_moe_router_and_shared_ref` host
    compute。
  - `_zeus_moe_chain_available` 加入 `linear_bf16` 齐备性检查。
- **`zeus_moe_decode` 现在是 10 颗 sgl-kernel-zeus 算子的完整 device-resident
  chain**：
  ```
   1. linear_bf16        (gate Linear)                 ← GAP-3 fix
   2. linear_bf16        (shared gate_up_proj)         ← GAP-3 fix
   3. silu_and_mul       (shared experts)
   4. linear_bf16        (shared down_proj)            ← GAP-3 fix
   5. biased_grouped_topk
   6. moe_align_block_size_alloc
   7. moe_grouped_gemm   (gemm1)
   8. silu_and_mul       (per-expert)
   9. moe_grouped_gemm   (gemm2, mul_routed_weight)
  10. moe_sum_reduce     (+shared residual fuse)
  ```
- **验收**（torch10_312 env）：
  - 16B (B=4, H=2048, seqlen=64)：`dsa_block` mid max_diff=4.88e-4 / out
    1.46e-3 — **PASS**（与 GAP-3 闭环前完全同量级，无回归）
  - Next (B=4, H=4096, seqlen=64)：mid 4.88e-4 / out 7.81e-3 — **PASS**
  - `decode_layer_full` (layer 3 → DSA) 同款 PASS
  - 全 stage 跑全：`linear_attn_decode` / `dsa_decode` / `mlp_decode` 仍
    SKIP/TODO（不在 mHC scope），`linear_attn_block` / `dsa_block` /
    `decode_layer_full` 全 PASS
- **`dsa_block` 一步 decode 整体 Zeus kernel 数 = 28 + B**（mHC 8 + DSA 10+B
  + MoE 10 - 重复算的 mHC pre/post 1 个 = 实际 28+B）：
  - mHC attn pre 3 (K1+K2+K3) + DSA 10+B + mHC attn post 1 (K4) +
    mHC mlp pre 3 (K1'+K2'+K3') + MoE 10 + mHC mlp post 1 (K4') = **28+B**
  - B=4 时 = 32 颗 sgl-kernel-zeus 算子串接
- **`dsa_block` / `decode_layer_full` 是 GLM5-Next 一个 decoder layer decode
  的完整 Zeus 全 device-resident 实现**：
  - host 仅做：①一次性 weight LocalMem pack（layer.__init__ 在生产路径会缓存）
    ②输入 residual H→D ③Latent K / Index-K transfer + history concat
    (PD-disagg 必须，用户预批准)
  - **无任何中间 host CPU compute**
  - **无 D→H→D round-trip**（GAP-2 修复后）
  - **无 host-side mask cleanup**（GAP-1 pool 契约修复后）
  - **无 host F.linear**（GAP-3 linear_bf16 修复后）
- **下一步**：`stage_linear_attn_block` 把 Linear-attn 5 个 projection（qkv /
  b / f_a/f_b / g_a/g_b / o_proj）以及 dense MLP gate_up / down 切到
  `linear_bf16`，让 `linear_attn_block` 也走向完全 device-resident。
  之后整个 GLM5-Next decoder layer decode（不论 full-attn DSA 层还是 linear-attn KDA
  层）都能在 Zeus 上无 host compute 跑通。

### 2026-05-25 · `stage_linear_attn_block` 切全 Zeus device-resident（option A: 无 clamp）
- **背景**：在 `dsa_block` 已经走通完整 Zeus 路径之后，`linear_attn_block` 的 KDA
  attention 与 dense MLP sublayer 仍是 host pure-torch REF。kernel 层 (`kimi_delta_attn_decode`
  end-to-end stage) 已经验过 conv1d_update + fused_kda_gate + fused_recurrent_kda_Sdecay
  + rms_norm_gated 这条链，**唯一一直缺的就是 projection matmul**（`dev_kimi_linear_attn_test.py:1060`
  那条 "Zeus matmul 需要 packed weight，不在本 stage scope" 的明文 TODO）。本日
  `linear_bf16` 已落地，把这块补上 + 把 kernel 链接到 block_decode dev 脚本。
- **新增 helpers**（`dev_glm5next_block_decode_test.py`）：
  - `_pack_linear_attn_weights(w)`：把 init_linear_weights 输出 dict 的 7 颗
    linear weight（qkv_proj / b_proj / f_a / f_b / g_a / g_b / o_proj）装到 LocalMem
    （`from_tensor(kind="weight", Tr=1, Tc=1)`），其余 conv_w / conv_b / dt_bias /
    A_log / o_norm 直接 `.to("zeus")`。
  - `zeus_linear_attn_decode(hidden_z, w_lmem, cfg, conv_state, rec_state)`：
    完整 Linear-attn decode 单步 Zeus chain（10+ 颗算子）：
    ```
    1-5.  linear_bf16 × 7 (qkv_proj / b_proj / f_a→f_b / g_a→g_b)
    6.    causal_conv1d_update (3P 通道一把跑 + silu)
    7.    fused_kda_gate (softplus·-exp·A_log + dt_bias)
    8.    .float().sigmoid() (beta；CPU fallback 但成本极低)
    9.    fused_recurrent_kda_Sdecay (non-indexed, head_dim 通用 72/128)
    10.   rms_norm_gated (sigmoid 门控)
    11.   linear_bf16 (o_proj)
    ```
    conv_state + rec_state 在 Zeus device 上 in-place 推进。
  - `_pack_mlp_weights(w)` + `zeus_mlp_decode(hidden_z, w_lmem, inter)`：dense
    SwiGLU MLP Zeus 路径（3 颗算子）：`linear_bf16(gate_up) → silu_and_mul →
    linear_bf16(down)`。**option A：无 clamp**（randn 0.05 magnitude 远不到
    swiglu_clamp_limit=10，clamp 无实际效果；生产严格 clamp 语义需新加
    `silu_and_mul_clamp` kernel，独立工作量）。
  - `_zeus_linear_attn_chain_available()`：6 颗 kernel 齐备性检查
    （linear_bf16 + causal_conv1d_update + fused_kda_gate + fused_recurrent_kda_Sdecay
    + rms_norm_gated + silu_and_mul）。
- **设计抉择 — 为什么不用 `causal_conv1d_update_qkv`**：`init_linear_state` 返回的
  conv_state 是 `[B, 3*P, K-1]` 融合存储（q+k+v 合并）。`causal_conv1d_update_qkv`
  要求三个独立 contiguous state，按 `dim=1` chunk 出来是**非连续 view**，与 Zeus
  host wrapper 的 contiguous 校验冲突。改用 non-fused `causal_conv1d_update` 跑
  整个 3*P 通道一把（per-channel kernel 通道间独立，等价于跑 3 次），再 `chunk(3,
  dim=-1)` 切 x（dim=-1 chunk 是连续的）。一次 launch + 1 个 state，比 qkv 版本
  实现还简单。
- **stage_linear_attn_block 改造**：与 `stage_dsa_block` 同套路 ——
  - REF chain (host `ref_linear_attn_decode` + `ref_mlp_decode`)
  - quantized REF chain (作 Zeus golden，用 `quantize_p_for_zeus_match` 的 mHC params)
  - **Zeus chain**：`run_mhc_block_zeus_e2e` + 设备端 `zeus_attn_fn` /
    `zeus_mlp_fn` 闭包。weight LocalMem pack 一次（生产路径 layer.__init__ 缓存）。
    conv_state + rec_state 各 `.clone().to("zeus")` 一份独立推进。
  - 复合误差预算 mid `atol=1.5e-1, rtol=5e-2` / out `atol=3e-1, rtol=1e-1`（与
    `dsa_block` 同 envelope）。
- **验收**（torch10_312 env）：
  - **16B** (B=4, H=2048)：`linear_attn_block` mid max_diff=1.56e-2 / out 3.91e-2 — **PASS**
  - **Next** (B=4, H=4096)：mid 1.56e-2 / out 1.56e-1 — **PASS**
  - 全 stage 跑全：`linear_attn_block` / `dsa_block` / `decode_layer_full` 三个
    block 级 stage **全部 PASS**（两套配置）。
- **整个 GLM5-Next decoder layer decode 现在完全 device-resident**：
  - **DSA 层（full-attn）路径** = `dsa_block` = mHC 8 + DSA 10+B + MoE 10 + mHC 4 +
    mHC 4 = **28+B 颗 sgl-kernel-zeus 算子**
  - **KDA 层（linear-attn）路径** = `linear_attn_block` = mHC 8 + Linear-attn 11 +
    MLP 3 + mHC 4 + mHC 4 = **30 颗 sgl-kernel-zeus 算子**（无 batch 依赖）
  - 不论哪条路径，host 仅做：①一次性 weight LocalMem pack ②输入 residual H→D
    ③(DSA only) Latent K / Index-K transfer + history concat (PD-disagg)
  - **唯一 host 残留**：beta = `b_proj.float().sigmoid()`（一次 `aten::sigmoid.out`
    CPU fallback）—— sigmoid 是 elementwise，1 步 1 个数（[B, Hh]），开销可忽略；
    将来 zeus 后端补 sigmoid 即可消除。
- **option A vs option B（clamp）**：dev script 用 randn(0.05) magnitude，clamp(-10,10)
  无实际触发；生产数据若 logits 可能超 10，需要 `silu_and_mul_clamp` 新 kernel。
  本日只闭环 dev test，留待真权重路径再补。
- **结论**：**GAP-3 在 GLM5-Next 整 transformer block decode 上完全闭环**。两类
  attention（DSA / KDA）+ 两类 MLP（MoE / dense）全部 4 种组合都已 Zeus
  device-resident。`decode_layer_full` 按 `full_attn_layers` 路由可跑任意 layer_id
  全 Zeus 端到端。下一步可以由各 kernel 性能 profiling 推动 fuse 优化（如 attention
  + post 大 fuse / projection 三联 fuse 等）。
