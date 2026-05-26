# GLM5-Next DSA Decode (CP / Route B) Zeus 适配开发追踪（V7）

> 对齐目标：GLM5-Next 的 DSA 全注意力子层（`Glm5NextDecoderLayer.self_attn = DeepseekV2AttentionMLA`，
> 见 REPO `python/sglang/srt/models/glm5_next.py`），把它的 **decode 路径在多卡
> Context-Parallel + "Route B"（KV 按 token 位置分片、partial sparse MQA + online-softmax merge）下**
> 在纯 Zeus（无 CUDA）环境里逐算子跑通。本文档记录这条路径的 porting 工作。
>
> 校对依据：`/root/project/sglang-feat-v0.5.10-prerelease-glm`（以下 **REPO**）。同目录另两份文档
> （V5 单卡完整 / V6 决策与推导）作为附录式背景，本文聚焦"工程实现"。

## 范围与方法

- **起点**：`hidden_new [B, H]` 进 `layer_communicator.prepare_attn(...)`（B = decode batch；
  每 request 1 个新 token，位于序列位置 `pos(req)`）。
- **终点**：`o_proj` 输出 `out [B, H]`，可直接喂 layer communicator 的 post-attn / MHC、再到下层。
- **切片**：单 DSA 层（layer_id ∈ `full_attn_layers`，6 层 / 11 层）、`use_mha=False`（decode 永远走
  absorb，不走 `MHA_ONE_SHOT` dense fallback）。**只覆盖 decode**；prefill / extend / KDA / MoE-FFN
  / TP / PP / EP / NextN / speculative 都不在本切片。
- **CP 模型**：`cp_size` 张卡（举例 16）；**latent KV 按 token 位置 round-robin 分片**到各卡
  （`position p → rank r = p mod cp_size`，rank-local 槽位 ≈ `p // cp_size`，实际经 paged page-table
  映射，`page_size=64`）；**index K 每卡复制全量**（每 token 132 B，相对 latent KV 1152 B 只占
  ~10%，换 top-k 全本地、零通信 —— 决策见附录 C）。新 token 的 owner（latent KV 视角）=
  `pos(req) mod cp_size`；index K 的"owner"= 所有卡（all-append）。本文档不叠 TP；叠 TP 是正交
  的另一层 all-reduce，附录 B 简注。
- **路由选择**：decode 选 **Route B**（partial sparse MQA + LSE merge）。理由见附录 A：MLA-absorb 下
  key 单 head（~1.15KB/token）、query/output Nh 头（~33–37KB/token），decode 只 1 query token，
  搬 query / reduce output 都极便宜；Route A（gather KV）反而是 ~2.4MB/层/step 的不规则 all-to-all。
- **fused stage 划分**：按"数据依赖 + 通信边界 + 并行机会"分成 **7 个 fused stage**（见下表）。
  D0..D1 合并成 F-PRE；D4.5+D5+D6 合并成 F-MERGE-POST；中间三条并行支线（F-Q-MAIN /
  F-KV-STORE / F-IDX）跑 `alt_streams`。
- **对齐方式**：沿用 `dev_glm4_moe_test.py` / `dev_kimi_linear_attn_test.py` 的 "REF vs Zeus" 模式
  —— 同一份权重、同一份输入，REF（pure-torch，必要时落 CUDA Triton）产生 golden，Zeus 侧跑完后
  逐算子 `compare_tensors`。**Zeus 路径禁止落到 torchnative fallback**，必须对齐到 `sgl_kernel_zeus`
  的 kernel 调用。**正确性硬判据**：LSE merge 是 exact，所以 `--cp 1` / `--cp 4` / `--cp 16` 的端到端
  输出必须**逐元素一致到浮点精度**——是检测 partial / merge / 分片实现 bug 最直接的方式。
- **dev 脚本**：`zeus_dev/dev_glm_moe_dsa_decode_test_v7.py`，stage 化，按 `--stage` 单独跑，
  `--cp` 指定 CP 度数。未实现的 Zeus stage 显式 raise `NotImplementedError` 并标 TODO，
  **不要 silent fallback**。

## GLM5-Next 两个配置（来自 `zeus_dev/model_state_dev/`）

REPO 的 `is_deepseek_nsa()` 白名单已含 `"Glm5NextForCausalLM"`，触发条件 `index_topk is not None`
（GLM5 = 2048）⇒ 启用 NSA backend。两个 config 的 **DSA cache 形态完全相同**（latent KV
`Rkv+Dro = 576`、index K 132 B/token、Ktop=2048），区别只在 projection 头数 / hidden 维 / DSA 层数。

| 字段 | `config_16b.json`（小） | `config.json`（大） |
|---|---:|---:|
| `architectures` | `["Glm5NextForCausalLM"]` | `["Glm5NextForCausalLM"]` |
| `hidden_size` (H) | 2048 | 4096 |
| `num_attention_heads` (Nh) | 32 | 64 |
| `q_lora_rank` (Rq) | 768 | 1536 |
| **`kv_lora_rank` (Rkv)** | **512** | **512** |
| `qk_nope_head_dim` (Dnope) | 128 | 192 |
| **`qk_rope_head_dim` (Dro)** | **64** | **64** |
| `qk_head_dim` (Dqk) | 192 | 256 |
| `v_head_dim` (Dv) | 128 | 256 |
| **`index_head_dim` (Di)** | **128** | **128** |
| `index_n_heads` (I) | 8 | 8 |
| **`index_topk` (Ktop)** | **2048** | **2048** |
| `index_dsa_use_layernorm` | true | true |
| `mla` / `mla_nope` | true / **true** | true / **true** |
| `multi_query_attention` | true | true |
| `mhc` / `mhc_num_residual_streams` | true / 4 | true / 4 |
| `num_hidden_layers` | 27 | 45 |
| `linear_attn_config.full_attn_layers` | `[3,7,11,15,19,23]` | `[3,7,11,…,43]` |
| **DSA 层数** | **6** | **11** |
| `num_key_value_heads` | 8（MLA 不用，仅 GQA 路径字段；与 `index_n_heads=8` 数值相同纯属巧合） | 64（同上） |
| `rope_theta` | 10000（仅 indexer 用） | 10000（仅 indexer 用） |
| attention scaling | `Dqk**-0.5 = 192**-0.5` | `Dqk**-0.5 = 256**-0.5` |

关键观察：**`mla_nope=True ⇒ DeepseekV2AttentionMLA(skip_rope=True) ⇒ self.rotary_emb = None ⇒
主 MLA 路径完全不做 RoPE**（`q_pe / k_pe` 是未旋转原始 slice，原样进 attn 与 cache）。只 indexer 前
64 维做 NeoX RoPE（`is_neox_style = not getattr(config, "indexer_rope_interleave", False) = True`）。

派生量（per token，进 cache）：
- 主 latent KV 行：`concat(kv_a_layernorm 输出 [Rkv=512], k_pe [Dro=64]) → 576 elem`，BF16 = 1152 B
- index K 行：`128 B FP8 + 4 B fp32 scale = 132 B`，`head_dim_with_sf = 132`

CP 分片下，每卡持久 latent KV ∝ `tokens_owned ≈ seqlen / cp_size`；index K 每卡持全量（见 F-IDX）。

## 算子依赖表（单 DSA 层，decode，CP + Route B；TP=1，非量化）

> 源头链路：`Glm5NextDecoderLayer.forward()`（REPO `models/glm5_next.py`）
> → `layer_communicator.prepare_attn` → `DeepseekV2AttentionMLA.forward`
> （REPO `models/deepseek_v2.py::dispatch_attn_forward_method` → `AttnForwardMethod.MLA`）
> → `forward_absorb_prepare / forward_absorb_core`（REPO `models/deepseek_common/attention_forward_methods/forward_mla.py`）
> → `Indexer.forward_cuda`（REPO `layers/attention/nsa/nsa_indexer.py`）
> → `NativeSparseAttnBackend.forward_decode`（REPO `layers/attention/nsa_backend.py`）。
>
> "CUDA sgl-kernel / sglang 参考" 一栏指向**单卡 REPO 现有实现**（V5/V6 的 D0..D6）；
> "Zeus 现状" 全部 `× TODO`（绿地项目）。⚙ 标的两个 stage（F-TOPK-CP、F-MERGE-POST 的 merge
> 部分）是 V7 新算子，**REPO 单卡路径不存在直接对应**，需要新写——其余 5 个 stage 基于
> REPO 已有 kernel 改/包装即可。
>
> 表里所有 shape 标到 per-rank、含 owner / 非 owner 差异；`Nh_local = Nh`（TP=1）。

| # | Fused Stage | 子步骤 / shape 变换 | CUDA sgl-kernel / sglang 参考 | Zeus 现状 |
|---|---|---|---|---|
| 0 | **F-PRE** | `hidden_new [B,H] → q_lora_raw [B,Rq] + latent_cache [B,Rkv+Dro=576]` (GEMM); `q_lora = q_a_layernorm(q_lora_raw) [B,Rq]`; `k_nope = kv_a_layernorm(latent_cache[:, :Rkv]) [B,Rkv]` (RMSNorm 只 norm 前 Rkv); `k_pe = latent_cache[:, Rkv:] [B,Dro]` (未 norm、未 RoPE); `k_new = concat(k_nope, k_pe).unsqueeze(1) [B,1,576]`。所有卡跑相同的 per-token forward。 | `DeepseekV2AttentionMLA.prepare_qkv_latent`（在 `communicator.py::fetch_qkv_latent` 内调用）+ `forward_absorb_prepare` 起手（`q_a_layernorm` / `kv_a_layernorm`）。RMSNorm 单算子参考 `sgl_kernel.rmsnorm`。 | × TODO 待 fused kernel `dsa_pre_attn_qkv_latent_fused`（GEMM + 2× RMSNorm + concat），所有卡跑 |
| 1 | **F-Q-MAIN** | `q_lora [B,Rq] →(q_b_proj GEMM)→ q [B,Nh*Dqk] → view [B,Nh,Dqk] → split([Dnope],[Dro]) → q_nope [B,Nh,Dnope], q_pe [B,Nh,Dro]`（q_pe **未 RoPE**，mla_nope=True）；`q_nope_out = bmm(q_nope.transpose(0,1), w_kc[Nh,Dnope,Rkv]).transpose(0,1) [B,Nh,Rkv=512]`；`q_new = concat(q_nope_out, q_pe) [B,Nh,Rkv+Dro=576]`。 | `forward_absorb_prepare`（`q_b_proj` + split + `bmm w_kc`）；`q_b_proj` = `ColumnParallelLinear` (GEMM)；`bmm` 用 `torch.bmm`。`w_kc/w_vc` 拆分见 `models/deepseek_common/deepseek_weight_loader.py:565-585`（`post_load_weights`）。 | × TODO 待 fused kernel `dsa_q_proj_absorb_fused`（GEMM + split + batched-GEMM），所有卡跑 |
| 2 | **F-KV-STORE** | (owner-only) `latent_KV_pool_r[ slot(pos(req)) ] = k_new[req]  [1,576]`；非 owner no-op（host 端跳过 launch）。 | `RadixAttention` 内部 `set_mla_kv_buffer`（由 `attn_mqa(..., save_kv_cache=True)` 触发）；REPO 的 V5 dev kernel `dsa_kv_proj_cache_store_fused` 已经是单卡 ✓ 的 fused store 形态。CP 下加 owner 判断。 | × TODO 基于 V5 `dsa_kv_proj_cache_store_fused` + `cache_indices` 寻址 + owner-mask（host 端 group-by-owner launch） |
| 3 | **F-IDX** | (输入 `hidden_new [B,H]`, `q_lora [B,Rq]`) →<br/>`q_idx = wq_b(q_lora).view(B,I,Di)`；<br/>`k_idx = wk(hidden_new) → k_norm(k_idx)`（**完整 LayerNorm**, 含 bias, fp32 计算 → 写回 bf16）`[B,Di]`；<br/>NeoX RoPE on `(q_idx[..., :Dro=64], k_idx[..., :64])`（positions `[B]`），后 64 维不动；<br/>`q_idx = rotate_activation(q_idx)`（Hadamard, scale=Di^-0.5, bf16）；`k_idx = rotate_activation(k_idx)`；<br/>`q_idx_fp8, q_scale = act_quant(q_idx, 128, "ue8m0")`；<br/>`k_idx_fp8, k_idx_scale = act_quant(k_idx, ...)`；<br/>`gate = weights_proj(hidden_new) * I^-0.5`；<br/>`weights = gate.unsqueeze(-1) * q_scale * Di^-0.5  [B,I,1]`（fp32）；<br/>**副作用**：**所有卡** append `(k_idx_fp8, k_idx_scale)` 到各自（复制的）index pool —— 由于 hidden_new / q_lora 在所有卡上 bit-exact 相同，各卡算出来的 k_idx_fp8 也 bit-exact 相同，append 后 index K pool 跨卡保持镜像。 | `nsa_indexer.py::Indexer.forward_cuda` → `_get_q_k_bf16` + `rotate_activation` (l.135) + `act_quant` + `_store_index_k_cache`；`weights_proj` 在 `forward_cuda` 内。底层依赖：`jit_kernel/hadamard.py::hadamard_transform`；`act_quant`（fp8 block-quant，block=128, fmt="ue8m0"）；`jit_kernel/fused_store_index_cache.py::fused_store_index_k_cache`（fast path）。 | × TODO 待 fused kernel `dsa_indexer_prep_store_fused`（GEMM×2 + LayerNorm + RoPE-first-64 + Hadamard + FP8 quant + gate fp32 + cache append）；CP 下没有 owner-gating（all-append） |
| 4 | **F-TOPK-CP** | (per rank, 输入完整的 index K pool) `logits [B, seqlen] = fp8_paged_mqa_logits(q_idx_fp8 [B,1,I,Di], index_K_pool.view(num_pages, 64, 1, 132), weights [B,I], seqlens, page_table, schedule_md, max_seq_len)`；<br/>`topk_positions [B, Ktop] = metadata.topk_transform(logits, Ktop)`（int32, 尾部 -1 填充）—— 各卡因 index K pool 镜像而算出**逐 bit 相同**的 topk_positions，**零通信**；<br/>(per rank r) page-table 变换：`topk_slots_r [B, Ktop] = map(topk_positions, page_table_local_r)`（不归 r 的位置 → -1，因为它的 latent KV 不在本卡上）。<br/>**注意**：与单卡 `_get_topk_paged` 几乎相同，唯一新增的是 per-rank `topk_positions → topk_slots_r` 这步（用本卡的 latent KV page table，非 r 位置自然映成 -1）。 | `nsa_indexer.py::_get_topk_paged`（`deep_gemm.fp8_paged_mqa_logits` + `metadata.topk_transform`，见 `layers/attention/nsa/transform_index.py`）—— 几乎照搬，每卡都跑一份；page-table 变换用 `transform_index.py::transform_index_page_table_decode` 的 per-rank 版本，非 r 位置 → -1。 | × TODO 复用 V5 / V6 单卡的 paged top-k 实现（每卡跑一份），在尾端 page-table 变换上加 "非 r 位置 → -1" 的 mask；**无通信** |
| 5 | **F-MQA-PARTIAL** | (per rank r) `partial_out_r [B, Nh, Rkv], partial_lse_r [B, Nh] = attn_mqa_partial(q_new [B,Nh,576], latent_KV_pool_r, topk_slots_r [B, Ktop], return_lse=True, scaling=Dqk^-0.5, num_kv_heads=1, v_head_dim=Rkv=512)`。<br/>`topk_slots_r` 里 `-1` 的槽位 → 不读 cache、当 -inf、不计入分母。<br/>若某 req 在 rank r 上 `S_local = ∅` → `partial_lse_r[req] = -inf`、`partial_out_r[req]` 任意（D4.5 里被 weight 0 掉）。<br/>"q 看自己"（position `pos(req)` ∈ top-Ktop）自然落在 owner(req) 的 `S_local` 里。 | `nsa_backend.py::NativeSparseAttnBackend.forward_decode` → `_forward_flashmla_kv` / `_forward_flashmla_sparse` / `_forward_tilelang`（底层 `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` / `tilelang_kernel_glm.sparse_mla_fwd_interface`），**要求 `return_lse=True`**。`attn_mqa = RadixAttention(num_kv_heads=1, head_dim=576, v_head_dim=Rkv=512, scaling=Dqk^-0.5)`（`models/deepseek_v2.py:1269`）。 | × TODO 待 fused kernel `dsa_decode_sparse_mqa_partial`：在 V6 sparse MQA kernel 上加 `return_lse=True` 出口；-1 槽位作 mask；空集 lse=-inf 处理 |
| 6 | ⚙ **F-MERGE-POST** | **通信**：all-gather `(partial_out, partial_lse)` 跨 cp_size 卡 → 每卡持 `cp_size` 份 partial。通信量 ≈ `cp_size · B · Nh · (Rkv+1) · 2 B`（all-gather 版，~526 KB/层 cp=16/B=1/Nh=32）或 `2 · B · Nh · (Rkv+1) · 2 B`（ring-allreduce 版，~66 KB/层）。<br/>**Local LSE merge**：`m = max_r partial_lse_r [B,Nh]`；`Z = Σ_r exp(partial_lse_r - m) [B,Nh]`；`attn_out_latent [B, Nh, Rkv] = (Σ_r exp(partial_lse_r - m) * partial_out_r) / Z`。<br/>**V absorb**：`attn_out [B, Nh, Dv] = bmm(attn_out_latent.transpose(0,1), w_vc[Nh,Rkv,Dv]).transpose(0,1)`。<br/>**o_proj**：`out [B, H] = RowParallelLinear(attn_out.reshape(B, Nh*Dv))`。<br/>三步本地融合在一个 post-collective kernel：merge 留 fp32 在 SMEM/寄存器，直接 stream 进 `bmm w_vc` → `o_proj`，省两次 HBM round-trip。 | **REPO 单卡无 merge** —— 单卡 sparse MQA 输出直接是最终 `attn_out_latent`。merge 部分 ⚙ 新写。下游 V absorb + o_proj 参考：`forward_absorb_core` 末段（`bmm w_vc`）+ `RowParallelLinear`。collective 走 `attn_cp_group` 的 all-gather / ring-allreduce（参考 `layers/attention/nsa/utils.py::cp_all_gather_rerange_output` 形态）。 | × TODO ⚙ 新 kernel `dsa_cp_merge_post_fused`：collective（all-gather 或 ring-AR）+ LSE merge（fp32）+ `bmm w_vc` + `o_proj`，融在一个 post-collective kernel 里；**绝对不能用裸 all_reduce(sum)** |

**不在本文档 scope**：
- prefill / extend 路径（默认走 Route A = all-gather latent KV + 本地全量 sparse MQA；见 V5 §14、V6 §2）。
- KDA 线性注意层（21 / 34 层；不支持 CP，进 KDA 前 `cp_all_gather_rerange_output`、出来再 split）。
- MoE-FFN（见 `glm4_moe_ffn_dev.md`）。
- TP / EP / DP-attention（与 CP 正交；TP 切权重头数、CP 切 token；可叠加，见附录 B）。
- NextN / MTP / speculative decoding。
- `MHA_ONE_SHOT` dense fallback（decode 永远 `use_mha=False`）。

## Decode-CP-RouteB 计算流（算子组合 + 中间变量传递）

下图给出单 DSA 层、`cp_size` 卡、Route B 下的完整 decode 算子串联。所有卡跑相同的
per-token forward（hidden_new 复制），**仅"副作用"按 owner 区分**。三条并行支线
F-Q-MAIN / F-KV-STORE / F-IDX 跑 `alt_streams`，在 F-TOPK-CP 前 sync。括号内序号对应
算子依赖表的 `#` 列。符号约定：`B`=decode batch，`H`=hidden_size，`Nh`=num_attention_heads，
`Rq`=q_lora_rank, `Rkv`=kv_lora_rank=512, `Dnope`=qk_nope_head_dim, `Dro`=qk_rope_head_dim=64,
`Dv`=v_head_dim, `I`=index_n_heads=8, `Di`=index_head_dim=128, `Ktop`=index_topk=2048。

```
   hidden_new [B,H] bf16   (来自 layer_communicator.prepare_attn；所有卡相同)
                  │
                  ▼
   ┌───────────────────────────────────────────────────────────────┐
(0)│ F-PRE  dsa_pre_attn_qkv_latent_fused                          │ 所有卡相同
   │   fused_qkv_a_proj_with_mqa(hidden_new)                       │
   │     → q_lora_raw [B,Rq], latent_cache [B,Rkv+Dro=576]         │
   │   q_lora    = q_a_layernorm(q_lora_raw)        [B,Rq]    ★ 喂 (1)(3)
   │   k_nope    = kv_a_layernorm(latent_cache[:,:Rkv]) [B,Rkv=512]
   │   k_pe      = latent_cache[:, Rkv:]            [B,Dro=64]    (未 RoPE)
   │   k_new     = concat(k_nope, k_pe).unsqueeze(1) [B,1,576]    ★ 喂 (2)
   └─────┬────────────────────────────┬──────────────────────────┬──┘
         │                            │                          │
         │ q_lora                     │ k_new                    │ q_lora, hidden_new
         ▼                            ▼                          ▼
   ┌──────────────┐         ┌──────────────────┐      ┌─────────────────────────────┐
(1)│ F-Q-MAIN     │      (2)│ F-KV-STORE       │   (3)│ F-IDX                       │
   │ dsa_q_proj_  │         │ (owner-only)     │      │ dsa_indexer_prep_store_     │
   │  absorb_fused│         │ host 端: rank == │      │  fused                       │
   │              │         │ pos % cp_size 才 │      │   q_idx = wq_b(q_lora)       │
   │ q_b_proj(q_  │         │ launch 写盘     │      │     .view(B,I,Di)            │
   │  lora)       │         │                  │      │   k_idx = wk(hidden_new)     │
   │  [B,Nh*Dqk]  │         │ latent_KV_pool_r │      │   k_idx = k_norm(k_idx)     │
   │ view(B,Nh,   │         │  [slot(pos)] =   │      │     ★ LayerNorm 含 bias fp32 │
   │   Dqk)→split │         │  k_new           │      │   (q_idx[:,:,:64], k_idx[:,:64])│
   │ q_nope[B,Nh, │         │                  │      │     = NeoX RoPE(positions)   │
   │  Dnope]      │         │ 非 owner: no-op  │      │     (后 64 维不动)           │
   │ q_pe [B,Nh,  │         │                  │      │   q_idx = rotate_activation │
   │  Dro] (未    │         └──────────────────┘      │     (q_idx)  ★ Hadamard      │
   │  RoPE)       │                                   │   k_idx = rotate_activation │
   │              │                                   │     (k_idx)                  │
   │ q_nope_out = │                                   │   q_idx_fp8,q_scale =        │
   │  bmm(q_nope, │                                   │     act_quant(q_idx,128,     │
   │   w_kc)      │                                   │       "ue8m0")               │
   │  [B,Nh,Rkv]  │                                   │   k_idx_fp8,k_idx_scale =    │
   │              │                                   │     act_quant(k_idx,...)     │
   │ q_new =      │                                   │   gate = weights_proj(       │
   │  concat(     │                                   │     hidden_new) * I^-0.5     │
   │  q_nope_out, │                                   │   weights = gate.unsqueeze   │
   │   q_pe)      │                                   │    (-1) * q_scale *          │
   │  [B,Nh,576]  │                                   │     Di^-0.5  [B,I,1] fp32    │
   │              │                                   │                              │
   │              │                                   │ **所有卡 append**（镜像）:   │
   │              │                                   │   index_K_pool[slot(pos)] =  │
   │              │                                   │     (k_idx_fp8, k_idx_scale) │
   │              │                                   │   ★ hidden_new 跨卡复制 ⇒   │
   │              │                                   │     k_idx_fp8 bit-exact 一致│
   │              │                                   │   ★ index K 跨卡保持镜像    │
   └──────┬───────┘                                   └─────────────┬────────────────┘
          │                                                         │
          │       (三条并行支线 stream sync)                          │
          └─────────────────────────┬──────────────────────────────┬─┘
                                    │                              │
                                    │ q_new, q_idx_fp8, q_scale,   │
                                    │ k_idx_fp8/scale (已入 pool), │
                                    │ weights                       │
                                    ▼                              │
                  ┌──────────────────────────────────────────────────┐
              (4)│ F-TOPK-CP   dsa_decode_topk_cp_fused             │ ★ 零通信
                 │   (per rank, 输入完整镜像的 index K pool)         │
                 │   logits [B, seqlen] =                            │
                 │     deep_gemm.fp8_paged_mqa_logits(              │
                 │       q_idx_fp8 [B,1,I,Di],                      │
                 │       index_K_pool.view(num_pages,64,1,132),     │
                 │       weights, seqlens, page_table,              │
                 │       schedule_md, max_seq_len)                  │
                 │   topk_positions [B, Ktop] =                     │
                 │     metadata.topk_transform(logits, Ktop)        │
                 │     ★ 各卡因 index K 镜像 ⇒ topk_positions       │
                 │       bit-exact 一致；无 all-gather              │
                 │                                                   │
                 │   (per rank r) topk_slots_r [B, Ktop] =          │
                 │     map_through_page_table(topk_positions,        │
                 │       latent_KV_page_table_local_r)               │
                 │     ★ 不归 r 的 position → -1                    │
                 │       (因 latent KV 是分片的，本卡只持有它的子集) │
                 └────────────────────┬─────────────────────────────┘
                                      │ topk_slots_r [B,Ktop]
                                      │ (非 r 位置 = -1)
                                      ▼
                  ┌──────────────────────────────────────────────────┐
              (5)│ F-MQA-PARTIAL                                    │ 各卡本地
                 │ dsa_decode_sparse_mqa_partial                    │
                 │   (per rank r) partial_out_r [B,Nh,Rkv],         │
                 │     partial_lse_r [B,Nh] =                       │
                 │   attn_mqa_partial(                              │
                 │       q_new [B,Nh,576], latent_KV_pool_r,        │
                 │       topk_slots_r [B,Ktop], return_lse=True,    │
                 │       scaling=Dqk^-0.5, num_kv_heads=1,           │
                 │       v_head_dim=Rkv=512)                         │
                 │   ★ -1 槽位: 不读 cache, 当 -inf, 不计入分母      │
                 │   ★ S_local = ∅: lse = -inf, out = 任意           │
                 │   ★ "q 看自己" 在 owner(req) 的 S_local 里         │
                 └────────────────────┬─────────────────────────────┘
                                      │ (partial_out_r, partial_lse_r) per rank
                                      ▼
                  ┌──────────────────────────────────────────────────┐
              (6)│ F-MERGE-POST  ⚙  dsa_cp_merge_post_fused         │ 含通信 #2 +
                 │   ★ comm #2: all-gather (partial_out, partial_lse)│  本地融合 3 步
                 │     across cp_size ranks                          │
                 │     all-gather 版 ≈ 526 KB / 层 / step            │
                 │     ring-AR 版  ≈ 66  KB / 层 / step              │
                 │                                                   │
                 │   ★ LSE merge (fp32 in registers/SMEM):           │
                 │     m = max_r partial_lse_r          [B,Nh]      │
                 │     Z = Σ_r exp(partial_lse_r - m)   [B,Nh]      │
                 │     attn_out_latent =                            │
                 │       (Σ_r exp(partial_lse_r - m) *               │
                 │              partial_out_r) / Z                  │
                 │       [B,Nh,Rkv=512]                              │
                 │                                                   │
                 │   ★ V absorb (本地, 在 fp32 域上 stream):          │
                 │     attn_out = bmm(attn_out_latent.T,             │
                 │       w_vc[Nh,Rkv,Dv]).T  [B,Nh,Dv]               │
                 │                                                   │
                 │   ★ o_proj (本地):                                │
                 │     out = RowParallelLinear(                      │
                 │       attn_out.reshape(B, Nh*Dv))  [B,H]          │
                 │                                                   │
                 │   ⚠ 绝不能用裸 all_reduce(sum) —— partial 各自的   │
                 │     softmax 分母不同，必须 LSE 重对齐              │
                 └────────────────────┬─────────────────────────────┘
                                      │ out [B,H] bf16 (所有卡相同)
                                      ▼
                          (layer_communicator.postprocess_layer →
                                进下一层 input_layernorm)
```

**关键数据流说明**：

- **F-PRE 既出 `q_lora` 又出 `k_new`**：`q_lora` 是 F-Q-MAIN 和 F-IDX 共享的起点；`k_new` 是
  F-KV-STORE-owner 写盘的全部内容、也是 owner partial sparse MQA 间接读到的内容（owner 已经
  写过 cache，F-MQA-PARTIAL 从 cache 读，不需要直传）。F-PRE 把 `fused_qkv_a` + 两个 layernorm
  + concat 全 fuse 在一个 kernel，输出三件套（`q_lora`、`latent_cache`/`k_new`）。
- **F-Q-MAIN / F-KV-STORE / F-IDX 三条并行支线**：互无依赖，跑 `alt_streams`。F-KV-STORE 在非
  owner 上是 no-op（host 端跳过 launch，非 ckpt-mask 在 kernel 内分支）。F-IDX 的 cache append
  类似（变体 b 下非 owner 跳过）。
- **F-IDX 的 `q_lora` 输入是 F-PRE 的 `q_a_layernorm` 输出**，不是 `q_lora_raw` —— `Indexer` 的
  `wq_b` 接收 norm 后的 q_lora（V5 §11 / REPO `nsa_indexer.py` 的输入约定）。
- **主 MLA 全程无 RoPE**（`mla_nope=True ⇒ rotary_emb=None`）；只 F-IDX 内部对 indexer Q/K 的
  **前 64 维**做 NeoX RoPE，后 64 维不动；之后整 128 维 Hadamard，再 FP8 量化。
- **F-TOPK-CP 零通信**：
  - index K 每卡持镜像 ⇒ 各卡 `fp8_paged_mqa_logits` 输入完全相同 ⇒ `logits` 跨卡 bit-exact 一致
    ⇒ `topk_positions` 跨卡 bit-exact 一致。
  - 每卡接着用**本卡的 latent KV page table** 做 `topk_positions → topk_slots_r`，**这一步各卡结果不同**
    （不归 r 的 position 在本卡的 page table 里查不到 ⇒ 自然映成 -1）。
  - 与 V5 / V6 单卡 `_get_topk_paged` 几乎一致；唯一新增是末端 page-table 变换的"非 r → -1" mask。
- **F-MQA-PARTIAL 必须 `return_lse=True`**：merge 需要 LSE 才能合并各 partial 的 softmax 分母。
  -1 槽位作 mask、空集 partial 输出 lse=-inf —— D4.5 merge 用 `exp(-inf-m)=0` 自然消掉。
- **F-MERGE-POST 三步本地融合**：LSE merge 留 fp32 在 SMEM/寄存器，直接 stream 进 `bmm w_vc`
  → `o_proj`，省两次 HBM round-trip + 两次 kernel launch。merge 之后所有卡持相同 `attn_out_latent`
  ⇒ 下游 V absorb / o_proj 各卡跑相同 GEMM ⇒ `out [B,H]` 各卡一致 ⇒ 下一层 lockstep。
- **正确性是 exact 的**：online-softmax merge 在浮点精度内**精确**等于"在完整 top-Ktop 上做一次
  attention"——所以 REF `--cp k` 必须与 `--cp 1` 逐元素一致，这是 dev 脚本的硬判据。

## Dev 脚本 Stage 顺序

`zeus_dev/dev_glm_moe_dsa_decode_test_v7.py`（绿地，所有 stage 均待落地）。Stage 命名沿用
fused stage 缩写：

1. `cp_pre` —— F-PRE（D0+D1）。`hidden_new [B,H] → q_lora [B,Rq] + k_nope [B,Rkv] + k_pe [B,Dro] + k_new [B,1,576]`。
   两个配置（16b / big）的 shape 各跑一次，验证 fused GEMM + 两个 RMSNorm + concat 与 REF 一致。
2. `cp_q_main` —— F-Q-MAIN（D2）。`q_lora → q_b_proj + split + bmm w_kc → q_new [B,Nh,576]`。
   主 MLA 路径**不插任何 RoPE**；REF 显式按 `(q_pe is not None and rotary_emb is None)` 跳过。
3. `cp_kv_store` —— F-KV-STORE。模拟 cp_size 张卡的 latent KV pool（list of tensors），按 owner
   规则把 `k_new` 写入对应 rank 的 pool；验证 owner pool 内容、非 owner pool 不变。
4. `cp_idx` —— F-IDX（D3）。indexer Q/K + LayerNorm + NeoX RoPE 前 64 + Hadamard + FP8 act_quant
   + gate(fp32) + index K cache append（两个变体各跑）。**注意 LayerNorm 是完整含 bias 的**，
   不是 RMSNorm —— V5 §3.1 的踩坑点。
5. `cp_topk_cp` —— F-TOPK-CP。每卡用复制的完整 index K 直接走 `_get_topk_paged`（**零通信**），
   得到跨卡 bit-exact 一致的 `topk_positions`；再用 per-rank latent-KV page-table 变换得
   `topk_slots_r`（非 r 位置 = -1）。**判据**：`topk_positions(cp=k)` 与 `topk_positions(cp=1)`
   逐元素一致。
6. `cp_mqa_partial` —— F-MQA-PARTIAL（D4）。在 V6 sparse MQA kernel 上加 `return_lse=True`；输入
   `topk_slots_r`（含 -1）和 rank-local KV pool；输出 `(partial_out_r [B,Nh,Rkv], partial_lse_r [B,Nh])`。
   **判据**：单卡退化（cp=1）下 = V5 单卡 D4 输出。
7. `cp_merge_post` —— F-MERGE-POST（⚙ 新算子，含 D4.5+D5+D6 融合）。把 cp_size 份 partial all-gather
   到每卡，本地 LSE merge 出 `attn_out_latent`，紧接 `bmm w_vc` + `o_proj`。**判据**：`out(cp=k)`
   逐元素 == `out(cp=1)`（exact merge）。
8. `cp_full_path` —— D0..D6 端到端，proxy shape 跑通 `--cp ∈ {1, 4, 16}`，硬判据：三个 cp 度数
   下端到端 `out [B,H]` 逐元素一致。

每个 stage：REF 侧 pure-torch、`cp_size` 维度用 list 模拟；Zeus 侧标 `× TODO` `raise NotImplementedError`，
逐 stage 攻下后回填本表的 "Zeus 现状" 一栏（与 `glm4_moe_ffn_dev.md` / `kimi_linear_attn_dev.md` 同款节奏）。

---

# 附录

## 附录 A：为什么 decode 选 Route B（成本对照）

decode 下 sparse MQA 有两条多卡实现路径：

- **Route A**（先 all-gather KV 再本地做满）：每卡每层 ≈ `2048 × 1.15 KB ≈ 2.4 MB / step`，**不规则**
  all-to-all（owner 卡收 ~15 份），负载不均衡。
- **Route B**（本文）：每卡每层 ≈ `~0.3–0.8 MB / step`（top-k merge 256 KB + LSE merge 66–530 KB），
  **规则**集合通信，负载均衡。

为什么这种非对称？**MLA-absorb 下 key 是单 head**（576 elem ≈ 1.15 KB / token）、**query/output 是 Nh 头**
（37 KB / 33 KB / token）—— **搬 key 比搬 query/output 便宜 ~Nh = 32–64 倍**。Route A 搬 KV ✓；
Route B 在 prefill 下要么搬 query（很贵）要么搬 output partial（也贵），所以 prefill 选 A。
decode 下"搬 query"和"reduce output"都只是 1 个 token，绝对量都微不足道（几十 KB），所以 B 完胜
A 的 ~2.4 MB 不规则 gather。完整推导见 V6 §2。

## 附录 B：与 TP / DP-attention / PP 的关系

- **TP 不省 KV**：MLA latent 是单 head，TP 下被复制；TP 只切权重（按 Nh 切）。叠 TP 后 `Nh_local =
  Nh / tp_size`，F-Q-MAIN / F-MERGE-POST 的 Nh 维变成 Nh_local；`o_proj` 多一个 hidden 维 all-reduce
  —— **与 CP 的 merge 是两个不同的 reduce**，CP merge 沿 cp_size 维合 partial，TP all-reduce 沿
  hidden 维合 head。可叠加。
- **DP-attention** 切 request 维（每卡跑自己那批 requests），KV 按 request 分；与 CP 切 token 维不同
  but 可叠加（DP × CP × TP）。多用户高吞吐场景首选 DP-attention；超长上下文（你的 1M 场景）首选 CP。
- **PP** 切层；DSA 层 / 总层数 = 6/27 或 11/45 比较稀疏，PP 切层 KV 也只 ÷pp_size 一次，远不如 CP
  灵活。一般不为了省 KV 而 PP。

## 附录 C：为什么 index K 选"每卡复制"而不是分片

| | 选择：index K 每卡复制（本文）| 备选：index K 分片（**不采**）|
|---|---|---|
| 每卡 index K 持久 HBM | `seqlen · 132 B · num_DSA_layers`（全量；1M·11 ≈ **1.45 GB / 卡**）| `(seqlen/cp_size) · 132 B · num_DSA_layers`（≈ 0.09 GB / cp=16）|
| F-IDX 副作用 | 所有卡 append（hidden_new 复制 ⇒ k_idx_fp8 跨卡 bit-exact 一致 ⇒ pool 保持镜像）| 仅 owner append |
| F-TOPK-CP 通信 | **零** —— 各卡用镜像的完整 index K 跑同一遍 `_get_topk_paged` | all-gather ≤cp_size·Ktop 个 `(pos, logit)` pairs（**+256 KB / 层 / step**），还要写一套全局 top-k merge |
| 与 V5 单卡 kernel 关系 | 几乎照搬，每卡跑一份，**末端加 "非 r 位置 → -1" 的 page-table mask** | 需要新写本地 top-k + all-gather + 全局 merge + page-table 映射 |
| 实现复杂度 | 低 | 中（多一处 collective + 全局 top-k 合并逻辑）|

**决策**：index K 只占 latent KV 的 ~10%（132 B vs 1152 B BF16），复制它换来 top-k 全本地、零通信、几乎照搬单卡 kernel，**值得**。"每卡 KV 空间很少"主要被 latent KV 的分片解决（1M·11层 BF16 ÷ 16 ≈ 0.79 GB/卡），即使再加 1.45 GB index K 复制，整体仍远低于不切分时的 14.2 GB（**省 ~83%**）。

## 附录 D：F-MERGE-POST 的 LSE merge 数学

flash attention 约定：`o_r = Σ_j exp(s_j - lse_r) v_j`（已按 partial r 的"本地分母" `lse_r =
log Σ exp s_j` 归一化）。合并两个不相交子集 A、B：

```
lse_AB = lse_max + log(exp(lse_A - lse_max) + exp(lse_B - lse_max))
o_AB   = exp(lse_A - lse_AB) o_A + exp(lse_B - lse_AB) o_B
```

merge 算子满足结合律 & 交换律，所以可以 tree / ring reduce。`cp_size` 份合一次：

```
m   = max_r partial_lse_r                  # [B, Nh]
Z   = Σ_r exp(partial_lse_r - m)           # [B, Nh]
attn_out_latent = (Σ_r exp(partial_lse_r - m) * partial_out_r) / Z
```

特殊情况：某 rank `S_local = ∅` → `partial_lse_r = -inf` → `exp(-inf - m) = 0`，自然消除。
实现两种：

- **all-gather + 本地 merge**（最简单）：通信 ≈ `cp_size · B · Nh · (Rkv+1) · 2 B`，~526 KB / 层
  cp=16 / B=1 / Nh=32。merge + `bmm w_vc` + `o_proj` 一个 kernel。
- **ring-AR with merge as reduce op**（更省带宽）：通信 ≈ `2 · B · Nh · (Rkv+1) · 2 B` ~66 KB / 层。
  reduce 算子要自己实现（NCCL 没现成 flash-merge），可用若干 all-gather 或自定义 kernel + send/recv 拼。

两种都做成 all-reduce 形态（结果落回所有卡），下游 `bmm w_vc` + `o_proj` 各卡跑相同 GEMM。

## 附录 E：通信量 / 显存估算（cp=16, 1M context, B=1）

每层每 decode step：

| 项 | 通信 |
|---|---:|
| F-TOPK-CP | **0**（index K 每卡复制 ⇒ 零通信）|
| F-MERGE-POST all-gather (partial_out, partial_lse)（all-gather 版） | ~526 KB |
| F-MERGE-POST 通信（ring-AR 版） | ~66 KB |
| **小计（ring-AR）** | **~66 KB / 层** |

× 11 个 DSA 层（大 GLM-5）：≈ **~0.7 MB / decode step**。在 400 GB/s NVLink 上 ~1.8 µs，完全可忽略。

per 卡持久 HBM（1M context；latent KV 分片、index K 复制）：

| | latent KV（分片，per 卡） | index K（每卡复制全量） | 合计 / 卡 |
|---|---|---|---|
| 16B（6 DSA 层） | 6·1M·1.15 KB / 16 ≈ **0.43 GB** | 6·1M·132 B ≈ **0.79 GB** | ≈ **1.22 GB** |
| 大 GLM-5（11 DSA 层） | 11·1M·1.15 KB / 16 ≈ **0.79 GB** | 11·1M·132 B ≈ **1.45 GB** | ≈ **2.24 GB** |

参考（不切 CP，单卡全量）：16B ≈ 7.87 GB / 大 GLM-5 ≈ 14.42 GB —— **CP 16 卡 + index K 复制
省 ~85% / ~84% 持久 KV HBM**。latent KV 若按 `fp8_e4m3` 存可再 ÷2。

## 附录 F：开放问题 / 注意点

- **REPO 现状是 prefill CP**（`is_nsa_enable_prefill_cp` / `nsa_cp_metadata` / `cp_all_gather_rerange_output`）。
  本文档延伸到 decode；F-TOPK-CP 沿用 V5 单卡 `_get_topk_paged` + per-rank page-table mask（小改动），
  F-MERGE-POST 是 ⚙ 真正的新算子（LSE-aware reduce + fused tail），REPO 单卡路径不存在直接对应。
- **decode 沿用 prefill 的分片规则**：本文假设 round-robin（`pos % cp_size`）；若 prefill 用 zigzag block
  split，decode 阶段要么转 round-robin（推荐）、要么新 token 一律落"tail" rank（会随时间失衡）。
- **B > 1**：不同 request 的 owner 不同、top-Ktop 不同；上述链路逐 request 成立，集合通信在 B 维 batch；
  F-MERGE-POST 的 merge 逐 `(req, head)` 独立。
- **`seqlen ≤ Ktop`**：top-k 退化为"全选可见"，`topk_positions = [0, 1, …, seqlen-1, -1, …]`；
  各卡的 `K_local` 就是它持有的全部可见 token，链路不变。
- **`return_lse=True` 的可用性**：`flash_mla_with_kvcache` / `flash_mla_sparse_fwd` /
  `tilelang sparse_mla_fwd_interface` 大多支持 LSE 返回；要确认 Zeus 端的 sparse MQA kernel
  在 v1 就把 LSE 出口开出来，否则 F-MERGE-POST 没法实现。
- **MHC**：CP 下 layer communicator 是 `MHCHybridNSACPLayerCommunicator`；本文按非-MHC 普通语义，
  MHC 残差作外层包装单独处理。
- **owner 切换的 boundary case**：当 `pos(req)` 跨过 `cp_size` 边界（新 decode 步），新 owner 切到下一
  rank —— 不需要 KV 迁移，新 token 的 KV 在新 owner 上 append；老 token 的 KV 在老 owner 上保留。
  这是 round-robin 的天然性质。

## 附录 G：参考源码索引（REPO 路径）

| 标签 | 文件 | 关注点 |
|---|---|---|
| `R-ENTRY` | `models/glm5_next.py::Glm5NextForCausalLM` / `Glm5NextDecoderLayer` | DSA 层入口；DSA 子层 = `DeepseekV2AttentionMLA(skip_rope=mla_nope)` |
| `R-CFG` | `configs/model_config.py::is_deepseek_nsa`；`configs/glm_linear.py::GlmLinearConfig` | NSA 判定（白名单已含 `Glm5NextForCausalLM`）；层分流 |
| `R-MLA` | `models/deepseek_v2.py::DeepseekV2AttentionMLA`（`__init__`、`prepare_qkv_latent`、`attn_mqa`/`attn_mha` l.1269/1280） | F-PRE / F-Q-MAIN / F-MERGE-POST 的 GEMM 来源 |
| `R-ABSORB` | `models/deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare / forward_absorb_core` | F-PRE / F-Q-MAIN / F-MERGE-POST 的 `bmm w_kc` / `bmm w_vc` |
| `R-WLOADER` | `models/deepseek_common/deepseek_weight_loader.py::post_load_weights` (l.565-585) | `kv_b_proj.weight` → `w_kc [Nh,Dnope,Rkv]` / `w_vc [Nh,Rkv,Dv]` |
| `R-INDEXER` | `layers/attention/nsa/nsa_indexer.py::Indexer.forward_cuda`（`_get_q_k_bf16`、`rotate_activation` l.135、`act_quant`、`_store_index_k_cache`、`_get_topk_paged`） | F-IDX / F-TOPK-CP（变体 a）的来源 |
| `R-TOPK` | `layers/attention/nsa/transform_index.py::topk_transform`、`transform_index_page_table_decode` | F-TOPK-CP 的 top-k + page-table 变换；`metadata.topk_transform` 含 -1 padding 处理 |
| `R-BACKEND` | `layers/attention/nsa_backend.py::NativeSparseAttnBackend.forward_decode`（`_forward_flashmla_kv` / `_forward_flashmla_sparse` / `_forward_tilelang`） | F-MQA-PARTIAL 的 sparse MQA kernel；需扩 `return_lse=True` |
| `R-CACHE` | `mem_cache/memory_pool.py::NSATokenToKVPool`（继承 `MLATokenToKVPool`）；`layers/attention/nsa/index_buf_accessor.py`（`GetK`/`GetS`/`SetKAndS`） | F-KV-STORE / F-IDX append 的 cache layout |
| `R-QUANT` | `layers/attention/nsa/quant_k_cache.py` / `dequant_k_cache.py` | fp8 latent KV 量化/反量化（叠 fp8 时用） |
| `R-FUSEDSTORE` | `jit_kernel/fused_store_index_cache.py::fused_store_index_k_cache` / `can_use_nsa_fused_store`；`jit_kernel/hadamard.py::hadamard_transform` | F-IDX 的 fused index K store 与 Hadamard 实现 |
| `R-CP` | `layers/attention/nsa/utils.py`（`can_cp_split`、`cp_all_gather_rerange_output`、`cp_split_and_rebuild_data/position`、`is_nsa_enable_prefill_cp`）；`forward_batch.nsa_cp_metadata` | CP 原语（REPO 现为 prefill CP，本文延伸到 decode） |
| `R-COMM` | `layers/communicator.py`（`fetch_qkv_latent`）；`layers/communicator_mhc_hybrid_cp.py::MHCHybridNSACPLayerCommunicator` | layer communicator：`prepare_attn` 内执行 `prepare_qkv_latent`；CP+MHC 下用 hybrid 版 |

---

> 配套文档：
> - V5 `GlmMoeDsa_dev_V5.md` —— 单卡完整链路（decode + prefill + dense fallback）的真值校对版本。
> - V6 `GlmMoeDsa_dev_V6.md` —— Route B 的设计推导、cost analysis、"owner 非对称"详解，以及与 V5 单卡 D0..D6 的对应。
> - 本文档 V7 —— 工程实现版（fused stage 划分、shape 表、计算流、dev stage），按 `glm4_moe_ffn_dev.md` / `kimi_linear_attn_dev.md` 风格组织。

## 开发日志

### 2026-05-12 · 起点
- 创建本文档、`dev_glm_moe_dsa_decode_test_v7.py` 骨架。
- 7 个 stage 全部 `× TODO`，Zeus 路径 `raise NotImplementedError`。
- 下一个 focus：`cp_pre`（F-PRE）的 REF 跑通 + Zeus fused kernel `dsa_pre_attn_qkv_latent_fused`
  在 `sgl-kernel-zeus` 落地。优先用 16B 配置 proxy shape（B=2, H=2048）验证；大配置（H=4096）
  shape 通过后跟进。
