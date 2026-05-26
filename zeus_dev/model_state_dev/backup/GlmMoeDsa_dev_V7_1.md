# GLM5-Next DSA Decode (CP / Route B) Zeus 适配开发追踪（V7.1, config_16b_v2）

> V7 的修订版。针对新的 `zeus_dev/model_state_dev/config_16b_v2.json`，
> 主要改动是 **`qk_rope_head_dim: 64 → 0`**（主 MLA + indexer 全链路**无 RoPE**），
> 以及 **MHC 开启**（`mhc=true`、4 残差流）。其余目标 / 方法 / 范围与 V7 一致：
> GLM5-Next 的 DSA 全注意力子层（`Glm5NextDecoderLayer.self_attn = DeepseekV2AttentionMLA`，
> 见 REPO `python/sglang/srt/models/glm5_next.py`）的 decode 路径，**多卡 Context-Parallel +
> "Route B"（KV 按 token 位置分片、partial sparse MQA + online-softmax merge）下**，
> 在纯 Zeus（无 CUDA）环境里逐算子跑通。
>
> 校对依据：`/root/project/sglang-feat-v0.5.10-prerelease-glm`（以下 **REPO**）。同目录 V5
> 单卡完整、V6 决策推导、V7 工程实现的逻辑全部沿用，本文只标注 v2 配置带来的 **shape /
> 算子 / 通信变更**。

## 与 V7（config_16b）的差异速览

| 维度 | V7 / config_16b | **V7.1 / config_16b_v2** |
|---|---:|---:|
| `qk_rope_head_dim` (Dro) | 64 | **0** |
| 主 MLA 路径 RoPE | `rotary_emb=None`（mla_nope）但保留未旋转的 `q_pe / k_pe` slice | **完全无 RoPE，无 q_pe / k_pe slice** |
| `qk_head_dim` (Dqk) | `Dnope+Dro = 192` | `Dnope+Dro = 128` |
| attention scaling | `Dqk**-0.5 = 192**-0.5` | `Dqk**-0.5 = 128**-0.5` |
| latent KV 行 / token | `concat(k_nope, k_pe) = Rkv+Dro = 576` elem, **1152 B** BF16 | `k_nope = Rkv = 512` elem, **1024 B** BF16 |
| q_new 形状 | `[B, Nh, Rkv+Dro = 576]` | `[B, Nh, Rkv = 512]` |
| `attn_mqa` `head_dim` / `v_head_dim` | 576 / 512 | **512 / 512** |
| Indexer RoPE | NeoX RoPE on first 64 dims of `(q_idx, k_idx)` | **无**（config 蕴含：`Indexer(rope_head_dim=qk_rope_head_dim=0)` → split 出 0 维 slice → 语义 noop）|
| `index_head_dim` (Di) | 128 | 128（不变）|
| `index_topk` (Ktop) | 2048 | 2048（不变）|
| `index_n_heads` (I) | 8 | 8（不变）|
| MHC | （v1 配置未声明，按非-MHC 普通语义）| **`mhc=true, streams=4, tau=1.0, sinkhorn_iter=20, no_norm_weight=true, post_mult_value=2, hc_eps=1e-6`** |
| layer communicator | `NSACPLayerCommunicator`（或非 CP 版）| **`MHCHybridNSACPLayerCommunicator`** |
| `swiglu_clamp_limit` | 未声明 | 10.0（FFN，不在本文 scope）|
| 其它（H / Nh / Rq / Rkv / Dnope / Dv / num_hidden_layers / DSA 层数 / 总层数）| 见 V7 | **不变** |

**直观影响**：
- 主 MLA 路径"latent_cache 拆 nope + pe"这一步消失，`latent_cache` 自身就是 `k_nope`；
- F-Q-MAIN 的 `split([Dnope], [Dro])` 与末端的 `concat(q_nope_out, q_pe)` 都不再需要；
- F-IDX 砍掉 NeoX RoPE 子步骤，整 128 维 LayerNorm 之后直接 Hadamard + FP8 量化；
- F-KV-STORE 写一行从 1152 B 缩到 1024 B；
- F-MQA-PARTIAL 的 K 维从 576 缩到 512，单 step 计算量 ↓ ~11%；
- F-MERGE-POST 的 partial all-gather 通信量同步缩（576+1 → 512+1，~12% 节省）；
- 持久 HBM：latent KV 占用 ÷ 16 后再 ×(512/576) ≈ ×0.89 的额外缩量。

## v2 配置（来自 `zeus_dev/model_state_dev/config_16b_v2.json`）关键字段

REPO 的 `is_deepseek_nsa()` 白名单已含 `"Glm5NextForCausalLM"`，触发条件 `index_topk is not None`
（GLM5 = 2048）⇒ 启用 NSA backend。

| 字段 | `config_16b_v2.json`（v2 小）|
|---|---:|
| `architectures` | `["Glm5NextForCausalLM"]` |
| `hidden_size` (H) | 2048 |
| `num_attention_heads` (Nh) | 32 |
| `q_lora_rank` (Rq) | 768 |
| **`kv_lora_rank` (Rkv)** | **512** |
| `qk_nope_head_dim` (Dnope) | 128 |
| **`qk_rope_head_dim` (Dro)** | **0** ★ |
| `qk_head_dim` (Dqk = Dnope+Dro) | **128** ★ |
| `v_head_dim` (Dv) | 128 |
| **`index_head_dim` (Di)** | **128** |
| `index_n_heads` (I) | 8 |
| **`index_topk` (Ktop)** | **2048** |
| `index_dsa_use_layernorm` | true |
| `mla` / `mla_nope` | true / **true** |
| `multi_query_attention` | true |
| **`mhc`** | **true** ★ |
| **`mhc_num_residual_streams`** | **4** ★ |
| **`mhc_tau`** | **1.0** ★ |
| **`mhc_sinkhorn_iterations`** | **20** ★ |
| **`mhc_no_norm_weight`** | **true** ★ |
| **`mhc_post_mult_value`** | **2** ★ |
| **`hres_vwnstyle`** | false |
| **`hc_eps`** | 1e-06 |
| `swiglu_clamp_limit` | 10.0（FFN, 非本文 scope）|
| `num_hidden_layers` | 27 |
| `linear_attn_config.full_attn_layers` | `[3,7,11,15,19,23]` |
| **DSA 层数** | **6** |
| `num_key_value_heads` | 8（MLA 不用）|
| `rope_theta` | 10000（**全链路无 RoPE 后此字段无消费者**）|

派生量（per token，进 cache）：
- 主 latent KV 行：`kv_a_layernorm 输出 [Rkv=512]`，BF16 = **1024 B / token**（**v1 是 1152 B**）。
- index K 行：`128 B FP8 + 4 B fp32 scale = 132 B`，`head_dim_with_sf = 132`（不变）。

CP 分片下，每卡持久 latent KV ∝ `tokens_owned ≈ seqlen / cp_size`；index K 每卡持全量（见 F-IDX）。

## 算子依赖表（单 DSA 层，decode，CP + Route B；TP=1，非量化；v2 = 无 RoPE + MHC）

> 源头链路（与 V7 相同）：`Glm5NextDecoderLayer.forward()` → `layer_communicator.prepare_attn`
> → `DeepseekV2AttentionMLA.forward` → `forward_absorb_prepare / forward_absorb_core`
> → `Indexer.forward_cuda` → `NativeSparseAttnBackend.forward_decode`。
>
> ★ 标 v2 相对 V7 的变更点。⚙ 标 V7 已说明的"新算子（REPO 无单卡直接对应）"。
>
> 表里所有 shape 标到 per-rank、含 owner / 非 owner 差异；`Nh_local = Nh`（TP=1）。

| # | Fused Stage | 子步骤 / shape 变换（v2）| CUDA sgl-kernel / sglang 参考 | Zeus 现状 |
|---|---|---|---|---|
| 0 | **F-PRE** ★ | `hidden_new [B,H] → q_lora_raw [B,Rq] + latent_cache [B,Rkv=512]` (GEMM, ★ **无 `+Dro` tail**); `q_lora = q_a_layernorm(q_lora_raw) [B,Rq]`; `k_nope = kv_a_layernorm(latent_cache) [B,Rkv=512]` (★ **整段 norm，不再切前 Rkv**); ★ **无 k_pe、无 concat**; `k_new = k_nope.unsqueeze(1) [B,1,512]`。所有卡跑相同 per-token forward。 | `prepare_qkv_latent`（`communicator.py::fetch_qkv_latent` 内）+ `forward_absorb_prepare` 起手（`q_a_layernorm` / `kv_a_layernorm`）；`sgl_kernel.rmsnorm` | × TODO 待 fused kernel `dsa_pre_attn_qkv_latent_fused_v2`（GEMM + 2× RMSNorm，**无 concat / 无 split**），所有卡跑 |
| 1 | **F-Q-MAIN** ★ | `q_lora [B,Rq] →(q_b_proj GEMM)→ q [B,Nh*Dqk=Nh*128] → view [B,Nh,Dqk=128]`（★ **不再 split [Dnope],[Dro]**，整段就是 `q_nope`）；`q_nope_out = bmm(q_nope.transpose(0,1), w_kc[Nh,Dnope,Rkv]).transpose(0,1) [B,Nh,Rkv=512]`；★ **`q_new = q_nope_out [B,Nh,512]`，无 concat q_pe**。 | `forward_absorb_prepare`（`q_b_proj` + `bmm w_kc`）；`q_b_proj` = `ColumnParallelLinear`；`bmm` = `torch.bmm`。`w_kc/w_vc` 拆分见 `deepseek_weight_loader.py::post_load_weights` l.565-585。 | × TODO 待 fused kernel `dsa_q_proj_absorb_fused_v2`（GEMM + batched-GEMM，★ 删除 split/concat 子步骤），所有卡跑 |
| 2 | **F-KV-STORE** ★ | (owner-only) `latent_KV_pool_r[ slot(pos(req)) ] = k_new[req]  [1, 512]`（★ **写一行 512 elem / 1024 B**，v1 是 576 elem / 1152 B）；非 owner no-op。 | `RadixAttention` 内部 `set_mla_kv_buffer`（由 `attn_mqa(..., save_kv_cache=True)` 触发）；V5 dev kernel `dsa_kv_proj_cache_store_fused` 已是单卡 ✓ 的 fused store 形态，CP 下加 owner 判断。 | × TODO 基于 V5 `dsa_kv_proj_cache_store_fused` + `cache_indices` 寻址 + owner-mask（host 端 group-by-owner launch），★ 行宽改 512 |
| 3 | **F-IDX** ★ | (输入 `hidden_new [B,H]`, `q_lora [B,Rq]`) →<br/>`q_idx = wq_b(q_lora).view(B,I,Di=128)`；<br/>`k_idx = wk(hidden_new) → k_norm(k_idx)`（**完整 LayerNorm**, 含 bias, fp32 计算 → 写回 bf16）`[B,Di=128]`；<br/>★ **无 NeoX RoPE 子步骤**（v1 在这里对前 64 维做 NeoX RoPE，v2 整段不旋转）；<br/>`q_idx = rotate_activation(q_idx)`（Hadamard, scale=Di^-0.5, bf16）；`k_idx = rotate_activation(k_idx)`；<br/>`q_idx_fp8, q_scale = act_quant(q_idx, 128, "ue8m0")`；`k_idx_fp8, k_idx_scale = act_quant(k_idx, ...)`；<br/>`gate = weights_proj(hidden_new) * I^-0.5`；`weights = gate.unsqueeze(-1) * q_scale * Di^-0.5 [B,I,1]` fp32；<br/>**副作用**：**所有卡** append `(k_idx_fp8, k_idx_scale)` 到各自（复制的）index pool（hidden_new 跨卡 bit-exact 相同 ⇒ k_idx_fp8 bit-exact 相同 ⇒ pool 跨卡镜像）。 | `nsa_indexer.py::Indexer.forward_cuda` → `_get_q_k_bf16` + `rotate_activation` (l.135) + `act_quant` + `_store_index_k_cache`；底层：`jit_kernel/hadamard.py::hadamard_transform`；`act_quant`（fp8 block-quant，block=128, fmt="ue8m0"）；`jit_kernel/fused_store_index_cache.py::fused_store_index_k_cache`。 | × TODO 待 fused kernel `dsa_indexer_prep_store_fused_v2`（GEMM×2 + LayerNorm + **★ 跳过 RoPE** + Hadamard + FP8 quant + gate fp32 + cache append）；CP 下没有 owner-gating（all-append）|
| 4 | **F-TOPK-CP** | (per rank, 输入完整镜像 index K pool) `logits [B, seqlen] = fp8_paged_mqa_logits(q_idx_fp8 [B,1,I,Di=128], index_K_pool.view(num_pages, 64, 1, 132), weights [B,I], seqlens, page_table, schedule_md, max_seq_len)`；<br/>`topk_positions [B, Ktop] = metadata.topk_transform(logits, Ktop)`（int32, 尾部 -1 填充）—— 各卡 bit-exact 一致，**零通信**；<br/>(per rank r) page-table 变换：`topk_slots_r [B, Ktop] = map(topk_positions, page_table_local_r)`（不归 r 的位置 → -1）。 | `nsa_indexer.py::_get_topk_paged`（`deep_gemm.fp8_paged_mqa_logits` + `metadata.topk_transform`，见 `transform_index.py`）—— 几乎照搬。 | × TODO 复用 V5 / V6 单卡 paged top-k 实现（每卡跑一份），尾端 page-table 变换加 "非 r → -1" mask；**无通信** |
| 5 | **F-MQA-PARTIAL** ★ | (per rank r) `partial_out_r [B, Nh, Rkv=512], partial_lse_r [B, Nh] = attn_mqa_partial(q_new [B,Nh,512], latent_KV_pool_r, topk_slots_r [B, Ktop], return_lse=True, scaling=Dqk^-0.5 = 128^-0.5, num_kv_heads=1, head_dim=512, v_head_dim=Rkv=512)`（★ **head_dim 由 576 → 512**，**scaling 由 192^-0.5 → 128^-0.5**）。<br/>`topk_slots_r` 里 `-1` 槽位 → 不读 cache、当 -inf、不计入分母。<br/>若某 req 在 rank r 上 `S_local = ∅` → `partial_lse_r[req] = -inf`、`partial_out_r[req]` 任意（merge 中被 weight 0 掉）。 | `nsa_backend.py::NativeSparseAttnBackend.forward_decode` → `_forward_flashmla_kv` / `_forward_flashmla_sparse` / `_forward_tilelang`（底层 `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` / `tilelang sparse_mla_fwd_interface`），**要求 `return_lse=True`**。`attn_mqa = RadixAttention(num_kv_heads=1, head_dim=512, v_head_dim=Rkv=512, scaling=128^-0.5)`（v2 配置下）。 | × TODO 待 fused kernel `dsa_decode_sparse_mqa_partial_v2`：V6 sparse MQA kernel + `return_lse=True` 出口 + -1 mask + 空集 lse=-inf；★ 模板 head_dim 改 512 |
| 6 | ⚙ **F-MERGE-POST** ★ | **通信**：all-gather `(partial_out, partial_lse)` 跨 cp_size 卡 → 每卡持 `cp_size` 份 partial。★ 通信量 `cp_size · B · Nh · (Rkv+1) · 2 B` = `cp_size · B · Nh · 513 · 2 B`（**v1 是 577**，all-gather 版 ~467 KB / 层 cp=16/B=1/Nh=32，v1 ~526 KB）或 `2 · B · Nh · 513 · 2 B`（ring-AR ~58 KB / 层，v1 ~66 KB）。<br/>**Local LSE merge**：`m = max_r partial_lse_r`；`Z = Σ_r exp(partial_lse_r - m)`；`attn_out_latent [B, Nh, Rkv=512] = (Σ_r exp(partial_lse_r - m) * partial_out_r) / Z`。<br/>**V absorb**：`attn_out [B, Nh, Dv=128] = bmm(attn_out_latent.transpose(0,1), w_vc[Nh,Rkv,Dv]).transpose(0,1)`。<br/>**o_proj**：`out [B, H=2048] = RowParallelLinear(attn_out.reshape(B, Nh*Dv))`。<br/>三步本地融合在一个 post-collective kernel：merge 留 fp32 在 SMEM/寄存器，直接 stream 进 `bmm w_vc` → `o_proj`。 | REPO 单卡无 merge —— 单卡 sparse MQA 输出直接是最终 `attn_out_latent`。merge 部分 ⚙ 新写。下游 V absorb + o_proj 参考：`forward_absorb_core` 末段 + `RowParallelLinear`。collective 走 `attn_cp_group`（参考 `nsa/utils.py::cp_all_gather_rerange_output`）。 | × TODO ⚙ 新 kernel `dsa_cp_merge_post_fused_v2`：collective + LSE merge(fp32) + `bmm w_vc` + `o_proj`，融在一个 post-collective kernel；**绝不能用裸 all_reduce(sum)** |

**不在本文档 scope**：
- prefill / extend 路径（Route A）。
- KDA 线性注意层。
- MoE-FFN（含 v2 的 `swiglu_clamp_limit=10.0`）。
- TP / EP / DP-attention（与 CP 正交）。
- NextN / MTP / speculative decoding。
- **MHC 本身的 4 残差流融合 / Sinkhorn / post-mult**（本文按 attention 子层视角看 MHC 影响：只换 layer communicator 包装；MHC 残差融合算子作外层处理）。

## v2 关键改动详解

### 1. 主 MLA 全链路无 RoPE

v1 的 "rotary_emb=None + 保留未旋转的 q_pe/k_pe slice" 在 v2 直接退化为 "**完全不存在 pe slice**"：
- `qk_rope_head_dim=0 ⇒ Dro=0 ⇒ Dqk=Dnope=128`；
- `q_b_proj` 输出 `[B, Nh*128]`，整段就是 `q_nope`，没有"前 Dnope / 后 Dro"的分块；
- `latent_cache = kv_a_proj(hidden_new)` 输出 `[B, Rkv=512]`，没有"前 Rkv / 后 Dro"的分块；
- `k_new = k_nope.unsqueeze(1) [B, 1, 512]`，**不存在 concat 步骤**；
- `q_new = q_nope_out [B, Nh, 512]`，**不存在 concat 步骤**；
- `attn_mqa.head_dim = 512`（v1 = 576）、scaling = `128^-0.5`（v1 = `192^-0.5`）。

### 2. Indexer 也无 RoPE（由 config 直接蕴含）

不是口头假设，是 config 蕴含的事实。Indexer 的 `rope_head_dim` 来源链路：

```
config_16b_v2.json:  qk_rope_head_dim = 0
    ↓
models/deepseek_v2.py:1188-1207  (`use_nsa` 分支)
    self.indexer = Indexer(
        ...
        rope_head_dim=qk_rope_head_dim,   # ← 直接透传
        is_neox_style=not getattr(config, "indexer_rope_interleave", False),  # = True
        ...
    )
    ↓
layers/attention/nsa/nsa_indexer.py:150-228  (`Indexer.__init__`)
    self.rope_head_dim = rope_head_dim                                 # = 0
    self.rotary_emb = get_rope_wrapper(rope_head_dim, rotary_dim=rope_head_dim, ...)
    ↓
layers/attention/nsa/nsa_indexer.py:274-364  (`_get_q_k_bf16`)
    q_rope, _ = torch.split(query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
                                    # = [0, 128] → q_rope.shape[-1] = 0
    q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)        # 0 维输入
    query[..., : self.rope_head_dim] = q_rope.clone()                  # = query[..., :0] = ... → noop
```

⇒ v1 (`rope_head_dim=64`) 真的旋转前 64 维；v2 (`rope_head_dim=0`) 的 split 出空 slice、
写回是 noop ⇒ **语义上整段 q_idx / k_idx 不旋转**。

v1 的 F-IDX 在 LayerNorm 后、Hadamard 前对 `(q_idx[..., :64], k_idx[..., :64])` 做 NeoX RoPE。
v2 这一步在**语义层面**整段移除：

```
q_idx ← wq_b(q_lora).view(B, I, Di=128)
k_idx ← LayerNorm(wk(hidden_new))         [B, Di=128]
# ★ v1 这里有 NeoX RoPE on first 64 dims —— v2 跳过
q_idx ← rotate_activation(q_idx)          # Hadamard, Di^-0.5
k_idx ← rotate_activation(k_idx)
q_idx_fp8, q_scale ← act_quant(q_idx, 128, "ue8m0")
k_idx_fp8, k_idx_scale ← act_quant(k_idx, ...)
gate    ← weights_proj(hidden_new) * I^-0.5
weights ← gate.unsqueeze(-1) * q_scale * Di^-0.5   # [B, I, 1] fp32
```

`fp8_paged_mqa_logits` 的输入 shape 完全一致（`[B,1,I,Di]` / `[num_pages,64,1,132]`），
所以 F-TOPK-CP 阶段不变。

**实现层 caveat（已查实）**：`nsa_indexer.py::_get_q_k_bf16` (line 289-323) **没有
`if rope_head_dim > 0` 短路分支**，照样会跑 `torch.split([0, 128])` + `self.rotary_emb(...)`
+ `query[..., :0] = ...`。语义上每步都是 noop，但 `Indexer.__init__` 里的
`get_rope_wrapper(0, rotary_dim=0, ...)` （`nsa_indexer.py:217`）大概率会在 cos/sin cache
初始化时**构造期 fail**（除零 / 形状错误，取决于 wrapper 实现）。

⇒ Zeus 侧 `cp_idx_v2` 落地策略（二选一，落地时定）：
- (a) **显式跳过**：在 dev script 里 monkey-patch `Indexer._get_q_k_bf16`，删掉 split / rotary_emb /
  写回三行；构造期短路掉 `get_rope_wrapper` 调用（或传入 dummy `rope_head_dim=1` 但 forward 不调用）；
- (b) **新写子类** `IndexerNoRope(Indexer)`，`__init__` 跳过 `rotary_emb` 创建，`_get_q_k_bf16`
  override 成无 RoPE 版本（LayerNorm 后直出 Hadamard）。

推荐 (b)：清晰、与 V7 共享主体代码、未来 REPO 上游真给 `rope_head_dim==0` 加分支时容易切换回去。

### 3. MHC 包装

v2 `mhc=True`、4 残差流：
- layer communicator 换成 `MHCHybridNSACPLayerCommunicator`（REPO `layers/communicator_mhc_hybrid_cp.py`）；
- `prepare_attn` / `postprocess_layer` 的语义变了（4 路残差流的分流 + 合流），attention 子层
  收到的 `hidden_new` 是 MHC 选出的"主分支"；
- attention 子层**对内**只看 `hidden_new`，不感知 MHC，所以 F-PRE..F-MERGE-POST 的算子串
  与本文 v2 描述完全一致；
- **对外**，attention 输出 `out [B, H]` 会被 MHC 包装层再做 sinkhorn 投影 / post-mult /
  残差合流 —— 这部分属于 MHC 包装算子，单列独立 dev stage（不在本文 scope）。

实现层面：dev script 的 stage 1..7 都按"裸 attention"语义跑（不套 MHC），stage 8
端到端时打开 MHC 包装以验证 communicator 兼容性。

### 4. 持久 HBM / 通信缩量

| 项 | v1 (16b) | **v2 (16b_v2)** | Δ |
|---|---:|---:|---:|
| latent KV 行 / token | 576 elem / 1152 B | **512 elem / 1024 B** | **−11.1%** |
| F-MQA-PARTIAL K 维 | 576 | **512** | −11.1% |
| F-MERGE-POST all-gather 通信 (cp=16, B=1, Nh=32) | ~526 KB | **~467 KB** | −11.2% |
| F-MERGE-POST ring-AR 通信 | ~66 KB | **~58 KB** | −12.1% |
| 持久 latent KV / 卡 (1M ctx, 6 DSA 层, cp=16) | 0.43 GB | **0.38 GB** | −11.6% |
| 持久 index K / 卡 (1M, 6 DSA 层, 复制) | 0.79 GB | 0.79 GB | 0 |
| **合计 / 卡 (1M, 16B v2)** | 1.22 GB | **~1.17 GB** | −4.1% |

主导是 latent KV 行宽，index K 132 B 不动。

## Decode-CP-RouteB 计算流（v2 算子组合 + 中间变量传递）

下图给出 v2 单 DSA 层、`cp_size` 卡、Route B 下的完整 decode 算子串联。
与 V7 相比 ★ 处为本版本变更点。

```
   hidden_new [B,H=2048] bf16   (来自 MHCHybridNSACPLayerCommunicator.prepare_attn)
                  │
                  ▼
   ┌───────────────────────────────────────────────────────────────┐
(0)│ F-PRE  dsa_pre_attn_qkv_latent_fused_v2                       │ 所有卡相同
   │   fused_qkv_a_proj_with_mqa(hidden_new)                       │
   │     → q_lora_raw [B,Rq=768], latent_cache [B,Rkv=512] ★       │
   │   q_lora = q_a_layernorm(q_lora_raw)        [B,768]    ★ 喂(1)(3)
   │   k_nope = kv_a_layernorm(latent_cache)     [B,512] ★ 整段 norm
   │   ★ 无 k_pe / 无 concat                                       │
   │   k_new  = k_nope.unsqueeze(1)              [B,1,512]   ★ 喂 (2)
   └─────┬────────────────────────────┬──────────────────────────┬──┘
         │ q_lora                     │ k_new                    │ q_lora, hidden_new
         ▼                            ▼                          ▼
   ┌──────────────┐         ┌──────────────────┐      ┌─────────────────────────────┐
(1)│ F-Q-MAIN     │      (2)│ F-KV-STORE       │   (3)│ F-IDX                       │
   │ dsa_q_proj_  │         │ (owner-only)     │      │ dsa_indexer_prep_store_     │
   │ absorb_fused │         │ rank == pos %    │      │  fused_v2                    │
   │ _v2          │         │ cp_size 才 launch│      │                              │
   │              │         │                  │      │  q_idx = wq_b(q_lora)        │
   │ q_b_proj(q_  │         │ latent_KV_pool_r │      │    .view(B,I=8,Di=128)       │
   │  lora)       │         │  [slot(pos)] =   │      │  k_idx = wk(hidden_new)      │
   │  [B,Nh*128]  │         │   k_new          │      │  k_idx = k_norm(k_idx)       │
   │ view(B,Nh,   │         │   ★ 行宽 1024B  │      │    ★ LayerNorm 含 bias fp32  │
   │  Dqk=128)    │         │                  │      │                              │
   │ ★ 无 split   │         │ 非 owner: no-op  │      │  ★ 无 NeoX RoPE             │
   │  (整段 q_nope)│         └──────────────────┘      │                              │
   │              │                                   │  q_idx = rotate_activation   │
   │ q_nope_out = │                                   │    (q_idx)  ★ Hadamard       │
   │  bmm(q_nope, │                                   │  k_idx = rotate_activation   │
   │   w_kc)      │                                   │    (k_idx)                   │
   │  [B,Nh,512]  │                                   │  q_idx_fp8, q_scale =        │
   │              │                                   │    act_quant(q_idx, 128,     │
   │ q_new =      │                                   │      "ue8m0")                │
   │  q_nope_out  │                                   │  k_idx_fp8, k_idx_scale =    │
   │ ★ 无 concat  │                                   │    act_quant(k_idx, ...)     │
   │  [B,Nh,512]  │                                   │  gate = weights_proj(        │
   │              │                                   │    hidden_new) * I^-0.5      │
   │              │                                   │  weights = gate.unsqueeze    │
   │              │                                   │   (-1) * q_scale *           │
   │              │                                   │    Di^-0.5  [B,I,1] fp32     │
   │              │                                   │                              │
   │              │                                   │ **所有卡 append**（镜像）:   │
   │              │                                   │   index_K_pool[slot(pos)] =  │
   │              │                                   │     (k_idx_fp8, k_idx_scale) │
   │              │                                   │   ★ k_idx_fp8 跨卡 bit-exact │
   │              │                                   │     一致 ⇒ pool 跨卡镜像     │
   └──────┬───────┘                                   └─────────────┬────────────────┘
          │                                                         │
          │       (三条并行支线 stream sync)                          │
          └─────────────────────────┬──────────────────────────────┬─┘
                                    │ q_new [B,Nh,512] ★          │
                                    │ q_idx_fp8 [B,1,I,Di]         │
                                    │ q_scale, weights, k_idx_*    │
                                    ▼                              │
                  ┌──────────────────────────────────────────────────┐
              (4)│ F-TOPK-CP   dsa_decode_topk_cp_fused             │ ★ 零通信
                 │ (per rank, 输入完整镜像 index K pool)             │
                 │   logits [B, seqlen] =                            │
                 │     deep_gemm.fp8_paged_mqa_logits(              │
                 │       q_idx_fp8 [B,1,I,Di=128],                  │
                 │       index_K_pool.view(num_pages,64,1,132),     │
                 │       weights, seqlens, page_table,              │
                 │       schedule_md, max_seq_len)                  │
                 │   topk_positions [B, Ktop=2048] =                │
                 │     metadata.topk_transform(logits, Ktop)        │
                 │     ★ 跨卡 bit-exact 一致；无 all-gather         │
                 │   (per rank r) topk_slots_r [B, Ktop] =          │
                 │     map_through_page_table(topk_positions,        │
                 │       latent_KV_page_table_local_r)               │
                 │     ★ 不归 r 的 position → -1                    │
                 └────────────────────┬─────────────────────────────┘
                                      │ topk_slots_r [B,Ktop] (非r=-1)
                                      ▼
                  ┌──────────────────────────────────────────────────┐
              (5)│ F-MQA-PARTIAL                                    │ 各卡本地
                 │ dsa_decode_sparse_mqa_partial_v2                 │
                 │   partial_out_r [B,Nh,Rkv=512],                  │
                 │   partial_lse_r [B,Nh] =                         │
                 │   attn_mqa_partial(                              │
                 │     q_new [B,Nh,512],   ★ K=512                  │
                 │     latent_KV_pool_r,                            │
                 │     topk_slots_r [B,Ktop],                       │
                 │     return_lse=True,                              │
                 │     scaling=Dqk^-0.5=128^-0.5,  ★                │
                 │     num_kv_heads=1, head_dim=512, ★              │
                 │     v_head_dim=Rkv=512)                          │
                 │   ★ -1 槽位: 当 -inf；S_local=∅: lse=-inf        │
                 └────────────────────┬─────────────────────────────┘
                                      │ (partial_out_r, partial_lse_r) per rank
                                      ▼
                  ┌──────────────────────────────────────────────────┐
              (6)│ F-MERGE-POST  ⚙  dsa_cp_merge_post_fused_v2      │ 含通信 + 本地融合
                 │   ★ comm: all-gather (partial_out, partial_lse)   │
                 │     ★ all-gather 版 ~467 KB / 层 / step (v1 526) │
                 │     ★ ring-AR 版    ~58  KB / 层 / step (v1 66)  │
                 │   ★ LSE merge (fp32 in registers/SMEM):           │
                 │     m = max_r partial_lse_r          [B,Nh]      │
                 │     Z = Σ_r exp(partial_lse_r - m)   [B,Nh]      │
                 │     attn_out_latent =                            │
                 │       (Σ_r exp(partial_lse_r - m) *               │
                 │              partial_out_r) / Z                  │
                 │       [B,Nh,Rkv=512]                              │
                 │   ★ V absorb (本地, fp32 stream):                 │
                 │     attn_out = bmm(attn_out_latent.T,             │
                 │       w_vc[Nh,Rkv=512,Dv=128]).T  [B,Nh,Dv=128]   │
                 │   ★ o_proj (本地):                                │
                 │     out = RowParallelLinear(                      │
                 │       attn_out.reshape(B, Nh*Dv))  [B,H=2048]    │
                 │   ⚠ 绝不能用裸 all_reduce(sum)                    │
                 └────────────────────┬─────────────────────────────┘
                                      │ out [B,H=2048] bf16 (所有卡相同)
                                      ▼
                  (MHCHybridNSACPLayerCommunicator.postprocess_layer →
                                MHC 残差合流 → 下一层 input_layernorm)
```

**与 V7 的数据流差异**：

- F-PRE 输出**两件套**（`q_lora` + `latent_cache`/`k_nope` = `k_new`），不再有"576 拆 512 + 64"
  的中间分块。
- F-Q-MAIN 输出 `q_new = q_nope_out`，**`q_pe` 完全不存在**。
- F-IDX 的内部子步骤少一个 RoPE（其它子步骤、调用顺序与 V7 一致）。
- F-TOPK-CP **完全不变**（input shape `[B,1,I,Di=128]` 与 v1 相同）。
- F-MQA-PARTIAL 把 K/Q 维从 576 收缩到 512、scaling 从 `192^-0.5` 改 `128^-0.5`。
- F-MERGE-POST 的 partial shape 从 `[B,Nh,576]` 收到 `[B,Nh,512]`；merge 数学不变。

**正确性硬判据（不变）**：online-softmax merge 在浮点精度内**精确**等于"在完整 top-Ktop 上
做一次 attention"——`--cp 1 / 4 / 16` 端到端 `out [B,H]` 必须逐元素一致。

## Dev 脚本 Stage 顺序

新文件：`zeus_dev/dev_glm_moe_dsa_decode_test_v7_1.py`（绿地，全部 stage 待落地）。
Stage 名称与 V7 同款，**算子内部全部按 v2 shape 实现**（带 `_v2` 后缀以与 V7 区分）。

1. `cp_pre_v2` —— F-PRE。`hidden_new [B,H] → q_lora [B,Rq] + k_nope [B,Rkv=512] + k_new [B,1,512]`。
   验证 fused GEMM + 两个 RMSNorm（**无 concat / 无 split**）。
2. `cp_q_main_v2` —— F-Q-MAIN。`q_lora → q_b_proj + bmm w_kc → q_new [B,Nh,512]`（**不 split / 不 concat**）。
3. `cp_kv_store_v2` —— F-KV-STORE。模拟 cp_size 张卡的 latent KV pool（行宽 512 elem / 1024 B），
   按 owner 规则把 `k_new` 写入对应 rank 的 pool；验证 owner pool 内容、非 owner pool 不变。
4. `cp_idx_v2` —— F-IDX。indexer Q/K + LayerNorm + **★ 跳过 RoPE** + Hadamard + FP8 act_quant
   + gate(fp32) + index K cache append。LayerNorm 含 bias 仍是 V5 的踩坑点。
   **实现**：新写 `IndexerNoRope(Indexer)`，跳过 `rotary_emb` 创建 + `_get_q_k_bf16` 里
   split/rotary_emb/写回三行 —— REPO 路径无 `rope_head_dim==0` 短路分支，直接跑会在
   `get_rope_wrapper(0, ...)` 构造期 fail。
5. `cp_topk_cp_v2` —— F-TOPK-CP。每卡用复制的完整 index K 走 `_get_topk_paged`（**零通信**），
   `topk_positions(cp=k)` 与 `topk_positions(cp=1)` 逐元素一致。
6. `cp_mqa_partial_v2` —— F-MQA-PARTIAL。**head_dim=512、scaling=128^-0.5**，`return_lse=True`；
   单卡退化（cp=1）下 = v2 单卡 D4 输出。
7. `cp_merge_post_v2` —— F-MERGE-POST（⚙ 新算子）。`cp_size` 份 partial all-gather + 本地 LSE merge
   + `bmm w_vc[Nh,512,128]` + `o_proj`。判据：`out(cp=k)` 逐元素 == `out(cp=1)`。
8. `cp_full_path_v2` —— D0..D6 端到端，proxy shape 跑通 `--cp ∈ {1, 4, 16}`；
   **可选**：打开 MHC 包装层（`mhc_num_residual_streams=4` + Sinkhorn `iter=20` + `tau=1.0` +
   `post_mult_value=2` + `hc_eps=1e-6`）验证 `MHCHybridNSACPLayerCommunicator` 兼容性。

每个 stage：REF 侧 pure-torch、`cp_size` 维度用 list 模拟；Zeus 侧标 `× TODO` `raise NotImplementedError`。

---

# 附录

## 附录 A：为什么 decode 选 Route B
与 V7 附录 A 完全一致。v2 的 key 单 head 由 576 elem (1.15 KB) 收到 512 elem (1 KB)，比例
relative to query/output (Nh 头 ≈ 33 KB / token, Nh=32) **更悬殊** —— Route A 更不划算，
Route B 进一步占优。

## 附录 B：与 TP / DP-attention / PP 的关系
与 V7 附录 B 一致。叠 TP 后 `Nh_local = Nh/tp_size`，v2 下 q_b_proj 输出维变 `Nh_local·128`，
o_proj all-reduce 沿 hidden=2048。

## 附录 C：为什么 index K 选"每卡复制"
与 V7 附录 C 一致。v2 下 latent KV 1 KB / token vs index K 132 B / token，index K 占比
~13%（v1 是 ~11.5%），复制它换 top-k 全本地、零通信、几乎照搬单卡 kernel，仍然值得。

## 附录 D：F-MERGE-POST 的 LSE merge 数学
与 V7 附录 D 完全一致。merge 数学跟 RoPE 与否无关，只跟 partial 的 `(out, lse)` 接口有关。

## 附录 E：通信量 / 显存估算（cp=16, 1M context, B=1, v2）

每层每 decode step：

| 项 | 通信 |
|---|---:|
| F-TOPK-CP | **0**（index K 每卡复制 ⇒ 零通信）|
| F-MERGE-POST all-gather (partial_out, partial_lse)（all-gather 版） | ~467 KB（v1 ~526 KB）|
| F-MERGE-POST ring-AR | ~58 KB（v1 ~66 KB）|
| **小计（ring-AR）** | **~58 KB / 层** |

× 6 个 DSA 层（v2 16B，`full_attn_layers=[3,7,11,15,19,23]`）：≈ **~0.35 MB / decode step**。
400 GB/s NVLink ~0.9 µs，可忽略。

per 卡持久 HBM（1M context；latent KV 分片、index K 复制）：

| | latent KV（分片，per 卡，行宽 1024 B）★ | index K（每卡复制全量，132 B）| 合计 / 卡 |
|---|---|---|---|
| **16B v2（6 DSA 层）** | 6·1M·1024 B / 16 ≈ **0.38 GB** | 6·1M·132 B ≈ **0.79 GB** | ≈ **1.17 GB** |
| 16B v1（参考）| 0.43 GB | 0.79 GB | 1.22 GB |

不切 CP 单卡全量：v2 ≈ 6·1M·1024 + 6·1M·132 ≈ 6.91 GB —— **CP 16 卡 + index K 复制
省 ~83% 持久 KV HBM**。latent KV 若按 `fp8_e4m3` 存可再 ÷2。

## 附录 F：开放问题 / 注意点（v2 新增）

- ~~**Indexer RoPE 是否真的跟着 main path 一起消失**~~ ✅ **已确认 / closed**：
  - **语义层面**：config 蕴含的事实，`qk_rope_head_dim=0 → Indexer.rope_head_dim=0` →
    `torch.split` 出 0 维 q_rope/k_rope slice → 写回是 noop ⇒ **无 indexer RoPE**。
    不是 "假设"，是配置链路直接闭合的结论。详见正文 §"v2 关键改动详解 §2"。
  - **实现层面**：REPO 路径 `nsa_indexer.py::_get_q_k_bf16` (line 289-323) 无
    `if rope_head_dim > 0` 短路分支；`Indexer.__init__` 里 `get_rope_wrapper(0, rotary_dim=0, ...)`
    构造期大概率 fail。⇒ Zeus 侧需要新写 `IndexerNoRope` 子类（推荐）或 dev script monkey-patch。
- **`rope_theta=10000` 字段保留无消费者**：v2 配置仍带 `rope_theta`，但全链路无 RoPE 后这个字段
  可能只是历史保留 / 给其他 backend 留口。Zeus 侧不消费即可。
- **MHC 接入边界**：layer communicator 换成 `MHCHybridNSACPLayerCommunicator`，attention 子层
  本身的算子串不变。MHC 的 4 残差流 / Sinkhorn / post-mult 是独立 dev stage，需要：
  - 验证 attention 输出 `out [B, H]` 的 dtype / shape 与 MHC 包装层期望一致；
  - 与 `MHCHybridNSACPLayerCommunicator.prepare_attn` / `postprocess_layer` 的接口对齐
    （prepare 出 `hidden_new`、post 接 `out`）。
- **`mla_nope=True` 在 v2 下的语义**：仍然是"skip rotary"，由于 `Dro=0` 已经从根上消除了 pe，
  `mla_nope` 字段实际成 noop。
- **`v_head_dim=128` + `Rkv=512` 的 `bmm w_vc`**：`w_vc [Nh=32, 512, 128]`，与 v1 完全一致
  （v1 `Dv=128, Rkv=512`），所以 F-MERGE-POST 的 `bmm w_vc` 模板可复用 V7。
- 其余（`return_lse=True` 可用性、owner 切换、`seqlen ≤ Ktop`、B>1）与 V7 附录 F 同。

## 附录 G：参考源码索引（REPO 路径）
与 V7 附录 G 一致。v2 不引入新的源码文件，**只是**：
- `R-MLA` / `R-ABSORB` 在 `Dro=0` 分支上少走两条 split/concat；
- `R-INDEXER` 在 `Dro=0` 时跳过 NeoX RoPE 调用（按字段动态分流的话）；
- `R-COMM` 走 `MHCHybridNSACPLayerCommunicator` 而不是非-MHC 普通版。

---

> 配套文档：
> - V5 `GlmMoeDsa_dev_V5.md` —— 单卡完整链路真值校对版本。
> - V6 `GlmMoeDsa_dev_V6.md` —— Route B 的设计推导 / cost analysis。
> - V7 `GlmMoeDsa_dev_V7.md` —— config_16b（有 RoPE，无 MHC）的工程实现版。
> - **V7.1（本文档）** —— config_16b_v2（**无 RoPE，开 MHC**）的工程实现版；与 V7 共享
>   stage 划分 / 计算流 / 数学推导，仅替换 shape / scaling / 通信量与 F-IDX 的 RoPE 子步骤。

## 开发日志

### 2026-05-12 · 起点
- 基于 V7 创建 V7.1，针对 `config_16b_v2.json`（无 RoPE + MHC）落版。
- 新建 `dev_glm_moe_dsa_decode_test_v7_1.py` 骨架（待落地），所有 stage 后缀 `_v2`。
- 全部 stage `× TODO`，Zeus 路径 `raise NotImplementedError`。
- 下一个 focus：`cp_pre_v2`（F-PRE）REF 跑通 + Zeus fused kernel
  `dsa_pre_attn_qkv_latent_fused_v2` 落地，**关键验证点是 latent_cache 不再做 split**、
  `k_new` 形状是 `[B, 1, 512]`。
- ✅ 已验真：`Indexer.forward_cuda` / `_get_q_k_bf16` **不按字段分流**（硬编码 split + rotary_emb +
  写回），且 `get_rope_wrapper(0, ...)` 构造期会 fail ⇒ `cp_idx_v2` 走"新写 `IndexerNoRope` 子类"策略。
