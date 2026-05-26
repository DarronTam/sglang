# GLM5-Next / GLM-MoE-DSA Zeus 算子流程文档（V4_b）

> 参考仓库：`/root/project/sglang-feat-v0.5.10-prerelease-glm`
>
> 本文是独立文档，按该仓库中 `Glm5NextForCausalLM` 的真实运行流程梳理 DSA full-attention
> 算子链路。配置基线为本目录 `config_16b.json`
> （`architectures: ["Glm5NextForCausalLM"]`, `model_type: "glm4_moe"`）。

## 1. 范围

- **入口**：GLM5-Next decoder layer 的 attention 输入 `hidden_states [N, H]`。
- **出口**：full-attention 子层 `o_proj` 输出 `[N, H]`，随后进入 layer communicator 的 MLP 前处理。
- **关注层**：`linear_attn_config.full_attn_layers = [3, 7, 11, 15, 19, 23]`。
- **不展开**：KDA 线性注意层、MoE FFN 内部、TP/PP/EP 通信细节、NextN、MHC、Eagle/MTP、NPU 专用 DSA path。
- **运行模式**：Decode、Prefill/Extend、Target verify / Draft extend 的 NSA 共享骨架；Zeus 首轮按 Decode 与普通 Prefill/Extend 对齐即可。

## 2. 真实模型入口

`glm5_next.py` 是 GLM5-Next 在该 repo 中的主入口。

| 层级 | 类 / 函数 | 作用 |
|---|---|---|
| LM 入口 | `Glm5NextForCausalLM` | 注册在 `EntryClass = [Glm5NextForCausalLM]`；初始化 `Glm5NextModel`；`use_nsa = is_deepseek_nsa(config)` |
| Backbone | `Glm5NextModel` | embedding、decoder layers、final RMSNorm、CP split/gather、跨层 `topk_indices` 传递 |
| Decoder layer | `Glm5NextDecoderLayer` | 按 `config.is_kda_layer(layer_id)` 选择 KDA 或 MLA/DSA attention |
| DSA attention | `Glm5NextMLAAttention = DeepseekV2AttentionMLA` | full-attention 层复用 DeepSeek MLA + NSA indexer |
| NSA backend | `NativeSparseAttnBackend` | 初始化 NSA metadata；选择 dense fallback / sparse MLA backend；提供 top-k transform 与 sparse attention kernel |

`is_deepseek_nsa(config)` 已包含 `"Glm5NextForCausalLM"`，触发条件是：

```text
architectures[0] in [
  "DeepseekV3ForCausalLM", "DeepseekV32ForCausalLM",
  "DeepseekV3ForCausalLMNextN", "MistralLarge3ForCausalLM",
  "PixtralForConditionalGeneration", "GlmMoeDsaForCausalLM",
  "Glm5NextForCausalLM"
]
and index_topk is not None
```

因此 `config_16b.json` 在模型判定层面可直接识别为 NSA/DSA，不需要再改 architecture 名。
启动服务时仍要确认 `attention_backend` 进入 `nsa`：该 repo 的
`server_args._handle_model_specific_adjustments()` 自动设置 NSA backend 的 architecture 列表中显式包含
`GlmMoeDsaForCausalLM`，但不在同一处列出 `Glm5NextForCausalLM`。

## 3. 关键配置

### 3.1 DSA / MLA 维度

| 字段 | 值 | 运行时含义 |
|---|---:|---|
| `hidden_size` H | 2048 | residual hidden dim |
| `num_attention_heads` Nh | 32 | MLA Q heads；本地 heads = `Nh / attn_tp_size` |
| `num_key_value_heads` | 8 | DSA/MLA absorb path 不按 GQA KV heads 使用 |
| `q_lora_rank` Rq | 768 | Q low-rank latent |
| `kv_lora_rank` Rkv | 512 | latent K/V dim；cache 中主 KV 的 value dim |
| `qk_nope_head_dim` Dnope | 128 | Q/K no-position segment |
| `qk_rope_head_dim` Dro | 64 | Q/K position segment 宽度 |
| `qk_head_dim` Dqk | 192 | `Dnope + Dro` |
| `v_head_dim` Dv | 128 | `w_vc` 解吸收后的 per-head V dim |
| `index_n_heads` I | 8 | NSA indexer heads |
| `index_head_dim` Di | 128 | indexer head dim |
| `index_topk` Ktop | 2048 | 每个 query 选出的 sparse KV 数 |

### 3.2 层类型

`GlmLinearConfig.is_kda_layer(layer_idx)` 直接检查 `layer_idx in linear_attn_config["kda_layers"]`。

| 层集合 | 值 | attention 类 |
|---|---|---|
| KDA layers | `[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26]` | `Glm5NextLinearAttention` |
| DSA full-attn layers | `[3,7,11,15,19,23]` | `DeepseekV2AttentionMLA` + `Indexer` |

### 3.3 RoPE / nope 约定

`config_16b.json` 中 `mla_nope=true`。`Glm5NextDecoderLayer` 构造 full-attn 时传：

```text
skip_rope = getattr(config, "mla_nope", False)
```

所以主 MLA path 中 `DeepseekV2AttentionMLA.rotary_emb = None`，`q_pe/k_pe` 仍作为
`Dro=64` 的位置段参与 concat/cache，但 SGLang 不在主 MLA path 上调用 RoPE。

Indexer 是另一条路径：`Indexer` 总是构造自己的 `rotary_emb`，对 query/key 的前
`rope_head_dim = Dro = 64` 维做 RoPE，默认：

```text
is_neox_style = not getattr(config, "indexer_rope_interleave", False) = True
```

随后对完整 `Di=128` 向量做 Hadamard rotation，再量化进 FP8 index cache。

## 4. 权重与 cache layout

### 4.1 主要权重

| 参数 | shape | 说明 |
|---|---:|---|
| `fused_qkv_a_proj_with_mqa.weight` | `[Rq + Rkv + Dro, H] = [1344, 2048]` | 权重加载时由 `q_a_proj` 与 `kv_a_proj_with_mqa` 拼接 |
| `q_a_layernorm.weight` | `[768]` | Q latent RMSNorm |
| `q_b_proj.weight` | `[Nh*Dqk, Rq] = [6144, 768]` | Q latent -> `[Nh, 192]` |
| `kv_a_layernorm.weight` | `[512]` | latent K RMSNorm |
| `kv_b_proj.weight` | `[Nh*(Dnope+Dv), Rkv] = [8192, 512]` | 加载后拆为 `w_kc / w_vc` |
| `o_proj.weight` | `[H, Nh*Dv] = [2048, 4096]` | attention output projection |
| `indexer.wq_b.weight` | `[I*Di, Rq] = [1024, 768]` | indexer Q |
| `indexer.wk.weight` | `[Di, H] = [128, 2048]` | indexer K |
| `indexer.weights_proj.weight` | `[I, H] = [8, 2048]` | indexer per-head gate |
| `indexer.k_norm.weight` | `[128]` | indexer K LayerNorm |

### 4.2 `w_kc / w_vc`

`Glm5NextForCausalLM.load_weights()` 在完成 GLM5 权重映射后，调用
`DeepseekV2WeightLoaderMixin.post_load_weights()`，保持与 `DeepseekV2AttentionMLA`
一致的 MLA 后处理。

`kv_b_proj.weight [Nh*(Dnope+Dv), Rkv]` 被离线拆为：

| 矩阵 | logical shape | 运行时用途 |
|---|---:|---|
| `w_kc` | `[Nh_local, Dnope, Rkv]` 或等价转置布局 | `q_nope [N, Nh_local, Dnope] -> q_nope_out [N, Nh_local, Rkv]` |
| `w_vc` | `[Nh_local, Rkv, Dv]` | `attn_out_latent [N, Nh_local, Rkv] -> [N, Nh_local, Dv]` |

### 4.3 主 KV cache

`NSATokenToKVPool` 继承 `MLATokenToKVPool`，主 KV 保存 MLA latent：

```text
main_kv_cache[layer][page, offset, 1, Rkv + Dro]
  = concat(k_nope_norm [Rkv], k_pe [Dro])
  = 512 + 64 = 576
```

注意这是 single latent KV head，不是 8 个 GQA KV heads。

### 4.4 Index K cache

`NSATokenToKVPool.index_k_with_scale_buffer[layer]` 按 page 存储 indexer K：

```text
shape: [num_pages, page_size * (Di + Di / 128 * 4)]
CUDA page_size = 64
Di = 128
每 token: 128 bytes FP8 K + 4 bytes scale = 132 bytes
```

写入优先走 `fused_store_index_k_cache(key, buf, out_cache_loc, page_size)`；
fallback 为 `act_quant(key, block_size=128, scale_fmt="ue8m0")` 后调用
`set_index_k_scale_buffer()`。

## 5. Decoder layer 总流程

每个 `Glm5NextDecoderLayer.forward()` 的 attention 部分按如下顺序执行：

```text
hidden_states, residual
  │
  ├─ layer_communicator.prepare_attn(...)
  │    - 执行 input_layernorm / scatter / collective 相关逻辑
  │    - 对 DSA 层调用 qkv_latent_func = self_attn.prepare_qkv_latent
  │    - prepare_qkv_latent(hidden) -> fused_qkv_a_proj_with_mqa(hidden)
  │    - 结果写入 get_attn_tp_context()，供 attention 内 fetch_qkv_latent()
  │
  ├─ self_attn.forward(...)
  │    - dispatch_attn_forward_method()
  │    - NSA backend 决定 MLA absorb 或 MHA_ONE_SHOT
  │    - full-attn DSA 默认走 MLA absorb
  │
  ├─ layer_communicator.prepare_mlp(...)
  ├─ mlp(...)
  └─ layer_communicator.postprocess_layer(...)
```

因此 DSA 的第一个投影算子不在 `forward_absorb_prepare()` 内直接发起，而是在 layer
communicator 的 pre-attn 阶段通过 `prepare_qkv_latent()` 产生。

## 6. DSA absorb path 算子流

这是 Decode 与普通 Prefill/Extend 的共同主线。`N=B` 表示 decode；`N=T=sum(extend_seq_lens)`
表示 prefill/extend。

```text
hidden [N,H], positions [N], out_cache_loc [N]
  │
  ├─ A0 pre_attn_qkv_latent
  │    fused_qkv_a_proj_with_mqa(hidden)
  │    -> q_raw [N,Rq], latent_cache [N,Rkv+Dro]
  │
  ├─ A1 qkv_a_norm
  │    q = q_a_layernorm(q_raw)                  -> [N,Rq]
  │    k_nope = kv_a_layernorm(latent_cache[:Rkv])-> [N,Rkv]
  │    k_pe = latent_cache[Rkv:]                 -> [N,Dro]
  │
  ├─ A2 q_b_proj_and_k_absorb
  │    q_b_proj(q) -> q_full [N,Nh_local,Dqk]
  │    split q_full -> q_nope [N,Nh_local,Dnope], q_pe [N,Nh_local,Dro]
  │    q_nope_out = bmm(q_nope^T, w_kc)^T -> [N,Nh_local,Rkv]
  │    main MLA RoPE skipped because mla_nope=true
  │
  ├─ A3 indexer_prepare_store_topk
  │    Indexer(hidden, q_lora=q, positions, metadata)
  │    -> topk_indices / transformed page table [N,Ktop]
  │
  ├─ A4 attn_mqa_sparse
  │    attn_mqa(q_nope_out, k_nope, k_nope, q_rope=q_pe, k_rope=k_pe, topk_indices)
  │    - writes main KV cache when save_kv_cache=True
  │    - attention is sparse MQA in latent space
  │    -> attn_out_latent [N,Nh_local,Rkv]
  │
  ├─ A5 v_absorb
  │    bmm(attn_out_latent^T, w_vc)^T -> [N,Nh_local,Dv]
  │    flatten -> [N, Nh_local*Dv]
  │
  └─ A6 o_proj
       RowParallelLinear -> [N,H]
```

## 7. Indexer 子流程

Indexer 由 `DeepseekV2AttentionMLA` 在 `self.use_nsa=True` 时创建。其输入是 attention
输入 `hidden` 与 A1 中保留下来的 `q_lora`。

### 7.1 Query / Key / Gate

```text
q_lora [N,Rq]
  └─ wq_b -> query [N,I,Di]

hidden [N,H]
  ├─ wk -> key [N,Di]
  ├─ k_norm(key)
  └─ weights_proj -> weights [N,I]
       weights *= I^-0.5
```

### 7.2 Indexer RoPE + Hadamard

```text
query[..., :Dro], key[..., :Dro] = indexer.rotary_emb(positions, ...)
query = rotate_activation(query)  # Hadamard, scale=Di^-0.5
key   = rotate_activation(key)
```

Prefill CP 开启时，`key` 会在 RoPE + Hadamard 后通过
`cp_all_gather_rerange_output()` 汇合到完整 indexer K 视图。

### 7.3 FP8 量化与 cache 写入

```text
q_fp8, q_scale = act_quant(query, block_size=128, scale_fmt="ue8m0")
weights = weights.unsqueeze(-1) * q_scale * Di^-0.5

store key:
  preferred: fused_store_index_k_cache(key, index_k_with_scale_buffer, out_cache_loc, page_size)
  fallback : k_fp8, k_scale = act_quant(key, 128, "ue8m0")
             set_index_k_scale_buffer(layer_id, out_cache_loc, k_fp8, k_scale)
```

### 7.4 TopK

Decode / target-verify / draft-extend 走 paged topk：

```text
kv_cache_fp8 = index_k_with_scale_buffer[layer]
logits = deep_gemm.fp8_paged_mqa_logits(
  q_fp8[:, next_n=1, I, Di],
  kv_cache_fp8.view(num_pages, 64, 1, 132),
  weights,
  seqlens,
  block_tables,
  paged_mqa_schedule_metadata,
)
topk_result = metadata.topk_transform(logits, Ktop)
```

Prefill/extend 走 ragged topk：

```text
k_fp8, k_scale = token_to_kv_pool.get_index_k_scale_buffer(...)
logits = deep_gemm.fp8_mqa_logits(q_fp8, (k_fp8, k_scale), weights, ks, ke)
topk_result = metadata.topk_transform(logits, Ktop, ks=ks)
```

当 prefill `max_kv_len <= index_topk` 且未启用 CP 时，Indexer 可跳过 logits 计算：

```text
只计算并写入 index K cache；
用 dummy logits 触发 topk_transform 快速生成 [0..valid_len-1, -1 padding]。
```

## 8. NSA backend 与 sparse attention

`NativeSparseAttnBackend.init_forward_metadata()` 为每个 batch 构造：

| 字段 | 用途 |
|---|---|
| `cache_seqlens_int32` | 当前真实 KV 长度 |
| `page_table_1` | token 粒度 page table |
| `real_page_table` | 按真实 `page_size` 折算后的 page table；CUDA NSA indexer paged path 使用 page_size=64 |
| `nsa_cache_seqlens_int32` | `min(seq_len, index_topk)` 后的 sparse attention KV 长度 |
| `nsa_seqlens_expanded` | prefill 每个 query row 的可见 KV 长度 |
| `topk_indices_offset` | ragged topk 结果转连续 KV 时的 row offset |
| `paged_mqa_schedule_metadata` | decode/verify/draft paged MQA logits 的 DeepGEMM schedule |

### 8.1 Prefill dense / sparse 判定

`set_nsa_prefill_impl()` 在普通 extend 中决定是否走 `MHA_ONE_SHOT`：

```text
self.use_mha =
  device_sm in {90, 100..109}
  and max_kv_len <= SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD
  and kv_cache_dtype in {bfloat16, fp8_e4m3}
  and sum_seq_lens <= forward_batch.get_max_chunk_capacity()
  and not is_nsa_enable_prefill_cp()
  and hisparse_coordinator is None
```

对自动进入 `_handle_model_specific_adjustments()` 的 DSA architecture，server args 会在未手动设置时把
dense threshold 设为模型 `index_topk=2048`。`Glm5NextForCausalLM` 启动时需要显式确认
`attention_backend=nsa` 以及 dense threshold 是否已被设置；Decode / verify 默认不走
`MHA_ONE_SHOT`，始终 `use_mha=False`。

### 8.2 Sparse attention backend

主 sparse MLA 调用由 `RadixAttention` 转到 `NativeSparseAttnBackend.forward_decode()` 或
`forward_extend()`。

| backend | Decode / Prefill sparse kernel | 备注 |
|---|---|---|
| `flashmla_sparse` | `flash_mla_sparse_fwd(q, kv, indices, d_v=Rkv)` | prefill 若 topk method 是 RAGGED，可能直接使用当前 chunk concat KV 或 dequantized paged KV |
| `flashmla_kv` | `flash_mla_with_kvcache(..., indices, head_dim_v=Rkv)` | 要求 indices last dim = `index_topk` |
| `fa3` | `flash_attn_with_kvcache(q_rope, k_cache_rope, v_cache_latent, qv=q_nope)` | 仍输出 latent `Rkv` |
| `tilelang` | `tilelang_sparse_fwd` 或 `tilelang_kernel_glm.sparse_mla_fwd_interface` | `q_all.shape[-1] == v_head_dim` 时识别 no-rope GLM path |
| `trtllm` | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` | fp8 path 可融合 query rope/quant/cache |

所有 sparse MLA backend 在 absorb path 的输出语义都是：

```text
attn_out_latent [N, Nh_local, Rkv]
```

之后必须经过 `w_vc` 解吸收与 `o_proj`。

## 9. Decode 算子链

Decode 中 `N=B`，每个 request 当前 1 个 query token。

| # | 算子 | 输入 -> 输出 | 参考路径 | Zeus 目标 |
|---|---|---|---|---|
| D0 | `dsa_pre_attn_qkv_latent` | `hidden [B,2048] -> q_raw [B,768] + latent_cache [B,576]` | `Glm5NextDecoderLayer.layer_communicator.prepare_attn` + `DeepseekV2AttentionMLA.prepare_qkv_latent` | fused GEMM |
| D1 | `dsa_qkv_a_norm` | `q_raw -> q_lora [B,768]`; `latent[:512] -> k_nope [B,512]`; `latent[512:] -> k_pe [B,64]` | `forward_absorb_prepare` | fused RMSNorm/split |
| D2 | `dsa_q_b_absorb` | `q_lora -> q_nope_out [B,Nh_local,512] + q_pe [B,Nh_local,64]` | `q_b_proj` + `bmm(q_nope,w_kc)` | GEMM + batched GEMM |
| D3 | `dsa_indexer_store_topk_decode` | `hidden + q_lora + index cache -> topk [B,2048]` | `Indexer.forward_cuda` + `_get_topk_paged` | RoPE/Hadamard/FP8/store/topk |
| D4 | `dsa_sparse_mqa_decode` | `q=[q_nope_out,q_pe] + main KV cache + topk -> latent [B,Nh_local,512]` | `NativeSparseAttnBackend.forward_decode` | sparse MLA |
| D5 | `dsa_v_absorb` | `latent [B,Nh_local,512] -> [B,Nh_local,128]` | `forward_absorb_core` | batched GEMM |
| D6 | `dsa_o_proj` | `[B,Nh_local*128] -> [B,2048]` | `RowParallelLinear` | GEMM + reduce as needed |

## 10. Prefill / Extend 算子链

Prefill/extend 中 `N=T=sum(extend_seq_lens)`，metadata 约束每个 query row 只能看见对应 request
的 prefix 与当前 chunk 的 causal 前缀。

| # | 算子 | 输入 -> 输出 | 参考路径 | Zeus 目标 |
|---|---|---|---|---|
| P0 | `dsa_pre_attn_qkv_latent` | `hidden [T,2048] -> q_raw [T,768] + latent_cache [T,576]` | 同 D0 | batched GEMM |
| P1 | `dsa_qkv_a_norm` | `q_lora [T,768] + k_nope [T,512] + k_pe [T,64]` | 同 D1 | fused RMSNorm/split |
| P2 | `dsa_q_b_absorb` | `q_nope_out [T,Nh_local,512] + q_pe [T,Nh_local,64]` | 同 D2 | GEMM + batched GEMM |
| P3 | `dsa_indexer_store_topk_prefill` | `hidden + q_lora + ragged metadata -> topk [T,2048]` | `_get_topk_ragged` 或 `_forward_cuda_k_only` | ragged logits/topk |
| P4 | `dsa_sparse_mqa_prefill` | sparse latent attention -> `[T,Nh_local,512]` | `NativeSparseAttnBackend.forward_extend` | sparse MLA |
| P5 | `dsa_v_absorb` | `[T,Nh_local,512] -> [T,Nh_local,128]` | `forward_absorb_core` | batched GEMM |
| P6 | `dsa_o_proj` | `[T,Nh_local*128] -> [T,2048]` | `RowParallelLinear` | GEMM + reduce as needed |

### Dense fallback

当 `NativeSparseAttnBackend.use_mha=True`，`handle_attention_nsa()` 返回
`AttnForwardMethod.MHA_ONE_SHOT`，attention 走 dense MHA：

```text
q/k/v = forward_normal_one_shot_prepare(...)
flash_attn_varlen_func 或 trtllm_ragged_attention_deepseek
-> [T, Nh_local, Dv]
-> o_proj
```

Dense fallback 不走 `q_nope @ w_kc -> latent attention -> w_vc` 这条 absorb sparse path。
Zeus DSA sparse 算子先按 `use_mha=False` 的主路径实现；dense fallback 可作为独立策略处理。

## 11. CP 与 topk 复用

### 11.1 Prefill CP

`Glm5NextForCausalLM.forward()` 在 `enable_nsa_prefill_context_parallel` 且满足
`can_cp_split()` 时准备 `forward_batch.nsa_cp_metadata`。

`Glm5NextModel.forward()` 会：

```text
hidden_states = cp_split_and_rebuild_data(...)
positions     = cp_split_and_rebuild_position(...)
```

KDA 层暂不支持 CP，进入 KDA 前会 `cp_all_gather_rerange_output()`，KDA 后再 split。
DSA 层的 main KV 与 index K 都有各自的 CP all-gather/rerange 处理。

Zeus 首版可先不实现 CP，但接口设计需要保留：

- `nsa_cp_metadata`
- split 后 `positions`
- indexer K all-gather
- `rebuild_cp_kv_cache(latent_cache, ...)`

### 11.2 跨层 topk 复用

`DeepseekV2AttentionMLA` 支持 `index_topk_freq` / `index_topk_pattern`：

```text
skip_topk: 当前层是否复用上一层 topk_indices
next_skip_topk: 当前层输出是否传给下一层复用
```

`Glm5NextModel.forward()` 在层循环中维护：

```text
topk_indices = None
for layer:
    hidden_states, residual, topk_indices = layer(..., prev_topk_indices=topk_indices)
```

当 `next_skip_topk=True` 时，`forward_absorb_core()` 返回 `(output, topk_indices)`。
Zeus 端如果实现跨层 index cache/复用，必须保持这个返回语义。

## 12. Zeus 开发拆分建议

| 优先级 | 任务 | 验收 |
|---|---|---|
| P0 | 按 `Glm5NextForCausalLM` 真实入口更新 REF：architecture、层分流、`mla_nope=true` 主 MLA 跳过 RoPE | REF 中 DSA full layer 与 prerelease repo 中间张量 shape 对齐 |
| P0 | 修 dev stage tensor contract：sparse MLA 输出是 `[*,Nh_local,Rkv]`，之后显式 `w_vc` 到 `[*,Nh_local,Dv]` | Decode/Prefill pure torch path 端到端对齐 |
| P1 | D0/D1：pre-attn fused qkv latent + q/kv RMSNorm | `q_lora/k_nope/k_pe` 对齐 |
| P1 | D2/P2：`q_b_proj + q_nope @ w_kc`，且不对主 MLA `q_pe/k_pe` 做 RoPE | `q_nope_out/q_pe/k_pe` 对齐 |
| P2 | D3/P3：Indexer RoPE(NeoX) + Hadamard + FP8 quant + index K cache store | index cache 读回与 REF 一致 |
| P2 | Decode paged topk | `fp8_paged_mqa_logits + topk_transform` 支持 `-1` padding |
| P3 | Prefill ragged topk / short prefill skip logits fast path | ragged causal 范围不跨 request、不看未来 |
| P3 | Sparse MLA decode/prefill | 输出 `[*,Nh_local,Rkv]` 对齐 |
| P4 | `w_vc` + `o_proj` | DSA layer `[*,H] -> [*,H]` 对齐 |
| P5 | Dense fallback、CP、跨层 topk 复用 | 分策略补齐 |

## 13. 参考源码索引

| 标签 | 文件 | 关注点 |
|---|---|---|
| `GLM5-M0` | `python/sglang/srt/models/glm5_next.py` | `Glm5NextForCausalLM / Glm5NextModel / Glm5NextDecoderLayer` |
| `GLM5-M1` | `python/sglang/srt/configs/glm_linear.py` | `GlmLinearConfig.is_kda_layer()` 与 full/KDA layer 判定 |
| `DSA-A0` | `python/sglang/srt/models/deepseek_v2.py` | `DeepseekV2AttentionMLA` 初始化、`skip_rope=mla_nope`、Indexer 创建 |
| `DSA-A1` | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` | `forward_absorb_prepare/core` |
| `DSA-W0` | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` | `kv_b_proj` 拆 `w_kc/w_vc` |
| `DSA-I0` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | Indexer Q/K/gate、RoPE、Hadamard、FP8 store、paged/ragged topk |
| `DSA-B0` | `python/sglang/srt/layers/attention/nsa_backend.py` | NSA metadata、MHA_ONE_SHOT 判定、sparse MLA backend |
| `DSA-C0` | `python/sglang/srt/mem_cache/memory_pool.py` | `NSATokenToKVPool` 主 KV 与 index K cache layout |
| `DSA-H0` | `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` | `handle_attention_nsa()` 根据 `backend.use_mha` 选择 MLA / MHA_ONE_SHOT |
| `DSA-S0` | `python/sglang/srt/server_args.py` | NSA backend 默认值、dense fallback threshold 默认设置 |
