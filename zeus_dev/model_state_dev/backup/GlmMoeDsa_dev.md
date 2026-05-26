# GLM-MoE-DSA / GLM5-Next DSA Zeus 适配开发追踪

> 对齐目标：GLM5-Next 内部模型 `Glm5NextForCausalLM` 的 DSA full-attention
> 子层。当前 checkpoint 未发布模型代码，因此本文档以 Transformers v5.3.0
> `GlmMoeDsaAttention + GlmMoeDsaIndexer`、SGLang NSA/DSA 路径，以及本地
> `hub/16b_hf/config.json` / safetensors 权重命名为参考。

## 范围与方法

- **起点**：DSA layer 输入的 `hidden_states [N,H]`。
- **终点**：`o_proj` 输出 `[N,H]`，可直接进入 post-attention residual / layernorm。
- **切片**：单 device、单 layer、DSA full-attention only；不覆盖 KDA linear attention、
  MoE-FFN、TP/PP/EP、NextN。
- **层范围**：`linear_attn_config.full_attn_layers = [3, 7, 11, 15, 19, 23]`。
- **文档组织**：按 runtime path 拆分：
  - Decode path：单步 token，历史来自 paged cache。
  - Prefill / extend path：多 token ragged chunk，当前 chunk 批量写 cache。
- **状态约定**：`×` 表示 Zeus DSA kernel 尚未落地或尚未接入 path 级对齐测试；
  后续实现后逐项改为 `△` / `✓`。

## GLM5-Next DSA 关键配置

来自 `/datau38020T/Application/tanzh/hf_cache/hub/16b_hf/config.json`：

| 字段 | 值 | 含义 |
|---|---:|---|
| `hidden_size` (H) | 2048 | residual hidden 维度 |
| `num_attention_heads` (Nh) | 32 | main attention head 数 |
| `q_lora_rank` (Rq) | 768 | Q low-rank hidden |
| `kv_lora_rank` (Rkv) | 512 | KV low-rank latent |
| `qk_nope_head_dim` (Dnope) | 128 | Q/K non-RoPE 维度 |
| `qk_rope_head_dim` (Dro) | 64 | Q/K RoPE 维度 |
| `qk_head_dim` (Dqk) | 192 | `Dnope + Dro` |
| `v_head_dim` (Dv) | 128 | value head dim |
| `index_n_heads` (I) | 8 | DSA indexer head 数 |
| `index_head_dim` (Di) | 128 | indexer 每 head 维度 |
| `index_topk` (Ktop) | 2048 | 每 query sparse token 数 |
| `max_position_embeddings` | 202752 | 长上下文上限 |

代表性权重：

| 参数 | shape | 说明 |
|---|---:|---|
| `q_a_proj.weight` | `[768, 2048]` | hidden -> q_lora |
| `q_a_layernorm.weight` | `[768]` | q_lora norm |
| `q_b_proj.weight` | `[6144, 768]` | q_lora_norm -> 32 * 192 |
| `kv_a_proj_with_mqa.weight` | `[576, 2048]` | hidden -> 512 kv_lora + 64 k_rope |
| `kv_a_layernorm.weight` | `[512]` | kv_lora norm |
| `kv_b_proj.weight` | `[8192, 512]` | kv_lora_norm -> 32 * (128 k_nope + 128 v) |
| `o_proj.weight` | `[2048, 4096]` | 32 * 128 -> hidden |
| `indexer.wq_b.weight` | `[1024, 768]` | q_lora_norm -> 8 * 128 |
| `indexer.wk.weight` | `[128, 2048]` | hidden -> index key |
| `indexer.weights_proj.weight` | `[8, 2048]` | per-token index head gate |

## 参考入口

| 标签 | 入口 | 用途 |
|---|---|---|
| `TF-DSA` | `transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py` | `GlmMoeDsaAttention` / `GlmMoeDsaIndexer` 数学结构 |
| `SG-I0` | `/datau38020T/Application/tanzh/project/ref/sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | `DeepseekV3AttentionMLAIndexer._get_q_k_bf16`、`_get_topk_paged`、`_get_topk_ragged`、`_store_index_k_cache` |
| `SG-I1` | `/datau38020T/Application/tanzh/project/ref/sglang/python/sglang/jit_kernel/fused_store_index_cache.py` + `jit_kernel/csrc/nsa/fused_store_index_cache.cuh` | `fused_store_index_k_cache` / `fused_store_indexer_cache`：bf16 index K -> FP8+scale 并写 index cache |
| `SG-I2` | `/datau38020T/Application/tanzh/project/ref/sglang/python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` | `SetKAndS`、`GetKAndS` Triton：index K/scale cache 写入与 gather |
| `SG-I3` | `/datau38020T/Application/tanzh/project/ref/sglang/python/sglang/srt/layers/attention/nsa/triton_kernel.py` / `tilelang_kernel.py` | `act_quant`、`fp8_index` 等 indexer FP8 量化/打分参考 |
| `SG-A0` | `/datau38020T/Application/tanzh/project/ref/sglang/python/sglang/srt/layers/attention/nsa_backend.py` | `forward_decode` / `forward_extend` dispatch、dense fallback policy、FlashMLA 调用 |
| `SG-C0` | `/datau38020T/Application/tanzh/project/ref/sglang/python/sglang/srt/mem_cache/memory_pool.py` + `srt/mem_cache/utils.py` | `set_mla_kv_buffer`、`set_mla_kv_buffer_triton`、`set_mla_kv_buffer_triton_fp8_quant` |
| `SG-E0` | `/datau38020T/Application/tanzh/project/ref/sglang/sgl-kernel/csrc/elementwise/concat_mla.cu` | `concat_mla_k_kernel`、`concat_mla_absorb_q_kernel` |
| `SG-T0` | `/datau38020T/Application/tanzh/project/ref/sglang/sgl-kernel/python/sgl_kernel/top_k.py` + `sgl-kernel/csrc/elementwise/topk.cu` | `fast_topk_transform_fused` / `fast_topk_transform_ragged_fused`；CUDA `topk_transform_decode_kernel` / prefill kernels |
| `SG-A1` | `/datau38020T/Application/tanzh/project/ref/sglang/sgl-kernel/python/sgl_kernel/flash_mla.py::flash_mla_with_kvcache` | decode sparse MLA，底层 `torch.ops.sgl_kernel.fwd_kvcache_mla` |
| `SG-A2` | `/datau38020T/Application/tanzh/project/ref/sglang/sgl-kernel/python/sgl_kernel/flash_mla.py::flash_mla_sparse_fwd` | prefill sparse MLA，底层 `torch.ops.sgl_kernel.sparse_prefill_fwd` |
| `SG-H0` | `/datau38020T/Application/tanzh/project/ref/sglang/sgl-kernel/include/sgl_kernel_ops.h` | `concat_mla_*`、`fast_topk_*`、FlashMLA op 声明入口 |
| `DS-F0` | `https://github.com/deepseek-ai/FlashMLA` | FlashMLA sparse decode/prefill kernel 形态 |
| `DS-T0` | `https://github.com/deepseek-ai/TileKernels` | TileLang sparse attention 分块参考 |

## Decode Path

Decode 的 `N=B`，每个 request 当前只有 1 个 query token；历史 K/V 与 index K
来自 paged cache。本文与
`glm_moe_dsa_decode_slides.html` 对齐，按 fused-based runtime 流程描述。

### Decode 计算流

```
hidden_t [B,H]
  │
  ├─ D0 dsa_q_proj_fused
  │    hidden_t -> q_lora_norm [B,Rq], Q [B,Nh,Dqk]
  │
  ├─ D1 dsa_kv_proj_cache_store_fused
  │    hidden_t -> Knew/Vnew
  │    side effect: main K/V cache[new_slots] = Knew/Vnew
  │
  ├─ D2 dsa_indexer_prep_store_fused
  │    hidden_t + q_lora_norm -> q_idx [B,I,Di], gate [B,I], index_k_new [B,Di]
  │    side effect: index K cache[new_slots] = index_k_new
  │
  ├─ D3 dsa_decode_indexer_topk_fused
  │    q_idx x index_k_cache[history_slots] + gate
  │    -> topk_local/topk_slots [B,Ktop]
  │
  ├─ D4 dsa_decode_sparse_attn_fused
  │    Q attends main K/V cache[topk_slots]
  │    -> attn_out [B,Nh,Dv]
  │
  └─ D5 o_proj
       attn_out [B,Nh*Dv] -> out [B,H]
```

### Decode 算子依赖表

| # | 子步骤 | shape / IO | CUDA sgl-kernel / 开源参考 | Zeus 状态 | dev stage 对应 |
|---|---|---|---|---|---|
| D0 | `dsa_q_proj_fused` | `hidden_t [B,2048] -> q_lora_norm [B,768] + Q [B,32,192]` | **无直接单 kernel**。0.5.11 upstream 是模型层 GEMM/RMSNorm/RoPE glue；最近似参考为 `SG-I0::_get_q_k_bf16` 的 `wq_b(q_lora)` + RoPE，以及 `SG-E0::concat_mla_absorb_q_kernel` 只做 `q_nope+q_rope` 拼接。Zeus 已新增 hidden-in fused Q path，当前 sim.c 执行、Triton 参考用于后续 porting。 | ✓ | `decode_q_proj_fused --mode zeus` |
| D1 | `dsa_kv_proj_cache_store_fused` | `hidden_t -> main K/V cache[new_slots]`；debug 可返回 `Knew [B,32,192]`, `Vnew [B,32,128]` | **部分对应**。cache store 参考 `SG-A0::forward_decode` 调 `token_to_kv_pool.set_mla_kv_buffer`，落到 `SG-C0::set_mla_kv_buffer_triton` / `set_mla_kv_buffer_triton_fp8_quant`；`SG-E0::concat_mla_k_kernel` 只对应 K nope/rope pack。无 hidden projection + cache store 单 kernel。 | × | `decode_kv_proj_cache_store_fused`；Zeus mode 当前 SKIP |
| D2 | `dsa_indexer_prep_store_fused` | `hidden_t + q_lora_norm -> q_idx [B,8,128] + gate [B,8]`；写 `index K cache[new_slots]` | **部分直接对应**。index K store 有直接参考：`SG-I1::fused_store_index_k_cache` -> `fused_store_indexer_cache`，在 `SG-I0::_store_index_k_cache` 中优先调用；fallback 为 `SG-I3::act_quant` + `SG-I2::SetKAndS`。但 `q_idx/gate` projection 与 store 尚未 upstream 融成单 kernel。 | × | `decode_indexer_prep_store_fused`；Zeus mode 当前 SKIP |
| D3 | `dsa_decode_indexer_topk_fused` | `q_idx + gate + index K cache/history_slots -> topk_slots [B,2048]` | **大部分对应**。paged logits 在 `SG-I0::_get_topk_paged` 中由 `deep_gemm.fp8_paged_mqa_logits` 生成；top-k+page-table 变换参考 `SG-T0::fast_topk_transform_fused`，CUDA 为 `topk_transform_decode_kernel`，op 声明在 `SG-H0::fast_topk_transform_interface`。 | × | `decode_indexer_topk_fused`；Zeus mode 当前 SKIP |
| D4 | `dsa_decode_sparse_attn_fused` | `Q + topk_slots + main K/V cache -> attn_out [B,32,128]` | **直接对应 attention kernel 形态**。`SG-A0::_forward_flashmla_kv` 调 `SG-A1::flash_mla_with_kvcache(indices=...)`，底层 `torch.ops.sgl_kernel.fwd_kvcache_mla`；可参考 `DS-F0` 的 FlashMLA sparse decode。 | × | `decode_sparse_attn_fused`；Zeus mode 当前 SKIP |
| D5 | `o_proj` | `attn_out [B,4096] -> out [B,2048]` | **非 DSA-specific**。普通 GEMM / `linear_zeus`；0.5.11 无 DSA 专用 `o_proj` kernel。 | × | `decode_o_proj`；Zeus mode 当前 SKIP |

### Decode 与 dev 脚本对应

| dev stage | 当前覆盖 | 对应 decode 子步骤 | Zeus 状态 |
|---|---|---|---|
| `decode_q_proj_fused` | Zeus `dsa_q_proj_fused` 对齐 REF 的 `q_lora_norm` 与 Q | D0 | ✓ |
| `decode_kv_proj_cache_store_fused` | KV projection + main KV cache store REF；Zeus mode SKIP | D1 | × |
| `decode_indexer_prep_store_fused` | indexer Q/K/gate + index K cache store REF；Zeus mode SKIP | D2 | × |
| `decode_indexer_topk_fused` | decode paged top-k REF；Zeus mode SKIP | D3 | × |
| `decode_sparse_attn_fused` | decode sparse attention REF；Zeus mode SKIP | D4 | × |
| `decode_o_proj` | output projection REF；Zeus mode SKIP | D5 | × |
| `decode_full_path` | D0-D5 端到端 REF；Zeus mode SKIP until D1-D5 land | D0-D5 | × |

## Prefill / Extend Path

Prefill 的 `N=T=sum(extend_seq_lens)`，多个 request 的新增 token 被展平。每个 query
row 的可见 key 范围是 `prefix(req) + current_chunk(req, <= row_pos)`。本文与
`glm_moe_dsa_prefill_slides.html` 对齐，按 fused-based runtime 流程描述。

### Prefill 计算流

```
hidden_states [T,H], positions [T], out_cache_loc [T], ragged metadata
  │
  ├─ P0 dsa_q_proj_fused
  │    hidden_states -> q_lora_norm [T,Rq], Q [T,Nh,Dqk]
  │
  ├─ P1 dsa_kv_proj_cache_store_fused
  │    hidden_states -> Knew/Vnew
  │    side effect: main K/V cache[out_cache_loc] = Knew/Vnew
  │
  ├─ P2 dsa_indexer_prep_store_fused
  │    hidden_states + q_lora_norm -> q_idx [T,I,Di], gate [T,I], index_k_new [T,Di]
  │    side effect: index K cache[out_cache_loc] = index_k_new
  │
  ├─ P3 dsa_prefill_ragged_indexer_topk_fused
  │    per row valid slots = same request prefix + current chunk causal span
  │    -> topk_local/topk_slots [T,Ktop]
  │
  ├─ P4 dsa_prefill_sparse_attn_fused
  │    Q[row] attends main K/V cache[topk_slots[row]]
  │    -> attn_out [T,Nh,Dv]
  │
  └─ P5 o_proj
       attn_out [T,Nh*Dv] -> out [T,H]
```

### Prefill 算子依赖表

| # | 子步骤 | shape / IO | CUDA sgl-kernel / 开源参考 | Zeus 状态 | dev stage 对应 |
|---|---|---|---|---|---|
| P0 | `dsa_q_proj_fused` | `hidden [T,2048] -> q_lora_norm [T,768] + Q [T,32,192]` | **无直接单 kernel**。同 D0：upstream 以模型层 GEMM/RMSNorm/RoPE 组合实现；`SG-I0::_get_q_k_bf16` 和 `SG-E0::concat_mla_absorb_q_kernel` 只能作为拆分参考。Zeus 已新增 hidden-in fused Q path，可直接覆盖 prefill batch `[T,H]`。 | ✓ | `prefill_q_proj_fused --mode zeus` |
| P1 | `dsa_kv_proj_cache_store_fused` | `hidden -> main K/V cache[out_cache_loc]`；debug 可返回 `Knew [T,32,192]`, `Vnew [T,32,128]` | **部分对应**。prefill cache 写入参考 `SG-A0::forward_extend` 调 `SG-C0::set_mla_kv_buffer`，底层为 `set_mla_kv_buffer_triton` / FP8 quant store；`SG-E0::concat_mla_k_kernel` 可参考 K layout pack。无 hidden projection + cache store 单 kernel。 | × | `prefill_kv_proj_cache_store_fused`；Zeus mode 当前 SKIP |
| P2 | `dsa_indexer_prep_store_fused` | `hidden + q_lora_norm -> q_idx [T,8,128] + gate [T,8]`；写 `index K cache[out_cache_loc]` | **部分直接对应**。`SG-I0::_store_index_k_cache` 优先走 `SG-I1::fused_store_index_k_cache`，CUDA `fused_store_indexer_cache` 完成 index K FP8 量化+scale+store；fallback 为 `SG-I3::act_quant` + `SG-I2::SetKAndS`。q_idx/gate 仍需新 fused。 | × | `prefill_indexer_prep_store_fused`；Zeus mode 当前 SKIP |
| P3 | `dsa_prefill_ragged_indexer_topk_fused` | `q_idx + gate + index K cache + ragged metadata -> topk_slots [T,2048]` | **大部分对应**。`SG-I0::_get_topk_ragged` 用 `SG-I2::GetKAndS` gather index cache，再由 `deep_gemm.fp8_mqa_logits` 产生 ragged logits；`SG-T0::fast_topk_transform_ragged_fused` 对应 CUDA `topk_transform_prefill_ragged_kernel`。 | × | `prefill_ragged_indexer_topk_fused`；Zeus mode 当前 SKIP |
| P4 | `dsa_prefill_sparse_attn_fused` | `Q + topk_slots + main K/V cache + ragged metadata -> attn_out [T,32,128]` | **直接对应 attention kernel 形态**。`SG-A0::_forward_flashmla_sparse` 调 `SG-A2::flash_mla_sparse_fwd`，底层 `torch.ops.sgl_kernel.sparse_prefill_fwd`；TileLang fallback/参考见 `DS-T0` 与 `SG-A0::_forward_tilelang`。 | × | `prefill_sparse_attn_fused`；Zeus mode 当前 SKIP |
| P5 | `o_proj` | `attn_out [T,4096] -> out [T,2048]` | **非 DSA-specific**。普通 GEMM / `linear_zeus`；0.5.11 无 DSA 专用 `o_proj` kernel。 | × | `prefill_o_proj`；Zeus mode 当前 SKIP |
| P-policy | dense fallback | `valid_len(row) <= index_topk` 或短序列阈值时可走 dense/MHA one-shot | **直接策略参考**。`SG-I0::forward_cuda` 在 `max_kv_len <= index_topk` 时跳过 logits，只 store index K 并生成顺序 top-k；`SG-A0::set_nsa_prefill_impl` 用 `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` 决定 `MHA_ONE_SHOT`，执行 `SG-A0::_forward_standard_mha`。 | × | `dense_fallback_policy`；Zeus mode 当前 SKIP |

### Prefill 与 dev 脚本对应

| dev stage | 当前覆盖 | 对应 prefill 子步骤 | Zeus 状态 |
|---|---|---|---|
| `prefill_q_proj_fused` | Zeus `dsa_q_proj_fused` 对齐 REF 的 batch `q_lora_norm` 与 Q | P0 | ✓ |
| `prefill_kv_proj_cache_store_fused` | KV projection + main KV cache store REF；Zeus mode SKIP | P1 | × |
| `prefill_indexer_prep_store_fused` | indexer Q/K/gate + index K cache store REF；Zeus mode SKIP | P2 | × |
| `prefill_ragged_indexer_topk_fused` | ragged causal top-k REF；Zeus mode SKIP | P3 | × |
| `prefill_sparse_attn_fused` | prefill sparse attention REF；Zeus mode SKIP | P4 | × |
| `prefill_o_proj` | output projection REF；Zeus mode SKIP | P5 | × |
| `dense_fallback_policy` | dense/MHA fallback policy REF；Zeus mode SKIP | P-policy | × |
| `prefill_full_path` | P0-P5 端到端 REF；Zeus mode SKIP until P1-P5 land | P0-P5 | × |

## Decode / Prefill Slides 对齐

| slides stage | Decode path | Prefill path | 对齐状态 |
|---|---|---|---|
| Q projection / Batch Q projection | D0 `dsa_q_proj_fused` | P0 `dsa_q_proj_fused` | 对齐：均从 `hidden` 产出 `q_lora_norm` 和 `Q` |
| KV projection / Batch KV projection | D1 compute 部分 | P1 compute 部分 | 对齐：slides 拆开讲，kernel 与 cache store 融合 |
| Indexer prep / Batch indexer prep | D2 compute 部分 | P2 compute 部分 | 对齐：均使用 `wq_b(q_lora_norm)` |
| Cache store / Batch cache store | D1/D2 side effect 写 `new_slots` | P1/P2 side effect 写 `out_cache_loc` | 对齐：decode append，prefill 批量写 |
| TopK / Ragged topk | D3 paged history top-k | P3 ragged visible-span top-k | 对齐：两条 path 不共用同一 top-k kernel |
| Sparse attention / Sparse prefill | D4 decode sparse MLA | P4 ragged sparse MLA | 对齐：两条 path 不共用同一 attention kernel |
| Output / `o_proj` | D5 | P5 | 对齐：普通 GEMM，不是 DSA-specific fused kernel |

已同步修正：

- `glm_moe_dsa_decode_slides.html` 与 `glm_moe_dsa_prefill_slides.html` 中 indexer-Q
  输入均写成 `wq_b(q_lora_norm)`。
- prefill slides 的 stage 合同已拆成 `Batch Q projection` 与 `Batch KV projection`，
  与 decode slides 和本文粒度一致。

## 里程碑

| Milestone | 目标 | 通过标准 |
|---|---|---|
| M0 | 文档 + pure-torch REF 脚本 | `dev_glm_moe_dsa_decode_test.py --mode ref` 与 `dev_glm_moe_dsa_prefill_test.py --mode ref` PASS；兼容 wrapper `dev_glm_moe_dsa_test.py --mode ref` |
| M1 | D0/P0 Q projection | `decode_q_proj_fused --mode zeus` 与 `prefill_q_proj_fused --mode zeus` 对齐 REF |
| M2 | Decode D1/D2 cache store | `decode_kv_proj_cache_store_fused` / `decode_indexer_prep_store_fused` Zeus stage 不再 SKIP，写入/读回一致 |
| M3 | Decode D3 top-k | `decode_indexer_topk_fused --mode zeus` 对齐 REF，支持 padding `-1` |
| M4 | Decode D4 sparse attention | `decode_sparse_attn_fused --mode zeus` 对齐 REF |
| M5 | Prefill P1/P2 cache store | `prefill_kv_proj_cache_store_fused` / `prefill_indexer_prep_store_fused` Zeus stage 不再 SKIP，写入/读回一致 |
| M6 | Prefill P3 ragged top-k | `prefill_ragged_indexer_topk_fused --mode zeus` 多请求不串行、不看未来、不跨 request |
| M7 | Prefill P4 sparse attention | `prefill_sparse_attn_fused --mode zeus` 对齐 REF；`dense_fallback_policy` 继续覆盖短序列策略 |
| M8 | GLM5-Next real shape smoke | DSA layer `[3,7,11,15,19,23]` 单层跑通 |
| M9 | FP8 / IndexCache | 评估 FP8 index K cache 与 `index_topk_pattern` |
