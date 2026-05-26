# GLM5-Next / GLM-MoE-DSA Zeus 适配评估 V2_b

> 基于旧版 `zeus_dev/model_state_dev/GlmMoeDsa_dev.md`、本地
> `zeus_dev/model_state_dev/config_16b.json`，以及
> `/root/project/sglang-v0.5.10-prerelease` 的 SGLang 0.5.10 prerelease
> 代码路径重新核对。V2_b 重点修正三件事：
>
> 1. 明确 decode / prefill 的 MLA absorb 路径。
> 2. 区分 query attention head、config 中的 dense/GQA KV head、MLA latent KV head。
> 3. 核对 0.5.10 prerelease 是否有更具体的 GLM5-next 线索与 RoPE 配置。

## 结论摘要

- `config_16b.json` 的 `architectures = ["Glm5NextForCausalLM"]`，但
  SGLang 0.5.10 prerelease 代码中没有 `Glm5NextForCausalLM` 模型类；
  真正接入 DSA/NSA 路径的是 `GlmMoeDsaForCausalLM`。
- prerelease 对 GLM5 的更具体线索主要是：
  - server/parser/test 中出现 `glm5` 命名；
  - `GlmMoeDsaForCausalLM` 被纳入 NSA/MLA 特判；
  - GB300/AMD 测试有 `test_glm5_*`，但模型层仍未提供 `Glm5NextForCausalLM` 类。
- 旧版文档里把 DSA attention 输出写成直接 `Q attends main K/V cache -> [N,32,128]`
  容易误导。SGLang 的 MLA fast path 是 absorb：
  - 先用 `w_kc` 把 `q_nope [N,Nh,128]` 吸收到 latent 维度：
    `q_nope_out [N,Nh,512]`。
  - attention kernel 对 `q_nope_out + q_rope` 与 latent KV cache 做 sparse MLA，
    输出仍是 latent value：`attn_output [N,Nh,512]`。
  - 再用 `w_vc` 把 latent value 投回每 head value：
    `[N,Nh,512] -> [N,Nh,128]`，最后 `o_proj -> [N,2048]`。
- `num_key_value_heads = 8` 不能直接当作 DSA sparse MLA 的 cache head 数。
  在 SGLang MLA absorb path 中，`RadixAttention(attn_mqa)` 使用
  `num_kv_heads=1`，KV cache 是 latent MQA 形态。
- `config_16b.json` 不是没有 RoPE；它有 `rope_theta=10000`、
  `rope_scaling=null`、`max_position_embeddings=202752`、`qk_rope_head_dim=64`。
  结论应写成：有 RoPE 维度与 base theta，但没有动态 rope scaling。

## 配置核对

来源：`zeus_dev/model_state_dev/config_16b.json`

| 字段 | 值 | V2_b 解释 |
|---|---:|---|
| `architectures` | `["Glm5NextForCausalLM"]` | 本地 config 名称；0.5.10 prerelease 未直接注册该类 |
| `model_type` | `glm4_moe` | GLM MoE 系列模型类型 |
| `hidden_size` | 2048 | residual hidden size |
| `num_attention_heads` | 32 | full-attn / DSA query heads，即 `Nh` |
| `head_dim` | 128 | dense attention 传统 head dim；MLA full-attn 的 QK dim 另看 `qk_*` |
| `num_key_value_heads` | 8 | config 中的 dense/GQA KV heads；不要混到 MLA latent cache head |
| `q_lora_rank` | 768 | Q low-rank hidden |
| `kv_lora_rank` | 512 | MLA latent KV dimension，也是 absorb attention 的 latent value dim |
| `qk_nope_head_dim` | 128 | per-query-head non-RoPE Q/K dim |
| `qk_rope_head_dim` | 64 | per-query-head RoPE dim |
| `qk_head_dim` | 192 | `128 + 64`，SGLang 中由代码计算 |
| `v_head_dim` | 128 | absorb 后投回的 per-query-head value dim |
| `index_n_heads` | 8 | DSA indexer heads |
| `index_head_dim` | 128 | DSA indexer head dim |
| `index_topk` | 2048 | sparse token top-k |
| `linear_num_key_heads` | 32 | KDA / linear attention 路径的 key heads，不属于 DSA full-attn |
| `linear_num_value_heads` | 32 | KDA / linear attention 路径的 value heads，不属于 DSA full-attn |
| `linear_attn_config.full_attn_layers` | `[3,7,11,15,19,23]` | DSA full-attn 层 |
| `linear_attn_config.kda_layers` | 其余 21 层 | KDA linear attention 层，本文只作为边界说明 |
| `rope_theta` | 10000 | RoPE base |
| `rope_scaling` | null | 无 YaRN/dynamic scaling |
| `partial_rotary_factor` | 0.5 | config 保留字段；DSA MLA 主要按 `qk_rope_head_dim=64` 接入 |
| `max_position_embeddings` | 202752 | 长上下文上限 |

### Head 语义纠偏

V2_b 推荐统一以下术语：

| 名称 | 数量 / 维度 | 用途 |
|---|---:|---|
| Query heads (`Nh`) | 32 | `q_b_proj -> [N,32,192]`；attention 输出按 32 个 query head 返回 |
| Dense/GQA KV heads | 8 | `config_16b.json` 字段；SGLang MLA absorb path 不直接用它建 cache |
| MLA latent KV heads | 1 | `attn_mqa` 的 `num_kv_heads=1`；cache head 是 latent MQA |
| MLA latent KV dim | 512 | `kv_lora_rank`；attention 中 value dim 暂为 512 |
| Final per-head V dim | 128 | `v_head_dim`；由 `w_vc` 投影后得到 |
| Indexer heads | 8 | `index_n_heads`；只用于 DSA top-k 检索 |

因此旧版表中的 `Knew [N,32,192]` / `Vnew [N,32,128]` 只适合作为
debug 或 dense-MHA 展开视角。SGLang MLA absorb fast path 更准确的 cache 视角是：

```
latent_cache = [kv_lora_rank=512, qk_rope_head_dim=64]
k_nope       = [N,1,512]
k_rope       = [N,1,64]
kv_cache     = [tokens/pages, h_kv=1, 576]
```

## 0.5.10 prerelease 的 GLM5-next 线索

### 直接命名线索

- `python/sglang/srt/models/glm4_moe.py`
  - 存在 `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`。
  - 没有独立 `Glm5NextForCausalLM`。
- `python/sglang/srt/configs/model_config.py`
  - `is_deepseek_nsa()` 把 `GlmMoeDsaForCausalLM` 纳入 NSA 判断。
  - 判断条件是 architecture 在列表中，且 `index_topk` 存在。
- `python/sglang/srt/server_args.py`
  - 对 `GlmMoeDsaForCausalLM` 启用 `attention_backend="nsa"`。
  - CUDA 下强制 page size 64。
  - Blackwell 上对 `GlmMoeDsaForCausalLM` 强制 prefill 使用 sparse MLA，
    禁用 MHA one-shot fallback。
- 测试/前端命名：
  - `test/registered/gb300/test_glm5_fp8.py`
  - `test/registered/gb300/test_glm5_nvfp4.py`
  - AMD accuracy 下有 `test_glm5_eval_*`
  - parser / function call 支持 `glm5`、`glm5stream`

### 对本地 `config_16b.json` 的影响

本地 config 写的是 `Glm5NextForCausalLM`。如果直接交给 0.5.10 prerelease，
按当前源码搜索结果，它不会命中 `GlmMoeDsaForCausalLM` 的 NSA 特判。
要把 prerelease 路径作为参考或直接跑通，需要确认内部加载层是否会：

1. 把 `Glm5NextForCausalLM` 映射/重写成 `GlmMoeDsaForCausalLM`；
2. 或者通过 `--json-model-override-args` 把 architecture 改为
   `GlmMoeDsaForCausalLM`；
3. 或者在本仓库新增 `Glm5NextForCausalLM` entry class，并复用
   `DeepseekV2ForCausalLM + NSA`。

V2_b 评估时建议把 `GlmMoeDsaForCausalLM` 视为 SGLang 对 GLM5 DSA 的实现入口，
但不要声称 0.5.10 prerelease 已经直接支持 `Glm5NextForCausalLM` 这个类名。

## SGLang MLA/DSA 核心路径

### 模型层入口

| 路径 | 作用 |
|---|---|
| `/root/project/sglang-v0.5.10-prerelease/python/sglang/srt/models/glm4_moe.py` | `GlmMoeDsaForCausalLM` 继承 `DeepseekV2ForCausalLM` |
| `/root/project/sglang-v0.5.10-prerelease/python/sglang/srt/models/deepseek_v2.py` | `DeepseekV2AttentionMLA` 初始化 Q/KV low-rank、indexer、`attn_mqa`、`attn_mha` |
| `/root/project/sglang-v0.5.10-prerelease/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` | absorb prepare/core：`w_kc`、`attn_mqa`、`w_vc`、`o_proj` |
| `/root/project/sglang-v0.5.10-prerelease/python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | DSA indexer Q/K/gate、index K cache、paged/ragged top-k |
| `/root/project/sglang-v0.5.10-prerelease/python/sglang/srt/layers/attention/nsa_backend.py` | decode / prefill sparse MLA dispatch |

### Kernel / op 入口

| 路径 | 作用 |
|---|---|
| `python/sglang/jit_kernel/fused_store_index_cache.py` | index K bf16 -> fp8+scale -> index cache |
| `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` | fused index cache store CUDA |
| `sgl-kernel/python/sgl_kernel/top_k.py` | `fast_topk_transform_fused` / `fast_topk_transform_ragged_fused` |
| `sgl-kernel/csrc/elementwise/topk.cu` | decode/prefill top-k transform kernels |
| `sgl-kernel/python/sgl_kernel/flash_mla.py` | `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` |
| `sgl-kernel/csrc/flashmla_extension.cc` | `fwd_kvcache_mla` / `sparse_prefill_fwd` op 注册 |
| `sgl-kernel/csrc/elementwise/concat_mla.cu` | `concat_mla_k` / `concat_mla_absorb_q` |
| `python/sglang/srt/mem_cache/memory_pool.py` | `NSATokenToKVPool`、MLA cache、index cache |
| `python/sglang/srt/mem_cache/utils.py` | `set_mla_kv_buffer_triton`、FP8 quant store |

## Decode Path V2_b

Decode 的 `N=B`。每个 request 当前 1 个 query token，历史 KV 与 index K 来自 cache。
V2_b 中 decode 要显式关注 absorb 矩阵，因为 SGLang 的 fast path 不是先 materialize
`K [B,32,192]` / `V [B,32,128]` 再做普通 MHA。

### Decode 计算流

```
hidden_t [B,2048]
  │
  ├─ D0 fused_qkv_a_proj_with_mqa / q_a + kv_a
  │    hidden_t -> q_lora [B,768] + latent_cache_new [B,576]
  │    latent_cache_new = kv_lora [B,512] + k_rope_raw [B,64]
  │
  ├─ D1 q_a_layernorm / kv_a_layernorm
  │    q_lora_norm [B,768]
  │    k_nope_new [B,1,512]
  │
  ├─ D2 q_b_proj
  │    q_lora_norm -> q [B,32,192]
  │    q_nope [B,32,128], q_rope [B,32,64]
  │
  ├─ D3 absorb-Q with w_kc
  │    q_nope^T [32,B,128] x w_kc [32,128,512]
  │    -> q_nope_out [B,32,512]
  │
  ├─ D4 RoPE
  │    q_rope [B,32,64], k_rope_new [B,1,64]
  │
  ├─ D5 indexer prep/store
  │    q_idx = indexer.wq_b(q_lora_norm) -> [B,8,128]
  │    index_k = indexer.wk(hidden_t) -> [B,128]
  │    gate = indexer.weights_proj(hidden_t) -> [B,8]
  │    side effect: index K FP8+scale cache[new_slots]
  │
  ├─ D6 indexer top-k
  │    q_idx + gate + index K cache -> topk/page_table_1 [B,2048]
  │
  ├─ D7 MLA cache store
  │    side effect: main MLA KV cache[new_slots] = [k_nope_new, k_rope_new]
  │
  ├─ D8 sparse MLA attention
  │    q_all = concat(q_nope_out [B,32,512], q_rope [B,32,64])
  │    kv_cache h_kv=1, dim=576, topk=2048
  │    -> attn_latent [B,32,512]
  │
  ├─ D9 absorb-output with w_vc
  │    attn_latent^T [32,B,512] x w_vc [32,512,128]
  │    -> attn_value [B,32,128]
  │
  └─ D10 o_proj
       attn_value.flatten [B,4096] -> out [B,2048]
```

### Decode 与 SGLang 0.5.10 对照

| 阶段 | prerelease 参考 | Zeus 关注点 |
|---|---|---|
| D0-D2 low-rank Q/KV + q_b | `DeepseekV2AttentionMLA` + `forward_absorb_prepare` | 旧版 D0/D1 可继续拆，但要保留 `q_lora_norm` 给 indexer |
| D3 `w_kc` absorb-Q | `forward_mla.py` 中 `q_nope @ w_kc` | decode path 必须纳入；否则 Q/K 维度会错 |
| D4 RoPE | `get_rope_wrapper(qk_rope_head_dim=64)` | `rope_scaling=null` 时仍要做 RoPE |
| D5 indexer prep/store | `nsa_indexer.py::_get_q_k_bf16` / `_store_index_k_cache` | indexer 的 heads 是 8，不是 attention heads 32 |
| D6 top-k | `deep_gemm.fp8_paged_mqa_logits` + `fast_topk_transform_fused` | topk 输出可直接是 transformed page table |
| D7 cache store | `nsa_backend.forward_decode` -> `set_mla_kv_buffer` | cache head 为 1，dim 为 512+64 |
| D8 sparse MLA | `nsa_backend._forward_flashmla_kv` 或 `_forward_flashmla_sparse` | `indices` shape 是 `[B,1,topk]` 或 `[B,seq_q,topk]` |
| D9 `w_vc` absorb-output | `forward_absorb_core` 中 `attn_output @ w_vc` | 输出从 latent 512 回到 per-head 128 |
| D10 `o_proj` | `RowParallelLinear` | 普通 GEMM，不是 DSA-specific |

### Decode backend 差异

SGLang 0.5.10 对 NSA decode 支持多种 backend：

| backend | 入口 | 备注 |
|---|---|---|
| `flashmla_kv` | `_forward_flashmla_kv` -> `flash_mla_with_kvcache` | decode sparse MLA 常用；page size 64；`indices.shape[-1] == index_topk` |
| `flashmla_sparse` | `_forward_flashmla_sparse` -> `flash_mla_sparse_fwd` | decode 也可用 sparse prefill kernel 形态 |
| `tilelang` | `_forward_tilelang` | ROCm/gfx95 还有 fused rope+cache 特化 |
| `fa3` | `_forward_fa3` | q_rope + q_nope 分开传给 FA3 |
| `trtllm` | `_forward_trtllm` | FP8 / fused rope 特化 |

Zeus decode kernel 对齐时，不建议只按旧版 D4 的“Q attends K/V cache”描述实现。
更稳妥的 contract 是：

```
input: q_nope_out [B,Nh,512], q_rope [B,Nh,64],
       kv_cache [pages/tokens,h_kv=1,576], topk/page_table
output: attn_latent [B,Nh,512]
post: attn_latent @ w_vc -> [B,Nh,128] -> o_proj
```

## Prefill / Extend Path V2_b

Prefill 的 `N=T=sum(extend_seq_lens)`，多个 request 展平成 ragged batch。Prefill
同样走 absorb；差异在于 top-k 与 sparse attention 的 metadata 是 ragged causal span。

### Prefill 计算流

```
hidden_states [T,2048], positions [T], out_cache_loc [T]
  │
  ├─ P0 fused_qkv_a_proj_with_mqa / q_a + kv_a
  │    hidden -> q_lora [T,768] + latent_cache_new [T,576]
  │
  ├─ P1 q_a_layernorm / kv_a_layernorm
  │    q_lora_norm [T,768]
  │    k_nope_new [T,1,512]
  │
  ├─ P2 q_b_proj
  │    q [T,32,192] -> q_nope [T,32,128], q_rope [T,32,64]
  │
  ├─ P3 absorb-Q with w_kc
  │    q_nope -> q_nope_out [T,32,512]
  │
  ├─ P4 RoPE
  │    q_rope [T,32,64], k_rope_new [T,1,64]
  │
  ├─ P5 indexer prep/store
  │    q_idx [T,8,128], gate [T,8], index_k [T,128]
  │    side effect: index K cache[out_cache_loc]
  │
  ├─ P6 ragged indexer top-k
  │    valid range per row = same request prefix + current chunk causal prefix
  │    -> topk/page_table_1 [T,2048]
  │
  ├─ P7 MLA cache store
  │    side effect: main MLA KV cache[out_cache_loc] = [k_nope_new, k_rope_new]
  │
  ├─ P8 sparse MLA attention
  │    q_all [T,32,576], kv_cache h_kv=1 dim=576, ragged topk
  │    -> attn_latent [T,32,512]
  │
  ├─ P9 absorb-output with w_vc
  │    attn_latent -> attn_value [T,32,128]
  │
  └─ P10 o_proj
       [T,4096] -> [T,2048]
```

### Prefill 与 SGLang 0.5.10 对照

| 阶段 | prerelease 参考 | Zeus 关注点 |
|---|---|---|
| P0-P4 projection / absorb / RoPE | `forward_absorb_prepare` | 和 decode 共享数学结构，只是 `N=T` |
| P5 indexer store | `fused_store_index_k_cache` 或 fallback accessor | 写 `out_cache_loc`，不是 decode append only |
| P6 ragged top-k | `nsa_indexer.py::_get_topk_ragged` | `ks/ke` 保证不跨 request、不看未来 |
| P7 cache store | `nsa_backend.forward_extend` -> `set_mla_kv_buffer` | cache 仍是 h_kv=1 latent MLA |
| P8 sparse attention | `_forward_flashmla_sparse` -> `flash_mla_sparse_fwd` | `indices` shape `[s_q,h_kv=1,topk]` |
| P-policy dense fallback | `set_nsa_prefill_impl` | GLM on Blackwell 强制 sparse MLA；否则 threshold 默认 index_topk |

## RoPE 结论

`config_16b.json` 有 RoPE 相关字段：

```
max_position_embeddings = 202752
qk_rope_head_dim        = 64
partial_rotary_factor   = 0.5
rope_theta              = 10000
rope_scaling            = null
```

V2_b 的判断：

- 不能写“没有 RoPE 部分”。正确说法是：有 RoPE dim 和 theta，但没有
  `rope_scaling` / YaRN scaling。
- SGLang `DeepseekV2AttentionMLA` 对 main MLA RoPE 使用：
  `get_rope_wrapper(qk_rope_head_dim, base=rope_theta, rope_scaling=rope_scaling)`。
- 如果 config 没有 `rope_interleave`，SGLang main MLA 默认
  `is_neox_style = not getattr(config, "rope_interleave", True)`，也就是 `False`。
- Indexer RoPE 另有默认：
  `is_neox_style = not getattr(config, "indexer_rope_interleave", False)`，
  config 缺省时为 `True`。
- 因此 Zeus 对齐时要把 main-attention RoPE 与 indexer RoPE 的 interleave 默认
  分开核对，不能只看 `partial_rotary_factor`。

## V2_b 算子依赖表

| # | 子步骤 | shape / IO | 0.5.10 prerelease 参考 | Zeus 状态建议 |
|---|---|---|---|---|
| A0 | fused A projection | `hidden -> q_lora[*,768] + latent[*,576]` | `fused_qkv_a_proj_with_mqa` | 可作为 D0/P0 前半 |
| A1 | q/kv RMSNorm | `q_lora[*,768]`, `kv_lora[*,512]` | `q_a_layernorm`, `kv_a_layernorm` | 需保留 q_lora_norm 给 indexer |
| A2 | Q B projection | `q_lora_norm -> q[*,32,192]` | `q_b_proj` | 旧 D0/P0 |
| A3 | absorb-Q | `q_nope[*,32,128] -> q_nope_out[*,32,512]` | `w_kc` bmm / deep_gemm | V2_b 新增重点 |
| A4 | RoPE | `q_rope[*,32,64]`, `k_rope[*,1,64]` | `rotary_emb` | config 有 RoPE，无 scaling |
| A5 | indexer prep | `hidden + q_lora_norm -> q_idx[*,8,128], gate[*,8], index_k[*,128]` | `nsa_indexer.py` | heads=8 |
| A6 | index cache store | `index_k bf16 -> fp8+scale cache` | `fused_store_index_cache.py` | 可先独立实现 |
| A7 | decode top-k | paged logits -> transformed page table | `_get_topk_paged`, `fast_topk_transform_fused` | page size 64 on CUDA |
| A8 | prefill top-k | ragged logits -> ragged indices | `_get_topk_ragged`, `fast_topk_transform_ragged_fused` | 关注 ks/ke |
| A9 | MLA cache store | `k_nope[*,1,512] + k_rope[*,1,64]` | `set_mla_kv_buffer` | h_kv=1 |
| A10 | sparse MLA decode | `q_all[B,32,576] + kv_cache h_kv=1 + topk` | `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` | 输出 latent 512 |
| A11 | sparse MLA prefill | `q_all[T,32,576] + ragged kv + topk` | `flash_mla_sparse_fwd` | 输出 latent 512 |
| A12 | absorb-output | `attn_latent[*,32,512] -> [*,32,128]` | `w_vc` bmm / deep_gemm | V2_b 新增重点 |
| A13 | output projection | `[*,4096] -> [*,2048]` | `o_proj` | 普通 GEMM |

## 开发优先级建议

1. 先修正文档/测试 REF 的 tensor contract：
   cache 与 sparse attention 用 latent MLA 视角，避免把 `num_key_value_heads=8`
   当作 cache head。
2. Decode path 优先加入 absorb-Q / absorb-output 两个 stage：
   `q_nope @ w_kc` 与 `attn_latent @ w_vc`。
3. Prefill path 复用同一 absorb 数学，但 top-k metadata 独立测试：
   `ks/ke`、`topk_indices_offset`、prefix sharing、padding `-1`。
4. RoPE 测试至少拆两组：
   main MLA RoPE (`qk_rope_head_dim=64`) 与 indexer RoPE (`index_head_dim=128`
   内前 64 维)。
5. 若要直接复用 0.5.10 prerelease 跑 GLM5-next，需要先解决 architecture 名称：
   `Glm5NextForCausalLM` 与 `GlmMoeDsaForCausalLM` 的映射。

## V1 到 V2_b 的主要变更

- 旧版 `D4/P4 sparse_attn_fused` 改写为 sparse MLA latent attention：
  输入 `q_all [*,32,576]`，输出 `attn_latent [*,32,512]`。
- 新增 `w_kc` / `w_vc` absorb 矩阵说明，decode 与 prefill 都必须关注。
- 明确 `num_key_value_heads=8` 不等于 SGLang MLA cache 的 `h_kv`；
  MLA cache 使用 `h_kv=1`。
- 补充 0.5.10 prerelease 的实际 GLM5 线索：
  代码支持 `GlmMoeDsaForCausalLM`，未直接支持本地 config 的
  `Glm5NextForCausalLM` 类名。
- 修正 RoPE 结论：config 有 RoPE，但 `rope_scaling=null`。
