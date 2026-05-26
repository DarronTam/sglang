# GLM-MoE-DSA / GLM5-Next DSA Zeus 适配开发追踪 (V2_c)

> **对齐目标**：GLM5-Next 内部模型 `Glm5NextForCausalLM` 的 DSA full-attention 子层。
> **参考基准**：当前 checkpoint 暂未发布对应模型代码，本文档结合 `Transformers v5.3.0 GlmMoeDsaAttention`、最新 `sglang v0.5.10-prerelease` (NSA/MLA 底层机制) 以及本地 `config_16b.json` 的权重特征和维度定义进行全面梳理。

## 核心澄清与修正 (V2_c 重点关注)

1. **SGLang v0.5.10-prerelease 针对 GLM5-Next 的线索**：
   - 当前的 prerelease `srt/models/` 目录中**暂未**包含直接命名为 `glm5`、`glm_moe_dsa` 的模型实现。最近似的结构为 `glm4_moe.py` 和用于投机解码的 `glm4_moe_nextn.py`。
   - 但是，GLM5-Next DSA 的底层计算逻辑 (尤其是 MLA 和 Sparse Indexing) 与 SGLang 当前实现的 **DeepSeek NSA / MLA** 逻辑 (`srt/layers/attention/nsa_backend.py` 和 `deepseek_v2.py`) **高度同源**。因此本文档的算子对齐将直接映射到这些 SGLang 内部机制。
2. **Head 数量与配置的澄清 (Attention vs. K/V vs. Indexer)**：
   - 之前的分析曾将 `num_key_value_heads: 8` 误解为普通 GQA 的 K/V heads。
   - **实际情况**：`config_16b.json` 中 `kv_b_proj.weight` 是 `[8192, 512]`，即 `512 -> 32 * (128 + 128)`。这意味着它通过隐式分解（MLA机制）实际上恢复出的是 **32个 KV Heads**，而不是8个。
   - 配置文件中的 `"num_key_value_heads": 8` 极有可能是历史遗留字段，或者是用来映射 **Indexer** 的 head 数量（因为配置中同时明确 `"index_n_heads": 8`）。主 Attention 是以 32 Q Heads 和 32 隐式 KV Heads (MLA) 运行的。
3. **RoPE (旋转位置编码) 的确认**：
   - `config_16b.json` 明确定义了 `"qk_rope_head_dim": 64` 和 `"rope_theta": 10000`。
   - 这表明 DSA 层**绝对包含 RoPE 计算**。RoPE 部分（64维）在解码和计算过程中不能省略，必须在拼接 `q_nope` 和 `k_nope` 前单独处理。
4. **吸收矩阵 (Absorb Matrix) 的应用机制**：
   - 在 **Decode Path** 中，为了避免每次迭代将庞大的 `kv_lora_rank (512)` 展开为 32 个 head 的 `k_nope` 和 `v`，SGLang 采用了**吸收矩阵**的极致优化。
   - **Q的吸收**：SGLang `deepseek_v2.py::forward_absorb_fused_mla_rope_prepare` 会将 `k` 的投影矩阵 `w_kc` (`kv_b_proj` 的 `k_nope` 权重转置) 提前乘到 `q_nope` 上。即：`q_nope_out = q_nope @ W_KC`。
   - 这样 Attention 计算直接演变为：`q_nope_out @ latent_cache^T`，完全省去了 `K` 的展开。
   - **V的吸收**：同理，算出的 Attention Score 与 `latent_cache` 相乘后，再统一乘以 `W_VC` (`kv_b_proj` 的 `v` 权重) 恢复最终输出。

---

## 范围与方法

- **起点**：DSA layer 输入的 `hidden_states [N,H]`。
- **终点**：`o_proj` 输出 `[N,H]`，可直接进入 post-attention residual / layernorm。
- **切片**：单 device、单 layer、DSA full-attention only；不覆盖 KDA linear attention、MoE-FFN、TP/PP/EP、NextN。
- **层范围**：`linear_attn_config.full_attn_layers = [3, 7, 11, 15, 19, 23]`。
- **状态约定**：`×` 表示 Zeus DSA kernel 尚未落地；`△` / `✓` 表示已有对齐。

## GLM5-Next DSA 关键配置

来自 `config_16b.json`：

| 字段 | 值 | 含义与解析 |
|---|---:|---|
| `hidden_size` (H) | 2048 | residual hidden 维度 |
| `num_attention_heads` (Nh) | 32 | 主 attention query head 数 |
| `q_lora_rank` (Rq) | 768 | Q low-rank hidden |
| `kv_lora_rank` (Rkv) | 512 | KV low-rank latent |
| `qk_nope_head_dim` (Dnope) | 128 | Q/K non-RoPE 维度 |
| `qk_rope_head_dim` (Dro) | 64 | Q/K RoPE 维度 (确认启用) |
| `v_head_dim` (Dv) | 128 | value head dim |
| `index_n_heads` (I) | 8 | DSA indexer head 数 (注意：这不是主attention的KV head) |
| `index_head_dim` (Di) | 128 | indexer 每 head 维度 |
| `index_topk` (Ktop) | 2048 | 每 query sparse token 数 |

---

## Decode Path (强调吸收矩阵)

Decode 的 `N=B`，每个 request 当前只有 1 个 query token；历史 K/V 与 index K 来自 paged cache。

### Decode 计算流与吸收矩阵 (Absorb Matrix) 逻辑

```text
hidden_t [B,H]
  │
  ├─ D0 dsa_q_proj_absorb_fused
  │    1. hidden_t -> q_lora_norm [B,Rq]
  │    2. q_lora_norm -> q_nope [B,Nh,Dnope], q_rope [B,Nh,Dro]
  │    3. 吸收矩阵：q_nope_out = q_nope @ W_KC (W_KC为 kv_b_proj 的 k_nope 权重转置)
  │    4. q_rope 正常应用 RoPE
  │
  ├─ D1 dsa_kv_proj_cache_store_fused
  │    hidden_t -> Knew(latent_cache)[B,Rkv] / k_pe(RoPE)[B,Dro]
  │    side effect: cache[new_slots] = latent_cache, k_pe
  │
  ├─ D2 dsa_indexer_prep_store_fused
  │    q_lora_norm -> q_idx [B,I,Di], gate [B,I]
  │    hidden_t -> index_k_new [B,Di]
  │    side effect: index K cache[new_slots] = index_k_new
  │
  ├─ D3 dsa_decode_indexer_topk_fused
  │    q_idx x index_k_cache[history_slots] + gate -> topk_slots [B,Ktop]
  │
  ├─ D4 dsa_decode_sparse_attn_fused (FlashMLA)
  │    直接计算: (q_nope_out @ latent_cache^T) + (q_rope @ k_pe^T)
  │    -> attn_score
  │    -> attn_inter = attn_score @ latent_cache
  │    -> attn_out = attn_inter @ W_VC (吸收矩阵还原)
  │
  └─ D5 o_proj
       attn_out -> out [B,H]
```

### Decode 算子依赖表

| # | 子步骤 | CUDA sgl-kernel / 开源参考 | Zeus 状态 |
|---|---|---|---|
| D0 | `dsa_q_proj_absorb_fused` | 参考 `SG: deepseek_v2.py::forward_absorb_fused_mla_rope_prepare`，完成 `W_KC` 吸收。 | ✓ |
| D1 | `dsa_kv_proj_cache_store_fused` | 直接调用 `memory_pool.py::set_mla_kv_buffer` (底层为 Triton kernel) 保存 512维 latent 和 64维 RoPE。 | × |
| D2 | `dsa_indexer_prep_store_fused` | 参考 `nsa/index_buf_accessor.py::SetKAndS` (Triton) 写入 index cache。 | × |
| D3 | `dsa_decode_indexer_topk_fused` | 参考 `top_k.py::fast_topk_transform_fused` (Paged history logits)。 | × |
| D4 | `dsa_decode_sparse_attn_fused` | 参考 `flash_mla.py::flash_mla_with_kvcache` (底层 `torch.ops.sgl_kernel.fwd_kvcache_mla`)，内置对吸收矩阵的兼容。 | × |
| D5 | `o_proj` | 普通 GEMM。 | × |

---

## Prefill / Extend Path

Prefill 的 `N=T=sum(extend_seq_lens)`，多请求拉平。Prefill 阶段通常**不使用吸收矩阵** (因为序列长时，显式展开 K/V 做 FlashAttention/FlashMLA 的效率更高)。

### Prefill 计算流

```text
hidden_states [T,H]
  │
  ├─ P0 dsa_q_proj_fused
  │    hidden_states -> q_nope [T,Nh,Dnope], q_rope [T,Nh,Dro] (不使用吸收)
  │
  ├─ P1 dsa_kv_proj_cache_store_fused
  │    hidden_states -> latent_cache, k_pe
  │    side effect: 批量写入 main K/V cache
  │
  ├─ P2 dsa_indexer_prep_store_fused
  │    hidden_states -> q_idx, gate, index_k_new
  │    side effect: 批量写入 index K cache
  │
  ├─ P3 dsa_prefill_ragged_indexer_topk_fused
  │    -> topk_slots [T,Ktop] (Ragged causal span)
  │
  ├─ P4 dsa_prefill_sparse_attn_fused
  │    -> attn_out [T,Nh,Dv]
  │
  └─ P5 o_proj
```

### Prefill 算子依赖表

| # | 子步骤 | CUDA sgl-kernel / 开源参考 | Zeus 状态 |
|---|---|---|---|
| P0 | `dsa_q_proj_fused` | 参考 `nsa_indexer.py::_get_q_k_bf16` 及 `concat_mla_absorb_q_kernel` 进行拼接映射，注意保留显式的 `q_nope`。 | ✓ |
| P1 | `dsa_kv_proj_cache_store_fused` | `memory_pool.py::set_mla_kv_buffer` (底层批量写入)。 | × |
| P2 | `dsa_indexer_prep_store_fused` | 参考 `nsa/index_buf_accessor.py` 写入 index cache。 | × |
| P3 | `dsa_prefill_ragged_indexer_topk_fused` | 参考 `top_k.py::fast_topk_transform_ragged_fused` (Ragged logits)。 | × |
| P4 | `dsa_prefill_sparse_attn_fused` | 参考 `flash_mla.py::flash_mla_sparse_fwd` (底层 `torch.ops.sgl_kernel.sparse_prefill_fwd`)。 | × |
| P5 | `o_proj` | 普通 GEMM。 | × |
| P-policy | dense fallback | 当 `max_kv_len <= index_topk` 走标准 MHA fallback。参考 `nsa_backend.py::_forward_standard_mha`。 | × |

## 总结

在 V2_c 版本中，我们理清了最重要的几点：
1. SGLang 官方通过 `NSA/MLA` 模块天然支持了此类架构。GLM5-Next 并没有另起炉灶建立专门的 "GLM-DSA" 内核，直接重用 SGLang 中为 DeepSeek 准备的融合/解耦算子即可。
2. **吸收矩阵（Absorb Matrix）** 是 DECODE Path 的灵魂。通过重构 Q，省掉了庞大且冗余的 K 展开。
3. 不要被 `num_key_value_heads: 8` 误导。真实的 KV 容量其实是由 `kv_lora_rank: 512` 和它的映射层 `kv_b_proj` (`8192` = `32 * 256`) 决定的，其等效于拥有 32 个 KV Heads。`8` 这个值是留给 Indexer Sparse Head 用的。
4. RoPE（64维）在 Q/K 中都是实打实存在的。
