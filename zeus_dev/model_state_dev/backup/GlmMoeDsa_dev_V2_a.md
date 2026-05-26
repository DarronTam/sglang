# GLM-MoE-DSA / GLM5-Next DSA Zeus 适配开发追踪（V2_a）

> 对齐目标：`Glm5NextForCausalLM`（config `architectures: ["Glm5NextForCausalLM"]`，
> `model_type: "glm4_moe"`）的 DSA full-attention 子层。
>
> 相对 V1（`GlmMoeDsa_dev.md`）的主要修订：
> 1. canonical 配置切到本目录 `config_16b.json`，并补齐 RoPE 字段。
> 2. 显式区分三种 "head" 数：Q 头、潜空间 KV 头、indexer 头；并指出 `num_key_value_heads`
>    在 MLA 路径下不被使用。
> 3. decode 与 prefill **都走 MLA absorb 路径**（吸收矩阵 `w_kc`/`w_vc`），sparse attention
>    发生在 latent 空间；V1 把 D4/P4 写成 `Q [B/T, Nh, Dv]` 的标准 sparse MHA 是错的，
>    本版本重写。
> 4. 新增 v0.5.10-prerelease 中和 GLM-MoE-DSA / `GlmMoeDsaForCausalLM` 直接相关的脚手架引用。

## 1. 范围与方法

- **起点**：DSA layer 输入的 `hidden_states [N,H]`。
- **终点**：`o_proj` 输出 `[N,H]`，可直接进入 post-attention residual / layernorm。
- **切片**：单 device、单 layer、DSA full-attention only；不覆盖 KDA linear attention、
  MoE-FFN、TP/PP/EP、NextN、CP (context parallel)、speculative decoding。
- **层范围**：`linear_attn_config.full_attn_layers = [3, 7, 11, 15, 19, 23]`，
  其余层是 KDA 线性注意（`kda_layers`，受 `gated_attention_layers` 选择 gated KDA）。
- **runtime path 划分**：
  - Decode path：单步 token，历史来自 paged cache。
  - Prefill / extend path：多 token ragged chunk，当前 chunk 批量写 cache。
- **状态约定**：`×` 表示 Zeus DSA kernel 尚未落地或尚未接入 path 级对齐测试；
  后续实现后逐项改为 `△` / `✓`。

## 2. GLM5-Next 16B 关键配置

来自 `./config_16b.json`（`architectures: ["Glm5NextForCausalLM"]`，
`model_type: "glm4_moe"`）。

### 2.1 维度

| 字段 | 值 | 含义 |
|---|---:|---|
| `hidden_size` (H) | 2048 | residual hidden 维度 |
| `num_attention_heads` (Nh) | 32 | **MLA Q 头数**（`q_b_proj` 输出 `Nh*Dqk`） |
| `num_key_value_heads` | 8 | **MLA 路径下不被使用**（见 §2.3） |
| `q_lora_rank` (Rq) | 768 | Q low-rank hidden |
| `kv_lora_rank` (Rkv) | 512 | KV low-rank latent（=潜空间 V dim，见 §3） |
| `qk_nope_head_dim` (Dnope) | 128 | Q/K non-RoPE 维度 |
| `qk_rope_head_dim` (Dro) | 64 | Q/K RoPE 维度 |
| `qk_head_dim` (Dqk) | 192 | `Dnope + Dro` |
| `v_head_dim` (Dv) | 128 | 解吸收后的 value head dim |
| `index_n_heads` (I) | 8 | **indexer 头数**（与 Nh 无关） |
| `index_head_dim` (Di) | 128 | indexer 每 head 维度 |
| `index_topk` (Ktop) | 2048 | 每 query sparse token 数 |
| `index_dsa_use_layernorm` | true | indexer K 上挂 LayerNorm（对齐 SG-I0 中 `k_norm`） |

### 2.2 位置编码 / Norm

`config_16b.json` **不缺 RoPE 配置**，相关字段是：

| 字段 | 值 | 含义 |
|---|---:|---|
| `max_position_embeddings` | 202752 | 长上下文上限 |
| `rope_theta` | 10000 | RoPE 基频 |
| `rope_scaling` | null | **不开 YaRN/线性 scaling** |
| `partial_rotary_factor` | 0.5 | 仅一半 head dim 做旋转（128×0.5 = 64 = `qk_rope_head_dim`） |
| `use_qk_norm` | true | Q/K 在 RoPE 之后做 RMSNorm（GLM-4 系列风格） |
| `rms_norm_eps` | 1e-05 | norm eps |

注意 indexer 共用主路径的 RoPE 配置（`qk_rope_head_dim=64`、`rope_theta=10000`），
对 `[*, I=8, Di=128]` indexer Q 的前 64 维做旋转，对 `[*, Di=128]` indexer K 的前 64 维
做旋转。indexer 的 K 之上额外有 `k_norm`（由 `index_dsa_use_layernorm: true` 触发，对应
SG-I0 中 `Indexer.k_norm`）。

### 2.3 "head 数" 三件套（V1 doc 混淆点）

| 名称 | config 字段 | 值 | 在 MLA/DSA 路径中的角色 |
|---|---|---:|---|
| Q 头 (Nh) | `num_attention_heads` | 32 | `q_b_proj` 输出 reshape 成 `[*, 32, 192]`；`attn_mqa.tp_q_head_num=Nh/tp` |
| GQA KV 头 | `num_key_value_heads` | 8 | **不被 MLA/DSA 路径使用**。只在 `Glm4MoeAttention`（普通 GQA 路径，非 MLA）下当作 KV 头数。`mla: true` 时直接忽略 |
| 潜空间 KV 头 | —（不在 config 里） | 1 | MLA 吸收后真实落到 cache 的 KV 头数。`self.attn_mqa = RadixAttention(num_kv_heads=1, v_head_dim=kv_lora_rank=512)`，即在 `[Rkv+Dro=576]` 维度上做 MQA |
| Indexer 头 (I) | `index_n_heads` | 8 | indexer Q 形状 `[*, 8, 128]`；indexer K 是 `[*, 128]`（共享单 head）。和 Q 头 / KV 头都无关 |

KV cache 实际 layout（`NSATokenToKVPool`，详见 SG-C1）：
- 主 KV：`[num_pages, page_size, 1, Rkv+Dro = 576]`，单 head（即潜空间 KV 头）。
- index K：`[num_pages, page_size, 1, Di = 128]`（uint8 packed FP8 + 4 bytes scale per quant_block）。
  page_size=64（CUDA）或 1（HIP）。

### 2.4 GLM5-Next 特有杂项

| 字段 | 值 | 说明 |
|---|---:|---|
| `num_hidden_layers` | 27 | 总层数 |
| `num_nextn_predict_layers` | 1 | NextN（投机解码）层数；本 dev doc 切片不覆盖 |
| `mla` / `mla_nope` | true / true | 启用 MLA + MLA-nope-only 风格 |
| `multi_query_attention` | true | latent 侧 MQA |
| `use_gated_attention` | true | gated_attention_layers 上启用 gated KDA |
| `linear_attn_config.kda_layers` | 21 层 | KDA 层索引 |
| `linear_attn_config.full_attn_layers` | `[3,7,11,15,19,23]` | DSA full-attn 层索引 |
| `first_k_dense_replace` | 1 | 第 0 层 FFN 是 dense（非 MoE） |
| `n_routed_experts` / `num_experts_per_tok` | 64 / 6 | MoE 路由 |

## 3. 代表性权重（DSA layer）

| 参数 | shape | 备注 |
|---|---:|---|
| `q_a_proj.weight` | `[768, 2048]` | hidden -> q_lora；模型层常和 `kv_a_proj_with_mqa` 合并成 `fused_qkv_a_proj_with_mqa [1344, 2048]`（=`Rq + Rkv + Dro`） |
| `q_a_layernorm.weight` | `[768]` | q_lora norm |
| `q_b_proj.weight` | `[6144, 768]` | q_lora_norm -> `Nh*Dqk = 32*192` |
| `kv_a_proj_with_mqa.weight` | `[576, 2048]` | hidden -> `Rkv + Dro = 512+64` |
| `kv_a_layernorm.weight` | `[512]` | kv_lora norm（只 norm 前 Rkv） |
| `kv_b_proj.weight` | `[8192, 512]` | kv_lora_norm -> `Nh*(Dnope+Dv) = 32*(128+128)`；权重加载后拆成吸收矩阵 `w_kc [Nh, Rkv, Dnope]` 与 `w_vc [Nh, Rkv, Dv]`（见 §4） |
| `o_proj.weight` | `[2048, 4096]` | `Nh*Dv = 32*128` -> hidden |
| `indexer.wq_b.weight` | `[1024, 768]` | q_lora_norm -> `I*Di = 8*128` |
| `indexer.wk.weight` | `[128, 2048]` | hidden -> index key |
| `indexer.weights_proj.weight` | `[8, 2048]` | per-token index head gate（`f32` 计算） |
| `indexer.k_norm.weight` | `[128]` | indexer K 上的 LayerNorm |

## 4. 吸收矩阵（V1 doc 缺失的关键点）

`kv_b_proj.weight` shape `[Nh*(Dnope+Dv), Rkv] = [8192, 512]`，在模型权重加载流程中
**离线拆分**为：

- `w_kc [Nh, Rkv, Dnope] = [32, 512, 128]`（K 吸收矩阵）
- `w_vc [Nh, Rkv, Dv] = [32, 512, 128]`（V 吸收矩阵）

参考 `SG-W0`：`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:565-610`
中 `w_kc, w_vc = w.unflatten(0, (-1, Dnope+Dv)).split([Dnope, Dv], dim=1)`。

DSA full-attn 子层运行时**不会**显式调 `kv_b_proj` 做 Q/K decompress，而是用吸收：

1. **Q 侧 K 吸收** (decode + prefill 都做)：
   `q_nope [T, Nh, Dnope] → bmm(q_nope.T, w_kc) → q_nope_out [T, Nh, Rkv]`
   含义：把 `q_nope` 投影到 `kv_b_proj^K` 的列空间，使后续 attention 直接在
   潜空间 K（cache 里只存 Rkv 维 latent）上做点积，无需把 cache 里的 latent
   反 decompress 成 `[Nh, Dnope]`。
2. **Sparse MQA 在潜空间**：
   `q = concat(q_nope_out, q_pe) [T, Nh, Rkv+Dro=576]`
   attends `kv_cache[topk_slots] [Ktop, 1, Rkv+Dro]`（潜空间 1 个 KV 头，`Rkv` 维 latent 同时充当 K 和 V）
   → `attn_out_latent [T, Nh, Rkv=512]`
3. **V 侧解吸收** (decode + prefill 都做)：
   `attn_out_latent [T, Nh, Rkv] → bmm(attn_out_latent.T, w_vc) → attn_out [T, Nh, Dv=128]`
4. `o_proj [T, Nh*Dv=4096] -> out [T, H=2048]`

参考实现：
- `SG-F0`：`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare`（生成 `q_nope_out`、`k_nope/k_pe`、可选 indexer `topk_indices`）
  与 `forward_absorb_core`（拼 q、调 `attn_mqa(..., topk_indices=...)`、bmm `w_vc`、`o_proj`）。
  decode / prefill 共用同一对函数。
- `SG-D0`：`python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA.dispatch_attn_forward_method`
  + `models/deepseek_common/attention_backend_handler.py::handle_attention_nsa`：
  在 NSA backend 下默认返回 `AttnForwardMethod.MLA`（即 absorb），只有
  `NativeSparseAttnBackend.set_nsa_prefill_impl` 把 `use_mha=True` 时才切到 `MHA_ONE_SHOT`
  （dense fallback，见 P-policy）。

## 5. v0.5.10-prerelease 中的 GLM5-Next 脚手架（V2_a 新增）

`/root/project/sglang-v0.5.10-prerelease` 已经为 `GlmMoeDsaForCausalLM` 留好了挂载点，
但 GLM 侧 decoder layer 还没接入。Zeus 适配时**可以直接用这套 dispatch**，不必重写。

| 标签 | 入口 | 内容 |
|---|---|---|
| `SG-G0` | `python/sglang/srt/models/glm4_moe.py:1417` | `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` 空壳，已加入 `EntryClass`。说明 DSA full-attn 子层期望沿用 `DeepseekV2AttentionMLA` 的实现，GLM 侧只补 KDA + 调度 |
| `SG-G1` | `python/sglang/srt/configs/model_config.py::is_deepseek_nsa` (l.55-78) | architecture 白名单已包含 `"GlmMoeDsaForCausalLM"`；触发条件还需 `index_topk is not None`。`config_16b.json` 的 `architectures` 是 `Glm5NextForCausalLM`，**不在白名单**——这是要么改 config、要么在白名单加 `Glm5NextForCausalLM` 的一个明确取舍点 |
| `SG-G2` | `python/sglang/srt/server_args.py:1541-1567` 注释 `# DeepSeek 3.2/GLM 5` | NSA prefill dense 阈值 = `index_topk`（即 2048）；Blackwell 上强制 sparse MLA（关掉 `MHA_ONE_SHOT`）。即 `dense_fallback_policy` 在 GLM5 上默认会触发到 `max_kv_len <= 2048` 的 prefill |
| `SG-G3` | `python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA` 中 `self.use_nsa = is_deepseek_nsa(config)` (l.1122) + `index_topk_freq / index_topk_pattern` 跨层共享 topk 索引的逻辑 (l.1207-1219) | GLM5 若引入"S/N 模式"（部分层跳过 topk 复用上一层）就走这套；本 dev doc 切片不覆盖跨层 |

## 6. 参考入口（V1 全部沿用 + V2_a 新增）

V1 中 `SG-I0..I3 / SG-A0..A2 / SG-C0 / SG-E0 / SG-T0 / SG-H0 / DS-F0 / DS-T0 / TF-DSA`
全部在 v0.5.10-prerelease 同路径仍然有效（只是 `DeepseekV3AttentionMLAIndexer` 改名
`Indexer`，路径不变）。**只把 base path 从 `/datau38020T/.../ref/sglang` 切到
`/root/project/sglang-v0.5.10-prerelease` 即可**。

V2_a 新增：

| 标签 | 入口 | 用途 |
|---|---|---|
| `SG-F0` | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare` + `forward_absorb_core` | DSA decode + prefill 共用的 absorb 流；含 `bmm(q_nope, w_kc)`、`attn_mqa(..., topk_indices=...)`、`bmm(out_latent, w_vc)` |
| `SG-M0` | `python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA`（`fused_qkv_a_proj_with_mqa [H -> Rq+Rkv+Dro]`、`dispatch_attn_forward_method`、`forward_prepare/core`） | hidden-in fused projection 与 attention dispatch 的官方参照 |
| `SG-D0` | `python/sglang/srt/models/deepseek_common/attention_backend_handler.py::handle_attention_nsa` | NSA backend 强制走 absorb（`AttnForwardMethod.MLA`），prefill dense fallback 由 `NativeSparseAttnBackend.use_mha` 决定 |
| `SG-W0` | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:565-610` | `kv_b_proj` 加载后拆 `w_kc`/`w_vc` 的精确公式与 transpose 约定 |
| `SG-G0..G3` | 见 §5 | GLM5-Next 脚手架（架构白名单、prefill 阈值默认、空壳类） |
| `SG-N0` | `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py::NPUFusedMLAPreprocess.forward_mlapo` | NPU 上把 `fused_qkv_a` + q_a_norm + q_b_proj + kv_a_norm + RoPE + KV cache store 融成 `torch.ops.npu.mla_preprocess` 单算子；Zeus 端 D0+D1(+D2) 融合形态的现成参考 |
| `SG-N1` | `sgl_kernel_npu.norm.fused_split_qk_norm::fused_split_qk_norm`（被 NPU 路径调用） | split + q_a_layernorm + kv_a_layernorm 单 kernel |
| `SG-J0` | `python/sglang/jit_kernel/fused_qknorm_rope.py::fused_qk_norm_rope` + `csrc/elementwise/fused_qknorm_rope.cuh` | JIT CUDA：fused QK RMSNorm + RoPE in-place（支持 partial RoPE / YaRN）；对应 GLM5-Next `use_qk_norm=true` + `partial_rotary_factor=0.5` 的语义 |
| `SG-J1` | `python/sglang/jit_kernel/concat_mla.py` + `csrc/elementwise/concat_mla.cuh` | `concat_mla_absorb_q(q_nope_out, q_pe)`、`concat_mla_k(k, k_nope, k_pe)` 的 JIT 包装，比 SG-E0 多一层 Python entry |
| `SG-T1` | `python/sglang/srt/layers/attention/nsa/transform_index.py::transform_index_page_table_decode_fast` / `transform_index_page_table_prefill_fast` | Triton 版 `fast_topk_transform_*`；Zeus porting 起步友好 |
| `SG-C1` | `python/sglang/srt/mem_cache/memory_pool.py::NSATokenToKVPool` (l.1846-2063) | 主 KV cache + `index_k_with_scale_buffer` 的具体 layout（uint8 packed FP8+scale），`set_index_k_scale_buffer` / `get_index_k_continuous` / `get_index_k_scale_buffer` 入口 |
| `SG-Q0` | `python/sglang/srt/layers/attention/nsa/quant_k_cache.py::quantize_k_cache_separate` + `dequant_k_cache.py::dequantize_k_cache_paged` | M9 FP8 IndexCache：K cache FP8 量化/反量化 Triton |
| `SG-H1` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py::rotate_activation` (l.135) + `sglang.jit_kernel.hadamard.hadamard_transform` | indexer 在 `fp8_mqa_logits` 之前对 q/k 做 Hadamard rotation；**V1 doc 漏写，Zeus 对齐 REF 必须加这步** |

## 7. Decode Path

Decode 的 `N=B`，每个 request 当前只有 1 个 query token；历史 K/V 与 index K 来自 paged cache。
**默认走 MLA absorb 路径**（`SG-D0`+`SG-F0`），sparse MQA attention 在潜空间完成，然后用 `w_vc`
解吸收回 `[B, Nh, Dv]`，最后 `o_proj`。

### 7.1 Decode 计算流（V2_a 修订版）

```
hidden_t [B,H]
  │
  ├─ D0 dsa_qkv_a_proj_norm_fused
  │    hidden_t -> q_lora_norm [B,Rq], kv_lora_norm_t [B,Rkv], k_pe_t [B,Dro]
  │    （ref/upstream 把 q_a + kv_a 合在 fused_qkv_a_proj_with_mqa，
  │      然后分别走 q_a_layernorm / kv_a_layernorm）
  │
  ├─ D1 dsa_q_b_proj_split_rope_absorb
  │    q_lora_norm -> Q [B,Nh,Dqk] -> split (q_nope [B,Nh,Dnope], q_pe [B,Nh,Dro])
  │    RoPE on (q_pe, k_pe_t)；q_nope_out = bmm(q_nope, w_kc) [B,Nh,Rkv]
  │    side effect: main KV cache[new_slots] = concat(kv_lora_norm_t, k_pe_t) [1,Rkv+Dro]
  │
  ├─ D2 dsa_indexer_prep_store_fused
  │    hidden_t + q_lora_norm -> q_idx [B,I,Di] (RoPE on first Dro dims) + gate [B,I]
  │    -> k_idx_t [B,Di] (k_norm + RoPE on first Dro dims)
  │    -> Hadamard rotate on (q_idx, k_idx_t)
  │    -> FP8 quant: q_idx_fp8 [B,I,Di], k_idx_fp8 [B,Di] + scale
  │    side effect: index K cache[new_slots] = (k_idx_fp8, scale_t)
  │
  ├─ D3 dsa_decode_indexer_topk_fused
  │    fp8_paged_mqa_logits(q_idx_fp8, index_k_cache, gate, seqlens, page_table)
  │    -> logits [B, max_kv_len]
  │    -> fast_topk_transform_fused(logits, page_table_1) -> topk_slots [B, Ktop=2048]
  │
  ├─ D4 dsa_decode_sparse_mqa_fused   ★ absorb 后在 latent 空间做 sparse MQA
  │    q = concat(q_nope_out, q_pe) [B,Nh,Rkv+Dro=576]
  │    flash_mla_with_kvcache(q, kv_cache, topk_slots, num_kv_heads=1, d_v=Rkv)
  │    -> attn_out_latent [B,Nh,Rkv=512]
  │
  ├─ D5 dsa_v_absorb
  │    attn_out = bmm(attn_out_latent, w_vc) -> [B,Nh,Dv=128]
  │
  └─ D6 o_proj
       attn_out [B, Nh*Dv=4096] -> out [B,H=2048]
```

### 7.2 Decode 算子依赖表

| # | 子步骤 | shape / IO | 参考实现 | Zeus 状态 |
|---|---|---|---|---|
| D0 | `dsa_qkv_a_proj_norm_fused` | `hidden [B,2048] -> q_lora_norm [B,768] + kv_lora_norm [B,512] + k_pe [B,64]` | **模型层已 fuse 一半**：`SG-M0` 的 `fused_qkv_a_proj_with_mqa` 做单一 GEMM (`[H -> Rq+Rkv+Dro]`)；之后 q_a_layernorm / kv_a_layernorm 分开做。NPU 端 `SG-N0`/`SG-N1` 把这部分进一步融成 `mla_preprocess` / `fused_split_qk_norm` 单 kernel。Zeus 自己 fuse 是合理选项 | × → V1 D0 的 ✓ 实际只覆盖了 q_lora_norm + Q（D1 前半），需重新分块 |
| D1 | `dsa_q_b_proj_split_rope_absorb` | `q_lora_norm [B,768] -> q_nope_out [B,32,512]; q_pe [B,32,64]; k_pe [B,1,64]`；写 main KV cache | `SG-F0::forward_absorb_prepare` 的 `q = q_b_proj(q_lora).view(-1,Nh,Dqk)` -> split -> RoPE -> `bmm(q_nope.T, w_kc)`。cache store 用 `SG-C0::set_mla_kv_buffer_triton`（写入 `concat(k_nope, k_pe) [Rkv+Dro]`） | × |
| D2 | `dsa_indexer_prep_store_fused` | `hidden + q_lora_norm -> q_idx [B,8,128] + gate [B,8]`；写 `index K cache[new_slots]` | `SG-I0::Indexer.forward_indexer` -> `_get_q_k_bf16` + `_project_and_scale_head_gates` + `SG-H1::rotate_activation` + `SG-I3::act_quant`；cache store 走 `SG-I1::fused_store_index_k_cache` -> CUDA `fused_store_indexer_cache`，fallback `SG-I2::SetKAndS` | × |
| D3 | `dsa_decode_indexer_topk_fused` | `q_idx_fp8 + index_k_cache + gate -> topk_slots [B,2048]` | `SG-I0::_get_topk_paged` 调 `deep_gemm.fp8_paged_mqa_logits` -> `metadata.topk_transform` -> `SG-T0::fast_topk_transform_fused`（CUDA `topk_transform_decode_kernel`）。Triton 版见 `SG-T1` | × |
| D4 | `dsa_decode_sparse_mqa_fused` | `q [B,32,576] + main KV cache[topk_slots] -> attn_out_latent [B,32,512]` | `SG-A0::_forward_flashmla_kv` -> `SG-A1::flash_mla_with_kvcache(indices=...)`，底层 `torch.ops.sgl_kernel.fwd_kvcache_mla`。**注意 `num_kv_heads=1`、`d_v=kv_lora_rank=512`**（不是配置里的 `v_head_dim=128`） | × |
| D5 | `dsa_v_absorb` | `attn_out_latent [B,32,512] -> attn_out [B,32,128]` | `SG-F0::forward_absorb_core` 末尾 `bmm(attn_output, w_vc)`（bf16）或 fp8 deep_gemm grouped bmm 路径 | × |
| D6 | `o_proj` | `[B, 4096] -> [B, 2048]` | `RowParallelLinear`（普通 GEMM），DSA 不特殊 | × |

### 7.3 Decode 与 dev 脚本对应

| dev stage | 当前覆盖 | 对应 decode 子步骤 | Zeus 状态 |
|---|---|---|---|
| `decode_qkv_a_proj_norm_fused` （需新增） | hidden -> q_lora_norm + kv_lora_norm + k_pe；REF + Zeus 对齐 | D0 | × |
| `decode_q_proj_fused` | 现行覆盖 q_lora_norm + Q (含 split/RoPE)，**需要扩展加入 `bmm(q_nope, w_kc)` 形成 absorb 后的 `q_nope_out`** | D1 | △（V1 的 ✓ 不完整） |
| `decode_kv_cache_store` | KV latent + k_pe 写 cache；REF；Zeus mode SKIP | D1 cache side-effect | × |
| `decode_indexer_prep_store_fused` | indexer Q/K/gate + Hadamard + FP8 quant + index K cache store | D2 | × |
| `decode_indexer_topk_fused` | paged MQA logits + topk transform | D3 | × |
| `decode_sparse_mqa_fused` | latent 空间 sparse MQA；输出 `[B,Nh,Rkv]` | D4 | × |
| `decode_v_absorb` | `bmm(out_latent, w_vc)`；输出 `[B,Nh,Dv]` | D5 | × |
| `decode_o_proj` | 输出 projection | D6 | × |
| `decode_full_path` | D0-D6 端到端 | D0-D6 | × |

## 8. Prefill / Extend Path

Prefill 的 `N=T=sum(extend_seq_lens)`，多个 request 的新增 token 被展平。每个 query
row 的可见 key 范围 = `prefix(req) + current_chunk(req, <= row_pos)`。

**默认仍走 absorb 路径**（与 decode 共用 `forward_absorb_prepare/core`），sparse 调用换成
`flash_mla_sparse_fwd`。只有当 `max_kv_len <= SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`
（默认 = `index_topk` = 2048，见 `SG-G2`）才切到 `MHA_ONE_SHOT`（dense MHA，不吸收，
直接 decompress KV 做 FlashAttention varlen，见 `SG-A0::_forward_standard_mha`）。

### 8.1 Prefill 计算流（V2_a 修订版）

```
hidden [T,H], positions [T], out_cache_loc [T], ragged metadata
  │
  ├─ P0 dsa_qkv_a_proj_norm_fused          （同 D0 形态，T batched）
  ├─ P1 dsa_q_b_proj_split_rope_absorb     （同 D1 形态；KV cache 批量写 out_cache_loc）
  ├─ P2 dsa_indexer_prep_store_fused       （同 D2 形态，T batched；index K 批量写 out_cache_loc）
  │
  ├─ P3 dsa_prefill_ragged_indexer_topk_fused
  │    fp8_mqa_logits ragged + GetKAndS(index_k_cache + page indices)
  │    -> logits [T, max_kv_len]，可见区由 ragged metadata 限制（prefix + causal 当前 chunk）
  │    -> fast_topk_transform_ragged_fused -> topk_slots [T, Ktop]
  │
  ├─ P4 dsa_prefill_sparse_mqa_fused       ★ absorb 后 sparse MQA（默认）
  │    q = concat(q_nope_out, q_pe) [T,Nh,Rkv+Dro]
  │    flash_mla_sparse_fwd(q, kv_cache, indices=topk_slots, d_v=Rkv)
  │    -> attn_out_latent [T,Nh,Rkv]
  │
  ├─ P4-alt dense_fallback (短序列)        当 max_kv_len <= 2048
  │    走 MHA_ONE_SHOT：kv_b_proj 解 latent -> [T,Nh,Dnope+Dv]
  │    FlashAttention varlen on full prefix+current
  │    -> attn_out_dense [T,Nh,Dv]，**跳过吸收，直接进 P6 o_proj**
  │
  ├─ P5 dsa_v_absorb                       仅 absorb path：bmm(out_latent, w_vc) -> [T,Nh,Dv]
  └─ P6 o_proj                             [T, Nh*Dv] -> [T, H]
```

### 8.2 Prefill 算子依赖表

| # | 子步骤 | shape / IO | 参考实现 | Zeus 状态 |
|---|---|---|---|---|
| P0 | `dsa_qkv_a_proj_norm_fused` | `hidden [T,2048] -> q_lora_norm + kv_lora_norm + k_pe` | 同 D0；`SG-M0`/`SG-N0`/`SG-N1` | × |
| P1 | `dsa_q_b_proj_split_rope_absorb` | `q_lora_norm [T,768] -> q_nope_out [T,32,512] + q_pe [T,32,64] + k_pe [T,1,64]`；批量写 KV cache | 同 D1；`SG-F0::forward_absorb_prepare` + `SG-C0::set_mla_kv_buffer_triton` | × |
| P2 | `dsa_indexer_prep_store_fused` | 同 D2，T batched | 同 D2 | × |
| P3 | `dsa_prefill_ragged_indexer_topk_fused` | `q_idx_fp8 + index_k_cache + ragged metadata -> topk_slots [T,2048]` | `SG-I0::_get_topk_ragged`（`GetKAndS` gather + `deep_gemm.fp8_mqa_logits`）-> `SG-T0::fast_topk_transform_ragged_fused`（CUDA `topk_transform_prefill_ragged_kernel`）。可见区由 `nsa_extend_len_cpu` + `extend_prefix_lens` 决定 | × |
| P4 | `dsa_prefill_sparse_mqa_fused` | `q [T,32,576] + kv_cache + topk_slots -> attn_out_latent [T,32,512]` | `SG-A0::_forward_flashmla_sparse` -> `SG-A2::flash_mla_sparse_fwd`（`torch.ops.sgl_kernel.sparse_prefill_fwd`）；TileLang fallback 见 `DS-T0` 与 `SG-A0::_forward_tilelang` | × |
| P4-alt | `dense_fallback_policy` | `valid_len(row) <= 2048` 切到 dense MHA | `SG-A0::set_nsa_prefill_impl` (use_mha=True) + `_forward_standard_mha`（FlashAttention varlen）。GLM5 默认阈值 = `index_topk = 2048`（`SG-G2`） | × |
| P5 | `dsa_v_absorb` | `[T,32,512] -> [T,32,128]` | `SG-F0::forward_absorb_core` 末段 bmm `w_vc` | × |
| P6 | `o_proj` | 同 D6 | RowParallelLinear | × |

### 8.3 Prefill 与 dev 脚本对应

| dev stage | 当前覆盖 | 对应 prefill 子步骤 | Zeus 状态 |
|---|---|---|---|
| `prefill_qkv_a_proj_norm_fused` （需新增） | D0/P0 形态 batched 版 | P0 | × |
| `prefill_q_proj_fused` | 现行覆盖 q_lora_norm + Q (含 split/RoPE)，需扩展加入 `bmm(q_nope, w_kc)` | P1 | △ |
| `prefill_kv_cache_store` | KV latent + k_pe 批量写 cache | P1 side-effect | × |
| `prefill_indexer_prep_store_fused` | indexer Q/K/gate + Hadamard + FP8 + index K cache store | P2 | × |
| `prefill_ragged_indexer_topk_fused` | ragged causal top-k | P3 | × |
| `prefill_sparse_mqa_fused` | latent 空间 sparse MQA；输出 `[T,Nh,Rkv]` | P4 | × |
| `dense_fallback_policy` | `max_kv_len <= 2048` 时 dense MHA_ONE_SHOT；不吸收 | P4-alt | × |
| `prefill_v_absorb` | `bmm(out_latent, w_vc)` | P5 | × |
| `prefill_o_proj` | output projection | P6 | × |
| `prefill_full_path` | P0-P6 端到端（含 absorb 与 fallback 二选一） | P0-P6 | × |

## 9. Decode / Prefill Slides 对齐

slides 在 V1 时按 "Q proj / KV proj / indexer prep / cache store / topk / sparse attn / o_proj"
分块；V2_a 把这些精确映射到 absorb 后的子步骤。

| slides stage | Decode path | Prefill path | 对齐说明 |
|---|---|---|---|
| Q projection / Batch Q projection | D0 + D1（`fused_qkv_a` 中的 Q part + q_b_proj + split + RoPE + `bmm w_kc`） | P0 + P1 | slides 名义上叫 "Q projection"，实际包含 absorb 的 `bmm w_kc`——slides 需要补图标注 |
| KV projection / Batch KV projection | D0（`fused_qkv_a` 中的 KV part + kv_a_layernorm）+ D1 cache store | P0 + P1 cache store | KV 在 absorb 路径下只走 latent；不再有 `kv_b_proj` 单步 |
| Indexer prep / Batch indexer prep | D2 | P2 | **slides 缺 Hadamard 步骤**，应补 |
| Cache store / Batch cache store | D1/D2 side effects | P1/P2 side effects | decode 单点 append，prefill 批量写 |
| TopK / Ragged topk | D3 (paged) | P3 (ragged) | 两 path 不共用 |
| Sparse attention / Sparse prefill | D4 (sparse MQA, `flash_mla_with_kvcache`) | P4 (sparse MQA, `flash_mla_sparse_fwd`) 或 P4-alt (dense MHA) | **slides 把 sparse attention 画在 `[Nh, Dv]` 维度是错的，应改成 `[Nh, Rkv]`（latent 空间），并在末端加 `v_absorb` 块** |
| V absorb / Batch V absorb（slides 需新增） | D5 | P5 | slides V1 没有这一块 |
| Output / `o_proj` | D6 | P6 | DSA 不特殊 |

待办：`glm_moe_dsa_decode_slides.html` 与 `glm_moe_dsa_prefill_slides.html` 需按上表
增加 "V absorb" 步骤，并把 Sparse attention 的 KV head 数从 32 改成 1、value dim 从 128
改成 512（latent）。

## 10. 里程碑（V2_a 调整）

V1 milestone 以 D0/D1/D2 + topk + sparse attn 的命名为粒度；V2_a 按吸收路径重新切片。

| Milestone | 目标 | 通过标准 |
|---|---|---|
| M0 | 文档 + pure-torch REF 脚本 | `dev_glm_moe_dsa_{decode,prefill}_test.py --mode ref` PASS；REF 内部已经按 §7.1 / §8.1 拆分子步骤，并显式生成中间张量 `q_nope_out`、`q_pe`、`k_nope/k_pe`、`attn_out_latent` |
| M1 | D0/P0 hidden-in fused projection | `*_qkv_a_proj_norm_fused --mode zeus` 对齐 REF（`q_lora_norm` + `kv_lora_norm` + `k_pe`） |
| M2 | D1/P1 Q 通路含 absorb | `*_q_proj_fused --mode zeus` 输出 `q_nope_out [*,Nh,Rkv]` 与 `q_pe`，对齐 REF；KV cache store side-effect 一致 |
| M3 | D2/P2 indexer 通路（含 Hadamard、k_norm、FP8 quant、index cache store） | `*_indexer_prep_store_fused --mode zeus` 输出 `q_idx_fp8`、`k_idx_fp8`、scale、gate；index cache 写入与读回一致 |
| M4 | D3 decode top-k | `decode_indexer_topk_fused --mode zeus` 对齐 REF，支持 padding `-1` |
| M5 | D4 decode sparse MQA | `decode_sparse_mqa_fused --mode zeus` 输出 `[B,Nh,Rkv]` 对齐 REF |
| M6 | D5/P5 V absorb + D6/P6 o_proj | `*_v_absorb` 与 `*_o_proj` Zeus 不再 SKIP |
| M7 | P3 ragged top-k | `prefill_ragged_indexer_topk_fused --mode zeus` 多请求不串行、不看未来、不跨 request |
| M8 | P4 prefill sparse MQA + P4-alt dense fallback | `prefill_sparse_mqa_fused --mode zeus` 对齐 REF；`dense_fallback_policy --mode zeus` 在 `max_kv_len <= 2048` 时与 MHA_ONE_SHOT REF 对齐 |
| M9 | GLM5-Next real shape smoke | DSA layer `[3,7,11,15,19,23]` 单层端到端 `[T,H] -> [T,H]` 对齐 transformers 参考 |
| M10 | FP8 / IndexCache | `SG-Q0` + `SG-C1` 落地；评估 `index_topk_pattern` 跨层共享 |

## 11. V1 → V2_a 主要差异速查

| 项 | V1 | V2_a |
|---|---|---|
| canonical config | `hub/16b_hf/config.json`（外部） | 本目录 `config_16b.json`（`Glm5NextForCausalLM`） |
| "head 数" 表达 | 仅 Nh=32 | 明确区分 Nh=32 / `num_key_value_heads`=8（MLA 不用）/ 潜空间 KV=1 / I=8 |
| RoPE 字段 | 未列出 | 列出 `rope_theta`、`rope_scaling=null`、`partial_rotary_factor=0.5`、`use_qk_norm` |
| Decode/Prefill attention | 描述为 `Q [*,Nh,Dv]` 标准 sparse MHA | 改为潜空间 sparse MQA：`Q [*,Nh,Rkv+Dro] → out_latent [*,Nh,Rkv]`，再 V absorb |
| 吸收矩阵 `w_kc`/`w_vc` | 未提 | 单独章节 §4 与 `SG-W0`/`SG-F0` 引用 |
| Hadamard rotation | 未提 | `SG-H1` 显式列入 D2/P2 |
| GLM5-Next 脚手架 | 未提 | §5 `SG-G0..G3`，含 `GlmMoeDsaForCausalLM` 空壳与 `is_deepseek_nsa` 白名单 |
| Dense fallback 默认阈值 | 通用描述 | 指出 GLM5 默认 = `index_topk=2048`（`SG-G2`） |
| dev stage 粒度 | 6 步 (D0..D5 / P0..P5) | 7 步 (D0..D6 / P0..P6)，新增 V absorb 与 hidden-in fused projection 拆分 |
| ref base path | `/datau38020T/.../ref/sglang` | `/root/project/sglang-v0.5.10-prerelease` |
