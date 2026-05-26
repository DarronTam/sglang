# GLM5-Next DSA（GlmMoeDsa）Zeus 适配开发追踪（V4_a）

> 校对依据：完整仓库 `/root/project/sglang-feat-v0.5.10-prerelease-glm`
> （以下简称 **REPO**）。
> canonical config：本目录 `config_16b.json`
> （`architectures: ["Glm5NextForCausalLM"]`，`model_type: "glm4_moe"`）。
>
> 本文档按 REPO 的真实算子链路重写，是一份完整、独立的开发追踪文档；
> 不再罗列与旧版本的差异。
>
> 状态约定：`×` = Zeus 端尚未落地或尚未接入 path 级对齐测试；
> `△` = 部分落地；`✓` = Zeus kernel 已落地且默认 PASS。

---

## 1. 范围与方法

- **起点**：DSA 全注意力子层（`Glm5NextDecoderLayer.self_attn`）拿到 `hidden_states [N,H]`。
- **终点**：`o_proj` 输出 `[N,H]`，可直接进入 layer communicator 的 post-attention residual / MHC。
- **切片**：单 device、单层、DSA full-attention only。**不覆盖**：KDA 线性注意（`Glm5NextLinearAttention`）、
  MoE-FFN、MHC（multi-head hyper-connection）的具体张量运算、TP/PP/EP/DP、NextN、CP（context parallel）、
  speculative decoding。
- **DSA 层范围**：`linear_attn_config.full_attn_layers = [3, 7, 11, 15, 19, 23]`；
  其余 21 层（`kda_layers`）是 KDA 线性注意，**不在本切片**。
- **runtime path 划分**：
  - **Decode**：每 request 单 query token，历史 K / index K 来自 paged cache（page_size=64）。
  - **Prefill / extend**：ragged chunk，多 token 批量写 cache；可能再走两条 sub-path：
    - **MLA absorb sparse**（默认）；
    - **MHA_ONE_SHOT dense fallback**（短序列 + 特定硬件，见 §10）。

---

## 2. 模型脚手架（REPO 实况）

### 2.1 入口与层构成

| 标签 | 入口（REPO 路径） | 内容 |
|---|---|---|
| `M-ENTRY` | `python/sglang/srt/models/glm5_next.py::Glm5NextForCausalLM`（`EntryClass = [Glm5NextForCausalLM]`） | GLM5-Next 完整实现类（不是空壳）；`config_16b.json` 的 `architectures = ["Glm5NextForCausalLM"]` 直接命中 |
| `M-NSA` | `python/sglang/srt/configs/model_config.py::is_deepseek_nsa` (l.55-79) | architecture 白名单已含 `"Glm5NextForCausalLM"`（与 `"GlmMoeDsaForCausalLM"` 并列）；还需 `index_topk is not None`（GLM5 = 2048）→ 满足，启用 NSA backend |
| `M-LAYER` | `glm5_next.py::Glm5NextDecoderLayer.__init__` | `is_kda_layer(layer_id)` 为真 → `Glm5NextLinearAttention`；否则 → `Glm5NextMLAAttention` |
| `M-MLA` | `glm5_next.py` l.99：`from ...models.deepseek_v2 import DeepseekV2AttentionMLA as Glm5NextMLAAttention` | **DSA 全注意子层直接复用 `DeepseekV2AttentionMLA`**；构造时传 `skip_rope=getattr(config, "mla_nope", False)`（GLM5 = `True`，含义见 §4） |
| `M-INDEXER` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py::Indexer` | `DeepseekV2AttentionMLA.__init__` 在 `use_nsa` 时构造；`index_n_heads / index_head_dim / index_topk` 由 `get_nsa_index_*(config)` 读出 |
| `M-MHC` | `glm5_next.py` + `python/sglang/srt/layers/communicator_mhc.py::MHCLayerCommunicator` | `config.mhc = True`（默认）→ attn / mlp 外包 `HyperConnection`，`LayerCommunicator` 换成 `MHCLayerCommunicator`（CP 时 `MHCHybridNSACPLayerCommunicator`） |
| `M-QKVLATENT` | `glm5_next.py`：`qkv_latent_func=self.self_attn.prepare_qkv_latent`（传给 communicator） | **`fused_qkv_a_proj_with_mqa` 这个 GEMM 在 layer communicator 的 `prepare_attn` 里执行**（见 `communicator.py::fetch_qkv_latent`），attention 模块通过 `get_attn_tp_context().fetch_qkv_latent()` 取结果再 split |

### 2.2 关键运行时事实

- DSA 子层的 MLA forward 路径由 `DeepseekV2AttentionMLA.dispatch_attn_forward_method` 经
  `AttentionBackendRegistry.get_handler("nsa")` 决定，即
  `deepseek_common/attention_backend_handler.py::handle_attention_nsa`：
  读 `NSA backend.use_mha`——`use_mha=True` 返回 `AttnForwardMethod.MHA_ONE_SHOT`，否则 `AttnForwardMethod.MLA`（即 absorb）。
  `use_mha` 的判定见 `nsa_backend.py::set_nsa_prefill_impl`（§10）。
- `index_topk_freq` 默认 1、`index_topk_pattern` 默认 None ⇒ `skip_topk / next_skip_topk` 全 False ⇒
  **本切片内不发生跨层 topk 复用**（每层各自算 topk）。跨层复用是后续优化点（见 §13）。
- 全程**不会**用到 `num_key_value_heads = 8`（那是非-MLA GQA 路径的字段）。

---

## 3. GLM5-Next 16B 关键配置（来自 `config_16b.json`）

### 3.1 维度

| 字段 | 值 | 含义 |
|---|---:|---|
| `hidden_size` (H) | 2048 | residual hidden 维度 |
| `num_attention_heads` (Nh) | 32 | MLA Q 头数（`q_b_proj` 输出 `Nh*Dqk`） |
| `num_key_value_heads` | 8 | **MLA/DSA 路径下完全不用**（仅非-MLA GQA 路径字段） |
| `q_lora_rank` (Rq) | 768 | Q low-rank hidden |
| `kv_lora_rank` (Rkv) | 512 | KV low-rank latent（潜空间同时充当 K 与 V） |
| `qk_nope_head_dim` (Dnope) | 128 | Q/K non-RoPE 维度 |
| `qk_rope_head_dim` (Dro) | 64 | Q/K "rope" 分区维度（GLM5 下不实际旋转，见 §4） |
| `qk_head_dim` (Dqk) | 192 | `Dnope + Dro` |
| `v_head_dim` (Dv) | 128 | 解吸收后的 value head dim |
| `index_head_dim` (Di) | 128 | indexer 每 head 维度 |
| `index_topk` (Ktop) | 2048 | 每 query 选中的 sparse token 数 |
| `index_n_heads` (I) | 8 | indexer 头数（与 Nh 无关；数值上与 `num_key_value_heads` 相同纯属巧合） |
| `index_dsa_use_layernorm` | true | indexer K 上挂 LayerNorm（→ `Indexer.k_norm`，**完整 LayerNorm，含 bias，fp32**） |

### 3.2 位置编码 / Norm

| 字段 | 值 | 说明 |
|---|---:|---|
| `max_position_embeddings` | 202752 | 长上下文上限 |
| `rope_theta` | 10000 | RoPE 基频（main & indexer 共用） |
| `rope_scaling` | null | 不开 YaRN / 线性 scaling |
| `partial_rotary_factor` | 0.5 | 名义上一半 head dim 旋转（128×0.5=64）；**对 main MLA 路径无意义**（被 `mla_nope` 跳过） |
| `mla` / `mla_nope` | true / true | `mla_nope=true` ⇒ `DeepseekV2AttentionMLA(skip_rope=True)` ⇒ `self.rotary_emb = None` ⇒ main MLA 路径**不做 RoPE**（见 §4） |
| `multi_query_attention` | true | latent 侧 MQA（潜空间 1 个 KV 头） |
| `use_qk_norm` | true | GLM-4 GQA 风格字段；**DSA MLA 路径不读它**（MLA 用的是 `q_a_layernorm` / `kv_a_layernorm`） |
| `use_gated_attention` / `gated_attention_layers` | true / 21 层 | gated KDA，作用于 `kda_layers`；不在本切片 |
| `rms_norm_eps` | 1e-05 | norm eps |

### 3.3 其它

| 字段 | 值 | 说明 |
|---|---:|---|
| `num_hidden_layers` | 27 | 总层数 |
| `num_nextn_predict_layers` | 1 | NextN 投机层；不覆盖 |
| `linear_attn_config.kda_layers` | 21 层 | KDA 层索引 |
| `linear_attn_config.full_attn_layers` | `[3,7,11,15,19,23]` | **DSA full-attn 层索引（本切片对象）** |
| `first_k_dense_replace` | 1 | 第 0 层 FFN 为 dense（非 MoE） |
| `n_routed_experts` / `num_experts_per_tok` | 64 / 6 | MoE 路由 |
| `n_shared_experts` | 2 | 共享专家数（注意 `Glm5NextForCausalLM.determine_num_fused_shared_experts` 断言 fused 数 == 1，按需 fuse） |
| `mhc` / `mhc_num_residual_streams` | true / 4 | multi-head hyper-connection |
| `intermediate_size` / `moe_intermediate_size` | 10944 / 1408 | dense FFN / MoE 专家 FFN 尺寸 |

---

## 4. 位置编码：main MLA 不旋转，indexer NeoX 旋转

这是 GLM5-Next 与 DeepSeek-V3.2 最容易踩坑的差别，单列一节。

| 路径 | 字面公式（REPO） | GLM5 实际行为 |
|---|---|---|
| main MLA Q/K RoPE | `DeepseekV2AttentionMLA.__init__`：`if not skip_rope: ... is_neox_style = not getattr(config, "rope_interleave", True)`；GLM5 传入 `skip_rope = config.mla_nope = True` | **`self.rotary_emb = None`** ⇒ `forward_absorb_prepare` / `forward_normal_prepare` 中 `if self.rotary_emb is not None: q_pe, k_pe = self.rotary_emb(...)` 这一支被整段跳过。即 `q_pe [*,Nh,64]`、`k_pe [*,1,64]` 是**未旋转**的原始 slice，原样进 attention 与 cache。（参考 `tilelang_kernel_glm.py:28` 注释 "the 'no RoPE tail' / mla_nope case — all tail ops are skipped"。） |
| Indexer Q/K RoPE | `DeepseekV2AttentionMLA.__init__`：`is_neox_style = not getattr(config, "indexer_rope_interleave", False)` → 默认 `indexer_rope_interleave=False` → `is_neox_style = True`（**NeoX**）；传给 `Indexer(rope_head_dim=qk_rope_head_dim=64, ...)` | indexer **总是**建 `self.rotary_emb`；在 `nsa_indexer.py::_get_q_k_bf16` 中对 indexer Q `[*,I=8,128]`、indexer K `[*,128]` 的**前 64 维**做 NeoX RoPE，后 64 维不动；之后再对整 128 维做 Hadamard rotation，再 FP8 量化。 |

> 结论：Zeus 对齐 REF 时，main MLA 路径**不要**插任何 RoPE；只有 indexer 路径的前 64 维需要 NeoX RoPE。

---

## 5. 代表性权重（单个 DSA 层）

按 REPO 的命名 / shape（TP=1）。注意 `q_a_proj` 与 `kv_a_proj_with_mqa` 在权重加载时会被
`Glm5NextForCausalLM.load_weights` 的 `packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = ["q_a_proj", "kv_a_proj_with_mqa"]`
合并成一张 `fused_qkv_a_proj_with_mqa`。

| 参数 | shape | 备注 |
|---|---:|---|
| `fused_qkv_a_proj_with_mqa.weight` | `[Rq+Rkv+Dro, H] = [1344, 2048]` | hidden → `q_lora ‖ kv_lora ‖ k_pe`；`ReplicatedLinear`（不切 TP） |
| `q_a_layernorm.weight` | `[768]` | q_lora RMSNorm |
| `q_b_proj.weight` | `[Nh*Dqk, Rq] = [6144, 768]` | q_lora_norm → `[*, 32, 192]`；`ColumnParallelLinear` |
| `kv_a_layernorm.weight` | `[512]` | kv_lora RMSNorm（只 norm 前 Rkv） |
| `kv_b_proj.weight` | `[Nh*(Dnope+Dv), Rkv] = [8192, 512]` | `ColumnParallelLinear`；**加载后离线拆为吸收矩阵 `w_kc` / `w_vc`**（§6）。absorb path 运行时**不调** `kv_b_proj` 做 decompress，只有 `MHA_ONE_SHOT` fallback 才调 |
| `o_proj.weight` | `[H, Nh*Dv] = [2048, 4096]` | `RowParallelLinear` |
| `attn_mqa`（RadixAttention 实例） | num_heads=32, head_dim=`Rkv+Dro=576`, scaling=`Dqk**-0.5 = 192**-0.5`, num_kv_heads=1, v_head_dim=`Rkv=512` | 潜空间 sparse MQA；attn 入口接 `topk_indices` kwarg |
| `attn_mha`（RadixAttention 实例） | num_heads=32, head_dim=`Dnope+Dro=192`, num_kv_heads=32, v_head_dim=`Dv=128`；`attn_mha.kv_b_proj = kv_b_proj`（forward 时绑定） | 仅 `MHA_ONE_SHOT` dense fallback 用 |
| `indexer.wq_b.weight` | `[I*Di, Rq] = [1024, 768]` | q_lora_norm → indexer Q `[*, 8, 128]`；`ReplicatedLinear` |
| `indexer.wk.weight` | `[Di, H] = [128, 2048]` | hidden → indexer K `[*, 128]`（单 head 共享）；`ReplicatedLinear` |
| `indexer.weights_proj.weight` | `[I, H] = [8, 2048]` | per-token per-head gate；CUDA 上参数 dtype = bf16，**计算结果转 fp32** |
| `indexer.k_norm.weight` / `.bias` | `[128]` / `[128]` | indexer K 上的**完整 LayerNorm（fp32）**，`index_dsa_use_layernorm: true` 触发 |

---

## 6. 吸收矩阵 `w_kc` / `w_vc`

`kv_b_proj.weight` shape `[Nh*(Dnope+Dv), Rkv] = [8192, 512]`，在
`Glm5NextForCausalLM.load_weights` 末尾 `DeepseekV2WeightLoaderMixin.post_load_weights`
里离线拆分（`deepseek_common/deepseek_weight_loader.py:565-585`）：

```python
w_kc, w_vc = w.unflatten(0, (-1, Dnope + Dv)).split([Dnope, Dv], dim=1)
# w_kc: [Nh, Dnope, Rkv] = [32, 128, 512]
# w_vc: [Nh, Dv,    Rkv] = [32, 128, 512]
self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)   # 仍是 [32, 128, 512]，仅改内存布局
self_attn.w_vc = w_vc.contiguous().transpose(1, 2)                   # → [Nh, Rkv, Dv] = [32, 512, 128]
```

absorb 路径运行时（decode 与 prefill **共用** `forward_absorb_prepare` / `forward_absorb_core`）：

1. **Q 侧 K 吸收**（`forward_absorb_prepare` 末段）：
   `q_nope [T, Nh, Dnope] →(transpose 0,1)→ [Nh, T, Dnope]` `bmm` `w_kc [Nh, Dnope, Rkv]` `→ [Nh, T, Rkv] →(transpose)→ q_nope_out [T, Nh, Rkv=512]`。
2. **拼 latent Q / K**（`forward_absorb_core`）：
   `q = cat(q_nope_out, q_pe) [T, Nh, Rkv+Dro=576]`；`k = cat(k_nope, k_pe) [T, 1, 576]`（`k_nope = kv_a_layernorm 输出.unsqueeze(1)`，`k_pe = latent_cache[..., Rkv:].unsqueeze(1)`，**均未旋转**）。
3. **潜空间 sparse MQA**：`attn_mqa(q, k, k_nope, forward_batch, topk_indices=topk_indices)` —— NSA backend 用 `topk_indices` 在 paged latent cache 上做 sparse MQA，`num_kv_heads=1`、`v_head_dim=Rkv`，输出 `attn_output [T, Nh, Rkv=512]`。
4. **V 侧解吸收**：`attn_output [T, Nh, Rkv] →(transpose 0,1)→ [Nh, T, Rkv]` `bmm` `w_vc [Nh, Rkv, Dv]` `→ [Nh, T, Dv] →` reshape `→ [T, Nh*Dv=4096]`。
5. **o_proj**：`[T, 4096] → out [T, H=2048]`。

---

## 7. Indexer 内部流程（`nsa_indexer.py::Indexer.forward_cuda`）

输入：`x = hidden_states [T, H]`、`q_lora = q_a_layernorm 输出 [T, Rq]`（注意是 norm 后、`q_b_proj` 前的那一份）、`positions [T]`、`forward_batch`、`layer_id`。

1. **Q / K 投影 + RoPE + Hadamard**（`_get_q_k_bf16`）：
   - `query = wq_b(q_lora)` → reshape `[T, I=8, Di=128]`；split 出前 `Dro=64` 维 `q_rope`。
   - `key = wk(x)` → `key = k_norm(key)`（完整 LayerNorm，fp32 计算）→ `[T, 128]`；split 出前 64 维 `k_rope`。
   - `q_rope, k_rope = rotary_emb(positions, q_rope, k_rope)`（NeoX）；写回 `query[..., :64] = q_rope`、`key[..., :64] = k_rope`。
   - `query = rotate_activation(query)`、`key = rotate_activation(key)` —— Hadamard 正交旋转（`scale = 128**-0.5`；bf16 输入；对 REF 数值是正交变换，主要为 FP8 量化把 outlier 摊开）。
2. **FP8 量化 + index K cache store**：
   - `q_fp8, q_scale = act_quant(query, block_size=128, scale_fmt="ue8m0")`。
   - `_store_index_k_cache`：`act_quant(key, ...)` → `(k_fp8, k_scale)`，写 `index_k_with_scale_buffer`（CUDA fast path 走 `fused_store_index_k_cache`；否则 `set_index_k_scale_buffer`）。
3. **gate**：`weights = weights_proj(x)`（bf16→fp32）`* n_heads**-0.5`；用于 logits 时再 `* q_scale * softmax_scale`（`softmax_scale = Di**-0.5 = 128**-0.5`）→ `weights [T, I, 1]`。
4. **logits + topk transform**：
   - **decode / target-verify / draft-extend**（`_get_topk_paged`）：
     `deep_gemm.fp8_paged_mqa_logits(q_fp8 [B,1,I,Di], kv_cache_fp8 [num_pages, 64, 1, 132], weights [B,I], seqlens_int32, block_table_64, schedule_metadata, max_seq_len)` → `logits [B, max_seq_len]` → `metadata.topk_transform(logits, Ktop)` → `topk_result [B, Ktop=2048]`（int32，序列短于 Ktop 时尾部填 `-1`）。
   - **prefill ragged**（`_get_topk_ragged`）：
     从 index K cache gather `(k_fp8 [Ktot, Di], k_scale [Ktot])`；`ks, ke` = 每 token 可见 key 区间（`prefix(req) + causal 当前 chunk`）；`deep_gemm.fp8_mqa_logits(q_fp8 [T,I,Di], (k_fp8, k_scale), weights [T,I], ks, ke)` → `logits [T, Ktot]` → `metadata.topk_transform(...)` → `topk_result [T, Ktop]`。OOM 时按行 chunk。
   - **prefill 短序列快捷路径**（`_forward_cuda_k_only`，当 `max_kv_len <= index_topk = 2048` 且非 CP）：**只**算 K 并写 index cache，跳过所有 q / weights / logits 运算；`topk_transform` 直接用 `dummy_logits` 走 kernel fast path 生成 `[0,1,...,len-1,-1,...]`（等价于"全选可见 token" → sparse 退化为 dense）。
   - **NPU**：走 `forward_npu` + `torch_npu.npu_lightning_indexer`，行为对齐但 kernel 不同；本切片以 CUDA / 通用语义为准。

---

## 8. KV / Index cache layout（`mem_cache/memory_pool.py::NSATokenToKVPool`，page_size=64）

| buffer | per-layer shape | dtype | 内容 |
|---|---|---|---|
| 主 latent KV cache（继承自 `MLATokenToKVPool`） | `[(size+page+1), page_size=64, 1, Rkv+Dro=576]`（按 page 组织） | bf16（或 fp8_e4m3，视 `kv_cache_dtype`） | 每 token 一行 `concat(kv_a_layernorm 输出 [Rkv=512], k_pe [Dro=64])`；单 latent 头 |
| `index_k_with_scale_buffer` | `[ceil((size+page+1)/64), 64 * (Di + Di//128*4)] = [num_pages, 64*132]` | uint8 | 每 token：`buf[..., :128]` = indexer K 的 FP8 数据（128 字节）；`buf[..., 128:132].view(fp32)` = block scale。`head_dim_with_sf = 132` |

要点：absorb path 落 cache 的是**单 latent 头、576 维**（不是 "32 个 KV 头"）。"32 头 KV" 仅在 `MHA_ONE_SHOT` dense fallback 里出现——那里 `kv_b_proj` 把 latent 解成 `[T, Nh=32, Dnope+Dv=256]`，FlashAttention varlen，输出 `[T, 32, Dv=128]`，**跳过 V 吸收**直接进 `o_proj`。

---

## 9. "head 数" 四件套

| 名称 | config 字段 | 值 | 运行时角色 | cache layout |
|---|---|---:|---|---|
| Q 头 (Nh) | `num_attention_heads` | 32 | `q_b_proj` 输出 `[*, 32, Dqk=192]`；absorb 后 `bmm(q_nope, w_kc)` → `[*, 32, Rkv=512]` | — |
| GQA KV 头 | `num_key_value_heads` | 8 | **MLA/DSA 路径完全不用**（仅非-MLA `Glm4MoeAttention` GQA 用） | — |
| 潜空间 KV 头 (h_kv) | —（不在 config） | **1** | absorb path 真正落 cache 的 KV 头数（latent MQA）；`attn_mqa = RadixAttention(num_kv_heads=1, v_head_dim=Rkv=512)` | `[num_pages, 64, 1, 576]` |
| Decompress 视角（仅 dense fallback） | `num_attention_heads` | 32 | `MHA_ONE_SHOT` 时 `kv_b_proj` 把 latent 解成 `[*, 32, Dnope+Dv=256]`，MHA 32 头 KV | 不另落 cache，借 absorb 的 latent 再 decompress |
| Indexer 头 (I) | `index_n_heads` | 8 | indexer Q `[*, 8, 128]`；indexer K `[*, 128]`（单 head 共享） | `index_k_with_scale_buffer` `[num_pages, 64*132]`，uint8 packed FP8+scale |

---

## 10. Decode Path

`N = B`（每 request 1 个 query token）。**走 `AttnForwardMethod.MLA`（absorb）**：sparse MQA 在潜空间完成，再 `w_vc` 解吸收回 `[B, Nh, Dv]`，最后 `o_proj`。

### 10.1 计算流

```
hidden_t [B,H]   (来自 layer_communicator.prepare_attn；MHC 残差已处理)
  │
  ├─ D0  dsa_qkv_a_proj  (在 layer communicator 内执行：prepare_qkv_latent)
  │     qkv_latent = fused_qkv_a_proj_with_mqa(hidden_t)            [B, Rq+Rkv+Dro = 1344]
  │     attention 模块: q_lora_raw, latent_cache = qkv_latent.split([Rq], [Rkv+Dro])
  │
  ├─ D1  dsa_qk_norm + q_b_proj + split (+ absorb-K)
  │     q_lora      = q_a_layernorm(q_lora_raw)            [B, Rq=768]
  │     k_nope      = kv_a_layernorm(latent_cache[..., :Rkv])       [B, Rkv=512] -> unsqueeze(1)
  │     k_pe        = latent_cache[..., Rkv:].unsqueeze(1)          [B, 1, Dro=64]   (未旋转)
  │     q           = q_b_proj(q_lora).view(B, Nh=32, Dqk=192)
  │     q_nope, q_pe = q.split([Dnope=128], [Dro=64])               q_pe 未旋转
  │     q_nope_out  = bmm(q_nope.T[Nh,B,Dnope], w_kc[Nh,Dnope,Rkv]).T  [B, Nh, Rkv=512]
  │     side effect: 主 latent KV cache[out_cache_loc] <- concat(k_nope, k_pe)  [1, 576]
  │
  ├─ D2  dsa_indexer  (与 q_b_proj 在 alt_stream 上 overlap；见 §7)
  │     x = hidden_t,  q_lora = D1 的 q_lora (norm 后)
  │     query[B,I,Di] = rotate_activation( rope_neox_first64( wq_b(q_lora) ) )
  │     key  [B,Di]   = rotate_activation( rope_neox_first64( k_norm(wk(x)) ) )
  │     q_fp8,q_scale = act_quant(query);   k_fp8,k_scale = act_quant(key)
  │     side effect: index_k_with_scale_buffer[out_cache_loc] <- (k_fp8, k_scale)
  │     gate[B,I]     = weights_proj(x)*I^-0.5;  weights = gate.unsqueeze(-1)*q_scale*Di^-0.5
  │
  ├─ D3  dsa_decode_indexer_topk  (_get_topk_paged)
  │     logits[B, max_seq_len] = deep_gemm.fp8_paged_mqa_logits(q_fp8[B,1,I,Di],
  │                               index_k_cache[num_pages,64,1,132], weights[B,I],
  │                               seqlens_int32, block_table_64, schedule_md, max_seq_len)
  │     topk_slots[B, 2048]    = metadata.topk_transform(logits, Ktop)   (尾部 -1 填充)
  │
  ├─ D4  dsa_decode_sparse_mqa   ★ latent 空间 sparse MQA
  │     q = cat(q_nope_out, q_pe)  [B, Nh=32, Rkv+Dro = 576]
  │     k = cat(k_nope,    k_pe)   [B, 1, 576]                  (写 cache 用，本步 attn 从 cache 读)
  │     attn_out_latent[B, Nh, Rkv=512] = attn_mqa(q, k, k_nope, forward_batch, topk_indices=topk_slots)
  │         (NSA backend: flashmla_kv / flashmla_sparse / tilelang，按 cache dtype 选；num_kv_heads=1, d_v=Rkv)
  │
  ├─ D5  dsa_v_absorb
  │     attn_out[B, Nh, Dv=128] = bmm(attn_out_latent.T[Nh,B,Rkv], w_vc[Nh,Rkv,Dv]).T
  │
  └─ D6  o_proj
        out[B, H=2048] = o_proj( attn_out.reshape(B, Nh*Dv=4096) )
```

### 10.2 Decode 算子依赖表

| # | 子步骤 | shape / IO | REF 入口（REPO） | Zeus 状态 |
|---|---|---|---|---|
| D0 | `dsa_qkv_a_proj` | `hidden [B,2048] -> qkv_latent [B,1344]` | `deepseek_v2.py::DeepseekV2AttentionMLA.prepare_qkv_latent`（在 `communicator.py::fetch_qkv_latent` 内调用） | × |
| D1 | `dsa_qk_norm_q_b_proj_split_absorb` | `qkv_latent -> q_nope_out [B,32,512] + q_pe [B,32,64] + k_nope [B,1,512] + k_pe [B,1,64]`；写 latent KV cache | `deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare`（`q_a_layernorm` / `kv_a_layernorm` / `q_b_proj` / `bmm w_kc`）；cache store 由 `attn_mqa` 内部完成 | × |
| D2 | `dsa_indexer_prep_store` | `hidden + q_lora -> q_fp8 [B,8,128] + weights [B,8,1]`；写 index K cache | `nsa_indexer.py::Indexer.forward_cuda` → `_get_q_k_bf16` + `rotate_activation` + `act_quant` + `_store_index_k_cache` | × |
| D3 | `dsa_decode_indexer_topk` | `q_fp8 + index_k_cache + weights -> topk_slots [B,2048]` | `nsa_indexer.py::_get_topk_paged`（`deep_gemm.fp8_paged_mqa_logits` + `metadata.topk_transform`，见 `layers/attention/nsa/transform_index.py`） | × |
| D4 | `dsa_decode_sparse_mqa` | `q [B,32,576] + latent KV cache[topk_slots] -> attn_out_latent [B,32,512]` | `forward_absorb_core` → `attn_mqa(..., topk_indices=...)` → `nsa_backend.py::_forward_flashmla_kv` / `_forward_flashmla_sparse` / `_forward_tilelang`（`num_kv_heads=1, d_v=Rkv=512`） | × |
| D5 | `dsa_v_absorb` | `attn_out_latent [B,32,512] -> attn_out [B,32,128]` | `forward_absorb_core` 末段 `bmm w_vc` | × |
| D6 | `o_proj` | `[B,4096] -> [B,2048]` | `RowParallelLinear` | × |

---

## 11. Prefill / Extend Path

`N = T = sum(extend_seq_lens)`；每 query row 可见 key 范围 = `prefix(req) + current_chunk(req, <= row_pos)`（causal）。

**默认走 `AttnForwardMethod.MLA`（absorb）**，与 decode 共用 `forward_absorb_prepare/core`，sparse 调用换成 ragged 版（`flash_mla_sparse_fwd` / `flashmla_kv` / `tilelang_sparse_fwd`）。

**`MHA_ONE_SHOT` dense fallback** 仅当 `nsa_backend.py::set_nsa_prefill_impl` 判 `use_mha=True`，需同时满足：
`device_sm == 90` 或 `100 <= device_sm < 110`（H200/B200 一类）、`max_kv_len <= SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`（默认 2048）、`kv_cache.dtype ∈ {bf16, fp8_e4m3}`、`sum_seq_lens <= max_chunk_capacity`、未开 prefill CP、无 hisparse coordinator。否则一律 absorb。

### 11.1 计算流

```
hidden [T,H], positions [T], out_cache_loc [T], ragged metadata
  │
  ├─ P0  dsa_qkv_a_proj           (同 D0，T batched；在 layer communicator 内)
  ├─ P1  dsa_qk_norm_q_b_proj_split_absorb   (同 D1；KV cache 批量写 out_cache_loc)
  ├─ P2  dsa_indexer_prep_store   (同 D2，T batched；index K 批量写 out_cache_loc)
  │
  ├─ P3  dsa_prefill_ragged_indexer_topk
  │     若 max_kv_len <= 2048 (非 CP): _forward_cuda_k_only —— 只写 index K，
  │       topk = topk_transform(dummy_logits) = [0..len-1, -1, ...]  (全选可见 token)
  │     否则: _get_topk_ragged
  │       gather (k_fp8[Ktot,Di], k_scale[Ktot]); ks,ke = 每 token 可见区间
  │       logits[T, Ktot] = deep_gemm.fp8_mqa_logits(q_fp8[T,I,Di], (k_fp8,k_scale), weights[T,I], ks, ke)
  │       topk_slots[T, 2048] = metadata.topk_transform(logits, Ktop, ks=ks)
  │
  ├─ P4  dsa_prefill_sparse_mqa   ★ absorb 后 sparse MQA（默认）
  │     q = cat(q_nope_out, q_pe)  [T, Nh=32, 576]
  │     attn_out_latent[T, Nh, 512] = attn_mqa(q, k, k_nope, fb, topk_indices=topk_slots)
  │         (NSA backend ragged: flash_mla_sparse_fwd / flashmla_kv / tilelang_sparse_fwd; d_v=Rkv)
  │
  ├─ P4-alt  dense_fallback (use_mha=True, 短序列+特定硬件)  —— forward_normal_one_shot_prepare/core
  │     q = q_b_proj(q_a_layernorm(...)).view(T, Nh, Dqk=192)   (q_pe 未旋转)
  │     kv_a = kv_a_layernorm(latent[..., :Rkv]);  k_pe = latent[..., Rkv:]   (未旋转)
  │     写 latent KV cache(attn_mha)；indexer 仍跑(return_indices=False) 只为填 index K cache
  │     kv = kv_b_proj(kv_a).view(T, Nh=32, Dnope+Dv=256) -> k_nope[Dnope], v[Dv]
  │     k = cat(k_nope, k_pe broadcast)  [T, 32, 192]
  │     attn_out_dense[T, Nh, Dv=128] = attn_mha(q, k, v, fb)    (FlashAttention varlen, 全 prefix+current)
  │       ★ 跳过 V 吸收，直接进 P6
  │
  ├─ P5  dsa_v_absorb            仅 absorb path：attn_out[T,Nh,128] = bmm(attn_out_latent.T, w_vc).T
  └─ P6  o_proj                  [T, Nh*Dv=4096] -> [T, H=2048]
```

### 11.2 Prefill 算子依赖表

| # | 子步骤 | shape / IO | REF 入口（REPO） | Zeus 状态 |
|---|---|---|---|---|
| P0 | `dsa_qkv_a_proj` | `hidden [T,2048] -> qkv_latent [T,1344]` | 同 D0 | × |
| P1 | `dsa_qk_norm_q_b_proj_split_absorb` | `qkv_latent -> q_nope_out [T,32,512] + q_pe [T,32,64] + k_nope [T,1,512] + k_pe [T,1,64]`；批量写 latent KV cache | `forward_mla.py::forward_absorb_prepare` | × |
| P2 | `dsa_indexer_prep_store` | 同 D2，T batched；批量写 index K cache | `nsa_indexer.py::Indexer.forward_cuda`（`_get_q_k_bf16` + Hadamard + `act_quant` + `_store_index_k_cache`） | × |
| P3 | `dsa_prefill_ragged_indexer_topk` | `q_fp8 + index_k_cache + ragged metadata -> topk_slots [T,2048]`；短序列走 K-only 快捷路径 | `nsa_indexer.py::_get_topk_ragged`（`deep_gemm.fp8_mqa_logits` + `topk_transform`）/ `_forward_cuda_k_only` | × |
| P4 | `dsa_prefill_sparse_mqa` | `q [T,32,576] + latent KV cache + topk_slots -> attn_out_latent [T,32,512]` | `forward_absorb_core` → `attn_mqa(..., topk_indices=...)` → `nsa_backend.py::_forward_flashmla_sparse` / `_forward_flashmla_kv` / `_forward_tilelang` | × |
| P4-alt | `dense_fallback_policy` | `use_mha` 判定 + MHA_ONE_SHOT | `nsa_backend.py::set_nsa_prefill_impl` + `handle_attention_nsa` + `forward_mha.py::forward_normal_one_shot_prepare/core`（含 indexer `return_indices=False`） | × |
| P5 | `dsa_v_absorb` | `[T,32,512] -> [T,32,128]` | `forward_absorb_core` 末段 `bmm w_vc` | × |
| P6 | `o_proj` | 同 D6 | `RowParallelLinear` | × |

---

## 12. dev 脚本对应

dev 脚本侧 stage 名沿用现有 `dev_glm_moe_dsa_{decode,prefill}_test*.py` 的命名，按 V4_a 的真实链路调整覆盖范围。

| dev stage | 覆盖范围 | 对应子步骤 | Zeus 状态 |
|---|---|---|---|
| `decode_qkv_a_proj_norm_fused` / `prefill_*` | `hidden -> qkv_latent`，再 split + `q_a_layernorm` + `kv_a_layernorm` | D0/P0 + D1/P1 前半 | × |
| `decode_q_proj_fused` / `prefill_*` | `q_lora_norm -> q_b_proj -> split -> bmm(q_nope, w_kc)`，输出 `q_nope_out [*,Nh,Rkv]` + `q_pe`（未旋转） | D1/P1 | × |
| `decode_kv_cache_store` / `prefill_*` | latent KV 行 `concat(k_nope, k_pe) [1, Rkv+Dro=576]` 写 cache（Zeus kernel `dsa_kv_proj_cache_store_fused` 已落地） | D1/P1 side-effect | ✓ |
| `decode_indexer_prep_store_fused` / `prefill_*` | indexer Q/K（含 NeoX RoPE 前 64 维）+ k_norm（LayerNorm）+ Hadamard + FP8 quant + index K cache store + gate | D2/P2 | × |
| `decode_indexer_topk_fused` | paged MQA logits + topk transform；支持 `-1` 填充 | D3 | × |
| `prefill_ragged_indexer_topk_fused` | ragged causal logits + topk；短序列 K-only 快捷路径（dummy → 全选可见 token） | P3 | × |
| `decode_sparse_mqa_fused` / `prefill_sparse_mqa_fused` | latent 空间 sparse MQA；输出 `[*,Nh,Rkv=512]` | D4/P4 | × |
| `dense_fallback_policy` | `use_mha` 判定 + `MHA_ONE_SHOT`（不吸收，输出 `[T,Nh,Dv]`） | P4-alt | × |
| `decode_v_absorb` / `prefill_v_absorb` | `bmm(attn_out_latent, w_vc)`，输出 `[*,Nh,Dv=128]` | D5/P5 | × |
| `decode_o_proj` / `prefill_o_proj` | output projection `[*,4096] -> [*,2048]` | D6/P6 | × |
| `decode_full_path` / `prefill_full_path` | D0–D6 / P0–P6 端到端 | 全链路 | × |

> 注意 §4：dev REF 脚本里 main MLA 路径**不要**插 RoPE；只 indexer 前 64 维做 NeoX RoPE。
> `derive_w_kc_w_vc` 要匹配 §6 的 `w_kc [Nh,Dnope,Rkv]` / `w_vc [Nh,Rkv,Dv]` 形态。

---

## 13. 里程碑与开发优先级

按"先正确性、再吸收路径、再 sparse、再 fallback、再 FP8"排序。

| 优先级 / Milestone | 任务 | 通过判据 |
|---|---|---|
| M0 | 文档 + pure-torch REF（按 V4_a 链路：main 无 RoPE、indexer NeoX、absorb 在潜空间） | `dev_glm_moe_dsa_{decode,prefill}_test*.py --mode ref` 端到端与 transformers / REPO REF 数值一致；REF 显式产出 `qkv_latent`、`q_lora_norm`、`q_nope_out`、`q_pe`、`k_nope`、`k_pe`、`q_fp8`/`weights`、`topk_slots`、`attn_out_latent` |
| M1 | D0/P0 `dsa_qkv_a_proj`（fused `fused_qkv_a_proj_with_mqa` GEMM） | `*_qkv_a_proj_norm_fused --mode zeus` 输出 `q_lora_norm + kv_lora_norm + k_pe` 对齐 REF |
| M2 | D1/P1 Q 通路含 absorb（`q_a_layernorm` + `q_b_proj` + split + `bmm w_kc`） | `*_q_proj_fused --mode zeus` 输出 `q_nope_out [*,Nh,Rkv]` + `q_pe` 对齐 REF；latent KV cache store side-effect 一致 |
| M3 | D2/P2 indexer 通路（`wq_b`/`wk` + NeoX RoPE 前 64 维 + `k_norm` LayerNorm + Hadamard + FP8 `act_quant` + index K cache store + gate） | `*_indexer_prep_store_fused --mode zeus` 写入/读回一致；`weights` fp32 数值对齐 |
| M4 | D3 decode paged top-k | `decode_indexer_topk_fused --mode zeus` 支持尾部 `-1` 填充，对齐 REF |
| M5 | D4 decode sparse MQA（潜空间，`num_kv_heads=1, d_v=Rkv`） | `decode_sparse_mqa_fused --mode zeus` 输出 `[B,Nh,Rkv=512]` 对齐 REF |
| M6 | D5/P5 V absorb + D6/P6 o_proj | `*_v_absorb` / `*_o_proj --mode zeus` 对齐 REF |
| M7 | P3 ragged top-k（含短序列 K-only 快捷路径） | `prefill_ragged_indexer_topk_fused --mode zeus` 多请求不串行、不看未来、不跨 request；短序列退化为全选可见 |
| M8 | P4 prefill sparse MQA + P4-alt dense fallback policy | absorb 路径 `[T,Nh,Rkv]` 对齐 REF；`dense_fallback_policy --mode zeus` 在 `max_kv_len <= 2048` + 模拟硬件条件下与 `MHA_ONE_SHOT` REF（`[T,Nh,Dv]`，不吸收）一致 |
| M9 | GLM5-Next real-shape smoke | DSA 层 `[3,7,11,15,19,23]` 单层 `[T,H] -> [T,H]` 端到端 |
| M10 | FP8 K cache / index cache + 跨层 topk 共享评估 | `quant_k_cache`/`dequant_k_cache` + `NSATokenToKVPool` 落地；评估 `index_topk_freq` / `index_topk_pattern`（当前默认不复用） |

---

## 14. 参考入口速查（REPO 路径）

| 标签 | 入口 | 用途 |
|---|---|---|
| `R-ENTRY` | `models/glm5_next.py::Glm5NextForCausalLM` / `Glm5NextDecoderLayer` / `Glm5NextLinearAttention` | 模型主体；DSA 子层 = `DeepseekV2AttentionMLA`（`skip_rope=mla_nope`） |
| `R-MLA` | `models/deepseek_v2.py::DeepseekV2AttentionMLA`（`__init__` l.1090；`prepare_qkv_latent` l.1511；`dispatch_attn_forward_method`；`forward_prepare/core`） | hidden-in fused projection + attention dispatch + absorb 路径骨架 |
| `R-ABSORB` | `models/deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare / forward_absorb_core` | decode + prefill 共用 absorb：`q_a/kv_a layernorm` → `q_b_proj` → split → `bmm w_kc` → `attn_mqa(..., topk_indices=...)` → `bmm w_vc` → `o_proj` |
| `R-MHA1S` | `models/deepseek_common/attention_forward_methods/forward_mha.py::forward_normal_prepare / forward_normal_one_shot_prepare/core` | `MHA_ONE_SHOT` dense fallback（`kv_b_proj` decompress → FlashAttention varlen，不吸收）；indexer 以 `return_indices=False` 仅填 index K cache |
| `R-DISPATCH` | `models/deepseek_common/attention_backend_handler.py::handle_attention_nsa` | 读 `NSA backend.use_mha` → `MHA_ONE_SHOT` / `MLA` |
| `R-WLOADER` | `models/deepseek_common/deepseek_weight_loader.py::post_load_weights` (l.399-)，`w_kc/w_vc` 拆分 l.565-585 | `kv_b_proj.weight` → `w_kc [Nh,Dnope,Rkv]` / `w_vc [Nh,Rkv,Dv]` |
| `R-INDEXER` | `layers/attention/nsa/nsa_indexer.py::Indexer`（`forward_cuda` / `_get_q_k_bf16` / `_get_topk_paged` / `_get_topk_ragged` / `_forward_cuda_k_only` / `_store_index_k_cache`；`rotate_activation` l.135） | indexer Q/K/gate + NeoX RoPE + Hadamard + FP8 quant + paged/ragged logits + topk transform + index K cache store |
| `R-TOPK` | `layers/attention/nsa/transform_index.py`（`metadata.topk_transform` 实现）+ `triton_kernel.py` / `tilelang_kernel*.py` | logits → topk slots（含 `-1` padding、causal 约束、跨 backend 变体） |
| `R-BACKEND` | `layers/attention/nsa_backend.py`（`set_nsa_prefill_impl` l.2090；`_forward_flashmla_sparse` l.1651；`_forward_flashmla_kv` l.1700；`_forward_tilelang` l.1810；`get_indexer_metadata`） | NSA attention backend：dense/sparse 调度、sparse MQA kernel 包装 |
| `R-CACHE` | `mem_cache/memory_pool.py::NSATokenToKVPool`（l.1858-）+ `MLATokenToKVPool`；`layers/attention/nsa/index_buf_accessor.py`（`GetK`/`GetS`/`GetKAndS`/`SetKAndS`） | 主 latent KV cache `[*,64,1,576]` + `index_k_with_scale_buffer [*,64*132]` |
| `R-QUANT` | `layers/attention/nsa/quant_k_cache.py` / `dequant_k_cache.py` | FP8 K cache 量化 / 反量化（`flashmla_kv` 分支用） |
| `R-CFG` | `configs/model_config.py::is_deepseek_nsa / get_nsa_index_*`；`configs/glm_linear.py::GlmLinearConfig` | NSA 判定、`GlmLinearConfig`（`is_mla` 触发条件含 `mla_nope is True`） |
| `R-COMM` | `layers/communicator.py`（`fetch_qkv_latent`）；`layers/communicator_mhc.py::MHCLayerCommunicator`；`layers/mhc.py::HyperConnection` | layer communicator：在 `prepare_attn` 内执行 `prepare_qkv_latent`；MHC 残差 |
| `R-FUSEDSTORE` | `jit_kernel/fused_store_index_cache.py`（`fused_store_index_k_cache` / `can_use_nsa_fused_store`）；`jit_kernel/hadamard.py`（`hadamard_transform`） | index K cache 融合写；Hadamard JIT |
