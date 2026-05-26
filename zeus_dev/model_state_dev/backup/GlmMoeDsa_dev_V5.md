# GLM5-Next / GLM-MoE-DSA Zeus 适配开发追踪（V5）

> 参考仓库（以下简称 **REPO**）：`/root/project/sglang-feat-v0.5.10-prerelease-glm`
> canonical config：本目录 `config_16b.json`
> （`architectures: ["Glm5NextForCausalLM"]`，`model_type: "glm4_moe"`）。
>
> 本文档整合 V4_a + V4_b，按 REPO 中 `Glm5NextForCausalLM` 的真实运行链路梳理 DSA
> full-attention 子层的算子流程，是一份完整、独立的开发追踪文档；不再罗列与旧版本的差异。
>
> 状态约定：`×` = Zeus 端尚未落地或尚未接入 path 级对齐测试；`△` = 部分落地；
> `✓` = Zeus kernel 已落地且默认 PASS。

---

## 1. 范围与方法

- **入口**：DSA 全注意力子层（`Glm5NextDecoderLayer.self_attn`，即 `DeepseekV2AttentionMLA`）拿到 `hidden_states [N,H]`。
  实际上 DSA 的第一个投影 GEMM（`fused_qkv_a_proj_with_mqa`）发生在 layer communicator 的 pre-attn 阶段（见 §9），
  attention 模块通过 `get_attn_tp_context().fetch_qkv_latent()` 取结果，所以严格说"入口"是 `hidden_states` 进 `layer_communicator.prepare_attn(...)`。
- **出口**：`o_proj` 输出 `[N,H]`，随后进入 layer communicator 的 post-attention residual / MHC，再到 MLP 前处理。
- **关注层**：`linear_attn_config.full_attn_layers = [3, 7, 11, 15, 19, 23]`（6 层 DSA full-attn）。
- **不展开**：KDA 线性注意层（`Glm5NextLinearAttention`，21 层）、MoE FFN 内部、MHC（multi-head hyper-connection）的张量细节、
  TP/PP/EP/DP 通信细节、NextN/MTP、CP（context parallel）的实现细节（仅在 §15 给接口约束）、NPU 专用 DSA path。
- **runtime path**：
  - **Decode**：每 request 1 个 query token，历史 latent K / index K 来自 paged cache（CUDA `page_size=64`）。
  - **Prefill / Extend**：ragged chunk，多 token 批量写 cache；可能再分两条 sub-path——
    - **MLA absorb sparse**（默认，与 decode 共用 `forward_absorb_prepare/core`）；
    - **MHA_ONE_SHOT dense fallback**（短序列 + 特定硬件，见 §12）。
  - **Target verify / Draft extend** 与 decode 共享 NSA 骨架；Zeus 首轮按 Decode 与普通 Prefill/Extend 对齐即可。

---

## 2. 模型脚手架（REPO 实况）

### 2.1 入口与层构成

| 标签 | 入口（REPO 路径） | 内容 |
|---|---|---|
| `M-ENTRY` | `python/sglang/srt/models/glm5_next.py::Glm5NextForCausalLM`（`EntryClass = [Glm5NextForCausalLM]`） | GLM5-Next 完整实现类（**不是空壳**）；`config_16b.json` 的 `architectures = ["Glm5NextForCausalLM"]` 直接命中；构造时 `use_nsa = is_deepseek_nsa(config)` |
| `M-BACKBONE` | `glm5_next.py::Glm5NextModel` | embedding → decoder layers → final RMSNorm；负责 CP split/gather、跨层 `topk_indices` 传递 |
| `M-LAYER` | `glm5_next.py::Glm5NextDecoderLayer.__init__` | `config.is_kda_layer(layer_id)`（= `layer_id in linear_attn_config["kda_layers"]`）为真 → `Glm5NextLinearAttention`；否则 → `Glm5NextMLAAttention` |
| `M-MLA` | `glm5_next.py` l.99：`from ...models.deepseek_v2 import DeepseekV2AttentionMLA as Glm5NextMLAAttention` | **DSA 全注意子层直接复用 `DeepseekV2AttentionMLA`**；构造时传 `skip_rope=getattr(config, "mla_nope", False)`（GLM5 = `True`，见 §4） |
| `M-INDEXER` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py::Indexer` | `DeepseekV2AttentionMLA.__init__` 在 `use_nsa` 时构造；`index_n_heads / index_head_dim / index_topk` 由 `get_nsa_index_*(config)` 读出 |
| `M-COMM` | `layers/communicator.py`（`fetch_qkv_latent`） / `communicator_mhc.py::MHCLayerCommunicator` / `communicator_mhc_hybrid_cp.py::MHCHybridNSACPLayerCommunicator` / `communicator_nsa_cp.py::NSACPLayerCommunicator` / `communicator.py::LayerCommunicator` | `config.mhc=True`（默认）→ `MHCLayerCommunicator`（CP 时 `MHCHybridNSACPLayerCommunicator`）；attn/mlp 外包 `HyperConnection`。`qkv_latent_func=self.self_attn.prepare_qkv_latent` 传入 communicator，**`fused_qkv_a_proj_with_mqa` 这个 GEMM 在 `prepare_attn` 内执行** |
| `M-BACKEND` | `layers/attention/nsa_backend.py::NativeSparseAttnBackend` | 初始化 NSA metadata；判 dense fallback / 选 sparse MLA backend；提供 topk transform 与 sparse attention kernel |

### 2.2 关键运行时事实

- DSA 子层 MLA forward 由 `DeepseekV2AttentionMLA.dispatch_attn_forward_method` 经
  `AttentionBackendRegistry.get_handler("nsa")` → `deepseek_common/attention_backend_handler.py::handle_attention_nsa` 决定：
  读 `NSA backend.use_mha`——`True` 返回 `AttnForwardMethod.MHA_ONE_SHOT`，否则 `AttnForwardMethod.MLA`（即 absorb）。
  `use_mha` 判定见 `nsa_backend.py::set_nsa_prefill_impl`（§12.1）。Decode / verify 始终 `use_mha=False`。
- `is_deepseek_nsa(config)` 的 architecture 白名单**已含** `"Glm5NextForCausalLM"`（与 `"GlmMoeDsaForCausalLM"` 并列），触发条件还需 `index_topk is not None`（GLM5 = 2048）⇒ 满足，启用 NSA backend。**不需要再改 architecture 名。**
  注意 `server_args._handle_model_specific_adjustments()` 自动设置的 NSA architecture 列表里显式列了 `GlmMoeDsaForCausalLM` 而未在同处列 `Glm5NextForCausalLM`——启动服务时要确认 `attention_backend` 已进 `nsa`、dense threshold 已被设为 `index_topk`。
- `index_topk_freq` 默认 1、`index_topk_pattern` 默认 None ⇒ `skip_topk / next_skip_topk` 全 None/False ⇒ **本切片内不发生跨层 topk 复用**（每层各自算 topk）。跨层复用是后续优化点（§15.2）。
- 全程**不会**用到 `num_key_value_heads = 8`（那是非-MLA GQA 路径的字段）。

---

## 3. GLM5-Next 16B 关键配置（`config_16b.json`）

### 3.1 DSA / MLA 维度

| 字段 | 值 | 运行时含义 |
|---|---:|---|
| `hidden_size` (H) | 2048 | residual hidden 维度 |
| `num_attention_heads` (Nh) | 32 | MLA Q 头数；本地 heads `Nh_local = Nh / attn_tp_size` |
| `num_key_value_heads` | 8 | **MLA/DSA 路径完全不用**（仅非-MLA `Glm4MoeAttention` GQA 字段；与 `index_n_heads=8` 数值相同纯属巧合） |
| `q_lora_rank` (Rq) | 768 | Q low-rank latent |
| `kv_lora_rank` (Rkv) | 512 | latent K/V dim；潜空间同时充当 K 与 V；主 KV cache 的 value dim |
| `qk_nope_head_dim` (Dnope) | 128 | Q/K no-position segment |
| `qk_rope_head_dim` (Dro) | 64 | Q/K "rope" 分区宽度（GLM5 主 MLA 路径下不实际旋转，见 §4） |
| `qk_head_dim` (Dqk) | 192 | `Dnope + Dro` |
| `v_head_dim` (Dv) | 128 | `w_vc` 解吸收后的 per-head V dim |
| `index_head_dim` (Di) | 128 | indexer 每 head 维度 |
| `index_n_heads` (I) | 8 | NSA indexer 头数（与 Nh 无关） |
| `index_topk` (Ktop) | 2048 | 每个 query 选出的 sparse KV 数 |
| `index_dsa_use_layernorm` | true | indexer K 上挂 LayerNorm（→ `Indexer.k_norm`，**完整 LayerNorm，含 bias，fp32 计算**） |

### 3.2 位置编码 / Norm

| 字段 | 值 | 说明 |
|---|---:|---|
| `max_position_embeddings` | 202752 | 长上下文上限 |
| `rope_theta` | 10000 | RoPE 基频（main MLA 与 indexer 共用基频；但 main MLA 不实际旋转） |
| `rope_scaling` | null | 不开 YaRN / 线性 scaling |
| `partial_rotary_factor` | 0.5 | 名义上一半 head dim 旋转（128×0.5=64）；**对 main MLA 路径无意义**（被 `mla_nope` 跳过） |
| `mla` / `mla_nope` | true / true | `mla_nope=true` ⇒ `DeepseekV2AttentionMLA(skip_rope=True)` ⇒ `self.rotary_emb=None` ⇒ main MLA 路径**不做 RoPE**（§4） |
| `multi_query_attention` | true | latent 侧 MQA（潜空间 1 个 KV 头） |
| `use_qk_norm` | true | GLM-4 GQA 风格字段；**DSA MLA 路径不读它**（MLA 用的是 `q_a_layernorm` / `kv_a_layernorm`） |
| `use_gated_attention` / `gated_attention_layers` | true / 21 层 | gated KDA，作用于 `kda_layers`；不在本切片 |
| `rms_norm_eps` | 1e-05 | norm eps |

### 3.3 其它

| 字段 | 值 | 说明 |
|---|---:|---|
| `num_hidden_layers` | 27 | 总层数 |
| `num_nextn_predict_layers` | 1 | NextN 投机层；不覆盖 |
| `linear_attn_config.kda_layers` | 21 层 `[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26]` | KDA 层 → `Glm5NextLinearAttention` |
| `linear_attn_config.full_attn_layers` | `[3,7,11,15,19,23]` | **DSA full-attn 层（本切片对象）** → `DeepseekV2AttentionMLA + Indexer` |
| `first_k_dense_replace` | 1 | 第 0 层 FFN 为 dense（非 MoE） |
| `n_routed_experts` / `num_experts_per_tok` | 64 / 6 | MoE 路由 |
| `n_shared_experts` | 2 | 共享专家数（`Glm5NextForCausalLM.determine_num_fused_shared_experts` 按需 fuse，断言 fused 数 == 1） |
| `mhc` / `mhc_num_residual_streams` | true / 4 | multi-head hyper-connection |
| `intermediate_size` / `moe_intermediate_size` | 10944 / 1408 | dense FFN / MoE 专家 FFN 尺寸 |

---

## 4. 位置编码：main MLA 不旋转，indexer NeoX 旋转

这是 GLM5-Next 与 DeepSeek-V3.2 最容易踩坑的差别，单列一节。

| 路径 | 字面公式（REPO） | GLM5 实际行为 |
|---|---|---|
| **main MLA Q/K RoPE** | `DeepseekV2AttentionMLA.__init__`：传入 `skip_rope = config.mla_nope = True` → 不构造 rotary，`self.rotary_emb = None`；否则才有 `is_neox_style = not getattr(config, "rope_interleave", True)` | **`self.rotary_emb is None`** ⇒ `forward_absorb_prepare` 中 `if self.rotary_emb is not None: q_pe, k_pe = self.rotary_emb(...)` 整段跳过。`q_pe [*,Nh,64]`、`k_pe [*,1,64]` 是**未旋转**的原始 slice，原样进 attention 与 cache。（参考 `tilelang_kernel_glm.py` 注释 "no RoPE tail / mla_nope case — all tail ops skipped"） |
| **Indexer Q/K RoPE** | `is_neox_style = not getattr(config, "indexer_rope_interleave", False)` → 默认 `indexer_rope_interleave=False` → `is_neox_style = True`（**NeoX**）；传给 `Indexer(rope_head_dim=qk_rope_head_dim=64, ...)` | indexer **总是**建 `self.rotary_emb`；`nsa_indexer.py::_get_q_k_bf16` 对 indexer Q `[*,I=8,128]`、indexer K `[*,128]` 的**前 64 维**做 NeoX RoPE，后 64 维不动；之后再对整 128 维做 Hadamard rotation，再 FP8 量化。 |

> 结论：Zeus 对齐 REF 时，main MLA 路径**不要**插任何 RoPE；只有 indexer 路径的前 64 维需要 NeoX RoPE。

---

## 5. 代表性权重（单个 DSA 层，TP=1）

`q_a_proj` 与 `kv_a_proj_with_mqa` 在权重加载时由 `Glm5NextForCausalLM.load_weights` 的
`packed_modules_mapping["fused_qkv_a_proj_with_mqa"] = ["q_a_proj", "kv_a_proj_with_mqa"]` 合并成一张 `fused_qkv_a_proj_with_mqa`。

| 参数 | shape | 备注 |
|---|---:|---|
| `fused_qkv_a_proj_with_mqa.weight` | `[Rq+Rkv+Dro, H] = [1344, 2048]` | hidden → `q_lora_raw ‖ kv_lora ‖ k_pe`；`ReplicatedLinear`（不切 TP） |
| `q_a_layernorm.weight` | `[768]` | q_lora_raw RMSNorm |
| `q_b_proj.weight` | `[Nh*Dqk, Rq] = [6144, 768]` | q_lora_norm → `[*, 32, 192]`；`ColumnParallelLinear` |
| `kv_a_layernorm.weight` | `[512]` | latent K RMSNorm（只 norm 前 Rkv） |
| `kv_b_proj.weight` | `[Nh*(Dnope+Dv), Rkv] = [8192, 512]` | `ColumnParallelLinear`；**加载后离线拆为吸收矩阵 `w_kc` / `w_vc`**（§6）。absorb path 运行时**不调** `kv_b_proj` 做 decompress，只有 `MHA_ONE_SHOT` fallback 才调 |
| `o_proj.weight` | `[H, Nh*Dv] = [2048, 4096]` | `RowParallelLinear` |
| `attn_mqa`（`RadixAttention` 实例） | `num_heads=Nh_local=32`、`head_dim=Rkv+Dro=576`、`scaling=self.scaling`（即 `Dqk**-0.5 = 192**-0.5`，可被 YaRN mscale 调整但 GLM5 不开）、`num_kv_heads=1`、`v_head_dim=Rkv=512` | 潜空间 sparse MQA；forward 接 `topk_indices` kwarg |
| `attn_mha`（`RadixAttention` 实例） | `num_heads=32`、`head_dim=Dnope+Dro=192`、`num_kv_heads=32`、`v_head_dim=Dv=128`；`attn_mha.kv_b_proj` 初始为 `None`，dense fallback forward 时绑定 `kv_b_proj` | 仅 `MHA_ONE_SHOT` dense fallback 用 |
| `indexer.wq_b.weight` | `[I*Di, Rq] = [1024, 768]` | q_lora_norm → indexer Q `[*, 8, 128]`；`ReplicatedLinear` |
| `indexer.wk.weight` | `[Di, H] = [128, 2048]` | hidden → indexer K `[*, 128]`（单 head 共享）；`ReplicatedLinear` |
| `indexer.weights_proj.weight` | `[I, H] = [8, 2048]` | per-token per-head gate；CUDA 上参数 dtype = bf16，**计算结果转 fp32** |
| `indexer.k_norm.weight` / `.bias` | `[128]` / `[128]` | indexer K 上的**完整 LayerNorm（fp32）**，`index_dsa_use_layernorm: true` 触发 |

---

## 6. 吸收矩阵 `w_kc` / `w_vc`

`kv_b_proj.weight [Nh*(Dnope+Dv), Rkv] = [8192, 512]` 在 `Glm5NextForCausalLM.load_weights` 末尾
`DeepseekV2WeightLoaderMixin.post_load_weights`（`deepseek_common/deepseek_weight_loader.py:565-585`）离线拆分：

```python
w_kc, w_vc = w.unflatten(0, (-1, Dnope + Dv)).split([Dnope, Dv], dim=1)
# w_kc: [Nh, Dnope, Rkv] = [32, 128, 512]
# w_vc: [Nh, Dv,    Rkv] = [32, 128, 512]
self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)  # 仍是 [32, 128, 512]，仅改内存布局
self_attn.w_vc = w_vc.contiguous().transpose(1, 2)                  # → [Nh, Rkv, Dv] = [32, 512, 128]
```

| 矩阵 | logical shape | 运行时用途 |
|---|---:|---|
| `w_kc` | `[Nh_local, Dnope, Rkv]`（或等价转置内存布局） | `q_nope [N, Nh_local, Dnope]` → `q_nope_out [N, Nh_local, Rkv]` |
| `w_vc` | `[Nh_local, Rkv, Dv]` | `attn_out_latent [N, Nh_local, Rkv]` → `[N, Nh_local, Dv]` |

absorb 路径运行时（decode 与 prefill **共用** `forward_absorb_prepare` / `forward_absorb_core`）：

1. **Q 侧 K 吸收**（`forward_absorb_prepare` 末段）：`q_nope [T,Nh,Dnope] →transpose(0,1)→ [Nh,T,Dnope]` `bmm` `w_kc [Nh,Dnope,Rkv]` `→ [Nh,T,Rkv] →transpose→ q_nope_out [T,Nh,Rkv=512]`。
   GLM5 因 `mla_nope`：`q_pe` 不旋转、`k_pe` 不旋转。
2. **拼 latent Q / K**（`forward_absorb_core`）：`q = cat(q_nope_out, q_pe) [T,Nh,Rkv+Dro=576]`；`k = cat(k_nope, k_pe) [T,1,576]`（`k_nope = kv_a_layernorm(latent_cache[..., :Rkv]).unsqueeze(1)`，`k_pe = latent_cache[..., Rkv:].unsqueeze(1)`，**均未旋转**）。
   注：部分 backend（FA3 类）实际以 `attn_mqa(q_nope_out, k_nope, k_nope, forward_batch, q_rope=q_pe, k_rope=k_pe, topk_indices=...)` 形式调用（rope 段单独传），数值语义与 cat 形式等价。
3. **潜空间 sparse MQA**：`attn_output [T,Nh,Rkv=512] = attn_mqa(..., topk_indices=topk_indices)` —— NSA backend 在 paged latent cache 上做 sparse MQA，`num_kv_heads=1`、`v_head_dim=Rkv`。`save_kv_cache=True` 时由 attention 内部把 `concat(k_nope, k_pe) [1,576]` 写进主 KV cache。
4. **V 侧解吸收**：`attn_output [T,Nh,Rkv] →transpose(0,1)→ [Nh,T,Rkv]` `bmm` `w_vc [Nh,Rkv,Dv]` `→ [Nh,T,Dv] →` reshape `→ [T, Nh*Dv=4096]`。
5. **o_proj**：`[T,4096] → out [T,H=2048]`。

---

## 7. KV / Index cache layout（`mem_cache/memory_pool.py::NSATokenToKVPool`，CUDA `page_size=64`）

| buffer | per-layer shape | dtype | 内容 |
|---|---|---|---|
| 主 latent KV cache（继承 `MLATokenToKVPool`） | 按 page 组织：`[(size+page+1) 折成 page, page_size=64, 1, Rkv+Dro=576]` | bf16（或 fp8_e4m3，视 `kv_cache_dtype`） | 每 token 一行 `concat(kv_a_layernorm 输出 [Rkv=512], k_pe [Dro=64])`；**单 latent 头**（不是 8 个 GQA KV 头） |
| `index_k_with_scale_buffer` | `[ceil((size+page+1)/64), 64 * (Di + Di//128*4)] = [num_pages, 64*132]` | uint8 | 每 token 132 bytes：`buf[..., :128]` = indexer K 的 FP8 数据；`buf[..., 128:132].view(fp32)` = block scale。`head_dim_with_sf = 132` |

要点：absorb path 落 cache 的是**单 latent 头、576 维**。"32 个 KV 头" 仅在 `MHA_ONE_SHOT` dense fallback 里出现——那里 `kv_b_proj` 把 latent 解成 `[T, Nh=32, Dnope+Dv=256]`，FlashAttention varlen，输出 `[T, 32, Dv=128]`，**跳过 V 吸收**直接进 `o_proj`。

index K 写入优先走 `fused_store_index_k_cache(key, buf, out_cache_loc, page_size)`（`can_use_nsa_fused_store` 为真时）；
fallback 为 `act_quant(key, block_size=128, scale_fmt="ue8m0")` → `(k_fp8, k_scale)` → `set_index_k_scale_buffer(layer_id, out_cache_loc, k_fp8, k_scale)`。

---

## 8. "head 数" 四件套

| 名称 | config 字段 | 值 | 运行时角色 | cache layout |
|---|---|---:|---|---|
| Q 头 (Nh) | `num_attention_heads` | 32 | `q_b_proj` 输出 `[*, 32, Dqk=192]`；absorb 后 `bmm(q_nope, w_kc)` → `[*, 32, Rkv=512]` | — |
| GQA KV 头 | `num_key_value_heads` | 8 | **MLA/DSA 路径完全不用**（仅非-MLA `Glm4MoeAttention` GQA 用） | — |
| 潜空间 KV 头 (h_kv) | —（不在 config） | **1** | absorb path 真正落 cache 的 KV 头数（latent MQA）；`attn_mqa = RadixAttention(num_kv_heads=1, v_head_dim=Rkv=512)` | `[num_pages, 64, 1, 576]` |
| Decompress 视角（仅 dense fallback） | `num_attention_heads` | 32 | `MHA_ONE_SHOT` 时 `kv_b_proj` 把 latent 解成 `[*, 32, Dnope+Dv=256]`，MHA 32 头 KV | 不另落 cache，借 absorb 的 latent 再 decompress |
| Indexer 头 (I) | `index_n_heads` | 8 | indexer Q `[*, 8, 128]`；indexer K `[*, 128]`（单 head 共享） | `index_k_with_scale_buffer [num_pages, 64*132]`，uint8 packed FP8+scale |

---

## 9. Decoder layer 总流程（layer communicator）

每个 `Glm5NextDecoderLayer.forward()` 的 attention 部分顺序：

```text
hidden_states, residual
  │
  ├─ layer_communicator.prepare_attn(...)
  │    - input_layernorm / scatter / collective（MHC: 残差流处理）
  │    - DSA 层调用 qkv_latent_func = self_attn.prepare_qkv_latent
  │      → fused_qkv_a_proj_with_mqa(hidden) → 写入 get_attn_tp_context()
  │
  ├─ self_attn.forward(...)            ← DSA 子层正文（§10 起）
  │    - 通过 fetch_qkv_latent() 取上一步结果
  │    - dispatch_attn_forward_method() → handle_attention_nsa()
  │    - full-attn DSA 默认走 AttnForwardMethod.MLA（absorb）
  │
  ├─ layer_communicator.prepare_mlp(...)
  ├─ mlp(...)                          ← MoE 或 dense FFN（不在本切片）
  └─ layer_communicator.postprocess_layer(...)
```

因此 DSA 的第一个投影算子（`fused_qkv_a_proj_with_mqa`）**不在** `forward_absorb_prepare()` 内直接发起，而是在 layer communicator 的 pre-attn 阶段由 `prepare_qkv_latent()` 产生；`forward_absorb_prepare()` 起手就是从 `fetch_qkv_latent()` 取到的 `q_lora_raw [N,Rq]` + `latent_cache [N,Rkv+Dro]`。

`config.mhc=True`（默认）→ communicator 是 `MHCLayerCommunicator`；`mhc + nsa_enable_prefill_cp` → `MHCHybridNSACPLayerCommunicator`；非 mhc + cp → `NSACPLayerCommunicator`；都不开 → 普通 `LayerCommunicator`。Zeus 首版按非-MHC 普通 `LayerCommunicator` 语义对齐即可，MHC 残差作为外层包装单独处理。

---

## 10. DSA absorb path 算子流（decode + 普通 prefill/extend 共用骨架）

`N=B` = decode（每 request 1 query token）；`N=T=sum(extend_seq_lens)` = prefill/extend。

```text
hidden [N,H], positions [N], out_cache_loc [N]   (+ ragged metadata for prefill)
  │
  ├─ A0  pre_attn_qkv_latent  (在 layer communicator 内执行：prepare_qkv_latent)
  │    qkv_latent = fused_qkv_a_proj_with_mqa(hidden)        [N, Rq+Rkv+Dro = 1344]
  │    attention 侧: q_lora_raw, latent_cache = qkv_latent.split([Rq], [Rkv+Dro])
  │                  q_lora_raw [N,768], latent_cache [N,576]
  │
  ├─ A1  qkv_a_norm  (forward_absorb_prepare 起手)
  │    q_lora  = q_a_layernorm(q_lora_raw)                   [N, Rq=768]
  │    k_nope  = kv_a_layernorm(latent_cache[..., :Rkv]) → unsqueeze(1)   [N, 1, Rkv=512]
  │    k_pe    = latent_cache[..., Rkv:] → unsqueeze(1)      [N, 1, Dro=64]   (未旋转)
  │
  ├─ A2  q_b_proj + split + K-absorb
  │    q       = q_b_proj(q_lora).view(N, Nh_local=32, Dqk=192)
  │    q_nope, q_pe = q.split([Dnope=128], [Dro=64])         q_pe 未旋转 (mla_nope)
  │    q_nope_out  = bmm(q_nope.T[Nh,N,Dnope], w_kc[Nh,Dnope,Rkv]).T   [N, Nh, Rkv=512]
  │
  ├─ A3  indexer_prepare_store_topk   (§11；与 q_b_proj 在 alt_stream 上 overlap)
  │    Indexer(hidden, q_lora=q_lora, positions, forward_batch, layer_id)
  │    → topk_indices  [N, Ktop=2048]   (尾部 -1 填充；prefill 短序列退化为"全选可见")
  │    side effect: index_k_with_scale_buffer[out_cache_loc] ← (k_fp8, k_scale)
  │
  ├─ A4  sparse MQA in latent space   (forward_absorb_core)
  │    q = cat(q_nope_out, q_pe) [N, Nh, Rkv+Dro=576] ;  k = cat(k_nope, k_pe) [N, 1, 576]
  │    attn_out_latent [N, Nh, Rkv=512] = attn_mqa(q, k, k_nope, forward_batch, topk_indices=topk_indices)
  │        - save_kv_cache=True 时由 attn 内部把 [1,576] 写主 latent KV cache[out_cache_loc]
  │        - NSA backend kernel 见 §12.2（num_kv_heads=1, v_head_dim=Rkv）
  │
  ├─ A5  v_absorb
  │    attn_out [N, Nh, Dv=128] = bmm(attn_out_latent.T[Nh,N,Rkv], w_vc[Nh,Rkv,Dv]).T
  │    flatten → [N, Nh*Dv = 4096]
  │
  └─ A6  o_proj
       out [N, H=2048] = RowParallelLinear(attn_out)
       (next_skip_topk 机制见 §15.2；默认返回 out)
```

---

## 11. Indexer 子流程（`nsa_indexer.py::Indexer.forward_cuda`）

输入：`x = hidden_states [N,H]`、`q_lora = q_a_layernorm 输出 [N,Rq]`（注意是 norm 后、`q_b_proj` 前的那一份）、`positions [N]`、`forward_batch`、`layer_id`。

### 11.1 Query / Key / Gate 投影

```text
q_lora [N,Rq]  ── wq_b ──→  query [N, I=8, Di=128]
hidden [N,H]   ── wk   ──→  key   [N, Di=128]  ── k_norm (完整 LayerNorm, fp32) ──→  key
hidden [N,H]   ── weights_proj ──→  weights [N, I=8]   (bf16→fp32) ;  weights *= I**-0.5
```

### 11.2 Indexer RoPE（NeoX，前 64 维） + Hadamard rotation

```text
q_rope = query[..., :Dro=64] ;  k_rope = key[..., :Dro=64]
q_rope, k_rope = indexer.rotary_emb(positions, q_rope, k_rope)        # NeoX
query[..., :64] = q_rope ;  key[..., :64] = k_rope ;  后 64 维不动
query = rotate_activation(query)   # Hadamard 正交旋转, scale = Di**-0.5 = 128**-0.5, bf16
key   = rotate_activation(key)     # 对 REF 是正交变换；主要为把 outlier 摊开利于 FP8 量化
```

> Prefill CP 开启时，`key` 在 RoPE + Hadamard 后经 `cp_all_gather_rerange_output()` 汇合成完整 indexer K 视图（§15.1）。

### 11.3 FP8 量化 + index K cache store

```text
q_fp8, q_scale = act_quant(query, block_size=128, scale_fmt="ue8m0")
weights = weights.unsqueeze(-1) * q_scale * softmax_scale         # softmax_scale = Di**-0.5 = 128**-0.5 → weights [N, I, 1]

store key:
  preferred: fused_store_index_k_cache(key, index_k_with_scale_buffer, out_cache_loc, page_size)
  fallback : k_fp8, k_scale = act_quant(key, 128, "ue8m0"); set_index_k_scale_buffer(layer_id, out_cache_loc, k_fp8, k_scale)
```

### 11.4 logits + topk transform

- **decode / target-verify / draft-extend**（`_get_topk_paged`）：

```text
logits [B, max_seq_len] = deep_gemm.fp8_paged_mqa_logits(
    q_fp8 [B, next_n=1, I, Di], index_k_with_scale_buffer.view(num_pages, 64, 1, 132),
    weights [B, I], seqlens_int32, block_table_64, paged_mqa_schedule_metadata, max_seq_len)
topk_result [B, Ktop=2048] = metadata.topk_transform(logits, Ktop)   # int32；序列短于 Ktop 时尾部填 -1
```

- **prefill ragged**（`_get_topk_ragged`）：

```text
(k_fp8 [Ktot, Di], k_scale [Ktot]) = gather from index K cache (page indices)
ks, ke = 每 token 可见 key 区间 = [prefix(req) 起, prefix(req) + causal 当前 chunk 前缀]
logits [T, Ktot] = deep_gemm.fp8_mqa_logits(q_fp8 [T, I, Di], (k_fp8, k_scale), weights [T, I], ks, ke)
topk_result [T, Ktop] = metadata.topk_transform(logits, Ktop, ks=ks)
# OOM 时按行 chunk 重算
```

- **prefill 短序列快捷路径**（`_forward_cuda_k_only`，当 `max_kv_len <= index_topk = 2048` 且非 CP）：**只**算 K 并写 index cache，跳过所有 q / weights / logits 运算；`topk_transform` 用 `dummy_logits` 走 kernel fast path 直接生成 `[0, 1, ..., valid_len-1, -1, ...]`（等价"全选可见 token" → sparse 退化为 dense）。

- **NPU**：`forward_npu` + `torch_npu.npu_lightning_indexer`，行为对齐但 kernel 不同；本切片以 CUDA / 通用语义为准。

---

## 12. NSA backend 与 sparse attention（`nsa_backend.py::NativeSparseAttnBackend`）

`init_forward_metadata()` 为每个 batch 构造：

| 字段 | 用途 |
|---|---|
| `cache_seqlens_int32` | 当前真实 KV 长度 |
| `page_table_1` | token 粒度 page table |
| `real_page_table` | 按真实 `page_size` 折算后的 page table；CUDA NSA indexer paged path 用 `page_size=64` |
| `nsa_cache_seqlens_int32` | `min(seq_len, index_topk)` 后的 sparse attention KV 长度 |
| `nsa_seqlens_expanded` | prefill 每个 query row 的可见 KV 长度 |
| `topk_indices_offset` | ragged topk 结果转连续 KV 时的 row offset |
| `paged_mqa_schedule_metadata` | decode/verify/draft paged MQA logits 的 DeepGEMM schedule |

### 12.1 Prefill dense / sparse 判定（`set_nsa_prefill_impl`）

普通 extend 中决定是否走 `MHA_ONE_SHOT`：

```text
self.use_mha =
      device_sm in {90} or 100 <= device_sm < 110          # H200 / B200 一类
  and max_kv_len <= SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD   # 默认 2048（= index_topk）
  and kv_cache.dtype in {bfloat16, fp8_e4m3}
  and sum_seq_lens <= forward_batch.get_max_chunk_capacity()
  and not is_nsa_enable_prefill_cp()
  and hisparse_coordinator is None
```

否则一律 absorb。Decode / target-verify / draft-extend 始终 `use_mha=False`。对自动进入 `_handle_model_specific_adjustments()` 的 DSA architecture，server args 在未手动设置时把 dense threshold 设为模型 `index_topk = 2048`。

### 12.2 Sparse MLA attention backend 变体

主 sparse MLA 调用经 `RadixAttention` → `NativeSparseAttnBackend.forward_decode()` / `forward_extend()`：

| backend | sparse kernel | 备注 |
|---|---|---|
| `flashmla_sparse` | `flash_mla_sparse_fwd(q, kv, indices, d_v=Rkv)` | prefill 若 topk method 是 RAGGED，可能直接用当前 chunk concat KV 或 dequantized paged KV |
| `flashmla_kv` | `flash_mla_with_kvcache(..., indices, head_dim_v=Rkv)` | 要求 indices last dim == `index_topk`；FP8 K cache 分支会先 `dequant_k_cache` |
| `fa3` | `flash_attn_with_kvcache(q_rope, k_cache_rope, v_cache_latent, qv=q_nope)` | 仍输出 latent `Rkv` |
| `tilelang` | `tilelang_sparse_fwd` 或 `tilelang_kernel_glm.sparse_mla_fwd_interface` | `q_all.shape[-1] == v_head_dim` 时识别 no-rope GLM path |
| `trtllm` | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` | fp8 path 可融合 query rope/quant/cache |

所有 sparse MLA backend 在 absorb path 的输出语义统一为 `attn_out_latent [N, Nh_local, Rkv]`，之后必须经 `w_vc` 解吸收 + `o_proj`。

### 12.3 MHA_ONE_SHOT dense fallback

`use_mha=True` 时 `handle_attention_nsa()` 返回 `AttnForwardMethod.MHA_ONE_SHOT`，走 `forward_mha.py::forward_normal_one_shot_prepare/core`：

```text
q = q_b_proj(q_a_layernorm(q_lora_raw)).view(T, Nh, Dqk=192)        # q_pe 未旋转 (mla_nope)
kv_a = kv_a_layernorm(latent_cache[..., :Rkv]) ;  k_pe = latent_cache[..., Rkv:]   # 未旋转
indexer 仍跑（return_indices=False）只为填 index K cache；写 latent KV cache(attn_mha)
kv = kv_b_proj(kv_a).view(T, Nh=32, Dnope+Dv=256) → k_nope[Dnope], v[Dv]
k  = cat(k_nope, k_pe broadcast) [T, 32, 192]
attn_out_dense [T, Nh, Dv=128] = attn_mha(q, k, v, forward_batch)   # FlashAttention varlen / trtllm_ragged，全 prefix+current causal
   ★ 跳过 V 吸收，直接进 o_proj
```

Zeus DSA sparse 算子先按 `use_mha=False` 主路径实现；dense fallback 作为独立策略后补。

---

## 13. Decode Path（`N=B`，每 request 1 query token）

**走 `AttnForwardMethod.MLA`（absorb）**：sparse MQA 在潜空间完成，再 `w_vc` 解吸收回 `[B,Nh,Dv]`，最后 `o_proj`。

### 13.1 计算流

```text
hidden_t [B,H]   (来自 layer_communicator.prepare_attn；MHC 残差已处理)
  │
  ├─ D0  dsa_pre_attn_qkv_latent  (layer communicator: prepare_qkv_latent)
  │     fused_qkv_a_proj_with_mqa(hidden_t) → q_lora_raw [B,768] + latent_cache [B,576]
  │
  ├─ D1  dsa_qkv_a_norm  (forward_absorb_prepare 起手)
  │     q_lora = q_a_layernorm(q_lora_raw) [B,768]
  │     k_nope = kv_a_layernorm(latent_cache[..., :512]).unsqueeze(1) [B,1,512]
  │     k_pe   = latent_cache[..., 512:].unsqueeze(1) [B,1,64]   (未旋转)
  │
  ├─ D2  dsa_q_b_absorb
  │     q = q_b_proj(q_lora).view(B, 32, 192) → q_nope [B,32,128], q_pe [B,32,64] (未旋转)
  │     q_nope_out = bmm(q_nope.T, w_kc).T [B,32,512]
  │
  ├─ D3  dsa_indexer_store_topk_decode  (§11；alt_stream 上与 D2 overlap)
  │     x=hidden_t, q_lora=D1 的 q_lora
  │     query[B,8,128] = rotate_activation(rope_neox_first64(wq_b(q_lora)))
  │     key  [B,128]   = rotate_activation(rope_neox_first64(k_norm(wk(x))))
  │     q_fp8,q_scale = act_quant(query); k_fp8,k_scale = act_quant(key)
  │     side effect: index_k_with_scale_buffer[out_cache_loc] ← (k_fp8,k_scale)
  │     gate[B,8]=weights_proj(x)*8^-0.5; weights=gate.unsqueeze(-1)*q_scale*128^-0.5
  │     logits[B,max_seq_len] = deep_gemm.fp8_paged_mqa_logits(q_fp8[B,1,8,128],
  │                              index_k_cache[num_pages,64,1,132], weights[B,8],
  │                              seqlens_i32, block_table_64, schedule_md, max_seq_len)
  │     topk_slots[B,2048] = metadata.topk_transform(logits, 2048)   (尾部 -1)
  │
  ├─ D4  dsa_sparse_mqa_decode  ★ latent 空间 sparse MQA  (forward_absorb_core)
  │     q = cat(q_nope_out, q_pe) [B,32,576] ;  k = cat(k_nope,k_pe) [B,1,576]
  │     attn_out_latent[B,32,512] = attn_mqa(q,k,k_nope,forward_batch,topk_indices=topk_slots)
  │         (NSA backend: flashmla_kv / flashmla_sparse / tilelang / fa3 / trtllm；num_kv_heads=1,d_v=512;
  │          save_kv_cache=True → 写主 latent KV cache[out_cache_loc] = [1,576])
  │
  ├─ D5  dsa_v_absorb
  │     attn_out[B,32,128] = bmm(attn_out_latent.T, w_vc).T
  │
  └─ D6  dsa_o_proj
        out[B,2048] = o_proj(attn_out.reshape(B, 4096))
```

### 13.2 Decode 算子依赖表

| # | 子步骤 | shape / IO | REF 入口（REPO） | Zeus 状态 |
|---|---|---|---|---|
| D0 | `dsa_pre_attn_qkv_latent` | `hidden [B,2048] → q_lora_raw [B,768] + latent_cache [B,576]` | `deepseek_v2.py::DeepseekV2AttentionMLA.prepare_qkv_latent`（在 `communicator.py::fetch_qkv_latent` 内调用） | × |
| D1 | `dsa_qkv_a_norm` | `q_lora_raw → q_lora [B,768]`；`latent[:512] → k_nope [B,1,512]`；`latent[512:] → k_pe [B,1,64]` | `forward_mla.py::forward_absorb_prepare`（`q_a_layernorm` / `kv_a_layernorm`） | × |
| D2 | `dsa_q_b_absorb` | `q_lora → q_nope_out [B,32,512] + q_pe [B,32,64]` | `forward_absorb_prepare`（`q_b_proj` + split + `bmm w_kc`），main MLA RoPE 跳过 | × |
| D3 | `dsa_indexer_store_topk_decode` | `hidden + q_lora + index cache → topk_slots [B,2048]`；写 index K cache | `nsa_indexer.py::Indexer.forward_cuda` → `_get_q_k_bf16` + `rotate_activation` + `act_quant` + `_store_index_k_cache` + `_get_topk_paged`（`deep_gemm.fp8_paged_mqa_logits` + `metadata.topk_transform`，见 `transform_index.py`） | × |
| D4 | `dsa_sparse_mqa_decode` | `q [B,32,576] + 主 latent KV cache[topk_slots] → attn_out_latent [B,32,512]` | `forward_absorb_core` → `attn_mqa(..., topk_indices=...)` → `nsa_backend.py::forward_decode`（`_forward_flashmla_kv` / `_forward_flashmla_sparse` / `_forward_tilelang`，`num_kv_heads=1, d_v=Rkv=512`） | × |
| D5 | `dsa_v_absorb` | `attn_out_latent [B,32,512] → attn_out [B,32,128]` | `forward_absorb_core` 末段 `bmm w_vc` | × |
| D6 | `dsa_o_proj` | `[B,4096] → [B,2048]` | `RowParallelLinear` | × |

---

## 14. Prefill / Extend Path（`N=T=sum(extend_seq_lens)`）

每 query row 可见 key 范围 = `prefix(req) + current_chunk(req, <= row_pos)`（causal）。
**默认走 `AttnForwardMethod.MLA`（absorb）**，与 decode 共用 `forward_absorb_prepare/core`，sparse 调用换 ragged 版。`MHA_ONE_SHOT` dense fallback 仅当 `set_nsa_prefill_impl` 判 `use_mha=True`（§12.1）。

### 14.1 计算流

```text
hidden [T,H], positions [T], out_cache_loc [T], ragged metadata
  │
  ├─ P0  dsa_pre_attn_qkv_latent      (同 D0，T batched；layer communicator 内)
  ├─ P1  dsa_qkv_a_norm               (同 D1；KV cache 批量写 out_cache_loc)
  ├─ P2  dsa_q_b_absorb               (同 D2)
  │
  ├─ P3  dsa_indexer_store_topk_prefill  (§11.4)
  │     若 max_kv_len <= 2048 (非 CP): _forward_cuda_k_only —— 只写 index K，
  │       topk = topk_transform(dummy_logits) = [0..valid_len-1, -1, ...]   (全选可见)
  │     否则: _get_topk_ragged
  │       gather (k_fp8[Ktot,Di], k_scale[Ktot]); ks,ke = 每 token 可见区间
  │       logits[T,Ktot] = deep_gemm.fp8_mqa_logits(q_fp8[T,8,128],(k_fp8,k_scale),weights[T,8],ks,ke)
  │       topk_slots[T,2048] = metadata.topk_transform(logits, 2048, ks=ks)
  │     side effect: index_k_with_scale_buffer[out_cache_loc] ← (k_fp8,k_scale)
  │
  ├─ P4  dsa_sparse_mqa_prefill  ★ absorb 后 sparse MQA（默认）  (forward_absorb_core)
  │     q = cat(q_nope_out, q_pe) [T,32,576]
  │     attn_out_latent[T,32,512] = attn_mqa(q,k,k_nope,forward_batch,topk_indices=topk_slots)
  │         (NSA backend ragged: flash_mla_sparse_fwd / flashmla_kv / tilelang_sparse_fwd; d_v=512)
  │
  ├─ P4-alt  dense_fallback (use_mha=True, 短序列+特定硬件)  —— forward_normal_one_shot_prepare/core
  │     见 §12.3；输出 attn_out_dense [T,32,128]，★ 跳过 V 吸收直接进 P6
  │
  ├─ P5  dsa_v_absorb            仅 absorb path：attn_out[T,32,128] = bmm(attn_out_latent.T, w_vc).T
  └─ P6  dsa_o_proj              [T, 4096] → [T, 2048]
```

### 14.2 Prefill 算子依赖表

| # | 子步骤 | shape / IO | REF 入口（REPO） | Zeus 状态 |
|---|---|---|---|---|
| P0 | `dsa_pre_attn_qkv_latent` | `hidden [T,2048] → q_lora_raw [T,768] + latent_cache [T,576]` | 同 D0 | × |
| P1 | `dsa_qkv_a_norm` | `q_lora [T,768] + k_nope [T,1,512] + k_pe [T,1,64]`；批量写 latent KV cache | `forward_absorb_prepare` | × |
| P2 | `dsa_q_b_absorb` | `q_nope_out [T,32,512] + q_pe [T,32,64]` | `forward_absorb_prepare` | × |
| P3 | `dsa_indexer_store_topk_prefill` | `hidden + q_lora + ragged metadata → topk_slots [T,2048]`；短序列走 K-only 快捷路径；批量写 index K cache | `nsa_indexer.py::Indexer.forward_cuda` → `_get_topk_ragged` / `_forward_cuda_k_only`（`deep_gemm.fp8_mqa_logits` + `topk_transform`） | × |
| P4 | `dsa_sparse_mqa_prefill` | `q [T,32,576] + 主 latent KV cache + topk_slots → attn_out_latent [T,32,512]` | `forward_absorb_core` → `attn_mqa(..., topk_indices=...)` → `nsa_backend.py::forward_extend`（`_forward_flashmla_sparse` / `_forward_flashmla_kv` / `_forward_tilelang`） | × |
| P4-alt | `dense_fallback_policy` | `use_mha` 判定 + `MHA_ONE_SHOT`（不吸收，输出 `[T,32,128]`） | `nsa_backend.py::set_nsa_prefill_impl` + `attention_backend_handler.py::handle_attention_nsa` + `forward_mha.py::forward_normal_one_shot_prepare/core`（indexer `return_indices=False`） | × |
| P5 | `dsa_v_absorb` | `[T,32,512] → [T,32,128]` | `forward_absorb_core` 末段 `bmm w_vc` | × |
| P6 | `dsa_o_proj` | 同 D6 | `RowParallelLinear` | × |

---

## 15. CP 与跨层 topk 复用（接口约束，本切片不实现）

### 15.1 Prefill CP

`Glm5NextForCausalLM.forward()` 在 `enable_nsa_prefill_context_parallel` 且 `can_cp_split()` 满足时准备 `forward_batch.nsa_cp_metadata`；`Glm5NextModel.forward()` 做 `hidden_states = cp_split_and_rebuild_data(...)`、`positions = cp_split_and_rebuild_position(...)`。KDA 层不支持 CP，进 KDA 前 `cp_all_gather_rerange_output()`、出 KDA 后再 split。DSA 层的主 latent KV 与 index K 各有 CP all-gather / rerange 处理（`rebuild_cp_kv_cache(latent_cache, ...)`、indexer K 在 RoPE+Hadamard 后 all-gather）。

Zeus 首版不实现 CP，但接口需保留：`nsa_cp_metadata`、split 后 `positions`、indexer K all-gather、`rebuild_cp_kv_cache`。

### 15.2 跨层 topk 复用

`DeepseekV2AttentionMLA` 支持 `index_topk_freq` / `index_topk_pattern`，推出每层的 `skip_topk`（是否复用上一层 `topk_indices`，跳过自己的 indexer logits/topk）与 `next_skip_topk`（是否把本层 topk 传给下一层）。`Glm5NextModel.forward()` 层循环里维护：

```text
topk_indices = None
for layer:
    hidden_states, residual, topk_indices = layer(..., prev_topk_indices=topk_indices)
```

`forward_absorb_core()` 在 `next_skip_topk is not None` 时返回 `(output, topk_indices_or_None)`，否则只返回 `output`。当前 config 默认 `index_topk_pattern=None` ⇒ `next_skip_topk` 全 None ⇒ 不复用、每层各算。Zeus 端若实现跨层 index cache/复用，必须保持这个返回语义。

---

## 16. dev 脚本对应

dev 脚本侧 stage 名沿用现有 `dev_glm_moe_dsa_{decode,prefill}_test*.py` 命名，按 V5 真实链路调整覆盖。

| dev stage | 覆盖范围 | 对应子步骤 | Zeus 状态 |
|---|---|---|---|
| `decode_qkv_a_proj_norm_fused` / `prefill_*` | `hidden → fused_qkv_a_proj_with_mqa → qkv_latent`，再 split + `q_a_layernorm` + `kv_a_layernorm` | D0/P0 + D1/P1 | × |
| `decode_q_proj_fused` / `prefill_*` | `q_lora_norm → q_b_proj → split → bmm(q_nope, w_kc)`，输出 `q_nope_out [*,Nh,Rkv]` + `q_pe`（未旋转） | D2/P2 | × |
| `decode_kv_cache_store` / `prefill_*` | latent KV 行 `concat(k_nope, k_pe) [1, Rkv+Dro=576]` 写主 KV cache（Zeus kernel `dsa_kv_proj_cache_store_fused` 已落地） | D1/P1 side-effect | ✓ |
| `decode_indexer_prep_store_fused` / `prefill_*` | indexer Q/K（含 NeoX RoPE 前 64 维）+ `k_norm`（完整 LayerNorm）+ Hadamard + FP8 `act_quant` + index K cache store + gate（fp32） | D3/P3（store 部分） | × |
| `decode_indexer_topk_fused` | paged MQA logits + topk transform；支持尾部 `-1` 填充 | D3 | × |
| `prefill_ragged_indexer_topk_fused` | ragged causal logits + topk；短序列 K-only 快捷路径（dummy → 全选可见 token） | P3 | × |
| `decode_sparse_mqa_fused` / `prefill_sparse_mqa_fused` | latent 空间 sparse MQA；输出 `[*,Nh,Rkv=512]` | D4/P4 | × |
| `dense_fallback_policy` | `use_mha` 判定 + `MHA_ONE_SHOT`（不吸收，输出 `[T,Nh,Dv=128]`） | P4-alt | × |
| `decode_v_absorb` / `prefill_v_absorb` | `bmm(attn_out_latent, w_vc)`，输出 `[*,Nh,Dv=128]` | D5/P5 | × |
| `decode_o_proj` / `prefill_o_proj` | output projection `[*,4096] → [*,2048]` | D6/P6 | × |
| `decode_full_path` / `prefill_full_path` | D0–D6 / P0–P6 端到端 | 全链路 | × |

> 注意 §4：dev REF 脚本里 main MLA 路径**不要**插 RoPE；只 indexer 前 64 维做 NeoX RoPE。
> `derive_w_kc_w_vc` 要匹配 §6 的 `w_kc [Nh,Dnope,Rkv]` / `w_vc [Nh,Rkv,Dv]` 形态。
> sparse MLA 输出契约是 `[*,Nh_local,Rkv]`，之后显式 `w_vc` → `[*,Nh_local,Dv]`，不要直接出 `Dv`。

---

## 17. 里程碑与开发优先级

按"先正确性 → 吸收路径 → sparse → fallback → FP8/CP/复用"排序。

| 优先级 / Milestone | 任务 | 通过判据 |
|---|---|---|
| M0 | 文档 + pure-torch REF（按 V5 链路：main MLA 无 RoPE、indexer NeoX、absorb 在潜空间、sparse 输出 `[*,Nh,Rkv]` 再 `w_vc`） | `dev_glm_moe_dsa_{decode,prefill}_test*.py --mode ref` 端到端与 REPO REF 中间张量 shape / 数值一致；REF 显式产出 `qkv_latent`、`q_lora`、`k_nope`、`k_pe`、`q_nope_out`、`q_pe`、`q_fp8`/`weights`、`topk_slots`、`attn_out_latent` |
| M1 | D0/P0 `dsa_pre_attn_qkv_latent`（fused `fused_qkv_a_proj_with_mqa` GEMM） + D1/P1 `q_a_layernorm` / `kv_a_layernorm` split | `*_qkv_a_proj_norm_fused --mode zeus` 输出 `q_lora + k_nope + k_pe` 对齐 REF |
| M2 | D2/P2 Q 通路含 absorb（`q_b_proj` + split + `bmm w_kc`，**不对主 MLA `q_pe/k_pe` 做 RoPE**） | `*_q_proj_fused --mode zeus` 输出 `q_nope_out [*,Nh,Rkv]` + `q_pe` 对齐 REF；latent KV cache store side-effect 一致 |
| M3 | D3/P3 indexer 通路（`wq_b`/`wk` + NeoX RoPE 前 64 维 + `k_norm` LayerNorm + Hadamard + FP8 `act_quant` + index K cache store + gate fp32） | `*_indexer_prep_store_fused --mode zeus` 写入/读回一致；`weights` fp32 数值对齐 |
| M4 | D3 decode paged top-k | `decode_indexer_topk_fused --mode zeus` 支持尾部 `-1` 填充，对齐 REF |
| M5 | D4 decode sparse MQA（潜空间，`num_kv_heads=1, d_v=Rkv`） | `decode_sparse_mqa_fused --mode zeus` 输出 `[B,Nh,Rkv=512]` 对齐 REF |
| M6 | D5/P5 V absorb（`bmm w_vc`） + D6/P6 `o_proj` | `*_v_absorb` / `*_o_proj --mode zeus` 对齐 REF；DSA layer `[*,H] → [*,H]` 端到端 |
| M7 | P3 prefill ragged top-k（含短序列 K-only 快捷路径） | `prefill_ragged_indexer_topk_fused --mode zeus` 多请求不串行、不看未来、不跨 request；短序列退化为全选可见 |
| M8 | P4 prefill sparse MQA + P4-alt dense fallback policy | absorb 路径 `[T,Nh,Rkv]` 对齐 REF；`dense_fallback_policy --mode zeus` 在 `max_kv_len <= 2048` + 模拟硬件条件下与 `MHA_ONE_SHOT` REF（`[T,Nh,Dv]`，不吸收）一致 |
| M9 | GLM5-Next real-shape smoke | DSA 层 `[3,7,11,15,19,23]` 单层 `[T,H] → [T,H]` 端到端 |
| M10 | FP8 K cache / index cache + CP + 跨层 topk 复用 | `quant_k_cache`/`dequant_k_cache` + `NSATokenToKVPool` 落地；评估 `index_topk_freq`/`index_topk_pattern`（当前默认不复用）；CP 接口对齐 |

---

## 18. 参考源码索引（REPO 路径）

| 标签 | 文件 | 关注点 |
|---|---|---|
| `R-ENTRY` | `models/glm5_next.py::Glm5NextForCausalLM` / `Glm5NextModel` / `Glm5NextDecoderLayer` / `Glm5NextLinearAttention` | 模型主体；DSA 子层 = `DeepseekV2AttentionMLA`（`skip_rope=mla_nope`）；`qkv_latent_func=prepare_qkv_latent` |
| `R-CFG` | `configs/model_config.py::is_deepseek_nsa / get_nsa_index_*`；`configs/glm_linear.py::GlmLinearConfig`（`is_kda_layer`、`is_mla` 触发含 `mla_nope is True`） | NSA 判定、层分流 |
| `R-MLA` | `models/deepseek_v2.py::DeepseekV2AttentionMLA`（`__init__`、`prepare_qkv_latent`、`dispatch_attn_forward_method`、`forward_prepare/core`、`attn_mqa`/`attn_mha` RadixAttention 构造 l.1269/1280） | hidden-in fused projection + attention dispatch + absorb 骨架 |
| `R-ABSORB` | `models/deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare / forward_absorb_core` | decode + prefill 共用 absorb：`q_a/kv_a layernorm` → `q_b_proj` → split → `bmm w_kc` → `attn_mqa(..., topk_indices=...)` → `bmm w_vc` → `o_proj`；`next_skip_topk` 返回语义 |
| `R-MHA1S` | `models/deepseek_common/attention_forward_methods/forward_mha.py::forward_normal_prepare / forward_normal_one_shot_prepare/core` | `MHA_ONE_SHOT` dense fallback（`kv_b_proj` decompress → FlashAttention varlen，不吸收）；indexer `return_indices=False` 仅填 index K cache |
| `R-DISPATCH` | `models/deepseek_common/attention_backend_handler.py::handle_attention_nsa` | 读 `NSA backend.use_mha` → `MHA_ONE_SHOT` / `MLA` |
| `R-WLOADER` | `models/deepseek_common/deepseek_weight_loader.py::post_load_weights`（`w_kc`/`w_vc` 拆分 l.565-585） | `kv_b_proj.weight` → `w_kc [Nh,Dnope,Rkv]` / `w_vc [Nh,Rkv,Dv]` |
| `R-INDEXER` | `layers/attention/nsa/nsa_indexer.py::Indexer`（`forward_cuda` / `_get_q_k_bf16` / `_get_topk_paged` / `_get_topk_ragged` / `_forward_cuda_k_only` / `_store_index_k_cache`；`rotate_activation` l.135；`forward_npu`） | indexer Q/K/gate + NeoX RoPE 前 64 维 + `k_norm` LayerNorm + Hadamard + FP8 `act_quant` + paged/ragged logits + topk transform + index K cache store |
| `R-TOPK` | `layers/attention/nsa/transform_index.py`（`metadata.topk_transform`）+ `triton_kernel.py` / `tilelang_kernel*.py` | logits → topk slots（`-1` padding、causal 约束、跨 backend 变体） |
| `R-BACKEND` | `layers/attention/nsa_backend.py::NativeSparseAttnBackend`（`init_forward_metadata`、`set_nsa_prefill_impl`、`forward_decode`/`forward_extend`、`_forward_flashmla_sparse`/`_forward_flashmla_kv`/`_forward_tilelang`、`get_indexer_metadata`） | NSA metadata、dense/sparse 调度、sparse MLA kernel 包装 |
| `R-CACHE` | `mem_cache/memory_pool.py::NSATokenToKVPool`（继承 `MLATokenToKVPool`）；`layers/attention/nsa/index_buf_accessor.py`（`GetK`/`GetS`/`GetKAndS`/`SetKAndS`） | 主 latent KV cache `[*,64,1,576]` + `index_k_with_scale_buffer [*,64*132]` |
| `R-QUANT` | `layers/attention/nsa/quant_k_cache.py` / `dequant_k_cache.py` | FP8 K cache 量化 / 反量化（`flashmla_kv` 分支用） |
| `R-FUSEDSTORE` | `jit_kernel/fused_store_index_cache.py`（`fused_store_index_k_cache` / `can_use_nsa_fused_store`）；`jit_kernel/hadamard.py`（`hadamard_transform`） | index K cache 融合写；Hadamard JIT |
| `R-COMM` | `layers/communicator.py`（`fetch_qkv_latent`、`get_attn_tp_context`）；`layers/communicator_mhc.py::MHCLayerCommunicator`；`layers/communicator_mhc_hybrid_cp.py`；`layers/communicator_nsa_cp.py`；`layers/mhc.py::HyperConnection` | layer communicator：`prepare_attn` 内执行 `prepare_qkv_latent`；MHC 残差 |
| `R-CP` | `layers/attention/nsa/utils.py`（`can_cp_split`、`cp_all_gather_rerange_output`、`cp_split_and_rebuild_data/position`、`is_nsa_enable_prefill_cp`、`nsa_use_prefill_cp`、`prepare_input_dp_with_cp_dsa`） | prefill context parallel 切分/汇合 |
| `R-SRV` | `server_args.py`（`_handle_model_specific_adjustments`：NSA backend architecture 列表、dense fallback threshold 默认 = `index_topk`） | 启动期默认值 |
