# GLM-MoE-DSA / GLM5-Next DSA Zeus 适配开发追踪（V3）

> 基线：V2_a；融合 V2_b 的 RoPE interleave 细节、head 语义表、测试线索；并保留对 V2_c
> 的两处实质性纠正（prerelease 实有 `GlmMoeDsaForCausalLM`；DSA prefill 默认走 absorb）。
> 详见同目录 `GlmMoeDsa_dev_v2_comparison.md`。
>
> 校对依据：`/root/project/sglang-v0.5.10-prerelease`，
> canonical config：`./config_16b.json`（`architectures: ["Glm5NextForCausalLM"]`）。

## 0. V3 相对 V2_a 的修订

| 段 | V2_a → V3 改动 | 来源 |
|---|---|---|
| §2.2 RoPE | 新增 main MLA vs Indexer 的 `is_neox_style` 默认值对照（一个缺省 interleave、一个缺省 NeoX——非常容易踩坑） | V2_b（已回 `deepseek_v2.py:1181/1244` 核对） |
| §2.3 head 语义 | 表头扩 4 列：`运行时 KV 头数 / latent dim / decompress 后 V dim / cache layout`，避免 V2_c 那种 "32 个隐式 KV heads" 误述 | V2_b 表 + 源码 |
| §5 GLM5 脚手架 | 加测试 / parser 线索：`test_glm5_fp8.py` / `test_glm5_nvfp4.py` / `Glm5MoeDetector` / reasoning `GLM5Detector`；并指出生产 HF 模型路径 `zai-org/GLM-5-FP8` 的 architecture 必然是 `GlmMoeDsaForCausalLM` | V2_b |
| §6.5（新） | 开发优先级建议清单（dev-stage 不变前提下的工程排序） | V2_b 风格 |
| §11 V1→V3 速查 | 替换原 V1→V2_a 速查表 | — |

## 1. 范围与方法

- **起点**：DSA layer 输入的 `hidden_states [N,H]`。
- **终点**：`o_proj` 输出 `[N,H]`，可直接进入 post-attention residual / layernorm。
- **切片**：单 device、单 layer、DSA full-attention only；不覆盖 KDA linear attention、
  MoE-FFN、TP/PP/EP、NextN、CP (context parallel)、speculative decoding。
- **层范围**：`linear_attn_config.full_attn_layers = [3, 7, 11, 15, 19, 23]`，
  其余层是 KDA 线性注意。
- **runtime path 划分**：Decode（单步 token，paged cache）与 Prefill / extend（多 token ragged chunk，批量写 cache）。
- **状态约定**：`×` 表示 Zeus DSA kernel 尚未落地或尚未接入 path 级对齐测试。

## 2. GLM5-Next 16B 关键配置

来自 `./config_16b.json`（`architectures: ["Glm5NextForCausalLM"]`，`model_type: "glm4_moe"`）。

### 2.1 维度

| 字段 | 值 | 含义 |
|---|---:|---|
| `hidden_size` (H) | 2048 | residual hidden 维度 |
| `num_attention_heads` (Nh) | 32 | **MLA Q 头数**（`q_b_proj` 输出 `Nh*Dqk`） |
| `num_key_value_heads` | 8 | **MLA 路径下不被使用**（见 §2.3） |
| `q_lora_rank` (Rq) | 768 | Q low-rank hidden |
| `kv_lora_rank` (Rkv) | 512 | KV low-rank latent（=潜空间 V dim） |
| `qk_nope_head_dim` (Dnope) | 128 | Q/K non-RoPE 维度 |
| `qk_rope_head_dim` (Dro) | 64 | Q/K RoPE 维度 |
| `qk_head_dim` (Dqk) | 192 | `Dnope + Dro` |
| `v_head_dim` (Dv) | 128 | 解吸收后的 value head dim |
| `index_n_heads` (I) | 8 | **indexer 头数**（与 Nh 无关） |
| `index_head_dim` (Di) | 128 | indexer 每 head 维度 |
| `index_topk` (Ktop) | 2048 | 每 query sparse token 数 |
| `index_dsa_use_layernorm` | true | indexer K 上挂 LayerNorm（对应 SG-I0 `Indexer.k_norm`） |

### 2.2 位置编码 / Norm

`config_16b.json` **不缺 RoPE 配置**，相关字段：

| 字段 | 值 | 含义 |
|---|---:|---|
| `max_position_embeddings` | 202752 | 长上下文上限 |
| `rope_theta` | 10000 | RoPE 基频 |
| `rope_scaling` | null | **不开 YaRN/线性 scaling** |
| `partial_rotary_factor` | 0.5 | 仅一半 head dim 做旋转（128×0.5 = 64 = `qk_rope_head_dim`） |
| `use_qk_norm` | true | Q/K 在 RoPE 之后做 RMSNorm（GLM-4 系列风格） |
| `rms_norm_eps` | 1e-05 | norm eps |

#### 2.2.1 main MLA vs Indexer 的 RoPE 风格默认值（V3 新增）

源码：`python/sglang/srt/models/deepseek_v2.py`

| RoPE 路径 | 字面公式 | config 未显式指定时缺省值 | 结果 `is_neox_style` |
|---|---|---|---:|
| main MLA Q/K RoPE | `is_neox_style = not getattr(config, "rope_interleave", True)` (l.1244) | `rope_interleave = True` | **False（即 interleave / GPT-J 风格）** |
| Indexer Q/K RoPE | `is_neox_style = not getattr(config, "indexer_rope_interleave", False)` (l.1181) | `indexer_rope_interleave = False` | **True（即 NeoX 风格）** |

`config_16b.json` 两个字段都没写，所以**两路 RoPE 走的风格是反的**——main 是 interleave、indexer 是 NeoX。
Zeus 对齐 REF 时这两路必须分别核 cos/sin 排布，不能共用同一段 RoPE 代码。

indexer 共用主路径的 `qk_rope_head_dim=64` 与 `rope_theta=10000`，对 `[*, I=8, Di=128]`
indexer Q 与 `[*, Di=128]` indexer K 的**前 64 维**做旋转，且 indexer K 之上额外有
`k_norm`（`index_dsa_use_layernorm: true` 触发）。

### 2.3 "head 数" 四件套（V3 强化）

| 名称 | config 字段 | 值 | 运行时角色 | KV cache layout |
|---|---|---:|---|---|
| Q 头 (Nh) | `num_attention_heads` | 32 | `q_b_proj` 输出 reshape 为 `[*, 32, Dqk=192]`；absorb 后 `q_nope @ w_kc` 转成 `[*, 32, Rkv=512]` | — |
| GQA KV 头 | `num_key_value_heads` | 8 | **MLA/DSA 路径下不被使用**。只在 `Glm4MoeAttention`（非 MLA 的 GLM-4.x GQA 路径）下当作 KV 头数。`mla: true` 时直接忽略 | — |
| 潜空间 KV 头 (h_kv) | —（不在 config 里） | **1** | absorb fast path 下真正落 cache 的 KV 头数（MQA in latent space）；`attn_mqa = RadixAttention(num_kv_heads=1, v_head_dim=Rkv=512)` | `[num_pages, page_size, 1, Rkv+Dro = 576]`，单 head latent |
| Decompress 视角 (仅 dense fallback) | `num_attention_heads` | 32 | 当 `MHA_ONE_SHOT` 触发时，`kv_b_proj` 解 latent 得 `[*, 32, Dnope+Dv = 256]` 全头；此时也是 MHA 32 头 KV | 不直接落这条 cache，借 absorb path 的 KV 再 decompress |
| Indexer 头 (I) | `index_n_heads` | 8 | indexer Q `[*, 8, 128]`；indexer K `[*, 128]`（单 head 共享） | `[num_pages, page_size, 1, Di=128]`，uint8 packed FP8+scale |

**关键澄清**（V2_c 易混点）：
- "MLA absorb fast path 下 KV cache 只有 **1 个 latent 头，dim 576**"——这是 cache 的真实形态。
- "32 个 KV 头" 只在 `MHA_ONE_SHOT` dense fallback 这种**少见路径**里出现，不是 DSA decode/prefill 的常态。
- `num_key_value_heads=8` 与 `index_n_heads=8` **数值相同是巧合**，不能套用任何映射关系。

### 2.4 GLM5-Next 特有杂项

| 字段 | 值 | 说明 |
|---|---:|---|
| `num_hidden_layers` | 27 | 总层数 |
| `num_nextn_predict_layers` | 1 | NextN（投机解码）层数；切片不覆盖 |
| `mla` / `mla_nope` | true / true | 启用 MLA + nope-only 风格 |
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
| `kv_b_proj.weight` | `[8192, 512]` | kv_lora_norm -> `Nh*(Dnope+Dv) = 32*(128+128)`；**权重加载后离线拆成吸收矩阵 `w_kc [Nh, Rkv, Dnope]` 与 `w_vc [Nh, Rkv, Dv]`**（见 §4） |
| `o_proj.weight` | `[2048, 4096]` | `Nh*Dv = 32*128` -> hidden |
| `indexer.wq_b.weight` | `[1024, 768]` | q_lora_norm -> `I*Di = 8*128` |
| `indexer.wk.weight` | `[128, 2048]` | hidden -> index key |
| `indexer.weights_proj.weight` | `[8, 2048]` | per-token index head gate（`f32` 计算） |
| `indexer.k_norm.weight` | `[128]` | indexer K 上的 LayerNorm |

## 4. 吸收矩阵（V1 缺、V3 强调）

`kv_b_proj.weight` shape `[Nh*(Dnope+Dv), Rkv] = [8192, 512]`，在模型权重加载流程中
**离线**拆为：

- `w_kc [Nh, Rkv, Dnope] = [32, 512, 128]`（K 吸收矩阵）
- `w_vc [Nh, Rkv, Dv] = [32, 512, 128]`（V 吸收矩阵）

参考 `SG-W0`：`deepseek_common/deepseek_weight_loader.py:565-610`，
`w_kc, w_vc = w.unflatten(0, (-1, Dnope+Dv)).split([Dnope, Dv], dim=1)`。

DSA full-attn 子层运行时**不调** `kv_b_proj` 做 Q/K decompress（除非 `MHA_ONE_SHOT`
dense fallback）。流程如下，**decode 与 prefill 共用**：

1. **Q 侧 K 吸收**：`q_nope [T, Nh, Dnope] → bmm(q_nope.T, w_kc) → q_nope_out [T, Nh, Rkv]`
   含义：把 `q_nope` 投到 `kv_b_proj^K` 列空间，attention 直接在潜空间 K 上点积。
2. **Sparse MQA 在潜空间**：
   `q = concat(q_nope_out, q_pe) [T, Nh, Rkv+Dro=576]`
   attends `kv_cache[topk_slots] [Ktop, 1, Rkv+Dro=576]`（潜空间 1 个 KV 头，`Rkv` 维 latent 同时充当 K 与 V）
   → `attn_out_latent [T, Nh, Rkv=512]`
3. **V 侧解吸收**：`attn_out_latent [T, Nh, Rkv] → bmm(attn_out_latent.T, w_vc) → attn_out [T, Nh, Dv=128]`
4. `o_proj [T, Nh*Dv=4096] → out [T, H=2048]`

参考实现（同 V2_a）：
- `SG-F0`：`models/deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare / forward_absorb_core`。
- `SG-D0`：`models/deepseek_common/attention_backend_handler.py::handle_attention_nsa` —— NSA backend 默认返回 `AttnForwardMethod.MLA`（即 absorb），`use_mha=True` 时才 `MHA_ONE_SHOT`。

## 5. v0.5.10-prerelease 中的 GLM5-Next 脚手架（V3 扩充）

### 5.1 核心脚手架

| 标签 | 入口 | 内容 |
|---|---|---|
| `SG-G0` | `models/glm4_moe.py:1417` | `class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` 空壳，已加入 `EntryClass`。意味着 DSA full-attn 子层期望沿用 `DeepseekV2AttentionMLA + Indexer`，GLM 侧只补 KDA + 调度 |
| `SG-G1` | `configs/model_config.py::is_deepseek_nsa` (l.55-78) + 其他引用 (l.297 / 443 / 1357) | architecture 白名单包含 `"GlmMoeDsaForCausalLM"`；触发条件还需 `index_topk is not None` |
| `SG-G2` | `server_args.py:1541-1567`（注释 `# DeepSeek 3.2/GLM 5`） | NSA prefill dense fallback 阈值默认 = `index_topk`（GLM5 = 2048）；Blackwell 上强制 sparse MLA（关掉 `MHA_ONE_SHOT`） |
| `SG-G3` | `models/deepseek_v2.py:1122` + `:1207-1219` | `use_nsa = is_deepseek_nsa(config)`；`index_topk_freq / index_topk_pattern` 跨层共享 topk 索引（GLM5 若用"S/N 模式"走这套；本切片不覆盖） |

### 5.2 测试与前端线索（V3 新增）

| 标签 | 入口 | 内容 |
|---|---|---|
| `SG-G4-test` | `test/registered/gb300/test_glm5_fp8.py` | HF 模型路径 = `zai-org/GLM-5-FP8`，GB300 (4×B200 NVL4, tp=4) FP8 跑通用例；含 TP4 / TP4+DP4+DPA / TP4+DP4+DPA+MTP 三套 |
| `SG-G4-test` | `test/registered/gb300/test_glm5_nvfp4.py` | 同上 NVFP4 版本 |
| `SG-G4-amd` | `test/registered/amd/accuracy/mi3{0,5}x/test_glm5_eval_*.py` | AMD MI30x / MI35x 精度评估 |
| `SG-G5-fc` | `function_call/glm5_moe_detector.py::Glm5MoeDetector / Glm5MoeStreamDetector` + `function_call_parser.py:56 "glm5"` | 工具调用解析器已注册 |
| `SG-G5-reasoning` | `parser/reasoning_parser.py::GLM5Detector` + `:519 "glm5"` | reasoning parser 已注册 |
| `SG-G6-launcher` | `entrypoints/openai/serving_chat.py:141 "glm5": "glm47"` | OpenAI 服务侧的 tool-call parser map |

### 5.3 取舍点（V3 强调）

本地 `config_16b.json` 的 `architectures = ["Glm5NextForCausalLM"]` **不在** `is_deepseek_nsa()`
白名单内。但 `zai-org/GLM-5-FP8` 跑得通——所以它的 HF config 一定写着
`"GlmMoeDsaForCausalLM"`。**`Glm5NextForCausalLM` 是 dev-time / 上游 transformers
命名，`GlmMoeDsaForCausalLM` 是 SGLang 跑通的命名**——两者要在以下三选一打通：

1. （推荐）改 `config_16b.json` 的 architecture 为 `GlmMoeDsaForCausalLM`，dev 端口直接走现有脚手架。
2. 在 `is_deepseek_nsa()` 白名单加 `"Glm5NextForCausalLM"`，并在 `EntryClass` 加 alias。
3. 单独写一个 `Glm5NextForCausalLM(GlmMoeDsaForCausalLM)` 空壳并注册。

## 6. 参考入口（V1 全部沿用 + V2_a 新增）

V1 中 `SG-I0..I3 / SG-A0..A2 / SG-C0 / SG-E0 / SG-T0 / SG-H0 / DS-F0 / DS-T0 / TF-DSA`
在 v0.5.10-prerelease 同路径仍有效（class 名 `DeepseekV3AttentionMLAIndexer` 改为
`Indexer`）。**只需把 base path 从 `/datau38020T/.../ref/sglang` 切到
`/root/project/sglang-v0.5.10-prerelease`。**

V3 沿用 V2_a 的新增标签：

| 标签 | 入口 | 用途 |
|---|---|---|
| `SG-F0` | `models/deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare/core` | DSA decode + prefill 共用的 absorb 流；`bmm(q_nope, w_kc)`、`attn_mqa(..., topk_indices=...)`、`bmm(out_latent, w_vc)` |
| `SG-M0` | `models/deepseek_v2.py::DeepseekV2AttentionMLA`（`fused_qkv_a_proj_with_mqa`、`dispatch_attn_forward_method`、`forward_prepare/core`） | hidden-in fused projection + attention dispatch |
| `SG-D0` | `models/deepseek_common/attention_backend_handler.py::handle_attention_nsa` | NSA backend 强制走 absorb |
| `SG-W0` | `models/deepseek_common/deepseek_weight_loader.py:565-610` | `kv_b_proj` 拆 `w_kc`/`w_vc` |
| `SG-G0..G6` | 见 §5 | GLM5-Next 脚手架全集 |
| `SG-N0` | `hardware_backend/npu/attention/mla_preprocess.py::NPUFusedMLAPreprocess.forward_mlapo` | NPU `torch.ops.npu.mla_preprocess` 单算子（D0+D1+D2 部分） |
| `SG-N1` | `sgl_kernel_npu.norm.fused_split_qk_norm::fused_split_qk_norm` | split + q_a_layernorm + kv_a_layernorm |
| `SG-J0` | `jit_kernel/fused_qknorm_rope.py` + `csrc/elementwise/fused_qknorm_rope.cuh` | JIT CUDA fused QK RMSNorm + RoPE（支持 partial RoPE / YaRN） |
| `SG-J1` | `jit_kernel/concat_mla.py` + `csrc/elementwise/concat_mla.cuh` | `concat_mla_absorb_q / concat_mla_k` JIT 包装 |
| `SG-T1` | `srt/layers/attention/nsa/transform_index.py::transform_index_page_table_{decode,prefill}_fast` | Triton 版 `fast_topk_transform_*` |
| `SG-C1` | `srt/mem_cache/memory_pool.py::NSATokenToKVPool` (l.1846-2063) | 主 KV cache + `index_k_with_scale_buffer` layout |
| `SG-Q0` | `srt/layers/attention/nsa/quant_k_cache.py / dequant_k_cache.py` | FP8 K cache 量化/反量化 |
| `SG-H1` | `srt/layers/attention/nsa/nsa_indexer.py::rotate_activation` (l.135) + `jit_kernel/hadamard.hadamard_transform` | **`fp8_mqa_logits` 之前对 q/k 做 Hadamard rotation；V1 漏写，Zeus 对齐 REF 必加** |

### 6.5 开发优先级建议（V3 新增）

按"先正确性、再吸收路径、再 sparse、再 fallback"的顺序，避免一上来就堆 fused kernel。

| 优先级 | 任务 | 通过判据 |
|---|---|---|
| P0 | **修 tensor contract**：把 dev 脚本 D4/P4 输出从 `[*,Nh,Dv]` 改成 `[*,Nh,Rkv]`，并在 D5/P5 加 `bmm w_vc`；REF 端先跑通 | `decode_full_path --mode ref` 与 `prefill_full_path --mode ref` 端到端数值与 transformers 参考一致 |
| P0 | **架构名映射**：选 §5.3 三选一，让 `config_16b.json` 能进 `is_deepseek_nsa()` 白名单 | `python -c "from sglang.srt.configs.model_config import is_deepseek_nsa; ..."` 返回 True |
| P1 | Decode D1 含 absorb 的 Q 通路（`q_b_proj` + split + RoPE + `bmm w_kc`） | `decode_q_proj_fused --mode zeus` 输出 `q_nope_out [B,Nh,Rkv]` + `q_pe [B,Nh,Dro]` 对齐 REF |
| P1 | Decode D5/P5 V absorb + D6/P6 o_proj | `*_v_absorb --mode zeus`、`*_o_proj --mode zeus` 对齐 REF |
| P2 | D0/P0 hidden-in fused projection（`fused_qkv_a` + norm） | `*_qkv_a_proj_norm_fused --mode zeus` 输出 `q_lora_norm + kv_lora_norm + k_pe` 对齐 REF |
| P2 | D2/P2 indexer 通路（含 Hadamard、k_norm、FP8 quant、index cache store） | `*_indexer_prep_store_fused --mode zeus` 写入/读回一致 |
| P2 | **RoPE 分两组测**：main MLA 走 interleave、indexer 走 NeoX；两路独立 unit test | `*_rope --mode zeus` 两路分别对齐 REF |
| P3 | D3 decode topk / P3 prefill ragged topk | `decode_indexer_topk_fused --mode zeus` 支持 `-1` 填充；`prefill_ragged_indexer_topk_fused --mode zeus` 多请求不串行、不看未来、不跨 request |
| P3 | D4 decode sparse MQA / P4 prefill sparse MQA | `*_sparse_mqa_fused --mode zeus` 输出 `[*,Nh,Rkv]` 对齐 REF |
| P3 | P4-alt dense fallback policy | `dense_fallback_policy --mode zeus` 在 `max_kv_len <= 2048` 与 MHA_ONE_SHOT REF 对齐 |
| P4 | 端到端 + 真实 shape smoke | DSA layer `[3,7,11,15,19,23]` 单层 `[T,H] -> [T,H]` |
| P5 | FP8 / IndexCache、跨层 topk 共享 (`index_topk_pattern`) | `SG-Q0` + `SG-C1` 落地 |

## 7. Decode Path

Decode 的 `N=B`，每个 request 当前只有 1 个 query token；历史 K/V 与 index K 来自 paged cache。
**默认走 MLA absorb 路径**（`SG-D0`+`SG-F0`），sparse MQA attention 在潜空间完成，
然后用 `w_vc` 解吸收回 `[B, Nh, Dv]`，最后 `o_proj`。

### 7.1 Decode 计算流（V3）

```
hidden_t [B,H]
  │
  ├─ D0 dsa_qkv_a_proj_norm_fused
  │    hidden_t -> q_lora_norm [B,Rq], kv_lora_norm_t [B,Rkv], k_pe_t [B,Dro]
  │
  ├─ D1 dsa_q_b_proj_split_rope_absorb
  │    q_lora_norm -> Q [B,Nh,Dqk] -> split (q_nope, q_pe)
  │    RoPE on (q_pe, k_pe_t)（main MLA：interleave 缺省）
  │    q_nope_out = bmm(q_nope, w_kc) [B,Nh,Rkv]
  │    side effect: main KV cache[new_slots] = concat(kv_lora_norm_t, k_pe_t) [1,Rkv+Dro]
  │
  ├─ D2 dsa_indexer_prep_store_fused
  │    hidden_t + q_lora_norm -> q_idx [B,I,Di] (Indexer RoPE：NeoX 缺省，前 Dro 维)
  │    -> k_idx_t [B,Di] (k_norm + Indexer RoPE 前 Dro 维)
  │    -> Hadamard rotate on (q_idx, k_idx_t)         ★ SG-H1
  │    -> FP8 quant: q_idx_fp8, k_idx_fp8 + scale
  │    -> gate [B,I]
  │    side effect: index K cache[new_slots] = (k_idx_fp8, scale_t)
  │
  ├─ D3 dsa_decode_indexer_topk_fused
  │    fp8_paged_mqa_logits(q_idx_fp8, index_k_cache, gate, seqlens, page_table)
  │    -> logits [B, max_kv_len]
  │    -> fast_topk_transform_fused(logits, page_table_1) -> topk_slots [B, 2048]
  │
  ├─ D4 dsa_decode_sparse_mqa_fused              ★ latent 空间 sparse MQA
  │    q = concat(q_nope_out, q_pe) [B,Nh,Rkv+Dro=576]
  │    flash_mla_with_kvcache(q, kv_cache, topk_slots, num_kv_heads=1, d_v=Rkv)
  │    -> attn_out_latent [B,Nh,Rkv=512]
  │
  ├─ D5 dsa_v_absorb
  │    attn_out = bmm(attn_out_latent, w_vc) -> [B,Nh,Dv=128]
  │
  └─ D6 o_proj
       attn_out [B, Nh*Dv=4096] -> out [B, H=2048]
```

### 7.2 Decode 算子依赖表（同 V2_a §7.2）

| # | 子步骤 | shape / IO | 参考实现 | Zeus 状态 |
|---|---|---|---|---|
| D0 | `dsa_qkv_a_proj_norm_fused` | `hidden [B,2048] -> q_lora_norm [B,768] + kv_lora_norm [B,512] + k_pe [B,64]` | `SG-M0` `fused_qkv_a_proj_with_mqa` + q_a_layernorm + kv_a_layernorm；NPU 端 `SG-N0/N1` 合到 `mla_preprocess` | × |
| D1 | `dsa_q_b_proj_split_rope_absorb` | `q_lora_norm [B,768] -> q_nope_out [B,32,512]; q_pe [B,32,64]; k_pe [B,1,64]`；写 main KV cache | `SG-F0::forward_absorb_prepare`；cache store `SG-C0::set_mla_kv_buffer_triton` | × |
| D2 | `dsa_indexer_prep_store_fused` | `hidden + q_lora_norm -> q_idx [B,8,128] + gate [B,8]`；写 `index K cache[new_slots]` | `SG-I0::Indexer.forward_indexer` + `SG-H1::rotate_activation` + `SG-I3::act_quant`；cache store `SG-I1::fused_store_index_k_cache` / `SG-I2::SetKAndS` | × |
| D3 | `dsa_decode_indexer_topk_fused` | `q_idx_fp8 + index_k_cache + gate -> topk_slots [B,2048]` | `SG-I0::_get_topk_paged` + `deep_gemm.fp8_paged_mqa_logits` + `SG-T0::fast_topk_transform_fused`（CUDA `topk_transform_decode_kernel`）。Triton 版 `SG-T1` | × |
| D4 | `dsa_decode_sparse_mqa_fused` | `q [B,32,576] + main KV cache[topk_slots] -> attn_out_latent [B,32,512]` | `SG-A0::_forward_flashmla_kv` -> `SG-A1::flash_mla_with_kvcache`；**`num_kv_heads=1, d_v=Rkv=512`** | × |
| D5 | `dsa_v_absorb` | `attn_out_latent [B,32,512] -> attn_out [B,32,128]` | `SG-F0::forward_absorb_core` 末段 `bmm w_vc` | × |
| D6 | `o_proj` | `[B, 4096] -> [B, 2048]` | RowParallelLinear | × |

### 7.3 Decode 与 dev 脚本对应

| dev stage | 当前覆盖 | 对应 decode 子步骤 | Zeus 状态 |
|---|---|---|---|
| `decode_qkv_a_proj_norm_fused`（需新增） | hidden -> q_lora_norm + kv_lora_norm + k_pe | D0 | × |
| `decode_q_proj_fused` | V1 现行只覆盖 `q_lora_norm + Q`（含 split/RoPE），**需扩展加入 `bmm(q_nope, w_kc)`**，原来的 ✓ 实际是 △ | D1 | △ |
| `decode_kv_cache_store` | KV latent + k_pe 写 cache（Zeus kernel `dsa_kv_proj_cache_store_fused` V3 落地，默认 zeus 已 PASS） | D1 side-effect | ✓ |
| `decode_indexer_prep_store_fused` | indexer Q/K/gate + **Hadamard** + FP8 quant + index K cache store | D2 | × |
| `decode_indexer_topk_fused` | paged MQA logits + topk transform | D3 | × |
| `decode_sparse_mqa_fused` | latent 空间 sparse MQA；输出 `[B,Nh,Rkv]` | D4 | × |
| `decode_v_absorb` | `bmm(out_latent, w_vc)`；输出 `[B,Nh,Dv]` | D5 | × |
| `decode_o_proj` | 输出 projection | D6 | × |
| `decode_full_path` | D0-D6 端到端 | D0-D6 | × |

## 8. Prefill / Extend Path

Prefill 的 `N=T=sum(extend_seq_lens)`；每个 query row 的可见 key 范围 = `prefix(req) + current_chunk(req, <= row_pos)`。

**默认走 absorb 路径**（与 decode 共用 `forward_absorb_prepare/core`），sparse 调用换成
`flash_mla_sparse_fwd`。只有当 `max_kv_len <= SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`
（默认 = `index_topk` = 2048，见 `SG-G2`）才切到 `MHA_ONE_SHOT`（dense，不吸收）。

### 8.1 Prefill 计算流（V3）

```
hidden [T,H], positions [T], out_cache_loc [T], ragged metadata
  │
  ├─ P0 dsa_qkv_a_proj_norm_fused          （同 D0 形态，T batched）
  ├─ P1 dsa_q_b_proj_split_rope_absorb     （同 D1；KV cache 批量写 out_cache_loc）
  ├─ P2 dsa_indexer_prep_store_fused       （同 D2，T batched；index K 批量写 out_cache_loc）
  │
  ├─ P3 dsa_prefill_ragged_indexer_topk_fused
  │    fp8_mqa_logits ragged + GetKAndS(index_k_cache + page indices)
  │    -> logits [T, max_kv_len]，可见区由 ragged metadata（prefix + causal 当前 chunk）限制
  │    -> fast_topk_transform_ragged_fused -> topk_slots [T, 2048]
  │
  ├─ P4 dsa_prefill_sparse_mqa_fused       ★ absorb 后 sparse MQA（默认）
  │    q = concat(q_nope_out, q_pe) [T,Nh,576]
  │    flash_mla_sparse_fwd(q, kv_cache, indices=topk_slots, d_v=Rkv)
  │    -> attn_out_latent [T,Nh,512]
  │
  ├─ P4-alt dense_fallback (短序列, max_kv_len <= 2048)
  │    MHA_ONE_SHOT：kv_b_proj 解 latent -> [T,Nh,Dnope+Dv]
  │    FlashAttention varlen on full prefix+current
  │    -> attn_out_dense [T,Nh,Dv]   ★ 跳过吸收，直接进 P6
  │
  ├─ P5 dsa_v_absorb                       仅 absorb path：bmm(out_latent, w_vc) -> [T,Nh,128]
  └─ P6 o_proj                             [T, Nh*Dv] -> [T, H]
```

### 8.2 Prefill 算子依赖表（同 V2_a §8.2）

| # | 子步骤 | shape / IO | 参考实现 | Zeus 状态 |
|---|---|---|---|---|
| P0 | `dsa_qkv_a_proj_norm_fused` | `hidden [T,2048] -> q_lora_norm + kv_lora_norm + k_pe` | 同 D0 | × |
| P1 | `dsa_q_b_proj_split_rope_absorb` | `q_lora_norm [T,768] -> q_nope_out [T,32,512] + q_pe [T,32,64] + k_pe [T,1,64]`；批量写 KV cache | `SG-F0::forward_absorb_prepare` + `SG-C0::set_mla_kv_buffer_triton` | × |
| P2 | `dsa_indexer_prep_store_fused` | 同 D2，T batched | 同 D2 | × |
| P3 | `dsa_prefill_ragged_indexer_topk_fused` | `q_idx_fp8 + index_k_cache + ragged metadata -> topk_slots [T,2048]` | `SG-I0::_get_topk_ragged`（`GetKAndS` gather + `deep_gemm.fp8_mqa_logits`）+ `SG-T0::fast_topk_transform_ragged_fused`（CUDA `topk_transform_prefill_ragged_kernel`） | × |
| P4 | `dsa_prefill_sparse_mqa_fused` | `q [T,32,576] + kv_cache + topk_slots -> attn_out_latent [T,32,512]` | `SG-A0::_forward_flashmla_sparse` -> `SG-A2::flash_mla_sparse_fwd`；TileLang fallback `DS-T0`/`SG-A0::_forward_tilelang` | × |
| P4-alt | `dense_fallback_policy` | `valid_len(row) <= 2048` 切 dense MHA | `SG-A0::set_nsa_prefill_impl` + `_forward_standard_mha` | × |
| P5 | `dsa_v_absorb` | `[T,32,512] -> [T,32,128]` | `SG-F0::forward_absorb_core` 末段 `bmm w_vc` | × |
| P6 | `o_proj` | 同 D6 | RowParallelLinear | × |

### 8.3 Prefill 与 dev 脚本对应

| dev stage | 当前覆盖 | 对应 prefill 子步骤 | Zeus 状态 |
|---|---|---|---|
| `prefill_qkv_a_proj_norm_fused`（需新增） | D0/P0 形态 batched 版 | P0 | × |
| `prefill_q_proj_fused` | V1 现行只覆盖 `q_lora_norm + Q`，**需扩展加入 `bmm(q_nope, w_kc)`** | P1 | △ |
| `prefill_kv_cache_store` | KV latent + k_pe 批量写 cache（Zeus kernel `dsa_kv_proj_cache_store_fused` V3 落地，默认 zeus 已 PASS） | P1 side-effect | ✓ |
| `prefill_indexer_prep_store_fused` | indexer Q/K/gate + **Hadamard** + FP8 + index K cache store | P2 | × |
| `prefill_ragged_indexer_topk_fused` | ragged causal top-k | P3 | × |
| `prefill_sparse_mqa_fused` | latent 空间 sparse MQA；输出 `[T,Nh,Rkv]` | P4 | × |
| `dense_fallback_policy` | `max_kv_len <= 2048` 时 dense MHA_ONE_SHOT；不吸收 | P4-alt | × |
| `prefill_v_absorb` | `bmm(out_latent, w_vc)` | P5 | × |
| `prefill_o_proj` | output projection | P6 | × |
| `prefill_full_path` | P0-P6 端到端 | P0-P6 | × |

## 9. Decode / Prefill Slides 对齐

旧 slides（`glm_moe_dsa_{decode,prefill}_slides.html`）按 V1 粒度画的：sparse attention 画在
`[Nh, Dv]` 维度、缺 V absorb、缺 Hadamard。

V3 不改旧 slides，单独出 `glm_moe_dsa_slides_v3.html`（一个文件覆盖 decode + prefill +
关键澄清）。映射如下：

| slides V3 章节 | Decode 子步骤 | Prefill 子步骤 |
|---|---|---|
| Q proj + Absorb-K | D0 + D1 (含 `bmm w_kc`) | P0 + P1 |
| KV latent + cache store | D0 (KV part) + D1 cache store | P0 + P1 cache store |
| Indexer prep (含 Hadamard) | D2 | P2 |
| TopK | D3 (paged) | P3 (ragged) |
| Sparse MQA in latent | D4 (`flash_mla_with_kvcache`, `[*,Nh,Rkv]`) | P4 (`flash_mla_sparse_fwd`, `[*,Nh,Rkv]`) |
| Dense fallback (仅 prefill) | — | P4-alt (`MHA_ONE_SHOT`, `[T,Nh,Dv]`) |
| V absorb | D5 | P5 |
| o_proj | D6 | P6 |

## 10. 里程碑（V3）

| Milestone | 目标 | 通过标准 |
|---|---|---|
| M0 | 文档 + pure-torch REF 脚本 | `dev_glm_moe_dsa_{decode,prefill}_test.py --mode ref` PASS；REF 已显式生成中间张量 `q_nope_out`、`q_pe`、`k_nope/k_pe`、`attn_out_latent` |
| M0.5 | 架构名映射（§5.3 三选一） | `is_deepseek_nsa(config_16b.json)` 返回 True；模型能被 SGLang `EntryClass` 识别 |
| M1 | D0/P0 hidden-in fused projection | `*_qkv_a_proj_norm_fused --mode zeus` 对齐 REF |
| M2 | D1/P1 Q 通路含 absorb | `*_q_proj_fused --mode zeus` 输出 `q_nope_out [*,Nh,Rkv]` 与 `q_pe`，对齐 REF；KV cache store side-effect 一致 |
| M3 | D2/P2 indexer 通路（含 Hadamard、k_norm、FP8、index cache store） | `*_indexer_prep_store_fused --mode zeus` |
| M4 | D3 decode top-k | `decode_indexer_topk_fused --mode zeus` 支持 padding `-1` |
| M5 | D4 decode sparse MQA | `decode_sparse_mqa_fused --mode zeus` 输出 `[B,Nh,Rkv]` |
| M6 | D5/P5 V absorb + D6/P6 o_proj | Zeus 不再 SKIP |
| M7 | P3 ragged top-k | `prefill_ragged_indexer_topk_fused --mode zeus` |
| M8 | P4 sparse MQA + P4-alt dense fallback | 两条路径都对齐 REF |
| M9 | GLM5-Next real shape smoke | DSA layer `[3,7,11,15,19,23]` 单层 `[T,H] -> [T,H]` |
| M10 | FP8 / IndexCache + 跨层 topk 共享 | `SG-Q0` + `SG-C1` 落地；评估 `index_topk_pattern` |

## 11. V1 → V3 差异速查

| 项 | V1 | V3 |
|---|---|---|
| canonical config | `hub/16b_hf/config.json` | 本目录 `config_16b.json`（`Glm5NextForCausalLM`） |
| "head 数"表达 | 仅 Nh=32 | 四件套：Nh=32（Q头）/ `num_key_value_heads`=8（MLA 不用）/ 潜空间 KV=1（absorb）/ 32（decompress，仅 fallback）/ I=8（indexer） |
| RoPE 字段 | 未列 | 列出 `rope_theta=10000`、`rope_scaling=null`、`partial_rotary_factor=0.5`、`use_qk_norm=true`；且**额外标出 main MLA vs Indexer 的 `is_neox_style` 默认值相反** |
| Decode/Prefill attention | `Q [*,Nh,Dv]` 标准 sparse MHA | 潜空间 sparse MQA：`Q [*,Nh,Rkv+Dro] → out_latent [*,Nh,Rkv]`，再 V absorb |
| 吸收矩阵 `w_kc`/`w_vc` | 未提 | §4 + `SG-W0`/`SG-F0` |
| Hadamard rotation | 未提 | `SG-H1` 显式列入 D2/P2 |
| GLM5-Next 脚手架 | 未提 | §5 含 `SG-G0..G6`（类、白名单、prefill 阈值、tests、function/reasoning parser） |
| 架构名映射 | 未提 | §5.3 列出 `Glm5NextForCausalLM` 与 `GlmMoeDsaForCausalLM` 的三选一打通方案 |
| Dense fallback 默认阈值 | 通用描述 | GLM5 默认 = `index_topk=2048`（`SG-G2`） |
| dev stage 粒度 | 6 步 (D0..D5 / P0..P5) | 7 步 (D0..D6 / P0..P6)；新增 V absorb 与 hidden-in fused projection 拆分 |
| ref base path | `/datau38020T/.../ref/sglang` | `/root/project/sglang-v0.5.10-prerelease` |
| 开发优先级 | 无 | §6.5 P0..P5 清单（先正确性、再吸收路径、再 sparse、再 fallback） |
