# GLM-MoE-DSA / GLM5-Next：V2_a / V2_b / V2_c 三份评估对比分析

> 输入：同一组提问的三份独立回答
> `GlmMoeDsa_dev_V2_a.md` / `GlmMoeDsa_dev_V2_b.md` / `GlmMoeDsa_dev_v2_c.md`，
> 均基于 `config_16b.json` 与 `/root/project/sglang-v0.5.10-prerelease`。
> 本文核对三者一致 / 分歧之处，并对分歧给出谁对谁错的判定（判定均回源码验证）。

## 0. 一句话结论

| 维度 | V2_a | V2_b | V2_c |
|---|---|---|---|
| 整体准确度 | **最高** | 高（RoPE interleave 唯一答对） | 最低（2 处实质性错误 + 1 处臆测） |
| 整体完整度 | 最高（7 段重切 + 新里程碑 + Hadamard） | 中（结论摘要 + head 语义表清晰） | 最低（沿用 V1 6 段） |
| 高层方向 | 对 | 对 | 大体对（"decode=吸收是灵魂" 的直觉成立） |

排序：**V2_a > V2_b > V2_c**。V2_c 不是全错——它的核心澄清方向（吸收矩阵、head 混淆、RoPE 存在）都对，但它漏看了 prerelease 里实际存在的 GLM5 脚手架，并把 vanilla MLA 的 prefill 行为误套到 DSA/NSA 路径上。

---

## 1. 三者一致、且经核对正确的点（可直接采信）

1. **`config_16b.json` 有 RoPE，只是没有 dynamic scaling。**
   三者都指出 `rope_theta=10000`、`qk_rope_head_dim=64`、`partial_rotary_factor=0.5`、`max_position_embeddings=202752`、`rope_scaling=null`。正确说法是"有 RoPE 维度/base，但无 YaRN/线性 scaling"，不能说"没有 RoPE 部分"。✓

2. **Decode path 必须走吸收矩阵 `w_kc` / `w_vc`。**
   三者都把 decode 改写成：`q_nope @ w_kc → q_nope_out`，在 latent 空间做 sparse attention，输出再 `@ w_vc` 还原。✓（与 SGLang `forward_absorb_prepare/core` 一致）

3. **V1 文档把 DSA attention 写成 `Knew[*,32,192] / Vnew[*,32,128]` + 标准 sparse MHA 是误导性的。**
   三者都纠了这一点（V2_c 只纠了 decode 半边，见 §2-B）。真实 cache 是 latent MLA 形态。✓

4. **GLM5-Next 没有另起炉灶的 "GLM-DSA" 内核，直接复用 DeepSeek 的 NSA/MLA 机制。**
   三者一致。✓

5. **`num_attention_heads=32` 是 Q 头；V1 把它和 K/V 头混为一谈是错的。**
   三者都点了 V1 的混淆。✓（但对"`num_key_value_heads=8` 到底是什么"三者结论不同，见 §2-C）

---

## 2. 三者分歧点 + 谁对谁错（已回源码判定）

### A. prerelease 里 GLM5-Next 的"具体线索"

| | 结论 |
|---|---|
| **V2_a** | 找到 `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)`（`glm4_moe.py:1417`，已进 `EntryClass`）；`is_deepseek_nsa` 白名单含 `GlmMoeDsaForCausalLM`（还需 `index_topk` 非空）；`server_args` 对它启用 `attention_backend="nsa"`、CUDA 强制 page_size=64、Blackwell 强制 sparse MLA。**并明确指出 `config_16b.json` 的 `Glm5NextForCausalLM` 不在白名单内——这是个取舍点**（改 config / 改白名单 / 加 entry class）。 |
| **V2_b** | 同 V2_a 的发现，**外加**：`test/registered/gb300/test_glm5_fp8.py`、`test_glm5_nvfp4.py`、AMD `test_glm5_eval_*`、parser/前端的 `glm5`/`glm5stream`。结论：把 `GlmMoeDsaForCausalLM` 视为 GLM5 DSA 的实现入口，但 prerelease 没直接支持 `Glm5NextForCausalLM` 这个类名。 |
| **V2_c** | "prerelease `srt/models/` 暂未包含直接命名为 `glm5`、`glm_moe_dsa` 的模型实现，最近似的是 `glm4_moe.py` 和 `glm4_moe_nextn.py`。" 未提 `GlmMoeDsaForCausalLM` 类，未提任何 `test_glm5_*`。 |

**判定：V2_a / V2_b 对，V2_c 漏。** 实测：

```
python/sglang/srt/models/glm4_moe.py:1417  class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)
python/sglang/srt/models/glm4_moe.py:1422  EntryClass = [Glm4MoeForCausalLM, GlmMoeDsaForCausalLM]
python/sglang/srt/configs/model_config.py:75 / 297 / 443 / 1357  "GlmMoeDsaForCausalLM"
test/registered/gb300/test_glm5_fp8.py, test_glm5_nvfp4.py
test/registered/amd/accuracy/mi3{0,5}x/test_glm5_eval_*.py
```

V2_c 只按"文件名"搜，没搜类名和测试目录，所以漏掉了实际存在的脚手架。V2_b 在这一项上线索覆盖最全。

### B. Prefill path 是否使用吸收矩阵？

| | 结论 |
|---|---|
| **V2_a** | **decode 和 prefill 都默认走 absorb**；prefill 只有当 `max_kv_len <= 阈值`（GLM5 默认 = `index_topk` = 2048）才切 `MHA_ONE_SHOT`（dense，不吸收）。 |
| **V2_b** | prefill 同样走 absorb，差异只在 top-k / sparse attention 的 metadata 是 ragged causal span；dense fallback 由 `set_nsa_prefill_impl` 的 `use_mha` 决定，Blackwell 上强制 sparse MLA。 |
| **V2_c** | "**Prefill 阶段通常不使用吸收矩阵**（序列长时，显式展开 K/V 做 FlashAttention/FlashMLA 更高效）"，并相应地把 P4 输出写成 `attn_out [T,Nh,Dv]`。 |

**判定：V2_a / V2_b 对（针对 DSA/NSA 路径），V2_c 错位。** 实测 `attention_backend_handler.py`：

- `handle_attention_nsa()`（NSA backend，GLM5-Next DSA 走这条）：
  `if backend.use_mha: return MHA_ONE_SHOT; else return AttnForwardMethod.MLA`（即 absorb），decode/prefill 都一样。
- `_handle_attention_backend()`（**非** NSA 的 flashinfer/flashmla/fa3 等）：extend 阶段会走 `MHA_ONE_SHOT` / `MHA_CHUNKED_KV`（解开 K/V）。

V2_c 描述的"prefill 展开 K/V"是**vanilla DeepSeek-V2 MLA 非稀疏路径**的行为——这本身没错，只是套错了对象。GLM5-Next DSA 是 NSA backend，prefill 默认仍是 absorb，sparse prefill kernel（`flash_mla_sparse_fwd`）也在 latent 空间工作。所以 V2_c 的 P4 张量契约 `[T,Nh,Dv]` 是错的，应为 `[T,Nh,Rkv=512]` 再 `w_vc` 还原（V2_a/V2_b 的写法）。

### C. `num_key_value_heads = 8` 到底是什么？

| | 结论 |
|---|---|
| **V2_a** | MLA/DSA 路径**不使用**它；它只在普通 GQA 的 `Glm4MoeAttention`（非 MLA）下当 KV 头数，`mla: true` 时直接忽略。真实落 cache 的是"潜空间 KV 头 = 1"（`attn_mqa` `num_kv_heads=1`、`v_head_dim=kv_lora_rank=512`）。 |
| **V2_b** | "config 中的 dense/GQA KV heads = 8，SGLang MLA absorb path 不直接用它建 cache"；MLA latent KV heads = 1，latent dim = 512，最终 per-head V dim = 128（由 `w_vc` 投回）。 |
| **V2_c** | "`num_key_value_heads:8` **极有可能是历史遗留字段，或用来映射 Indexer 的 head 数**（config 里也有 `index_n_heads:8`）；主 attention 以 32 Q heads + **32 个隐式 KV heads (MLA)** 运行。" |

**判定：V2_a / V2_b 对；V2_c 部分对、部分臆测。**
- "MLA 路径不用 `num_key_value_heads`" —— 对（SGLang DeepseekV2 MLA 用 `kv_lora_rank` 重建全部 `num_attention_heads`，`num_key_value_heads` 仅 GQA 模块读）。
- "8 可能映射到 indexer" —— **无依据的猜测**。`index_n_heads=8` 是独立字段；两者都等于 8 是巧合。`num_key_value_heads=8` 就是个 vestigial GQA 字段。
- "32 个隐式 KV heads" —— 这是 **decompress / MHA 视角**（`kv_b_proj [8192,512] = 512→32×(128+128)` 解开后确实是 32 头）。但运行时 absorb fast path 落 cache 的是 `h_kv=1` 的 latent MQA（`[tokens, 1, Rkv+Dro=576]`）。V2_c 没区分这两个视角，容易再造成"K/V head 数"的新混淆——而这恰恰是这轮提问要纠正的点。V2_a/V2_b 的"Q头32 / GQA头8(不用) / 潜空间KV头1 / indexer头8"四件套表达最准确。

### D. RoPE 的 interleave / neox 风格

| | 结论 |
|---|---|
| **V2_a** | 指出 `use_qk_norm=true`（Q/K 在 RoPE 后做 RMSNorm）、`partial_rotary_factor=0.5` ↔ `128×0.5=64=qk_rope_head_dim`、indexer 共用主 RoPE 配置。**没提 neox/interleave 默认值。** |
| **V2_b** | **唯一深入**：main MLA RoPE `is_neox_style = not getattr(config,"rope_interleave",True)` → 缺省 `False`；indexer RoPE `is_neox_style = not getattr(config,"indexer_rope_interleave",False)` → 缺省 `True`。提醒两路 interleave 默认不同，要分别核对。 |
| **V2_c** | 只确认"RoPE 64 维实打实存在，拼接前要单独处理"。最浅。 |

**判定：V2_b 对且最有价值（实测 `deepseek_v2.py:1181` 与 `:1244` 与其引用逐字一致）。** V2_a 的 `use_qk_norm` 与 `partial_rotary_factor` 解释也对、且是 V2_b 没强调的点。两者互补。

### E. Indexer 里的 Hadamard rotation

| | 结论 |
|---|---|
| **V2_a** | **唯一捕捉到**：`fp8_mqa_logits` / `fp8_paged_mqa_logits` 之前对 q_idx / k_idx 要做 `rotate_activation`（Hadamard transform），并明确标注"V1 漏写，Zeus 对齐 REF 必须加这步"。 |
| **V2_b** | 提到 indexer 的 FP8 量化（`act_quant`）与 fp8+scale cache store，**未提 Hadamard**。 |
| **V2_c** | **未提 Hadamard**，indexer 部分最简略。 |

**判定：V2_a 对（实测 `nsa_indexer.py:135 rotate_activation` + `:328/331/338/341/353` 多处调用）。** V2_b / V2_c 这里有遗漏。

### F. dev 脚本 / 里程碑重切粒度

| | 做法 |
|---|---|
| **V2_a** | 把 dev stage 从 V1 的 6 步重切成 7 步（D0..D6 / P0..P6），新增 `*_qkv_a_proj_norm_fused`、`*_v_absorb`；新里程碑 M0–M10；**明确指出 V1 的 "D0 ✓" 其实只覆盖了 `q_lora_norm + Q`，缺 `bmm(q_nope, w_kc)` 那步，应降级为 △**。 |
| **V2_b** | 给出 A0–A13 算子依赖表 + "开发优先级建议" 列表（先修 tensor contract → decode 加 absorb-Q/absorb-output → prefill ragged top-k 单独测 → RoPE 分两组测 → 解决 architecture 名称映射）。 |
| **V2_c** | 基本沿用 V1 的 6 步（D0..D5），`dsa_q_proj_absorb_fused` 仍标 ✓，里程碑没大动。 |

**判定：无对错，V2_a 最细、对后续实现最可执行；V2_b 的优先级清单实用；V2_c 偏保守。** 注意 V2_a 与 V2_c 对 "Q proj 那步是否已完成（✓）" 的态度相反——V2_a 认为 V1 的 ✓ 不完整（因为没含 `w_kc` 吸收），这个判断更严谨。

---

## 3. 可互相借鉴的地方（合并出一份"最佳版本"应包含）

- 取 **V2_a §4**（`w_kc`/`w_vc` 离线拆分公式 + absorb 数学推导）作为吸收矩阵章节主体。
- 取 **V2_b 的 "head 语义表"**（Q头32 / GQA头8不用 / latent KV头1 / latent dim 512 / 最终V dim 128 / indexer头8）作为 head 澄清的标准表述——比 V2_c 的 "32 隐式 KV heads" 更不容易再造混淆。
- 取 **V2_b 的 RoPE interleave 段**（main vs indexer 的 `is_neox_style` 默认不同）补进 V2_a / V2_c——这是三份里只有 V2_b 抓到的实质细节。
- 取 **V2_a 的 Hadamard rotation 步**（`SG-H1::rotate_activation`）补进 V2_b / V2_c 的 indexer prep——D2/P2 漏了它，REF 对齐会差。
- 取 **V2_a 的 GLM5 脚手架取舍点**（`Glm5NextForCausalLM` 不在 `is_deepseek_nsa` 白名单 → 改 config / 改白名单 / 加 entry class 三选一）+ **V2_b 的测试线索清单**（`test_glm5_*`、parser `glm5`）合并成 "prerelease GLM5 现状" 一节。
- 取 **V2_a 的 dev stage 7 步重切 + M0–M10** 作为里程碑；并采纳 "V1 的 Q proj ✓ 实为 △（缺 `w_kc` 吸收）" 这个修正。
- **务必修掉 V2_c 的两处错**：① prerelease 确实有 `GlmMoeDsaForCausalLM`；② DSA/NSA 的 prefill **默认走 absorb**，不是"展开 K/V"——V2_c 那段描述的是 vanilla MLA 非稀疏路径，不适用于本模型。

---

## 4. 错误清单（按严重度）

| 严重度 | 出处 | 错误 | 正确说法 |
|---|---|---|---|
| 高 | V2_c | "prerelease `srt/models/` 没有 glm5 / glm_moe_dsa 实现" | `glm4_moe.py:1417` 有 `GlmMoeDsaForCausalLM`，已进 `EntryClass`；`model_config.py` 多处特判；`test/registered/.../test_glm5_*` 存在 |
| 高 | V2_c | "Prefill 通常不用吸收矩阵，展开 K/V 做 FlashAttention" | NSA backend 的 `handle_attention_nsa` 对 decode/prefill 默认都返回 `AttnForwardMethod.MLA`(absorb)；只有短序列 `use_mha=True` 才 `MHA_ONE_SHOT`。展开 K/V 是 vanilla MLA 路径，不是 DSA 路径 |
| 中 | V2_c | "`num_key_value_heads=8` 可能映射 Indexer head 数" | 无依据；它是未用的 GQA 字段，与 `index_n_heads=8` 仅数值巧合 |
| 中 | V2_c | "主 attention 以 32 个隐式 KV heads 运行"（作为主结论表述） | 那是 decompress/MHA 视角；absorb fast path 落 cache 的是 `h_kv=1` latent MQA（dim 576）。表述上应以 latent 视角为主、decompress 视角为辅 |
| 低 | V2_b / V2_c | 漏了 indexer 的 Hadamard rotation（`rotate_activation`） | `fp8_mqa_logits` 前要对 q_idx/k_idx 做 Hadamard transform（`nsa_indexer.py:135`） |
| 低 | V2_b / V2_c | RoPE 未区分 main vs indexer 的 neox/interleave 默认 | `is_neox_style = not getattr(config,"rope_interleave",True)`(main, 缺省 False) vs `not getattr(config,"indexer_rope_interleave",False)`(indexer, 缺省 True) |
| 提示 | V1（被三者引用纠正） | `Knew[*,32,192]/Vnew[*,32,128]` + 标准 sparse MHA、把 attention head 和 KV head 混为一谈、Q proj 标 ✓ | 见上文 §1.3 / §2-C / §2-F |

---

## 5. 一致性总评

- **方法论层面三者一致**：都认同"复用 DeepSeek NSA/MLA、关注吸收矩阵、纠正 head 混淆、确认 RoPE 存在"。这部分可放心采信。
- **事实细节层面 V2_c 偏弱**：漏看 prerelease 现成脚手架、把 prefill 的 absorb 行为说反、对 `num_key_value_heads` 做了臆测。
- **V2_a 与 V2_b 高度一致且互补**：V2_a 胜在 absorb 数学、Hadamard、dev-stage 重切的工程颗粒度；V2_b 胜在 RoPE interleave 细节、head 语义表的清爽表达、测试线索覆盖。把两者合并 + 修掉 V2_c 的两处错，就是当前能给出的最准版本。
- 建议后续 **以 V2_a 为骨架**，吸收 V2_b 的 head 语义表 / RoPE interleave 段 / 测试线索，作为 `GlmMoeDsa_dev_V2.md` 定稿。
