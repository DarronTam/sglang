# GLM5-Next / GLM5-Next-16B Decode · 16-Device KV 内存放置（v2）

> 16 device 一个 instance、4–8 req / instance、512 K 平均上下文下的 KV cache 放置与并行设计。
> 涵盖 Linear Attention (KDA) 与 DSA（Index K + Sparse MQA）两条 attention 子层；MoE / dense 不在本文范围。
> 本文**自包含**，不依赖 v1_a / v1_b / v1_c。

---

## 1. 模型配置

下文用 **GLM5-Next-16B** 指 `config_16b_v2.json`，**GLM5-Next** 指 `config.json`。

### 1.1 配置对照表

| 分组 | 维度 | **GLM5-Next-16B**<br>(`config_16b_v2.json`) | **GLM5-Next**<br>(`config.json`) |
|---|---|---:|---:|
| 基本 | `hidden_size` (H) | **2048** | **4096** |
| 基本 | `num_hidden_layers` | **27** | **45** |
| 基本 | `torch_dtype` | bfloat16 | bfloat16 |
| MLA | `num_attention_heads` (Nh) | **32** | **64** |
| MLA | `q_lora_rank` (Rq) | 768 | 1536 |
| MLA | **`kv_lora_rank` (Rkv)** | **512** | **512** |
| MLA | `qk_nope_head_dim` (Dnope) | 128 | 192 |
| MLA | **`qk_rope_head_dim` (Dro)** | **0** | **实际为 0**（config 标 64；见 §1.2） |
| MLA | `v_head_dim` (Dv) | 128 | 256 |
| MLA | `mla` / `mla_nope` | true / true | true / —（部署等价 mla_nope） |
| 层分布 | **DSA（`full_attn_layers`）** | **6 层** `[3,7,11,15,19,23]` | **11 层** `[3,7,11,...,39,43]` |
| 层分布 | **KDA（`kda_layers`）** | **21 层** `[0..2, 4..6, ..., 24..26]` | **34 层** `[0..2, 4..6, ..., 40..42, 44]` |
| KDA | `linear_num_value_heads` / `linear_num_key_heads` | 32 / 32 | 64 / 64 |
| KDA | `linear_key_head_dim` / `linear_value_head_dim` (Kd / Vd) | 72 / 72 | 128 / 128 |
| KDA | `linear_conv_kernel_dim` (k_conv) | 4 | 4 |
| Indexer | `index_head_dim` (Di) | 128 | 128 |
| Indexer | `index_n_heads` (I) | 8 | 8 |
| Indexer | `index_topk` (Ktop) | 2048 | 2048 |
| MoE | `moe_intermediate_size` | 1408 | 2048 |
| MoE | `n_routed_experts` / `num_experts_per_tok` | 64 / 6 | 288 / 7 |

### 1.2 关键提示：**两个 config 在部署中 Dro = 0**

GLM5-Next-16B 配置文件即写 `qk_rope_head_dim = 0`；GLM5-Next 虽然 config 写 64，但**实际部署的业务版本去掉了 rope 维度**（按 mla_nope 范式）。因此本文统一按 **Dro = 0** 处理：

- **每个 token 的 latent KV cache 维度 = Rkv = 512 elem**（不含 k_pe）。
- 主 MLA 不做 RoPE（`rotary_emb is None`）；只 indexer 内 NeoX RoPE 前 64 维（与 cache 无关，是 query/key 旋转后再走 FP8 量化）。
- 在 sparse MQA 的 absorb 计算空间，head_dim 也是 **512**（= Rkv，因为 q_pe 维度为 0）。

→ 下文所有 "latent K per token" 都按 512 elem 算。

---

## 2. 硬件

### 2.1 拓扑

| 层级 | 数量 |
|---|---|
| Instance | 16 device |
| Device | 2 core |
| Core | 4 vector core + 1 tensor core（本文假设 1 TC/core，若实际不同按比例缩放） |

### 2.2 两类存储

| | Gmem | Lmem |
|---|---|---|
| 容量 / device | **1.8 GB** | **29,440 MB ≈ 28.75 GB**（= 14,720 MB × 2 core） |
| 私有度 | device 范围共享 | **per-core private**（14.38 GB / core） |
| Host 写效率 | 高，通道独立 | 较低 |
| Host 写 vs TC | 不冲突 | **与 TC 竞争 BW** |
| TC 读距离 / BW | 远 | 近，**3.2 TB/s per core**（= 6.4 TB/s per device） |
| 对齐 | 16 KB | 16 KB |

### 2.3 Tensor Core (TC) 性能

| 数据类型 | MMA 形状 / cycle | 频率 | per-TC 峰值 | per-device 峰值（2 TC） |
|---|---|---|---:|---:|
| FP8 × FP8 | `[1, 128] × [128, 128]` | 400 MHz | **13.1 TFLOPS** | **26.2 TFLOPS** |
| BF16 × BF16 | `[1, 64]  × [64, 128]`  | 400 MHz | **6.55 TFLOPS** | **13.1 TFLOPS** |

**算密度拐点**：FP8 TC 13.1 T / Lmem BW 3.2 T = **4.1 FLOPs/B**——算密度超过此值的 GEMM 计算受限，低于则 BW 受限。Memory 时钟也是 400 MHz，每 cycle 8 KB BW。

### 2.4 Vector Core 性能

每个 core 含 4 个 vector core；每 vc SIMD lane：

| dtype | lanes / vc | lanes / core | lanes / device (×2 core) |
|---|---:|---:|---:|
| BF16 / FP8 (8-bit) | 64 | 256 | **512** |
| FP32 | 32 | 128 | **256** |

vec core 主要承担 top-K 选择、LSE merge、softmax、量化/反量化、地址计算、激活归一化等非 GEMM 工作。

---

## 3. 部署场景

| 假设 | 值 |
|---|---|
| 一个 instance 的 device 数 | **16** |
| 一个 instance 的并发 request 数 | **4 – 8** |
| 每个 request 的平均上下文 | **512 K token** |
| 阶段 | **PD-disaggregation 的 decode worker**（KV 通过 host 写从 prefill 转入） |

下文用 `R` 表示 request 数。常考虑 R=4 / R=8 两个端点。

---

## 4. Linear Attention (KDA) 并行与放置

### 4.1 并行：固定 **TP=16**（head 切）

KDA 是 recurrent 结构，每个 step 输入 / 输出都按 head 独立。直接沿 head 维做 TP=16 是最自然、最便宜的并行方式（无跨 step 通信，只在 prefill→decode 切换时一次 state 迁移）。

每 device 持有的 head 数：

| 模型 | KDA 总 head 数 | head / device (TP=16) | head / core (2 core) |
|---|---:|---:|---:|
| GLM5-Next-16B | 32 | **2** | **1** |
| GLM5-Next | 64 | **4** | **2** |

> 注：不做 CP / DP 给 KDA。CP 对 recurrent state 不友好（state 跟序列位置有依赖），DP 在 R<16 时会浪费 device。

### 4.2 State 形状与存储

每 KDA 层每 request 持两块 state：

- **recurrent state `S`**：`[H_kda, Kd, Vd]`
- **short-conv state**：`3 · H_kda · (k_conv - 1) · Kd`（q/k/v 三份 lane，存 `k_conv - 1 = 3` 个历史 token）

**单 request 全部 KDA 层 state（无 TP，bf16，参考）**：

| 模型 | per layer (elem) | 全部层 (elem) | bf16 字节 |
|---|---:|---:|---:|
| GLM5-Next-16B | 32·72·72 + 3·32·72·3 = **186,624** | 21 · 186,624 ≈ 3.92 M | **7.48 MB** |
| GLM5-Next | 64·128·128 + 3·64·128·3 = **1,122,304** | 34 · 1,122,304 ≈ 38.16 M | **72.8 MB** |

**TP=16 后 per-device 每 request**：

| 模型 | per layer per device (elem) | 全部层 per device (bf16) | 全部层 per device (recurrent fp32 + conv bf16) |
|---|---:|---:|---:|
| GLM5-Next-16B | 11,664 | **478 KB** | ≈ 904 KB |
| GLM5-Next | 70,144 | **4.55 MB** | ≈ 8.80 MB |

**R requests per device 合计**：

| | R=4 | R=8 |
|---|---:|---:|
| GLM5-Next-16B bf16 | 1.87 MB | 3.74 MB |
| GLM5-Next-16B fp32+bf16 | 3.53 MB | 7.06 MB |
| GLM5-Next bf16 | **18.2 MB** | **36.4 MB** |
| GLM5-Next fp32+bf16 | 35.2 MB | 70.4 MB |

### 4.3 放置决策：**Gmem**

每条 state 几百 KB – 几 MB，**条数多**（16 device × 21–34 层 × 4–8 req = 几千个独立 buffer），且 PD-disaggregation 时高频小写。理由：

1. **Gmem 写不与 TC 竞争**——KV-transfer 的高频小写不会扰动 decode TC；
2. **容量绰绰有余**——最大占用（GLM5-Next、R=8、fp32 recurrent）也只是 70 MB / device，相对 Gmem 1.8 GB 占 4%；
3. recurrent state 在 decode 每步只被读/写一次（不像 latent K 要 gather 大量 entries），Gmem→TC 这一次读完全 hide 在 KDA kernel 的整体延迟内。

**Layout**：无特殊layout要求。

### 4.4 KV-transfer 考量

prefill→decode 转 KDA state 时，每 device 按 head TP=16 接收自己负责的 head 子集。写入 Gmem 与 decode 计算解耦；可以在 decode 启动前完成、或与其他层的 decode 重叠（PD 分离的标准做法）。

---

## 5. DSA 并行：三个方案

DSA 子层 6 层（GLM5-Next-16B）/ 11 层（GLM5-Next），是全量 attention，对 KV cache 有真实的存储与访问压力。

### 5.1 三个候选方案

**方案 1 · CP=16**

```
所有 16 device 组成一个 CP 组。每个 request 的 KV 按 token round-robin
分到 16 个 device，每 device 持 1/16。同一 device 处理所有 R 个 request
的自己那 1/16 token shard。
```

**方案 2 · CP=8 × DP=2**

```
16 device 分 2 个 DP 组（每组 8 device 组成 CP=8）。R 个 request 分到 2
组，每组 R/2 个。组内每 device 持 1/8 token shard。
```

**方案 3 · CP=4 × DP=4**

```
16 device 分 4 个 DP 组（每组 4 device 组成 CP=4）。R 个 request 分到 4
组，每组 R/4 个。组内每 device 持 1/4 token shard。
```

### 5.2 per-device 容量公式（三方案等价）

```
per_device_KV(R, dtype) = R · per_req_KV(dtype) / 16
```

这是关键观察：三方案都把 16 个 device 全部用满，per-device 持有 KV 比例 = R / 16，**与 CP/DP 划分无关**。三方案的差异只在：

- **通信作用域**（top-K AG、LSE merge 等的"范围"）
- **单 req 计算并行度**（直接影响单 req 延迟）
- **PD-transfer 拓扑**（scatter 的"宽度"）
- **R 不能被 DP 整除时的余 1 处理**

### 5.3 DSA 子结构与本文章节分配

DSA decode 路径每层做两件事：

1. **Indexer**：用 indexer Q（基于 hidden）扫所有历史 index K → 算 logits → 取 top-2048 → 输出 indices；
2. **Sparse MQA**：在 top-2048 选定的 latent KV 上做 MQA（在 absorb 空间）。

这两步用的 cache 是分开的：

| cache | 每 token 大小 | 用途 |
|---|---|---|
| **Index K cache** | 128 elem FP8（**Lmem**）+ 4 B fp32 scale（**Gmem**）= **132 B** | indexer GEMM 的 K 矩阵；scale 仅在 GEMM 之后 vec dequant 用一次（§6.3） |
| **Latent K cache** | 512 elem (= Rkv，Dro=0) → **1024 B bf16 / ~520 B fp8** | sparse MQA 的 K/V（在 absorb 形式下 V = K[:Rkv]） |

**§6 讲 Index K**（只考虑 Lmem）；**§7 讲 Sparse MQA**（Lmem / Gmem 都要考虑）。

---

## 6. DSA 之 Index K

### 6.1 Cache 形态与访问模式

每 token 每 DSA 层一行 `index_K [128] FP8 + scale fp32`。decode 每步：

1. 计算当前 token 的 `q_idx [I=8, Di=128]` 并 FP8 量化；
2. 对**全 seqlen × Di** 做 indexer GEMM（FP8 inner product per head），得到 `logits [I, seqlen]`；
3. 按 gate 加权 reduce 到 `[seqlen]`，取 top-2048 → 全局 indices。

→ index K cache 的访问是**全量扫描**（每 token、每 step、每 DSA 层都扫一次），是 BW-heavy 的读流。

### 6.2 容量矩阵（per-device，三方案等价）

#### 6.2.1 每 token 每层存储构成（= **132 B**）

每个 token 每个 DSA 层在 index K cache 里占两部分，**放在不同存储里**：

| 组件 | 元素数 | 元素 dtype | 字节 | 放置 |
|---|---:|---|---:|---|
| `index_K` 向量 | Di = **128** | FP8 (1 B / elem) | 128 · 1 = **128 B** | **Lmem**（进 TC 热路径） |
| 量化 scale | **1**（per-token, 整个 128-elem 行共享一个） | fp32 (4 B / elem) | 1 · 4 = **4 B** | **Gmem**（GEMM 之后才用） |
| **合计** | — | — | **132 B / token / 层** | Lmem 128 B + Gmem 4 B |

> **scale 是 per-token 一个**：FP8 量化用 per-row（= per-token）单 scale，块大小恰为 Di=128（与 indexer head_dim 一致）。计算流：GEMM 出 `logits[I, seqlen]` 后**先做 gate-weighted I-reduce 到 `[seqlen]`、再与 `k_scale[seqlen]` element-wise 相乘**（避免对 [I, seqlen] 做 broadcast 乘）。后面可以考虑BF16的scale。

#### 6.2.2 推到 per request、per device

**per request 全 DSA 层 cache 大小**（拆 Lmem 主体 vs Gmem scale）：

```
per_req_Lmem (K body) = 128 B · n_dsa_layer · seqlen
per_req_Gmem (scale)  =   4 B · n_dsa_layer · seqlen
```

| 模型 | n_dsa_layer | per req Lmem (K body) | per req Gmem (scale) | per req 合计 |
|---|---:|---:|---:|---:|
| GLM5-Next-16B | 6 | 128·6·512K = **384 MB** | 4·6·512K = **12 MB** | **396 MB** |
| GLM5-Next | 11 | 128·11·512K = **704 MB** | 4·11·512K = **22 MB** | **726 MB** |

**per-device**（三方案等价，= R · per_req / 16）：

| | R=4 Lmem | R=4 Gmem | R=8 Lmem | R=8 Gmem |
|---|---:|---:|---:|---:|
| GLM5-Next-16B | 96 MB | 3 MB | 192 MB | 6 MB |
| **GLM5-Next** | **176 MB** | **5.5 MB** | **352 MB** | **11 MB** |

### 6.3 放置：**K 主体 → Lmem，scale → Gmem**

- **`index_K` 向量（128 B / token，FP8）→ Lmem**
- **量化 scale（4 B / token，fp32）→ Gmem**

理由：

1. **K 主体 Lmem**——Inedx-K无需额外加工，进Lmem后可以直接参与计算；dim是128的FP8，与Lmem的Tile Layout 直接匹配(Lmem的Tile Layout要求在每行按128字节分段)，因此无需额外Re-Layout。
2. **scale Gmem**——scale 只在 GEMM **之后** 的 dequant 步骤（vec core，§6.2.1 的 `[seqlen] · [seqlen]` element-wise 乘）用一次，**不进 TC 热路径**；放 Gmem 让 Lmem 通道全留给 K 主体的全量扫描读。容量 11 MB / device（GLM5-Next R=8），占 Gmem 1.8 GB 不到 1%。

### 6.4 2-core 切分：**N-split 强制**

> 同 device 内 2 core 各自有 private Lmem，必须想清楚 index K 怎么分。

切法选择：

| 切法 | 是否可行 | 原因 |
|---|---|---|
| M-split (Q heads = I) | × | 依赖两个Core Duplicate Index-K，增大KV Transfer压力 |
| K-split (Di = 128) | × | TC FP8 MMA 的 K 最小粒度 = 128，不能再切 |
| **N-split (token)** | ✅ | 每 core 持 device 持有的 n_local 中的一半 token；GEMM 各自独立 |

**布局**：

- core 0 存 device's 一半 token；
- core 1 存另一半。

这把"device-level CP"在内部进一步加倍成"core-level CP"：方案 1 实际是 CP=32，方案 2 是 CP=16，方案 3 是 CP=8。但 **inter-device 通信不变**——核间归并 (2-way) 在 device 内通过 shmem 完成，不出 device。

### 6.5 Indexer GEMM 时间预算

每个 DSA 层、每 request、本 device 做一个独立 GEMM（不同 req 的 K 不同）：

```
[M = I = 8] · [K = Di = 128]  ×  [K = 128] · [N = n_local]^T  →  [I, n_local]   FP8 in, FP32 accum
其中 n_local = 512K / (CP-domain-size)
```

| 方案 | CP域 | n_local per device | R_per_device |
|---|---:|---:|---:|
| 方案 1 | 16 | 32 K | R |
| 方案 2 | 8 | 64 K | R / 2 |
| 方案 3 | 4 | 128 K | R / 4 |

**算密度**：`AI ≈ 2·I·N / (I + N) → 16 FP8 FLOPs/B`（N≫I 时），**远超 4.1 拐点 → 计算受限**。

**Per-device per-step FLOPs（不变于方案，因 R·CP 抵消）**：

```
FLOPs / step / device = R · 2 · I · (ctx / CP) · K · (CP_in_DP_group / CP_in_DP_group) · n_layer
                       = R · 2 · I · ctx · K · n_layer / 16     ← 与 CP/DP 无关
```

| 模型 | n_layer | R=4 FP8 FLOPs / step | R=8 FP8 FLOPs / step |
|---|---:|---:|---:|
| GLM5-Next-16B | 6 | 1.57 G | 3.15 G |
| GLM5-Next | 11 | 2.88 G | **5.77 G** |

**Roofline 时间**（FP8, per device, 2 TC 并行 @ 26.2 TFLOPS；**整段 = 一个 decode step 内所有 DSA 层的 indexer GEMM 累加**）：

| 模型 | R=4 | R=8 |
|---|---:|---:|
| GLM5-Next-16B (6 层 / step) | 60 µs | **120 µs** |
| GLM5-Next (11 层 / step) | 110 µs | **220 µs** |

> 注：单层一次 indexer GEMM 的纯 TC 时间 = 上表 / n_layer。例：GLM5-Next R=8 单层 ≈ 220/11 = **20 µs**。

**Lmem BW 占用核对**（per step per device，GLM5-Next R=8 为例）：
- K 读总量 = `R · ctx · 128 · n_layer / 16` = `8 · 512K · 128 · 11 / 16` = **352 MB / step**
- 在 step 内 220 µs 中读完 → **352 MB / 220 µs ≈ 1.6 TB/s** 实际使用
- 占 per-device Lmem BW 6.4 TB/s 的 **≈ 25%**

> 25% 的占用是"计算受限的合理表现"——AI = 5.77 G / 352 MB ≈ **16.4 FLOPs/B**，超 4.1 拐点 **4×**，所以 BW 用率正好是峰值的 1/4。瓶颈是 TC 计算。

### 6.6 三方案在 Index K 上的差异

| 维度 | 方案 1 (CP=16) | 方案 2 (CP=8×DP=2) | 方案 3 (CP=4×DP=4) |
|---|---|---|---|
| per-device 容量 | 同 | 同 | 同 |
| per-device GEMM 时间 | 同 | 同 | 同 |
| top-K 候选 AG 通信（per layer per req） | 16·2K·8B = **256 KB** | 8·2K·8B = 128 KB | 4·2K·8B = **64 KB** |
| top-K 候选 AG 通信（全 step，R=8、11 层、GLM5-Next） | 22 MB | 11 MB | **5.5 MB** |
| 单 req 通信作用域 | 全 16 device | 8 device | 4 device |
| 单 req top-K 延迟 | 最高 | 中 | **最低** |

→ Index K 这一面，**方案 3 通信最少**，**方案 1 单 req 延迟最高但简单**。

---

## 7. DSA 之 Sparse MQA

### 7.1 Sparse MQA 流程（per req per layer）

```
Step 1   q_new [Nh, 512]   ←  q_b_proj + split + bmm(w_kc)          # 与 §6 共用 hidden 之前
Step 2   topk_slots [2048] ←  来自 §6 indexer 的 global top-K (经 page-table 映射)
Step 3   K_topk [N_eff, 512]  ←  gather latent K 从 cache（只取本 device 持有的部分，N_eff ≈ 2048/CP-domain）
Step 4   logits [Nh, N_eff]   ←  q_new · K_topk^T               GEMM-1
Step 5   attn  [Nh, N_eff]    ←  softmax(logits)                   (vec core)
Step 6   out_latent [Nh, 512] ←  attn · K_topk[:, :Rkv]            GEMM-2  (absorb 形式下 V = K[:Rkv])
Step 7   (跨-CP-rank LSE merge → 全局 out_latent)
Step 8   bmm(w_vc) → out [Nh, Dv]  → o_proj 到 [H]                 # 后续，不在本节
```

GEMM-1 与 GEMM-2 是 sparse MQA 的两个 TC 重头戏，本节聚焦。

### 7.2 Latent K 容量矩阵（per-device，三方案等价）

每 token 512 elem。`per_req @ 512K`：

| 模型 | n_layer | per token 全层 | bf16 per req | fp8 per req |
|---|---:|---:|---:|---:|
| GLM5-Next-16B | 6 | 6·1024 = 6 KB | **3.00 GB** | **1.53 GB** |
| GLM5-Next | 11 | 11·1024 = 11 KB | **5.50 GB** | **2.80 GB** |

per-device:

| | R=4 bf16 | R=4 fp8 | R=8 bf16 | R=8 fp8 |
|---|---:|---:|---:|---:|
| GLM5-Next-16B | **0.75 GB** | **0.38 GB** | **1.50 GB** | **0.77 GB** |
| GLM5-Next | **1.375 GB** | **0.70 GB** | **2.75 GB** | **1.40 GB** |

### 7.3 Gmem 可行性（容量端）

per-device Gmem 容量 = **1.8 GB**。要把 latent K 放 Gmem，需要 latent K + KDA state + 其他临时 buffer 总和 ≤ 1.8 GB。

KDA state 最大 = GLM5-Next R=8 ≈ 36 MB（bf16）。其他临时 buffer 预留 ~200 MB 比较稳。所以 latent K 实际可用预算 ≈ **1.5 GB**。

| | R=4 bf16 | R=4 fp8 | R=8 bf16 | R=8 fp8 |
|---|---:|---:|---:|---:|
| GLM5-Next-16B | ✅ 0.75 GB | ✅ 0.38 GB | ✅ 1.50 GB（紧） | ✅ 0.77 GB |
| **GLM5-Next** | ✅ 1.375 GB | ✅ 0.70 GB | ❌ 2.75 GB | ✅ **1.40 GB**（80% Gmem 占用） |

→ **GLM5-Next R=8 必须 fp8 才能进 Gmem**；其他组合都有进 Gmem 的可能性。**GLM5-Next-16B 任意组合都能进 Gmem**。

### 7.4 Gmem vs Lmem 路径对比

**路径 X · Latent K in Gmem**

```
host (prefill) ─KV-transfer ─► Gmem  (full latent K shard)         ── 写不扰 TC
                                │
                  decode step   ▼
                               Gmem ──gather (Gmem read, N_eff × 512)──►  shmem
                                                                            │
                                                                            ▼
                                                                           Lmem ──► TC sparse MQA
```

收益：
- gather 走 Gmem，**与 TC 读 Lmem 完全解耦**（不同存储通道）；
- KV-transfer 走 Gmem 高效通道；
- Lmem 完全留给 indexer 流（index K 读 + index K cache）。

代价：
- Gmem 容量受限（见 §7.3）；
- gather 端依赖 Gmem 读 BW（Gmem BW 通常低于 Lmem，但 gather 数据量小，几乎免费）。

**路径 Y · Latent K in Lmem**

```
host ─KV-transfer─►  Lmem (full latent K cache)                    ── 与 TC 抢 BW，要错峰
                       │
       decode step     ▼
                      Lmem ──gather (Lmem read)──►  shmem  ──►  Lmem  ──► TC
                                ▲                                         ▲
                                └─── 与 indexer K read / TC 抢 Lmem    ───┘
```

收益：
- 容量充足，bf16 也能塞下；

代价：
- KV-transfer 与 decode TC 抢 Lmem BW，需 PD 协议错峰；
- **需要测试评估"从Lmem做Gather最后写回Lmem"与"从Gmem做Gather最后写回Lmem"的性能差异**。

### 7.5 2-core 切分：三选项

sparse MQA 的 **M = Nh ∈ {32, 64}**（远大于 indexer 的 I=8），所以 **M 也能切**。三选项都数学等价（输出逐元素相同）：

#### 7.5.1 (a) M-split：Nh / 2 头 / core

每 core 算 Nh/2 个 Q head，**K 是 device 共享**：

```
Core 0:  q_half[Nh/2, 512] · K_full[N_eff, 512]^T  →  logits_half[Nh/2, N_eff]
         softmax + GEMM-2  →  out_half_0[Nh/2, 512]
Core 1:  同上但用另一半 Q head                       →  out_half_1[Nh/2, 512]
末尾沿 head 维 concat → out[Nh, 512]   （零通信）
```

- **K 存储**：需要"两 core 都能访问完整 N_local × 512 的 K"。
  - 配 **Gmem-K（路径 X）**：自然成立——Gmem 是 device 共享，两 core 各做一次 gather 把同一份 K 拉到自己 private Lmem scratch。✅
  - 配 **Lmem-K（路径 Y）**：要么 K cache 在两 core Lmem 复制（per-core 容量 2×），要么gather到Gmem后再Load。
- **跨核通信**：**零**（concat 是 head 维拼接，结果落不同位置）。
- **复杂度**：最低。

#### 7.5.2 (b) N-split：N_local / 2 token / core

每 core 持 device 一半 token、完整 head_dim：

```
Core 0:  q[Nh, 512] · K_half_0[N_eff/2, 512]^T  →  partial_out_0[Nh, 512] + lse_0[Nh]
Core 1:  同上但 K 的另一半 token                  →  partial_out_1[Nh, 512] + lse_1[Nh]
末尾  →  online-softmax LSE merge (2-way)        →  out[Nh, 512]
```

- **K 存储**：每 core 半数 token × 全 head_dim；total per-device 不变。
- **跨核通信**：LSE merge ≈ `(out[Nh,512]+lse[Nh])·2B ≈ 65 KB / 层 / req`（all-gather）或 33 KB（ring）。
- **复杂度**：中——需要 online-softmax merge 算子。
- 与 indexer 的 N-split 协议一致（KV-transfer 端按 token round-robin 分 2 core，两类 K 一致）。

#### 7.5.3 (c) K-split：head_dim 256 / core

每 core 持完整 N_local × 半 head_dim（256 维）：

```
Core 0:  q_dim0[Nh, 256] · K_dim0[N_local, 256]^T  →  partial_logits[Nh, N_eff]
Core 1:  q_dim1 · K_dim1                          →  partial_logits[Nh, N_eff]
reduce-sum 跨核 → logits = partial_0 + partial_1
softmax → broadcast attn
Core 0:  attn · K_dim0  →  out_dim0[Nh, 256]
Core 1:  attn · K_dim1  →  out_dim1[Nh, 256]
concat 沿 head_dim → out[Nh, 512]
```

- **K 存储**：每 core 完整 N_local × 半 dim。
- **跨核通信**：reduce-sum logits + broadcast attn ≈ `Nh·N_eff·6B ≈ 48 KB / 层 / req`。
- **复杂度**：中——需要 GEMM-1 后的同步 reduce-sum。

#### 7.5.4 推荐组合（取决于 §7.4 的 latent K 位置）

| latent K 位置 | sparse MQA 切法 | 理由 |
|---|---|---|
| **Gmem（路径 X）** | **(a) M-split** | 最干净：K 自然 device 共享、零跨核通信、零归并算子；只末尾 concat。Lmem 不用装 latent K 蓄水池。 |
| **Lmem（路径 Y）** | **(b) N-split** | 协议与 indexer 统一（按 token round-robin）；端到端独立，只末尾一次 LSE merge；不需要 2× Lmem 复制 K。 |
| K-split 何时上 | 仅当 DMA 对 "按 head_dim stride-2 加载" 特别友好时考虑 | 其他场景没特别优势 |

### 7.6 Kernel 内 tile 粒度与 two-pass 设计

> 上文给的"per-device per-step FLOPs"是按"实际拥有 token 数"算的理想值。真实 kernel 走 **tile-rounded 变长 sparse attention**——每 device 处理 `ceil(N_actual / 128) · 128` 行 K（128 是 TC FP8/BF16 N-tile 最小粒度）。下表的工程选择决定圆整代价和 kernel 结构。

#### 7.6.1 GEMM-1 `BLOCK_SIZE_N` = 128（推荐）

GEMM-1：`Q[Nh, 512] · K_topk[N_actual, 512]^T → logits[Nh, N_actual]`。

- 硬件下限：TC FP8 / BF16 N-tile 均为 **128** → `BLOCK_N` 必须是 128 的倍数。
- 选 **128** 而非 256/512 的原因：
  - **圆整浪费最小**——每个 N-tile 最坏浪费 127 列；方案 1 平均 N=128 时 1 tile 命中（0 浪费），方案 3 平均 N=512 时 4 tile 命中。
  - **load-imbalance 容差**——round-robin 下 max N per device ≈ mean + 3·std；BLOCK_N=128 让 "max 比 mean 多一个 tile" 这种最坏情况只多 50% 工作（方案 1 从 1→2 tile），更大 BLOCK_N 比例会更糟。

→ `BLOCK_N_GEMM1 = 128`，`BLOCK_K_GEMM1 = 512`（K=Rkv=512）。`BLOCK_M_GEMM1` = 同 2-core 切法（M-split 时 32，否则 64）。

#### 7.6.2 Softmax **不沿 N 切**，只对 BLOCK_M tile

N_actual 上限 2048（全 mask padding 最坏情况），logits 全量 fp32 scratch：

```
logits scratch = Nh · N_aligned · 4B
              ≤ 64 · 2048 · 4 = 512 KB     (最坏)
               ≈ 64 · 128  · 4 = 32 KB      (方案 1 实际)
```

Lmem per core 14.38 GB 完全装得下；甚至放 shmem 也行。

**Two-pass 优于在线 softmax 的判据**：

| 维度 | Two-pass（推荐） | FlashAttention 在线 softmax |
|---|---|---|
| Softmax 沿 N tile？ | 否，单遍全 N | 是，每 N-tile 维护 running max/sum |
| Logits scratch | 需要 32–512 KB / req | 不需要 |
| 跨 tile 簿记 | 无 | 每 tile：reduce + rescale 旧 partial out |
| Kernel 复杂度 | **低**（GEMM-1 → softmax → GEMM-2 三段独立） | 高（softmax 与 GEMM-2 融合，每 tile rescale 之前累加） |
| 适用 N 范围 | 小 N (≤2K) 时是干净的选择 | 大 N (≥8K) 时强制使用 |

→ 本场景 N ≤ 2048、Lmem scratch 富裕 → **two-pass 是正确选择**。

**Softmax kernel**（per device per req per layer）：

```
for m in 0..Nh step BLOCK_M_SOFTMAX (=32 或 64):
    max_block  = vec_reduce_max(logits[m:m+BM, :N_actual])
    logits[m:m+BM] = vec_exp(logits[m:m+BM] - max_block)
    sum_block  = vec_reduce_sum(logits[m:m+BM, :N_actual])
    attn[m:m+BM]  = logits[m:m+BM] / sum_block
    lse_block  = max_block + log(sum_block)          # 供跨 device merge
```

vec ops per device per step（R=8, 11 层, 方案 1 N=128 平均）：
- reduce-max + exp + reduce-sum + normalize ≈ 4 · Nh · N ≈ 32 K vec ops / req / 层
- × 8 req × 11 层 ≈ **2.8 M FP32 ops / step / device** → 256 fp32 lane @ 400 MHz ≈ **30 µs**。便宜。

#### 7.6.3 GEMM-2 tile 选择

GEMM-2：`attn[Nh, N_actual] · K_topk[N_actual, 512] → out[Nh, 512]`

注意：GEMM-1 的 **N 维（N_actual）变成 GEMM-2 的 K 维（内积维）**；head_dim 512 变成 GEMM-2 的 N 维（输出）。

| 维度 | 值 | 理由 |
|---|---|---|
| `BLOCK_M_GEMM2` | = `BLOCK_M_GEMM1`（32/64） | 与 GEMM-1 M-flow 一致，attn scratch 直接复用 |
| `BLOCK_K_GEMM2`（内积维=旧 N） | **256/512/1024？** | 待定，只不过Tensor Core期望更大一点的K提效率 |
| `BLOCK_N_GEMM2`（输出 dim） | **512** | 直接覆盖完整 head_dim |


#### 7.6.4 端到端 phase 流（per req per layer 内部）

```
┌─ Phase 1 · gather ──────────────────────────────────────┐
│  N_actual 行 latent K → K_scratch[N_aligned, 512]       │
│  数据量：~32–128 KB（视方案）                             │
├─ Phase 2 · GEMM-1 (TC) ─────────────────────────────────┤
│  BLOCK_M=32/64, BLOCK_N=128, BLOCK_K=512                │
│  输出 logits[Nh, N_aligned] (fp32, ~32–512 KB scratch)   │
├─ Phase 3 · Softmax (vec, two-pass) ─────────────────────┤
│  仅 BLOCK_M tile, N 全长一遍                             │
│  输出 attn[Nh, N_aligned] + lse[Nh]                     │
├─ Phase 4 · GEMM-2 (TC) ─────────────────────────────────┤
│  BLOCK_M=同 GEMM-1, BLOCK_K=128, BLOCK_N=512            │
│  输出 out_latent[Nh, 512]                               │
└─ Phase 5 · 跨 device LSE merge ─────────────────────────┘
   ring all-reduce 用 online-softmax merge 算子（V6 §10）
```

#### 7.6.5 对三方案的影响

- 三方案的 N_actual 平均 = 128 / 256 / 512，BLOCK_N=128 让 1/2/4 tile 自然命中，**FLOPs 与 §7.7 表的"理想均分"值基本一致**（仅有 +25%/+17%/+8% 的 load-balance tail，由 max-per-device 决定）。
- 若 kernel 改成朴素 pad-2048（不推荐），FLOPs 退化到固定 `R_per_dev · 268 M·n_layer`，方案 3 会变成方案 1 的 **4×**。本节的 tile-rounded 设计避免了这个退化。

### 7.7 Sparse MQA GEMM 时间预算

**Per req per layer FLOPs**（GEMM-1 + GEMM-2，per device）：

```
= 2 · 2 · Nh · 512 · N_eff                   N_eff = 2048 / CP-domain-size
```

**Per-device per-step FLOPs**（聚合 R_per_device, n_layer, 三方案等价）：

```
= R · Nh · 2048 · 512 · 4 / 16 · n_layer    (R 与 CP 抵消)
```

| 模型 | n_layer | R=4 总 FLOPs / step | R=8 总 FLOPs / step |
|---|---:|---:|---:|
| GLM5-Next-16B (Nh=32) | 6 | 201 M | 403 M |
| GLM5-Next (Nh=64) | 11 | **738 M** | **1.48 G** |

**时间**（per device, 2 TC 并行；前提：kernel 走 §7.6 的 tile-rounded 变长 sparse attention）：

| 模型 | R | BF16（13.1 TFLOPS）| FP8（26.2 TFLOPS）|
|---|---:|---:|---:|
| GLM5-Next-16B | 4 | 15 µs | 8 µs |
| GLM5-Next-16B | 8 | **31 µs** | **15 µs** |
| GLM5-Next | 4 | 56 µs | 28 µs |
| GLM5-Next | 8 | **113 µs** | **56 µs** |

**算密度**：`AI ≈ 2·Nh / (1 + Nh/N_eff) ≈ 64–100`，**远超 4.1 → 计算受限**（确认）。

> **load-balance tail**：round-robin 下 max-N-per-device ≈ mean + 3·std。方案 1 (max ≈ 160) 比 mean 多 25%，方案 3 (max ≈ 550) 多 8%。绝对 wall-clock 取 max → 方案 3 在 tail latency 上**略好** 5–10%。

### 7.8 三方案在 Sparse MQA 上的差异

| 维度 | 方案 1 (CP=16) | 方案 2 (CP=8×DP=2) | 方案 3 (CP=4×DP=4) |
|---|---|---|---|
| N_eff per device (mean) | 128 | 256 | 512 |
| N-tile 数 / req（GEMM-1, BLOCK_N=128） | 1 | 2 | 4 |
| GEMM 时间 per step (理想均分) | 同 | 同 | 同 |
| Load-balance tail (max/mean) | 1.25× | 1.17× | **1.08×** |
| LSE merge 范围 | 16 device | 8 device | 4 device |
| LSE merge 通信（per layer per req, ring） | 33 KB | 33 KB | 33 KB |
| LSE merge 通信（全 step, R=8、11 层、GLM5-Next） | 11·8·33 KB ≈ 2.9 MB | 同 | 同 |
| 配 M-split（Gmem-K）时跨核 comm | 0 | 0 | 0 |
| 配 N-split（Lmem-K）时跨核 comm | 极小 | 极小 | 极小 |

→ Sparse MQA 这一面，三方案的差异**主要在 load-balance tail**（5–10%，方案 3 略好）——LSE merge 数据量本就小，且 ring 后总传输量与作用域无关。**Sparse MQA 不是选方案的决定因素，决定方案的是 §6.6 的 indexer top-K 通信**。

---

## 8. 端到端 DSA Step 时间预算

> Per-device, R=4 或 R=8, GLM5-Next 11 层 / GLM5-Next-16B 模型 6 层。
> 三方案对 GEMM 时间相同；通信差异体现在 top-K AG 与 LSE merge。

### 8.1 各组件时间表（per step per device）

| 组件 | GLM5-Next-16B R=4 | GLM5-Next-16B R=8 | GLM5-Next R=4 | GLM5-Next R=8 |
|---|---:|---:|---:|---:|
| Indexer GEMM (FP8) | 60 µs | 120 µs | 110 µs | **220 µs** |
| Sparse MQA GEMM (BF16 latent) | 15 µs | 31 µs | 56 µs | 113 µs |
| Sparse MQA GEMM (FP8 latent) | 8 µs | 15 µs | 28 µs | 56 µs |
| 本地 top-K (vec) | ~55 µs | ~105 µs | ~55 µs | ~105 µs |
| LSE merge (vec) | ~35 µs | ~67 µs | ~35 µs | ~67 µs |
| top-K AG + page-table | ~5 µs | ~10 µs | ~5 µs | ~10 µs |
| **小计 (BF16 latent)** | **~170 µs** | **~333 µs** | **~260 µs** | **~515 µs** |
| **小计 (FP8 latent)** | **~163 µs** | **~317 µs** | **~232 µs** | **~458 µs** |

> 由于有各类流水线延时，实际计算效率与带宽效率损失；保守的实际值取上述 3× 。

### 8.2 哪里是瓶颈

- **Indexer GEMM** 是单 step 最大头（GLM5-Next、R=8 → 220 µs，占总 ~45%）。
- **Sparse MQA GEMM** 第二大（最长 113 µs）。
- vec 工作（top-K + LSE merge）合计 ~150–170 µs（R=8），可与 GEMM 流水（不同 core / 不同 unit），实际并不完全串行计入。
- **通信很小**（< 22 MB / step），对总时间贡献 µs 级。

→ **降总时间最大杠杆**：FP8 latent（省 sparse MQA 一半 GEMM）+ 流水化 vec 与 TC。

---


## 9. 开放问题 / 待 benchmark

1. **fp8 latent K 落地** —— GLM5-Next R=8 强依赖 fp8 把 latent K 压进 Gmem；量化/反量化 kernel + sparse MQA FP8 kernel 是否就绪？精度回归是否过关？
2. Index-K 使用 BF16 的scale？
3. **Gmem 实测 BW** —— 需要测试评估"从Lmem做Gather最后写回Lmem"与"从Gmem做Gather最后写回Lmem"的性能差异。
4. **DP 不整除 R** —— R=5/6/7 时怎么分？最简单是退方案 1，或方案 2 接受 ±1 不均衡；待具体调度协议设计。
5. **针对CP时不均的问题，需要专门设计算子，主要是避免直接每个padding到2048**
6. **粗略估时间看，8req / instance 应该是比较合适的了，再多req容易不满足SLO**

---

<!-- ## 9. 容量总账与推荐组合

### 9.1 Per-device 容量总账（4 个端点组合）

> "Gmem 主路线" = §7.4 路径 X（latent K → Gmem）；"Lmem 备路线" = 路径 Y。

#### 9.1.1 GLM5-Next，R=8

| 类别 | 位置 | bf16 | fp8 latent |
|---|---|---:|---:|
| KDA state (TP=16) | Gmem | 36 MB | 36 MB |
| DSA index K · K body (CP shard) | Lmem | 352 MB | 352 MB |
| DSA index K · scale | Gmem | 11 MB | 11 MB |
| DSA latent K (CP shard) | **Lmem 备路线** / **Gmem 主路线** | 2.75 GB Lmem (bf16 不能进 Gmem) | **1.40 GB Gmem** |
| **Gmem 占用** | — | 47 MB / 1.8 GB (3%) | **1.45 GB / 1.8 GB (80%)** |
| **Lmem 占用 per device** | — | **3.10 GB / 28.75 GB (10.8%)** | **352 MB / 28.75 GB (1.2%)** |
| **Lmem 占用 per core** | — | 1.55 GB / 14.38 GB (10.8%) | 176 MB / 14.38 GB (1.2%) |

→ R=8 GLM5-Next：**fp8 latent + Gmem 主路线** 比 bf16 + Lmem 备路线 Lmem 占用低 ~9×。

#### 9.1.2 GLM5-Next，R=4

| 类别 | 位置 | bf16 latent | fp8 latent |
|---|---|---:|---:|
| KDA state | Gmem | 18.2 MB | 18.2 MB |
| Index K · K body | Lmem | 176 MB | 176 MB |
| Index K · scale | Gmem | 5.5 MB | 5.5 MB |
| Latent K | **Gmem 主路线** | **1.375 GB Gmem** | **0.70 GB Gmem** |
| **Gmem 占用** | — | **1.40 GB / 1.8 GB (78%)** | **0.72 GB / 1.8 GB (40%)** |
| **Lmem 占用** | — | 176 MB (0.6%) | 176 MB (0.6%) |

→ R=4 GLM5-Next：两种 dtype 都能走 Gmem 主路线；fp8 更宽松。

#### 9.1.3 GLM5-Next-16B（GLM5-Next-16B），R=8

| 类别 | 位置 | bf16 | fp8 |
|---|---|---:|---:|
| KDA state | Gmem | 3.7 MB | 3.7 MB |
| Index K · K body | Lmem | 192 MB | 192 MB |
| Index K · scale | Gmem | 6 MB | 6 MB |
| Latent K | Gmem | 1.50 GB | 0.77 GB |
| Gmem 占用 | — | 1.51 GB / 1.8 GB (84%) | 0.78 GB / 1.8 GB (43%) |
| Lmem 占用 | — | 192 MB (0.7%) | 192 MB (0.7%) |

→ GLM5-Next-16B R=8：两种 dtype 都进 Gmem，bf16 偏紧。

#### 9.1.4 GLM5-Next-16B（GLM5-Next-16B），R=4

| 类别 | 位置 | bf16 | fp8 |
|---|---|---:|---:|
| KDA state | Gmem | 1.9 MB | 1.9 MB |
| Index K · K body | Lmem | 96 MB | 96 MB |
| Index K · scale | Gmem | 3 MB | 3 MB |
| Latent K | Gmem | 0.75 GB | 0.38 GB |
| Gmem 占用 | — | 0.75 GB / 1.8 GB (42%) | 0.38 GB / 1.8 GB (21%) |
| Lmem 占用 | — | 96 MB (0.3%) | 96 MB (0.3%) |

→ GLM5-Next-16B R=4：宽松到几乎随便选。

### 9.2 推荐组合

#### 9.2.1 并行方案选择

| R | 推荐方案 | 备选 | 理由 |
|---|---|---|---|
| **R=4** | **方案 3 (CP=4×DP=4)** | 方案 1 (CP=16) | R=4 完美整除 DP=4（每组 1 req），通信作用域最小，PD-transfer 拓扑友好 |
| **R=8** | **方案 2 (CP=8×DP=2)** | 方案 1 (CP=16) | R=8 完美整除 DP=2（每组 4 req），通信中等，负载均衡好 |
| 追求最低单 req 延迟 | 方案 1 (CP=16) | — | 16 路并行计算 partial，单 req 延迟最低 |
| R 不能整除 DP | 方案 1 (CP=16) | — | DP 路线需要 DP 整除 R；CP=16 没这个限制 |

#### 9.2.2 数据放置主路线

| 模型 | R | 推荐 dtype | latent K 位置 | sparse MQA 2-core 切法 |
|---|---|---|---|---|
| GLM5-Next-16B | 4 | bf16 / fp8 | **Gmem** | **M-split** |
| GLM5-Next-16B | 8 | fp8 优先 / bf16 OK | **Gmem** | **M-split** |
| GLM5-Next | 4 | bf16 / fp8 | **Gmem** | **M-split** |
| **GLM5-Next** | **8** | **fp8（必须）** | **Gmem** | **M-split** |
| GLM5-Next (R=8 fallback 若 fp8 不可) | 8 | bf16 | **Lmem** | **N-split** |

Indexer 始终：**Lmem + N-split**。

### 9.3 何时切回 Lmem 备路线

只有这一种情况强制走 Lmem 备路线：**GLM5-Next + R=8 + 不能用 fp8 latent**（精度或 kernel 没就绪）。此时 latent K 2.75 GB / device 装不进 Gmem 1.8 GB，必须 Lmem；sparse MQA 切法相应改 N-split。

---

## 10. 速查（一页本）

```
======= GLM5-Next/GLM5-Next-16B Decode · 16-device · 4–8 req · 512K ctx · 速查 =======

模型 (Dro=0, latent per token = 512 elem)
  GLM5-Next-16B    : Nh=32, DSA 6 层, KDA 21 层
  GLM5-Next     : Nh=64, DSA 11 层, KDA 34 层

硬件
  device : 2 core; 每 core 1 TC + 4 vec core
  TC     : FP8 13.1 TFLOPS, BF16 6.55 TFLOPS (per core; ×2 = per device)
  Vec    : 64 (bf16/fp8) / 32 (fp32) lane per vc; 4 vc per core
  Gmem   : 1.8 GB/device, host I/O 独立, 不扰 TC
  Lmem   : 14.72 GB/core (= 28.75 GB/device), per-core private, host 写扰 TC
  Lmem BW: 3.2 TB/s per core (= 6.4 TB/s per device)

并行
  Linear Attention (KDA) : TP=16 固定 (head 切), state → Gmem
                          per-device per-req: GLM5-Next-16B 478 KB / GLM5-Next 4.55 MB
                          per-device R=8 GLM5-Next: ~36 MB (bf16) / 70 MB (fp32+bf16)
  DSA                    : 三方案
                          方案 1: CP=16
                          方案 2: CP=8×DP=2  → R=8 主推
                          方案 3: CP=4×DP=4  → R=4 主推
                          三方案 per-device 容量等价: R · per_req / 16

DSA Index K
  位置                  : Lmem (固定, BW-bound on K read)
  per-device 容量       : GLM5-Next R=4 177 MB / R=8 354 MB; GLM5-Next-16B R=4 99 MB / R=8 198 MB
  2-core 切法           : N-split 强制 (head_dim 128 是 TC 最小粒度)
  Indexer GEMM 时间     : GLM5-Next R=8 220 µs; GLM5-Next R=4 110 µs; GLM5-Next-16B R=8 120 µs; GLM5-Next-16B R=4 60 µs

DSA Sparse MQA
  位置                  : Gmem 主路线 (路径 X) / Lmem 备路线 (路径 Y)
  Gmem 容量预算         : ~1.5 GB 给 latent K (留 200 MB transfer + 36 MB KDA)
  bf16 latent 进 Gmem   : GLM5-Next-16B 任意 R / GLM5-Next R=4
  fp8  latent 进 Gmem   : 任意组合 (GLM5-Next R=8 也行, 1.40 GB)
  bf16 必走 Lmem        : GLM5-Next R=8 唯一
  2-core 切法           : M-split (Gmem-K) / N-split (Lmem-K) / K-split (备选)
  sparse MQA GEMM 时间  : GLM5-Next R=8 BF16 113 µs / FP8 56 µs
                          GLM5-Next R=4 BF16 56 µs / FP8 28 µs

每 device 每 DSA step 时间 (粗预算)
                      bf16 latent  fp8 latent
  GLM5-Next-16B  R=4         :  ~170 µs      ~163 µs
  GLM5-Next-16B  R=8         :  ~333 µs      ~317 µs
  GLM5-Next    R=4         :  ~260 µs      ~232 µs
  GLM5-Next    R=8         :  ~515 µs      ~458 µs   ← 主推 fp8 + Gmem 路径

通信总量 (per step, GLM5-Next R=8, 11 层)
  方案 1: top-K AG 22 MB + LSE merge 1 MB ≈ ~23 MB
  方案 2: ~12 MB
  方案 3: ~6.5 MB

主推组合
  R=4: 方案 3 + Gmem-K + M-split + bf16 或 fp8 latent
  R=8: 方案 2 + Gmem-K + M-split + fp8 latent (必须)
       fallback: Lmem-K + N-split + bf16 latent (GLM5-Next R=8 fp8 不可时)
  Indexer 永远 Lmem + N-split
```

---

## 11. 开放问题 / 待 benchmark

1. **fp8 latent K 落地** —— GLM5-Next R=8 强依赖 fp8 把 latent K 压进 Gmem；量化/反量化 kernel + sparse MQA FP8 kernel 是否就绪？精度回归是否过关？
2. **TC 数 / core** —— §2.3 假设 1 TC / core；实际可能是 0.5 / 2 / 其他；按比例缩放 §8.1 即可。
3. **Gmem 实测 BW** —— sparse MQA 走 Gmem-K 时，每 step gather ~22 MB（M-split 翻倍）；Gmem BW 是否够支撑无延迟？理论上几 µs，实测要确认。
4. **Lmem 双流并发**（备路线下）—— index K 流（352 MB / step）和 sparse MQA gather 流（~11 MB / step）同时在 Lmem 上跑，issue port 与 cache-line 是否互扰？
5. **2-core M-split + Gmem-K 的 gather 翻倍** —— 两 core 各自从 Gmem 拉同一份 K，是否能通过共享 shmem 单次 gather + 拆分到两 core 来减半？取决于 shmem 是否跨 core 可见。
6. **DP 不整除 R** —— R=5/6/7 时怎么分？最简单是退方案 1，或方案 2 接受 ±1 不均衡；待具体调度协议设计。
7. **方案 3 在 R=8 时的负载** —— 每 DP-group 2 req，组间通信小但单 req 计算并行度只 4，单 req 延迟更高；是否影响 SLO？
8. **PD-transfer 协议与 2-core 布局耦合** —— N-split 要求按 token round-robin 写到两 core；M-split 不要求；transfer 端是否能两套都支持？

---

## 12. 附录：关键公式快查

```
Latent K per token (Dro=0): 512 elem  →  bf16 1024 B / fp8 ~520 B
Index  K per token        : 128 elem FP8 + 4 B scale = 132 B
KDA per req per layer (TP=16, bf16):
  GLM5-Next-16B  = 23,328 B ≈ 22.8 KB
  GLM5-Next       = 140,288 B ≈ 137 KB

per-device 容量 (三方案等价):
  KV(R, dtype) = R · per_req_KV(dtype) / 16

Indexer GEMM FLOPs/step/device (三方案等价):
  = R · 2 · I · ctx · Di · n_layer / 16
  = R · 2 · 8 · 512K · 128 · n_layer / 16
  GLM5-Next-16B (n_layer=6): R · 393 M
  GLM5-Next  (n_layer=11): R · 720 M

Sparse MQA GEMM FLOPs/step/device (三方案等价):
  = R · 4 · Nh · 512 · 2048 · n_layer / 16  (= R · Nh · 2^18 · n_layer / 16)
  GLM5-Next-16B (Nh=32, n_layer=6):  R · 50 M
  GLM5-Next  (Nh=64, n_layer=11): R · 184 M

TC 时间 = FLOPs / TFLOPS:
  FP8  per device: 26.2 TFLOPS
  BF16 per device: 13.1 TFLOPS

算密度拐点: 4.1 FP8 FLOPs/B (compute > BW above this)
Indexer AI: ~16 → 计算受限
Sparse AI:  ~64–100 → 计算受限
``` -->
