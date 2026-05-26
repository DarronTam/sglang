# GLM5-Next Decode · 16-Device KV 内存放置与并行方案分析（v1）

> 场景：**PD-disaggregation 的 decode worker**，16 device、每 device 2 core。
> 关注三类 state：**KDA recurrent/conv state**、**DSA indexer K**、**DSA latent K**。
> 模型配置以 `config_16b_v2.json`（小，6 个 DSA 层 / 21 个 KDA 层）和 `config.json`（大，11 个 DSA 层 / 34 个 KDA 层）为准；注意两者 `qk_rope_head_dim = 0`，**latent 每 token = Rkv = 512 elem**，没有 k_pe 分量。
> 配套：`GlmMoeDsa_dev_V5.md`（单卡 decode 链路）、`GlmMoeDsa_dev_V6.md`（CP Route B 设计）。

---

## 0. 一句话

- **KDA state** → 强制 TP=16，**放 Gmem**：单条很小（几百 KB / req / device）但条数多（KV-transfer 写入频繁），放 Lmem 容易反复打扰 TC。
- **DSA indexer K** → **放 Lmem**：层数少、容量可控（每 req 0.4–0.7 GB / 全 16 device），且本来就是 GEMM 的直接输入，没有"中间加工"的余地，直接近 TC 最划算。
- **DSA latent K** → 必然 Lmem（≈4 GB/device 量级超 Gmem 1.8 GB 容量上限）；待评估的是 **gather 路径**：`Gmem→shmem→Lmem` vs `Lmem→shmem→Lmem`。
- **DSA 并行**：在 **CP / DP / 混合（±TP）** 中选，本文给出 5 种组合的容量、通信、KV-transfer 影响对比，未定论，待 benchmark。

---

## 1. 硬件约束（输入条件）

| 维度 | Gmem (per device) | Lmem (per device) | Lmem (per core, 2 core/device) |
|---|---:|---:|---:|
| 容量 | **1.8 GB** | **29,440 MB ≈ 28.75 GB** | **14,720 MB ≈ 14.38 GB** |
| Host 写效率 | 高（独立通道） | 较低 | 较低 |
| Host 写 vs TC | **不冲突**（独立） | **会与 TC 竞争** | 同上 |
| TC 读延迟/BW | 远（远存） | 近（近存） | 近 |
| 对齐要求 | 16 KB | 16 KB | 16 KB |

**16 device 总容量**：Gmem 28.8 GB / Lmem 460 GB。
**关键含义**：

1. KV-transfer 是 host→device 写。**写得多但每次小** → 优先 Gmem（避免与 TC 抢 Lmem）；**单次大块写** → 容量不允许只能用 Lmem，但可以与 decode 计算在时间上错开（PD 分离的好处）。
2. **TC 在 decode 阶段读 Lmem**：gather 路径若从 Lmem 出发，会与 TC 抢 Lmem read BW；若从 Gmem 出发，gather 与 TC 算 GEMM 可以错开/并行。

---

## 2. 模型 state 概览（per request）

> 维度沿用 `config_16b_v2.json` / `config.json`，bf16 字节数 = 2，FP8 = 1，索引位置用 int32 = 4。

### 2.1 KDA（linear attention）—— 跟 seqlen 无关

| 项 | 形状 / 公式 | 16B | 大 GLM-5 |
|---|---|---:|---:|
| 每层 recurrent state `S` | `[H, Kd, Vd]` | 32·72·72 = 165,888 elem | 64·128·128 = 1,048,576 elem |
| 每层 short-conv state | `3·H·Kd·(k−1)` | 3·32·72·3 = 20,736 elem | 3·64·128·3 = 73,728 elem |
| 每层合计（bf16） | — | **373 KB** | **2.14 MB** |
| KDA 层数 | — | 21 | 34 |
| **per-request KDA 全部（bf16）** | — | **≈ 7.65 MB** | **≈ 72.8 MB** |
| 同上（recurrent 用 fp32 + conv bf16） | — | ≈ 14.4 MB | ≈ 141 MB |

> recurrent state 通常推荐 fp32（数值稳定），conv state 跟激活 dtype；下文容量预算两种 dtype 都列。

### 2.2 DSA indexer K cache —— 线性正比 S

每 token 每层 FP8 128 elem + fp32 scale 4 B = **132 B/token**。

| | 16B（6 DSA 层） | 大（11 DSA 层） |
|---|---:|---:|
| per token 全部 DSA 层 | 6·132 = **792 B** | 11·132 = **1,452 B** |
| @ 512 K context | ≈ **396 MB** | ≈ **708 MB** |

### 2.3 DSA latent K cache —— 线性正比 S（无 V、无 rope）

每 token 每层 = Rkv = 512 elem。

| dtype | per token per 层 | 16B（6 DSA 层）@ 512K | 大（11 DSA 层）@ 512K |
|---|---:|---:|---:|
| bf16 | 1,024 B | ≈ **3.00 GB** | ≈ **5.50 GB** |
| fp8 (+128-block scale) | ≈ 520 B | ≈ **1.53 GB** | ≈ **2.80 GB** |

---

## 3. 容量预算：12 req × 512 K，分摊到 16 device

> "12 req / 16 device" 取用户给的中位负载（区间 8–12）；"8 req"按比例缩 2/3。

### 3.1 KDA（TP=16，所有 device 必参与每个 req）

每个 device 持 `H/16` 头：16B 32/16 = **2 头** / device、大 64/16 = **4 头** / device；进一步 2 头 / core（16B：1 头/core；大：2 头/core）。

| | per-req per-device | × 12 req | × 12 req per-core |
|---|---:|---:|---:|
| 16B（bf16） | 7.65 MB / 16 ≈ **478 KB** | ≈ **5.6 MB** | ≈ **2.8 MB** |
| 大（bf16） | 72.8 MB / 16 ≈ **4.55 MB** | ≈ **54.6 MB** | ≈ **27.3 MB** |
| 大（recurrent fp32） | ≈ **8.8 MB** | ≈ **106 MB** | ≈ **53 MB** |

**结论**：放 Gmem（1.8 GB）富余 ≥ 15×。Lmem 当然也够；但**KV-transfer 写入是 12 条小 buffer × 21~34 层 × 16 device，频次极高**——放 Gmem 让 host 写入与 TC 完全脱耦。**Gmem 是首选**。

### 3.2 DSA indexer K（无并行 / CP / DP 三种情况）

| 方案 | per-device index K（12 req @ 512K） | Lmem 占比 | Gmem 占比 |
|---|---:|---:|---:|
| 复制（每 device 全量） | 16B 6·512K·132·12 = **4.75 GB** ；大 ≈ **8.5 GB** | 16%/30% | **放不下** |
| **CP=16 分片** | 16B ≈ **297 MB** ；大 ≈ **531 MB** | 1%/2% | 16B 勉强 / 大 放不下 |
| DP（每 device 0.75 req） | 16B ≈ **297 MB** ；大 ≈ **531 MB** | 同上 | 同上 |

**结论**：复制方案大模型放不下 Gmem，且即便 Lmem 也吃 8.5 GB，浪费；**index K 分片（CP）或随 req 分配（DP），统一放 Lmem**。理由：
1. indexer 在 decode 每步、每 DSA 层都要做一次 `q_idx · index_K` 的全量扫描（生成 logits 再 top-2048），是 **bandwidth-bound on `index_K` read** → 必须近 TC。
2. 容量 0.3–0.5 GB / device 在 Lmem 14 GB/core 里是噪声。
3. 没有"先 gather 再算"的中间状态可以拆出来——直接进 GEMM。

### 3.3 DSA latent K（容量直接决定放哪儿）

| 方案 | per-device latent K（12 req @ 512K，bf16） | 大模型 Gmem (1.8 GB) | 大模型 Lmem (28.75 GB) |
|---|---:|---|---|
| 全复制 | 16B 36 GB / 大 **66 GB** | ❌ | ❌ |
| **CP=16 分片** | 16B **2.25 GB** / 大 **4.13 GB** | ❌（大）/ ❌（16B 单 req 已 188 MB×12=2.25 GB 也超） | ✅ 富余 6–7× |
| CP=8 + DP=2 | 同上量级 | ❌ | ✅ |
| DP（≤1 req/device） | 16B 3.0 GB / 大 **5.5 GB** | ❌ | ✅ |

**结论**：**latent K 必须放 Lmem**。这不是性能选择，是容量硬约束（Gmem 1.8 GB × 16 = 28.8 GB，连大模型 12 req latent 66 GB 的一半都装不下）。
fp8 latent 可把 4.13 GB → 2.07 GB，仍超 Gmem。

---

## 4. 决策矩阵（汇总）

| State | per-device 量级 | 频次特征 | 选位置 | 决定理由 |
|---|---|---|---|---|
| KDA recurrent + conv | < 100 MB / device | 写入很多次（高频小写） | **Gmem** | TC-friendly：host 写 Gmem 与 TC 独立；容量充足 |
| DSA indexer K | ~0.5 GB / device | 全量扫描 read（GEMM 输入） | **Lmem** | read BW 决定性；容量小 → Lmem 占比 < 2% |
| DSA latent K | ~4 GB / device | 2048-gather + 6/11 层 × decode step | **Lmem**（必然） | 容量超 Gmem 数十倍；gather 路径见 §5 |

---

## 5. DSA latent K 的 gather 路径分析（待定）

decode 每层、每 step、每 req 都要：

```
top-2048 indices (来自 indexer) → gather 出 2048 个 latent 行（[2048, Rkv]）→ 进入 sparse MQA
```

每次 gather 的搬运量（per req per 层 per step）：

- bf16 latent: `2048 · 512 · 2B = 2 MB`
- fp8 latent: `2048 · 512 · 1B + scale ≈ 1.05 MB`

12 req × 11 层（大）= **132 次 gather / step**，总搬运 ≈ 264 MB / step（bf16）。

### 路径 A：`Gmem → shmem → Lmem`（gather 源在 Gmem）

> 注：**latent K 整体仍在 Lmem**（容量不允许放 Gmem）。"Gmem 源"路径意味着维护一份 **mirror / 影子副本** 在 Gmem 里，或者把 latent K **双写**（host 写时同时写 Gmem 与 Lmem）。需要权衡这份 mirror 的开销。

- **优点**：gather 从 Gmem 出发，**与 TC 的 Lmem read 不竞争 BW**，可与本层其他 GEMM 时间重叠。
- **代价**：
  - Gmem 容量根本不够装 mirror（≥ 4 GB / device）。**A 在此场景实质不可行**，除非只 mirror "热区"（很难界定）。
  - 双写消耗 host BW 与 Gmem 容量。
- **结论**：除非未来 Gmem 扩容或只 mirror 部分 token（如最近 N K），A **基本排除**。

### 路径 B：`Lmem → shmem → Lmem`（gather 源在 Lmem）

- **优点**：单一存储，零 mirror，省容量；KV-transfer 写一次到位。
- **代价**：gather 的 Lmem read 与 TC 算 GEMM 抢 Lmem BW，**需要在时间维度上错开/流水**：
  - 若 sparse MQA kernel 自带 prefetch，gather 在前一个 tile 算的时候就开始读，可以重叠掉大部分。
  - 极端串行情况下，264 MB / step 的 Lmem 读会"挤"TC，但 Lmem BW 通常足够高，**估计是可接受**。

### 推荐与待 benchmark 项

- **缺省主线：路径 B（Lmem→shmem→Lmem）**，配合 kernel 内 double-buffer / pipeline。
- 待 benchmark：
  1. **B 的 TC 实际利用率**（带 gather vs 不带 gather 的 GEMM throughput 比）。
  2. 若 B 损失 > X%（阈值待定），考虑 "**部分 mirror**"：把每层最后 K 个 token 的 latent 双写 Gmem，gather 时优先从 Gmem 取（命中率取决于 indexer 的局部性，可统计）。
  3. fp8 latent 下 gather 量减半，B 的压力进一步下降——**fp8 latent + B** 可能直接落地。

---

## 6. DSA 并行方案分析

> 前提：**KDA 已固定 TP=16**（每 device 2~4 头），所以 16 个 device 是一个紧密耦合的推理单元，不能拆成"独立服务器"。DSA 的并行轴在这个 16-pack 内部组合。
>
> 三个候选轴：
> - **TP**（head 维度切 q_b/kv_b/o_proj 等权重）—— *不切 latent K cache*（latent 在 absorb 下是单 head）。
> - **CP**（token 位置维度切 latent K + index K）—— V6 Route B，**唯一能切 KV cache 的轴**。
> - **DP**（不同 request 分到不同 device 子组）—— 每子组复制 KV cache，但只服务部分 req。

### 6.1 选项矩阵

| 选项 | 配置 | latent K 分布 | index K 分布 | per-device latent (大, bf16, 12 req @ 512K) | 跨卡通信 / decode step / DSA 层 | 单 req 延迟 | 备注 |
|---|---|---|---|---:|---|---|---|
| **O1 · 纯 TP=16** | TP=16 only | **复制 16 份** | 复制或分片 | **66 GB**（**不可行**） | 0 | 最低 | latent 不能分→Lmem 装不下 |
| **O2 · 纯 CP=16**（V6 Route B） | CP=16, 无 TP / DP | **token round-robin 16-shard** | 同 token shard | **4.13 GB** | top-k all-gather ≈ 0.26 MB + LSE merge（ring）≈ 0.07 MB ≈ **~0.33 MB/层** | 最低（16× 并行算 partial） | 主线候选；与 KDA 的 TP=16 **完全正交** |
| **O3 · TP=16 + CP=16 共用 16 卡** | head 切 + token 切，**沿同 16 device 双切** | 16-shard | 16-shard | **4.13 GB** | 同 O2 | 同 O2，每 device 多省 head 维 GEMM | head/device 减到 2（16B）/4（大）；projection 显存 ÷16 |
| **O4 · CP=4 × DP=4** | 4-token-shard × 4-req-group | per group 4-shard | 同 | (12/4) × 5.5 / 4 = **4.13 GB** | 仅 4-device merge：top-k ~0.06 MB + merge ~0.13 MB ≈ **~0.19 MB/层（4-范围）** | 4× 并行 | merge scope 小→ 通信延迟更低；req 间隔离更好（cache friendly） |
| **O5 · 纯 DP=12（≤1 req/device，4 idle）** | 12 device 各一 req | 1 req full | 同 | 5.5 GB（单 req） | 0 | 单 device 速度（最慢） | 4 卡完全空转；不推荐 |

> 显存数字按"大模型 + bf16 latent"算；fp8 latent ×0.5。"index K" 跟随 latent 同分布（CP 下分片，DP 下随 req）。

### 6.2 indexer 计算的划分

| 选项 | indexer Q 计算 | indexer K 计算 | logits / top-k |
|---|---|---|---|
| O1 TP=16 | 每 device 算 `I/16` 头 Q (= 0.5 头) | K 复制 → 每 device 算全 seqlen × 0.5 头 logits | 不需要分布式 top-k；但 KV cache 复制不可行→O1 实质废 |
| **O2/O3 CP=16** | 复制 Q（O2）/ 切 head（O3，单 device 几乎 0 头不可行） → **现实是 O2 复制 indexer Q** | K shard → 各 device 算自己 shard 的 logits | **分布式 top-k**：local top-2048 → all-gather (pos, logit) ≈ 256 KB/层 → global top-2048（见 V6 §8） |
| O4 CP=4×DP=4 | 同 O2 但范围 4 | K 4-shard | 4-device 分布式 top-k ≈ 64 KB/层 |
| O5 DP=12 | 每 device 全本地 | 全本地 | 本地 top-2048，零通信 |

> **indexer Q 的 head 维度（I=8）很小**，沿 head 切的 TP 在 16 device 上不可行（< 1 头/device），所以 indexer Q **现实只能复制**，不依赖 TP 切。 

### 6.3 主干 sparse MQA 的划分

sparse MQA 的 query 头数 Nh = 32（16B）/ 64（大），key 单头（MQA-absorb）。

| 选项 | 每 device 算的形状 | 输出处理 |
|---|---|---|
| **O2 CP=16** | `q_new [Nh, 576]` 复制；K 取本地 shard 的 2048-子集；输出 `partial_out [Nh, 512] + lse [Nh]` | online-softmax merge（V6 §10），16-device all-reduce / ring |
| **O3 CP=16+TP=16** | `q_new [Nh/16, 576]` 切；K 取本地 shard 的 2048-子集；输出 `partial_out [Nh/16, 512] + lse [Nh/16]` | merge 仍沿 CP=16（不沿 TP）；最后 o_proj 沿 TP 做 RowParallel all-reduce |
| O4 CP=4×DP=4 | 同 O2 但范围 4 | 4-device merge |
| O5 DP=12 | 全本地 sparse MQA | 无 merge |

### 6.4 KV-transfer（PD 分离）影响

decode worker 接收来自 prefill 的 KV，layout 必须匹配自己的 cache 分布：

| 选项 | KV-transfer pattern | 复杂度 |
|---|---|---|
| O2/O3 CP=16 | prefill 端按 round-robin 把 token KV 散给 16 个 decode device（每 device ~ S/16 行） | 16-way scatter，每 device 写量均衡 ≈ 256 MB latent + 30 MB index（大，512K） |
| O4 CP=4×DP=4 | prefill 端按 req 分 4 组、组内 4-way scatter | 同样均衡，但 transfer 拓扑分 4 组，更适合 NIC 拓扑分簇 |
| O5 DP=12 | prefill 端把每个 req 的全部 KV 一次性给一个 decode device | 单点写 5.5 GB / req，写量集中、不均衡；但 transfer 协议简单 |

→ **CP 类方案的 transfer 量更均衡**，是 Lmem 写入与 TC 的时序错开更容易做（每 device 写量小、可分批），呼应"Lmem 写竞争 TC"的硬件约束。

### 6.5 主线推荐

- **首选 O2（纯 CP=16，V6 Route B）**：
  1. latent K 容量分摊 → Lmem 装得下；
  2. KV-transfer 写量均衡，每 device ~ 256 MB 量级，host→Lmem 与 decode TC 容易错开；
  3. 通信成本 < 1 MB/层/step，可忽略；
  4. 单 req 延迟最低；
  5. 与 KDA 的 TP=16 **完全正交**（KDA TP 切 head，DSA CP 切 token），两层切法不冲突。
- **次选 O3（CP=16 + TP=16 复用同 16 device）**：在 O2 上再切 projection 权重，省 projection 显存 / FLOPs（每 device 2~4 q-head），merge 范围不变。**TP 与 CP 复用同一组 16 device 是合法的**——只要分清"沿哪个轴 reduce"。
- **备选 O4（CP=4×DP=4）**：merge scope 缩到 4-device → 通信延迟更小；适合 NIC 分簇拓扑。但 12 req / 4 组 = 3 req/组，组间负载有±1 波动。
- **不推荐 O1 / O5**。

---

## 7. 容量总账（推荐方案 O2 / O3，大模型 / 12 req / 512 K / bf16 latent）

per-device：

| 类别 | 位置 | 容量 | 占比（vs 物理容量） |
|---|---|---:|---:|
| KDA state（TP=16） | Gmem | ≈ 55 MB（bf16）/ ≈ 106 MB（recurrent fp32） | 3% / 6% of 1.8 GB |
| DSA index K（CP=16 分片） | Lmem | ≈ 0.53 GB | 1.8% of 28.75 GB |
| DSA latent K（CP=16 分片，bf16） | Lmem | ≈ 4.13 GB | 14% of 28.75 GB |
| **小计** | — | **Gmem ≈ 55–106 MB ; Lmem ≈ 4.66 GB** | — |
| 同上 fp8 latent | Lmem | ≈ 2.07 + 0.53 = 2.60 GB | 9% of 28.75 GB |

per-core（每 device 2 core，简单对半分）：

| 类别 | per-core | 占比（vs 14.38 GB） |
|---|---:|---:|
| DSA index K | ≈ 265 MB | 1.8% |
| DSA latent K（bf16） | ≈ 2.07 GB | 14% |
| KDA state（如果改放 Lmem） | ≈ 30 MB | 0.2% |

→ Lmem 还有大量余量，可以容纳：activation scratch、shmem 溢出、模型权重切片（如果走 O3 的 TP 切权重）。

---

## 8. 开放问题 / 待 benchmark

1. **latent K gather 路径（§5）**：B（Lmem 源）是否真的会被 TC 干扰显著？需要实测带 gather 的 sparse MQA kernel vs 纯算的吞吐比。若损失 > 阈值，考虑"部分 mirror"或 fp8 latent。
2. **KDA state dtype**：recurrent state 是否必须 fp32？若 bf16 精度够，Gmem 与 Lmem 都更宽松；fp32 也仍在预算内。
3. **fp8 latent K 落地**：量化/反量化 kernel 是否已就绪（参 V6 §14 `R-QUANT`）；与 sparse MQA fp8 kernel 的衔接。
4. **O3（CP=16+TP=16）是否真比 O2 有收益**：projection 权重在 16 device 复制（O2）的显存代价是 q_b/kv_b/o_proj 三个 GEMM 的权重 ≈ 几十 MB——也许不切就行。
5. **prefill 端 layout 协议**：CP=16 round-robin 是 V6 §1 假设，需要确认 prefill 真按这个 split 写出 KV；否则 decode 端需要在 KV-transfer 时做 re-layout。
6. **KDA 在 PD 分离下的 transfer**：state 大小固定但有 21~34 层 × 16 device 个独立 buffer。每个 buffer 几百 KB～几 MB，**写次数远多于 DSA**——验证 Gmem 写通道是否扛得住峰值（如 12 req 同时 handoff）。
7. **8 req vs 12 req 的下限校准**：8 req 时 Lmem 占用 = 12 req 的 2/3，余量更大，但 batch dim 缩小可能让某些 kernel 触发低占用退化；需测 sparse MQA 在 B=8 vs B=12 的 throughput。
8. **混合 batch（部分 req 接近 1M）**：上下文上限拉到 1M 时，单 req latent K（大）= 11 GB，CP=16 后 = 0.69 GB/device，仍合理；混合长短时按 max-context 估上限即可。

---

## 9. 速查对照（一页本）

```
=== 硬件 ===
Gmem  : 1.8 GB/device  (host I/O 独立, 高效, 与 TC 不冲突)
Lmem  : 28.75 GB/device (14.38 GB/core, 近 TC, host 写竞争 TC)

=== 模型 (大 GLM-5, 11 DSA 层, 34 KDA 层) ===
per-req KDA state        : 73 MB bf16 / 141 MB fp32+bf16
per-req per-token latent : 11·1024 B = 11.0 KB   (bf16, 6 个 16B 模型则 6.0 KB)
per-req per-token index  : 11·132 B  = 1.42 KB
per-req @ 512K latent    : 5.5 GB bf16 / 2.8 GB fp8
per-req @ 512K index     : 708 MB

=== 12 req / 16 device, CP=16 ===
per-device KDA (TP=16)        : 4.55 MB × 12 = 54.6 MB  → Gmem ✓
per-device latent (bf16)      : 5.5 GB × 12 / 16 = 4.13 GB → Lmem ✓
per-device latent (fp8)       : 2.07 GB               → Lmem ✓
per-device index (CP shard)   : 708 MB × 12 / 16 = 531 MB → Lmem ✓
per-device 合计 Lmem          : ≈ 4.66 GB (bf16) / 2.60 GB (fp8)

=== 通信 (CP=16, 大模型, per step, 全 11 层累计) ===
top-k 分布式 all-gather  : 11 · 16 · 12 · 2048 · 8 B  ≈ 33 MB / step
LSE merge (ring)         : 11 · 2 · 12 · 64 · 513 · 2 B ≈ 17 MB / step
→ 加起来 ~50 MB / step, 在 NVLink/CXL 量级是 μs 级
```

---

## 10. 与现有文档的关系

- V5 单卡 decode 链路：把 §6.5 的并行换成 cp=1，本文与 V5 等价。
- V6 Route B：本文的 §6.5 主线（O2）= V6 全文，只是补了硬件视角下的内存放置约束。
- V7 fused-stage：V7 的 stage 划分（D0/D1/D2/D3/D3.5/D4/D4.5/D5/D6）正交于本文的内存放置；本文不改 stage，只决定每个 stage 输入/输出住哪儿。
