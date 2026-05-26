# GLM5-Next Decode · 16-Device KV 内存放置 + indexer / top-K 并行（v1_b · 8 req）

> 在 [`KvPlacement_Decode_16dev_v1.md`](KvPlacement_Decode_16dev_v1.md) 基础上：
> - **负载切到 8 req @ 512 K**（V1 用的是 12 req）；
> - 加入 **indexer K "每 device 重复" vs "分片"** 的正面对比；
> - 加入 **index GEMM / top-K / gather** 的并行展开，结合用户提供的 **vector-core 硬件参数**做工作量估算；
> - 内存放置主结论与 V1 不变（KDA→Gmem, index K→Lmem, latent K→Lmem）。
>
> 模型仍以 `config_16b_v2.json`（6 DSA / 21 KDA 层，Nh=32, I=8）和 `config.json`（11 DSA / 34 KDA 层，Nh=64, I=8）为准；`qk_rope_head_dim = 0` ⇒ latent per token = **Rkv = 512 elem**。

---

## 0. 与 V1 的差异

| 维度 | V1 (12 req) | **V1_b (8 req)** |
|---|---|---|
| per-device Lmem（大模型 + CP=16 + bf16 latent） | ≈ 4.66 GB | **≈ 3.10 GB** |
| index K 放 Lmem 复制 vs 分片 | 主线分片 | **复制方案变得可行**（容量从 8.5 GB → 5.8 GB / device，Lmem 仍 28 GB，占比 20% → 可接受） |
| 通信压力 | 较高 | **较低**：candidate 集合 ÷ 1.5，merge buffer ÷ 1.5 |
| 主要新增分析 | — | indexer GEMM + top-K + gather 的 vector-core 工作量预算 |

> 8 req 比 12 req 的真正质变是：**index K 复制方案重新进入候选**（不再因为容量爆 Lmem 被排除），需要拿"零通信 vs 16× 重复计算"重新权衡。

---

## 1. 硬件参数（含 vector core）

| 维度 | 单位 |
|---|---|
| Device 数 | 16 |
| Core / device | 2 |
| Vector-core / core | **4** |
| Vec-core SIMD lane（BF16 / FP8） | **64** lane/cycle |
| Vec-core SIMD lane（FP32） | **32** lane/cycle |
| Gmem | 1.8 GB / device，host I/O 独立，**不与 TC 抢资源** |
| Lmem | 29,440 MB ≈ 28.75 GB / device（14.38 GB/core），近 TC，host 写 / vec 读会与 TC 抢 BW |
| 对齐 | 16 KB |

**vector-core 聚合算力**（每 core，所有 4 个 vec core 全开）：

| dtype | lanes / core | lanes / device (×2 core) | lanes / 16-device 总 |
|---|---:|---:|---:|
| BF16 / FP8 (8-bit) | 4·64 = **256** | **512** | **8,192** |
| FP32 | 4·32 = **128** | **256** | **4,096** |

> 下文按"1 个 op = 1 个 lane·cycle"折算。具体频率 / op 类型 / latency hiding 待 benchmark；本文用作 **相对比较**，不是绝对时间表。

---

## 2. 单 request 各类 state（与 V1 §2 一致）

| 类别 | 每 token / 每层 | 公式 | 16B | 大 |
|---|---|---|---:|---:|
| KDA recurrent + conv (per layer, bf16) | — | `(H·Kd·Vd + 3·H·Kd·3)·2 B` | 373 KB | 2.14 MB |
| KDA state per request (全部 KDA 层) | — | × 层数 | **7.65 MB** | **72.8 MB** |
| DSA latent K (Rkv=512, bf16) | 1,024 B | `512 · 2 B` | — | — |
| DSA latent K (fp8 + scale) | ≈ 520 B | — | — | — |
| DSA index K (FP8 + scale) | **132 B** | `128 + 4` | — | — |

@ 512 K context per request（单 req）：

| | latent (bf16) | latent (fp8) | index |
|---|---:|---:|---:|
| 16B (6 层) | 3.00 GB | 1.53 GB | 396 MB |
| 大 (11 层) | 5.50 GB | 2.80 GB | 708 MB |

---

## 3. 8 req / 16 device 容量重算

> 假设 CP=16 分片，KDA 用 TP=16。**所有数字 per device**。

### 3.1 KDA

| | per-req per-device | × 8 req | 占 Gmem (1.8 GB) |
|---|---:|---:|---:|
| 16B bf16 | 478 KB | **3.7 MB** | 0.2% |
| 大 bf16 | 4.55 MB | **36 MB** | 2.0% |
| 大 (recurrent fp32) | 8.83 MB | **70 MB** | 3.9% |

→ 放 Gmem，毫无压力；与 V1 同结论。

### 3.2 index K（4 种放置 × 2 种 dtype）

| 方案 | per-device 容量 (16B) | per-device 容量 (大) | Gmem | Lmem |
|---|---:|---:|---|---|
| **A · 复制 + 全量复制 (replicated K, replicated compute)** | 6·512K·132·8 = **3.17 GB** | 11·512K·132·8 = **5.81 GB** | ❌ 超容 | ✅ 占 20% (大) |
| **B · 复制 K + 切计算 (replicated K, sharded compute)** | 同 A 3.17 GB | 同 A 5.81 GB | ❌ | ✅ 同 A，但 GEMM/top-K 切到 1/16 |
| **C · 分片 K + 分片计算 (CP=16, V6 主线 V6 §8 变体 b)** | 3.17/16 ≈ **198 MB** | 5.81/16 ≈ **363 MB** | 16B ❌ / 大 ❌ (单段也超) | ✅ 占 1.3% (大) |
| D · 全分布 (按 req 切 DP) | 同 C 量级 | 同 | — | — |

→ **8 req 下 A/B 都进入 Lmem 容量预算**（占 20% 是大的，但 28 GB Lmem 装得下），可以正面比较 §4。

### 3.3 latent K

| 方案 | per-device (16B, bf16) | per-device (大, bf16) | per-device (大, fp8) |
|---|---:|---:|---:|
| 全复制 | 24 GB ❌ | 44 GB ❌ | 22 GB（也超 Lmem） |
| **CP=16** | 1.50 GB ✅ | **2.75 GB** ✅ | **1.38 GB** ✅ |

→ latent K 仍**必然 CP=16 分片** + Lmem，与 V1 同结论。

### 3.4 per-device Lmem 合计（推荐路线：latent CP=16，index 见 §4）

> 下表 "合计 Lmem" 均为 **per-device**（= 2 core 的 Lmem 总和 = 14,720 MB × 2 = **29,440 MB ≈ 28.75 GB**），不是单 core。

| 路线 | latent (大, bf16) | index | KDA 在 Lmem？ | per-device 合计 Lmem | per-device 剩余余量 (28.75 GB − 合计) |
|---|---:|---:|---:|---:|---:|
| 主线 V1_b（latent CP, **index 分片**） | 2.75 GB | 363 MB | 不 (KDA 在 Gmem) | **~3.10 GB** | **~25.65 GB** |
| 替代 V1_b' (latent CP, **index 复制**) | 2.75 GB | 5.81 GB | 不 | **~8.56 GB** | **~20.19 GB** |

→ 两种都装得下；**两条路线之间的差额是 5.46 GB**——即"index 复制方案的额外存储代价"。剩余余量（25.6 / 20.2 GB）才是可以给 weight 切片、activation scratch、shmem 溢出、KV-transfer 双缓冲等用途的空间。
> per-core 视角：若 latent K + index K 平均铺到 2 个 core 的 Lmem 上，主线 per-core ≈ 1.55 GB（占 14.38 GB Lmem 的 10.8%），替代路线 per-core ≈ 4.28 GB（占 29.8%）；§11 给出更详细的 per-core 分解。

---

## 4. **index K 放置详对比**：复制 (A) vs 分片 (C)

> 这是 V1_b 的核心新讨论。8 req 让"复制"重新成为现实选项，但**实测能否赢看通信代价与冗余 BW**。

### 4.1 一次 decode step 的 index 阶段（大模型，11 个 DSA 层，8 req）

| 资源 | A · 复制 K + 复制 compute | C · 分片 K + 分片 compute (V6 主线) |
|---|---|---|
| 每 device index K 容量 | 5.81 GB | **0.36 GB** |
| 每 device 每层 index GEMM 读量（FP8） | 8·512K·128 = **512 MB** | 8·32K·128 = **32 MB** |
| 全 11 层每 step 每 device 的 K 读量 | **5.63 GB** | **0.35 GB** |
| 每 device 每层 GEMM ops（FP8） | 8·8·128·512K·2 ≈ **8.6 G** | **0.54 G** |
| 全 11 层每 step 每 device 的 GEMM ops | **94 G** | **5.9 G** |
| 每 device 每层 top-K 候选规模 | 全 512K | 32K |
| top-K 跨卡通信（11 层全部） | **0** | candidate all-gather 11·16·8·2048·8 B ≈ **22 MB** |
| KV-transfer（PD 写） | prefill 端要把同一份 index K **scatter 到 16 device 16 次**（写 5.81 GB × 16 ≈ 93 GB 总 BW） | prefill 端 round-robin scatter，每 device 只写 363 MB（5.81 GB 总 BW，**省 16×**） |

### 4.2 资源占用对比表（直观）

```
                     A (replicated)        C (CP=16 shard)        ratio (A/C)
index K Lmem         5.81  GB              0.36  GB               16×
GEMM read BW         5.63  GB/step         0.35  GB/step          16×
GEMM compute         94    G ops/step      5.9   G ops/step       16×
top-K work (vec)     ~ 45  M ops/step      ~ 3   M ops/step       15×
inter-device comm    0                     ~ 22  MB/step          (A 零)
PD write total BW    93    GB/req-batch    5.81  GB/req-batch     16×
single-req latency   稍快 (无 merge wait)  几 µs 慢 (加 all-gather)
```

### 4.3 决策

- **A（复制）在 8 req 下不爆容**，但其他每一项几乎都是 C 的 **16×**——它把 16 个 device 当作 16 份独立的复制，浪费了硬件并行性。**唯一收益是节省 ~22 MB/step 的 all-gather 通信**，但这点通信在 NVLink/CXL 级互连上是 µs 级，相对 94 G ops 的冗余完全划不来。
- **建议主线：C · 分片**（V6 §8 变体 b 的"分布式 top-k"）。
- **保留 A 的场景**：互连真的很弱（< 50 GB/s）且 prefill 端能负担 16× scatter 的写 BW。本系统假设不属于此情况。
- 中间方案 **B（K 复制 + compute 分片）**：每 device 只算 1/16 的 GEMM、做 1/16 的 local top-K、再 all-gather merge——**与 C 在通信和算力上等价，但白白多存 16× index K**，没意义，淘汰。

> **小结：8 req 让复制方案"装得下"，但装得下不等于划算。主线仍是 CP=16 分片。下文 §5–§8 的工作量都按 C 计算。**

---

## 5. indexer GEMM 的并行展开（CP=16，主线）

### 5.1 计算分解（每 device，per DSA 层，per step）

每 device 持有 `n_local = seqlen/16 = 32 K` 个 index K 行（FP8 [128] + scale）。query 侧不分片（`q_idx [B, I=8, Di=128]` FP8 在每 device 上都有）：

```
logits_local [B=8, I=8, n_local=32K]
  = q_idx_fp8 @ index_K_local_fp8^T            (FP8 GEMM, K=128)
weighted_local [B=8, n_local]
  = Σ_i ( gate[B,i,1] * logits_local[:,i,:] )  (vec-core, BF16/FP32)
```

### 5.2 GEMM 形状与硬件适配

| 维度 | 值 | 硬件适配 |
|---|---|---|
| M（query 数） | B·I = 8·8 = **64** | 与 BF16/FP8 lane 数（per core 256）等同 → 一次 issue 1–4 个 M-tile，单 core 容易吃满 |
| K（内积长度） | Di = **128** | 短，FP8 GEMM 的 K-tile 通常 32/64/128 → 1–4 个 K-tile |
| N（key 数 per device） | n_local = **32K** | 长边，沿 N 切片到所有 vec core / pipeline depth |
| dtype | FP8 a/b, BF16/FP32 accum | accum FP32 → vec/fp32 lane（32/vc, 128/core） |

GEMM 应由 **tensor core / MMA 单元** 完成，vec core 只做：
- gate 缩放（`weighted = Σ gate * logits`）
- act_quant / dequant
- 后续 top-K（§6）

→ 这一步**主要 BW-bound 在 index K 读上**（K=128 太短，算密度 = `2·I·Di / Di = 16 FLOPs/byte`，FP8 算力远高于 Lmem BW，所以读 K 是瓶颈）。

### 5.3 工作量估算（每 device 每 step，11 层 / 大模型）

| 资源 | 数值 |
|---|---|
| index K 读 | **0.35 GB / step**（已含 8 req × 11 层） |
| GEMM FP8 ops | **5.9 G ops / step**（accum FP32：5.9 G adds + 5.9 G muls） |
| 后处理（gate 加权求和） | per-layer `B · n_local = 8·32K = 256K` BF16 muladds → 11 层 ≈ **2.8 M ops** → 在 256-lane BF16 vec 下 ≈ 11K cycles，**很小** |

→ **结论**：indexer GEMM 阶段 vec core 几乎空闲（gate 加权那点工作几乎不计入），瓶颈是 Lmem→TC 的 K 读。这正合"index K 放 Lmem"的设计意图。

---

## 6. top-K 并行（vector-core 主战场）

> 这是 vector core 真正干活的地方。下文按 **B=8 req、I=8 head（已加权后只剩 [B, n_local]）、K=2048** 估算。

### 6.1 算法选择

候选有三类：

| 算法 | 优势 | 劣势 |
|---|---|---|
| **Bitonic top-K (sorting-based)** | 完全数据无关、确定性、容易在 SIMD 上展开 | O(N · log² K) 工作量；对 N >> K 不最优 |
| **Radix top-K** | 渐近最优 O(N) (按位分桶) | 控制流较复杂；FP32 logit 取整需要 careful |
| **Tournament + heap（每 lane 一个本地 heap）** | O(N·log K) 总工作；对 SIMD 友好 | 需要每 lane 私有 K 容量；K=2048 在 lane 内放不下 |

**推荐主线：分层 tournament**：
1. 把 32K 候选切成 P 个 chunk（P = vec lane 数 / lane 内 K 大小）
2. 每 lane 维护一个长度 16 的 mini-heap，扫完 chunk 输出 top-16
3. 跨 lane 合并到 256 → 跨 vec core 合并到 1024 → 跨 core 合并到 2048

### 6.2 工作量估算（CP=16，每 device 每层）

设每 lane 每 cycle 做 1 次"compare + conditional update"（≈ 4 vec ops 估）：

```
per device per layer:  candidates = 8 (req) × 32K = 256K
扫描成本              ≈ 256K · 4 ops = 1.0 M vec ops (FP32)
合并 (log P 阶段)     ≈ 0.2 M vec ops
小计 per layer        ≈ 1.2 M FP32 vec ops
× 11 层 (大)          ≈ 13 M FP32 vec ops / step / device
```

在 **256 FP32 lane / device** 下 (= 4 vc/core × 32 lane × 2 core)：

```
13 M ops / 256 lanes ≈ 51K cycles (理想)
@ 1 GHz 假设       ≈ 51 µs / step / device
```

> 50 µs 量级是松弛上限——实际 vec core 还有 latency hiding、shmem 溢出等开销，但这数量级支持"top-K 不是瓶颈"的判断。

### 6.3 全局 top-K（分布式合并）

```
本地 top-2048 输出 (per device per layer per req)
   → all-gather (pos:int32, logit:fp32) 8 B/entry
   → 候选总数 16 · 2048 = 32K (per req per layer)
   → 全局 top-2048 (在每 device 各算一份，结果一致)
```

| 资源 | 大模型 / step |
|---|---|
| all-gather BW | 11 · 16 · 8 · 2048 · 8 B = **22 MB** |
| 全局 top-K vec ops | 32K 候选 → 2048 top，再来一轮分层 tournament ≈ **0.2 M ops/层 × 11 ≈ 2.2 M ops / step** |

→ 全局 merge 比本地扫描便宜得多（候选少 8×），vec core 完全 hold 得住。

### 6.4 与 indexer GEMM 的流水

```
┌── per layer ──────────────────────────────────────────────┐
│  TC: index GEMM (BW-bound on K read, ~32 MB Lmem read)    │
│  ─────────────────────────────────────────────────────────│
│   vec: gate-weighted sum (cheap, ~0.25M ops)              │
│   vec: local top-K scan (~1.2M ops, ~5 µs)                │
│   ─────── all-gather candidates (256 KB) ──────────────── │
│   vec: global top-K merge (~0.2M ops, ~1 µs)              │
│   vec: page-table 变换 (slot mapping, 16K addr ops)       │
└────────────────────────────────────────────────────────────┘
```

GEMM 与本地 top-K 可流水：GEMM 输出第 i 个 N-tile 的时候，vec 已经在 reduce 第 i-1 个 tile。这是 fused indexer kernel 的关键。

---

## 7. latent K gather 路径（vec-core 视角）

> 内容承接 V1 §5，补 vec-core 工作量。

### 7.1 gather 工作量

每 device 每层每 step：
- top-2048 indices（global） → 取 `topk_slots_local`（不归本 device 的填 -1，可能 < 2048 个有效）
- 实际 gather 行数（per req）≈ 2048 / 16 = **128 行**（均匀分布的期望）
- gather 每行 = Rkv = 512 elem = 1 KB (bf16) / 0.5 KB (fp8)

| 项 | 数值 (大, bf16, 8 req) |
|---|---|
| 有效 gather 行数 per layer | 8·128 = 1,024 行 |
| gather 数据量 per layer | 1,024 · 1 KB = **1 MB** |
| 全 11 层 per step | **11 MB** |

### 7.2 vec-core 工作量（gather 地址计算）

每 gather 需要：page-table 查找（页号→物理槽位）+ 地址计算。粗算每行 5 ops：

```
1024 行/层 × 5 ops × 11 层 ≈ 56K addr ops / step
```

→ 几乎免费（在 256-lane BF16 vec 下 < 1K cycles），**gather 的瓶颈是数据搬运 BW，不是 vec 计算**。

### 7.3 主线推荐

主线仍是 **`Lmem → shmem → Lmem`（路径 B）**，理由同 V1 §5：
- A 路径 (Gmem 源) 需要在 Gmem 维护 mirror，4 GB 远超 Gmem 容量；
- B 路径每 step 11 MB Lmem read，相对 indexer GEMM 的 0.35 GB Lmem read 是噪声（< 5%）；
- vec core 的 gather 地址计算开销可忽略。

---

## 8. LSE merge（Route B 收尾，vec-core 工作）

> V6 §10 的 online-softmax merge。

### 8.1 工作量

每 device 每层 per req：
- 输入：`partial_out [Nh, Rkv=512]` + `partial_lse [Nh]`（大模型 Nh=64）
- merge ops: max + exp + add + mul，每 element ~6 ops（fp32）

```
per device per layer:  8 (req) · 64 (Nh) · 513 ≈ 263K · 6 = 1.6 M FP32 vec ops
× 11 层:                ≈ 17 M / step
```

→ 在 256 FP32 lane / device 下 ≈ 67K cycles，**~67 µs 量级**，跟 top-K 同数量级。

### 8.2 通信

ring-allreduce 版（V6 §11）：

```
全 11 层 per step: 2 · 8 · 64 · 513 · 2 B ≈ 1.05 MB total
```

可忽略。

---

## 9. 每 device 每 step 工作量预算（大模型，8 req @ 512 K，CP=16）

| 阶段 | TC (Lmem read) | TC FLOPs | vec ops | 跨卡通信 |
|---|---:|---:|---:|---:|
| indexer GEMM (§5) | 0.35 GB | 5.9 G FP8 | 2.8 M (gate) | 0 |
| 本地 top-K (§6) | (复用上面) | — | **13 M** | 0 |
| 全局 top-K all-gather (§6.3) | — | — | 2.2 M | **22 MB** |
| latent gather (§7) | 11 MB | — | 0.06 M | 0 |
| sparse MQA (主干 attention) | latent gather 已含 | 11·8·64·2048·512·2 ≈ **12 G** (BF16) | — | 0 |
| LSE merge (§8) | — | — | **17 M** | **1.05 MB** |
| **合计** | **~0.36 GB** | **~18 G FLOPs** | **~35 M vec ops** | **~23 MB** |

> 假设 1 GHz、TC 算力远过剩、vec 256 fp32 lane → **理论 vec 时间 35 M / 256 ≈ 137K cycles ≈ 137 µs / step**。
> Lmem read 0.36 GB / step 在 ~1 TB/s 量级的近存 BW 下 ≈ 360 µs；二者大致同级，可期望 step 时间 200–400 µs（粗）。
> 这只是 DSA 子层；KDA 子层、MoE、其他 dense 都不在本预算内。

---

## 10. DSA 并行选项（8 req 下重新打分）

| 选项 | latent K | index K | per-device Lmem (大, bf16) | 跨卡 comm | 评价 |
|---|---|---|---:|---:|---|
| **O2 · 纯 CP=16** | CP shard | CP shard | **3.10 GB** | top-k 22 MB + merge 1 MB ≈ **23 MB / step** | **主线**：均衡、低延迟、与 KDA TP 正交 |
| O3 · CP=16 + TP=16（复用 16 device） | CP shard | CP shard | 同 O2，加 projection weight ÷16 | 同 O2 + o_proj all-reduce | 略省 weight，复杂度↑ |
| O4 · CP=4 × DP=4 | 4-shard / 组 | 4-shard / 组 | 3.10 GB | 4-device merge ~6 MB / step | 通信拓扑友好；req 间隔离好 |
| **O6 · CP=16 + index 复制**（V1_b 新增） | CP shard | **每 device 复制** | 3.10 + 5.45 ≈ **8.55 GB** | merge 1 MB（**top-k 零通信**） | 见 §4：8 req 装得下，但 GEMM 多花 16× → 不推荐 |
| O7 · 纯 DP=8 | 全本地 | 全本地 | 5.5 + 0.71 = **6.21 GB** | 0 | 8 device 在算，8 device 空转；浪费 |

→ **主线：O2**。O4 在 NIC 分簇硬件上可能更优，O6 仅在互连极弱时考虑。

---

## 11. 容量总账（8 req · 大模型 · CP=16 · bf16 latent）

> Lmem 容量：per-device **28.75 GB** (= 14,720 MB × 2 core)，per-core **14.38 GB**。
> 下文假设 latent K / index K 在 device 内 **平均铺到 2 个 core 的 Lmem**（per-core = per-device ÷ 2）。若 kernel 实现是单 core 独占，per-core 占用翻倍、另一 core 这一项为 0。

| 类别 | 位置 | per-device | per-core | per-device 占比 (28.75 GB) | per-core 占比 (14.38 GB) |
|---|---|---:|---:|---:|---:|
| KDA state (TP=16) | **Gmem** | 36 MB | 18 MB | 2.0% of 1.8 GB | — |
| DSA index K (CP shard) | **Lmem** | 363 MB | **182 MB** | 1.3% | **1.3%** |
| DSA latent K (CP shard, bf16) | **Lmem** | 2.75 GB | **1.38 GB** | 9.6% | **9.6%** |
| **小计 (主线 V1_b)** | — | Gmem 36 MB / **Lmem 3.11 GB** | Gmem 18 MB / **Lmem 1.56 GB** | **10.8%** | **10.8%** |
| 同上但 fp8 latent | Lmem | 1.74 GB | **0.87 GB** | 6.1% | **6.1%** |
| 替代路线 V1_b' / O6 (index 复制) | Lmem | **8.55 GB** | **4.28 GB** | 29.7% | **29.7%** |

→ 主线方案 per-core Lmem 占用 ≈ 1.56 GB（10.8%），per-core 剩余约 **12.82 GB** 给权重切片、KV-transfer 双缓冲、shmem 溢出等。
→ 即便走 O6（index 复制），per-core 占用也只到 4.28 GB（29.7%），剩余 ~10.10 GB；容量不是 O6 的卡点（参 §4 的 GEMM / BW 代价才是）。

---

## 12. 与 V1 的差异回顾

1. **容量更松**：per-device Lmem 从 4.66 GB → 3.10 GB，整套设计余量更大。
2. **index 复制方案重新进入候选**：但 §4 分析表明仍不划算（16× 冗余计算 vs 22 MB/step 通信节省），主线还是分片。
3. **vector-core 工作量首次量化**（§6/§8）：top-K + LSE merge 共 ~30 M FP32 ops/step/device，在 256 lane 下是 ~100K cycles 量级，**不会成为瓶颈**——瓶颈仍是 Lmem read BW（index GEMM 的 K 流）。
4. **indexer GEMM 形状**确认（§5.2）：M=64, K=128, N=32K（per device）——非常 BW-bound，正好对应 index K 放 Lmem 的选择。
5. KDA 与 latent K 的放置结论与 V1 完全一致。

---

## 13. 开放问题 / 待 benchmark

继承 V1 §8 全部，新增：

1. **vec-core 频率与单 op 实际 cycle**：本文以"1 op/lane/cycle @ 1 GHz"做估算，需要实测校准。
2. **top-K 算法**：分层 tournament vs radix vs bitonic 在 64-lane SIMD 上的实测开销比较——尤其 32K → 2048 这个规模点。
3. **fused indexer kernel**：能否做到 "TC 算 GEMM + vec 同时跑 top-K reduce" 完全流水？需要 kernel 工程验证。
4. **gather → sparse MQA 的 shmem 容量**：每 layer per req 的 active gather 行 ≈ 128 行 × 1 KB = 128 KB；shmem 是否够？需要落实 kernel 的 tile 切法。
5. **8 req vs 12 req 的 throughput 拐点**：MQA / GEMM 的 batch 维利用率在 B=8 vs B=12 时差别多大；是否值得为了塞 12 req 而升级到更激进的 latent fp8。
6. **index K 复制（O6）是否在某些拓扑下值得**：NIC 分簇 + 跨簇延迟高的场景下，22 MB all-gather 可能贵；这时 O6 的"零通信"才有意义。需要拿目标互连带宽实测拐点。

---

## 14. 速查（一页本）

```
======== 8 req @ 512K, 大 GLM-5, CP=16, bf16 latent ========

PLACEMENT
  KDA state (TP=16)   →  Gmem,   36 MB / device  ( 18 MB / core)
  DSA index K         →  Lmem,  363 MB / device  (182 MB / core, CP shard)
  DSA latent K        →  Lmem, 2.75 GB / device  (1.38 GB / core, CP shard, bf16)
  ─────────────────────────────────────────────
  Gmem used: 36 MB / 1.8 GB         (2.0%)
  Lmem used (per-device): 3.11 GB / 28.75 GB    (10.8%)
  Lmem used (per-core)  : 1.56 GB / 14.38 GB    (10.8%)

PER-STEP (per device, 11 DSA 层)
  Lmem read  : ~0.36 GB  (主要 index K 流)
  TC FLOPs   : ~18 G     (FP8 GEMM 6 G + BF16 sparse MQA 12 G)
  Vec ops    : ~35 M     (top-K + LSE merge)
  Comm       : ~23 MB    (top-k AG 22 MB + merge 1 MB, ring)

HARDWARE
  per device  : 2 core × 4 vec core × 64 (bf16/fp8) / 32 (fp32) lane
              ≈ 512 bf16 / 256 fp32 lane/device
  per 16-dev  ≈ 8192 bf16 / 4096 fp32 lane

VEC-CORE BUDGET
  top-K local (per dev): ~13 M FP32 ops/step → ~50 µs @1GHz, 256 lane
  LSE merge   (per dev): ~17 M FP32 ops/step → ~67 µs
  → vec 时间 ~130 µs / step (远小于 Lmem read 时间 ~360 µs)
  → DSA step 大概率 BW-bound on Lmem read, not compute-bound

CALL OUT
  index 复制方案 (O6) 在 8 req 下装得下 (8.55 GB Lmem)
    但 GEMM/读 BW 全部 ×16 → 仅在互连极弱时考虑
  主线仍是 CP=16 分片
```
