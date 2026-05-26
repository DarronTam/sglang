# GLM5-Next Decode · 16-Device KV 放置（v1_c · 4–8 req · CP / CP-DP · latent K 是否能进 Gmem · top-K merge 详解）

> 在 [`KvPlacement_Decode_16dev_v1_b.md`](KvPlacement_Decode_16dev_v1_b.md) 基础上迭代：
>
> 1. **裁掉**：TP-only（V1 的 O1/O3）、纯 DP（O5/O7）、index 复制（O6）。
> 2. **聚焦**：O2（CP=16）、O4（CP=4×DP=4）；**新增 CP=8×DP=2**。
> 3. **request 范围**：4–8（不再固定 8 / 12）。
> 4. **新议题 A**：latent K 是否能塞进 Gmem（在小 CP 域 + fp8 量化下）。
> 5. **新议题 B**：top-K 取 global top-2048 的合并机制——这一步具体怎么干。
>
> 模型仍以 `config_16b_v2.json` / `config.json` 为准；下文未特别说明时按**大 GLM-5（11 DSA / 34 KDA 层，Nh=64, I=8）**估。

---

## 0. 与 V1_b 的差异速看

| 维度 | V1_b | **V1_c** |
|---|---|---|
| 候选方案数 | 6（含 TP / DP / O6） | **3**（O2 / **CP=8×DP=2** / O4） |
| Req 数 | 8 | **4–8（区间）** |
| latent K 位置 | 一律 Lmem | **重新评估：4 req 任意 dtype / 8 req + fp8 → 可放 Gmem**；§3 |
| index K 位置 | 主线分片 Lmem（已淘汰复制） | 同 V1_b，分片 Lmem，固定 |
| KDA state 位置 | Gmem | Gmem，固定 |
| 新分析 | indexer GEMM + vec 预算 | **top-K 全局合并机制详解（§5）** + **TC GEMM 时间预算 + 2-core 切分（§6）** |

---

## 1. 三方案的共同点：per-device 容量

设 R = request 数（4–8），D = 16 设备：

```
per_device_KV(R, dtype) = R · per_req_KV(dtype) / 16
```

> **关键观察**：O2 / CP=8×DP=2 / O4 三种方案 **per-device 容量完全相同**，因为它们都把全部 16 device 用满。三者的差异只在 **(a) merge / 通信的"作用域"**、**(b) 单 req 延迟**、**(c) PD 拓扑**——不在 per-device 存储。

### 1.1 per-device 容量表（大 GLM-5，512 K context）

| 类别 | dtype | per-req（参考） | R=4 per-device | R=8 per-device |
|---|---|---:|---:|---:|
| KDA state（TP=16，per-req） | bf16 | 4.55 MB | **18.2 MB** | 36.4 MB |
| DSA latent K | bf16 | 5.50 GB | **1.375 GB** | **2.75 GB** |
| DSA latent K | fp8 (+block scale) | 2.80 GB | **0.70 GB** | **1.40 GB** |
| DSA index K | FP8+scale | 708 MB | **177 MB** | **354 MB** |

### 1.2 与硬件容量的对比

| 单 device 容量 | Gmem 1.8 GB | Lmem 28.75 GB |
|---|---|---|
| R=4 latent bf16 (1.375 GB) | **✅ 装得下**（占 76%） | ✅（4.8%） |
| R=4 latent fp8 (0.70 GB) | **✅**（39%） | ✅（2.4%） |
| R=8 latent bf16 (2.75 GB) | ❌ 装不下 | ✅（9.6%） |
| **R=8 latent fp8 (1.40 GB)** | **✅**（78%） | ✅（4.9%） |
| index K（R=8, 354 MB） | 单独装得下，但与 latent 共占 Gmem 大概率挤兑 | ✅（1.2%） |

**结论**（容量预算先看一眼）：
- **R=4** 任意 dtype：latent K **可以**进 Gmem；
- **R=8 + bf16 latent**：latent K **不能**进 Gmem，强制 Lmem；
- **R=8 + fp8 latent**：latent K **可以**进 Gmem（占 78%，留 22% ≈ 0.4 GB 给 KDA state + KV-transfer 缓冲）；
- **index K** 不挤 Gmem（继续 Lmem），KDA state 也不挤（继续 Gmem）。

---

## 2. 决策矩阵（V1_c 锁定）

| State | 位置 | 备注 |
|---|---|---|
| KDA recurrent + conv | **Gmem**（固定） | 高频小写 → 不打扰 TC（V1 §4） |
| DSA index K | **Lmem**（CP shard，固定） | indexer GEMM 是 K-read BW-bound，必须近 TC（V1_b §5） |
| **DSA latent K** | **看 §3**（R/dtype 决定 Gmem 或 Lmem） | **本版的新决策点** |

---

## 3. latent K 放 Gmem 的可行性（V1_c 核心新分析）

### 3.1 为什么要重新考虑 Gmem？

V1 §5 当时排除 Gmem 是因为想"把整份 latent K mirror 到 Gmem"——容量爆。
V1_c 不再 mirror，而是把 **latent K 的主拷贝直接放 Gmem**。在 R=4–8 且容量允许时，这条路有三个独立收益：

1. **gather 时机**：sparse MQA 阶段，TC 读 Lmem scratch（128 行/req gathered rows），同时 gather kernel 从 Gmem 读源——**两条 BW 通道独立**，不抢资源。  
   对比：latent K 在 Lmem 时，gather read 与 TC read 都走 Lmem，必须靠 kernel 内 double-buffer / tile pipelining 错开。
2. **KV-transfer**：PD 分离时 prefill→decode 是高 BW host 写。写到 Gmem 与 TC 解耦；写到 Lmem 要么排开 TC 要么挤进 idle window。
3. **gather 数据量本来就小**：每 device 每步全部 11 层 ≈ **5.6 MB（fp8）/ 11 MB（bf16）**——以任何合理 Gmem 读 BW 都几乎免费。

### 3.2 各 (R, dtype) 组合的可行性

| 配置 | latent 占 Gmem | 余下 Gmem（给 KDA + scratch） | 评估 |
|---|---:|---:|---|
| R=4, bf16 | 1.375 GB (76%) | 0.42 GB | **可行**，但 Gmem 较紧；KV-transfer 双缓冲需小心 |
| **R=4, fp8** | 0.70 GB (39%) | 1.10 GB | **首选**：宽松，KDA + scratch + transfer buffer 都够 |
| R=8, bf16 | — | — | **不可行**（超容） |
| **R=8, fp8** | 1.40 GB (78%) | 0.40 GB | **可行**：边缘紧，需把 KDA + scratch 总和压在 0.4 GB 内（KDA ~36 MB，留给 transfer 双缓冲 ~360 MB → 充裕） |

### 3.3 sparse MQA 数据路径对比

**路径 X · latent K in Gmem**（V1_c 新主线，当容量允许）

```
host (prefill) ─KV-transfer─►  Gmem  (full latent K shard)        ── 高效写，不扰 TC
                                │
                  decode step  ▼
                               Gmem  ──gather (Gmem read)──►  shmem  ──staging──►  Lmem scratch
                                                                                       │
                                                                                       ▼
                                                                                  TC reads → sparse MQA
```

**路径 Y · latent K in Lmem**（V1_b 原主线，当容量不允许 Gmem）

```
host ─KV-transfer─► Lmem (full)        ── 需与 TC 错开
                      │
       decode step   ▼
                     Lmem ──gather (Lmem read)──►  shmem  ──►  Lmem scratch ──► TC
                              ▲                                   ▲
                              └─── 与 TC read 抢 Lmem BW ─────────┘
```

### 3.4 BW 量级估算（参考）

| 项 | 量级 / step / device |
|---|---|
| gather data (R=8, fp8, 11 层) | ~5.6 MB |
| gather data (R=8, bf16, 11 层) | ~11 MB |
| index K 读（indexer GEMM） | ~350 MB |
| KV-transfer 突发写（per req @ 512 K 启动时） | latent fp8: 1.40 GB / 16 dev = 89 MB；index: 354 MB / 16 ≈ 22 MB；KDA: ~5 MB |

> latent gather 是 BW 上的"零头"，但它的**位置**决定要不要和 TC 抢 Lmem 通道。把 latent K 移到 Gmem 是**减负 Lmem 通道**的关键。

### 3.5 推荐

| R | 推荐 latent K 位置 | 理由 |
|---|---|---|
| 4 | **Gmem（fp8 优先）** | 容量宽松；解耦 gather 与 TC；KV-transfer 高效 |
| 8 | **Gmem，仅在 fp8 下**；否则 **Lmem (bf16)** | fp8 + 78% 占用是可接受的设计点 |

> 副作用：选 Gmem 路线后，gather kernel 的源寻址要切到 Gmem 空间；page-table 也要相应在 Gmem 侧维护（或两边都维护元数据）。这部分实现细节见 §6.4。

---

## 4. 三种并行方案对比

| 维度 | O2 · CP=16 | **CP=8 × DP=2** | O4 · CP=4 × DP=4 |
|---|---|---|---|
| CP 域大小 | 16 device | 8 device | 4 device |
| DP 域数 | 1 | 2 | 4 |
| 每 DP 域 req 数（R=8） | 8 | 4 | 2 |
| 每 DP 域 req 数（R=4） | 4 | 2 | 1 |
| 每 device 持 token shard | seqlen / 16 = 32 K | 32 K | 32 K |
| **per-device 容量** | 同表 §1.1 | **同上** | **同上** |
| top-K 候选合并范围 | 16 device | **8 device** | 4 device |
| LSE merge 范围 | 16 device | 8 device | 4 device |
| 单 req 计算并行度 | 16× | 8× | 4× |
| 单 req 延迟（理论） | 最低 | 中 | 最高（同 R/CP 量级） |
| 负载均衡（R=4 时） | 完美（4 req × 16 设备打满） | 平均 2 req/CP-group | **完美**（1 req/CP-group/DP） |
| 负载均衡（R=8 时） | 完美 | 平均 4 req/CP-group | 平均 2 req/DP-group |
| PD-transfer 拓扑 | 1×16 scatter | 2×8 | 4×4，最契合 NIC 分簇 |
| 互连压力（每层每步） | 见 §5 详细 | 中 | 最低 |

### 4.1 通信量（top-K all-gather + LSE merge）汇总（大模型 11 层、R=8）

| 方案 | top-K AG（per device） | LSE merge（per device, ring） | 合计 / step |
|---|---:|---:|---:|
| O2 CP=16 | 11·16·8·2048·8 B = **22 MB** | 11·2·8·64·513·2 B ≈ 1.05 MB | **~23 MB** |
| CP=8×DP=2 | 11·8·8·2048·8 B = **11 MB** | ≈ 1.05 MB | **~12 MB** |
| O4 CP=4×DP=4 | 11·4·8·2048·8 B = **5.5 MB** | ≈ 1.05 MB | **~6.5 MB** |

> 小作用域 → 通信少，但单 req 计算并行度也小。**weak interconnect 倾向 O4；strong interconnect 倾向 O2**；CP=8×DP=2 是中间。

### 4.2 R=4 的特殊性

R=4 + O4 (CP=4×DP=4) 是**最优雅的组合**：每个 DP 组内正好 1 个 req，零跨 req 干扰；CP=4 内做 partial + merge，作用域最小。

R=4 + O2 仍可行（16-device 并行算一个 req 的 partial），延迟最低，但通信作用域大。

R=4 + CP=8×DP=2：2 req / DP 组，CP=8 内做 partial。中间。

### 4.3 R=8 的特殊性

R=8 + O4：每 DP 组 2 req，4 device 内做 partial——CP=4 的并行度对单 req 是足够的（每 device 32 K K shard），通信小。
R=8 + O2：所有 8 req 跨 16 device 并行，单 req 延迟最低，通信最大。
R=8 + CP=8×DP=2：折中。

### 4.4 选型建议（V1_c）

- **R=4 默认 O4**（最干净、通信最小）；
- **R=8 默认 CP=8×DP=2**（折中：通信 ≈ O4 + O2 中点；负载均衡 OK）；
- **追求最低单 req 延迟时切 O2**（任意 R）；
- O2 / CP=8×DP=2 / O4 在 per-device 容量上**完全等价**——切换无需改 KV 布局，只改 collective scope。

---

## 5. top-K 全局合并：取 top-2048 index 这一步怎么干（V1_c 重点）

> 用户原话："感觉 top-K 得到 top-2048 index 这一步还比较麻烦"——展开。

### 5.1 问题定义

在 CP=N 组内（N ∈ {4, 8, 16}），每 device r 已经有：

```
local_logits_r[B, n_local]    n_local = seqlen / N = 32 K (when seqlen=512K, N=16) / 64 K (N=8) / 128 K (N=4)
                                                    （注意 N 越小，每 device n_local 越大）
```

**目标**：得到全局 top-2048 的 (position, logit) 对，使每个 device 都能映射出自己的 `topk_slots_r[B, 2048]`（不归本 device 的位置 → -1）。

**等价条件**：全局 top-2048 必出现在 N 个 device 各自的 local top-2048 内（因为某 token 若全局排前 2048，则它在本地子集里也至多被 2047 个本地 token 压过 → 一定进本地 top-2048）。

所以核心是两步：

1. **本地 top-K**：每 device 独立从 n_local 选出 local top-2048；
2. **全局合并**：从 N×2048 个候选里再选 top-2048。

### 5.2 本地 top-K（每 device 内部，vec core 工作）

候选规模：

| N (CP域) | n_local @ 512K | 本地 top-K 候选 |
|---:|---:|---:|
| 16 | 32 K | 32K → 2048 |
| 8 | 64 K | 64K → 2048 |
| 4 | 128 K | 128K → 2048 |

**推荐算法：分层 tournament + 排序输出**

每 lane 维护一个长度 16 的 mini-heap（heap-as-array，SIMD-friendly compare-update），扫完自己负责的那段；然后跨 lane / 跨 vec core 做 log₂ 级 merge → 最终输出 2048 个**已按 logit 降序排序**的候选。

为什么强制"已排序输出"——为下面 §5.3 的合并节省一大块工作。

工作量（vec FP32 ops，per req per layer，N=8 / n_local=64K 为例）：

```
扫描候选          ≈ n_local · 4 ops    = 256 K ops
跨-lane merge     ≈ log₂(lanes)·K·2    = 8 · 4096 = 33 K ops
跨-vc / 跨-core 合并到一致排序的 2K  ≈ K · log₂(local merges) ≈ 22 K ops
                                       ────────────────────
                                       ≈ 311 K ops / req / 层
```

R=8 × 11 层 ≈ **27 M ops / step / device**（FP32）；在 256 FP32 lane / device 下 ≈ 105 K cycle ≈ **~105 µs**。

### 5.3 全局合并：N×2048 候选 → top-2048

> 这是用户问的"麻烦的那一步"。给四种实现，从最简单到最优。

#### 方法 A · 全 all-gather + 每 device 各自冗余合并

```
[per device r] send local_top_K_r[2048]   (pos:int32 + logit:fp32 = 8 B per entry)
all-gather  →  cand[N · 2048]   buf_size_per_dev = N·K·8 B
[per device r] global-top-K(cand) → top_K_global[2048]    (16K-32K → 2K)
```

通信（per device per req per 层）：(N-1) · K · 8 B ≈ N·K·8 B
- N=16: 256 KB, N=8: 128 KB, N=4: 64 KB

合并 vec 工作（per device per req per 层）：

`N·K → K`：把 N 个**已排序** K-list（来自 §5.2）做 N-way merge，只取 top-K。
经典 SIMD-friendly 算法：bitonic merge tree，log₂(N) 个 round：

| N | merge rounds | per-round 工作 (2K→K) | 总 vec ops |
|---:|---:|---|---:|
| 16 | 4 | ≈ K = 2 K | **8 K** |
| 8 | 3 | ≈ K | **6 K** |
| 4 | 2 | ≈ K | **4 K** |

R=8 × 11 层 × N=16 → ≈ **0.7 M ops / step / device**（FP32）→ 在 256-lane 下 < 3 µs。**几乎免费**。

**结论**：A 方法的合并本身极便宜，瓶颈是**冗余**——N 个 device 各算一遍同样的事；以及 all-gather 的带宽（N·K·8B / device）。

#### 方法 B · 树状 reduce-with-merge + broadcast

```
log₂(N) 个 round 的 butterfly 配对，每 round：
  pair_rank ← XOR partner; exchange K-list (8 B/entry · K)
  在本地 merge 两条**已排序** K-list → 取 top-K（O(K) 工作）
log₂(N) 后所有 device 持有全局 top-K（butterfly all-reduce 形态，结果自动复制到所有 rank）
```

通信（per device per req per 层）：`log₂(N) · K · 8 B`（每 round 双向交换 K 候选）
- N=16: 4·16 KB = 64 KB（**比 A 省 4×**）
- N=8: 3·16 KB = 48 KB
- N=4: 2·16 KB = 32 KB

vec 工作（per device per req per 层）：`log₂(N) · K` ≈ 8 K（N=16）、6 K（N=8）、4 K（N=4）——同 A 量级；但 **不再冗余**（每 round 工作只算一次）。

**结论**：B 在通信上比 A 省 ~4×（N=16）/~2.7×（N=8）/~2×（N=4），vec 工作量同 A。**需要写自定义 collective**（带 merge reduce op），不是开箱即用 NCCL。

#### 方法 C · 单 leader 计算 + broadcast

```
[device 0] reduce (gather) N 个 K-list → 本地算 global top-K → broadcast
```

延迟最高（reduce → 算 → broadcast 串行），单设备热点；不推荐。

#### 方法 D · 不取 global top-K 的索引集，直接 reduce attention 输出

**激进版**：跳过全局 top-K，每 device 直接用 local top-K 子集做 partial sparse MQA，然后 merge。
- 数学上**不严格等价**于"先 global top-K 再算 attention"（因为局部 top-K 的并集 ⊇ 全局 top-K，partial 会包含全局排名 2049–2050+ 的 token）
- 但若分布稀疏度足够好，差异可能在精度噪声内
- **不推荐**作为默认；只作为 ablation 项

#### 5.3 汇总

| 方法 | 通信 (N=16) | 通信 (N=8) | 通信 (N=4) | vec 合并工作 | 实现复杂度 | 推荐 |
|---|---:|---:|---:|---:|---|---|
| **A** all-gather + 冗余 merge | 256 KB | 128 KB | 64 KB | 0.7 M ops / step | **低**（NCCL 直出） | **R=8 main**：通信仍小 |
| **B** tree-reduce with merge | 64 KB | 48 KB | 32 KB | 0.7 M ops / step | 高（custom op） | **R=4 main 或** 互连弱时 |
| C leader+broadcast | ~256 KB | ~128 KB | ~64 KB | + 串行延迟 | 中 | × |
| D 跳 global top-K | 0 | 0 | 0 | 0 | 低 | 仅 ablation |

### 5.4 page-table 映射（拿到 global top-K 之后）

```
对 device r, for each i in [0, 2048):
    pos = global_top_K[i].pos
    if (pos % N) == r:                                # round-robin owner
        topk_slots_r[i] = page_table_r[ pos // N ]    # 实际由 paged page-table 索引（page_size=64）
    else:
        topk_slots_r[i] = -1
```

vec 工作：2048 个 mod / div / page lookup，~10 vec ops/entry × 2048 = 20 K ops / req / 层。
R=8 × 11 层 ≈ **1.8 M ops / step / device**。在 256-lane 下 < 8 µs。**便宜**。

### 5.5 sorted-runs 一句话

**整条 top-K 链路最关键的工程优化**：本地 top-K 输出**已排序**，让全局合并降为 O(N·K) 的 merge（不是 O(N·K · log)）。这把方法 A/B 的 vec 工作都压到 < 1 M ops / step，比扫描阶段（27 M ops / step）少一个数量级。**严格按"sorted local top-K + 排序-merge 合并"实现**。

### 5.6 综合 latency 预算（CP=8, R=8, 大模型, 11 层）

```
本地 top-K 扫描+排序     ≈ 27 M ops    ≈ 105 µs   (256 FP32 lane)
方法 A all-gather        ≈ 128 KB      ≈ µs 级 (NVLink/CXL)
方法 A 本地合并 (sorted) ≈ 0.7 M ops   ≈ 3 µs
page-table 映射          ≈ 1.8 M ops   ≈ 8 µs
                          ──────────
                          ≈ ~116 µs / decode step / device on top-K
```

> 与 V1_b §9 估的总 vec 时间 ~130 µs / step 一致——大部分还是本地扫描，全局合并真正的额外开销 < 15 µs。**"麻烦"是工程实现层面（自定义 reduce / sorted-runs / SIMD merge）；运行时开销其实小**。

---

## 6. TC GEMM 时间预算 + 2-core 内部切分（V1_c 新增）

> 范围限定：本节**只算 indexer GEMM 与 sparse MQA 两个 GEMM**的 TC 时间。其他 dense 路径（q_b / kv_b / o_proj 等）不在本节范围。
>
> Lmem 是 **per-core private**——所以"两 core 怎么切"既是计算并行问题，也是数据放置问题。

### 6.1 TC 硬件参数

| 维度 | 数值 |
|---|---|
| TC MMA 形状 (FP8) | `[1, 128] × [128, 128]` per cycle |
| TC MMA 形状 (BF16) | `[1, 64]  × [64, 128]`  per cycle |
| Compute 时钟 | 400 MHz |
| Memory 时钟 | 400 MHz |
| Lmem→TC BW | **3.2 TB/s per core**（= 8 KB / cycle） |
| TC 数 | 假设 **1 TC / core, 2 TC / device**（用户未明示，按 vec-core/core 类比） |

**单 TC 峰值算力**：

| dtype | 每 cycle FLOPs | @ 400 MHz |
|---|---:|---:|
| FP8 | 2 · 1 · 128 · 128 = 32,768 | **13.1 TFLOPS** |
| BF16 | 2 · 1 · 64 · 128 = 16,384 | **6.55 TFLOPS** |

**Per-device 峰值**（2 TC）：FP8 **26.2 TFLOPS** / BF16 **13.1 TFLOPS**；Lmem BW **6.4 TB/s**。

**算密度 (AI) 拐点**：FP8 13.1 T / 3.2 T/s ≈ **4.1 FLOPs/B**；超过此算密度 → 计算受限。

### 6.2 Indexer GEMM 形状与时间

每 device 每 DSA 层执行 `R_per_dev` 个独立 GEMM（不同 req 的 index K 不同），每个 GEMM：

```
[M=I=8 queries] · [K=Di=128] × [K=128] · [N=n_local keys]^T  →  [M=8, N=n_local]  (FP8 in, FP32 accum)
```

| 方案 | R_per_dev | n_local | per-device per-layer FLOPs |
|---|---:|---:|---:|
| O2 CP=16 | 8 | 32 K | 2·8·32K·128·8 = **524 M** |
| CP=8×DP=2 | 4 | 64 K | 524 M |
| O4 CP=4×DP=4 | 2 | 128 K | 524 M |

→ **per-device FLOPs 同**：约 **524 M FP8 / 层 / step**，11 层 = **5.76 G FP8 / step / device**。

**算密度**：AI ≈ `2·I·N·K / (I·K + N·K) = 2·I·N / (I + N)` ≈ **16 FP8 FLOPs/B**（N≫I 时） → **远大于 4.1 → 计算受限**（不是 BW 受限）。

**时间**：
```
5.76e9 FLOPs / 26.2e12 FLOPs/s ≈ 220 µs / step / device  （2 TC 并行）
```

### 6.3 Sparse MQA GEMM 形状与时间

每层每 req 两个连续 GEMM（latent-absorb 形式，head_dim = Rkv = 512）：

```
GEMM-1  qK^T:  q[Nh=64, 512] · K_topk[N_eff, 512]^T  →  logits[Nh, N_eff]
GEMM-2  attn·K: attn[Nh, N_eff] · K_topk[N_eff, 512] →  out_latent[Nh, 512]
其中 N_eff ≈ 2048 / N_CP   (本 device 持有的 top-2048 子集)
```

| 方案 | N_CP | N_eff per device | R_per_dev | 两 GEMM FLOPs / 层 |
|---|---:|---:|---:|---:|
| O2 | 16 | ~128 | 8 | 8·(2·64·128·512 + 2·64·128·512) ≈ **134 M** |
| CP=8×DP=2 | 8 | ~256 | 4 | 4·(2·64·256·512)·2 ≈ 134 M |
| O4 | 4 | ~512 | 2 | 2·(2·64·512·512)·2 ≈ 134 M |

→ per-device 同：约 **134 M / 层**，11 层 = **1.48 G / step / device**。

**时间**：

| dtype | 时间 |
|---|---:|
| BF16（latent 是 bf16） | 1.48e9 / 13.1e12 = **113 µs / step / device** |
| FP8（latent 是 fp8 且 Q 也量化） | 1.48e9 / 26.2e12 = **56 µs / step / device** |

> sparse MQA 的算密度 ≈ `2·Nh·N_eff·512 / (Nh·512 + N_eff·512) ≈ 2·Nh / (1 + Nh/N_eff)` —— Nh=64, N_eff≥128 → AI ≈ **64–100 FLOPs/B**，**远超 4.1，计算受限**。

### 6.4 两 core 切分策略

**关键约束**：
- TC FP8 MMA 的 K 最小粒度 = **128**；BF16 MMA 的 K 最小粒度 = **64**。
- Lmem **per-core private**，跨核数据要么走 shmem 交换，要么走对方 Lmem（慢）。

#### 6.4.1 Indexer：**只能切 N**（intra-device CP=2 进一步分 token）

| 选项 | 是否可行 | 原因 |
|---|---|---|
| 切 M (Q heads) | M = I = 8，太小，切 4/core 后 TC M=4 利用率差 | × |
| 切 K (Di = 128) | K=128 是 TC FP8 最小粒度，不能再切 | × |
| **切 N (token)** | 各 core 持 n_local/2 个 token 的 index K shard，独立扫各自一半 | ✅ |

**布局**：
- core 0 存 `index_K[r, even_pos_in_local]`（或前一半 token），core 1 存后一半；
- 各 core 独立做 GEMM + 本地 top-2048 from n_local/2；
- **intra-device 2-way merge**：两 core 经 shmem 交换两条已排序 K-list → device-level top-2048（小，sorted-merge ≈ K 次 compare）；
- 然后参与 inter-device top-K（§5 方法 A/B），device-level 起步。

→ 把"device-level CP=N"在内部变成"core-level CP=2N"：O2 实际是 32-way CP，CP=8×DP=2 实际是 16-way CP，等等。**inter-device 通信不变**（核间归并不出 device）。

**单 core 时间**：220 µs / 2 ≈ **110 µs / step / core**（理想线性 scaling，FP8 计算受限，BW 还有 4× 富余）。

#### 6.4.2 Sparse MQA：M / K / N 三条切法都可

> 与 indexer 不同，sparse MQA 的 **M = Nh = 64（大）/ 32 (16B)** 足够大，切 M 后每 core 仍有 Nh/2 = 32 / 16 行查询——远大于 TC 一周期 1 行的 issue rate，**M-split 完全可行**。head_dim = 512 也能切（256/core ≥ TC 最小粒度 128）。所以这一段比 indexer 自由度高得多。

| 选项 | 数据布局 (per core) | GEMM 流程 | 跨核通信 | latent K 存储代价 |
|---|---|---|---|---|
| **(a) K-split**：head_dim 256/core | `latent_K[N_local, 256]`，core 0 持 dim[0:256]、core 1 持 dim[256:512]；每 core 看**全部** N_local token | GEMM-1: 各 core 算 partial logits[Nh, N_eff] → **reduce-sum 跨核** → softmax / broadcast attn → GEMM-2: 各 core `attn · K_half → out_half[Nh, 256]`；末尾沿 head_dim concat | reduce-sum + broadcast ≈ `Nh·N_eff·6B`（O2: 48 KB）/ layer / req | per core = device / 2（dim 切片，非重复） |
| **(b) N-split**：N_local/2 token / core，head_dim 完整 | `latent_K[N_local/2, 512]`，每 core 持半数 token 全部 dim | GEMM-1+softmax+GEMM-2 各 core **完全独立**（在自己的 N_eff/2 子集上） → **online-softmax LSE merge 跨核**（V6 §10，2-way） | merge buffer ≈ `(out[Nh,512]+lse[Nh])·2B ≈ 65 KB`（AG）/ **33 KB**（ring）/ layer / req | per core = device / 2（token 切片，非重复） |
| **(c) M-split**：Nh / 2 头 / core，K 共享 | `q_half[Nh/2, 512]` 各 core 不同；**`latent_K[N_local, 512]` 两 core 共享/复制** | GEMM-1+softmax+GEMM-2 各 core **完全独立**（不同 Q 头，同一份 K）→ 末尾沿 head 维 concat（**零通信**） | **0**（不需要跨核 reduce / merge） | **看 K 在哪**——见下表 |

**(c) M-split 的 K 存储依赖 latent K 放在哪儿**：

| latent K 位置 | M-split 下 K 的处理 | 代价 |
|---|---|---|
| **Gmem（§3 path X）** | Gmem 是 device 共享存储，**两 core 自然访问同一份**；各 core gather 到自己 private Lmem scratch | **零额外开销**：K 在 Gmem 里只有一份；两 core 各做一次 gather（~17 MB / step 总量翻倍到 34 MB，相对 Gmem BW 仍是噪声） |
| Lmem (§3 path Y) | Lmem 是 per-core private，两 core 要么**复制 latent K cache**（2× 容量），要么走慢的跨核读 | per-core Lmem 占用从 1.38 GB → **2.75 GB**（R=8 + bf16 时，仍在 14.38 GB Lmem 余量内，但和 N-split / K-split 比多一倍） |

**等价性**：三种切法**逐元素等价**于"在完整 N_eff 上做一次 attention"——
- (a) K-split：内积分块求和（K-axis 分布律）；
- (b) N-split：online softmax 标准 partition；
- (c) M-split：head 维独立（attention 各头本来就独立）——**数学上最干净**，因为不需要任何跨核归并算子，只需 concat。

#### 6.4.3 推荐组合（**取决于 latent K 在哪**）

| latent K 位置（§3） | Indexer 切法 | Sparse MQA 切法（推荐） | 理由 |
|---|---|---|---|
| **Gmem（path X，R=4 或 R=8+fp8）** | N-split（强制） | **M-split** | M-split 在 Gmem-K 下是**最干净的方案**：K 自然 device 共享、零跨核通信、零归并算子；只需末尾 concat。Lmem 不需要装 latent K 蓄水池 → §6.6 的 Lmem BW 完全留给 indexer 流 |
| **Lmem（path Y，R=8+bf16 fallback）** | N-split（强制） | **N-split** | M-split 在 Lmem-K 下要 2× per-core 容量（虽然装得下，但耗 Lmem 余量、KV-transfer 协议要"两份 K"写）；N-split 与 indexer 共享 token-round-robin 协议，端到端独立，只末尾一次 LSE merge |

**两条路线的差异是真正的架构选择，不是同一方案的不同实现**：
- **Gmem-K + M-split**：sparse MQA 阶段两 core 不互相通信，indexer 与 sparse MQA 用不同 Lmem 内容（indexer 只用 index K shard，sparse MQA 只用 Lmem scratch），互不干扰。
- **Lmem-K + N-split**：所有 KV（index + latent）都按 token round-robin 在 2 core 间均分，protocol 统一；代价是 sparse MQA 多一个 2-way LSE merge。

**K-split 何时上**：仅在 KV-transfer DMA 引擎对"按 head_dim 半切"特别友好时考虑（如硬件原生 stride-2 加载），其他场景 K-split 的跨核 reduce-sum 没有特别优势。

### 6.5 端到端 DSA step 时间预算（per device, R=8, 11 层, 大模型）

| 子项 | 时间 / step / device |
|---|---:|
| Indexer GEMM (FP8, 2 TC 并行) | **~220 µs** |
| Sparse MQA GEMM (BF16) | ~113 µs |
| Sparse MQA GEMM (FP8 latent + Q 量化) | ~56 µs |
| top-K 链路（本地扫描+sorted merge+page-table，§5.6） | ~115 µs |
| LSE merge (V6 §10) | ~67 µs |
| latent gather (BW 极小，可与 GEMM 重叠) | < 5 µs（流水掩盖） |
| **小计（BF16 latent）** | **~515 µs / step / device** |
| **小计（FP8 latent）** | **~458 µs / step / device** |

> 三方案（O2 / CP=8×DP=2 / O4）TC 时间**完全一致**（per-device FLOPs 相同）；差异只在 inter-device 通信延迟，本节不计。
> 实际 step time 会受 latency hiding、kernel issue overhead、Lmem 冲突等影响——本预算用作**相对比较和工程上限**。

### 6.6 Lmem BW 利用率核对

| 阶段 | per-step per-device Lmem read | 对应 GEMM 时间 | 实际 BW 使用 | vs 6.4 TB/s |
|---|---:|---:|---:|---:|
| Indexer K 读 | 352 MB FP8 | 220 µs | 1.6 GB/s | 0.025% |
| Sparse MQA K 读 | ~17 MB | 113 µs | 0.15 GB/s | 0.002% |

→ **Lmem BW 完全富余**，瓶颈是 TC 计算（特别是 indexer GEMM 的 220 µs），不是 BW。可以在同一片 Lmem 上同时跑 indexer 流和别的 KDA / activation 流而不互相挤兑。

### 6.7 对 latent K 放 Gmem 决策（§3）的回声

- §3 的 path X（latent K in Gmem）只影响 sparse MQA 的 K **来源**，不影响 §6.3 的 GEMM 时间——因为 K 数据量 17 MB 在任何合理 Gmem BW 下都是几微秒级，远小于 113 µs 的 TC 时间。
- 真正的收益还是**解耦 Lmem 通道**：indexer 流（220 µs Lmem 读 352 MB）和 sparse MQA 流（如果 latent K 在 Lmem）会争用同一 Lmem 通道；如果 latent K 在 Gmem，sparse MQA 的 K 走 Gmem 不挤 Lmem，两条流并行更干净。

---

## 7. 容量总账（V1_c · 大模型 · R=8 · fp8 latent · O2/CP-DP 任选）

> R=8 + fp8 是 V1_c 的"最满负载、最压力 Gmem"组合。R=4 / bf16 是它的子集。

**Route P · latent K → Gmem**（V1_c 推荐主线，R=4 或 R=8+fp8）

| 类别 | 位置 | per-device | per-core |
|---|---|---:|---:|
| KDA state (TP=16) | Gmem | 36 MB | 18 MB |
| **DSA latent K (R=8, fp8)** | **Gmem** | **1.40 GB** | n/a (Gmem 单 device 空间) |
| DSA index K (CP shard) | Lmem | 363 MB | 182 MB |
| **小计 Gmem** | — | **1.44 GB / 1.8 GB（80%）** | — |
| **小计 Lmem** | — | **0.36 GB / 28.75 GB（1.3%）** | 0.18 GB / 14.38 GB (1.3%) |

**Route Q · latent K → Lmem**（R=8 + bf16 fallback；与 V1_b 同）

| 类别 | 位置 | per-device | per-core |
|---|---|---:|---:|
| KDA state | Gmem | 36 MB | 18 MB |
| DSA latent K (R=8, bf16) | Lmem | 2.75 GB | 1.38 GB |
| DSA index K | Lmem | 363 MB | 182 MB |
| **小计 Gmem** | — | **36 MB** | — |
| **小计 Lmem** | — | **3.11 GB / 28.75 GB（10.8%）** | 1.56 GB / 14.38 GB（10.8%） |

---

## 8. 速查（一页本）

```
======== V1_c · 4–8 req · 大 GLM-5 · CP / CP-DP ========

CAPACITY (per device, 各方案一致)
  R=4 bf16 latent : 1.375 GB → fits Gmem
  R=4 fp8  latent : 0.70  GB → fits Gmem easily
  R=8 bf16 latent : 2.75  GB → MUST Lmem
  R=8 fp8  latent : 1.40  GB → fits Gmem (80%)

PLACEMENT
  KDA state       → Gmem (固定)
  DSA index K     → Lmem (固定, CP shard)
  DSA latent K    → Gmem (R=4 or R=8+fp8) / Lmem (R=8+bf16)

PARALLELISM (per-device 容量三者等价)
  O2  CP=16       : 单 req 延迟最低,  top-K AG 256 KB / 层 / req
  CP=8×DP=2       : 折中,             top-K AG 128 KB / 层 / req
  O4  CP=4×DP=4   : 通信最小,         top-K AG  64 KB / 层 / req
  推荐:
    R=4 → O4  (1 req / DP-group, 干净)
    R=8 → CP=8×DP=2 (折中)
    需最低延迟 → O2

TOP-K MERGE (在 CP 域内)
  本地 top-K       : tournament + sorted 输出, ~27 M vec ops / step (~105 µs)
  全局合并方法 A   : all-gather + 冗余 sorted-merge, ~128 KB (N=8), ~3 µs
  全局合并方法 B   : tree reduce-with-merge, ~48 KB (N=8), 需自定义 collective
  page-table 映射 : ~2 M vec ops / step (~8 µs)
  → top-K 链路总开销 ~115 µs / step, 是 vec 主战场但不是瓶颈

TC GEMM TIME (per device per step, R=8, 11 层, §6)
  TC peak          : FP8  13.1 TFLOPS/core × 2 = 26.2 TFLOPS/device
                     BF16 6.55 TFLOPS/core × 2 = 13.1 TFLOPS/device
                     Lmem→TC BW 3.2 TB/s per core (× 2 cores)
  Indexer GEMM     : ~220 µs (FP8, 计算受限, AI≈16)
  Sparse MQA GEMM  : ~113 µs (BF16) / ~56 µs (FP8)
  DSA step (BF16)  : ~515 µs;  DSA step (FP8 latent): ~458 µs

2-CORE SPLIT (§6.4)
  Indexer    : N-split 强制 (head_dim=128 是 TC 最小粒度, K 不能切; M=I=8 太小)
  Sparse MQA : 依 latent K 位置选 ——
               · latent K in Gmem → M-split (Nh/2 头/core, K 自然 device 共享, 零跨核通信)
               · latent K in Lmem → N-split (与 indexer 统一, 末尾 LSE merge)
               · K-split 备选 (DMA 半-dim 友好时考虑)
  Lmem 布局  : 各 core private, 按 token round-robin 分给 2 core (N-split 路线)
              或 Lmem 只放 scratch, K 走 Gmem (M-split 路线)
```

---

## 9. 开放问题（V1_c 增量）

继承 V1_b §13 全部，新增：

1. **latent K 在 Gmem 的 KV-transfer 双缓冲**：R=8+fp8 时 Gmem 占用 80%，留给 transfer buffer 仅 ~0.4 GB → 是否够做接收 + 后台清理？需要 per-req transfer 时序模型。
2. **gather kernel 的 Gmem 寻址**：和 paged-page-table 的元数据放哪儿？建议放 Lmem（小，热查），实际 Gmem-read 走数据通道、Lmem-read 走元数据通道，两端不互扰。
3. **方法 A vs 方法 B 的实测拐点**：在 R=4 / N=4 时通信只差 2×（64 KB vs 32 KB），不一定值得为方法 B 写自定义 collective。R=8 / N=8 时差 2.7×（128 KB vs 48 KB），更值得。
4. **sorted-runs 在 vec 上的成本**：本地 top-K 强制按 logit 降序输出，相比"只取 top-K 但顺序任意"会增加多少 vec cycle？需要 benchmark。
5. **方法 D（直接用本地子集做 partial）的精度影响**：稀疏度高时 local top-K 的并集 ≈ global top-K，差异在浮点噪声内；需 ablation 验证是否能把全部全局合并通信干掉。
6. **2-core 切分实测（M/K/N 三选）**：§6.4 给的推荐（Gmem-K → M-split / Lmem-K → N-split）是工程判断，需要实测：
   - **M-split + Gmem-K**：两 core 各自 gather 一份 N_eff × 512 K 到 private Lmem scratch，Gmem read 数据量翻倍但仍 < 50 MB / step——Gmem BW 是否能 hide 这部分延迟？
   - **N-split + Lmem-K**：online-softmax 2-way merge 的 vec 工作 ≈ §5 LSE merge 的 1/8（只 2 路），实际延迟应该 < 10 µs / step，但跨核 shmem/Lmem 通路的带宽决定了它真实成本。
   - **K-split**：GEMM-1 后 reduce-sum 是 fp32 元素加法，跨核数据量 ~48 KB / 层 / req——这条路线在 DMA 引擎对 stride-2 dim 加载特别友好的硬件上才有优势。
7. **TC 数 / core 的确认**：§6.1 假设 1 TC / core（按 vec-core/core 类推），若实际是 2 TC / core 或 0.5 TC / core，§6.5 的 GEMM 时间预算按 ÷2 / ×2 缩放。
8. **indexer GEMM 的 R 个独立 B 矩阵**：当前估算把 R reqs 当 R 个独立 GEMM 串行算（共享 Q 但 K 不同）。如果 deep_gemm 的 paged FP8 MQA logits kernel 能把 R 个 GEMM 在 N 维 padding 后合批，理论上 issue overhead 会摊薄；需要实测哪种调度更接近 220 µs 上限。

---

## 10. 与 V1_b 的并存关系

V1_c 是 V1_b 在"4–8 req + 三窄方案"下的精简+深挖版：
- 容量结论：R=8 沿用 V1_b 的 Lmem 主线；**R=4 或 R=8+fp8 时 latent K 进 Gmem 是 V1_c 新主张**。
- 并行方案：V1_b 的 6 选 → V1_c 的 3 选；推荐由"统一推 O2"细化为"R=4→O4 / R=8→CP=8×DP=2 / 低延迟→O2"。
- top-K：V1_b 提了"分布式 top-K"，V1_c **具体给出方法 A / B 的通信、vec 工作量、推荐**。

如果 R 后续放宽到 12+，回看 V1_b（它把 12 req 跑过一遍）。
