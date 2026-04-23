# Zeus 缺失 ATen 算子 × 大算子来源分析

> 日期：2026-04-21
> 对照基线：`sglang_zeus_manual.md` 第 10.3 节 "仍缺失的关键 ATen 算子"
> 目的：把每个缺失算子"上溯"到它所属的大算子（attention metadata / allocator / sampler / scheduler / logits / overlap），说明为什么这些位置必须走 CPU，并给出对应的代码修改建议

---

## 概览表

| # | 缺失 ATen 算子 | 上游大算子（按出现次数排序） | CPU bounce 根因 | 每步数据量 |
|---|------------|---------------------------|----------------|---------|
| 1 | `aten::cumsum` | **Attention Metadata** (kv_indptr/qo_indptr)、**Logits Processor** (extend→last-token 索引)、**Forward Batch Info** (chunked prefix、extend_start_loc) | "per-seq 长度 → CSR 前缀和" 的标准模式；cumsum 是语义上必须在 tensor 上做的 reduce-scan，Zeus 无 native | 小 (bs×4B) 但 ≥5 处 |
| 2 | `aten::index.Tensor_out`（花式索引读） | **Allocator / alloc_for_decode** (req_to_token 行取 last_loc)、**Attention Metadata** (req_to_token 行取 kv_indices)、**Schedule Batch** (filter_batch 保留存活请求)、**Logits Processor** (extend 取每条请求最后 token 的 hidden) | `t[idx]`、`t[idx_i, :len]` 的底层 kernel；attention/sampling 所有"按 req 抓" 场景全走它 | **大头**：P0 整矩阵 64 MB + P1 bs 行 16 MB |
| 3 | `aten::argmax` | **Sampler** (greedy 分支) | greedy 采样 = logits argmax；无 kernel ⇒ 整个 logits 回 CPU | 9.3 MB/step |
| 4 | `aten::clamp` | **Overlap Utils** (解析 future_indices 前 clamp 负号)、**Forward Batch Info** (`clamp_position`: seq_lens-1 保正) | 边界保护（空 seq、future idx 负值）语义；Zeus 无 min/max clamp kernel | <1 KB/处 |
| 5 | `aten::where` | **Overlap Utils** (`_resolve_future_token_ids` 三态选择)、**Sampler** (NaN 检测替换) | "条件选择"是 overlap 调度和 NaN 健壮性的核心语义；Zeus 无 kernel | <1 KB/处 |
| 6 | `aten::arange` (device) | **Overlap** (future_indices 生成)、**Forward Batch Info** (extend_start_loc、positions)、**Allocator 初始化** (free_pages 全量列表) | 需要在 device 上直接生成等差数列；Zeus 只能 CPU arange 后 `.to(device)` | 初始化大块、运行时小 |
| 7 | `aten::neg` (int) | **Scheduler** (`-future_indices.indices` 标记 future) | overlap 机制用"负 id" 编码 future token；Zeus 只注册了 float neg | <1 KB/step |

> 备注：`aten::cat`、`aten::add/sub/mul`、`aten::sum/mean/norm`、`aten::index_put_` 均已由 torch_zeus 原生实现（2026-04 更新），不列入本次分析。

---

## 1. `aten::cumsum`（高优先级）

### 所属大算子

**1.1 Attention metadata 构建（CSR 前缀和）** — 最核心用途

| 位置 | 代码 | 用途 |
|----|------|-----|
| `layers/attention/zeus_backend.py:99` | `kv_indptr_cpu[1:] = torch.cumsum(seq_lens_cpu, dim=0).to(int32)` | Graph decode 路径构 kv_indptr |
| `layers/attention/zeus_backend.py:148` | 同上（非 graph 路径） | 运行时构 kv_indptr |
| `layers/attention/zeus_backend.py:167` | `qo_indptr[1:] = torch.cumsum(extend_seq_lens_cpu, dim=0)` | 构 qo_indptr（extend 模式） |
| `model_executor/forward_batch_info.py:1081` | `prefix_chunk_cu_seq_lens[:, 1:] = ...cpu().cumsum(1).to(device)` | Chunked MHA prefix 的累积长度 |
| `model_executor/forward_batch_info.py:1118` | `kv_indptr[1:] = torch.cumsum(seq_lens.cpu(), 0).to(device)` | `fetch_mha_one_shot_kv_indices` 兜底路径 |
| `model_executor/forward_batch_info.py:1258` | `extend_start_loc[1:] = torch.cumsum(s_cpu[:-1], 0)` | `compute_position_torch`（非 triton 后端） |

**1.2 Logits Processor（prefill 取每条请求最后一个 hidden state）**

| 位置 | 代码 |
|----|------|
| `layers/logits_processor.py:425` | `indices = (torch.cumsum(seq_lens_cpu, 0) - 1).tolist()` |

### 为什么必须走 CPU

`cumsum` 语义上是 **串行前缀和 reduction-scan**：输出位置 `i` 需要全部 `[0, i)` 的累加值；并行实现需要 segment-scan kernel。Zeus 当前无 native reduction-scan，**任何 `cumsum(tensor_on_zeus)` 都会隐式 D2H→CPU scan→H2D**。  
显式写 `seq_lens.cpu().cumsum(0).to(device)` 的目的是把一次隐式传输改成一次 **显式** 传输 + 若干后续 CPU 计算（因为 kv_indptr 构建后还要紧接着做 index/cat，都在 CPU 更便宜）。

### 代码修改建议

**短期（不改 torch_zeus）：** 全部 cumsum 位置合并成一个 `build_kv_metadata_on_cpu(seq_lens, req_pool_indices, req_to_token_mirror)` 工具函数，一次 `.cpu()` 就完成 cumsum + kv_indices gather + qo_indptr 三件事，避免现在一次 forward 多次 D2H。建议放到 `utils/zeus_ops.py`（第 8.5.4 节规划已提到但未建）。

**中期：** torch_zeus 注册 `aten::cumsum`（triton 实现 block-scan + inter-block）。预期位置：`operators/zenl/reduceOps.cpp` 旁边新增 `scan.cpp`。落地后可直接删除以上 6 处 `_is_zeus` 分支。

---

## 2. `aten::index.Tensor_out`（花式索引读取，高优先级）

### 所属大算子

**2.1 Allocator / alloc_for_decode（P1 热点，16 MB/step）**

| 位置 | 代码 |
|----|------|
| `mem_cache/common.py:467` | `r2t_rows_cpu = batch.req_to_token_pool.req_to_token[rpi_cpu].cpu()` |
| `mem_cache/common.py:468` | `last_loc = r2t_rows_cpu[torch.arange(bs), sl_cpu - 1].to(device)` |

**2.2 Attention metadata（P0 热点，64 MB/step — 整矩阵）**

| 位置 | 代码 |
|----|------|
| `layers/attention/zeus_backend.py:112` | `kv_indices_cpu[off:off+sl] = req_to_token_cpu[req_idx, :sl]`（graph 路径） |
| `layers/attention/zeus_backend.py:155` | `req_to_token_cpu[req_idx, : seq_lens_cpu[b]]`（非 graph） |

**2.3 Schedule Batch 管理（请求生命周期）**

| 位置 | 代码 |
|----|------|
| `managers/schedule_batch.py:1842-1845` | `req_pool_indices.cpu()[ki_cpu]`、`seq_lens.cpu()[ki_cpu]`、`orig_seq_lens.cpu()[ki_cpu]`、`output_ids.cpu()[ki_cpu]` |

**2.4 Logits Processor（prefill 取 per-req 最后 hidden）**

| 位置 | 代码 |
|----|------|
| `layers/logits_processor.py:437-448` | 用 Python `for` + `pruned_states[i].copy_(hidden_states[idx])` 逐行拷贝 |

### 为什么必须走 CPU

`t[idx]`、`t[idx_i, :len_i]` 这类"花式索引"（fancy/advanced indexing）在 PyTorch 里对应 `aten::index.Tensor_out`——需要 gather kernel。Zeus 没实现，任何此类访问都会隐式把 **被索引的 base tensor**（可能很大，比如整个 `req_to_token`）拷到 CPU。  
现在的 sglang 代码是"显式 CPU + 只拷需要的行"，对 P1 比隐式 fallback 好，但对 P0 仍做了整矩阵 `.cpu()`。

### 代码修改建议（**本次代码修改主力方向**）

**A. 在 `ReqToTokenPool` 内维护一份 CPU mirror**（推荐 — 同时消除 P0+P1 共 80 MB/step）
- `ReqToTokenPool.__init__` 在 zeus 上新增 `self.req_to_token_cpu = torch.zeros(size, max_context_len, dtype=int32)`（CPU pinned memory）
- `ReqToTokenPool.write(indices, values)` 同时写 CPU mirror 和 Zeus 本体（Zeus 本体走已原生的 `aten::index_put_`）
- 把 `zeus_backend.py:init_forward_metadata` 里的 `req_to_token.cpu()` 改成直接读 `self.req_to_token_cpu`，**砍掉 64 MB 整矩阵拷贝**
- 把 `mem_cache/common.py:alloc_for_decode` 的 Zeus 分支改成读 mirror 的对应行，**砍掉 16 MB 行拷贝**
- 入侵点：`mem_cache/memory_pool.py` 里的 `ReqToTokenPool`（~30 行新增） + `zeus_backend.py` / `mem_cache/common.py` 两处 diff（各 ~5 行）

**B. torch_zeus 侧注册 `aten::index.Tensor_out`**
- 位置：`torch_zeus/csrc/aten/operators/zenl/` 下新增 `index_select.cpp`
- 一旦就绪，2.1 / 2.2 / 2.3 / 2.4 四处 `_is_zeus` 分支都可删除（但 A 方案对 P0/P1 的效果一样，先做 A 代价更低）

---

## 3. `aten::argmax`（中优先级，9.3 MB/step）

### 所属大算子

**Sampler — greedy 分支**

| 位置 | 代码 |
|----|------|
| `layers/sampler.py:103` | `batch_next_token_ids = torch.argmax(logits, -1)` |

### 为什么必须走 CPU

Greedy 采样就是选 `argmax(logits, dim=-1)`。Zeus 没有 `aten::argmax` 注册，`torch.argmax(logits_on_zeus, -1)` 会触发隐式 fallback：把 `[bs, vocab] = [32, 151936]` bf16 logits（9.3 MB）搬到 CPU 做 argmax，再把 `[bs]` 结果搬回。

### 代码修改建议

**A. 复用已有 zeus sampling kernel**（零新 kernel，最快）
- `sgl_kernel_zeus.sampling_from_logits` 已支持 temperature + softmax + top-k/p + sample 融合。可以用 **temperature 极小值**（例如 `1e-4`）模拟 greedy：logits 除以极小数 → softmax 变成 one-hot → sample 必取最大。
- 修改 `layers/sampler.py` 的 is_all_greedy 分支：当 `_is_zeus` 时走 `zeus_sampling_from_logits`（需要准备一个 all-ones temperatures = ε 的 buffer）。

**B. 新增 `sgl_kernel_zeus.argmax`**（最干净）
- 独立的 block-reduce kernel，运行时远快于 softmax 链路。

两个方案可以并存：先 A 消除瓶颈，等 B 就位再切换。

---

## 4. `aten::clamp`（低优先级）

### 所属大算子

**4.1 Overlap Scheduler — 解析 future token 负索引**

| 位置 | 代码 |
|----|------|
| `managers/overlap_utils.py:24` | `buf[torch.clamp(-ids, min=0)]`（CPU 侧，在 zeus 分支内） |
| `managers/overlap_utils.py:29` | `future_token_ids_map[torch.clamp(-input_ids, min=0)]`（非 zeus 走这条） |

**4.2 Forward Batch Info — position 计算**

| 位置 | 代码 |
|----|------|
| `model_executor/forward_batch_info.py:1278` | `torch.clamp((seq_lens.cpu() - 1), min=0).to(int64).to(device)` |

### 为什么必须走 CPU

`clamp` 是"元素级 min/max 夹取"；Zeus 当前只注册了 add/sub/mul 等点积算术，没有 clamp。Overlap 里 `-input_ids` 会产出 ≤0（正 token id 变负）和 ≥0（future 索引保持），需要 clamp 到 0 避免负下标；position 需要 `seq_lens - 1` 在空 seq 下保非负。

### 代码修改建议

**torch_zeus 侧注册 `aten::clamp` / `aten::clamp.Tensor`**（点积扩展，工作量小）
- 可以在已有的 dispatch stub 模式下加一层（参考 add/sub/mul 的 stub）。
- 落地后可去掉 `forward_batch_info.py:clamp_position` 和 `overlap_utils.py:_resolve_future_token_ids` 的 zeus 分支。

短期替代：如果对 position 的 clamp 值在 CPU 已经拿到（`seq_lens_cpu` 本就维护在 CPU），可直接用 `seq_lens_cpu - 1` 再 max(0)，不需要 device 侧 clamp。

---

## 5. `aten::where`（低优先级）

### 所属大算子

**5.1 Overlap Scheduler — future vs real token 三态选择**

| 位置 | 代码 |
|----|------|
| `managers/overlap_utils.py:24` | `torch.where(ids < 0, buf[clamp(-ids, 0)], ids)` |

**5.2 Sampler — NaN 替换**

| 位置 | 代码 |
|----|------|
| `layers/sampler.py:65` | `torch.where(torch.isnan(logits), -1e5, logits)` |

### 为什么必须走 CPU

`aten::where` 是"条件选择"，需要 mask + select kernel。Zeus 没实现。Overlap 的 future-token 解析依赖它（负值是 future 索引，正值是真实 token）；NaN 健壮性检查在 `logits` 上 apply mask 替换。

### 代码修改建议

**torch_zeus 侧注册 `aten::where.self` / `aten::where.ScalarOther`**。
- NaN 替换这条可以由 sampler 在 `_preprocess_logits` 判空；极端情况下才会触发（enable_nan_detection 默认关闭），短期影响较小。
- Overlap 是默认开启的热路径，优先级高于 NaN。

---

## 6. `aten::arange` (device 版本)（低优先级）

### 所属大算子

**6.1 Overlap Scheduler — future_indices 生成**

| 位置 | 代码 |
|----|------|
| `managers/overlap_utils.py:119` | `torch.arange(start, end, int64).to(device)`（已显式走 CPU 了） |

**6.2 Forward Batch Info — extend_start_loc / positions**

| 位置 | 代码 |
|----|------|
| `model_executor/forward_batch_info.py:844-846` | `torch.arange(bs, int32).to(device)` |
| `model_executor/forward_batch_info.py:1252` | `torch.arange(p, p+s)`（CPU 构，后 `.to(device)`） |

**6.3 Allocator 初始化**

| 位置 | 代码 |
|----|------|
| `mem_cache/zeus_allocator.py:107` | `torch.arange(0, num_pages, int64)` (CPU，作为 free_pages) |
| `mem_cache/zeus_allocator.py:131` | `out_pages[:, None] * page_size + torch.arange(page_size)`（CPU） |

### 为什么必须走 CPU

`torch.arange(..., device="zeus")` 需要 device 上生成等差数列；Zeus 没注册，隐式 fallback。当前 sglang 改成 `torch.arange(...)` + 显式 `.to(device)`，避开了隐式路径。

### 代码修改建议

**torch_zeus 侧注册 `aten::arange.start_step` / `aten::arange.default`**（实现简单，iota 类 kernel）。  
落地后可把上述 6.1/6.2 两处的 `.to(device)` 去掉，让 arange 在 device 上直接生成；6.3 allocator 因为保留 CPU 影子状态的设计，不需要改（本来就在 CPU 上）。

---

## 7. `aten::neg`（int tensor 版本，低优先级）

### 所属大算子

**Scheduler — future indices 标记**

| 位置 | 代码 |
|----|------|
| `managers/scheduler.py:2063` | `(-future_indices.indices.cpu()).to(device)` |

### 为什么必须走 CPU

Overlap 调度机制用"负整数"标记 future token（正 = 真实 token，负 = future 索引）。`-int_tensor` 对应 `aten::neg.Tensor` (int dtype)。torch_zeus 当前 neg 的 dispatch 只覆盖 float，int 走 fallback。

### 代码修改建议

**扩充 torch_zeus neg dispatch 到 int dtype**（约等于 add stub 复用，< 20 行 C++）。  
短期：因为 `future_indices.indices` 本身长度是 bs × int64，bounce 数据量极小（<1 KB），可以搁置。

---

## 大算子 × 缺失算子 交叉矩阵

| 大算子模块 | cumsum | index.Tensor_out | argmax | clamp | where | arange | neg | 总计 |
|---------|:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Attention metadata (`zeus_backend.py`, `forward_batch_info.py 1080/1117`) | ✓✓✓ | ✓✓ | | | | | | 5 |
| Allocator / alloc_for_decode (`mem_cache/common.py`) | | ✓ | | | | | | 1 |
| Sampler (`sampler.py`) | | | ✓ | | ✓ | | | 2 |
| Schedule Batch filter/merge (`schedule_batch.py`) | | ✓ | | | | | | 1 |
| Logits Processor (`logits_processor.py`) | ✓ | ✓ | | | | | | 2 |
| Overlap Scheduler (`overlap_utils.py`, `scheduler.py`) | | | | ✓ | ✓ | ✓ | ✓ | 4 |
| Forward Batch Info (position/pad) | ✓ | | | ✓ | | ✓ | | 3 |

**两条清晰的主线：**

1. **Attention metadata + Allocator** 这条线被 `cumsum` 和 `index.Tensor_out` 两兄弟绑死（P0+P1 = 90% 传输量）。解决方向是 **ReqToTokenPool 维护 CPU mirror + 补 cumsum/index kernel**。
2. **Overlap 调度** 这条线被 `clamp/where/arange/neg` 的"小 kernel 全家桶"绑死（数据量小但每步都有）。解决方向是一次性补齐 torch_zeus 的点积 + 条件 + 生成类 kernel。

其他（Sampler argmax、Logits Processor）是孤立问题，可单独处理。

---

## 建议的代码修改优先级

| 优先级 | 动作 | 触达缺失算子 | 预期收益 |
|------|-----|---------|---------|
| 🔴 P0 | 在 `ReqToTokenPool` 里加 CPU mirror；`zeus_backend.init_forward_metadata` 和 `mem_cache/common.alloc_for_decode` 改读 mirror | `index.Tensor_out` 两处 | -80 MB/step |
| 🔴 P0 | `sampler.is_all_greedy` 分支在 zeus 上走 `zeus_sampling_from_logits` (ε temp) 替代 argmax | `argmax` | -9.3 MB/step |
| 🟡 P1 | torch_zeus 注册 `aten::cumsum` | `cumsum` 全部 6 处 | 解锁未来在 device 侧构 metadata |
| 🟡 P1 | torch_zeus 注册 `aten::index.Tensor_out` | `index.Tensor_out` 4 处 | 可替代 P0 方案 A，允许删除所有 `_is_zeus` 索引分支 |
| 🟢 P2 | torch_zeus 点积全家桶：`clamp` / `where` / `arange.start_step` / `neg.int` | clamp/where/arange/neg | 清理 overlap 路径 `_is_zeus` 分支 |
| 🟢 P2 | 建 `utils/zeus_ops.py` 集中工具层 | — | 把分散在 15+ 文件的 44 个 bounce 收编 |

---

## 附录：`ZeusPagedTokenToKVPoolAllocator` 函数级分析

> 补充说明：本节把 `mem_cache/zeus_allocator.py`（以及 `mem_cache/allocator.py` 中 Zeus 走到的辅助函数）里每个函数的职责、在推理流水中的调用节拍、以及与缺失 ATen 算子的关联逐一展开。读者读完本节后，应能回答"为什么 Zeus 需要一个独立于父类的 allocator"。

### 0. 背景：KV Cache 的分页分配模型

LLM 每层 K/V tensor 要持续追加。SGLang 用 **Paged KV Cache**：把 token 池按 `page_size`（Zeus=128）切成等大 page，每条请求按需"申请若干 page"拼出它的 KV 序列。Allocator 管的是 page 级别的 free list，加上把分配结果翻译成 token 粒度的槽位索引 `out_indices`；上层再把 `out_indices` 写进 `req_to_token[req_idx, :]` 和 KV buffer。

### 0.1 为什么需要一个 Zeus 专用 allocator？

父类 `PagedTokenToKVPoolAllocator`（`mem_cache/allocator.py`）的 free_pages 常驻 device，alloc/free 过程用到 `cumsum / arange / cat / sort / unique / 花式索引`——这些在 Zeus 上恰好**全部**落在本文第 1、2、6 节的缺失算子清单里。每调用一次 alloc/free 会触发 6+ 次隐式 D2H↔H2D。Zeus 分叉出来的 `ZeusPagedTokenToKVPoolAllocator` 把所有记账常驻 CPU，只有最终 `out_indices` 才 `.to(zeus_device)`——把多次隐式 bounce 压缩成一次显式 H2D。

### 1. `ZeusPagedTokenToKVPoolAllocator.__init__`

**功能**：调父类构造，然后立刻把 `self.free_pages` 和 `self.release_pages` 迁到 CPU；`self._zeus_device` 记目标设备。

**推理流程里的作用**：**Scheduler 启动时构造一次**。此时根据 `profile_max_num_reqs` 估出来的可用 KV 槽位数 `size` 初始化 free page 池（如 `size / page_size` = 1024 个 page）。

### 2. `clear()`

**功能**：`free_pages = torch.arange(0, num_pages, int64)`（CPU），清空 `release_pages` 和 `free_group`，`is_not_in_free_group = True`。

**推理流程里的作用**：batch 全部完成或测试 reset 时调用，"清空 KV 池"。生产热路径上不会频繁触发。

### 3. `merge_and_sort_free()`

**功能**：把延迟释放队列 `free_group`（list[tensor]）和 `release_pages` `torch.cat` 进 `free_pages` 再 `torch.sort`。

**推理流程里的作用**：当 alloc 检测到 free_pages 不足时触发的 **lazy GC**。为什么延迟合并——decode/extend 过程中频繁释放（请求结束、radix cache 淘汰），每次都 sort 成本高；改成只在"真的不够用"时 flush 一次。对应调度节拍：**批次末尾调度新请求时的按需整理**。

### 4. `alloc(need_size)`

**功能**：从 free_pages 头部取 `need_size / page_size` 个 page，展开成 token 槽位：
```python
out_indices = (out_pages[:, None] * page_size + torch.arange(page_size)).reshape(-1)
return out_indices.to(zeus_device)
```

**推理流程里的作用**：给 **token-slot 级连续分配**用的兜底接口，在 `page_size == 1` / non-paged 路径或 radix cache 的大块写入时调用。Zeus 默认 `page_size=128` 的标准链路里不走这条。

### 5. `alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens)` — **Prefill 阶段的 KV 分配**

**功能**：
1. 把 prefix/seq/last_loc 都迁 CPU 做整数运算
2. 计算新增 page 数 `num_new_pages = Σ (ceil(sl/ps) - ceil(pl/ps))`
3. free page 不够先 `merge_and_sort_free()`，仍不够则返回 `None` 让 scheduler 重排
4. 委托给 `_alloc_extend_naive` 按 3 段拼出 `out_indices`（见第 8 节）
5. 从 free_pages 砍掉前 `num_new_pages` 个
6. `out_indices.to(zeus_device)` 返回

**推理流程里的作用**：**Prefill（或 chunked prefill 的每一片）** 进来时一次性为新 tokens 开空间。一条从 `prefix=0 → seq=1024` 的请求，就是这里一次性申请 1024 个 token 槽位，返回长度 1024 的 `out_cache_loc`；上层把它写进 `req_to_token[req_idx, 0:1024]`，attention kernel 后续按页表读写。

### 6. `alloc_decode(seq_lens, seq_lens_cpu, last_loc)` — **Decode 阶段每步必调**

**功能**：decode 每条请求只加 1 个 token，绝大多数情况**不需要新 page**（写到 `last_loc + 1` 即可）；只有请求恰好到 page 边界（`seq_len % page_size == 1`）时才需要新 page。代码核心：
```python
need_new_pages = (sl % page_size == 1).int()   # 每条 req 是否需要新 page
out_indices    = (ll + 1) * (1 - need_new_pages)         # 不需要：续写
               + free_pages[start_new_pages] * page_size \
               * need_new_pages                          # 需要：新 page 的 [0]
```

**推理流程里的作用**：**每个 decode step 必调**。LLM 自回归生成"下一 token"时扩 KV 一格的操作。bs=32 的 batch 里可能只有 1~2 条跨 page，其余 30 条就是简单的 `last_loc + 1`。这也是为什么这条路径被极度优化为 **CPU-only 记账**：一次 D2H（last_loc / seq_lens，≤256B）+ 一次 H2D（out_indices，256B）。

### 7. `free(free_index)`

**功能**：
1. `free_index_cpu = free_index.cpu()`
2. `torch.unique(free_index_cpu // page_size)` 算出要还的 page id（`unique` 用在 Zeus 上会 fallback，所以才要在 CPU 做）
3. 非 free_group 模式：直接 `cat` 进 `release_pages`（need_sort）或 `free_pages`（!need_sort）
4. free_group 模式：追加到 `self.free_group` 列表，等 `merge_and_sort_free` 统一处理

**推理流程里的作用**：请求**生成完成 / 被 evict / radix cache 降级**时调用。传入的是 token 槽位粒度（因为 page 可能被多条请求前缀共享），内部 `// page_size + unique` 转成 page 粒度再还池子。

### 8. `_alloc_extend_naive(prefix_lens, seq_lens, last_loc, free_pages, out_indices, page_size)` — 模块级工具

**功能**：`alloc_extend` 的核心算法纯 Python CPU 实现，按 3 段把 extend_num_tokens 个槽位铺出来：
- **part1 "补齐"**：prefix 所在 page 剩下的槽位，续写 `last_loc + 1, +2, ...`
- **part2 "整 page 批量"**：取 N 个整 page，每个填满 page_size 个槽
- **part3 "尾部不满 page"**：取一个新 page，但只用开头几格

**推理流程里的作用**：对应 CUDA 路径的 `alloc_extend_kernel`（triton）。Zeus 上 triton 不可用，纯 Python for-loop 在 CPU 上跑——因为 bs 通常 ≤32，batch 内串行成本可接受。

### 9. `_to_cpu_tensor(x)` — 模块级工具

**功能**：把 list / CPU tensor / device tensor 统一归一化成 CPU int64 tensor。

**作用**：allocator 入口的防御性转换——上层有时传 CPU 镜像的 `seq_lens_cpu`，有时传 Zeus tensor，有时是 Python list，这里统一接住。

### 10. 与推理大图的映射

```
Prefill 请求到达
  └── alloc_extend(prefix=0, seq=prompt_len)     ──▶ §5
        out_cache_loc[prompt_len] → ReqToTokenPool.write(req_idx, 0:prompt_len, ...)
        → KV kernel 按 out_cache_loc 写 K/V buffer

每步 Decode（循环至 EOS）
  └── alloc_decode(seq_lens, last_loc)           ──▶ §6
        out_cache_loc[bs] → ReqToTokenPool.write(req_idx, seq_lens, ...)
        → KV kernel 写入本 step 的 K/V

请求结束
  └── free(req 的全部 token 槽位)                ──▶ §7
        page id unique 后回收

(偶发) free 累积过多或 free page 不足
  └── merge_and_sort_free()                     ──▶ §3
```

### 11. 与缺失 ATen 算子的关联

| allocator 函数 | 依赖的缺失算子 | 现状 |
|-------------|-------------|-----|
| `__init__` / `clear` / free_pages 初始化 | `aten::arange` | 显式 CPU arange，已规避 |
| `merge_and_sort_free` | `aten::cat`(已原生)、`aten::sort`、`aten::unique` | cat 已不需要 CPU；sort/unique 因状态已在 CPU，无影响 |
| `alloc` / `alloc_extend` | `aten::arange`、`aten::cumsum`、页表 gather（等价于 `index.Tensor_out`） | 全部在 CPU 做 |
| `alloc_decode` | `aten::cumsum`、`aten::where`（被 `(1 - need_new_pages)` 配合算术巧妙避开） | **已无 where 依赖**，这是该函数优化的亮点 |
| `free` | `aten::unique`、`aten::cat`(已原生) | unique 在 CPU 做 |
| `_alloc_extend_naive` | 纯 Python 循环 | 不依赖 ATen |

**推论**：一旦 torch_zeus 补齐了 `aten::cumsum` + `aten::index.Tensor_out` + `aten::arange.start_step` + `aten::unique`（以及已完成的 cat），`ZeusPagedTokenToKVPoolAllocator` 这层分叉**就可以退化为父类的一层薄 wrapper**（最多保留 `_alloc_extend_naive` 作为 triton 不可用时的兜底），free_pages 也可以回到 device 上，和 CUDA 路径对齐。这也是文档 §8.5.1 "清理 SGLang 中已失效的 `_is_zeus` 分支" 的终极形态。

---

## 参考

- `sglang_zeus_manual.md` §7.1、§8.5.1、§10.3
- `torch_zeus_operator_analysis-20260415.md`（torch_zeus 侧算子清单，与本文档互补）
- `mem_cache/zeus_allocator.py` / `mem_cache/allocator.py`（本附录源码对象）
