# torch_zeus 已支持算子与 SGLang `_is_zeus` CPU bounce 审计

> 日期：2026-04-23  
> 本次更新：按**当前代码**重新整理，只统计**仍然存在**的 CPU bounce；已被最新代码修复的项不再列为现存问题。

---

## 1. 当前代码结论

基于最新代码，Zeus 路径里的 CPU 行为现在应分成 3 类：

1. **已解决，不再计入当前 CPU bounce**
   - `ReqToTokenPool.write()` 的整表 CPU roundtrip 已去掉
   - `alloc_for_decode()` 从 `req_to_token` 读 `last_loc` 的行拷贝已去掉
   - `ZeusAttnBackend.init_forward_metadata()` 的 non-graph 整表 `req_to_token.cpu()` 已去掉
   - greedy `argmax` 不再有 Zeus 专门的 CPU workaround

2. **仍存在的 CPU bounce**
   - 这些分支背后依赖的算子当前仍缺失
   - 典型：`cumsum`、`index` 读取、`clamp`、`where`、`arange`、`neg(int)`

3. **设计上保留的 CPU bookkeeping**
   - 这些不是“不必要 bounce”
   - 而是 Zeus 分支主动保留在 CPU 的状态管理
   - 典型：`ZeusPagedTokenToKVPoolAllocator`

一句话：

> 当前主链路最明显的 `req_to_token` 读写 bounce 已经被新代码消掉了；现在更值得继续关注的是 `cumsum / index(read)` 驱动的 metadata、batch 处理和 overlap 小算子路径。

---

## 2. 本地文档汇总：`torch_zeus` 相关算子现状

本节只保留对当前审计仍有影响的算子。

### 2.1 已确认可直接用的关键算子

依据：

- [torch_zeus_operator_analysis-20260415.md](/root/workspace/sglang/zeus_dev/torch_zeus_operator_analysis-20260415.md:1)
- [sglang_zeus_manual.md](/root/workspace/sglang/zeus_dev/sglang_zeus_manual.md:1131)

| 类别 | 算子 | 当前判断 |
|---|---|---|
| 拼接 | `cat`, `cat.out` | ✅ 原生 |
| 算术 | `add`, `sub`, `mul` | ✅ 原生 |
| 归约 | `sum`, `mean`, `norm` | ✅ 原生 |
| GEMM | `mm`, `addmm`, `linear` | ✅ 原生 |
| 索引写入 | `index_put_`, `index_put` | ✅ 原生 |
| greedy 归约 | `argmax` | ✅ 当前代码已直接使用 |

### 2.2 仍缺失、并且仍影响当前代码的关键算子

| 算子 | 当前影响 |
|---|---|
| `cumsum` | attention metadata、forward_batch_info、logits_processor |
| `index.Tensor_out` / `tensor[indices]` | graph metadata、schedule_batch.filter、logits gather |
| `clamp` | overlap、position |
| `where` | overlap、NaN 替换 |
| `arange(device=zeus)` | overlap、forward_batch_info |
| `neg(int)` | overlap future index 标记 |

### 2.3 关于 `argmax`

本地旧文档存在时间差，但**当前代码现状**是：

- [sampler.py:101](/root/workspace/sglang/python/sglang/srt/layers/sampler.py:101) 直接调用 `torch.argmax(logits, -1)`
- 当前 `sampler.py` 不再有 Zeus 专门的 `logits.cpu() -> argmax -> to(device)` workaround

因此本审计中，`argmax` **不再计入当前仍存在的 CPU bounce**。

---

## 3. 已解决：不再计入当前 CPU bounce 统计

### 3.1 `ReqToTokenPool.write()` 整表 CPU roundtrip

当前代码：

- [memory_pool.py:107](/root/workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py:107)

现状：

- Zeus 下已改为：
  - CPU mirror 同步
  - `self.req_to_token[indices] = values`

因此：

- **不再存在** `req_to_token.cpu() -> CPU index_put -> to(device)` 整表 bounce

### 3.2 `alloc_for_decode()` 的 `last_loc` 行拷贝

当前代码：

- [common.py:461](/root/workspace/sglang/python/sglang/srt/mem_cache/common.py:461)

现状：

- Zeus 下已改为直接读 `req_to_token_cpu[rpi_cpu]`

因此：

- **不再存在** 每步从 device `req_to_token` 拉行到 CPU 的 bounce

### 3.3 `ZeusAttnBackend.init_forward_metadata()` 的 non-graph 整表拷贝

当前代码：

- [zeus_backend.py:136](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:136)

现状：

- non-graph 路径已改为直接读 `forward_batch.req_to_token_pool.req_to_token_cpu`

因此：

- **不再存在** non-graph metadata 构建前的整表 `req_to_token.cpu()`

### 3.4 greedy `argmax` CPU workaround

当前代码：

- [sampler.py:101](/root/workspace/sglang/python/sglang/srt/layers/sampler.py:101)

现状：

- 直接 `torch.argmax`
- 无 Zeus CPU workaround 分支

因此：

- **不再计入当前 CPU bounce**

---

## 4. 当前仍存在的 Zeus CPU bounce

本节只统计**最新代码里还存在**的项。

## 4.0 主链路 vs Overlap / Scheduler：功能边界与调用流

这两条路线虽然都会影响一次推理 step 的执行，但职责并不相同：

- **主链路**：负责“真实 token 如何进入 KV cache，并被后续 attention 消费”
- **Overlap / Scheduler 路线**：负责“在真实结果尚未完全落地前，如何用 future 占位把调度和前向重叠起来”

换句话说：

- 主链路偏 **模型状态推进 / cache 存取**
- Overlap / Scheduler 偏 **异步调度 / 结果占位与回填**

### 4.0.1 主链路在做什么

主链路完成的是一次 decode / extend 中，和 paged KV cache 直接相关的工作：

1. 根据当前请求长度，为新 token 分配 KV slot
2. 把 `req -> token -> kv_slot` 写进 `req_to_token`
3. 从 `req_to_token` 组装出 attention 所需的 `kv_indptr / kv_indices / qo_indptr`
4. 让 paged attention kernel 依据这些 metadata 读取 KV cache

它的核心目标是：

- **让“这个 token 写到哪里、之后 attention 去哪里读”保持一致**

### 4.0.2 主链路调用流图

```text
Decode / Extend Main Path

ScheduleBatch
   |
   | seq_lens / req_pool_indices / reqs
   v
alloc_for_decode()                       [mem_cache/common.py]
   |
   | read last_loc from req_to_token_cpu mirror
   v
ZeusPagedTokenToKVPoolAllocator
   |
   | CPU bookkeeping:
   | free_pages / release_pages / page alloc result
   v
out_cache_loc
   |
   | device index_put + CPU mirror sync
   v
ReqToTokenPool.write()                   [mem_cache/memory_pool.py]
   |
   | req_to_token[req, token_pos] = kv_slot
   v
ZeusTokenToKVPool.set_kv_buffer()
   |
   | store_kv_cache kernel writes K/V data
   v
Paged KV Cache
   |
   | build kv_indptr / kv_indices / qo_indptr
   v
ZeusAttnBackend.init_forward_metadata()  [layers/attention/zeus_backend.py]
   |
   | attention metadata
   v
extend_attention / decode_attention
   |
   v
Paged attention reads KV cache
```

### 4.0.3 主链路模块说明

| 模块 | 作用 | 典型数据 |
|------|------|----------|
| `alloc_for_decode()` | 组织 decode 分配前后的页表读写 | `seq_lens`、`req_pool_indices`、`last_loc` |
| `ZeusPagedTokenToKVPoolAllocator` | 维护页级分配/回收 bookkeeping | `free_pages`、`release_pages`、`out_cache_loc` |
| `ReqToTokenPool` | 维护请求到 KV slot 的页表 | `req_to_token`、`req_to_token_cpu` |
| `ZeusTokenToKVPool` | 真正把 K/V 写进 paged KV cache | `out_cache_loc`、K/V tensors |
| `ZeusAttnBackend` | 从页表构 attention metadata | `kv_indptr`、`kv_indices`、`qo_indptr` |
| paged attention kernel | 按 metadata 读取 KV cache | `kv_indices` 指向的 cache pages |

### 4.0.3.1 主链路流程图（带 CPU bounce 标注）

```text
Decode / Extend Main Path

ScheduleBatch
   |
   | seq_lens / req_pool_indices / reqs
   v
alloc_for_decode()                       [mem_cache/common.py]
   |
   | read last_loc from req_to_token_cpu mirror
   | [已优化] 不再做 req_to_token.device -> cpu 的行读取 bounce
   v
ZeusPagedTokenToKVPoolAllocator
   |
   | CPU bookkeeping:
   | free_pages / release_pages / page alloc result
   | [设计保留] 这是 CPU bookkeeping，不归类为“不必要 bounce”
   v
out_cache_loc
   |
   | device index_put + CPU mirror sync
   | [已优化] 不再做 req_to_token 整表 cpu() -> 写 -> to(device)
   v
ReqToTokenPool.write()                   [mem_cache/memory_pool.py]
   |
   | req_to_token[req, token_pos] = kv_slot
   v
ZeusTokenToKVPool.set_kv_buffer()
   |
   | store_kv_cache kernel writes K/V data
   | [device kernel] 无 CPU bounce
   v
Paged KV Cache
   |
   | build kv_indptr / kv_indices / qo_indptr
   | [已优化] indptr prefix-sum runs on Zeus:
   |   - cumsum(seq_lens)
   |   - cumsum(extend_seq_lens)
   | [仍存在] CPU mirror gather:
   |   - seq_lens.cpu()
   |   - req_pool_indices.cpu()
   |   - ragged gather from req_to_token_cpu mirror
   | [已优化] 不再有 req_to_token 整表 .cpu() bounce
   v
ZeusAttnBackend.init_forward_metadata()  [layers/attention/zeus_backend.py]
   |
   | attention metadata
   v
extend_attention / decode_attention
   |
   | [device kernel] paged attention reads KV cache
   | [无 CPU bounce]
   v
Paged attention output
```

### 4.0.4 Overlap / Scheduler 路线在做什么

Overlap / Scheduler 路线完成的是“先占位、后兑现”的异步调度：

1. scheduler 先为当前 batch 申请一批 future slots
2. 用负数 future index 暂时写进 `output_ids`
3. forward 真正跑完后，把结果写入 `FutureMap`
4. 后续谁要消费这些 token，就把负数占位符解引用成真实 token

它的核心目标是：

- **让调度准备、前向执行、结果消费尽量重叠，减少等待**

这里它管理的不是 KV cache，而是：

- “未来某一步会产生的 token / draft 结果”

### 4.0.5 Overlap / Scheduler 调用流图

```text
Overlap / Scheduler Path

Scheduler.run_batch()
   |
   | bs = len(model_worker_batch.seq_lens)
   v
FutureMap.alloc_future_indices(bs)      [managers/overlap_utils.py]
   |
   | allocate future slots: [f1, f2, ...]
   v
future_indices
   |
   | encode as negative placeholders
   v
(-future_indices.indices)               [managers/scheduler.py]
   |
   | write placeholder ids into batch state
   v
batch.output_ids / next-step inputs
   |
   | before forward: resolve old futures if input_ids contain negative refs
   v
FutureMap.resolve_future()
   |
   | where(ids < 0, future_buf[clamp(-ids)], ids)
   v
ModelWorker.forward_batch_generation()
   |
   | real next_token_ids / draft result produced
   v
GenerationBatchResult
   |
   | store real result into circular future buffer
   v
FutureMap.store_to_map()
   |
   v
Later batch preparation / later consumer resolves future ids
```

### 4.0.6 Overlap / Scheduler 模块说明

| 模块 | 作用 | 典型数据 |
|------|------|----------|
| `Scheduler` | 组织一次 batch 的调度、forward 与 future 占位 | `future_indices`、`output_ids` |
| `FutureMap.alloc_future_indices()` | 给未来结果分配 circular buffer 槽位 | `indices = [start, ..., end)` |
| `neg(int)` | 把 future slot 编码成负占位符 | `-future_indices` |
| `FutureMap.resolve_future()` | 把负占位符解引用为真实 token | `where + clamp + index(read)` |
| `FutureMap.store_to_map()` | forward 完成后把真实结果写回 future buffer | `next_token_ids` / eagle draft data |

### 4.0.6.1 Overlap / Scheduler 流程图（带 CPU bounce 标注）

```text
Overlap / Scheduler Path

Scheduler.run_batch()
   |
   | bs = len(model_worker_batch.seq_lens)
   v
FutureMap.alloc_future_indices(bs)      [managers/overlap_utils.py]
   |
   | allocate future slots: [f1, f2, ...]
   | [仍存在小 bounce] arange 在 Zeus 下仍走历史 CPU 特判
   v
future_indices
   |
   | encode as negative placeholders
   | [仍存在小 bounce] (-future_indices.indices.cpu()).to(device)
   v
(-future_indices.indices)               [managers/scheduler.py]
   |
   | write placeholder ids into batch state
   v
batch.output_ids / next-step inputs
   |
   | before forward: resolve old futures if input_ids contain negative refs
   v
FutureMap.resolve_future()
   |
   | where(ids < 0, future_buf[clamp(-ids)], ids)
   | [仍存在小 bounce]
   |   - input_ids.cpu()
   |   - future_token_ids_map.cpu()
   |   - CPU where / clamp / index(read)
   v
ModelWorker.forward_batch_generation()
   |
   | real next_token_ids / draft result produced
   | [device forward] 本身不是 CPU bounce 点
   v
GenerationBatchResult
   |
   | store real result into circular future buffer
   | [device write] 无 CPU bounce
   v
FutureMap.store_to_map()
   |
   v
Later batch preparation / later consumer resolves future ids
```

### 4.0.7 两条路线的关系

它们的关系可以概括为：

1. **Overlap / Scheduler 决定“结果还没正式落地时，系统怎么继续往前走”**
2. **主链路决定“结果一旦落地，这个 token 的 KV 写到哪里、attention 之后怎么读”**

所以：

- Overlap / Scheduler 主要解决 **流水线重叠**
- 主链路主要解决 **cache 一致性与 attention 消费**

也因此它们的 Zeus 改造重点不同：

- 主链路最关心：`req_to_token` 读写、metadata 构建、KV page attention
- Overlap / Scheduler 最关心：`where / clamp / arange / neg(int)` 这组小算子

## 4.1 主链路状态更新

### A. `layers/attention/zeus_backend.py`

参考：

- [zeus_backend.py:94](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:94)
- [zeus_backend.py:103](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:103)
- [zeus_backend.py:141](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:141)

按当前代码，这里的结论应拆成两层：

1. **主链路上“不必要的大块 CPU bounce”已经基本清掉**
   - graph / non-graph 两条 metadata 路径都已改为直接读 `req_to_token_cpu mirror`
   - 已经不存在为了 `kv_metadata` 构建而额外执行整表 `req_to_token.cpu()` 的 bounce
2. **attention metadata 的 CPU 计算范围进一步缩小**
   - `kv_indptr / qo_indptr` 的 `cumsum` 已回到 Zeus device
   - 仍在 CPU 上的是 `seq_lens / req_pool_indices` 小张量读取，以及基于 `req_to_token_cpu mirror` 的 ragged gather

为什么还要关注：

- 如果目标只是消除主链路上的明显 CPU bounce：这一阶段已经基本完成
- 如果目标是进一步减少 host 参与、靠近 CUDA 路径：仍需要 `build_kv_metadata` kernel，或补齐 `index(read)` 后把 ragged gather 搬回 device

怎么继续优化：

- 二阶段：补 `build_kv_metadata`
- 或者分步补齐 `index(read)`，逐步把 metadata 组装从 CPU 挪走

### B. `mem_cache/common.py`

参考：

- [common.py:485](/root/workspace/sglang/python/sglang/srt/mem_cache/common.py:485)

当前仍存在：

- `locs = batch.seq_lens.cpu().to(batch.seq_lens.device)`

为什么还在：

- 这不是大头，但当前写法仍保留了 Zeus 特判

能否直接删：

- **可能可以**

怎么消除：

- 如果验证 `locs = batch.seq_lens.clone()` 在 Zeus 下已无问题，可直接统一到普通路径

---

## 4.2 Overlap / Scheduler 路线

### A. `managers/overlap_utils.py`

参考：

- [overlap_utils.py:21](/root/workspace/sglang/python/sglang/srt/managers/overlap_utils.py:21)
- [overlap_utils.py:118](/root/workspace/sglang/python/sglang/srt/managers/overlap_utils.py:118)

当前仍存在：

- `ids = input_ids.cpu()`
- `buf = future_token_ids_map.cpu()`
- `torch.where(...)`
- `torch.clamp(...)`
- `torch.arange(...).to(self.device)`

为什么还在：

- `where` / `clamp` / `arange(device)` / `index(read)` 缺失

能否直接删：

- **不能**

怎么消除：

- 补齐 `where + clamp + arange + neg(int)` 这一组 overlap 小算子

### B. `managers/scheduler.py`

参考：

- [scheduler.py:2061](/root/workspace/sglang/python/sglang/srt/managers/scheduler.py:2061)

当前仍存在：

- `(-future_indices.indices.cpu()).to(device)`

为什么还在：

- `neg(int)` 缺失

能否直接删：

- **不能**

怎么消除：

- 补 `neg` 的 int dtype dispatch

---

## 4.3 Forward Batch Info 路线

参考：

- [forward_batch_info.py:841](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:841)
- [forward_batch_info.py:1080](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1080)
- [forward_batch_info.py:1117](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1117)
- [forward_batch_info.py:1246](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1246)
- [forward_batch_info.py:1276](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1276)

当前仍存在：

- `self.extend_prefix_lens = (self.seq_lens.cpu() - 1).to(device)`
- `torch.arange(...).to(device)`
- `compute_position_torch()` 里的 CPU `arange + cumsum`
- `clamp_position()` 里的 CPU `clamp`

为什么还在：

- `arange(device)`、`clamp` 缺失
- `compute_position_torch()` 仍混合 CPU `arange/cat`，不只依赖 `cumsum`

能否直接删：

- **不能整体直接删**

怎么消除：

- 已处理纯 `cumsum` 路径：
  - `prefix_chunk_cu_seq_lens`
  - `fetch_mha_one_shot_kv_indices()` 中的 `kv_indptr`
- 后续再补 `arange`
- 后续再补 `clamp`

---

## 4.4 Schedule Batch 路线

参考：

- [schedule_batch.py:1775](/root/workspace/sglang/python/sglang/srt/managers/schedule_batch.py:1775)
- [schedule_batch.py:1840](/root/workspace/sglang/python/sglang/srt/managers/schedule_batch.py:1840)
- [schedule_batch.py:1890](/root/workspace/sglang/python/sglang/srt/managers/schedule_batch.py:1890)

当前仍存在：

- `self.seq_lens = (self.seq_lens.cpu() + 1).to(device)`
- `self.orig_seq_lens = (self.orig_seq_lens.cpu() + 1).to(device)`
- `tensor.cpu()[ki_cpu].to(self.device)` 的 batch 过滤
- `torch.cat([x.cpu(), y.cpu()]).to(self.device)` 的 merge

为什么还在：

- `index(read)` 缺失
- 虽然 `cat` 已支持，但当前代码尚未清理历史分支
- `+1` 小算术分支也还没回归统一 device 路径

能否直接删：

- **部分能**

怎么消除：

- `cat` 相关分支：可以直接清理
- `+1` 小算术分支：若 Zeus 上 `add` 路径验证无误，也可直接清理
- `filter_batch` 的索引读取：仍需等 `index(read)` 或改造数据流

---

## 4.5 Logits Processor 路线

参考：

- [logits_processor.py:420](/root/workspace/sglang/python/sglang/srt/layers/logits_processor.py:420)

当前仍存在：

- `seq_lens_cpu = ...cpu()`
- `torch.cumsum(seq_lens_cpu, dim=0)`
- `torch.arange(len(seq_lens_cpu))`
- 用 Python `for` 循环逐行 `copy_(hidden[idx])`

为什么还在：

- `index(read)` 缺失
- `arange` 缺失
- 虽然 `cumsum` 已支持，但该分支还依赖后续 gather / index 逻辑，所以暂不单独修改

能否直接删：

- **不能**

怎么消除：

- 补 `cumsum`
- 补 `index(read)` / gather
- `arange` 配套解决

---

## 4.6 `radix_cache.py`

参考：

- [radix_cache.py:42](/root/workspace/sglang/python/sglang/srt/mem_cache/radix_cache.py:42)
- [radix_cache.py:416](/root/workspace/sglang/python/sglang/srt/mem_cache/radix_cache.py:416)
- [radix_cache.py:543](/root/workspace/sglang/python/sglang/srt/mem_cache/radix_cache.py:543)
- [radix_cache.py:631](/root/workspace/sglang/python/sglang/srt/mem_cache/radix_cache.py:631)

当前仍存在：

- `_cat_zeus()`：`torch.cat([t.cpu() for t in tensors]).to(device)`

为什么还在：

- 这是历史遗留，当前 `cat` 已支持

能否直接删：

- **能**

怎么消除：

- 删除 `_cat_zeus()`，直接统一成 `torch.cat`

---

## 5. 设计上保留的 CPU 路径：不计入“不必要 bounce”

### 5.1 `zeus_allocator.py`

参考：

- [zeus_allocator.py:84](/root/workspace/sglang/python/sglang/srt/mem_cache/zeus_allocator.py:84)

这里的：

- `free_pages`
- `release_pages`
- alloc/free bookkeeping

是 Zeus 分支主动保留在 CPU 的设计。  
它不是这份文档要优先清理的“不必要 CPU bounce”。

---

## 6. 当前代码下，哪些项应该优先做

### P0：继续清理已失效的历史分支

1. `radix_cache.py` 里的 `_cat_zeus()`
2. `schedule_batch.py` 里的 `cat` CPU 分支
3. `schedule_batch.py` / `common.py` 里仅依赖 `add/sub` 的小算术 Zeus 分支

这些项不依赖新 kernel，最容易清掉。

### P1：主链路第二阶段优化

1. 继续减少 attention metadata 的 host 计算
2. 视情况清理 `common.py` 里 `locs = seq_lens.cpu().to(device)` 这种轻量残留

### P2：补缺失算子 / 应用已支持算子

建议顺序：

1. `index(read)` / gather
2. `arange`
3. `where`
4. `clamp`
5. `neg(int)`
6. 将已支持的 `cumsum` 继续应用到 batch / logits 路径

## 6.1 剩余 `cumsum` 调用点优先级表

这里把当前 Zeus 还值得关注的 `cumsum` 调用点，按 `主链路 / batch / logits` 三类重新整理如下。

| 类别 | 代表位置 | 当前作用 | 优先级 | 原因 |
|------|----------|----------|--------|------|
| 主链路 | [zeus_backend.py:100](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:100), [zeus_backend.py:149](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:149), [zeus_backend.py:168](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:168) | 构 `kv_indptr / qo_indptr` | 已处理 | 已改为 Zeus device `torch.cumsum`；剩余问题转为 `kv_indices` 的 ragged gather |
| Batch | [forward_batch_info.py:1081](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1081), [forward_batch_info.py:1118](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1118) | 构 `prefix_chunk_cu_seq_lens`、`kv_indptr` | 已处理 | 这两处是纯 `cumsum` CPU 绕路，已改为 Zeus device `torch.cumsum` |
| Batch | [forward_batch_info.py:1258](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1258) | 构 `extend_start_loc` | 暂不处理 | 该分支还混合 CPU `arange/cat`，不只依赖 `cumsum` |
| Logits | [logits_processor.py:420](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/logits_processor.py:420) | 构最后 token 的线性下标 | 暂不处理 | 该分支还依赖 gather / index 逻辑，先不只改 `cumsum` |

补充说明：

- 主链路里的 `cumsum` 已经回到 Zeus device 上执行
- 纯 `cumsum` 的 batch 路径已处理
- logits 和混合 batch 路径还依赖其他 CPU/index 能力，暂不单独修改
- Paged KV metadata 的剩余 host 工作，核心已经变成 `kv_indices` 的 ragged gather

---

## 7. 一句话总结

这份文档按最新代码重排后，最重要的变化是：

> `req_to_token` 读写和 `kv_metadata` 相关的主链路大块 CPU bounce 已经被解决，不应再继续作为“当前未解决问题”统计。  
> 现在更准确的说法是：主链路上已基本没有“不必要的大块 CPU bounce”，`kv_indptr / qo_indptr` 的 `cumsum` 也已 device 化；Paged KV metadata 剩余的 host 工作主要是 `kv_indices` ragged gather。此外真正残留的热点，主要是 batch / logits 上的 `cumsum`、`index(read)` 路径，以及 overlap 上的 `clamp/where/arange/neg` 小算子分支。
