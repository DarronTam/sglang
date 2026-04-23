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

## 4.1 主链路仍存在的问题

### A. `layers/attention/zeus_backend.py`

参考：

- [zeus_backend.py:94](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:94)
- [zeus_backend.py:103](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:103)
- [zeus_backend.py:141](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:141)

当前仍存在：

1. graph 路径里：
   - `seq_lens[:bs].cpu()`
   - `req_pool_indices[:bs].cpu()`
   - `self.req_to_token.cpu()`
2. non-graph 路径里：
   - `cumsum(seq_lens_cpu)`
   - `cumsum(extend_seq_lens_cpu)`

为什么还在：

- graph 路径仍直接从 device `req_to_token` 读
- `cumsum` 缺失

能否直接删：

- **不能整体直接删**

怎么消除：

- 短期：graph 路径也改读 `req_to_token_cpu mirror`
- 中期：补 `build_kv_metadata` Zeus 专用 kernel 或补 `cumsum + index(read)`

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
- `prefix_chunk_seq_lens_cuda.cpu().cumsum(...)`
- `torch.cumsum(self.seq_lens.cpu(), dim=0).to(...)`
- `compute_position_torch()` 里的 CPU `arange + cumsum`
- `clamp_position()` 里的 CPU `clamp`

为什么还在：

- `cumsum`、`arange(device)`、`clamp` 缺失

能否直接删：

- **不能整体直接删**

怎么消除：

- 先补 `cumsum`
- 再补 `arange`
- 再补 `clamp`

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

- `cumsum` 缺失
- `index(read)` 缺失
- `arange` 缺失

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

### P1：继续优化主链路

1. `zeus_backend.py` graph 路径改读 `req_to_token_cpu mirror`
2. 视情况清理 `common.py` 里 `locs = seq_lens.cpu().to(device)` 这种轻量残留

### P2：补缺失算子

建议顺序：

1. `cumsum`
2. `index(read)` / gather
3. `arange`
4. `where`
5. `clamp`
6. `neg(int)`

---

## 7. 一句话总结

这份文档按最新代码重排后，最重要的变化是：

> `req_to_token` 读写相关的主链路大块 CPU bounce 已经被解决，不应再继续作为“当前未解决问题”统计。  
> 现在还真正残留的热点，主要是 `cumsum`、`index(read)` 驱动的 metadata / batch / logits 路径，以及 overlap 上的 `clamp/where/arange/neg` 小算子分支。
