# Zeus 主链路 CPU Bounce 重新评估与实现方案

> 日期：2026-04-27  
> 前提更新：`torch_zeus` 已支持 `aten::index_put`  
> 目标：重新评估 `allocator + req_to_token 读写 + attention metadata + kv page attention` 这条链路上的 CPU bounce，并给出在当前前提下的最佳实现方案。

---

## 1. 结论先行
  
在 `torch_zeus` 已支持 `aten::index_put` 的前提下，**最优的近期方案不是先补 3 个 Zeus 专用 kernel**，而是：

1. 在 `ReqToTokenPool` 中为 Zeus 维护一份 **CPU mirror**：`req_to_token_cpu`
2. `ReqToTokenPool.write()` 在 Zeus 上：
   - 直接对 device tensor 执行 `self.req_to_token[indices] = values`
   - 同步更新 `req_to_token_cpu`
3. `alloc_for_decode()` 和 `ZeusAttnBackend.init_forward_metadata()` 改为直接读取 `req_to_token_cpu`
4. **保留 `ZeusPagedTokenToKVPoolAllocator` 的 CPU bookkeeping 设计**

原因是：

- 现在 `write_req_to_token` 的最大痛点已经被 `index_put` 原生支持化解，因此 **`write_req_to_token` 专用 kernel 不再是第一优先级**
- 当前最大的 bounce 已经集中在：
  - `alloc_for_decode()` 里的 `req_to_token[rpi].cpu()` 行拷贝
  - `zeus_backend.py` 里的 `req_to_token.cpu()` 整表拷贝
- 这两个问题都可以靠 **CPU mirror** 以极小改动解决
- allocator 这层虽然在 CPU 上运行，但它是 **Zeus 分支有意保留的 CPU bookkeeping**，不是当前最“不必要”的 bounce 大头

一句话概括：

> **先做 CPU mirror + 直接 device index_put，是当前收益/改动比最好的方案。**  
> `get_last_loc` / `build_kv_metadata` 专用 kernel 仍然值得做，但应排在第二阶段。  
> `write_req_to_token` 专用 kernel 由于 `aten::index_put` 已可用，优先级大幅下降。

---

## 2. 有无 `aten::index_put` 时的最优方案对比

`aten::index_put` 是否可用，会直接改变“最佳方案”的排序。

### 2.1 没有 `aten::index_put` 时

这时 `ReqToTokenPool.write()` 不能直接在 Zeus device 上写页表，主链路会卡在：

- device 上的 `req_to_token` 无法原地 scatter 写
- 只能走 `cpu() -> index_put -> to(device)` 整表往返

因此最佳方案通常是：

1. `write_req_to_token` Zeus 专用 kernel
2. `get_last_loc` Zeus 专用 kernel
3. `build_kv_metadata` Zeus 专用 kernel

原因：

- `write_req_to_token` 是最硬的 blocker
- 只做 CPU mirror 仍然无法避免 device 页表写回的整表 roundtrip
- 所以必须优先把“写页表”搬到 device 端

### 2.2 已有 `aten::index_put` 时

这时 `ReqToTokenPool.write()` 已经可以直接：

```python
self.req_to_token[indices] = values
```

因此最优方案变成：

1. `ReqToTokenPool` 增加 CPU mirror
2. `write()` 改成 device `index_put` + mirror 同步
3. `alloc_for_decode()` 直接读 mirror
4. `ZeusAttnBackend.init_forward_metadata()` 直接读 mirror
5. 二阶段再补 `build_kv_metadata`

原因：

- “写页表”的核心 blocker 已经消失
- 当前最明显的额外传输转而集中在“读页表”
- CPU mirror 能以最小改动同时覆盖 decode 的 `last_loc` 读取和 attention metadata 构建

### 2.3 对比结论

| 前提 | 最佳近期方案 | 为什么 |
|---|---|---|
| **没有 `aten::index_put`** | 优先补 `write_req_to_token`，再补 `get_last_loc` / `build_kv_metadata` | 页表写入本身就是 blocker，不先解决写就无法消掉整表 bounce |
| **已有 `aten::index_put`** | 优先做 `CPU mirror + device index_put`，再补 `build_kv_metadata` | 页表写入已通，剩余大头转为页表读取与 metadata 构建 |

因此，**在 2026-04-23 这个前提下，最佳方案已经从“先补 3 个专用 kernel”切换成“先做 CPU mirror + 统一 device 写”**。

---

## 3. 模块结构图与推理流程图

## 3.1 模块结构图：主链路里的调用关系

`allocator + req_to_token + attention metadata + kv page attention`
在 Zeus 分支里是一条完整链路：

1. `allocator` 决定新 token 该落到哪个 KV page / slot。
2. `req_to_token` 是请求级页表，记录 `req_idx -> token_pos -> kv_slot` 的映射。
3. `attention metadata` 把这张页表压成 paged attention kernel 可直接消费的 CSR 形式：
   `kv_indptr` 表示每个请求的段边界，`kv_indices` 表示实际需要读的 KV slot 列表。
4. `kv page attention` 内核再按 `kv_indices` 去读 paged KV cache。
   
下面这张图强调的是“模块之间怎么调用”，不是时序细节。

```text
+-------------------+
|   ScheduleBatch   |
| seq_lens / req_id |
+---------+---------+
          |
          v
+-------------------------------+
| mem_cache/common.py           |
| alloc_for_decode / extend     |
+---------+---------------------+
          |
          | read last_loc
          v
+-------------------------------+
| ReqToTokenPool                |
| - req_to_token        (device)|
| - req_to_token_cpu    (mirror)|
+---------+---------------------+
          |
          | allocate slots
          v
+-------------------------------+
| ZeusPagedTokenToKVPoolAllocator|
| CPU bookkeeping: free_pages    |
+---------+----------------------+
          |
          | out_cache_loc
          v
+-------------------------------+
| ReqToTokenPool.write()        |
| device index_put + mirror sync|
+---------+---------------------+
          |
          | loc + K/V
          v
+-------------------------------+
| ZeusTokenToKVPool             |
| store_kv_cache (device kernel)|
+---------+---------------------+
          |
          | before attention
          v
+-------------------------------+
| ZeusAttnBackend               |
| init_forward_metadata         |
| -> kv_indptr / kv_indices     |
+---------+---------------------+
          |
          v
+-------------------------------+
| sgl_kernel_zeus               |
| extend_attention / decode_attn|
+-------------------------------+
```

## 3.2 推理流程图：decode / extend 主链路

下面这张图强调的是“实际一次推理里数据怎么流动”。

```text
[输入 batch]
  req_pool_indices / seq_lens / prefix_lens / K / V / Q
        |
        v
[读取页表状态]
  decode: 读 last_loc
  extend: 读 prefix 信息
        |
        v
[allocator]
  CPU bookkeeping
  -> 计算 out_cache_loc
        |
        v
[页表更新]
  req_to_token[req_idx, token_pos] = out_cache_loc
  同步到 req_to_token_cpu
        |
        +--------------------------+
        |                          |
        v                          v
[KV 写入]                     [metadata 构建]
  store_kv_cache               kv_indptr
  -> paged KV cache            kv_indices
                               qo_indptr / prefix_lens
        |                          |
        +------------+-------------+
                     |
                     v
            [Zeus paged attention]
            extend_attention / decode_attention
                     |
                     v
                  [输出 O]
```

---

## 4. 现状：哪些 CPU 工作是“设计上的”，哪些是“不必要的”

### 3.1 allocator：CPU bookkeeping 是设计上的

Zeus allocator 当前设计就是把 free list 和记账常驻 CPU。

参考 [python/sglang/srt/mem_cache/zeus_allocator.py](/root/workspace/sglang/python/sglang/srt/mem_cache/zeus_allocator.py:84)：

- `free_pages` / `release_pages` 在 CPU 上维护
- `alloc_extend()` / `alloc_decode()` 的页分配算术在 CPU 上做
- 最终只把 `out_indices` 拷到 Zeus device

关键位置：

- [zeus_allocator.py:102](/root/workspace/sglang/python/sglang/srt/mem_cache/zeus_allocator.py:102)
- [zeus_allocator.py:144](/root/workspace/sglang/python/sglang/srt/mem_cache/zeus_allocator.py:144)
- [zeus_allocator.py:165](/root/workspace/sglang/python/sglang/srt/mem_cache/zeus_allocator.py:165)

这层 CPU 路径的意义是：**用一次显式 CPU bookkeeping，避开多次 device fallback**。  
因此它不属于当前第一批要消灭的“不必要 bounce”。

### 3.2 req_to_token 写：当前 Zeus 分支存在不必要的整表 bounce

参考 [python/sglang/srt/mem_cache/memory_pool.py](/root/workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py:100)：

当前 Zeus `write()` 逻辑是：

1. `self.req_to_token.cpu()`
2. CPU 上做 `index_put`
3. 整张表 `.to(device)` 回去

关键位置：

- [memory_pool.py:100](/root/workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py:100)

在 `aten::index_put` 已原生支持后，这段整表往返已经没有必要。

### 3.3 req_to_token 读：当前存在不必要的整表/整行 bounce

#### decode 入口读 `last_loc`

参考 [python/sglang/srt/mem_cache/common.py](/root/workspace/sglang/python/sglang/srt/mem_cache/common.py:441)：

- 先取 `req_to_token[rpi_cpu]`
- 再 `.cpu()`
- 再取每行最后一个位置

关键位置：

- [common.py:461](/root/workspace/sglang/python/sglang/srt/mem_cache/common.py:461)

#### attention metadata 构建

参考 [python/sglang/srt/layers/attention/zeus_backend.py](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:130)：

- `req_to_token.cpu()`
- CPU 上构建 `kv_indices`

关键位置：

- [zeus_backend.py:141](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:141)

这里是当前主链路最大的多余传输。

### 3.4 KV 写入和 paged attention 本身已经在 device 端

KV cache 写入已经通过 Zeus 专用 kernel 执行：

- [zeus_memory_pool.py:72](/root/workspace/sglang/python/sglang/srt/mem_cache/zeus_memory_pool.py:72)
- [zeus_memory_pool.py:96](/root/workspace/sglang/python/sglang/srt/mem_cache/zeus_memory_pool.py:96)

attention kernel 也已经在 Zeus device 上：

- `extend_attention`
- `decode_attention`

参考：

- [zeus_backend.py:178](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:178)
- [zeus_backend.py:231](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:231)

所以问题并不在 KV page attention kernel 本身，而在它前面的 **页表读写与 metadata 组装**。

---

## 5. 重新评估 `attention metadata` Zeus 专用 kernel

`attention metadata`，也就是把页表压成 paged attention kernel 直接消费的 CSR metadata：

- `kv_indptr`
- `kv_indices`
- extend 路径的 `qo_indptr / prefix_lens`

在这个边界下，原先 3 个候选 kernel 中只有：

```python
build_kv_metadata(req_to_token, req_pool_indices, seq_lens, extend_seq_lens)
```

### 5.1 当前第 3 步还在 CPU 上做什么

当前 `ZeusAttnBackend.init_forward_metadata()` 的状态是：

- `kv_indptr`：已经在 Zeus device 上用 `torch.cumsum(seq_lens)` 构建
- `qo_indptr`：已经在 Zeus device 上用 `torch.cumsum(extend_seq_lens)` 构建
- `prefix_lens`：直接使用 device tensor 转 int32
- `kv_indices`：仍然通过 CPU mirror 做 ragged gather，再拷回 device

参考 [zeus_backend.py:143](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:143)：

```python
seq_lens_cpu = seq_lens.cpu()
req_pool_indices_cpu = req_pool_indices.cpu()
req_to_token_cpu = forward_batch.req_to_token_pool.req_to_token_cpu

kv_indices_list = []
for b in range(batch_size):
    req_idx = req_pool_indices_cpu[b]
    kv_indices_list.append(
        req_to_token_cpu[req_idx, : seq_lens_cpu[b]].to(torch.int32)
    )
kv_indices = torch.cat(kv_indices_list)
```

所以现在第 3 步的 CPU 残留不是 `indptr` prefix sum，而是：

1. `seq_lens.cpu()` / `req_pool_indices.cpu()` 用来驱动 Python 变长切片
2. 从 `req_to_token_cpu` mirror 做每个 request 的变长 row gather
3. Python for-loop 管理每个 request 的 offset 和 slice 长度
4. 最终 `kv_indices.to(device)` 把拼好的 compact CSR indices 搬回 Zeus

### 5.2 `cat` 已有 device aten 后
 
如果每个 per-request slice 已经是 device tensor，那么 `torch.cat(kv_indices_list)`
可以在 Zeus device 上完成。

但这并不自动让 `kv_indices` 构建完全 device 化，因为当前困难点变成了：

- 每个 request 的 slice 长度 `seq_lens[b]` 是动态的
- Python slice `:seq_lens[b]` 需要 CPU 可见的边界
- compact CSR 输出需要按 `kv_indptr[b]` 写到连续区间
- 仅靠 device `cat` 仍需要先得到一组合法的变长 device slice

因此，在 `cat` 已支持 device 的前提下，可以把方案分成两级：

| 方案 | 做法 | CPU 残留 | 评价 |
|------|------|----------|------|
| A. 最小 Python 改造 | 从 device `req_to_token` 取每行 slice，然后 device `cat` | 仍需要 CPU `seq_lens` 控制 Python slice；可能仍有逐行同步 | 只适合作为验证，不建议作为最终优化 |
| B. `build_kv_metadata` 专用 kernel | device 端按 `kv_indptr` 把 `req_to_token[req_idx, 0:seq_len]` 直接 scatter 到 `kv_indices` | 基本无 CPU 参与 | 推荐方案 |

真正要 device 化第 3 步，关键仍然是：

```text
kv_indices[kv_indptr[b] + j] = req_to_token[req_pool_indices[b], j]
```

这个变长二维 gather 到 compact 一维 CSR buffer 的过程。

### 5.3 只处理第 3 步时的推荐优先级

这里要特别区分两类问题：

1. **不必要的 CPU bounce**
   - 已经靠 `req_to_token_cpu mirror` 基本消掉了
   - 也就是：不再为了读页表而把整张 `req_to_token` 从 device 临时拉回 CPU
2. **仍在 CPU 上执行的 metadata 计算**
   - 还包括 ragged gather、Python 变长切片、offset 管理
   - 这不一定伴随“大块 device→CPU 拷贝”，但仍意味着 host 参与

所以当前主链路更准确的状态是：

- **大块 bounce 已基本消除**
- **`kv_indptr / qo_indptr` cumsum 已 device 化**
- **`cat` 已具备 device 化条件**
- **`kv_indices` ragged gather 仍未真正 device 化**

因此只处理步骤 3 时，优先级应改为：

1. P1：实现 `build_kv_metadata` Zeus 专用 kernel，重点只放在 `kv_indices` ragged gather
2. P2：把 graph/eager 两条路径统一到同一个 device metadata builder


### 5.4 剩余 `cumsum` 调用点优先级表

按当前代码状态，剩余值得关注的 `cumsum` 调用点可以分成三类：

| 类别 | 代表位置 | 当前作用 | 建议优先级 | 说明 |
|------|----------|----------|------------|------|
| 主链路 | [zeus_backend.py:100](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:100), [zeus_backend.py:149](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:149), [zeus_backend.py:168](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:168) | 构 `kv_indptr / qo_indptr` | 已处理 | 已改为 Zeus device `torch.cumsum`；剩余问题转为 `kv_indices` 的 ragged gather |
| Batch | [forward_batch_info.py:1081](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1081), [forward_batch_info.py:1118](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1118) | 构 `prefix_chunk_cu_seq_lens`、`kv_indptr` | 已处理 | 这两处是纯 `cumsum` CPU 绕路，已改为 Zeus device `torch.cumsum` |
| Batch | [forward_batch_info.py:1258](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1258) | 构 `extend_start_loc` | 暂不处理 | 该分支还混合动态 `arange`、变长拼接和索引逻辑，不只依赖 `cumsum` |
| Logits | [logits_processor.py:420](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/logits_processor.py:420) | 构最后 token 的线性下标 | 暂不处理 | 该分支还依赖 gather / index 逻辑，先不只改 `cumsum` |

因此这里对 `cumsum` 的定位应更新为：

- **主链路 indptr 构建已经可以直接使用 device cumsum**
- **纯 `cumsum` 的 batch 路径已处理**
- **logits 和混合 batch 路径还依赖其他 CPU/index 能力，暂不单独修改**

---

## 6. 最佳实现方案

## 6.1 Phase 1：最小改动、最大收益

### 改动 1：为 Zeus 增加 `req_to_token_cpu` mirror

在 `ReqToTokenPool.__init__` 中新增：

```python
if _is_zeus:
    self.req_to_token_cpu = torch.zeros(
        (size, max_context_len), dtype=torch.int32
    )
```

目的：

- 把“CPU 可读页表”常驻下来
- 避免后续每步从 device 拉整行/整表

### 改动 2：`ReqToTokenPool.write()` 直接 device 写，并同步 mirror

建议逻辑：

```python
def write(self, indices, values):
    if _is_zeus:
        idx_cpu = ...
        val_cpu = ...
        self.req_to_token_cpu[idx_cpu] = val_cpu

    self.req_to_token[indices] = values
```

意义：

- 直接利用 `aten::index_put`
- 去掉当前整表 `cpu() -> to(device)` roundtrip
- 保持 mirror 与 device 页表一致

### 改动 3：`alloc_for_decode()` 改读 mirror

把：

```python
r2t_rows_cpu = batch.req_to_token_pool.req_to_token[rpi_cpu].cpu()
```

改成：

```python
r2t_rows_cpu = batch.req_to_token_pool.req_to_token_cpu[rpi_cpu]
```

这样 decode 每步不再从 device 拉行。

### 改动 4：`ZeusAttnBackend.init_forward_metadata()` 改读 mirror

把：

```python
req_to_token_cpu = req_to_token.cpu()
```

改成：

```python
req_to_token_cpu = forward_batch.req_to_token_pool.req_to_token_cpu
```

这样砍掉 attention metadata 前最大的整表 `.cpu()`。

---

## 6.2 Phase 1 实施后，模块调用关系会变成什么样

```text
allocator (CPU bookkeeping, 保留)
   |
   | out_cache_loc -> Zeus device
   v
ReqToTokenPool.write
   |\
   | \__ req_to_token_cpu[idx] = val     (CPU mirror)
   |
   \____ req_to_token[indices] = values  (Zeus device, index_put)
         |
         v
Zeus KV store_kv_cache (device)
         |
         v
ZeusAttnBackend.init_forward_metadata
   |
   | 直接读 req_to_token_cpu
   | device 上做 kv_indptr / qo_indptr cumsum
   | CPU mirror 上做 kv_indices ragged gather
   v
extend_attention / decode_attention (device)
```

这个方案的核心收益是：

- **消除 `req_to_token.write()` 的整表 CPU roundtrip**
- **消除 decode 路径的逐步行拷贝**
- **消除 attention metadata 路径的整表 device→CPU 拷贝**

同时：

- 不动 allocator 设计
- 不需要马上新增 Zeus kernel
- 改动集中在 3 个 Python 文件

---

## 7. 为什么这比“马上补 3 个 kernel”更优

### 6.1 改动更小

只需要修改：

- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/common.py`
- `python/sglang/srt/layers/attention/zeus_backend.py`

### 6.2 收益更直接

当前最大的多余 bounce 都来自 `req_to_token` 的“读”和“写”。  
而 `index_put` 已经把“写”的最大 blocker 去掉了。

### 6.3 风险更低

相比新增 Zeus kernel：

- 不需要 C++/kernel 实现与调试
- 不引入新的 ABI 或 graph 兼容问题
- 更容易对照 CUDA/CPU 路径验证正确性

---

## 8. 第二阶段建议

如果 Phase 1 完成后继续推进，并且本轮明确只处理图中的第 3 步
`attention metadata`，建议按下面顺序做。

### 7.1 P1：`build_kv_metadata` Zeus 专用 kernel

目标：

- 保持 `kv_indptr / qo_indptr` 继续使用 device `torch.cumsum`
- 把 `kv_indices` 的 ragged gather 从 CPU mirror + Python for-loop 搬到 device
- graph/eager 路径复用同一个 metadata builder，只是输出 buffer 不同

价值：

- 进一步减少 host 参与
- 让 Zeus attention 前处理更接近 CUDA 路径
- graph capture/replay 也更容易统一

建议 kernel 语义：

```text
for b in batch:
    req_idx = req_pool_indices[b]
    start = kv_indptr[b]
    len = seq_lens[b]
    for j in range(len):
        kv_indices[start + j] = req_to_token[req_idx, j]
```

输入：

- `req_to_token`
- `req_pool_indices`
- `seq_lens`
- `kv_indptr`

输出：

- `kv_indices`

说明：

- `cat` 已有 Zeus device aten 后，不需要把 `cat` 本身视为 blocker
- 但变长 slice 的边界和 compact CSR 写入仍然需要 device 端处理
- 因此推荐做专用 kernel，而不是仅靠 Python list + device `cat`

### 7.2 P2：Python-level device `cat` 过渡版本

前提：

- 仅用于快速验证 device `cat` 和 device row gather 的可行性
- 不作为最终性能方案

原因：

- 即使 `cat` 在 device 上，Python 仍然需要逐 request 决定 slice 长度
- 动态 slice 很可能继续依赖 CPU scalar 或触发同步
- 对大 batch / 长上下文，Python for-loop 本身也会变成调度开销

## 9. 最终建议

**最佳方案：**

1. **立即实现 CPU mirror + 直接 device index_put**
2. 保留 allocator 的 CPU bookkeeping，不急着 device 化
3. 在只优化步骤 3 的前提下，把 `build_kv_metadata` 收敛为 `kv_indices` device ragged gather kernel
4. `get_last_loc` / `write_req_to_token` 属于步骤 1/2，本轮不纳入

补充说明：

- 这套方案完成后，主链路上已经**基本没有不必要的大块 CPU bounce**
- `kv_indptr / qo_indptr` 的 `cumsum` 已经可以在 Zeus device 上执行
- `cat` 已具备 device aten 实现，不再是第 3 步的核心 blocker
- Paged KV metadata 剩余的 host 工作主要是 `kv_indices` ragged gather 和 Python 变长 slice 管理

**优先级总结：**

- 🔴 P0：`ReqToTokenPool` CPU mirror + 统一 device `index_put`
- 🔴 P0：`alloc_for_decode()` 改读 mirror
- 🔴 P0：`zeus_backend.init_forward_metadata()` 改读 mirror
- 🟡 P1：步骤 3 专用 `build_kv_metadata`，重点是 device `kv_indices` ragged gather
- 🟢 P2：Python-level device `cat` 过渡验证

---

## 10. 对应代码位置

- `ReqToTokenPool.write()`  
  [python/sglang/srt/mem_cache/memory_pool.py](/root/workspace/sglang/python/sglang/srt/mem_cache/memory_pool.py:100)

- `alloc_for_decode()`  
  [python/sglang/srt/mem_cache/common.py](/root/workspace/sglang/python/sglang/srt/mem_cache/common.py:441)

- `ZeusAttnBackend.init_forward_metadata()`  
  [python/sglang/srt/layers/attention/zeus_backend.py](/root/workspace/sglang/python/sglang/srt/layers/attention/zeus_backend.py:130)

- `ZeusPagedTokenToKVPoolAllocator`  
  [python/sglang/srt/mem_cache/zeus_allocator.py](/root/workspace/sglang/python/sglang/srt/mem_cache/zeus_allocator.py:84)

- `ZeusTokenToKVPool.set_kv_buffer()` / `store_kv_cache`  
  [python/sglang/srt/mem_cache/zeus_memory_pool.py](/root/workspace/sglang/python/sglang/srt/mem_cache/zeus_memory_pool.py:72)
