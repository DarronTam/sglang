# torch_zeus 已支持算子与 SGLang `_is_zeus` CPU bounce 审计

> 创建日期：2026-04-27 | 最后更新：2026-05-08  
> 本次更新：按**当前代码**重新整理，只统计**仍然存在**的 CPU bounce；已被最新代码修复的项不再列为现存问题。
> 前提更新（04-27）：`torch_zeus` 已支持 `aten::cumsum` 和 `aten::index_put / index_put_`。
> 前提更新（05-08）：`torch_zeus` `neg / clamp / where / arange` 均已支持。

---

## 1. 当前代码结论

基于最新代码，Zeus 路径里的 CPU 行为现在应分成 3 类：

1. **已解决，不再计入当前 CPU bounce**
   - `ReqToTokenPool.write()` 的整表 CPU roundtrip 已去掉
   - `alloc_for_decode()` 从 `req_to_token` 读 `last_loc` 的行拷贝已去掉
   - `ZeusAttnBackend.init_forward_metadata()` 的 non-graph 整表 `req_to_token.cpu()` 已去掉
   - greedy `argmax` 不再有 Zeus 专门的 CPU workaround

2. **仍存在的 CPU bounce**
   - 这些分支背后依赖的能力当前仍缺失，或代码里还保留历史 CPU workaround
   - 典型：`index` 读取（`aten::index_select` / gather 尚不支持）

3. **设计上保留的 CPU bookkeeping**
   - 这些不是“不必要 bounce”
   - 而是 Zeus 分支主动保留在 CPU 的状态管理
   - 典型：`ZeusPagedTokenToKVPoolAllocator`

一句话：

> 当前主链路最明显的 `req_to_token` 读写 bounce 已经被新代码消掉了；`cumsum` / `index_put` / `neg` / `clamp` / `where` / `arange` 也已具备 Zeus aten 实现。`overlap_utils` 和 `forward_batch_info` 的相关 CPU bounce 已随之消减。现在更值得继续关注的是 `index(read)` / gather 驱动的 metadata、batch 处理路径。

---

## 2. 本地文档汇总：`torch_zeus` 相关算子现状

本节只保留对当前审计仍有影响的算子。

### 2.1 已确认可直接用的关键算子

依据：

- [torch_zeus_operator_analysis-20260415.md](/root/workspace/sglang/zeus_dev/torch_zeus_operator_analysis-20260415.md:1)
- [sglang_zeus_manual.md](/root/workspace/sglang/zeus_dev/sglang_zeus_manual.md:1131)
- torch_zeus `csrc/aten/operators/zenl/` + `zenl/src/host/` 源码（2026-05-08 核查）

| 类别 | 算子 | 支持 dtype | 备注 |
|---|---|---|---|
| 拼接 | `cat`, `cat.out` | dtype 透传（不做限制） | ✅ 原生 |
| 算术 | `add`（scalar 加） | fp32, bf16, int32 | ✅ scalar fast path；int64 自动降级到 int32（带一次性警告） |
| 算术 | `add`（element-wise vector） | fp32, bf16, int8, fp8_e4m3fn | ✅ optensor 路径 |
| 算术 | `sub`, `mul` | fp32, bf16, int8, fp8_e4m3fn | ✅ optensor 路径 |
| 归约 | `sum`, `mean`, `norm` | fp32, bf16, int8, fp8_e4m3fn | ✅ reduce 内核 |
| 前缀和 | `cumsum` | fp32, bf16, int32 | ✅ 原生；int64 输入按 PyTorch promote_integers 规则提升 |
| GEMM | `mm`, `addmm`, `linear` | bf16, fp8_e4m3fn | ✅ 原生；权重须先 `pack_weights` 放入 LocalMem |
| 索引写入 | `index_put_`, `index_put` | value dtype 透传；index: int32/int64 | ✅ 原生 |
| 元素选取 | `neg` | fp32, bf16, int8, int32, fp8_e4m3fn | ✅ 原生 |
| 元素选取 | `clamp` | fp32, bf16, int8, int32, fp8_e4m3fn | ✅ 原生；int8 遇浮点 bound 自动升 fp32 |
| 条件选取 | `where` | value: fp32, bf16, int8, int32, fp8_e4m3fn；condition: bool | ✅ 原生 |
| 序列生成 | `arange` | fp32, bf16, int8, int32, fp8_e4m3fn | ✅ 原生 |
| Embedding | `embedding` | weight dtype 透传；index: int32/int64 | ✅ 原生 |
| 非零索引 | `nonzero` | fp16, fp32, bf16, int8, uint8, int16, int32, bool, fp8_e4m3fn | ✅ 原生；output index 为 int64 |
| greedy 归约 | `argmax` | fp32, bf16, int8, fp8_e4m3fn（经 reduce 路径） | ✅ 当前代码已直接使用 |

### 2.2 `sgl_kernel_zeus` 已支持 / 已接入的自定义 kernel

这里和上一节的 `torch_zeus` aten 算子区分开：`torch_zeus` 负责
PyTorch ATen dispatch，`sgl_kernel_zeus` 则是 SGLang 在 Zeus 上使用的
硬件专用融合 kernel / attention kernel。

依据：

- [sglang_zeus_manual.md](/root/workspace/sglang/zeus_dev/sglang_zeus_manual.md:407)
- [zeus_graph_integration_plan.md](/root/workspace/sglang/zeus_dev/zeus_graph_integration_plan.md:476)
- 当前 Python 代码中的 `from sgl_kernel_zeus import ...` 调用点

| 类别 | Kernel | 用途 | 主要调用位置 / 状态 |
|------|--------|------|--------------------|
| Attention | `extend_attention` | Prefill / extend paged attention | `zeus_backend.py:forward_extend` |
| Attention | `decode_attention` | Decode paged attention | `zeus_backend.py:forward_decode` |
| KV Cache | `store_kv_cache` | 写入 Zeus tiled KV cache | `zeus_memory_pool.py:set_kv_buffer` |
| Position | `rotary_embedding` | RoPE 位置编码 | `rotary_embedding.py:forward_zeus` |
| Norm | `rmsnorm` | RMSNorm | `layernorm.py:forward_zeus` |
| Norm | `fused_add_rmsnorm` | residual add + RMSNorm 融合 | `layernorm.py:forward_zeus` |
| Activation | `silu_and_mul` | SiLU + gate 乘法融合 | `activation.py:forward_zeus` |
| Embedding | `embedding` | LocalMem 转置布局下的 embedding gather | 本地 Zeus manual 列出；当前代码需按实际接入情况确认 |
| Sampling | `sampling_from_logits` | fused sampling from logits | `sampler.py` Zeus 分支 |
| Sampling | `top_k_renorm_prob` | top-k 后概率重归一化 | `sampler.py` Zeus 分支 |
| Sampling | `top_p_renorm_prob` | top-p 后概率重归一化 | `sampler.py` Zeus 分支 |
| Sampling | `top_k_top_p_sampling_from_probs` | top-k + top-p 采样 | `sampler.py` Zeus 分支 |
| Sampling | `min_p_sampling_from_probs` | min-p 采样 | `sampler.py` Zeus 分支 |

补充说明：

- `zeus_graph_integration_plan.md` 中已经验证过 7 个核心 decode graph 相关 op：
  `rmsnorm`、`fused_add_rmsnorm`、`silu_and_mul`、`rotary_embedding`、
  `store_kv_cache`、`decode_attention`、`embedding`。
- 本地旧文档有“14 个算子”的口径，但展开表格显式列出的条目是上面这些。
  当前 audit 以显式 kernel 名称和 SGLang 调用点为准。
- `argmax` 当前走 `torch.argmax` / `torch_zeus` aten 路径，不再作为
  `sgl_kernel_zeus` CPU bounce workaround 统计。

### 2.3 仍缺失、并且仍影响当前代码的关键算子

| 算子 | 当前影响 |
|---|---|
| `index.Tensor_out` / `tensor[indices]` | graph metadata、schedule_batch.filter、logits gather |

以下算子已于 2026-05-08 支持，相关 bounce 已消减：

| 算子 | 完整支持 dtype（05-08 后） | 已修复的影响位置 |
|---|---|---|
| `clamp(int32)` | fp32, bf16, int8, int32, fp8_e4m3fn | `overlap_utils.py` clamp(-input_ids)、`forward_batch_info.py` clamp_position ✅ |
| `where(int32)` | fp32, bf16, int8, int32, fp8_e4m3fn | `overlap_utils.py` torch.where(ids<0, ...) ✅ |
| `arange(int32)` | fp32, bf16, int8, int32, fp8_e4m3fn | `overlap_utils.py` alloc_future_indices、`forward_batch_info.py` extend_start_loc ✅ |
| `neg(int32)` | fp32, bf16, int8, int32, fp8_e4m3fn | `scheduler.py` -future_indices.indices、`overlap_utils.py` -input_ids ✅ |

### 2.4 关于 `argmax`

本地旧文档存在时间差，但**当前代码现状**是：

- [sampler.py:101](/root/workspace/sglang/python/sglang/srt/layers/sampler.py:101) 直接调用 `torch.argmax(logits, -1)`
- 当前 `sampler.py` 不再有 Zeus 专门的 `logits.cpu() -> argmax -> to(device)` workaround

因此本审计中，`argmax` **不再计入当前仍存在的 CPU bounce**。

### 2.5 torch_zeus 算子支持 dtype 速查矩阵

> 数据来源：`torch_zeus/csrc/aten/operators/zenl/` ATen 层 + `zenl/src/host/` host 层源码，2026-05-08 核查。

ZENL 内部 dtype 枚举（`zenl.h`）：

| 枚举值 | 含义 | PyTorch `kType` |
|---|---|---|
| `ZENL_DTYPE_FLOAT32 = 1` | FP32 | `at::kFloat` |
| `ZENL_DTYPE_FLOAT16 = 2` | FP16 | `at::kHalf` |
| `ZENL_DTYPE_BFLOAT16 = 3` | BF16 | `at::kBFloat16` |
| `ZENL_DTYPE_INT8 = 4` | INT8（有符号） | `at::kChar` |
| `ZENL_DTYPE_UINT8 = 5` | UINT8 | `at::kByte` |
| `ZENL_DTYPE_INT16 = 6` | INT16 | `at::kShort` |
| `ZENL_DTYPE_INT32 = 7` | INT32 | `at::kInt` |
| `ZENL_DTYPE_INT64 = 8` | INT64（无硬件向量支持） | `at::kLong` |
| `ZENL_DTYPE_BOOL = 9` | Bool | `at::kBool` |
| `ZENL_DTYPE_FLOAT8_E4M3FN = 10` | FP8 E4M3 | `at::kFloat8_e4m3fn` |

> Zeus 芯片不支持 int64 / fp64 向量。int64 标量在 `add/sub` 等少数路径下可自动降级到 int32（带一次性警告）。

各算子支持 dtype 矩阵：

| 算子 | fp32 | bf16 | fp16 | int8 | uint8 | int16 | int32 | bool | fp8_e4m3fn | 备注 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `neg` | ✅ | ✅ | — | ✅ | — | — | ✅ | — | ✅ | int32 于 05-08 新增 |
| `clamp` | ✅ | ✅ | — | ✅ | — | — | ✅ | — | ✅ | int8 遇浮点 bound 自动提升 fp32；int32 于 05-08 新增 |
| `where` | ✅ | ✅ | — | ✅ | — | — | ✅ | — | ✅ | condition 必须 bool；int32 于 05-08 新增 |
| `arange` | ✅ | ✅ | — | ✅ | — | — | ✅ | — | ✅ | int32 于 05-08 新增 |
| `cumsum` | ✅ | ✅ | — | — | — | — | ✅ | — | — | int64 按 PyTorch promote_integers 规则提升 |
| `add`（scalar） | ✅ | ✅ | — | — | — | — | ✅ | — | — | int64 自动降级 int32（有警告） |
| `add`（vector） | ✅ | ✅ | — | ✅ | — | — | — | — | ✅ | optensor element-wise 路径 |
| `sub` | ✅ | ✅ | — | ✅ | — | — | — | — | ✅ | optensor |
| `mul` | ✅ | ✅ | — | ✅ | — | — | — | — | ✅ | optensor |
| `sum` / `mean` / `norm` | ✅ | ✅ | — | ✅ | — | — | — | — | ✅ | reduce 内核 |
| `argmax` | ✅ | ✅ | — | ✅ | — | — | — | — | ✅ | 经 reduce 路径 |
| `mm` / `addmm` / `linear` | — | ✅ | — | — | — | — | — | — | ✅ | 权重须先 `pack_weights` |
| `index_put` / `index_put_` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | value dtype 透传；index 须 int32/int64 |
| `embedding` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | weight dtype 透传；index 须 int32/int64 |
| `cat` / `cat.out` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | dtype 透传 |
| `nonzero` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | output index 为 int64 |

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
   | [已优化] arange(int32) 现原生 Zeus，直接在 device 上分配
   v
future_indices
   |
   | encode as negative placeholders
   | [已优化] neg(int32) 原生 Zeus；-future_indices.indices 直接在 device 上执行
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
   | [已优化] neg / clamp / where 现原生 Zeus(int32)
   | [仍存在] index(read): buf[gather_indices] 仍需 CPU（缺 aten::index_select）
   |   - future_token_ids_map.cpu()
   |   - gather_indices.cpu() + buf[...]
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
- Overlap / Scheduler 最关心：~~`where / clamp / arange / neg(int)`~~ ✅ 已补齐 int32 支持；当前剩余关注点为 `index(read)` / `aten::index_select`

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

- [common.py:461](/root/workspace/sglang/python/sglang/srt/mem_cache/common.py:461)
- [common.py:467](/root/workspace/sglang/python/sglang/srt/mem_cache/common.py:467)
- [common.py:485](/root/workspace/sglang/python/sglang/srt/mem_cache/common.py:485)

当前仍存在：

- Zeus 分支仍通过 `req_to_token_cpu` mirror 读取 `last_loc`
- 因为普通路径里的 `req_to_token[req_pool_indices, seq_lens - 1]` 依赖 `index(read)` / gather

当前已清理：

- `seq_lens_next` 不再走 `(sl_cpu + token_per_req).to(device)`
- 已统一为 `seq_lens_next = batch.seq_lens + token_per_req`
- `locs` 已经是 `batch.seq_lens.clone()`，不再有旧的 `seq_lens.cpu().to(device)` 写法

为什么还在：

- `aten::index_put` 只解决 `req_to_token` 写入，不解决这里的高级索引读取
- 在 `index(read)` / gather 没确认可用前，直接删除整个 Zeus 分支会让 `last_loc` 回到 device 高级索引路径，风险较高

能否直接删：

- **不能整段直接删**
- 可以继续清理纯 `add/sub/clone` 这类小算术残留；目前本节相关的 `seq_lens_next` 已处理

怎么消除：

- 等 `index(read)` / gather 可用并验证后，把 `last_loc` 也统一到普通路径：

```python
last_loc = batch.req_to_token_pool.req_to_token[
    batch.req_pool_indices, batch.seq_lens - 1
]
seq_lens_next = batch.seq_lens + token_per_req
```

- 或者补一个 Zeus 专用 `get_last_loc` kernel，只把 `last_loc` 读取搬到 device

---

## 4.2 Overlap / Scheduler 路线

### A. `managers/overlap_utils.py`

参考：

- [overlap_utils.py:21](/root/workspace/sglang/python/sglang/srt/managers/overlap_utils.py:21)
- [overlap_utils.py:118](/root/workspace/sglang/python/sglang/srt/managers/overlap_utils.py:118)

当前仍存在：

- `buf = future_token_ids_map.cpu()`（future buffer 整表拉回 CPU）
- `gather_indices.cpu()` + `buf[gather_indices.cpu()]`（CPU index read）

已解决（2026-05-08）：

- `neg(int32)`：`-input_ids` 直接在 Zeus 上执行 ✅
- `clamp(int32)`：`torch.clamp(-input_ids, min=0)` 直接在 Zeus 上执行 ✅
- `where(int32)`：`torch.where(input_ids < 0, looked_up, input_ids)` 直接在 Zeus 上执行 ✅
- `arange(int32)`：`alloc_future_indices` 中 `torch.arange(..., device=self.device)` 直接分配 ✅

为什么还在：

- `index(read)` / `aten::index_select` 缺失：`buf[gather_indices]` 无法在 Zeus 上做

能否直接删：

- **剩余 CPU 行不能删**

怎么消除：

- 补 `aten::index_select` / gather 后，`buf[gather_indices.cpu()]` 可移至 device

### B. `managers/scheduler.py`

参考：

- [scheduler.py:2061](/root/workspace/sglang/python/sglang/srt/managers/scheduler.py:2061)

当前状态：✅ 已修复（2026-05-08）

- `future_indices_or_next_token_ids = -future_indices.indices`
- `neg(int32)` 现已原生 Zeus；`-future_indices.indices` 直接在 Zeus device 上执行

---

## 4.3 Forward Batch Info 路线

参考：

- [forward_batch_info.py:841](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:841)
- [forward_batch_info.py:1080](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1080)
- [forward_batch_info.py:1117](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1117)
- [forward_batch_info.py:1246](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1246)
- [forward_batch_info.py:1276](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1276)

已解决（2026-05-08）：

- `extend_prefix_lens = self.seq_lens - 1`（`sub` 原生 Zeus）✅
- `extend_start_loc = torch.arange(bs, dtype=torch.int32, device=...)` 直接 device ✅
- `clamp_position()` 改为 `torch.clamp((seq_lens - 1), min=0)` 直接 Zeus ✅
- `compute_position_torch()` Zeus 分支改为 `arange(int32, device=device)` + `cumsum` ✅

当前仍存在：

- `compute_position_torch()` 中仍需 `extend_prefix_lens.cpu().tolist()` / `extend_seq_lens.cpu().tolist()` 获取 Python 标量来驱动 `range()`（`zip(p_list, s_list)` 迭代边界）

为什么还在：

- `range()` 需要 Python 整数；读取标量本身代价小，不属于大块 bounce

能否直接删：

- 不需要再做额外改动；`.cpu().tolist()` 用于 Python 标量属于正常模式

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
- `torch.arange(len(seq_lens_cpu))`
- 用 Python `for` 循环逐行 `copy_(hidden[idx])`

为什么还在：

- `index(read)` 缺失
- `arange` 缺失
- 虽然 `cumsum` 已支持，但该分支当前把 prefix sum、`arange`、gather/index 逻辑混在 CPU 控制流里，所以不能只靠替换一行 `cumsum` 完成 device 化

能否直接删：

- **不能**

怎么消除：

- 补 `index(read)` / gather
- `arange` 配套解决
- 再把 prefix sum 留在 Zeus device 上，与 gather/index 路径一起重写

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
3. `schedule_batch.py` 里仅依赖 `add/sub` 的小算术 Zeus 分支

这些项不依赖新 kernel，最容易清掉。

### P1：主链路第二阶段优化

1. 继续减少 attention metadata 的 host 计算
2. 等 `index(read)` / gather 可用后，再考虑删除 `common.py` 里的 `last_loc` mirror 读取分支

### P2：补缺失算子 / 应用已支持算子

以下已完成（2026-05-08）：

- ✅ `arange(int32)`、`neg(int32)`、`clamp(int32)`、`where(int32)` — torch_zeus 全 4 层已支持
- ✅ 相关 SGLang 侧 bounce 已消减（overlap_utils / scheduler / forward_batch_info）

剩余（依赖 `aten::index_select` / gather）：

1. `index(read)` / gather — 消除后可继续清理：
   - `overlap_utils._resolve_future_token_ids` 中的 `buf[gather_indices.cpu()]`
   - `logits_processor.py:420` 中的 gather loop
   - `common.py` 中的 `last_loc` mirror 读取分支
2. 清理 `schedule_batch.py` 中的 `tensor.cpu()[ki_cpu].to(device)` filter 分支

## 6.1 剩余 `cumsum` 调用点优先级表

`aten::cumsum` 已由 `torch_zeus` 支持。这里不再把 `cumsum` 当作缺失算子，
而是把代码中仍值得关注的历史 CPU workaround 按 `主链路 / batch / logits`
三类重新整理如下。

| 类别 | 代表位置 | 当前作用 | 优先级 | 原因 |
|------|----------|----------|--------|------|
| 主链路 | [zeus_backend.py:100](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:100), [zeus_backend.py:149](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:149), [zeus_backend.py:168](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:168) | 构 `kv_indptr / qo_indptr` | 已处理 | 已改为 Zeus device `torch.cumsum`；剩余问题转为 `kv_indices` 的 ragged gather |
| Batch | [forward_batch_info.py:1081](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1081), [forward_batch_info.py:1118](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1118) | 构 `prefix_chunk_cu_seq_lens`、`kv_indptr` | 已处理 | 这两处是纯 `cumsum` CPU 绕路，已改为 Zeus device `torch.cumsum` |
| Batch | [forward_batch_info.py:compute_position_torch](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1257) | 构 `extend_start_loc` via cumsum | ✅ 已处理 | `arange(int32)` + `cumsum` 均已原生 Zeus；仍需 `.cpu().tolist()` 获取 Python 标量 |
| Logits | [logits_processor.py:420](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/logits_processor.py:420) | 构最后 token 的线性下标 | 暂不处理 | 该分支还依赖 gather / index 逻辑，先不只改 `cumsum` |

补充说明：

- 主链路里的 `cumsum` 已经回到 Zeus device 上执行
- 纯 `cumsum` 的 batch 路径已处理
- logits 和混合 batch 路径还依赖其他 CPU/index 能力，不能只替换 `cumsum`
- Paged KV metadata 的剩余 host 工作，核心已经变成 `kv_indices` 的 ragged gather

---

## 7. 一句话总结

这份文档按最新代码重排后，最重要的变化是：

> `req_to_token` 读写和 `kv_metadata` 相关的主链路大块 CPU bounce 已经被解决，不应再继续作为“当前未解决问题”统计。  
> 现在更准确的说法是：主链路上已基本没有"不必要的大块 CPU bounce"；`neg / clamp / where / arange(int32)` 于 2026-05-08 完成 torch_zeus int32 支持，overlap / scheduler / forward_batch_info 相关 CPU bounce 已消减。当前真正残留的热点，主要集中在 `index(read)` / `aten::index_select` 缺失导致的 `buf[gather_indices]` gather（overlap_utils）、logits gather loop（logits_processor），以及 `kv_indices` ragged gather（可由 `build_kv_indices` device kernel 替代）。
