# torch_zeus 已支持算子与 SGLang `_is_zeus` CPU bounce 审计

> 创建日期：2026-04-27 | 最后更新：2026-05-08③  
> 本次更新：`aten::index.Tensor`（`index_get.cpp`）确认可用，所有 `index(read)` CPU bounce 已全部消除。
> 前提更新（04-27）：`torch_zeus` 已支持 `aten::cumsum` 和 `aten::index_put / index_put_`。
> 前提更新（05-08）：`torch_zeus` `neg / clamp / where / arange` 均已支持。
> 前提更新（05-08②）：`sgl_kernel_zeus.build_kv_indices` 五件套已落地；`zeus_backend.py` 已切换到 device kernel 路径，CPU fallback 改为一次性 `logger.warning`。
> 前提更新（05-08③）：`torch_zeus` 已注册 `aten::index.Tensor / index.Tensor_out`（`index_get.cpp`），支持 2-D fancy index + int32/int64 indices。`alloc_for_decode` last_loc、`filter_batch`、`_resolve_future_token_ids`、`logits_processor` gather loop 的 Zeus CPU 分支已全部删除，统一走 device 路径。

---

## 1. 当前代码结论

基于最新代码，Zeus 路径里的 CPU 行为现在应分成 3 类：

1. **已解决，不再计入当前 CPU bounce**
   - `ReqToTokenPool.write()` 的整表 CPU roundtrip 已去掉
   - `alloc_for_decode()` 从 `req_to_token` 读 `last_loc` 的行拷贝已去掉（05-08③ 改为 device `aten::index.Tensor`）
   - `ZeusAttnBackend.init_forward_metadata()` 的 non-graph 整表 `req_to_token.cpu()` 已去掉
   - greedy `argmax` 不再有 Zeus 专门的 CPU workaround
   - `index(read)` / gather 驱动的 CPU bounce 已全部消除（05-08③）：`filter_batch`、`_resolve_future_token_ids`、`logits_processor` gather loop 的 `_is_zeus` CPU 分支均已删除

2. **仍存在的 CPU bounce**
   - 截至 05-08③，因算子缺失导致的不必要 CPU bounce **已全部消除**，本类别当前无内容

3. **设计上保留的 CPU bookkeeping**
   - 这些不是"不必要 bounce"
   - 而是 Zeus 分支主动保留在 CPU 的状态管理
   - 典型：`ZeusPagedTokenToKVPoolAllocator`
   - 另：`req_to_token_cpu` mirror 现仅服务 `_build_kv_indices_cpu` 回退路径，`last_loc` 已不依赖

一句话：

> 截至 05-08③，SGLang Zeus 路径中不再有"因算子缺失导致的不必要 CPU bounce"：`cumsum` / `index_put` / `neg` / `clamp` / `where` / `arange` / `index.Tensor`（2-D fancy index）均已具备原生 Zeus 实现；`alloc_for_decode` last_loc、`filter_batch`、`_resolve_future_token_ids`、`logits_processor` gather loop 的 `_is_zeus` CPU 分支已全部删除。仅保留设计上必要的 CPU bookkeeping（`ZeusPagedTokenToKVPoolAllocator`）和 `_build_kv_indices_cpu` 回退 mirror。后续关注点转为 fp32 linearize 精度上限和 `req_pool_indices` dtype 统一。

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
| Attention | `build_kv_indices` | 页表 ragged gather → compact CSR kv_indices | `zeus_backend.py:_build_kv_indices_device`；五件套于 05-08② 落地 |
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

截至 2026-05-08③，主链路和 batch/logits 处理路径影响的关键算子均已具备：

| 算子 | 状态 | 备注 |
|---|---|---|
| `index.Tensor` / `index.Tensor_out` | ✅ 已支持（05-08③） | `index_get.cpp` 注册，支持 2-D fancy index，int32/int64 indices；flat offset 用 fp32 累加，上限约 $2^{24}$（见风险注记） |

**fp32 linearize 精度风险注记**：`index_get.cpp` 的 `zenl_linearize_indices` 用 fp32 中间量累加 flat offset。对 `req_to_token[pool_size, max_ctx_len]`，当 `pool_size × max_ctx_len > 2^{24} ≈ 16.7M`（如 `4096 × 8192`）时可能精度溢出，建议向 `index_get.cpp` 反馈改用 int32/int64 累加器。

以下算子已于 2026-05-08 / 05-08② 支持，相关 bounce 已消减：

| 算子 / kernel | 完整支持 dtype / 状态 | 已修复的影响位置 |
|---|---|---|
| `clamp(int32)` | fp32, bf16, int8, int32, fp8_e4m3fn | `overlap_utils.py` clamp(-input_ids)、`forward_batch_info.py` clamp_position ✅ |
| `where(int32)` | fp32, bf16, int8, int32, fp8_e4m3fn | `overlap_utils.py` torch.where(ids<0, ...) ✅ |
| `arange(int32)` | fp32, bf16, int8, int32, fp8_e4m3fn | `overlap_utils.py` alloc_future_indices、`forward_batch_info.py` extend_start_loc ✅ |
| `neg(int32)` | fp32, bf16, int8, int32, fp8_e4m3fn | `scheduler.py` -future_indices.indices、`overlap_utils.py` -input_ids ✅ |
| `sgl_kernel_zeus.build_kv_indices` | int32 ragged gather（五件套落地） | `zeus_backend.py` kv_indices 构建：CPU mirror gather → device kernel ✅ |
| `index.Tensor` / `index.Tensor_out` | int32/int64 indices，2-D fancy index（05-08③） | `alloc_for_decode` last_loc、`filter_batch`、`_resolve_future_token_ids`、`logits_processor.py` gather loop ✅ |

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

- [common.py](/root/workspace/sglang/python/sglang/srt/mem_cache/common.py)

✅ **已解决（05-08③）**：`alloc_for_decode` 中 Zeus 分支的 `req_to_token_cpu` mirror 读取已删除。`aten::index.Tensor` 确认可用后，统一改为：

```python
last_loc = batch.req_to_token_pool.req_to_token[
    batch.req_pool_indices, batch.seq_lens - 1
]
seq_lens_next = batch.seq_lens + token_per_req
```

`_is_zeus` 变量及 `is_zeus` import 已从 `common.py` 删除（该文件不再有 Zeus 特判）。

---

## 4.2 Overlap / Scheduler 路线

### A. `managers/overlap_utils.py`

参考：

- [overlap_utils.py](/root/workspace/sglang/python/sglang/srt/managers/overlap_utils.py)

✅ **已解决（05-08③）**：`_resolve_future_token_ids` 中的 `_is_zeus` 分支（`buf.cpu() + gather_indices.cpu()` CPU gather）已删除，统一走 device 路径：

```python
input_ids[:] = torch.where(
    input_ids < 0,
    future_token_ids_map[torch.clamp(-input_ids, min=0)],
    input_ids,
)
```

保留的 `_is_zeus` 用途：`alloc_future_indices` 中 dtype 选择（`int32` vs `int64`，Zeus 无 int64 向量支持）；`token_ids_buf` dtype 选择（`int32`）。这些不属于 CPU bounce，正常保留。

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

✅ **已解决（05-08③）**：`filter_batch` 中的 `_is_zeus` 分支（`tensor.cpu()[ki_cpu].to(device)`）已删除，统一使用 `tensor[keep_indices_device]`（device 上已有 `keep_indices_device = torch.tensor(keep_indices, dtype=torch.int64).to(self.device)`）。

保留的 `_is_zeus`（line 1779）：`if self.enable_overlap or _is_zeus:` 选择 `+= 1` vs `add_()` 的 in-place 写法，与 CPU bounce 无关，正常保留。

---

## 4.5 Logits Processor 路线

参考：

- [logits_processor.py:420](/root/workspace/sglang/python/sglang/srt/layers/logits_processor.py:420)

✅ **已解决（05-08③）**：`_is_zeus` 分支（CPU for 循环 `copy_` gather）已删除。`aten::index.Tensor`、`cumsum(int32)`、`arange(int32)`、`sub/mul` 均已原生支持，统一走 device 路径：

```python
last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
pruned_states = hidden_states[last_index]
```
（padded 路径同理，`arange + mul + add - 1` 全在 device 上执行）

---

## 4.6 `radix_cache.py`

参考：

- [radix_cache.py](/root/workspace/sglang/python/sglang/srt/mem_cache/radix_cache.py)

✅ `_cat_zeus()` 已不存在于当前代码，`_is_zeus` 变量声明保留但无其他使用点（可后续一并清理）。

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

### P0（已全部完成）

- ✅ `radix_cache.py` `_cat_zeus()`：已不存在
- ✅ `schedule_batch.py` filter_batch CPU 分支：05-08③ 删除
- ✅ `schedule_batch.py` / `common.py` 小算术 Zeus 分支：已清理

### P1（已全部完成）

- ✅ `alloc_for_decode` last_loc mirror 读取：05-08③ 改为 device index
- ✅ `overlap_utils._resolve_future_token_ids` gather：05-08③ 统一 device
- ✅ `logits_processor.py` gather loop：05-08③ 统一 device

### P2：剩余关注点（非 CPU bounce）

1. **fp32 linearize 精度上限**：`index_get.cpp` flat offset 用 fp32 累加，`pool_size × ctx_len > 2^{24}` 时可能溢出 → 向 `index_get.cpp` 反馈改用 int32/int64 累加器
2. **`req_pool_indices` dtype**：`alloc_for_extend` 仍以 `int64` 构造；`zeus_backend._build_kv_indices_device` 和 `attention.py` 各有一次防御性 cast → 可将 `alloc_for_extend` 改用 `zeus_index_dtype` 从根源消除 cast
3. **`req_to_token_cpu` mirror 弱化**：`last_loc` 已 device 化；mirror 现只服务 `_build_kv_indices_cpu` 回退路径，等 device kernel 覆盖率稳定后可降级为 debug-only

## 6.1 剩余 `cumsum` 调用点优先级表

`aten::cumsum` 已由 `torch_zeus` 支持。这里不再把 `cumsum` 当作缺失算子，
而是把代码中仍值得关注的历史 CPU workaround 按 `主链路 / batch / logits`
三类重新整理如下。

| 类别 | 代表位置 | 当前作用 | 优先级 | 原因 |
|------|----------|----------|--------|------|
| 主链路 | [zeus_backend.py:100](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:100), [zeus_backend.py:149](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:149), [zeus_backend.py:168](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/attention\/zeus_backend.py:168) | 构 `kv_indptr / qo_indptr` | 已处理 | 已改为 Zeus device `torch.cumsum`；剩余问题转为 `kv_indices` 的 ragged gather |
| Batch | [forward_batch_info.py:1081](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1081), [forward_batch_info.py:1118](\/root\/workspace\/sglang\/python\/sglang\/srt\/model_executor\/forward_batch_info.py:1118) | 构 `prefix_chunk_cu_seq_lens`、`kv_indptr` | 已处理 | 这两处是纯 `cumsum` CPU 绕路，已改为 Zeus device `torch.cumsum` |
| Batch | [forward_batch_info.py:compute_position_torch](/root/workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py:1257) | 构 `extend_start_loc` via cumsum | ✅ 已处理 | `arange(int32)` + `cumsum` 均已原生 Zeus；仍需 `.cpu().tolist()` 获取 Python 标量 |
| Logits | [logits_processor.py:420](\/root\/workspace\/sglang\/python\/sglang\/srt\/layers\/logits_processor.py:420) | 构最后 token 的线性下标 | ✅ 已处理（05-08③） | `index.Tensor` 可用后，`_is_zeus` gather loop 已删除；统一走 `hidden_states[last_index]` |

补充说明：

- 主链路里的 `cumsum` 已经回到 Zeus device 上执行
- 纯 `cumsum` 的 batch 路径已处理
- logits 和混合 batch 路径还依赖其他 CPU/index 能力，不能只替换 `cumsum`
- Paged KV metadata 的剩余 host 工作，核心已经变成 `kv_indices` 的 ragged gather

---

## 7. 一句话总结

这份文档按最新代码重排后，最重要的变化是：

> `aten::index.Tensor`（`index_get.cpp`）于 2026-05-08③ 确认可用后，主链路和 batch/logits 处理路径的所有 `index(read)` CPU bounce 已全部消除：`alloc_for_decode` last_loc 直接读 device `req_to_token`，`filter_batch` 统一用 `keep_indices_device`，`_resolve_future_token_ids` 和 `logits_processor` gather 统一走 device 路径。当前 SGLang Zeus 路径中不再有"因 operator 缺失导致的不必要 CPU bounce"，仅保留设计上必要的 CPU bookkeeping（`ZeusPagedTokenToKVPoolAllocator`）、dtype 选择（`int32` vs `int64`）和 `_build_kv_indices_cpu` 回退 mirror。后续关注点转为 fp32 linearize 精度上限和 `req_pool_indices` dtype 统一。
