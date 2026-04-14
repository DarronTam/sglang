# SGLang Zeus Graph 集成 — 代码变更记录

> 日期：2026-04-13
> 分支：dev
> 关联计划：[zeus_graph_integration_plan.md](./zeus_graph_integration_plan.md)
> 覆盖 Phase：Phase 1（骨架）、Phase 3（CPU Bounce 处理）、Phase 4（多 BS 支持与调优）

---

## 一、变更概览

| # | 文件 | 变更类型 | 关联 Phase | 行数变化 |
|---|------|----------|------------|----------|
| 1 | `python/sglang/srt/hardware_backend/zeus/graph_runner/zeus_graph_runner.py` | **新建** | Phase 1 | +46 |
| 2 | `python/sglang/srt/layers/attention/zeus_backend.py` | 修改 | Phase 1 + 4 | +86 |
| 3 | `python/sglang/srt/model_executor/model_runner.py` | 修改 | Phase 1 | +4 |
| 4 | `python/sglang/srt/server_args.py` | 修改 | Phase 1 + 4 | +8 / -2 |
| 5 | `python/sglang/srt/layers/sampler.py` | 修改 | Phase 3 | -5 |

---

## 二、逐文件详细变更

### 2.1 新建 `zeus_graph_runner.py`（Phase 1）

**路径**：`python/sglang/srt/hardware_backend/zeus/graph_runner/zeus_graph_runner.py`

**目的**：创建 `ZeusGraphRunner`，继承 `CudaGraphRunner`，最小覆写 3 个方法即可适配 Zeus Graph API。

**完整代码**：

```python
class ZeusGraphRunner(CudaGraphRunner):
    """A ZeusGraphRunner runs the forward pass of a model with Zeus graph capture/replay."""

    def _create_device_graph(self):
        return self.device_module.ZEUSGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        # Zeus graph context manager uses 'zeus_graph=' kwarg (not 'cuda_graph=')
        with self.device_module.graph(zeus_graph=graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _cache_loc_dtype(self):
        # Zeus attention kernels use int32 indices
        return torch.int32
```

**关键设计决策**：

- 采用与 `NPUGraphRunner` 相同的"继承 + 最小覆写"策略
- `_create_device_graph()`：Zeus 使用 `ZEUSGraph()` 而非 `CUDAGraph()`
- `_capture_graph()`：Zeus `graph()` 上下文管理器使用 `zeus_graph=` 参数名（而非基类的 `cuda_graph=`）
- `_cache_loc_dtype()`：Zeus attention kernel 使用 `int32` 索引（而非 CUDA 默认的 `int64`）

---

### 2.2 修改 `zeus_backend.py`（Phase 1 + Phase 4）

**路径**：`python/sglang/srt/layers/attention/zeus_backend.py`

**目的**：为 `ZeusAttnBackend` 增加 graph capture/replay 所需的 4 个钩子方法。

#### 2.2.1 构造函数新增属性

```python
# 新增（Phase 1）
self.max_context_len = model_runner.model_config.context_len
self.req_to_token = model_runner.req_to_token_pool.req_to_token
```

- `max_context_len`：用于预分配 `kv_indices` buffer 的最大长度（`max_bs * max_context_len`）
- `req_to_token`：CSR 索引构建时需要从 req_to_token pool 查找 token 位置

#### 2.2.2 新增 `init_cuda_graph_state()`

```python
def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
    """Pre-allocate fixed-size tensors for graph capture/replay."""
    self.cuda_graph_kv_indptr = torch.zeros(
        max_bs + 1, dtype=torch.int32, device=self.device
    )
    max_total_kv = max_bs * self.max_context_len
    self.cuda_graph_kv_indices = torch.zeros(
        max_total_kv, dtype=torch.int32, device=self.device
    )
```

- 预分配固定大小的 `kv_indptr` 和 `kv_indices` buffer
- `kv_indices` 使用 `max_bs * max_context_len` 作为上界，保证任何 replay 场景下都不越界

#### 2.2.3 新增 `init_forward_metadata_capture_cuda_graph()` / `init_forward_metadata_replay_cuda_graph()`

两者共用内部方法 `_fill_decode_metadata_for_graph()`：

```python
def _fill_decode_metadata_for_graph(self, bs, req_pool_indices, seq_lens):
    # 1. seq_lens / req_pool_indices 拉到 CPU
    # 2. CPU 上计算 kv_indptr（cumsum）和 kv_indices（gather from req_to_token）
    # 3. copy_ 到预分配的 device buffer
    # 4. 构建 ZeusAttnMetadata
```

**关键设计**：

- CSR 索引构建全部在 CPU 上完成，`_fill_decode_metadata_for_graph` 在 capture/replay **之前**被调用（不在 graph 内），所以 CPU 工作不影响 graph capture
- `kv_indptr` 使用 `[:bs+1]` slice — 每个 BS 有独立的 graph，shape 在 capture/replay 间不变
- `kv_indices` 使用**完整预分配 buffer**（不做 slice）— 这是 Phase 4 的关键修复

#### 2.2.4 Phase 4 修复：kv_indices 使用完整 buffer

**问题**：最初使用 `cuda_graph_kv_indices[:total_kv]`（变长 slice），graph capture 时 `total_kv=bs`（因为 capture 时 seq_lens 全为 1），replay 时实际 `total_kv` 可能远大于 `bs`，导致 kernel 越界。

**修复**：

```python
# 修复前（有 bug）：
self.forward_metadata = ZeusAttnMetadata(
    kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
    kv_indices=self.cuda_graph_kv_indices[:total_kv],   # 变长 slice，shape 不一致
)

# 修复后：
self.forward_metadata = ZeusAttnMetadata(
    kv_indptr=self.cuda_graph_kv_indptr[: bs + 1],
    kv_indices=self.cuda_graph_kv_indices,               # 完整 buffer，kernel 通过 kv_indptr 确定有效范围
)
```

#### 2.2.5 新增 `get_cuda_graph_seq_len_fill_value()`

```python
def get_cuda_graph_seq_len_fill_value(self):
    return 1
```

基类 `CudaGraphRunner.replay()` 对 padding 位使用此值填充 seq_lens，避免 kernel 访问到非法的 seq_len=0。

#### 2.2.6 TYPE_CHECKING import 更新

```python
# 修改前
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# 修改后
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
```

新增方法签名中使用了 `ForwardMode` 类型注解。

---

### 2.3 修改 `model_runner.py`（Phase 1）

**路径**：`python/sglang/srt/model_executor/model_runner.py`

**目的**：在 graph runner 注册表中添加 Zeus。

**变更**：

```python
# 新增 import
from sglang.srt.hardware_backend.zeus.graph_runner.zeus_graph_runner import (
    ZeusGraphRunner,
)

# 注册表新增 "zeus" 映射
graph_runners = defaultdict(
    lambda: CudaGraphRunner,
    {
        "cpu": CPUGraphRunner,
        "npu": NPUGraphRunner,
        "zeus": ZeusGraphRunner,  # 新增
    },
)
```

`model_runner.py` 通过 `self.device`（值为 `"zeus"`）选择对应的 graph runner。

---

### 2.4 修改 `server_args.py`（Phase 1 + Phase 4）

**路径**：`python/sglang/srt/server_args.py`

**目的**：移除 Zeus 的硬编码 `disable_cuda_graph=True`，启用 graph 支持；修复初始化顺序。

#### 2.4.1 `_handle_zeus_backends()` 改动

```python
# 修改前
def _handle_zeus_backends(self):
    if is_zeus():
        if self.attention_backend is None:
            self.attention_backend = "zeus"
        self.disable_cuda_graph = True                 # 硬编码禁用 graph
        if self.page_size is None:
            self.page_size = 128

# 修改后
def _handle_zeus_backends(self):
    if is_zeus():
        if self.attention_backend is None:
            self.attention_backend = "zeus"
        # Zeus graph capture/replay is supported — don't force-disable
        if self.cuda_graph_max_bs is None:
            self.cuda_graph_max_bs = 32                # 保守默认值
        if self.page_size is None:
            self.page_size = 128
```

- 删除 `self.disable_cuda_graph = True`
- 新增 `cuda_graph_max_bs = 32` 默认值（保守值，默认 capture 8 个 graph：bs=1,2,4,8,12,16,24,32）
- 用户仍可通过 `--cuda-graph-max-bs` 覆盖

#### 2.4.2 Phase 4 修复：初始化顺序

**问题**：`_handle_gpu_memory_settings()` 在 `_handle_zeus_backends()` 之前运行，会根据 GPU 显存设置 `cuda_graph_max_bs`（如 160/256），导致 Zeus 的保守默认值 32 永远不生效。

**修复**：将 `_handle_zeus_backends()` 移到 `_handle_gpu_memory_settings()` 之前：

```python
# 修改前的调用顺序：
self._handle_gpu_memory_settings(gpu_mem)      # 可能设置 cuda_graph_max_bs=160
self._handle_hpu_backends()
self._handle_cpu_backends()
self._handle_npu_backends()
self._handle_zeus_backends()                   # 此时 cuda_graph_max_bs 已非 None，32 不生效

# 修改后的调用顺序：
self._handle_zeus_backends()                   # 先设 cuda_graph_max_bs=32
self._handle_gpu_memory_settings(gpu_mem)      # 看到非 None，不覆写
self._handle_hpu_backends()
self._handle_cpu_backends()
self._handle_npu_backends()
```

---

### 2.5 修改 `sampler.py`（Phase 3）

**路径**：`python/sglang/srt/layers/sampler.py`

**目的**：移除 greedy 采样路径中的 Zeus CPU roundtrip。

**变更**：

```python
# 删除的全局变量
_is_zeus = is_zeus()

# 修改前的 greedy 路径
if sampling_info.is_all_greedy:
    if _is_zeus:
        batch_next_token_ids = torch.argmax(logits.cpu(), -1).to(logits.device)
    else:
        batch_next_token_ids = torch.argmax(logits, -1)

# 修改后
if sampling_info.is_all_greedy:
    batch_next_token_ids = torch.argmax(logits, -1)
```

**背景**：早期 `torch_zeus` 不支持原生 `argmax`，需要 `.cpu()` → `argmax` → `.to(device)` 绕行。现在 `torch_zeus` 已支持原生 `argmax`，每个 decode step 节省一次 D2H + H2D 往返。

---

## 三、与 Plan Phase 对应关系

| Phase | 状态 | 本次变更覆盖的任务 |
|-------|------|-------------------|
| Phase 0 — 前置验证 | ✅ 已完成（之前） | — |
| Phase 1 — ZeusGraphRunner 骨架 | ✅ 已完成 | 新建 `zeus_graph_runner.py`；`zeus_backend.py` 增加 4 个 graph 方法；`model_runner.py` 注册；`server_args.py` 移除硬编码 disable |
| Phase 2 — Attn Backend Graph 支持 | ✅ 已合并到 Phase 1 | `zeus_backend.py` 的 graph 方法实现 |
| Phase 3 — CPU Bounce 处理 | ✅ 已完成 | `sampler.py` 移除 `_is_zeus` + argmax CPU roundtrip |
| Phase 4 — 多 BS 支持与调优 | ✅ 已完成 | `server_args.py` 初始化顺序修复；`zeus_backend.py` kv_indices 完整 buffer 修复 |
| Phase 5 — 端到端验证 | ⬜ 待真实硬件 | 测试脚本已就绪（`test_zeus_graph_e2e.py`），代码变更不涉及 |

---

## 四、设计要点总结

### 4.1 继承 + 最小覆写

`ZeusGraphRunner` 仅覆写 3 个方法（`_create_device_graph` / `_capture_graph` / `_cache_loc_dtype`），多 BS capture、replay 调度、input buffer 管理全部复用基类 `CudaGraphRunner`。

### 4.2 CSR 构建在 graph 外

`_fill_decode_metadata_for_graph` 在 capture/replay **之前**被基类调用，CPU 上的 `cumsum` / `gather` 不会被 graph 录制。后续可作为 P1 优化移到 device 端（需 `cumsum` kernel）。

### 4.3 kv_indices 使用完整 buffer

Graph capture 会录制 tensor shape。`kv_indices` 始终使用完整预分配 buffer，kernel 通过 `kv_indptr` 确定有效访问范围，避免 capture/replay 间 shape 不一致导致越界。

### 4.4 初始化顺序保证默认值

Zeus 的 `_handle_zeus_backends()` 必须在 `_handle_gpu_memory_settings()` 之前调用，确保保守的 `cuda_graph_max_bs=32` 不被通用 GPU 显存启发式覆盖。

---

## 五、下一步

1. **真实硬件验证**：在 Zeus 硬件上运行 `python zeus_dev/test_zeus_graph_e2e.py`，验证 graph mode 正确性和性能
2. **性能数据**：收集 graph vs eager 的吞吐量 / 延迟 / 显存对比，结果存入 `zeus_dev/zeus_graph_perf_results.json`
3. **P1 优化**（可选）：将 CSR 构建移到 device 端，消除 metadata 构建的 CPU bounce（需 `cumsum` kernel）
