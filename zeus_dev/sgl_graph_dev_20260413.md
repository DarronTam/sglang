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

---

## 六、2026-04-16 补充：Zeus metadata fill 的 CPU 协助

### 6.1 问题本质

本轮补丁处理的不是模型主计算 kernel，而是 **graph input / metadata tensor 的标量填充能力缺口**。

当前 `torch_zeus` 在 Zeus device 上对以下路径支持不完整：

| 类别 | 典型调用 | 当前表现 |
|------|----------|----------|
| device 端整型填充 | `torch.full(..., dtype=torch.int32, device="zeus")` | ❌ 不支持 |
| device 端 bool 填充 | `torch.ones(..., dtype=torch.bool, device="zeus")` | ❌ 不支持（底层同样会走 fill 路径） |
| 原位标量填充 | `tensor.fill_(x)`，其中 `tensor.dtype in {bool, int16, int32, int64}` | ❌ 不支持 |
| 标量广播赋值 | `tensor[...] = scalar`，若底层退化到 fill | ❌ 不支持 |

服务启动时实际见到的报错形式包括：

- `Zeus fill_ does not support dtype Int`
- `Zeus fill_ does not support dtype Bool`

这类张量多数不是模型主数据，而是 graph capture / replay 需要的固定 shape 控制信息：

- `seq_lens`
- `custom_mask`
- `encoder_lens`
- `global_num_tokens_gpu`
- `global_num_tokens_for_logprob_gpu`
- `num_token_non_padded`

因此补丁的目标不是“把计算放回 CPU”，而是更窄地做：

1. **在 CPU 上构造 filled template**
2. **再 `copy_` 到 Zeus 常驻 buffer**

模型 forward、attention、KV cache 更新仍然在 Zeus 上执行。

### 6.2 当前 workaround 依赖的算子支持

要让 CPU 协助路径成立，Zeus 至少需要具备下面这些基础能力：

| 能力 | 例子 | 用途 |
|------|------|------|
| Zeus 上分配零值 tensor | `torch.zeros(shape, dtype=..., device="zeus")` | 先建出目标 buffer |
| CPU 上构造 filled tensor | `torch.full(shape, fill_value, dtype=..., device="cpu")` | 在 CPU 侧生成模板值 |
| CPU -> Zeus 拷贝 | `zeus_tensor.copy_(cpu_tensor)` | 把 filled template 写回 Zeus |
| Zeus -> Zeus 拷贝 | `zeus_tensor.copy_(other_zeus_tensor)` | 正常 replay 时用真实 batch 数据覆盖 buffer |
| 普通 slice / view | `buffer[:bs]`、`buffer[:num_tokens]` | graph buffer 切片传给 `ForwardBatch` |

这意味着 workaround 依赖的是：

- `zeros`
- `copy_`
- 常规 tensor slicing

而不是依赖：

- `fill_`
- Zeus device 上的 `torch.full`
- Zeus device 上的 `torch.ones(bool)`

### 6.3 新增 helper

本轮在 `python/sglang/srt/model_executor/input_buffers.py` 中新增两个 helper：

- `create_filled_tensor(...)`
- `fill_tensor_(...)`

实现策略：

```python
if not zeus_bool_or_int_tensor:
    # 原生路径
    return torch.full(...) / tensor.fill_(...)

# Zeus bool/int metadata tensor:
tensor = torch.zeros(..., device="zeus")
src = torch.full(..., device="cpu")
tensor.copy_(src)
```

只对 `dtype in {bool, int16, int32, int64}` 且 `device.type == "zeus"` 生效，避免影响 CUDA / CPU / NPU 路径。

### 6.4 已落地的 CPU 协助点

下表汇总当前已经加了 CPU 协助的地方、它们各自做什么，以及成本类别。

| 位置 | 调用点 | CPU 协助方式 | 这些 tensor 的作用 | 成本类别 |
|------|--------|--------------|--------------------|----------|
| `input_buffers.py:108` | `seq_lens = create_filled_tensor(...)` | CPU `full` -> Zeus `copy_` | 为 graph buffer 预填充每个 request 的默认 `seq_len`，给 padding 槽位一个合法 sentinel 值 | **启动一次性成本**（每次 graph create / recapture 一次） |
| `input_buffers.py:118` | `custom_mask = create_filled_tensor(..., True, dtype=torch.bool)` | CPU `full(True)` -> Zeus `copy_` | 构造 decode graph 的默认 custom attention mask | **启动一次性成本** |
| `input_buffers.py:138` | `encoder_lens = create_filled_tensor(...)` | CPU `full` -> Zeus `copy_` | encoder-decoder 模型下，为 padding 槽位提供默认 encoder length | **启动一次性成本** |
| `input_buffers.py:197` | `fill_tensor_(self.seq_lens, seq_len_fill_value)` | CPU `full` -> Zeus `copy_` | replay 前把未使用 batch 槽位恢复到默认 `seq_len`，保证 graph 输入 shape 固定且数值合法 | **每次请求重复成本**（graph replay 前执行） |
| `input_buffers.py:221` | `fill_tensor_(self.global_num_tokens_gpu, bs * num_tokens_per_bs)` | CPU `full` -> Zeus `copy_` | 为 DP/TP gather 路径写入“本轮 graph replay 的 token 总数” | **每次请求重复成本** |
| `input_buffers.py:222` | `fill_tensor_(self.global_num_tokens_for_logprob_gpu, ...)` | CPU `full` -> Zeus `copy_` | 为 logprob 相关 gather 路径写入 token 总数 | **每次请求重复成本** |
| `cuda_graph_runner.py:558` | `fill_tensor_(buffers.num_token_non_padded, num_tokens)` | CPU `full` -> Zeus `copy_` | capture 时写入“真实 token 数”，区分 graph 固定 shape 与真实非 padding token 数 | **启动一次性成本**（每个 capture BS 一次；recapture 会重复） |
| `model_runner.py:2318` | `fill_tensor_(buffers.num_token_non_padded, num_tokens)` | CPU `full` -> Zeus `copy_` | dummy warmup / 初始化路径里同步设置真实 token 数 | **启动一次性成本** |
| `model_runner.py:2325` | `extend_seq_lens = create_filled_tensor(...)` | CPU `full` -> Zeus `copy_` | 非 generation 的 dummy extend 路径预填默认 `seq_len` | **启动一次性成本** |

### 6.5 各字段的具体职责

#### 6.5.1 `seq_lens`

- **语义**：每个 request 当前的序列长度
- **为什么需要默认值**：graph capture 使用固定 `max_bs`，当真实 batch 小于 `max_bs` 时，后面的 padding 槽位仍然必须有合法 `seq_len`
- **下游用途**：
  - attention metadata 构建
  - `ForwardBatch.seq_lens`
  - `ForwardBatch.seq_lens_sum`

#### 6.5.2 `custom_mask`

- **语义**：自定义 attention mask buffer
- **为什么需要默认全 True**：graph 初始化时先给出“允许全部”的基础 mask，后续按具体路径覆盖或切片使用
- **下游用途**：attention backend / 特殊 masking 路径的元数据输入

#### 6.5.3 `encoder_lens`

- **语义**：encoder-decoder 模型中 encoder 侧长度
- **为什么需要默认值**：padding 槽位也要满足固定 shape graph 的输入契约
- **下游用途**：enc-dec attention metadata

#### 6.5.4 `global_num_tokens_gpu`

- **语义**：DP/TP gather 视角下的 token 总数
- **为什么每次 replay 都要重写**：不同请求的 `bs` / `num_tokens_per_bs` 会变化
- **下游用途**：
  - `ForwardBatch.global_num_tokens_gpu`
  - DP/TP gather buffer 长度计算

#### 6.5.5 `global_num_tokens_for_logprob_gpu`

- **语义**：logprob 相关路径使用的 token 总数
- **下游用途**：
  - `ForwardBatch.global_num_tokens_for_logprob_gpu`
  - logprob 路径的 gather / reduce 元数据

#### 6.5.6 `num_token_non_padded`

- **语义**：当前 graph batch 中真实非 padding token 数
- **为什么 capture 和 dummy init 都要设置**：很多下游逻辑需要区分“固定 shape graph token 数”与“真实 token 数”
- **下游用途**：
  - `ForwardBatch.num_token_non_padded`
  - `compute_local_num_token_non_padded(...)`
  - 一些 attention / logits / gather 元数据路径

#### 6.5.7 `extend_seq_lens`

- **语义**：非 generation dummy extend 路径中的序列长度模板
- **下游用途**：warmup / extend graph 初始化时构造 `ForwardBatch`

### 6.6 启动一次性成本 vs 每次请求重复成本

#### 6.6.1 启动一次性成本

这类成本只在 graph 初始化、dummy run、或 recapture 时发生：

- `seq_lens` 初始模板创建
- `custom_mask` 初始模板创建
- `encoder_lens` 初始模板创建
- capture 阶段的 `num_token_non_padded`
- dummy init 阶段的 `num_token_non_padded`
- dummy extend 阶段的 `extend_seq_lens`

特点：

- 次数少
- 多数发生在服务启动、模型加载完成后的 graph capture 阶段
- 即使 tensor 较大（如 `custom_mask`），也不是每个请求都付费

#### 6.6.2 每次请求重复成本

这类成本发生在 graph replay 前的 buffer 复用阶段：

- `seq_lens` reset
- `global_num_tokens_gpu` 重写
- `global_num_tokens_for_logprob_gpu` 重写

特点：

- 会随每次 graph replay 重复发生
- 但张量规模通常较小，主要是 `bs` 或 `dp_size` 级别
- 本质是 metadata 更新成本，不是主算子计算成本

### 6.7 当前评估

这批 CPU 协助的影响应这样理解：

1. **不是主计算回退**
   forward、attention、KV cache 仍在 Zeus 上执行。

2. **主要是 metadata 填充补丁**
   问题集中在 `fill_` / `torch.full` / bool-int 控制张量。

3. **短期工程上可接受**
   现有实现已经能让 `sglang.launch_server` 在 Zeus graph 模式下成功启动。

4. **长期最值得补齐的仍是 Zeus 原生 fill 支持**
   如果 `torch_zeus` 后续补上以下能力，这一整批 CPU 协助都可以显著简化：

   - `torch.full(..., dtype=int32/bool, device="zeus")`
   - `zeus_int_tensor.fill_(scalar)`
   - `zeus_bool_tensor.fill_(True/False)`
   - 标量广播赋值到 Zeus int/bool tensor
