# SGLang → Zeus NPU 移植手册

> 日期：2026-04-09（更新）  
> 代码版本：sglang_dev  
> 基线分叉点：`94e125113`（sglang main）  
> 状态：Dense LLM (Qwen2.5/Llama) 推理已跑通，torch_zeus 新增 cat/add/sub/mul/reduce 原生算子，CPU bounce 可大幅消除

---

## 目录

**第一部分：移植现状**

1. [移植总览：为什么需要移植以及移植了什么](#1-移植总览)
   - 1.1 背景
   - 1.2 移植目标
   - 1.3 当前状态
   - 1.4 变更规模

**第二部分：架构与方法**

2. [SGLang 架构与 Zeus 移植切入点](#2-sglang-架构与-zeus-移植切入点)
   - 2.1 SGLang 推理架构
   - 2.2 移植切入点总结
3. [移植方法论：六大改造维度](#3-移植方法论六大改造维度)
   - 3.1 设备识别与全局配置
   - 3.2 CustomOp 算子派发
   - 3.3 Zeus Attention Backend
   - 3.4 KV Cache 内存管理
   - 3.5 模型权重加载与 LocalMem 打包
   - 3.6 ATen 缺失算子的 CPU Bounce 规避

**第三部分：技术细节**

4. [Zeus 自定义算子清单 (sgl_kernel_zeus)](#4-zeus-自定义算子清单)
5. [CUDA vs Zeus 全流程对比](#5-cuda-vs-zeus-全流程对比)
   - 5.1 初始化阶段
   - 5.2 单次 Decode Step 对比
   - 5.3 关键模块性能差距
   - 5.4 量化估算
6. [已修改文件完整清单与修改原因](#6-已修改文件完整清单)
   - 6.1 新增文件
   - 6.2 修改文件
7. [已知性能瓶颈与 CPU↔Zeus 传输热点](#7-已知性能瓶颈)
   - 7.1 CPU↔Zeus 传输热点汇总
   - 7.2 严重问题分级

**第四部分：后续开发规划**

8. [后续开发路线图](#8-后续开发路线图)
   - 8.1 总体路线：四大阶段
   - 8.2 阶段一：Graph 接入（计算图捕获/重放）
   - 8.3 阶段二：分布式接入（多卡 TP/PP）
   - 8.4 阶段三：PD 分离接入（Prefill-Decode Disaggregation）
   - 8.5 阶段四：继续理清算子流程
   - 8.6 辅助性 TODO（持续推进）
   - 8.7 路线图甘特视图

**第五部分：参考**

9. [开发避坑指南](#9-开发避坑指南)
   - 9.1 LocalMem 权重转置
   - 9.2 _is_zeus 判断顺序
   - 9.3 cos_sin_cache 构建
   - 9.4 隐式 CPU Fallback 的识别
   - 9.5 req_to_token 矩阵拷贝优化
   - 9.6 lm_head 权重来源检测
10. [附录：推理全流程调用链路图](#10-附录推理全流程调用链路图)
    - 10.1 Engine 初始化完整链路
    - 10.2 Decode Step 完整链路
    - 10.3 Zeus 缺失的 ATen 算子汇总

---

## 1. 移植总览

### 1.1 背景

SGLang 是一个高性能 LLM 推理服务框架，原生以 NVIDIA CUDA GPU 为第一公民构建，核心依赖：

- **FlashInfer**：CUDA 专用 PagedAttention kernel
- **Triton**：CUDA PTX 代码生成
- **NCCL**：CUDA GPU 集合通信
- **CUDA Graph**：计算图捕获/重放，消除 kernel launch 开销

Zeus NPU 是全新硬件平台，以上依赖 **全部不可用**，需要用 Zeus 原生替代方案逐一替换。

### 1.2 移植目标

将 SGLang 的 LLM 推理链路完整跑通在 Zeus NPU 上，包括：

- **Prefill**（一次性处理整个 prompt）
- **Decode**（逐 token 自回归生成）
- **Paged KV Cache**（分页式 KV 缓存管理）
- **多种采样策略**（greedy / top-k / top-p / min-p）

### 1.3 当前状态

| 项目 | 状态 |
|------|------|
| Dense Transformer 推理 (Llama/Qwen) | ✅ 功能正常 |
| Paged KV Cache + Zeus Attention | ✅ 可用 |
| 采样 (greedy / top-k / top-p / min-p) | ✅ 可用 |
| 权重 LocalMem 打包 | ✅ 可用 |
| sgl-kernel-zeus 算子 | ✅ 14 个已实现 |
| SWA / MoE / 量化 / Speculative | ❌ 未支持 |
| CUDA Graph (计算图重放) | ❌ 已禁用 |
| 生产级性能优化 | 🔧 进行中 |

### 1.4 变更规模

| 指标 | 数值 |
|------|------|
| 新增文件 | 9 个（3 个核心 + 6 个文档/工具） |
| 修改文件 | 21 个 |
| 总变更行数 | +3508 / −131 |
| `_is_zeus` 检查点 | ~44 个分布在 15+ 文件中 |

---

## 2. SGLang 架构与 Zeus 移植切入点

### 2.1 SGLang 推理架构

```
用户代码 (demo_zeus_llm.py)
  │
  ├── sgl.Engine(model_path=..., device="zeus", ...)
  │     │
  │     ├── ServerArgs 解析
  │     │     └── _handle_zeus_backends()
  │     │           → attention_backend="zeus"
  │     │           → sampling_backend="zeus"
  │     │           → page_size=128
  │     │           → disable_cuda_graph=True
  │     │
  │     └── _launch_subprocesses()
  │           ├── [主进程] TokenizerManager          ← 分词
  │           ├── [子进程] Scheduler                  ← 调度 + 前向推理
  │           │     ├── TpModelWorker
  │           │     │     └── ModelRunner              ← 模型加载、KV Cache、forward
  │           │     │           ├── ZeusAttnBackend     ← 注意力算子后端
  │           │     │           ├── ZeusTokenToKVPool   ← KV Cache 存储池
  │           │     │           ├── ZeusPagedTokenToKVPoolAllocator  ← KV Cache 分配
  │           │     │           └── Sampler (zeus)      ← 采样后端
  │           │     └── 事件循环 (event_loop_normal)
  │           └── [子进程] DetokenizerManager         ← 反分词
  │
  └── llm.generate(prompts, sampling_params)
        └── TokenizerManager → [ZMQ IPC] → Scheduler → ModelRunner.forward()
```

### 2.2 移植切入点总结

SGLang 的架构为硬件适配提供了天然的插件点，Zeus 移植主要在以下 6 个层面展开：

| 层面 | 切入点 | 作用 |
|------|--------|------|
| 设备识别 | `DeviceConfig` / `ServerArgs` / `utils/common.py` | 让框架认识和正确配置 Zeus 设备 |
| 算子派发 | `CustomOp.dispatch_forward()` → `forward_zeus` | 将计算路由到 Zeus 专用 kernel |
| Attention 后端 | `attention_registry` → `ZeusAttnBackend` | 封装 Zeus 硬件 attention kernel |
| KV Cache | `ZeusTokenToKVPool` + `ZeusPagedTokenToKVPoolAllocator` | 适配 Zeus tiled 内存布局 |
| 权重管理 | `loader.py` → `pack_weights()` | 将权重打包到 LocalMem |
| ATen 缺失算子 | 各处 `if _is_zeus:` CPU bounce | 规避 Zeus 未实现的 ATen 算子 |

---

## 3. 移植方法论：六大改造维度

### 3.1 设备识别与全局配置

**核心原理：** SGLang 的设备抽象层原本只覆盖 CUDA/XPU/HPU/CPU/NPU，Zeus 作为通过 `torch.register_privateuse1_backend("zeus")` 注册的 PrivateUse1 设备，不在任何白名单中。必须让整个框架正确识别 Zeus。

**涉及文件：**
- `python/sglang/srt/utils/common.py` — `is_zeus()` 检测等设备工具函数
- `python/sglang/srt/configs/device_config.py` — 设备白名单增加 "zeus"
- `python/sglang/srt/server_args.py` — 自动配置 Zeus 默认参数

**关键配置项：**

| 配置项 | Zeus 值 | CUDA 默认值 | 设置原因 |
|--------|---------|-------------|----------|
| `attention_backend` | `"zeus"` | `"flashinfer"` | Zeus 有专用硬件 attention kernel |
| `sampling_backend` | `"zeus"` | `"flashinfer"` | Zeus 有融合采样 kernel |
| `page_size` | `128` (默认，可自定义但必须为 128 的倍数) | `1` | Zeus attention kernel 硬件要求 |
| `disable_cuda_graph` | `True` | `False` | Zeus 的 graph capture 机制未就绪 |
| 分布式后端 | `"zecl"` | `"nccl"` | Zeus 专用集合通信库 |

**移植模式：**
```python
# server_args.py — Zeus 自动配置
def _handle_zeus_backends(self):
    self.attention_backend = "zeus"
    self.disable_cuda_graph = True
    if self.page_size is None:
        self.page_size = 128
    assert self.page_size % 128 == 0, "Zeus requires page_size aligned to 128"
```

---

### 3.2 CustomOp 算子派发

**核心原理：** SGLang 所有计算密集型算子（RMSNorm、SiluAndMul、RotaryEmbedding 等）继承 `CustomOp` 基类，通过 `dispatch_forward()` 按设备类型选择实现。Zeus 需要新增一个 `forward_zeus` 派发分支。

**涉及文件：**
- `python/sglang/srt/custom_op.py` — 新增 `forward_zeus` 方法和派发逻辑
- `python/sglang/srt/layers/activation.py` — `SiluAndMul.forward_zeus`
- `python/sglang/srt/layers/layernorm.py` — `RMSNorm.forward_zeus`
- `python/sglang/srt/layers/rotary_embedding.py` — `RotaryEmbedding.forward_zeus`
- `python/sglang/srt/layers/vocab_parallel_embedding.py` — `VocabParallelEmbedding.forward` Zeus 分支

**派发机制：**
```python
# custom_op.py
def dispatch_forward(self):
    if _is_zeus:                    # ← Zeus 判断放在 _is_cuda 之前
        return self.forward_zeus    #    因为两者可能同时为 True（容器环境）
    elif _is_cuda:
        return self.forward_cuda
    elif _is_hip:
        return self.forward_hip
    ...

def forward_zeus(self, *args, **kwargs):
    return self.forward_native(*args, **kwargs)  # 基类默认回退到纯 PyTorch
```

**每个算子的 Zeus 实现都调用 `sgl_kernel_zeus` 中的硬件专用融合 kernel：**

| 算子 | 纯 PyTorch 步骤 | Zeus 融合 kernel | 收益 |
|------|-----------------|-----------------|------|
| `rmsnorm` | 3 步 (mean→rsqrt→mul) | 1 次 kernel | 减少 2 次访存 |
| `fused_add_rmsnorm` | 4 步 (add+mean+rsqrt+mul) | 1 次 kernel | 省 1 次全量读写 |
| `silu_and_mul` | 2 步 (silu→mul) | 1 次 kernel | 减少 1 次访存 |
| `rotary_embedding` | 6+ 步 (查表+切片+旋转+拼接) | 1 次 kernel | 全部在片上完成 |
| `embedding` | F.embedding (行索引) | sgl_kernel_zeus.embedding (列 gather) | 适配 LocalMem 转置布局 |

---

### 3.3 Zeus Attention Backend

**核心原理：** SGLang 的 attention 后端是插件式架构（flashinfer/triton/torch_native），通过注册装饰器。Zeus NPU 有自己的硬件 attention 计算单元，需要独立后端封装 Zeus 特有的 CSR 格式 metadata 构建和 kernel 调用。

**涉及文件（新增）：**
- `python/sglang/srt/layers/attention/zeus_backend.py` — Zeus Attention 后端实现 (+194 行)
- `python/sglang/srt/layers/attention/attention_registry.py` — 注册 "zeus" 后端名

**设计要点：**

| 设计选择 | 原因 |
|---------|------|
| metadata (kv_indptr/kv_indices) 在 CPU 侧构建 | Zeus 缺少 cumsum/arange/花式索引的 ATen 支持，CPU 构建后一次性传到 device |
| 分离 `forward_extend()` 和 `forward_decode()` | Zeus 的 prefill (变长 Q) 和 decode (Q 长度=1) 使用不同硬件 kernel |
| KV buffer 为 4D tiled 布局 `[pages, heads, page_size, head_dim]` | Zeus attention 计算单元直接从此布局读取 |

**init_forward_metadata 流程（每次 forward 调用）：**
```
init_forward_metadata(forward_batch)
  ├── seq_lens_cpu = forward_batch.seq_lens.cpu()          # T1: Zeus→CPU
  ├── req_pool_indices_cpu = forward_batch.req_pool_indices.cpu()  # T2
  ├── req_to_token_cpu = req_to_token.cpu()                # T3: ⚠️ 整个矩阵！
  ├── [CPU] 构建 kv_indptr (cumsum), kv_indices (for 循环拼接)
  └── kv_indptr.to(device), kv_indices.to(device)          # T4: CPU→Zeus
```

> **⚠️ 性能问题：** T3 拷贝整个 req_to_token 矩阵（**64 MB**，128×131072×4B int32），是当前最大的传输瓶颈。详见 [第 7 节](#7-已知性能瓶颈) P0。

---

### 3.4 KV Cache 内存管理

**核心原理：** Zeus 的 attention kernel 要求 KV cache 以 4D tiled 布局存储，且写入必须通过专用 kernel 处理硬件 tiled 格式。原版 flat 2D 布局不兼容。

**涉及文件（新增）：**
- `python/sglang/srt/mem_cache/zeus_memory_pool.py` — `ZeusTokenToKVPool` (+122 行)
- `python/sglang/srt/mem_cache/zeus_allocator.py` — `ZeusPagedTokenToKVPoolAllocator` (+212 行)

#### ZeusTokenToKVPool vs MHATokenToKVPool

| 方面 | CUDA (MHATokenToKVPool) | Zeus (ZeusTokenToKVPool) |
|------|------------------------|--------------------------|
| Buffer 布局 | 扁平: `[total_tokens, heads, head_dim]` | Tiled: `[num_pages, heads, page_size, head_dim]` |
| 写入方式 | `k_buffer[loc] = k` 直接索引 | `sgl_kernel_zeus.store_kv_cache()` 专用 kernel |
| 异步写入 | ✅ alt_stream 异步 | ❌ 同步写入 |

#### ZeusPagedTokenToKVPoolAllocator

**为什么不能复用原版：** 原版 `PagedTokenToKVPoolAllocator` 的 alloc/free 逻辑使用 `torch.cat`、`torch.sort`、`torch.unique`、`torch.cumsum` 等大量 ATen 算子，在 Zeus 上每次都会触发隐式 CPU fallback（6+ 次/step）。

**解决方案：** 新 allocator 将所有记账状态保持在 CPU tensor 上（`free_pages`、`release_pages` 始终在 CPU），中间计算全在 CPU 完成，仅最终的 `out_indices` 做一次 `.to(zeus_device)`。

```
原版 (Zeus 上): 每次 alloc/free → 6+ 次隐式 D2H/H2D
新版:           每次 alloc/free → 1 次显式 H2D (仅 output)
```

---

### 3.5 模型权重加载与 LocalMem 打包

**核心原理：** Zeus GEMM 高性能路径要求权重存在 **LocalMem**（片上近存，带宽远高于 GDG 全局显存），且布局必须是 **(K, N) 转置格式**。

**涉及文件：**
- `python/sglang/srt/model_loader/loader.py` — 权重加载后 pack 到 LocalMem (+107 行)
- `python/sglang/srt/layers/quantization/unquant.py` — GEMM dispatch 适配
- `python/sglang/srt/layers/logits_processor.py` — LM Head 矩阵乘法适配

**加载后的权重处理流程：**
```
load_model()
  ├── model = get_model_cls()(config)           # 构建模型结构
  ├── model.load_weights(weights)                # 加载权重到 Zeus GDG
  │     └── _tracking_iter(weights)              # 追踪 lm_head.weight 是否在 checkpoint 中
  ├── _zeus_init_lm_head_from_embed()            # tied 模型: embed→lm_head 权重拷贝
  └── pack_weights()                             # ★ 转为 LocalMem 格式
        ├── LinearBase: weight (N,K) → 转置 → LocalMem (K,N)
        ├── ParallelLMHead: 同上
        └── tie=False 时:
              embed_tokens → 保持 GDG (embedding lookup 需要行索引)
              lm_head     → LocalMem   (GEMM 需要高带宽)
```

**关键约束：**
- `F.linear(x, weight)` 内部做 `x @ weight.T`，但 LocalMem 权重已是 (K,N)，再 `.T` 会破坏布局
- **解法：** 用 `torch.mm(x, weight)` 替代 `F.linear`，绕过内部转置

**Tied Embedding 处理：**

| 场景 | embed_tokens | lm_head | 策略 |
|------|-------------|---------|------|
| tie=True | LocalMem (被迫) | LocalMem (共享) | embedding lookup 走 `sgl_kernel_zeus.embedding` (列 gather) |
| tie=False | GDG (最优) | LocalMem (最优) | 各自使用最优存储 |

---

### 3.6 ATen 缺失算子的 CPU Bounce 规避

**核心原理：** Zeus NPU 的 ATen 算子覆盖率有限。当 PyTorch 在 Zeus 设备上调用未注册算子时，会自动触发 CPU fallback（D2H→CPU 计算→H2D），这是**同步操作，会阻塞 Zeus 计算流水线**。

**策略：** 用显式 `.cpu()` → 计算 → `.to(device)` 替代隐式 fallback，将多次隐式拷贝合并为一次显式来回。

> **📢 2026-04 更新：** torch_zeus 已新增 `aten::cat`/`cat.out`（TensorOps.cpp）、`aten::add`/`sub`/`mul`（dispatch stub）、`aten::sum`/`mean`/`norm`（reduce dispatch stub）的原生实现。这意味着下表中标注 🆕 的项目对应的 CPU bounce **可以安全移除**。

**三种主要的 CPU Bounce 模式：**
```python
# 模式 A: 一元/二元运算 bounce  → aten::add/sub/mul 已原生，大部分可移除 🆕
result = (tensor.cpu() OP value).to(device)
# 示例: seq_lens += 1, aten::neg, aten::clamp

# 模式 B: 索引 bounce  → aten::index.Tensor_out 仍未注册，需保留
result = tensor.cpu()[index.cpu()].to(device)
# 示例: filter_batch, alloc_for_decode

# 模式 C: cat bounce  → aten::cat 已原生，可移除 🆕
result = torch.cat([a.cpu(), b.cpu()]).to(device)
# 示例: radix_cache, memory_pool
```

**涉及文件（共 10+ 个，按影响频率排序）：**

| 文件 | 改动点 | 频率 | 当前状态 |
|------|--------|------|----------|
| `memory_pool.py` — `write()` | ✅ 已实现 Zeus 原生 `aten::index_put_`，无需 CPU bounce | **每个 decode step** | ✅ 已解决 |
| `schedule_batch.py` — `merge_batch` | torch.cat 拼接 4 个 tensor | **batch merge 时** | 🆕 cat 已原生，可移除 |
| `schedule_batch.py` — `update_batch` | seq_lens += 1 标量加法 | **每个 decode step** | 🆕 add 已原生，可移除 |
| `schedule_batch.py` — `filter_batch` | `tensor[int64_index]` 花式索引 | **请求结束时** | ❌ index 未注册，需保留 |
| `common.py` — `alloc_for_decode` | 二维花式索引 → 仅拷贝所需行 | **每个 decode step** | ❌ index 未注册，需保留 |
| `forward_batch_info.py` | pad/cumsum/clamp/arange | **每次 forward** | 混合：pad 可移除 🆕，cumsum/arange 需保留 |
| `sampler.py` — `argmax` | logits 搬到 CPU 做 argmax | **每个 greedy decode** | ❌ argmax 未注册，需保留 |
| `logits_processor.py` | 花式索引 → 逐条 copy_ | **每次 prefill** | ❌ index 未注册，需保留 |
| `radix_cache.py` | `_cat_zeus()` 封装 | 请求结束时 | 🆕 cat 已原生，可移除 |
| `memory_pool.py` — `free/data_ptrs` | torch.cat 拼接 | 初始化/释放时 | 🆕 cat 已原生，可移除 |
| `overlap_utils.py` | where/clamp/arange | overlap 调度时 | ❌ where/clamp/arange 未注册 |
| `scheduler.py` | neg(int tensor) | 每步 | ❌ neg 未注册 |

---

## 4. Zeus 自定义算子清单

所有 Zeus 自定义算子来自 `sgl_kernel_zeus` 包：

| 算子 | 功能 | 调用位置 |
|------|------|----------|
| `extend_attention` | Prefill 阶段注意力计算 | `zeus_backend.py:forward_extend` |
| `decode_attention` | Decode 阶段注意力计算 | `zeus_backend.py:forward_decode` |
| `store_kv_cache` | 写入 KV Cache (tiled layout) | `zeus_memory_pool.py:set_kv_buffer` |
| `rotary_embedding` | RoPE 位置编码 | `rotary_embedding.py:forward_zeus` |
| `rmsnorm` | RMS 归一化 | `layernorm.py:forward_zeus` |
| `fused_add_rmsnorm` | 残差加 + RMS 归一化 (原地) | `layernorm.py:forward_zeus` |
| `silu_and_mul` | SiLU 激活 + 门控乘法 | `activation.py:forward_zeus` |
| `embedding` | Embedding 查表 (GDG 列方向 gather) | `vocab_parallel_embedding.py` |
| `sampling_from_logits` | 融合采样: temp/softmax/top-k/top-p/sample | `sampler.py` |
| `top_k_renorm_prob` | top-k 后重归一化 | `sampler.py` |
| `top_p_renorm_prob` | top-p 后重归一化 | `sampler.py` |
| `top_k_top_p_sampling_from_probs` | top-k + top-p 采样 | `sampler.py` |
| `min_p_sampling_from_probs` | min-p 采样 | `sampler.py` |

**共计 14 个算子**，覆盖了 Dense Transformer 推理的全部关键路径。

---

## 5. CUDA vs Zeus 全流程对比

### 5.1 初始化阶段

| 步骤 | CUDA | Zeus | 差异 |
|------|------|------|------|
| cuBLAS 初始化 | `init_cublas()` | ❌ 不需要 | Zeus 用自有 GEMM 引擎 (ZENL) |
| Attention 后端 | FlashInfer (GPU workspace 128~2048MB, 预分配 buffer) | ZeusAttnBackend (极简, 无预分配) | Zeus 每次 forward 需重建 indices |
| Kernel 热身 | FlashInfer autotune (选最优 tile 配置) | ❌ 无 | Zeus 缺少 kernel tuning |
| CUDA Graph | CudaGraphRunner 捕获 ~20 个不同 bs 的 graph | ❌ `graph_runner=None` | **Zeus 每步重新 launch 所有 kernel** |

### 5.2 单次 Decode Step 对比

**CUDA 路径（零 CPU 传输）：**
```
GPU:  alloc_for_decode → init_metadata → CUDA Graph replay → argmax
      (GPU index_put)    (GPU Triton)    (完整 forward)       (GPU)
CPU:  (空闲，无数据传输)
```

**Zeus 路径（多次 CPU↔Zeus 传输）：**
```
Zeus:   ═══╗           ╔══════════════════════════╗           ╔═══
            ║           ║  model.forward() 纯 Zeus  ║           ║
            ↓T1-T3     ↑T4                         ↓T11     ↑T12
CPU:  alloc_for_decode  init_forward_metadata     argmax    next
      ⚠️ 多次 bounce    ⚠️ 整矩阵传输             (greedy)   step
```

### 5.3 关键模块性能差距

| 模块 | CUDA | Zeus | 差距根因 |
|------|------|------|----------|
| Attention metadata | GPU Triton kernel 构建 | CPU for 循环 + 整矩阵拷贝 (**64 MB/step**) | Zeus 缺 cumsum/index ATen 支持 |
| req_to_token write | GPU 1 行 `index_put_` | ✅ Zeus 原生 `aten::index_put_` (零传输) | ~~Zeus 缺 `index_put_`~~ 已解决 |
| tensor 拼接/算术 | GPU 原生 cat/add/sub/mul | ✅ Zeus 原生 (dispatch stub) | ~~Zeus 缺 cat/add~~ 已解决 🆕 |
| Decode forward | 1 次 CUDA Graph replay (~1μs) | ~200 次 kernel launch | Zeus 无 Graph capture |
| Greedy argmax | GPU 原生 | CPU round-trip (**9.3 MB** logits 搬回) | Zeus 缺 `argmax` |

### 5.4 量化估算

```
假设: bs=32, max_reqs=128, max_ctx_len=131072, vocab=151936, 28 层模型, bf16

CUDA Decode Step:  ~18μs  (0 次 CPU 传输)
Zeus Decode Step:  ~25.5ms (7+ 次 CPU 传输, ~89 MB 数据搬运)

传输量拆解 (bs=32):
  P0  init_forward_metadata req_to_token 整矩阵   64.0 MB  (72%)
  P1  alloc_for_decode      req_to_token bs 行     16.0 MB  (18%)
  P2  greedy argmax         logits                  9.3 MB  (10%)
  --  metadata 小量          seq_lens/rpi/indptr     ~130 KB (<1%)
  合计:                                            ~89.4 MB / step

注: index_put_ 已解决后传输量已从 ~200MB 降至 ~89MB，
    cat/add/sub/mul 已原生后约 ~10KB 的 bounce 也可清除。
    下一步实现 Zeus 端 kv_indices 构建可再消除 ~80MB。

差距: ~1400× (时间), ~89MB vs 0 (数据传输)
```

---

## 6. 已修改文件完整清单

### 6.1 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `layers/attention/zeus_backend.py` | +194 | Zeus Attention 后端 |
| `mem_cache/zeus_allocator.py` | +212 | Zeus 分页 KV Cache 分配器 |
| `mem_cache/zeus_memory_pool.py` | +122 | Zeus KV Cache 内存池 |
| `docs/zeus_gap_analysis.md` | +175 | Gap 分析文档 |
| `zeus_dev/demo_cuda_llm.py` | +143 | CUDA 对照推理脚本 |
| `zeus_dev/demo_zeus_llm.py` | +129 | Zeus 推理 demo |
| `zeus_dev/demo_zeus_layer_compare.py` | +1579 | 逐层数值对比工具 |
| `zeus_dev/zeus_adaptation_dev_doc.md` | +68 | 适配开发文档 |
| `zeus_dev/zeus_adaptation_report.md` | +181 | 适配进度报告 |

### 6.2 修改文件

| 文件 | 变更 | 核心作用 |
|------|------|----------|
| `configs/device_config.py` | +1/−1 | 设备白名单加 "zeus" |
| `custom_op.py` | +9/−1 | `forward_zeus` 派发 |
| `layers/activation.py` | +5 | SiluAndMul Zeus kernel |
| `layers/attention/attention_registry.py` | +7 | 注册 zeus 后端 |
| `layers/layernorm.py` | +13 | RMSNorm Zeus kernel |
| `layers/logits_processor.py` | +50/−24 | LM Head GEMM + 花式索引规避 |
| `layers/quantization/unquant.py` | +16 | LocalMem GEMM dispatch |
| `layers/rotary_embedding.py` | +32/−7 | RoPE Zeus kernel |
| `layers/sampler.py` | +97/−32 | 融合采样 + greedy CPU bounce |
| `layers/vocab_parallel_embedding.py` | +14 | Zeus embedding (列 gather) |
| `managers/overlap_utils.py` | +15 | where/clamp/arange 规避 |
| `managers/schedule_batch.py` | +48/−9 | seq_lens/filter/merge CPU bounce |
| `managers/scheduler.py` | +9 | neg() CPU bounce |
| `mem_cache/common.py` | +55 | alloc_for_decode Zeus 优化 |
| `mem_cache/memory_pool.py` | +32/−5 | write() + data_ptrs CPU bounce |
| `mem_cache/radix_cache.py` | +16/−6 | _cat_zeus() 封装 |
| `model_executor/forward_batch_info.py` | +48/−14 | pad/cumsum/clamp 等 CPU bounce |
| `model_executor/model_runner.py` | +38/−5 | Zeus KV Pool / allocator / zecl |
| `model_loader/loader.py` | +107 | 权重 LocalMem 打包 |
| `server_args.py` | +25 | Zeus 默认参数 |
| `utils/common.py` | +66 | is_zeus() 及设备工具函数 |

---

## 7. 已知性能瓶颈

> **📢 2026-04 更新：** torch_zeus 新增了 `aten::cat`、`aten::add/sub/mul`、`aten::sum/mean/norm` 的原生实现。标注 🆕 的 bounce 可通过移除 `_is_zeus` 分支直接消除。

### 7.1 CPU↔Zeus 传输热点汇总（含精确数据量分析）

以下估算基于典型配置：**bs=32, max_reqs=128, max_ctx_len=131072, vocab=151936, 28 层, bf16**

| # | 位置 | 方向 | 数据 | 大小估算 | 频率 | 严重度 |
|---|------|------|------|----------|------|--------|
| **T1** | `zeus_backend.py:init_forward_metadata` L48 | Zeus→CPU | **req_to_token 整个矩阵** | **64 MB** (128×131072×4B) | 每 decode step | 🔴 **致命** |
| **T2** | `zeus_backend.py:init_forward_metadata` | Zeus→CPU | seq_lens + req_pool_indices | 256 B (32×4B×2) | 每 decode step | 🟢 忽略 |
| **T3** | `zeus_backend.py:init_forward_metadata` | CPU→Zeus | kv_indptr + kv_indices | ~128 KB (33×4B + total_kv×4B) | 每 decode step | 🟡 可接受 |
| **T4** | `common.py:alloc_for_decode` L53-58 | Zeus→CPU | req_to_token (仅 bs 行) | **16 MB** (32×131072×4B) | 每 decode step | 🔴 **严重** |
| **T5** | `common.py:alloc_for_decode` | Zeus→CPU | rpi + seq_lens | 256 B | 每 decode step | 🟢 忽略 |
| **T6** | `sampler.py` greedy argmax L104-105 | Zeus→CPU | logits tensor | **9.3 MB** (32×151936×2B) | 每 greedy step | 🟡 中等 |
| **T7** | `sampler.py` greedy result | CPU→Zeus | argmax 结果 | 256 B (32×8B) | 每 greedy step | 🟢 忽略 |
| **T8** 🆕 | `schedule_batch.py:merge_batch` L1890-1916 | Zeus→CPU→Zeus | req_pool_indices/seq_lens/orig_seq_lens/output_ids ×2 | ~1.6 KB (4×bs×4B×2) | batch merge 时 | 🟢 可移除 |
| **T9** 🆕 | `schedule_batch.py:update_batch` L1775-1779 | Zeus→CPU→Zeus | seq_lens +1, orig_seq_lens +1 | ~512 B (2×bs×4B×2) | 每 decode step | 🟢 可移除 |
| **T10** | `schedule_batch.py:filter_batch` L1840-1845 | Zeus→CPU→Zeus | 4 个 tensor 花式索引 | ~1.5 KB | 请求结束时 | 🟢 小 |
| **T11** 🆕 | `radix_cache.py:_cat_zeus` L42-48 | Zeus→CPU→Zeus | 变长 index tensor list | ~几 KB | 请求结束时 | 🟢 可移除 |
| **T12** 🆕 | `memory_pool.py:MambaPool.free` L276-279 | Zeus→CPU→Zeus | free_slots cat | ~几 KB | 释放时 | 🟢 可移除 |
| **T13** | `forward_batch_info.py:compute_position` L1246-1257 | Zeus→CPU→Zeus | arange+cumsum+cat 混合 | ~bs×seq_len×8B | 每次 extend | 🟡 中等 |
| **T14** | `forward_batch_info.py:cumsum` L1081, L1118 | Zeus→CPU→Zeus | seq_lens cumsum | ~bs×4B | 每次 forward | 🟢 小 |
| **T15** 🆕 | `forward_batch_info.py:_pad_tensor` L762-772 | Zeus→CPU→Zeus | padding cat | ~bs×4B | 每次 forward | 🟢 可移除 |

### 7.2 严重问题分级（按数据量排序）

#### 🔴 P0: init_forward_metadata() — 整矩阵拷贝 (64 MB/step)

**位置:** `zeus_backend.py:48` — `req_to_token.cpu()`

**问题：** 每个 decode step 无条件拷贝**整个** req_to_token 矩阵到 CPU 用于构建 kv_indices。
- 矩阵尺寸：`max_reqs × max_ctx_len = 128 × 131072 = 16M` 个 int32
- 数据量：**64 MB / step**
- 100 token 生成 = **6.4 GB** 无效传输

**这是当前最大的性能瓶颈**，远超其他热点之和。

**解法方案：**
```python
# 方案 A（快速修复）：只拷贝需要的 bs 行
rpi_cpu = req_pool_indices.cpu()  # 已有
req_to_token_needed = req_to_token[rpi_cpu].cpu()  # bs 行 vs 全矩阵
# → 64MB 降至 16MB (bs=32)，但仍需 aten::index 支持

# 方案 B（最优）：在 Zeus 端直接构建 kv_indices
kv_indices = sgl_kernel_zeus.create_kv_indices(
    req_to_token, req_pool_indices, seq_lens, kv_indptr
)
# → 零 CPU 传输
```

#### 🔴 P1: alloc_for_decode() — bs 行拷贝 (16 MB/step)

**位置:** `common.py:53-58` — `req_to_token[rpi_cpu].cpu()`

**问题：** 已优化为只拷贝 bs 行，但 bs=32 时仍拷贝 `32 × 131072 × 4B = 16 MB / step`。

**解法：** 与 P0 共用方案 B（Zeus 端构建），或改写 `last_loc` 计算逻辑。

#### 🟡 P2: Greedy argmax 在 CPU 执行 (9.3 MB/step)

**位置:** `sampler.py:104-105`

`torch.argmax(logits.cpu(), -1).to(device)` 把完整 logits 搬到 CPU。
- 数据量：`bs × vocab × 2B = 32 × 151936 × 2 ≈ 9.3 MB / step`

**解法方案:**
```python
# 方案 A：实现 Zeus argmax 算子
batch_next_token_ids = sgl_kernel_zeus.argmax(logits, dim=-1)

# 方案 B：复用 Zeus sampling kernel (极小 temperature 模拟 greedy)
```

#### ✅ P3: ReqToTokenPool.write() — 已解决

**位置:** `memory_pool.py:write()`

**原问题:** 每个 decode step 把整个 `req_to_token` 矩阵（64 MB）从 Zeus 拷到 CPU，写入几个元素后整块拷回。

**解决方案:** 在 `torch_zeus` 中注册了 `aten::index_put_` 和 `aten::index_put` 的 Zeus PrivateUse1 原生实现。

**实现细节:**
- ATen 注册层：`torch_zeus/csrc/aten/operators/zenl/index_put.cpp`
  - 多维索引线性化为 flat offset (`idx0 * stride0 + idx1 * stride1 + ...`)
  - 自动处理 dtype 转换和 contiguous 保证
- ZENL kernel 层：`zenl/src/host/index_put.cpp` → `zenl/src/triton/index_put.py`
  - Triton scatter-write kernel，≥1024 indices 自动双核并行

**收益:** 消除每 step ~128 MB (D2H+H2D) 传输。

#### 🆕 可立即移除的 CPU bounce（aten::cat/add/sub/mul 已原生）

以下 bounce 因 torch_zeus 新增算子而**不再需要**：

| 位置 | 操作 | 对应新算子 | 估计节省 |
|------|------|-----------|---------|
| `radix_cache.py` L416, L543, L631 — `_cat_zeus()` | 变长 list cat | `aten::cat` | ~几 KB/次 |
| `schedule_batch.py:merge_batch` L1891-1914 | 2-tensor cat ×4 | `aten::cat` | ~1.6 KB/次 |
| `schedule_batch.py:update_batch` L1776-1779 | seq_lens + 1 | `aten::add` | ~512 B/step |
| `memory_pool.py:MambaPool.free` L277-279 | 2-tensor cat | `aten::cat` | ~几 KB/次 |
| `memory_pool.py:data_ptrs` L653-655 | uint64 tensor cat | `aten::cat` | ~几 KB (初始化) |
| `forward_batch_info.py:_pad_tensor` L762-772 | padding cat | `aten::cat` | ~bs×4B/次 |

**移除方法：** 删除对应的 `if _is_zeus:` 分支，让 Zeus 走与 CUDA 相同的代码路径。

> **注意：** `forward_batch_info.py:compute_position_torch` (L1246-1257) 虽含 cat，但还混合了 `torch.arange`（标量参数需从 Zeus tensor 取值）和 `torch.cumsum`，这两个算子仍未注册，因此**该处 bounce 暂不能移除**。

### 7.3 CPU bounce 数据量汇总

```
每次 Decode Step 的 CPU↔Zeus 数据传输估算 (bs=32, max_ctx=131072, vocab=151936):

┌─ 不可避免的传输 ──────────────────────────────────────────────────┐
│  P0  init_forward_metadata  req_to_token 整矩阵   64.0 MB  🔴   │
│  P1  alloc_for_decode       req_to_token bs 行     16.0 MB  🔴   │
│  P2  greedy argmax          logits                  9.3 MB  🟡   │
│  --  metadata 小量传输      seq_lens/rpi/indptr     ~130 KB  🟢   │
├──────────────────────────────────────────────────────────────────┤
│  合计每 step                                       ~89.4 MB      │
│  生成 100 token                                    ~8.9 GB       │
└───────────────────────────────────────────────────────────────────┘

┌─ 可通过移除 _is_zeus 分支消除的传输 (aten::cat/add/sub/mul 已原生) ┐
│  merge_batch cat ×4          1.6 KB                               │
│  update_batch add ×2         512 B                                │
│  _cat_zeus ×3                几 KB                                │
│  _pad_tensor cat             几 KB                                │
│  memory_pool cat ×2          几 KB                                │
├───────────────────────────────────────────────────────────────────┤
│  合计                        ~10 KB/step (占比 <0.01%)            │
└───────────────────────────────────────────────────────────────────┘

结论：
  1. P0 (64 MB) 独占 72% 传输量 → 最高优先级
  2. P1 (16 MB) 占 18% → 可与 P0 共同解决 (Zeus 端 kv_indices 构建)
  3. P2 (9.3 MB) 占 10% → 实现 Zeus argmax 可消除
  4. 已可移除的 cat/add bounce 数据量极小，但消除后简化代码路径
```

---

## 8. 后续开发路线图

### 8.1 总体路线：四大阶段

后续 Zeus 适配 SGLang 的大方向按以下顺序推进：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  阶段一          │     │  阶段二          │     │  阶段三          │     │  阶段四          │
│  Graph 接入     │────→│  分布式接入      │────→│  PD 分离接入     │────→│  理清算子流程    │
│                 │     │                 │     │                 │     │                 │
│ 计算图捕获/重放 │     │ 多卡 TP/PP      │     │ Prefill-Decode  │     │ 消除 CPU bounce │
│ decode 吞吐翻倍 │     │ ZECL 集合通信   │     │ 分离式推理      │     │ 扩充 ATen 算子  │
│                 │     │ 大模型支撑      │     │ KV 跨节点传输   │     │ 模型覆盖扩展    │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
  ★ 单卡性能基础          ★ 规模化基础            ★ 生产部署架构          ★ 持续打磨完善
```

---

### 8.2 阶段一：Graph 接入（计算图捕获/重放）

**目标：** 让 Zeus 支持类似 CUDA Graph 的计算图捕获与重放，消除 decode 阶段的 kernel launch 开销。

**背景：** 当前 Zeus 硬编码 `disable_cuda_graph=True`，每次 decode step 需要逐个 launch ~200 个 kernel（28 层 × ~7 kernel/层），Python 调度开销约 2ms/step。CUDA 路径通过 `CudaGraphRunner` 一次 `graph.replay()` 完成全部 forward，开销约 1μs。

#### CUDA Graph 在 SGLang 中的工作机制

SGLang 的 `CudaGraphRunner`（`model_executor/cuda_graph_runner.py`）工作流程：

```
初始化阶段：
  ├── 枚举 batch sizes [1, 2, 4, 8, 16, 32, ...]
  ├── 对每个 bs:
  │     ├── 构造 ForwardBatch（预分配 GPU buffer）
  │     ├── attn_backend.init_cuda_graph_state(bs, num_tokens)  ← 预分配 attention metadata
  │     ├── attn_backend.init_forward_metadata_capture_cuda_graph(...)  ← 准备捕获
  │     ├── 运行 2 次 warmup（确保所有算子路径稳定）
  │     └── torch.cuda.CUDAGraph.capture(run_once)              ← 捕获完整 forward
  └── 保存 self.graphs[bs] = graph, self.output_buffers[bs] = output

推理阶段：
  ├── bisect 找到匹配的 bs
  ├── replay_prepare(): 填充输入 buffer（copy_ 到预分配区域）
  ├── attn_backend.init_forward_metadata_replay_cuda_graph(...)  ← 增量更新 metadata
  └── self.graphs[bs].replay()  ← 一次调用，零 Python 开销
```

#### Zeus 侧需要实现的内容

| # | 任务 | 说明 | 依赖 |
|---|------|------|------|
| G1 | `torch_zeus` 实现 Graph Capture/Replay API | Zeus 的 `zertGraphLaunch` 需要能工作，或实现等效机制。需要提供 `torch.zeus.ZeusGraph()` 类，支持 `capture()` 上下文管理器和 `replay()` 方法 | torch_zeus 底层 |
| G2 | `ZeusAttnBackend` 实现 `init_cuda_graph_state()` | 预分配 attention metadata buffer（kv_indptr, kv_indices 等），使其在 graph capture 时是固定地址 | G1 |
| G3 | `ZeusAttnBackend` 实现 `init_forward_metadata_capture_cuda_graph()` | 在 capture 阶段构建 attention metadata | G2 |
| G4 | `ZeusAttnBackend` 实现 `init_forward_metadata_replay_cuda_graph()` | 在 replay 阶段增量更新 metadata（仅更新 kv_indptr/kv_indices 内容，不重新分配） | G2 |
| G5 | 创建 `ZeusGraphRunner` 类 | 参考 `CudaGraphRunner`，处理 Zeus 特有的 capture/replay 逻辑。注册到 `model_runner.py` 的 `graph_runners` 字典 | G1-G4 |
| G6 | `server_args.py` 改为条件式 `disable_cuda_graph` | 当 Zeus graph API 就绪后，不再强制禁用 | G5 |

#### 关键约束与注意事项

- **Graph 内不能有 CPU↔Zeus 传输**：当前 `init_forward_metadata` 中的 CPU bounce（T1-T4）必须在 graph 外完成，或在 graph 前用 Zeus 端算子替代
- **Graph 内不能有动态 shape**：所有 tensor 的 shape 必须在 capture 时固定，所以需要 padding 到 capture_bs
- **Graph 内不能有 Python 控制流**：所有 `if _is_zeus:` 分支必须在 trace 时确定
- **ZECL 通信与 Graph 的兼容性**：需要验证 ZECL allreduce 操作是否能在 graph 中 capture

#### 预期收益

- Decode 吞吐量提升 **2×~5×**（消除 ~200 次 kernel launch 的 Python 调度开销）
- 为后续分布式和 PD 分离打下性能基础

---

### 8.3 阶段二：分布式接入（多卡 TP/PP）

**目标：** 让 Zeus 支持多卡推理（Tensor Parallel / Pipeline Parallel），支撑大模型部署。

**背景：** SGLang 的分布式通信通过 `GroupCoordinator`（`distributed/parallel_state.py`）管理，支持多种通信后端。Zeus 当前使用 `"zecl"` 作为 `torch.distributed` 后端，单卡推理已验证，但多卡路径尚未完整测试。

#### SGLang 分布式架构

```
分布式初始化：
  ├── init_torch_distributed(backend="zecl")       ← Zeus 集合通信后端
  ├── initialize_model_parallel(tp_size, ep_size, pp_size)
  │     ├── init_model_parallel_group(tp_ranks, backend)  ← Tensor Parallel 组
  │     ├── init_model_parallel_group(pp_ranks, backend)  ← Pipeline Parallel 组
  │     └── init_model_parallel_group(ep_ranks, backend)  ← Expert Parallel 组 (MoE)
  └── GroupCoordinator(group_ranks, torch_distributed_backend, ...)
        ├── pynccl_comm     ← Zeus 上不可用，需 ZECL 替代
        ├── ca_comm         ← CustomAllReduce (IPC/NVLink)，Zeus 上不可用
        └── device_group    ← torch.distributed ProcessGroup (ZECL 后端)

推理时通信：
  ├── all_reduce(tensor)    ← TP 中的 QKV/FFN 结果同步
  ├── all_gather(tensor)    ← TP 中的 Vocab Parallel 结果收集
  └── send/recv(tensor)     ← PP 中的层间数据传输
```

#### Zeus 侧需要实现的内容

| # | 任务 | 说明 | 依赖 |
|---|------|------|------|
| D1 | 验证 ZECL 多卡 `all_reduce` 正确性 | 使用 TP=2 跑 dense 模型，验证 `all_reduce` 数值正确。ZECL 需要支持 bf16/fp16 的 SUM reduce | ZECL 基础功能 |
| D2 | 验证 ZECL `all_gather` / `reduce_scatter` | TP 中 `VocabParallelEmbedding` 和 `RowParallelLinear` 需要这些操作 | D1 |
| D3 | `GroupCoordinator` Zeus 适配 | 当前 `all_reduce` 路径中 PyNccl / CustomAllReduce / TorchSymmMem 等通信器在 Zeus 上均不可用。需确保走 `torch.distributed.all_reduce` (ZECL backend) 的 fallback 路径。可能需要新增 `use_zeus_communicator` 分支或禁用 CUDA-only 通信器 | D1 |
| D4 | Pipeline Parallel 支持 | PP 需要 `send()`/`recv()` 点对点通信。验证 ZECL 的 P2P 支持 | D1 |
| D5 | Graph + 分布式联调 | Graph capture 中的 `all_reduce` 需要特殊处理。CUDA 路径中 PyNccl 在 graph 中使用 `change_state(enable=True)`。Zeus 需要验证 ZECL 在 graph 中的行为 | 阶段一 + D1-D3 |
| D6 | 多节点通信测试 | 跨节点 ZECL 通信延迟和带宽验证 | D1-D4 |

#### 关键注意事项

- **通信器选择**：SGLang 的 `GroupCoordinator.all_reduce()` 有复杂的通信器选择逻辑（CustomAllReduce → PyNccl → torch.distributed）。Zeus 需要确保最终 fallback 到 `torch.distributed` (ZECL) 时能正常工作
- **IPC 相关**：`CustomAllreduce` 依赖 CUDA IPC (`cudaIpcGetMemHandle`)，Zeus 上不可用，确保被正确跳过
- **Device Mesh**：`torch.distributed.init_device_mesh("zeus", (tp_size,))` 需要 Zeus 作为合法 device 类型

#### 预期收益

- 支持超出单卡显存的大模型（如 7B+ 模型 TP=2/4）
- 为 PD 分离架构提供通信基础

---

### 8.4 阶段三：PD 分离接入（Prefill-Decode Disaggregation）

**目标：** 让 Zeus 支持 Prefill-Decode 分离式推理，P 节点专做 prefill，D 节点专做 decode，通过 KV Cache 传输实现协作。

**背景：** SGLang 已有完整的 PD 分离架构（`sglang/srt/disaggregation/`），支持 Mooncake 和 NIXL 两种传输引擎。Ascend NPU 也有参考实现 `AscendTransferEngine`。

#### PD 分离架构

```
Prefill 节点：                              Decode 节点：
┌──────────────────────────┐              ┌──────────────────────────┐
│ 1. Bootstrap Queue       │              │ 1. PreallocQueue         │
│    ├── 初始化 KVSender    │              │    ├── 初始化 KVReceiver  │
│    └── 握手 + 预分配      │   ←握手→    │    └── 预分配 KV 空间     │
│                          │              │                          │
│ 2. Waiting Queue         │              │ 2. TransferQueue         │
│    └── 执行 Prefill       │              │    └── Poll 等待传输完成  │
│                          │              │                          │
│ 3. Inflight Queue        │              │ 3. WaitingQueue          │
│    ├── KVSender.send()   │   →KV传输→  │    └── 构建 PrebuiltExtend│
│    │   发送 KV Cache      │              │       跳过 prefill forward│
│    └── Poll 等待完成      │              │                          │
│                          │              │ 4. RunningBatch          │
│                          │              │    └── 直接开始 Decode     │
└──────────────────────────┘              └──────────────────────────┘

核心组件：
  ├── BaseKVManager / BaseKVSender / BaseKVReceiver   ← 抽象接口
  ├── CommonKVManager / CommonKVSender / CommonKVReceiver  ← 通用逻辑
  ├── MooncakeTransferEngine  ← RDMA 传输引擎
  ├── NixlKVManager           ← NIXL 传输引擎
  ├── AscendTransferEngine    ← Ascend NPU 参考 (可借鉴)
  └── transfer_kv_*()         ← KV Cache 跨层/跨节点拷贝 kernel
```

#### Zeus 侧需要实现的内容

| # | 任务 | 说明 | 依赖 |
|---|------|------|------|
| PD1 | 设计 Zeus KV Cache 传输机制 | 评估可选方案：① 参考 `AscendTransferEngine` 实现 `ZeusTransferEngine`；② 基于 RDMA/共享内存的自定义传输；③ 通过 CPU 中转（简单但慢） | 阶段二 |
| PD2 | 实现 `ZeusTransferEngine` | 继承 `MooncakeTransferEngine` 或独立实现，封装 Zeus 设备间的 KV 数据传输。需要支持 `register_memory()` / `send()` / `poll()` 接口 | PD1 |
| PD3 | 适配 `transfer_kv_*` kernel | 当前 `sgl_kernel.transfer_kv_per_layer` 等系列 kernel 是 CUDA 实现。Zeus 的 tiled KV 布局 `[pages, heads, page_size, head_dim]` 与 CUDA 的 flat 布局不同，需要适配或重新实现 | PD2 |
| PD4 | KV 布局转换 kernel | 如果 P 节点和 D 节点使用不同 page_size 或不同 KV 布局，需要实现 `transfer_kv_per_layer_pf_lf`（paged→flat）和 `transfer_kv_per_layer_ph_lf`（paged→flat with head interleave）等变体 | PD3 |
| PD5 | Prefill 节点集成测试 | 在 Zeus 上运行 `DisaggregationMode.PREFILL` 模式，验证 prefill → KV 发送链路 | PD2-PD4 |
| PD6 | Decode 节点集成测试 | 在 Zeus 上运行 `DisaggregationMode.DECODE` 模式，验证 KV 接收 → decode 链路。关键点：`prepare_for_prebuilt()` 中的 `req_to_token` 操作需适配 Zeus | PD5 |
| PD7 | 端到端联调 | P 节点 + D 节点联调，验证完整 PD 分离推理链路 | PD5 + PD6 |

#### 关键注意事项

- **KV 布局兼容性**：Zeus 使用 4D tiled 布局，CUDA 使用 flat 布局。跨设备 PD 分离（如 P 在 CUDA，D 在 Zeus）需要布局转换
- **V Cache 特殊格式**：Zeus 的 V cache 使用 16-byte column-group interleaved 格式，传输前后需要确保格式一致
- **内存注册**：Mooncake/NIXL 引擎需要注册 GPU 内存区域用于 RDMA。Zeus 的 `torch.zeus` 内存管理接口需要支持 `get_data_ptr()` 等
- **参考 Ascend**：`AscendTransferEngine` 是最直接的 NPU PD 分离参考实现，可借鉴其 `mf_adapter` 封装思路

#### 预期收益

- P/D 分离后 decode 延迟更稳定（不受 prefill 干扰）
- 支持更灵活的资源分配（P 节点少 + D 节点多的异构部署）

---

### 8.5 阶段四：继续理清算子流程

**目标：** 系统性消除残留的 CPU bounce，扩充 ATen 算子覆盖率，扩大模型覆盖，打磨生产级性能。

此阶段贯穿前三个阶段持续推进，但作为独立方向需要系统化梳理。

#### 8.5.1 消除关键 CPU↔Zeus 数据传输

| 优先级 | 任务 | 类型 | 对应问题 | 预期收益 |
|--------|------|------|----------|----------|
| ✅ 完成 | ~~实现 Zeus 原生 `aten::index_put_`~~ | kernel 开发 | ~~P3 write 整矩阵~~ | **已实现** (消除 ~128MB/step) |
| ✅ 完成 | ~~实现 Zeus 原生 `aten::cat`~~ | kernel 开发 | ~~cat bounce ×10~~ | **已实现** (TensorOps.cpp) |
| ✅ 完成 | ~~实现 Zeus 原生 `aten::add/sub/mul`~~ | dispatch stub | ~~arithmetic bounce~~ | **已实现** (dispatch stub) |
| 🔴 P0 | `init_forward_metadata` 改为按行拷贝或实现 Zeus 端 kv_indices 构建 | sglang + kernel | P0 (整矩阵 64MB/step) | 消除最大热点 |
| 🔴 P0 | 实现 Zeus argmax 或复用 sampling kernel | kernel 开发 | P2 (logits 9.3MB/step) | 消除 greedy decode 瓶颈 |
| 🟡 P1 | 实现 `aten::index.Tensor_out` | kernel 开发 | filter/alloc 花式索引 | 消除 ~4 处 bounce |
| 🟡 P1 | 实现 `aten::cumsum` | kernel 开发 | kv_indptr 等 5+ 处 bounce | 解锁 Zeus 端 metadata 构建 |
| 🟢 P2 | 清理 SGLang 中已失效的 `_is_zeus` cat/add bounce 分支 | sglang 清理 | 代码简化 | ~10 处分支可删除 |

#### 8.5.2 扩充 ATen 高频算子

消除 CPU round-trip，推动 torch_zeus 原生实现：

| 优先级 | ATen 操作 | 用途 | 当前状态 |
|--------|-----------|------|----------|
| ✅ 已完成 | `aten::index_put_` / `aten::index_put` | req_to_token 写入 | **原生** (index_put.cpp) |
| ✅ 已完成 | `aten::cat` / `aten::cat.out` | kv_indices 拼接, padding, radix cache | **原生** (TensorOps.cpp) |
| ✅ 已完成 | `aten::add` / `aten::sub` / `aten::mul` | seq_lens±1, extend_prefix_lens 等 | **原生** (dispatch stub) |
| ✅ 已完成 | `aten::sum` / `aten::mean` / `aten::norm` | reduce 操作 | **原生** (reduceOps.cpp) |
| 🔴 高 | `aten::cumsum` | kv_indptr, qo_indptr, chunk 索引 | ❌ 未注册 (5+ 处 bounce) |
| 🔴 高 | `aten::index.Tensor_out` (花式索引读取) | filter_batch, alloc_for_decode | ❌ 未注册 (4+ 处 bounce) |
| 🟡 中 | `aten::argmax` | greedy 采样 | ❌ 未注册 (9.3 MB/step) |
| 🟢 低 | `aten::clamp` / `aten::where` | position 计算, last_loc 计算 | ❌ 未注册 |
| 🟢 低 | `aten::arange` (device) | position/start_loc 生成 | ❌ 未注册 |
| 🟢 低 | `aten::neg` (int tensor) | future_indices 取反 | ❌ 未注册 |

#### 8.5.3 扩大模型覆盖

| 优先级 | 任务 | 影响模型 | 类型 |
|--------|------|----------|------|
| 🔴 P0 | SWA (滑动窗口注意力) | Mistral / Gemma / Qwen2-VL | attention kernel 改造 |
| 🟡 P1 | `GeluAndMul` kernel | Phi-1/2/3 / GPT-NeoX / StarCoder | sgl_kernel_zeus 新增 |
| 🟡 P1 | `GemmaRMSNorm` (+1 offset) | Gemma 系列 | sgl_kernel_zeus 新增 |
| 🟡 P1 | MoE Forward | Mixtral / DeepSeek-V2/V3 | grouped GEMM + fused kernel |

#### 8.5.4 代码质量改进

| 任务 | 说明 |
|------|------|
| 创建 `zeus_ops.py` 集中工具层 | 将分散在 15+ 文件中的 44 个 CPU bounce 统一封装为 `zeus_cat()` / `zeus_index()` / `zeus_cumsum()` 等 |
| logits_processor 优化 | 逐元素 `copy_` 循环改为 batch 操作（`torch.index_select` 或 CPU batch 索引） |
| 预分配 metadata buffer | 类似 FlashInfer 的 `indices_updater`，避免每步重新分配 kv_indptr/kv_indices |
| 异步 KV Cache 写入 | 若 Zeus 支持多流机制，将 KV 写入放在辅助流上 |
| Fused RoPE + KV Store | 合并为单一 kernel，减少 1 次 kernel launch/层 |

---

### 8.6 辅助性 TODO（持续推进）

以下任务不属于四大阶段主线，但随时可以推进：

| 任务 | 说明 | 优先级 |
|------|------|--------|
| FP8/INT8 量化 | 需 `fp8_scaled_mm` / `int8_scaled_mm`，显存减半 | 🟢 P2 |
| Speculative Decoding | tree verify kernels，延迟优化 | 🟢 P2 |
| Logit Softcap | Gemma2 attention 需要 | 🟢 P2 |
| `forward_mixed` | 混合 prefill+decode，continuous batching 优化 | 🟢 P2 |
| Grammar Enforcement | constrained decoding | 🟢 P3 |
| CustomAllReduce | 超越标准 ZECL 的高级集合通信 | 🟢 P3 |

---

### 8.7 路线图甘特视图

```
时间线 ──────────────────────────────────────────────────────────────────→

阶段一: Graph 接入
█████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
├─ G1: torch_zeus Graph API
├─ G2-G4: ZeusAttnBackend graph 方法
├─ G5: ZeusGraphRunner
└─ G6: 条件式 disable_cuda_graph

阶段二: 分布式接入
░░░░░░░░░░░░█████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
├─ D1-D2: ZECL all_reduce/all_gather 验证
├─ D3: GroupCoordinator 适配
├─ D4: PP 支持
└─ D5-D6: Graph + 分布式联调 / 多节点测试

阶段三: PD 分离接入
░░░░░░░░░░░░░░░░░░░░░░░░██████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
├─ PD1-PD2: ZeusTransferEngine
├─ PD3-PD4: transfer_kv kernel 适配
└─ PD5-PD7: P/D 节点集成测试 + 端到端联调

阶段四: 理清算子流程 (贯穿始终)
████████████████████████████████████████████████████████████████████████
├─ 消除关键 CPU bounce (index_put_✅, cat✅, add/sub/mul✅, argmax, kv_indices)
├─ 扩充 ATen 高频算子 (cumsum, index.Tensor_out, clamp...)
├─ 清理 SGLang 已失效的 _is_zeus bounce 分支 (cat/add ~10 处)
├─ 扩大模型覆盖 (SWA, GeluAndMul, MoE...)
└─ 代码质量改进 (zeus_ops.py, 预分配 buffer, 异步写入)
```

---

## 9. 开发避坑指南

### 9.1 LocalMem 权重转置（最常见的坑）

Zeus GEMM 期望 LocalMem 权重布局为 `(K, N)`，而 PyTorch `nn.Linear.weight` 默认为 `(N, K)`。

```python
# ❌ 错误：直接 to_local_mem 不转置
weight_lm = to_local_mem(weight, ...)  # 布局还是 (N,K)，GEMM 结果错乱

# ✅ 正确：先转置再打包
weight_lm = to_local_mem(weight.t().contiguous(), ...)  # 布局变成 (K,N)
```

**推论：** 打包后不能再用 `F.linear(x, weight)` — 它内部做 `x @ weight.T` 会破坏布局。必须用 `torch.mm(x, weight)` 直接乘。

### 9.2 _is_zeus 判断必须在 _is_cuda 之前

容器环境中 CUDA 和 Zeus runtime 可能同时存在：

```python
# ❌ 错误：Zeus 会走到 CUDA 分支
if _is_cuda:     return self.forward_cuda
elif _is_zeus:   return self.forward_zeus

# ✅ 正确：优先匹配 Zeus
if _is_zeus:     return self.forward_zeus
elif _is_cuda:   return self.forward_cuda
```

### 9.3 cos_sin_cache 必须在 CPU 构建

RotaryEmbedding 的 cos_sin_cache 初始化用 `torch.arange` + `torch.einsum`，在 Zeus 上会 fallback。

```python
# ✅ 正确做法
cos_sin_cache = compute_cos_sin_cache(device="cpu")  # CPU 构建
cos_sin_cache = cos_sin_cache.to(zeus_device)         # 一次性传到 Zeus
```

### 9.4 隐式 CPU Fallback 的识别

Zeus 上任何看似"正常"的 PyTorch 操作都可能触发隐式 fallback，产生意外的性能开销：

```python
# ✅ 以下操作在 Zeus 上已有原生实现，不再触发 fallback (2026-04 更新):
result = a + b                     # aten::add → zenl_add_kernel ✅
result = a - b                     # aten::sub → zenl_sub_kernel ✅
result = a * b                     # aten::mul → zenl_mul_kernel ✅
result = torch.cat([a, b])         # aten::cat → TensorOps.cpp ✅
result = torch.sum(x)              # aten::sum → sum_zeus_kernel ✅
self.req_to_token[idx] = values    # aten::index_put_ → zenlIndexPut ✅

# ❌ 以下操作在 Zeus 上仍会触发隐式 CPU fallback:
result = tensor[bool_mask]         # aten::index (花式索引读取)
result = torch.cumsum(x, dim=0)    # aten::cumsum
result = torch.where(cond, a, b)   # aten::where
result = torch.argmax(x, dim=-1)   # aten::argmax
result = torch.arange(n, device="zeus")  # aten::arange
result = torch.clamp(x, min=0)    # aten::clamp
result = -int_tensor               # aten::neg (int)
```

**识别方法：** 在 torch_zeus 中开启 fallback 日志，观察哪些算子触发了 CPU fallback。

### 9.5 req_to_token 矩阵拷贝优化策略

**当前状态：** `alloc_for_decode` 已优化为仅拷贝 bs 行（~KB 级），但 `write()` 和 `init_forward_metadata` 仍拷贝整个矩阵（~MB 级）。

**优化原则：**
- 能不拷就不拷（实现 Zeus 端算子）
- 不得不拷时只拷需要的行（用 `req_to_token[indices].cpu()` 而非 `req_to_token.cpu()`）
- 整矩阵拷贝是最后手段

### 9.6 lm_head 权重来源检测

tied embedding 模型的 checkpoint 中可能没有 `lm_head.weight`，此时该参数为随机值。

```python
# ✅ 正确检测方式（已实现）：
# 在遍历 safetensors 权重时用 nonlocal 变量追踪
def _tracking_iter(weights):
    lm_head_found = False
    for name, tensor in weights:
        if name.endswith('lm_head.weight'):
            lm_head_found = True
        yield name, tensor
    # lm_head_found 通过 nonlocal 传递给后续逻辑

# ❌ 早期错误做法（已废弃）：
# torch.equal(lm_head.weight, embed.weight)  ← 在 Zeus 上触发数十 MB 的 fallback
# model._zeus_lm_head_loaded = True          ← monkey-patch 反模式
```

---

## 10. 附录：推理全流程调用链路图

### 10.1 Engine 初始化完整链路

```
sgl.Engine(model_path, device="zeus")
  │
  ├── ServerArgs.__init__()
  │     └── _handle_zeus_backends()
  │           ├── attention_backend = "zeus"
  │           ├── sampling_backend = "zeus"
  │           ├── disable_cuda_graph = True
  │           └── page_size = 128 (default, assert % 128 == 0)
  │
  └── _launch_subprocesses()
        ├── [主进程] TokenizerManager
        ├── [子进程] Scheduler
        │     └── TpModelWorker
        │           └── ModelRunner.__init__()
        │                 ├── init_torch_distributed(backend="zecl")
        │                 ├── load_model()
        │                 │     ├── model = get_model_cls()(config)
        │                 │     ├── model.load_weights(weights)
        │                 │     │     └── _tracking_iter() 追踪 lm_head
        │                 │     ├── _zeus_init_lm_head_from_embed()
        │                 │     └── pack_weights() → LocalMem
        │                 │
        │                 ├── profile_max_num_reqs()
        │                 │     └── torch.zeus.mem_get_info()
        │                 │
        │                 ├── init_memory_pool()
        │                 │     ├── ReqToTokenPool(device="zeus")
        │                 │     ├── ZeusTokenToKVPool (4D tiled buffer)
        │                 │     └── ZeusPagedTokenToKVPoolAllocator (CPU 记账)
        │                 │
        │                 └── init_attention_backend()
        │                       └── ZeusAttnBackend(page_size=128)
        │
        └── [子进程] DetokenizerManager
```

### 10.2 Decode Step 完整链路（含所有 CPU↔Zeus 传输）

```
decode_step:
  │
  ├── alloc_for_decode(batch)
  │     ├── rpi_cpu = req_pool_indices.cpu()              # [Zeus→CPU] bs×4B
  │     ├── seq_lens_cpu = seq_lens.cpu()                  # [Zeus→CPU] bs×4B
  │     ├── r2t_rows = req_to_token[rpi_cpu].cpu()         # [Zeus→CPU] bs×max_ctx×4B (仅 bs 行)
  │     ├── [CPU] last_loc 计算
  │     ├── last_loc.to(device)                            # [CPU→Zeus] bs×8B
  │     │
  │     ├── ZeusPagedAllocator.alloc_decode()              # [CPU 记账]
  │     │     └── out_indices.to(device)                   # [CPU→Zeus] bs×8B
  │     │
  │     └── ReqToTokenPool.write()
  │           └── self.req_to_token[indices] = values      # ✅ Zeus 原生 aten::index_put_
  │
  ├── ZeusAttnBackend.init_forward_metadata()
  │     ├── seq_lens.cpu(), req_pool_indices.cpu()         # [Zeus→CPU] 小
  │     ├── req_to_token.cpu()                             # ⚠️ [Zeus→CPU] 整个矩阵！
  │     ├── [CPU] 构建 kv_indptr/kv_indices/qo_indptr
  │     └── .to(device)                                    # [CPU→Zeus] 中等
  │
  ├── model.forward(input_ids=[1 token], positions, forward_batch)
  │     └── 逐层计算 (全部在 Zeus 上):
  │           ├── RMSNorm → sgl_kernel_zeus.rmsnorm
  │           ├── QKV proj → ZENL LocalMem GEMM
  │           ├── RoPE → sgl_kernel_zeus.rotary_embedding
  │           ├── KV Store → sgl_kernel_zeus.store_kv_cache
  │           ├── Attention → sgl_kernel_zeus.decode_attention
  │           ├── Fused Add RMSNorm → sgl_kernel_zeus.fused_add_rmsnorm
  │           ├── FFN Up/Gate → ZENL LocalMem GEMM
  │           ├── SiLU+Mul → sgl_kernel_zeus.silu_and_mul
  │           ├── FFN Down → ZENL LocalMem GEMM
  │           └── LM Head → torch.mm (LocalMem GEMM)
  │
  └── Sampler.forward(logits)
        ├── [greedy] logits.cpu() → argmax → .to(device)   # [Zeus↔CPU] vocab_size×2B
        └── [sampling] sgl_kernel_zeus.sampling_from_logits # Zeus 上完成 ✅
```

### 10.3 Zeus ATen 算子覆盖汇总

#### 已原生注册的 torch_zeus ATen 算子

| 类别 | Ops | 文件 | 加速方式 |
|------|-----|------|----------|
| 张量创建 | `empty.memory_format`, `empty_strided`, `zeros`, `ones`, `full` | TensorFactory.cpp, TensorOps.cpp | 原生 |
| 内存操作 | `copy_`, `fill_.Scalar`, `zero_`, `_to_copy`, `_copy_from`, `_copy_from_and_resize`, `resize_` | TensorOps.cpp, Resize.cpp | 原生 |
| 拼接 | `cat`, `cat.out` | TensorOps.cpp | **原生 (新增 🆕)** |
| 视图操作 | `view`, `_reshape_alias`, `clone`, `as_strided`, `as_strided_` | ViewOps.cpp, AsStridedOps.cpp | 原生 |
| 存储操作 | `set_*` (4 种), `is_set_to` | StorageOps.cpp | 原生 |
| 标量操作 | `_local_scalar_dense` | ScalarOps.cpp | 原生 |
| GEMM | `mm`, `addmm`, `linear` | operators/zenl/gemm.cpp | ZENL 加速 |
| GEMM (CPU fallback) | `bmm`, `mv`, `addmv`, `baddbmm` | operators/zenl/gemm.cpp | CPU 回退 |
| 算术 | `add`, `sub`, `mul` | operators/zenl/add.cpp, sub.cpp, mul.cpp | **ZENL 加速 (新增 🆕)** |
| 归约 | `sum`, `mean`, `norm` | operators/zenl/reduceOps.cpp | **ZENL 加速 (新增 🆕)** |
| 嵌入 | `embedding` | operators/zenl/embedding.cpp | ZENL 加速 |
| 索引写入 | `index_put_`, `index_put` | operators/zenl/index_put.cpp | **ZENL 加速 (新增 🆕)** |
| 随机数 | `uniform_`, `normal_`, `random_*` (3 种), `bernoulli_*` (2 种), `exponential_` | random/*.cpp | 原生 |
| 流管理 | `record_stream` | RecordStream.cpp | 原生 |
| Autocast | ~90 种操作 | autocast/*.cpp | 原生 |

#### 仍缺失的关键 ATen 算子（产生 CPU bounce）

| 优先级 | ATen 操作 | SGLang 用途 | CPU bounce 影响 |
|--------|-----------|-------------|----------------|
| 🔴 高 | `aten::cumsum` | kv_indptr, qo_indptr, chunk 索引 | 5+ 处，阻碍 Zeus 端 metadata 构建 |
| 🔴 高 | `aten::index.Tensor_out` (花式索引读取) | filter_batch, alloc_for_decode | 4+ 处，每处 ~KB 级 |
| 🟡 中 | `aten::argmax` | greedy 采样 | 1 处，9.3 MB/step |
| 🟢 低 | `aten::clamp` | position 计算 | 2+ 处 |
| 🟢 低 | `aten::where` | last_loc 计算 | 1+ 处 |
| 🟢 低 | `aten::arange` (device 参数) | position/start_loc 生成 | 3+ 处 |
| 🟢 低 | `aten::neg` (int tensor) | future_indices 取反 | 1 处 |

---

## 参考文档

| 文档 | 说明 |
|------|------|
| `zeus_dev/sglang_dev_vs_main_diff_report.md` | Dev vs Main 分支差异对比报告 |
| `zeus_dev/zeus_inference_flow_trace.md` | 推理全流程追踪与 CUDA vs Zeus 深度对比 |
| `docs/zeus_gap_analysis.md` | Zeus 适配 Gap 分析 (P0/P1/P2/P3 分级) |
| `zeus_dev/zeus_adaptation_dev_doc.md` | 适配开发笔记 (含 LocalMem 转置坑点) |
| `zeus_dev/zeus_adaptation_report.md` | 适配进度总结报告 |
| `zeus_dev/demo_zeus_llm.py` | Zeus 推理端到端 demo |
| `zeus_dev/demo_zeus_layer_compare.py` | 逐层数值对比工具 |
