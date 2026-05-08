# SGLang 适配 Zeus NPU 开发文档

本文档总结了在 SGLang 中为支持 Zeus NPU 所做的大量底层适配工作。主要涵盖了“做了什么（What）”以及“为什么要这么做（Why）”，旨在为后续的开发、维护和性能优化提供参考。

## 1. 设备接入与全局识别 (Device Detection & Registration)

### 做了什么？
*   **设备注册**：在 `DeviceConfig` 和 `ServerArgs` 中添加了 `"zeus"` 作为合法的设备和 Attention/Sampling 后端选项。
*   **硬件信息读取**：在 `sglang/srt/utils/common.py` 中实现了 `is_zeus()`，以及专门针对 Zeus 的内存、设备核心数、容量的读取接口（封装了 `torch.zeus` API）。
*   **默认配置拦截**：在启动参数解析时，识别到 Zeus 硬件会自动将 `attention_backend` 和 `sampling_backend` 设置为 `"zeus"`，开启 `disable_cuda_graph=True`，并强制 `page_size=128`。

### 为什么这么做？
SGLang 原本是以 CUDA 为第一公民构建的，内部大量依赖 `torch.cuda` 的状态查询。为了让 SGLang 的引擎能够启动，调度器能够准确获取可用显存并合理划分 KV Cache，我们必须让系统全局正确认识 Zeus 设备，而不是将其误判为 CPU 或报错。设定默认参数是为了防止框架走到不支持的 CUDA Graph 或不兼容的 Page Size 分支导致崩溃。

---

## 2. 算子分发与专用 Kernel 接入 (Operator Dispatch & Kernels)

### 做了什么？
*   **统一分发拦截**：在 `CustomOp` 中新增了 `forward_zeus()` 接口，并通过 `dispatch_forward` 将计算流引导至 Zeus 分支。
*   **核心算子替换**：对于 `SiluAndMul`、`RMSNorm`、`RotaryEmbedding` 等核心算子，在 `forward_zeus` 中调用了 `sgl_kernel_zeus` 库中高度优化的专用内核。
*   **Attention 与 Sampling 后端**：新增了 `ZeusAttnBackend` 处理 Paged Attention 的 Extend/Decode 阶段；Sampling 则针对 Zeus 实现了基于 `sgl_kernel_zeus` 的 Fused 采样核，对于单纯的 greedy 采样则转移至 CPU 执行 `argmax`。

### 为什么这么做？
Zeus NPU 具有与 GPU 完全不同的底层指令集和存储架构，无法直接运行 Triton 或 CUDA C++ Kernel。如果依赖 PyTorch Native 的 fallback，会导致严重的性能问题（无法利用算力）。因此我们必须在执行图的关键节点，将计算请求精准路由到为 Zeus 量身定制的底层 Kernel 上。

---

## 3. 内存管理与 KV Cache 定制 (Memory Management & KV Pool)

### 做了什么？
*   **定制化 KV 内存池**：实现了 `ZeusTokenToKVPool` 和对应的分配器 `ZeusPagedTokenToKVPoolAllocator`。
*   **数据排布与后端对接**：将 KV Cache 的存储调用绑定到 `sgl_kernel_zeus.store_kv_cache`，并在调度层管理内存页。

### 为什么这么做？
Zeus NPU 的底层存储对数据对齐有严格要求（如 Paged Attention 必须满足 Page Size 为 128 的倍数）。标准的 GPU 内存池分配策略并不符合 Zeus 的内存粒度和接口规范，定制化内存池能够确保 KV Cache 高效、安全地驻留在 NPU 显存中，避免越界或存取效率低下的问题。

---

## 4. ATen Fallback 规避与 CPU 卸载 (ATen Workarounds)

### 做了什么？
*   **大规模重写控制流张量操作**：在 `LogitsProcessor`、`ForwardBatch`、`RadixCache`、`MemoryPool` 等模块中，针对索引计算（Indexing）、`cumsum`、`where`、`cat`、`sub` 等操作，做了大量 `tensor.cpu() -> 运算 -> .to(device)` 的显式转移逻辑。
*   **前处理与后处理逻辑剥离**：例如 Prefill 阶段的 Pruning 提取，改为在 CPU 上计算好 index 后通过循环或显式的 `copy_` 覆盖，替代原有的高级索引语法。

### 为什么这么做？
**这是适配过程中非常耗时的一环。** 目前 Zeus 上的 PyTorch (ATen) 对于许多基础的张量控制流、索引操作缺乏原生支持。如果让框架自动 Fallback 到 CPU，不仅触发开销巨大，还可能由于隐式的上下文切换导致严重的同步阻塞。通过**主动将低算力密度的调度簿记（Bookkeeping）计算转移到 CPU 上执行**，可以规避缺失算子导致的报错，同时把宝贵的 Zeus 算力集中在 GEMM 和 Attention 等密集计算上。

---

## 5. 模型权重加载与内存布局优化 (Model Loading & Weight Packing)

### 做了什么？
*   **权重格式转换 (Packing)**：在 `model_loader` 中，模型加载完成后，利用 `torch_zeus.zeus.pack_weights` 将 `LinearBase` 和 `ParallelLMHead` 的权重转换打包到 **LocalMem** 中。
    *   **⚠️ 避坑指南 (权重转置)**：在使用底层的 `to_local_mem` 接口打包权重时，Zeus GEMM 算子期望的 LocalMem 权重布局是 `(K, N)`，而 PyTorch 中 `nn.Linear.weight` 默认形状为 `(out_features, in_features)` 即 `(N, K)`。因此在手动打包权重时（如在编写算子对比测试用例时），**必须先显式调用 `.t().contiguous()` 进行转置**，例如 `to_local_mem(weight.t().contiguous(), ...)`。如果不进行转置，会导致计算时维度完全错乱并引发诸如 `RuntimeError: shape '...' is invalid for input of size ...` 的隐蔽错误。
*   **解绑词嵌入 (Tie-embeddings Override)**：特殊处理了 `tie_word_embeddings`。强制要求 `embed_tokens` 留在普通的 **GDG (Global Memory)**，而为其单独拷贝一份独立权重到 `lm_head` 并打包进 **LocalMem**。

### 为什么这么做？
这完全是针对 Zeus 存储层级的深度优化：
1. **LocalMem 提速**：Zeus 的矩阵乘法 (GEMM) 如果直接在普通显存（GDG）上读取，带宽极差。将线性层权重打包进 Tiled 布局的 LocalMem，能彻底释放 GEMM 的算力。
2. **读写特性差异**：虽然 GEMM 在 LocalMem 极快，但词表查找（Embedding Lookup）这种离散 Gather 操作在 LocalMem 中异常缓慢。如果 `embed_tokens` 和 `lm_head` 共享权重（Tie），必然顾此失彼。因此我们选择“断开绑定”，用空间换时间：让 Embedding 用 GDG 跑得快，让 LM Head 用 LocalMem 算得快。

---

## 总结

本次 Zeus 适配的核心哲学是：**顺应硬件特性，规避软件短板**。
通过显式设备注册让框架畅通无阻，通过替换定制 Kernel 释放稠密算力，通过切断隐式的 ATen Fallback 保证调度流畅，最后利用 LocalMem 布局规划将硬件性能压榨到极致。这些工作共同构成了 `demo_zeus_llm.py` 得以成功跑通的基石。