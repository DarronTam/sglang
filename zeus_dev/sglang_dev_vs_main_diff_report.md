# SGLang 仓库 dev 与 main 分支差异对比报告

> 仓库地址：https://github.com/DarronTam/sglang.git
> 生成时间：2026-03-23
> 分叉点（merge-base）：`94e125113` — *enable flashinfer-jit-cache in image build and ci install to speed up model launch (#14959)*

---

## 一、总体概况

| 项目 | 数值 |
|------|------|
| dev 独有提交数 | **8** |
| main 独有提交数 | **2897**（上游社区持续合入） |
| dev 新增/修改文件数 | **30** |
| 总变更行数 | **+3508 / −131** |
| dev 分支主题 | **Zeus NPU 硬件后端适配** |

### 结论

dev 分支是在 main 的一个早期快照（`94e1251`）上，叠加了 **8 个 Zeus NPU 适配提交**。main 上游此后已推进约 2897 个提交，dev 尚未回合（rebase/merge）上游变更。

---

## 二、dev 独有提交列表（按时间顺序）

| # | Commit | 日期 | 标题 | 修改文件数 |
|---|--------|------|------|-----------|
| 1 | `2776493b` | 2026-03-17 14:18 | feat(zeus): add initial adaptation for Zeus NPU | 30 |
| 2 | `2d0b42f1` | 2026-03-17 14:51 | fix(zeus): transpose weights before packing to LocalMem to avoid shape mismatches | 1 |
| 3 | `56ba3124` | 2026-03-17 15:02 | docs(zeus): add LocalMem weight transposition pitfall to dev doc | 1 |
| 4 | `e7b145c6` | 2026-03-17 15:42 | fix(zeus): workaround aten::equal fallback when checking if lm_head weight is loaded | 1 |
| 5 | `1f9cdc56` | 2026-03-17 17:08 | fix(zeus): remove heavy tensor equality check and rely on _zeus_lm_head_loaded flag instead | 1 |
| 6 | `3c4ef4a1` | 2026-03-17 17:57 | chore(zeus): update demo_zeus_llm.py with explicit prompt formatting and greedy decoding | 1 |
| 7 | `7141fe7c` | 2026-03-19 12:59 | update demo template | 2 |
| 8 | `f14344e1` | 2026-03-20 11:36 | update device env selection | 3 |

---

## 三、修改文件清单与变更统计

### 3.1 新增文件（dev 独有）

| 文件路径 | 行数 | 说明 | 新增原因 |
|---------|------|------|----------|
| `python/sglang/srt/layers/attention/zeus_backend.py` | +194 | Zeus Attention 后端（extend/decode attention 调度） | SGLang 的 attention 后端是插件式架构（flashinfer/triton/torch_native），Zeus NPU 有自己的硬件 attention kernel（通过 sgl-kernel-zeus 暴露），需要一个独立后端来封装 Zeus 特有的 CSR 格式 metadata 构建和 kernel 调用，无法复用已有的 CUDA/Triton 后端 |
| `python/sglang/srt/mem_cache/zeus_allocator.py` | +212 | Zeus 分页 KV cache 分配器（CPU 侧记账） | Zeus NPU 不支持 `torch.cat`、`torch.unique`、`torch.cumsum` 等 ATen 算子原生执行，原版 `PagedTokenToKVPoolAllocator` 的 alloc/free 逻辑大量使用这些算子，直接运行会反复触发隐式 CPU fallback（每次 D2H→计算→H2D），性能极差。新 allocator 将所有记账状态保持在 CPU，仅最终 output indices 搬到设备，一次性消除数十次隐式拷贝 |
| `python/sglang/srt/mem_cache/zeus_memory_pool.py` | +122 | Zeus KV cache 内存池（sgl-kernel-zeus tiled 布局） | Zeus NPU 的 attention kernel 要求 KV cache 以 4D tiled 布局存储（`[num_pages, num_kv_heads, page_size, head_dim]`），而原版 MHATokenToKVPool 使用 flat 2D 布局（`[max_tokens, num_kv_heads, head_dim]`），两者内存排列不兼容；此外 Zeus 写入 KV 必须通过 `sgl_kernel_zeus.store_kv_cache` 专用 kernel 处理硬件 tiled 格式（K: block-tiled row-major, V: 16-byte column-group interleaved），不可使用 PyTorch 索引赋值 |
| `docs/zeus_gap_analysis.md` | +175 | Zeus 适配 Gap 分析文档 | 系统性梳理当前适配未覆盖的算子/特性/模型，按 P0（阻塞更多模型）/P1（生产性能）分级，用于指导后续开发优先级 |
| `zeus_dev/demo_cuda_llm.py` | +143 | CUDA 对照推理脚本 | 提供 CUDA 端「黄金参考」输出，用于和 Zeus 端做逐 token 数值对比验证 |
| `zeus_dev/demo_zeus_llm.py` | +129 | Zeus 推理 demo 脚本 | 端到端验证 Zeus 推理链路是否跑通（prompt 格式化→prefill→decode→greedy sampling）的最小可运行脚本 |
| `zeus_dev/demo_zeus_layer_compare.py` | +1579 | Zeus 与 CUDA 逐层数值对比工具 | Zeus 是全新硬件，每一层（embedding→attention→FFN→logits）的数值误差都需要逐层验证以定位是哪个算子引入偏差，此工具自动化了 1579 行的逐层 tensor diff |
| `zeus_dev/zeus_adaptation_dev_doc.md` | +68 | 适配开发文档 | 记录开发中遇到的陷阱（如 LocalMem 权重转置）和解法，避免后续开发者踩同样的坑 |
| `zeus_dev/zeus_adaptation_report.md` | +181 | 适配进度报告 | 汇总适配进展和待办，便于团队同步 |

### 3.2 修改文件

| 文件路径 | 变更 | 说明 | 修改原因 |
|---------|------|------|----------|
| `python/sglang/srt/configs/device_config.py` | +1/−1 | 设备配置增加 zeus 类型 | 原 `DeviceConfig` 白名单只有 `cuda/xpu/hpu/cpu/npu`，Zeus 设备名 `"zeus"` 不在其中，初始化时会抛 `RuntimeError`，必须加入白名单才能启动 |
| `python/sglang/srt/custom_op.py` | +9/−1 | CustomOp 增加 `forward_zeus` 派发 | SGLang 所有算子层（RMSNorm、SiluAndMul、RotaryEmbedding 等）都继承 `CustomOp`，通过 `dispatch_forward()` 按设备类型选择具体实现。Zeus 注册为 `PrivateUse1` 设备，没有对应的派发分支就会走 CUDA 路径报错。新增 `forward_zeus` 方法及对应派发，默认回退到 `forward_native`，子类可以 override 为 sgl-kernel-zeus 加速版本 |
| `python/sglang/srt/layers/activation.py` | +5 | SiluAndMul 增加 `forward_zeus` | 原版 `forward_cuda` 使用 flashinfer 的 `silu_and_mul`（CUDA kernel），Zeus 上不可用。`forward_native` 虽然可以 fallback 但走纯 PyTorch 实现要两次访存（silu + mul），Zeus 有融合 kernel `sgl_kernel_zeus.silu_and_mul` 一次完成，节省约一半访存带宽 |
| `python/sglang/srt/layers/attention/attention_registry.py` | +7 | 注册 zeus attention 后端 | SGLang 的 attention 后端是通过 `@register_attention_backend("name")` 装饰器注册的插件系统。不注册 `"zeus"` 名字，`server_args.attention_backend="zeus"` 时会触发 KeyError 找不到后端 |
| `python/sglang/srt/layers/layernorm.py` | +13 | RMSNorm 增加 `forward_zeus` | 原版走 flashinfer 的 `rmsnorm`/`fused_add_rmsnorm`（CUDA），Zeus 上不可用。sgl-kernel-zeus 提供了 Zeus 硬件优化的 rmsnorm kernel，特别是 `fused_add_rmsnorm`（norm + residual add 融合为一次 kernel），避免额外的 residual 读写开销 |
| `python/sglang/srt/layers/logits_processor.py` | +50/−24 | Logits 裁剪/LM head 矩阵乘法避免 Zeus fallback | **两个问题**：① Prefill 阶段用 `hidden_states[last_index]` 做花式索引（fancy indexing），Zeus 的 `aten::index` 未实现会 fallback，改为逐条 `copy_` 逐元素拷贝避免；② `F.linear(hidden, weight)` 内部做 `hidden @ weight.T`，但 Zeus LocalMem 权重已经预转置为 (K,N)，再 `.T` 会破坏内存布局，改为 `torch.mm(h, weight)` 直接乘，并在 weight 加载时确保布局正确 |
| `python/sglang/srt/layers/quantization/unquant.py` | +16 | UnquantizedLinearMethod 增加 Zeus weight packing | Zeus GEMM 要求权重存在 LocalMem 中。`pack_weights` 后权重变成 LocalMem 格式 (K,N)，`F.linear` 内部的 `weight.T` 会导致结果错误。此处检测如果权重已在 LocalMem（`is_local_mem(layer.weight)=True`），则用 `torch.mm(x, weight)` / `torch.addmm(bias, x, weight)` 替代 `F.linear`，绕过内部转置 |
| `python/sglang/srt/layers/rotary_embedding.py` | +62/−7 | RotaryEmbedding/DeepseekScaling 增加 `forward_zeus` | **三个改动**：① `forward_zeus` 调用 `sgl_kernel_zeus.rotary_embedding`（硬件融合 kernel，一次完成 cos/sin 查表+旋转+写回），替代 `forward_native` 的多步 PyTorch 算子；② cos_sin_cache 初始化时强制在 CPU 构建（`init_device="cpu"`），因为 `torch.arange` + `torch.einsum` 在 Zeus 上会 fallback；③ `torch.arange(max_position_embeddings)` 加上 `device=inv_freq.device` 确保和 inv_freq 在同设备，避免跨设备 einsum |
| `python/sglang/srt/layers/sampler.py` | +97/−32 | 采样路径增加 zeus_sampling_from_logits 融合实现 | **核心原因**：原版采样分 5 步（div temp → softmax → top-k renorm → top-p renorm → sample），每步都是独立 kernel launch。Zeus 提供 `sampling_from_logits` 融合算子，一次 kernel 完成全部步骤，减少 5 次 kernel launch 为 1 次。此外 `torch.argmax` 在 Zeus 上未实现，greedy 路径改为 `.cpu()` 后 argmax 再回传 |
| `python/sglang/srt/layers/vocab_parallel_embedding.py` | +14/−0 | Embedding 增加 Zeus forward | Zeus 权重在 LocalMem 中是转置的 (K,N) 格式以兼容 GEMM，普通的 `F.embedding` 按行索引会取到错误数据。`sgl_kernel_zeus.embedding` 是列方向 gather，能正确从转置权重中取出 embedding 向量 |
| `python/sglang/srt/managers/overlap_utils.py` | +15/−0 | Overlap 调度增加 Zeus guard | `_resolve_future_token_ids` 用到 `torch.where` + 负数索引 + `torch.clamp + 花式索引`，这些算子在 Zeus 上均不支持原生执行。改为 CPU 侧执行后写回。另外 `torch.arange(..., device=zeus)` 在 Zeus 上也不支持，改为先在 CPU 创建再 `.to(device)` |
| `python/sglang/srt/managers/schedule_batch.py` | +48/−9 | Batch 管理操作 CPU bounce 避 Zeus fallback | **三处热路径**：① `seq_lens += 1`（每个 decode step）— Zeus 不支持 int tensor 的 `add_.Scalar`，隐式 fallback 会每步触发 D2H+H2D；② `filter_batch` 中的 `tensor[keep_indices]` 花式索引 — Zeus 不支持 `aten::index`；③ `merge_batch` 中的 `torch.cat` — Zeus 不支持。全部改为显式 `.cpu()` 操作后 `.to(device)`，将隐式多次拷贝合并为一次显式拷贝 |
| `python/sglang/srt/managers/scheduler.py` | +9/−0 | Scheduler 中 future_indices 取反避免 fallback | `future_indices_or_next_token_ids = -future_indices.indices` 用到 `aten::neg`（int tensor），Zeus 不支持。改为 `(-x.cpu()).to(device)`。虽然语义相同但避免了隐式 fallback 的额外内存分配和同步开销 |
| `python/sglang/srt/mem_cache/common.py` | +55/−0 | alloc_for_decode 增加 Zeus CPU 侧索引 | `alloc_for_decode` 是 decode 热路径，每步执行。内部用 `req_to_token[req_pool_indices, seq_lens - 1]` 做二维花式索引，Zeus 不支持 `aten::index` 的多维 tensor 索引。改为把 r2t/rpi/sl 全部拉到 CPU 做索引后一次性传回，将每步 3 次隐式 D2H 合并为 1 次显式 |
| `python/sglang/srt/mem_cache/memory_pool.py` | +32/−5 | ReqToTokenPool.write/MambaPool.free/MHA data_ptrs CPU bounce | **三处**：① `ReqToTokenPool.write` 用 `index_put_`（tensor indices），Zeus 不支持，改为 CPU 侧 index_put 后整块回传；② `MambaPool.free` 用 `torch.cat` 拼接 free_slots，Zeus 不支持；③ `MHATokenToKVPool` 初始化时 `torch.cat([k_data_ptrs, v_data_ptrs])` 拼接 uint64 指针，Zeus 的 cat 不支持 uint64 dtype |
| `python/sglang/srt/mem_cache/radix_cache.py` | +16/−6 | torch.cat → _cat_zeus（CPU 拼接后回传） | RadixCache 在 `match_prefix`、`cache_finished_req`、`total_allocated_tokens` 三处用 `torch.cat` 拼接 int64 token indices。Zeus 对 `torch.cat` 的支持不完善（尤其是变长 list of tensors），封装 `_cat_zeus()` 辅助函数统一处理：`.cpu()` → `torch.cat` → `.to(device)` |
| `python/sglang/srt/model_executor/forward_batch_info.py` | +48/−14 | pad_tensor/extend 字段/cumsum 等 CPU bounce | **四处**：① `_pad_tensor_to_size` 用 `torch.cat` + `new_zeros`/`new_full` pad tensor，Zeus 不支持；② `extend_prefix_lens = seq_lens - 1` 和 `extend_start_loc = torch.arange(...)` 在 Zeus 上 fallback；③ `cumsum(dim=1)` 构建 prefix_chunk_cu_seq_lens，Zeus 不支持 cumsum；④ 以上全是 prefill/decode 准备阶段的 metadata 构建，在 Zeus 上没有对应 kernel，必须在 CPU 侧完成 |
| `python/sglang/srt/model_executor/model_runner.py` | +38/−5 | 初始化 Zeus KV pool / allocator / zecl 通信后端 | **三处**：① 分布式通信后端设为 `"zecl"`（Zeus 的集合通信库，类似 NCCL），否则 PyTorch distributed 找不到后端；② KV pool 初始化时选用 `ZeusTokenToKVPool`（tiled 布局），否则会用 MHA 的 flat 布局，Zeus attention kernel 无法正确读取；③ allocator 选用 `ZeusPagedTokenToKVPoolAllocator`（CPU 记账），否则每次 alloc/free 都会大量 fallback |
| `python/sglang/srt/model_loader/loader.py` | +107/−0 | Zeus 专用权重加载器（LocalMem 转置+打包） | **核心原因**：Zeus GEMM 的高性能路径要求权重存放在 LocalMem（片上近存）中，且布局必须是 (K,N)（转置的）。原版 `load_weights` 把权重加载到普通 GDG（Global Device Memory）中。新增逻辑：① 遍历 `load_weights` 的权重迭代器，标记 `lm_head.weight` 是否在 checkpoint 中（判断 tie embedding）；② 对所有 LinearBase/ParallelLMHead 模块调用 `pack_weights` 转置并搬入 LocalMem；③ 处理 tied embedding 场景（embed_tokens 和 lm_head 共享权重时，需要把 VocabParallelEmbedding 也 pack 到 LocalMem，牺牲 embedding lookup 性能换取 GEMM 性能） |
| `python/sglang/srt/server_args.py` | +25/−0 | zeus 设备识别 / sampling_backend / attention_backend 默认值 | **四处自动配置**：① `attention_backend` 自动设为 `"zeus"`（Zeus 有专用 attention kernel，不能走 flashinfer/triton）；② `sampling_backend` 设为 `"zeus"`（使用融合采样 kernel）；③ `disable_cuda_graph=True`（Zeus 的 graph capture 机制 `zertGraphLaunch` 尚未就绪）；④ `page_size=128`（Zeus attention kernel 硬性要求 page size 为 128 的倍数，对齐硬件 tiled 内存单元） |
| `python/sglang/srt/utils/common.py` | +66/−0 | is_zeus() / get_zeus_memory_capacity() / 设备工具 | SGLang 的设备抽象层（`get_device`/`get_device_count`/`get_device_capability` 等）没有 Zeus 分支，任何用到这些工具函数的地方都会忽略 Zeus 设备。新增完整的设备检测和查询函数，让 Zeus 设备在整个框架中被正确识别和使用 |

---

## 四、核心改动详解

### 4.1 Zeus 设备识别与基础设施

**涉及文件**：`utils/common.py`、`server_args.py`、`configs/device_config.py`

#### 修改原因

SGLang 的设备抽象层原本只覆盖 CUDA/XPU/HPU/CPU/NPU，Zeus 作为通过 `torch.register_privateuse1_backend("zeus")` 注册的 PrivateUse1 设备，框架中没有任何地方能识别到它。这导致：
- `get_device()` 返回 `"cuda"` 而非 `"zeus"`，后续所有设备相关逻辑全部走错
- `DeviceConfig` 白名单校验直接抛异常
- 显存查询、设备数量、SM 数量等运行时信息无法获取

因此需要在整个设备抽象层补齐 Zeus 分支。

#### 具体改动

| 改动点 | 为什么要改 |
|--------|------------|
| `is_zeus()` 检测函数 | 提供统一的设备判断入口，后续所有 `if _is_zeus:` 分支都依赖此函数 |
| `get_device()` 增加 zeus 分支 | 框架启动时通过此函数确定运行设备，不加则永远检测不到 Zeus |
| `get_device_count()` / `get_device_core_count()` / `get_device_capability()` | 框架用这些函数做 TP 并行度推理、内存预算计算、算子选择，缺失会导致崩溃或使用错误默认值 |
| `get_zeus_memory_capacity()` | 框架根据显存容量计算 `max_total_num_tokens`（KV cache 容量），没有这个函数则无法自动计算，需要用户手动指定 |
| `attention_backend = "zeus"` 自动设置 | Zeus 有专用 attention kernel，不能走 flashinfer（CUDA only）/triton（PTX only），必须路由到 zeus 后端 |
| `sampling_backend = "zeus"` 自动设置 | Zeus 有融合采样 kernel（一次 kernel 完成 temp+softmax+topk/p+sample），比 flashinfer/pytorch 采样更高效 |
| `disable_cuda_graph = True` | Zeus 的 graph capture 机制（`zertGraphLaunch`）尚未在 torch_zeus 中完整实现，强制开启会崩溃 |
| `page_size = 128` | Zeus attention kernel 硬件要求 page size 必须是 128 的倍数，对齐片上 tiled 内存单元，否则 kernel 会读到错误的 KV 偏移 |
| `os.environ.setdefault("SGLANG_DEVICE", ...)` | 多进程场景下子进程需要通过环境变量感知设备类型，不设置则子进程的 `is_zeus()` 可能返回错误结果 |
| 分布式后端 `"zecl"` | Zeus 的集合通信库名为 zecl（类似 NCCL），PyTorch distributed 需要显式指定后端名，否则会尝试用 NCCL 导致找不到 GPU 报错 |

### 4.2 CustomOp 派发机制

**涉及文件**：`custom_op.py`

#### 修改原因

SGLang 的所有计算密集型算子（RMSNorm、SiluAndMul、RotaryEmbedding 等）都继承 `CustomOp` 基类，通过 `dispatch_forward()` 按设备类型选择实现：

```python
def dispatch_forward(self):
    if _is_cuda:  return self.forward_cuda     # flashinfer/sgl_kernel
    elif _is_hip: return self.forward_hip       # ROCm
    elif _is_npu: return self.forward_npu       # Ascend
    ...
```

Zeus 注册为 `PrivateUse1` 设备类型，没有对应分支的话会走到 `forward_cuda` 然后因为 CUDA 不可用而崩溃。

#### 为什么这么改

- 新增 `forward_zeus` 方法，**基类默认回退到 `forward_native()`**（纯 PyTorch 实现，功能正确但性能一般）
- 子类可以 override `forward_zeus` 为 `sgl_kernel_zeus` 加速版本（如 RMSNorm、SiluAndMul 等）
- **将 `_is_zeus` 判断放在 `_is_cuda` 之前**：因为 Zeus 的 `torch.zeus.is_available()` 和 `torch.cuda.is_available()` 可能同时为 True（容器中同时存在 CUDA 和 Zeus runtime），必须优先匹配 Zeus，否则会错误地走 CUDA 路径

### 4.3 算子层 Zeus 实现

#### 修改原因

SGLang 原版的计算密集型算子全部依赖 flashinfer（CUDA kernel）或 Triton（生成 PTX），Zeus NPU 两者都不支持。如果全部回退到 `forward_native`（纯 PyTorch），虽然功能正确但性能损失巨大（逐元素 Python 循环 + 多次 kernel launch + 多次访存）。

`sgl_kernel_zeus` 是针对 Zeus 硬件 ISA 编写的融合 kernel 包，每个算子都做了硬件特化优化（tiled 内存访问、向量化运算、流水线调度）。

#### 为什么每个算子都需要单独实现

| 算子 | 对应文件 | 为什么不能用 native fallback | Zeus 实现的优势 |
|------|---------|---------------------------|----------------|
| `rmsnorm` / `fused_add_rmsnorm` | `layernorm.py` | native 版分 3 步（mean → rsqrt → mul），3 次 kernel launch + 3 次全量读写 | 融合为 1 次 kernel，reduce + normalize + weight mul 在片上完成；`fused_add_rmsnorm` 额外融合 residual add，省一次全量读写 |
| `silu_and_mul` | `activation.py` | native 版分 2 步（silu → mul），2 次访存 | 融合为 1 次 kernel，SiLU 激活和逐元素乘法在同一次读写中完成 |
| `rotary_embedding` | `rotary_embedding.py` | native 版需 `cos_sin_cache` 查表 + 切片 + 旋转 + 拼接，6+ 步 | 融合为 1 次 kernel，position → cos/sin → rotate 全部在片上完成 |
| `extend_attention` / `decode_attention` | `zeus_backend.py` | native attention（`torch_native_backend`）是纯 PyTorch 实现，无法利用 Zeus 硬件的 tiled attention 单元 | Zeus 硬件有专用 attention 计算单元，需要特定的 CSR 格式输入和 tiled KV 布局 |
| `store_kv_cache` | `zeus_memory_pool.py` | 普通 `tensor[indices] = value` 赋值会用 `aten::index_put_`（Zeus 不支持） | 专用 kernel 处理 Zeus tiled 内存布局（K: block-tiled row-major, V: 16-byte column-group interleaved） |
| `sampling_from_logits` | `sampler.py` | 原版分 5 步（div temp → softmax → top-k → top-p → sample），5 次 kernel launch | **融合为 1 次 kernel**：temp scaling + softmax + top-k/p filtering + multinomial sample 一次完成 |
| `top_k_renorm_prob` / `top_p_renorm_prob` | `sampler.py` | 当不走融合路径时（如需要 logprob 返回）的独立 renorm | Zeus 原生实现比 PyTorch 的 `sort + cumsum + scatter` 组合更高效 |
| `top_k_top_p_sampling_from_probs` | `sampler.py` | 同上，非融合路径的联合采样 | 避免分两步（top-k filter → top-p filter）的额外排序开销 |
| `min_p_sampling_from_probs` | `sampler.py` | min-p 需要动态阈值裁剪，PyTorch 实现需要多步 | 硬件加速的 threshold filter + sample |
| `embedding` | `vocab_parallel_embedding.py` | 权重在 LocalMem 中是转置的 (K,N)，`F.embedding` 按行索引取到的是错误数据 | 列方向 gather，能正确从转置权重中取出 embedding 向量 |

### 4.4 Zeus Attention Backend（全新）

**文件**：`layers/attention/zeus_backend.py`（+194 行）

#### 修改原因

SGLang 的 attention 计算是模型推理的核心瓶颈（占总 FLOPS 的 60%+），且每个硬件平台都有自己专用的 attention 实现：
- CUDA：flashinfer（PagedAttention with CUDA cores + Tensor Cores）
- AMD：aiter / triton
- Ascend NPU：ascend attention

Zeus NPU 有自己的硬件 attention 计算单元，需要特定的输入格式（CSR 索引 + tiled KV buffer），无法复用上述任何后端。

#### 为什么要这么设计

| 设计选择 | 原因 |
|---------|------|
| metadata 在 CPU 侧构建（kv_indptr/kv_indices/kv_last_page_len） | 这些是 int32/int64 的小 tensor（长度 = batch_size），在 Zeus 上用 `cumsum`/`arange`/花式索引构建会反复 fallback；CPU 构建后一次性 `.to(device)` 更高效 |
| 分离 `forward_extend()` 和 `forward_decode()` | Zeus 的 prefill（变长 Q）和 decode（Q 长度=1）使用不同的硬件 kernel，输入约束不同 |
| 不支持 SWA | Zeus 的 attention kernel 目前没有 sliding_window_size 参数，需要在 kernel 层面添加支持后才能在此后端启用 |

### 4.5 Zeus KV Cache 管理

#### 4.5.1 ZeusTokenToKVPool（`zeus_memory_pool.py`，+122 行）

**为什么不能复用 MHATokenToKVPool：**

原版 `MHATokenToKVPool` 使用 flat 2D 布局（`[max_tokens, num_kv_heads, head_dim]`），通过 PyTorch `tensor[indices] = value` 写入。Zeus 的 attention kernel 有两个硬件约束：

1. **KV buffer 必须是 4D tiled 布局** `[num_pages, num_kv_heads, page_size, head_dim]`，这是硬件 tiled 内存单元的物理排列方式，attention 计算单元直接从这个布局读取，flat 布局会导致 cache miss 和数据错位
2. **写入必须通过 `store_kv_cache` kernel**：Zeus 的 V cache 使用 16-byte column-group interleaved 格式（为了对齐硬件向量化宽度），普通 PyTorch 索引赋值写入的字节排列不正确

因此需要一个新的 pool 类来管理不同形状的 buffer 并使用专用 kernel 写入。

#### 4.5.2 ZeusPagedTokenToKVPoolAllocator（`zeus_allocator.py`，+212 行）

**为什么不能复用 PagedTokenToKVPoolAllocator：**

原版 allocator 在 Zeus 设备上执行时，以下操作每次都会触发隐式 CPU fallback：

| 操作 | 原版调用 | Zeus 缺失的算子 | 每次 alloc/free 触发次数 |
|------|---------|---------------|---------------------|
| 空闲页合并 | `torch.cat([free_pages, release_pages])` | `aten::cat` | 1 |
| 空闲页排序 | `torch.sort(free_pages)` | `aten::sort` | 1 |
| 页去重 | `torch.unique(indices // page_size)` | `aten::unique` | 1 |
| 累加求和 | `torch.cumsum(need_new_pages, 0)` | `aten::cumsum` | 2 |
| 条件判断 | `(sl % page_size == 1).int()` | `aten::remainder` | 1 |

每次隐式 fallback = 1 次 D2H + CPU 计算 + 1 次 H2D，每个 decode step 的 alloc+free 会产生 **6+ 次隐式拷贝**。

新的 `ZeusPagedTokenToKVPoolAllocator` 将所有记账状态保持在 CPU tensor 上（`free_pages`、`release_pages` 始终在 CPU），中间计算全在 CPU 完成，仅最终的 `out_indices` 做一次 `.to(zeus_device)`，将 6+ 次隐式拷贝降为 **1 次显式拷贝**。

### 4.6 CPU Bounce 避免 Zeus Fallback（大面积修改）

#### 修改原因

Zeus NPU 的 ATen 算子覆盖率有限。当 PyTorch 在 Zeus 设备上调用一个未注册的算子时，会自动触发 **CPU fallback 机制**：

```
Zeus tensor → D2H copy → CPU 计算 → H2D copy → Zeus tensor
```

这个机制虽然保证了功能正确性，但有严重的性能问题：
1. **每次 fallback 都是同步操作**，会阻塞 Zeus 计算流水线
2. **隐式拷贝不可控**：一条看似简单的 `seq_lens += 1` 实际产生 D2H + CPU add + H2D 三步
3. **热路径叠加效应**：decode 每步都执行的代码中有 13+ 处 fallback，累积延迟巨大

#### 为什么选择显式 CPU bounce 而非实现 ATen kernel

- 这些算子（cat/index_put_/argmax/cumsum 等）操作的都是**小型管理 tensor**（batch_size 大小，通常 < 1024 元素），不是计算瓶颈
- 为这些杂项算子逐个实现 Zeus kernel 的工程量大、收益低
- 显式 `.cpu()` → 计算 → `.to(device)` 可以把多次隐式拷贝**合并为一次显式来回**，实际更快

#### 逐模块改法与原因

| 模块 | 改法 | 具体原因 |
|------|------|----------|
| `sampler.py` — `argmax` | `torch.argmax(logits.cpu(), -1).to(device)` | `aten::argmax` 在 Zeus 上未实现，greedy decode 每步必调。logits shape = (bs, vocab_size)，搬到 CPU 做 argmax 后只回传 (bs,) 的 token_id，回传量远小于 logits 本身 |
| `schedule_batch.py` — `seq_lens += 1` | `(seq_lens.cpu() + 1).to(device)` | `aten::add_.Scalar`（int tensor in-place）未实现。每个 decode step 必执行，隐式 fallback 每步产生 1 次 D2H + 1 次 H2D |
| `schedule_batch.py` — filter/merge | `.cpu()[indices].to(device)` | `aten::index`（花式索引）未实现。filter_batch 在请求结束时调用，merge_batch 在新请求加入时调用，两者都用花式索引筛选 tensor |
| `memory_pool.py` — `ReqToTokenPool.write` | CPU 侧 index_put 后整体回传 | `aten::index_put_`（tensor indices 版本）未实现。write 在每个 decode step 调用，原版对整个 req_to_token 矩阵做二维 index_put_，隐式 fallback 会拷贝整个矩阵来回。改为 CPU 侧 put 后一次性替换整个 tensor |
| `memory_pool.py` — `MambaPool.free` | `.cpu()` cat 后回传 | `aten::cat` 对 Zeus int64 tensor 未实现 |
| `memory_pool.py` — `data_ptrs` cat | CPU 拼接后回传 | `aten::cat` 对 uint64 dtype 未实现（data_ptrs 是指针地址） |
| `radix_cache.py` — `torch.cat` | `_cat_zeus()` 辅助函数 | RadixCache 多处拼接变长 int64 列表，统一封装避免重复代码。`_cat_zeus` = `.cpu()` → `cat` → `.to(device)` |
| `forward_batch_info.py` — `_pad_tensor_to_size` | CPU pad 后回传 | pad 操作用 `torch.cat([tensor, zeros])` 实现，`aten::cat` 未实现 |
| `forward_batch_info.py` — `extend_prefix_lens`/`extend_start_loc` | CPU 构建后回传 | `seq_lens - 1` 用到 `aten::sub.Scalar`（int tensor），`torch.arange` 在 Zeus 上不支持直接创建 |
| `forward_batch_info.py` — `cumsum` | `.cpu().cumsum().to(device)` | `aten::cumsum` 未实现，用于构建 prefix_chunk_cu_seq_lens |
| `common.py` — `alloc_for_decode` | r2t/rpi/sl 全部 CPU 侧索引 | decode 热路径，每步用 `req_to_token[req_pool_indices, seq_lens-1]` 做二维花式索引。Zeus 的 `aten::index` 不支持多维 tensor 索引 |
| `overlap_utils.py` — `_resolve_future_token_ids` | CPU 侧 where+clamp+索引 | `torch.where` + `torch.clamp` + 负数花式索引组合，三个算子都需要 fallback |
| `overlap_utils.py` — `FutureMap` | `torch.arange` CPU 创建后传回 | `torch.arange(..., device="zeus")` 不支持 |
| `scheduler.py` — `future_indices` 取反 | `(-x.cpu()).to(device)` | `aten::neg`（int tensor）未实现 |

### 4.7 模型加载器 Zeus 适配

**文件**：`model_loader/loader.py`（+107 行）

#### 修改原因

Zeus NPU 的 GEMM 高性能路径要求权重存放在 **LocalMem**（片上近存，带宽远高于 GDG 全局显存），且布局必须是 **(K, N) 转置格式**。原版 `load_weights` 把权重加载到普通 GDG 中，布局是标准的 (N, K)（`nn.Linear` 的 `weight` shape = `(out_features, in_features)`）。

如果不做 LocalMem 打包，所有 GEMM 都走 GDG 路径，带宽利用率不到 LocalMem 的 1/4。

#### 逐改动点原因

| 改动点 | 为什么要改 |
|--------|------------|
| `_tracking_iter(weights)` 包装权重迭代器 | 在 `load_weights` 遍历 safetensors 权重时，标记 `lm_head.weight` 是否出现在 checkpoint 中。原因：很多模型（如 Qwen、Llama）使用 `tie_word_embeddings=True`，checkpoint 里没有单独的 `lm_head.weight`，此时 lm_head 参数保持随机初始化值。必须检测这种情况，再决定是否从 embed_tokens 拷贝权重 |
| `_zeus_init_lm_head_from_embed()` | 当 checkpoint 是 tied 但 Zeus 适配层将 `tie_word_embeddings` override 为 False 时（因为 tied weight 在 LocalMem 打包时有额外限制），lm_head 的权重是垃圾值。此函数从 embed_tokens.weight 拷贝初始化 lm_head，确保 logits 计算正确 |
| 使用 `_zeus_lm_head_loaded` flag 替代 `torch.equal()` 检测 | **commit `e7b145c6` → `1f9cdc56` 迭代**：最初用 `torch.equal(lm_head.weight, embed.weight)` 判断是否同源，但 `aten::equal` 在 Zeus 上触发 fallback（需要把两个大权重矩阵拷到 CPU 逐元素比较），开销巨大。改为在遍历权重时自己记录 flag，零开销 |
| `pack_weights(model, target_modules=targets)` | 遍历模型所有 LinearBase/ParallelLMHead 模块，将 weight 转置 (N,K) → (K,N) 并搬入 LocalMem。使用 `_GEMM_TRANSPOSE_PARAMS` 注册表告知 `pack_weights` 哪些 (module_class, attr_name) 组合需要转置 |
| tie_word_embeddings 分支处理 | **tie=True 时**：embed_tokens 和 lm_head 是同一个 `VocabParallelEmbedding` 对象，必须把 VPE 也 pack 到 LocalMem（否则 lm_head 的 GEMM 没有 LocalMem 权重），代价是 embedding lookup 需要额外的 `to_gdg` 转换。**tie=False 时**：embed_tokens 保持在 GDG（embedding lookup 高效），只 pack ParallelLMHead |

### 4.8 Logits Processor Zeus 路径

**文件**：`logits_processor.py`（+50/−24）

#### 修改原因与逐点说明

| 改动点 | 原版代码 | 问题 | Zeus 改法 | 为什么这样改 |
|--------|---------|------|-----------|-------------|
| Prefill hidden_states 裁剪 | `pruned_states = hidden_states[last_index]` | `aten::index`（花式索引）在 Zeus 上未实现，会触发整个 `(seq_len, hidden_dim)` tensor 的 D2H fallback | 在 CPU 计算 `last_index`（tensor 很小，= batch_size），然后逐条 `pruned_states[i].copy_(hidden_states[idx])` | `copy_` 是逐行的连续内存拷贝（Zeus 支持），避免了花式索引的 fallback。CPU 侧只计算 index list（几个 int），而非搬动整个 hidden_states |
| LM head 矩阵乘法 | `F.linear(hidden, lm_head.weight)` | `F.linear` 内部调用 `hidden @ weight.T`，但 Zeus 的 weight 已在 loader 中转置并打包到 LocalMem (K,N)，再 `.T` 回 (N,K) 会破坏 LocalMem 布局导致结果错误 | `torch.mm(h, lm_head.weight)` 直接乘 | weight 已是 (K,N) = (hidden_dim, vocab_size)，`mm(h, weight)` = `(bs, hidden_dim) × (hidden_dim, vocab_size)` = `(bs, vocab_size)`，语义正确 |
| dtype 对齐 | 无 | lm_head.weight 在 LocalMem 可能是 fp16，hidden_states 可能是 bf16 | `h = hidden_states.to(lm_head.weight.dtype)` | Zeus GEMM kernel 要求两侧 dtype 一致 |

---

## 五、Zeus 适配已知限制与 Gap（来自 docs/zeus_gap_analysis.md）

### P0 — 阻塞更多模型推理

| 项目 | 说明 | 影响模型 |
|------|------|---------|
| 滑动窗口注意力(SWA) | ZeusAttnBackend 未处理 sliding_window_size | Mistral, Gemma, Qwen2-VL |
| GeluAndMul 算子 | 无 `forward_zeus` 实现 | Phi-1/2/3, GPT-NeoX, StarCoder |
| GemmaRMSNorm | 带 +1 偏移的变体未实现 | Gemma 系列 |
| MoE Forward | 走 native 逐专家循环，不可用于生产 | Mixtral, DeepSeek-V2/V3 |

### P1 — 生产性能关键

| 项目 | 说明 |
|------|------|
| Graph Capture/Replay | 硬编码 `disable_cuda_graph=True`，decode 无法批量重放 |
| 高频 CPU Bounce | argmax / seq_lens+=1 / req_to_token 全量拷贝 / clamp / CSR 构建 — 每步 decode 均触发 |

### 当前可用状态

- Dense transformer 推理（Llama/Qwen 系列）：✅ 功能正常
- 分页 KV cache + Zeus attention：✅ 可用
- 采样（greedy / top-k / top-p / min-p）：✅ 可用
- 权重 LocalMem 打包：✅ 可用
- sgl-kernel-zeus 已实现 14 个算子

---

## 六、开发辅助文件

| 文件 | 用途 |
|------|------|
| `zeus_dev/demo_cuda_llm.py` | CUDA 推理对照脚本，用于功能验证 |
| `zeus_dev/demo_zeus_llm.py` | Zeus 推理 demo，含显式 prompt 格式化和 greedy 解码 |
| `zeus_dev/demo_zeus_layer_compare.py` | 1579 行逐层数值对比工具（Zeus vs CUDA） |
| `zeus_dev/zeus_adaptation_dev_doc.md` | 适配开发笔记（含 LocalMem 转置坑点） |
| `zeus_dev/zeus_adaptation_report.md` | 适配进度总结报告 |

---

## 七、Dev 修改合理性审查

### 7.1 总体评价

| 维度 | 评分 (1-10) | 说明 |
|------|-------------|------|
| 功能正确性 | **7** | 能跑通 Dense LLM 推理，但部分 edge case 和性能陷阱未处理 |
| 代码质量 | **5** | CPU bounce 散布在 15+ 文件中，缺少抽象层，有 copy-paste 代码 |
| 性能 | **4** | `memory_pool.write()` 和 `alloc_for_decode` 的全矩阵 CPU bounce 是生产瓶颈 |
| 可维护性 | **4** | 44 个 `_is_zeus` 检查散布各处，未来 upstream rebase 将非常痛苦 |
| CUDA 路径安全 | **9** | 所有改动都在 `if _is_zeus:` 守卫下，不影响 CUDA 路径正确性 |
| 架构设计 | **5** | CustomOp 派发部分很好，数据管理层缺少 strategy 抽象 |

**综合评分：5.5/10**

### 7.2 必要的修改（无法省略）

| 文件 | 评判 | 理由 |
|------|------|------|
| `device_config.py` | ✅ 必要且干净 | 一行改动，不加则无法启动 |
| `custom_op.py` | ✅ 必要且干净 | 利用已有 CustomOp 分发机制，设计模式正确 |
| `activation.py` | ✅ 必要且干净 | 干净的 `forward_zeus` 覆盖，调用 sgl_kernel_zeus |
| `layernorm.py` | ✅ 必要且干净 | 同上，rmsnorm/fused_add_rmsnorm kernel 适配 |
| `attention_registry.py` | ✅ 必要且干净 | 注册 zeus backend，不加则 KeyError |
| `zeus_backend.py` | ✅ 必要 | Zeus 有专用硬件 attention 单元，必须独立后端 |
| `zeus_memory_pool.py` | ✅ 必要 | Zeus tiled 布局与 MHA flat 布局不兼容 |
| `zeus_allocator.py` | ✅ 必要 | 原版 allocator 在 Zeus 上每步 6+ 次隐式拷贝 |
| `unquant.py` | ✅ 必要 | LocalMem GEMM dispatch，避免 F.linear 隐式转置 |
| `model_runner.py` | ✅ 必要且结构合理 | Zeus KV pool / allocator / zecl 初始化 |
| `utils/common.py` | ✅ 必要 | 设备检测和查询函数，无此则所有设备逻辑走错 |

### 7.3 必要但实现有问题的修改

| 文件 | 评判 | 具体问题 | 改进建议 |
|------|------|---------|---------|
| `logits_processor.py` | ⚠️ 实现差 | 逐元素 `copy_` Python 循环（O(batch_size)），大 batch 时性能极差 | 用 `torch.index_select` 一次完成；或 `.cpu()[indices].to(device)` 做一次 batch 操作。另外 `torch.mm(h, weight)` 缺少 shape 断言，weight 未被 pack 时会静默产生错误结果 |
| `schedule_batch.py` | ⚠️ 实现差 | 4 个独立的 `.cpu()[idx].to(device)` 操作未合并；`seq_lens += 1` 每步 bounce 一次 | **根本解法**：schedule metadata（seq_lens、req_pool_indices 等小 tensor）在 Zeus 上直接**保持在 CPU**，仅在 attention kernel 需要时搬到 device。消除所有管理层 bounce |
| `memory_pool.py` — `write()` | ⚠️ **性能杀手** | 每次调用**复制整个 `req_to_token` 矩阵到 CPU 再复制回来**。典型配置 (max_reqs=4096, max_seq_len=8192) = 128MB round-trip，**每个 decode step 至少触发一次** | 实现 `sgl_kernel_zeus.index_put_` kernel；或将 `req_to_token` 保持在 CPU |
| `common.py` — `alloc_for_decode` | ⚠️ 浪费 | `req_to_token_pool.req_to_token.cpu()` 每步复制整个矩阵；`.cpu().clone()` 的 `.clone()` 多余 | 只复制需要的行：`req_to_token[rpi_cpu]` 而非整个矩阵 |
| `loader.py` | ⚠️ 复杂度高 | `_zeus_lm_head_loaded` 通过 monkey-patch 到 model 对象（反模式）；`'lm_head.weight' in name` 字符串匹配可能误匹配 `shared_lm_head.weight`；直接修改 `_GEMM_TRANSPOSE_PARAMS` 全局变量有副作用风险 | 用返回值/context 对象替代 monkey-patch；用 `name.endswith('lm_head.weight')` 精确匹配 |
| `server_args.py` | ⚠️ 过于死板 | `page_size = 128` **强制覆盖**用户设置，而非仅在默认值时设置；`disable_cuda_graph = True` 无条件覆盖 | 改为 `if self.page_size is None: self.page_size = 128` + `assert self.page_size % 128 == 0`；cuda_graph 同理 |
| `rotary_embedding.py` | ⚠️ 代码重复 | `RotaryEmbedding` 和 `DeepseekScalingRotaryEmbedding` 的 `forward_zeus` **完全重复**（~30 行 copy-paste） | 提取到基类或共用函数 |

### 7.4 值得商榷的修改

| 文件 | 评判 | 问题 |
|------|------|------|
| `scheduler.py` — `neg()` bounce | ❓ 可能不必要 | `aten::neg`（int tensor 取反）真的需要 CPU bounce 吗？这是最基本的一元运算。如果连 neg 都不支持，问题应该在 torch_zeus 侧修复 |
| `overlap_utils.py` — `@torch.compile` 内分支 | ❓ 设计有风险 | 在 `@torch.compile` 装饰的函数内部加 `if _is_zeus:` 运行时分支。虽然 `_is_zeus` 是模块级常量，compile 应该能在 trace 时消除死分支，但**依赖 compiler 行为不够稳健**。应在外部分发 |
| `sampler.py` — greedy `argmax` bounce | ❓ 代价偏高 | `torch.argmax(logits.cpu(), -1).to(device)` 把整个 (bs, vocab_size) 的 logits 矩阵搬到 CPU。vocab_size=128K, bs=64 时 = 32MB round-trip。`argmax` 是如此基础的算子，应该推动 torch_zeus 原生实现 |
| `sampler.py` — 缺少 `sampling_seed` | ❓ 功能缺失 | `zeus_sampling_from_logits` 没有传递 `sampling_seed` 参数，导致 Zeus 路径**不支持确定性采样**，影响可复现性 |

### 7.5 系统性问题

#### 问题一：CPU Bounce 未抽象化（最严重）

~31 个 `.cpu()...to(device)` 调用散布在 15+ 文件中，同样的模式反复出现：

```python
# 模式 A: 一元运算 bounce       — scheduler.py, schedule_batch.py, forward_batch_info.py
result = (tensor.cpu() OP).to(device)

# 模式 B: 索引 bounce           — schedule_batch.py, common.py, memory_pool.py
result = tensor.cpu()[index.cpu()].to(device)

# 模式 C: cat bounce            — radix_cache.py, memory_pool.py, forward_batch_info.py
result = torch.cat([a.cpu(), b.cpu()]).to(device)
```

**建议**：创建 `python/sglang/srt/utils/zeus_ops.py`，集中提供 `zeus_cat()`, `zeus_index()`, `zeus_cumsum()`, `zeus_neg()` 等工具函数。`radix_cache.py` 的 `_cat_zeus()` 是唯一做了正确抽象的。

#### 问题二：管理层 Metadata 应常驻 CPU

`schedule_batch` / `memory_pool` / `forward_batch_info` 中的 metadata tensor（seq_lens, req_pool_indices, kv_indices 等）都是小尺寸（batch_size 级别），在 Zeus 上反复 bounce 毫无必要。更好的设计是：**这些 tensor 在 Zeus 场景下始终保持在 CPU**，仅在喂给 attention/sampling kernel 时一次性搬到 device。这能消除约 20 个 bounce 点。

#### 问题三：Upstream Rebase 风险

44 个 `_is_zeus` 检查散布各处，main 上游已推进 2897 个 commit，这些文件大概率都有变更。Rebase 时每个文件都可能冲突。集中化的抽象层不仅提升代码质量，也能大幅减少 merge conflict。

### 7.6 结论

| 分类 | 文件数 | 占比 |
|------|--------|------|
| ✅ 必要且实现良好 | 11 | 37% |
| ⚠️ 必要但实现需改进 | 7 | 23% |
| ❓ 值得商榷 | 3 | 10% |
| 📝 文档/脚本（不影响运行时） | 9 | 30% |

**没有发现完全不必要的修改**——每个代码变更都有真实的技术原因（Zeus 缺失 ATen 算子、LocalMem 布局约束、硬件 attention kernel 需求等）。但约 **1/3 的修改在实现质量上有明显改进空间**，主要集中在：
1. CPU bounce 模式缺少抽象层，导致代码散布且不可维护
2. 两处全矩阵 CPU round-trip 是生产性能瓶颈（`memory_pool.write()` 和 `alloc_for_decode`）
3. 几处缺少防御性断言（logits_processor 的 weight shape、server_args 的 page_size）

**优先改进建议**（按影响排序）：
1. 🔴 **P0**：修复 `memory_pool.write()` 全矩阵拷贝 → 实现 `index_put_` kernel 或 CPU-resident metadata
2. 🔴 **P0**：修复 `alloc_for_decode` 全矩阵拷贝 → 只拷贝需要的行
3. 🟡 **P1**：创建 `zeus_ops.py` 集中工具层，消除分散的 bounce 代码
4. 🟡 **P1**：logits_processor 的逐元素循环改为 batch 操作
5. 🟢 **P2**：推动 torch_zeus 实现 `argmax`、`neg` 等基础 ATen 算子，消除不必要的 bounce
6. 🟢 **P2**：消除 `rotary_embedding.py` 的 copy-paste forward_zeus

---

## 八、合并建议

1. CPU bounce 模式是临时方案，长期应推动 Zeus ATen kernel 的原生实现（argmax、cat、index_put_ 等）。
2. SWA / MoE / Graph Capture 是扩大模型覆盖面和提升生产性能的关键缺口。
3. 建议在合入前完成 7.6 中 P0 级别的改进，P1 可作为后续迭代。
