# SGLang Zeus Adaptation — Gap Analysis

> Generated: 2026-03-17 | Current state: Qwen2.5-0.5B demo fully working (zero fallback, zero to_gdg)

## Current Working State

- Dense transformer inference (Llama/Qwen family): fully functional
- Paged KV cache with Zeus attention backend: working
- Sampling (greedy / top-k / top-p / min-p): working
- Weight packing to LocalMem for GEMM: working
- Tied / untied embedding handling: working
- sgl-kernel-zeus: 14 ops (rmsnorm, silu_and_mul, rotary_embedding, embedding, attention, sampling, store_kv_cache)

---

## P0 — Blocks Basic Inference on More Models

### 1. Sliding Window Attention (SWA)

`ZeusAttnBackend` 没有处理 `sliding_window_size`。Mistral、Gemma、Qwen2-VL 等使用 SWA 的模型会**静默产生错误结果**（KV cache 会无限增长，attention 计算范围错误）。

**需要做**：在 `init_forward_metadata` 中裁剪 kv_indices 到 window 范围内；`sgl_kernel_zeus.extend_attention` / `decode_attention` 可能需要 window_size 参数。

**影响模型**：Mistral, Gemma, Gemma2, Qwen2-VL, 所有 hybrid-SWA 架构

### 2. GeluAndMul 算子

GPT-NeoX、Phi、StarCoder 等模型使用 `GeluAndMul` 激活函数，当前没有 `forward_zeus`，会 fallback 到 `forward_native`（纯 PyTorch 实现，可用但慢）。

**需要做**：在 sgl-kernel-zeus 中实现 `gelu_and_mul` kernel，添加 `forward_zeus` override。

**影响模型**：Phi-1/2/3, GPT-NeoX, StarCoder, CodeLlama

### 3. GemmaRMSNorm

Gemma 系列模型使用带 `+1` 偏移的 RMSNorm 变体。当前 fallback 到 `forward_native`，可用但每层多一次开销。

**需要做**：实现 `gemma_rmsnorm` / `gemma_fused_add_rmsnorm` kernel。

**影响模型**：Gemma, Gemma2, Gemma3

### 4. MoE Forward 路径

当前 MoE 模型（Mixtral, DeepSeek-MoE, Qwen-MoE）走 `moe_forward_native`：逐 expert 循环做 matmul，性能比 fused kernel 慢 10-100x。实际不可用于生产。

**需要做**：
- 短期：实现 `moe_align_block_size` + grouped GEMM kernel
- 长期：fused MoE kernel（topk gating + expert dispatch + GEMM）

**影响模型**：Mixtral, DeepSeek-V2/V3, Qwen-MoE, DBRX

---

## P1 — 生产性能关键

### 5. Graph Capture / Replay

`disable_cuda_graph=True` 是硬编码的。每次 decode step 都付出完整的 kernel launch 开销，无法 batch replay。这是 decode 吞吐量的最大瓶颈之一。

**需要做**：
- `torch_zeus` 侧：让 `zertGraphLaunch` 真正工作（or 等效机制）
- SGLang 侧：在 `ZeusAttnBackend` 实现 `init_cuda_graph_state` / `capture` / `replay` 方法

### 6. 高频 CPU Bounce（Decode 热路径）

每次 decode step 都执行的 CPU 往返操作：

| 操作 | 来源文件 | 影响 |
|------|---------|------|
| `argmax` on CPU | `sampler.py:104` | 每个 greedy token |
| `seq_lens += 1` on CPU | `schedule_batch.py:1776` | 每步 |
| `req_to_token` 全量 CPU↔device 拷贝 | `memory_pool.py:101-116` | 每步每请求 |
| `clamp(seq_lens-1, min=0)` on CPU | `forward_batch_info.py:1276` | 每步 |
| `alloc_for_decode` 索引 on CPU | `common.py:461-486` | 每步 |
| Attention metadata CSR 构建 on CPU | `zeus_backend.py:56-58` | 每步 |

**需要做**：按优先级实现 Zeus ATen ops：
1. `aten::argmax` — 消除 greedy decode 的 D2H+H2D
2. `aten::add_.Scalar` (int tensors) — 消除 seq_lens 每步 bounce
3. `aten::index_put_` (tensor indices) — 消除 req_to_token 全量拷贝
4. `aten::cumsum` — 消除 metadata 构建的多次 bounce
5. `aten::clamp` — 消除 clamp_position bounce
6. `aten::arange` (device) — 消除 arange 分配 bounce

### 7. Speculative Decoding

`ZeusAttnBackend` 没有处理 `spec_info` / `speculative_step_id`。sgl-kernel-zeus 也缺少 tree verification kernels。

**需要做**：
- Attention backend 支持 spec metadata
- 实现 `build_tree_kernel_efficient`, `verify_tree_greedy` 等 spec kernels

### 8. 量化支持

当前只有 `UnquantizedLinearMethod` 在 Zeus 上工作。所有量化方案（FP8, INT8, AWQ, GPTQ, Marlin, GGUF）都需要 CUDA-specific kernel。

**需要做**（按需求优先级）：
- FP8 GEMM（`fp8_scaled_mm`）— 最常用的量化方案
- INT8 GEMM（`int8_scaled_mm`）
- 其他按需

---

## P2 — 功能缺口

### 9. forward_mixed (混合 prefill + decode)

`ZeusAttnBackend` 缺少 `forward_mixed`，不影响基础功能但影响 continuous batching 的 overlap 调度效率。

### 10. Logit Softcap

Gemma2 在 attention 中使用 logit softcap（`tanh(logit / softcap_value) * softcap_value`）。当前未处理。

### 11. 非 Decode 热路径的 CPU Bounce

`filter_batch`, `merge_batch`, radix cache `torch.cat`, overlap utils `torch.where` 等。频率低于 decode 热路径，但在高并发场景可能成为瓶颈。

### 12. Mamba / Jamba 模型

需要 `causal_conv1d_fwd` / `causal_conv1d_update` kernel，完全 CUDA-only。

### 13. Grammar Enforcement

`apply_token_bitmask_inplace_cuda` — 用于 constrained decoding。当前无 Zeus 实现。

---

## P3 — Nice to Have

### 14. DualChunkRotaryEmbedding

`__init__` 调用 `torch.cuda.current_device()`，在 Zeus 上会 crash。仅影响 DCA 架构（LongCAT）。

### 15. Custom AllReduce

NCCL-style ring/tree allreduce 优化。当前走标准 ZECL 集合通信。

### 16. KV Cache Transfer

`transfer_kv_*` ops 用于 disaggregated inference。

---

## 按工作量排序的建议路线

### 短期（扩大模型覆盖）
1. 实现 `GeluAndMul` kernel → 解锁 Phi/StarCoder 系列
2. 实现 SWA in attention backend → 解锁 Mistral/Gemma
3. 实现 `GemmaRMSNorm` kernel → 解锁 Gemma 系列

### 中期（生产性能）
4. 注册核心 ATen ops（argmax, add_.Scalar, cumsum, clamp, arange）→ 消除 decode 热路径 CPU bounce
5. 实现 index_put_ → 消除 req_to_token 全量拷贝
6. Graph capture/replay → decode 吞吐量翻倍

### 长期（完整生态）
7. MoE fused kernels → DeepSeek/Mixtral 可用
8. FP8/INT8 量化 → 显存减半
9. Speculative decoding → 延迟优化
10. forward_mixed → continuous batching 优化

---

## sgl-kernel-zeus vs sgl-kernel 算子对比

| 类别 | sgl-kernel-zeus 已有 | sgl-kernel 还需要的关键算子 |
|------|---------------------|---------------------------|
| Elementwise | rmsnorm, fused_add_rmsnorm, silu_and_mul, rotary_embedding | gelu_and_mul, gemma_rmsnorm, gemma_fused_add_rmsnorm |
| Embedding | embedding | — |
| Attention | extend_attention, decode_attention | sliding window, logit softcap, MLA decode |
| Memory | store_kv_cache | — |
| Sampling | 7 ops (top-k/p/min-p, sampling_from_logits) | argmax (via ATen) |
| MoE | — | moe_align_block_size, grouped GEMM, topk_softmax |
| Quant | — | fp8_scaled_mm, int8_scaled_mm |
| Speculative | — | tree verify, reconstruct indices |
