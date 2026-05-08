# SGLang Zeus Adaptation Report

> Generated: 2026-03-17 | Base: SGLang main branch | Target: Zeus NPU via torch_zeus

## Overview

Total changes: **23 files modified**, ~901 lines added, ~134 lines removed, **3 new files** created.

All modifications serve one purpose: make SGLang run on Zeus NPU hardware via the `torch_zeus` PrivateUse1 backend. Changes fall into 6 categories:

| Category | Files | Description |
|---|---|---|
| Device Detection | 4 | Register Zeus as a valid device throughout SGLang |
| Operator Dispatch | 7 | Route compute kernels to `sgl_kernel_zeus` |
| Attention Backend | 3 (1 new) | Paged attention for Zeus |
| Sampling Backend | 2 | Fused sampling kernel + greedy argmax |
| Memory Management | 5 (2 new) | KV pool, page allocator, CPU-side bookkeeping |
| Model Loading | 2 | LocalMem weight packing + tied embedding handling |
| ATen Workarounds | 8 | CPU fallback for missing Zeus ATen ops |

---

## 1. Device Detection & Registration

### `python/sglang/srt/utils/common.py`
- Added `is_zeus()` detection (lru-cached, checks `torch.zeus.is_available()`)
- Zeus branches in: `get_available_gpu_memory`, `get_device_memory_capacity`, `get_device`, `get_device_count`, `get_device_core_count`, `get_device_capability`
- Excluded Zeus from `support_triton()` (Zeus has no Triton support)
- Guarded `set_cuda_arch()` to skip on non-CUDA

### `python/sglang/srt/configs/device_config.py`
- Added `"zeus"` to allowed device type list

### `python/sglang/srt/server_args.py`
- Added `"zeus"` to `ATTENTION_BACKEND_CHOICES` and sampling backend choices
- Added `_handle_zeus_backends()`: sets `attention_backend="zeus"`, `disable_cuda_graph=True`, `page_size=128`
- Sampling backend defaults to `"zeus"` when on Zeus device

### `python/sglang/srt/custom_op.py`
- Added `forward_zeus()` method (defaults to `forward_native`)
- Modified `dispatch_forward()`: Zeus checked before CUDA in dispatch chain

---

## 2. Operator Dispatch (sgl_kernel_zeus)

Each layer registers a `forward_zeus()` override that calls the corresponding `sgl_kernel_zeus` custom op.

### `python/sglang/srt/layers/activation.py`
- `SiluAndMul.forward_zeus` → `sgl_kernel_zeus.silu_and_mul`

### `python/sglang/srt/layers/layernorm.py`
- `RMSNorm.forward_zeus` → `sgl_kernel_zeus.rmsnorm` / `sgl_kernel_zeus.fused_add_rmsnorm`

### `python/sglang/srt/layers/rotary_embedding.py`
- `RotaryEmbedding.forward_zeus` → `sgl_kernel_zeus.rotary_embedding`
- `DeepseekScalingRotaryEmbedding.forward_zeus` → same kernel
- `inv_freq` initialized on CPU for Zeus (hardware constraint)

### `python/sglang/srt/layers/vocab_parallel_embedding.py`
- `VocabParallelEmbedding.forward`: when weight is in LocalMem, calls `sgl_kernel_zeus.embedding` (column-gather from transposed weight)

### `python/sglang/srt/layers/quantization/unquant.py`
- `UnquantizedLinearMethod.apply`: when weight is in LocalMem, uses `torch.mm` / `torch.addmm` directly (weight is `(K,N)` in LocalMem, bypasses `F.linear` transpose)

### `python/sglang/srt/layers/logits_processor.py`
- **Prefill pruning** (~30 lines): replaces `hidden_states[last_index]` (fancy indexing) with CPU index computation + per-element `copy_` to avoid ATen index fallback
- **Logits computation**: `torch.mm(h, lm_head.weight)` for Zeus (weight is `(K,N)` in LocalMem)

---

## 3. Attention Backend

### `python/sglang/srt/layers/attention/attention_registry.py`
- Registered `"zeus"` → `ZeusAttnBackend`

### `python/sglang/srt/layers/attention/zeus_backend.py` *(new)*
~195 lines. Complete attention backend:
- `ZeusAttnMetadata`: CSR-format indices (kv_indptr, kv_indices, qo_indptr)
- `init_forward_metadata`: builds metadata on CPU, moves to device
- `forward_extend` → `sgl_kernel_zeus.extend_attention`
- `forward_decode` → `sgl_kernel_zeus.decode_attention`

---

## 4. Sampling Backend

### `python/sglang/srt/layers/sampler.py`
- **Greedy path**: `torch.argmax` offloaded to CPU (Zeus lacks native argmax)
- **Non-greedy path**: `"zeus"` backend branch using `zeus_sampling_from_logits` — fused temp-div + softmax + top-k/p/min-p + sample kernel from `sgl_kernel_zeus`

---

## 5. Memory Management

### `python/sglang/srt/mem_cache/zeus_memory_pool.py` *(new)*
~123 lines. `ZeusTokenToKVPool(MHATokenToKVPool)`:
- Paged buffer layout: `[num_pages, num_kv_heads, page_size, head_dim]` per layer
- `set_kv_buffer` → `sgl_kernel_zeus.store_kv_cache`

### `python/sglang/srt/mem_cache/zeus_allocator.py` *(new)*
~213 lines. `ZeusPagedTokenToKVPoolAllocator`:
- All internal state (`free_pages`, `release_pages`) on CPU
- Allocation outputs moved to Zeus device only at return

### `python/sglang/srt/mem_cache/memory_pool.py`
- `ReqToTokenPool.write`: Zeus path does `index_put` on CPU (see Performance Notes)
- `MHATokenToKVPool._create_buffers`: data_ptrs built on CPU

### `python/sglang/srt/mem_cache/radix_cache.py`
- Added `_cat_zeus` helper: cat on CPU → move back to device

### `python/sglang/srt/mem_cache/common.py`
- Added `_get_last_loc_cpu`: CPU implementation of page-last-location lookup
- `alloc_for_decode`: Zeus path does indexing/arithmetic on CPU

---

## 6. Model Loading

### `python/sglang/srt/model_loader/loader.py`
~100 lines added:
- **Weight tracking**: wraps weight iterator to detect if `lm_head.weight` exists in checkpoint
- **`_zeus_init_lm_head_from_embed()`**: when `tie_word_embeddings` is overridden to `False` but checkpoint has no separate `lm_head.weight`, copies `embed_tokens.weight` → `lm_head.weight`
- **`pack_weights` integration**: after `load_weights`, converts `LinearBase` and `ParallelLMHead` weights to LocalMem (Zeus tiled memory) via `torch_zeus.zeus.pack_weights`
- **Tied embedding logic**: when `tie_word_embeddings=True`, also packs `VocabParallelEmbedding` (same object as lm_head); when `False`, only packs `ParallelLMHead` (embed_tokens stays in GDG for fast gather)

### `python/sglang/srt/model_executor/model_runner.py`
- Distributed backend: `"zecl"` for Zeus
- KV pool init: uses `ZeusTokenToKVPool`
- Allocator init: uses `ZeusPagedTokenToKVPoolAllocator`

---

## 7. ATen Workarounds (CPU Fallback Pattern)

Zeus NPU currently lacks native implementations for several ATen ops (`cumsum`, `sub`, `arange`, `clamp`, `where`, `index_put`, `argmax`). The workaround pattern is:

```python
if _is_zeus:
    result = op(tensor.cpu(), ...).to(device)
else:
    result = op(tensor, ...)
```

Files using this pattern:

| File | Ops Worked Around |
|---|---|
| `forward_batch_info.py` | cumsum, zeros, cat, clamp, arange, sub (6 blocks) |
| `schedule_batch.py` | add(+1), indexing, cat |
| `overlap_utils.py` | where, clamp, arange |
| `scheduler.py` | negation |
| `common.py` | where, indexing |
| `memory_pool.py` | index_put, cat |
| `radix_cache.py` | cat |
| `model_runner.py` | sub(-1) |
| `logits_processor.py` | cumsum, sub, index |
| `sampler.py` | argmax |

---

## 8. Demo Script

### `demo_zeus_llm.py`
- Runs Qwen2.5-0.5B-Instruct on Zeus
- Sets `json_model_override_args='{"tie_word_embeddings": false}'` so embed_tokens stays in GDG (fast gather) and lm_head goes to LocalMem (fast GEMM)

---

## Performance Notes

1. **`ReqToTokenPool.write`** (memory_pool.py): copies entire `req_to_token` tensor CPU↔device on every decode step. High-frequency hot path — candidate for a Zeus `index_put` kernel.

2. **`logits_processor.py` prefill pruning**: per-element for-loop with `copy_`. Correct but O(batch_size) kernel launches. Candidate for a batched gather kernel.

3. **`sampler.py` greedy argmax**: offloaded to CPU. Adds H2D+D2H latency per decode step. Candidate for a Zeus argmax kernel.

4. **Embedding to_gdg overhead** (when `tie_word_embeddings=True`): each forward pass converts 896×151936 tiled→linear. Mitigated by the C++ linear cache (D2D memcpy after first call). Eliminated entirely when `tie_word_embeddings=False`.

---