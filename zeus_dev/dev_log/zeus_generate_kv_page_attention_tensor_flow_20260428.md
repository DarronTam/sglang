# Zeus `llm.generate` 到 KV Page Attention Tensor Shape/Dtype 推理

> 文档版本：2026-05-08③（05-08③：`aten::index.Tensor` 确认可用，`last_loc` 改为 device 直读，CPU mirror 依赖已消除）

本文从 `zeus_dev/demo_zeus_llm.py` 中的 `llm.generate(...)` 开始，梳理 SGLang Zeus 路径里一次文本生成请求如何进入 prefill/extend、decode，并最终构造 KV page attention 所需 metadata。重点关注 tensor 的 shape、dtype，以及 runtime 侧需要与 host 实现对齐的数据类型约定。

示例入口配置：

```python
llm = sgl.Engine(
    model_path="Qwen/Qwen2.5-0.5B-Instruct",
    device="zeus",
    dtype="bfloat16",
    disable_cuda_graph=True,
    disable_radix_cache=True,
)

prompts = ["<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"]
sampling_params = {"max_new_tokens": 64, "temperature": 0.0}
out = llm.generate(prompts, sampling_params)
```

因此本文默认：

- `device = zeus`
- model activation/cache dtype 以 `bfloat16` 为主
- `disable_radix_cache=True`，首轮 prompt 基本按完整 extend/prefill 处理
- `disable_cuda_graph=True`，走 eager metadata 构建路径
- 示例 batch size 为 `bs = 1`，但表格按通用 `bs` 描述

## 1. 主调用链路

`llm.generate` 后的主要链路可以理解为：

```text
demo_zeus_llm.py
  llm.generate(prompts, sampling_params)
    -> sglang.srt.entrypoints.engine.Engine.generate
      -> GenerateReqInput
      -> tokenizer_manager.generate_request(...)
      -> scheduler 接收 tokenized request
      -> ScheduleBatch.prepare_for_extend(...)
      -> ModelWorkerBatch
      -> ForwardBatch.init_new(...)
      -> attn_backend.init_forward_metadata(...)
      -> model forward
      -> RadixAttention.forward
      -> ZeusAttnBackend.forward_extend / forward_decode
      -> token_to_kv_pool.set_kv_buffer(...)
      -> sgl_kernel_zeus.store_kv_cache(...)
      -> sgl_kernel_zeus.extend_attention / decode_attention
      -> logits / sampler
      -> 下一轮 decode
```

更具体的代码路径和运行位置如下：

| 阶段 | 代码位置 | 主要职责 | 运行位置判断 |
| --- | --- | --- | --- |
| demo 入口 | `zeus_dev/demo_zeus_llm.py` | 创建 `sgl.Engine(device="zeus", dtype="bfloat16")`，调用 `llm.generate` | Python 控制流在 CPU；后续 tensor/device 由 engine 配置为 Zeus |
| 同步 generate API | `python/sglang/srt/entrypoints/engine.py::Engine.generate` | 构造 `GenerateReqInput`，调用 `tokenizer_manager.generate_request`，等待首个输出 | CPU Python 控制流 |
| 请求 tokenize/分发 | `python/sglang/srt/managers/tokenizer_manager.py::TokenizerManager.generate_request` | prompt tokenize，组装 tokenized request，发送给 scheduler | CPU 为主；token ids 仍是请求侧数据 |
| scheduler 调度 | `python/sglang/srt/managers/scheduler.py::Scheduler.handle_generate_request` / `get_new_batch_prefill` / `run_batch` | 接收请求，构造 `ScheduleBatch`，决定 extend/decode batch | CPU Python 控制流；batch tensor 会逐步搬到 Zeus |
| extend batch 准备 | `python/sglang/srt/managers/schedule_batch.py::ScheduleBatch.prepare_for_extend` | 构造 `input_ids/seq_lens/req_pool_indices/out_cache_loc`，分配 prompt KV slot | 控制流在 CPU；`input_ids/seq_lens/out_cache_loc` 等关键 tensor 在 Zeus |
| decode batch 准备 | `python/sglang/srt/managers/schedule_batch.py::ScheduleBatch.prepare_for_decode` | 使用上一轮 sampled token，给每个 request 分配新 KV slot，更新 `seq_lens` | 控制流在 CPU；主要 batch tensor 在 Zeus；`last_loc` 已直接从 device `req_to_token` 读取（05-08③）|
| worker batch 转换 | `python/sglang/srt/managers/schedule_batch.py::ScheduleBatch.get_model_worker_batch` | 从 scheduler batch 提取 model forward 所需字段 | CPU Python dataclass/对象组织；内部 tensor 保持各自 device |
| model worker | `python/sglang/srt/managers/tp_worker.py::TpModelWorker.forward_batch_generation` | 调用 `ForwardBatch.init_new`，进入 `model_runner.forward`，再做 sample | CPU 控制流；`ForwardBatch` 内模型输入 tensor 在 Zeus；sample 依赖 Zeus/torch 算子实现 |
| forward batch 初始化 | `python/sglang/srt/model_executor/forward_batch_info.py::ForwardBatch.init_new` | 生成 positions、extend lens、绑定 req/token KV pool、attention backend | CPU 控制流；`positions/extend_seq_lens/extend_prefix_lens` 等在 Zeus |
| model runner 总入口 | `python/sglang/srt/model_executor/model_runner.py::ModelRunner.forward` | 判断 graph/eager、decode/extend/split_prefill，并派发 | CPU 控制流；本 demo `disable_cuda_graph=True`，主要走 eager |
| extend forward | `python/sglang/srt/model_executor/model_runner.py::ModelRunner.forward_extend` | 初始化 attention metadata，然后调用 `self.model.forward(...)` | metadata 已 device 化：`kv_indptr/qo_indptr` 用 device `cumsum`，`kv_indices` 用 `sgl_kernel_zeus.build_kv_indices`；模型计算在 Zeus |
| decode forward | `python/sglang/srt/model_executor/model_runner.py::ModelRunner.forward_decode` | 初始化 decode metadata，然后调用 `self.model.forward(...)` | decode metadata 已 device 化：`kv_indptr` 用 device `cumsum`，`kv_indices` 用 `build_kv_indices`；模型计算在 Zeus |
| CausalLM forward | `python/sglang/srt/models/qwen2.py::Qwen2ForCausalLM.forward` | 调 `self.model(...)`，最后接 `logits_processor` | 模型 tensor 计算在 Zeus |
| Transformer forward | `python/sglang/srt/models/qwen2.py::Qwen2Model.forward` | embedding，循环 decoder layers，最后 norm | Zeus 主计算 |
| Decoder layer | `python/sglang/srt/models/qwen2.py::Qwen2DecoderLayer.forward` | RMSNorm -> self attention -> RMSNorm -> MLP | Zeus 主计算 |
| Qwen2 attention | `python/sglang/srt/models/qwen2.py::Qwen2Attention.forward` | `qkv_proj`，split Q/K/V，RoPE，调用 `RadixAttention`，再 `o_proj` | Zeus 主计算 |
| RadixAttention | `python/sglang/srt/layers/radix_attention.py::RadixAttention.forward` | reshape K/V，按 `forward_batch.attn_backend` 派发 attention | Zeus tensor path；具体 kernel 由 backend 决定 |
| Zeus metadata | `python/sglang/srt/layers/attention/zeus_backend.py::ZeusAttnBackend.init_forward_metadata` | 构建 `kv_indptr/kv_indices/qo_indptr/prefix_lens` | `kv_indptr/qo_indptr` 在 Zeus；`kv_indices` 由 `sgl_kernel_zeus.build_kv_indices` 在 Zeus 上构建 |
| Zeus extend attention | `python/sglang/srt/layers/attention/zeus_backend.py::ZeusAttnBackend.forward_extend` | 写 K/V cache，调用 `sgl_kernel_zeus.extend_attention` | Zeus kernel |
| Zeus decode attention | `python/sglang/srt/layers/attention/zeus_backend.py::ZeusAttnBackend.forward_decode` | 写本轮 K/V cache，调用 `sgl_kernel_zeus.decode_attention` | Zeus kernel |

所以这条主流程不是全部都在 Zeus 上执行，而是：

- CPU：Python API、tokenizer、scheduler、batch 决策、worker 调度等 control plane。
- Zeus：模型 forward 主计算、Q/K/V projection、norm/MLP、KV cache 写入、paged attention kernel、`cumsum/index_put/cat/build_kv_indices` 等 tensor/kernel 操作。
- 当前 CPU 路径：KV metadata 已完全 device 化；decode allocator 的 `last_loc` 已于 05-08③ 改为直接从 device `req_to_token` 读取，不再依赖 CPU mirror。

生成过程分两类 forward：

- `EXTEND/PREFILL`：处理 prompt token，把 prompt 的 K/V 写入 paged KV cache，同时为所有历史 token 构建 `kv_indptr/kv_indices`。
- `DECODE`：每轮只新增每个 request 的 1 个 token，分配新的 KV slot，写入 req-to-token 映射，再构建当前完整上下文的 paged attention metadata。

## 2. Extend / Prefill 阶段 tensor

设：

- `bs`：batch size
- `L_i`：第 `i` 个 request 当前总序列长度
- `P_i`：第 `i` 个 request prefix 长度
- `E_i = L_i - P_i`：本轮 extend token 数
- `T_ext = sum(E_i)`：本轮总 extend token 数
- `Hq`：TP 后 query head 数，代码中常见为 `layer.tp_q_head_num`
- `Hkv`：TP 后 KV head 数，代码中常见为 `layer.tp_k_head_num`
- `Dqk`：Q/K head dim，代码中常见为 `layer.qk_head_dim`
- `Dv`：V head dim，代码中常见为 `layer.v_head_dim`
- `page_size`：paged KV cache page size
- `num_pages = ceil(total_kv_slots / page_size)`

### 2.1 ScheduleBatch.prepare_for_extend

`ScheduleBatch.prepare_for_extend` 主要把 Python request 列表压成 device tensor，并调用 allocator 分配本轮新增 token 的 KV slot。

SGLang 通过 `zeus_index_dtype(device)` 函数判断当前设备：Zeus device 返回 `torch.int32`，否则返回 `torch.int64`（默认）。因此 `input_ids/seq_lens/out_cache_loc` 等用 `idx_dtype` 构造的 tensor 在 Zeus 上均为 `int32`，非 Zeus 上为 `int64`。

| Tensor / 字段 | Shape | Dtype（Zeus） | Device | 含义 |
| --- | --- | --- | --- | --- |
| `input_ids` | `[T_ext]` | **`int32`**（`zeus_index_dtype`） | `zeus` | 本轮需要送入模型的 token ids |
| `seq_lens` | `[bs]` | **`int32`**（`zeus_index_dtype`） | `zeus` | 每个 request 当前总长度 `L_i` |
| `seq_lens_cpu` | `[bs]` | `int64`（固定） | `cpu` | `seq_lens` 的 CPU mirror，调度/fallback 用；始终为 int64 |
| `orig_seq_lens` | `[bs]` | `int32`（固定） | `zeus` | extend 前原始长度 |
| `extend_seq_lens` | `[bs]` | `int32` | `zeus`（in `ForwardBatch`） | 每个 request 的 `E_i` |
| `extend_prefix_lens` | `[bs]` | `int32` | `zeus`（in `ForwardBatch`） | 每个 request 的 `P_i` |
| `req_pool_indices` | `[bs]` | **`int64`**（alloc 时用 `torch.int64` 构造） | `zeus` | request 在 req pool 中的行号；进入 metadata kernel 前需 cast 为 int32 |
| `out_cache_loc` | `[T_ext]` | **`int32`**（`zeus_index_dtype`，由 `alloc_for_extend` 返回） | `zeus` | 本轮新增 token 分配到的 KV slot |

> **注意**：`req_pool_indices` 在 `alloc_for_extend` 中以 `torch.int64` 创建（`req_pool_indices_cpu = torch.tensor(..., dtype=torch.int64)`），与 `seq_lens/out_cache_loc` 不同，**仍为 int64**。`zeus_backend.py` 的 `_build_kv_indices_device` 在调用 kernel 前会显式 cast 到 `int32`。

首轮无 radix prefix 时，通常 `P_i = 0`，`E_i = L_i`，所以 `T_ext = sum(L_i)`。

### 2.2 ForwardBatch.init_new

`ModelWorkerBatch` 进入 model worker 后会转换成 `ForwardBatch`，补齐 position、extend metadata 输入等。

| Tensor / 字段 | Shape | Dtype | Device | 含义 |
| --- | --- | --- | --- | --- |
| `positions` | `[T_ext]` | `int64` | `zeus` | 本轮输入 token 的 position ids |
| `extend_seq_lens` | `[bs]` | `int32` | `zeus` | 每个 request 本轮 query token 数 |
| `extend_prefix_lens` | `[bs]` | `int32` | `zeus` | 每个 request prefix token 数 |
| `extend_start_loc` | `[bs]` | 通常 `int32/int64` | `zeus` | 每个 request 在 flat extend token 中的起点 |

### 2.3 ZeusAttnBackend.init_forward_metadata

Zeus attention backend 会把 req-to-token 页表压成 paged attention kernel 可直接消费的 CSR-like metadata。

| Tensor | Shape | Dtype | Device | 含义 |
| --- | --- | --- | --- | --- |
| `kv_indptr` | `[bs + 1]` | `int32` | `zeus` | CSR row pointer，`kv_indptr[i+1]-kv_indptr[i]=L_i` |
| `kv_indices` | `[sum(L_i)]` | `int32` | `zeus` | 每个 request 需要读的历史 KV slot 列表 |
| `qo_indptr` | `[bs + 1]` | `int32` | `zeus` | extend query/output 的 CSR row pointer，差值为 `E_i` |
| `prefix_lens` | `[bs]` | `int32` | `zeus` | 每个 request prefix 长度 `P_i` |

关键不变量：

```text
kv_indptr[0] = 0
kv_indptr[-1] = len(kv_indices) = sum(seq_lens)
qo_indptr[0] = 0
qo_indptr[-1] = T_ext = sum(extend_seq_lens)
```

当前 eager 路径里，`kv_indptr` 的 cumsum 已在 device 上完成；`kv_indices` 由 `sgl_kernel_zeus.build_kv_indices` 从 device `req_to_token` 直接构建，不再需要 `req_to_token_cpu` ragged gather。

> **int64→int32 转换位置**：`seq_lens`（Zeus 上已为 int32）和 `req_pool_indices`（仍为 int64）在 `_build_kv_indices_device` 中均被检查并 cast 到 int32，再传给 `sgl_kernel_zeus.build_kv_indices`（kernel ABI 要求 int32）。Python API 层（`sgl_kernel_zeus/attention.py`）也做了相同的防御性 cast。

### 2.4 Attention Q/K/V 与 KV cache

进入每层 `RadixAttention.forward` 后，Zeus backend 会先写 KV cache，再调用 page attention kernel。

| Tensor | Logical Shape | Dtype | Device | 说明 |
| --- | --- | --- | --- | --- |
| `q` | `[T_ext, Hq * Dqk]` | `bfloat16` | `zeus` | 模型线性层输出，attention 前 view |
| `q_` | `[T_ext, Hq, Dqk]` | `bfloat16` | `zeus` | `q.view(...)` 后传给 kernel |
| `k` | `[T_ext, Hkv * Dqk]` | `bfloat16` | `zeus` | 本轮新增 token 的 K |
| `k_` | `[T_ext, Hkv, Dqk]` | `bfloat16` | `zeus` | 写入 paged KV cache |
| `v` | `[T_ext, Hkv * Dv]` | `bfloat16` | `zeus` | 本轮新增 token 的 V |
| `v_` | `[T_ext, Hkv, Dv]` | `bfloat16` | `zeus` | 写入 paged KV cache |
| `k_cache` | `[num_pages, Hkv, page_size, Dqk]` | store dtype，通常 `bfloat16` | `zeus` | Zeus paged/tiled K cache |
| `v_cache` | `[num_pages, Hkv, page_size, Dv]` | store dtype，通常 `bfloat16` | `zeus` | Zeus paged/tiled V cache |
| `o_` | `[T_ext, Hq, Dv]` | `bfloat16` | `zeus` | attention 输出 |

Zeus KV cache 不是普通的 `[max_tokens, Hkv, D]` flat layout，而是：

```text
[num_pages, num_kv_heads, page_size, head_dim]
```

kernel 通过 `kv_indices` 中的绝对 KV slot id 和 `page_size` 计算 page 内偏移。

## 3. Decode 阶段 tensor

首轮 extend 结束后，sampler 产出 next token。之后每轮 decode 每个 active request 通常只处理 1 个新 token。

设当前 decode 前每个 request 长度为 `L_i_old`，本轮新增后为 `L_i_new = L_i_old + 1`。

### 3.1 ScheduleBatch.prepare_for_decode

| Tensor / 字段 | Shape | Dtype（Zeus） | Device | 含义 |
| --- | --- | --- | --- | --- |
| `input_ids` | `[bs]` | **`int32`**（`zeus_index_dtype`） | `zeus` | 上一轮 sampler 产出的 token ids |
| `seq_lens` | `[bs]` | **`int32`**（`zeus_index_dtype`，由上轮保持） | `zeus` | decode 前加 1（`seq_lens = seq_lens + 1`），表示 `L_i_new` |
| `seq_lens_cpu` | `[bs]` | `int64`（固定） | `cpu` | CPU mirror，同步加 1 |
| `req_pool_indices` | `[bs]` | `int64`（alloc 时构造） | `zeus` | request pool 行号 |
| `out_cache_loc` | `[bs]` | **`int32`**（`zeus_index_dtype`，allocator 返回） | `zeus` | 本轮每个 request 新 token 的 KV slot |

当 `page_size > 1` 时，decode allocator 需要知道上一个 token 的 KV slot。05-08③ 后已直接从 device `req_to_token` 读取（`aten::index.Tensor` 在 Zeus 上已注册）：

```python
# alloc_for_decode (mem_cache/common.py) 统一路径（05-08③）：
last_loc = batch.req_to_token_pool.req_to_token[
    batch.req_pool_indices, batch.seq_lens - 1
]                                                        # device int32 2-D fancy index
seq_lens_next = batch.seq_lens + token_per_req           # device int32 + 1
```

`page_size == 1` 时不需要 `last_loc`，直接 alloc token slots。`seq_lens_next` 以及 `last_loc` 均在 device 上完成，无 CPU 参与。

### 3.2 ReqToTokenPool 写入

decode 分配完成后，会写入请求级页表：

```text
req_to_token[req_pool_indices, L_i_old] = out_cache_loc
```

相关 shape/dtype：

| Tensor | Shape | Dtype | Device | 含义 |
| --- | --- | --- | --- | --- |
| `req_to_token` | `[req_pool_size, max_context_len]` | `int32` | `zeus` | request/token_pos 到 KV slot 的映射 |
| `req_to_token_cpu` | `[req_pool_size, max_context_len]` | `int32` | `cpu` | Zeus mirror，与 device 同步；`last_loc` 已改为 device 直读（05-08③）；mirror 现仅服务 `_build_kv_indices_cpu` 回退路径 |
| `locs` | `[bs]` | **`int32`**（`= batch.seq_lens.clone()`，Zeus 上为 int32） | `zeus` | 本轮写入 token position，值为 `L_i_old`（decode 前的 seq_len） |
| `out_cache_loc` 写入时 | `[bs]` | **`int32`**（Zeus 上已是 int32，写前调用 `.to(int32)` 确保） | `zeus` | 写入页表的 KV slot |

`ReqToTokenPool.write` 在 Zeus 路径下同时更新 device `req_to_token` 和 CPU mirror `req_to_token_cpu`。`index_put` 已有 torch_zeus 实现，device 写入可以在线完成。CPU mirror 写入保留，仅服务 `_build_kv_indices_cpu` 回退路径（device kernel 不可用时）；`last_loc` 已不再依赖 mirror（05-08③）。

### 3.3 Decode metadata 与 attention

decode metadata 不需要 `qo_indptr/prefix_lens`，只需要当前完整上下文的 KV CSR：

| Tensor | Shape | Dtype | Device | 含义 |
| --- | --- | --- | --- | --- |
| `kv_indptr` | `[bs + 1]` | `int32` | `zeus` | 每个 request 历史 KV 的边界 |
| `kv_indices` | `[sum(L_i_new)]` | `int32` | `zeus` | 每个 request 所有历史 KV slot |
| `q_` | `[bs, Hq, Dqk]` | `bfloat16` | `zeus` | 本轮新 token 的 Q |
| `k_` | `[bs, Hkv, Dqk]` | `bfloat16` | `zeus` | 本轮新 token 的 K，先写 cache |
| `v_` | `[bs, Hkv, Dv]` | `bfloat16` | `zeus` | 本轮新 token 的 V，先写 cache |
| `o_` | `[bs, Hq, Dv]` | `bfloat16` | `zeus` | decode attention 输出 |

decode kernel 调用概念上是：

```text
decode_attention(
    q_, o_,
    k_cache, v_cache,
    kv_indptr, kv_indices,
    num_q_heads=Hq,
    num_kv_heads=Hkv,
    head_dim=Dqk,
    page_size=page_size,
    sm_scale=layer.scaling,
)
```

## 4. KV Page Attention 的指针/页表机制

KV page attention 里真正传给硬件/kernel 的不是“每个 request 一段连续变长 buffer 指针”，而是一套固定大 buffer + 小整数索引页表：

```text
req_id + token_pos
  -> req_to_token[req_id, token_pos]
  -> kv_slot
  -> page_id = kv_slot // page_size
  -> page_offset = kv_slot % page_size
  -> k_cache/v_cache[page_id, kv_head, page_offset, head_dim]
```

这里的核心思想是：**硬件看到的是固定 shape 的 paged KV cache buffer，变长序列由 metadata 指针数组描述**。

### 4.1 固定 buffer 与变长逻辑序列

Zeus KV cache 的物理 buffer 是预分配的固定形状：

```text
k_cache: [num_pages, Hkv, page_size, Dqk]
v_cache: [num_pages, Hkv, page_size, Dv]
```

其中：

- `num_pages` 是 allocator 根据最大 token capacity 预先算出来的固定页数。
- `page_size` 是每页固定 token 数。
- `Hkv/Dqk/Dv` 是模型结构固定参数。
- 单个 request 的上下文长度 `seq_len` 是变长的，但不会改变 `k_cache/v_cache` 的物理 shape。

也就是说，循环生成时 KV cache buffer 本身不 resize。每生成一个 token，只是 allocator 找一个新的空闲 `kv_slot`，然后把这个 slot 记录到 request 的页表里。

### 4.2 `out_cache_loc`：新 token 写到哪里

每轮 extend/decode allocator 会返回：

```text
out_cache_loc: [num_new_tokens]
```

它表示本轮新 token 对应的绝对 KV slot id。

extend 阶段：

```text
out_cache_loc.numel() = T_ext = sum(extend_seq_lens)
```

decode 阶段：

```text
out_cache_loc.numel() = bs
```

每个 block 得到本层的 K/V 后，会用同一份 `out_cache_loc` 把 K/V 写入该 layer 自己的 KV cache：

```text
set_kv_buffer(layer, out_cache_loc, k, v)
  -> store_kv_cache(k_cache[layer], v_cache[layer], out_cache_loc, k, v, page_size)
```

因此 `out_cache_loc` 可以理解成“本轮写 KV cache 的写指针数组”。它不是 CPU 指针，也不是裸地址，而是 device/kernel 可解释的 slot index。

### 4.3 `req_to_token`：request 级逻辑页表

`req_to_token` 保存每个 request 的逻辑 token 位置到物理 KV slot 的映射：

```text
req_to_token: [req_pool_size, max_context_len] int32
req_to_token[req_idx, token_pos] = kv_slot
```

当一个 request 在 decode 阶段新增 token 时，会写入：

```text
req_to_token[req_idx, old_seq_len] = out_cache_loc[batch_idx]
```

这样 request 的逻辑序列虽然是连续的 `token_pos = 0..seq_len-1`，但物理 KV slot 可以分散在不同 page 中。page attention kernel 后续不要求这些 slot 物理连续。

### 4.4 `kv_indptr/kv_indices`：kernel 消费的 CSR 指针

attention kernel 不直接按二维 `req_to_token` 遍历，而是消费压缩后的 CSR-like metadata：

```text
kv_indptr:  [bs + 1] int32
kv_indices: [sum(seq_lens)] int32
```

对第 `b` 个 request：

```text
start = kv_indptr[b]
end   = kv_indptr[b + 1]
slots = kv_indices[start:end]
```

其中 `slots` 就是该 request 当前完整上下文要读取的所有 KV slot。

例如 `bs=3`，三个 request 长度分别是 `[4, 2, 5]`：

```text
seq_lens  = [4, 2, 5]
kv_indptr = [0, 4, 6, 11]

request 0 读取 kv_indices[0:4]
request 1 读取 kv_indices[4:6]
request 2 读取 kv_indices[6:11]
```

这就是“变长”的表达方式：每个 request 的长度由 `kv_indptr[b+1]-kv_indptr[b]` 给出，硬件/kernel 只需要遍历对应区间。

### 4.5 kernel 内的 page 地址计算

`kv_indices` 中每个元素是绝对 `kv_slot`，不是 page id。kernel 内部会做：

```text
page_id     = kv_slot // page_size
page_offset = kv_slot % page_size
```

然后访问：

```text
k_cache[page_id, kv_head, page_offset, d]
v_cache[page_id, kv_head, page_offset, d]
```

如果从硬件视角看，可以把 base address + stride 展开成近似地址计算：

```text
k_addr = k_base
       + (((page_id * Hkv + kv_head) * page_size + page_offset) * Dqk + d) * sizeof(dtype)

v_addr = v_base
       + (((page_id * Hkv + kv_head) * page_size + page_offset) * Dv + d) * sizeof(dtype)
```

所以硬件不是维护一堆变长 buffer 指针，而是维护：

- 一个固定 base pointer：`k_cache/v_cache` 的起始地址。
- 固定 stride/layout：`[num_pages, Hkv, page_size, D]`。
- 变长 metadata：`kv_indptr/kv_indices`。
- 当前写入位置：`out_cache_loc`。

### 4.6 循环生成时指针如何变化

每轮 decode 的变化可以简化为：

```text
第 t 轮 decode:
  1. allocator 分配新 kv_slot，得到 out_cache_loc[b]
  2. req_to_token[req_idx[b], old_seq_len[b]] = out_cache_loc[b]
  3. seq_lens[b] += 1
  4. 根据新的 seq_lens 和 req_to_token 构建 kv_indptr/kv_indices
  5. 每个 block 用同一份 out_cache_loc 写本层 K/V cache
  6. page attention 用 kv_indptr/kv_indices 读取完整历史 KV
```

其中第 5 步对每一层都会发生，因为每层有自己的 K/V cache；第 1-4 步是本轮 forward/batch 级别的 metadata 准备，不是每层都重新分配。

### 4.7 为什么固定 size 硬件能处理变长序列

固定 size 体现在物理资源上：

```text
KV cache buffer 固定最大容量
page_size 固定
kernel block/thread tile 固定策略
```

变长体现在 metadata 上：

```text
seq_lens 决定每个 request 有多长
kv_indptr 决定每个 request 的起止边界
kv_indices 决定每个逻辑 token 实际落在哪个 KV slot
```

所以硬件处理变长不是靠动态分配不同长度的 buffer，而是靠“固定 buffer + 动态索引表”。这和 CPU 上的 CSR sparse matrix 很像：values/index buffer 是一维连续数组，row pointer 告诉 kernel 每一行长度是多少。

当前 Zeus 路径中，`req_to_token -> kv_indices` 的 ragged gather 已由 `sgl_kernel_zeus.build_kv_indices` 承接，metadata 指针链可以在 device 上闭环：

```text
allocator out_cache_loc
  -> device index_put 写 req_to_token
  -> device cumsum 生成 kv_indptr
  -> device ragged gather 生成 kv_indices
  -> page attention kernel 读取 paged KV cache
```

## 5. Runtime 侧 dtype 对齐建议

runtime 与 SGLang/Zeus kernel 对齐时，建议把类型分成两层：调度层可以保留 SGLang 当前习惯，kernel 边界必须收敛成稳定 ABI。

### 5.1 Kernel metadata 建议统一为 int32

以下 tensor 建议 runtime/kernel ABI 明确要求 `int32`：

| Tensor | 建议 dtype | 原因 |
| --- | --- | --- |
| `req_to_token` | `int32` | 保存 KV slot id，当前实现已是 int32 |
| `kv_indptr` | `int32` | CSR pointer，kernel 当前使用 int32 |
| `kv_indices` | `int32` | KV slot id 列表，kernel 当前使用 int32 |
| `qo_indptr` | `int32` | extend query/output CSR pointer |
| `prefix_lens` | `int32` | attention mask/prefix 判断只需 int32 |
| 写入 `req_to_token` 的 `out_cache_loc` | `int32` | 页表存储 int32，写入前必须 cast |

注意：`kv_indices` 表示绝对 KV slot id，不是 page id。kernel 会结合 `page_size` 自己计算：

```text
page_id = kv_slot // page_size
offset  = kv_slot % page_size
```

### 5.2 调度层 dtype：Zeus 已部分收敛为 int32

SGLang 通过 `zeus_index_dtype(device)` 在 Zeus device 上将若干 index tensor 直接构造为 `int32`，避免了下游 cast 开销；少数字段由于历史或 PyTorch indexing 原因仍保留 `int64`：

| Tensor | Zeus 实际 dtype | 非 Zeus 实际 dtype | 说明 |
| --- | --- | --- | --- |
| `input_ids` | **`int32`**（`zeus_index_dtype`） | `int64` | embedding 层会做 int32→int64 promote（如需要）|
| `seq_lens`（device） | **`int32`**（`zeus_index_dtype`） | `int64` | cumsum 时指定 `dtype=torch.int32` 输出 int32 的 `kv_indptr` |
| `seq_lens_cpu` | `int64`（固定） | `int64` | CPU mirror，始终为 int64 |
| `req_pool_indices` | **`int64`**（`alloc_for_extend` 中 `torch.tensor(..., dtype=torch.int64)` 构造） | `int64` | 进入 `build_kv_indices` 前 cast 为 int32（两处防御：`attention.py` + `zeus_backend.py`） |
| `out_cache_loc` | **`int32`**（`zeus_index_dtype`） | `int64` | 写 `req_to_token` 时已经是 int32；`store_kv_cache` 直接使用 |
| `locs`（decode write） | **`int32`**（`= seq_lens.clone()`） | `int64` | 写页表的 row index |
| `positions` | `int64`（未经 `zeus_index_dtype`） | `int64` | RoPE 输入，保持 int64 没有问题 |

也就是说，runtime 侧可以采用：

```text
scheduler/control plane: int64 is acceptable
attention/kernel metadata ABI: int32
model activation/cache: bfloat16 or configured model dtype
```

### 5.3 Activation 与 KV cache dtype

demo 中 `dtype="bfloat16"`，因此模型 attention 主路径通常应对齐为：

| Tensor 类别 | 建议 dtype |
| --- | --- |
| Q/K/V activation | `bfloat16` |
| attention output | `bfloat16` |
| paged K/V cache | `bfloat16`，或 `token_to_kv_pool.store_dtype` |
| logits | 通常由模型 head 决定，可能 bf16/fp32 |

`ZeusTokenToKVPool.set_kv_buffer` 会在写 cache 前检查 `cache_k/cache_v` dtype，不一致时 cast 到 pool dtype，并调用 contiguous。因此 runtime 最好在上游就保证 Q/K/V dtype 与 KV pool dtype 一致，避免隐式 cast 和 layout 开销。

## 6. Device / Runtime Buffer 对齐建议

`build_kv_indices` 支持后，attention metadata 的核心数据已经可以在 Zeus device 上构建。runtime 侧最重要的是把 buffer 的 shape、dtype、layout 和“值的语义”固定下来，避免出现 int64/int32、slot/page、flat/tiled layout 混用。

### 6.1 KV metadata buffer

| Buffer | Shape | Dtype | Device | 生产者 | 消费者 | 对齐要求 |
| --- | --- | --- | --- | --- | --- | --- |
| `seq_lens` | `[bs]` | 调度层常见 `int64`；kernel 边界转 `int32` | `zeus` | scheduler / `ForwardBatch` | `cumsum`、`build_kv_indices` | runtime 可保留 int64，但进入 metadata kernel 前必须可转 int32 |
| `extend_seq_lens` | `[bs]` | `int32` | `zeus` | `ForwardBatch.init_new` | `qo_indptr` cumsum | extend-only；decode 为 `None` |
| `extend_prefix_lens` | `[bs]` | `int32` | `zeus` | `ForwardBatch.init_new` | `prefix_lens` / extend attention | extend-only；值是每个 request 的 prefix token 数 |
| `kv_indptr` | `[bs + 1]` | `int32` | `zeus` | `torch.cumsum(seq_lens)` | `build_kv_indices`、attention kernel | `kv_indptr[0]=0`，`kv_indptr[-1]=sum(seq_lens)` |
| `kv_indices` | `[sum(seq_lens)]`；graph 可用 `[max_bs * max_context_len]` 预分配 buffer | `int32` | `zeus` | `sgl_kernel_zeus.build_kv_indices` | `extend_attention/decode_attention` | 元素是绝对 KV slot id，不是 page id |
| `qo_indptr` | `[bs + 1]` | `int32` | `zeus` | `torch.cumsum(extend_seq_lens)` | `extend_attention` | extend-only；`qo_indptr[-1]=sum(extend_seq_lens)=T_ext` |
| `prefix_lens` | `[bs]` | `int32` | `zeus` | `extend_prefix_lens.to(int32)` | `extend_attention` | extend-only；和 `qo_indptr` 同 batch 顺序 |

`build_kv_indices` 的 ABI 建议固定为：

```text
build_kv_indices(
    req_to_token:     int32[req_pool_size, max_context_len],
    req_pool_indices: int32[bs],
    seq_lens:         int32[bs],
    kv_indptr:        int32[bs + 1],
    kv_indices_out:   int32[sum(seq_lens)] or preallocated graph buffer,
)
```

### 6.2 Req/KV slot 页表与 allocator buffer

| Buffer | Shape | Dtype | Device | 含义 | 对齐要求 |
| --- | --- | --- | --- | --- | --- |
| `req_to_token` | `[req_pool_size, max_context_len]` | `int32` | `zeus` | request/token_pos 到绝对 KV slot 的页表 | runtime 必须按绝对 slot id 写，不要写 page id |
| `req_to_token_cpu` | 同上 | `int32` | `cpu` | 兼容/调试 mirror | metadata 已不依赖它；若 runtime 全 device 化，可逐步弱化 |
| `req_pool_indices` | `[bs]` | 调度层可 `int64`，metadata kernel 用 `int32` | `zeus` | 当前 batch 每个 request 的 `req_to_token` 行号 | 进入 `build_kv_indices` 前转 int32 |
| `out_cache_loc` | extend: `[T_ext]`；decode: `[bs]` | allocator 可 `int64`，写页表/kernel 用 `int32` | `zeus` | 本轮新 token 分配到的绝对 KV slot | 写入 `req_to_token` 和 `store_kv_cache` 前转 int32 |
| `positions` | extend: `[T_ext]`；decode: `[bs]` | `int64` | `zeus` | RoPE / position embedding 位置 | 保持 int64 即可，不属于 metadata ABI |

关键写入关系：

```text
req_to_token[req_pool_indices[b], token_pos] = out_cache_loc[token_idx]
```

extend 时 `token_idx` 遍历 flat prompt token；decode 时每个 request 通常只有一个新 token。

### 6.3 KV cache / model data buffer

| Buffer | Shape | Dtype | Device | Layout / 语义 |
| --- | --- | --- | --- | --- |
| `q` | extend `[T_ext, Hq * Dqk]`；decode `[bs, Hq * Dqk]` | model dtype，demo 为 `bfloat16` | `zeus` | attention 前 view 成 `[tokens, Hq, Dqk]` |
| `k` | extend `[T_ext, Hkv * Dqk]`；decode `[bs, Hkv * Dqk]` | model/cache dtype，demo 为 `bfloat16` | `zeus` | 写入 K cache 前 view 成 `[tokens, Hkv, Dqk]` |
| `v` | extend `[T_ext, Hkv * Dv]`；decode `[bs, Hkv * Dv]` | model/cache dtype，demo 为 `bfloat16` | `zeus` | 写入 V cache 前 view 成 `[tokens, Hkv, Dv]` |
| `k_cache` | `[num_pages, Hkv, page_size, Dqk]` | `token_to_kv_pool.store_dtype`，通常 `bfloat16` | `zeus` | Zeus paged/tiled K cache |
| `v_cache` | `[num_pages, Hkv, page_size, Dv]` | `token_to_kv_pool.store_dtype`，通常 `bfloat16` | `zeus` | Zeus paged/tiled V cache |
| `o` | extend `[T_ext, Hq * Dv]`；decode `[bs, Hq * Dv]` | model dtype，demo 为 `bfloat16` | `zeus` | attention output，后接 `o_proj` |

runtime 侧要特别避免两类错位：

- **slot/page 混淆**：`out_cache_loc`、`req_to_token`、`kv_indices` 都保存绝对 KV slot id；kernel 内再用 `page_size` 计算 `page_id/offset`。
- **flat/tiled layout 混淆**：runtime 若有自己的 KV cache buffer，必须和 Zeus kernel 约定一致，即 `[num_pages, Hkv, page_size, head_dim]`，不能传 `[max_tokens, Hkv, head_dim]`。

### 6.4 已支持算子与对应 dtype/shape

| 操作 | 当前实现 | 输入 dtype/shape | 输出 dtype/shape | 说明 |
| --- | --- | --- | --- | --- |
| 页表写入 | `aten::index_put` | indices: `req_pool_indices/locs`；values: `out_cache_loc.to(int32)` | `req_to_token[int32]` 原位更新 | 写入 device `req_to_token` |
| `kv_indptr` 构建 | `aten::cumsum` | `seq_lens[bs]`，调度层常见 int64 | `kv_indptr[bs+1] int32` | `kv_indptr[1:] = cumsum(seq_lens, dtype=int32)` |
| `qo_indptr` 构建 | `aten::cumsum` | `extend_seq_lens[bs] int32` | `qo_indptr[bs+1] int32` | extend-only |
| `kv_indices` 构建 | `sgl_kernel_zeus.build_kv_indices` | `req_to_token[int32]`、`req_pool_indices[int32]`、`seq_lens[int32]`、`kv_indptr[int32]` | `kv_indices[sum(seq_lens)] int32` | 替代 CPU ragged gather |
| KV cache 写入 | `sgl_kernel_zeus.store_kv_cache` | `loc[int32]`、`k/v bf16[tokens,Hkv,D]` | `k_cache/v_cache` 原位更新 | `loc` 对齐 `out_cache_loc.to(int32)` |
| page attention | `sgl_kernel_zeus.extend_attention/decode_attention` | Q/O/KV cache bf16；metadata int32 | `o bf16` | 消费 `kv_indptr/kv_indices` |

## 7. 当前仍涉及 CPU 的点

在只考虑 KV metadata 构建和 paged KV attention 主流程时，`kv_indptr/qo_indptr/kv_indices/prefix_lens` 已经可以在 device 上构建。当前还需要继续关注的是：

| 位置 | CPU 操作 | 确认状态 | 影响 | 可优化方向 |
| --- | --- | --- | --- | --- |
| Python control plane | tokenizer、scheduler、batch 决策 | 正常控制面 | 正常控制流，不属于无谓 tensor bounce | 保持 CPU 控制面即可 |
| decode `last_loc`（paged，`page_size > 1`） | `alloc_for_decode` 从 device `req_to_token` 直接 2-D fancy index | ✅ **已解决（05-08③）**：`aten::index.Tensor` 注册后统一 device 路径，`_is_zeus` CPU mirror 分支已删除 | 无额外 CPU bounce | — |
| `req_to_token_cpu` 写入同步 | `ReqToTokenPool.write` Zeus 分支同时写 device 和 CPU mirror | 保留，服务 `_build_kv_indices_cpu` 回退路径 | 每轮 decode 写一次 CPU mirror（代价极小） | device kernel 覆盖率稳定后可降级为 debug-only |
| `req_pool_indices` dtype | `alloc_for_extend` 以 `int64` 构造，进 kernel 前 cast | `zeus_backend._build_kv_indices_device` 和 `attention.py` 两处 cast | 一次小 cast，延迟可忽略 | 上游改用 `zeus_index_dtype` 构造 `req_pool_indices`，彻底消除 cast |

已经在 device 上的 metadata 操作：

- `torch.cumsum(seq_lens, dtype=torch.int32)` 构建 `kv_indptr`。
- `torch.cumsum(extend_seq_lens, dtype=torch.int32)` 构建 `qo_indptr`。
- `sgl_kernel_zeus.build_kv_indices` 构建 `kv_indices`。
- `req_to_token[indices] = values` 通过 `aten::index_put` device 写入。

## 8. Runtime 对齐检查清单

接 runtime 时建议逐项检查：

- `req_to_token` 和 runtime 页表是否都是 `[req_pool_size, max_context_len] int32`，值为绝对 KV slot。
- `out_cache_loc` 写入页表前是否 cast 到 `int32`。
- `kv_indptr/kv_indices/qo_indptr/prefix_lens` 是否都是 device `int32`。
- `kv_indptr[-1] == kv_indices.numel() == sum(seq_lens)`。
- extend 阶段 `qo_indptr[-1] == input_ids.numel() == out_cache_loc.numel()`。
- decode 阶段 `input_ids.numel() == out_cache_loc.numel() == bs`。
- KV cache layout 是否是 `[num_pages, Hkv, page_size, head_dim]`。
- runtime 传入 kernel 的 `kv_indices` 是否是 KV slot id，而不是 page id。
- Q/K/V dtype 是否与 `token_to_kv_pool.store_dtype` 一致，demo 下应为 `bfloat16`。
- 如果 runtime 内部使用 int64 管理 `seq_lens/req_pool_indices/out_cache_loc`，进入 kernel ABI 前是否显式转换为 int32。

## 9. 小结

从 `llm.generate` 到 KV page attention 的关键数据流是：

```text
tokenized prompt
  -> input_ids / seq_lens / req_pool_indices
  -> allocator 分配 out_cache_loc
  -> req_to_token 写入 token_pos -> kv_slot
  -> attention metadata 压缩成 kv_indptr + kv_indices
  -> Q/K/V 写入 paged KV cache
  -> page attention kernel 按 kv_indices 读取历史 KV
```

对 runtime 来说，最需要稳定对齐的是两类边界：

- metadata ABI：`int32` 的 `kv_indptr/kv_indices/qo_indptr/prefix_lens/req_to_token`，其中 `kv_indices` 由 `build_kv_indices` 在 device 上构建。
- data ABI：`bfloat16` 的 Q/K/V activation 和 `[num_pages, Hkv, page_size, head_dim]` paged KV cache。

当前 KV metadata 构建和主链路 index(read) 操作已经完成 device 化闭环（05-08③）。后续关注点仅剩：`req_pool_indices` dtype 统一（改用 `zeus_index_dtype` 消除 `int64→int32` cast）；以及 `index_get.cpp` fp32 linearize 精度上限（`pool_size × max_ctx_len > 2^{24}` 时需改用 int32/int64 累加器）。

## 10. Dtype 转换流水线（附录）

以下展示 Zeus 路径关键 tensor 从构造到 kernel 消费的完整 dtype 变化链路，供 runtime 对接参考。

### 10.1 Extend 路径

```text
prepare_for_extend (schedule_batch.py)
  input_ids      : zeus_index_dtype → int32[T_ext]   @zeus
  seq_lens       : zeus_index_dtype → int32[bs]      @zeus
  seq_lens_cpu   : int64 固定       → int64[bs]      @cpu
  req_pool_indices: int64 固定      → int64[bs]      @zeus   ← 注意：未用 zeus_index_dtype
  out_cache_loc  : zeus_index_dtype → int32[T_ext]   @zeus
  orig_seq_lens  : int32 固定       → int32[bs]      @zeus

write_cache_indices / ReqToTokenPool.write
  写 req_to_token[req_pool_indices, pos] = out_cache_loc.to(int32)
  写 req_to_token_cpu 同步

init_forward_metadata (zeus_backend.py)
  kv_indptr[1:] = cumsum(seq_lens, dtype=int32)      → int32[bs+1] @zeus
  _build_kv_indices_device:
    req_pool_indices.to(int32) if int64             # cast
    seq_lens.to(int32) if int64                     # 已是 int32，no-op
    build_kv_indices(req_to_token, rpi32, sl32, kv_indptr, kv_indices_out)
  kv_indices                                        → int32[T_kv]  @zeus
  qo_indptr[1:] = cumsum(extend_seq_lens, int32)    → int32[bs+1]  @zeus
  prefix_lens = extend_prefix_lens.to(int32)        → int32[bs]    @zeus

extend_attention kernel ABI
  q_  : bfloat16[T_ext, Hq, Dqk]
  o_  : bfloat16[T_ext, Hq, Dv]
  k_cache / v_cache : bfloat16[num_pages, Hkv, page_size, D]
  kv_indptr, kv_indices, qo_indptr, prefix_lens : int32
```

### 10.2 Decode 路径

```text
prepare_for_decode (schedule_batch.py)
  seq_lens  = seq_lens + 1                          → int32[bs]    @zeus  (继承 extend 后的 int32)
  out_cache_loc : zeus_index_dtype                  → int32[bs]    @zeus

alloc_for_decode (mem_cache/common.py) 统一路径（05-08③）
  last_loc = req_to_token[req_pool_indices, seq_lens - 1]   # device 2-D index，int32
  seq_lens_next = seq_lens + token_per_req          → int32[bs]    @zeus

ReqToTokenPool.write
  locs = seq_lens.clone()                           → int32[bs]    @zeus
  req_to_token[req_pool_indices, locs] = out_cache_loc.to(int32)
  req_to_token_cpu 同步

init_forward_metadata (zeus_backend.py)
  kv_indptr[1:] = cumsum(seq_lens, dtype=int32)    → int32[bs+1]  @zeus
  _build_kv_indices_device:
    req_pool_indices cast to int32                 # int64 → int32
    build_kv_indices(...)
  kv_indices                                       → int32[T_kv]  @zeus
  (qo_indptr, prefix_lens = None for decode)

decode_attention kernel ABI
  q_  : bfloat16[bs, Hq, Dqk]
  o_  : bfloat16[bs, Hq, Dv]
  k_cache / v_cache : bfloat16[num_pages, Hkv, page_size, D]
  kv_indptr, kv_indices : int32
```

### 10.3 Runtime 对接时需要特别关注的转换点

| 转换点 | 当前位置 | 方向 | 必要原因 |
| --- | --- | --- | --- |
| `req_pool_indices` int64→int32 | `zeus_backend._build_kv_indices_device` + `attention.py` | int64 → int32 | kernel ABI 要求 int32；SGLang 内部用 int64 构造 |
| `cumsum(seq_lens, dtype=int32)` | `zeus_backend.init_forward_metadata` | int32 in → int32 out（Zeus），int64 in → int32 out（non-Zeus） | 显式 `dtype` 参数保证输出类型 |
| `extend_prefix_lens.to(int32)` | `zeus_backend.init_forward_metadata` | int32 → int32（no-op，defensive） | 防御性 cast |
| `out_cache_loc.to(int32)` 写页表 | `alloc_for_decode` / `write_cache_indices` | int32 on Zeus → int32（no-op）；int64 on non-Zeus → int32 | `req_to_token` 要求 int32 |
