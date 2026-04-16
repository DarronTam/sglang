# Zeus 适配更新日志 — 2026-04-15

本次更新分两个阶段：

**阶段一（demo 层跑通）** —— 围绕 "让 `zeus_dev/demo_zeus_layer_compare.py`
能在纯 Zeus 无 CUDA 环境跑通"，涉及 11 个文件、净新增约 200 行。

**阶段二（server 端到端跑通）** —— 进一步让 `python -m sglang.launch_server
--device zeus` 能完整启动 Qwen2.5-0.5B 并响应 `/generate` 请求，涉及新增
`srt_zeus` extra + 6 个 runtime 补丁 + 1 份安装文档。

核心目标：

1. 让 `is_cuda()` 真正受 `SGLANG_DEVICE` 控制，阻断 CUDA 代码路径。
2. 抽出后端无关的 `device_empty_cache` / `device_synchronize` 辅助函数，
   替换代码中硬编码的 `torch.cuda.xxx()` 调用。
3. 把若干 `sgl_kernel`（仅 CUDA 构建）的顶层 import 加上 `is_zeus()` 短路，
   避免在 zeus 环境下因缺 `sgl_kernel` 直接 ImportError。
4. 重写 `demo_zeus_layer_compare.py`，在没有 CUDA 的机器上用 CPU 作为黄金参考。
5. 新增 `srt_zeus` extra + 安装指南，把 zeus 对齐到 AMD / NPU 的"独立硬件分支"
   安装 flow。
6. 清理 server 启动链上阻断端到端生成的 runtime 兼容项（分配器、KV 写入、
   attention、tied lm_head、torch.compile backend、memory capacity 查询）。

---

## 一、核心基础设施 — `python/sglang/srt/utils/common.py`

- `is_cuda()` 在返回前先读 `SGLANG_DEVICE` 环境变量：当其被显式设为
  非 `"cuda"` 值（例如 `"zeus"`）时，即使机器上有 CUDA GPU + CUDA 版 torch，
  也强制返回 `False`。这是关掉 ~80 个 `_is_cuda = is_cuda()` 分支的总开关。
- 新增两个后端无关的辅助函数：
  - `device_empty_cache(device)` — 通过 `torch.get_device_module(device)`
    取到具体后端模块，若其暴露 `empty_cache` 就调用，否则静默跳过
    （CPU 没这个 API）。
  - `device_synchronize(device)` — 同样走 device module，先找
    `synchronize`，再回退到 `device_synchronize`，以兼容
    `torch.zeus` 命名差异。

这两个函数是后面多处替换 `torch.cuda.synchronize()` / `torch.cuda.empty_cache()`
的统一入口。

## 二、Layer 层 — import 层面的 zeus 短路

以下文件都做了同一模式的改造：加入 `is_zeus` 导入 → 定义 `_is_zeus = is_zeus()`
→ 把 `sgl_kernel` 的顶层 import 或 fallback 判断纳入 zeus 分支：

- `python/sglang/srt/layers/activation.py`
  `_is_zeus` 进入“sgl-kernel 不可用即回退”判断，避免在 zeus 环境下
  误报“Fallback to other kernel libraries”。
- `python/sglang/srt/layers/layernorm.py`
  同上，layernorm 回退逻辑增加 zeus 分支。
- `python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`
  `if not (_is_npu or _is_hip or _is_zeus): from sgl_kernel import silu_and_mul`
  ——zeus 环境下完全不 import `sgl_kernel`。
- `python/sglang/srt/mem_cache/memory_pool_host.py`
  `sgl_kernel.kvcacheio` 的 import 同样对 zeus 短路。

## 三、移除硬编码 `torch.cuda.*` 调用

用新的 `device_empty_cache` / `device_synchronize` 替换：

- ~~`python/sglang/srt/model_executor/model_runner.py`~~
  **已 revert**：`torch_zeus` 已修复，`torch.zeus` 现在同时暴露
  `synchronize()` 和 `device_synchronize()`，因此原写法
  `torch.get_device_module(self.device).synchronize()` 在 zeus 上可直接工作，
  不再需要 `device_synchronize` wrapper。
- `python/sglang/srt/model_loader/loader.py`
  `RemoteInstanceModelLoader` 中两处 `torch.cuda.synchronize()` →
  `device_synchronize(device_config.device)`。注：`torch.zeus` 已暴露
  `synchronize()`，这两处也可改回
  `torch.get_device_module(device).synchronize()`，但 `device_synchronize`
  wrapper 同样正确且 RemoteInstanceModelLoader 短期不会在 zeus 上使用。
- `python/sglang/srt/models/llama.py`
  `LlamaForCausalLM.set_embed_and_head` / `set_embed` 中的
  `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` →
  `device_empty_cache(embed.device)` + `device_synchronize(embed.device)`。
- `python/sglang/srt/models/qwen2.py`
  `Qwen2ForCausalLM.set_embed_and_head` 中同样的替换。

## 四、`torchao_utils.py` 的惰性 import

- `apply_torchao_config_to_model` 中把 `if torchao_config in (None, "")` 的
  早退提前到 `from torchao.quantization import ...` 之前。
  原因：纯 zeus / CPU 环境通常没装 torchao，不应该为了“可能走 fallback 路径”
  就无条件 import。

## 五、Demo — `zeus_dev/demo_zeus_layer_compare.py`

整体思路：把原先硬写 `"cuda"` 的“黄金参考”端改成 `REF_DEVICE`，
在无 CUDA 机器上退化为 `"cpu"`。

主要改动：

- 顶部新增 `REF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`。
- 新增 `_ref_forward(layer, *args, **kwargs)` 辅助：CUDA 下走 `forward_cuda`，
  CPU 下走 `forward_native`（`sgl_kernel` 在 zeus 环境下不可用）。
- 所有 per-stage 测试函数（`test_embedding`、`test_rmsnorm`、`test_silu_and_mul`、
  `test_rope`、`test_qkv_proj`、`test_o_proj`、`test_mlp`、`test_store_kv_cache`、
  `test_extend_attention`、`test_decode_attention`、`test_transformer_block`、
  `test_lm_head`、`test_full_model`）中的 `.to("cuda")` 一律替换为
  `.to(REF_DEVICE)`，打印文案 `CUDA output` 同步替换为 `{REF_DEVICE.upper()} output`。
- `test_rmsnorm` 显式把 `SGLangRMSNorm` 的 `weight_dtype` 设为 `bfloat16`，
  让 `forward_native` / `forward_cuda` / `forward_zeus` 的权重乘法
  保持数值一致（否则 fp32 权重会导致 native 路径漂移）。
- `test_silu_and_mul` 放宽了 CPU vs zeus 比较的容差到
  `atol=5e-2, rtol=1e-2`，对应 bf16 CPU silu 与 zeus fused kernel 的末位差。
- `test_store_kv_cache` 中对 `torch.ops.sgl_kernel.store_kv_cache` 的调用
  改为 try/except，CUDA build 才会成功，CPU/zeus 退回 plain index assign。
- `main()` 中删除 `has_cuda` 判断与 `skip_stage_without_cuda` 调用，
  该函数被保留但标记为“backwards-compat，no longer used”，
  新增一行 `Reference device: {REF_DEVICE}` 打印。

---

# 阶段二：让 server 端到端跑通 Qwen2.5-0.5B on zeus

阶段一解决的是"demo 能跑到 forward 对齐"。这一段进一步让
`python -m sglang.launch_server --device zeus` 能完整启动、warmup、
响应一次 `curl /generate` 请求。需要触达 launch_server → scheduler →
model_runner → attention backend → logits processor 这条主路径上每一个
原本假设 CUDA 的点。

## 六、安装 flow — `python/pyproject_other.toml` + `zeus_dev/zeus_install.md`

仿照 `srt_hip` / `srt_npu` / `srt_hpu` 的结构新增 `srt_zeus` extra：

```toml
srt_zeus = [
  "sglang[runtime_common]",
  "torch",
]
all_zeus = ["sglang[srt_zeus]"]
dev_zeus = ["sglang[all_zeus]", "sglang[test]"]
```

extra 内部有意**不列** `torch_zeus` 和 `sgl_kernel_zeus`：这两个包还在
内部开发、没上任何公共或内部 PyPI index，如果硬写进依赖声明 `pip install
sglang[srt_zeus]` 会直接解不出依赖链。取而代之的做法是在紧挨着的注释里
说明"这两个包需要先从源码装"，并指向 `zeus_dev/zeus_install.md`。

新增 `zeus_dev/zeus_install.md` —— zeus 的完整安装指南：

- **Prerequisites**：`torch_zeus` 与 `sgl_kernel_zeus` 当前仅内部源码，
  从它们各自的源码目录 `pip install -e .`；
- **独立 smoke-test 命令**：`import torch_zeus, torch_zeus._C; import
  sgl_kernel_zeus; torch.zeus.is_available()`；
- **srt_zeus extra 的两种装法**：软链 `pyproject_other.toml` 或者
  `--config-settings pyproject=pyproject_other.toml`；
- **标准 launch_server 命令** 和每个 flag 的 "为什么"（`--device zeus`、
  `--disable-cuda-graph`、`--attention-backend torch_native`、
  `--sampling-backend pytorch`）；
- **curl /generate 烟雾测试**；
- **Known slow / missing**：列出当前 `[ZEUS Fallback]` 主要 op，明确
  说后续性能工作主战场在 `torch_zeus` / `sgl_kernel_zeus` 那边补 op，
  不在 sglang 侧继续改。

## 七、sglang 启动前置修复 — `python/sglang/srt/utils/common.py` 追加两处

阶段一改过 `is_cuda()`，阶段二又在同一个文件里补了两处：

- **`get_zeus_memory_capacity()`**：原实现直接访问 `torch.zeus.mem_get_info(0)`，
  但 `torch.zeus` 属性只有在 `torch_zeus` 被 import 一次之后才会被注册到
  `torch` 上。sglang server 启动时 `server_args.__post_init__` 就要读
  GPU memory capacity，这时很可能还没碰过 `torch_zeus`，于是直接
  `AttributeError: module 'torch' has no attribute 'zeus'`。补一行
  `import torch_zeus` 在访问前强制触发。
- **`get_compiler_backend()`**：原实现对 zeus 没有分支，一路 fall through
  到 `"inductor"`。`overlap_utils._resolve_future_token_ids` 上的
  `@torch.compile(backend=get_compiler_backend())` 在模块导入时就把
  backend 绑死成 inductor；首次 scheduler 调度时立刻炸成
  `torch._inductor.exc.InductorError: RuntimeError: device zeus nyi`。
  改成 `is_zeus()` 时直接返回 `"eager"`。

两处都对 CUDA 用户零影响（`is_zeus()` 要么由 `SGLANG_DEVICE=zeus`
显式启用，要么需要 CUDA 不可用 + `torch_zeus` 已装）。

## 八、paged 分配器 — `python/sglang/srt/mem_cache/allocator.py`

`PagedTokenToKVPoolAllocator.alloc_extend` / `alloc_decode` 走
`@triton.jit` 写的 `alloc_extend_kernel` / `alloc_decode_kernel`。Triton
在 zeus 上没有 driver，scheduler 第一次 prefill 就报
`RuntimeError: 0 active drivers ([]).`

补两个纯 Python fallback：

- `_alloc_extend_native(prefix_lens, seq_lens, last_loc, free_pages,
  out_indices, page_size)` —— 把 prefill/extend 的"填尾页 → 填满页 →
  填新页首"三段拆成 python 循环；`_alloc_decode_native(...)` —— decode
  每个 seq 只加一个 token，要么复用上一个槽位、要么取新页的首位。
- 模块顶层读 `_is_zeus = is_zeus()`，`alloc_extend` / `alloc_decode` 里
  加分支：zeus 走 native，其他后端保持原 triton kernel 调用。

实现思路是把所有 bookkeeping tensor `.cpu().tolist()` 后纯 Python 算好，
再一次性 `torch.tensor(...).copy_(out_indices)` 回 zeus，避免逐 op 触发
zeus fallback；bs/extend_num_tokens 都很小，成本可忽略。

## 九、KV 写入 — `python/sglang/srt/mem_cache/memory_pool.py`

`MHATokenToKVPool.set_kv_buffer` 原实现：

```python
self.k_buffer[layer][loc] = cache_k
self.v_buffer[layer][loc] = cache_v
```

在 zeus 上 `index_put_` 对多维 value tensor 处理有 bug，报
`values size mismatch: 768 vs num_indices 6`（把 `[6, H, D]` 的 cache_k
当成 flat `[N*H*D]`）。新增 `elif _is_zeus:` 分支改走 `scatter_`：

```python
idx = loc.view(-1, *([1] * (k_buf.dim() - 1))).expand_as(cache_k)
k_buf.scatter_(0, idx, cache_k)
v_buf.scatter_(0, idx, cache_v)
```

`scatter_` 在 zeus 上目前走 CPU fallback，数值正确、功能正常。

## 十、Attention backend — ~~`torch_native_backend.py`~~ → `zeus_backend.py`

**已 revert `torch_native_backend.py` 的改动。** Review 时确认 zeus 环境下
`server_args._handle_zeus_backends()` 默认设置 `attention_backend = "zeus"`，
走 `ZeusAttnBackend`（调用 `sgl_kernel_zeus.extend_attention` /
`decode_attention`），不会走到 `TorchNativeAttnBackend`。因此
`torch_native_backend.py` 中的 SDPA CPU round-trip workaround 不必要，已清理。

## 十一、Tied lm_head packing — `python/sglang/srt/model_loader/loader.py`

这条是阶段二里最绕的一个 bug，也是唯一一次**修错方向后又修回来**的：

**Qwen2 tied model 的实际情况**：`Qwen2ForCausalLM.__init__` 里写的是
`self.lm_head = self.model.embed_tokens`，`model.lm_head` **整个就是**
`embed_tokens` 这个 `VocabParallelEmbedding` 模块本身，**不是**一个独立
的 `ParallelLMHead`。`lm_head.weight is embed_tokens.weight`，同一个
`nn.Parameter` 对象。

**原 loader 的 zeus 分支意图**：对 tied model `if tie: pass`，把
`weight` 留在 GDG (N,K)；在 `logits_processor` 里用 `weight.t()` 翻成
(K,N) 给 `torch.mm` 作 mat2。

**这个 intent 在代码里没真正跑通**，两个隐藏 bug：
1. `_zeus_init_lm_head_from_embed` 用 `isinstance(module, ParallelLMHead)`
   遍历 `named_modules()`，对 tied model 永远找不到 lm_head —— 因为它
   是 VPE 不是 PLM。这个函数对 tied model 一直是 no-op。
2. `pack_weights(target_modules={ParallelLMHead})` 也按 isinstance 匹配
   —— 同样找不到 tied 的 lm_head，于是整个 lm_head.weight 根本没被
   pack 进 LocalMem，一直留在 GDG。zeus mm 看到 `mat2` 不是 LocalMem，
   第一次 logits 投影就炸。

**正确修复**：新增 `_zeus_decouple_tied_lm_head(model, ParallelLMHead,
VocabParallelEmbedding)`，在 `pack_weights` 之前显式 decouple：

- 通过**属性访问** `model.lm_head` / `model.model.embed_tokens` 拿到
  真实对象，**不**走 `named_modules` isinstance 筛选；
- 若 `lm_head.weight.data_ptr() != embed.weight.data_ptr()`，说明根本没
  tied，no-op 早退；
- 否则 `cloned = embed.weight.data.detach().clone()`；
- 如果 `lm_head` 本来就是 `ParallelLMHead` 实例，只替换它的 `weight`；
- **如果 `lm_head` 是 VocabParallelEmbedding**（Qwen2 tied 情况），构造一个
  全新的 `ParallelLMHead(num_embeddings, embedding_dim, params_dtype=...)`，
  把 `.weight` 设成那份 clone，再 `model.lm_head = new_lm_head`。这样后续
  `pack_weights` 的 isinstance 才能命中它、把它 transpose 打进 LocalMem。

`_zeus_init_lm_head_from_embed` 的调用也从"仅 non-tied 分支"改成无条件
调用（对 tied model 它依旧是 no-op，但对"checkpoint 原本是 tied、config
改成 non-tied"的边界情况负责 copy embed → lm_head）。

代价：额外一份 `vocab × hidden` bf16 tensor 内存（Qwen2.5-0.5B 上
~250 MB）。推理从不训练 embedding，decouple 完全安全。

**回滚**：前一轮我在 `logits_processor.py` 的 zeus 分支里加了一个 CPU
round-trip 作为 workaround，这次 loader 修好后整个 revert 掉，
`logits_processor.py` 这次会话的净改动归零，走的仍然是原本设计的
`_zeus_transposed_params` 双路径。

## 十二、运行验证

无 preload、标准入口（注意不指定 `--attention-backend` 和 `--sampling-backend`，
让 `_handle_zeus_backends` / `_handle_sampling_backend` 自动选择 zeus 原生后端）：

```bash
SGLANG_DEVICE=zeus python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-0.5B-Instruct --device zeus \
  --dtype bfloat16 --disable-cuda-graph \
  --disable-radix-cache \
  --mem-fraction-static 0.5 --max-running-requests 1 --tp-size 1
```

日志里依次出现：

- `Zeus: decoupled tied lm_head from embed_tokens ...`
- `Zeus: packed model weights into LocalMem.`
- `The server is fired up and ready to roll!`

```bash
curl -X POST http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is",
       "sampling_params": {"max_new_tokens": 6, "temperature": 0}}'
```

→ `" Paris. It is the largest"`, token ids `[12095, 13, 1084, 374, 279, 7772]`.
~42 秒 / 6 token，**没有任何 CPU fallback** 在 logits mm 这一环节 （注意，这里的30000端口可能有变化，要看server terminal显示的是多少）。

---

## 影响面与下一步

- 所有改动对 CUDA 用户保持行为一致：`SGLANG_DEVICE` 未设或为 `cuda` 时，
  `is_cuda()` / `is_zeus()` / `get_compiler_backend()` 等分支全部走原路径；
  `device_{empty_cache,synchronize}` 在 CUDA 上等价于原 `torch.cuda.*` 调用。
- 纯 zeus 环境现在可以：
  - 跑 `demo_zeus_layer_compare.py --stage all`；
  - 通过 `python -m sglang.launch_server --device zeus` 启动 server
    并响应 `/generate` 请求。
- **下一步性能工作主战场在 `torch_zeus` / `sgl_kernel_zeus` 侧**，
  按启动日志里 `[ZEUS Fallback]` 频次最高的 op 往下补原生实现：
  - `aten::scatter.src_out`（KV 写入）
  - `aten::add.Tensor`（各种 residual）
  - `aten::arange.start_out`
  - `aten::index.Tensor_out`
  - `Zeus fill_ does not support dtype Bool`（SDPA 因果 mask）
  - `zenl_add_kernel: inputs must match output device/numel`
  每补一个就能在 sglang 这边对应删掉一条 CPU round-trip / native fallback。
- **sglang 侧 Python fallback / workaround，待 zeus OP 补齐后可移除**：
  - `allocator.py` 的 `_alloc_extend_native` / `_alloc_decode_native`：
    Triton kernel 的纯 Python 翻写，把 bookkeeping tensor 搬到 CPU 做
    per-batch 循环。若 `sgl_kernel_zeus` 后续实现 `alloc_extend` /
    `alloc_decode` kernel 就可以直接替换，不过这两个操作是 per-batch
    级别的整数运算，优先级低。
  - `memory_pool.py` 的 `MHATokenToKVPool.set_kv_buffer` zeus 分支：
    因 `aten::index_put_` 对多维 value tensor 的 broadcast 不正确
    （把 `[N, H, D]` 当 flat `[N*H*D]` 处理），改用 `scatter_` +
    手动 expand index。待 zeus 侧修复 `index_put_` 的多维 broadcast
    后可回退到原始写法 `k_buffer[loc] = cache_k`。
  - `memory_pool.py` 的 `ReqToTokenPool.write` zeus 分支：
    同样因 `index_put_` 问题，把 `req_to_token` 搬到 CPU 做写入再搬回。
    待 `index_put_` 修复后可移除。
  - `memory_pool.py` 的 `MambaPool.free` zeus 分支：
    `torch.cat` 在 zeus 上可能存在兼容问题（防御性写法），搬到 CPU
    做 cat 再搬回。待确认 zeus 的 `aten::cat` 是否稳定后决定是否移除。
- 后续仍需关注的硬编码 `torch.cuda.*` 调用点（本轮未覆盖）可通过
  `grep -rn "torch.cuda\." python/sglang/srt` 继续清理。
- 想 commit 成 PR，建议分两份：
  - **PR1（安装 flow）**：`pyproject_other.toml` + `zeus_dev/zeus_install.md`；
  - **PR2（runtime 兼容补丁）**：`utils/common.py` / `allocator.py` /
    `memory_pool.py` / `loader.py` 以及阶段一的 demo 重写和 import 短路
    改动，外加本文档。（`torch_native_backend.py` 和 `model_runner.py`
    的改动已 revert，不再包含在 PR 中。）
