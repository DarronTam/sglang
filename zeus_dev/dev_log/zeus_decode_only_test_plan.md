# Plan: Zeus 上 GLM5-next 16B decode-only 测试脚本(CUDA dump → Zeus inject)

## Context

用户想在 `zeus_dev/` 下加一个测试脚本,模拟 PD 分离里 **decode 节点的部署**:
- 不跑 prefill,只跑 decode
- 用一次性 dump 出来的真实 prefill KV cache + KDA state cache 作为 mock 数据喂给 decode
- **dump 必须在 CUDA 设备上跑生成,decode-only 测试在 Zeus 上跑** —— 设备之间通过 CPU 中转
- 调用方式要和 `demo_zeus_llm.py` 一致(`sgl.Engine(...).generate(...)` 走生产路径),不允许手拼 forward
- 模型: `Glm5NextForCausalLM` @ `/infra/Linear/16b_hf/`(hybrid: MLA full-attn + KDA linear-attn)

目的是验证 decode-only 部署时,Zeus 上完整的模型加载 + decode 路径(MLA + KDA 两种层都要走通)。

产出两个脚本,都在 `zeus_dev/`:
1. `dump_glm5_next_prefill_cache.py` — **在 CUDA 上跑一次**生成 dump 文件,后续 Zeus 端复用
2. `test_zeus_decode_only_llm.py` — **在 Zeus 上**反复跑,加载 dump 后做 decode

### CUDA vs Zeus KV cache layout(已重新核对,2026-05-17 修订)

**前次结论已作废**。Zeus 上 `attention_backend="zeus"` 触发的 `ZeusTokenToKVPool`(`mem_cache/zeus_memory_pool.py:79-180`)与 CUDA `MHATokenToKVPool` 的 layout 完全不同:

| 维度 | CUDA `MHATokenToKVPool` | Zeus `ZeusTokenToKVPool` (attention_backend="zeus") |
| --- | --- | --- |
| `k_buffer[layer]` shape | `[size+page_size, num_kv_heads, head_dim]` 平铺 | `[num_pages, num_kv_heads, page_size, head_dim]` 分页 |
| K 内部 layout | row-major | **block-tiled row-major**(`sgl_kernel_zeus.store_kv_cache` 写) |
| V 内部 layout | row-major | **16-byte column-group interleaved**(`store_kv_cache` 写) |
| 直接 PyTorch index | 可 | 不可 — 必须走 `store_kv_cache` |

→ **结论**:不能用 `kv_buffer[idx].copy_(t)` 直接注入。本计划改为:**inject 端调 `token_to_kv_pool.set_kv_buffer(layer=None, loc=token_indices, cache_k=k_flat, cache_v=v_flat, layer_id_override=layer_id)`** —— 这是生产 prefill 写池子的同一条路径,`ZeusTokenToKVPool.set_kv_buffer` 内部就是 `store_kv_cache(...)`,所有 tile 转换在内核里完成,我们的 hook 不需要懂 tiling 规则。

**KDA mamba state**:`MambaPool.mamba_cache.conv/temporal` 是普通 PyTorch tensor,shape device-agnostic,可以直接 `.copy_(t.to("zeus"))`。

### 已知风险:GLM5-next 在 Zeus 上的可运行性未验证

Zeus 当前只支持 MHA(`attention_backend="zeus"` → `ZeusTokenToKVPool` 只分配 MHA-shape buffer),但 GLM5-next 是 MLA + KDA hybrid:

- `attention_backend="zeus"` 分支(`model_runner_kv_cache_mixin.py:478`)在 MLA / mambaish 分支之前,会无视 model 的 MLA 配置,分配出 MHA-shape `[num_pages, num_kv_heads, page_size, head_dim]` 池子
- 这与 MLA layer 写 KV 时期望的 latent KV shape(`[size+page_size, 1, kv_lora_rank+qk_rope_head_dim]`)不匹配,**Engine 启动时 / 第一次 forward 时大概率崩**
- `sgl-kernel-zeus` 没有任何 MLA kernel(`extend_attention` / `decode_attention` / `store_kv_cache` 都是 MHA 路径)

**用户决策**:先把脚本搭起来,**不解决 MLA 适配问题**,运行后根据具体报错再去 sgl-kernel-zeus / `zeus_backend.py` 加缺的 kernel / pool 分支。这两个脚本本身的逻辑(hook 安装、dump 落盘、inject 走生产 `set_kv_buffer`)与 MLA 适配是正交的,先期可独立验证 hook 链路是否打通。

---

## 关键代码路径

### Disaggregation = decode 流程
- `python/sglang/srt/disaggregation/decode.py:881-933` (`_pre_alloc`) — decode 端按 `len(origin_input_ids) + max(len(output_ids)-1, 0)` 预分配 KV 槽
- `python/sglang/srt/disaggregation/decode.py:760-807` — 调 `kv_receiver.send_metadata(page_indices, metadata_buffer_index, state_indices)` 上报已分配的槽位
- `python/sglang/srt/disaggregation/decode.py:1026-1053` — fake 模式下读 `metadata_buffers.output_ids[idx][0]` 作为 prefill 采的第一个 token,append 到 `req.output_ids`,然后进 `process_prebuilt` 路径
- `python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py:22-101` (`prepare_for_prebuilt`) — 把 batch 标成 `ForwardMode.PREBUILT`,跳过 prefill forward
- `python/sglang/srt/disaggregation/fake/conn.py:79-115` — `FakeKVReceiver`,瞬时返回 Success,不真传数据(我们的注入点)
- `python/sglang/srt/disaggregation/utils.py:135-296` — `MetadataBuffers`,共享内存里存第一个采样 token + cached_tokens 等

### Prefill 端 hook
- `python/sglang/srt/managers/scheduler.py:2867-2885` — `process_batch_result` 分发,prefill 走 `process_batch_result_prefill`
- `python/sglang/srt/disaggregation/prefill.py:464-583` — disagg prefill 路径的 hook(我们用 null mode 不走这条)

### KV / state pool 数据结构
- `python/sglang/srt/mem_cache/memory_pool.py:497-689` — `HybridReqToTokenPool`
  - `.req_to_token` shape `[size, max_context_len]` int32, 存每个 req 的 token→kv-slot 映射
  - `.mamba_pool` → `MambaPool`
  - `.req_index_to_mamba_index_mapping` shape `[size]` — req_pool_idx → mamba_pool_slot
- `python/sglang/srt/mem_cache/memory_pool.py:222-395` — `MambaPool`
  - `.mamba_cache.conv` 是 `List[Tensor]`,每个 `[size+1, 3*Hh, K-1]` bf16(每个 conv 子组一个)
  - `.mamba_cache.temporal` shape `[num_mamba_layers, size+1, H, K, V]` fp32
- `python/sglang/srt/mem_cache/memory_pool.py:1281+` — `HybridLinearKVPool`(token_to_kv_pool)
  - `.full_kv_pool.kv_buffer[layer_id]` 是 full-attn (MLA) 层的 KV 缓冲

### Engine 进程拓扑
- `python/sglang/srt/entrypoints/engine.py:143` — `Engine` 类,`run_scheduler_process_func = staticmethod(run_scheduler_process)` (line 161)
- `python/sglang/srt/managers/scheduler.py:3616` — `run_scheduler_process` 是子进程入口
- TP=1 时 scheduler 跑在独立子进程(spawn),要装钩子必须在子进程的入口包一层

### 参考样板
- `zeus_dev/demo_zeus_llm.py` — Engine 启动 + generate 风格
- `zeus_dev/dev_glm5_next_kda_test.py:736-779` — mamba pool / req_to_token_pool stub 结构(读参考用)

---

## 设计

### 强制对齐(CUDA dump 端 ↔ Zeus decode 端)

两端 Engine 必须用**完全相同**的下列 server args,否则 KV slot 索引算不齐:

| 参数 | 强制值 | 理由 |
| --- | --- | --- |
| `model_path` | `/infra/Linear/16b_hf/` | 同一份权重,config 一致 |
| `dtype` | `"bfloat16"` | KV / state 张量精度 |
| `kv_cache_dtype` | `"bfloat16"`(显式传) | CUDA 默认可能用 fp8/auto;显式禁掉 |
| `page_size` | `128`(显式传同值) | **Zeus 硬件要求 `page_size % 128 == 0`**(`server_args.py:1522`);CUDA 端配同值对齐 slot 索引 |
| `max_running_requests` | `1` | 单 req 简化 |
| `disable_radix_cache` | `True` | 避免 prefix cache 影响 KV slot 分配模式 |
| `disable_cuda_graph` | `True` | 两边都关,避免 graph capture 干扰 hook |
| `mem_fraction_static` | `0.5` | 仅影响 pool 总 size,不影响 slot 索引 |
| `attention_backend` | CUDA: `"triton"` / Zeus: `"zeus"` | **Zeus 端坚持用 `"zeus"`,即使当前对 MLA 不兼容(见上节风险)**;CUDA 端用 `"triton"` 兼容 MLA absorb |

prompt + sampling 也必须一致:
- `PROMPT = "中国的首都是"`(字符串字面量同步,两脚本都写死)
- `temperature=0.0`(确定性采样)

新建一个文件存 monkey-patch 代码,两个主脚本都 import 它。**这个模块只在 scheduler 子进程里被装载**(通过 Engine.`run_scheduler_process_func` 包装),不会污染主进程。

主要 API:
```
install_dump_hook(dump_path: str)   # prefill 完写盘
install_inject_hook(dump_path: str) # decode 端 send_metadata 写池子
```

实现要点(2026-05-17 修订):
- `install_dump_hook`: monkey-patch `Scheduler.process_batch_result_prefill`,原函数走完之后,对 batch 第一个 `req` snapshot 以下内容并 `torch.save` 落盘:
  - 元数据: `origin_input_ids` (list[int])、`first_sampled_token = req.output_ids[-1]`、`seq_len`、`req_pool_idx`、`mamba_pool_idx`
  - 服务端参数: `page_size`、`kv_cache_dtype`、`head_num`、`head_dim`、`kv_lora_rank+qk_rope_head_dim`(用于 inject 端断言)
  - `kv_slots = req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len].cpu()` (int)
  - **Full-attn 每层 KV**: `kv_buffer[layer_pool_idx][kv_slots].detach().cpu().clone()`(直接读 CUDA pool 的 logical 切片,shape 可能是 MLA latent `[seq_len, 1, kv_cache_dim]` 或 MHA `[seq_len, head_num, head_dim]`,取决于 CUDA 上 pool 类型)
  - **KDA 每层 state**: `conv[g][mamba_pool_idx]` 和 `temporal[mamba_layer_idx, mamba_pool_idx]` 都 `.detach().cpu().clone()`
  - 全部 tensor 设备解耦后落盘,文件名: `/tmp/glm5_next_16b_prefill_dump.pt`
  - 打印 `[DUMP] saved to {dump_path}` 后让原 generate 正常返回(不影响 Engine 关闭)

- `install_inject_hook`: 模块级全局缓存 dump(第一次加载),monkey-patch `FakeKVReceiver.send_metadata(self, kv_indices, aux_index, state_indices)`:
  - 通过模块级 `_SCHEDULER_REF` 拿到 scheduler → `model_runner` → `token_to_kv_pool`(`ZeusTokenToKVPool`)、`req_to_token_pool`、`mamba_pool`、`disagg_metadata_buffers`
  - **断言对齐**: dump 元数据里的 `page_size` / `kv_cache_dtype` 必须与 Zeus 端 server args 严格相等,否则 raise(不做静默兼容)
  - 把 `kv_indices`(page 索引数组)展开成 token 级 `kv_token_indices`(长度 == dump 的 `seq_len`),转 Zeus tensor
  - **Full-attn 写入(走生产路径)**:对每个 full-attn `layer_id`,调
    ```python
    token_to_kv_pool.set_kv_buffer(
        layer=None,
        loc=kv_token_indices_zeus,         # zeus device, int
        cache_k=k_flat.to("zeus"),         # 来自 dump
        cache_v=v_flat.to("zeus"),
        layer_id_override=layer_id,
    )
    ```
    `ZeusTokenToKVPool.set_kv_buffer`(`mem_cache/zeus_memory_pool.py:132-163`)内部调 `sgl_kernel_zeus.store_kv_cache(...)` —— **tile 转换全部由 kernel 完成**,本 hook 不感知 layout 细节
  - **KDA mamba state**: `mamba_pool.mamba_cache.conv[g][state_idx].copy_(c.to("zeus"))` + `.temporal[layer_idx, state_idx].copy_(t.to("zeus"))`(shape device-agnostic,直接 copy)
  - **第一个 token**: `disagg_metadata_buffers.output_ids[aux_index][0] = first_sampled_token`(decode 端 `decode.py:1053` 会读这里 append 到 `req.output_ids`)
  - **bootstrap_room**: `disagg_metadata_buffers.bootstrap_room[aux_index, 0] = ...`(fake 路径下 bypass,但保险)
  - 调原 `send_metadata` 让 `has_sent_metadata = True`

辅助 patch:
- 包一层 `Scheduler.__init__` 末尾,把 `self` 塞进 `FakeKVManager` 实例(实际操作是 patch `FakeKVManager.__init__`,从 `server_args` 反向找不到 scheduler,所以更稳的做法是 patch 一个模块全局 `_SCHEDULER_REF`,Scheduler.__init__ 末尾赋值)。
- 注入端 patch 读 `_decode_only_hooks._SCHEDULER_REF` 拿 model_runner。

### 入口包装

每个脚本各自定义一个 `_InstallHooksEngine(Engine)` 子类,override `run_scheduler_process_func`:

```
def _run_scheduler_with_hook(*args, **kwargs):
    from zeus_dev import _decode_only_hooks as h
    h.install_dump_hook(os.environ["ZEUS_DECODE_DUMP_PATH"])  # or install_inject_hook
    from sglang.srt.managers.scheduler import run_scheduler_process
    run_scheduler_process(*args, **kwargs)
```

环境变量 `ZEUS_DECODE_DUMP_PATH` 决定 dump 文件位置(默认 `/tmp/glm5_next_16b_prefill_dump.pt`),两个脚本都从这里读。

### 脚本 1: `zeus_dev/dump_glm5_next_prefill_cache.py` (CUDA)

**头部不设 `SGLANG_DEVICE=zeus`,也不 `import torch_zeus`。这是 CUDA 路径。**

主体:
```
DUMP_PATH = "/tmp/glm5_next_16b_prefill_dump.pt"
MODEL_PATH = "/infra/Linear/16b_hf/"
PROMPT = "中国的首都是"

os.environ["ZEUS_DECODE_DUMP_PATH"] = DUMP_PATH

class _DumpEngine(sgl.Engine):
    run_scheduler_process_func = staticmethod(_run_scheduler_with_dump_hook)

llm = _DumpEngine(
    model_path=MODEL_PATH,
    dtype="bfloat16",
    kv_cache_dtype="bfloat16",       # 显式禁掉 fp8 / auto
    page_size=128,                   # 与 Zeus 端严格一致(Zeus 硬性要求 128 倍数)
    disable_cuda_graph=True,
    disable_radix_cache=True,
    attention_backend="triton",      # CUDA 上能 cover MLA 的 backend
    mem_fraction_static=0.5,
    max_running_requests=1,
    # device 默认 cuda,不显式传
)
out = llm.generate([PROMPT], {"max_new_tokens": 1, "temperature": 0.0})
print("[DUMP] prefill output:", out)
llm.shutdown()
print(f"[DUMP] dump saved to {DUMP_PATH}")
```

- `max_new_tokens=1` 保证 dump 时 `output_ids` 只有 1 个 token(就是 prefill 采的第一个),hook 直接拿 `req.output_ids[-1]`
- `temperature=0.0` 保证 decode 复跑时第一个 token 的 logit 行为可重复
- dump 张量全部 `.cpu().clone()`,文件可在不同 host / 不同设备上加载

### 脚本 2: `zeus_dev/test_zeus_decode_only_llm.py` (Zeus)

**头部必须 `SGLANG_DEVICE=zeus` + `import torch_zeus`(在 `import sglang` 之前),与 `demo_zeus_llm.py` 一致。**

```
os.environ["SGLANG_DEVICE"] = "zeus"
import torch_zeus  # noqa: F401

DUMP_PATH = "/tmp/glm5_next_16b_prefill_dump.pt"
MODEL_PATH = "/infra/Linear/16b_hf/"
PROMPT = "中国的首都是"           # 必须和 dump 时完全一致

os.environ["ZEUS_DECODE_DUMP_PATH"] = DUMP_PATH

assert os.path.exists(DUMP_PATH), f"先在 CUDA 上跑 dump_glm5_next_prefill_cache.py 生成 {DUMP_PATH}"

class _DecodeOnlyEngine(sgl.Engine):
    run_scheduler_process_func = staticmethod(_run_scheduler_with_inject_hook)

llm = _DecodeOnlyEngine(
    model_path=MODEL_PATH,
    device="zeus",
    dtype="bfloat16",
    kv_cache_dtype="bfloat16",       # 与 dump 一致
    page_size=128,                   # Zeus 要求 128 倍数
    disable_cuda_graph=True,
    disable_radix_cache=True,
    attention_backend="zeus",        # 坚持 zeus 路径(即使 MLA 不兼容,先跑起来再看报错)
    mem_fraction_static=0.5,
    max_running_requests=1,
    disaggregation_mode="decode",
    disaggregation_transfer_backend="fake",
)
out = llm.generate(
    [PROMPT],
    {"max_new_tokens": 16, "temperature": 0.0},
)
print("[DECODE-ONLY] output:", out)
llm.shutdown()
```

执行流程(运行时,Zeus 端):
1. Engine init → scheduler 子进程起来,`install_inject_hook` 装好 monkey-patch + 加载 dump 到模块全局
2. `llm.generate(...)` 提交一个普通 request
3. decode-side scheduler 通过 `DecodePreallocQueue._pre_alloc` 给请求分 KV 槽 + mamba 槽
4. `send_metadata` 被调用 → patch 校验 dump 元数据匹配 → 把 dump 张量 `.to("zeus")` 后拷到那些槽里 + 写 `metadata_buffers.output_ids[idx][0] = first_sampled_token`
5. `FakeKVReceiver.poll` 返回 Success
6. scheduler 用 PREBUILT forward mode 进入 decode 循环,基于注入的 KV/state + 第一个 token 开始生成
7. decode 出 16 个 token,打印 detokenize 后的输出

---

## 列表: 文件改动

**新增**
- `zeus_dev/_decode_only_hooks.py` — monkey-patch 公共模块(install_dump_hook / install_inject_hook + _SCHEDULER_REF 机制)
- `zeus_dev/dump_glm5_next_prefill_cache.py` — dump 脚本
- `zeus_dev/test_zeus_decode_only_llm.py` — decode-only 测试脚本

**不修改任何 SGLang 源码**(全部通过 monkey-patch 在 scheduler 子进程里完成)

---

## 验证步骤

1. 装环境:
   - CUDA host: 用默认 `cd python && pip install -e .` 装 CUDA 路径
   - Zeus host: `bash zeus_dev/setup_zeus_dev.sh`
2. **CUDA host** 上第一次跑 dump:
   ```
   python zeus_dev/dump_glm5_next_prefill_cache.py
   ```
   期望:打印 prefill 输出,`/tmp/glm5_next_16b_prefill_dump.pt` 文件生成,大小至少几 GB(prompt 短 → 主要占用是 MambaPool 那一份 conv/temporal × 所有 KDA 层)
3. 把 dump 文件**拷到 Zeus host 的同样路径** `/tmp/glm5_next_16b_prefill_dump.pt`(`scp` 或共享文件系统)
4. **Zeus host** 上跑 decode-only 测试:
   ```
   python zeus_dev/test_zeus_decode_only_llm.py
   ```
   期望:不再做 prefill forward(scheduler 日志能看到走 PREBUILT path),只跑 decode,输出一段连续的 token 序列,文本合理(例如 prompt "中国的首都是" 后面接出 "北京" 等)
5. 验证 decode 正确性(可选):
   - CUDA host:同样 prompt 用普通 `sgl.Engine`(无 disaggregation)跑 `max_new_tokens=16, temperature=0.0`,记下 token 序列 `T_cuda`
   - Zeus host:test_zeus_decode_only_llm.py 输出的 token 序列 `T_zeus`
   - 期望第 1 个 token 完全一致(`first_sampled_token` 来自 dump);后续 token 因 Zeus vs CUDA kernel 数值微差,可能从某个位置开始发散,这是已知的设备差异,不算 bug

---

## 风险 / 注意点

- **dump 文件巨大**:16B × 所有 layer 的 conv/temporal state(每个 layer 至少几十 MB),容易几个 GB 到几十 GB。`/tmp` 空间要够;若不够改 `DUMP_PATH`
- **跨设备 dtype / page_size 漂移**:CUDA 默认 `kv_cache_dtype=auto` 在某些 GPU 上会自动切 fp8;**必须显式传 bfloat16 + page_size=64**。inject hook 里要断言 dump 元数据匹配,失败直接 raise
- **GLM5-next 在 CUDA 上的 attention_backend 选项**:常见为 `triton` 或 `fa3`;需确认能 cover MLA absorb 路径。若实际跑发现 attention_backend 选错,在 dump 脚本里换 `flashinfer` / `fa3` 即可,**不影响 KV 池子里存的数据**(layout 与 backend 无关)
- **第一个 token 来自 dump**:test 脚本里"第 1 个 decode token"实际上是 dump 时 prefill 采的 token,这是设计上正确的(模拟 PD 分离 prefill→decode 的 hand-off)
- **prompt + sampling 必须严格一致**:dump 和 decode 共用同一份 prompt 字符串、`temperature=0.0`、`max_new_tokens=1`(dump 端);一旦改 prompt 必须重跑 dump
- **scheduler 子进程引用 `_SCHEDULER_REF`**:依赖 monkey-patch `Scheduler.__init__` 末尾把 self 写进 hook 模块的全局,这是子进程内部的全局,不跨进程,安全。`MetadataBuffers` 是子进程内的本地对象,inject 直接写就能被 PREBUILT 路径读到
- **CUDA host 必须能跑通 16B 模型**:dump 端要做完整的 prefill,这需要 ~32GB+ 显存(16B bf16 权重)。如果 CUDA host 显存不够,无法 dump
- **Zeus 端不要装 CUDA `import` 路径**:Zeus 脚本必须先 `os.environ["SGLANG_DEVICE"]="zeus"` 再 `import sglang`,否则 `is_zeus()` 时序不对(`demo_zeus_llm.py` 注释里有说明)
