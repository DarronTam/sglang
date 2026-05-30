# Zeus Graph 回归修复：`next_token_logits_buffer` dtype 导致 decode 输出冻结

> 日期：2026-05-30
> 分支：`dev-merge-with-zhipu`
> 症状：`zeus_dev/smoke_test/test_zeus_graph_e2e.py` graph 模式输出全是重复 token（如 `The[][][][]...`），eager 模式正常。
> 结论：合并 zhipu 时**回退**了一处 Zeus 专属修复（该修复在 `dev` 分支上存在），属于回归。

---

## 1. 症状

```
Prompt 0: [MISMATCH]
  Eager : 'The capital of France is Paris.'
  Graph : 'The[][][][][][][][][][][][][][][][][]...'
```

- eager 模式：完全正确。
- graph 模式：prefill 出的第一个 token 正确（"The"），之后每个 decode step 都吐同一个 token，且 **4 个 prompt 的结果完全一样**——输出与输入无关。

## 2. 根因

**Zeus 芯片规则：** 在 captured graph 内，一个**转换 dtype**（bf16→float32）或**跨设备**的 `copy_`，会走同步 **CPU-bounce 路径，不会被录进 graph**。replay 时它根本不执行，目标 buffer 因此一直停留在 **capture 时的 dummy 值**上。

具体链路：

- `cuda_graph_runner.py` 的 `DecodeInputBuffers.create()` 把 `next_token_logits_buffer` 建成 `dtype=torch.float`（float32）。
- lm_head 输出 logits 是 **bf16**。
- `LogitsProcessor._copy_logits_to_buffer()` 执行 `logits_buffer.copy_(logits)` → bf16→float32 **跨 dtype 拷贝** → CPU bounce → **未录进 graph**。
- replay 时该拷贝不执行 → logits buffer 永远是 capture 时 dummy 输入（`input_ids=0`）算出来的那一份 → argmax 恒定 → 重复 token。

eager 不受影响：CPU-bounce 拷贝每步都真实执行。

## 3. 定位过程（关键证据）

1. 逐一验证 Zeus 运行时 / kernel 在 graph replay 下都正确反映输入变化：
   - 单算子：`embedding`（`sgl_kernel_zeus` 与 `aten::embedding` 两条路径）、`mm`/linear、`rmsnorm`、`fused_add_rmsnorm`、`decode_attention`+`store_kv_cache` —— 全部 OK。
   - 链式 + 反馈循环 + 完整 2/8 层 decode pipeline（含 residual 别名模式）—— 全部 VARIES（正确）。
   - 结论：运行时、kernel、graph 机制、pipeline 组合都没问题，bug 在 SGLang 集成层。（npu_simulator 不涉及。）
2. 在 `ZeusGraphRunner.replay` 加 `[ZEUS_GRAPH_DEBUG]` 打印：
   - `input_ids` buffer **每步都正确更新**（785→1294→1294…）。
   - 但 logits 的 `min/max/mean/argmax` **逐字节相同**——证实输出与输入完全无关，即某个 buffer 在 replay 时从未被重写。
3. 因 graph replay 时 Python 模型 forward 不执行，普通 print 只在 capture 时触发；改用 `copy_` 写入持久 buffer 仍无效（`.float()/.detach()` 等 ATen 走 CPU fallback 未被捕获）——这反过来印证了「跨 dtype/非捕获 copy 在 replay 不生效」的机制。
4. 用户提示「`dev` 分支能跑过」→ 定位为回归。对比 `dev` 与 HEAD：`dev` 的 `next_token_logits_buffer` 在 Zeus 上用模型 dtype（bf16），并带有解释此 bug 的注释；zhipu 合并把它改回了 `torch.float`。

## 4. 修复（2 处）

**a) `python/sglang/srt/model_executor/cuda_graph_runner.py`** —— buffer 在 Zeus 上匹配模型 dtype：

```python
next_token_logits_buffer = torch.zeros(
    (max_num_token, vocab_size),
    # Zeus chip rule: bf16(lm_head)->float32 的 copy_ 跨 dtype，走未被捕获的
    # CPU-bounce；replay 时不执行，logits buffer 冻结在 capture 值 -> 重复 token。
    # Zeus 上匹配模型 dtype，让 copy_ 走 dtype 一致的 D2D 快路 (zenlMemcpy，被捕获)。
    dtype=dtype if torch.device(device).type == "zeus" else torch.float,
)
```

**b) `python/sglang/srt/layers/logits_processor.py`** —— 放宽 `_copy_logits_to_buffer` 的断言以接受 Zeus 的 bf16 buffer：

```python
assert logits_buffer.dtype in (torch.float, logits.dtype), (
    f"next_token_logits_buffer dtype {logits_buffer.dtype} "
    f"is incompatible with logits dtype {logits.dtype}"
)
```

其它后端（CUDA 等）仍保持 float32，行为不变。

## 5. 验证

`python test_zeus_graph_e2e.py --test correctness` → **PASS**，4 个 prompt graph 与 eager 全部 MATCH。
（完整 5 项测试结果见下方「6. 完整测试结果」。）

## 6. 经验 / 注意

- **判 Zeus graph「重复同一 token」的 bug，先怀疑 captured 区内有跨 dtype / 跨设备的 `copy_`。** 同 dtype 同设备的 `copy_` 才会走被捕获的 `zenlMemcpy` D2D 快路。
- 这类 Zeus 专属特判（buffer dtype、`zeus_index_dtype` int32、CPU-bounce 规避）在大型合并（zhipu）中**容易被整体回退**。Zeus 专属功能一旦回归，优先 `git diff dev HEAD -- <file>` 对比。
- 相关既有保护：`zeus_graph_runner.py` 里的 `_check_zeus_int32_fields`（int32 forward 字段守卫），原理同源。

## 7. 完整测试结果

`python test_zeus_graph_e2e.py`（全量，2026-05-30，torch10_312 环境）：

```
Phase 5 Summary
  correctness    : PASS     # 4/4 prompt graph 与 eager 全部 MATCH
  multi_bs       : PASS     # BS=1/4/8 全部与 baseline 一致
  long_seq       : PASS     # max_tokens=16/64/128/256 graph==eager
  stability      : PASS     # 5 轮 decode 输出完全一致，无 drift
  perf           : DONE
```

性能（模拟器环境，绝对吞吐很低，只看相对）：

```
  BS   Eager (tok/s)   Graph (tok/s)   Speedup
   1            1.11            1.20     1.08x
   4            1.52            1.89     1.24x
   8            1.93            1.93     1.00x
  Memory: eager=4096 MB, graph=4096 MB, delta=+0 MB
```

结果文件：`zeus_dev/smoke_test/zeus_graph_perf_results.json`。

> 注：上述运行在 Zeus 模拟器上（每 kernel fork+exec 子进程），绝对延迟不代表真实硬件；
> 关键结论是 4 项正确性测试全部通过，graph 与 eager 数值一致。
