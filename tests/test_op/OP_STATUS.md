# Zeus ATen 算子实现状态

> 最后更新: 2026-04-14（引入 MLU 风格 codegen 流水线：`codegen/zeus_functions.yaml` + `gen_zeus_stubs.py` 自动生成 `generated/RegisterZeus.cpp`，替换手写 `StructuredOpsDispatch.cpp`）

本文档跟踪所有已注册到 `PrivateUse1` 后端的 ATen 算子的实现状态。
重点标注哪些算子仍在通过 **CPU roundtrip** 执行，这些是临时方案，后续需要替换为原生 ZENL kernel。

## 状态说明

| 标记 | 含义 |
|------|------|
| ZENL | 调用原生 ZENL kernel（zenlXxx API），在设备端完成计算 |
| Device | 使用设备级内存操作（zertMemcpy / zenlMemcpy），无计算 |
| Metadata | 纯元数据操作（view、stride 变更等），无数据搬移 |
| **CPU Roundtrip** | **临时方案**：数据下载到 CPU → CPU 上计算 → 结果上传回设备 |

---

## ZENL 原生 kernel 算子

这些算子已有完整的设备端实现，通过 ZENL C API 调度到 sim/NPU kernel。

| 算子 | 注册方式 | ZENL API | 备注 |
|------|----------|----------|------|
| `add` | Codegen (Level 1) + `REGISTER_PRIVATEUSE1_DISPATCH(add_stub)` (Level 2) | `zenlAdd` | ATen dispatch: f32 / bf16 / int8（wrapping）；ZENL API / Triton: f32 / bf16 / int8（saturation）/ fp8e4m3；`.Tensor` / `._` / `.out` 三个 variant 由 codegen 统一生成包装 |
| `sub` | Codegen (Level 1) + `REGISTER_PRIVATEUSE1_DISPATCH(sub_stub)` (Level 2) | `zenlSub` | 同上 |
| `mul` | Codegen (Level 1) + `REGISTER_PRIVATEUSE1_DISPATCH(mul_stub)` (Level 2) | `zenlMul` | 同上 |
| `fill_.Scalar` | `TORCH_LIBRARY_IMPL` | `zenlFill` | f32 / bf16 / int8 / fp8e4m3；fill\_(0) 走 `zertMemsetAsync` 快速路径（任意 dtype），非零值走 `zenlFill` kernel；LocalMem tensor 显式拒绝；Triton: `fill.py`；sim: `fill_{dtype}_sim.c` |
| `zero_` | 委托给 `fill_(0)` | `zertMemsetAsync` | 经由 fill\_(0) 走 memset 快速路径，不经过 kernel |
| `sum` | Codegen (Level 1) + `REGISTER_PRIVATEUSE1_DISPATCH(sum_stub)` (Level 2) | `zenlReduce(ADD)` | f32 / bf16 / int8 / fp8e4m3 均有 per-dtype 快速路径；last-dim / mid-dim 单块直调 + 非连续 reduce dims 自动分解为多步快速路径（右向左逐块 reduce）；**仅支持 C-contiguous 输入**，非 contiguous 或其他 dtype 直接报错（不走 generic fallback）；Triton: `reduce_sum.py` / `reduce_sum_mid.py`；sim: `reduce_sum_{dtype}_sim.c` / `reduce_sum_mid_{dtype}_sim.c` |
| `mean` | Codegen (Level 1) + `REGISTER_PRIVATEUSE1_DISPATCH(mean_stub)` (Level 2) | `zenlReduce(AVG)` | f32 / bf16 / int8 / fp8e4m3 |
| `norm` | Codegen (Level 1) + `REGISTER_PRIVATEUSE1_DISPATCH(norm_stub)` (Level 2) | `zenlReduce(NORM*)` | f32 / bf16；L1 / L2 / Lp |
| `argmax` | Codegen (Level 1) + `REGISTER_PRIVATEUSE1_DISPATCH(argmax_stub)` (Level 2) | `zenlReduce(MAX, ONLY_INDICES)` | f32 / bf16；输出 int64 索引；支持指定 dim / keepdim / 全局 reduce |
| `argmin` | Codegen (Level 1) + `REGISTER_PRIVATEUSE1_DISPATCH(argmin_stub)` (Level 2) | `zenlReduce(MIN, ONLY_INDICES)` | f32 / bf16；输出 int64 索引；支持指定 dim / keepdim / 全局 reduce |
| `embedding` | `TORCH_LIBRARY_IMPL` | `zenlEmbedding` | LocalMem weight 需先转 linear 再 gather |
| `index_put_` / `index_put` | `TORCH_LIBRARY_IMPL` | `zenlIndexPut` | scatter write；op layer 做多维 index 线性化；**不支持 accumulate**；不支持 bool mask 索引；不支持 AdvancedIndexing（None 维度、子空间转置）；self 必须 contiguous |

## GEMM 算子

Zeus 上所有 GEMM 操作的右矩阵（权重）**必须**在 LocalMem 中。
使用 `zeus.pack_weights(model, Tr=..., Tc=...)` 预先打包模型权重。

| 算子 | 状态 | 备注 |
|------|------|------|
| `mm` | ZENL (`zenlGemm`) | mat2 必须是 LocalMem；否则 **RuntimeError**；BF16 dense + aligned(仅 `alignedSize=1024KB`, `Tr/Tc=4x4` 或 `8x8`)；Triton: `gemm.py` 仅 dense；sim: `gemm_bf16_dense_sim.c` + `gemm_bf16_as1024_tr{4,8}_tc{4,8}_sim.c` |
| `addmm` | ZENL (`zenlGemm` + bias) | mat2 必须是 LocalMem；否则 **RuntimeError**；支持同上 aligned 配置；`F.linear` 内部调用 |
| `linear` | ZENL (`zenlGemm`) | weight 必须是 LocalMem；否则 **RuntimeError**；支持任意 input 维度；支持同上 aligned 配置 |
| `bmm` | **CPU Roundtrip** | 无 ZENL batched GEMM kernel（LocalMem 和非 LocalMem 均 fallback） |
| `mv` | **CPU Roundtrip** | 无 ZENL matrix-vector kernel |
| `addmv` | **CPU Roundtrip** | 同上 |
| `baddbmm` | **CPU Roundtrip** | 无 ZENL batched GEMM kernel |

### 权重打包（LocalMem）

```python
import torch_zeus.zeus as zeus

model = MyModel().to('zeus')
zeus.pack_weights(model, Tr=4, Tc=4)   # → weights 转为 ZeusLocalMemTensor
output = model(input)                   # → 透明调度到 ZENL GEMM kernel
```

`ZeusLocalMemTensor` 是 `torch.Tensor` subclass，通过 `__torch_dispatch__`
拦截 GEMM 算子并将其路由到 C++ ZENL kernel；非 GEMM 算子透明地降级到 GDG（线性）内存。
用户不需要直接调用 `to_local_mem()` 或 `wrap_as_dispatch_tensor()`。

## Triton kernel 文件索引

| Triton 文件 | 对应算子 | 支持 dtype | 备注 |
|-------------|---------|-----------|------|
| `zenl/src/triton/add.py` | `add` | f32 / bf16 / int8 / fp8e4m3 | fp8e4m3 / int8-saturation 仅 Triton 直调可用，ATen 不经此路径 |
| `zenl/src/triton/sub.py` | `sub` | f32 / bf16 / int8 / fp8e4m3 | 同上 |
| `zenl/src/triton/mul.py` | `mul` | f32 / bf16 / int8 / fp8e4m3 | 同上 |
| `zenl/src/triton/fill.py` | `fill_` / `zero_` | f32 / bf16 / int8 / fp8e4m3 | |
| `zenl/src/triton/reduce_sum.py` | `sum` / `mean` / `norm`（last-dim） | f32 / bf16 / int8 / fp8e4m3 | 合并自 reduce_sum_f32.py + reduce_sum_bf16.py |
| `zenl/src/triton/reduce_sum_mid.py` | `sum` / `mean`（mid-dim） | f32 / bf16 / int8 / fp8e4m3 | 合并自 reduce_sum_mid_f32.py + reduce_sum_mid_bf16.py |
| `zenl/src/triton/gemm.py` | `mm` / `addmm` / `linear` | bf16 | Triton 侧仍只有 dense 模式；aligned `1024KB` 目前由 host + sim 路径支持 |
| `zenl/src/triton/embedding.py` | `embedding` | bf16 / f32 | |

---

## 需要替换的 CPU Roundtrip 算子

### 计算类（优先级高）

这些算子有明确的计算语义，应实现 ZENL kernel。

| 算子 | 当前实现 | 替换方案 |
|------|----------|----------|
| `bmm` | 始终 CPU roundtrip | ZENL batched GEMM kernel |
| `mv` | 始终 CPU roundtrip | ZENL matrix-vector kernel 或复用 GEMM (M=1) |
| `addmv` | 始终 CPU roundtrip | 同上 |
| `baddbmm` | 始终 CPU roundtrip | ZENL batched GEMM kernel |

### 随机数类（优先级中）

所有随机算子都是 CPU roundtrip，需要设备端 RNG 实现。

| 算子 | 当前实现 | 替换方案 |
|------|----------|----------|
| `random_` | CPU 生成 + copy 回设备 | 设备端 PRNG kernel |
| `random_.to` | 同上 | 同上 |
| `random_.from` | 同上 | 同上 |
| `uniform_` | 同上 | 同上 |
| `normal_` | 同上 | 同上 |
| `bernoulli_.float` | 同上 | 同上 |
| `bernoulli_.Tensor` | 同上 | 同上 |
| `exponential_` | 同上 | 同上 |

## 设备级内存操作算子

这些算子使用 `zertMemcpy` / `zenlMemcpy` 做数据搬移，不涉及计算，状态正常。

| 算子 | 实现方式 | 备注 |
|------|----------|------|
| `copy_` | `zertMemcpy` (H2D/D2H) / `zenlMemcpy` (D2D) | contiguous 直接 memcpy；non-contiguous 走 storage roundtrip |
| `_to_copy` | 委托给 `copy_` | |
| `_copy_from` | 委托给 `copy_` | |
| `_copy_from_and_resize` | 委托给 `copy_` | |
| `clone` | `empty_like` + `copy_` | |
| `_local_scalar_dense` | `zertMemcpyAsync(D2H)` + stream sync | |
| `resize_` | 存储重分配 + `zenlMemcpy` (D2D) | 扩容时迁移旧数据；缩小只改元数据不释放 storage |

## 纯元数据操作算子

无数据搬移，仅操作 TensorImpl 的 size/stride/offset，状态正常。

| 算子 | 备注 |
|------|------|
| `empty.memory_format` | Zeus allocator 分配存储 |
| `empty_strided` | Zeus allocator 分配存储 |
| `zeros` / `ones` / `full` | `empty` + `fill_`（→ `zenlFill`）组合；存储分配为 Metadata，填值为 ZENL |
| `view` | 计算新 stride，共享存储 |
| `_reshape_alias` | 委托给 PyTorch native |
| `as_strided` / `as_strided_` | 设置任意 size/stride/offset |
| `set_` (4 个变体) | 存储和元数据操作 |
| `is_set_to` | 检查存储共享 |
| `record_stream` | 内存管理标记 |
| `cat` / `cat.out` | 分配输出 + `narrow` (view) + `copy_` |

## 已知问题

| 问题 | 状态 | 描述 |
|------|------|------|
| `copy_` non-contiguous dest | 已修复 | Zeus→Zeus 路径中 dest 为 non-contiguous view 时 flat memcpy 写错位置，影响 `cat(dim≠0)` |
| add/sub/mul fp8e4m3 无法走 ATen dispatch | 上游设计限制 | PyTorch 将 `Float8_e4m3fn` 定位为 storage-only dtype。`torch.add(fp8, fp8)` 在 `TensorIterator::build() → compute_types()` 阶段即被拒绝（`NotImplementedError: "add_stub" not implemented for 'Float8_e4m3fn'`），**早于** `REGISTER_PRIVATEUSE1_DISPATCH` 注册的 backend 函数被调用。这是 PyTorch 框架对所有 backend（含 CUDA）的统一限制——fp8 标准用法为量化存储 + 专用算子（如 `torch._scaled_mm`），不走通用 element-wise 路径。ZENL kernel 和 Triton kernel 本身完整支持 fp8e4m3 计算，可通过 ZENL C API 或 Triton 直调使用 |
| add/sub/mul int8 溢出语义差异 | 设计限制 | ATen TensorIterator 路径为 wrapping（与 CPU 一致），ZENL kernel / Triton 路径为 saturation（clamp 到 [-128, 127]） |
| fp8e4m3 sim `f_to_fp8e4m3` 截断 | 已修复 | 原实现直接截断 mantissa 低位，已改为 IEEE 754 round-to-nearest-even |
| `index_put_` 功能不完备 | 待实现 | 对比 MLU CNNL 实现缺少：(1) accumulate=True 支持 (2) bool mask 索引 (3) AdvancedIndexing（None 维度、子空间转置、`make_info`）(4) 非 contiguous self 支持 (5) overlap 检测 (6) 跨设备 value 自动搬运。当前覆盖 SGLang 推理的主要场景（contiguous tensor + int64 索引 + 非累加 scatter write） |
| Structured ops 两级分发 | 已修复（2026-04-14 迁移至 codegen） | PyTorch structured kernels（add/sub/mul/sum 等）需要两级分发：Level 1 = `TORCH_LIBRARY_IMPL` 在 PrivateUse1 key 注册 `.Tensor`/`._`/`.out` 三个 variant 的 wrapper；Level 2 = `REGISTER_PRIVATEUSE1_DISPATCH` 注册 kernel 函数。Level 1 的 wrapper 需要 override `set_output_strided/raw_strided` 把输出分配到 Zeus device，然后调用 `op.meta()` + `op.impl()`，impl 内部通过 DispatchStub 走到 Level 2 注册的 Zeus kernel。**此前**手写 `StructuredOpsDispatch.cpp` 维护 9 个 op × 多 variant 的样板代码，每加一个 structured op 要改多处，不可扩展。**现在**通过 `codegen/zeus_functions.yaml` 声明 + `gen_zeus_stubs.py` 生成 `generated/RegisterZeus.cpp`，参考 `torch_mlu` 流水线实现，详见下方「Structured Op Codegen 流水线」章节 |

---

## Structured Op Codegen 流水线

> 2026-04-14 引入。参考 `torch_mlu/codegen/` 架构，为 structured ops 的 Level 1 dispatch 注册代码生成替代手写 `StructuredOpsDispatch.cpp`。

### 目录结构

```
codegen/
  zeus_functions.yaml           # 单一真相源：声明 Zeus 支持的 aten ops
  gen_zeus_stubs.py             # 入口：调用 torchgen 解析 YAML
  dest/
    gen_external_zeus.py        # 核心生成器：RegisterZeus / StructuredRegisterZeus / GenExternalZeus
  templates/
    RegisterZeus.cpp            # C++ 模板，含 helper（create_out/resize_out/check_inplace/maybe_create_proxy）和 op_call<> fallback wrapper
    ZeusFunctions.h             # namespaced 声明模板
    KernelFunctions.h           # kernel 声明模板（用于 zenl_kernel.h）

torch_zeus/csrc/aten/generated/  # 生成输出（.gitignore，不入库）
  RegisterZeus.cpp               # 19 个 m.impl 注册 + 结构化 op wrapper 类
  ZeusFunctions.h
  zenl_kernel.h                  # 当前为空（只在需要自定义 impl 时才填充）
```

### 生成流程

1. `setup.py` 的 `ZeusBuildExtension.build_extensions()` 在 **build_ext 阶段**（不是 import 阶段）调用 `run_codegen()`，触发：
   ```bash
   python -m codegen.gen_zeus_stubs \
       -s codegen/zeus_functions.yaml \
       -o torch_zeus/csrc/aten/generated/
   ```
   `run_codegen()` 使用 mtime 短路：对比 `RegisterZeus.cpp` 与 `codegen/zeus_functions.yaml` / `gen_zeus_stubs.py` / `gen_external_zeus.py` / `templates/*` 的 mtime，新时跳过。**注意**：stamp file 只用 `RegisterZeus.cpp`，因为 torchgen 的 `FileManager` 在内容不变时不会重写文件，辅助输出（`ZeusFunctions.h` / `zenl_kernel.h`）的 mtime 可能是陈旧的，若用它们做 `min()` 短路会永远失效
2. `gen_zeus_stubs.py` 使用 `torchgen.gen.parse_native_yaml` 解析 PyTorch 的 `native_functions.yaml`，与 `zeus_functions.yaml` 交叉匹配，对 structured op 从 `backend_indices[DispatchKey.CPU]` 取 metadata（kernel name）
3. `RegisterZeus` / `StructuredRegisterZeus` 生成 3 个 target：
   - `ANONYMOUS_DEFINITION` — wrapper class（继承 `at::native::structured_{kernel}`）+ `impl_{name}` inner 函数 + `wrapper_{name}` fallback 外壳
   - `REGISTRATION` — `m.impl("aten::xxx", TORCH_FN(wrapper_xxx));`
   - `NAMESPACED_DEFINITION` — `torch_zeus::zeus::xxx()` C++ 调用接口
4. 生成的 `RegisterZeus.cpp` 被 `setup.py` 纳入 sources 编译

### 生成内容（Phase 1 覆盖 9 ops / 19 variants）

| YAML 声明 | 生成 m.impl |
|-----------|-------------|
| `add.Tensor` / `add_.Tensor` / `add.out` | `add.Tensor` / `add_.Tensor` / `add.out` |
| `sub.Tensor` / `sub_.Tensor` / `sub.out` | `sub.Tensor` / `sub_.Tensor` / `sub.out` |
| `mul.Tensor` / `mul_.Tensor` / `mul.out` | `mul.Tensor` / `mul_.Tensor` / `mul.out` |
| `sum.dim_IntList` / `sum.IntList_out` | `sum.dim_IntList` / `sum.IntList_out` |
| `mean.dim` / `mean.out` | `mean.dim` / `mean.out` |
| `argmax` / `argmax.out` | `argmax` / `argmax.out` |
| `argmin` / `argmin.out` | `argmin` / `argmin.out` |
| `norm.ScalarOpt_dim_dtype` / `norm.dtype_out` / `norm.out` | `norm.ScalarOpt_dim_dtype` / `norm.dtype_out` / `norm.out` |

### Wrapper class 生成逻辑

- **复用 PyTorch 类**（默认路径）：生成的 wrapper class 继承 `at::native::structured_{cpu_kernel_name}`（例如 `structured_ufunc_add_CPU`、`structured_sum_out`、`structured_argmax_out`）。override `set_output_strided/raw_strided` 把输出分配到 Zeus device，`op.impl(...)` 内部通过 `add_stub/sub_stub/sum_stub/...` 这条 DispatchStub 派发到 Level 2 注册的 Zeus kernel——所以复用 CPU 的 class 是安全的
- **TensorIteratorBase 特殊处理（bridge ownership）**：对 `structured_inherits: TensorIteratorBase` 的 op（add/sub/mul），codegen wrapper 在 `op.impl()` 之前注入 `TensorIteratorBridge iter_bridge; iter_bridge.to_build(op, "name");`，kernel 侧（`operators/zenl/add.cpp` 等）**直接从 iter 读取 operands**，不得再次调用 `TensorIteratorBridge::to_build`（否则会 double-bridge）。对于 reduce op（sum/mean/argmax/argmin/norm），由于 YAML 中不是 `structured_inherits: TensorIteratorBase`，codegen 不注入 bridge，kernel 侧（`operators/zenl/reduceOps.cpp`）**需自行构建 bridge**——这是 structured vs. reduce 的 bridge ownership 不对称点
- **自定义 class 扩展点**：YAML 支持 `override_meta` / `override_impl` 开关；一旦设置，codegen 会在 `zenl_kernel.h` 生成 `structured_{kernel}_zeus` 的 forward declaration，开发者可在 C++ 代码中提供自定义 `impl`。Phase 1 未使用该机制

### Fallback 行为

生成的 `wrapper_{name}` 通过 `op_call<Return>(impl_call, fallback_call, args...)` 包装，inner impl 名为 `impl_{name}`：

```cpp
static const bool enable_zeus_fail_fallback = []() {
  const char* env = std::getenv("ZEUS_DISABLE_FAIL_FALLBACK");
  if (!env) return true;
  std::string s(env);
  return !(s == "1" || s == "ON" || s == "on" || s == "true");
}();

template <typename Return, typename F1, typename F2, typename... Args>
static Return op_call(F1 impl_call, F2 fallback_call, Args&&... args) {
  if (!enable_zeus_fail_fallback) {
    return impl_call(std::forward<Args>(args)...);
  }
  try {
    return impl_call(std::forward<Args>(args)...);
  } catch (const std::exception& e) {
#ifdef ZEUS_DEBUG_FALLBACK
    std::cerr << "[ZEUS FailFallback] kernel threw, delegating to CPU: " << e.what() << std::endl;
#endif
    return fallback_call(std::forward<Args>(args)...);
  }
}
```

`fallback_call` 展开为 `at::native::call_fallback_fn<&zeus_fail_fallback, ATEN_OP2(add, Tensor)>::call`，最终由 `ZEUSFallback.cpp` 中的 `zeus_fail_fallback` 委托给全局 `zeus_fallback`（即 PyTorch 的 `at::native::cpu_fallback`）。这让 kernel 内的 `TORCH_CHECK` 抛出（例如 dtype/shape 不支持）自动降级到 CPU 而不是硬 error。

**调试开关**：设置 `ZEUS_DISABLE_FAIL_FALLBACK=1`（或 `ON` / `on` / `true`）后 `op_call<>` 停止 try/catch，让原始异常直接抛出到用户栈——定位 `TORCH_CHECK` 触发点 / 看完整 backtrace 时非常有用（此前 fail fallback 会把异常信息吞掉只剩下 CPU 结果）。参考 MLU 的 `getFailFallbackEnabledEnvVar()` 模式

### 新增一个 structured op 的工作量

1. 在 `codegen/zeus_functions.yaml` 加 3 行（`.Tensor` / `._` / `.out`）
2. 在 `operators/zenl/xxx.cpp` 写 kernel + `REGISTER_PRIVATEUSE1_DISPATCH(xxx_stub, &zenl_xxx_kernel)`
3. `python setup.py build_ext --inplace`

对比此前手写流程需要改 5 个文件（StructuredOpsDispatch.cpp 的 3 处 + 2 处 helper + include），工作量降一个数量级

### 与 `torch_mlu` 的差异

| 项 | MLU | Zeus |
|---|---|---|
| kernel 库 | BANG + MLUOP 两套，需 `--use_bang/--use_mluop` 开关 | 只有 zenl，无开关 |
| 自定义命名空间 | 有（vision / audio / custom），需 `parse_mlu_custom_yaml` | 无，仅 `aten:` 一节 |
| `override_meta` / `override_impl` | 对 CPU!=CUDA metadata 的 op 总生成自己的 class | 简化：仅判断 CPU metadata 存在，默认复用 PyTorch 的 class，只有显式写 `override_*` 才生成自定义 class |
| backward stubs | 生成 backward kernel 声明 | Phase 1 不涉及 |
| unstructured ops | 全面接入 | Phase 1 仅 structured；`gen_unstructured` 代码路径已写好但当前未触发，预留 Phase 2 |

### 2026-04-14 打磨批次

第一轮 codegen 流水线落地后，对本轮 commit 做了一次 review，下表 9 项已全部修复：

| # | 项 | 修复内容 |
|---|---|---|
| 1 | binary op double-bridging | `operators/zenl/add.cpp` / `sub.cpp` / `mul.cpp` 移除 kernel 内的 `TensorIteratorBridge iter_bridge; iter_bridge.to_build(...)`，改为直接 `iter.output(0)` / `iter.input(0/1)`——因为 codegen wrapper 已经在 `op.impl()` 之前 build 过 bridge，kernel 再 build 一次是重复工作 |
| 2 | bridge ownership 代码注释缺失 | `operators/zenl/add.cpp` 顶部加了一段长注释解释 bridge 所有权规则：binary op（`structured_inherits: TensorIteratorBase`）的 bridge 由 codegen wrapper 构建，kernel 不得重复 bridge；`sub.cpp` / `mul.cpp` 指向 add.cpp 作为规范。`operators/zenl/reduceOps.cpp` 的 `reduce_stub()` 注释反向说明：reduce ops 不继承 `TensorIteratorBase`，所以 codegen 不注入 bridge，kernel 必须自行 build——防止后续再出现 double-bridge 或漏 bridge |
| 3 | `ZeusFunctions.h` / `NAMESPACED` 是死代码 | `gen_zeus_stubs.py` 移除 `ZeusFunctions.h` + `NAMESPACED_DEFINITION` / `NAMESPACED_DECLARATION` 的写入，并主动删除陈旧文件；`codegen/templates/ZeusFunctions.h` 模板删除；`codegen/dest/gen_external_zeus.py` 的 `Target` 枚举、`RegisterZeus` / `StructuredRegisterZeus` 的 NAMESPACED 分支、`CppSignature` / `CppSignatureGroup` 导入全部清理；`codegen/templates/RegisterZeus.cpp` 去掉 `namespace zeus { ${dispatch_namespaced_definitions} }` 块。`torch_zeus::zeus::xxx()` 入口无调用方，走 (b) 方案保持 codegen 输出精简 |
| 4 | `zenl_kernel.h` 空文件 | `gen_zeus_stubs.py` 扫描 `aux` 判断是否有任何 op 声明了 `override_meta` / `override_impl`；无 override 时不再写 `zenl_kernel.h`，同时把模板里 `#include "aten/generated/zenl_kernel.h"` 替换为 `${kernel_declarations_include}` 并传空串——Phase 1 下该文件彻底从生成输出树中消失；若之前存在陈旧的 `zenl_kernel.h`（比如曾经开启过 override）会在下次 codegen 运行时被主动删除。一旦有 override 声明，自动恢复写入 + 注入 include |
| 5 | `create_out` dispatcher 往返 | 模板 `RegisterZeus.cpp` 的 `create_out` 不再调 `at::empty` / `at::empty_strided`，改为直接取 `ZEUSCachingAllocator::getAllocator()` + `at::detail::empty_{generic,strided_generic}`，并用 `TORCH_INTERNAL_ASSERT_DEBUG_ONLY` 断言 `options.device()` 是 `PrivateUse1`（structured wrapper 里由 `guard_.reset_device` 保证）。`maybe_create_proxy` 转为复用 `create_out` 同步受益。fast path 跳过了 `at::empty → aten::empty.memory_format → TensorFactory.cpp::empty_memory_format` 的 dispatcher 往返。参考 MLU `torch_mlu/csrc/aten/operators/cnnl/internal/create_out.cpp` |
| 6 | fail fallback 缺少调试开关 | 模板 `RegisterZeus.cpp` 新增 `ZEUS_DISABLE_FAIL_FALLBACK` 环境变量门，`op_call<>` 根据该开关走 try/catch 或直通（见上方「Fallback 行为」） |
| 7 | `SKIP_FALLBACK_OPS` 语义注释缺失 | `gen_external_zeus.py` 里 `SKIP_FALLBACK_OPS` 前加了长注释说明准入规则：任何被 `at::native::cpu_fallback` 内部依赖的 op（`empty.memory_format` / `empty_strided` / `copy_` / `_copy_from_and_resize`，这些在 cpu_fallback 里用于 CPU staging + data copy）都必须 bypass `op_call<>` 壳，否则 fail fallback 会递归触发自身形成死循环；另外这些 op 的真实签名（SymIntArrayRef / 定制 copy 机制）与 codegen wrapper 签名不兼容。同时在 `SKIP_DEVICE_GUARD_OPS` 旁加了 `_copy_from_and_resize` 跨设备拷贝的说明 |
| 8 | codegen 在 import 阶段执行 | `setup.py` 新增 `ZeusBuildExtension(BuildExtension)`，在 `build_extensions()` 里触发 `run_codegen()`；`pip install` / `egg_info` / `sdist` 等不再支付生成开销。同步添加 mtime 短路（stamp file 为 `RegisterZeus.cpp`，对比 YAML + 生成器 + 模板的 max mtime） |
| 9 | `wrapper_wrapper_*` 命名冗余 | `gen_external_zeus.py` 将 inner 函数前缀从 `wrapper_` 改为 `impl_`，fallback 外壳从 `wrapper_wrapper_` 改为 `wrapper_`，既与 PyTorch `torchgen` 惯例一致也更易读 |

### 待完善项

- **Phase 2**：将简单 unstructured op（`embedding`、`index_put`、`fill_` 等）纳入 codegen；`gen_unstructured` 分支已就绪
- **Phase 3**：复杂 op（`copy_`、GEMM 系）暂保持手写，视后续需要再迁移
- **回归保护**：YAML 里的 schema 字段手抄自 PyTorch `native_functions.yaml`，升级 PyTorch 版本时需确认 structured 关键字、overload 名称未变
