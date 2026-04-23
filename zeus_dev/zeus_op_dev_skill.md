# Zeus 算子开发指引

这份指引覆盖两条算子开发链路：

1. `zenl`：基础算子库，面向 `torch aten` / `torch_zeus` 集成。
2. `sgl-kernel-zeus`：面向 `sglang` 部署场景的算子库，最终通过 torch 自定义 op 打到 Python 层。

目标不是只说明“文件放哪里”，而是把每次开发都要重复做的事情固定下来：先确认功能和签名，再做 kernel / host / PyTorch 接入，最后用测试和状态文档收尾。

## 1. 先确认功能，再开始写代码

在动手前，先和开发者对齐下面 4 件事。这里不能省，因为很多 Zeus 算子当前只需要覆盖 ATen 对应 op 的一个子集。

### 1.1 先确认功能范围

至少明确：

- 这个算子要对齐哪个 PyTorch op / schema。
- 当前版本必须支持哪些输入形态、dtype、layout、维度组合。
- 哪些功能暂时不做，必须明确标记为 `unsupported`。
- ATen 语义是否允许做 Zeus 特化约束，例如只支持 contiguous、只支持部分 dtype、只支持非广播场景等。

### 1.2 先确认签名

签名确认时，必须同时对照：

- `torch native`
- `torch_mlu`

推荐参考位置：

- `torch native`：upstream `aten/src/ATen/native/` 下对应实现，例如 `BinaryOps.cpp`、`Embedding.cpp`、`ReduceOps.cpp`。
- `torch_mlu`：`/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/`
- `torch_mlu internal`：`/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/internal/`

当前仓库里已经有一些直接对齐 `torch_mlu` 分层方式的实现，可先参考：

- `torch_zeus/csrc/aten/operators/zenl/embedding.cpp`
- `torch_zeus/csrc/aten/operators/zenl/reduceOps.cpp`
- `torch_zeus/csrc/aten/operators/zenl/internal/embedding_internal.cpp`
- `torch_zeus/csrc/aten/operators/zenl/internal/zenl_internal.h`

### 1.3 先确认 Zeus 侧约束

Zeus 设备侧开发目前按下面约束收敛：

- 支持的核心 dtype：`bf16`、`fp32`、`fp8e4m3`、`int8`、`int4`
- `int4` 不能按普通裸数据单独处理，必须配合 LocalMem 相关 API / packed layout 使用
- kernel 边界上的标量参数、shape、stride、循环变量、kernel args 字段，默认统一使用 `int32`
- 如果上层 API 来自 PyTorch，输入经常是 `int64`，应当在 op 层做范围检查后再下沉为 `int32`

不要把 PyTorch 前端的宽类型直接原样带到 Zeus kernel 边界。

### 1.4 输出一份最小开发约定

开工前建议至少写清楚下面这张表：

| 项目 | 要确认的内容 |
|---|---|
| 功能范围 | 和 PyTorch 对齐到什么程度 |
| 输入限制 | dtype / contiguous / broadcast / LocalMem 要求 |
| 输出语义 | 是否与 CPU 完全一致，哪些地方会降级或报错 |
| unsupported | 暂不支持的行为、报错方式 |
| 签名 | PyTorch schema、C++ internal 签名、ZENL / sgl kernel 签名 |
| 测试范围 | zenl 单测、`tests/test_op`、`sgl-kernel-zeus/tests` |

## 2. 总体分层

### 2.1 `zenl` 链路

`zenl` 面向基础算子，主要服务 `torch_zeus` 的 ATen 对接。

典型分层如下：

```text
torch_zeus/csrc/aten/operators/zenl/*.cpp
  ↓
torch_zeus/csrc/aten/operators/zenl/internal/*.cpp
  ↓
zenl/include/zenl.h
  ↓
zenl/src/host/*.cpp
  ↓
zenl/src/sim/*_sim.c
zenl/src/triton/*.py
```

### 2.2 `sgl-kernel-zeus` 链路

`sgl-kernel-zeus` 面向 SGLang 运行时，最终通过 torch 自定义 op 暴露到 Python。

典型分层如下：

```text
sgl-kernel-zeus/python/sgl_kernel_zeus/*.py
  ↓
torch.ops.sgl_kernel_zeus.*
  ↓
sgl-kernel-zeus/csrc/common_extension.cpp
  ↓
sgl-kernel-zeus/csrc/**/xxx_zeus.cpp
  ↓
sgl-kernel-zeus/include/sgl_kernel_zeus_ops.h
  ↓
sgl-kernel-zeus/csrc/**/xxx_kernel.c
sgl-kernel-zeus/csrc/**/xxx_kernel.py
```

### 2.3 矩阵计算的额外要求

如果算子涉及矩阵计算，尤其是下面这些场景：

- `mm` / `addmm` / `linear` / `bmm` / `mv` / `addmv`
- 直接读取或写入 LocalMem weight
- 需要按 `(k, n)` 访问 pack 后的权重
- 任何依赖 tiled weight / aligned-space / `Tr` / `Tc` / `aligned_size` 的 kernel

实现前必须同时参考下面三份文档：

- `docs/skill/zeus_weight_skill.md`
- `docs/localmem_runtime_api.md`
- `docs/zeus_weight_layout_runtime.md`

原因是这类算子不能只按普通 row-major 矩阵理解，必须结合 LocalMem 的专门 layout 和当前 `torch_zeus` 的 dispatch 方式一起实现。

至少要先确认下面几点：

- `nn.Linear.weight` 原始是 `(N, K)`，pack 后是 `(K, N)`，后续不能再按 `(N, K)` 理解
- Dense、no-padding、aligned-space 是不同布局分支，寻址和 kernel 分发条件不同
- `Tr` / `Tc` / `aligned_size` / `alignedSpaceBytes` 会直接影响物理地址计算和 host dispatch
- `int4` 不能按普通线性矩阵处理，必须结合 packed `uint8` + LocalMem layout 一起实现
- `torch_zeus` 里 LocalMem tensor 的分发、fallback、`to_gdg()`、`pack_weights()` 也会影响最终调用链

如果矩阵类算子没有把这三份文档一起看完，通常很容易在下面几处出错：

- 权重方向写反
- 把 Dense 和 PATCH 布局混为一谈
- 忽略 LocalMem dispatch，导致 ATen 路径和底层 kernel 假设不一致
- 用普通连续内存公式直接访问已经 pack 的 weight

## 3. `zenl` 算子开发流程

### 3.1 需要补的文件

新增一个 `zenl` 算子时，至少检查下面几个位置：

- `zenl/src/host/`：host 分发与 launch
- `zenl/src/sim/`：sim kernel
- `zenl/src/triton/`：Triton 参考实现
- `zenl/include/zenl.h`：公开 API 声明
- `zenl/tests/`：单测

如果是库内公共类型或工具有变化，再同步看：

- `zenl/include/zenl_types.h`
- `zenl/include/zenl_api.h`

### 3.2 Host 层职责

`zenl/src/host/*.cpp` 不是简单透传。它通常要负责：

- 参数校验
- dtype / layout / 子功能分发
- 将一个上层算子拆成多个 kernel 子功能
- 打包 kernel args
- 调用 `zertLaunchKernel(...)`

现有实现可以直接参考：

- `zenl/src/host/optensor.cpp`：按 dtype 分发 add/sub/mul/fill
- `zenl/src/host/reduce.cpp`：按 reduce 模式和 shape pattern 分发多个子 kernel
- `zenl/src/host/gemm.cpp`：按 dtype、layout、`Tr/Tc`、`alignedSize` 分发多个 kernel

一个算子往往不止一个 kernel。常见拆分维度包括：

- 不同 dtype
- 不同 shape pattern
- 不同 layout
- 不同快路径 / fallback

不要强行让一个 kernel 覆盖所有场景，再把复杂度堆到 device 侧。

### 3.3 参数与 dtype 约束

Zeus 侧 kernel args 建议遵循下面规则：

- 指针直接传 device pointer
- shape / stride / numel / loop trip count 用 `int32`
- enum / mode 也优先压成 `int32`
- 只在 ATen / PyTorch 边界保留 `int64`

建议模式是：

1. op 层接收 PyTorch 的 `int64`
2. 做合法范围检查
3. internal / host 层转换为 `int32`
4. kernel args 统一按 `int32` 组织

### 3.4 `sim.c` 命名规则

`sim` 文件名要显式区分“子功能 + dtype”，不要只保留一个模糊总名。

现有命名模式：

- `add_f32_sim.c`
- `reduce_sum_mid_bf16_sim.c`
- `gemm_bf16_dense_sim.c`
- `gemm_fp8e4m3_as1024_tr4_tc4_sim.c`

建议原则：

- 同一算子不同 dtype 拆成不同 `sim` 文件
- 同一算子不同 layout / 子路径也拆成不同 `sim` 文件
- 文件名直接反映 host 分发条件

这样 host 的 `extern const unsigned char zenl_kernel_xxx_data[];`、Triton 参考代码、测试命名都更容易对齐。

### 3.5 Triton 参考实现要求

`zenl/src/triton/*.py` 的主要价值有两点：

1. 给真实 Zeus Triton 编译链保留 source of truth
2. 给后续同事提供可读的参考实现

写 Zeus Triton 时，和 CUDA Triton 的思路有一个关键差异：

- CUDA Triton：很多工作由 `grid_x * grid_y` 的 block 调度隐式完成
- Zeus Triton：需要把 grid 层切分显式展开成 `core_num * for_loop_x * for_loop_y`

也就是原来 CUDA 里一个二维 grid 的任务，在 Zeus 上通常要拆成：

```text
core_id 负责固定的 core 维度
+ 显式 for 循环负责剩余 tile 维度
```

写法上建议遵循下面原则：

- 用 `tl.program_id(0)` 表示 core 维度
- 把其余切分显式写成 `for` / `tl.range`
- 优先用 `tl.make_block_ptr(...)` 描述输入输出块
- 对向量计算，不要依赖显式 cast 才能正确计算
- 默认按“硬件内部提升到 fp32”思路写计算
- 输出类型主要由输出变量 / `tl.store` 的目标 dtype 决定，常见为 `fp32` 或 `bf16`

特别注意：

- 不要把 CUDA 的 grid 映射直接照搬到 Zeus
- 不要把多核调度留给片上隐式机制，Zeus 这里要写成显式循环
- `int4` 路径如果需要读写 packed 数据，必须明确它和 LocalMem / packed layout 的关系，不能把 `int4` 当普通线性元素访问

### 3.6 `zenl.h` 与 host / sim 结构体一致性

每次新增 ZENL API 时：

1. 在 `zenl/include/zenl.h` 增加公开函数声明
2. 在 `zenl/src/host/*.cpp` 定义与 kernel 对齐的 args struct
3. 确认 host / sim / triton 对同一参数结构的字段顺序、类型、语义一致

尤其注意：

- 不同文件里同一个 args struct 的字段顺序必须完全一致
- 如果 host 压成了 `int32`，sim / triton 也必须按相同约定取值

### 3.7 `zenl` 测试要求

每个新增算子都要在 `zenl/tests/` 增加对应测试，覆盖至少：

- 基本功能
- 关键 dtype
- 边界 shape
- 关键快路径 / 分发分支
- 空 tensor / 小规模 / 大规模场景
- 明确不支持场景的报错

构建和测试命令：

```bash
cd zenl && make
cd zenl && make test
```

或者在仓库根目录：

```bash
make runtime
make zecc
make zenl
```

如果是首次编译，优先走根目录这组命令，确保 `runtime` 和 `zecc` 依赖已准备好。

## 4. `zenl` 接入 `torch aten` 的流程

`zenl` 算子完成后，通常还需要进一步集成到 `aten`，也就是 `torch_zeus/csrc/aten`。

### 4.1 需要关注的目录

- `torch_zeus/csrc/aten/operators/zenl/`
- `torch_zeus/csrc/aten/operators/zenl/internal/`

当前已经存在的代表性实现：

- `torch_zeus/csrc/aten/operators/zenl/add.cpp`
- `torch_zeus/csrc/aten/operators/zenl/embedding.cpp`
- `torch_zeus/csrc/aten/operators/zenl/reduceOps.cpp`
- `torch_zeus/csrc/aten/operators/zenl/internal/binary_internal.cpp`
- `torch_zeus/csrc/aten/operators/zenl/internal/embedding_internal.cpp`
- `torch_zeus/csrc/aten/operators/zenl/internal/reduce_internal.cpp`

### 4.2 两层职责划分

建议严格保持两层分工：

#### op layer: `operators/zenl/*.cpp`

职责：

- 对齐 PyTorch / ATen schema
- 输入合法性检查
- contiguous / memory format 处理
- output 分配
- TensorIteratorBridge 或 shape 推导
- 处理 Zeus 特有前置逻辑，例如 LocalMem 转换

#### internal layer: `operators/zenl/internal/*.cpp`

职责：

- 提取 data pointer
- 将 `at::ScalarType` 映射到 Zeus / ZENL dtype
- 获取 stream
- 调用 `zenlXxx(...)`

不要把 PyTorch 语义处理和裸 kernel 调用混在一个文件里。

### 4.3 注册方式

当前仓库主要有两种 ATen 接入模式：

- `REGISTER_PRIVATEUSE1_DISPATCH(...)`
  - 常用于依赖 `TensorIterator` 的通用算子
  - 参考：`torch_zeus/csrc/aten/operators/zenl/add.cpp`
- `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`
  - 常用于直接对接 ATen schema 的算子
  - 参考：`torch_zeus/csrc/aten/operators/zenl/embedding.cpp`

选哪种方式，取决于这个算子在 PyTorch 里本来是 stub 路径还是 library impl 路径。

### 4.4 对照 `torch native` 与 `torch_mlu`

这一步是必须项，不是可选项。

最少要比较下面几类内容：

- schema / 参数顺序
- dtype 规则
- broadcast / contiguous / keepdim / out 参数语义
- unsupported 场景的处理方式
- 是否需要和 PyTorch CPU 结果完全一致

推荐优先找两个参考：

- `torch native`：确认“规范应该是什么”
- `torch_mlu`：确认“设备后端通常怎样分层接入”

本地 `torch_mlu` 参考入口：

- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/add.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/sub.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/mul.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/fill.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/index_put.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/embedding.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/reduceOps.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/internal/embedding_internal.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/internal/index_put_internal.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/internal/fill_internal.cpp`
- `/home/tanzh/project/torch_mlu/torch_mlu/csrc/aten/operators/cnnl/internal/reduceOps_internal.cpp`

### 4.5 `tests/test_op` 端到端测试

ATen 接入完成后，必须在 `tests/test_op/` 增加完整测试。

现有测试示例：

- `tests/test_op/test_binary_ops.py`
- `tests/test_op/test_embedding.py`
- `tests/test_op/test_reduce_ops.py`
- `tests/test_op/test_gemm.py`
- `tests/test_op/test_index_put.py`

建议覆盖：

- PyTorch 前端真实调用路径
- 与 CPU / reference 对比
- 关键 dtype
- 关键形状
- 关键 unsupported 场景
- 如果有 Zeus 特化路径，也要验证分支是否真的走到

`torch_zeus` 重新构建与安装：

```bash
make torch_zeus
pip install -e . --no-build-isolation
```

如果本次修改同时影响 `runtime` 或 `zenl`，先补前置构建：

```bash
make runtime
make zenl
make torch_zeus
pip install -e . --no-build-isolation
```

推荐按算子定点执行：

```bash
pytest tests/test_op/test_xxx.py -v
```

## 5. `sgl-kernel-zeus` 算子开发流程

`sgl-kernel-zeus` 的 kernel 开发和 `zenl` 很像，但需要额外把自定义 op 暴露到 Python 层。

### 5.1 需要补的文件

通常需要同时改下面几类文件：

- `sgl-kernel-zeus/csrc/<domain>/xxx_kernel.c`
- `sgl-kernel-zeus/csrc/<domain>/xxx_kernel.py`
- `sgl-kernel-zeus/csrc/<domain>/xxx_zeus.cpp`
- `sgl-kernel-zeus/include/sgl_kernel_zeus_ops.h`
- `sgl-kernel-zeus/csrc/common_extension.cpp`
- `sgl-kernel-zeus/python/sgl_kernel_zeus/*.py`
- `sgl-kernel-zeus/python/sgl_kernel_zeus/__init__.py`
- `sgl-kernel-zeus/setup.py`
- `sgl-kernel-zeus/tests/test_xxx.py`

可直接参考当前 embedding 路径：

- `sgl-kernel-zeus/csrc/embedding/embedding_kernel.c`
- `sgl-kernel-zeus/csrc/embedding/embedding_zeus.cpp`
- `sgl-kernel-zeus/python/sgl_kernel_zeus/embedding.py`
- `sgl-kernel-zeus/tests/test_embedding.py`

### 5.2 C++ wrapper 层职责

`xxx_zeus.cpp` 通常负责：

- contiguous / dtype / shape 检查
- 必要的 dtype 转换
- 将 PyTorch 输入压成 Zeus kernel 可接受的参数
- 调用 `sgl_xxx_kernel_*`

参考：

- `sgl-kernel-zeus/csrc/embedding/embedding_zeus.cpp`

建议和 `zenl` 一样，kernel 边界统一使用 `int32` 控制参数；如果 Python 或 torch 输入是 `int64`，先在 wrapper 或 Python API 层转换 / 检查。

### 5.3 注册与 Python 暴露

新增算子时要打通下面三层：

1. 在 `sgl-kernel-zeus/include/sgl_kernel_zeus_ops.h` 增加 kernel 声明和 C++ wrapper 声明
2. 在 `sgl-kernel-zeus/csrc/common_extension.cpp` 里 `m.def(...)` + `m.impl(...)`
3. 在 `sgl-kernel-zeus/python/sgl_kernel_zeus/*.py` 提供 Python API，并在 `__init__.py` 导出

也就是最终调用链应当是：

```text
Python API
  -> torch.ops.sgl_kernel_zeus.xxx
  -> C++ wrapper
  -> Zeus kernel
```

### 5.4 `setup.py` 别漏

`sgl-kernel-zeus/setup.py` 里的 `sources=[...]` 需要把新增 `.cpp` / `.c` 文件显式加进去，否则构建时不会编译。

### 5.5 `sgl-kernel-zeus` 测试要求

每个新增算子都要在 `sgl-kernel-zeus/tests/` 增加 `.py` 测试，覆盖至少：

- 基本功能
- 典型 batch / shape
- 边界输入
- dtype / layout 限制
- Python API 的输入转换逻辑

构建与测试命令：

```bash
cd sgl-kernel-zeus
pip install -e . --no-build-isolation
```

或：

```bash
cd sgl-kernel-zeus
python setup.py build_ext --inplace
```

测试：

```bash
cd sgl-kernel-zeus
LD_LIBRARY_PATH=$(python -c "import torch; print(torch.__path__[0])")/lib:$LD_LIBRARY_PATH \
    pytest tests/ -v
```

也可以只跑单算子测试：

```bash
cd sgl-kernel-zeus
pytest tests/test_xxx.py -v
```

## 6. 每次开发完成后的收尾要求

### 6.1 必做验证

每完成一个算子，至少完成下面三层验证：

1. `zenl` 或 `sgl-kernel-zeus` 自身测试通过
2. 如果接入了 `torch_zeus`，`tests/test_op/test_xxx.py` 通过
3. 必要时补充端到端 smoke test，确认前端实际调用链打通

### 6.2 必须更新 `OP_STATUS.md`

每完成一次算子开发，都要更新：

- `tests/test_op/OP_STATUS.md`

更新内容至少包括：

- 当前算子是否已接入 ATen
- 当前走的是 `ZENL` / `Device` / `Metadata` / `CPU Roundtrip`
- 已支持的 dtype / layout / 关键限制
- 仍未支持的功能

如果本次只完成了底层 kernel，但还没接入 `torch_zeus`，也要在状态文档中明确写出“底层已完成，但尚未进入 ATen 调度 / 仍未对外可用”的状态，避免后续误判。

## 7. 推荐开发顺序

如果是一个新的 `zenl` 基础算子，推荐按这个顺序推进：

1. 和开发者确认功能范围、unsupported、签名
2. 对照 `torch native` 和 `torch_mlu` 整理语义
3. 设计 host 分发和 kernel 子功能拆分
4. 补 `zenl/src/sim/` 与 `zenl/src/triton/`
5. 补 `zenl/src/host/` 与 `zenl/include/zenl.h`
6. 补 `zenl/tests/` 并完成 `make test`
7. 接入 `torch_zeus/csrc/aten/operators/zenl` 与 `internal`
8. 补 `tests/test_op/test_xxx.py`
9. 重建并验证 `torch_zeus`
10. 更新 `tests/test_op/OP_STATUS.md`

如果是 `sgl-kernel-zeus` 算子，推荐顺序是：

1. 确认功能范围和 Python API 形式
2. 实现 `xxx_kernel.c` / `xxx_kernel.py`
3. 实现 `xxx_zeus.cpp`
4. 更新 `sgl_kernel_zeus_ops.h`
5. 更新 `common_extension.cpp`
6. 更新 Python 包装和 `__init__.py`
7. 更新 `setup.py`
8. 增加 `sgl-kernel-zeus/tests/test_xxx.py`
9. 构建并跑测试
10. 如果最终还要进入 `torch_zeus`，继续补 ATen 集成和 `OP_STATUS.md`

## 8. 一个简单判断标准

如果你准备提交一个新算子，但下面任一问题答不上来，说明开发还没收口：

- 这个算子当前到底对齐了哪一部分 PyTorch 语义？
- 哪些输入场景明确 unsupported？
- host 为什么要拆成这些子 kernel？
- sim / triton / host 的参数结构是否完全一致？
- `tests/test_op` 是否已经覆盖真实前端调用路径？
- `tests/test_op/OP_STATUS.md` 是否已经反映当前状态？

这几项都明确后，再交付代码，后续维护成本会低很多。
