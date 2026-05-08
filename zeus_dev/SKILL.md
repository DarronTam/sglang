---
name: sgl-kernel-zeus-dev
description: Develop a new operator in sgl-kernel-zeus end-to-end — find a CUDA reference, produce the five artifacts (triton reference kernel, sim.c, host cpp, python api, tests) plus a docs/ markdown, all aligned with Zeus idioms. Use whenever the user asks to port / add an operator into sgl-kernel-zeus (e.g. "给 sgl-kernel-zeus 加一个 rmsnorm/silu_and_mul/moe_align_block_size/..."), refactor an existing one to the Zeus block_ptr + CORE_NUM + explicit-for-loop pattern, or update one of its five artifacts and the docs in lockstep.
---

# sgl-kernel-zeus 算子开发 Skill

sgl-kernel-zeus 是 SGLang 在 Zeus NPU 上的 kernel 库（对标 CUDA 侧的 `sgl-kernel`）。一个算子的完整交付包含 **五份代码 + 一份文档**，任意一份变动都要同步到其它份。本 skill 规范这整套流程。

## 何时用

- 用户要求给 sgl-kernel-zeus 加/改一个算子（porting、fuse、接口对齐）。
- 用户要求把已有算子改造成 Zeus 习惯（`make_block_ptr` / `CORE_NUM` / 显式 for 循环）。
- 用户更新了 triton 参考实现，需要级联更新 sim.c / 文档。

不要单独改五件套里某一份而不同步其它份。

## 算子的五件套 + 文档

目录约定（以算子名 `<op>`、类别 `<cat>` ∈ {`elementwise`, `moe`, `attention`, `norm`, ...} 为占位）：

| 角色 | 路径 | 作用 |
|---|---|---|
| Triton 参考 | `sgl-kernel-zeus/csrc/<cat>/<op>_kernel.py` | Zeus-friendly 参考实现，porting 蓝本；**当前不进执行路径**，仅供硬件后端 porting 和文档展示 |
| CPU sim | `sgl-kernel-zeus/csrc/<cat>/sgl_<op>_sim.c` | 实际跑的 kernel（zecc 编译成 `.zbin`），**必须和 triton 行为一致** |
| Host wrapper | `sgl-kernel-zeus/csrc/<cat>/<op>_zeus.cpp` | TORCH_CHECK + 结构体打包 + `zertLaunchKernel` |
| Python API | `sgl-kernel-zeus/python/sgl_kernel_zeus/<cat>.py` | 调用入口 |
| Tests | `sgl-kernel-zeus/tests/test_<op>.py` | 正向 + 拒绝用例，参考值用 pure torch 算 |
| Docs | `sgl-kernel-zeus/docs/<op>.md` | 严格按 `docs/silu_and_mul.md` / `docs/moe_sum_reduce.md` 的章节结构 |

另外可能需要动：
- `sgl-kernel-zeus/csrc/common_extension.cpp`（`TORCH_LIBRARY` 注册）
- `sgl-kernel-zeus/include/sgl_kernel_zeus_ops.h`（函数声明）
- `sgl-kernel-zeus/setup.py`（`.zbin` 构建规则）
- `sgl-kernel-zeus/python/sgl_kernel_zeus/__init__.py`（re-export）

改动时先看仓库里同类算子怎么做，照搬结构。

## 标准工作流

### Step 1 — 找 CUDA 参考（不要 from scratch）

优先顺序：

1. **SGLang 本家**：`/root/project/sglang/sgl-kernel/csrc/**/*.cu`。名字对得上的直接对照。
2. **flashinfer**：纯 kernel 库，attention / norm / sampling 往这里找。
3. **vLLM**：`vllm/csrc/**/*.cu` 或 `vllm/_custom_ops.py`。MoE、activation、quant 类有参考。
4. **triton 官方 tutorials / triton-inference-server**：如果前三处都没有，找最接近的单 kernel。

**如果目标算子是几个已有 op 融合出来的**（比如 `moe_sum_reduce` 带 shared-expert 残差 fuse），那就找**最接近的单 op 作为骨架**，fuse 部分在 triton 和 sim.c 里对齐补上，并在文档 §1 "算子定位" 里写清相对 CUDA 扩展了什么。

记下这些信息开始写代码：
- CUDA 源路径
- 输入输出张量形状 / dtype
- 数学语义（一条公式写清楚）
- 有没有 in-place、有没有可选参数

### Step 2 — 写 Triton 参考实现（`<op>_kernel.py`）

允许从 CUDA 做"直译"起步，但必须按下面的 **Zeus 习惯清单**改造：

1. **显式 for 循环代替 grid**
   - CUDA 里 `blockIdx.x` 维度的并行 → Zeus 里用 `for t_block in range(T_blocks)` 等显式循环。
   - `grid = (CORE_NUM,)`，program_id 只代表"哪个核"。

2. **CORE_NUM 编译期常数**
   - 声明 `CORE_NUM: tl.constexpr`，Zeus 当前默认 2。
   - 在**最能均匀切分**的维度上切：例如 `silu_and_mul` 切 `dim`，`moe_sum_reduce` 切 `H`，`moe_align_block_size` 只有单核就不切。
   - 推导 `*_per_core = axis // CORE_NUM`，派生 `core_*_offset = core_id * *_per_core`。
   - host 侧要加 `TORCH_CHECK(axis % CORE_NUM == 0, ...)`。

3. **指针用 `tl.make_block_ptr` 描述**
   - 每个张量建一个 `*_block_ptr_template`（`offsets=(0, 0, ...)` 在原点）。
   - 循环内用 `tl.advance(template, (absolute_offsets...))` 走到本 tile，再 `tl.load / tl.store`。
   - load/store 配 `boundary_check=(...)` + `padding_option="zero"`，尾部不整除自动掩码。
   - 两个张量如果共享物理 buffer 的不同半区（如 `silu_and_mul` 的 `[x | y]`），用**同一 strides + base ± offset** 的两套模板描述——这是 Zeus 里表达"gate/up 合并张量"或类似交错布局的关键手法。

4. **OOB 非零 sentinel：禁用 raw-pointer load，改用 `block_ptr + tl.where`**

   Zeus 芯片**不支持** `tl.load(ptr + offs, mask=mask, other=X)` 的 raw-pointer 形态——该路径在中端会退化成 scalar scatter/gather（每 lane 一次标量 DMA），带宽灾难级。**所有 load 必须走 `make_block_ptr` 形态。**

   OOB 填 `0` 时直接 `padding_option="zero"` 即可。OOB 需要**非零 sentinel**（如 `-2.0`、`-inf`）时，用 `make_block_ptr + tl.where` 重建：

   ```python
   # ❌ 禁用：raw-pointer + other sentinel，Zeus 中端退化为 scalar load
   t = tl.load(ptr + offs, mask=(offs < N), other=-2.0)

   # ✅ 正确：block_ptr 先补零，再 tl.where 换成目标 sentinel
   tpl = tl.make_block_ptr(
       base=ptr,
       shape=(1, N),          # 1D 数据包成 [1, N] 2D 视图（Zeus V3 对纯 1D 支持偏弱）
       strides=(N, 1),
       offsets=(0, 0),
       block_shape=(1, BLOCK_N),
       order=(1, 0),
   )
   blk     = tl.advance(tpl, (0, base))
   t_raw   = tl.load(blk, boundary_check=(0, 1), padding_option="zero")    # OOB = 0
   in_bnd  = (tile_n[None, :] + base) < N                                  # [1, BLOCK_N] bool
   t       = tl.where(in_bnd, t_raw, tl.full((1, BLOCK_N), -2.0, tl.float32))  # OOB = -2.0
   ```

   代价：多 2 条向量 ALU 指令（compare + select），Zeus 上向量 ALU 单 cycle 且与 load 并流，**性能差异可忽略**，换来 structured lowering 路径的稳定性。

   多维 tile（`[BLOCK_M, BLOCK_N]`）时 `in_bnd` 需对两维同时 compare：
   ```python
   in_bnd = (offs_m[:, None] < M) & (offs_n[None, :] < N)
   ```

   详细分析见 `sgl-kernel-zeus/docs/raw_ptr_to_block_ptr_sentinel.md`。

5. **规避 where / bool 向量操作（计算逻辑中）**
   - Zeus 向量 ALU 对 bool / where 不友好。判等、计数 hit 用 **`tl.relu(1 - tl.abs(a - b))`** 得 `{0, 1}` 权重向量；具体例子见 `csrc/moe/moe_align_block_size_kernel.py` Step 2 / Step 5。
   - 需要 bin-count：`hit = tl.relu(1 - tl.abs(bin_ids - e)); count += tl.sum(hit)`。
   - 需要 scatter 路由：`packed = (values * hit) @ permutation_matrix`。
   - 注：`tl.where` 用于 OOB sentinel 补填（见第 4 条）是**允许的**，这里的"规避"针对计算逻辑中的条件选择。

6. **GEMM 权重内存标签**
   - GEMM 的右矩阵（weight）在 Zeus 上放专用 DRAM，triton block_ptr 里需要指明 memory type（`weight` / 等）——具体 API 以仓库里已落地的 GEMM kernel（如 `csrc/moe/moe_grouped_gemm_kernel.py` 或 `docs/gemm_normal_dense_*.py`）为准，照搬它的标注方式。

7. **精度约定**
   - bf16 I/O，fp32 内部累加/运算：`.to(tl.float32)` 后算，最后 `.to(tl.bfloat16)` RNE 写回。
   - 有 reduce 的算子：fp32 累加器 `tl.zeros((...), dtype=tl.float32)`。

8. **分块超参**
   - `BLOCK_T` / `BLOCK_H` / `BLOCK_D` / `BLOCK_K` 等都做 `tl.constexpr`；默认值保守（`BLOCK_T=4~16`，`BLOCK_H/D=128`），便于 porting 时再调。

9. **单核 kernel 约定**
   - 若算子本质串行（如 `moe_align_block_size`）或不值得并行，写成 `grid=(1,)` 单核，仍保留 `_ = tl.program_id(0)` 一行。

10. **循环融合：对同一数据块的多次操作合并成一次遍历**

   **触发条件**：kernel 里出现两个（或更多）顺序 for 循环，且它们操作**同一份数据**——即第二个循环读取的 tile 与第一个循环读取过的 tile 完全重合。

   **判断方法**：对每个中间 tile，问自己：
   - "此时这个 tile 的值还在寄存器里吗？"
   - "如果是，下一阶段需要它吗？"
   - "如果两个答案都是 yes → 直接复用寄存器，删掉第二次 load。"

   **典型模式（causal_conv1d_update 例）**：
   ```python
   # ❌ 两次遍历：第一遍累加，第二遍重新 load 做状态移位
   for k in static_range(Ks):
       s_k = load(state[:, :, k])      # 第一次 load
       acc += s_k * w_k[None, :]

   for k in static_range(Ks):         # 第二遍：重复 load 同一批 tile
       new_val = load(state[:, :, k+1]) if k < Ks-1 else x_val
       store(state[:, :, k], new_val)

   # ✅ 单次遍历：s_k 从寄存器直接复用，省去 K-2 次 DRAM tile 读
   for k in static_range(Ks):
       s_k = load(state[:, :, k])      # 读一次
       acc += s_k * w_k[None, :]
       if k >= 1:                       # 编译期分支
           store(state[:, :, k-1], s_k.to(bf16))   # 从寄存器写回
   store(state[:, :, Ks-1], x_val.to(bf16))        # 单独处理尾列
   ```

   **注意安全性**：融合前确认"写目标"与"后续读来源"不重叠：
   - 若循环写 `state[k]` 而后续还要读 `state[k]`（读写同一地址）→ **不能融合**。
   - 若循环写 `state[k-1]`，而下一迭代读 `state[k]`（写目标落后于读来源）→ **可以融合**，
     因为 `state[k]` 在被写之前就已经作为 `s_k` 读完并驻留寄存器。

   **sim.c 对比**：CPU sim 通常预先把所有 state 列读进 `float old_state[Ks]` 再统一写回，
   天然已是最优（每元素读一次）。triton 里的循环融合是对应的向量化等价——**sim.c 不需要
   因此改动，但 triton 每次改完都要确认 sim.c 数值仍然对齐**。

11. **只能依赖两类 Zeus 扩展，不要依赖改过的 Python 运算符**
   Zeus compiler 扩展惯例只覆盖两种形态：
   - **新 `tl.xxx` / `tl.zeus.xxx` intrinsic**：前端新加函数，后端配新 MLIR op。当前已落地的扩展面包括：
     - `tl.topk(x, k)` — 返回 packed `[V0..V{k-1}, I0..I{k-1}]`（老契约，已有 kernel 在用）。
     - `tl.zeus.topk(x, k=..., axis=-1)` — 返回 `(values, ids)` tuple，带 axis kwarg，更贴近芯片原生 top-K 接口。新 kernel 可直接用。
     - `tl.zeus.sigmoid(x, approx="fast")` — 硬件 sigmoid，`approx="fast"` 走快速近似路径（bf16 I/O 内部 fp32）。比 `tl.sigmoid` 或 `1/(1+exp(-x))` 展开更贴近硬件。
     - `tl.gather(x, indices, axis=...)` — 按索引抽取，支持 axis 参数。
   - **已有 op 加属性**：如 `make_block_ptr` 的 `memory="weight"` kwarg，落到 `tt.make_tensor_ptr` 的 attribute 上。

   两类 topk 共存期内的选择原则：新 kernel 倾向 `tl.zeus.topk`（tuple 返回，无需 helper 解 packed）；动老 kernel 时保留原来的 `tl.topk` + helper 不必强行迁移，但**同一算子的 triton / sim.c / docs 契约描述必须一致**。

   **绝对不要依赖那些"默默改 Python 运算符语义"的写法**，典型反例：

   - `x_tile[bt, :]`：上游 `tl.tensor.__getitem__` 只接受 `None` / `slice(None)`，不接受 `int` / `tl.constexpr int`。TTIR 也没有"按静态下标抽行"的通用 op（`tt.split` 只能对最后一维按 2 等分）。这种写法前端会 `raise ValueError`，下降不到 TTIR。
   - 别的 `__getitem__` / `__iter__` / `reduce` 等隐式糖——出错点藏在用户代码里、没有显式新 op，不是 Zeus 喜欢挑的扩展方向。

   正确做法：要 per-token / per-row 处理，就用 per-row block_ptr + `tl.advance(template, (t, 0))` + `load` + `reshape`——这能干净下降到 `tt.make_tensor_ptr` / `tt.advance` / `tt.load`：

   ```python
   x_row_blk_template = tl.make_block_ptr(
       base=input_ptr, shape=(T, E), strides=(E, 1),
       offsets=(0, 0), block_shape=(1, E), order=(1, 0),
   )
   for bt in tl.static_range(TOKENS_PER_CORE):
       t = t_start + bt
       if t < T:
           cur_x = tl.advance(x_row_blk_template, (t, 0))
           x = tl.load(cur_x, boundary_check=(0, 1), padding_option="zero").reshape((E,))
   ```

   判断一条写法是否在扩展边界内的经验法则：**能不能在 AST 上指出一个具体的 `tl.xxx` 调用 / 一个具体的 kwarg？** 能 → 属于合法扩展面；只是改 Python 符号行为 → 不属于。

### Step 3 — 写 sim.c（**必须跟 triton 行为一致**）

sim.c 是 triton 行为的 **CPU simulator**，不是独立的 reference 实现。它的目的：让上层 host/测试在真 kernel 编译器还没接通时也能跑数值对照。

- 结构体 `struct Sgl<Op>Args` 字段和 host cpp 里的**逐字节对齐**。
- 入口签名固定：`void zenl_sgl_<op>_kernel_sim(void** args)`，`args[0]` 强转结构体指针。
- 只负责产出正确数值，**不必复刻 triton 的并行结构**——scalar 嵌套 for 循环把每个输出元素算对即可。
- bf16 用 `bf16_to_float` / `float_to_bf16` helper（参考 `sgl_silu_and_mul_sim.c` 里已有的 RNE 实现）。
- 每次改 triton → **立即回来改 sim.c**，确保两者最终输出 bit-近似一致（bf16 允许 1 ULP）。测试跑过即视为对齐通过。
- **sim.c 改完必须重新装包再跑测试**。`.c` 文件由 zecc 编译成 `.zbin` 嵌进 C++ 扩展，只改源码不装包，`torch.ops.sgl_kernel_zeus.*` 还是老 `.zbin`，测试看上去通过的是旧行为。硬流程：

  ```bash
  cd sgl-kernel-zeus
  pip install -e . --no-build-isolation     # 重新编 .zbin + 装包
  pytest tests/ -v                          # 或只跑受影响的 tests/test_<op>.py
  ```

  `--no-build-isolation` 用现有 venv 的工具链、避免重新下载构建依赖；CI / 本地环境都这样跑。改完 sim.c 不装包就提交 = 交了一份没被验证过的文件。

### Step 4 — 写 host wrapper（`<op>_zeus.cpp`）

模板：

```cpp
#include "sgl_kernel_zeus_ops.h"
#include "zeus_runtime.h"
#include "core/ZEUSStream.h"

extern "C" { extern const unsigned char zenl_kernel_sgl_<op>_data[]; }

struct Sgl<Op>Args { /* 与 sim.c 一一对应 */ };

void <op>(at::Tensor& out, at::Tensor& input /*, ...*/) {
  TORCH_CHECK(input.is_contiguous(), ...);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, ...);
  // 其它 shape / dim % CORE_NUM / 可选参数同形 校验
  Sgl<Op>Args kargs{ /* 打包指针 + 标量 */ };
  void* args[] = { &kargs, nullptr };
  auto stream = torch_zeus::getCurrentZEUSStream(out.device().index());
  zertLaunchKernel(zenl_kernel_sgl_<op>_data, /*grid=*/1, args,
                   static_cast<zertStream_t>(stream));
}
```

当前 sim 路径 `grid = 1`；切到真 triton zbin 时同步改成 `CORE_NUM`，并补 `axis % CORE_NUM == 0` 校验（如果 kernel 有切分）。

别忘了：
- `include/sgl_kernel_zeus_ops.h` 加函数声明
- `csrc/common_extension.cpp` 加 `m.def("<op>(...) -> ()")` 和 `m.impl("<op>", torch::kPrivateUse1, &<op>)`

### Step 5 — 写 Python API（`python/sgl_kernel_zeus/<cat>.py`）

- 转发到 `torch.ops.sgl_kernel_zeus.<op>.default(...)`。
- `out` 可选时自动按 `input.shape[:-1] + (dim,)` 分配。
- 在 `python/sgl_kernel_zeus/__init__.py` re-export 并追加到 `__all__`。

### Step 6 — 写测试（`tests/test_<op>.py`）

必备：

1. **正向多形状**：典型 `(batch, hidden)` 组合 + 目标模型的实战 shape（GLM-4.7: `H=5120`，DeepSeek-V2: `topk=9` 等）。
2. **参考值用 pure torch fp32 算**，结果 `.to(bf16)`，`atol/rtol ≈ 1e-2`。
3. **拒绝用例**：非 bf16 / 非 contiguous / 不整除 CORE_NUM / shape 不匹配；`pytest.raises(RuntimeError, match=...)`。
4. 如果有可选参数（如 shared residual、scale），覆盖 "带/不带" 两条路径。

运行方式在文件头写一行 `Run: pytest tests/test_<op>.py -v`。

### Step 7 — Triton 语法自检（Level 1）

在 `<op>_kernel.py` 末尾加自检块（参考 `csrc/elementwise/activation_kernel.py`）：

```python
if __name__ == "__main__":
    expected_args = [...]
    expected_constexprs = [...]  # constexpr 参数在 arg_names 里的索引
    jit = sgl_<op>_kernel_bf16
    assert jit.arg_names == expected_args
    assert list(jit.constexprs) == expected_constexprs
    print(f"OK: {jit}")
```

跑法：`python <op>_kernel.py`。这层检查只验证：

- Python AST 解析通过
- `@triton.jit` 接受源码
- 参数表 / constexpr 位置没漂

真正的 IR 编译得在后端接通后才跑得了；本 skill 范围内**做到 Level 1 通过即可**。

### Step 8 — 写文档（`docs/<op>.md`）

严格按 `docs/silu_and_mul.md` 的章节骨架：

1. **标题 + 对应代码路径列表**（五件套 + 测试）
2. **§1 算子定位**：上下游语境、fuse 了什么、相对 CUDA 扩展/裁剪了什么
3. **§2 输入输出与语义**：2.1 输入 / 2.2 输出 / 2.3 维度约束

   **§2 核心要求：每个维度必须写清语义，不能只写形状符号。**
   例子对比：

   | ❌ 只写形状 | ✅ 维度 + 语义 |
   |---|---|
   | `x : [N, C] bf16` | `x : [N, C] bf16 — 当前 decode token；N = decode batch 大小；C = num_heads × head_dim（head 折进 C，不单独成维）` |
   | `conv_state : [N, C, Ks] bf16` | `conv_state : [N, C, Ks] bf16 — rolling window；Ks = K−1（K=卷积核宽度）；in-place 更新` |
   | `weight : [C, K] bf16` | `weight : [C, K] bf16 — per-channel 一维卷积核；K 通常 4` |

   需要特别说明的几类情况：
   - **折叠维度**：如 `C = num_heads × head_dim`（head 隐式编码在 C 里），要在表格注释里点出 head 映射关系（`head = c // head_dim`）。
   - **派生尺寸**：如 `Ks = K − 1`、`D_kv = qk_nope_head_dim + v_head_dim`，要写出公式。
   - **上下文依赖含义**：如 `N` 在 decode 路径 = batch size（每请求 1 token），在 prefill 路径 = total tokens（各请求 token 拼成 1D），需分路径说明。
   - **in-place / side-effect**：凡是被原位修改的张量（如 `conv_state`），在表格里加 **原位更新** 标注，并在 §2.2 输出列中再次列出（以 side-effect 形式）。
4. **§3 数据流**：
   - §3.1 端到端宏观视图（ASCII 图示输入→输出的整体变换 + 核级切分归属）
   - §3.2 具体例子（取一组与默认常量对齐的小 shape，列表展示每个核每轮 tile 的 `(t_start, ...)` 和读写位置）
   - §3.3 单 tile 计算过程（advance → load → fp32 算 → RNE 写回）
5. **§4 编译期常量 & 中间变量**：
   - §4.1 Constexpr 表：名称 / 默认 / **作用与选择取向**
   - §4.2 运行时派生变量表：名称 / 公式 / **含义及行为细节**（约束、边界兜底、绝对 vs. 相对偏移等）
6. **§5 三层职责**：Python API / host wrapper / kernel 层各做什么
7. **§6 测试**：覆盖面 + 参考值公式
8. **§7 向芯片原生算子 porting 时的注意事项**：grid 对齐 / 整除校验 / 内存布局 / 精度顺序 / tile 形状 / scratchpad 等

写作语气保持中文、短句、表格优先。

## 同步更新纪律

**任何一次 triton 参考实现的改动，必须同时处理：**

1. ✅ `sgl_<op>_sim.c` — 行为对齐（这是 simulator，不是独立实现）
2. ✅ `docs/<op>.md` — §3 数据流 / §4 中间变量表如有新变量或新策略同步更新
3. ✅ `<op>_kernel.py` 末尾的 Level 1 自检块（参数表 / constexpr 索引改了要更新）
4. ✅ 如果改了常量默认值或校验约束 → host `TORCH_CHECK` 同步
5. ✅ **sim.c 改动后必装包再测**：`pip install -e . --no-build-isolation` + `pytest tests/ -v`（或 `tests/test_<op>.py -v`）。`.c` 源码不经 zecc 重编 → 还是老 `.zbin` → 测试看到的是旧行为。跳过这步 = 提交未验证的文件。

三者（triton / sim.c / 文档）之中任何一份落后，都视为半成品。

## 常见坑

- **`tl.load(ptr+offs, mask, other=X)` raw-pointer load**：Zeus 中端无法将其折叠为 structured DMA，退化成 scalar scatter/gather，每 lane 一次标量 load，带宽极差。一律改成 `make_block_ptr + padding_option="zero"` + `tl.where` 补 sentinel（见 Step 2 第 4 条）。
- **多阶段循环未融合**：kernel 对同一 tile 做了两次遍历（第一遍读做计算，第二遍重新读做写回），第二次 load 完全是冗余的——第一遍的寄存器值还在。写完 triton 后主动检查：是否有两个 for 循环操作同一份数据？能否把第二个循环的写操作提前到第一个循环内（在寄存器驻留期间）完成？
- **host `grid = 1` vs. triton `grid = CORE_NUM`**：当前 sim 路径单核跑，porting 到真 zbin 时切成 `CORE_NUM`——这是最容易漏改的一条。
- **`dim % CORE_NUM == 0`**：triton 里整数除法不补齐尾部，host 忘加这个校验会在奇数 `dim` 上悄悄漏写。
- **`[x | y]` 合并张量的 stride**：行跨度是 `2 * dim` 不是 `dim`，否则会把另一半当自己的数据读。
- **bf16 直接做 sigmoid**：误差会大 10× 以上，必须先升 fp32。
- **sim.c 漂了**：triton 更新后忘同步 sim.c，测试通过但上真机崩。把"同步更新"当硬规则。
- **sim.c 改完没 `pip install -e . --no-build-isolation`**：`.c` 不重编 `.zbin` 就装不进扩展，`torch.ops.sgl_kernel_zeus.*` 继续跑老 kernel。`pytest` 看起来 pass 的是旧行为。改 sim.c 的硬流程：编辑 → 装包 → 跑测。
- **`x_tile[bt, :]` 这种整数轴下标**：Level 1 自检能过（只做 AST 解析），但真编译时 `tl.tensor.__getitem__` 在前端就 `raise`，到不了 TTIR。改成 per-row block_ptr + `advance` + `load` + `reshape`。详见 Step 2 第 9 条。

## 快速起步清单

拿到一个新算子需求，按顺序过一遍：

```
[ ] 找到 CUDA 参考源（sglang / flashinfer / vllm）
[ ] 明确输入/输出 shape & dtype & 数学公式；**每个维度写清语义**（折叠维度如 C=H×D 要说明，派生尺寸如 Ks=K-1 要写公式，in-place 张量要标注）
[ ] 决定切分轴（dim? H? topk?）— 选最均匀、无 cross-tile 依赖的
[ ] 检查循环融合机会：kernel 里是否有多个阶段对同一 tile 操作？若是，把后续阶段的写操作提前到寄存器驻留期内完成，省去重复 DRAM 读（详见 Step 2 第 10 条）
[ ] 写 <op>_kernel.py（block_ptr 模板、CORE_NUM、显式 for、fp32 中间、relu 小把戏）
[ ] 末尾加 if __name__ == "__main__" 自检块；python <op>_kernel.py 过 Level 1
[ ] 写 sgl_<op>_sim.c（scalar 嵌套 for，与 triton 语义对齐）
[ ] 写 <op>_zeus.cpp + 头文件声明 + common_extension.cpp 注册
[ ] 写 python/sgl_kernel_zeus/<cat>.py + __init__.py 导出
[ ] 写 tests/test_<op>.py（正向 + 拒绝用例）
[ ] 写 docs/<op>.md（对照 silu_and_mul.md 结构）
[ ] pip install -e . --no-build-isolation  # 让 sim.c 的改动生效为新 .zbin
[ ] pytest tests/test_<op>.py -v            # 数值对齐
[ ] sim.c 后续再改？每次都要重复上面两步
```
