# GLM5 Next MoE CPU Bounce Analysis

分析对象：

- 测试脚本：`zeus_dev/dev_glm5_next_moe_test.py`
- 环境：`torch_zeus_py12_torch10`
- 重点 stage：
  - `moe_block_e2e`
  - `moe_block_e2e_prod_vs_dev`

## 实测结论

打开 Zeus fallback 统计后运行：

```bash
ENABLE_ZEUS_FALLBACK_TO_CPU=1 ZEUS_DUMP_FALLBACK=1 \
  /root/miniconda3/envs/torch_zeus_py12_torch10/bin/python \
  zeus_dev/dev_glm5_next_moe_test.py --stage moe_block_e2e --num-tokens 16
```

结果：

```text
moe_block_e2e: PASS
Fallback call counts:
  aten::silu : 1
```

再运行：

```bash
ENABLE_ZEUS_FALLBACK_TO_CPU=1 ZEUS_DUMP_FALLBACK=1 \
  /root/miniconda3/envs/torch_zeus_py12_torch10/bin/python \
  zeus_dev/dev_glm5_next_moe_test.py --stage moe_block_e2e_prod_vs_dev --num-tokens 16
```

结果：

```text
moe_block_e2e_prod_vs_dev: PASS
Fallback call counts:
  aten::silu : 1
  aten::mul_ : 1
```

因此，当前测试里的 CPU bounce 可以分成两类：

1. 测试/诊断需要的显式 `.cpu()`，不属于生产 forward 计算。
2. Zeus 计算路径触发的 ATen CPU fallback，目前实测只有 `aten::silu` 和 `aten::mul_`。

## CPU Bounce 明细

| 类型 | 位置 | 做什么 | 是否生产计算关键路径 | 处理建议 |
| --- | --- | --- | --- | --- |
| REF golden `.cpu()` | `dev_glm5_next_moe_test.py:223-230` | 把 REF 侧输入、权重、topk 结果搬到 CPU，调用 `_ref_moe_core` 做 pure torch golden | 否 | 保留。proxy 测试故意让 REF 可秒级跑完 |
| 诊断 `.cpu().item()` | `dev_glm5_next_moe_test.py:282` | 读取 `num_tokens_post_pad` 打印 block 数 | 否 | 保留或仅在 verbose/debug 下执行 |
| 打印 `.cpu().tolist()` | `dev_glm5_next_moe_test.py:326` | 打印 Zeus 输出前几个元素 | 否 | 保留或仅在 verbose/debug 下执行 |
| compare `.cpu()` | `dev_glm5_next_moe_test.py:337`，prod compare 同类 | 把 Zeus 输出搬回 CPU 做断言 | 否 | 测试必须有，生产无关 |
| `aten::silu` fallback | `dev_glm5_next_moe_test.py:258-261` | shared experts MLP 中手写 `F.silu(gate) * up` | 是，dev pipeline 中是 Zeus 计算路径 | 用已有 `sgl_kernel_zeus.silu_and_mul` 替代 |
| `aten::mul_` fallback | `deepseek_v2.py:695-702` | production `forward_normal` 对 MoE routed 输出做 `final_hidden_states *= routed_scaling_factor` | 是，prod path 中是计算路径 | 优先把 scale 融入 MoE kernel/topk，而不是为这条路径单独走 CPU |

## 具体任务拆解

### 1. REF golden CPU bounce

脚本在 REF 路径中显式调用：

```python
moe_core_ref = _ref_moe_core(
    x_ref.cpu(),
    w13_ref.cpu(), w2_ref.cpu(),
    w_ref.cpu(), ids_ref.cpu(),
    mI=mI,
)
```

这部分任务是：用小 proxy shape 在 CPU 上跑一个可读、可对齐的 MoE reference，包括 per-token/per-expert 的两段 GEMM、SiLU、topk 权重乘法和 topk reduce。

这不是生产路径问题，也不应该用 Zeus ATen 替换。它的存在价值是提供 golden。

### 2. shared experts 的 SiLU CPU fallback

dev Zeus path 当前手写：

```python
sh_silu_z = (
    torch.nn.functional.silu(sh_gu_out_z[:, :mI].float())
    * sh_gu_out_z[:, mI:].float()
).to(torch.bfloat16)
```

实测触发：

```text
aten::silu : 1
```

这一步做的是 GLU 激活：

```text
out = silu(gate_half) * up_half
```

当前已有 Zeus fused kernel：

- Python API：`sgl_kernel_zeus.silu_and_mul`
- SGLang production wrapper：`SiluAndMul.forward_zeus`
- 源码位置：`python/sglang/srt/layers/activation.py:82-85`

因此这部分不需要新 ATen kernel。直接用已有 fused op 更合适：

```python
sh_silu_z = sgl_kernel_zeus.silu_and_mul(sh_gu_out_z)
```

注意：当前手写代码显式 `.float()` 后再 cast bf16；替换前需要确认 `sgl_kernel_zeus.silu_and_mul` 的 bf16 I/O 和内部累加/舍入语义是否与期望 tolerance 一致。按现有 MoE core 中 `C1_z -> silu_and_mul -> C1_silu_z` 已经在同一测试中 PASS，预计风险低。

### 3. production `mul_` CPU fallback

prod path 在 `DeepseekV2MoE.forward_normal` 中有：

```python
if (
    not _is_cuda
    and not _is_xpu
    and not _use_aiter
    or isinstance(self.experts.quant_method, KTEPWrapperMethod)
):
    final_hidden_states *= self.routed_scaling_factor
```

在 Zeus 上这会触发：

```text
aten::mul_.Tensor
```

fallback 原因不是没有任何 `mul` ATen，而是当前 `torch_zeus` 的 `zenl_mul_kernel` 要求两侧都是同 device 且 `numel` 完全一致。PyTorch 对 `tensor *= python_float` 会走 TensorIterator scalar/broadcast 形态，其中 scalar 侧不是同 shape Zeus tensor，于是 `zenl_mul_kernel` 抛错并委托 CPU fallback。

已有 `add.cpp` 已经为 scalar add 做了专门 fast path；`mul.cpp` 目前没有类似 scalar fast path。

对 GLM5 Next MoE 来说，不建议优先为这条路径新写通用 ATen scalar-mul。更好的处理是把 `routed_scaling_factor` 融入已有 MoE kernel 链：

- `biased_grouped_topk(..., apply_routed_scaling_factor_on_output=True)`：把 scale 融进 topk weights。
- `moe_grouped_gemm(..., topk_weights=..., mul_routed_weight=True)`：gemm2 输出时乘 routed weight。
- `moe_sum_reduce(..., routed_scaling_factor=...)`：reduce 阶段统一乘 scale。

当前 dev pipeline 已经采用第一种归属：topk weights 中融入 scale，`moe_sum_reduce` 传 `routed_scaling_factor=1.0`。这也是最贴近当前测试注释的方式。

如果 production Zeus runner 无法保证 topk 已经融合 scale，则建议在 Zeus MoE runner 收口处改为调用 `moe_sum_reduce(routed_scaling_factor=self.routed_scaling_factor)`，避免 Python 层 `final_hidden_states *= scalar`。

## 是否可以用目前 ATen 算子处理？

结论：

- `linear/mm/add/copy/to/empty/view/reshape/contiguous` 等当前路径用到的基础 ATen 基本已经有 Zeus 实现或未触发 fallback。
- shared experts 的 `silu * mul` 不建议拆成现有 ATen。`aten::silu` 当前没有 Zeus 实现，但已有更合适的 `sgl_kernel_zeus.silu_and_mul`。
- production 的 scalar `mul_` 当前 ATen 不完整，不能无 bounce 处理 `tensor *= float`。但 MoE 路径不需要依赖它，可以把 scale 合入现有 MoE fused kernels。

## 是否需要开发新的 ATen kernel？

按优先级：

1. 不需要为 shared experts 开发 `aten::silu`。使用已有 `sgl_kernel_zeus.silu_and_mul` 即可，还能减少一个中间 op。
2. 不建议为了 GLM5 MoE 优先开发通用 `aten::mul_.Tensor` scalar fast path。MoE 的 scale 本来就应归属 topk/gemm/reduce fused pipeline。
3. 如果希望提高 torch_zeus 通用 ATen 覆盖率，可以给 `mul.cpp` 增加类似 `add.cpp` 的 scalar fast path：
   - `tensor * scalar`
   - `tensor *= scalar`
   - scalar 为 CPU 0-dim tensor 或 Python float/int 包装形态
   - dtype 至少覆盖 bf16/fp32/int32

这属于通用能力增强，但不是当前 GLM5 Next MoE 消除 CPU bounce 的首选路径。

## 是否和其他操作合并？

建议合并。

### shared experts

合并 `silu + mul`，使用已有：

```python
sgl_kernel_zeus.silu_and_mul(gate_up)
```

不建议先做 `gate_up GEMM + silu_and_mul` 更大融合。原因是当前已有 fused activation kernel，改动小、风险低；GEMM+activation 融合会牵涉 LocalMem packed weight、输出布局、down GEMM 输入契约，收益需要单独 profiling 证明。

### routed scaling

合并到 MoE fused pipeline，不保留独立 `final_hidden_states *= scalar`：

- 最优先：topk weights 内融合 scale。
- 备选：gemm2 乘 topk weight 时融合。
- 备选：`moe_sum_reduce` 阶段融合。

不要让 routed scale 落到 Python 层 ATen scalar `mul_`，否则即使补了 ATen，也多一个 kernel launch。

## 建议落地顺序

1. 修改 dev pipeline shared experts，把手写 `F.silu(...)*...` 改成 `sgl_kernel_zeus.silu_and_mul(sh_gu_out_z)`。
2. 跑 `moe_block_e2e`，确认 fallback 统计从 `aten::silu : 1` 变成无真实计算 fallback。
3. 检查 production Zeus MoE runner 的 scaling 归属，确保 `forward_normal` 不再执行 line 702 的 scalar `mul_`，而是由 topk/gemm/reduce 之一承担。
4. 跑 `moe_block_e2e_prod_vs_dev`，目标 fallback 统计中不再出现 `aten::mul_`。
5. 如果后续其他模型也频繁出现 `tensor *= scalar`，再补 `torch_zeus/csrc/aten/operators/zenl/mul.cpp` 的 scalar fast path。

## 当前判断

当前 GLM5 Next MoE 测试里，真正值得处理的 CPU bounce 很少：

- `aten::silu`：dev 手写路径问题，用已有 fused op 解决。
- `aten::mul_`：production scaling 归属问题，应该并入 MoE fused kernel 链。

测试中的 `.cpu()` 主要是 golden、打印、断言，不是生产性能问题。
