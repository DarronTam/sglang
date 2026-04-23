# torch_zeus 原生算子支持分析（更新于 2026-04-15）

## 一、ZENL 原生内核算子

真正在 Zeus 设备上执行的原生内核，**无 CPU roundtrip**：

| 算子 | 文件 | 注册方式 | 关键限制 |
|------|------|---------|---------|
| **add** / add.Tensor / add_ / add.out | zenl/add.cpp + RegisterZeus.cpp | REGISTER_PRIVATEUSE1_DISPATCH(add_stub) + codegen | 支持 alpha 缩放 |
| **sub** / sub.Tensor / sub_ / sub.out | zenl/sub.cpp + RegisterZeus.cpp | REGISTER_PRIVATEUSE1_DISPATCH(sub_stub) + codegen | |
| **mul** / mul.Tensor / mul_ / mul.out | zenl/mul.cpp + RegisterZeus.cpp | REGISTER_PRIVATEUSE1_DISPATCH(mul_stub) + codegen | |
| **mm** | zenl/gemm.cpp | m.impl("mm") | mat2 必须 LocalMem |
| **addmm** | zenl/gemm.cpp | m.impl("addmm") | weight(K,N) 必须 LocalMem |
| **linear** | zenl/gemm.cpp | m.impl("linear") | weight 必须 LocalMem，自动 reshape |
| **embedding** | zenl/embedding.cpp | m.impl("embedding") | LocalMem 权重先转 GDG |
| **index_put_** / index_put | zenl/index_put.cpp | m.impl | accumulate=False，self 须 contiguous |
| **sum** / sum.dim_IntList / sum.IntList_out | zenl/reduceOps.cpp + RegisterZeus.cpp | REGISTER_PRIVATEUSE1_DISPATCH(sum_stub) | mode=ZENL_REDUCE_ADD |
| **mean** / mean.dim / mean.out | zenl/reduceOps.cpp + RegisterZeus.cpp | REGISTER_PRIVATEUSE1_DISPATCH(mean_stub) | mode=ZENL_REDUCE_AVG |
| **norm** (L1/L2/Lp) | zenl/reduceOps.cpp + RegisterZeus.cpp | REGISTER_PRIVATEUSE1_DISPATCH(norm_stub) | |
| **argmax** / argmax.out | RegisterZeus.cpp | codegen structured wrapper | 通过 ZENL reduce 内核 |
| **argmin** / argmin.out | RegisterZeus.cpp | codegen structured wrapper | 通过 ZENL reduce 内核 |
| **fill_.Scalar** | TensorOps.cpp | m.impl("fill_.Scalar") | 0 值用 memset；非零用 zenl_fill_internal（仅 fp32/bf16/int8/fp8） |

## 二、TensorOps 组合实现（设备端完成，无 CPU roundtrip）

| 算子 | 文件 | 实现方式 |
|------|------|---------|
| **cat** / cat.out | TensorOps.cpp | empty + narrow + copy_（设备端） |
| **zeros** | TensorOps.cpp | empty + zertMemsetAsync |
| **ones** | TensorOps.cpp | empty + fill_(1) |
| **full** | TensorOps.cpp | empty + fill_(value) |
| **zero_** | TensorOps.cpp | fill_(0) |
| **copy_** | TensorOps.cpp | zertMemcpy/zenlMemcpy，支持 LocalMem 双向 |
| **_to_copy** | TensorOps.cpp | empty_like + copy_ |
| **_copy_from** / _copy_from_and_resize | TensorOps.cpp | 委托 copy_ |

## 三、基础设施/元数据算子（设备端，零拷贝或低开销）

| 算子 | 文件 | 实现方式 |
|------|------|---------|
| **empty.memory_format** / empty_strided | TensorFactory.cpp | Zeus allocator |
| **view** / _reshape_alias | ViewOps.cpp | 纯元数据操作，零拷贝 |
| **clone** | ViewOps.cpp | empty_like + copy_ |
| **as_strided** / as_strided_ | AsStridedOps.cpp | 纯元数据操作 |
| **resize_** | Resize.cpp | 元数据 + allocator resize |
| **_local_scalar_dense** | ScalarOps.cpp | zertMemcpyAsync D2H + sync |
| **record_stream** | RecordStream.cpp | CachingAllocator 记录 |
| **set_** (多种变体) | StorageOps.cpp | 元数据操作 |

## 四、注册但实际 CPU roundtrip 的算子

⚠️ 这些算子虽在 gemm.cpp 中注册，但内部走 CPU fallback（无真正的 ZENL 内核）：

| 算子 | 文件 | 说明 |
|------|------|------|
| **bmm** | zenl/gemm.cpp | 无 ZENL batched GEMM 内核 |
| **mv** | zenl/gemm.cpp | 无 ZENL matrix-vector 内核 |
| **addmv** | zenl/gemm.cpp | CPU roundtrip |
| **baddbmm** | zenl/gemm.cpp | CPU roundtrip |

## 五、随机数算子（CPU 生成 + H2D 拷贝）

| 算子 | 文件 |
|------|------|
| uniform_, normal_, exponential_ | random/Uniform.cpp, Normal.cpp, Exponential.cpp |
| bernoulli_.float, bernoulli_.Tensor | random/Bernoulli.cpp |
| random_, random_.to, random_.from | random/Random.cpp |

## 六、确认无原生实现的算子（走全局 CPU Fallback）

| 算子 | 状态 | 说明 |
|------|------|------|
| **neg** | ❌ CPU fallback | ZeusOpParams 有参数注册但无 dispatch stub |
| **div** | ❌ CPU fallback | ZeusOpParams 有类型信息但无 ZENL 内核 |
| **max** / **min** | ❌ CPU fallback | reduceOps.cpp 无 max_stub/min_stub |
| **prod** | ❌ CPU fallback | 无 ZENL 实现 |
| **cumsum** / **cumprod** | ❌ CPU fallback | 仅 autocast fp32_set_opt_dtype |
| **where** | ❌ CPU fallback | 无原生注册 |
| **clamp** | ❌ CPU fallback | 无原生注册 |
| **arange** | ❌ CPU fallback | 无原生注册 |
| **index** (tensor[indices]) | ❌ CPU fallback | 无原生注册（仅 index_put 有） |
| **index_select** / gather / scatter | ❌ CPU fallback | 无原生注册 |
| **stack** / split / chunk | ❌ CPU fallback | 无原生注册 |
| **permute** / transpose / expand | ❌ CPU fallback | 无原生注册（但 as_strided 已注册） |
| **softmax** / log_softmax | ❌ CPU fallback | 仅 autocast |
| **relu** / gelu / silu | ❌ CPU fallback | 无内核 |

## 七、全局 CPU Fallback 机制

**文件**: `csrc/aten/ZEUSFallback.cpp`

- 通过 `TORCH_LIBRARY_IMPL(_, PrivateUse1, m)` 注册 `zeus_fallback` 捕获所有未注册算子
- 自动转换 Zeus 张量 → CPU → 执行 CPU 实现 → 转换回 Zeus
- 初始化检测算子（100ms 窗口内静默）：abs, ne, eq, mul, bitwise_and, masked_select, min, max, ceil, div, gt, lt, floor, sqrt
- 环境变量控制：`ZEUS_DISABLE_FALLBACK=1` 禁用，`ZEUS_DUMP_FALLBACK=1/2` 导出统计
- **这是所有未注册算子的隐式 CPU bounce 源头**

---

## 八、SGLang CPU Bounce 完整分析（基于 torch_zeus_sglang/sglang 实际代码）

共扫描 42 处 `_is_zeus` 引用，分布 13 个文件，27 个有效分支（含辅助函数）。

### 全部分支逐条分析

| # | 文件 | 行号 | 涉及算子 | 类型 | 可移除 |
|---|------|------|---------|------|--------|
| 1-A | memory_pool.py | 101-118 | `index_put`（高级索引赋值） | CPU bounce | ✅ zenl 原生 index_put |
| 1-B | memory_pool.py | 296-299 | `cat` | CPU bounce | ✅ TensorOps 原生 cat |
| 1-C | memory_pool.py | 671-676 | `cat` | CPU bounce | ✅ TensorOps 原生 cat |
| 2-A | forward_batch_info.py | 765-779 | `cat`, `zeros`, `full` | CPU bounce | ✅ 均有原生实现 |
| 2-B | forward_batch_info.py | 841-852 | `sub`, `arange` | CPU bounce | ⚠️ sub ✅ / arange ❌ |
| 2-C | forward_batch_info.py | 1082-1087 | `cumsum` | CPU bounce | ❌ cumsum 无原生 |
| 2-D | forward_batch_info.py | 1117-1118 | `cumsum` | CPU bounce | ❌ cumsum 无原生 |
| 2-E | forward_batch_info.py | 1250-1270 | `cat`, `arange`, `cumsum` | CPU bounce | ❌ cumsum/arange 无原生 |
| 2-F | forward_batch_info.py | 1281-1283 | `clamp`, `sub` | CPU bounce | ❌ clamp 无原生 |
| 3-A | common.py | 472-484 | `index`(2D), `sub`, `add` | CPU bounce | ⚠️ sub/add ✅ / index ❌ 无原生 |
| 3-B | common.py | 494-497 | `clone`（roundtrip 等效） | CPU bounce | ✅ clone 有原生实现 |
| 4-A | schedule_batch.py | 1775-1790 | `add`（+1） | CPU bounce | ✅ zenl 原生 add |
| 4-B | schedule_batch.py | 1840-1851 | `index`（tensor[ki]） | CPU bounce | ⚠️ index 无原生，走全局 fallback |
| 4-C | schedule_batch.py | 1890-1905 | `cat` ×3 | CPU bounce | ✅ TensorOps 原生 cat |
| 4-D | schedule_batch.py | 1910-1916 | `cat` | CPU bounce | ✅ TensorOps 原生 cat |
| 5-A | scheduler.py | 2061-2066 | `neg`（取负） | CPU bounce | ❌ neg 无原生 zenl 内核 |
| 6-A | overlap_utils.py | 21-28 | `where`, `clamp`, `neg`, `index` | CPU bounce | ❌ where/clamp 无原生 |
| 6-B | overlap_utils.py | 123-126 | `arange` | CPU bounce | ❌ arange 无原生 |
| 7-A | logits_processor.py | 420-451 | `cumsum`, `sub`, `arange`, `index` | CPU bounce | ❌ cumsum 无原生 |
| 7-B | logits_processor.py | 887-903 | `mm`（GEMM LocalMem 路由） | **功能性** | ❌ Zeus GEMM 路由 |
| 8-A | custom_op.py | 96-97 | 无（forward 分发） | **功能性** | ❌ 平台调度必需 |
| 9-A | loader.py | 678-694 | 无（lm_head 追踪） | **功能性** | ❌ pack_weights 依赖 |
| 9-B | loader.py | 706-741 | `pack_weights` 到 LocalMem | **功能性** | ❌ Zeus 初始化必需 |
| 10-A | rotary_embedding.py | 119-121 | 无（平台条件判断） | **功能性** | ❌ 平台路由 |
| 10-B | rotary_embedding.py | 155-164 | `arange`,`pow`,`div`（CPU 初始化） | **功能性** | ❌ 数值精度保证 |
| 12-A | unquant.py | 145-157 | `mm`, `addmm`（LocalMem GEMM） | **功能性** | ❌ ZENL GEMM 核心路径 |
| 13 | radix_cache.py | 42-46 + 416/543/631 | `cat`（_cat_zeus 辅助 + 3 调用点） | CPU bounce | ✅ 可删函数+3 调用 |

### 统计

- **CPU bounce 类型**：19 个分支
- **功能性分支**（不可移除）：8 个（GEMM 路由、dispatch、pack_weights、rotary 初始化）  
- **安全可移除**：**12 个**（涉及 add/sub/cat/index_put/argmax/clone）
- **部分可移除**：2 个（混合原生+非原生算子）
- **需保留**：5 个 CPU bounce（cumsum/arange/clamp/where/neg 无原生支持）

### 明确可安全移除的 CPU Bounce（12 个）

| 文件 | 分支# | 涉及算子 | 移除方式 |
|------|-------|---------|---------|
| radix_cache.py | 13 | cat ×3 | 删除 _cat_zeus 函数 + 3 调用点改为 `torch.cat(...)` |
| memory_pool.py | 1-A | index_put | 删除 if/else，保留 else 分支 |
| memory_pool.py | 1-B | cat | 删除 if/else，保留 else 分支 |
| memory_pool.py | 1-C | cat | 删除 if/else，保留 else 分支 |
| forward_batch_info.py | 2-A | cat+zeros+full | 删除 if 分支，保留非 zeus 路径 |
| schedule_batch.py | 4-A | add(+1) | 删除 if 分支，使用 elif/else 路径 |
| schedule_batch.py | 4-C | cat ×3 | 删除 if/else，保留 else 分支 |
| schedule_batch.py | 4-D | cat | 删除 if/else，保留 else 分支 |
| common.py | 3-B | clone | 替换为 `locs = batch.seq_lens.clone()` |

注：以下项的算子部分原生但含非原生算子（index/arange），建议整体保留或仅拆分可移除部分：
- common.py 3-A：index(2D 高级索引)无原生 → 保留
- schedule_batch.py 4-B：index(tensor[ki]) 无原生 → 保留
- forward_batch_info.py 2-B：sub ✅ 但 arange ❌ → 保留（不拆分）

### 需保留的 CPU Bounce（7 个）

| 文件 | 分支# | 算子 | 原因 |
|------|-------|------|------|
| forward_batch_info.py | 2-B | sub + **arange** | arange 无原生 |
| forward_batch_info.py | 2-C | **cumsum** | 无原生 |
| forward_batch_info.py | 2-D | **cumsum** | 无原生 |
| forward_batch_info.py | 2-E | cat + **arange** + **cumsum** | cumsum/arange 无原生 |
| forward_batch_info.py | 2-F | **clamp** + sub | clamp 无原生 |
| scheduler.py | 5-A | **neg** | 无原生 zenl 内核 |
| overlap_utils.py | 6-A | **where** + **clamp** | 均无原生 |
| overlap_utils.py | 6-B | **arange** | 无原生 |
| logits_processor.py | 7-A | **cumsum** + arange | 无原生 |
| common.py | 3-A | **index**(2D) + sub + add | index 2D 高级索引无原生 |
| schedule_batch.py | 4-B | **index**(tensor[ki]) | index 无原生 |

### 保留项——功能性分支（8 个，非 CPU bounce）

| 文件 | 分支# | 用途 |
|------|-------|------|
| logits_processor.py | 7-B | lm_head mm GEMM LocalMem 路由 |
| custom_op.py | 8-A | Zeus forward 分发（dispatch） |
| loader.py | 9-A | lm_head 加载追踪 |
| loader.py | 9-B | pack_weights 到 LocalMem |
| rotary_embedding.py | 10-A | 平台条件守卫 |
| rotary_embedding.py | 10-B | CPU 初始化（数值精度） |
| unquant.py | 12-A | LocalMem GEMM 调度 |
| vocab_parallel_embedding.py | — | 仅声明，无分支 |

---

## 九、sgl-kernel-zeus 算子支持分析

### 已实现算子（16 个内核，18 个导出接口）

所有内核均通过 **zecc 编译 → CPU 仿真 + Zeus 调度器** 模式运行，Triton 内核编译尚待完成。

#### 1. Elementwise / 归一化（4 个）

| 内核 | 状态 | 实现文件 | 说明 |
|------|------|---------|------|
| **rmsnorm** | ✅ 完整 | sgl_rmsnorm_sim.c + fused_add_rms_norm_zeus.cpp | Root Mean Square Norm，BF16 |
| **fused_add_rmsnorm** | ✅ 完整 | sgl_fused_add_rmsnorm_sim.c | 融合残差加 + RMSNorm，in-place |
| **silu_and_mul** | ✅ 完整 | sgl_silu_and_mul_sim.c + activation_zeus.cpp | SiLU(x) × gate，BF16 |
| **rotary_embedding** | ✅ 完整 | sgl_rotary_embedding_sim.c + rotary_embedding_zeus.cpp | RoPE (Neox-style)，in-place Q/K |

#### 2. Attention（2 个）

| 内核 | 状态 | 实现文件 | 说明 |
|------|------|---------|------|
| **decode_attention** | ✅ 完整 | sgl_decode_attention_sim.c + decode_attention_cpp.cpp | 解码阶段 MHA/GQA，支持 logit capping |
| **extend_attention** | ✅ 完整 | sgl_extend_attention_sim.c + extend_attention_cpp.cpp | 扩展阶段 MHA/GQA，因果遮罩+前缀 |

#### 3. KV Cache（1 个）

| 内核 | 状态 | 实现文件 | 说明 |
|------|------|---------|------|
| **store_kv_cache** | ✅ 完整 | sgl_store_kv_cache_sim.c + store_cpp.cpp | K 分块行主序 + V 列组交错，写入 LocalMem 分页缓存 |

#### 4. Sampling（8 个导出，6 个独立内核）

| 内核 | 状态 | 实现文件 | 说明 |
|------|------|---------|------|
| **top_k_renorm_probs** | ✅ 完整 | sgl_top_k_renorm_probs_sim.c | Top-K 过滤 + 重归一化 |
| **top_k_renorm_prob** | ✅ 别名 | → top_k_renorm_probs | 兼容别名 |
| **top_p_renorm_probs** | ✅ 完整 | sgl_top_p_renorm_probs_sim.c | Top-P 过滤 + 重归一化 |
| **top_p_renorm_prob** | ✅ 别名 | → top_p_renorm_probs | 兼容别名 |
| **top_p_sampling_from_probs** | ✅ 完整 | sgl_top_p_sampling_sim.c | Top-P 采样 |
| **top_k_top_p_sampling_from_probs** | ✅ 完整 | sgl_top_k_top_p_sampling_sim.c | 联合 Top-K+Top-P 采样 |
| **min_p_sampling_from_probs** | ✅ 完整 | sgl_min_p_sampling_sim.c | Min-P 阈值采样 |
| **sampling_from_logits** | ✅ 完整 | sgl_sampling_from_logits_sim.c | 融合 Softmax → Top-K/P/Min-P → 采样 |

#### 5. Embedding（1 个）

| 内核 | 状态 | 实现文件 | 说明 |
|------|------|---------|------|
| **embedding** | ✅ 完整 | sgl_embedding_sim.c + embedding_zeus.cpp | 词表行聚集查找，BF16 |

### SGLang Zeus 分支实际调用映射

| SGLang 文件 | 导入的 sgl_kernel_zeus 函数 | 用途 |
|------------|---------------------------|------|
| layers/layernorm.py | `rmsnorm`, `fused_add_rmsnorm` | 层归一化 |
| layers/activation.py | `silu_and_mul` | 激活函数 |
| layers/rotary_embedding.py | `rotary_embedding` | 位置编码 |
| layers/sampler.py | `top_k_renorm_prob`, `top_p_renorm_prob`, `top_k_top_p_sampling_from_probs`, `min_p_sampling_from_probs`, `sampling_from_logits` | 采样 |
| layers/vocab_parallel_embedding.py | `embedding` | 词嵌入 |
| layers/attention/zeus_backend.py | `extend_attention`, `decode_attention` | 注意力计算 |
| mem_cache/zeus_memory_pool.py | `store_kv_cache` | KV 缓存写入 |

### 待支持算子（原版 sgl-kernel 有但 sgl-kernel-zeus 尚未实现）

按 SGLang Zeus 运行所需的优先级排列：

#### 高优先级（当前 Zeus 推理流程可能用到）

| 算子 | 原版 sgl-kernel 分类 | Zeus 当前状态 | 说明 |
|------|---------------------|-------------|------|
| **gelu_and_mul** | Elementwise | ❌ 未实现 | 部分模型 (GPT-J 等) 使用 GELU 激活 |
| **gelu_tanh_and_mul** | Elementwise | ❌ 未实现 | LLaMA-3 等模型的 GELU-tanh 变体 |
| **gemma_rmsnorm** | Elementwise | ❌ 未实现 | Gemma 模型专用 (weight+1) |
| **gemma_fused_add_rmsnorm** | Elementwise | ❌ 未实现 | Gemma 模型专用融合版 |
| **top_k_mask_logits** | Sampling | ❌ 未实现 | Top-K logit 掩码（采样前置步骤） |
| **fast_topk** / **fast_topk_v2** | Sampling | ❌ 未实现 | 快速 Top-K 选择 |
| **merge_state** / **merge_state_v2** | Attention | ❌ 未实现 | 多轮注意力状态合并 |
| **transfer_kv_per_layer** | KV Cache | ❌ 未实现 | 分层级 KV 传输（分布式/迁移场景） |
| **gelu_quick** | Elementwise | ❌ 未实现 | GELU 快速近似 |

#### 中优先级（特定模型/特性需要）

| 算子 | 原版 sgl-kernel 分类 | Zeus 当前状态 | 说明 |
|------|---------------------|-------------|------|
| **moe_fused_gate** | MoE | ❌ 未实现 | MoE 专家路由门控（DeepSeek-V2 等） |
| **moe_align_block_size** | MoE | ❌ 未实现 | MoE token 分组对齐 |
| **moe_sum** / **moe_sum_reduce** | MoE | ❌ 未实现 | MoE 专家输出求和/规约 |
| **topk_softmax** / **topk_sigmoid** | MoE | ❌ 未实现 | MoE 路由 Top-K 选择 |
| **causal_conv1d_fwd** / **update** | Mamba | ❌ 未实现 | Mamba 模型因果卷积 |
| **concat_mla_k** / **concat_mla_absorb_q** | MLA | ❌ 未实现 | Multi-Latent Attention 张量拼接 |
| **cutlass_mla_decode** | MLA | ❌ 未实现 | MLA 解码阶段 CUTLASS 加速 |
| **build_tree_kernel_efficient** | Speculative | ❌ 未实现 | 推测解码树构建 |
| **verify_tree_greedy** | Speculative | ❌ 未实现 | 推测解码验证 |
| **segment_packbits** | Speculative | ❌ 未实现 | 推测解码位打包 |

#### 低优先级（量化/分布式/GPU 特性，Zeus 暂不需要）

| 算子类别 | 数量 | Zeus 当前状态 | 说明 |
|---------|------|-------------|------|
| **量化 GEMM**（fp8/int8/int4/awq/gptq/marlin） | ~30 | ❌ 未实现 | Zeus 当前仅支持 FP32/BF16 非量化推理 |
| **GGUF 量化** | ~6 | ❌ 未实现 | GGML 格式反量化和矩阵乘 |
| **Hadamard 变换** | ~5 | ❌ 未实现 | QuIP# 量化方案专用 |
| **分布式 AllReduce** (custom_ar/quick_reduce) | ~19 | ❌ 未实现 | Zeus 目前单卡推理，无 multi-NPU 通信 |
| **GPU SM 管理** | ~2 | ❌ 不适用 | GPU Green Context，Zeus 架构不适用 |
| **Flash Attention 变体** (flash_attn_with_kvcache 等) | ~3 | ❌ 不需要 | Zeus 有自己的 decode/extend attention |
| **NSA 稀疏注意力** | ~3 | ❌ 未实现 | Sparse Attention 高级特性 |
| **ES MoE GEMM** (expert specialization) | ~3 | ❌ 未实现 | Expert Specialization 量化 MoE |

### 统计总览

| 维度 | 数量 |
|------|------|
| 原版 sgl-kernel 总算子数 | ~120+ |
| **sgl-kernel-zeus 已实现** | **16 个内核（18 个导出）** |
| SGLang Zeus 分支实际调用 | 16 个（全部已覆盖） |
| 高优先级待实现 | ~9 |
| 中优先级待实现 | ~12 |
| 低优先级/暂不需要 | ~80+ |
| 覆盖率（按 SGLang Zeus 实际调用） | **100%** |
| 覆盖率（按原版全量算子） | **~13%** |

---

## 十、未原生支持算子的代码位置、推理语义与支持改造建议

本节聚焦第六节列出的未原生支持算子：`cumsum / arange / clamp / where / neg / index / relu / prod / softmax / permute / stack / div`。

分析范围分为两层：

1. **SGLang Zeus 实际运行路径**：这些算子在 `torch_zeus_sglang/sglang/python/sglang/srt/` 中出现在哪里、参与了什么推理过程、当前是显式 CPU bounce 还是会进入 `ZEUSFallback`。
2. **torch_zeus 后端证据**：这些算子为何当前没有原生 Zeus ATen 支持，是没有注册、只有 autocast 占位、还是已经在 `sgl-kernel-zeus` 中单独以自定义 kernel 形式支持。

### 1. torch_zeus 侧不支持证据汇总

| 算子 | 当前状态 | 证据 |
|------|---------|------|
| `cumsum` / `cumprod` | ❌ 无 Zeus ATen 计算内核 | 仅在 `autocast_mode.cpp` 里有 dtype 规则，没有 `TORCH_LIBRARY_IMPL` 计算实现 |
| `arange` | ❌ 无原生注册 | 只能走 `ZEUSFallback` |
| `clamp` | ❌ 无原生注册 | 只能走 `ZEUSFallback` |
| `where` | ❌ 无原生注册 | 只能走 `ZEUSFallback` |
| `neg` | ❌ 无原生注册 | 只能走 `ZEUSFallback` |
| `div` | ❌ 无原生注册 | 只能走 `ZEUSFallback` |
| `softmax` / `log_softmax` | ❌ 无 Zeus ATen 注册 | 采样场景由 `sgl-kernel-zeus` 自定义 kernel 单独实现 |
| `stack` | ❌ 无原生注册 | 只能走 `ZEUSFallback` |
| `permute` | ❌ 无原生注册 | 仅能依赖 fallback；`view/as_strided` 已支持但 `permute` 本身未接通 |
| `relu` | ❌ 无原生注册 | 只能走 `ZEUSFallback` |
| `prod` | ❌ 无 Zeus ATen 注册 | `zenl_types.h` 仅有 reduce mode 枚举，占位未接入 ATen |
| `index` | ❌ 仅 `index_put` 支持 | `tensor[indices]` / gather 类读取仍无原生 Zeus 实现 |

补充：`ZEUSFallback.cpp` 通过 `TORCH_LIBRARY_IMPL(_, PrivateUse1, m)` 统一兜底，所有未注册 Zeus 算子最终都会走 Zeus Tensor -> CPU -> 执行 CPU 实现 -> 回拷 Zeus 的回退链路。

### 2. SGLang Zeus 路径中的真实调用点

#### 2.1 `cumsum`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `model_executor/forward_batch_info.py` | 1071-1077 | `init_prefix_chunks` | prefix chunk 累积长度 | `_is_zeus` 下先 `cpu().cumsum()` |
| `model_executor/forward_batch_info.py` | 1105-1112 | `fetch_mha_one_shot_kv_indices` | 构建 `kv_indptr` 前缀和 | `_is_zeus` 下先 `cpu().cumsum()` |
| `model_executor/forward_batch_info.py` | 1244-1260 | `compute_position_torch` | extend 模式位置编码起始偏移 | 一部分直接在 CPU 上计算 |
| `layers/attention/fla/index.py` | 20-25 | `prepare_chunk_indices` | FLA chunk 索引整理 | 直接 `indices.eq(0).cumsum(0)`，当前 Zeus 路径未专门守护 |

**在做什么**

- `cumsum` 本质是在把每个 request 的长度数组转换成“段起始偏移”或“indptr”。
- 在推理系统里，这类前缀和主要用于：
	- 构造 packed sequence 的 `start_loc`
	- 构造 KV cache 的 `indptr`
	- 计算 extend / chunk decode 时每个子段在大缓冲中的写入起点

**从模型推理角度的原理**

- LLM serving 的 batch 通常是 ragged batch，不同请求长度不同。
- 真正送入 kernel 前，必须把 ragged 结构映射为扁平连续缓冲区，因此要有：
	- `len[i]`
	- `offset[i] = sum(len[:i])`
- `cumsum` 就是在做这个 offset 生成。

**支持改造建议**

1. **第一阶段**：先支持 1D `int32/int64` exclusive / inclusive prefix-sum。
2. **接口建议**：新增 `cumsum.int` Zeus kernel，仅支持 `dim=0`、连续张量。
3. **实现建议**：
	 - Host 侧约束 `ndim==1 or 2`、`contiguous`
	 - Device 侧先做块内 scan，再做 block sums，再做 add-back
4. **优先级**：高。因为它直接决定 batch 打平和 KV 索引生成，属于推理调度基础设施。

#### 2.2 `arange`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `model_executor/forward_batch_info.py` | 833-841 | `init_extend_compute` | extend token 的起始位置 | `_is_zeus` 下分支拆到 CPU/设备混合 |
| `mem_cache/memory_pool.py` | 265 | `clear` / 初始化空闲槽 | KV/slot 管理 | 直接 `torch.arange(..., device=self.device)` |
| `mem_cache/memory_pool.py` | 305 | `clear` / 初始化空闲槽 | Mamba cache slot 管理 | 直接 `torch.arange(..., device=self.device)` |

**在做什么**

- 在调度层，`arange` 常用来生成 token 序号、batch 内索引、空闲槽位编号。
- 在推理框架里这类张量不是模型数值计算本身，而是“元数据张量”。

**从模型推理角度的原理**

- 推理时要不停分配与回收 token slot、构造 `[0, 1, ..., n-1]` 型索引，驱动：
	- request 对齐
	- KV cache 写入位置计算
	- extend chunk 内偏移计算

**支持改造建议**

1. 优先支持 `arange(start, end, step=1, dtype=int32/int64)`。
2. 这是**低算力、高频调用**算子，适合直接做简单 fill kernel：`out[i] = start + i * step`。
3. 如果短期不做 kernel，也可先在 host 侧做“小张量 arange CPU 生成 + async H2D”的 runtime shortcut，避免进入通用 fallback。

#### 2.3 `clamp`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `model_executor/forward_batch_info.py` | 1259-1266 | `clamp_position` | 位置 ID 边界保护 | `_is_zeus` 下先转 CPU |
| `model_executor/forward_batch_info.py` | 1009-1015 | `init_prefix_chunks` | chunk 长度裁剪到非负 | 操作数大多来自 CPU |

**在做什么**

- 把 `seq_len - 1`、`chunk_end - chunk_start` 之类的中间量裁到合法区间，防止负长度、负位置索引。

**从模型推理角度的原理**

- 推理调度里大量存在 “边界截断” 操作，例如：
	- `max(seq_len - 1, 0)` 取最后一个有效 token 位置
	- chunk 切分后保证长度不小于 0
- 它本质是逐元素比较和截断，属于非常典型的 elementwise min/max 算子。

**支持改造建议**

1. 优先支持 `clamp_min` / `clamp_max` 两个子集，再组合成通用 `clamp`。
2. Device kernel 直接走 elementwise compare-select，难度低。
3. 建议和 `where` 一起做，因为二者底层都可归结为逐元素选择。

#### 2.4 `where`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `mem_cache/common.py` | 146-157 | `_get_last_loc_cpu` | 获取 prefix 最后一个缓存位置 | 显式 CPU bounce |
| `mem_cache/common.py` | 160-166 | `get_last_loc_torch` | 同上 | 普通张量路径直接 `torch.where` |

**在做什么**

- 给每个 request 找 “prefix 的最后一个 token 对应的 cache slot”；如果 prefix 长度为 0，则返回 `-1`。

**从模型推理角度的原理**

- prefix cache 命中后，调度器要知道“缓存命中的末尾在哪里”，这样才能接着分配新 token 的写入位置。
- `where(prefix_len > 0, last_slot, -1)` 是典型的条件选择元数据逻辑。

**支持改造建议**

1. 优先只做 `where(bool/int mask, a, b)` 的 contiguous 版本。
2. Kernel 本质是 `out[i] = cond[i] ? a[i] : b[i]`，实现难度低。
3. 一旦支持 `where`，`clamp`、`masked_fill`、部分 logits 后处理都能复用底层模板。

#### 2.5 `neg`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `managers/scheduler.py` | 2061-2066 | 草稿 token / future index 处理 | speculative 调度 | `_is_zeus` 下显式 CPU 取负 |

**在做什么**

- 对 `future_indices.indices` 取负，编码一种“未来 token / draft token”的特殊标记约定。

**从模型推理角度的原理**

- speculative decoding 往往需要在调度器里区分：
	- 已确认 token
	- draft token
	- future placeholder
- 用负号做标记是一种很常见的轻量编码方式。

**支持改造建议**

1. `neg` 应和 `add/sub/mul` 共用同一类 unary / binary elementwise 框架实现。
2. Host 侧只需补注册；device 侧本质是 `out[i] = -x[i]`。
3. 优先级中等。性能收益有限，但能减少大量小张量 CPU bounce。

#### 2.6 `index`（高级索引读取）

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `mem_cache/common.py` | 148-156 | `_get_last_loc_cpu` | prefix cache 最后位置查询 | 与 `where` 组合，显式 CPU |
| `mem_cache/common.py` | 162-166 | `get_last_loc_torch` | 同上 | 普通路径 `req_to_token[req_pool_indices_tensor, prefix_lens_tensor - 1]` |
| `mem_cache/memory_pool.py` | 101-118 | `ReqToTokenPool.write` | 2D slot 写入 | 当前 Zeus 仍保留 CPU bounce |

**在做什么**

- 读路径：`tensor[row_idx, col_idx]` 从二维映射表中取某个 request 某个 token 的 cache 位置。
- 写路径：把新分配的 token slot 写回 `req_to_token` 映射表。

**从模型推理角度的原理**

- `req_to_token` 是 serving 框架最核心的元数据结构之一，等价于 “逻辑 token 序号 -> 物理 KV slot”。
- 所有 prefix 匹配、extend 写入、decode 追踪，本质都依赖这个映射表做 gather/scatter。

**支持改造建议**

1. 当前 `index_put` 已支持，但**读取型高级索引**仍缺失，应优先补 `index.Tensor` / `gather-like` 子集。
2. 建议先只做二维整型索引：
	 - 输入 `base[M, N]`
	 - 索引 `(row_idx[K], col_idx[K])`
	 - 输出 `out[K]`
3. 这本质是一个 gather kernel，和 embedding 的 row-gather 很像；可复用地址计算框架。
4. 优先级很高。因为它直接决定元数据表访问是否能完全留在 Zeus 侧。

#### 2.7 `div`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `mem_cache/memory_pool.py` | 794-798 | `set_kv_buffer` 路径 | KV 量化前缩放 | 可能命中 Zeus fallback |
| `mem_cache/memory_pool.py` | 949-953 | `get_kv_buffer` 路径 | KV 反量化 / 还原 | 可能命中 Zeus fallback |
| `layers/sampler.py` | 118-123 | `forward_cuda` | 按 temperature 缩放 logits | Zeus sampling backend 已绕开，普通路径仍会调 `div` |
| `multimodal/processors/step3_vl.py` | 34-36 | 图像预处理 | 像素 `÷255` 归一化 | 通常在 CPU/CUDA 图像预处理侧 |

**在做什么**

- 两大类：
	- **logits 除温度**：`logits / T`
	- **KV / 图像缩放**：按比例归一化

**从模型推理角度的原理**

- `div` 在推理中最典型的语义就是“重标度”：
	- 采样时调节分布尖锐程度
	- 量化/反量化时恢复物理量级
	- 图像像素归一化到 `[0,1]`

**支持改造建议**

1. 先支持 `Tensor / Scalar` 与 `Tensor / Tensor` 两种 contiguous elementwise 除法。
2. 采样链路可直接复用 `sgl-kernel-zeus sampling_from_logits` 中已实现的 `div(temp) + softmax` 设计。
3. KV 缩放场景要求支持 bf16/fp16/fp32；可先不覆盖整型除法。

#### 2.8 `stack`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `model_executor/forward_batch_info.py` | 637-644 | 多模态 MROPE 位置偏移整理 | 多模态位置编码 | `torch.stack(...).to(device)`，通常先在 CPU 列表上构造 |
| `layers/rotary_embedding.py` | 60 / 90 / 1086 | RoPE 辅助张量构造 | rotary 预计算 | 常见为小张量拼装 |

**在做什么**

- 把多个小张量沿新维度拼成更高维结构，例如：
	- 多模态三轴位置索引 `[t, h, w]`
	- RoPE 的 `[cos, sin]` 成对堆叠

**从模型推理角度的原理**

- `stack` 和 `cat` 的区别是会新增一个维度，非常适合把“多个物理意义不同但对应同一 token 的分量”打包在一起。

**支持改造建议**

1. `stack` 可以先在 host 侧重写为：`unsqueeze + cat`。
2. 由于 `cat` 已支持，短期可在 Zeus dispatch 层把简单 `stack` 自动降解为组合实现，而不必立刻写新 kernel。
3. 中长期仍建议补 `stack` 原生注册，减少 graph break。

#### 2.9 `softmax`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `layers/sampler.py` | 105-145 | `Sampler.forward_*` | logits -> probs 采样 | Zeus backend 下改走 `sampling_from_logits` 自定义 kernel；普通路径仍是 `torch.softmax/log_softmax` |
| `layers/logits_processor.py` | 319 / 1219-1224 | logprob / normalized probs | 输出后处理 | 可能走 fallback |
| `speculative/eagle_worker.py` | 655 / 1020 等 | speculative 验证与采样 | 草稿 token 概率 | 可能走 fallback |

**在做什么**

- 把 logits 转为概率分布或 log-prob 分布，用于：
	- 正常采样
	- logprob 返回
	- speculative verify

**从模型推理角度的原理**

- 这是生成模型后处理最核心的归一化：
	$$p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
- 常与 top-k / top-p / min-p / temperature 串联使用。

**支持改造建议**

1. **采样链路**已经有 `sgl-kernel-zeus` 融合 kernel，应优先把更多 Zeus 路径切到该 kernel，而不是单独补普通 `torch.softmax`。
2. 若要补 ATen softmax，建议先支持：
	 - 最后一维 softmax
	 - bf16/fp16 输入、fp32 累加
3. `log_softmax` 可以由 softmax + log 组合，或直接做数值稳定版本。

#### 2.10 `relu`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `layers/activation.py` | 160-165 | `ReLU2.forward` | 特定模型 MLP 激活 | 当前无 Zeus 专门处理 |
| `layers/sparse_pooler.py` | 67 | token weight 生成 | pooling/重排序 | 可能走 fallback |
| 多个模型文件 | 如 `models/gemma3n_causal.py:129` | 模型层激活 | 通用模型算子 | 可能走 fallback |

**在做什么**

- 典型前馈层激活：`relu(x) = max(0, x)`，也常作为 `relu^2`、门控或轻量音频/视觉子模块激活。

**从模型推理角度的原理**

- 与 SiLU / GELU 相比，ReLU 更便宜，但在现代 LLM 主干里并不主流；更多出现在：
	- 视觉编码器
	- 音频模块
	- 一些 reward / pooling / adapter 子模块

**支持改造建议**

1. 可直接用 `clamp_min(x, 0)` 语义实现，底层和 `clamp` 复用。
2. 优先级中低，除非明确要跑视觉/音频/奖励模型。

#### 2.11 `permute`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `speculative/eagle_worker.py` | 610-614 | speculative cache reshape | speculative 调度 | 直接 `permute(...).reshape(...)` |
| `multimodal/processors/step3_vl.py` | 34 | 图像 HWC -> CHW | 多模态预处理 | 一般在 CPU/CUDA 侧 |
| 多个视觉/MoE模型文件 | 如 `models/deepseek_janus_pro.py`、`layers/moe/...` | 张量布局转换 | 模型层/预处理 | 目前没有 Zeus 原生支持 |

**在做什么**

- 纯布局变换，把张量解释成不同维度顺序。

**从模型推理角度的原理**

- 注意力、视觉 patch、MoE 权重排布中经常需要把：
	- `[B, T, H, D]` 变成 `[B, H, T, D]`
	- `[H, W, C]` 变成 `[C, H, W]`
- 理想情况下，`permute` 应该只是改 strides / metadata，不应真实拷贝。

**支持改造建议**

1. 优先检查 Zeus 张量的 stride 体系是否足够支持 view-based `permute`。
2. 若运行时对任意 strides 支持不足，可先支持常见模式：
	 - 2D transpose
	 - 4D attention 常见交换
	 - HWC <-> CHW
3. 这类算子更像 metadata op，优先级高于数值 kernel，因为一旦支持，很多模型路径会自然解锁。

#### 2.12 `prod`

| 文件 | 行号 | 函数 | 推理阶段 | 当前处理方式 |
|------|------|------|---------|-------------|
| `layers/rotary_embedding.py` | 1956 / 1987 / 2061 | 图像/视频 token 数计算 | 多模态位置编码 | `tensor.prod()`，潜在 fallback |
| 其他大多数位置 | `np.prod` / `math.prod` | shape/numel 计算 | Python 侧元数据 | 与 Zeus ATen 无关 |

**在做什么**

- 多模态场景中，`grid_thw.prod()` 用来把三维网格 `(T, H, W)` 转成 token 数。

**从模型推理角度的原理**

- 视觉或视频 patch token 数通常等于时间、高、宽维度的乘积，再除以 merge ratio。
- 它属于小规模 reduce，但在多模态位置编码里很常见。

**支持改造建议**

1. 若已有 reduce 框架，`prod` 可以直接在 `sum/mean/norm` 的 reduce host 框架上扩展一个 `ZENL_REDUCE_MUL` 模式。
2. 优先支持“小张量最后一维/全维 reduce”，先满足 grid 乘积与少量统计需求。

### 3. 当前几类算子的总体结论

#### 3.1 真实高优先级待补

| 算子 | 原因 |
|------|------|
| `cumsum` | 直接决定 ragged batch 到 packed buffer 的 offset/indptr 生成 |
| `index`（读取） | 直接决定 `req_to_token` 等核心元数据表的 gather 能否留在 Zeus 侧 |
| `arange` | 高频元数据张量构造算子，几乎所有调度与 cache 管理都要用 |
| `where` / `clamp` | 大量边界裁剪和条件选择都依赖这组算子 |
| `div` | 采样温度缩放、量化/反量化、图像归一化都会用到 |

#### 3.2 中优先级待补

| 算子 | 原因 |
|------|------|
| `softmax` / `log_softmax` | 采样主链已有 `sgl-kernel-zeus`，但其他后处理路径仍缺通用支持 |
| `neg` | 性能价值一般，但补齐成本很低 |
| `stack` | 可先用 `unsqueeze + cat` 组合替代 |
| `prod` | 有明确多模态用途，但频率低于 `sum/cumsum` |

#### 3.3 结构性待补

| 算子 | 原因 |
|------|------|
| `permute` | 更偏张量布局/stride 能力，补齐后会自然解锁更多模型路径 |
| `relu` | 视觉/音频/特种模型更依赖，不是当前纯文本 LLM 最核心瓶颈 |

### 4. 推荐实施顺序

1. **调度元数据优先**：`arange -> cumsum -> index(read) -> where/clamp`
2. **数值后处理第二批**：`div -> neg -> prod`
3. **概率分布第三批**：优先扩用 `sgl-kernel-zeus sampling_from_logits`，再决定是否补通用 `softmax/log_softmax`
4. **布局能力第四批**：`permute`
5. **模型激活第五批**：`relu`

### 5. 支持改造的工程路线建议

#### 路线 A：优先补 Zeus ATen 原生能力

- 适合：`arange / cumsum / where / clamp / neg / div / prod / index(read)`
- 优点：SGLang 与普通 PyTorch 代码都能直接受益，减少 `_is_zeus` 分支
- 建议：先做最小子集，限定 dtype、dim、contiguous 约束

#### 路线 B：优先补 SGLang 专用融合 kernel

- 适合：`softmax` 相关链路
- 当前已经有例子：`sgl-kernel-zeus sampling_from_logits`
- 建议：继续把 speculative / logits_processor 的 softmax/log_softmax 场景逐步收敛到融合 kernel，而不是单独让普通 softmax 频繁 fallback

#### 路线 C：优先补 metadata/stride 能力

- 适合：`permute`
- 关键不是算力，而是 Zeus Tensor 是否能稳定支持非 contiguous stride 语义

#### 路线 D：短期 runtime rewrite

- 适合：`stack`
- 先在 dispatch 层改写为 `unsqueeze + cat`，用已有 `cat` 能力快速消化

### 6. 最终判断

从 **纯文本 LLM 推理** 的短期收益看，最值得优先补的是：

1. `cumsum`
2. `index`（读取）
3. `arange`
4. `where`
5. `clamp`
6. `div`

从 **中长期模型覆盖率** 看，下一批应补：

1. `softmax/log_softmax`
2. `permute`
3. `prod`
4. `relu`

它们的共同目标不是“减少几个 Python 分支”，而是把 **调度元数据、概率后处理、布局变换** 这三类高频非 GEMM 操作稳定留在 Zeus 设备侧，逐步把当前“GEMM/embedding/采样 kernel 已原生，但周边 ATen 元算子仍频繁 bounce”的状态，推进到“完整推理链大部分驻留 Zeus”。
