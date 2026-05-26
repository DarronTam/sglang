# GLM5-Next DSA Decode Zeus 适配开发追踪

> 对齐目标：`/root/project/sglang-feat-v0.5.10-prerelease-glm` 中的
> `Glm5NextForCausalLM` 结构。DSA 层对应
> `Glm5NextDecoderLayer.self_attn = DeepseekV2AttentionMLA`，decode 路径走
> MLA absorb + NSA indexer + sparse MQA。
>
> 本文同时覆盖两套配置：
> - **GLM5-Next-16B**：`zeus_dev/model_state_dev/config_16b_v2.json`
> - **GLM5-Next**：`zeus_dev/model_state_dev/config.json`，其中配置文件写
>   `qk_rope_head_dim=64`，但当前实际部署版本按 **Dro=0** 处理。
>
> 文档格式参考 `glm4_moe_ffn_dev.md`：先定义范围与方法，再给配置表、算子依赖表、
> 计算流、dev 脚本 stage 顺序，细节统一收在附录。

## 范围与方法

- **起点**：DSA attention 模块收到的 `hidden_state [B,H]`。在完整
  `Glm5NextDecoderLayer.forward()` 中，它来自
  `layer_communicator.prepare_attn(...)` 之后；MHC/LayerCommunicator 的外层残差流不在本切片内。
- **终点**：`self_attn.o_proj` 输出 `out [B,H]`。不包含后续
  `prepare_mlp`、post-attention layernorm、MLP/MoE、MHC postprocess、NextN。
- **切片**：单个 DSA 模块、decode only、TP=1 起步。CP 作为同一算子链上的可选维度：
  - `CP=1`：Indexer topK 与 sparse MQA 都本地完成。
  - `CP>1`：Indexer 在本地 `S_local` 上算 logits/topK 后，额外做
    local-topK all-gather 与 global merge-topK；Sparse MQA 先算 partial
    `(out,lse)`，再做 partial all-gather 和 FA-reduce。
- **对齐方式**：沿用 `moe_ffn_dev` 的 stage 化 "REF vs Zeus" 模式。同一份随机权重和输入，
  REF 侧用 pure-torch / 目标 repo 语义生成 golden；Zeus 侧逐 stage 对拍。
  **Zeus 路径禁止静默落到 torchnative fallback**，未落地 kernel 必须显式 `TODO` / fail。
- **dev 脚本**：本文档完成后新增
  `zeus_dev/model_state_dev/dev_glm5next_dsa_decode_test.py`，按 `--stage` 单独跑。

不在本文 scope：
- prefill / extend / draft extend / target verify。
- KDA linear attention。
- MoE-FFN。
- MHC 内部 4 残差流、Sinkhorn、post-mult 算子。
- TP/EP/DP-attention 与真实跨进程 collective wiring。
- latent KV 的 FP8 存储路线；首版按 BF16 latent K。

## v2 设计迭代：Paged-Attention 集成 + Device-Side History

**背景**：v1 文档（2026-05 起）按 "per-rank single-pool latent KV cache + host-coordinated
history concat" 路径落地，dev_dsa_attn 包装时观察到 forward_zeus 内部有大量
host↔device round-trip（`scratch["body_cache"].cpu()` 触发 `[ZEUS ToGDG Stub]`，
`torch.cat([history.*, new_*])` 走 CPU concat 等）。v2 把 DSA 模块定位为
**paged-attention 的消费者**，由上游 KV cache manager 管理共享 pool 与 block_table，
DSA 只接 slot 接口、history 状态完全在 device。

### Paged 设计契约

**State（device-resident，由 paged-attention manager 持有 / 管理）**：

```
latent_kv_pool   [total_slots, Rkv]            bf16    共享池（多序列共用 page）
index_body_pool  [total_slots, Di]             fp8 LocalMem
index_scale_pool [total_slots]                 fp32
block_table      [B, max_pages_per_seq]        int32   logical page -> physical page; -1 未分配
seq_lens         [B]                           int32   每序列当前 valid token 数
new_slot_mapping [B]                           int32   本步要写入的物理 slot（上游算好）
```

其中：
- `total_slots = total_pages * page_size`；`page_size` 由 manager 配置（典型 16/32/64）
- `new_slot_mapping[b] = block_table[b, seq_lens[b] // page_size] * page_size
   + (seq_lens[b] % page_size)`
- v1 的 "per-rank single pool + slot_mapping" 实际上就是 paged 的 `page_size=1` /
  block_table 是 identity 的退化情况；v2 不改 pool 结构，只是让 caller 按 paged 规则算 slot

### DSA 模块的 paged 兼容性盘点

| Kernel | v1 接口 | v2 paged 适配 |
|---|---|---|
| `dsa_q_a_proj_norm` | hidden → q_lora（无 cache） | ✅ 与 paging 无关，无需改 |
| `dsa_kv_a_proj_norm_store` | `cache [num_slots, Rkv]` + `slot_mapping [B]` | ✅ **已 paged-ready** —— caller 算好 `new_slot_mapping` 喂进来即可；kernel 签名不动 |
| `dsa_q_main_absorb` | q_lora → q_new（无 cache） | ✅ 与 paging 无关，无需改 |
| `dsa_indexer_q_weights` | q_lora + hidden → q_body / weights（无 cache） | ✅ 与 paging 无关，无需改 |
| `dsa_indexer_k_prep_store` | `body_cache [num_slots, Di] LocalMem` + `scale_cache [num_slots]` + `slot_mapping [B]` | ✅ **已 paged-ready** —— 同上 |
| **`dsa_index_logits`** | `body [B, S, Di] LocalMem + scale [B, S]`（per-batch 连续 history） | ❌ **需要 paged 变种**：接 `(body_pool, scale_pool, block_table, seq_lens, page_size)`，kernel 内每 (b, s) 算物理 slot 后从 pool 取；s >= seq_lens[b] 出 -inf |
| `dsa_local_topk_radix` | `logits [B, S]` + `positions [S]` | ✅ 不感知 paging。positions 缓存 `arange(max_logical_S)` 一次；invalid 位置因 -inf 自动被 topk 排除 |
| `dsa_latent_k_gather` (2026-05-26 batched 版) | `cache [B, num_slots, Rkv]` + `slot_indices [B, Ktop]` | ⚠️ **需要决策**：见下节"`dsa_latent_k_gather` paged 适配" |
| `dsa_sparse_mqa_partial` | q_new + (K_local, mask) | ✅ 与 paging 无关 |
| `dsa_post_o_proj_no_cp` | attn_latent → out | ✅ 与 paging 无关 |

### `dsa_latent_k_gather` paged 适配

2026-05-26 把 `dsa_latent_k_gather` 改成 `cache [B, num_slots, Rkv]` batched 版本，
本质是 paged 的 page_size=num_slots / 每序列独占一池的退化形式。要兼容真正的 paged，
**应回到单池接口 + slot_indices 内存放物理 slot**：

```
v2 paged 接口：
  slot_indices    [B, Ktop]                    int32   物理 slot (caller 通过 block_table 翻译)
  latent_kv_pool  [total_slots, Rkv]           bf16    共享池
  → K_local / mask 同现状
```

**翻译规则**（caller 端做）：
```
phys_slot[b, k] = block_table[b, top_pos[b, k] // page_size] * page_size
                + (top_pos[b, k] % page_size)
```

但 top_pos 来源是 `dsa_local_topk_radix` 输出，本身是 logical position（在 logits 数组里
的索引，0..seq_lens[b]-1 内）。所以需要一步 "logical → physical" 翻译。

**两种实现选择**：
- **Option A**：caller 显式做翻译（一行 `phys_slot = block_table.gather(1, top_pos //
  page_size) * page_size + top_pos % page_size`），喂给单池版 `dsa_latent_k_gather`
- **Option B**：把翻译融进 kernel —— `dsa_latent_k_gather` 加 `block_table / page_size`
  参数，内部翻译 + gather

Option A 接口干净（gather kernel 不感知 paging），Option B 减少一次 device op。
**首版选 Option A**，等 profile 看到翻译开销显著再融。

**回退路径**：v1.5 的 batched 接口仍可保留作为 dev/test path，paged 路径独立出
`dsa_latent_k_gather_paged`（或 caller 端先 reshape 把 paged 池映射成 batched 视图）。

### 新增 helper ops（按需）

| 新 op | 作用 | 必要性 |
|---|---|---|
| `dsa_compute_new_slot` | 给 (block_table, seq_lens) 算 new_slot_mapping[b] | 弱必需，可上游 paged manager 做 |
| `dsa_translate_topk_positions` | logical top_pos [B, Ktop] → physical slot_indices [B, Ktop] via block_table + page_size | 中等必需（每步都要），避免 device aten 散包 |
| `dsa_advance_seqlens` | seq_lens += 1 inplace device side | 可选，aten in-place add 兜底即可 |

首版只新增 **`dsa_index_logits_paged`** 一颗 kernel，其它用 aten / caller code 兜底，
profile 后再决定哪些值得融。

### v2 后的 dev_dsa_attn forward_zeus 形状

```python
class Glm5NextDsaAttn:
    def __init__(self, which, seed):
        # 只持 model weights, 不持 history
        ...

    def forward_zeus(
        self,
        hidden_z,
        paged_state,           # 上游 paged-attention manager 给的状态:
                               #   latent_kv_pool, index_body_pool, index_scale_pool,
                               #   block_table, seq_lens, new_slot_mapping, page_size
        max_logical_s,         # 当前 batch 最大 seq_lens + 1 (positions arange 上界)
    ):
        # #0 / #1 / #2 同 v1 (paged-ready 接口, caller 算好 new_slot_mapping)
        # #3 改调 dsa_index_logits_paged(body_pool, scale_pool, block_table, seq_lens, ...)
        # #7 caller 先把 top_pos 翻译成 phys_slot_indices, 再调单池版 dsa_latent_k_gather
        # #8/#10 不变
```

`forward_zeus` 主体长度从 v1 ~150 行（含 host concat）降到 ~70 行（纯 device-side ops 串接）。
所有 `.cpu()` / `torch.cat` / `from_tensor(full_*)` 消失。

## GLM5-Next DSA 相关配置

| 字段 | **GLM5-Next-16B**<br>`config_16b_v2.json` | **GLM5-Next**<br>`config.json` |
|---|---:|---:|
| `architectures` | `["Glm5NextForCausalLM"]` | `["Glm5NextForCausalLM"]` |
| `torch_dtype` | bfloat16 | bfloat16 |
| `hidden_size` (H) | 2048 | 4096 |
| `num_hidden_layers` | 27 | 45 |
| DSA layers (`full_attn_layers`) | 6 层 `[3,7,11,15,19,23]` | 11 层 `[3,7,11,15,19,23,27,31,35,39,43]` |
| `num_attention_heads` (Nh) | 32 | 64 |
| `q_lora_rank` (Rq) | 768 | 1536 |
| `kv_lora_rank` (Rkv) | 512 | 512 |
| `qk_nope_head_dim` (Dnope) | 128 | 192 |
| `qk_rope_head_dim` (Dro, 实际) | 0 | **0**（config 写 64，实际版本为 0） |
| `qk_head_dim` (Dqk, 实际) | 128 | 192 |
| `v_head_dim` (Dv) | 128 | 256 |
| attention scaling | `128^-0.5` | `192^-0.5` |
| `mla` / `mla_nope` | true / true | true / true |
| `multi_query_attention` | true | false（DSA MLA absorb 路径仍按 latent KV） |
| `index_head_dim` (Di) | 128 | 128 |
| `index_n_heads` (I) | 8 | 8 |
| `index_topk` (Ktop) | 2048 | 2048 |
| `index_dsa_use_layernorm` | true | true |
| `rope_theta` | 10000 | 10000 |
| `mhc` | true | true |
| `mhc_num_residual_streams` | 4 | 4 |
| `mhc_tau` | 1.0 | 0.05 |
| `mhc_sinkhorn_iterations` | 20 | 未显式写，按实现默认 |
| `mhc_no_norm_weight` | true | 未显式写，按实现默认 |
| `mhc_post_mult_value` | 2 | 未显式写，按实现默认 |
| `hres_vwnstyle` | false | true |
| `hc_eps` | 1e-6 | 未显式写，按实现默认 |
| latent KV 行宽（Dro=0） | `Rkv=512` elem = 1024 B BF16 | `Rkv=512` elem = 1024 B BF16 |
| index K 行宽 | `Di=128` FP8 + scale = 132 B | `Di=128` FP8 + scale = 132 B |
| `fused_qkv_a_proj_with_mqa` 输出维度（实际） | `Rq+Rkv=1280` | `Rq+Rkv=2048` |
| `q_b_proj` 输出维度（实际） | `Nh*Dqk=4096` | `Nh*Dqk=12288` |
| `w_kc` | `[Nh,Dnope,Rkv]=[32,128,512]` | `[64,192,512]` |
| `w_vc` | `[Nh,Rkv,Dv]=[32,512,128]` | `[64,512,256]` |

关键约定：
- 两个模型在本文中都按 **Dro=0**，主 MLA 与 Indexer 都不走 RoPE。
- `Rkv=512` 固定，所以 sparse MQA 的 absorb 空间 head_dim 也是 512。
- Indexer `Di=128/I=8/Ktop=2048` 两个模型相同，local topK 与 CP merge 逻辑可共用。

## DSA-Decode 算子依赖表

符号：
- `B`：decode batch / request 数。
- `S`：当前 request 的逻辑历史长度。
- `CP`：context parallel degree。
- `S_local`：本 rank 负责的历史长度，近似 `ceil(S / CP)`；实际按 page/block interleave。
- `Ktop=2048`，`I=8`，`Di=128`，`Rkv=512`。

| # | 子步骤 | shape | CUDA sgl-kernel / sglang 参考 | Zeus 现状 |
|---|---|---|---|---|
| 0 | Q/KV-A projection + q/kv RMSNorm + latent KV store。**当前 Zeus 设计拆成两个独立 kernel**，不 fuse：<br>**#0.Q** `dsa_q_a_proj_norm`：`hidden -> q_a_proj -> RMSNorm -> q_lora_out`。<br>**#0.KV** `dsa_kv_a_proj_norm_store`：`hidden -> kv_a_proj -> RMSNorm -> latent_kv_cache[slot_mapping[t]]`（in-place scatter）。<br>两个 kernel 各自 `grid=(1,)`、hidden 各读一次。fuse 决策留给 porting 阶段。 | **#0.Q**：`hidden [B,H] -> q_a_norm [B,Rq]`。16B: `[B,2048]→[B,768]`；Next: `[B,4096]→[B,1536]`。<br>**#0.KV**：`hidden [B,H] + slot [B] -> cache[slot, :] [num_slots,Rkv]`。16B/Next 同样写 `Rkv=512` 宽。 | `DeepseekV2AttentionMLA.prepare_qkv_latent`；`forward_mla.py::forward_absorb_prepare`；`token_to_kv_pool.set_mla_kv_buffer`；V5/V7 dev 的 `dsa_kv_proj_cache_store_fused` 可参考 store 逻辑。 | ✅ `sgl_kernel_zeus.dsa_q_a_proj_norm` + `sgl_kernel_zeus.dsa_kv_a_proj_norm_store`（`torch_zeus/sgl-kernel-zeus/csrc/glm5next_dsa/`）。当前执行路径是 CPU sim；Triton 蓝本已落地，待后端接通。 |
| 1 | Q-main absorb。`q_b_proj(q_lora)` 后按 `Dqk` reshape；Dro=0，因此不 split `q_pe`，不做 RoPE；直接用 `w_kc` 做 absorb。 | `q_lora [B,Rq] -> q [B,Nh,Dqk] -> q_new [B,Nh,Rkv]`。16B: `[B,32,512]`；Next: `[B,64,512]`。 | `forward_mla.py::forward_absorb_prepare` 中 `q_b_proj` + `torch.bmm(q_nope.T, w_kc)`；`deepseek_weight_loader.py` 的 `kv_b_proj -> w_kc/w_vc` 拆分。 | ✅ `sgl_kernel_zeus.dsa_q_main_absorb`（`torch_zeus/sgl-kernel-zeus/csrc/glm5next_dsa/`）。两次 GEMM 中间 bf16 round；per-head 外循环，q_lora 一次 load 跨 head 共享。当前执行路径是 CPU sim；Triton 蓝本已落地，待后端接通。 |
| 2 | Indexer prep + index K store。**当前 Zeus 设计拆成两个独立 kernel**，不 fuse：<br>**#2.Q** `dsa_indexer_q_weights`：`q_lora → wq_b → reshape [B,I,Di] → Hadamard (× H_Di / √Di) → row-wise int8-proxy quant → q_body/q_scale`；同 kernel 内 `hidden → weights_proj.T → gate → weights = gate · I^{-1/2} · q_scale · Di^{-1/2}`。<br>**#2.K** `dsa_indexer_k_prep_store`：`hidden → wk → 完整 LayerNorm(weight, bias) → Hadamard (× H_Di / √Di) → row-wise int8-proxy quant → k_body/k_scale`，最后按 `slot_mapping[t]` scatter 到 `index_k_body_cache[slot]` / `index_k_scale_cache[slot]`。<br>两个 kernel 各自 `grid=(1,)`、Dro=0（不做 NeoX RoPE）。`k_scale` 与 `k_body` 在 cache 中**作为独立的张量**保存，不 pack 成 132B 行宽。fuse 决策留给 porting 阶段。 | **#2.Q**：`q_lora [B,Rq] + hidden [B,H] → q_body [B,I,Di] bf16(int8 vals) + q_scale [B,I] fp32 + weights [B,I] fp32`。16B: `q_body [B,8,128]`、`q_scale [B,8]`、`weights [B,8]`。<br>**#2.K**：`hidden [B,H] + slot [B] → cache_body [num_slots,Di] bf16(int8 vals, in-place scatter) + cache_scale [num_slots] fp32(in-place scatter)`。16B: `Di=128`。 | `nsa_indexer.py::Indexer.forward_cuda` / `_get_q_k_bf16`；`rotate_activation = hadamard_transform(x, N**-0.5)`；`act_quant(..., block=128, fmt="ue8m0")`；`_get_logits_head_gate`；`fused_store_index_k_cache`。 | ✅ `sgl_kernel_zeus.dsa_indexer_q_weights` + `sgl_kernel_zeus.dsa_indexer_k_prep_store`（`torch_zeus/sgl-kernel-zeus/csrc/glm5next_dsa/`）。当前执行路径是 CPU sim；Triton 蓝本已落地，待后端接通。首版按 **Sylvester ±1** Hadamard 矩阵做真旋转，body 用 bf16 容器存 int8 值（Zeus 边界限制 bf16/int32），scale 用 fp32 独立张量。 |
| 3 | Local Index logits。按 **Index GEMM -> gate -> scale** 次序做：先用 FP8 q/k 做 `[I,S_local]` dot；再按 `weights` 做 head reduce；最后乘 `k_scale[S_local]`，避免对 `[I,S_local]` 做 scale broadcast。 | `q_idx_fp8 [B,I,Di] x index_K_local [S_local,Di] -> raw [B,I,S_local] -> logits_local [B,S_local]`。 | `deep_gemm.fp8_paged_mqa_logits`；`nsa_indexer.py::_get_topk_paged`；`KvPlacement_Decode_16dev_v2.md` §6.2.1 的 scale 归属。 | ✅ `sgl_kernel_zeus.dsa_index_logits`（`torch_zeus/sgl-kernel-zeus/csrc/glm5next_dsa/`）。当前执行路径是 CPU sim；Triton 蓝本（fp8 dot + S-tile）已落地，待后端接通。 |
| 4 | Local topK。若 `S_local <= Ktop`，kernel 内分支直接返回所有有效位置并 padding；若 `S_local > Ktop`，走 radix select。输出 local position 需要带全局 position/base offset，供 CP merge。 | `logits_local [B,S_local] -> local_topk_logits [B,Ktop] + local_topk_pos [B,Ktop]`，无效位填 `-1/-inf`。 | `NSAIndexerMetadata.topk_transform`；`sgl_kernel.fast_topk_v2` / `fast_topk_transform_fused`；算法参考 `topk_fused_radix_select.md`。 | ✅ `sgl_kernel_zeus.dsa_local_topk_radix`（`torch_zeus/sgl-kernel-zeus/csrc/glm5next_dsa/`）。当前执行路径是 CPU sim（三分支 + min-heap top-K）；Triton 蓝本（radix-select 7-phase 直译，PWLF / core_send / barrier 占位 intrinsic）已落地，待后端接通。CP=1 与 CP>1 共用同一个 kernel，区别仅在传入的 `S_local` 与 `positions` slice。 |
| 5 | Optional CP local-topK all-gather。仅 `CP>1`。收集各 rank 的 local topK `(logit,pos)` pair。`CP=1` 直接跳过。 | `[CP,B,Ktop]` logits + `[CP,B,Ktop]` positions。通信量约 `CP*B*Ktop*(4+4)`。 | 参考 `attn_cp_group` collective；`nsa/utils.py` 的 CP all-gather/rerange 接口形态。 | ❌ TODO collective wrapper。 |
| 6 | Optional global merge-topK。仅 `CP>1`。对 `CP*Ktop` 个候选再选全局 topK；算法与 local radix select 同源，只是输入已经是候选集。`CP=1` 时 `global_topk = local_topk`。 | `local_topk[*] -> global_topk_pos [B,Ktop] + global_topk_logits [B,Ktop]`。 | 新增 Zeus 侧逻辑；可复用 `topk_fused_radix_select.md` 的候选二次 select 思路。 | ❌ TODO `dsa_cp_merge_topk`。 |
| 7 | Latent K gather。基于 `global_topk_pos` 做 owner/page-table 过滤：只把 **属于本 device/rank 的 latent K** 从本 device 的 **Gmem** gather 到 **Lmem**；非本 rank owner 的 topK 位置不写 K_local（保留 caller garbage），但**额外产出 `mask [B, Ktop] bf16`** 告诉下游 #8 哪些位置有效。两个 core 都跑完整 Ktop（broadcast gather，duplicate work），各自 Lmem 拿到一份完整副本，为下游 #8 按 Q-head 切核做铺垫。首版假设 latent K 为 BF16 且能放在 Gmem。 | `global_topk_pos [B,Ktop] -> local_slots [B,Ktop]`；gather `K_local [B,Ktop,Rkv]` 到 Lmem（valid 行真数据，invalid 行不动），同时产 `mask [B,Ktop]` bf16（1.0/0.0）。 | `transform_index_page_table_decode`；`NativeSparseAttnBackend.forward_decode` 中 topK indices 到 `page_table_1` 的变换；KV 放置参考 `KvPlacement_Decode_16dev_v2.md` §7。 | ✅ `sgl_kernel_zeus.dsa_latent_k_gather`（`torch_zeus/sgl-kernel-zeus/csrc/glm5next_dsa/`）。当前执行路径是 CPU sim；Triton 蓝本（两核 broadcast、per-position 标量 if 跳过 invalid + 同步产 bf16 mask）已落地，待后端接通。CP=4 短 history 实测 invalid ≈ 99%，新版省 ~99% Gmem read + K_local write（vs always-load+mask）。未来 TODO：切核间 DMA，两核分工 + 互发广播，Gmem 读再省一半。 |
| 8 | Sparse MQA partial。用 `q_new` 与本 rank gather 到的 latent K 做 sparse MQA，two-pass softmax，输出 partial latent out 与 LSE。kernel 始终输出 `(partial_out, partial_lse)`，是否后续 reduce 由 CP 决定。 | `q_new [B,Nh,512] + K_local [B,N_eff,512] -> partial_out [B,Nh,512] + partial_lse [B,Nh]`。`N_eff=Ktop` when CP=1。 | `NativeSparseAttnBackend.forward_decode`；`_forward_flashmla_sparse` / `_forward_flashmla_kv`；`flash_mla_sparse_fwd` / `flash_mla_with_kvcache`；TileLang sparse MLA fallback。 | ✅ `sgl_kernel_zeus.dsa_sparse_mqa_partial`（`torch_zeus/sgl-kernel-zeus/csrc/glm5next_dsa/`）。FA-v2 在线 softmax；**Q-head 切核 → 无核间 reduce**（两核处理 disjoint head 段，#7 broadcast K 已让两核 Lmem 各拥完整 K）。当前执行路径是 CPU sim；Triton 蓝本（CORE_NUM=2 沿 Nh 切核 + 在线 softmax + 空 rank guard）已落地，待后端接通。 |
| 9 | Optional CP partial all-gather。仅 `CP>1`。收集各 rank 的 `partial_out/partial_lse`，供 FA-reduce。`CP=1` 跳过。 | `[CP,B,Nh,512]` + `[CP,B,Nh]`。 | `attn_cp_group` collective；V7/V7.1 的 F-MERGE-POST 设计可参考。 | ❌ TODO collective wrapper。 |
| 10 | No-CP post kernel。`CP=1` 时不做 FA-reduce，直接把 sparse MQA latent out 做 V absorb，再做 `o_proj`。 | `attn_out_latent [B,Nh,512] -> bmm w_vc -> [B,Nh,Dv] -> reshape [B,Nh*Dv] -> o_proj -> [B,H]`。 | `forward_mla.py::forward_absorb_core` 末段 `torch.bmm(attn_output.T, w_vc)` + `RowParallelLinear(o_proj)`。 | ✅ `sgl_kernel_zeus.dsa_post_o_proj_no_cp`（`torch_zeus/sgl-kernel-zeus/csrc/glm5next_dsa/`）。dual-GEMM 单 kernel：per-head V absorb fp32 → bf16 round → o_proj GEMM 跨 head accumulate 到单 fp32 `o_acc [B, H]` → 一次 RNE 写出。当前执行路径是 CPU sim；Triton 蓝本（CORE_NUM=1，预留 H 切核）已落地，待后端接通。 |
| 11 | CP post kernel。`CP>1` 时先做 FA-reduce：online-softmax merge partial `(out,lse)`，再做 V absorb 和 `o_proj`。这是 CP 版本的最终 kernel。 | `partial_out/lse [CP,B,Nh,512] -> attn_out_latent [B,Nh,512] -> [B,Nh,Dv] -> out [B,H]`。 | REPO 单卡无直接对应；数学参考 FlashAttention LSE merge；V absorb/o_proj 同 #10。 | ❌ TODO `dsa_fa_reduce_o_proj_cp`。 |

## DSA-Decode 计算流

```
hidden_state [B,H] bf16
        │
        ▼
(0) pre_store
    fused_qkv_a_proj_with_mqa + q_a_rmsnorm + kv_a_rmsnorm
        ├── q_lora [B,Rq] ──────────────┬──────────────────────────────┐
        └── k_new [B,1,512]             │                              │
             owner store latent KV      │                              │
                                        ▼                              ▼
                              (1) q_main_absorb              (2.Q) indexer_q_weights         (2.K) indexer_k_prep_store
                                  q_b_proj                    wq_b(q_lora) → [B,I,Di]         wk(hidden) → [B,Di]
                                  no RoPE                     Hadamard × H_Di / √Di           完整 LayerNorm(weight, bias)
                                  bmm w_kc                    row int8-proxy quant             Hadamard × H_Di / √Di
                                                              gate = weights_proj(hidden)      row int8-proxy quant
                                                              weights = gate · I^-½ · q_scale  scatter cache[slot]
                                        │                              │                              │
                                        ▼                              ▼                              ▼
                                  q_new [B,Nh,512]             q_body [B,I,Di]                 cache_body [slot,Di]
                                                               q_scale [B,I]                   cache_scale [slot]
                                                               weights [B,I]
                                                                       │
                                                                       ▼
                                                        (3) index_logits_local
                                                        Index GEMM -> gate -> scale
                                                                       │
                                                                       ▼
                                                        logits_local [B,S_local]
                                                                       │
                                                                       ▼
                                                        (4) local_topk
                                                        short path if S_local<=2048,
                                                        radix select otherwise
                                                                       │
                           ┌─────────────────────────────── CP=1 ──────┴────── CP>1 ───────────────┐
                           │                                                                        │
                           ▼                                                                        ▼
                 global_topk_pos = local_topk                                      (5) all-gather local topK
                                                                                   (6) merge global topK
                                                                                              │
                                                                                              ▼
                                                                                     global_topk_pos
                           └──────────────────────────────────────┬────────────────────────────┘
                                                                  ▼
                                                        (7) latent_K gather
                                                        owned Gmem BF16 -> Lmem scratch
                                                                  │
                                                                  ▼
                                                        (8) sparse_mqa_partial
                                                        q_new x K_local
                                                        -> partial_out, partial_lse
                                                                  │
                           ┌─────────────────────────────── CP=1 ─┴────── CP>1 ────────────────────┐
                           │                                                                        │
                           ▼                                                                        ▼
                 (10) no_cp post                                                     (9) all-gather partial
                 V absorb + o_proj                                                   (11) FA-reduce
                           │                                                          + V absorb + o_proj
                           ▼                                                                        │
                    out [B,H] bf16                                                                  ▼
                                                                                              out [B,H] bf16
```

关键数据流说明：

- #0 拆成两个独立 kernel（`dsa_q_a_proj_norm` + `dsa_kv_a_proj_norm_store`），hidden 各读一次。
  早期讨论曾考虑 fused 单 kernel 让 hidden 读一次产 `q_lora` 与 `latent_cache` 两份；最终为了
  host 路径和契约的极简性，选择拆开。fuse 决策留给后续 porting。Q 侧 `q_lora_norm` 供
  下游 #1 / #2 复用；KV 侧直接 scatter 到 latent cache，避免 `kv_a_layernorm` 输出落 DRAM。
- #1 与 #2 都消费 `q_lora`，但计算性质不同：#1 是 q-main heavy GEMM + absorb bmm，
  #2 是 indexer prep/store。`#2.Q` 与 `#2.K` 互不依赖（输入只有 `q_lora`、`hidden`、`slot_mapping`），
  可在调度层并发；首版 dev 脚本分两个 stage 对齐。Hadamard 旋转**首版就用真 Sylvester ±1 矩阵**
  做 GEMM，不再走 identity 占位。
- Index logits 必须按 `Index GEMM -> gate reduce -> k_scale` 排序。这样 `k_scale` 只乘
  `[B,S_local]`，不对 `[B,I,S_local]` 做广播。
- CP 对 #3 之前的 Indexer 本地计算基本透明，只改变 `S_local` 与全局 position offset。
  `CP=ncp` 时近似 `S_local=S/ncp`；实际 page attention 下按 page/block interleave。
- #4 对 `S_local<=2048` 做直接返回；`S_local>2048` 才进入 radix select。#6 的 global merge-topK
  是同类 select，只是输入是 `CP*Ktop` 个候选。
- Sparse MQA 首版按 BF16 latent K。#7 只对本 device/rank owner 的 topK 位置生效，
  把本 device Gmem 中的 latent K gather 到 Lmem/scratch；非 owner 位置保留无效标记，
  由 #8 partial MQA 忽略。
  #8 无论 CP 是否为 1 都输出 `(partial_out, partial_lse)`，让 #10/#11 共用同一个 partial 接口。
- 最终 post 有两个版本：`CP=1` 走 #10，只做 V absorb + o_proj；`CP>1` 走 #11，
  在同一个 post kernel 里做 FA-reduce + V absorb + o_proj。

## Dev 脚本 Stage 顺序

新文件：`zeus_dev/model_state_dev/dev_glm5next_dsa_decode_test.py`。

建议参数：
- `--config {16b,next}`：选择 GLM5-Next-16B 或 GLM5-Next。
- `--stage ...`：单 stage 对齐。
- `--cp {1,4,8,16}`：in-process 模拟 CP ranks。
- `--batch B`、`--seqlen S`、`--seed`。
- `--mode {ref,zeus,both}`：Zeus 模式必须只调用已落地 kernel。

Stage 顺序：

1. `q_a_proj_norm` —— 对齐 #0.Q（`dsa_q_a_proj_norm`）。`hidden -> q_lora_norm`，验证 q_a_proj + RMSNorm 的精度链。
2. `kv_a_proj_norm_store` —— 对齐 #0.KV（`dsa_kv_a_proj_norm_store`）。`hidden + slot_mapping -> latent_kv_cache[slot] in-place`，验证只写被 slot 标记的行。
3. `q_main` —— 对齐 #1。`q_lora -> q_new [B,Nh,512]`，Dro=0，不 split / 不 RoPE。
4. `idx_q_weights` —— 对齐 #2.Q（`dsa_indexer_q_weights`）。`q_lora + hidden -> q_body [B,I,Di] + q_scale [B,I] + weights [B,I]`，
   验证 `wq_b` GEMM + Hadamard + 行 int8-proxy quant + gate × q_scale 的精度链。
5. `idx_k_prep_store` —— 对齐 #2.K（`dsa_indexer_k_prep_store`）。`hidden + slot_mapping ->
   index_k_body_cache[slot] + index_k_scale_cache[slot] in-place`，验证完整 LayerNorm + Hadamard +
   行 int8-proxy quant + scatter store。
6. `idx_logits` —— 对齐 #3。验证 `Index GEMM -> gate -> scale` 次序；输出
   `logits_local [B,S_local]`。
7. `local_topk` —— 对齐 #4。覆盖 `S_local<=2048` 直接返回与 `S_local>2048` radix select。
8. `cp_topk_merge` —— 对齐 #5/#6。`CP=1` 验证 bypass；`CP>1` 验证 all-gather local topK
   与 global merge-topK，输出应等价于在完整 `S` 上直接 topK。
9. `latent_gather` —— 对齐 #7。验证 global position 到本 rank page/slot 的映射，非 owner 位置置无效；
   首版只把本 device Gmem 里的 BF16 latent K gather 到 Lmem/scratch。
10. `sparse_mqa_partial` —— 对齐 #8。输出 `partial_out [B,Nh,512]` 与 `partial_lse [B,Nh]`；
    `CP=1` 时应等价于完整 sparse MQA latent out。
11. `post_o_proj_nocp` —— 对齐 #10。`CP=1` 跑 no-reduce 版本，只做 V absorb + o_proj。
12. `post_o_proj_cp` —— 对齐 #11。`CP>1` 跑 FA-reduce + V absorb + o_proj。
13. `decode_full_nocp` —— #0..#10 端到端 no-CP decode，验证本地 topK / sparse MQA / post 输出。
14. `decode_full_cp` —— #0..#11 端到端 CP decode，对 `--cp 4/8/16` 做同一输入同一权重的
    `out(cp=k)` vs `out(cp=1)` 对拍。

每个 stage 的 REF 侧都用 list 模拟 `cp_size` 个 rank；真实 collective 后续在
`sgl-kernel-zeus` 测试中替换。Zeus 侧未落地 stage 必须显式报 TODO，不能用 torch 拼装冒充 kernel。

## 附录 A：CP page/block interleave 约定

- DSA cache 配合 page attention。CP 之间按 page/block 做 interleave，而不是按连续长段切分。
- 令 `block_id = floor(pos / page_size)`。逻辑上 owner 可以写成
  `owner = block_group_id % CP`，其中 `block_group_id` 的粒度由实现决定。
- 当前 Zeus 设计里，每个 device 有 2 个 core，且每个 core 有独立的 Index Lmem。
  为了让两个 core 的 Index Lmem 都连续访问，先把同一 device 内的两个 core 看成
  **连续两个相邻 block**。例如 CP4 时，可以按 `2 blocks` 为单位做 round-robin：

```
block pair 0 -> rank0/core0, rank0/core1
block pair 1 -> rank1/core0, rank1/core1
block pair 2 -> rank2/core0, rank2/core1
block pair 3 -> rank3/core0, rank3/core1
block pair 4 -> rank0/core0, rank0/core1
...
```

- 对 Index logits/topK 来说，#3/#4 只需要看到本 rank 的 `S_local` 和全局 position offset。
  是否 CP、CP 是几路，在 all-gather local topK 之前都不改变 kernel 主体。

## 附录 B：TopK 细节

- `Ktop=2048`。
- `S_local <= Ktop`：直接返回 `[0..S_local)` 对应的有效 position，尾部 padding `-1`，logit padding `-inf`。
- `S_local > Ktop`：走 radix select，参考 `topk_fused_radix_select.md`：
  1. 分桶统计；
  2. 找阈值 bucket；
  3. collect candidates；
  4. 对候选做二次 select / 精排；
  5. 写出 `(logit, global_pos)`。
- CP merge-topK 对 `CP*Ktop` 个候选做同样的二次 select。输入规模小于原始 `S`，但仍要保留
  tie-break 规则，保证 `CP=1` 与 `CP>1` 的 REF 对齐稳定。

## 附录 C：FA-reduce 数学

Sparse MQA partial 在 rank `r` 上输出：

```
partial_out_r [B,Nh,Rkv]
partial_lse_r [B,Nh]
```

全局 merge：

```
m = max_r(partial_lse_r)
z = sum_r exp(partial_lse_r - m)
out_latent = sum_r exp(partial_lse_r - m) * partial_out_r / z
```

空集 rank 约定：
- `partial_lse_r = -inf`
- `partial_out_r` 任意，但 merge 权重为 0

`out_latent [B,Nh,512]` 随后进入：

```
attn_out = bmm(out_latent.transpose(0,1), w_vc[Nh,512,Dv]).transpose(0,1)
out = o_proj(attn_out.reshape(B, Nh*Dv))
```

不能用裸 `all_reduce(sum)` 替代 FA-reduce；必须带 LSE 权重。

## 附录 D：参考源码索引

- `python/sglang/srt/models/glm5_next.py`
  - `Glm5NextDecoderLayer.__init__`：DSA 层创建 `Glm5NextMLAAttention`。
  - `Glm5NextDecoderLayer.forward`：`prepare_attn -> self_attn -> prepare_mlp` 边界。
  - MHC 开启时 communicator 为 `MHCHybridNSACPLayerCommunicator` 或 `MHCLayerCommunicator`。
- `python/sglang/srt/models/deepseek_v2.py`
  - `DeepseekV2AttentionMLA.__init__`：`q_lora_rank/kv_lora_rank/qk_*_head_dim`、
    `Indexer`、`attn_mqa`、`o_proj`。
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`
  - `forward_absorb_prepare`：`qkv_a` split、q/kv norm、`q_b_proj`、`bmm w_kc`。
  - `forward_absorb_core`：`attn_mqa`、`bmm w_vc`、`o_proj`。
- `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`
  - `_get_q_k_bf16`、`_get_logits_head_gate`、`_get_topk_paged`。
- `python/sglang/srt/layers/attention/nsa_backend.py`
  - `NativeSparseAttnBackend.forward_decode`。
  - `_forward_flashmla_sparse` / `_forward_flashmla_kv`。
- `python/sglang/srt/layers/attention/nsa/transform_index.py`
  - topK position 到 page table / slot 的变换。

## 开发日志

### 2026-05-15 · 起点

- 创建 `glm5next_dsa_decode_dev.md`。
- 范围收敛为单个 DSA decode 模块：`hidden_state [B,H] -> self_attn.o_proj out [B,H]`。
- 配置合表覆盖 GLM5-Next-16B 与 GLM5-Next，并明确 GLM5-Next 实际 `Dro=0`。
- 算子表按 #0..#11 拆分，显式加入 CP>1 的 local-topK all-gather、global merge-topK、
  partial all-gather 与 FA-reduce post。
- 下一个任务：落 `dev_glm5next_dsa_decode_test.py` 骨架，先完成 REF stages 与 Zeus TODO guard。

### 2026-05-17 · 子步骤 #0 落地，拆成两个独立 kernel

- sgl-kernel-zeus 侧新增独立目录 `csrc/glm5next_dsa/`，落地两个独立 kernel：
  - `dsa_q_a_proj_norm`：hidden → q_a_proj → RMSNorm → `q_lora_out [T, Rq]`。
  - `dsa_kv_a_proj_norm_store`：hidden → kv_a_proj → RMSNorm → in-place 写
    `latent_kv_cache[slot_mapping[t]]`。
- 两个 kernel 互不 fuse。讨论结论：fuse 单 kernel 让 hidden 读一次同产 q_lora 和 latent
  cache 看起来更省 DRAM，但 host 契约（slot_mapping vs q_lora 输出形状不一致）会复杂
  很多；当前阶段以简洁为先，把 fuse 决策放到 porting 后再讨论。
- 完整 5 件套 + 独立 docs/slides 落地：
  - Triton 蓝本：`csrc/glm5next_dsa/dsa_{q_a_proj_norm, kv_a_proj_norm_store}_kernel.py`
  - CPU sim：`csrc/glm5next_dsa/sgl_dsa_{q_a_proj_norm, kv_a_proj_norm_store}_sim.c`
  - Host wrapper：`csrc/glm5next_dsa/dsa_{q_a_proj_norm, kv_a_proj_norm_store}_zeus.cpp`
  - Python API：`python/sgl_kernel_zeus/glm5next_dsa.py`
  - 测试：`tests/test_dsa_q_a_proj_norm.py`、`tests/test_dsa_kv_a_proj_norm_store.py`
  - 文档：`docs/dsa_q_a_proj_norm.md`、`docs/dsa_kv_a_proj_norm_store.md`
  - 幻灯：`docs/dsa_q_a_proj_norm_slides.html`、`docs/dsa_kv_a_proj_norm_store_slides.html`
- 当前执行路径是 CPU sim。GLM5-Next-16B 实形（H=2048, Rq=768, Rkv=512）测试均 PASS。
- 同步把本文件 #0 表项、计算流说明、dev 脚本 stage 顺序拆成 `q_a_proj_norm` /
  `kv_a_proj_norm_store` 两个 stage。
- 下一个任务：把 dev 脚本的 Zeus 侧也接通这两个 kernel（替换 SKIP TODO）。

### 2026-05-17 · 子步骤 #1 落地，q_main_absorb

- sgl-kernel-zeus 侧在 `csrc/glm5next_dsa/` 新增 `dsa_q_main_absorb`，把 `q_b_proj` +
  per-head `w_kc` absorb 收成一个 kernel。Dro=0 → 不 split q_pe / 不 RoPE，整个
  `Dqk=Dnope` 块直接喂 absorb。
- 数据流：`q_lora [T, Rq] bf16 → (per head)`
  - ① `q_h = bf16_round(q_lora · W_b[h].T)`，fp32 累加，匹配 model REF 的 bf16 边界；
  - ② `q_new[:, h, :] = bf16(q_h · w_kc[h])`，fp32 累加。
  `q_h` 在两次 GEMM 之间寄存器驻留，没有 DRAM scratch。
- 切核取向：当前契约 `grid=(1,)`、`CORE_NUM=1`；porting 时**按 head 切核**最自然——两次 GEMM
  都无需跨 core reduce。16B `Nh=32`、Next `Nh=64`，对 2/4/8/16 核都能整除。
- 完整 5 件套 + 文档：
  - Triton 蓝本：`csrc/glm5next_dsa/dsa_q_main_absorb_kernel.py`（Level-1 自检通过）
  - CPU sim：`csrc/glm5next_dsa/sgl_dsa_q_main_absorb_sim.c`
  - Host wrapper：`csrc/glm5next_dsa/dsa_q_main_absorb_zeus.cpp`
  - Python API：`python/sgl_kernel_zeus/glm5next_dsa.py::dsa_q_main_absorb`
  - 测试：`tests/test_dsa_q_main_absorb.py`（含 GLM5-Next-16B 实形
    `T=2, Rq=768, Nh=32, Dqk=128, Rkv=512`）
  - 文档：`docs/dsa_q_main_absorb.md`
  - 幻灯：`docs/dsa_q_main_absorb_slides.html`（10 张，含整体数据流 + 核切分页）
- dev 脚本 `stage_q_main` 已从 `finish_stage(... "dsa_q_main_absorb_fused" ...)` 改成
  跟 `stage_q_a_proj_norm` 一致的 REF vs Zeus 对拍写法，调用
  `sgl_kernel_zeus.dsa_q_main_absorb(q_lora, q_b_proj, w_kc)`。`--stage q_main --config 16b`
  在 `B=2, S=64` 下 `max_diff=0`（sim.c 与 REF 走同一份 bf16 round 顺序）。
- 同步更新本文件 #1 表项的 Zeus 现状栏。
- 下一个任务：开始 #2 `dsa_indexer_prep_store`（Indexer Q/K + LayerNorm + Hadamard +
  FP8 quant + index K cache store）。

### 2026-05-17 · 子步骤 #2 落地，按 Q / K 拆成两个独立 kernel

- 按对齐 #0 的 Q/KV 拆分模式，把原本一行的 #2 拆成两个独立 kernel：
  - **#2.Q** `dsa_indexer_q_weights`：`q_lora + hidden → q_body [B,I,Di] + q_scale [B,I]
    + weights [B,I]`。内部依次跑 `wq_b` GEMM → reshape `[B,I,Di]` → Hadamard 旋转
    (`× H_Di / √Di`) → 行 int8-proxy quant；同 kernel 还把 `weights_proj(hidden)` 算出 gate，
    再乘上 `q_scale` 与 `Di^-½` 得到 weights。把 q-quant 和 weights 收在一起，是因为
    `weights` 必须用 q_scale 的最终值，先把 q_scale 留在寄存器里直接消费比再读 DRAM 便宜。
  - **#2.K** `dsa_indexer_k_prep_store`：`hidden + slot_mapping →
    index_k_body_cache[slot] + index_k_scale_cache[slot]`。内部跑 `wk` GEMM → 完整
    LayerNorm（带 mean、var、weight、bias）→ Hadamard 旋转 → 行 int8-proxy quant →
    scatter store。body 与 scale 在 cache 中**作为独立张量**，不 pack 成 132B 行宽。
- Hadamard 决定：**首版就落地真 Sylvester ±1 Hadamard**，而不是 identity 占位。实现上把
  `H_Di [Di,Di]` 作为 bf16 weight 张量传入 kernel，做 `[..,Di] · H_Di · Di^-½` 的小型 GEMM。
  REF 同样改成真 Hadamard，dev 脚本里加 `hadamard_matrix(N)` helper。这样 REF 与 sglang
  `rotate_activation = hadamard_transform(x, N**-0.5)` 数值意义一致。
- FP8 决定：Zeus kernel 边界强制 bf16 / int32，所以 body 用 bf16 容器存 int8 量化值
  （范围 `[-127, 127]`），scale 用独立 fp32 张量。这两者通过两次 scatter 写进各自的
  cache 张量。porting 到芯片原生 `float8_e4m3fn` 时再调整 dtype。
- 完整 5 件套 + 独立 docs/slides 落地：
  - Triton 蓝本：`csrc/glm5next_dsa/dsa_indexer_{q_weights, k_prep_store}_kernel.py`
  - CPU sim：`csrc/glm5next_dsa/sgl_dsa_indexer_{q_weights, k_prep_store}_sim.c`
  - Host wrapper：`csrc/glm5next_dsa/dsa_indexer_{q_weights, k_prep_store}_zeus.cpp`
  - Python API：`python/sgl_kernel_zeus/glm5next_dsa.py::dsa_indexer_q_weights` /
    `::dsa_indexer_k_prep_store`
  - 测试：`tests/test_dsa_indexer_q_weights.py`、`tests/test_dsa_indexer_k_prep_store.py`
  - 文档：`docs/dsa_indexer_q_weights.md`、`docs/dsa_indexer_k_prep_store.md`
  - 幻灯：`docs/dsa_indexer_q_weights_slides.html`、`docs/dsa_indexer_k_prep_store_slides.html`
- dev 脚本 `stage_idx_prep_store` 被拆成 `stage_idx_q_weights` 与 `stage_idx_k_prep_store`，
  REF 侧用真 Hadamard helper 重算 q/k 旋转结果，Zeus 侧分别对接两个新 kernel。
  下游 stage（`idx_logits` 起）的 REF 改成消费这两个新 kernel 的输出契约。
- 下一个任务：开始 #3 `dsa_index_logits_fused`（Index GEMM → gate reduce → k_scale）。

### 2026-05-18 · 子步骤 #3 落地，dsa_index_logits

- sgl-kernel-zeus 侧在 `csrc/glm5next_dsa/` 新增 `dsa_index_logits`，把 #2.Q 输出的
  `q_body / weights` 与 #2.K 输出的 `k_body / k_scale` 合并成本地 Index logits。命名
  沿用同目录无 `_fused` 后缀风格（`dsa_q_a_proj_norm / dsa_q_main_absorb / dsa_indexer_*`）；
  原 TODO 占位 `dsa_index_logits_fused` 同步改名。
- 算子顺序硬约束：**Index GEMM → head-gate reduce → k_scale**。若先乘 k_scale，会对
  `[B, I, S]` 做 broadcast，多 `I-1` 倍读写。kernel 里 raw / gate / logits_tile 全部
  寄存器驻留，无 scratch DRAM round-trip。
- 数据流：
  - q_body `[B, I, Di]` fp8e4m3 + k_body `[B, S, Di]` fp8e4m3 → `tl.dot(q_b, tl.trans(k_tile), out_dtype=fp32)`，输出 raw `[I, BLOCK_S]` fp32；
  - gate `[BLOCK_S]` = Σ_i raw · weights_b（fp32 head reduce）；
  - logits_tile = gate · k_scale_tile（fp32 标量乘）→ 写 `logits[b, s_start:s_end]`。
- 切核取向：当前契约 `grid=(1,)`、`CORE_NUM=1`；porting 时按 **S 维**切核最自然
  （核间无依赖）；host 需补 `S % (BLOCK_S * CORE_NUM) == 0`。fp8 dot 多数硬件需 M ≥ 16，
  porting 时把 `BLOCK_I` pad 到 16，weights / q_body 在 i ∈ [I, BLOCK_I) 补 0，gate 不变。
- k_body **不当 weight DRAM**：sglang 把 index K cache 视作 activation pool（每 step 读最新 +
  历史），block_ptr 不加 `memory_type="weight"`。
- page-table 形态留给上游：本 kernel 直接接 gather 完的 `[B, S, Di]` k_body，与 dev 脚本
  `RankCache` 完全对齐；prod 上游 `index_k_body_cache[num_slots, Di]` + 每请求
  `kv_indices[B, S]` gather 思路同 #7 latent gather，后续单独落地。
- 完整 5 件套 + 独立 docs/slides 落地：
  - Triton 蓝本：`csrc/glm5next_dsa/dsa_index_logits_kernel.py`（Level-1 自检通过）
  - CPU sim：`csrc/glm5next_dsa/sgl_dsa_index_logits_sim.c`（自带 fp8e4m3_to_f helper）
  - Host wrapper：`csrc/glm5next_dsa/dsa_index_logits_zeus.cpp`
  - Python API：`python/sgl_kernel_zeus/glm5next_dsa.py::dsa_index_logits`
  - 测试：`tests/test_dsa_index_logits.py`（含 GLM5-Next-16B 实形 B=2, I=8, Di=128, S=64
    与更大的 S=128；7 用例全部 PASS）
  - 文档：`docs/dsa_index_logits.md`
  - 幻灯：`docs/dsa_index_logits_slides.html`（10 张，含 Index GEMM / head-reduce /
    k_scale 三步分解 + 顺序硬约束说明 + porting checklist）
- dev 脚本 `stage_idx_logits` 由 SKIP 改为 REF vs Zeus 真对拍：
  - `--config 16b --batch 2 --seqlen 64 --cp 1`：`max_diff=3.7e-9` PASS
  - `--config 16b --batch 2 --seqlen 64 --cp 4`：4 个 rank 全 PASS，`max_diff ≤ 3.7e-9`
  - dev 脚本里 `rank.index_body` 仍是 bf16 整数序列（初始化时的代理）；wire 时显式
    `.to(torch.float8_e4m3fn)` 对齐 #2.K 实际 cache dtype，REF 这里也用同一份 fp8 重算
    确保 dtype 对齐的精度对拍。
- 下一个任务：开始 #4 `dsa_local_topk_radix`（`logits_local [B, S] → topk_logits + topk_pos [B, Ktop=2048]`，
  `S_local ≤ Ktop` 走直接返回 + padding，`S_local > Ktop` 走 radix select）。
  → 已完成，见下条记录。

### 2026-05-21 · #4 `dsa_local_topk_radix` 五件套落地

- 新增五件套（`torch_zeus/sgl-kernel-zeus/csrc/glm5next_dsa/`）：
  - Triton 蓝本 `dsa_local_topk_radix_kernel.py`：双分支 `DIRECT_BRANCH: tl.constexpr` 编译期分流；
    `S_local ≤ Ktop` 走 block_ptr + `tl.where(-inf/-1)` 单核拷贝；`S_local > Ktop` 直译
    `topk_fused_radix_select.md` 的 7-phase（PWLF / core_send / barrier / SRAM 候选区
    / above + eq cumsum scatter），其中 `tl.zeus.{make_pwlf, pwlf, core_send, barrier}`
    是 chip-native intrinsic 占位，sim 不参与。Level-1 自检通过（`arg_names` 14 + 7 个 constexpr）。
  - CPU sim `sgl_dsa_local_topk_radix_sim.c`：三分支处理 + size-K min-heap（tie-break
    按 position 小者优先），与 REF `ref_local_topk` 数值对齐 enough for set 比较。
  - Host wrapper `dsa_local_topk_radix_zeus.cpp`：`TORCH_CHECK` 强制
    `logits=fp32 + positions=i32 + 输出 fp32 + i32`、`positions.size(0)==logits.size(1)`、
    `top_*.shape==[B, Ktop]`、`S_local >= 0`、维度 ≤ INT32_MAX。
  - Python API `dsa_local_topk_radix(logits, positions, *, Ktop, top_logits=None, top_pos=None)`：
    int64 positions 走 `_DSA_LOCAL_TOPK_POS_BUFS` stable-buffer 兜底（graph-capture 安全）；
    输出未传时自动 `torch.empty`。
  - 测试 `tests/test_dsa_local_topk_radix.py`：11 个用例覆盖 direct partial / direct equal /
    radix / empty / CP-4 partition consistency / int64 positions / 非 fp32 logits reject /
    shape mismatch reject / GLM5-Next-16B real shape (S=2048 direct=, S=4096 radix) /
    preallocated outputs。
- 文档 + 幻灯：`docs/dsa_local_topk_radix.md`（§1 定位 / §2 输入输出 / §3 三分支 +
  7-phase 拆解 / §4 constexpr + 派生 / §5 三层职责 / §6 测试矩阵 / §7 porting checklist）+
  `docs/dsa_local_topk_radix_slides.html`（10 张 slide，dark theme，含 direct
  例子 + radix Phase 1–7 拆解 + CP 兼容性页 + porting checklist）。
- `csrc/common_extension.cpp` / `include/sgl_kernel_zeus_ops.h` / `setup.py` /
  `python/sgl_kernel_zeus/__init__.py` 四处注册 + re-export 同步。
- dev 脚本 `stage_local_topk` 由 SKIP 改为 REF vs Zeus 真对拍：
  - `--config 16b --batch 2 --seqlen 64 --cp 1`：rank 0 direct (S=64) PASS
  - `--config 16b --batch 2 --seqlen 64 --cp 4`：4 个 rank 全 PASS（S_local ≈ 16~17 全走 direct）
- 下一个任务：#5 `dsa_cp_topk_allgather` + #6 `dsa_cp_merge_topk`，把 CP&gt;1 的
  跨 rank 候选合并通路打通。

### 2026-05-18 · #3 Triton 蓝本切换到 CORE_NUM=2 沿 S 维切核

- `dsa_index_logits_kernel.py` 默认 `CORE_NUM` 从 1 改为 2，沿 **S 维**做 2-way 切核。
  每个核拿连续半段 `[core_id * s_per_core, (core_id+1) * s_per_core)`，内层
  `cdiv(s_per_core, BLOCK_S)` 个 S-tile 走完；`q_body / weights` 只读全核共享、
  `k_body / k_scale / logits` 沿 S 段独立写不冲突。
- prod 形 `Ktop=2048` + 默认 `BLOCK_S=64`：每核做 16 个内层 tile × 外层 batch 2 轮 = 32
  次 fp8 dot / core，正好 `2048 % (64 * 2) == 0` 满足整除约束。
- porting 真 zbin 时 host 同步：`zertLaunchKernel grid` 改成 `CORE_NUM`，并加
  `TORCH_CHECK(S % (BLOCK_S * CORE_NUM) == 0)`。当前执行路径仍是 `grid=(1,)` CPU sim，
  sim.c 是 scalar 嵌套 for 不感知 CORE_NUM，无需同步改动。
- 文档与幻灯同步：`docs/dsa_index_logits.md` §3.1 数据流图加 "q/weights 全核共享 vs
  k/scale/logits 沿 S 切核" 注，§3.2 例子换成 prod 形 `S=2048` 展示 2 核 × 16 tile，
  §4.1 Constexpr 表 CORE_NUM 默认改 2、§4.2 加 `core_id / s_per_core / s_start_core`
  三行派生变量，§7 porting 项 #3 更新；`docs/dsa_index_logits_slides.html` slide 1
  badge、slide 8 切核表 + 蓝本伪码、slide 10 checklist 全部更新。
- 不变项：sim.c、host wrapper（仍 `grid=1`）、Python API、test、common_extension /
  setup.py / __init__.py 都不动。`pytest tests/test_dsa_index_logits.py` 仍 7/7 PASS
  （走 sim.c grid=1 路径，与 CORE_NUM 改动正交）。

### 2026-05-18 · 子步骤 #7 落地，dsa_latent_k_gather

- sgl-kernel-zeus 侧在 `csrc/glm5next_dsa/` 新增 `dsa_latent_k_gather`，把上游
  `transform_index_page_table_decode` 输出的 `slot_indices [B, Ktop]` 按 slot 间接 gather
  本 rank `latent_kv_cache [num_slots, Rkv] bf16` 的对应行到 `K_local [B, Ktop, Rkv] bf16`；
  非 owner 位置（slot < 0）写零行。
- 数据流：纯数据搬运（bf16 in → bf16 out，无算术）。per-(b, k) scalar slot lookup →
  `slot_safe = max(slot, 0)` clamp 防 OOB → `is_valid_bf16 = (slot >= 0).to(bf16)` 标量 mask →
  `for r_blk: row = load(cache[slot_safe, r..]); row * is_valid; store(K_local[b, k, r..])`。
  invalid 也读 cache 第 0 行然后乘 0，避免 scalar 分支破坏流水线（vector ALU 乘比 control-flow
  分流便宜，符合 sgl-kernel-zeus dev skill §"规避 where / bool 向量操作"）。
- 切核取向：**沿 Ktop 维 2-way 切核**（CORE_NUM=2）。Core 0 写 `K_local[:, 0..1024, :]`，
  Core 1 写 `[:, 1024..2048, :]`；`slot_indices / latent_kv_cache` 两核共享只读。host 约束
  `Ktop % CORE_NUM == 0`（Ktop=2048 对 2/4/8/16 都满足）。"按 N_eff 维并行"的直接体现：
  N_eff ≤ Ktop 是 token 选择的天然维度，切 Ktop 比切 Rkv 更对称。
- 上游对接：依赖 sglang `transform_index_page_table_decode` 输出的 `slot_indices`，本算子
  **不重复做 owner 判断**，签名极简（slot_indices / cache / K_local）。`cache_index_gather`
  （mamba 侧）是直接蓝本——`(slot, 0)` advance pattern 完全相同，只是 indices 从 1D 升到 2D
  并加 invalid mask。
- 完整 5 件套 + 独立 docs/slides 落地：
  - Triton 蓝本：`csrc/glm5next_dsa/dsa_latent_k_gather_kernel.py`（Level-1 自检通过）
  - CPU sim：`csrc/glm5next_dsa/sgl_dsa_latent_k_gather_sim.c`（per-row `memcpy` / `memset`）
  - Host wrapper：`csrc/glm5next_dsa/dsa_latent_k_gather_zeus.cpp`
  - Python API：`python/sgl_kernel_zeus/glm5next_dsa.py::dsa_latent_k_gather`（含
    `_DSA_LATENT_GATHER_SLOT_BUFS` stable buffer cast，graph-capture 安全）
  - 测试：`tests/test_dsa_latent_k_gather.py`（11 用例全部 PASS，含 GLM5-Next-16B 实形
    `B=2, Ktop=2048, Rkv=512, num_slots=4096` × 3 种 ownership 模式 + int64→int32 cast +
    拒绝用例）
  - 文档：`docs/dsa_latent_k_gather.md`
  - 幻灯：`docs/dsa_latent_k_gather_slides.html`（10 张，含 GLM5-Next-16B 核切分例子 +
    上游 transform_index_page_table_decode 对接图 + porting checklist）
- dev 脚本 `stage_latent_gather` 由 SKIP 改为 REF vs Zeus 真对拍：
  - `--config 16b --batch 2 --seqlen 64 --cp 1`：`max_diff = 0.0` PASS（纯数据搬运，bf16 严格相等）
  - `--config 16b --batch 2 --seqlen 64 --cp 4`：4 个 rank × 2 batch = 8 次 kernel 调用全 PASS，
    `max_diff = 0.0`
- 下一个任务：开始 #8 `dsa_decode_sparse_mqa_partial_bf16`（`q_new [B, Nh, Rkv] + K_local [B, Ktop, Rkv]`
  → partial sparse MQA，two-pass softmax，输出 `partial_out [B, Nh, Rkv] + partial_lse [B, Nh]`）。

### 2026-05-19 · 子步骤 #7 切到 skip-invalid + 双输出（K_local + mask）

- 设计动因：v1 用 "always-load + 乘 bf16 mask" 把 invalid 行写 0，每个 invalid 位置仍吃 1 KB
  Gmem cache read + 1 KB Lmem K_local write。CP=4 短 history 实测 N_eff ≈ 17 / 2048（invalid
  占 99%），4 MB / 4 MB 的 DRAM 流量里 ~99% 都是无效操作。
- 新契约：kernel 内部用 per-position 标量 `if slot >= 0:` 跳过 invalid 位置——invalid 不 load
  cache、不 store K_local，K_local invalid 行保留 caller `torch.empty` 的 garbage；同时**新增
  第二个输出 `mask [B, Ktop] bf16`**（1.0=valid / 0.0=invalid），供下游 #8 sparse MQA 在
  `scores * mask + (1 - mask) * -1e9` 后做 softmax 时把 invalid 位置 score 设 -∞，garbage
  K_local 行在 softmax 输出中天然贡献 0。
- 实测收益（GLM5-Next-16B + CP=4 + S=64）：DRAM 流量从 4 MB R + 4 MB W 降到 ~17 KB R +
  ~17 KB K_local W + 4 KB mask W，**省 ~99%**。
- 为什么用标量 if 而非 vector where：sgl-kernel-zeus dev skill 禁的是 `tl.where` 向量分支和
  `tl.load(raw_ptr, mask=)`；per-position 标量分支是基本块谓词，lower 干净（与
  `cache_index_gather_kernel.py` 的 scalar advance pattern 同源）。
- mask 选 bf16 而非 bool/int8：下游 #8 拿到 fp32 scores 后直接 `scores * mask` broadcast 乘法，
  bf16 与 fp32 broadcast 多数硬件原生支持，省一次 cast。代价 mask 8 KB（vs bool 0.5 KB），
  对 Lmem 容量可忽略。
- 改动同步：
  - Triton 蓝本：新 sig `(slot_indices_ptr, latent_kv_cache_ptr, k_local_ptr, mask_ptr, batch,
    ktop, num_slots, rkv, CORE_NUM, BLOCK_R)`；外层 batch + Ktop 循环里每 (b, k) 先写 mask 标量、
    再标量 `if slot >= 0:` 进入 R-tile 内层；删 `slot_safe` / `is_valid_bf16` / `row * is_valid`。
  - CPU sim：args struct 加 `void* mask` 字段；invalid 分支 K_local 不动（不 memset）、mask 写
    `BF16_ZERO`；valid 分支 K_local memcpy（OOB 时 memset 0 兜底）、mask 写 `BF16_ONE`。
  - Host wrapper / `common_extension.cpp` / `ops.h`：函数签名加 `at::Tensor& mask`，op def 加
    `Tensor! mask`，TORCH_CHECK 加 mask dtype/shape，`ScopedSimMetadata` 加 `sim_out(mask)`。
  - Python API：返回 `(k_local, mask)` tuple；`mask` 可选 kwarg，缺省时 `torch.empty([B, Ktop],
    bf16, device='zeus')`。
  - 测试：新增 sentinel poison 测试（`-99.0` 预填 K_local，断言 invalid 行未被改动），比较前用
    `torch.where(mask>0, K_got, 0)` 归一化 Zeus garbage；新增 mask shape / mask dtype 拒绝用例。
    14/14 PASS（含 GLM5-Next-16B 实形 × 3 ownership 模式）。
  - dev 脚本 `stage_latent_gather`：解 tuple，先比 mask（bit-exact），再用 mask normalize
    K_local garbage 后比 K_local（bit-exact）。`--cp 1` 与 `--cp 4` 均 `max_diff=0.0` PASS。
- 下一个任务不变：开始 #8。新的契约里 #8 需要消费 `K_local + mask`（不是 `K_local + slot_indices`），
  在 softmax 前用 mask 把 invalid scores 压成 -∞。

### 2026-05-19 · 子步骤 #7 切核策略：从"分工切 Ktop"改为"broadcast 重复 gather"

- 设计动因：下游 #8 sparse MQA partial 计划按 **Q-head 维**切核（core 0 处理 head 0..Nh/2、
  core 1 处理 Nh/2..Nh）。每个 core 需要看完整 `[B, Ktop, Rkv]` K_local 才能算自己半数 Q-head
  对全部 Ktop 行的 attention 分数。如果 #7 沿 Ktop 切核（core 0 拿一半、core 1 拿另一半），
  #8 就必须做核间取数。为了让 #8 实现简单，#7 改成两核都跑完整 Ktop（duplicate work），
  各自的 Lmem 拿到一份完整 K_local + mask 副本。
- kernel 改动：删 `k_per_core = ktop // CORE_NUM` 和 `k_start_core = core_id * k_per_core`；
  外层循环从 `for k_off in range(k_per_core)` 改为 `for k in range(ktop)`，两核相同。
  `tl.program_id(0)` 保留但 kernel body 不消费（注释 `_core_id`），留给未来核间 DMA 方案。
- host 改动：去掉 `Ktop % CORE_NUM == 0` 校验（不再需要整除）。
- 代价：两核各自做完整 Ktop 的 Gmem read + Lmem write，**Gmem 带宽 2×、Lmem 占用 2×**（前者
  是浪费，后者是 #8 必需的）。GLM5-Next-16B + CP=4 实测每核 ~17 KB cache read + ~17 KB
  K_local write + 4 KB mask write，两核合计 ~76 KB——绝对值很小，没必要现在优化。
- **未来优化 TODO**（等 #8 落地、profile 后再决定）：切核间 DMA 方案——两核仍然分工各跑一半
  Ktop，但每核取一批 SRAM 行后，在写本核 Lmem 的同时发起 core-to-core 传输到对方 core；
  对方 core 接收后写入自己的 Lmem 互补半段。最终每核 Lmem 仍然完整（行为等价当前版本），
  但 Gmem 读带宽减半。这需要 Zeus 后端提供 core-to-core 同步/传输原语。
- 测试同步：删 `test_dsa_latent_k_gather_rejects_ktop_not_divisible_by_core_num`，新增
  `test_dsa_latent_k_gather_accepts_odd_ktop`（用 Ktop=7 验证无整除约束）。14/14 PASS。
- dev 脚本 `stage_latent_gather` 行为不变（仍 host 单线程 sim），`--cp 1` / `--cp 4` 均
  `max_diff = 0.0` PASS。

### 2026-05-19 · 子步骤 #8 落地，dsa_sparse_mqa_partial（FA-v2 + Q-head 切核 + 无核间 reduce）

- sgl-kernel-zeus 侧在 `csrc/glm5next_dsa/` 新增 `dsa_sparse_mqa_partial`，消费 #1 输出
  `q [B, Nh, Rkv]` 与 #7 输出 `(k_local [B, Ktop, Rkv], mask [B, Ktop])`，按
  **FlashAttention v2 在线 softmax** 做 sparse MQA，输出 `partial_out [B, Nh, Rkv] bf16`
  与 `partial_lse [B, Nh] fp32`。MQA：同 batch 内所有 Q head 共享同一份 K_local；Q 在
  latent 空间 (Rkv=512，#1 absorb 输出)，无 q_pe / 无 RoPE（Dro=0）。
- **核心设计决定 — Q-head 切核，无核间 reduce**：CORE_NUM=2 沿 Nh 维做 disjoint partition
  （core 0 → heads [0, Nh/2)、core 1 → [Nh/2, Nh)）。两核读完全相同的 K_local + mask（#7
  broadcast 到各自 Lmem 的副本），写不重叠的 `partial_out / partial_lse` head slice。
  **两核之间无 cross-core 通信、无 LSE-merge、无同步**——这恰恰是 #7 选 broadcast-gather
  的最终回报。如果走 K-axis split（每核处理一半 Ktop），才需要 intra-device LSE-merge。
- 算法：单 pass FA-v2 在线 softmax。per (b, core) 保 (m_state, l_state, o_state) 寄存器
  状态，K-tile 循环里：
  - score = scaling · Q · K_tile^T（fp32 dot）
  - score = score · mask + (1 − mask) · (−1e9)（用乘法掩码避免 NaN 风险）
  - row_max → m_new → alpha = exp(m_state − m_new) → p = exp(score − m_new)
  - l_state = alpha · l_state + Σ p
  - o_state = alpha · o_state + p.to(bf16) · K_tile（fp32 dot）
  - m_state = m_new
  最后 finalize：`no_valid = m_state ≤ −5e8` 走空 rank guard → lse=−inf, out=0；否则
  lse = m + log(l), out = (o_state / l_state).to(bf16)。
- `tl.where` 用法：只在 finalize 阶段对空 rank sentinel 用（lse → −inf / out → 0），不在主
  循环算分 / 累加里用——符合 sgl-kernel-zeus dev skill §"规避 where" 的 OOB sentinel 边界。
- 切核取向：当前契约 `grid=(1,)`、`CORE_NUM=1`；porting 时把 grid 改为 `CORE_NUM=2` 并补
  `TORCH_CHECK(Nh % CORE_NUM == 0)`（16B `Nh=32`、Next `Nh=64` 都满足 2/4/8/16 整除）。
- K_tile 寄存器驻留：score GEMM 与 O update 在同一 K-tile 内共用 K_tile（两次 `tl.dot(...
  k_tile)`），编译器自然分配同一寄存器，**K_tile 每 step 只读一次 Lmem**。
- 完整 5 件套 + 独立 docs/slides 落地：
  - Triton 蓝本：`csrc/glm5next_dsa/dsa_sparse_mqa_partial_kernel.py`（Level-1 自检通过）
  - CPU sim：`csrc/glm5next_dsa/sgl_dsa_sparse_mqa_partial_sim.c`（single-pass softmax，
    double 累加保住 fp32 精度，bf16 RNE 写出）
  - Host wrapper：`csrc/glm5next_dsa/dsa_sparse_mqa_partial_zeus.cpp`
  - Python API：`python/sgl_kernel_zeus/glm5next_dsa.py::dsa_sparse_mqa_partial`
  - 测试：`tests/test_dsa_sparse_mqa_partial.py`（12 用例：小 smoke + GLM5-Next-16B 实形
    `B=2, Nh=32, Ktop=2048, Rkv=512` × valid_frac {1.0, 0.25, 0.05} + 空 rank guard +
    sentinel poison 测试 + 预分配输出 + 3 个拒绝用例，全 PASS）
  - 文档：`docs/dsa_sparse_mqa_partial.md`
  - 幻灯：`docs/dsa_sparse_mqa_partial_slides.html`（10 张，含切核策略对比图 + FA-v2 伪码 +
    16B 实形例子 + porting checklist）
- dev 脚本 `stage_sparse_mqa_partial` 由 SKIP 改为 REF vs Zeus 真对拍：从 `decoded["ranks"]`
  + `decoded["global_topk_pos"]` 构造每 rank 的 `K_local + mask` 输入，转发 Zeus kernel；
  partial_lse 比较时把 `-inf` empty heads 单独 bit-exact 检查、finite 段 atol/rtol=5e-3。
  - `--config 16b --batch 2 --seqlen 64 --cp 1`：rank 0 PASS，`partial_out max_diff=0.0`、
    `partial_lse max_diff=4.8e-7`
  - `--config 16b --batch 2 --seqlen 64 --cp 4`：4 个 rank 全 PASS，`partial_out
    max_diff ≤ 1.5e-5`、`partial_lse max_diff ≤ 2.4e-7`
- ✅ #10 `dsa_post_o_proj_no_cp` 落地（CP=1 no-reduce post：V absorb + o_proj）：
  - 完整 5 件套 + 独立 docs/slides 落地：
    - Triton 蓝本：`csrc/glm5next_dsa/dsa_post_o_proj_no_cp_kernel.py`（Level-1 自检通过）。
      dual-GEMM 单 kernel：per-head V absorb `acc_v = attn_latent[:, h, :] · w_vc[h]` fp32
      → bf16 round 边界 → o_proj GEMM2 跨 head accumulate 到单 fp32 `o_acc [BLOCK_T, BLOCK_H]`
      → 最后一次 RNE 写出。`o_acc` 全程驻 SRAM（16B 仅 16 KB）；`attn_h` 寄存器驻留。
    - CPU sim：`csrc/glm5next_dsa/sgl_dsa_post_o_proj_no_cp_sim.c`（scalar h × t × d 的
      V absorb，再 h × t × j × d 的 o_proj contribution，全 fp32 累加，最终 RNE 写出）。
    - Host wrapper：`csrc/glm5next_dsa/dsa_post_o_proj_no_cp_zeus.cpp`。
    - Python API：`python/sgl_kernel_zeus/glm5next_dsa.py::dsa_post_o_proj_no_cp`。
    - 测试：`tests/test_dsa_post_o_proj_no_cp.py`（8 用例：3 个 proxy + GLM5-Next-16B 实形
      `B=2, Nh=32, Rkv=512, Dv=128, H=2048` + 预分配输出 + 3 个拒绝用例，全 PASS；
      proxy max_diff = 0，16B 实形误差量级 ≈ bf16 ULP，atol=3e-2 / rtol=2e-2）。
    - 文档：`docs/dsa_post_o_proj_no_cp.md`。
    - 幻灯：`docs/dsa_post_o_proj_no_cp_slides.html`（10 张，含 dual-GEMM 数据流 + 16B
      实形 + bf16 边界为什么不能省 + porting checklist；推荐 H 切核 vs. Nh 切核分析）。
- dev 脚本 `stage_post_o_proj_nocp` 由 SKIP 改为 REF vs Zeus 真对拍：用 `decoded["attn_latent"]`
  作输入（即 #8 partial_out 在 CP=1 路径下的结果），权重直接复用 `ctx.weights["w_vc"]` /
  `["o_proj"]`，转发 `sgl_kernel_zeus.dsa_post_o_proj_no_cp`。
  - `--config 16b --batch 2 --seqlen 64 --cp 1`：PASS，`out max_diff=3.7e-9`、`mean_diff=9.1e-13`
    （远小于 bf16 ULP——sim 与 REF 数学上完全等价；bit-level 差异源于 fp32 累加顺序）。
- 下一个任务：开始 #11 `dsa_fa_reduce_o_proj_cp`（CP&gt;1 后处理：先 online-softmax merge
  各 rank 的 `(partial_out, partial_lse)` → 得到 final `attn_latent` → V absorb + o_proj，
  其中 V absorb / o_proj 段语义与 #10 完全一致），或者补 #4 `dsa_local_topk_radix` /
  #6 `dsa_cp_merge_topk` 把 topK 通路打通。

### 2026-05-26 · dev_dsa_attn wrapper 落地 + dsa_latent_k_gather 加 batch 维

- 在 `zeus_dev/model_state_dev/glm5next_modules/` 加 `dev_dsa_attn.py`，把 DSA decode
  抽成 `Glm5NextDsaAttn` 模块（同 `Glm5NextLinearAttn` / `Glm5NextMoE` 的形态）。
  暴露 `__init__` + `init_state(B, seqlen)` + `forward (REF)` + `forward_zeus`。
- `_pack_zeus()` 一次性把 6 个 LocalMem weight (q_a / kv_a / q_b / w_kc / wk_idx /
  h_di) + 9 个 plain Zeus tensor 全部装包，**消除每 forward 重复 from_tensor 的
  per-forward repack**。
- `_get_scratch(B)` 池化 per-B buffer：slot_mapping / kv_new_cache / body_cache
  (fp8 LocalMem) / scale_cache。
- 关键：sparse-MQA 输入的 K_local / K_local_T / mask 三个 LocalMem buffer 在 scratch
  里 **zero-init 一次性分配**，跨 forward 用 `copy_from_linear` 原位写入。理由：
  gather 只写 valid 位置 → invalid 位置因为 buffer 初始 zero-init 始终 finite 0.0 →
  下游 sparse_mqa_partial 算术 mask 把 invalid 位置乘 0，数值上没影响。
- 同步把 `dsa_latent_k_gather` 加 batch 维：v1 `cache [num_slots, Rkv]` 共享池 →
  v1.5 `cache [B, num_slots, Rkv]` per-batch。kernel 内 batch loop 在最外层 BLOCK_B=1。
  改了 sim.c / triton blueprint / host cpp / Python wrapper / test。
  消除原 dev_dsa_attn 的 per-batch CPU loop。
- 验证：16b/both seqlen=64 PASS max_diff=1.22e-4；next/both seqlen=128 PASS。

### 2026-05-26 · v2 设计迭代：paged-attention 集成（**仅设计 + 单个 kernel 落地，其余 kernel 工作待续**）

- **触发**：dev_dsa_attn forward_zeus 跑起来后观察到 `[ZEUS ToGDG Stub]`（来自
  `scratch["body_cache"].cpu()` 拉 LocalMem 到 CPU 做 history concat）、`torch.cat([
  history.*, new_*])` CPU 操作、`from_tensor(full_*)` 每步 LocalMem repack 等
  host-coordinated history 的开销。讨论后明确**目标是 paged attention 集成**——history
  应由 paged-attention manager 持有，DSA 模块只接 slot 接口。
- **设计决策**：见上面新增的 "v2 设计迭代：Paged-Attention 集成 + Device-Side History"
  章节。核心结论：
  1. `dsa_kv_a_proj_norm_store` / `dsa_indexer_k_prep_store` v1 接口（单池 +
     slot_mapping）**已经是 paged 友好**，caller 改 slot_mapping 计算方式即可，
     kernel 不变。
  2. 2026-05-26 的 `dsa_latent_k_gather` batched 改动，其实是 paged 的 page_size=
     total_slots 退化形式。真 paged 需要回到单池接口 + 物理 slot；可保留 batched
     版作 dev/test path，paged 路径独立出 paged 变种（或 caller 端先翻译位置）。
  3. **`dsa_index_logits` 是唯一需要重写的 kernel**——v1 假设 history 在 per-batch
     连续 `[B, S, Di]`，paged 下需要通过 block_table + page_size 间接寻址 pool。
  4. helper ops：`dsa_translate_topk_positions`（中等必需）、`dsa_compute_new_slot`
     / `dsa_advance_seqlens`（可选）。
- **本次实施**：
  - 文档 v2 章节 + appendix E（见下）+ 本 dev log 条目。
  - 新增 `dsa_index_logits_paged` 五件套（sim.c / triton blueprint / host cpp /
    Python wrapper / 文档）。`dsa_index_logits` v1 保留不动，paged 变种作为
    新 op 共存（迁移期）。
- **后续 kernel 工作清单**（按优先级）：
  1. `dsa_latent_k_gather_paged`（或确认 caller-translate + 单池版的方案）—
     涉及 sim + triton + host + Python wrapper + test，工作量 ≈ 当前 batched 版改回去。
  2. `dsa_translate_topk_positions`（新 op）—— 简单 elementwise，从 block_table
     + page_size 翻译 logical → physical slot。
  3. `dsa_compute_new_slot`（新 op）—— 可选，elementwise 标量。
  4. 重写 `dev_dsa_attn.forward_zeus` 用 paged 接口；去掉所有 host-side
     coordination。需要先有 paged-attention manager mock（或 fake state）。
  5. 跑 paged + non-paged 两条路径的数值对拍（在小 page_size=1 等价情形下应 bit-exact）。
- **不在本次范围**：上游 paged-attention manager 本身（block table 分配 / page
  eviction / KV preemption）—— 那是 sglang 层面的事，DSA 模块只消费它的输出。

## 附录 E：Paged-Attention State 管理约定

DSA 模块本身不持有 history。每步 decode 由上层（paged-attention manager）传入：

```
paged_state = {
    # 共享池（manager 一次性分配, 跨步复用; page eviction 时由 manager 调整）
    "latent_kv_pool":   torch.Tensor[total_slots, Rkv]      bf16  Gmem,
    "index_body_pool":  LocalMemTensor[total_slots, Di]     fp8,
    "index_scale_pool": torch.Tensor[total_slots]           fp32  Gmem,

    # 每序列元数据（manager 维护; -1 表示未分配 page）
    "block_table":      torch.Tensor[B, max_pages_per_seq]  int32,
    "seq_lens":         torch.Tensor[B]                     int32,

    # 本步要写入的物理 slot（manager 在 step 开始时算好）
    "new_slot_mapping": torch.Tensor[B]                     int32,

    # 配置常量
    "page_size":        int,
}
```

### 一步 decode 内 manager 与 DSA 的交互序列

```
manager (step N+1 开始):
    if seq_lens[b] + 1 已超过当前 block_table 的容量:
        allocate new page; block_table[b, ...] 加一行 physical_page_id
    new_slot_mapping[b] = block_table[b, seq_lens[b] // page_size] * page_size
                       + (seq_lens[b] % page_size)

DSA forward_zeus(hidden_z, paged_state):
    # #0.KV / #2.K 写 pool[new_slot_mapping[b]]
    dsa_kv_a_proj_norm_store(hidden, ..., new_slot_mapping, latent_kv_pool)
    dsa_indexer_k_prep_store(hidden, ..., new_slot_mapping, index_body_pool,
                             index_scale_pool)

    # #3 paged 变种: 用 block_table + seq_lens 间接寻址
    logits = dsa_index_logits_paged(q_body, weights, index_body_pool,
                                    index_scale_pool, block_table, seq_lens,
                                    page_size, max_logical_s)

    # #4 local topk: positions 是 logical (0..max_logical_s)
    top_logits, top_pos = dsa_local_topk_radix(logits, arange_positions, Ktop)

    # #7 caller 翻译 logical top_pos -> physical slot
    phys_slot = block_table.gather(1, top_pos // page_size) * page_size
              + (top_pos % page_size)
    K_local, mask = dsa_latent_k_gather(phys_slot, latent_kv_pool)

    # #8 / #10 不变

manager (step N+1 结束):
    seq_lens += 1
```

### Slot 计算规则

```
new_slot_mapping[b] = block_table[b, seq_lens[b] // page_size] * page_size
                   + (seq_lens[b] % page_size)
```

### Logical → Physical 位置翻译

```
# top_pos[b, k] ∈ [0, seq_lens[b]) 是 logical position (在 logits 数组里的索引)
logical_page_idx = top_pos // page_size            # [B, Ktop]
slot_in_page     = top_pos %  page_size            # [B, Ktop]
phys_page        = block_table.gather(1, logical_page_idx)   # [B, Ktop]
phys_slot        = phys_page * page_size + slot_in_page      # [B, Ktop]
```

### 与 v1 单卡测试路径的兼容

把 page_size 设成一个很大的值（≥ max_seqlen）、block_table[b, 0] = b、
seq_lens[b] = current_step，即可让 paged 路径退化成 v1 single-pool 路径。这种
退化模式可作为 paged 实现的数值对拍 oracle（paged out vs v1 out 在 page_size=
total_slots / block_table=identity 下应该 bit-exact）。

### 2026-05-26 续 · 子步骤 #3 paged 变种落地（`dsa_index_logits_paged`）

- 在 `csrc/glm5next_dsa/` 新增完整五件套：
  - Sim：`sgl_dsa_index_logits_paged_sim.c`（scalar `(b, s)` 嵌套, 内部
    `phys_slot = block_table[b, s/PS] * PS + s%PS` 间接寻址 pool；`s >= seq_lens[b]`
    出 -1e30；`phys_page < 0` / `phys_slot >= total_slots` OOB sanity 兜底）
  - Triton 蓝本：`dsa_index_logits_paged_kernel.py`（CORE_NUM=2 沿 max_logical_s
    切核；`make_block_ptr` 仅用于 q_body / weights / logits；body_pool / scale_pool
    的间接 slot 索引用 scalar offset + `tl.load(base + phys_slot * stride + ...)`，
    这是 paged 的固有访问 pattern；Level-1 自检通过）
  - Host wrapper：`dsa_index_logits_paged_zeus.cpp`（dtype / dim 校验 + `max_pages *
    page_size >= max_logical_s` + int32 范围）
  - Python API：`python/sgl_kernel_zeus/glm5next_dsa.py::dsa_index_logits_paged`
    + `__init__.py` re-export + 注册进 `_LOCALMEM_AWARE_OPS` 默认集合
  - 文档：`docs/dsa_index_logits_paged.md`
- 注册链：`include/sgl_kernel_zeus_ops.h` 加 fwd decl；`csrc/common_extension.cpp`
  加 `m.def + m.impl`；`setup.py` SIM_SOURCES / HOST_SOURCES 双列表加新文件
- 验证：
  - **退化等价测试**：`page_size = max_logical_s` & `block_table[b, 0] = b` 时
    paged 输出与 v1 `dsa_index_logits` 在等效输入下 **bit-exact**（max_diff=0）
  - **非 trivial paged 测试**：`B=2, PS=16, seq_lens=[40,24], max_logical_s=48,
    block_table=[[3,1,5,-1],[2,6,-1,-1]]`；invalid 区段 (s >= seq_lens[b]) 全部 -1e30；
    手算两个 valid 位置 (seq 0 page 3 slot 0; seq 1 page 6 slot 1) 与 kernel 输出
    bit-exact 对得上
- v1 `dsa_index_logits` 保留不动，paged 变种作为新 op 共存（迁移期）

### 剩余 paged kernel 工作 roadmap

按优先级排序，下一批 PR 推进：

1. **`dsa_latent_k_gather_paged`** —— 把 v1.5 batched `[B, num_slots, Rkv]` 回退到单池
   `[total_slots, Rkv]` + slot_indices 内放物理 slot。caller 端先通过 block_table
   翻译 logical top_pos → physical slot 再喂入。涉及 sim + triton + host + Python
   wrapper + test 五件套。**工作量 ≈ 当前 batched 版改回去** + 增加迁移期共存
2. **`dsa_translate_topk_positions`** （新 op）—— 简单 elementwise：
   ```
   phys_slot[b, k] = block_table[b, top_pos[b, k] // page_size] * page_size
                   + (top_pos[b, k] % page_size)
   ```
   也可在 caller 端用 aten 三步散包搞定（gather + 除 + 加），先 aten 兜底跑通流程再
   评估是否单融一个 fused op
3. **`dsa_compute_new_slot`** （新 op，可选）—— 从 `(block_table, seq_lens)` 算
   `new_slot_mapping[b]`。可上游 paged manager 做；如果上游不做再加 op
4. **`dsa_advance_seqlens`** （新 op，可选）—— `seq_lens += 1`；aten in-place add 通常
   能 cover，profile 后定
5. **`dev_dsa_attn.forward_zeus` v2 重写**：用 paged 接口替代当前 host-coordinated
   path。需要先有 paged-attention manager mock / fake state（或者直接接 sglang 真实
   paged manager 出来对拍）
6. **数值对拍**：跑 paged + non-paged 两条路径在小 page_size=1 / 退化形式 下应
   bit-exact（已为 #3 验证过；其它 step 跟着 #3 模式建对拍）
