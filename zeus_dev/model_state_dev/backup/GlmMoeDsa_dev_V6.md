# GLM5-Next DSA Decode Path under Context-Parallel — "Route B"（V6）

> 本文是 `GlmMoeDsa_dev_V5.md` 单卡 decode 链路的 **多卡 Context-Parallel（CP）版**，按 **"Route B"**
> 实现：KV 按 token 位置分片到 `cp_size` 张卡，每张卡对自己持有的那部分 top-2048 槽位做
> **partial sparse MQA**，再用一个**带 LSE 的 online-softmax merge**（一个新增的 reduce 算子）
> 合成最终结果。
>
> 只覆盖 **decode path**（每 request 1 个 query token）的 DSA full-attention 子层（层
> `[3,7,11,15,19,23]` / 大 config `[…,43]`）；**不覆盖** prefill、KDA 线性注意、MoE-FFN、
> TP/PP/EP、NextN/MTP。
>
> 校对依据：`/root/project/sglang-feat-v0.5.10-prerelease-glm`（以下 **REPO**）的 prefill-CP
> 原语（zigzag / round-robin token split、`cp_all_gather_rerange_output`、`nsa_cp_metadata`、
> `transform_index.py`）。**注意：REPO 现有的 CP 主要是 _prefill_ CP（`is_nsa_enable_prefill_cp`）；
> 本文是把它延伸到 decode 的工程设计**，不是对现成代码的逐行描述——凡"现成"的地方会标 `R-*`，
> 凡"本文新增/设计"的会标 ⚙。
>
> 状态约定：`×` = Zeus 未落地；`△` = 部分；`✓` = 已落地默认 PASS；⚙ = 本文新增设计点。

---

## 0. 一句话

decode 时把每个 DSA 层的 **latent KV cache 按 token 位置 round-robin 分片**到 `cp_size` 张卡（position `p` → rank `p mod cp_size`）。新 token 的 latent KV `k_new [1,576]` **只 store 在它的 owner 卡**；其余卡虽然在 replicated 的 forward 里也算出了 `k_new`，但**算完即弃**——因为 `k_new` 只是一条新增 key，只进 owner 的 `K_local`。而 **`q_new [Nh,576]` 是每张卡都要的**（每卡都得拿 `q` 去和自己那份 `K_local` 算 partial attention）。每张卡用 `q_new` 对 `top2048(q) ∩ K_local` 做 **partial sparse MQA**（不归本卡的槽位填 `-1`，在该 partial 的 softmax 里当 -inf），输出 `(partial_out [Nh,Rkv], partial_lse [Nh])`；最后一个 **online-softmax all-reduce merge** 把 `cp_size` 份 partial 合成最终 `attn_out_latent [Nh,Rkv]`，再 `bmm w_vc` → `o_proj`。

为什么 decode 选 Route B（而 prefill 选 Route A）：见 §2。简言之——MLA-absorb 里 **key 是单 head（便宜），query/output 是 Nh 头（贵 ~32×）**；prefill 的 top-k 并集≈整段，搬 key 最划算（Route A）；decode 只 1 个 query，搬 query / reduce output 都只是 1 个 token，反而最便宜（Route B）。

---

## 1. 范围与约定

- **入口**：`hidden_new [B, H]`（B = decode batch；每 request 1 个新 token，位于序列位置 `pos(req)`）。
- **出口**：`o_proj` 输出 `out [B, H]`，进 layer communicator 的 post-attn / MHC。
- **CP 度数**：`cp_size`（举例 16）；本文不叠 TP（叠 TP 是正交的另一层 all-reduce，见 §13）。
- **分片规则（round-robin）**：position `p` 的 latent KV 落 rank `p mod cp_size`，rank-local 槽位 ≈ `p // cp_size`（实际经 paged page-table 映射，`page_size=64`）。decode 增量产生的新 token 自然轮流落到各 rank。
- **"owner(req)"**：拥有 `pos(req)` 的那张卡 = `pos(req) mod cp_size`。不同 request 的 owner 可能不同。
- **复制 vs 分片**（详见 §3）：
  - **latent KV cache** → 分片（这是省显存的本体）。
  - **index K cache** → 两个变体：(a) 复制（top-k 全本地、零通信，但 index K 不省显存）；(b) 分片 + **分布式 top-k**（index K 也省，付一次 ~256 KB/层/step 的小 all-gather）。本文主线走 (b)（"minimal HBM"），(a) 作为简化变体。
  - **权重**（`fused_qkv_a_proj_with_mqa` / `q_b_proj` / `kv_b_proj`→`w_kc`/`w_vc` / `o_proj` / `indexer.*`）→ 复制（纯 CP 不切权重）。
  - **`hidden_new` / `positions`** → 复制（每卡跑同一份 per-token forward）；可选"只 owner 算 + broadcast `q_new`"，但省的≈0、还多一处同步，本文不采。
- **dispatch**：DSA decode 永远 `AttnForwardMethod.MLA`（absorb），`use_mha` 始终 `False`（`MHA_ONE_SHOT` 是 prefill 才有的事）。
- **main MLA 不做 RoPE**：`mla_nope=True ⇒ rotary_emb=None`，`q_pe / k_pe` 是未旋转原始 slice，原样进 attention 与 cache（同 V5 §4）。只 indexer 前 64 维做 NeoX RoPE。

---

## 2. 为什么 decode 选 Route B（成本对照）

| 方案 | 做法 | 每层每 decode step 通信（cp=16, 1M 上下文） | 备注 |
|---|---|---|---|
| Route A · 全量 gather | all-gather 整段 latent KV → 本地做满 sparse MQA | ≈ `seqlen × 1.15 KB` ≈ **~1.15 GB** | 荒谬：只需要 2048 个 token 却搬了全部 |
| Route A · 按实际 top-k indices gather | owner 卡向各卡要"被选中的那 2048 个 latent" → 本地做满 | ≈ `2048 × 1.15 KB` ≈ **~2.4 MB**（不规则 all-to-all，负载集中在 owner 卡） | 比 B 贵 ~3–9×、不规则、不均衡 |
| **Route B（本文）** | 各卡对 `top2048(q) ∩ K_local` 做 partial → online-softmax merge | top-k merge ~256 KB + partial merge ~66–530 KB ≈ **~0.3–0.8 MB**，规则集合通信、cp 卡负载均衡 | 推荐 |
| Route B"搬 query 版" | all-gather `q_new`、各卡对全量 K_local 做 partial、再 reduce | `q` 是 Nh=32 头，~37 KB/token；reduce output 也是 Nh 头 —— 对 prefill 贵 ~90×；**对 decode（1 token）这两项都退化成 1 个 token，所以照样便宜** | 即上一行 |

> 关键非对称：**搬 key 比搬 query/output 便宜 ~Nh 倍**（key 单 head 576 elem ≈ 1.15 KB；query `[Nh,576]` ≈ 37 KB；output `[Nh,Rkv]+lse` ≈ 33 KB）。decode 只有 1 个 query token，所以"搬 query 一次 + reduce output 一次"= 几十 KB，秒杀 Route A 的 ~2.4 MB（更别说 1.15 GB）。
>
> ×11 个 DSA 层后，Route B 全 decode step 的额外通信 ≈ 几 MB，~10 µs 量级，可忽略。

---

## 3. 分片 / 复制矩阵

| 对象 | per-token 大小 | 决策 | 理由 / 代价 |
|---|---|---|---|
| 主 latent KV cache（`concat(kv_a_layernorm 输出 [Rkv=512], k_pe [Dro=64])`） | 576 elem ≈ 1.15 KB BF16（fp8_e4m3 ≈ 576 B） | **分片**，每 rank 持 `≈ seqlen/cp_size` 行 | 这是省显存的本体；1M·11层 BF16 ≈ 12.7 GB → /16 ≈ 0.79 GB/卡 |
| `index_k_with_scale_buffer`（128 B FP8 + 4 B fp32 scale = 132 B） | 132 B | 变体 (a) **复制** / 变体 (b) **分片** | (a)：top-k 全本地、零通信，但每卡都存 ≈ 1.45 GB（1M·11层）；(b)：每卡 ≈ 0.09 GB，付一次分布式-top-k 的 ~256 KB/层/step。**"每卡 KV 空间很少" → 选 (b)** |
| `w_kc [Nh,Dnope,Rkv]` / `w_vc [Nh,Rkv,Dv]` / `q_b_proj` / `kv_b_proj` / `o_proj` / `fused_qkv_a_proj_with_mqa` / `indexer.wq_b/wk/weights_proj/k_norm` | — | **复制** | CP 不切权重；要切权重请叠 TP（§13） |
| `hidden_new [B,H]` / `positions [B]` / `q_lora`/`q_nope_out`/`q_pe`/`q_new`/`q_idx`/`weights(gate)` | — | **复制**（每卡同一份 per-token forward） | 1 个 token 的 GEMM，重复 cp 次 ≈ 微秒，换简洁；唯一 owner-only 的是"append KV 行" |

---

## 4. "k 只去 owner，q 去所有卡" —— 非对称的来源

decode 每层一次 projection 出 4 样东西，去向不同：

| 产物（新 token，per layer） | 谁需要 | 落地 |
|---|---|---|
| `k_new = concat(kv_a_layernorm(latent[:Rkv]), latent[Rkv:]) [1,576]` | **只 owner(req)** —— 它只是一条新增 key，只进 owner 的 `K_local` | owner 把它写进自己的 latent KV pool 槽位 `pos//cp_size`；**非 owner 算完即弃** |
| `k_idx_new`（indexer K，FP8 `[128]`+scale） | 变体 (a)：所有卡（复制的 index pool）；变体 (b)：只 owner | (a) 各卡都 append；(b) 只 owner append |
| `q_new = concat(q_nope_out, q_pe) [Nh=32, 576]`（`q_nope_out = bmm(q_nope, w_kc)`） | **所有卡** —— 每卡都得拿 q 去和自己那份 `K_local` 算 partial attention | 不进 cache；每卡都要 |
| `q_idx [8,128]`（FP8）+ `weights/gate [8,1]` | 变体 (a)：所有卡（或一卡算完 broadcast 2048 indices）；变体 (b)：所有卡（各自对 `K_local` 算 local logits） | 不进 cache |

所以你观察到的"非 owner 把 `k_new` 算出来又扔了"完全正确——但**那点浪费 = 1 个 token 的 `q_b_proj`/`kv_a` GEMM × (cp_size−1) 张卡 × per layer ≈ 微秒级**，落在"复制整个 decode forward"的自然结果里。想"零浪费"可以改成"只 owner 跑 projection + broadcast `q_new`(~37 KB)/`q_idx`(~4 KB)"，但省下的≈0、还多一处通信和同步——**不值，本文不采**。

---

## 5. 维度速查（沿用 `config_16b.json`；大 `config.json` cache 结构相同）

| 符号 | 含义 | `config_16b.json` | `config.json`（大 GLM-5） |
|---|---|---:|---:|
| H | hidden | 2048 | 4096 |
| Nh | MLA Q 头数 | 32 | 64 |
| Rq | q_lora_rank | 768 | 1536 |
| **Rkv** | kv_lora_rank（= latent K=V dim，**进 cache**） | **512** | **512** |
| **Dro** | qk_rope_head_dim（**进 cache**） | **64** | **64** |
| Dnope | qk_nope_head_dim | 128 | 192 |
| Dqk | Dnope+Dro | 192 | 256 |
| Dv | v_head_dim | 128 | 256 |
| **Di** | index_head_dim（**进 index cache**） | **128** | **128** |
| I | index_n_heads | 8 | 8 |
| **Ktop** | index_topk（每 query 选的 sparse 数） | **2048** | **2048** |
| DSA 层数 | `full_attn_layers` 长度 | **6**（`[3,7,11,15,19,23]`） | **11**（`[3,7,…,43]`） |

> 注：`Nh / Dnope / Dv / Rq` 这些在 absorb 路径下**不进 cache**（只影响 `w_kc/w_vc`、dispatch、dense fallback——dense fallback decode 不走）。所以两个 config 的**每 token cache 形态完全一样**（latent KV 576、index K 132B），区别只在 DSA 层数（6 vs 11）和 query/output 的头数（影响 §2 的"搬 q/output 有多贵"，但 decode 下都只是 1 token）。
>
> attention scaling = `Dqk**-0.5`（16B: `192**-0.5`；大: `256**-0.5`），GLM5 不开 YaRN 无额外 mscale。

---

## 6. Decode-CP-RouteB 计算流（D0–D6 + ⚙ D3.5 / D4.5）

`B` = decode batch。除"⚙ owner-only" 标注的副作用外，**D0–D2、D5、D6 在所有 cp_size 卡上跑相同的 per-token forward**（输入 `hidden_new` 复制）。

```text
hidden_new [B,H]   (来自 layer_communicator.prepare_attn；MHC 残差已处理；每卡相同)
  │
  ├─ D0  dsa_pre_attn_qkv_latent          R-MLA      (所有卡相同)
  │    qkv_latent = fused_qkv_a_proj_with_mqa(hidden_new)            [B, Rq+Rkv+Dro]
  │    → q_lora_raw [B,Rq], latent_cache [B, Rkv+Dro=576]
  │
  ├─ D1  dsa_qkv_a_norm                    R-ABSORB   (所有卡算；append 只 owner)
  │    q_lora    = q_a_layernorm(q_lora_raw)                          [B, Rq]
  │    k_new_nope = kv_a_layernorm(latent_cache[:, :Rkv])             [B, Rkv=512]
  │    k_new_pe   = latent_cache[:, Rkv:]                             [B, Dro=64]   (未 RoPE)
  │    k_new      = concat(k_new_nope, k_new_pe)                      [B, 1, 576]
  │    ⚙ side effect (per req): if rank == owner(req):
  │         latent_KV_pool_rank[ slot(pos(req)) ] = k_new[req]        [1, 576]
  │      else: k_new[req] 算完即弃
  │
  ├─ D2  dsa_q_b_absorb                    R-ABSORB   (所有卡相同)
  │    q = q_b_proj(q_lora).view(B, Nh, Dqk) → q_nope [B,Nh,Dnope], q_pe [B,Nh,Dro]  (q_pe 未 RoPE)
  │    q_nope_out = bmm(q_nope.T[Nh,B,Dnope], w_kc[Nh,Dnope,Rkv]).T   [B, Nh, Rkv=512]
  │    q_new      = concat(q_nope_out, q_pe)                          [B, Nh, 576]
  │
  ├─ D3  dsa_indexer_prep_store            R-INDEXER  (所有卡算；append 见变体)
  │    q_idx [B,I,Di] = rotate_activation( rope_neox_first64( wq_b(q_lora) ) )       (Hadamard, NeoX 前64)
  │    k_idx [B,Di]   = rotate_activation( rope_neox_first64( k_norm( wk(hidden_new) ) ) )   (k_norm = 完整 LayerNorm fp32)
  │    q_idx_fp8,q_scale = act_quant(q_idx,128,"ue8m0");  k_idx_fp8,k_idx_scale = act_quant(k_idx,...)
  │    gate[B,I] = weights_proj(hidden_new)*I^-0.5;  weights = gate.unsqueeze(-1)*q_scale*Di^-0.5   → [B,I,1]
  │    ⚙ side effect (per req):
  │       变体(a) index K 复制: 所有卡 index_K_pool[ slot(pos) ] = (k_idx_fp8, k_idx_scale)[req]
  │       变体(b) index K 分片: if rank==owner(req): owner 的 index_K_pool[ slot(pos) ] = ...   (非 owner 弃)
  │
  ├─⚙D3.5 dsa_decode_topk_cp               ⚙ (本文新增；变体 b 主线)
  │    [每卡 r] local_logits_r[B, n_local_r] = deep_gemm.fp8_paged_mqa_logits(
  │                 q_idx_fp8[B,1,I,Di], index_K_pool_r.view(...), weights[B,I],
  │                 seqlens_local_r, page_table_local_r, schedule_md, max_local_len )
  │              local_top_r[B, min(Ktop, n_local_r)] = top-k(local_logits_r) → (positions, logits) 对
  │    all-gather local_top_r 跨 cp_size 卡  →  cand[B, ≤ cp_size*Ktop] (位置+logit)
  │    [每卡 r] global top-Ktop(cand) → topk_positions[B, Ktop]   (全局位置；尾部 -1 填充)
  │       —— 正确性：任何属于全局 top-Ktop 的 token，必在其 owner 的 local top-Ktop 内
  │    [每卡 r] page-table 变换: topk_slots_r[B, Ktop] = map(topk_positions, page_table_local_r)
  │       —— 不归 r 的位置 → -1
  │    （变体 (a)：跳过 all-gather，每卡用复制的完整 index K 直接算 topk_positions，再各自做 page-table 变换）
  │
  ├─ D4  dsa_decode_sparse_mqa_partial     ⚙ (基于 R-BACKEND，加 return_lse + 局部 indices)
  │    [每卡 r]  partial_out_r [B,Nh,Rkv], partial_lse_r [B,Nh]
  │        = attn_mqa_partial( q_new [B,Nh,576], latent_KV_pool_r, topk_slots_r [B,Ktop],  return_lse=True )
  │        —— topk_slots_r 里 -1 的槽位在本 partial 的 softmax 里当 -inf（不读 cache、不计入分母）
  │        —— 若某 req 在 rank r 上一个槽位都没有 → partial_lse_r[req] = -inf, partial_out_r[req] = 任意（被 merge 权重 0 掉）
  │        —— num_kv_heads=1, head_dim=576, v_head_dim=Rkv=512, scaling=Dqk^-0.5
  │
  ├─⚙D4.5 dsa_cp_merge                     ⚙ (本文新增的 reduce 算子；带 LSE 的 online-softmax merge)
  │    跨 cp_size 卡合并 {(partial_out_r, partial_lse_r)}：
  │       m   = max_r partial_lse_r                          # [B,Nh]，逐 (req,head)
  │       Z   = Σ_r exp(partial_lse_r - m)                   # [B,Nh]
  │       attn_out_latent = ( Σ_r exp(partial_lse_r - m) * partial_out_r ) / Z      # [B,Nh,Rkv=512]
  │    —— 实现：all-gather (partial_out, partial_lse) 后本地 merge（~530 KB/层），或 ring-allreduce 用 merge 作 reduce 算子（~66 KB/层）
  │    —— all-reduce 形态：merge 后所有卡拿到相同 attn_out_latent，保持 lockstep
  │    ★ 不能用裸 all_reduce(sum)：各 partial 的 softmax 分母不同，必须用 LSE 重新对齐
  │
  ├─ D5  dsa_v_absorb                      R-ABSORB   (所有卡相同)
  │    attn_out [B,Nh,Dv] = bmm(attn_out_latent.T[Nh,B,Rkv], w_vc[Nh,Rkv,Dv]).T
  │
  └─ D6  dsa_o_proj                        R-MLA      (所有卡相同)
       out [B,H] = o_proj( attn_out.reshape(B, Nh*Dv) )
       （所有卡输出相同 → 进下一层继续 lockstep）
```

---

## 7. 各 stage 的 IO / 副作用 / 通信

| # | stage | 输入（每卡） | 输出（每卡） | 副作用 | 跨卡通信 | 来源 / 状态 |
|---|---|---|---|---|---|---|
| D0 | `dsa_pre_attn_qkv_latent` | hidden_new [B,H]（复制） | q_lora_raw [B,Rq], latent_cache [B,576] | — | — | `R-MLA::prepare_qkv_latent`；× |
| D1 | `dsa_qkv_a_norm` | q_lora_raw, latent_cache | q_lora [B,Rq], k_new [B,1,576] | **⚙ owner-only**: latent_KV_pool_r[slot(pos)] = k_new | — | `R-ABSORB::forward_absorb_prepare`；× |
| D2 | `dsa_q_b_absorb` | q_lora | q_nope_out [B,Nh,512], q_pe [B,Nh,64], q_new [B,Nh,576] | — | — | `R-ABSORB`（含 `bmm w_kc`）；× |
| D3 | `dsa_indexer_prep_store` | hidden_new, q_lora | q_idx_fp8 [B,I,Di], q_scale, weights [B,I,1], k_idx_fp8/scale | 变体(a) 所有卡 / 变体(b) **⚙ owner-only**: index_K_pool_r[slot(pos)] = (k_idx_fp8,scale) | — | `R-INDEXER`（`_get_q_k_bf16`+`rotate_activation`+`act_quant`+`_store_index_k_cache`）；× |
| ⚙D3.5 | `dsa_decode_topk_cp` | q_idx_fp8, weights, index_K_pool_r, page_table_local_r, seqlens_local_r | topk_slots_r [B,Ktop]（非 r 位置 = -1） | — | **all-gather local-top-Ktop（pos+logit）pairs** ≈ `cp_size·B·Ktop·8 B`（变体 b；变体 a 无通信） | ⚙ 基于 `R-INDEXER::_get_topk_paged` + `R-TOPK::topk_transform`；× |
| D4 | `dsa_decode_sparse_mqa_partial` | q_new [B,Nh,576], latent_KV_pool_r, topk_slots_r | partial_out_r [B,Nh,Rkv], partial_lse_r [B,Nh] | — | — | ⚙ 基于 `R-BACKEND::forward_decode`（`flash_mla_with_kvcache` / `flash_mla_sparse_fwd`，需 `return_lse`，num_kv_heads=1, d_v=Rkv）；× |
| ⚙D4.5 | `dsa_cp_merge` | {(partial_out_r, partial_lse_r)} 跨 cp_size 卡 | attn_out_latent [B,Nh,Rkv] | — | **online-softmax merge（all-gather 或 ring-allreduce）** ≈ `cp_size·B·Nh·(Rkv+1)·2 B`（all-gather）或 `2·B·Nh·(Rkv+1)·2 B`（ring） | ⚙ 新增；× |
| D5 | `dsa_v_absorb` | attn_out_latent, w_vc | attn_out [B,Nh,Dv] | — | — | `R-ABSORB` 末段 `bmm w_vc`；× |
| D6 | `dsa_o_proj` | attn_out [B,Nh*Dv] | out [B,H] | — | （纯 CP 不需要；叠 TP 才有 o_proj all-reduce） | `RowParallelLinear`；× |

---

## 8. D3.5 详解：分布式 top-k（变体 b）

top-2048 选择**只依赖 index K + 该 query 的 indexer Q/gate**——这是个对每个 key 独立打分、再取 top 的操作，没有跨 key 的归一化（不像 attention 有 softmax）。所以可以分布式做：

1. **每卡本地打分 + 本地 top-k**：rank r 用 `deep_gemm.fp8_paged_mqa_logits(q_idx_fp8, index_K_pool_r, weights, …)` 对自己持有的 `n_local_r` 个 key 算 logits，取本地 top-`min(Ktop, n_local_r)`，连同它们的**全局位置**和 **logit 值**打包。
2. **all-gather 候选**：`cp_size` 张卡各自的 ≤Ktop 个 `(position, logit)` 收齐 → `≤ cp_size·Ktop` 个候选。通信量 ≈ `cp_size · B · Ktop · 8 B`（位置 int32 + logit fp32）；cp=16,B=1,Ktop=2048 → ≈ 256 KB/层/step。
3. **全局 top-k**：从候选里取 top-`Ktop` → `topk_positions [B, Ktop]`（按规范尾部填 `-1`）。
   - **正确性**：若 token `t`（owner = rank o）属于全局 top-Ktop，则全局至多 Ktop−1 个 token 的 logit 比它大 ⇒ 在 rank o 的 token 子集里至多 Ktop−1 个比它大 ⇒ `t` 在 rank o 的本地 top-`min(Ktop, n_local_o)` 内 ⇒ 一定被 all-gather 上来。✔
4. **page-table 变换**：每卡 r 把 `topk_positions` 经自己的 per-rank page table 映射成 rank-local cache 槽位 `topk_slots_r [B, Ktop]`；不属于 r 的位置 → `-1`。

> 变体 (a)（index K 复制）：跳过第 1–3 步——每卡用本地完整 index K 直接算 `topk_positions`（结果各卡一致），只做第 4 步的 page-table 变换。零通信，但 index K 不省显存。
>
> 注意：`-1` 在 `topk_positions` 尾部的语义是 **"凑不满 Ktop 的填充"**（`visible_len < Ktop` 时）；`-1` 在 `topk_slots_r` 里的（额外）语义是 **"这个被选中的位置不归本卡"** —— 两者在 D4 的 partial 里都当 -inf 处理，所以可以共用同一个值。

---

## 9. D4 详解：partial sparse MQA 与 `-1` 语义

rank r 对**它物理持有的那部分 top-2048 槽位**做一次标准的（带 softmax 的）latent sparse MQA，但要求 kernel **额外返回 LSE**：

```text
for each req, each head h ∈ [0, Nh):
    S_local = { s = top2048(req)[i] : topk_slots_r[req,i] != -1 }                  # rank r 持有的子集
    s_j = scaling * ( q_new[req,h,:576] · latent_KV_pool_r[ slot_j , :576 ] )       # slot_j ∈ S_local, scaling = Dqk^-0.5
    partial_lse_r[req,h]  = logsumexp_j( s_j )                                       # 仅本卡子集；空集 → -inf
    partial_out_r[req,h]  = Σ_j exp( s_j - partial_lse_r[req,h] ) * latent_KV_pool_r[ slot_j , :Rkv ]   # latent 前 Rkv=512 维当 V
```

- `topk_slots_r` 里 `-1` 的槽位：kernel 不读 cache、当 -inf、不计入 `S_local`、不进分母。
- 某 req 在 rank r 上 `S_local = ∅`（这条 query 的 top-2048 一个都不在 r 上）→ `partial_lse_r[req] = -inf`，`partial_out_r[req]` 任意值（D4.5 里权重 `exp(-inf - m) = 0` 把它消掉）。
- "q 看自己"那一项（position `pos(req)` ∈ top-2048，几乎必然，因为是 query 自身位置）落在 **owner(req)** 的 `S_local` 里——所以 owner 的 partial 含自注意力项，其它卡不含（也不该含）。
- kernel 候选：`R-A1::flash_mla_with_kvcache(..., return_lse=True)` / `R-A2::flash_mla_sparse_fwd(..., return_lse=True)` / tilelang 的 `sparse_mla_fwd_interface`——绝大多数 flash 系都能返回 LSE；这是本设计对 sparse MQA kernel 的唯一额外要求。

---

## 10. D4.5 详解：online-softmax merge（新增的 reduce 算子）⚙

把 `cp_size` 份 `(partial_out_r [B,Nh,Rkv], partial_lse_r [B,Nh])` 合成最终 `attn_out_latent`：

```text
m   = max_r  partial_lse_r                       # [B,Nh]，逐 (req, head)
Z   = Σ_r    exp(partial_lse_r - m)              # [B,Nh]
attn_out_latent = ( Σ_r  exp(partial_lse_r - m) * partial_out_r ) / Z      # [B,Nh,Rkv]
# 等价地：attn_out_latent = Σ_r w_r * partial_out_r，  w_r = exp(partial_lse_r - m) / Z，  Σ_r w_r = 1
```

为什么对：flash attention 的约定下，每份 partial 已经按"本卡子集的分母"归一化了（`o_r = Σ exp(s_j - lse_r) v_j`）。两个不相交子集 A、B 的合并是
`lse_AB = lse_max + log(exp(lse_A-lse_max)+exp(lse_B-lse_max))`，`o_AB = exp(lse_A-lse_AB) o_A + exp(lse_B-lse_AB) o_B`；这个 merge 算子**结合律 & 交换律**成立，所以可以 tree/ring reduce。`cp_size` 份就是上面那个加权平均。

实现两种：

- **all-gather + 本地 merge**（最简单）：收齐 `cp_size` 份 `(partial_out, partial_lse)` 后本地算上式。通信 ≈ `cp_size · B · Nh · (Rkv+1) · 2 B`；cp=16,B=1,Nh=32,Rkv=512 → ≈ 526 KB/层/step。
- **ring-allreduce，merge 作 reduce 算子**：通信 ≈ `2 · B · Nh · (Rkv+1) · 2 B` ≈ 66 KB/层/step；更省，但要自己实现 reduce 算子（NCCL 没有现成的"flash-merge"，用若干 all-gather 或自定义 kernel + send/recv 拼）。
- 两种都做成 **all-reduce 形态**（结果落回所有卡），让所有卡 merge 后 lockstep，继续跑相同的 D5/D6/下一层 D0–D2。

> ⚠ 绝不能用裸 `all_reduce(add)` 把 `partial_out_r` 直接加起来——各 partial 的 softmax 分母不同，必须 LSE 重对齐。

---

## 11. 通信量估算（cp_size = 16, 1M 上下文, B = 1）

| 项 | 每层每 decode step | ×11 个 DSA 层 |
|---|---|---|
| D3.5 分布式 top-k all-gather（变体 b） | `16 · 2048 · 8 B` ≈ **256 KB** | ≈ 2.8 MB |
| D4.5 merge（all-gather 版） | `16 · 32 · 513 · 2 B` ≈ **526 KB** | ≈ 5.8 MB |
| D4.5 merge（ring-allreduce 版） | `2 · 32 · 513 · 2 B` ≈ **66 KB** | ≈ 0.7 MB |
| **Route B 合计（ring 版）** | ≈ **~0.32 MB** | ≈ **~3.5 MB / decode step** |
| 对照 Route A（按 top-k indices gather，不规则） | `2048 · 1.15 KB` ≈ ~2.4 MB（集中在 owner 卡） | ≈ 26 MB / decode step |
| 对照 Route A（all-gather 全量 latent KV） | `seqlen · 1.15 KB` ≈ ~1.15 GB | ≈ 12.7 GB / decode step（不可行） |

显存（per 卡，1M 上下文）：

| | latent KV（分片） | index K（变体 a 复制 / 变体 b 分片） | 合计 |
|---|---|---|---|
| 16B（6 DSA 层） | 6·1M·1.15 KB / 16 ≈ **0.43 GB** | a: 6·1M·132 B ≈ 0.79 GB / b: ÷16 ≈ 0.05 GB | a ≈ 1.22 GB / **b ≈ 0.48 GB** |
| 大 GLM-5（11 DSA 层） | 11·1M·1.15 KB / 16 ≈ **0.79 GB** | a: 11·1M·132 B ≈ 1.45 GB / b: ÷16 ≈ 0.09 GB | a ≈ 2.24 GB / **b ≈ 0.88 GB** |

（latent KV 若按 `fp8_e4m3` 存再 ÷2。"每卡 KV 空间很少" → 选变体 b + fp8 latent KV。）

---

## 12. dev 脚本 stage 对应 + 里程碑

| dev stage | 覆盖 | 对应 | Zeus |
|---|---|---|---|
| `decode_cp_qkv_a_proj_norm_fused` | D0+D1（含 owner-only 的 latent KV append） | D0/D1 | × |
| `decode_cp_q_proj_fused` | D2：`q_b_proj`+split+`bmm w_kc`，输出 `q_nope_out [B,Nh,Rkv]`+`q_pe`（未 RoPE）+`q_new` | D2 | × |
| `decode_cp_indexer_prep_store_fused` | D3：indexer Q/K + NeoX RoPE(前64) + k_norm(LayerNorm) + Hadamard + FP8 + index K append（owner-only / 复制）+ gate(fp32) | D3 | × |
| `decode_cp_topk_distributed` | D3.5：本地 logits + 本地 top-k → all-gather → 全局 top-k → per-rank page-table 变换（含 -1） | ⚙D3.5 | × |
| `decode_cp_sparse_mqa_partial` | D4：partial sparse MQA over `top2048 ∩ K_local`（-1 当 -inf），返回 `(partial_out [B,Nh,Rkv], partial_lse [B,Nh])` | D4 | × |
| `decode_cp_merge` | D4.5：online-softmax merge（带 LSE），输出 `attn_out_latent [B,Nh,Rkv]` | ⚙D4.5 | × |
| `decode_cp_v_absorb` | D5：`bmm(attn_out_latent, w_vc)` → `[B,Nh,Dv]` | D5 | × |
| `decode_cp_o_proj` | D6：`[B,Nh*Dv]` → `[B,H]` | D6 | × |
| `decode_cp_full_path` | D0–D6 + D3.5 + D4.5 端到端（cp_size 卡） | 全链路 | × |

| Milestone | 目标 | 通过判据 |
|---|---|---|
| M0 | pure-torch REF（单进程模拟 cp_size 张卡：用 list 表示各 rank 的 KV shard；main MLA 不插 RoPE；indexer NeoX 前 64；sparse 输出 `[*,Nh,Rkv]` 再 `w_vc`） | `decode_cp_full_path --mode ref --cp 4` 端到端与 V5 单卡 REF **数值完全一致**（merge 是 exact） |
| M1 | D0–D2 复制 + owner-only latent KV append | 各 rank 的 KV shard 内容正确（位置 ↔ rank 映射、page-table 槽位） |
| M2 | D3 indexer 通路 + index K append（两个变体） | indexer Q/K/gate/FP8 与 V5 一致；index K shard/复制内容正确 |
| M3 | D3.5 分布式 top-k | `topk_positions` 与"全量 index K 上算的 top-k"逐元素一致（含 -1 填充）；`topk_slots_r` 的 -1 位置正确 |
| M4 | D4 partial sparse MQA（`return_lse`） | 单卡退化（cp=1）时 = V5 D4；多卡时各 partial 的 `(out, lse)` 与"只在该子集上算 attention"一致 |
| M5 | D4.5 merge | merge(cp 份 partial) 与"在完整 top-2048 上算 attention"逐元素一致（exact，到浮点精度） |
| M6 | D5/D6 + 端到端 | DSA 层 `[B,H] → [B,H]` 在 cp_size 卡上与 V5 单卡一致；通信量符合 §11 估算 |
| M7 | 真分布式（NCCL）+ ring-allreduce 版 merge | 多机/多卡跑通；merge 用 ring reduce 算子 |
| M8 | fp8 latent KV + 变体 b（index K 分片） | quant/dequant 落地；显存符合 §11；精度回归 |

---

## 13. 开放问题 / 注意点

- **REPO 现状是 prefill CP**（`is_nsa_enable_prefill_cp` / `nsa_cp_metadata` / `cp_all_gather_rerange_output`）。本文把它延伸到 decode，复用其 split 规则与 page-table 原语，但 D3.5（分布式 top-k）和 D4.5（LSE merge）是**新算子**——需要确认 REPO 的 `nsa_backend` decode 路径在 CP 下到底有没有现成实现；若有，按其实现校正本文。
- **decode 是否沿用 prefill 的分片**：本文假设 round-robin 自然延伸（新 token position `pos` → rank `pos mod cp_size`）。若 prefill 用的是 zigzag block split，decode 阶段要么转成 round-robin、要么新 token 一律落某个固定"tail" rank（会随时间失衡，不推荐）。
- **B > 1**：不同 request 的 owner 不同、top-2048 不同；上述链路逐 request 成立，集合通信在 B 维上 batch；D4.5 的 merge 逐 `(req, head)` 独立。
- **`seqlen ≤ Ktop`**：top-k 退化为"全选可见"，`topk_positions = [0,1,…,seqlen-1, -1, …]`；各卡的 `K_local` 就是它持有的全部可见 token，链路不变。
- **KDA 线性注意层**（21 / 34 层）不在本切片：它们不支持 CP，decode 时单 token 过 KDA，recurrent state 的归属是另一套问题（prefill CP 里 KDA 前后 all-gather/split）；这里只管 DSA。
- **MHC**：CP 下 layer communicator 是 `MHCHybridNSACPLayerCommunicator`；本文按非-MHC 普通语义，MHC 残差作外层包装。
- **叠 TP**：TP 切 `q_b_proj`/`kv_b_proj`/`o_proj`/`indexer.*` 等权重（按 Nh 切），与 CP 正交；叠 TP 后 `o_proj` 多一个沿 hidden 维的 all-reduce，`Nh_local = Nh / tp_size`，partial sparse MQA 也只算 `Nh_local` 个头——D4.5 的 merge 仍是沿 CP 维、对 `Nh_local` 个头做。TP all-reduce 与 CP merge 是两个不同的 reduce。
- **正确性是 exact 的**：online-softmax merge 在精度内**精确**等于"在完整 top-2048 上做一次 attention"——所以 REF 阶段 `--cp k` 的结果应当与 `--cp 1`（= V5 单卡）逐元素一致，这是 M0/M5 的硬判据。

---

## 14. 参考源码索引（REPO；⚙ = 本文新增设计）

| 标签 | 入口 | 用途 |
|---|---|---|
| `R-MLA` | `models/deepseek_v2.py::DeepseekV2AttentionMLA`（`prepare_qkv_latent`、`forward_absorb_prepare/core`、`attn_mqa` l.1269） | D0/D2/D5/D6 的算子链；`attn_mqa = RadixAttention(num_kv_heads=1, head_dim=576, v_head_dim=Rkv=512, scaling=Dqk^-0.5)` |
| `R-ABSORB` | `models/deepseek_common/attention_forward_methods/forward_mla.py::forward_absorb_prepare/core` | `q_a/kv_a layernorm` → `q_b_proj` → split → `bmm w_kc` →（sparse MQA）→ `bmm w_vc` → `o_proj`；main MLA 不插 RoPE（`rotary_emb is None`） |
| `R-INDEXER` | `layers/attention/nsa/nsa_indexer.py::Indexer`（`_get_q_k_bf16`、`rotate_activation` l.135、`act_quant`、`_store_index_k_cache`、`_get_topk_paged`、`forward_npu`） | D3 的 indexer Q/K/gate + NeoX RoPE 前 64 + `k_norm` LayerNorm + Hadamard + FP8 + index K cache；D3.5 的 `fp8_paged_mqa_logits` + `topk_transform` 的单卡版 |
| `R-TOPK` | `layers/attention/nsa/transform_index.py`（`metadata.topk_transform`、`transform_index_page_table_decode`）+ `triton_kernel.py` / `tilelang_kernel*.py` | logits → topk slots（-1 padding、page-table 映射）；D3.5 的 per-rank page-table 变换基于它 |
| `R-A1`/`R-A2` | `nsa_backend.py::_forward_flashmla_kv` / `_forward_flashmla_sparse`；`flash_mla_with_kvcache` / `flash_mla_sparse_fwd`；tilelang `sparse_mla_fwd_interface` | D4 的 sparse MQA kernel——本设计要求其支持 `return_lse` |
| `R-CACHE` | `mem_cache/memory_pool.py::NSATokenToKVPool`（继承 `MLATokenToKVPool`）；`index_buf_accessor.py` | 主 latent KV cache `[*,64,1,576]` + `index_k_with_scale_buffer [*,64*132]`——CP 下每 rank 持自己那份 shard |
| `R-CP` | `layers/attention/nsa/utils.py`（`can_cp_split`、`cp_all_gather_rerange_output`、`cp_split_and_rebuild_data/position`、`is_nsa_enable_prefill_cp`、`prepare_input_dp_with_cp_dsa`）；`forward_batch.nsa_cp_metadata` | CP token split / page-table 重排原语（REPO 现为 prefill CP，本文延伸到 decode） |
| `R-QUANT` | `layers/attention/nsa/quant_k_cache.py` / `dequant_k_cache.py` | fp8 latent KV cache 的量化/反量化（变体 b + fp8 时用） |
| `R-COMM` | `layers/communicator_mhc_hybrid_cp.py::MHCHybridNSACPLayerCommunicator`；`layers/communicator.py::fetch_qkv_latent` | CP + MHC 下的 layer communicator |
| ⚙D3.5 | `dsa_decode_topk_cp`（本文设计） | 本地 logits + 本地 top-k → all-gather (pos,logit) → 全局 top-k → per-rank page-table 变换 |
| ⚙D4.5 | `dsa_cp_merge`（本文设计） | 跨 cp_size 卡的 online-softmax merge（带 LSE）；all-gather 或 ring-allreduce 形态 |

---

> 配套：单卡基线 `GlmMoeDsa_dev_V5.md`（§4 RoPE / §6 absorb / §7 cache / §13 decode / §15 CP 接口）。本文只展开 decode + Route B；prefill 的 CP 走 Route A（all-gather latent KV 再本地做满），不在本文范围。
