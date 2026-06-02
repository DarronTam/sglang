"""
GLM5-Next DSA (Deepseek Sparse Attention) decode sublayer 独立模块 + REF↔Zeus 对拍.

设计与 ``dev_linear_attn.py`` / ``dev_moe.py`` / ``dev_mhc.py`` 同构. 把 DSA
decode (cp=1) 抽成 ``Glm5NextDsaAttn`` 类, 暴露:
  __init__ + init_state(B, seqlen) + init_paged_state(history) +
  forward (REF) + forward_zeus (paged Zeus chain).

KV cache 通过 paged-attention pool 管理: [total_slots, ...] 共享池 +
block_table (logical→physical) + seq_lens (per-seq 历史长度). 完全没有
host-side concat / per-batch CPU loop, 全链路 device-resident.

DSA decode Zeus chain (paged, 14 颗 sgl-kernel-zeus 算子):
  1.  dsa_q_a_proj_norm           Q low-rank projection + RMSNorm → q_lora
  2a. dsa_compute_new_slot        new_slot[b] = bt[b, sl/PS]*PS + sl%PS
  2b. dsa_kv_a_proj_norm_store    KV low-rank + norm + STORE latent_pool[slot]
  3.  dsa_q_main_absorb           Q upproject + absorb bmm(w_kc) → q_new
  4.  dsa_indexer_q_weights       indexer Q (rotate+fp8) + per-head weights
  5.  dsa_indexer_k_prep_store    indexer K prep (rotate+fp8 quant) + store
        ─ device: seq_lens += 1 (aten op, 1 launch)
  6.  dsa_index_logits_paged      paged indexer GEMM → logits [B, max_logical_s]
  7.  dsa_local_topk_radix        local top-K (cp=1 → IS global)
  8.  dsa_translate_topk_positions logical pos → phys slot in pool
  9.  dsa_latent_k_gather_paged   paged latent K gather (返回 c0/c1 per-core)
  10. dsa_sparse_mqa_partial      sparse MQA partial
  11. dsa_post_o_proj_no_cp       V absorb + o_proj → bf16 [B, H]

与 ``dev_glm5next_block_decode_test.zeus_dsa_decode`` 的关键改进:
  - **静态 weights 一次性 pack** (LocalMem + .to("zeus")) 进 ``_pack_zeus()``,
    forward 复用. 原版每次 forward 都 ``_lmem_pack(q_a_w)`` 等 11 个 LocalMem
    pack + 多个 ``.to("zeus")`` 搬运重做.
  - **paged-state 一次性搬上 device** (latent_kv_pool / index_body_pool / ...),
    跨 forward 持有 device 引用, 不再每步 D2H concat history + new step.

State:
  - history (REF only): ``dsa.GlobalHistory``  host-side
    - latent_kv  [B, S, Rkv]  bf16
    - index_body [B, S, Di]   bf16 (FP8 proxy)
    - index_scale [B, S]      fp32
  - paged_state (Zeus): host dict, 首次 forward_zeus 调用时全部 .to("zeus") +
    LocalMem pack, 之后跨 step 复用. seq_lens 每步 in-place +1, latent/body/
    scale pool 每步在 slot_mapping 位置写入新 K, history 自动累积.

用法:
  python glm5next_modules/dev_dsa_attn.py                 # 16b / both
  python glm5next_modules/dev_dsa_attn.py --config next
  python glm5next_modules/dev_dsa_attn.py --mode zeus --seqlen 128
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

# 公共脚手架
import _common
from _common import (
    ZEUS_IMPORT_ERROR, sgl_kernel_zeus,
    compare_tensors, zeus_chain_available,
    make_argparser, print_header, print_summary,
)

# DSA 底层 API (位于上一级 model_state_dev/)
import dev_glm5next_dsa_decode_test as dsa


# ── Module ──────────────────────────────────────────────────────
class Glm5NextDsaAttn:
    """GLM5-Next DSA decode sublayer (cp=1) — paged-attention Zeus chain.

    使用模式::

        attn = Glm5NextDsaAttn(which="16b", seed=42)
        history, block_span = attn.init_state(B=batch, seqlen=ctx_len, seed=...)
        # REF
        out = attn.forward(hidden, history, new_pos=ctx_len, block_span=block_span)
        # Zeus (paged)
        paged_state = attn.init_paged_state(history)
        out_z = attn.forward_zeus(hidden.to("zeus"), paged_state)

    multi-step decode: 同一个 ``paged_state`` 反复传给 ``forward_zeus``;
    每次 forward 内部 in-place 写 pool[slot_mapping[b]] + seq_lens += 1.
    """

    def __init__(self, which: str, seed: int = 0):
        self.cfg = dsa.select_config(which)
        torch.manual_seed(seed)
        self.weights = dsa.init_weights(self.cfg, seed)

        # Zeus device-resident state — lazy
        self._zeus_packed = False
        self._w_lmem: Dict[str, torch.Tensor] = {}
        self._w_z: Dict[str, torch.Tensor] = {}

    # ── State init ──────────────────────────────────────────────
    def init_state(self, B: int, seqlen: int, seed: int = 0, block_span: int = 16
                   ) -> Tuple[dsa.GlobalHistory, int]:
        """构造初始 KV history host tensors + block_span 常量 (REF 用)."""
        history = dsa.init_history(self.cfg, B, seqlen, seed)
        return history, block_span

    # ── REF forward ─────────────────────────────────────────────
    def forward(self, hidden: torch.Tensor, history: dsa.GlobalHistory,
                new_pos: int, block_span: int = 16) -> torch.Tensor:
        """REF DSA decode (cp=1). ``hidden: [B, H] bf16  ->  [B, H] bf16``."""
        ctx = dsa.DevContext(
            cfg=self.cfg, weights=self.weights, history=history,
            hidden=hidden, new_pos=new_pos, block_span=block_span,
        )
        return dsa.run_ref_decode(ctx, cp_size=1)["out"]

    # ── Zeus pack (lazy, 一次性) ────────────────────────────────
    def _pack_zeus(self) -> None:
        """一次性把静态 weights 装到 LocalMem (GEMM 路径) 或 Zeus device
        (norm / bias / non-LocalMem GEMM). 与 dev_glm5next_block_decode_test
        ``zeus_dsa_decode`` 的 weight 处理一致, 但避免 per-forward repack."""
        if ZEUS_IMPORT_ERROR is not None:
            raise RuntimeError(f"Zeus runtime unavailable: {ZEUS_IMPORT_ERROR}")

        cfg = self.cfg
        w = self.weights

        def _lmem(t: torch.Tensor) -> torch.Tensor:
            return torch.zeus.local_memory.from_tensor(
                t.to("zeus"), kind="weight", Tr=1, Tc=1,
            )

        # fused_qkv_a [Rq+Rkv, H] 在 host 切成 q_a [Rq, H] / kv_a [Rkv, H] 两块
        q_a_w  = w["fused_qkv_a"][:cfg.Rq].contiguous()
        kv_a_w = w["fused_qkv_a"][cfg.Rq:].contiguous()

        # LocalMem-packed weights (#1/#2/#3/#5 算子的 GEMM weight)
        self._w_lmem = {
            "q_a":    _lmem(q_a_w),                   # #1 dsa_q_a_proj_norm
            "kv_a":   _lmem(kv_a_w),                  # #2 dsa_kv_a_proj_norm_store
            "q_b":    _lmem(w["q_b_proj"]),           # #3 dsa_q_main_absorb (q upproject)
            "w_kc":   _lmem(w["w_kc"]),               # #3 dsa_q_main_absorb (absorb)
            "wk_idx": _lmem(w["wk_idx"]),             # #5 dsa_indexer_k_prep_store
            "h_di":   _lmem(w["hadamard_Di"]),        # #5 dsa_indexer_k_prep_store (Hadamard)
        }
        # Plain Zeus tensors (kernel 不要求 LocalMem 的: norm / bias / non-LocalMem GEMM)
        self._w_z = {
            "q_a_norm":      w["q_a_norm"].to("zeus"),
            "kv_a_norm":     w["kv_a_norm"].to("zeus"),
            "wq_b":          w["wq_b"].to("zeus"),               # #4 dsa_indexer_q_weights
            "h_di_z":        w["hadamard_Di"].to("zeus"),        # #4 (plain Zeus)
            "weights_proj": w["weights_proj"].to("zeus"),        # #4
            "k_norm_weight": w["k_norm_weight"].to("zeus"),      # #5
            "k_norm_bias":   w["k_norm_bias"].to("zeus"),        # #5
            "w_vc":          w["w_vc"].to("zeus"),               # #11 dsa_post_o_proj_no_cp
            "o_proj":        w["o_proj"].to("zeus"),             # #11
        }
        self._zeus_packed = True

    # ── Paged state init ───────────────────────────────────────
    def init_paged_state(
        self,
        history: dsa.GlobalHistory,
        *,
        page_size: Optional[int] = None,
        num_physical_pages: Optional[int] = None,
        dual_core: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """从 ``dsa.GlobalHistory`` 构造 paged-attention state.

        退化形式 (默认): ``page_size = S_history + 1``, ``block_table[b, 0] = b``,
        ``total_slots = B * page_size``. 每个 batch 独占一个 logical page, 装下
        full history + 一步 new write.

        真正 paged 场景: caller 传 ``page_size``, 自己构造 block_table
        (logical→physical 映射), 让 manager 跨 seq 复用 pool.

        Returns:
          dict 包含:
            latent_kv_pool   [total_slots, Rkv]   bf16    paged 共享池
            index_body_pool  [total_slots, Di]    fp8e4m3 paged 共享池
            index_scale_pool [total_slots]        fp32    paged 共享池
            block_table      [B, max_pages]       int32   logical_page → phys_page
            seq_lens         [B]                  int32   current history length
            page_size        int
        """
        cfg = self.cfg
        latent_kv = history.latent_kv             # [B, S, Rkv] bf16
        index_body = history.index_body           # [B, S, Di]  bf16 (FP8 proxy)
        index_scale = history.index_scale         # [B, S]      fp32
        B, S_hist, Rkv = latent_kv.shape
        assert Rkv == cfg.Rkv
        Di = index_body.shape[-1]
        assert Di == cfg.Di

        if dual_core:
            from sgl_kernel_zeus.dsa_index_k_dual_core import (
                owner_of_physical_page, local_page_of,
            )
            assert page_size is not None and num_physical_pages is not None, (
                "dual_core path requires explicit page_size and num_physical_pages"
            )
            P = int(num_physical_pages)
            assert P % 2 == 0, "num_physical_pages must be even"
            pages_per_seq = -(-S_hist // page_size)            # ceil
            num_local_pages = P // 2
            total_slots = P * page_size
            split = num_local_pages
            # Per-core capacity guard. Allocation alternates cores per logical
            # page (lp%2==0 → core0 region [0, split); lp%2==1 → core1 region
            # [split, P)). When pages_per_seq is odd, core0 demands
            # B*ceil(pages_per_seq/2) and core1 B*floor(pages_per_seq/2); a total
            # guard P >= B*pages_per_seq can pass while core0 overflows past
            # `split` into core1's region → same physical page handed to two
            # sequences. Check each core's region independently.
            need_c0 = B * math.ceil(pages_per_seq / 2)
            need_c1 = B * (pages_per_seq // 2)
            assert need_c0 <= num_local_pages and need_c1 <= num_local_pages, (
                f"dual_core pool too small: per-core need (c0={need_c0}, c1={need_c1}) "
                f"exceeds num_local_pages={num_local_pages} (P={P}, B={B}, "
                f"pages_per_seq={pages_per_seq}); increase num_physical_pages"
            )

            # Small-page block_table: consecutive logical pages of a sequence
            # alternate cores (forces page-wise cross-core split).
            # core0 owns physical pages [0, P/2); core1 owns [P/2, P).
            block_table = torch.full((B, pages_per_seq), -1, dtype=torch.int32)
            next_c0, next_c1 = 0, split
            for b in range(B):
                for lp in range(pages_per_seq):
                    if lp % 2 == 0:
                        pp = next_c0; next_c0 += 1
                    else:
                        pp = next_c1; next_c1 += 1
                    block_table[b, lp] = pp
            assert next_c0 <= split and next_c1 <= P, (
                "dual_core allocation overflowed its core region"
            )

            body_c0 = torch.zeros(
                (num_local_pages, page_size, Di), dtype=torch.float8_e4m3fn,
            )
            body_c1 = torch.zeros(
                (num_local_pages, page_size, Di), dtype=torch.float8_e4m3fn,
            )
            scale_pool = torch.zeros((total_slots,), dtype=torch.float32)
            latent_pool = torch.zeros((total_slots, Rkv), dtype=torch.bfloat16)
            for b in range(B):
                for s in range(S_hist):
                    lp_logical = s // page_size
                    sip = s % page_size
                    pp = int(block_table[b, lp_logical])
                    slot = pp * page_size + sip
                    scale_pool[slot] = index_scale[b, s]
                    latent_pool[slot] = latent_kv[b, s]
                    bank = body_c0 if owner_of_physical_page(pp, P) == 0 else body_c1
                    bank[local_page_of(pp, P), sip] = (
                        index_body[b, s].to(torch.float8_e4m3fn)
                    )

            seq_lens = torch.full((B,), S_hist, dtype=torch.int32)

            # 4 zero-init LocalMem gather buffers (same as degenerate path; #9
            # needs them — see the rationale comment in the degenerate path).
            Ktop = cfg.Ktop
            zero_K   = torch.zeros(B, Ktop, Rkv, dtype=torch.bfloat16)
            zero_K_T = torch.zeros(B, Rkv, Ktop, dtype=torch.bfloat16)
            k_local_c0_lmem   = torch.zeus.local_memory.from_tensor(
                zero_K,   kind="native", Tr=1, Tc=1,
            )
            k_local_c1_lmem   = torch.zeus.local_memory.from_tensor(
                zero_K,   kind="native", Tr=1, Tc=1,
            )
            k_local_t_c0_lmem = torch.zeus.local_memory.from_tensor(
                zero_K_T, kind="native", Tr=1, Tc=1,
            )
            k_local_t_c1_lmem = torch.zeus.local_memory.from_tensor(
                zero_K_T, kind="native", Tr=1, Tc=1,
            )

            return {
                "latent_kv_pool":   latent_pool,
                "index_scale_pool": scale_pool,
                "scale_cache":      scale_pool,
                "block_table":      block_table,
                "seq_lens":         seq_lens,
                "page_size":        int(page_size),
                "num_physical_pages": P,
                "body_cache_c0":    body_c0,
                "body_cache_c1":    body_c1,
                "dual_core":        True,
                "k_local_c0":       k_local_c0_lmem,
                "k_local_c1":       k_local_c1_lmem,
                "k_local_t_c0":     k_local_t_c0_lmem,
                "k_local_t_c1":     k_local_t_c1_lmem,
            }

        if page_size is None:
            page_size = S_hist + 1
        if page_size < S_hist + 1:
            raise ValueError(
                f"init_paged_state: page_size ({page_size}) must be ≥ "
                f"S_hist+1 ({S_hist + 1}) for the degenerate single-page layout"
            )

        max_pages = 1
        total_slots = B * page_size

        # Identity block_table: each batch owns one logical page mapped to phys page b
        block_table = torch.zeros((B, max_pages), dtype=torch.int32)
        for b in range(B):
            block_table[b, 0] = b

        latent_pool = torch.zeros((total_slots, Rkv), dtype=torch.bfloat16)
        body_pool = torch.zeros((total_slots, Di), dtype=torch.float8_e4m3fn)
        scale_pool = torch.zeros((total_slots,), dtype=torch.float32)

        # Lay history into pools at slots [b*PS + s for s in 0..S_hist)
        for b in range(B):
            base = b * page_size
            latent_pool[base: base + S_hist] = latent_kv[b]
            body_pool[base: base + S_hist] = index_body[b].to(torch.float8_e4m3fn)
            scale_pool[base: base + S_hist] = index_scale[b]

        seq_lens = torch.full((B,), S_hist, dtype=torch.int32)

        # K_local gather output buffers — pre-zeroed LocalMem (native mode),
        # 跨 step 复用同一组 pool 指针.
        # 动机: dsa_latent_k_gather_paged sim 跳过 invalid slot 的写入 (cp > 1
        # 时非本核 token 不写入 c0/c1; cp = 1 时无 invalid case 但 contract
        # 一致), 依赖 caller 端 buffer 是 zero-init 才能保证 sparse_mqa_partial
        # 的算术 mask 把 invalid 位置乘 0 时仍是有限值. 临时 alloc 的 device
        # buffer 含未初始化内存, 可能让 invalid 位置出现 NaN/Inf -> 算术 mask
        # × 0 = NaN, 污染整条 attn 输出.
        # 解法: 一次性 CPU zeros → from_tensor(kind='native') memcpy 到 LocalMem,
        # 后续 forward 直接复用这 4 个 LocalMem; gather 的 C++ wrapper 走
        # isLocalMem 路径, copyFromLocalMemSlice 把当前 LocalMem 内容 (= zero)
        # 拷到 tmp, sim 只覆盖 valid slot, copyToLocalMemSlice 写回 — invalid
        # slot 一直保持 zero, 永不引入 NaN.
        Ktop = cfg.Ktop
        zero_K   = torch.zeros(B, Ktop, Rkv, dtype=torch.bfloat16)
        zero_K_T = torch.zeros(B, Rkv, Ktop, dtype=torch.bfloat16)
        k_local_c0_lmem   = torch.zeus.local_memory.from_tensor(
            zero_K,   kind="native", Tr=1, Tc=1,
        )
        k_local_c1_lmem   = torch.zeus.local_memory.from_tensor(
            zero_K,   kind="native", Tr=1, Tc=1,
        )
        k_local_t_c0_lmem = torch.zeus.local_memory.from_tensor(
            zero_K_T, kind="native", Tr=1, Tc=1,
        )
        k_local_t_c1_lmem = torch.zeus.local_memory.from_tensor(
            zero_K_T, kind="native", Tr=1, Tc=1,
        )

        return {
            "latent_kv_pool":   latent_pool,
            "index_body_pool":  body_pool,
            "index_scale_pool": scale_pool,
            "block_table":      block_table,
            "seq_lens":         seq_lens,
            "page_size":        int(page_size),
            # K_local gather LocalMem pool (zero-init, 跨 step 复用)
            "k_local_c0":       k_local_c0_lmem,
            "k_local_c1":       k_local_c1_lmem,
            "k_local_t_c0":     k_local_t_c0_lmem,
            "k_local_t_c1":     k_local_t_c1_lmem,
        }

    # ── Zeus forward helpers ───────────────────────────────────
    def _forward_to_logits(
        self,
        hidden_z: torch.Tensor,
        paged_state: Dict[str, torch.Tensor],
        *,
        advance_seq_lens: bool = True,
    ) -> torch.Tensor:
        """DSA #1–#6: 首帧搬 device → 出 index logits ``[B, max_logical_s]``.

        single-core / dual_core 在 #5 STORE + #6 READ 分叉, 其余共用. #3 算出的
        ``q_new_z`` 经 ``paged_state["_q_new_z"]`` 临时槽传给 :meth:`forward_zeus`
        的 #7–#11 (避免改方法签名 / 多返回值).
        """
        cfg = self.cfg
        B = hidden_z.shape[0]

        # Move paged_state to device on first call; subsequent calls reuse device tensors.
        if "_z_loaded" not in paged_state:
            if paged_state.get("dual_core"):
                paged_state["latent_kv_pool"] = paged_state["latent_kv_pool"].to("zeus")
                paged_state["body_cache_c0"] = torch.zeus.local_memory.from_tensor(
                    paged_state["body_cache_c0"].to("zeus"), kind="weight", Tr=1, Tc=1,
                )
                paged_state["body_cache_c1"] = torch.zeus.local_memory.from_tensor(
                    paged_state["body_cache_c1"].to("zeus"), kind="weight", Tr=1, Tc=1,
                )
                paged_state["scale_cache"] = paged_state["scale_cache"].to("zeus")
                paged_state["block_table"] = paged_state["block_table"].to("zeus")
                paged_state["seq_lens"] = paged_state["seq_lens"].to("zeus")
                paged_state["_z_loaded"] = True
            else:
                paged_state["latent_kv_pool"] = paged_state["latent_kv_pool"].to("zeus")
                # body_pool 是 fp8 LocalMem — 需要装到 LocalMem 给 indexer_k_prep_store
                # 和 dsa_index_logits_paged 使用.
                paged_state["index_body_pool"] = torch.zeus.local_memory.from_tensor(
                    paged_state["index_body_pool"], kind="weight", Tr=1, Tc=1,
                )
                paged_state["index_scale_pool"] = paged_state["index_scale_pool"].to("zeus")
                paged_state["block_table"] = paged_state["block_table"].to("zeus")
                paged_state["seq_lens"] = paged_state["seq_lens"].to("zeus")
                paged_state["_z_loaded"] = True

        latent_pool_z = paged_state["latent_kv_pool"]
        block_table_z = paged_state["block_table"]
        seq_lens_z = paged_state["seq_lens"]
        page_size = paged_state["page_size"]
        max_logical_s = page_size * block_table_z.shape[1]

        # #1  q_a_proj + RMSNorm → q_lora
        q_lora_z = sgl_kernel_zeus.dsa_q_a_proj_norm(
            hidden_z, self._w_lmem["q_a"], self._w_z["q_a_norm"],
            eps=cfg.rms_norm_eps,
        )

        # #2a  Compute per-seq write slot: new_slot[b] = block_table[b, seq_lens[b]/PS] * PS + seq_lens[b]%PS
        slot_mapping_z = sgl_kernel_zeus.dsa_compute_new_slot(
            block_table_z, seq_lens_z, page_size=page_size,
        )

        # #2b  kv_a_proj + norm + STORE TO POOL (latent_pool_z[slot_mapping[b]] = new_k[b])
        sgl_kernel_zeus.dsa_kv_a_proj_norm_store(
            hidden_z, self._w_lmem["kv_a"], self._w_z["kv_a_norm"],
            slot_mapping_z, latent_pool_z,
            eps=cfg.rms_norm_eps,
        )

        # #3  q_b_proj + absorb bmm(w_kc) → q_new
        q_new_z = sgl_kernel_zeus.dsa_q_main_absorb(
            q_lora_z, self._w_lmem["q_b"], self._w_lmem["w_kc"],
        )
        # 经 paged_state 临时槽把 q_new_z 传给 forward_zeus 的 #7–#11.
        paged_state["_q_new_z"] = q_new_z

        # #4  indexer Q + weights
        q_body_z, _q_scale_z, weights_z = sgl_kernel_zeus.dsa_indexer_q_weights(
            q_lora_z, hidden_z, self._w_z["wq_b"], self._w_z["h_di_z"],
            self._w_z["weights_proj"],
            num_index_heads=cfg.I, index_head_dim=cfg.Di,
        )

        if paged_state.get("dual_core"):
            from sgl_kernel_zeus.dsa_index_k_dual_core import build_index_k_exec_tables
            P = paged_state["num_physical_pages"]
            bc0 = paged_state["body_cache_c0"]
            bc1 = paged_state["body_cache_c1"]
            scale_cache_z = paged_state["scale_cache"]

            # #5  STORE into the two persistent LocalMem banks
            sgl_kernel_zeus.dsa_indexer_k_prep_store_dual_core(
                hidden_z, self._w_lmem["wk_idx"],
                self._w_z["k_norm_weight"], self._w_z["k_norm_bias"],
                self._w_lmem["h_di"], slot_mapping_z,
                bc0, bc1, scale_cache_z,
                num_physical_pages=P, page_size=page_size,
                eps=cfg.rms_norm_eps,
            )

            if advance_seq_lens:
                seq_lens_z = seq_lens_z + 1
                paged_state["seq_lens"] = seq_lens_z

            # #5.5  host-side exec tables (outside device-resident chain)
            tables = build_index_k_exec_tables(
                block_table_z.cpu(), seq_lens_z.cpu(), P, page_size, bc0, bc1,
            )
            addr_tbl = tables["addr_table"]
            work_list = tables["work_list"]

            # #6  two per-core READs sharing a -1e30-preinit logits buffer
            logits_z = torch.full(
                (B, max_logical_s), -1e30, dtype=torch.float32, device="zeus",
            )
            for core_id in (0, 1):
                sgl_kernel_zeus.dsa_index_logits_lmem_addr_table(
                    q_body_z, weights_z,
                    bc0 if core_id == 0 else bc1,
                    scale_cache_z,
                    addr_tbl[core_id].to("zeus"), work_list[core_id].to("zeus"),
                    seq_lens_z, logits_z,
                    page_size=page_size, core_id=core_id,
                )
        else:
            body_pool_lmem = paged_state["index_body_pool"]
            scale_pool_z = paged_state["index_scale_pool"]

            # #5  indexer K prep + STORE TO POOL (body_pool[slot] + scale_pool[slot])
            sgl_kernel_zeus.dsa_indexer_k_prep_store(
                hidden_z, self._w_lmem["wk_idx"],
                self._w_z["k_norm_weight"], self._w_z["k_norm_bias"],
                self._w_lmem["h_di"], slot_mapping_z,
                body_pool_lmem, scale_pool_z,
                eps=cfg.rms_norm_eps,
            )

            # Advance seq_lens so subsequent reads see the new step.
            # aten op, single device kernel launch — device-side history advance.
            if advance_seq_lens:
                seq_lens_z = seq_lens_z + 1
                paged_state["seq_lens"] = seq_lens_z

            # #6  paged index GEMM → logits [B, max_logical_s].
            # max_logical_s = page_size * max_pages_per_seq (static upper bound).
            # Positions >= seq_lens[b] get -inf inside the kernel.
            logits_z = sgl_kernel_zeus.dsa_index_logits_paged(
                q_body_z, weights_z, body_pool_lmem, scale_pool_z,
                block_table_z, seq_lens_z,
                page_size=page_size,
                max_logical_s=max_logical_s,
            )

        return logits_z

    def forward_zeus_logits(self, hidden_z, paged_state):
        """跑到 #6 返回 index logits [B, max_logical_s](对拍用)。"""
        if not self._zeus_packed:
            self._pack_zeus()
        return self._forward_to_logits(hidden_z, paged_state)

    # ── Zeus forward (paged) ───────────────────────────────────
    def forward_zeus(
        self,
        hidden_z: torch.Tensor,
        paged_state: Dict[str, torch.Tensor],
        *,
        advance_seq_lens: bool = True,
    ) -> torch.Tensor:
        """Zeus DSA decode with paged-attention pool.

        ``hidden_z: [B, H] bf16 (zeus)  ->  [B, H] bf16 (zeus)``.

        Paged chain:
          - KV pool 是 [total_slots, ...] 共享池, block_table (logical→physical) +
            seq_lens (per-seq 长度) 完成 page-table 间接寻址
          - kv_a / indexer_k 的 store 直接落到 pool[slot_mapping[b], :]
            (slot_mapping = ``dsa_compute_new_slot(block_table, seq_lens, PS)``)
          - index logits 用 ``dsa_index_logits_paged`` (kernel 内 page-table 寻址)
          - top-k 出 logical position → ``dsa_translate_topk_positions`` → phys slot
          - latent K gather 用 ``dsa_latent_k_gather_paged`` (paged pool 直接 gather)
          - 全链路 device-resident, 没有 D2H concat / per-batch loop / host coordination

        Args:
          hidden_z:         [B, H] bf16 (zeus)
          paged_state:      由 :meth:`init_paged_state` 返回的 host-side dict.
                            首次调用时全部搬到 device, 之后跨 step 持有 device 引用.
          advance_seq_lens: True (默认) → 写完 pool 后 in-place 推进 ``seq_lens[b] += 1``,
                            使下游 read 看到新 step. multi-step decode 必需.
        """
        if not self._zeus_packed:
            self._pack_zeus()
        cfg = self.cfg

        logits_z = self._forward_to_logits(
            hidden_z, paged_state, advance_seq_lens=advance_seq_lens,
        )

        block_table_z = paged_state["block_table"]
        seq_lens_z = paged_state["seq_lens"]
        latent_pool_z = paged_state["latent_kv_pool"]
        page_size = paged_state["page_size"]
        max_logical_s = page_size * block_table_z.shape[1]
        positions_z = torch.arange(max_logical_s, dtype=torch.int32).to("zeus")

        # q_new_z 由 #3 在 helper 内算出, 通过 paged_state 临时槽传出 (避免改方法签名).
        q_new_z = paged_state.pop("_q_new_z")

        # #7  local top-K (cp=1 → IS global top-K). Output is **logical** position
        # in [0, max_logical_s); -inf positions naturally lose to valid ones.
        _top_lg_z, top_pos_z = sgl_kernel_zeus.dsa_local_topk_radix(
            logits_z, positions_z, Ktop=cfg.Ktop,
        )

        # #8  logical position → physical slot in pool
        phys_slot_z = sgl_kernel_zeus.dsa_translate_topk_positions(
            top_pos_z, block_table_z, page_size=page_size,
        )

        # #9  paged gather. K_local c0/c1/T_c0/T_c1 复用 paged_state 里 init 时
        # 创建的 zero-init LocalMem pool (kind='native'), 跨 step 不重新分配,
        # invalid slot 始终保持 zero (避免 sparse_mqa 的乘 0 mask 撞上未初始化
        # NaN). masks 由 sim 写满, 用临时 device buffer 即可.
        K_c0   = paged_state["k_local_c0"]
        K_c1   = paged_state["k_local_c1"]
        K_T_c0 = paged_state["k_local_t_c0"]
        K_T_c1 = paged_state["k_local_t_c1"]
        _, _, _, _, m_c0, m_c1 = (
            sgl_kernel_zeus.dsa_latent_k_gather_paged(
                phys_slot_z, latent_pool_z,
                k_local_c0=K_c0, k_local_c1=K_c1,
                k_local_t_c0=K_T_c0, k_local_t_c1=K_T_c1,
            )
        )

        # #10  sparse MQA partial
        po_z, _pl_z = sgl_kernel_zeus.dsa_sparse_mqa_partial(
            q_new_z, K_c0, K_c1, K_T_c0, K_T_c1, m_c0, m_c1,
            scaling=cfg.scaling,
        )

        # #11  V absorb + o_proj → bf16 [B, H]
        out_z = sgl_kernel_zeus.dsa_post_o_proj_no_cp(
            po_z, self._w_z["w_vc"], self._w_z["o_proj"],
        )
        return out_z


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = (
    "dsa_q_a_proj_norm", "dsa_kv_a_proj_norm_store", "dsa_q_main_absorb",
    "dsa_indexer_q_weights", "dsa_indexer_k_prep_store",
    "dsa_index_logits_paged", "dsa_local_topk_radix",
    "dsa_translate_topk_positions", "dsa_latent_k_gather_paged",
    "dsa_compute_new_slot", "dsa_sparse_mqa_partial", "dsa_post_o_proj_no_cp",
)


def _run_stage(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print(f"Stage: DSA sublayer ({args.config})  seqlen={args.seqlen}")
    print("=" * 60)

    torch.manual_seed(args.seed)
    attn = Glm5NextDsaAttn(args.config, seed=args.seed)
    cfg = attn.cfg
    B = args.num_tokens
    print(f"  cfg: {cfg.name}  H={cfg.H}  Nh={cfg.Nh}  "
          f"Rq={cfg.Rq} Rkv={cfg.Rkv} Dnope={cfg.Dnope} Dv={cfg.Dv}  "
          f"I={cfg.I} Di={cfg.Di} Ktop={cfg.Ktop}")

    hidden = (torch.randn(B, cfg.H, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    history, block_span = attn.init_state(B, args.seqlen, seed=args.seed + 1)

    # ── REF ───────────────────────────────────────────────────
    ref_out: Optional[torch.Tensor] = None
    if args.mode in ("ref", "both"):
        ref_out = attn.forward(
            hidden, history, new_pos=args.seqlen, block_span=block_span,
        )
        ok = ref_out.shape == (B, cfg.H) and ref_out.dtype == torch.bfloat16
        print(f"  REF out={tuple(ref_out.shape)} {ref_out.dtype}")
        print(f"  REF out[0,:4] = "
              f"{[round(v,4) for v in ref_out[0,:4].float().tolist()]}")
        if not ok:
            return False

    # ── Zeus (paged) ──────────────────────────────────────────
    zeus_ok: Optional[bool] = None
    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"  ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
        else:
            try:
                paged_state = attn.init_paged_state(history)
                z_out = attn.forward_zeus(hidden.to("zeus"), paged_state)
                z_out_cpu = z_out.cpu()
                finite = torch.isfinite(z_out_cpu).all().item()
                shape_ok = (z_out_cpu.shape == (B, cfg.H)
                            and z_out_cpu.dtype == torch.bfloat16)
                print(f"  ZEUS out={tuple(z_out_cpu.shape)} {z_out_cpu.dtype}  "
                      f"finite={finite}")
                print(f"  ZEUS out[0,:4] = "
                      f"{[round(v,4) for v in z_out_cpu[0,:4].float().tolist()]}")
                if ref_out is not None:
                    # DSA chain (11 算子) 累计误差预算, 与原 stage_dsa_decode 一致.
                    zeus_ok = compare_tensors(
                        f"dsa.{args.config}.out", ref_out, z_out_cpu,
                        atol=5e-2, rtol=5e-2,
                    ) and finite and shape_ok
                else:
                    zeus_ok = finite and shape_ok
            except Exception as e:
                import traceback
                print(f"  ZEUS EXCEPTION: {e!r}")
                traceback.print_exc()
                zeus_ok = False

    if args.mode == "ref":
        return True
    if args.mode == "zeus":
        return zeus_ok
    return True if zeus_ok is None else zeus_ok


def main():
    parser = make_argparser(
        "dev_dsa_attn",
        description="GLM5-Next DSA (Deepseek Sparse Attention) decode sublayer dev test",
    )
    parser.add_argument("--seqlen", type=int, default=64,
                        help="DSA decode 的历史长度 (KV cache 长度, 不含 new step)")
    args = parser.parse_args()
    print_header("GLM5-Next DSA decode sublayer", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_dsa_attn ({args.config})", ok)


if __name__ == "__main__":
    main()
