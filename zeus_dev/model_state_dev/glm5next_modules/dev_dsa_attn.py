"""
GLM5-Next DSA (Deepseek Sparse Attention) decode sublayer 独立模块 + REF↔Zeus 对拍.

设计与 ``dev_linear_attn.py`` / ``dev_moe.py`` / ``dev_mhc.py`` 同构. 把 DSA
decode (cp=1) 抽成 ``Glm5NextDsaAttn`` 类, 暴露:
  __init__ + init_state(B, seqlen) + init_paged_state(history) +
  forward (REF) + forward_zeus (paged Zeus chain).

KV cache 通过 paged-attention pool 管理: [total_slots, ...] 共享池 +
block_table (logical→physical) + seq_lens (per-seq 历史长度). 完全没有
host-side concat / per-batch CPU loop, 全链路 device-resident.

按 physical-page ownership 双核 (dual-core): physical page 连续范围对半切,
前 P/2 归 core0、后 P/2 归 core1; 同一 seq 的相邻 logical page 交替落核.

DSA decode Zeus chain (dual-core paged, 12 颗 sgl-kernel-zeus 算子):
  1.  dsa_q_a_proj_norm                  Q low-rank projection + RMSNorm → q_lora
  2a. dsa_compute_new_slot               new_slot[b] = bt[b, sl/PS]*PS + sl%PS
  2b. dsa_kv_a_proj_norm_store           KV low-rank + norm + STORE latent_pool[slot]
  3.  dsa_q_main_absorb                  Q upproject + absorb bmm(w_kc) → q_new
  4.  dsa_indexer_q_weights              indexer Q (rotate+fp8) + per-head weights
  5.  dsa_indexer_k_prep_store_dual_core indexer K prep (rotate+fp8) + STORE 两个 per-core bank
        ─ device: seq_lens += 1 (aten op, 1 launch) —— STORE/READ 分界
  6.  dsa_index_logits_lmem_addr_table   两核 paged indexer GEMM → logits [B, max_logical_s]
  7.  dsa_local_topk_radix               local top-K (cp=1 → IS global)
  8.  dsa_translate_topk_positions       logical pos → phys slot in pool
  9.  dsa_latent_k_gather_paged          paged latent K gather (返回 c0/c1 per-core)
  10. dsa_sparse_mqa_partial             sparse MQA partial
  11. dsa_post_o_proj_no_cp              V absorb + o_proj → bf16 [B, H]

host-side per-core exec tables (build_index_k_exec_tables → addr_table/work_list)
只依赖 block_table + 本步 seq_lens + 静态 bank 地址几何, 不碰任何 device 中间结果,
故在 device chain **之前** 一次性算好, 不穿插进算子链.

与 ``dev_glm5next_block_decode_test.zeus_dsa_decode`` 的关键改进:
  - **静态 weights 一次性 pack** (LocalMem + .to("zeus")) 进 ``_pack_zeus()``,
    forward 复用. 原版每次 forward 都 ``_lmem_pack(q_a_w)`` 等 11 个 LocalMem
    pack + 多个 ``.to("zeus")`` 搬运重做.
  - **paged-state 在 ``init_paged_state`` 内一次性搬上 device** (latent_kv_pool /
    body_cache_c0/c1 / ...), forward 进来即纯跑 kernel 链, 不再做任何 load.

State:
  - history (REF only): ``dsa.GlobalHistory``  host-side
    - latent_kv  [B, S, Rkv]  bf16
    - index_body [B, S, Di]   bf16 (FP8 proxy)
    - index_scale [B, S]      fp32
  - paged_state (Zeus): ``init_paged_state`` 返回时即 device-resident dict,
    跨 step 复用. seq_lens 每步 in-place +1, latent/body/scale pool 每步在
    slot_mapping 位置写入新 K, history 自动累积.

用法:
  python glm5next_modules/dev_dsa_attn.py                 # 16b / both
  python glm5next_modules/dev_dsa_attn.py --config next
  python glm5next_modules/dev_dsa_attn.py --mode zeus --seqlen 128
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

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


# ── Dual-core page ownership + exec-table builder ────────────────
# 原 sgl_kernel_zeus.dsa_index_k_dual_core, 迁到本 serving 侧 (exec table
# 构造是 caller 的职责, kernel 包只提供算子). 这样 dev_dsa_attn 不再依赖
# kernel 包的内部 Python 模块, 没装/没编 dual-core kernel 时仍可纯 REF 跑.
#
# CP=1, 按 physical-page ownership 分核: physical page 连续范围对半切, 前一半
# core0、后一半 core1. pool 固定 ⇒ 静态映射, 无 manager.
_CORE_NUM = 2
_PAGES_PER_BLOCK = 512  # 每 16KB native block 容纳的 page 数 (addr_table 内轴)
_REPEAT = 8             # addr_table 第三维: 同一 offset 的 8 份相同复制


def _check_even(P: int) -> None:
    if P % 2 != 0:
        raise ValueError(f"P (physical page total) must be even, got {P}")


def _check_pp(pp: int, P: int) -> None:
    if pp < 0 or pp >= P:
        raise ValueError(f"physical page out of range: pp={pp} not in [0, {P})")


def owner_of_physical_page(pp: int, P: int) -> int:
    _check_even(P)
    _check_pp(pp, P)
    return 0 if pp < P // 2 else 1


def local_page_of(pp: int, P: int) -> int:
    _check_even(P)
    _check_pp(pp, P)
    split = P // 2
    return pp if pp < split else pp - split


def pages_of_core(core: int, P: int) -> List[int]:
    _check_even(P)
    if core not in (0, 1):
        raise ValueError(f"core must be 0 or 1, got {core}")
    split = P // 2
    return list(range(0, split)) if core == 0 else list(range(split, P))


def build_index_k_exec_tables(
    semantic_block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    P: int,
    page_size: int,
    body_lmem_c0,
    body_lmem_c1,
) -> dict:
    """每个 step 从最新 block_table 全量重建 per-core 地址表 + work list。

    返回 dict：addr_table=[t0,t1] int32 [B, max_blocks_core, REPEAT, PAGES_PER_BLOCK]，
    work_list=[w0,w1] int32 [B, max_pages_core, 3]=(b, logical_page, physical_page)，
    外加 max_blocks_core, max_pages_core。
    entry = group_ptr(local_page) - base_ptr（相对本核 bank 基址的 int32 字节 offset），
    core 侧 addr = base_ptr + offset。几何空位填 0；有效性以 work_list 为准（READ
    只读 work_list 指向的格），故 offset 0（local_page 0）合法、不与空位混淆。
    """
    _check_even(P)
    if semantic_block_table.dim() != 2:
        raise ValueError("semantic_block_table must be [B, max_pages_per_seq]")
    B, max_pages_per_seq = semantic_block_table.shape
    bt = semantic_block_table.cpu().tolist()
    sl = seq_lens.cpu().tolist()
    # 兼容 to_local_mem(LocalMemTensor) 与 from_tensor(ZeusLocalMemTensor)
    lmems = [getattr(body_lmem_c0, "local_mem", body_lmem_c0),
             getattr(body_lmem_c1, "local_mem", body_lmem_c1)]

    owned = [[[] for _ in range(B)] for _ in range(_CORE_NUM)]  # owned[core][b]=list[(lp,pp)]
    for b in range(B):
        n_valid_pages = (int(sl[b]) + page_size - 1) // page_size
        for lp in range(min(n_valid_pages, max_pages_per_seq)):
            pp = int(bt[b][lp])
            if pp < 0:
                continue
            c = owner_of_physical_page(pp, P)
            owned[c][b].append((lp, pp))

    # 按 ownership 统计每个 (core, b) 实际拥有的 page 数
    max_pages_core = max(
        (len(owned[c][b]) for c in range(_CORE_NUM) for b in range(B)), default=0,
    )
    max_pages_core = max(max_pages_core, 1)
    max_blocks_core = (max_pages_core + _PAGES_PER_BLOCK - 1) // _PAGES_PER_BLOCK

    addr_tables, work_lists = [], []
    for c in range(_CORE_NUM):
        base_ptr_c = int(lmems[c].base_ptr)  # 本核 bank 基址
        t = torch.zeros((B, max_blocks_core, _REPEAT, _PAGES_PER_BLOCK), dtype=torch.int32)
        w = torch.full((B, max_pages_core, 3), -1, dtype=torch.int32)
        for b in range(B):
            for j, (lp, pp) in enumerate(owned[c][b]):
                # entry = bank-relative 字节 offset；core 侧 base_ptr + offset 定位
                offset = int(lmems[c].group_ptr(local_page_of(pp, P))) - base_ptr_c
                blk, within = j // _PAGES_PER_BLOCK, j % _PAGES_PER_BLOCK
                t[b, blk, :, within] = offset
                w[b, j, 0] = b
                w[b, j, 1] = lp
                w[b, j, 2] = pp
        addr_tables.append(t)
        work_lists.append(w)

    return {
        "addr_table": addr_tables,
        "work_list": work_lists,
        "max_blocks_core": max_blocks_core,
        "max_pages_core": max_pages_core,
    }


# ── Module ──────────────────────────────────────────────────────
class Glm5NextDsaAttn:
    """GLM5-Next DSA decode sublayer (cp=1) — dual-core paged-attention Zeus chain.

    使用模式::

        attn = Glm5NextDsaAttn(which="16b", seed=42)
        history, block_span = attn.init_state(B=batch, seqlen=ctx_len, seed=...)
        # REF
        out = attn.forward(hidden, history, new_pos=ctx_len, block_span=block_span)
        # Zeus (dual-core paged)
        paged_state = attn.init_paged_state(
            history, page_size=512, num_physical_pages=4,
        )
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
            "wk_idx": _lmem(w["wk_idx"]),             # #5 dsa_indexer_k_prep_store_dual_core
            "h_di":   _lmem(w["hadamard_Di"]),        # #5 dsa_indexer_k_prep_store_dual_core (Hadamard)
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

    # ── Paged state init (dual-core, device-resident) ───────────
    def init_paged_state(
        self,
        history: dsa.GlobalHistory,
        *,
        page_size: int,
        num_physical_pages: int,
    ) -> Dict[str, torch.Tensor]:
        """从 ``dsa.GlobalHistory`` 构造 **device-resident** dual-core paged state.

        physical page 连续范围对半切核 (前 P/2 → core0, 后 P/2 → core1, 见
        本模块顶部 ``owner_of_physical_page``); 同一 seq 的相邻 logical page 奇偶交替落核,
        强制 page-wise 跨核 split. history 直接摊进两个 per-core bank + 共享
        latent/scale pool, 返回时全部已搬上 zeus device + LocalMem pack ——
        ``forward_zeus`` 进来不再做任何 load.

        Returns dict (全部 device-resident):
          latent_kv_pool     [total_slots, Rkv]    bf16     共享 latent 池 (zeus)
          body_cache_c0/c1   [P/2, page_size, Di]  fp8e4m3  per-core indexer-K bank (LocalMem)
          scale_cache        [total_slots]         fp32     共享 scale 池 (zeus)
          block_table        [B, pages_per_seq]    int32    logical→physical (zeus)
          block_table_host   同上                  int32    host 副本 (供 host exec tables 复用)
          seq_lens           [B]                   int32    history length (zeus); prepare_decode_step 推进为 inclusive
          seq_lens_host      [B]                   int32    host 副本 (slot 计算 + exec tables 复用)
          page_size / num_physical_pages           int
          k_local_c0/c1/t_c0/t_c1                            #9 gather 复用的 zero-init LocalMem
        """
        if ZEUS_IMPORT_ERROR is not None:
            raise RuntimeError(f"Zeus runtime unavailable: {ZEUS_IMPORT_ERROR}")

        cfg = self.cfg
        latent_kv = history.latent_kv             # [B, S, Rkv] bf16
        index_body = history.index_body           # [B, S, Di]  bf16 (FP8 proxy)
        index_scale = history.index_scale         # [B, S]      fp32
        B, S_hist, Rkv = latent_kv.shape
        assert Rkv == cfg.Rkv
        Di = index_body.shape[-1]
        assert Di == cfg.Di

        P = int(num_physical_pages)
        assert P % 2 == 0, "num_physical_pages must be even"
        pages_per_seq = -(-S_hist // page_size)            # ceil
        num_local_pages = P // 2
        total_slots = P * page_size
        split = num_local_pages
        # Per-core capacity guard. 分配按 logical page 奇偶交替落核
        # (lp%2==0 → core0 区 [0, split); lp%2==1 → core1 区 [split, P)). 当
        # pages_per_seq 为奇数时 core0 需 B*ceil(pages_per_seq/2)、core1 需
        # B*floor(pages_per_seq/2); 仅靠总量 guard P >= B*pages_per_seq 可能放过
        # core0 溢出到 core1 区 → 同一物理页发给两条 seq. 逐核独立校验.
        need_c0 = B * math.ceil(pages_per_seq / 2)
        need_c1 = B * (pages_per_seq // 2)
        assert need_c0 <= num_local_pages and need_c1 <= num_local_pages, (
            f"dual-core pool too small: per-core need (c0={need_c0}, c1={need_c1}) "
            f"exceeds num_local_pages={num_local_pages} (P={P}, B={B}, "
            f"pages_per_seq={pages_per_seq}); increase num_physical_pages"
        )

        # block_table: 同一 seq 的相邻 logical page 交替落核
        # (core0 拥有 physical pages [0, P/2); core1 拥有 [P/2, P)).
        block_table_host = torch.full((B, pages_per_seq), -1, dtype=torch.int32)
        next_c0, next_c1 = 0, split
        for b in range(B):
            for lp in range(pages_per_seq):
                if lp % 2 == 0:
                    pp = next_c0; next_c0 += 1
                else:
                    pp = next_c1; next_c1 += 1
                block_table_host[b, lp] = pp
        assert next_c0 <= split and next_c1 <= P, (
            "dual-core allocation overflowed its core region"
        )

        # history 摊进 per-core bank + 共享 latent/scale pool
        body_c0 = torch.zeros((num_local_pages, page_size, Di), dtype=torch.float8_e4m3fn)
        body_c1 = torch.zeros((num_local_pages, page_size, Di), dtype=torch.float8_e4m3fn)
        scale_pool = torch.zeros((total_slots,), dtype=torch.float32)
        latent_pool = torch.zeros((total_slots, Rkv), dtype=torch.bfloat16)
        for b in range(B):
            for s in range(S_hist):
                lp_logical = s // page_size
                sip = s % page_size
                pp = int(block_table_host[b, lp_logical])
                slot = pp * page_size + sip
                scale_pool[slot] = index_scale[b, s]
                latent_pool[slot] = latent_kv[b, s]
                bank = body_c0 if owner_of_physical_page(pp, P) == 0 else body_c1
                bank[local_page_of(pp, P), sip] = (
                    index_body[b, s].to(torch.float8_e4m3fn)
                )

        seq_lens = torch.full((B,), S_hist, dtype=torch.int32)

        # #9 gather 输出 buffer —— 预置零的 LocalMem (native mode), 跨 step 复用同
        # 一组 pool 指针.
        # 动机: dsa_latent_k_gather_paged sim 跳过 invalid slot 的写入 (cp > 1 时
        # 非本核 token 不写入 c0/c1; cp = 1 时无 invalid case 但 contract 一致),
        # 依赖 caller 端 buffer 是 zero-init 才能保证 sparse_mqa_partial 的算术 mask
        # 把 invalid 位置乘 0 时仍是有限值. 临时 alloc 的 device buffer 含未初始化
        # 内存, 可能让 invalid 位置出现 NaN/Inf -> 算术 mask × 0 = NaN, 污染整条
        # attn 输出. 解法: 一次性 CPU zeros → from_tensor(kind='native') 进 LocalMem,
        # 后续 forward 直接复用; gather 只覆盖 valid slot, invalid slot 永保 zero.
        Ktop = cfg.Ktop
        zero_K   = torch.zeros(B, Ktop, Rkv, dtype=torch.bfloat16)
        zero_K_T = torch.zeros(B, Rkv, Ktop, dtype=torch.bfloat16)

        def _native_lmem(t: torch.Tensor) -> torch.Tensor:
            return torch.zeus.local_memory.from_tensor(t, kind="native", Tr=1, Tc=1)

        def _weight_lmem(t: torch.Tensor) -> torch.Tensor:
            return torch.zeus.local_memory.from_tensor(
                t.to("zeus"), kind="weight", Tr=1, Tc=1,
            )

        # ── 一次性搬上 device (forward 不再 load) ──
        paged_state = {
            "latent_kv_pool":     latent_pool.to("zeus"),
            "body_cache_c0":      _weight_lmem(body_c0),
            "body_cache_c1":      _weight_lmem(body_c1),
            "scale_cache":        scale_pool.to("zeus"),
            "block_table":        block_table_host.to("zeus"),
            "block_table_host":   block_table_host,
            "seq_lens":           seq_lens.to("zeus"),
            "seq_lens_host":      seq_lens.clone(),     # host 副本, prepare_decode_step 推进
            "page_size":          int(page_size),
            "num_physical_pages": P,
            # K_local gather LocalMem pool (zero-init, 跨 step 复用)
            "k_local_c0":         _native_lmem(zero_K),
            "k_local_c1":         _native_lmem(zero_K),
            "k_local_t_c0":       _native_lmem(zero_K_T),
            "k_local_t_c1":       _native_lmem(zero_K_T),
        }

        # 静态 weights 也在此一次性 pack —— forward 进来即可纯跑 kernel 链.
        if not self._zeus_packed:
            self._pack_zeus()
        return paged_state

    # ── Host per-step prepare (对齐上游 prepare_for_decode) ──────
    def prepare_decode_step(self, paged_state: Dict[str, torch.Tensor]) -> None:
        """Host 侧 per-step KV bookkeeping —— 对齐上游 SGLang ``prepare_for_decode``.

        在 forward 之前, host 一次性完成 (device 链只消费):
          1. 用当前 (pre-increment) seq_lens 从 block_table 算出本步新 token 的物理
             槽位 ``slot_mapping`` (= out_cache_loc, = block_table[b,L/PS]*PS+L%PS,
             L = seq_lens[b]); 越界/未分配页给 -1.
          2. 把 ``seq_lens`` 推进到含新 token 的 INCLUSIVE 长度。
          3. 同步 slot_mapping / seq_lens 给 device。

        forward_zeus 不再 compute_new_slot / advance —— 整条 device 链没有任何 ±1。
        **每个 decode step 必须先调本方法再调** :meth:`forward_zeus`。
        """
        page_size = paged_state["page_size"]
        bt_h = paged_state["block_table_host"]            # [B, max_pages] int32 host
        sl_h = paged_state["seq_lens_host"]               # [B] int32 host, pre-increment
        B = sl_h.shape[0]
        L = sl_h.to(torch.int64)                          # 本步新 token 的逻辑位置
        lp = L // page_size
        sip = L % page_size
        # 越界 (lp >= max_pages, caller 没给 decode 预留页) 由下面 gather 直接抛
        # index error, 不静默; 见 dsa-paged-state-no-decode-headroom.
        pp = bt_h.to(torch.int64).gather(1, lp.view(B, 1)).view(B)   # block_table_host[b, L/PS]
        slot = torch.where(
            pp >= 0, pp * page_size + sip, torch.full_like(pp, -1),
        ).to(torch.int32)
        sl_h_new = (sl_h + 1).to(torch.int32)             # advance 到 inclusive
        paged_state["seq_lens_host"] = sl_h_new
        paged_state["seq_lens"] = sl_h_new.to("zeus")
        paged_state["slot_mapping"] = slot.to("zeus")

    # ── Zeus forward (dual-core paged) ──────────────────────────
    def forward_zeus(
        self,
        hidden_z: torch.Tensor,
        paged_state: Dict[str, torch.Tensor],
        *,
        stop_at_logits: bool = False,
    ) -> torch.Tensor:
        """Zeus DSA decode (dual-core paged chain, 全 device-resident).

        ``hidden_z: [B, H] bf16 (zeus)  ->  [B, H] bf16 (zeus)``. ``paged_state``
        由 :meth:`init_paged_state` 返回, 已 device-resident, 跨 step 复用.

        所有 host 侧准备 (paged_state 取数 + host exec tables) 集中在 device chain
        之前; 其后是 #1→#11 一串纯 kernel 调用, 中途无穿插, 衔接一目了然.

        Paged chain:
          - KV pool 是 [total_slots, ...] 共享池, block_table (logical→physical) +
            seq_lens (per-seq 长度) 完成 page-table 间接寻址
          - kv_a / indexer_k 的 store 直接落到 pool[slot_mapping[b], :]
            (slot_mapping = ``dsa_compute_new_slot(block_table, seq_lens, PS)``)
          - index logits 用两核 ``dsa_index_logits_lmem_addr_table`` 共写一块 buffer
          - top-k 出 logical position → ``dsa_translate_topk_positions`` → phys slot
          - latent K gather 用 ``dsa_latent_k_gather_paged`` (paged pool 直接 gather)
          - 全链路 device-resident, 没有 D2H concat / per-batch loop / host coordination

        seq_lens / slot_mapping 由 host 侧 :meth:`prepare_decode_step` 在每步 forward
        前算好并写进 ``paged_state`` (对齐上游 SGLang: scheduler 在 forward 前
        allocate out_cache_loc 并 advance seq_lens). forward 只消费, 自己不再
        compute_new_slot / advance —— 整条 device 链没有任何 ±1.

        Args:
          hidden_z:         [B, H] bf16 (zeus)
          paged_state:      :meth:`init_paged_state` 返回 + 每步 :meth:`prepare_decode_step`
                            刷新过的 device-resident dict.
          stop_at_logits:   True → 跑到 #6 即返回 index logits ``[B, max_logical_s]``
                            (对拍/调试用).
        """
        cfg = self.cfg
        B = hidden_z.shape[0]

        # ── 取出 device-resident paged state (集中在 device chain 之前) ──
        # seq_lens / slot_mapping 已由 host prepare_decode_step 刷新: seq_lens 是
        # INCLUSIVE (含本步新 token), slot_mapping 是新 token 的物理槽位 (out_cache_loc).
        latent_pool   = paged_state["latent_kv_pool"]
        block_table   = paged_state["block_table"]            # zeus
        block_table_h = paged_state["block_table_host"]       # host, 静态
        seq_lens      = paged_state["seq_lens"]               # zeus, INCLUSIVE
        slot_mapping  = paged_state["slot_mapping"]           # zeus, host 算好
        bc0           = paged_state["body_cache_c0"]
        bc1           = paged_state["body_cache_c1"]
        scale_cache   = paged_state["scale_cache"]
        K_c0          = paged_state["k_local_c0"]
        K_c1          = paged_state["k_local_c1"]
        K_T_c0        = paged_state["k_local_t_c0"]
        K_T_c1        = paged_state["k_local_t_c1"]
        page_size     = paged_state["page_size"]
        P             = paged_state["num_physical_pages"]
        max_logical_s = page_size * block_table.shape[1]

        # ── host per-core exec tables (在 device chain 之前算好) ──
        # 只依赖 block_table + 本步 INCLUSIVE seq_lens + 静态 bank 地址几何, 不依赖
        # 任何 device 中间结果; addr_table entry = group_ptr(local_page) - base_ptr
        # (按 bank 实际地址反查, 非等距), bank 在 init 一次性分配后几何恒定, 每步唯一
        # 变化的输入是 seq_lens (决定每核 owned page 数). 无 ±1: seq_lens_host 已是
        # host prepare_decode_step 推进后的 inclusive 值.
        tables = build_index_k_exec_tables(
            block_table_h, paged_state["seq_lens_host"], P, page_size, bc0, bc1,
        )
        addr_c0 = tables["addr_table"][0].to("zeus")
        addr_c1 = tables["addr_table"][1].to("zeus")
        work_c0 = tables["work_list"][0].to("zeus")
        work_c1 = tables["work_list"][1].to("zeus")

        # ── device-resident DSA chain #1→#11 ──
        # #1  q_a_proj + RMSNorm → q_lora
        q_lora = sgl_kernel_zeus.dsa_q_a_proj_norm(
            hidden_z, self._w_lmem["q_a"], self._w_z["q_a_norm"], eps=cfg.rms_norm_eps,
        )
        # #2  kv_a_proj + norm + STORE TO POOL (latent_pool[slot_mapping[b]] = new_k[b]).
        # slot_mapping 由 host prepare_decode_step 算好 (= block_table[b,L/PS]*PS+L%PS,
        # L = 本步新 token 位置 = inclusive seq_lens - 1); 链里不再 compute_new_slot.
        sgl_kernel_zeus.dsa_kv_a_proj_norm_store(
            hidden_z, self._w_lmem["kv_a"], self._w_z["kv_a_norm"],
            slot_mapping, latent_pool, eps=cfg.rms_norm_eps,
        )
        # #3  q_b_proj + absorb bmm(w_kc) → q_new
        q_new = sgl_kernel_zeus.dsa_q_main_absorb(
            q_lora, self._w_lmem["q_b"], self._w_lmem["w_kc"],
        )
        # #4  indexer Q + per-head weights
        q_body, _q_scale, weights = sgl_kernel_zeus.dsa_indexer_q_weights(
            q_lora, hidden_z, self._w_z["wq_b"], self._w_z["h_di_z"],
            self._w_z["weights_proj"],
            num_index_heads=cfg.I, index_head_dim=cfg.Di,
        )
        # #5  indexer K prep + STORE 进两个 per-core bank (+ 共享 scale_cache)
        sgl_kernel_zeus.dsa_indexer_k_prep_store_dual_core(
            hidden_z, self._w_lmem["wk_idx"],
            self._w_z["k_norm_weight"], self._w_z["k_norm_bias"],
            self._w_lmem["h_di"], slot_mapping,
            bc0, bc1, scale_cache,
            num_physical_pages=P, page_size=page_size, eps=cfg.rms_norm_eps,
        )
        # #6  单个 dual-core kernel: 两核合一次 launch, 内部按 physical-page ownership
        # 写 disjoint 列, 并自行 init -1e30 (无需外部 torch.full); seq_lens INCLUSIVE
        # → kernel 内位置 >= seq_lens[b] 留 -1e30. logits 是纯输出, 由 wrapper 按
        # max_logical_s 内部分配返回 —— 链中途不再 torch.empty.
        logits = sgl_kernel_zeus.dsa_index_logits_lmem_addr_table_dual_core(
            q_body, weights, bc0, bc1, scale_cache,
            addr_c0, addr_c1, work_c0, work_c1,
            seq_lens, page_size=page_size, max_logical_s=max_logical_s,
        )
        if stop_at_logits:
            return logits

        # #7  local top-K (cp=1 → IS global top-K). 输出 logical position
        # in [0, max_logical_s); -inf 位置自然输给 valid 的. positions 省略 →
        # wrapper 内部按 identity (列号 == logical position) 生成, 链中途不再
        # torch.arange.
        _top_lg, top_pos = sgl_kernel_zeus.dsa_local_topk_radix(
            logits, Ktop=cfg.Ktop,
        )
        # #8  logical position → pool 内 physical slot
        phys_slot = sgl_kernel_zeus.dsa_translate_topk_positions(
            top_pos, block_table, page_size=page_size,
        )
        # #9  paged gather. K_local c0/c1/T_c0/T_c1 复用 init 时创建的 zero-init
        # LocalMem pool (kind='native'), 跨 step 不重新分配, invalid slot 恒 0
        # (避免 sparse_mqa 的乘 0 mask 撞上未初始化 NaN). masks 由 sim 写满.
        _, _, _, _, m_c0, m_c1 = sgl_kernel_zeus.dsa_latent_k_gather_paged(
            phys_slot, latent_pool,
            k_local_c0=K_c0, k_local_c1=K_c1,
            k_local_t_c0=K_T_c0, k_local_t_c1=K_T_c1,
        )
        # #10  sparse MQA partial
        po, _pl = sgl_kernel_zeus.dsa_sparse_mqa_partial(
            q_new, K_c0, K_c1, K_T_c0, K_T_c1, m_c0, m_c1, scaling=cfg.scaling,
        )
        # #11  V absorb + o_proj → bf16 [B, H]
        out = sgl_kernel_zeus.dsa_post_o_proj_no_cp(
            po, self._w_z["w_vc"], self._w_z["o_proj"],
        )
        return out

    def forward_zeus_logits(self, hidden_z, paged_state):
        """跑到 #6 返回 index logits [B, max_logical_s] (对拍/调试用)。"""
        return self.forward_zeus(hidden_z, paged_state, stop_at_logits=True)


# ── Stage runner ────────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = (
    "dsa_q_a_proj_norm", "dsa_kv_a_proj_norm_store",
    "dsa_q_main_absorb", "dsa_indexer_q_weights",
    "dsa_indexer_k_prep_store_dual_core",
    "dsa_index_logits_lmem_addr_table_dual_core",
    "dsa_local_topk_radix", "dsa_translate_topk_positions",
    "dsa_latent_k_gather_paged", "dsa_sparse_mqa_partial",
    "dsa_post_o_proj_no_cp",
)


def _run_stage(args) -> bool | None:
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
    ref_out: torch.Tensor | None = None
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

    # ── Zeus (dual-core paged) ────────────────────────────────
    zeus_ok: bool | None = None
    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"  ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
        else:
            try:
                # 退化几何: page_size = S_hist+1 → 每 seq 单 logical page (全落
                # core0), num_physical_pages = 2*B 满足逐核容量. dual-core 算子链
                # 完整跑通 (core1 work_list 空, 不贡献), 与 REF 对拍.
                page_size = args.seqlen + 1
                num_physical_pages = 2 * B
                paged_state = attn.init_paged_state(
                    history, page_size=page_size,
                    num_physical_pages=num_physical_pages,
                )
                # host prepare_for_decode 等价步: 算 slot_mapping + advance seq_lens
                attn.prepare_decode_step(paged_state)
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
                    # DSA chain (12 算子) 累计误差预算, 与原 stage_dsa_decode 一致.
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
