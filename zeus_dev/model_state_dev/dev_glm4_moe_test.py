"""
GLM-4.7 (== GLM-4.5 / 4.6 架构) MoE-FFN 段逐算子 REF-vs-Zeus 对齐

范围（见 zeus_dev/glm4_moe_ffn_dev.md）：
  - 起点：self_attn.o_proj 的输出 [T, H]
  - 终点：MoE Block 输出 [T, H]
  - 单 device，单 layer，不含 TP/EP/DeepEP/NextN

Stage:
  biased_grouped_topk         —— 通用版（sgl_kernel_zeus.biased_grouped_topk 对齐）
  glm4_biased_grouped_topk    —— GLM-4.7 特化快路径，E=160/K=8/G=1/Gk=1，
                                  routed_scaling_factor 已 fuse 进 weights
  moe_align_block_size        —— fused-MoE GEMM 前的 index bookkeeping
  moe_grouped_gemm            —— fused-experts GEMM driver（gemm1 gate_up +
                                  gemm2 down，后者带 mul_routed_weight）
  moe_sum_reduce              —— 收尾：gemm2 输出 [T, topk, H] 按 topk 聚合回
                                  [T, H]；可选把 shared_experts 的 [T, H] 作为
                                  residual fuse 进同一个 kernel（Zeus 扩展）
  moe_block_full              —— 端到端组装：biased_grouped_topk → moe_align →
                                  gemm1 → silu_and_mul → gemm2 →
                                  moe_sum_reduce(+shared)，对齐
                                  Glm4MoeSparseMoeBlock.forward_normal 单卡路径。

用法:
  python zeus_dev/dev_glm4_moe_test.py                          # 跑所有已实现 stage
  python zeus_dev/dev_glm4_moe_test.py --stage biased_grouped_topk
  python zeus_dev/dev_glm4_moe_test.py --stage glm4_biased_grouped_topk
  python zeus_dev/dev_glm4_moe_test.py --stage moe_align_block_size
  python zeus_dev/dev_glm4_moe_test.py --stage moe_grouped_gemm
  python zeus_dev/dev_glm4_moe_test.py --stage moe_sum_reduce
  python zeus_dev/dev_glm4_moe_test.py --stage moe_block_full
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch_zeus  # noqa: F401 — registers zeus backend

# ── SGLang server_args mock（与 demo_zeus_layer_compare.py 同策略） ──
import sglang.srt.server_args
_dummy_args = Mock()
_dummy_args.rl_on_policy_target = None
sglang.srt.server_args.get_global_server_args = lambda *a, **kw: _dummy_args


DEFAULT_CONFIG_PATH = Path(__file__).parent / "config_glm4.json"
REF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_glm4_config(path):
    cfg = json.loads(path.read_text())
    return SimpleNamespace(**cfg)


def compare_tensors(name, ref_out, zeus_out, atol=5e-3, rtol=5e-3):
    a = ref_out.detach().float().cpu()
    b = zeus_out.detach().float().cpu()
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: ref={a.shape} zeus={b.shape}")
        return False
    abs_diff = (a - b).abs()
    close = torch.allclose(a, b, atol=atol, rtol=rtol)
    status = "PASS" if close else "DIFF"
    print(
        f"  [{name}] {status} | max_diff={abs_diff.max().item():.6e} "
        f"mean_diff={abs_diff.mean().item():.6e} shape={list(a.shape)}"
    )
    return close


def compare_ids(name, ref_ids, zeus_ids, allow_permutation=True):
    """比较 topk_ids。allow_permutation=True 时允许同 token 内 id 排列不同
    （因为未 sorted 的 topk 在 score 相等时次序不定）。"""
    a = ref_ids.detach().cpu().to(torch.int64)
    b = zeus_ids.detach().cpu().to(torch.int64)
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: ref={a.shape} zeus={b.shape}")
        return False
    if allow_permutation:
        a_sorted, _ = a.sort(dim=-1)
        b_sorted, _ = b.sort(dim=-1)
        ok = torch.equal(a_sorted, b_sorted)
    else:
        ok = torch.equal(a, b)
    status = "PASS" if ok else "DIFF"
    mismatch = (a != b).sum().item()
    print(
        f"  [{name}] {status} | mismatch_cells={mismatch}/{a.numel()} "
        f"shape={list(a.shape)} (permutation_ok={allow_permutation})"
    )
    return ok


def compare_topk_weights_by_id(name, ref_w, ref_ids, zeus_w, zeus_ids,
                                atol=5e-3, rtol=5e-3):
    """Align both sides' [T, K] weights by their topk_ids (argsort), then
    compare. Needed because REF and Zeus may emit experts in different column
    order — only the set matters."""
    rw = ref_w.detach().float().cpu()
    zw = zeus_w.detach().float().cpu()
    ri = ref_ids.detach().cpu().to(torch.int64)
    zi = zeus_ids.detach().cpu().to(torch.int64)

    rw_aligned = torch.gather(rw, -1, ri.argsort(dim=-1))
    zw_aligned = torch.gather(zw, -1, zi.argsort(dim=-1))

    abs_diff = (rw_aligned - zw_aligned).abs()
    close = torch.allclose(rw_aligned, zw_aligned, atol=atol, rtol=rtol)
    status = "PASS" if close else "DIFF"
    print(
        f"  [{name}] {status} | max_diff={abs_diff.max().item():.6e} "
        f"mean_diff={abs_diff.mean().item():.6e} shape={list(rw_aligned.shape)}"
    )
    return close


# ── Stage: biased_grouped_topk ─────────────────────────────────
def test_biased_grouped_topk(cfg, num_tokens=16, seed=42):
    """对齐 Glm4MoeSparseMoeBlock 里 TopK(use_grouped_topk=True, correction_bias=...).

    REF 侧: `biased_grouped_topk_impl`（纯 torch）作为 golden。CUDA 上的
            `moe_fused_gate` 与它数值一致（除未排序的 id 排列）。
    Zeus 侧: `sgl_kernel_zeus.biased_grouped_topk` —— 签名与 CUDA 版一致。

    输入（与 CUDA call site 对齐，两侧都用 fp32 input）:
      router_logits   [T, E=160] fp32   —— 来自 gate Linear → .to(float32)
      correction_bias [E=160]    fp32   —— Glm4MoeGate.e_score_correction_bias
    输出:
      topk_weights [T, top_k=8] fp32
      topk_ids     [T, top_k=8] int32
    """
    print()
    print("=" * 60)
    print("Stage: biased_grouped_topk (GLM-4.7 MoE gate)")
    print("=" * 60)

    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl
    import sgl_kernel_zeus

    E = cfg.n_routed_experts           # 160
    top_k = cfg.num_experts_per_tok    # 8
    n_group = cfg.n_group              # 1
    topk_group = cfg.topk_group        # 1
    scaling = cfg.routed_scaling_factor
    H = cfg.hidden_size                # 5120

    print(f"  n_routed_experts = {E}")
    print(f"  top_k            = {top_k}")
    print(f"  n_group          = {n_group}  topk_group = {topk_group}")
    print(f"  routed_scaling   = {scaling}  norm_topk_prob = {cfg.norm_topk_prob}")

    torch.manual_seed(seed)
    hidden_states = torch.randn(num_tokens, H, dtype=torch.bfloat16)
    # router_logits 生成时用 bf16（对齐 F.linear(bf16, bf16)），但在喂给 kernel
    # 之前显式 .float()，这是 topk.py:745-754 的 CUDA 调用点的同款行为。这样
    # REF 和 Zeus 都从同一份 fp32 值算 sigmoid，数值可以紧对齐。
    router_logits = torch.randn(num_tokens, E, dtype=torch.bfloat16).float()
    correction_bias = torch.randn(E, dtype=torch.float32) * 0.01

    # ── Reference (REF_DEVICE, torch-native impl) ──
    # apply_routed_scaling_factor_on_output=False 以对齐 forward_normal 里 MoE
    # 之后显式的 `final_hidden_states *= routed_scaling_factor`——我们不把
    # scaling fuse 到 topk 里，保持 stage 边界清晰。
    hs_ref = hidden_states.to(REF_DEVICE)
    rl_ref = router_logits.to(REF_DEVICE)
    cb_ref = correction_bias.to(REF_DEVICE)
    with torch.no_grad():
        w_ref, ids_ref = biased_grouped_topk_impl(
            hidden_states=hs_ref,
            gating_output=rl_ref,
            correction_bias=cb_ref,
            topk=top_k,
            renormalize=cfg.norm_topk_prob,
            num_expert_group=n_group,
            topk_group=topk_group,
            num_fused_shared_experts=0,
            routed_scaling_factor=scaling,
            apply_routed_scaling_factor_on_output=False,
        )
    print(f"  REF topk_weights : {tuple(w_ref.shape)} {w_ref.dtype}")
    print(f"  REF topk_ids     : {tuple(ids_ref.shape)} {ids_ref.dtype}")
    print(f"  REF ids[0]       : {ids_ref[0].tolist()}")
    print(f"  REF weights[0]   : {[round(x, 4) for x in w_ref[0].tolist()]}")

    # ── Zeus ──
    rl_zeus = router_logits.to("zeus")
    cb_zeus = correction_bias.to("zeus")
    with torch.no_grad():
        w_zeus, ids_zeus = sgl_kernel_zeus.biased_grouped_topk(
            rl_zeus,
            cb_zeus,
            num_expert_group=n_group,
            topk_group=topk_group,
            topk=top_k,
            num_fused_shared_experts=0,
            routed_scaling_factor=scaling,
            apply_routed_scaling_factor_on_output=False,
        )
    print(f"  ZEUS topk_weights: {tuple(w_zeus.shape)} {w_zeus.dtype}")
    print(f"  ZEUS topk_ids    : {tuple(ids_zeus.shape)} {ids_zeus.dtype}")
    print(f"  ZEUS ids[0]      : {ids_zeus[0].cpu().tolist()}")
    print(f"  ZEUS weights[0]  : {[round(x, 4) for x in w_zeus[0].cpu().tolist()]}")

    # ── Compare ──
    # ids: REF 未排序（biased_grouped_topk_impl 在 num_fused_shared_experts==0
    # 时传 sorted=False），Zeus 返回降序；所以按集合比较。
    ok_ids = compare_ids(
        "biased_grouped_topk/ids", ids_ref, ids_zeus, allow_permutation=True)
    # weights: 按 id 对齐后比。
    ok_w = compare_topk_weights_by_id(
        "biased_grouped_topk/weights",
        w_ref, ids_ref, w_zeus, ids_zeus,
        atol=5e-5, rtol=5e-5,
    )
    return (ok_ids and ok_w), (w_ref, ids_ref)


# ── Stage: glm4_biased_grouped_topk（GLM-4.7 特化快路径） ──────────
def test_glm4_biased_grouped_topk(cfg, num_tokens=16, seed=42):
    """对齐 `sgl_kernel_zeus.glm4_biased_grouped_topk` —— 特化 E=160/K=8/G=1/Gk=1、
    `routed_scaling_factor` 已 fuse 进 weights。

    语义等价：
        glm4_biased_grouped_topk(input, bias, scale)
      == biased_grouped_topk(input, bias, 1, 1, 8, 0, scale, apply_scale_on_output=True)

    因此 REF 侧必须传 `apply_routed_scaling_factor_on_output=True`，否则
    weights 会差一个 scale 倍。
    """
    print()
    print("=" * 60)
    print("Stage: glm4_biased_grouped_topk (GLM-4.7 specialized)")
    print("=" * 60)

    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl
    import sgl_kernel_zeus

    E = cfg.n_routed_experts           # 160
    top_k = cfg.num_experts_per_tok    # 8
    scaling = cfg.routed_scaling_factor
    H = cfg.hidden_size

    assert E == 160 and top_k == 8, (
        f"glm4_biased_grouped_topk 要求 E=160/K=8，当前 E={E}/K={top_k}"
    )

    print(f"  fixed-shape: E=160 top_k=8  (GLM-4.7 specialized)")
    print(f"  dtype        : input/bias/topk_weights = bf16  (topk_ids = int32)")
    print(f"  routed_scaling   = {scaling}  (fused into weights)")

    torch.manual_seed(seed)
    hidden_states = torch.randn(num_tokens, H, dtype=torch.bfloat16)
    # Zeus 侧签名要 bf16 router_logits + bf16 bias。但 Zeus sim 里 top-k 的
    # compare domain 是 fp32(`choice_scores = fp32(scores) + fp32(bias)`)；
    # 为了让 REF 的 top-k 选择域与 Zeus 一致，我们给 REF 传 **fp32 bias** ——
    # biased_grouped_topk_impl 里 `scores(bf16) + bias(fp32)` 会被提升到 fp32，
    # 和 Zeus 一致。保留一个 bf16 副本喂 Zeus kernel（其 Python API 自动
    # bf16 cast，所以传 fp32 也行，但显式传 bf16 更明确）。
    router_logits = torch.randn(num_tokens, E, dtype=torch.bfloat16)
    correction_bias_fp32 = torch.randn(E, dtype=torch.float32) * 0.01
    correction_bias_bf16 = correction_bias_fp32.to(torch.bfloat16)

    # ── Reference ── 注意 apply_routed_scaling_factor_on_output=True
    hs_ref = hidden_states.to(REF_DEVICE)
    rl_ref = router_logits.to(REF_DEVICE)
    cb_ref = correction_bias_fp32.to(REF_DEVICE)  # fp32 bias → fp32 compare domain
    with torch.no_grad():
        w_ref, ids_ref = biased_grouped_topk_impl(
            hidden_states=hs_ref,
            gating_output=rl_ref,
            correction_bias=cb_ref,
            topk=top_k,
            renormalize=cfg.norm_topk_prob,
            num_expert_group=1,
            topk_group=1,
            num_fused_shared_experts=0,
            routed_scaling_factor=scaling,
            apply_routed_scaling_factor_on_output=True,  # ← key
        )
    print(f"  REF weights dtype: {w_ref.dtype}  ids dtype: {ids_ref.dtype}")
    print(f"  REF ids[0]       : {ids_ref[0].tolist()}")
    print(f"  REF weights[0]   : {[round(x, 4) for x in w_ref[0].tolist()]}")
    print(f"  REF row sum ≈ scale: {w_ref[0].sum().item():.4f} (expect {scaling})")

    # ── Zeus（走特化 kernel，bf16 IO） ──
    rl_zeus = router_logits.to("zeus")
    cb_zeus = correction_bias_bf16.to("zeus")
    with torch.no_grad():
        w_zeus, ids_zeus = sgl_kernel_zeus.glm4_biased_grouped_topk(
            rl_zeus, cb_zeus, routed_scaling_factor=scaling,
        )
    print(f"  ZEUS weights dtype: {w_zeus.dtype}  ids dtype: {ids_zeus.dtype}")
    print(f"  ZEUS ids[0]      : {ids_zeus[0].cpu().tolist()}")
    print(f"  ZEUS weights[0]  : {[round(x, 4) for x in w_zeus[0].float().cpu().tolist()]}")
    print(f"  ZEUS row sum     : {w_zeus[0].float().cpu().sum().item():.4f}")

    # ── Compare ──
    # bf16 weights 精度 ~2-3 位小数，用 bf16 级 tolerance（~1e-2）。
    # compare_topk_weights_by_id 内部 .float().cpu() 后再比，绝对误差阈值
    # 5e-3 对 scale=2.5 的 weights 来说够松（约 0.2% 相对误差）。
    ok_ids = compare_ids(
        "glm4_biased_grouped_topk/ids", ids_ref, ids_zeus, allow_permutation=True)
    ok_w = compare_topk_weights_by_id(
        "glm4_biased_grouped_topk/weights",
        w_ref, ids_ref, w_zeus, ids_zeus,
        atol=5e-3, rtol=5e-3,
    )
    return (ok_ids and ok_w), (w_ref, ids_ref)


# ── Stage: moe_align_block_size ─────────────────────────────────
def test_moe_align_block_size(cfg, num_tokens=16, seed=42):
    """对齐 `sgl_kernel.moe_align_block_size` 的 index 簿记：先通过
    `glm4_biased_grouped_topk` 拿到 topk_ids [T, K]，再喂给 align 得到
    (sorted_token_ids, expert_ids, num_tokens_post_pad)。

    REF 侧：纯 torch 复现（count → pad → prefix → 扫描填 expert_ids/sorted）。
    Zeus 侧：`sgl_kernel_zeus.moe_align_block_size`（含 Python 分配 wrapper
    `moe_align_block_size_alloc` 的同款分配策略）。

    对齐点：
      - num_tokens_post_pad：标量，必须完全相等。
      - expert_ids[0..num_blocks)：bin 索引 -1..real_num_experts-1，逐位相等。
      - sorted_token_ids 按 bin 分段比集合（同 bin 内 token 顺序不保证，
        与 CUDA atomic 语义一致）。
      - 尾部 sentinel 保持为 `numel`（kernel 自带融合 fill，caller 不需要预填）。
    """
    print()
    print("=" * 60)
    print("Stage: moe_align_block_size (GLM-4.7 MoE index bookkeeping)")
    print("=" * 60)

    import sgl_kernel_zeus

    real_num_experts = cfg.n_routed_experts          # 160
    top_k = cfg.num_experts_per_tok                  # 8
    block_size = 64                                   # 与 fused_moe 常用配置对齐

    num_experts = real_num_experts + 1               # +1 for EP-filtered / padding

    print(f"  real_num_experts = {real_num_experts}  top_k = {top_k}")
    print(f"  block_size       = {block_size}  (num_experts passed = {num_experts})")

    torch.manual_seed(seed)
    # 先用 argsort 生成 [T, K] 合法 topk_ids（范围 [0, real_num_experts)）。
    # 实际流程是上游 moe_fused_gate 产出；这里直接合成以保持 stage 独立。
    topk_ids = torch.argsort(
        torch.rand(num_tokens, real_num_experts), dim=1)[:, :top_k]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    numel = topk_ids.numel()

    # ── Reference（pure torch）──
    flat = topk_ids.flatten().to(torch.int64)
    count = torch.zeros(num_experts, dtype=torch.int64)
    for i in range(numel):
        count[int(flat[i].item()) + 1] += 1
    padded = ((count + block_size - 1) // block_size) * block_size
    prefix = torch.zeros(num_experts + 1, dtype=torch.int64)
    for e in range(num_experts):
        prefix[e + 1] = prefix[e] + padded[e]
    total = int(prefix[num_experts].item())

    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_m_blocks = (max_padded + block_size - 1) // block_size

    ref_sorted = torch.full((max_padded,), numel, dtype=torch.int32)
    ref_eid = torch.zeros(max_m_blocks, dtype=torch.int32)
    num_blocks = total // block_size
    for b in range(num_blocks):
        block_start = b * block_size
        e = 0
        while e + 1 <= num_experts and int(prefix[e + 1].item()) <= block_start:
            e += 1
        ref_eid[b] = e - 1
    cur = prefix.clone()
    for i in range(numel):
        bin_id = int(flat[i].item()) + 1
        slot = int(cur[bin_id].item())
        ref_sorted[slot] = i
        cur[bin_id] += 1

    print(f"  REF num_tokens_post_pad = {total}")
    print(f"  REF expert_ids[:num_blocks={num_blocks}] = "
          f"{ref_eid[:num_blocks].tolist()}")

    # ── Zeus ──
    z_tk = topk_ids.to("zeus")
    z_sorted, z_eid, z_ntpp = sgl_kernel_zeus.moe_align_block_size_alloc(
        z_tk, block_size, real_num_experts)

    z_ntpp_val = int(z_ntpp.cpu().item())
    print(f"  ZEUS num_tokens_post_pad = {z_ntpp_val}")
    print(f"  ZEUS expert_ids[:num_blocks] = "
          f"{z_eid.cpu()[:num_blocks].tolist()}")

    # ── Compare ──
    ok_ntpp = (z_ntpp_val == total)
    print(f"  [moe_align/num_tokens_post_pad] "
          f"{'PASS' if ok_ntpp else 'FAIL'} | zeus={z_ntpp_val} ref={total}")

    ok_eid = torch.equal(z_eid.cpu()[:num_blocks], ref_eid[:num_blocks])
    print(f"  [moe_align/expert_ids] "
          f"{'PASS' if ok_eid else 'FAIL'} | blocks={num_blocks}")

    # sorted_token_ids per-bin set compare（bin 内顺序不比）
    z_sorted_cpu = z_sorted.cpu()
    ok_sorted = True
    for e in range(num_experts):
        lo = int(prefix[e].item())
        hi = int(prefix[e + 1].item())
        if lo == hi:
            continue
        z_bin, _ = z_sorted_cpu[lo:hi].sort()
        r_bin, _ = ref_sorted[lo:hi].sort()
        if not torch.equal(z_bin, r_bin):
            ok_sorted = False
            print(f"    bin {e} mismatch [{lo},{hi})\n"
                  f"      ref={r_bin.tolist()}\n      got={z_bin.tolist()}")
            break
    print(f"  [moe_align/sorted_token_ids] "
          f"{'PASS' if ok_sorted else 'FAIL'} | per-bin set compare")

    # sentinel tail
    tail = z_sorted_cpu[total:]
    ok_tail = bool(torch.all(tail == numel).item())
    print(f"  [moe_align/sentinel_tail] "
          f"{'PASS' if ok_tail else 'FAIL'} | tail_len={tail.numel()} "
          f"expected_all={numel}")

    return (ok_ntpp and ok_eid and ok_sorted and ok_tail), None


# ── Stage: moe_grouped_gemm ─────────────────────────────────────
def _ref_moe_grouped_gemm(
    A_bf16, B_bf16,
    sorted_ids_fp32, expert_ids_fp32, num_tokens_post_pad,
    num_valid_tokens, top_k,
    topk_weights_bf16=None, mul_routed_weight=False,
    block_m=64,
):
    """纯 torch 参考 —— 块级循环镜像 `moe_grouped_gemm` 语义。

    语义完全对齐 `sgl_kernel.fused_moe_kernel` 的 v1 scope（bf16 A/B/C、fp32
    accum、c_sorted=False、filter_expert=True）。
    """
    N = B_bf16.shape[1]
    A_fp32 = A_bf16.float()
    B_fp32 = B_bf16.float()
    C_fp32 = torch.zeros(num_valid_tokens, N, dtype=torch.float32)
    EM = int(num_tokens_post_pad)
    num_blocks = (EM + block_m - 1) // block_m
    for b in range(num_blocks):
        e = int(expert_ids_fp32[b].item())
        for j in range(block_m):
            slot = b * block_m + j
            if slot >= EM:
                break
            flat = int(sorted_ids_fp32[slot].item())
            if flat >= num_valid_tokens:
                continue
            if e == -1:
                C_fp32[flat] = 0.0
                continue
            a_row = flat // top_k
            acc = A_fp32[a_row] @ B_fp32[e].T            # [N]
            if mul_routed_weight:
                w = float(topk_weights_bf16[flat].item())
                acc = acc * w
            C_fp32[flat] = acc
    return C_fp32.to(torch.bfloat16)


def test_moe_grouped_gemm(cfg, num_tokens=8, seed=42):
    """对齐 `moe_grouped_gemm` 的 gemm1 + gemm2 双路径 vs 纯 torch 参考。

    为了让 dev 脚本保持秒级响应（GLM-4.7 真实 w13 ≈ 4.8 GB / w2 ≈ 2.4 GB），
    stage 用 **proxy weight shape** 配 GLM 拓扑（topk=8 → 用户传 `num_tokens=8`
    依然实跑 T=8）。N 必须被 `CORE_NUM * BLOCK_N = 256` 整除，所以最小可用
    H=256 / 2·mI=256。

    sub-test:
      - gemm1（w13 / gate_up）：A=[T, H], B=[E, 2·mI, H], C=[T·topk, 2·mI],
        top_k=原始 topk, mul_routed_weight=False
      - gemm2（w2 / down）    ：A=[T·topk, mI], B=[E, H, mI], C=[T·topk, H],
        top_k=1, mul_routed_weight=True（路由权重在 accumulator 上融合）

    对齐方式：Zeus 输出 vs `_ref_moe_grouped_gemm`（纯 torch 块级循环）。
    """
    print()
    print("=" * 60)
    print("Stage: moe_grouped_gemm (GLM-4.7 fused-experts GEMM driver)")
    print("=" * 60)

    import sgl_kernel_zeus

    # ── Proxy shape ──
    T = num_tokens
    H = 256                      # proxy for hidden_size (>= CORE_NUM*BLOCK_N=256)
    mI = 128                     # proxy for moe_intermediate_size (→ 2·mI=256)
    E = 8                        # proxy for n_routed_experts（用小 E 省 CPU ref 时间）
    top_k = 4                    # proxy for num_experts_per_tok
    block_size = sgl_kernel_zeus.MOE_GROUPED_GEMM_BLOCK_M   # 64
    scaling = cfg.routed_scaling_factor

    print(f"  proxy shape: T={T}  H={H}  mI={mI}  E={E}  top_k={top_k}")
    print(f"  block_size = {block_size}  (BLOCK_M / BLOCK_N / CORE_NUM = 64/128/2)")
    print(f"  NOTE: 真实 GLM-4.7 权重过大（w13≈4.8GB），dev 脚本用 proxy shape；")
    print(f"        GLM-4.7 真实 shape 的端到端 align 走 sgl-kernel-zeus/tests/test_moe_grouped_gemm.py。")

    torch.manual_seed(seed)

    # ── 合成 topk_ids / topk_weights（跳过 gate linear，用 randperm 生成合法路由） ──
    topk_ids_list = [torch.randperm(E)[:top_k] for _ in range(T)]
    topk_ids = torch.stack(topk_ids_list).to(torch.int32).contiguous()   # [T, top_k]
    raw_w = torch.rand(T, top_k, dtype=torch.float32) + 0.1
    topk_weights_fp32 = raw_w / raw_w.sum(dim=-1, keepdim=True) * scaling
    topk_weights_bf16 = topk_weights_fp32.to(torch.bfloat16)             # [T, top_k] bf16

    # flat 一维视图 —— 和 moe_grouped_gemm 对 topk_weights 的索引方式对齐
    topk_weights_flat_bf16 = topk_weights_bf16.flatten().contiguous()    # [T*top_k]

    # ── moe_align_block_size（Zeus） ──
    z_topk_ids = topk_ids.to("zeus")
    sorted_ids_z, expert_ids_z, num_post_z = sgl_kernel_zeus.moe_align_block_size_alloc(
        z_topk_ids, block_size, E,
    )
    sorted_ids_cpu = sorted_ids_z.cpu()
    expert_ids_cpu = expert_ids_z.cpu()
    num_post = int(num_post_z.cpu().item())
    num_valid_tokens = T * top_k
    print(f"  moe_align: num_tokens_post_pad = {num_post} "
          f"({num_post // block_size} blocks)")

    # ── GEMM-1 路径（gate_up / w13） ──
    print()
    print("  --- gemm1 (w13 / gate_up) ---")
    hidden_states = torch.randn(T, H, dtype=torch.bfloat16) * 0.1
    w13 = torch.randn(E, 2 * mI, H, dtype=torch.bfloat16) * 0.1
    C1_z = torch.empty(T * top_k, 2 * mI, dtype=torch.bfloat16, device="zeus")

    with torch.no_grad():
        sgl_kernel_zeus.moe_grouped_gemm(
            hidden_states.to("zeus"),
            w13.to("zeus"),
            C1_z,
            sorted_ids_z, expert_ids_z, num_post_z,
            num_valid_tokens=num_valid_tokens,
            top_k=top_k,
        )
    C1_ref = _ref_moe_grouped_gemm(
        hidden_states, w13,
        sorted_ids_cpu, expert_ids_cpu, num_post,
        num_valid_tokens=num_valid_tokens, top_k=top_k,
        block_m=block_size,
    )
    ok1 = compare_tensors(
        "moe_grouped_gemm/gemm1", C1_ref, C1_z.cpu(),
        atol=5e-2, rtol=5e-2,
    )

    # ── GEMM-2 路径（down / w2 + mul_routed_weight） ──
    print()
    print("  --- gemm2 (w2 / down, mul_routed_weight=True) ---")
    A2 = torch.randn(T * top_k, mI, dtype=torch.bfloat16) * 0.1
    w2 = torch.randn(E, H, mI, dtype=torch.bfloat16) * 0.1
    C2_z = torch.empty(T * top_k, H, dtype=torch.bfloat16, device="zeus")

    with torch.no_grad():
        sgl_kernel_zeus.moe_grouped_gemm(
            A2.to("zeus"),
            w2.to("zeus"),
            C2_z,
            sorted_ids_z, expert_ids_z, num_post_z,
            num_valid_tokens=num_valid_tokens,
            top_k=1,
            topk_weights=topk_weights_flat_bf16.to("zeus"),
            mul_routed_weight=True,
        )
    C2_ref = _ref_moe_grouped_gemm(
        A2, w2,
        sorted_ids_cpu, expert_ids_cpu, num_post,
        num_valid_tokens=num_valid_tokens, top_k=1,
        topk_weights_bf16=topk_weights_flat_bf16, mul_routed_weight=True,
        block_m=block_size,
    )
    ok2 = compare_tensors(
        "moe_grouped_gemm/gemm2", C2_ref, C2_z.cpu(),
        atol=5e-2, rtol=5e-2,
    )

    return (ok1 and ok2), None


# ── Stage: moe_sum_reduce ───────────────────────────────────────
def _ref_moe_sum_reduce(x_bf16, scale, shared_bf16=None):
    """fp32 reference: upcast → sum(topk) → *scale → (+shared in fp32) → bf16 RNE。

    对齐 `sgl_kernel_zeus.moe_sum_reduce` 的语义：
        output[t, h] = scale * sum_k input[t, k, h]
                     + (shared_output[t, h] if provided else 0)

    注意：shared residual **不乘 scale**，在 fp32 累加域上加，再整体 RNE
    到 bf16（比独立 bf16+bf16 add 少一次舍入）。
    """
    acc = x_bf16.float().sum(dim=1) * scale                  # [T, H] fp32
    if shared_bf16 is not None:
        acc = acc + shared_bf16.float()
    return acc.to(torch.bfloat16)


def test_moe_sum_reduce(cfg, num_tokens=16, seed=42):
    """对齐 `sgl_kernel_zeus.moe_sum_reduce` —— fused-MoE FFN 流水线的**收尾**。

    gemm2 的输出以 `[T*topk, H]` 形态散落（每个 token 的 topk 份 expert 结果
    分散在 topk 行），`moe_sum_reduce` 沿 topk 轴求和，回到 `[T, H]`。
    Zeus 扩展：shared_experts 的 `[T, H]` 可作为 residual 直接 fuse 进同一个
    kernel（省一次 DRAM 往返 + 一次 kernel launch，精度更高）。

    **routed scaling 归属**（重要）：GLM-4.7 Zeus 路径下
    `glm4_biased_grouped_topk` 以 `apply_routed_scaling_factor_on_output=True`
    把 `routed_scaling_factor` 折进 `topk_weights`，`moe_grouped_gemm` 的
    gemm2 再以 `mul_routed_weight=True` 把该权重乘进 accumulator。所以
    **到 moe_sum_reduce 这一步典型调用是 `scale=1.0`**，它只负责求和。
    本 stage 仍保留 `scale != 1.0` 的 sub-test，用以校验签名。

    sub-test:
      - plain-sum：`scale ∈ {1.0, 2.5}`，两组 GLM-4.7 对齐形态。
      - shared-fuse：同上两组 + shared_output residual。
    """
    print()
    print("=" * 60)
    print("Stage: moe_sum_reduce (GLM-4.7 MoE-FFN tail reduction + shared fuse)")
    print("=" * 60)

    import sgl_kernel_zeus

    T = num_tokens
    top_k = cfg.num_experts_per_tok   # 8
    H = cfg.hidden_size               # 5120
    scale_fuse = 1.0                  # 典型调用（routed scaling 已在 gemm2 融进）
    scale_sig  = cfg.routed_scaling_factor  # 2.5, 只作为签名校验

    print(f"  shape: T={T}  topk={top_k}  H={H}")
    print(f"  scale (typical)   = {scale_fuse}  (routed scaling 已被 gemm2 融进 accumulator)")
    print(f"  scale (signature) = {scale_sig}   (留口以对齐 CUDA moe_sum_reduce 的签名)")
    print(f"  core / block      = CORE_NUM={sgl_kernel_zeus.MOE_SUM_REDUCE_CORE_NUM} "
          f"BLOCK_H={sgl_kernel_zeus.MOE_SUM_REDUCE_BLOCK_H}")

    torch.manual_seed(seed)
    x      = torch.randn(T, top_k, H, dtype=torch.bfloat16) * 0.05      # gemm2 输出模拟
    shared = torch.randn(T,        H, dtype=torch.bfloat16) * 0.05      # shared_experts 输出

    results = []

    # ── plain-sum path ────────────────────────────────────────
    for scale in (scale_fuse, scale_sig):
        print()
        print(f"  --- plain sum (scale={scale}) ---")
        y_z = torch.zeros(T, H, dtype=torch.bfloat16, device="zeus")
        with torch.no_grad():
            sgl_kernel_zeus.moe_sum_reduce(
                input=x.to("zeus"),
                output=y_z,
                routed_scaling_factor=scale,
            )
        y_ref = _ref_moe_sum_reduce(x, scale)
        ok = compare_tensors(
            f"moe_sum_reduce/plain@scale={scale}",
            y_ref, y_z.cpu(), atol=2e-2, rtol=2e-2,
        )
        results.append(ok)

    # ── shared-fuse path ──────────────────────────────────────
    for scale in (scale_fuse, scale_sig):
        print()
        print(f"  --- +shared fuse (scale={scale}) ---")
        y_z = torch.zeros(T, H, dtype=torch.bfloat16, device="zeus")
        with torch.no_grad():
            sgl_kernel_zeus.moe_sum_reduce(
                input=x.to("zeus"),
                output=y_z,
                shared_output=shared.to("zeus"),
                routed_scaling_factor=scale,
            )
        y_ref = _ref_moe_sum_reduce(x, scale, shared)
        ok = compare_tensors(
            f"moe_sum_reduce/+shared@scale={scale}",
            y_ref, y_z.cpu(), atol=2e-2, rtol=2e-2,
        )
        results.append(ok)

    return all(results), None


# ── Stage: moe_block_full ───────────────────────────────────────
def _ref_moe_core(
    x_bf16, w13_bf16, w2_bf16,
    topk_weights_fp32, topk_ids_int32,
    mI,
):
    """Pure-torch MoE core: per-token per-expert GEMM1 → SiluAndMul → GEMM2,
    accumulated with routing weights in fp32. Returns bf16 [T, H].

    Mirrors `FusedMoE.forward` with `fuse_routed_scaling_factor_in_topk=True`
    semantics: `topk_weights_fp32` is expected to already carry the routed
    scaling factor (ie biased_grouped_topk_impl was called with
    `apply_routed_scaling_factor_on_output=True`), so we do NOT multiply by
    `routed_scaling_factor` again outside.
    """
    T, H = x_bf16.shape
    top_k = topk_ids_int32.shape[1]
    E = w13_bf16.shape[0]
    device = x_bf16.device

    final = torch.zeros(T, H, dtype=torch.float32, device=device)
    x_fp32 = x_bf16.float()
    w13_fp32 = w13_bf16.float()
    w2_fp32 = w2_bf16.float()
    for t in range(T):
        for k in range(top_k):
            e = int(topk_ids_int32[t, k].item())
            w_tk = float(topk_weights_fp32[t, k].item())
            # gemm1: [H] @ [2*mI, H].T → [2*mI]
            a = x_fp32[t] @ w13_fp32[e].T
            # silu_and_mul: [2*mI] → [mI]
            a = torch.nn.functional.silu(a[:mI]) * a[mI:]
            # gemm2: [mI] @ [H, mI].T → [H], then scale by route weight
            a = a @ w2_fp32[e].T
            final[t] += w_tk * a
    return final.to(torch.bfloat16)


def test_moe_block_full(cfg, num_tokens=16, seed=42):
    """端到端组装：对齐 `Glm4MoeSparseMoeBlock.forward_normal`（单卡、不含 a2a /
    不含 fused_shared_experts）。

    设计决定（复用前序 stage 的 proxy-shape 策略）：
      - 真实 GLM-4.7 权重过大（w13≈4.8GB），本 stage 使用 proxy shape
        (T=16, H=256, mI=128, E=8, top_k=4)。GLM-4.7 真实 shape 的 kernel
        层对齐分别由 sgl-kernel-zeus/tests/ 的各 kernel test 承担。
      - `gate` Linear 与 `shared_experts` MLP 的 GEMM 不是本 stage 的测试对象
        （它们在 qwen demo / 早期 stage 已验证），我们在 **CPU/CUDA 上用纯
        torch 预计算** `router_logits` 和 `shared_output`，两条路径共享同一份
        输入，专心比较 **MoE 核心 6-kernel pipeline** 的组装是否对齐。
      - routed_scaling_factor 融合策略与生产 Zeus 路径一致：
        `biased_grouped_topk(apply_scale_on_output=True)` 把 scale 乘进 weights，
        `moe_grouped_gemm(gemm2, mul_routed_weight=True)` 把 weights 乘进
        accumulator，`moe_sum_reduce(scale=1.0, shared_output=…)` 只做 topk
        求和并融合 shared residual。

    REF:  biased_grouped_topk_impl → per-token per-expert pytorch loop →
          + shared_output
    Zeus: biased_grouped_topk → moe_align_block_size_alloc →
          moe_grouped_gemm(gemm1) → silu_and_mul → moe_grouped_gemm(gemm2) →
          moe_sum_reduce(shared_output=…)
    """
    print()
    print("=" * 60)
    print("Stage: moe_block_full (GLM-4.7 MoE-FFN end-to-end)")
    print("=" * 60)

    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl
    import sgl_kernel_zeus

    T = num_tokens
    H = 256
    mI = 128
    E = 8
    top_k = 4
    n_group = 1
    topk_group = 1
    scaling = cfg.routed_scaling_factor
    block_size = sgl_kernel_zeus.MOE_GROUPED_GEMM_BLOCK_M

    print(f"  proxy shape: T={T}  H={H}  mI={mI}  E={E}  top_k={top_k}")
    print(f"  routed_scaling_factor = {scaling}  block_size = {block_size}")
    print(f"  n_group = {n_group}  topk_group = {topk_group}  "
          f"norm_topk_prob = {cfg.norm_topk_prob}")
    print(f"  NOTE: proxy shape — 真实 GLM-4.7 权重无法在 CPU ref 跑；")
    print(f"        gate Linear / shared_experts MLP 走 REF_DEVICE 预计算，")
    print(f"        两条路径共享同一份 router_logits / shared_output。")

    torch.manual_seed(seed)

    # ── 合成输入与权重（仅 MoE 核心需要的部分在 zeus 上运行） ──
    x_bf16 = torch.randn(T, H, dtype=torch.bfloat16) * 0.1
    gate_w_bf16 = torch.randn(E, H, dtype=torch.bfloat16) * 0.1
    corr_bias_fp32 = torch.randn(E, dtype=torch.float32) * 0.01
    # shared-experts MLP 权重（仅用 torch ref 算）
    sh_gu_bf16 = torch.randn(2 * mI, H, dtype=torch.bfloat16) * 0.1
    sh_dp_bf16 = torch.randn(H, mI, dtype=torch.bfloat16) * 0.1
    # routed-experts 权重（MoE 核心，两侧都要用）
    w13_bf16 = torch.randn(E, 2 * mI, H, dtype=torch.bfloat16) * 0.1
    w2_bf16 = torch.randn(E, H, mI, dtype=torch.bfloat16) * 0.1

    # ── REF_DEVICE 预计算 router_logits 和 shared_output（共享输入） ──
    x_ref = x_bf16.to(REF_DEVICE)
    gate_w_ref = gate_w_bf16.to(REF_DEVICE)
    sh_gu_ref = sh_gu_bf16.to(REF_DEVICE)
    sh_dp_ref = sh_dp_bf16.to(REF_DEVICE)
    corr_bias_ref = corr_bias_fp32.to(REF_DEVICE)

    with torch.no_grad():
        # gate Linear: bf16 × bf16 → bf16, 进 topk 前 .float()
        router_logits_bf16 = torch.nn.functional.linear(x_ref, gate_w_ref)
        router_logits_fp32 = router_logits_bf16.float()
        # shared_experts: MLP bf16 pipeline (gate_up → silu_and_mul → down)
        sh_gu = torch.nn.functional.linear(x_ref, sh_gu_ref)              # [T, 2*mI]
        sh_silu = (
            torch.nn.functional.silu(sh_gu[:, :mI].float())
            * sh_gu[:, mI:].float()
        ).to(torch.bfloat16)
        shared_out_bf16 = torch.nn.functional.linear(sh_silu, sh_dp_ref)  # [T, H] bf16

    # ── REF 路径：biased_grouped_topk_impl → per-token MoE 循环 → + shared ──
    w13_ref = w13_bf16.to(REF_DEVICE)
    w2_ref = w2_bf16.to(REF_DEVICE)
    with torch.no_grad():
        w_ref, ids_ref = biased_grouped_topk_impl(
            hidden_states=x_ref,
            gating_output=router_logits_fp32,
            correction_bias=corr_bias_ref,
            topk=top_k,
            renormalize=cfg.norm_topk_prob,
            num_expert_group=n_group,
            topk_group=topk_group,
            num_fused_shared_experts=0,
            routed_scaling_factor=scaling,
            apply_routed_scaling_factor_on_output=True,  # scale fused into weights
        )
        # MoE core: pure torch
        moe_core_ref = _ref_moe_core(
            x_ref.cpu(), w13_ref.cpu(), w2_ref.cpu(),
            w_ref.cpu(), ids_ref.cpu(), mI=mI,
        )
        # residual add in fp32 (mirrors kernel's fused fp32 add + single RNE)
        final_ref = (moe_core_ref.float() + shared_out_bf16.float().cpu()).to(torch.bfloat16)

    print(f"  REF final:  shape={tuple(final_ref.shape)} dtype={final_ref.dtype}")
    print(f"  REF final[0, :6] = "
          f"{[round(v, 4) for v in final_ref[0, :6].float().tolist()]}")

    # ── Zeus 路径 ──
    # router_logits 在 REF 端已是 fp32，直接 .to('zeus') 喂 biased_grouped_topk。
    # shared_output 是 bf16，直接作为 residual 喂 moe_sum_reduce。
    router_logits_z = router_logits_fp32.to("zeus")
    corr_bias_z = corr_bias_fp32.to("zeus")
    shared_out_z = shared_out_bf16.to("zeus")
    x_z = x_bf16.to("zeus")
    w13_z = w13_bf16.to("zeus")
    w2_z = w2_bf16.to("zeus")

    with torch.no_grad():
        # 1) biased_grouped_topk（scale fused 进 weights）
        w_z, ids_z = sgl_kernel_zeus.biased_grouped_topk(
            router_logits_z, corr_bias_z,
            num_expert_group=n_group,
            topk_group=topk_group,
            topk=top_k,
            num_fused_shared_experts=0,
            routed_scaling_factor=scaling,
            apply_routed_scaling_factor_on_output=True,
        )
        # 2) moe_align
        sorted_ids_z, expert_ids_z, num_post_z = (
            sgl_kernel_zeus.moe_align_block_size_alloc(
                ids_z, block_size, E,
            )
        )
        num_valid_tokens = T * top_k
        num_post_val = int(num_post_z.cpu().item())
        print(f"  Zeus moe_align: num_tokens_post_pad = {num_post_val} "
              f"({num_post_val // block_size} blocks)")

        # 3) gemm1: [T, H] → [T*top_k, 2*mI]
        C1_z = torch.empty(T * top_k, 2 * mI, dtype=torch.bfloat16, device="zeus")
        sgl_kernel_zeus.moe_grouped_gemm(
            x_z, w13_z, C1_z,
            sorted_ids_z, expert_ids_z, num_post_z,
            num_valid_tokens=num_valid_tokens,
            top_k=top_k,
        )

        # 4) silu_and_mul: [T*top_k, 2*mI] → [T*top_k, mI]
        C1_silu_z = torch.empty(T * top_k, mI, dtype=torch.bfloat16, device="zeus")
        sgl_kernel_zeus.silu_and_mul(C1_z, C1_silu_z)

        # 5) gemm2 (mul_routed_weight=True): [T*top_k, mI] → [T*top_k, H]
        #    topk_weights 需要扁平 bf16
        w_z_flat_bf16 = w_z.to(torch.bfloat16).flatten().contiguous()
        C2_z = torch.empty(T * top_k, H, dtype=torch.bfloat16, device="zeus")
        sgl_kernel_zeus.moe_grouped_gemm(
            C1_silu_z, w2_z, C2_z,
            sorted_ids_z, expert_ids_z, num_post_z,
            num_valid_tokens=num_valid_tokens,
            top_k=1,
            topk_weights=w_z_flat_bf16,
            mul_routed_weight=True,
        )

        # 6) moe_sum_reduce (+shared residual; scale=1.0 因为 scale 已融进 weights)
        final_z = torch.empty(T, H, dtype=torch.bfloat16, device="zeus")
        sgl_kernel_zeus.moe_sum_reduce(
            input=C2_z.view(T, top_k, H),
            output=final_z,
            shared_output=shared_out_z,
            routed_scaling_factor=1.0,
        )

    print(f"  ZEUS final: shape={tuple(final_z.shape)} dtype={final_z.dtype}")
    print(f"  ZEUS final[0, :6] = "
          f"{[round(v, 4) for v in final_z[0, :6].float().cpu().tolist()]}")

    # ── Compare ──
    # Sanity check: topk_ids 应当集合一致（否则后面 MoE 核心路由到不同专家，
    # 数值一定对不上，先暴露）。
    ok_ids = compare_ids(
        "moe_block_full/topk_ids",
        ids_ref, ids_z, allow_permutation=True,
    )

    # 端到端 bf16 输出。bf16 多级累加 + scale=2.5 放大，容忍 ~5e-2 级别。
    ok_final = compare_tensors(
        "moe_block_full/final",
        final_ref, final_z.cpu(),
        atol=5e-2, rtol=5e-2,
    )
    return (ok_ids and ok_final), None


# ── Dispatch ───────────────────────────────────────────────────
STAGES = {
    "biased_grouped_topk": test_biased_grouped_topk,
    "glm4_biased_grouped_topk": test_glm4_biased_grouped_topk,
    "moe_align_block_size": test_moe_align_block_size,
    "moe_grouped_gemm": test_moe_grouped_gemm,
    "moe_sum_reduce": test_moe_sum_reduce,
    "moe_block_full": test_moe_block_full,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_glm4_config(args.config)
    print(f"Config path: {args.config}")
    print(f"GLM-4.7 config: H={cfg.hidden_size}  E={cfg.n_routed_experts}"
          f"  top_k={cfg.num_experts_per_tok}  mI={cfg.moe_intermediate_size}")
    print(f"Reference device: {REF_DEVICE}")

    results = {}
    for name, fn in STAGES.items():
        if args.stage not in (name, "all"):
            continue
        ok, _ = fn(cfg, num_tokens=args.num_tokens, seed=args.seed)
        results[name] = ok

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok is True else "FAIL" if ok is False else "SKIP"
        print(f"  {name:30s} : {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
