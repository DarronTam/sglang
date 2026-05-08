"""
mHC (Manifold-Constrained Hyper-Connections) 逐算子 REF-vs-Zeus 对齐

范围（见 zeus_dev/mhc_dev.md）：
  - 起点：multi-stream residual [T, mhc=4, H] 进入 block
  - 终点：mhc_post 之后的新 multi-stream residual [T, mhc=4, H]
  - 单 device，单 layer，FFN-side 一轮 wrap，attn-side 结构对称（本脚本暂不覆盖）
  - 不含训练反向；N 硬锁 4；bf16 IO + fp32 参数

REF 来源：
  `ref_tile_kernels_torch/mhc.py`（TileKernels 官方 pure-torch REF，85 行 / 7 函数）

Stage：
  mhc_expand             —— [T,H] → [T,mhc,H]（sequence 入口，非热路径）
  mhc_pre_norm_fn        —— RMSNorm(flatten(hc,H)) + F.linear(fn fp32) → mixes[T,24] fp32
  mhc_pre_split_mixes    —— mixes*scale + base → sigmoid → (pre[T,4,1], post[T,4,1], comb[T,4,4])
  mhc_sinkhorn           —— comb logits → doubly-stochastic via Sinkhorn-Knopp
  mhc_pre_apply_mix      —— (residual × pre).sum(-2).bfloat16() → [T, H]
  mhc_post               —— x + residual/comb → [T, mhc, H]，fp32 accum + 单次 RNE
  mhc_block_ffn_full     —— 端到端组装 (expand → pre 四步 → identity sublayer → post)

用法：
  python zeus_dev/dev_mhc_test.py                          # 跑所有 stage
  python zeus_dev/dev_mhc_test.py --stage mhc_sinkhorn
  python zeus_dev/dev_mhc_test.py --stage mhc_block_ffn_full
"""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch


# ── REF import (bypass ref_tile_kernels_torch/__init__.py's broader imports) ──
_REF_MHC_PATH = Path(__file__).parent / "ref_tile_kernels_torch" / "mhc.py"
_spec = importlib.util.spec_from_file_location("_ref_mhc", _REF_MHC_PATH)
ref_mhc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref_mhc)
# ref_mhc 提供：expand_to_mhc_ref, sinkhorn_normalize_ref,
#               mhc_pre_split_mixes_ref, mhc_pre_apply_mix_ref,
#               mhc_post_ref, mhc_pre_norm_fn_ref, mhc_head_compute_mix_ref


# ── Zeus runtime (optional: kernels land incrementally) ──
try:
    import torch_zeus  # noqa: F401
    import sgl_kernel_zeus

    # SGLang server_args mock（与 dev_glm4_moe_test.py 同策略）
    import sglang.srt.server_args
    _dummy = Mock()
    _dummy.rl_on_policy_target = None
    sglang.srt.server_args.get_global_server_args = lambda *a, **kw: _dummy

    ZEUS_AVAILABLE = True
except Exception as e:
    sgl_kernel_zeus = None
    ZEUS_AVAILABLE = False
    _ZEUS_IMPORT_ERR = repr(e)


CONFIG_PATH = Path(__file__).parent / "config.json"
REF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config():
    """Load config.json and return SimpleNamespace. Works for Glm5Next (mhc_*)
    and DeepSeek-V4-style (hc_*) — we read both prefixes and normalize."""
    cfg = json.loads(CONFIG_PATH.read_text())
    # Normalize hc_* → mhc_* if present (so the script works against either config)
    alias = {
        "hc_mult": "mhc_num_residual_streams",
        "hc_sinkhorn_iters": "mhc_sinkhorn_iters",
        "hc_eps": "mhc_sinkhorn_eps",
    }
    for src, dst in alias.items():
        if src in cfg and dst not in cfg:
            cfg[dst] = cfg[src]
    cfg.setdefault("mhc_num_residual_streams", 4)
    cfg.setdefault("mhc_sinkhorn_iters", 20)
    cfg.setdefault("mhc_sinkhorn_eps", 1e-6)
    cfg.setdefault("mhc_tau", None)
    cfg.setdefault("hres_vwnstyle", False)
    return SimpleNamespace(**cfg)


# ── Helpers ────────────────────────────────────────────────────
def compare_tensors(name, ref, zeus, atol=5e-3, rtol=5e-3):
    a = ref.detach().float().cpu()
    b = zeus.detach().float().cpu()
    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: ref={a.shape} zeus={b.shape}")
        return False
    diff = (a - b).abs()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    status = "PASS" if ok else "DIFF"
    print(
        f"  [{name}] {status} | max_diff={diff.max().item():.3e} "
        f"mean_diff={diff.mean().item():.3e} shape={list(a.shape)}"
    )
    return ok


def zeus_kernel_ready(op_name):
    """Return the Zeus kernel callable if registered, else None."""
    if not ZEUS_AVAILABLE:
        return None
    return getattr(sgl_kernel_zeus, op_name, None)


def print_zeus_skip(op_name, reason=None):
    if not ZEUS_AVAILABLE:
        print(f"  [zeus] SKIP: torch_zeus/sgl_kernel_zeus import failed "
              f"({_ZEUS_IMPORT_ERR})")
    else:
        msg = reason or f"sgl_kernel_zeus.{op_name} not registered yet"
        print(f"  [zeus] SKIP: {msg}")


def zeus_skip_result(ref_ok=True):
    """Return None for Summary SKIP unless the REF-side sanity check failed."""
    return None if ref_ok else False


def assert_doubly_stochastic(name, comb, tol=1e-4):
    """comb: [..., N, N] — check row/col sums ≈ 1."""
    row_err = (comb.sum(-1) - 1.0).abs().max().item()
    col_err = (comb.sum(-2) - 1.0).abs().max().item()
    ok = row_err < tol and col_err < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{name}/doubly_stochastic] {status} | "
          f"max|row_sum-1|={row_err:.3e} max|col_sum-1|={col_err:.3e} "
          f"tol={tol:.1e}")
    return ok


# ── Stage: mhc_expand ──────────────────────────────────────────
def test_mhc_expand(cfg, num_tokens=16, seed=42):
    """[T, H] → [T, mhc, H]. Called once at sequence entry (embedding → streams)."""
    print()
    print("=" * 60)
    print("Stage: mhc_expand (embedding → multi-stream residual)")
    print("=" * 60)

    T = num_tokens
    H = 256
    mhc = cfg.mhc_num_residual_streams

    print(f"  shape: T={T}  H={H}  mhc={mhc}")

    torch.manual_seed(seed)
    hidden = torch.randn(T, H, dtype=torch.bfloat16)

    # REF
    out_ref = ref_mhc.expand_to_mhc_ref(hidden, mhc)
    assert out_ref.shape == (T, mhc, H), out_ref.shape
    assert out_ref.is_contiguous()
    # Invariant: all mhc streams start identical (pure broadcast)
    for k in range(mhc):
        assert torch.equal(out_ref[:, k, :], hidden), "expand broke broadcast"
    print(f"  REF out: shape={tuple(out_ref.shape)} dtype={out_ref.dtype} "
          f"(all {mhc} streams == source hidden)")

    # Zeus
    kernel = zeus_kernel_ready("mhc_expand")
    if kernel is None:
        print_zeus_skip("mhc_expand")
        return zeus_skip_result(), None

    out_z = kernel(hidden.to("zeus"), mhc)
    ok = compare_tensors("mhc_expand", out_ref, out_z)
    return ok, None


# ── Stage: mhc_pre_norm_fn ─────────────────────────────────────
def test_mhc_pre_norm_fn(cfg, num_tokens=16, seed=42):
    """RMSNorm(residual.flatten(2,3)) + F.linear(fn fp32) → mixes [T, mix_hc] fp32.

    mix_hc = mhc * (2 + mhc) = 4 * 6 = 24.
    fn shape: [mix_hc, mhc * H].  Weights are **fp32**, residual is bf16.
    """
    print()
    print("=" * 60)
    print("Stage: mhc_pre_norm_fn (RMSNorm + fp32 Linear)")
    print("=" * 60)

    T = num_tokens
    H = 256
    mhc = cfg.mhc_num_residual_streams
    mix_hc = mhc * (2 + mhc)

    print(f"  shape: T={T}  H={H}  mhc={mhc}  mix_hc={mix_hc}")
    print(f"  dtype: residual=bf16  fn=fp32  norm_weight=fp32 (optional)  "
          f"mixes=fp32")

    torch.manual_seed(seed)
    residual = torch.randn(T, mhc, H, dtype=torch.bfloat16) * 0.1
    fn = torch.randn(mix_hc, mhc * H, dtype=torch.float32) * 0.02

    # REF: no norm_weight (set to None, matches TileKernels' "optional" path)
    mixes_ref = ref_mhc.mhc_pre_norm_fn_ref(
        residual.unsqueeze(0),       # [B=1, T, mhc, H]
        fn,
        mhc_norm_weight=None,
        mhc_norm_eps=1e-6,
    )
    # REF returns [B, T, mix_hc]; drop batch
    mixes_ref = mixes_ref.squeeze(0)
    assert mixes_ref.shape == (T, mix_hc), mixes_ref.shape
    assert mixes_ref.dtype == torch.float32
    print(f"  REF mixes: shape={tuple(mixes_ref.shape)} dtype={mixes_ref.dtype}"
          f"  |mean|={mixes_ref.abs().mean().item():.3e}  "
          f"|max|={mixes_ref.abs().max().item():.3e}")

    kernel = zeus_kernel_ready("mhc_pre_norm_fn")
    if kernel is None:
        print_zeus_skip("mhc_pre_norm_fn")
        return zeus_skip_result(), mixes_ref

    # Zeus: signature TBD; placeholder for when kernel lands
    mixes_z = kernel(residual.to("zeus"), fn.to("zeus"), None, 1e-6)
    ok = compare_tensors("mhc_pre_norm_fn", mixes_ref, mixes_z,
                         atol=5e-4, rtol=5e-4)
    return ok, mixes_ref


# ── Stage: mhc_pre_split_mixes ─────────────────────────────────
def test_mhc_pre_split_mixes(cfg, num_tokens=16, seed=42):
    """[T, mix_hc=24] fp32 → (pre[T,4,1], post[T,4,1], comb[T,4,4]) fp32.

    mixes * scale_bcast + base → 前 mhc 位 sigmoid + pre_eps = pre
                              → 中 mhc 位 sigmoid * post_mult_value = post
                              → 后 mhc² 位 view 成 [T, mhc, mhc] = comb (raw logits)
    """
    print()
    print("=" * 60)
    print("Stage: mhc_pre_split_mixes (sigmoid / scale / base split)")
    print("=" * 60)

    T = num_tokens
    mhc = cfg.mhc_num_residual_streams
    mix_hc = mhc * (2 + mhc)
    post_mult_value = 1.0
    pre_eps = 1e-6

    print(f"  shape: T={T}  mhc={mhc}  mix_hc={mix_hc}  "
          f"post_mult_value={post_mult_value}  pre_eps={pre_eps}")

    torch.manual_seed(seed)
    mixes = torch.randn(1, T, mix_hc, dtype=torch.float32)   # [B=1, T, 24]
    scale = torch.randn(3, dtype=torch.float32) * 0.5
    base = torch.randn(mix_hc, dtype=torch.float32) * 0.1

    pre_ref, post_ref, comb_ref = ref_mhc.mhc_pre_split_mixes_ref(
        mixes, scale, base, mhc, post_mult_value, pre_eps,
    )
    # REF shapes: [B, T, mhc, 1], [B, T, mhc, 1], [B, T, mhc, mhc]
    pre_ref = pre_ref.squeeze(0)
    post_ref = post_ref.squeeze(0)
    comb_ref = comb_ref.squeeze(0)

    assert pre_ref.shape == (T, mhc, 1), pre_ref.shape
    assert post_ref.shape == (T, mhc, 1), post_ref.shape
    assert comb_ref.shape == (T, mhc, mhc), comb_ref.shape

    print(f"  REF pre   : shape={tuple(pre_ref.shape)}  "
          f"range=[{pre_ref.min().item():.3e}, {pre_ref.max().item():.3e}]  "
          f"(sigmoid + pre_eps, expect in (eps, 1+eps))")
    print(f"  REF post  : shape={tuple(post_ref.shape)}  "
          f"range=[{post_ref.min().item():.3e}, {post_ref.max().item():.3e}]  "
          f"(sigmoid * post_mult_value)")
    print(f"  REF comb  : shape={tuple(comb_ref.shape)}  "
          f"|mean|={comb_ref.abs().mean().item():.3e} (raw logits, pre-Sinkhorn)")

    # invariants on REF
    assert (pre_ref >= pre_eps - 1e-9).all() and (pre_ref <= 1.0 + pre_eps + 1e-5).all()
    assert (post_ref >= 0).all() and (post_ref <= post_mult_value + 1e-5).all()

    kernel = zeus_kernel_ready("mhc_pre_split_mixes")
    if kernel is None:
        print_zeus_skip("mhc_pre_split_mixes")
        return zeus_skip_result(), (pre_ref, post_ref, comb_ref)

    pre_z, post_z, comb_z = kernel(
        mixes.to("zeus"), scale.to("zeus"), base.to("zeus"),
        mhc, post_mult_value, pre_eps,
    )
    ok1 = compare_tensors("mhc_pre_split_mixes/pre", pre_ref, pre_z,
                          atol=1e-5, rtol=1e-5)
    ok2 = compare_tensors("mhc_pre_split_mixes/post", post_ref, post_z,
                          atol=1e-5, rtol=1e-5)
    ok3 = compare_tensors("mhc_pre_split_mixes/comb", comb_ref, comb_z,
                          atol=1e-5, rtol=1e-5)
    return (ok1 and ok2 and ok3), (pre_ref, post_ref, comb_ref)


# ── Stage: mhc_sinkhorn ────────────────────────────────────────
def test_mhc_sinkhorn(cfg, num_tokens=16, seed=42):
    """comb logits [T, 4, 4] → doubly-stochastic [T, 4, 4] via Sinkhorn-Knopp.

    REF impl (copy from ref_tile_kernels_torch/mhc.py):
        x = x.softmax(-1) + eps
        x = x / (x.sum(-2, keepdim=True) + eps)
        for _ in range(repeat - 1):
            x = x / (x.sum(-1, keepdim=True) + eps)
            x = x / (x.sum(-2, keepdim=True) + eps)
    """
    print()
    print("=" * 60)
    print("Stage: mhc_sinkhorn (Birkhoff-polytope projection)")
    print("=" * 60)

    T = num_tokens
    mhc = cfg.mhc_num_residual_streams
    repeat = cfg.mhc_sinkhorn_iters
    eps = cfg.mhc_sinkhorn_eps

    print(f"  shape: T={T}  mhc={mhc}  repeat={repeat}  eps={eps}")
    print(f"  config source: mhc_sinkhorn_iters={repeat} "
          f"(DeepSeek-V4 prod=20, TileKernels API default=10)")

    torch.manual_seed(seed)
    comb_logits = torch.randn(T, mhc, mhc, dtype=torch.float32) * 0.5

    comb_ref = ref_mhc.sinkhorn_normalize_ref(comb_logits, repeat=repeat, eps=eps)
    assert comb_ref.shape == (T, mhc, mhc)
    assert comb_ref.dtype == torch.float32

    # REF invariants: doubly-stochastic (row & col sums → 1)
    # repeat=20 tol ≈ 1e-6；repeat=10 约 1e-3
    ds_tol = 1e-4 if repeat >= 15 else 5e-3
    ok_ds = assert_doubly_stochastic("mhc_sinkhorn", comb_ref, tol=ds_tol)
    # REF invariant: non-negative (Birkhoff polytope is in [0, 1])
    neg_count = (comb_ref < 0).sum().item()
    print(f"  [mhc_sinkhorn/non_negative] "
          f"{'PASS' if neg_count == 0 else 'FAIL'}  "
          f"negative_entries={neg_count}/{comb_ref.numel()}")
    ok_nn = (neg_count == 0)

    kernel = zeus_kernel_ready("mhc_sinkhorn") or \
             zeus_kernel_ready("sinkhorn_normalize")
    if kernel is None:
        print_zeus_skip("mhc_sinkhorn / sinkhorn_normalize")
        return zeus_skip_result(ok_ds and ok_nn), comb_ref

    comb_z = kernel(comb_logits.to("zeus"), repeat, eps)
    ok_num = compare_tensors("mhc_sinkhorn/numerical", comb_ref, comb_z,
                             atol=5e-5, rtol=5e-5)
    return (ok_ds and ok_nn and ok_num), comb_ref


# ── Stage: mhc_pre_apply_mix ───────────────────────────────────
def test_mhc_pre_apply_mix(cfg, num_tokens=16, seed=42):
    """(residual × pre).sum(-2).bfloat16() —— 按 mhc 轴加权求和回单流。

    Sanity: pre=1/mhc 且 residual 各流相同 → 输出 ≈ residual[:, 0, :]
    """
    print()
    print("=" * 60)
    print("Stage: mhc_pre_apply_mix (weighted sum over streams)")
    print("=" * 60)

    T = num_tokens
    H = 256
    mhc = cfg.mhc_num_residual_streams

    print(f"  shape: residual=[{T},{mhc},{H}] bf16  pre=[{T},{mhc},1] fp32")

    torch.manual_seed(seed)
    residual = torch.randn(T, mhc, H, dtype=torch.bfloat16) * 0.1
    pre = torch.rand(T, mhc, 1, dtype=torch.float32) * 0.5 + 0.05

    out_ref = ref_mhc.mhc_pre_apply_mix_ref(residual, pre)
    assert out_ref.shape == (T, H)
    assert out_ref.dtype == torch.bfloat16
    print(f"  REF out: shape={tuple(out_ref.shape)} dtype={out_ref.dtype}")

    # Sanity: uniform pre + identical streams → out == one stream
    uniform_pre = torch.full((T, mhc, 1), 1.0 / mhc, dtype=torch.float32)
    uniform_res = residual[:, 0:1, :].expand(T, mhc, H).contiguous()
    sanity_out = ref_mhc.mhc_pre_apply_mix_ref(uniform_res, uniform_pre)
    sanity_diff = (sanity_out.float() - residual[:, 0, :].float()).abs().max().item()
    print(f"  [mhc_pre_apply_mix/uniform_sanity] "
          f"{'PASS' if sanity_diff < 5e-3 else 'DIFF'}  "
          f"max_diff={sanity_diff:.3e}  (expect ≈ 0 for uniform pre)")

    kernel = zeus_kernel_ready("mhc_pre_apply_mix")
    if kernel is None:
        print_zeus_skip("mhc_pre_apply_mix")
        return zeus_skip_result(), out_ref

    out_z = kernel(residual.to("zeus"), pre.to("zeus"))
    ok = compare_tensors("mhc_pre_apply_mix", out_ref, out_z,
                         atol=5e-3, rtol=5e-3)
    return ok, out_ref


# ── Stage: mhc_post ────────────────────────────────────────────
def test_mhc_post(cfg, num_tokens=16, seed=42):
    """mhc_post(x, residual, post_mix, comb) = x·post + einsum(comb, residual)
    —— fp32 accumulator + 单次 bf16 RNE.

    Sanity: comb = I, post = 0 → out == residual (identity pass-through via comb).
    """
    print()
    print("=" * 60)
    print("Stage: mhc_post (scatter back to streams + residual mix)")
    print("=" * 60)

    T = num_tokens
    H = 256
    mhc = cfg.mhc_num_residual_streams

    print(f"  shape: x=[{T},{H}] bf16  residual=[{T},{mhc},{H}] bf16  "
          f"post=[{T},{mhc},1] fp32  comb=[{T},{mhc},{mhc}] fp32")

    torch.manual_seed(seed)
    x = torch.randn(T, H, dtype=torch.bfloat16) * 0.1
    residual = torch.randn(T, mhc, H, dtype=torch.bfloat16) * 0.1
    post = torch.rand(T, mhc, 1, dtype=torch.float32) * 0.3
    # comb: doubly-stochastic via Sinkhorn on random logits
    comb_logits = torch.randn(T, mhc, mhc, dtype=torch.float32) * 0.5
    comb = ref_mhc.sinkhorn_normalize_ref(comb_logits, repeat=20, eps=1e-6)

    # REF expects [B, T, ...] shapes via einsum 'abmn,abmc→abnc'. Wrap in B=1.
    out_ref = ref_mhc.mhc_post_ref(
        x.unsqueeze(0),
        residual.unsqueeze(0),
        post.unsqueeze(0),
        comb.unsqueeze(0),
    ).squeeze(0)
    assert out_ref.shape == (T, mhc, H)
    assert out_ref.dtype == torch.bfloat16
    print(f"  REF out: shape={tuple(out_ref.shape)} dtype={out_ref.dtype}")

    # Sanity: post=0, comb=I → out == residual
    id_comb = torch.eye(mhc).expand(T, mhc, mhc).contiguous()
    zero_post = torch.zeros_like(post)
    sanity = ref_mhc.mhc_post_ref(
        x.unsqueeze(0),
        residual.unsqueeze(0),
        zero_post.unsqueeze(0),
        id_comb.unsqueeze(0),
    ).squeeze(0)
    diff = (sanity.float() - residual.float()).abs().max().item()
    print(f"  [mhc_post/identity_sanity] "
          f"{'PASS' if diff < 5e-3 else 'DIFF'}  "
          f"max_diff={diff:.3e}  (post=0, comb=I → out≈residual)")

    kernel = zeus_kernel_ready("mhc_post")
    if kernel is None:
        print_zeus_skip("mhc_post")
        return zeus_skip_result(), out_ref

    out_z = kernel(x.to("zeus"), residual.to("zeus"),
                   post.to("zeus"), comb.to("zeus"))
    ok = compare_tensors("mhc_post", out_ref, out_z, atol=5e-3, rtol=5e-3)
    return ok, out_ref


# ── Stage: mhc_block_ffn_full ──────────────────────────────────
def test_mhc_block_ffn_full(cfg, num_tokens=16, seed=42):
    """端到端一轮 FFN-side mHC wrap：expand → pre(4 步) → identity sublayer → post.

    当前 sublayer 是 identity（`x_sub = x`），目的是先把 wrap 逻辑跑通；
    后续把 identity 替换成 `moe_block_full` 即得到 Glm5Next / DeepSeek-V4
    FFN-side 真正一层。
    """
    print()
    print("=" * 60)
    print("Stage: mhc_block_ffn_full (REF-only end-to-end, identity sublayer)")
    print("=" * 60)

    T = num_tokens
    H = 256
    mhc = cfg.mhc_num_residual_streams
    mix_hc = mhc * (2 + mhc)
    repeat = cfg.mhc_sinkhorn_iters
    eps = cfg.mhc_sinkhorn_eps

    print(f"  shape: T={T}  H={H}  mhc={mhc}  mix_hc={mix_hc}  "
          f"sinkhorn_iters={repeat}")

    torch.manual_seed(seed)
    hidden = torch.randn(T, H, dtype=torch.bfloat16) * 0.1
    fn = torch.randn(mix_hc, mhc * H, dtype=torch.float32) * 0.02
    scale = torch.randn(3, dtype=torch.float32) * 0.5
    base = torch.randn(mix_hc, dtype=torch.float32) * 0.1

    # (0) embedding → streams
    residual = ref_mhc.expand_to_mhc_ref(hidden, mhc)      # [T, mhc, H] bf16

    # (1) pre_norm_fn: residual → mixes fp32
    mixes = ref_mhc.mhc_pre_norm_fn_ref(
        residual.unsqueeze(0), fn, None, 1e-6,
    )                                                       # [1, T, 24] fp32

    # (2) split
    pre, post, comb_logits = ref_mhc.mhc_pre_split_mixes_ref(
        mixes, scale, base, mhc, 1.0, 1e-6,
    )                                                       # each in [1, T, ...]

    # (3) Sinkhorn on comb logits
    comb = ref_mhc.sinkhorn_normalize_ref(comb_logits, repeat=repeat, eps=eps)
    assert_doubly_stochastic(
        "mhc_block_ffn_full/sinkhorn", comb.squeeze(0),
        tol=1e-4 if repeat >= 15 else 5e-3,
    )

    # (4) pre_apply_mix: residual → layer_input
    layer_input = ref_mhc.mhc_pre_apply_mix_ref(
        residual.unsqueeze(0), pre,
    ).squeeze(0)                                            # [T, H] bf16
    assert layer_input.shape == (T, H)

    # Sublayer: identity (TODO: replace with moe_block_full once wiring is done)
    x_sub = layer_input

    # (5) post: scatter back to streams
    out = ref_mhc.mhc_post_ref(
        x_sub.unsqueeze(0),
        residual.unsqueeze(0),
        post, comb,
    ).squeeze(0)                                            # [T, mhc, H] bf16
    assert out.shape == (T, mhc, H)

    print(f"  REF end-to-end: out shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"  REF out[0, 0, :4] = "
          f"{[round(v, 4) for v in out[0, 0, :4].float().tolist()]}")

    # Not comparing against Zeus: this stage exercises ALL of (1)..(5). If any
    # individual Zeus kernel is missing, fall back to "REF-only smoke test".
    any_kernel = any(zeus_kernel_ready(op) for op in (
        "mhc_pre_norm_fn", "mhc_pre_split_mixes", "mhc_sinkhorn",
        "sinkhorn_normalize", "mhc_pre_apply_mix", "mhc_post",
    ))
    if not any_kernel:
        print("  [zeus] SKIP end-to-end: no mhc_* kernels registered on Zeus yet")
        return zeus_skip_result(), out
    print("  [zeus] end-to-end assembly not yet implemented "
          "(enable once all 5 sub-kernels land)")
    return zeus_skip_result(), out


# ── Dispatch ───────────────────────────────────────────────────
STAGES = {
    "mhc_expand": test_mhc_expand,
    "mhc_pre_norm_fn": test_mhc_pre_norm_fn,
    "mhc_pre_split_mixes": test_mhc_pre_split_mixes,
    "mhc_sinkhorn": test_mhc_sinkhorn,
    "mhc_pre_apply_mix": test_mhc_pre_apply_mix,
    "mhc_post": test_mhc_post,
    "mhc_block_ffn_full": test_mhc_block_ffn_full,
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
    args = parser.parse_args()

    cfg = load_config()
    print(f"mHC config: mhc_mult={cfg.mhc_num_residual_streams}  "
          f"sinkhorn_iters={cfg.mhc_sinkhorn_iters}  "
          f"sinkhorn_eps={cfg.mhc_sinkhorn_eps}")
    print(f"  GLM-extended (悬空): mhc_tau={cfg.mhc_tau}  "
          f"hres_vwnstyle={cfg.hres_vwnstyle}")
    print(f"Reference device: {REF_DEVICE}")
    print(f"Zeus runtime available: {ZEUS_AVAILABLE}")

    results = {}
    for name, fn in STAGES.items():
        if args.stage not in (name, "all"):
            continue
        try:
            ok, _ = fn(cfg, num_tokens=args.num_tokens, seed=args.seed)
            results[name] = ok
        except Exception as e:
            print(f"  [{name}] EXCEPTION: {e}")
            results[name] = False

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        if ok is True:
            status = "PASS"
        elif ok is False:
            status = "FAIL"
        else:
            status = "SKIP (REF-only; Zeus TODO)"
        print(f"  {name:28s} : {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
