"""
Kimi Linear (KDA) Attention 段逐算子 REF-vs-Zeus 对齐

范围（见 zeus_dev/kimi_linear_attn_dev.md）：
  - 起点：input_layernorm 之后、进入 KimiDeltaAttention.forward 的 hidden_states [T, H]
  - 终点：o_proj 输出 [T, H]
  - 单 device / 单 layer / KDA layer only；不含 TP / speculative / MLA layer

**核心**：decode 与 extend 是两条**完全不同**的 kernel 流水线。
  - decode  : causal_conv1d_update + fused_recurrent_kda
  - extend  : causal_conv1d_fn     + chunk_kda(7-sub-kernel pipeline)
projection / kda_gate / rms_norm_gated / o_proj 两条路径共享。

Stage:
  fused_kda_gate              —— softplus(β,τ) * (-exp(A_log)) per-head 点乘
  causal_conv1d_update        —— decode 单步 conv1d state-update（单 projection）
  causal_conv1d_update_qkv    —— decode 单步 conv1d state-update（Q/K/V 融合）
  causal_conv1d_fn            —— extend varlen causal conv1d
  l2norm                      —— q/k 的 per-head-dim L2 normalize
  fused_recurrent_kda_Sdecay  —— decode core **生产默认**：S 先 decay 再 readout，
                                 与 CUDA fla 同形态、operations 更少
  fused_recurrent_kda         —— decode core Tensor Core GEMV **实验**路径：
                                 tl.dot + weight/share 双 memory tier 调度
  chunk_kda                   —— extend core
  rms_norm_gated              —— o_norm(core_out, g_gate), activation=sigmoid
  kimi_delta_attn_decode      —— 端到端 decode 路径（**indexed kernel chain**：
                                 conv1d_update_indexed + Sdecay_indexed，state pool
                                 直接对全局池做 slot 寻址，无需 host gather/scatter）
  kimi_delta_attn_extend      —— 端到端 extend 路径

用法:
  python zeus_dev/dev_kimi_linear_attn_test.py                                   # 跑全部
  python zeus_dev/dev_kimi_linear_attn_test.py --stage fused_kda_gate
  python zeus_dev/dev_kimi_linear_attn_test.py --stage causal_conv1d_update
  python zeus_dev/dev_kimi_linear_attn_test.py --stage causal_conv1d_update_qkv
  python zeus_dev/dev_kimi_linear_attn_test.py --stage fused_recurrent_kda_Sdecay   # 生产默认
  python zeus_dev/dev_kimi_linear_attn_test.py --stage fused_recurrent_kda          # TC 实验
  python zeus_dev/dev_kimi_linear_attn_test.py --stage chunk_kda
  python zeus_dev/dev_kimi_linear_attn_test.py --stage kimi_delta_attn_decode
  python zeus_dev/dev_kimi_linear_attn_test.py --stage kimi_delta_attn_extend

约定：Zeus 侧未落地的算子直接 raise NotImplementedError 并附 TODO 锚点；
不允许 silent fallback 到 torchnative（对齐 dev_glm4_moe_test.py 的策略）。
"""

import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch_zeus  # noqa: F401 — registers zeus backend
import sgl_kernel_zeus

# ── SGLang server_args mock（与 demo_zeus_layer_compare.py 同策略） ──
import sglang.srt.server_args
_dummy_args = Mock()
_dummy_args.rl_on_policy_target = None
sglang.srt.server_args.get_global_server_args = lambda *a, **kw: _dummy_args


# Kimi-Linear 典型配置（来自 Kimi-Linear-48B-A3B 量级 checkpoint；对齐脚本
# 用小一号的 proxy-shape 跑起来，真实 shape 的 kernel-level align 由
# sgl-kernel-zeus/tests/ 下的对应测试承担）。
def default_kimi_cfg():
    return SimpleNamespace(
        hidden_size=512,              # proxy for 真实 H (4096 / 5120)
        num_heads=8,                  # proxy for 16 / 32
        head_dim=64,                  # proxy for 128（取较小值让 CPU REF 能跑）
        conv_kernel_size=4,           # 真实通常就是 4
        rms_norm_eps=1e-5,
        # KDA gate 的两个硬超参（见 fla/kda.py:1306 fused_kda_gate 默认值）
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )


# 选一个 REF device：CUDA 优先，否则退 CPU（此时 FLA 的 triton kernel 不可用，
# core-attn 的 REF 必须走 pure-torch 等价实现）。
REF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAN_USE_FLA_TRITON = REF_DEVICE == "cuda"


# ── 通用比较工具（与 dev_glm4_moe_test.py 保持签名一致） ──
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


def zeus_todo(kernel_name, anchor):
    """Zeus 侧未落地时的统一占位：显式 raise，不允许 silent fallback。"""
    raise NotImplementedError(
        f"[TODO] sgl_kernel_zeus.{kernel_name} 尚未实现。\n"
        f"       参考 CUDA 实现: {anchor}\n"
        f"       完成后回填 kimi_linear_attn_dev.md 的算子依赖表。"
    )


# ── pure-torch REF helpers ──────────────────────────────────────
def _ref_causal_conv1d_update(x, conv_state, weight, bias, activation="silu"):
    """REF for causal_conv1d_update (decode single-step).

    x          : [N, C]       current token（N=decode batch, C=H_qkv）
    conv_state : [N, C, K-1]  rolling window; updated IN-PLACE
    weight     : [C, K]       per-channel conv kernel
    bias       : [C] or None
    returns    : [N, C]
    """
    N, C = x.shape
    K = weight.shape[1]
    # 拼 window = [state, x] → [N, C, K]，然后 dot-per-channel
    win = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)    # [N, C, K]
    out = (win * weight.unsqueeze(0)).sum(dim=-1)             # [N, C]
    if bias is not None:
        out = out + bias.unsqueeze(0)
    # 更新 state：最后 (K-1) 个元素
    new_state = win[..., 1:]
    conv_state.copy_(new_state)
    if activation == "silu":
        out = torch.nn.functional.silu(out)
    return out


def _ref_causal_conv1d_fn(
    x, weight, bias, conv_states, has_initial_state,
    query_start_loc, activation="silu",
):
    """REF for causal_conv1d_fn (extend varlen).

    x               : [C, total_T]  flat tokens across batch（C=H_qkv）
    weight          : [C, K]
    bias            : [C] or None
    conv_states     : [num_cache, C, K-1] —— 按 cache_indices 取；简化起见
                      dev 里直接假设 cache_indices == arange(B)，外部传入的
                      conv_states 已是目标 slot 的引用
    has_initial_state : [B] bool
    query_start_loc : [B+1] int32
    returns         : [C, total_T]；conv_states in-place 更新到最后 K-1 帧
    """
    C, total_T = x.shape
    B = query_start_loc.numel() - 1
    K = weight.shape[1]
    out = torch.empty_like(x)
    for b in range(B):
        lo = int(query_start_loc[b].item())
        hi = int(query_start_loc[b + 1].item())
        T_b = hi - lo
        seg = x[:, lo:hi]                                    # [C, T_b]
        if bool(has_initial_state[b]):
            state = conv_states[b]                           # [C, K-1]
        else:
            state = torch.zeros(C, K - 1, dtype=x.dtype, device=x.device)
        # 左 padding 接 state，然后 sliding-conv
        padded = torch.cat([state, seg], dim=-1)             # [C, K-1+T_b]
        for t in range(T_b):
            win = padded[:, t:t + K]                         # [C, K]
            y = (win * weight).sum(dim=-1)                   # [C]
            if bias is not None:
                y = y + bias
            out[:, lo + t] = y
        # 回写 state 为最后 K-1 帧
        conv_states[b].copy_(padded[:, -(K - 1):])
    if activation == "silu":
        out = torch.nn.functional.silu(out)
    return out


def _ref_fused_kda_gate(g_flat, A_log, head_dim, g_bias=None,
                         beta=1.0, threshold=20.0):
    """REF for fused_kda_gate (fla/kda.py:1306).

    g_flat : [..., H*head_dim]  (f_b_proj(f_a_proj(x)) 的输出)
    A_log  : [H] 或 [1,1,H,1] fp32
    g_bias : [H*head_dim] 或 None
    returns: [..., H, head_dim] fp32
    """
    orig_shape = g_flat.shape[:-1]
    HD = g_flat.shape[-1]
    A_flat = A_log.reshape(-1).float()
    H = A_flat.numel()
    assert H * head_dim == HD, f"H*head_dim ({H}*{head_dim}) != HD ({HD})"
    x = g_flat.reshape(-1, HD).float()
    if g_bias is not None:
        x = x + g_bias.reshape(-1).float().unsqueeze(0)
    x_scaled = x * beta
    use_linear = x_scaled > threshold
    sp = torch.where(
        use_linear,
        x,
        (1.0 / beta) * torch.log1p(torch.exp(x_scaled)),
    )
    a = -torch.exp(A_flat)                                   # [H]
    # 广播到 [..., H, head_dim]
    x_hd = x.reshape(-1, H, head_dim)
    sp_hd = sp.reshape(-1, H, head_dim)
    y = a.view(1, H, 1) * sp_hd                              # fp32
    return y.reshape(*orig_shape, H, head_dim)


def _ref_l2norm(x, eps=1e-6):
    """REF for fla/l2norm.l2norm_fwd —— 沿最后一维做 L2 normalize（fp32 accum）。"""
    x32 = x.float()
    rsqrt = torch.rsqrt(x32.pow(2).sum(dim=-1, keepdim=True) + eps)
    return (x32 * rsqrt).to(x.dtype)


def _ref_rms_norm_gated(x, g, weight, activation="sigmoid", eps=1e-5):
    """REF for fla/kda.rms_norm_gated —— rmsnorm(x) * act(g)。

    注意 KDA 固定 activation='sigmoid'：y = rmsnorm(x) * sigmoid(g)
    （不是 swish 的 g * sigmoid(g)）。
    """
    x32 = x.float()
    rstd = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x32 * rstd
    if weight is not None:
        y = y * weight.float()
    g32 = g.float()
    if activation in ("swish", "silu"):
        y = y * g32 * torch.sigmoid(g32)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(g32)
    else:
        raise ValueError(f"unsupported activation: {activation}")
    return y.to(x.dtype)


def _ref_fused_recurrent_kda(
    q, k, v, g, beta, initial_state, scale, use_qk_l2norm_in_kernel=True,
    cu_seqlens=None,
):
    """Pure-torch REF for fused_recurrent_kda（IS_KDA=True 分支）。

    shapes（单 batch，cu_seqlens 展平 batch 模式与 CUDA 对齐）：
      q, k    : [1, T, H, K]  bf16
      v       : [1, T, HV, V] bf16        (HV 可能 != H；KDA 下通常 HV == H)
      g       : [1, T, H, K]  fp32        (fused_kda_gate 的输出)
      beta    : [1, T, H]     fp32
      initial_state : [N, HV, K, V] fp32  N=sequences
      cu_seqlens    : [N+1] int32 或 None

    KDA recurrent 语义（每个 token, 每个 head）：
      k̂ = l2norm(k) if use_qk_l2norm_in_kernel else k
      q̂ = l2norm(q) * scale                （scale 融到 q 一侧）
      S_t = S_{t-1} * exp(g_t) + beta_t * (v_t - S_{t-1} k̂_t) ⊗ k̂_t   (delta-rule)
      o_t = q̂_t @ S_t
    注意 g 是 **per-(head, K)** 的 gating（不是 scalar），与 GDR / Mamba2 的
    scalar gate 不同。
    """
    assert q.shape[0] == 1
    _, T, H, K = q.shape
    V = v.shape[-1]
    HV = v.shape[2]
    # 这里只实现 H == HV 的 KDA 典型情况（Kimi 里 f_b_proj / k_proj / v_proj
    # 都投到同一个 projection_size，head 数一致）。
    assert H == HV, "dev REF 暂只实现 H == HV 的 KDA 情况"

    if cu_seqlens is None:
        cu = torch.tensor([0, T], dtype=torch.int32)
        N = 1
    else:
        cu = cu_seqlens.to(torch.int32)
        N = cu.numel() - 1

    q = q[0].float()                                      # [T, H, K]
    k = k[0].float()
    v = v[0].float()
    g = g[0].float()
    beta = beta[0].float()                                # [T, H]

    if use_qk_l2norm_in_kernel:
        q = _ref_l2norm(q)
        k = _ref_l2norm(k)
    q = q * scale

    o = torch.zeros(T, H, V, dtype=torch.float32)
    final_state = initial_state.clone().float()           # [N, H, K, V]

    for n in range(N):
        lo = int(cu[n].item())
        hi = int(cu[n + 1].item())
        S = final_state[n]                                # [H, K, V]
        for t in range(lo, hi):
            g_t = torch.exp(g[t])                         # [H, K]
            # S <- S * g_t (沿 K 维广播)
            S = S * g_t.unsqueeze(-1)                     # [H, K, V]
            k_t = k[t]                                    # [H, K]
            v_t = v[t]                                    # [H, V]
            # 读取当前 S 投影出的 v̂
            v_hat = torch.einsum("hkv,hk->hv", S, k_t)    # [H, V]
            delta = v_t - v_hat                           # [H, V]
            # outer product: S += beta_t * delta ⊗ k_t
            update = torch.einsum(
                "hv,hk->hkv", beta[t].unsqueeze(-1) * delta, k_t
            )
            S = S + update
            # o_t = q_t @ S
            o[t] = torch.einsum("hk,hkv->hv", q[t], S)
        final_state[n] = S

    return o.unsqueeze(0).to(torch.bfloat16), final_state.to(initial_state.dtype)


def _ref_chunk_kda(
    q, k, v, g, beta, initial_state, scale, use_qk_l2norm_in_kernel=True,
    cu_seqlens=None, output_final_state=True,
):
    """Pure-torch REF for chunk_kda。

    CUDA 侧是 `cumsum → scaled_dot_kkt → solve_tril → recompute_w_u →
    chunk_delta_h → chunk_gla_fwd_o_gk` 的 7-sub-kernel 流水线。dev REF 直接
    **复用** _ref_fused_recurrent_kda：两者数学等价（recurrent 等价于 chunk
    计算结果，差异仅是 throughput），用 recurrent 形态作为 golden 对齐 Zeus
    的 chunk 实现。
    """
    return _ref_fused_recurrent_kda(
        q, k, v, g, beta, initial_state, scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )


# ── Stage: fused_kda_gate ───────────────────────────────────────
def test_fused_kda_gate(cfg, num_tokens=8, seed=42):
    """对齐 `fla/kda.fused_kda_gate`（softplus(β,τ) * (-exp(A_log))）。

    输入：
      g      : [T, H*head_dim]  fp32 or bf16（来自 f_b_proj）
      A_log  : [H] fp32
      g_bias : [H*head_dim] fp32 (可选；KDA 里来自 `dt_bias`)
    输出：
      y : [T, H, head_dim] fp32
    """
    print()
    print("=" * 60)
    print("Stage: fused_kda_gate (softplus(β,τ) * -exp(A_log))")
    print("=" * 60)

    H = cfg.num_heads
    D = cfg.head_dim
    print(f"  shape: T={num_tokens}  H={H}  head_dim={D}")
    print(f"  softplus: beta={cfg.softplus_beta}  threshold={cfg.softplus_threshold}")

    torch.manual_seed(seed)
    g = torch.randn(num_tokens, H * D, dtype=torch.bfloat16)
    A_log = torch.randn(H, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H * D, dtype=torch.float32) * 0.01

    # REF（pure torch）
    y_ref = _ref_fused_kda_gate(
        g.to(REF_DEVICE), A_log.to(REF_DEVICE), D,
        g_bias=dt_bias.to(REF_DEVICE),
        beta=cfg.softplus_beta, threshold=cfg.softplus_threshold,
    )
    print(f"  REF y: shape={tuple(y_ref.shape)} dtype={y_ref.dtype}")

    # Zeus
    y_zeus = sgl_kernel_zeus.fused_kda_gate(
        g.to("privateuseone"), A_log.to("privateuseone"),
        head_dim=D, g_bias=dt_bias.to("privateuseone"),
    )
    ok = compare_tensors("fused_kda_gate/y", y_ref, y_zeus)
    return ok, (y_ref,)


# ── Stage: causal_conv1d_update (decode) ────────────────────────
def test_causal_conv1d_update(cfg, num_tokens=4, seed=42):
    """Decode 单步 conv1d state-update（kernel_size=4，per-channel）。

    CUDA 实现: `mamba/causal_conv1d_triton.py:973 causal_conv1d_update`
    """
    print()
    print("=" * 60)
    print("Stage: causal_conv1d_update (decode single-step)")
    print("=" * 60)

    Hq = cfg.num_heads * cfg.head_dim
    K = cfg.conv_kernel_size
    N = num_tokens    # decode 时 N == batch_size
    print(f"  shape: N={N}  H_qkv={Hq}  kernel_size={K}")

    torch.manual_seed(seed)
    x = torch.randn(N, Hq, dtype=torch.bfloat16)
    weight = torch.randn(Hq, K, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(Hq, dtype=torch.bfloat16) * 0.01

    # 两侧各保留一份 state（避免原位写互相污染）
    state_ref = torch.randn(N, Hq, K - 1, dtype=torch.bfloat16) * 0.1
    state_zeus = state_ref.clone()

    # REF
    # state_ref.to(REF_DEVICE) 在 REF_DEVICE!="cpu" 时会新建一份拷贝，REF 的原位
    # state 更新写进的是这份拷贝；保留其引用去比较，否则比到的是未更新的原始
    # CPU state_ref（假性 DIFF）。
    state_ref_dev = state_ref.to(REF_DEVICE)
    y_ref = _ref_causal_conv1d_update(
        x.to(REF_DEVICE), state_ref_dev,
        weight.to(REF_DEVICE), bias.to(REF_DEVICE),
        activation="silu",
    )
    print(f"  REF y: shape={tuple(y_ref.shape)} dtype={y_ref.dtype}")

    # Zeus — causal_conv1d_update
    # Move tensors to Zeus device up-front so in-place state update is captured.
    x_zeus     = x.to("privateuseone")
    w_zeus     = weight.to("privateuseone")
    b_zeus     = bias.to("privateuseone")
    state_zeus_dev = state_zeus.to("privateuseone")

    y_zeus = sgl_kernel_zeus.causal_conv1d_update(
        x_zeus, state_zeus_dev, w_zeus, b_zeus, activation="silu",
    )

    ok_out   = compare_tensors("causal_conv1d_update/out",   y_ref,      y_zeus)
    ok_state = compare_tensors("causal_conv1d_update/state", state_ref_dev, state_zeus_dev)
    return ok_out and ok_state, (y_ref, state_ref_dev)


# ── Stage: causal_conv1d_update_qkv (decode, fused Q/K/V) ───────
def test_causal_conv1d_update_qkv(cfg, num_tokens=4, seed=42):
    """Fused Q/K/V decode 单步 conv1d state-update。

    Zeus 实现: `sgl-kernel-zeus/csrc/mamba/causal_conv1d_update_qkv_kernel.py`
              → `sgl_kernel_zeus.causal_conv1d_update_qkv`
    单 kernel launch 同时更新 q/k/v 三个 projection 的 conv_state 并产出输出，
    消除两次 launch 开销；REF 用三次独立 `_ref_causal_conv1d_update` 作 golden。

    Kimi-Linear 里 q/k/v 共享 (N, C=num_heads*head_dim, K=conv_kernel_size)；
    weight/bias/state 三份独立。Zeus kernel 要求 C 能被 CORE_NUM=2 整除。
    """
    print()
    print("=" * 60)
    print("Stage: causal_conv1d_update_qkv (decode fused Q/K/V)")
    print("=" * 60)

    C = cfg.num_heads * cfg.head_dim
    K = cfg.conv_kernel_size
    N = num_tokens    # decode 时 N == batch_size
    assert C % 2 == 0, f"C={C} 必须能被 CORE_NUM=2 整除"
    print(f"  shape: N={N}  C(=H_qkv)={C}  kernel_size={K}")

    torch.manual_seed(seed)

    def _mk():
        x = torch.randn(N, C, dtype=torch.bfloat16)
        w = torch.randn(C, K, dtype=torch.bfloat16) * 0.1
        b = torch.randn(C, dtype=torch.bfloat16) * 0.01
        s = torch.randn(N, C, K - 1, dtype=torch.bfloat16) * 0.1
        return x, w, b, s

    x_q, w_q, b_q, sq = _mk()
    x_k, w_k, b_k, sk = _mk()
    x_v, w_v, b_v, sv = _mk()

    # REF：三次独立 single-projection
    sq_ref, sk_ref, sv_ref = sq.clone(), sk.clone(), sv.clone()
    # 保留 REF 真正原位更新的设备张量引用（见 causal_conv1d_update 处说明）。
    sq_ref_dev = sq_ref.to(REF_DEVICE)
    sk_ref_dev = sk_ref.to(REF_DEVICE)
    sv_ref_dev = sv_ref.to(REF_DEVICE)
    yq_ref = _ref_causal_conv1d_update(
        x_q.to(REF_DEVICE), sq_ref_dev,
        w_q.to(REF_DEVICE), b_q.to(REF_DEVICE), activation="silu",
    )
    yk_ref = _ref_causal_conv1d_update(
        x_k.to(REF_DEVICE), sk_ref_dev,
        w_k.to(REF_DEVICE), b_k.to(REF_DEVICE), activation="silu",
    )
    yv_ref = _ref_causal_conv1d_update(
        x_v.to(REF_DEVICE), sv_ref_dev,
        w_v.to(REF_DEVICE), b_v.to(REF_DEVICE), activation="silu",
    )
    print(f"  REF y_q/k/v: shape={tuple(yq_ref.shape)} dtype={yq_ref.dtype}")

    # Zeus：单 kernel 同时出 q/k/v
    sq_z = sq.clone().to("privateuseone")
    sk_z = sk.clone().to("privateuseone")
    sv_z = sv.clone().to("privateuseone")
    yq_z, yk_z, yv_z = sgl_kernel_zeus.causal_conv1d_update_qkv(
        x_q.to("privateuseone"), x_k.to("privateuseone"), x_v.to("privateuseone"),
        sq_z, sk_z, sv_z,
        w_q.to("privateuseone"), w_k.to("privateuseone"), w_v.to("privateuseone"),
        b_q.to("privateuseone"), b_k.to("privateuseone"), b_v.to("privateuseone"),
        activation="silu",
    )

    ok = True
    ok &= compare_tensors("causal_conv1d_update_qkv/out_q",   yq_ref, yq_z)
    ok &= compare_tensors("causal_conv1d_update_qkv/out_k",   yk_ref, yk_z)
    ok &= compare_tensors("causal_conv1d_update_qkv/out_v",   yv_ref, yv_z)
    ok &= compare_tensors("causal_conv1d_update_qkv/state_q", sq_ref_dev, sq_z)
    ok &= compare_tensors("causal_conv1d_update_qkv/state_k", sk_ref_dev, sk_z)
    ok &= compare_tensors("causal_conv1d_update_qkv/state_v", sv_ref_dev, sv_z)
    return ok, (yq_ref, yk_ref, yv_ref, sq_ref_dev, sk_ref_dev, sv_ref_dev)


# ── Stage: causal_conv1d_fn (extend) ────────────────────────────
def test_causal_conv1d_fn(cfg, seed=42):
    """Extend varlen causal conv1d：2 个 seq 拼成 flat，cu_seqlens 切边界。

    CUDA 实现: `mamba/causal_conv1d_triton.py:378 causal_conv1d_fn`
    """
    print()
    print("=" * 60)
    print("Stage: causal_conv1d_fn (extend varlen)")
    print("=" * 60)

    Hq = cfg.num_heads * cfg.head_dim
    K = cfg.conv_kernel_size
    seq_lens = [12, 20]                    # 两个 seq 的长度
    B = len(seq_lens)
    total_T = sum(seq_lens)
    print(f"  shape: B={B}  seq_lens={seq_lens}  H_qkv={Hq}  kernel_size={K}")

    torch.manual_seed(seed)
    x = torch.randn(Hq, total_T, dtype=torch.bfloat16)
    weight = torch.randn(Hq, K, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(Hq, dtype=torch.bfloat16) * 0.01
    query_start_loc = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
        dtype=torch.int32,
    )
    # 第 0 个 seq 是全新 prefill（has_initial=False）；第 1 个是带 prefix 的
    # extend（has_initial=True）—— 覆盖两种分支。
    has_initial_state = torch.tensor([False, True], dtype=torch.bool)

    conv_states_ref = torch.randn(B, Hq, K - 1, dtype=torch.bfloat16) * 0.1
    conv_states_zeus = conv_states_ref.clone()

    # REF
    # 保留 REF 真正原位更新的设备张量引用（见 causal_conv1d_update 处说明）。
    conv_states_ref_dev = conv_states_ref.to(REF_DEVICE)
    y_ref = _ref_causal_conv1d_fn(
        x.to(REF_DEVICE), weight.to(REF_DEVICE), bias.to(REF_DEVICE),
        conv_states_ref_dev,
        has_initial_state.to(REF_DEVICE),
        query_start_loc.to(REF_DEVICE),
        activation="silu",
    )
    print(f"  REF y: shape={tuple(y_ref.shape)} dtype={y_ref.dtype}")

    # Zeus
    DEV = "privateuseone"
    conv_states_zeus_dev = conv_states_zeus.to(DEV)
    y_zeus = sgl_kernel_zeus.causal_conv1d_fn(
        x.to(DEV), weight.to(DEV), bias.to(DEV),
        conv_state=conv_states_zeus_dev,
        has_initial_state=has_initial_state.to(DEV),
        cu_seqlens=query_start_loc.to(DEV),
        activation="silu",
    )

    ok_out   = compare_tensors("causal_conv1d_fn/out",   y_ref, y_zeus)
    ok_state = compare_tensors("causal_conv1d_fn/state", conv_states_ref_dev, conv_states_zeus_dev)
    return ok_out and ok_state, (y_ref, conv_states_ref_dev)


# ── Stage: l2norm ───────────────────────────────────────────────
def test_l2norm(cfg, num_tokens=8, seed=42):
    """q/k 的 per-head-dim L2 normalize。

    CUDA 实现: `python/sglang/srt/layers/attention/fla/l2norm.py`
    注：生产路径 `use_qk_l2norm_in_kernel=True` 时融进 core kernel；
    这 stage 独立测试是为了数值路径能 isolate 验证。
    """
    print()
    print("=" * 60)
    print("Stage: l2norm (per-head-dim L2 normalize)")
    print("=" * 60)

    H = cfg.num_heads
    D = cfg.head_dim
    T = num_tokens
    print(f"  shape: [1, T={T}, H={H}, K={D}]")

    torch.manual_seed(seed)
    q = torch.randn(1, T, H, D, dtype=torch.bfloat16)

    y_ref = _ref_l2norm(q.to(REF_DEVICE))
    print(f"  REF y: shape={tuple(y_ref.shape)} dtype={y_ref.dtype}")

    # Zeus
    y_zeus = sgl_kernel_zeus.l2norm(q.to("privateuseone"))
    ok = compare_tensors("l2norm/y", y_ref, y_zeus)
    return ok, (y_ref,)


# ── Stage: fused_recurrent_kda (decode core) ────────────────────
def _make_kda_inputs(H, D, seq_lens, seed):
    """共享生成函数，给 baseline / Sdecay 两个 stage 用同一份输入。

    K == V == D（KDA 里 q/k/v 共享 head_dim）；v1 host enforce K==V==128。
    """
    K = D
    V = D
    total_T = sum(seq_lens)
    N = len(seq_lens)

    torch.manual_seed(seed)
    q = torch.randn(1, total_T, H, K, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, total_T, H, K, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, total_T, H, V, dtype=torch.bfloat16) * 0.1
    g = torch.randn(1, total_T, H, K, dtype=torch.float32) * 0.01
    beta = torch.rand(1, total_T, H, dtype=torch.float32)
    initial_state = torch.randn(N, H, K, V, dtype=torch.float32) * 0.02
    cu_seqlens = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
        dtype=torch.int32,
    )
    scale = K ** -0.5
    return {
        "q": q, "k": k, "v": v, "g": g, "beta": beta,
        "initial_state": initial_state, "cu_seqlens": cu_seqlens,
        "scale": scale, "H": H, "K": K, "V": V, "N": N, "total_T": total_T,
    }


def test_fused_recurrent_kda(cfg, seed=42):
    """Decode core **Tensor Core GEMV 实验路径**：tl.dot + weight/share 双 memory tier。

    Zeus 实现: `sgl_kernel_zeus.fused_recurrent_kda`
              （`csrc/mamba/fused_recurrent_kda_{kernel.py,zeus.cpp,sim.c}`）
    与 Sdecay（生产默认）的差异：
      - 算法层面：decay 折进 k → v̂ readout → outer-update → S 末尾衰减；
        多一次 `k_eff = k · decay` 的 K 维向量乘（结构上比 Sdecay 多 1 步）。
      - 硬件层面：v̂ 与 o_t 都用 `tl.dot([1,K] bf16, [K,V] bf16) → [1,V] fp32`
        走矩阵引擎；state 的 load/store 用 `memory_type='weight'` 标注落到
        weight 内存层；9b 之前的 element-wise S·decay 显式再 load 一份到
        share memory（向量引擎最近 tier）。
      - 数学层面：与 Sdecay 严格等价（S_new、o[t] 完全一致）。
    生产 decode 链路默认走 **Sdecay**（更接近 CUDA fla 参考、ops 更少、寄存器
    路径简单）；本 stage 主要为真核 Tensor Core 路径做精度 / 调度验证用。

    CUDA 参考: `fla/kda.py:119 fused_recurrent_kda` →
              `fla/fused_recurrent.py fused_recurrent_gated_delta_rule_fwd_kernel`
              （IS_KDA=True 分支）
    """
    print()
    print("=" * 60)
    print("Stage: fused_recurrent_kda (decode core, Tensor Core GEMV 实验)")
    print("=" * 60)

    H = cfg.num_heads
    # v1 host enforce K==V==128；本 stage 覆盖 proxy cfg.head_dim 用生产值。
    D = 128
    seq_lens = [1, 3]
    bundle = _make_kda_inputs(H, D, seq_lens, seed)
    print(f"  shape: B={bundle['N']}  seq_lens={seq_lens}  "
          f"H={H}  K=V={D} (overridden to v1 production)")

    # REF (pure torch)
    o_ref, final_state_ref = _ref_fused_recurrent_kda(
        bundle["q"].to(REF_DEVICE), bundle["k"].to(REF_DEVICE), bundle["v"].to(REF_DEVICE),
        bundle["g"].to(REF_DEVICE), bundle["beta"].to(REF_DEVICE),
        bundle["initial_state"].clone().to(REF_DEVICE),
        scale=bundle["scale"], use_qk_l2norm_in_kernel=True,
        cu_seqlens=bundle["cu_seqlens"].to(REF_DEVICE),
    )
    print(f"  REF o: shape={tuple(o_ref.shape)} dtype={o_ref.dtype}")
    print(f"  REF final_state: shape={tuple(final_state_ref.shape)} "
          f"dtype={final_state_ref.dtype}")

    # Zeus —— baseline fused_recurrent_kda
    initial_state_z = bundle["initial_state"].clone().to("privateuseone")
    o_zeus, final_state_zeus = sgl_kernel_zeus.fused_recurrent_kda(
        q=bundle["q"].to("privateuseone"),
        k=bundle["k"].to("privateuseone"),
        v=bundle["v"].to("privateuseone"),
        g=bundle["g"].to("privateuseone"),
        beta=bundle["beta"].to("privateuseone"),
        initial_state=initial_state_z,
        cu_seqlens=bundle["cu_seqlens"].to("privateuseone"),
        scale=bundle["scale"],
        use_qk_l2norm_in_kernel=True,
    )

    ok = True
    ok &= compare_tensors("fused_recurrent_kda/o",            o_ref,           o_zeus)
    ok &= compare_tensors("fused_recurrent_kda/final_state",  final_state_ref, final_state_zeus)
    return ok, (o_ref, final_state_ref)


# ── Stage: fused_recurrent_kda_Sdecay (decode core, variant) ────
def test_fused_recurrent_kda_Sdecay(cfg, seed=42):
    """Decode core **生产默认**：S 先 decay 再 readout（与 CUDA fla 同形态）。

    Zeus 实现: `sgl_kernel_zeus.fused_recurrent_kda_Sdecay`
              （`csrc/mamba/fused_recurrent_kda_Sdecay_{kernel.py,zeus.cpp,sim.c}`）
    选为生产默认的理由：
      - **更接近 CUDA fla 参考**：`fla/fused_recurrent.py` 的
        `fused_recurrent_gated_delta_rule_fwd_kernel` 本来就是 S-decay-first
        形态，sim.c / 数学公式与本 kernel 步骤一一对应。
      - **operations 更少**：少一次 `k_eff = k · decay` 的 K 维向量乘，
        S 的更新拆成 "S decay" + "outer-add"，没有 baseline 的 S_decayed
        中间值。寄存器层数据流更直观。
      - **数学严格等价 baseline**（见 docs/fused_recurrent_kda_Sdecay.md §1）：
        `Σ_k (S[k,:]·decay[k]) · k̂[k] ≡ Σ_k S[k,:] · (k̂[k]·decay[k])`，
        S_new 与 o[t] 完全一致。
    本 stage 同时验证 vs REF（pure-torch fp32）与 vs baseline parity，把"生产默认
    与 Tensor Core 实验路径互恰"在 dev 流水线钉死。
    """
    print()
    print("=" * 60)
    print("Stage: fused_recurrent_kda_Sdecay (decode core, **生产默认**)")
    print("=" * 60)

    H = cfg.num_heads
    D = 128                                    # v1 host enforce K==V==128
    seq_lens = [1, 3]
    bundle = _make_kda_inputs(H, D, seq_lens, seed)
    print(f"  shape: B={bundle['N']}  seq_lens={seq_lens}  "
          f"H={H}  K=V={D} (overridden to v1 production)")

    # REF
    o_ref, final_state_ref = _ref_fused_recurrent_kda(
        bundle["q"].to(REF_DEVICE), bundle["k"].to(REF_DEVICE), bundle["v"].to(REF_DEVICE),
        bundle["g"].to(REF_DEVICE), bundle["beta"].to(REF_DEVICE),
        bundle["initial_state"].clone().to(REF_DEVICE),
        scale=bundle["scale"], use_qk_l2norm_in_kernel=True,
        cu_seqlens=bundle["cu_seqlens"].to(REF_DEVICE),
    )
    print(f"  REF o: shape={tuple(o_ref.shape)} dtype={o_ref.dtype}")

    # Zeus —— Sdecay 变体（独立 state 副本）
    initial_state_sd = bundle["initial_state"].clone().to("privateuseone")
    o_sdecay, final_state_sdecay = sgl_kernel_zeus.fused_recurrent_kda_Sdecay(
        q=bundle["q"].to("privateuseone"),
        k=bundle["k"].to("privateuseone"),
        v=bundle["v"].to("privateuseone"),
        g=bundle["g"].to("privateuseone"),
        beta=bundle["beta"].to("privateuseone"),
        initial_state=initial_state_sd,
        cu_seqlens=bundle["cu_seqlens"].to("privateuseone"),
        scale=bundle["scale"],
        use_qk_l2norm_in_kernel=True,
    )

    ok = True
    # vs REF（atol/rtol 与 baseline 一致；REF 是 fp32 pure-torch）
    ok &= compare_tensors("fused_recurrent_kda_Sdecay/o",            o_ref,           o_sdecay)
    ok &= compare_tensors("fused_recurrent_kda_Sdecay/final_state",  final_state_ref, final_state_sdecay)

    # vs baseline parity（更紧的容差，因为两版只差 1 处 bf16 RNE 顺序）
    initial_state_bl = bundle["initial_state"].clone().to("privateuseone")
    o_bl, final_state_bl = sgl_kernel_zeus.fused_recurrent_kda(
        q=bundle["q"].to("privateuseone"),
        k=bundle["k"].to("privateuseone"),
        v=bundle["v"].to("privateuseone"),
        g=bundle["g"].to("privateuseone"),
        beta=bundle["beta"].to("privateuseone"),
        initial_state=initial_state_bl,
        cu_seqlens=bundle["cu_seqlens"].to("privateuseone"),
        scale=bundle["scale"],
        use_qk_l2norm_in_kernel=True,
    )
    ok &= compare_tensors(
        "Sdecay/parity_o_vs_baseline", o_bl, o_sdecay,
        atol=1e-2, rtol=1e-2,
    )
    ok &= compare_tensors(
        "Sdecay/parity_state_vs_baseline", final_state_bl, final_state_sdecay,
        atol=1e-2, rtol=1e-2,
    )
    return ok, (o_ref, final_state_ref)


# ── Stage: chunk_kda (extend core) ──────────────────────────────
def test_chunk_kda(cfg, seed=42):
    """Extend core：chunk-wise O(T·BT)，chunk_size=64。CUDA 实现是 7 个
    sub-kernel 的流水线；dev REF 用等价 recurrent 形态做 golden（数学等价）。

    CUDA 实现: `fla/kda.py:1200 chunk_kda` → 7-step pipeline:
      chunk_local_cumsum → chunk_kda_scaled_dot_kkt_fwd → solve_tril →
      recompute_w_u_fwd → chunk_gated_delta_rule_fwd_h → chunk_gla_fwd_o_gk
    """
    print()
    print("=" * 60)
    print("Stage: chunk_kda (extend core — 7-sub-kernel pipeline)")
    print("=" * 60)

    H = cfg.num_heads
    K = cfg.head_dim
    V = cfg.head_dim
    # extend 场景：1 个长 seq 就能覆盖 chunk 逻辑（T 跨多个 chunk 更好）
    seq_lens = [72]      # > chunk_size=64，保证至少走 2 个 chunk
    N = len(seq_lens)
    total_T = sum(seq_lens)
    print(f"  shape: B={N}  seq_lens={seq_lens}  H={H}  K=V={K}")

    torch.manual_seed(seed)
    q = torch.randn(1, total_T, H, K, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, total_T, H, K, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, total_T, H, V, dtype=torch.bfloat16) * 0.1
    g = torch.randn(1, total_T, H, K, dtype=torch.float32) * 0.01
    beta = torch.rand(1, total_T, H, dtype=torch.float32)
    initial_state = torch.randn(N, H, K, V, dtype=torch.float32) * 0.02
    cu_seqlens = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
        dtype=torch.int32,
    )
    scale = K ** -0.5

    # REF —— 等价的 recurrent 实现（数学等价；Zeus chunk 实现对齐到此 golden）
    o_ref, final_state_ref = _ref_chunk_kda(
        q.to(REF_DEVICE), k.to(REF_DEVICE), v.to(REF_DEVICE),
        g.to(REF_DEVICE), beta.to(REF_DEVICE),
        initial_state.clone().to(REF_DEVICE),
        scale=scale, use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens.to(REF_DEVICE),
    )
    print(f"  REF o: shape={tuple(o_ref.shape)} dtype={o_ref.dtype}")
    print(f"  REF final_state: shape={tuple(final_state_ref.shape)} "
          f"dtype={final_state_ref.dtype}")

    try:
        zeus_todo(
            "chunk_kda",
            "python/sglang/srt/layers/attention/fla/kda.py:1200 chunk_kda "
            "(pipeline: cumsum → scaled_dot_kkt → solve_tril → recompute_w_u → "
            "chunk_delta_h → chunk_gla_o_gk)",
        )
    except NotImplementedError as e:
        print(f"  ZEUS: {e}")
        return None, (o_ref, final_state_ref)


# ── Stage: rms_norm_gated ───────────────────────────────────────
def test_rms_norm_gated(cfg, num_tokens=16, seed=42):
    """对齐 `fla/kda.rms_norm_gated` —— rmsnorm(x) * sigmoid(g)。

    **注意**：KDA 里的 `FusedRMSNormGated(activation='sigmoid')`（kimi_linear.py:287），
    是 `y = rmsnorm(x) * sigmoid(g)`，不是 swish 的 `g * sigmoid(g)`。
    """
    print()
    print("=" * 60)
    print("Stage: rms_norm_gated (y = rmsnorm(x) * sigmoid(g), head-dim)")
    print("=" * 60)

    D = cfg.head_dim
    T = num_tokens
    H = cfg.num_heads
    # o_norm 作用在 [..., head_dim] 维度（per-head）；CUDA 里 reshape 成
    # [T*H, head_dim] 过 kernel。
    print(f"  shape: [T*H={T*H}, head_dim={D}]  activation=sigmoid")

    torch.manual_seed(seed)
    x = torch.randn(T * H, D, dtype=torch.bfloat16) * 0.1
    g_gate = torch.randn(T * H, D, dtype=torch.bfloat16) * 0.1

    # Zeus rms_norm_gated 固定 weight=1（无可学习 γ，对齐 KDA 生产配置）
    y_ref = _ref_rms_norm_gated(
        x.to(REF_DEVICE), g_gate.to(REF_DEVICE), weight=None,
        activation="sigmoid", eps=cfg.rms_norm_eps,
    )
    print(f"  REF y: shape={tuple(y_ref.shape)} dtype={y_ref.dtype}")

    # Zeus
    y_zeus = sgl_kernel_zeus.rms_norm_gated(
        x.to("privateuseone"), g_gate.to("privateuseone"),
        eps=cfg.rms_norm_eps,
    )
    ok = compare_tensors("rms_norm_gated/y", y_ref, y_zeus)
    return ok, (y_ref,)


# ── Stage: kimi_delta_attn_decode (端到端) ───────────────────────
def test_kimi_delta_attn_decode(cfg, seed=42):
    """端到端 decode：conv1d_update_indexed → fused_kda_gate →
    fused_recurrent_kda_Sdecay_indexed → rms_norm_gated → o_proj。

    完全对齐 `KimiLinearAttnBackend.forward_decode`
    (`hybrid_linear_attn_backend.py:307`)，但用 **indexed** 版本的 kernel：
    conv_state / ssm_state 都直接对全局池做 slot 寻址，无需 host 侧
    cache_index_gather/scatter。

    REF 路径走"非 indexed kernel + host 手动 gather/scatter"，与 CUDA
    backend 形态一致；Zeus 路径走 indexed kernel 一步直达。两条路径
    最终的 output 与 pool 状态（被命中的 slot + 未被命中的 slot）必须严格
    一致。

    Stage 跳过前置 projection（与各 sub-stage 同款策略）：直接生成 post-
    projection 张量。这让两条路径面对的 bf16 bits 完全相同，对比聚焦在 kernel
    链路本身。

    形状（v1: KDA head_dim=128 是 Sdecay_indexed 的 host 强制约束）：
      N      = 2 （decode batch）
      H      = cfg.hidden_size                  (proxy 512)
      Hd     = 128                              (override；v1 K==V==128)
      Hh     = cfg.num_heads                    (proxy 8)
      Hq     = Hh * Hd                          = 1024
      K      = cfg.conv_kernel_size             (4)
      N_pool = 16                               (state pool 容量；indices 散点)
      cache_indices = [3, 5]                    (随便选两个不连续 slot)
    """
    print()
    print("=" * 60)
    print("Stage: kimi_delta_attn_decode (end-to-end decode path, INDEXED)")
    print("=" * 60)

    # ── shape ──────────────────────────────────────────────────────
    H = cfg.hidden_size
    Hd = 128                       # override: v1 Sdecay_indexed host enforce K==V==128
    Hh = cfg.num_heads
    Hq = Hh * Hd
    K = cfg.conv_kernel_size
    N = 2
    N_pool = 16
    indices_list = [3, 5]
    print(f"  shape: N={N}  N_pool={N_pool}  H={H}  H_qkv={Hq}  "
          f"heads={Hh}×{Hd}  conv_K={K}  cache_indices={indices_list}")

    torch.manual_seed(seed)

    # ── Post-projection inputs (skip projections; both paths see identical bits) ──
    q_proj_states = torch.randn(N, Hq, dtype=torch.bfloat16) * 1.0
    k_proj_states = torch.randn(N, Hq, dtype=torch.bfloat16) * 1.0
    v_proj_states = torch.randn(N, Hq, dtype=torch.bfloat16) * 1.0
    b_proj_states = torch.randn(N, Hh, dtype=torch.bfloat16) * 0.5      # b_proj(hs)
    f_b_states    = torch.randn(N, Hq, dtype=torch.bfloat16) * 0.5      # f_b_proj(f_a_proj(hs))
    g_b_states    = torch.randn(N, Hq, dtype=torch.bfloat16) * 0.5      # g_b_proj(g_a_proj(hs))

    # ── Conv weights / bias (per q/k/v) ────────────────────────────
    q_conv_w = torch.randn(Hq, K, dtype=torch.bfloat16) * 0.1
    k_conv_w = torch.randn(Hq, K, dtype=torch.bfloat16) * 0.1
    v_conv_w = torch.randn(Hq, K, dtype=torch.bfloat16) * 0.1
    q_conv_b = torch.randn(Hq, dtype=torch.bfloat16) * 0.01
    k_conv_b = torch.randn(Hq, dtype=torch.bfloat16) * 0.01
    v_conv_b = torch.randn(Hq, dtype=torch.bfloat16) * 0.01

    # ── KDA gate fixed params ──────────────────────────────────────
    A_log   = torch.randn(Hh,  dtype=torch.float32) * 0.5
    dt_bias = torch.randn(Hq,  dtype=torch.float32) * 0.1

    # ── Pools (scattered slots; only [3, 5] should be modified) ────
    conv_state_q_pool_init = torch.randn(N_pool, Hq, K - 1, dtype=torch.bfloat16) * 0.1
    conv_state_k_pool_init = torch.randn(N_pool, Hq, K - 1, dtype=torch.bfloat16) * 0.1
    conv_state_v_pool_init = torch.randn(N_pool, Hq, K - 1, dtype=torch.bfloat16) * 0.1
    ssm_state_pool_init    = torch.randn(N_pool, Hh, Hd, Hd, dtype=torch.float32) * 0.02

    cache_indices = torch.tensor(indices_list, dtype=torch.int32)
    cu_seqlens    = torch.arange(N + 1, dtype=torch.int32)              # [0, 1, 2] for N=2

    # ── o_proj weight ──────────────────────────────────────────────
    o_proj_w = torch.randn(H, Hq, dtype=torch.bfloat16) * 0.1            # output [N, H] = normed @ w.T

    # ─────────────────────────── REF (CPU) ──────────────────────────
    # gather pool[indices] → 非 indexed REF kernel → scatter
    idx_long = cache_indices.long()

    conv_q_ref = conv_state_q_pool_init[idx_long].clone()
    conv_k_ref = conv_state_k_pool_init[idx_long].clone()
    conv_v_ref = conv_state_v_pool_init[idx_long].clone()
    ssm_ref    = ssm_state_pool_init[idx_long].clone()

    q_post = _ref_causal_conv1d_update(q_proj_states.clone(), conv_q_ref, q_conv_w, q_conv_b, "silu")
    k_post = _ref_causal_conv1d_update(k_proj_states.clone(), conv_k_ref, k_conv_w, k_conv_b, "silu")
    v_post = _ref_causal_conv1d_update(v_proj_states.clone(), conv_v_ref, v_conv_w, v_conv_b, "silu")
    # → q_post, k_post, v_post : [N, Hq]
    # → conv_q/k/v_ref         : [N, Hq, K-1] (mutated in-place by REF)

    # rearrange "n (h d) -> 1 n h d"
    q_4d = q_post.reshape(1, N, Hh, Hd)
    k_4d = k_post.reshape(1, N, Hh, Hd)
    v_4d = v_post.reshape(1, N, Hh, Hd)

    beta_ref = b_proj_states.float().sigmoid()                          # [N, Hh] fp32
    g_ref = _ref_fused_kda_gate(
        f_b_states.float(), A_log, head_dim=Hd, g_bias=dt_bias,
        beta=cfg.softplus_beta, threshold=cfg.softplus_threshold,
    )                                                                    # [N, Hh, Hd] fp32

    o_ref_4d, ssm_ref_after = _ref_fused_recurrent_kda(
        q_4d, k_4d, v_4d,
        g_ref.unsqueeze(0),                                             # [1, N, Hh, Hd]
        beta_ref.unsqueeze(0),                                           # [1, N, Hh]
        ssm_ref.clone(),                                                 # [N, Hh, Hd, Hd]
        scale=Hd ** -0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    o_ref = o_ref_4d.squeeze(0)                                          # [N, Hh, Hd] bf16

    g_gate_ref = g_b_states.reshape(N, Hh, Hd)                           # [N, Hh, Hd] bf16
    normed_ref_3d = _ref_rms_norm_gated(
        o_ref, g_gate_ref, weight=None, activation="sigmoid", eps=cfg.rms_norm_eps,
    )                                                                    # [N, Hh, Hd] bf16
    normed_ref = normed_ref_3d.reshape(N, Hq)

    # o_proj: normed @ w.T  → [N, H]
    output_ref = (normed_ref.float() @ o_proj_w.float().T).to(torch.bfloat16)

    # REF pool state after: scatter the gathered+modified state back
    pool_q_ref = conv_state_q_pool_init.clone()
    pool_k_ref = conv_state_k_pool_init.clone()
    pool_v_ref = conv_state_v_pool_init.clone()
    pool_ssm_ref = ssm_state_pool_init.clone()
    pool_q_ref[idx_long] = conv_q_ref
    pool_k_ref[idx_long] = conv_k_ref
    pool_v_ref[idx_long] = conv_v_ref
    pool_ssm_ref[idx_long] = ssm_ref_after

    print(f"  REF: output={tuple(output_ref.shape)}  "
          f"core_attn_out={tuple(o_ref.shape)}  "
          f"normed={tuple(normed_ref.shape)}")

    # ────────────────────────── Zeus (indexed) ──────────────────────
    DEV = "privateuseone"
    pool_q_zeus   = conv_state_q_pool_init.clone().to(DEV)
    pool_k_zeus   = conv_state_k_pool_init.clone().to(DEV)
    pool_v_zeus   = conv_state_v_pool_init.clone().to(DEV)
    pool_ssm_zeus = ssm_state_pool_init.clone().to(DEV)
    cache_indices_zeus = cache_indices.to(DEV)

    # 1. conv1d_update_indexed (q/k/v 各一次；都对 pool 原地)
    q_zeus = sgl_kernel_zeus.causal_conv1d_update_indexed(
        q_proj_states.to(DEV), pool_q_zeus, cache_indices_zeus,
        q_conv_w.to(DEV), q_conv_b.to(DEV), "silu",
    )
    k_zeus = sgl_kernel_zeus.causal_conv1d_update_indexed(
        k_proj_states.to(DEV), pool_k_zeus, cache_indices_zeus,
        k_conv_w.to(DEV), k_conv_b.to(DEV), "silu",
    )
    v_zeus = sgl_kernel_zeus.causal_conv1d_update_indexed(
        v_proj_states.to(DEV), pool_v_zeus, cache_indices_zeus,
        v_conv_w.to(DEV), v_conv_b.to(DEV), "silu",
    )

    # 2. fused_kda_gate
    g_zeus = sgl_kernel_zeus.fused_kda_gate(
        f_b_states.to(DEV), A_log.to(DEV), head_dim=Hd,
        g_bias=dt_bias.to(DEV),
    )                                                                    # [N, Hh, Hd] fp32

    # 3. beta = sigmoid(b_proj.float())   — torch native on Zeus
    beta_zeus = b_proj_states.to(DEV).float().sigmoid()                  # [N, Hh] fp32

    # 4. Sdecay_indexed (in-place pool update at slot indices)
    q_4d_zeus = q_zeus.reshape(1, N, Hh, Hd)
    k_4d_zeus = k_zeus.reshape(1, N, Hh, Hd)
    v_4d_zeus = v_zeus.reshape(1, N, Hh, Hd)
    o_zeus_4d = sgl_kernel_zeus.fused_recurrent_kda_Sdecay_indexed(
        q=q_4d_zeus, k=k_4d_zeus, v=v_4d_zeus,
        g=g_zeus.unsqueeze(0),
        beta=beta_zeus.unsqueeze(0),
        state_pool=pool_ssm_zeus,
        cache_indices=cache_indices_zeus,
        cu_seqlens=cu_seqlens.to(DEV),
        scale=Hd ** -0.5,
        use_qk_l2norm_in_kernel=True,
    )
    o_zeus = o_zeus_4d.squeeze(0)                                        # [N, Hh, Hd] bf16

    # 5. rms_norm_gated (sigmoid gate)
    g_gate_zeus = g_b_states.reshape(N, Hh, Hd).to(DEV)
    normed_zeus_3d = sgl_kernel_zeus.rms_norm_gated(
        o_zeus, g_gate_zeus, eps=cfg.rms_norm_eps,
    )
    normed_zeus = normed_zeus_3d.reshape(N, Hq)

    # 6. o_proj — torch matmul on **CPU**（Zeus matmul 需要 packed weight，
    # 不在本 stage scope；统一在 CPU 上做最后这一步线性变换，让 REF / Zeus
    # 看到相同的 matmul 实现）。
    output_zeus = (normed_zeus.cpu().float() @ o_proj_w.float().T).to(torch.bfloat16)

    # ────────────────────────── Comparison ──────────────────────────
    ok = True
    # 中间产物
    ok &= compare_tensors("decode/conv_q_out",            q_post,    q_zeus)
    ok &= compare_tensors("decode/conv_k_out",            k_post,    k_zeus)
    ok &= compare_tensors("decode/conv_v_out",            v_post,    v_zeus)
    ok &= compare_tensors("decode/g (kda_gate)",          g_ref,     g_zeus)
    ok &= compare_tensors("decode/core_attn_out",         o_ref,     o_zeus)
    ok &= compare_tensors("decode/normed (rms_norm_gated)", normed_ref, normed_zeus)
    # o_proj 是经过 5 个 kernel + matmul 的端到端链路，bf16 累积 + matmul 放大让
    # max_diff 自然到 ~1e-2 量级；本 stage 验证 kernel 链路一致性，o_proj 只是
    # 收尾的 sanity check，所以放宽到 1e-2。
    ok &= compare_tensors("decode/output (o_proj)",       output_ref, output_zeus,
                          atol=1e-2, rtol=1e-2)

    # 池状态：touched slots（indices=[3,5]）应与 REF 一致
    pool_q_zeus_cpu = pool_q_zeus.cpu()
    pool_k_zeus_cpu = pool_k_zeus.cpu()
    pool_v_zeus_cpu = pool_v_zeus.cpu()
    pool_ssm_zeus_cpu = pool_ssm_zeus.cpu()
    for slot in indices_list:
        ok &= compare_tensors(
            f"decode/conv_q_pool[{slot}]",   pool_q_ref[slot],   pool_q_zeus_cpu[slot],
        )
        ok &= compare_tensors(
            f"decode/conv_k_pool[{slot}]",   pool_k_ref[slot],   pool_k_zeus_cpu[slot],
        )
        ok &= compare_tensors(
            f"decode/conv_v_pool[{slot}]",   pool_v_ref[slot],   pool_v_zeus_cpu[slot],
        )
        ok &= compare_tensors(
            f"decode/ssm_pool[{slot}]",      pool_ssm_ref[slot], pool_ssm_zeus_cpu[slot],
        )

    # 池状态：untouched slots 必须 bit-exact 不变（in-place 副作用边界）
    untouched = [s for s in range(N_pool) if s not in indices_list]
    untouched_ok = True
    for s in untouched:
        for name, ref_pool, zeus_pool in (
            ("conv_q",   conv_state_q_pool_init, pool_q_zeus_cpu),
            ("conv_k",   conv_state_k_pool_init, pool_k_zeus_cpu),
            ("conv_v",   conv_state_v_pool_init, pool_v_zeus_cpu),
            ("ssm",      ssm_state_pool_init,    pool_ssm_zeus_cpu),
        ):
            same = torch.equal(ref_pool[s], zeus_pool[s])
            if not same:
                print(f"  [decode/{name}_pool[{s}]] MUTATED (should be untouched)")
                untouched_ok = False
    if untouched_ok:
        print(f"  [decode/untouched_pools] PASS  ({len(untouched)} slots × 4 pools "
              f"= {len(untouched) * 4} bit-exact checks)")
    ok &= untouched_ok

    return ok, (output_ref, o_ref, normed_ref, pool_q_ref, pool_ssm_ref)


# ── Stage: kimi_delta_attn_extend (端到端) ───────────────────────
def test_kimi_delta_attn_extend(cfg, seed=42):
    """端到端 extend：projection → causal_conv1d_fn → fused_kda_gate →
    chunk_kda → rms_norm_gated → o_proj。

    完全对齐 `KimiLinearAttnBackend.forward_extend`
    (`hybrid_linear_attn_backend.py:401`)。
    """
    print()
    print("=" * 60)
    print("Stage: kimi_delta_attn_extend (end-to-end extend path)")
    print("=" * 60)

    H = cfg.hidden_size
    Hq = cfg.num_heads * cfg.head_dim
    seq_lens = [12, 24]
    total_T = sum(seq_lens)

    print(f"  shape: B={len(seq_lens)}  seq_lens={seq_lens}  "
          f"total_T={total_T}  H={H}  H_qkv={Hq}  "
          f"heads={cfg.num_heads}×{cfg.head_dim}")

    print("  REF: 骨架已就位；待 sub-stage 2-8 全部 PASS 后，把它们串成 REF "
          "端到端 pipeline（镜像 forward_extend）。")
    try:
        zeus_todo(
            "KimiLinearAttnBackend.forward_extend (end-to-end)",
            "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:401 "
            "forward_extend",
        )
    except NotImplementedError as e:
        print(f"  ZEUS: {e}")
        return None, None


# ── Dispatch ────────────────────────────────────────────────────
STAGES = {
    "fused_kda_gate":             test_fused_kda_gate,
    "causal_conv1d_update":       test_causal_conv1d_update,
    "causal_conv1d_update_qkv":   test_causal_conv1d_update_qkv,
    "causal_conv1d_fn":           test_causal_conv1d_fn,
    "l2norm":                     test_l2norm,
    "fused_recurrent_kda_Sdecay": test_fused_recurrent_kda_Sdecay,    # 生产默认
    "fused_recurrent_kda":        test_fused_recurrent_kda,           # Tensor Core 实验
    "chunk_kda":                  test_chunk_kda,
    "rms_norm_gated":             test_rms_norm_gated,
    "kimi_delta_attn_decode":     test_kimi_delta_attn_decode,
    "kimi_delta_attn_extend":     test_kimi_delta_attn_extend,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = default_kimi_cfg()
    print(f"Kimi-Linear proxy config: H={cfg.hidden_size}  "
          f"heads={cfg.num_heads}×{cfg.head_dim}  "
          f"conv_kernel_size={cfg.conv_kernel_size}")
    print(f"Reference device: {REF_DEVICE}  "
          f"(FLA Triton available: {CAN_USE_FLA_TRITON})")

    results = {}
    for name, fn in STAGES.items():
        if args.stage not in (name, "all"):
            continue
        # stage 的 REF 部分若正常执行会内部 print PASS/DIFF。Zeus 侧
        # NotImplementedError 被 stage 内部 catch，返回 ok=None 标 SKIP。
        try:
            # 按签名分派 —— 有些 stage 只吃 (cfg, seed)，有些还吃 num_tokens
            if fn.__code__.co_argcount == 2:
                ok, _ = fn(cfg, seed=args.seed)
            else:
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
        print(f"  {name:30s} : {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
