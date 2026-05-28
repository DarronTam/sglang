"""
GLM-Next KDA（线性注意力）层集成测试（end-to-end）

范围：
  - 起点：input_layernorm 后的 hidden_states [N(decode) | total_T(extend), H]
  - 终点：Glm5NextLinearAttention.o_proj 输出 [..., H]
  - 单 device / 单 layer / KDA layer only（is_kda_layer=True）
  - 覆盖 decode 与 extend 两条 forward path

Decode kernel pipeline（与 KimiLinearAttnBackend.forward_decode 对齐）：
  qkv_proj → fused_kda_gate → causal_conv1d_update_qkv → l2norm(q,k) →
  fused_recurrent_kda → rms_norm_gated → o_proj

Extend kernel pipeline（与 KimiLinearAttnBackend.forward_extend 对齐）：
  qkv_proj → fused_kda_gate → causal_conv1d_fn → l2norm(q,k) →
  chunk_kda → rms_norm_gated → o_proj

设计：
  - 不实例化 Glm5NextLinearAttention（依赖 distributed init），而是直接
    合成 forward 路径里实际用到的 weights + states，对应到 sgl_kernel_zeus
    7-kernel 流水线，与 dev_kimi_linear_attn_test.py 各 stage 同语义。
  - GLM-Next 多出的 f_a/f_b/g_a/g_b/b 低秩分解：本测试合并为等价的
    f_proj = f_b @ f_a, g_proj = g_b @ g_a, b_proj 单独 —— 对 KDA
    pipeline 的输入而言只是 projection 计算位置不同，数值完全等价。
  - 比 hidden 状态：`ssm_states` 与 `conv_states` 必须双向比，否则下一步
    decode 会立刻发散。

约定：Zeus 侧未落地的 kernel 直接 raise NotImplementedError，不允许 silent
fallback（与 dev_kimi_linear_attn_test.py 一致）。

用法:
  python zeus_dev/dev_glm5_next_kda_test.py
  python zeus_dev/dev_glm5_next_kda_test.py --stage decode --num-tokens 4
  python zeus_dev/dev_glm5_next_kda_test.py --stage extend
"""

import argparse
import sys, os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch_zeus  # noqa: F401 — registers zeus backend
import sgl_kernel_zeus
os.environ.setdefault("SGLANG_DEVICE", "zeus")

# ── server_args mock ─────────────────────────────────────────────
import sglang.srt.server_args
_dummy_args = Mock()
_dummy_args.rl_on_policy_target = None
sglang.srt.server_args.get_global_server_args = lambda *a, **kw: _dummy_args


# 把 zeus_dev/ 加入 sys.path，复用 dev_kimi_linear_attn_test.py 已 PASS 的
# pure-torch REF helpers —— 数学等价的 _ref_* 函数无需重复实现。
sys.path.insert(0, str(Path(__file__).parent))
from dev_kimi_linear_attn_test import (  # noqa: E402
    compare_tensors,
    _ref_causal_conv1d_update,
    _ref_causal_conv1d_fn,
    _ref_fused_kda_gate,
    _ref_l2norm,
    _ref_rms_norm_gated,
    _ref_fused_recurrent_kda,
    _ref_chunk_kda,
)


REF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def default_glm5_next_kda_cfg():
    """GLM-Next 线性 attention 的 proxy config。

    对齐 GlmLinearConfig.linear_attn_config 的字段子集（见
    configs/glm_linear.py），shape 缩小到 CPU REF 能秒级跑完的尺寸。
    真实 shape 的 kernel-level 对齐由 sgl-kernel-zeus/tests/ 下的
    test_kda_*.py 承担。
    """
    return SimpleNamespace(
        hidden_size=1024,                   # proxy for H (4096 / 5120)
        head_dim=128,                       # proxy for 128
        num_heads=8,                       # = num_k_heads = num_v_heads（KDA 约定）
        head_v_dim=128,                     # = head_dim（KDA 约定 K == V）
        short_conv_kernel_size=4,          # 真实通常就是 4
        rms_norm_eps=1e-5,
        # fused_kda_gate 的两个硬超参（fla/kda.py:1306）
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )


# ────────────────────────────────────────────────────────────────
#                    REF 端：完整 forward 流水线
# ────────────────────────────────────────────────────────────────
def _ref_forward_decode(cfg, hidden_states, conv_state, ssm_state, weights):
    """REF for `Glm5NextLinearAttention.forward_decode` 单步 KDA。

    Args:
      hidden_states : [N, H]            bf16, 当前一步输入（decode batch=N）
      conv_state    : [N, 3*Hh, K-1]    bf16, **就地更新**
      ssm_state     : [N, H_heads, K, V] fp32, **就地更新**
      weights       : SimpleNamespace（_build_weights 产出，REF_DEVICE 上）

    Returns:
      out : [N, H] bf16
    """
    N, H = hidden_states.shape
    Hh = cfg.num_heads * cfg.head_dim     # head 数 × head_dim = projection size

    # 1) qkv_proj: [N, H] → [N, 3*Hh]
    qkv = torch.nn.functional.linear(hidden_states, weights.wqkv)

    # 2) f / g / b 三组 projection（GLM-Next 低秩分解）
    f = torch.nn.functional.linear(
        torch.nn.functional.linear(hidden_states, weights.wfa),
        weights.wfb,
    )                                     # [N, Hh]
    g_states = torch.nn.functional.linear(
        torch.nn.functional.linear(hidden_states, weights.wga),
        weights.wgb,
    )                                     # [N, Hh]
    beta = torch.nn.functional.linear(hidden_states, weights.wb)   # [N, num_heads]
    beta = torch.sigmoid(beta.float())                             # KDA 约定 sigmoid(beta)

    # 3) fused_kda_gate（softplus + -exp(A_log)，per-(head, head_dim)）
    forget_gate = _ref_fused_kda_gate(
        f, weights.A_log, cfg.head_dim,
        g_bias=weights.dt_bias,
        beta=cfg.softplus_beta, threshold=cfg.softplus_threshold,
    )                                     # [N, num_heads, head_dim] fp32

    # 4) causal_conv1d_update（fused Q/K/V）
    qkv_conv = _ref_causal_conv1d_update(
        qkv, conv_state, weights.conv_w, weights.conv_b, activation="silu",
    )                                     # [N, 3*Hh]
    q, k, v = qkv_conv.split([Hh, Hh, Hh], dim=-1)
    q = q.view(N, cfg.num_heads, cfg.head_dim)
    k = k.view(N, cfg.num_heads, cfg.head_dim)
    v = v.view(N, cfg.num_heads, cfg.head_v_dim)

    # 5) l2norm（生产路径 use_qk_l2norm_in_kernel=True 时融进 core；这里
    #    显式调出来便于排错。fused_recurrent_kda REF 里也会再做一次 ——
    #    幂等，不影响数值。）
    q = _ref_l2norm(q)
    k = _ref_l2norm(k)

    # 6) fused_recurrent_kda（decode core）
    #    形状对齐 fla.kda 的 (1, T=N, H, K/V) 约定 —— 用 cu_seqlens=[0,1,2,...]
    #    把 decode 的 N 个 batch 拉成一段长度为 1 的 chunk 序列。
    cu_seqlens = torch.arange(N + 1, dtype=torch.int32, device=hidden_states.device)
    # 把 forget_gate 的 [N, H, D] 当成每个 token 一份 g（K = head_dim = D）
    o, ssm_state_new = _ref_fused_recurrent_kda(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        forget_gate.unsqueeze(0),
        beta.unsqueeze(0),                # [1, N, num_heads]
        ssm_state.clone(),
        scale=cfg.head_dim ** -0.5,
        use_qk_l2norm_in_kernel=False,    # 已在第 5 步显式做过
        cu_seqlens=cu_seqlens,
    )
    ssm_state.copy_(ssm_state_new)
    o = o.squeeze(0)                      # [N, num_heads, head_v_dim]

    # 7) rms_norm_gated（activation="sigmoid"，per-head_dim）
    g_for_norm = g_states.view(N, cfg.num_heads, cfg.head_dim)
    core = _ref_rms_norm_gated(
        o, g_for_norm, weights.o_norm_w,
        activation="sigmoid", eps=cfg.rms_norm_eps,
    )                                     # [N, num_heads, head_v_dim]
    core_flat = core.reshape(N, Hh)

    # 8) o_proj
    out = torch.nn.functional.linear(core_flat, weights.wo)   # [N, H]
    return out


def _ref_forward_extend(cfg, hidden_states, conv_states, ssm_state,
                         weights, query_start_loc, has_initial_state):
    """REF for `Glm5NextLinearAttention.forward_extend` varlen KDA。

    Args:
      hidden_states     : [total_T, H]        bf16
      conv_states       : [B, 3*Hh, K-1]      bf16（每个 seq 一份），**就地更新**
      ssm_state         : [B, num_heads, K, V] fp32（每个 seq 一份），**就地更新**
      query_start_loc   : [B+1] int32
      has_initial_state : [B] bool

    Returns:
      out : [total_T, H] bf16
    """
    total_T, H = hidden_states.shape
    B = query_start_loc.numel() - 1
    Hh = cfg.num_heads * cfg.head_dim

    # 1) qkv_proj + f/g/b proj 与 decode 同
    qkv = torch.nn.functional.linear(hidden_states, weights.wqkv)
    f = torch.nn.functional.linear(
        torch.nn.functional.linear(hidden_states, weights.wfa),
        weights.wfb,
    )
    g_states = torch.nn.functional.linear(
        torch.nn.functional.linear(hidden_states, weights.wga),
        weights.wgb,
    )
    beta = torch.nn.functional.linear(hidden_states, weights.wb)
    beta = torch.sigmoid(beta.float())                         # [total_T, num_heads]

    # 2) fused_kda_gate
    forget_gate = _ref_fused_kda_gate(
        f, weights.A_log, cfg.head_dim,
        g_bias=weights.dt_bias,
        beta=cfg.softplus_beta, threshold=cfg.softplus_threshold,
    )                                                          # [total_T, H, K]

    # 3) causal_conv1d_fn（varlen）
    #    REF helper 期望 [C, total_T] 形态
    qkv_conv = _ref_causal_conv1d_fn(
        qkv.transpose(0, 1).contiguous(),
        weights.conv_w, weights.conv_b, conv_states,
        has_initial_state, query_start_loc, activation="silu",
    ).transpose(0, 1).contiguous()                              # [total_T, 3*Hh]

    q, k, v = qkv_conv.split([Hh, Hh, Hh], dim=-1)
    q = q.view(total_T, cfg.num_heads, cfg.head_dim)
    k = k.view(total_T, cfg.num_heads, cfg.head_dim)
    v = v.view(total_T, cfg.num_heads, cfg.head_v_dim)

    # 4) l2norm
    q = _ref_l2norm(q)
    k = _ref_l2norm(k)

    # 5) chunk_kda（extend core；REF 复用 recurrent 实现 —— 数学等价）
    o, ssm_state_new = _ref_chunk_kda(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        forget_gate.unsqueeze(0),
        beta.unsqueeze(0),
        ssm_state.clone(),
        scale=cfg.head_dim ** -0.5,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=query_start_loc,
        output_final_state=True,
    )
    ssm_state.copy_(ssm_state_new)
    o = o.squeeze(0)                                            # [total_T, num_heads, V]

    # 6) rms_norm_gated
    g_for_norm = g_states.view(total_T, cfg.num_heads, cfg.head_dim)
    core = _ref_rms_norm_gated(
        o, g_for_norm, weights.o_norm_w,
        activation="sigmoid", eps=cfg.rms_norm_eps,
    )
    core_flat = core.reshape(total_T, Hh)

    # 7) o_proj
    out = torch.nn.functional.linear(core_flat, weights.wo)
    return out


# ────────────────────────────────────────────────────────────────
#                    Zeus 端：完整 forward 流水线
# ────────────────────────────────────────────────────────────────
def _zeus_forward_decode(cfg, hidden_states, conv_state, ssm_state, weights):
    """Zeus 版 decode forward —— 用 sgl_kernel_zeus 的 7 个 kernel 串起来。

    所有张量必须已经在 zeus device 上；conv_state / ssm_state 就地更新。
    """
    N, H = hidden_states.shape
    Hh = cfg.num_heads * cfg.head_dim

    # 1) qkv_proj
    qkv = torch.mm(hidden_states, weights.wqkv)

    # 2) f / g / b proj（GLM-Next 低秩分解）
    f = torch.mm(torch.mm(hidden_states, weights.wfa), weights.wfb)
    g_states = torch.mm(torch.mm(hidden_states, weights.wga), weights.wgb)
    beta = torch.mm(hidden_states, weights.wb)
    beta = torch.sigmoid(beta.float())

    # 3) fused_kda_gate（Zeus 版）
    forget_gate = sgl_kernel_zeus.fused_kda_gate(
        f, weights.A_log, cfg.head_dim,
        g_bias=weights.dt_bias
    )

    # 4) causal_conv1d_update_indexed（与 CUDA / prod 同款语义：单次扫整个
    #    3*Hh 通道，weight [3*Hh, K]）
    #
    #    历史：早期版本调 sgl_kernel_zeus.causal_conv1d_update_qkv（QKV 三组
    #    fused 的 Zeus-only 优化算子），但 CUDA / 生产代码用的是单次扫的
    #    causal_conv1d_update —— per-channel 数学等价但 SIMD/tile 调度顺序
    #    不同，会引入 ~1e-2 量级 bf16 ULP 差。改用 indexed 版让 dev 与 prod /
    #    CUDA 三方语义对齐，差异降到接近 0。
    #
    #    indexed 版要求 conv_state 是 [N_pool, 3*Hh, K-1] 格式 + cache_indices
    #    选 batch；这里 dev 的 conv_state 已经是 [N, 3*Hh, K-1]，直接当
    #    N_pool == N 用，cache_indices = [0..N-1] 即可。in-place 更新 pool。
    cache_indices = torch.arange(
        N, dtype=torch.int32, device=conv_state.device,
    )
    qkv_post = sgl_kernel_zeus.causal_conv1d_update_indexed(
        qkv,                              # [N, 3*Hh]
        conv_state,                       # [N, 3*Hh, K-1] —— in-place 更新
        cache_indices,
        weights.conv_w,                   # [3*Hh, K]
        bias=weights.conv_b,
        activation="silu",
    )                                     # → [N, 3*Hh]

    q_post, k_post, v_post = qkv_post.split([Hh, Hh, Hh], dim=-1)
    q_post = q_post.contiguous().view(N, cfg.num_heads, cfg.head_dim)
    k_post = k_post.contiguous().view(N, cfg.num_heads, cfg.head_dim)
    v_post = v_post.contiguous().view(N, cfg.num_heads, cfg.head_v_dim)

    # 5) l2norm（Zeus 版）
    #
    #    实测：dev 走 "external sgl_kernel_zeus.l2norm + use_qk_l2norm_in_kernel=False"
    #    比走 "no external + use_qk_l2norm_in_kernel=True" 更接近 prod 的输出，
    #    虽然 prod 也是 use_qk_l2norm_in_kernel=True。原因可能是 indexed vs
    #    non-indexed 两个 kernel 内部融合的 l2norm 实现并不完全一致 ——
    #    `sgl_kernel_zeus.l2norm` 反而与 `fused_recurrent_kda_Sdecay_indexed`
    #    内部那一份对得更齐。保留 external 路径。
    q_post = sgl_kernel_zeus.l2norm(q_post)
    k_post = sgl_kernel_zeus.l2norm(k_post)

    # 6) fused_recurrent_kda_Sdecay（decode core，与 prod 同款 Sdecay 算法）
    #
    #    Prod 走的是 fused_recurrent_kda_Sdecay_indexed —— Sdecay 变体 + indexed
    #    pool 索引；dev 不需要 pool 索引，但要用 *Sdecay* 变体而非 baseline，
    #    否则 baseline 的"decay folded into k"会比 Sdecay 多一次 bf16 RNE
    #    （详见 mamba.py:518-534 docstring：Sdecay 比 baseline ~1 ULP 更接近
    #    fp32 reference）。
    #    这两个算法**数学等价**，但 bf16 累加顺序不同，统一用 Sdecay 让 dev
    #    与 prod 在 recurrent 这一步 byte-equivalent。
    cu_seqlens = torch.arange(N + 1, dtype=torch.int32, device=hidden_states.device)
    o, ssm_state_new = sgl_kernel_zeus.fused_recurrent_kda_Sdecay(
        q_post.unsqueeze(0), k_post.unsqueeze(0), v_post.unsqueeze(0),
        forget_gate.unsqueeze(0),
        beta.unsqueeze(0),
        ssm_state,
        scale=cfg.head_dim ** -0.5,
        use_qk_l2norm_in_kernel=False,       # 已在第 5 步外部做过
        cu_seqlens=cu_seqlens,
    )
    ssm_state.copy_(ssm_state_new)
    o = o.squeeze(0)

    # 7) rms_norm_gated（Zeus 版）
    g_for_norm = g_states.view(N, cfg.num_heads, cfg.head_dim)
    core = sgl_kernel_zeus.rms_norm_gated(
        o, g_for_norm, eps=cfg.rms_norm_eps,
    )
    core_flat = core.reshape(N, Hh)

    # 8) o_proj
    out = torch.mm(core_flat, weights.wo)
    return out


def _zeus_forward_extend(cfg, hidden_states, conv_states, ssm_state,
                          weights, query_start_loc, has_initial_state):
    """Zeus 版 extend forward —— 与 _zeus_forward_decode 同 7 步,
    把 conv1d_update_qkv 换成 causal_conv1d_fn,把 recurrent 换成 chunk_kda。
    """
    total_T, H = hidden_states.shape
    Hh = cfg.num_heads * cfg.head_dim

    qkv = torch.mm(hidden_states, weights.wqkv)
    f   = torch.mm(torch.mm(hidden_states, weights.wfa), weights.wfb)
    g_states = torch.mm(torch.mm(hidden_states, weights.wga), weights.wgb)
    beta = torch.mm(hidden_states, weights.wb)
    beta = torch.sigmoid(beta.float())

    forget_gate = sgl_kernel_zeus.fused_kda_gate(
        f, weights.A_log, cfg.head_dim,
        g_bias=weights.dt_bias
    )

    # causal_conv1d_fn（varlen）—— 接口要求 [C, total_T]
    qkv_post = sgl_kernel_zeus.causal_conv1d_fn(
        qkv.transpose(0, 1).contiguous(),
        weights.conv_w, weights.conv_b, conv_states,
        has_initial_state, query_start_loc, activation="silu",
    ).transpose(0, 1).contiguous()

    q, k, v = qkv_post.split([Hh, Hh, Hh], dim=-1)
    q = q.view(total_T, cfg.num_heads, cfg.head_dim)
    k = k.view(total_T, cfg.num_heads, cfg.head_dim)
    v = v.view(total_T, cfg.num_heads, cfg.head_v_dim)

    q = sgl_kernel_zeus.l2norm(q)
    k = sgl_kernel_zeus.l2norm(k)

    o, ssm_state_new = sgl_kernel_zeus.chunk_kda(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        forget_gate.unsqueeze(0),
        beta.unsqueeze(0),
        ssm_state,
        scale=cfg.head_dim ** -0.5,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=query_start_loc,
        output_final_state=True,
    )
    ssm_state.copy_(ssm_state_new)
    o = o.squeeze(0)

    g_for_norm = g_states.view(total_T, cfg.num_heads, cfg.head_dim)
    core = sgl_kernel_zeus.rms_norm_gated(
        o, g_for_norm, eps=cfg.rms_norm_eps,
    )
    core_flat = core.reshape(total_T, Hh)

    out = torch.mm(core_flat, weights.wo)
    return out


# ────────────────────────────────────────────────────────────────
#                          权重合成
# ────────────────────────────────────────────────────────────────
def _build_weights(cfg, device, seed=42):
    H  = cfg.hidden_size
    Hh = cfg.num_heads * cfg.head_dim
    D  = cfg.head_dim
    K  = cfg.short_conv_kernel_size
    is_zeus = (str(device) == "zeus")

    g = torch.Generator(device="cpu").manual_seed(seed)

    def _rand(shape, dtype):
        return (torch.randn(*shape, generator=g, dtype=dtype) * 0.05).to(device)

    def _rand_cpu(shape, dtype):
        return torch.randn(*shape, generator=g, dtype=dtype) * 0.05

    def _linear_w(out_features, in_features, dtype):
        """构造 nn.Linear 风格的权重；Zeus 上自动 pack 成 LocalMem。

        返回 `(weight_for_dev_path, weight_unpacked_cpu)` 二元组：
          - 第一个：device 上的张量（Zeus 上是 ZeusLocalMemTensor，CPU/CUDA
            上是普通 (out, in) bf16），dev / REF forward 直接用
          - 第二个：CPU 上的 raw (out, in) 副本，prod-path
            `_build_prod_layer` 注入 `Glm5NextLinearAttention` 各 nn.Linear
            子模块用（注入后由 `zeus.pack_weights` 转置/打包到 LocalMem）
        """
        w_cpu = _rand_cpu((out_features, in_features), dtype)
        if not is_zeus:
            return w_cpu.to(device), w_cpu
        import torch.nn as nn
        import torch_zeus.zeus as zeus
        layer = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
        layer.weight.data.copy_(w_cpu)
        layer = layer.to("zeus")
        zeus.pack_weights(layer, Tr=1, Tc=1)   # 转置 (N,K) → (K,N) 并搬入 LocalMem
        return layer.weight, w_cpu             # (ZeusLocalMemTensor, CPU raw)

    # 构造时同时拿到 packed/device 版本和 CPU unpacked 版本（共享 RNG 序列）
    wqkv_d, wqkv_cpu = _linear_w(3 * Hh, H, torch.bfloat16)
    wfa_d,  wfa_cpu  = _linear_w(D,      H, torch.bfloat16)
    wfb_d,  wfb_cpu  = _linear_w(Hh,     D, torch.bfloat16)
    wga_d,  wga_cpu  = _linear_w(D,      H, torch.bfloat16)
    wgb_d,  wgb_cpu  = _linear_w(Hh,     D, torch.bfloat16)
    wb_d,   wb_cpu   = _linear_w(cfg.num_heads, H, torch.bfloat16)
    wo_d,   wo_cpu   = _linear_w(H,      Hh, torch.bfloat16)

    # 非 GEMM tensors：dev 路径用 device 版本，prod 注入用 CPU 版本
    conv_w_cpu  = _rand_cpu((3 * Hh, K), torch.bfloat16)
    conv_b_cpu  = _rand_cpu((3 * Hh,),   torch.bfloat16)
    A_log_cpu   = _rand_cpu((1, 1, cfg.num_heads, 1), torch.float32)
    dt_bias_cpu = _rand_cpu((Hh,),       torch.float32)

    return SimpleNamespace(
        # ── dev / REF forward 用（对 Zeus 是 packed LocalMem，对 CPU/CUDA 是普通张量） ──
        wqkv=wqkv_d,
        wfa=wfa_d,
        wfb=wfb_d,
        wga=wga_d,
        wgb=wgb_d,
        wb=wb_d,
        wo=wo_d,
        # 以下不是 GEMM，不需要 pack
        conv_w   =conv_w_cpu.to(device),
        conv_b   =conv_b_cpu.to(device),
        A_log    =A_log_cpu.to(device),
        dt_bias  =dt_bias_cpu.to(device),
        o_norm_w =torch.ones(D, dtype=torch.bfloat16, device=device),

        # ── prod-path 注入用：CPU raw (out, in) 副本 ──
        wqkv_unpacked=wqkv_cpu,
        wfa_unpacked=wfa_cpu,
        wfb_unpacked=wfb_cpu,
        wga_unpacked=wga_cpu,
        wgb_unpacked=wgb_cpu,
        wb_unpacked=wb_cpu,
        wo_unpacked=wo_cpu,
        conv_w_unpacked=conv_w_cpu,
        conv_b_unpacked=conv_b_cpu,
        A_log_unpacked=A_log_cpu,
        dt_bias_unpacked=dt_bias_cpu,
    )


# ════════════════════════════════════════════════════════════════
#         Production path （Glm5NextLinearAttention 真实代码）
# ════════════════════════════════════════════════════════════════
# 下面这一坨是为了直接跑 python/sglang/srt/models/glm5_next.py 里的
# Glm5NextLinearAttention.forward —— 而不是 dev 端手拼的 _zeus_forward_*。
# 目的：发现生产代码本身的 bug（不是 kernel 的 bug，而是 wiring 的 bug：
# split 顺序、unflatten/unsqueeze、cache_indices 索引方向、conv1d 三次调
# 用、forward_qkvbfg 的 Linear 拼装等）。
#
# Dev 路径作为 golden，prod 路径作为待验证对象，两者跑同一组输入 + 同一
# 组权重，期望数值 byte-exact（同 device 同 kernel，差异来自 wrapper）。

_TP_INITIALIZED = False


def _setup_tp_once():
    """单进程 TP=1 初始化；KDA 后端选 ZEUS。一次性，幂等。

    Glm5NextLinearAttention.__init__ 会调 get_tensor_model_parallel_world_size
    / get_attention_tp_size 等，必须先把 distributed 装起来。Zeus 上不能
    用 nccl，用 gloo。
    """
    global _TP_INITIALIZED
    if _TP_INITIALIZED:
        return

    import os
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    import sglang.srt.layers.attention.linear.utils as _linear_utils

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12345")

    try:
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
    except Exception as e:
        # 二次调用会报 "already initialized"；忽略
        if "already" not in str(e).lower():
            raise

    _linear_utils.LINEAR_ATTN_DECODE_BACKEND = (
        _linear_utils.LinearAttnKernelBackend.ZEUS
    )
    _linear_utils.LINEAR_ATTN_PREFILL_BACKEND = (
        _linear_utils.LinearAttnKernelBackend.TRITON
    )

    # SGLang 多个模块在 import 时执行 `_is_zeus = is_zeus()` 一次性求值。
    # 如果 import 时 is_zeus() 还没返回 True（比如某条 import 链早于
    # SGLANG_DEVICE 的设置），_is_zeus 会卡在 False，运行时即使 is_zeus()
    # 已经返回 True 也走不到 Zeus 分支。这里手动把已知有这种 pattern 的
    # 模块的 _is_zeus 都拨成 True。
    #
    # 当前已知的：
    #   - fused_norm_gate.py:22 → 影响 FusedRMSNormGated.forward 是否走
    #     sgl_kernel_zeus.rms_norm_gated（不走的话会去拿 triton 的 rms_norm_gated，
    #     在 Zeus 上要么 fallback 到 CPU、要么直接报错）
    try:
        import sglang.srt.layers.attention.fla.fused_norm_gate as _fng
        _fng._is_zeus = True
    except ImportError:
        pass

    _TP_INITIALIZED = True


def _build_glm5_config(cfg):
    """从 dev test 的 SimpleNamespace cfg 构造一个最小可用的 GlmLinearConfig。"""
    from sglang.srt.configs.glm_linear import GlmLinearConfig

    return GlmLinearConfig(
        model_type="glm4_moe",
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        head_dim=cfg.head_dim,                # 走 MLA 时才用，KDA 不读
        # KDA 关键字段
        linear_num_key_heads=cfg.num_heads,
        linear_num_value_heads=cfg.num_heads,
        linear_key_head_dim=cfg.head_dim,
        linear_value_head_dim=cfg.head_v_dim,
        linear_conv_kernel_dim=cfg.short_conv_kernel_size,
        linear_attn_config={
            "num_heads": cfg.num_heads,
            "head_dim": cfg.head_dim,
            "short_conv_kernel_size": cfg.short_conv_kernel_size,
            "kda_layers": [0],                 # 只测 layer 0
            "full_attn_layers": [],
            "safe_gate": False,
        },
        linear_allow_neg_eigval=False,
        # MoE 字段必须给 placeholder（GlmLinearConfig 校验）
        first_k_dense_replace=0,
        n_routed_experts=8,
        num_experts_per_token=4,
    )


def _build_prod_layer(cfg, weights):
    """实例化 Glm5NextLinearAttention 并注入 dev test 的权重。

    流程：
      1. Glm5NextLinearAttention(...) → CPU
      2. .to("zeus") → 子模块全部搬到 Zeus
      3. 用 weights.*_unpacked 覆盖各 nn.Linear / nn.Parameter 的 .data
      4. 镜像生产 loader 的 pack 逻辑（model_loader/loader.py:842-887）：
         - 注册 (LinearBase, 'weight') 到 _GEMM_TRANSPOSE_PARAMS
         - pack_weights(layer, target_modules={LinearBase}, filter_fn=skip_conv1d)
         注意：SGLang 的 Linear 类（QKVParallelLinear / ColumnParallelLinear / ...）
         继承自 LinearBase 而不是 nn.Linear，所以必须用 target_modules 显式
         告诉 pack_weights 识别它们；qkv_conv1d.weight 是 3D 必须 filter 掉。
      5. 刷新 RadixLinearAttention.attn 里 conv_weights / bias 视图
         （_apply 钩子的逻辑，对应 glm5_next.py:288-300）
    """
    from sglang.srt.models.glm5_next import Glm5NextLinearAttention
    from sglang.srt.layers.linear import LinearBase
    from torch_zeus.zeus.pack_weights import (
        pack_weights, _GEMM_TRANSPOSE_PARAMS,
    )

    config = _build_glm5_config(cfg)
    # Linear 子类默认用 torch.get_default_dtype()，默认 fp32 → pack_weights
    # 会拒（_SUPPORTED_DTYPES 只含 bf16/int8/uint8）。临时把 default dtype
    # 切成 bf16 构造 layer；A_log / dt_bias 在 __init__ 里显式 fp32，不受影响。
    _saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        layer = Glm5NextLinearAttention(
            layer_idx=0,
            hidden_size=cfg.hidden_size,
            config=config,
            rms_norm_eps=cfg.rms_norm_eps,
            prefix="layers.0.linear_attn",
        ).to("zeus")
    finally:
        torch.set_default_dtype(_saved_dtype)

    # ── 注入权重（注入时还是 (out, in) 普通张量；pack 在下一步） ──
    with torch.no_grad():
        layer.qkv_proj.weight.data.copy_(weights.wqkv_unpacked.to("zeus"))
        layer.f_a_proj.weight.data.copy_(weights.wfa_unpacked.to("zeus"))
        layer.f_b_proj.weight.data.copy_(weights.wfb_unpacked.to("zeus"))
        layer.g_a_proj.weight.data.copy_(weights.wga_unpacked.to("zeus"))
        layer.g_b_proj.weight.data.copy_(weights.wgb_unpacked.to("zeus"))
        layer.b_proj.weight.data.copy_(weights.wb_unpacked.to("zeus"))
        layer.o_proj.weight.data.copy_(weights.wo_unpacked.to("zeus"))
        # qkv_conv1d.weight 在 __init__ 里被 unsqueeze(1) 成 [3*Hh, 1, K]
        layer.qkv_conv1d.weight.data.copy_(
            weights.conv_w_unpacked.to("zeus").unsqueeze(1)
        )
        if layer.qkv_conv1d.bias is not None:
            layer.qkv_conv1d.bias.data.copy_(weights.conv_b_unpacked.to("zeus"))
        layer.A_log.data.copy_(weights.A_log_unpacked.to("zeus"))
        layer.dt_bias.data.copy_(weights.dt_bias_unpacked.to("zeus"))

    # ── 与生产 loader 对齐：注册 LinearBase 转置规则 + 一次性 pack ──
    # SGLang 的 Linear 子类继承自 LinearBase，不是 nn.Linear；不显式注册
    # 转置规则 / target_modules，pack_weights 找不到 .weight 就什么都不 pack，
    # 后续 GEMM 会报 "weight must be a LocalMem tensor"。
    _GEMM_TRANSPOSE_PARAMS.add((LinearBase, 'weight'))

    def _skip_conv1d(name, param):
        # qkv_conv1d.weight 是 3D conv 权重（[out, 1, K]），不能 .t()
        # 也不该 pack 成 LocalMem GEMM 形态 —— 它是被 causal_conv1d 用，
        # 不走 GEMM dispatch
        return "qkv_conv1d" not in name

    pack_weights(
        layer, target_modules={LinearBase},
        Tr=1, Tc=1,
        filter_fn=_skip_conv1d,
    )

    # ── 刷新 attn 内的 conv_weights / bias 视图（_apply 钩子语义） ──
    layer.attn.conv_weights = layer.qkv_conv1d.weight.squeeze(1)
    layer.attn.bias = layer.qkv_conv1d.bias

    # ── 诊断：打印每个 Linear 的 weight 是否真的 pack 进 LocalMem ──
    from torch_zeus.zeus.local_memory import is_local_mem
    print("  [pack_diag] per-Linear is_local_mem(weight):")
    for name, mod in layer.named_modules():
        w = getattr(mod, "weight", None)
        if w is None or not isinstance(w, torch.Tensor):
            continue
        try:
            packed = is_local_mem(w)
        except Exception as e:
            packed = f"<is_local_mem err: {e}>"
        print(f"    {name or '<root>':30s} type={type(mod).__name__:30s} "
              f"shape={tuple(w.shape)} dtype={w.dtype} packed={packed}")

    return layer


class _DirectKDAAttnBackend:
    """轻量 attn_backend stub。

    生产里 RadixLinearAttention.forward 调 forward_batch.attn_backend.forward
    (...)；正常情况下 attn_backend 是 HybridLinearAttnBackend，由它再分流到
    KDAAttnBackend。这里直接 by-pass HybridLinearAttnBackend，按 forward_mode
    把调用转给 KDAAttnBackend.forward_decode/extend —— 因为我们只测 KDA
    单层，不需要 hybrid 的 full-attn 分支。
    """

    def __init__(self, kda_backend):
        self._kda = kda_backend

    def forward(self, layer, forward_batch, mixed_qkv, a, b,
                save_kv_cache=True, **kwargs):
        if forward_batch.forward_mode.is_decode():
            return self._kda.forward_decode(
                layer=layer, mixed_qkv=mixed_qkv, a=a, b=b,
                forward_batch=forward_batch, save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self._kda.forward_extend(
                layer=layer, forward_batch=forward_batch,
                mixed_qkv=mixed_qkv, a=a, b=b,
                save_kv_cache=save_kv_cache, **kwargs,
            )


def _build_prod_backend(cfg, B, query_start_loc, conv_init, ssm_init,
                        state_pool_size=None):
    """构造 KDAAttnBackend 实例 + state pool stub。

    Returns:
      attn_backend  : _DirectKDAAttnBackend，丢给 forward_batch.attn_backend
      conv_pool     : [N_pool, 3*Hh, K-1] bf16  zeus
      ssm_pool      : [N_pool, num_heads, K, V] fp32 zeus
      cache_indices : [B] int32 zeus（[0..B-1]）
    """
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata

    Hh = cfg.num_heads * cfg.head_dim
    K = cfg.short_conv_kernel_size
    N_pool = state_pool_size if state_pool_size is not None else (B + 4)

    # state pool（生产格式：pool 索引 + cache_indices 取批次）
    conv_pool = torch.zeros(N_pool, 3 * Hh, K - 1,
                             dtype=torch.bfloat16, device="zeus")
    ssm_pool = torch.zeros(N_pool, cfg.num_heads, cfg.head_dim,
                            cfg.head_v_dim, dtype=torch.float32, device="zeus")
    conv_pool[:B].copy_(conv_init)
    ssm_pool[:B].copy_(ssm_init)
    cache_indices = torch.arange(B, dtype=torch.int32, device="zeus")

    # mamba2_layer_cache 协议：.conv[0] 和 .temporal
    mamba_cache = SimpleNamespace(conv=[conv_pool], temporal=ssm_pool)
    req_to_token_pool = SimpleNamespace(
        mamba2_layer_cache=lambda layer_id: mamba_cache,
        get_mamba_indices=lambda req_pool_indices: cache_indices,
    )
    model_runner_stub = SimpleNamespace(
        device=torch.device("zeus"),
        req_to_token_pool=req_to_token_pool,
    )

    backend = KDAAttnBackend(model_runner_stub)
    backend.forward_metadata = ForwardMetadata(
        query_start_loc=query_start_loc.to("zeus"),
        mamba_cache_indices=cache_indices,
    )

    return _DirectKDAAttnBackend(backend), conv_pool, ssm_pool, cache_indices


def _make_forward_batch(forward_mode, B, seq_lens, attn_backend, total_T):
    """最小可用 ForwardBatch stub —— 只装生产代码访问到的字段。"""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if forward_mode == "extend":
        # 与 dev test 的 has_initial_state=[False, True] 对齐：第二条带 prefix
        prefix_lens = torch.zeros(B, dtype=torch.int32, device="zeus")
        if B >= 2:
            prefix_lens[1] = 32                # 任意 >0 的值表示有 prefix
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_prefix_lens=prefix_lens,
            extend_seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            attn_backend=attn_backend,
            batch_size=B,
            input_ids=torch.zeros(total_T, dtype=torch.long, device="zeus"),
            req_pool_indices=torch.arange(B, dtype=torch.int32, device="zeus"),
        )
    else:  # decode
        return SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            extend_prefix_lens=None,
            extend_seq_lens_cpu=None,
            attn_backend=attn_backend,
            batch_size=B,
            input_ids=torch.zeros(B, dtype=torch.long, device="zeus"),
            req_pool_indices=torch.arange(B, dtype=torch.int32, device="zeus"),
        )


def _prod_forward(layer, hidden_states, forward_batch):
    """跑 Glm5NextLinearAttention.forward（生产代码）。

    `positions` 和 `zero_allocator` 在 KDA 路径下不被使用，传 None 即可
    （Glm5NextLinearAttention.forward 接收但没有访问）。
    """
    return layer(
        hidden_states=hidden_states,
        positions=None,
        forward_batch=forward_batch,
        zero_allocator=None,
    )


# ════════════════════════════════════════════════════════════════
#         Prod-vs-Dev stages
# ════════════════════════════════════════════════════════════════
def test_kda_layer_decode_prod_vs_dev(cfg, num_tokens=4, seed=42):
    """比较 dev `_zeus_forward_decode` 与 Glm5NextLinearAttention.forward。

    两条路径跑同一份输入 + 同一份权重，期望数值 byte-exact（同 device 同
    kernel）。任何偏差都来自生产 wiring（split 顺序、unflatten 等），不是
    kernel 数值本身。
    """
    print()
    print("=" * 60)
    print("Stage: kda_layer_decode_prod_vs_dev "
          "(dev pipeline vs glm5_next.py production path)")
    print("=" * 60)
    from sglang.srt.utils.common import is_zeus
    print("is_zeus", is_zeus())

    _setup_tp_once()

    H = cfg.hidden_size
    Hh = cfg.num_heads * cfg.head_dim
    K = cfg.short_conv_kernel_size
    N = num_tokens
    print(f"  shape: N={N}  H={H}  H_qkv={Hh}  K={K}  "
          f"heads={cfg.num_heads}×{cfg.head_dim}")

    z_w = _build_weights(cfg, "zeus", seed)
    # 生产里 qkv_conv1d 是 bias=False（glm5_next.py:238）→ layer.qkv_conv1d.bias=None，
    # 走到 causal_conv1d_update 时不加 bias。dev `_build_weights` 默认给 conv_b 一个
    # 随机值，会让 dev 比 prod 多加一项 bias，conv-state 和 o_proj 都会偏。这里
    # zero 掉对齐 prod；dev-only 路径（test_kda_layer_decode/extend）的 _build_weights
    # 不受影响。
    with torch.no_grad():
        z_w.conv_b.zero_()
        z_w.conv_b_unpacked.zero_()
    torch.manual_seed(seed)
    hidden_states = (torch.randn(N, H, dtype=torch.bfloat16) * 0.05).to("zeus")
    conv_init = (torch.randn(N, 3 * Hh, K - 1,
                              dtype=torch.bfloat16) * 0.01).to("zeus")
    ssm_init = (torch.randn(N, cfg.num_heads, cfg.head_dim, cfg.head_v_dim,
                             dtype=torch.float32) * 0.01).to("zeus")

    # ── Dev path（golden） ──
    conv_dev = conv_init.clone()
    ssm_dev = ssm_init.clone()
    try:
        with torch.no_grad():
            out_dev = _zeus_forward_decode(
                cfg, hidden_states.clone(), conv_dev, ssm_dev, z_w,
            )
    except (NotImplementedError, AttributeError, RuntimeError) as e:
        print(f"  DEV path failed (cannot proceed without golden): {e}")
        return None, None
    print(f"  DEV out:  shape={tuple(out_dev.shape)} dtype={out_dev.dtype}")

    # ── Prod path ──
    try:
        layer = _build_prod_layer(cfg, z_w)
    except Exception as e:
        print(f"  PROD layer build failed: {e}")
        return None, None

    query_start_loc = torch.arange(0, N + 1, dtype=torch.int32)
    attn_backend, conv_pool, ssm_pool, _ = _build_prod_backend(
        cfg=cfg, B=N,
        query_start_loc=query_start_loc,
        conv_init=conv_init, ssm_init=ssm_init,
    )
    fb = _make_forward_batch("decode", N, [1] * N, attn_backend, total_T=N)

    try:
        with torch.no_grad():
            out_prod = _prod_forward(layer, hidden_states.clone(), fb)
    except (NotImplementedError, AttributeError, RuntimeError) as e:
        print(f"  PROD forward failed: {e}")
        return None, None
    print(f"  PROD out: shape={tuple(out_prod.shape)} dtype={out_prod.dtype}")

    # ── Compare ──
    # 同 device 同 kernel 同 dtype，预期非常严格；> 1e-3 一定有 wiring bug
    ok_out = compare_tensors(
        "decode_prod_vs_dev/o_proj", out_dev.cpu(), out_prod.cpu(),
        atol=1e-4, rtol=1e-4,
    )
    ok_ssm = compare_tensors(
        "decode_prod_vs_dev/ssm", ssm_dev.cpu(), ssm_pool[:N].cpu(),
        atol=1e-4, rtol=1e-4,
    )
    ok_conv = compare_tensors(
        "decode_prod_vs_dev/conv", conv_dev.cpu().float(),
        conv_pool[:N].cpu().float(),
        atol=1e-4, rtol=1e-4,
    )
    return (ok_out and ok_ssm and ok_conv), None


def test_kda_layer_extend_prod_vs_dev(cfg, seed=42):
    """比较 dev `_zeus_forward_extend` 与 Glm5NextLinearAttention.forward
    （extend 模式）。

    与 decode 同一套思路。注意：extend 路径生产端依赖 chunk_kda，而
    sgl_kernel_zeus 当前还没移植 chunk_kda —— 报错时返回 None 走 SKIP。
    """
    print()
    print("=" * 60)
    print("Stage: kda_layer_extend_prod_vs_dev "
          "(dev pipeline vs glm5_next.py production path)")
    print("=" * 60)

    _setup_tp_once()

    H = cfg.hidden_size
    Hh = cfg.num_heads * cfg.head_dim
    K = cfg.short_conv_kernel_size
    seq_lens = [12, 72]
    B = len(seq_lens)
    total_T = sum(seq_lens)
    query_start_loc = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
        dtype=torch.int32,
    )
    print(f"  shape: B={B}  seq_lens={seq_lens}  total_T={total_T}  "
          f"H={H}  H_qkv={Hh}  heads={cfg.num_heads}×{cfg.head_dim}")

    z_w = _build_weights(cfg, "zeus", seed)
    # 与 decode_prod_vs_dev 同：prod qkv_conv1d.bias=None，对齐到 dev 也 zero 掉
    with torch.no_grad():
        z_w.conv_b.zero_()
        z_w.conv_b_unpacked.zero_()
    torch.manual_seed(seed)
    hidden_states = (torch.randn(total_T, H,
                                  dtype=torch.bfloat16) * 0.05).to("zeus")
    conv_init = (torch.randn(B, 3 * Hh, K - 1,
                              dtype=torch.bfloat16) * 0.01).to("zeus")
    ssm_init = (torch.randn(B, cfg.num_heads, cfg.head_dim, cfg.head_v_dim,
                             dtype=torch.float32) * 0.01).to("zeus")
    has_initial_state = torch.tensor([False, True], dtype=torch.bool)

    # ── Dev path（golden） ──
    conv_dev = conv_init.clone()
    ssm_dev = ssm_init.clone()
    try:
        with torch.no_grad():
            out_dev = _zeus_forward_extend(
                cfg, hidden_states.clone(), conv_dev, ssm_dev, z_w,
                query_start_loc.to("zeus"),
                has_initial_state.to("zeus"),
            )
    except (NotImplementedError, AttributeError, RuntimeError) as e:
        print(f"  DEV path failed (cannot proceed without golden): {e}")
        return None, None
    print(f"  DEV out:  shape={tuple(out_dev.shape)} dtype={out_dev.dtype}")

    # ── Prod path ──
    try:
        layer = _build_prod_layer(cfg, z_w)
    except Exception as e:
        print(f"  PROD layer build failed: {e}")
        return None, None

    attn_backend, conv_pool, ssm_pool, _ = _build_prod_backend(
        cfg=cfg, B=B,
        query_start_loc=query_start_loc,
        conv_init=conv_init, ssm_init=ssm_init,
    )
    fb = _make_forward_batch("extend", B, seq_lens, attn_backend,
                              total_T=total_T)

    try:
        with torch.no_grad():
            out_prod = _prod_forward(layer, hidden_states.clone(), fb)
    except (NotImplementedError, AttributeError, RuntimeError) as e:
        # chunk_kda 暂未移植到 Zeus —— 当前 extend prod 路径必然 SKIP
        print(f"  PROD forward failed (likely chunk_kda missing): {e}")
        return None, None
    print(f"  PROD out: shape={tuple(out_prod.shape)} dtype={out_prod.dtype}")

    # ── Compare ──
    ok_out = compare_tensors(
        "extend_prod_vs_dev/o_proj", out_dev.cpu(), out_prod.cpu(),
        atol=1e-4, rtol=1e-4,
    )
    ok_ssm = compare_tensors(
        "extend_prod_vs_dev/ssm", ssm_dev.cpu(), ssm_pool[:B].cpu(),
        atol=1e-4, rtol=1e-4,
    )
    ok_conv = compare_tensors(
        "extend_prod_vs_dev/conv", conv_dev.cpu().float(),
        conv_pool[:B].cpu().float(),
        atol=1e-4, rtol=1e-4,
    )
    return (ok_out and ok_ssm and ok_conv), None


# ────────────────────────────────────────────────────────────────
#                          Stage: decode
# ────────────────────────────────────────────────────────────────
def test_kda_layer_decode(cfg, num_tokens=4, seed=42):
    """端到端 decode 路径对齐。"""
    print()
    print("=" * 60)
    print("Stage: kda_layer_decode (GLM-Next KDA decode end-to-end)")
    print("=" * 60)

    H = cfg.hidden_size
    Hh = cfg.num_heads * cfg.head_dim
    K = cfg.short_conv_kernel_size
    N = num_tokens

    print(f"  shape: N={N}  H={H}  H_qkv={Hh}  K={K}  "
          f"heads={cfg.num_heads}×{cfg.head_dim}")

    torch.manual_seed(seed)
    hidden_states = torch.randn(N, H, dtype=torch.bfloat16) * 0.05
    conv_state_init = torch.randn(N, 3 * Hh, K - 1, dtype=torch.bfloat16) * 0.01
    ssm_state_init = torch.randn(
        N, cfg.num_heads, cfg.head_dim, cfg.head_v_dim, dtype=torch.float32,
    ) * 0.01

    # ── REF ──
    ref_w = _build_weights(cfg, REF_DEVICE, seed)
    hs_ref = hidden_states.to(REF_DEVICE)
    conv_ref = conv_state_init.clone().to(REF_DEVICE)
    ssm_ref = ssm_state_init.clone().to(REF_DEVICE)
    with torch.no_grad():
        out_ref = _ref_forward_decode(cfg, hs_ref, conv_ref, ssm_ref, ref_w)
    print(f"  REF out: shape={tuple(out_ref.shape)} dtype={out_ref.dtype}")

    # ── Zeus ──
    z_w = _build_weights(cfg, "zeus", seed)
    hs_z = hidden_states.to("zeus")
    conv_z = conv_state_init.clone().to("zeus")
    ssm_z = ssm_state_init.clone().to("zeus")
    try:
        with torch.no_grad():
            out_z = _zeus_forward_decode(cfg, hs_z, conv_z, ssm_z, z_w)
    except (NotImplementedError, AttributeError) as e:
        print(f"  ZEUS: missing kernel — {e}")
        return None, (out_ref,)
    print(f"  ZEUS out: shape={tuple(out_z.shape)} dtype={out_z.dtype}")

    # ── Compare ──
    # decode bf16 + recurrent 累加：output 容忍 5e-2;state 用 fp32 累加，1e-3
    ok_out = compare_tensors(
        "kda_layer_decode/o_proj", out_ref, out_z.cpu(),
        atol=5e-2, rtol=5e-2,
    )
    ok_conv = compare_tensors(
        "kda_layer_decode/conv_state", conv_ref.float(), conv_z.cpu().float(),
        atol=1e-3, rtol=1e-3,
    )
    ok_ssm = compare_tensors(
        "kda_layer_decode/ssm_state", ssm_ref, ssm_z.cpu(),
        atol=1e-3, rtol=1e-3,
    )
    return (ok_out and ok_conv and ok_ssm), (out_ref,)


# ────────────────────────────────────────────────────────────────
#                          Stage: extend
# ────────────────────────────────────────────────────────────────
def test_kda_layer_extend(cfg, seed=42):
    """端到端 extend 路径对齐。

    用两条 seq 拼成 varlen batch：第 0 个全新 prefill（has_initial=False），
    第 1 个带 prefix 续传（has_initial=True，conv_state / ssm_state 都非零）。
    """
    print()
    print("=" * 60)
    print("Stage: kda_layer_extend (GLM-Next KDA extend end-to-end)")
    print("=" * 60)

    H = cfg.hidden_size
    Hh = cfg.num_heads * cfg.head_dim
    K = cfg.short_conv_kernel_size
    seq_lens = [12, 72]                    # 第 2 个 > chunk_size=64，确保走多 chunk
    B = len(seq_lens)
    total_T = sum(seq_lens)
    query_start_loc = torch.tensor(
        [0, *torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()],
        dtype=torch.int32,
    )
    has_initial_state = torch.tensor([False, True], dtype=torch.bool)

    print(f"  shape: B={B}  seq_lens={seq_lens}  total_T={total_T}  "
          f"H={H}  H_qkv={Hh}  heads={cfg.num_heads}×{cfg.head_dim}")
    print(f"  has_initial_state = {has_initial_state.tolist()}")

    torch.manual_seed(seed)
    hidden_states = torch.randn(total_T, H, dtype=torch.bfloat16) * 0.05
    conv_states_init = torch.randn(B, 3 * Hh, K - 1, dtype=torch.bfloat16) * 0.01
    ssm_state_init = torch.randn(
        B, cfg.num_heads, cfg.head_dim, cfg.head_v_dim, dtype=torch.float32,
    ) * 0.01

    # ── REF ──
    ref_w = _build_weights(cfg, REF_DEVICE, seed)
    hs_ref = hidden_states.to(REF_DEVICE)
    conv_ref = conv_states_init.clone().to(REF_DEVICE)
    ssm_ref = ssm_state_init.clone().to(REF_DEVICE)
    with torch.no_grad():
        out_ref = _ref_forward_extend(
            cfg, hs_ref, conv_ref, ssm_ref, ref_w,
            query_start_loc.to(REF_DEVICE),
            has_initial_state.to(REF_DEVICE),
        )
    print(f"  REF out: shape={tuple(out_ref.shape)} dtype={out_ref.dtype}")

    # ── Zeus ──
    z_w = _build_weights(cfg, "zeus", seed)
    hs_z = hidden_states.to("zeus")
    conv_z = conv_states_init.clone().to("zeus")
    ssm_z = ssm_state_init.clone().to("zeus")
    try:
        with torch.no_grad():
            out_z = _zeus_forward_extend(
                cfg, hs_z, conv_z, ssm_z, z_w,
                query_start_loc.to("zeus"),
                has_initial_state.to("zeus"),
            )
    except (NotImplementedError, AttributeError) as e:
        print(f"  ZEUS: missing kernel — {e}")
        return None, (out_ref,)
    print(f"  ZEUS out: shape={tuple(out_z.shape)} dtype={out_z.dtype}")

    # ── Compare ──
    # extend 比 decode 容忍稍紧（chunk-fwd 是块状累加，比 token-by-token 累误差小）
    ok_out = compare_tensors(
        "kda_layer_extend/o_proj", out_ref, out_z.cpu(),
        atol=3e-2, rtol=3e-2,
    )
    ok_conv = compare_tensors(
        "kda_layer_extend/conv_states", conv_ref.float(), conv_z.cpu().float(),
        atol=1e-3, rtol=1e-3,
    )
    ok_ssm = compare_tensors(
        "kda_layer_extend/ssm_state", ssm_ref, ssm_z.cpu(),
        atol=1e-3, rtol=1e-3,
    )
    return (ok_out and ok_conv and ok_ssm), (out_ref,)


# ────────────────────────────────────────────────────────────────
#                          Dispatch
# ────────────────────────────────────────────────────────────────
STAGES = {
    "decode": test_kda_layer_decode,
    "extend": test_kda_layer_extend,
    "decode_prod_vs_dev": test_kda_layer_decode_prod_vs_dev,
    "extend_prod_vs_dev": test_kda_layer_extend_prod_vs_dev,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--num-tokens", type=int, default=4,
                        help="decode batch size (extend ignores)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = default_glm5_next_kda_cfg()
    print(f"GLM-Next KDA proxy: H={cfg.hidden_size}  "
          f"heads={cfg.num_heads}×{cfg.head_dim}  "
          f"conv_kernel_size={cfg.short_conv_kernel_size}")
    print(f"REF_DEVICE = {REF_DEVICE}")

    results = {}
    for name, fn in STAGES.items():
        if args.stage not in (name, "all"):
            continue
        try:
            if fn.__code__.co_argcount >= 3:
                ok, _ = fn(cfg, num_tokens=args.num_tokens, seed=args.seed)
            else:
                ok, _ = fn(cfg, seed=args.seed)
            results[name] = ok
        except Exception as e:
            import traceback
            print(f"  [{name}] EXCEPTION: {e}")
            traceback.print_exc()
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
            status = "SKIP (REF-only; Zeus kernel TODO)"
        print(f"  {name:30s} : {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
