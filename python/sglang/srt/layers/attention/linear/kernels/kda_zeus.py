import torch
import torch_zeus                          # 注册 privateuseone
from sgl_kernel_zeus import (
    fused_kda_gate,
    fused_recurrent_kda_Sdecay_indexed,
)

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu
from sglang.srt.utils.common import is_zeus


if not is_cpu():
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.srt.layers.attention.fla.kda import chunk_kda


class ZeusKDAKernel(LinearAttnKernelBase):
    """Zeus kernel for KDA (Kimi Delta Attention) linear attention."""

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        ## TODO
        scale = None
        eps = 1e-6
        o = None
        # softplus_threshold: CUDA 端用 tl.where(z>20, z, log(1+exp(z))) 切分支
        # Zeus 端用数值稳定形式 max(z,0)+log(1+exp(-|z|))，等价于 threshold=∞
        # 任何 threshold 值都安全，直接忽略

        # ── 形状归一化 ────────────────────────────────────────────────
        # CUDA fla 经常用 [1, T, H, D] 这种带前导 batch-1 维的 4D 形式
        def _squeeze1(t):
            return t.squeeze(0) if (t.dim() >= 3 and t.size(0) == 1
                                    and t.dim() > 2) else t
        q = _squeeze1(q) if q.dim() == 4 else q
        k = _squeeze1(k) if k.dim() == 4 else k
        v = _squeeze1(v) if v.dim() == 4 else v

        T, H, D = q.shape                       # K == D, V == D（KDA v1: K=V=128）

        # a: 可能是 [T, H*D] / [T, H, D] / [1, T, H*D] / [1, T, H, D]
        if a.dim() == 4 and a.size(0) == 1:
            a = a.squeeze(0)
        if a.dim() == 3:                        # [T, H, D] → flat
            a = a.reshape(a.size(0), -1)
        assert a.shape == (T, H * D), f"a shape {tuple(a.shape)} != ({T}, {H*D})"

        # b (= beta): [T, H] 或 [1, T, H]
        if b.dim() == 3 and b.size(0) == 1:
            b = b.squeeze(0)
        assert b.shape == (T, H), f"beta shape {tuple(b.shape)} != ({T}, {H})"

        # ── dtype 归一化（Zeus 契约：a bf16；A_log/dt_bias/beta fp32）────
        if a.dtype       != torch.bfloat16: a       = a.to(torch.bfloat16)
        if A_log.dtype   != torch.float32:  A_log   = A_log.to(torch.float32)
        if dt_bias is not None and dt_bias.dtype != torch.float32:
            dt_bias = dt_bias.to(torch.float32)
        if b.dtype       != torch.float32:  b       = b.to(torch.float32)
        if cache_indices.dtype != torch.int32:
            cache_indices = cache_indices.to(torch.int32)
        if query_start_loc.dtype != torch.int32:
            query_start_loc = query_start_loc.to(torch.int32)

        # ── stage 4: gate ─────────────────────────────────────────────
        # g[t, h, d] = -exp(A_log[h]) * softplus(a[t, h*D+d] + dt_bias[h*D+d])
        g = fused_kda_gate(a, A_log, head_dim=D, g_bias=dt_bias)   # [T, H, D] fp32

        # CUDA 路径下 fused_sigmoid_gating_delta_rule_update 内部对 beta 做
        # sigmoid（kda_triton.py:33；从 kernel 名 "sigmoid_gating" 也能看出）；
        # Glm5NextLinearAttention.forward 在 decode 模式下显式 *不* 做
        # sigmoid（glm5_next.py:331-348，sigmoid 仅 prefill 才显式做），
        # 假定 decode kernel 自己处理。Zeus 的 fused_recurrent_kda_Sdecay_indexed
        # 不做 sigmoid，会直接把 raw beta 当成 delta-rule rate 用，数值与
        # CUDA 不对齐 —— 在这里补上 sigmoid。
        b = b.sigmoid()

        # ── stage 5: gated delta-rule recurrent，state pool in-place 更新 ─
        return fused_recurrent_kda_Sdecay_indexed(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g,                                        # 已经 fp32 [T, H, D]
            beta=b.contiguous(),                        # fp32 [T, H]
            state_pool=ssm_states,            # fp32 [N_pool, H, K, V] in-place
            cache_indices=cache_indices,
            cu_seqlens=query_start_loc,
            scale=scale,                                # None → K**-0.5
            eps=eps,
            use_qk_l2norm_in_kernel=True,
            o=o,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
        )
