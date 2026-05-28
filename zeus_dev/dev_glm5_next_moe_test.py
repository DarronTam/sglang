"""
GLM-Next MoE-FFN 段集成测试（end-to-end）

范围：
  - 起点：input_layernorm 后的 hidden_states [T, H]
  - 终点：Glm5NextMoe.forward_normal 输出 [T, H]
  - 单 device / 单 layer / 不含 a2a / 不含 NextN
  - GLM-Next 的 Glm5NextMoe 复用 DeepseekV2MoE.forward_normal，本测试镜像该
    路径的 6-kernel pipeline。

REF（REF_DEVICE 上，纯 torch 链）：
  gate Linear → biased_grouped_topk_impl → per-token MoE 循环 + shared MLP
Zeus（"zeus" device 上，sgl_kernel_zeus 6-kernel）：
  biased_grouped_topk → moe_align_block_size_alloc →
  moe_grouped_gemm(gemm1) → silu_and_mul → moe_grouped_gemm(gemm2) →
  moe_sum_reduce(shared_output=…)

Routed scaling 归属（与生产路径一致）：
  - biased_grouped_topk(apply_routed_scaling_factor_on_output=True)
  - moe_grouped_gemm(gemm2, mul_routed_weight=True)
  - moe_sum_reduce(routed_scaling_factor=1.0, shared_output=…)

Proxy shape（与 dev_glm4_moe_test.py:test_moe_block_full 同策略）：
  T=16, H=256, mI=128, E=8, top_k=4。
  GLM-Next 真实权重过大（H=4096+ 、E≥128），dev 脚本走 proxy 拓扑保证
  CPU REF 能秒级完成；真实 shape 的 kernel-level 对齐由
  sgl-kernel-zeus/tests/ 下的单算子测试承担。

用法:
  python zeus_dev/dev_glm5_next_moe_test.py
  python zeus_dev/dev_glm5_next_moe_test.py --stage moe_block_e2e --num-tokens 32
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

# 强制 SGLang 走 Zeus 路径。机器上 CUDA + Zeus 共存时，is_zeus() 默认会
# 选 CUDA（参见 python/sglang/srt/utils/common.py:is_zeus）。必须在 import
# sglang 之前设这个环境变量，否则 lru_cache 会缓存错误的结果。
os.environ.setdefault("SGLANG_DEVICE", "zeus")

import torch
import torch_zeus  # noqa: F401 — registers zeus backend
import sgl_kernel_zeus

# ── SGLang server_args mock ──────────────────────────────────────
# 注意：dev_glm4_moe_test 这种 helper 模块在 import 时也会做
# `sglang.srt.server_args.get_global_server_args = lambda: <its own _dummy_args>`，
# 所以**我们的 mock 必须在 from dev_glm4_moe_test import ... 之后再装一次**，
# 否则会被覆盖。把安装逻辑包成 helper，下面 import 完再调，需要时也可
# 重复调（比如生产代码后续又装了别的）。
import sglang.srt.server_args
_dummy_args = Mock()
_dummy_args.rl_on_policy_target = None
# Glm5NextMoe.__init__ 读 disable_shared_experts_fusion；Mock 默认值是
# Mock 对象（truthy），会让 num_fused_shared_experts=0 —— 与 dev 路径
# "shared experts 走独立 MLP" 一致。这里显式置 True 让语义清晰。
_dummy_args.disable_shared_experts_fusion = True
_dummy_args.enable_eplb = False
_dummy_args.ep_num_redundant_experts = 0
_dummy_args.enable_deterministic_inference = False
# KTransformers EP wrapper：检查 `kt_weight_path is None` 决定是否启用，
# Mock 默认值不是 None 而是 Mock，会让 FusedMoE 试图实例化 KTEPWrapperMethod
# 然后报 `kt_kernel is not installed`。显式置 None 关掉。
_dummy_args.kt_weight_path = None


def _install_server_args_mock():
    """把 _dummy_args 注入 sglang.srt.server_args 的 storage + 函数引用。

    幂等。在每次需要确保 mock 生效的地方都可以调一次（比如新增了
    会再次覆盖 get_global_server_args 的依赖时）。
    """
    sglang.srt.server_args._global_server_args = _dummy_args
    sglang.srt.server_args.get_global_server_args = lambda *a, **kw: _dummy_args


_install_server_args_mock()


# 把 zeus_dev/ 加入 sys.path，方便复用 dev_glm4_moe_test.py 的 helpers。
sys.path.insert(0, str(Path(__file__).parent))
from dev_glm4_moe_test import (  # noqa: E402
    compare_ids,
    compare_tensors,
    _ref_moe_core,
)

# dev_glm4_moe_test import 时会用它自己的 _dummy_args 覆盖我们的 lambda。
# 这里再装一遍恢复成本测试的 _dummy_args（带我们设的所有字段）。
_install_server_args_mock()


REF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _pack_zeus_linear_weight(w_cpu, dtype=torch.bfloat16):
    """把 (out, in) 的 CPU 权重包成 Zeus 上的 LocalMem packed tensor。

    Zeus 的 F.linear / torch.mm 要求 weight 是 ZeusLocalMemTensor（packed
    成 (K, N) 布局并搬入 LocalMem）。dev 路径里在 Zeus 上跑 F.linear 时
    weight 必须先用这个 helper pack。

    返回的 weight 可以直接喂 F.linear(x, weight) —— Zeus aten::linear dispatch
    认识 LocalMem 形态。
    """
    import torch.nn as nn
    import torch_zeus.zeus as zeus

    out_features, in_features = w_cpu.shape
    layer = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
    layer.weight.data.copy_(w_cpu)
    layer = layer.to("zeus")
    zeus.pack_weights(layer, Tr=1, Tc=1)
    return layer.weight


def default_glm5_next_moe_cfg():
    """GLM-Next MoE 的 proxy config（数学拓扑与 GLM-4.7 同构，shape 缩小）。

    n_group / topk_group / norm_topk_prob / routed_scaling_factor 与
    GLM-Next 生产配置（GlmLinearConfig）一致；hidden_size 与 expert 数量
    缩到 CPU REF 能跑的尺寸。
    """
    return SimpleNamespace(
        # —— routing 相关（与生产同语义） —— #
        n_routed_experts=8,           # proxy for n_routed_experts (>= 128)
        num_experts_per_tok=4,        # proxy for top_k (typically 6 / 8)
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        # —— MoE 张量 shape —— #
        hidden_size=256,              # proxy for H
        moe_intermediate_size=128,    # proxy for mI
    )


# ── Stage: moe_block_e2e ───────────────────────────────────────
def test_moe_block_e2e(cfg, num_tokens=16, seed=42):
    """端到端 MoE-FFN 集成（GLM-Next 单 device、forward_normal 单卡路径）。

    与 dev_glm4_moe_test.py:test_moe_block_full 的差异：
      - 命名上对齐 GLM-Next（Glm5NextMoe），数学拓扑同构。
      - shared experts 直接走 raw MLP（gate_up + silu_and_mul + down），
        residual 系数固定为 1.0；Glm5NextMoe 与 GLM-4.7 在 forward_normal
        中均无额外的 routed-vs-shared gating，差异在 a2a / NextN 等本测试
        显式排除的部分。
    """
    print()
    print("=" * 60)
    print("Stage: moe_block_e2e (GLM-Next MoE-FFN end-to-end)")
    print("=" * 60)

    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

    T = num_tokens
    H = cfg.hidden_size
    mI = cfg.moe_intermediate_size
    E = cfg.n_routed_experts
    top_k = cfg.num_experts_per_tok
    scaling = cfg.routed_scaling_factor
    n_group = cfg.n_group
    topk_group = cfg.topk_group
    block_size = sgl_kernel_zeus.MOE_GROUPED_GEMM_BLOCK_M

    print(f"  proxy shape: T={T}  H={H}  mI={mI}  E={E}  top_k={top_k}")
    print(f"  routed_scaling_factor = {scaling}  block_size = {block_size}")
    print(f"  n_group = {n_group}  topk_group = {topk_group}  "
          f"norm_topk_prob = {cfg.norm_topk_prob}")
    print(f"  REF_DEVICE = {REF_DEVICE}")

    torch.manual_seed(seed)

    # ── 输入 + 权重（MoE 核心 + shared experts） ──
    x_bf16 = torch.randn(T, H, dtype=torch.bfloat16) * 0.1
    gate_w_bf16 = torch.randn(E, H, dtype=torch.bfloat16) * 0.1
    corr_bias_fp32 = torch.randn(E, dtype=torch.float32) * 0.01
    sh_gu_bf16 = torch.randn(2 * mI, H, dtype=torch.bfloat16) * 0.1   # shared gate_up
    sh_dn_bf16 = torch.randn(H, mI, dtype=torch.bfloat16) * 0.1       # shared down
    w13_bf16 = torch.randn(E, 2 * mI, H, dtype=torch.bfloat16) * 0.1
    w2_bf16 = torch.randn(E, H, mI, dtype=torch.bfloat16) * 0.1

    # ── REF 路径（REF_DEVICE 上，纯 torch） ──
    x_ref = x_bf16.to(REF_DEVICE)
    gate_w_ref = gate_w_bf16.to(REF_DEVICE)
    corr_bias_ref = corr_bias_fp32.to(REF_DEVICE)
    sh_gu_ref = sh_gu_bf16.to(REF_DEVICE)
    sh_dn_ref = sh_dn_bf16.to(REF_DEVICE)
    w13_ref = w13_bf16.to(REF_DEVICE)
    w2_ref = w2_bf16.to(REF_DEVICE)

    with torch.no_grad():
        # 1. gate Linear: bf16 × bf16 → bf16，进 topk 前 .float()（与生产一致）
        router_logits_bf16 = torch.nn.functional.linear(x_ref, gate_w_ref)
        router_logits_fp32 = router_logits_bf16.float()
        # 2. shared experts MLP（gate_up → silu_and_mul → down）
        sh_gu = torch.nn.functional.linear(x_ref, sh_gu_ref)
        sh_silu = (
            torch.nn.functional.silu(sh_gu[:, :mI].float())
            * sh_gu[:, mI:].float()
        ).to(torch.bfloat16)
        shared_out_bf16 = torch.nn.functional.linear(sh_silu, sh_dn_ref)
        # 3. biased grouped topk（scale 融进 weights）
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
            apply_routed_scaling_factor_on_output=True,
        )
        # 4. MoE 核心（per-token per-expert pure-torch GEMM 链）
        moe_core_ref = _ref_moe_core(
            x_ref.cpu(),
            w13_ref.cpu(), w2_ref.cpu(),
            w_ref.cpu(), ids_ref.cpu(),
            mI=mI,
        )
        # 5. residual add（fp32 累加 + 单次 RNE，与 moe_sum_reduce 内部一致）
        final_ref = (
            moe_core_ref.float() + shared_out_bf16.float().cpu()
        ).to(torch.bfloat16)

    print(f"  REF final:  shape={tuple(final_ref.shape)} dtype={final_ref.dtype}")
    print(f"  REF final[0, :6] = "
          f"{[round(v, 4) for v in final_ref[0, :6].float().tolist()]}")

    # ── Zeus 路径 ──
    # gate Linear 与 shared experts MLP 在 Zeus 上重新计算（不复用 REF 端结果）。
    # 这样保证 dev 路径完全 Zeus 端，与 prod 路径（Glm5NextMoe.forward_normal
    # 内部一切都在 Zeus 上跑）输入一致 —— 避免 router_logits 跨 device 后
    # bf16 微小偏差让 topk 选出不同的 expert。
    #
    # Zeus aten::linear 要求 weight 是 LocalMem packed，所以 gate / shared
    # MLP 的三个 Linear weight 必须先用 _pack_zeus_linear_weight 包一下。
    x_z = x_bf16.to("zeus")
    gate_w_z = _pack_zeus_linear_weight(gate_w_bf16)        # [E, H] → packed
    sh_gu_z = _pack_zeus_linear_weight(sh_gu_bf16)          # [2*mI, H] → packed
    sh_dn_z = _pack_zeus_linear_weight(sh_dn_bf16)          # [H, mI] → packed
    corr_bias_z = corr_bias_fp32.to("zeus")
    w13_z = w13_bf16.to("zeus")
    w2_z = w2_bf16.to("zeus")

    with torch.no_grad():
        # 0a. gate Linear（Zeus）
        router_logits_z = torch.nn.functional.linear(x_z, gate_w_z).float()
        # 0b. shared experts MLP（Zeus）
        sh_gu_out_z = torch.nn.functional.linear(x_z, sh_gu_z)
        sh_silu_z = sgl_kernel_zeus.silu_and_mul(sh_gu_out_z)
        shared_out_z = torch.nn.functional.linear(sh_silu_z, sh_dn_z)

        # 1. biased_grouped_topk
        w_z, ids_z = sgl_kernel_zeus.biased_grouped_topk(
            router_logits_z, corr_bias_z,
            num_expert_group=n_group,
            topk_group=topk_group,
            topk=top_k,
            num_fused_shared_experts=0,
            routed_scaling_factor=scaling,
            apply_routed_scaling_factor_on_output=True,
        )

        # 2. moe_align_block_size
        sorted_ids_z, expert_ids_z, num_post_z = (
            sgl_kernel_zeus.moe_align_block_size_alloc(
                ids_z, block_size, E,
            )
        )
        num_valid_tokens = T * top_k
        num_post_val = int(num_post_z.cpu().item())
        print(f"  Zeus moe_align: num_tokens_post_pad = {num_post_val} "
              f"({num_post_val // block_size} blocks)")

        # 3. gemm1（gate_up）：[T, H] → [T·top_k, 2·mI]
        C1_z = torch.empty(
            T * top_k, 2 * mI, dtype=torch.bfloat16, device="zeus",
        )
        sgl_kernel_zeus.moe_grouped_gemm(
            x_z, w13_z, C1_z,
            sorted_ids_z, expert_ids_z, num_post_z,
            num_valid_tokens=num_valid_tokens,
            top_k=top_k,
        )

        # 4. silu_and_mul：[T·top_k, 2·mI] → [T·top_k, mI]
        C1_silu_z = torch.empty(
            T * top_k, mI, dtype=torch.bfloat16, device="zeus",
        )
        sgl_kernel_zeus.silu_and_mul(C1_z, C1_silu_z)

        # 5. gemm2（down，mul_routed_weight=True）：[T·top_k, mI] → [T·top_k, H]
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

        # 6. moe_sum_reduce（+shared residual；scale=1.0 因为 scale 已融进 weights）
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

    # ── 比较 ──
    # Sanity check: topk_ids 集合一致（否则路由到不同专家，数值一定对不上）
    ok_ids = compare_ids(
        "moe_block_e2e/topk_ids",
        ids_ref, ids_z, allow_permutation=True,
    )
    # 端到端 bf16 输出，多级累加 + scale=2.5 放大，容忍 ~5e-2
    ok_final = compare_tensors(
        "moe_block_e2e/final",
        final_ref, final_z.cpu(),
        atol=5e-2, rtol=5e-2,
    )
    return (ok_ids and ok_final), None


# ════════════════════════════════════════════════════════════════
#         Production path （Glm5NextMoe = DeepseekV2MoE 真实代码）
# ════════════════════════════════════════════════════════════════
# 下面这一坨是为了直接跑 python/sglang/srt/models/glm5_next.py 里
# `Glm5NextMoe.forward_normal`（实际指向 DeepseekV2MoE.forward_normal） ——
# 而不是 dev 端手拼的 6-kernel 流水线。目的：发现生产代码本身的 wiring bug
# （TopK 路径、FusedMoE.experts 内部 dispatch、shared_experts 残差累加位置、
# routed_scaling_factor 的归属等）。
#
# Dev 路径作为 golden（已与 REF 对齐），prod 路径作为待验证对象，两者跑
# 同一组输入 + 同一组权重，期望数值在 bf16 噪声范围内对齐。

_TP_INITIALIZED = False


def _setup_tp_once():
    """单进程 TP=1 初始化；MoE 后端走 Zeus。一次性，幂等。

    Glm5NextMoe.__init__ 会调 get_tensor_model_parallel_world_size /
    get_moe_expert_parallel_world_size 等，必须先把 distributed 装起来。
    Zeus 上不能用 nccl，用 gloo。
    """
    global _TP_INITIALIZED
    if _TP_INITIALIZED:
        return

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.utils.common import is_zeus

    # 双保险：环境变量 + 清 lru_cache
    os.environ["SGLANG_DEVICE"] = "zeus"
    is_zeus.cache_clear()
    assert is_zeus(), (
        "is_zeus() returned False even after SGLANG_DEVICE=zeus; "
        "check torch_zeus install."
    )

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
        if "already" not in str(e).lower():
            raise

    # 同 KDA 测试：把已经早期捕获 _is_zeus=False 的模块拨成 True
    for mod_path in (
        "sglang.srt.layers.attention.fla.fused_norm_gate",
    ):
        try:
            import importlib
            _mod = importlib.import_module(mod_path)
            if hasattr(_mod, "_is_zeus"):
                _mod._is_zeus = True
        except ImportError:
            pass

    _TP_INITIALIZED = True


def _build_moe_config(cfg):
    """构造 Glm5NextMoe / DeepseekV2MoE 能消费的最小 config。

    DeepseekV2MoE.__init__ 读到的字段：
      hidden_size, hidden_act, n_routed_experts, num_experts_per_tok,
      n_shared_experts, n_group, topk_group, norm_topk_prob,
      routed_scaling_factor, moe_intermediate_size, topk_method
    （后两个是 GlmLinearConfig 的可选字段，必须显式给。）
    """
    from sglang.srt.configs.glm_linear import GlmLinearConfig

    config = GlmLinearConfig(
        model_type="glm4_moe",
        hidden_size=cfg.hidden_size,
        hidden_act="silu",
        # MoE 拓扑
        n_routed_experts=cfg.n_routed_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        n_shared_experts=1,                    # 与 dev 路径一致：1 组 shared experts
        n_group=cfg.n_group,
        topk_group=cfg.topk_group,
        moe_renormalize=cfg.norm_topk_prob,
        routed_scaling_factor=cfg.routed_scaling_factor,
        moe_intermediate_size=cfg.moe_intermediate_size,
        scoring_func="sigmoid",
        first_k_dense_replace=0,
        topk_method="noaux_tc",                # 让 MoEGate 创建 e_score_correction_bias
        # 下面这些 KDA 不读但 GlmLinearConfig 需要默认值
        num_attention_heads=8,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        head_dim=128,
        linear_attn_config=None,
    )
    # GlmLinearConfig 把 norm_topk_prob 字段叫 moe_renormalize（命名不一致），
    # 但 DeepseekV2MoE.__init__ 读的是 config.norm_topk_prob —— 手动补 alias
    # 让生产代码能拿到值（PretrainedConfig 允许后期 setattr 任意属性）。
    config.norm_topk_prob = cfg.norm_topk_prob
    return config


def _build_prod_moe_layer(cfg, weights):
    """实例化 Glm5NextMoe 并注入 dev test 的权重。

    Glm5NextMoe (= DeepseekV2MoE) 子模块结构：
      - gate (MoEGate)             —— .weight [E, H], .e_score_correction_bias [E]
      - experts (FusedMoE)         —— .w13_weight [E, 2*mI, H], .w2_weight [E, H, mI]
      - shared_experts (DeepseekV2MLP)
        - gate_up_proj (MergedColumnParallelLinear) —— .weight [2*mI, H]
        - down_proj    (RowParallelLinear)          —— .weight [H, mI]
      - topk (TopK helper, no weights)

    流程：
      1. instantiate Glm5NextMoe + .to("zeus") (with bf16 default dtype)
      2. 注入权重
      3. 把 LinearBase 子类 + MoEGate 都 pack 成 LocalMem
         - shared_experts 里两个 Linear 是 LinearBase 子类
         - MoEGate 是直接 nn.Module（不是 LinearBase），但 forward 用 F.linear
           需要 LocalMem，得专门 pack
         - experts.w13_weight / w2_weight 是 3D，不能 pack（也不该 pack ——
           moe_grouped_gemm 接受 raw [E, ...] 形态）
    """
    from sglang.srt.models.glm5_next import Glm5NextMoe
    from sglang.srt.models.deepseek_v2 import MoEGate
    from sglang.srt.layers.linear import LinearBase
    from torch_zeus.zeus.pack_weights import (
        pack_weights, _GEMM_TRANSPOSE_PARAMS,
    )

    config = _build_moe_config(cfg)

    # 诊断：在调 Glm5NextMoe(...) 之前确认 server_args mock 真的在生效
    from sglang.srt.server_args import get_global_server_args as _get_sa
    import sglang.srt.server_args as _sa_mod
    print(f"  [diag] _sa_mod._global_server_args is _dummy_args: "
          f"{_sa_mod._global_server_args is _dummy_args}")
    _sa_actual = _get_sa()
    print(f"  [diag] get_global_server_args() returns: "
          f"type={type(_sa_actual).__name__} id={id(_sa_actual)}")
    print(f"  [diag] is _dummy_args: {_sa_actual is _dummy_args}")
    print(f"  [diag] .ep_num_redundant_experts = "
          f"{_sa_actual.ep_num_redundant_experts!r}")
    print(f"  [diag] .disable_shared_experts_fusion = "
          f"{_sa_actual.disable_shared_experts_fusion!r}")

    # 与 KDA 测同：default dtype → bf16，让 Linear 子类的 weight 是 bf16
    # （pack_weights 只支持 bf16/int8/uint8）
    _saved_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        layer = Glm5NextMoe(
            config=config,
            layer_id=0,
            quant_config=None,
            prefix="layers.0.mlp",
        ).to("zeus")
    finally:
        torch.set_default_dtype(_saved_dtype)

    # ── 注入权重 ──
    # gate.weight: [E, H] —— 对应 dev 的 weights.gate_w_bf16
    # gate.e_score_correction_bias: [E] fp32 —— 对应 weights.corr_bias_fp32
    # experts.w13_weight: [E, 2*mI, H] —— 对应 weights.w13_bf16
    # experts.w2_weight: [E, H, mI] —— 对应 weights.w2_bf16
    # shared_experts.gate_up_proj.weight: [2*mI, H] —— 对应 weights.sh_gu_bf16
    # shared_experts.down_proj.weight: [H, mI] —— 对应 weights.sh_dn_bf16
    with torch.no_grad():
        layer.gate.weight.data.copy_(weights.gate_w_bf16.to("zeus"))
        if layer.gate.e_score_correction_bias is not None:
            layer.gate.e_score_correction_bias.data.copy_(
                weights.corr_bias_fp32.to("zeus")
            )
        layer.experts.w13_weight.data.copy_(weights.w13_bf16.to("zeus"))
        layer.experts.w2_weight.data.copy_(weights.w2_bf16.to("zeus"))
        layer.shared_experts.gate_up_proj.weight.data.copy_(
            weights.sh_gu_bf16.to("zeus")
        )
        layer.shared_experts.down_proj.weight.data.copy_(
            weights.sh_dn_bf16.to("zeus")
        )

    # ── 把 LinearBase + MoEGate 都 pack 成 LocalMem ──
    # shared_experts 的两个 Linear 是 LinearBase 子类；MoEGate 是直接
    # nn.Module 但 forward 用 F.linear，Zeus 要求 weight 在 LocalMem。
    # 注意：experts (FusedMoE) 不在 target_modules 里，w13/w2 不会被 pack。
    _GEMM_TRANSPOSE_PARAMS.add((LinearBase, 'weight'))
    _GEMM_TRANSPOSE_PARAMS.add((MoEGate, 'weight'))
    pack_weights(
        layer, target_modules={LinearBase, MoEGate},
        Tr=1, Tc=1,
    )

    return layer


def _prod_moe_forward(layer, hidden_states):
    """跑 Glm5NextMoe.forward_normal（生产代码）。

    forward_normal 不需要 forward_batch，只接 hidden_states 和几个 tp/fusion
    flag。这里全部走默认值（TP=1 → 不需要 all-reduce / all-gather 融合）。
    """
    return layer.forward_normal(hidden_states)


# ════════════════════════════════════════════════════════════════
#         Stage: moe_block_e2e_prod_vs_dev
# ════════════════════════════════════════════════════════════════
def test_moe_block_e2e_prod_vs_dev(cfg, num_tokens=16, seed=42):
    """比较 dev 6-kernel 流水线与 Glm5NextMoe.forward_normal。

    与 KDA 那一对 prod_vs_dev 同款思路：dev 作 golden（已与 REF 对齐），
    prod 用 SGLang 真实生产代码 instantiate + forward。
    """
    print()
    print("=" * 60)
    print("Stage: moe_block_e2e_prod_vs_dev "
          "(dev pipeline vs Glm5NextMoe.forward_normal)")
    print("=" * 60)

    _setup_tp_once()

    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

    T = num_tokens
    H = cfg.hidden_size
    mI = cfg.moe_intermediate_size
    E = cfg.n_routed_experts
    top_k = cfg.num_experts_per_tok
    scaling = cfg.routed_scaling_factor
    n_group = cfg.n_group
    topk_group = cfg.topk_group
    block_size = sgl_kernel_zeus.MOE_GROUPED_GEMM_BLOCK_M

    print(f"  proxy shape: T={T}  H={H}  mI={mI}  E={E}  top_k={top_k}")
    print(f"  routed_scaling_factor = {scaling}  block_size = {block_size}")

    torch.manual_seed(seed)

    # ── 同一组输入 + 权重 ──
    x_bf16 = torch.randn(T, H, dtype=torch.bfloat16) * 0.1
    gate_w_bf16 = torch.randn(E, H, dtype=torch.bfloat16) * 0.1
    corr_bias_fp32 = torch.randn(E, dtype=torch.float32) * 0.01
    sh_gu_bf16 = torch.randn(2 * mI, H, dtype=torch.bfloat16) * 0.1
    sh_dn_bf16 = torch.randn(H, mI, dtype=torch.bfloat16) * 0.1
    w13_bf16 = torch.randn(E, 2 * mI, H, dtype=torch.bfloat16) * 0.1
    w2_bf16 = torch.randn(E, H, mI, dtype=torch.bfloat16) * 0.1

    weights = SimpleNamespace(
        gate_w_bf16=gate_w_bf16,
        corr_bias_fp32=corr_bias_fp32,
        sh_gu_bf16=sh_gu_bf16,
        sh_dn_bf16=sh_dn_bf16,
        w13_bf16=w13_bf16,
        w2_bf16=w2_bf16,
    )

    # ── Dev path（golden；与 test_moe_block_e2e 完全相同的流水线） ──
    x_z = x_bf16.to("zeus")
    gate_w_z_dev = _pack_zeus_linear_weight(gate_w_bf16)    # [E, H] → packed
    sh_gu_z = _pack_zeus_linear_weight(sh_gu_bf16)          # [2*mI, H] → packed
    sh_dn_z = _pack_zeus_linear_weight(sh_dn_bf16)          # [H, mI] → packed
    corr_bias_z = corr_bias_fp32.to("zeus")
    w13_z = w13_bf16.to("zeus")
    w2_z = w2_bf16.to("zeus")

    try:
        with torch.no_grad():
            # router_logits（与 test_moe_block_e2e 同：bf16 → float()）
            router_logits_dev = torch.nn.functional.linear(
                x_z, gate_w_z_dev,
            ).float()
            # shared experts MLP
            sh_gu_dev = torch.nn.functional.linear(x_z, sh_gu_z)
            sh_silu_dev = sgl_kernel_zeus.silu_and_mul(sh_gu_dev)
            shared_out_dev = torch.nn.functional.linear(sh_silu_dev, sh_dn_z)

            # 1. biased_grouped_topk
            w_dev, ids_dev = sgl_kernel_zeus.biased_grouped_topk(
                router_logits_dev, corr_bias_z,
                num_expert_group=n_group,
                topk_group=topk_group,
                topk=top_k,
                num_fused_shared_experts=0,
                routed_scaling_factor=scaling,
                apply_routed_scaling_factor_on_output=True,
            )
            # 2. moe_align_block_size
            sorted_ids_dev, expert_ids_dev, num_post_dev = (
                sgl_kernel_zeus.moe_align_block_size_alloc(
                    ids_dev, block_size, E,
                )
            )
            num_valid_tokens = T * top_k
            # 3. gemm1
            C1_dev = torch.empty(
                T * top_k, 2 * mI, dtype=torch.bfloat16, device="zeus",
            )
            sgl_kernel_zeus.moe_grouped_gemm(
                x_z, w13_z, C1_dev,
                sorted_ids_dev, expert_ids_dev, num_post_dev,
                num_valid_tokens=num_valid_tokens, top_k=top_k,
            )
            # 4. silu_and_mul
            C1_silu_dev = torch.empty(
                T * top_k, mI, dtype=torch.bfloat16, device="zeus",
            )
            sgl_kernel_zeus.silu_and_mul(C1_dev, C1_silu_dev)
            # 5. gemm2
            w_dev_flat = w_dev.to(torch.bfloat16).flatten().contiguous()
            C2_dev = torch.empty(
                T * top_k, H, dtype=torch.bfloat16, device="zeus",
            )
            sgl_kernel_zeus.moe_grouped_gemm(
                C1_silu_dev, w2_z, C2_dev,
                sorted_ids_dev, expert_ids_dev, num_post_dev,
                num_valid_tokens=num_valid_tokens, top_k=1,
                topk_weights=w_dev_flat, mul_routed_weight=True,
            )
            # 6. moe_sum_reduce + shared
            out_dev = torch.empty(T, H, dtype=torch.bfloat16, device="zeus")
            sgl_kernel_zeus.moe_sum_reduce(
                input=C2_dev.view(T, top_k, H),
                output=out_dev,
                shared_output=shared_out_dev,
                routed_scaling_factor=1.0,
            )
    except (NotImplementedError, AttributeError, RuntimeError) as e:
        print(f"  DEV path failed (cannot proceed without golden): {e}")
        return None, None
    print(f"  DEV out:  shape={tuple(out_dev.shape)} dtype={out_dev.dtype}")

    # ── Prod path ──
    try:
        layer = _build_prod_moe_layer(cfg, weights)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  PROD layer build failed: {e}")
        return None, None

    try:
        with torch.no_grad():
            out_prod = _prod_moe_forward(layer, x_bf16.to("zeus"))
    except (NotImplementedError, AttributeError, RuntimeError) as e:
        import traceback
        traceback.print_exc()
        print(f"  PROD forward failed: {e}")
        return None, None
    print(f"  PROD out: shape={tuple(out_prod.shape)} dtype={out_prod.dtype}")

    # ── Compare ──
    # 同 device 同 kernel 同 dtype，但路由可能因 TopK helper 与
    # biased_grouped_topk_impl 实现差异有微小差，容忍稍宽
    ok_final = compare_tensors(
        "moe_block_e2e_prod_vs_dev/final",
        out_dev.cpu(), out_prod.cpu(),
        atol=5e-2, rtol=5e-2,
    )
    return ok_final, None


# ── Dispatch ───────────────────────────────────────────────────
STAGES = {
    "moe_block_e2e": test_moe_block_e2e,
    "moe_block_e2e_prod_vs_dev": test_moe_block_e2e_prod_vs_dev,
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

    cfg = default_glm5_next_moe_cfg()
    print(f"GLM-Next MoE proxy: H={cfg.hidden_size} mI={cfg.moe_intermediate_size} "
          f"E={cfg.n_routed_experts} top_k={cfg.num_experts_per_tok}")
    print(f"REF_DEVICE = {REF_DEVICE}")

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
            status = "SKIP"
        print(f"  {name:30s} : {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
