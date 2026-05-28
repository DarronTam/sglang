"""
GLM5-Next sampler 独立模块 + REF↔Zeus 对拍.

把 logits[T, V] → next_token_ids[T] 这段抽成独立 module ``Glm5NextSampler``,
覆盖两种典型 mode:

  - **greedy**:      top_k=1, temperature=1.0  → bit-exact 等价 argmax(logits)
  - **stochastic**:  top_k=K, top_p=P, temperature=temp  → 输出 token 必须落在
    REF 的 top-K∩top-P 候选集内 (membership check; 不能 bit-equal 因为 Zeus
    `sampling_from_logits` 内部 RNG state 与 torch.multinomial 不可对齐).

形状 (从 config json 读真实 vocab):
  - 16b / next:  V=154880  → logits[T, V] fp32

Chain (decode-only, 单 device, TP=1):
  - logits[T, V] fp32   (上游 dev_lm_head.forward_zeus 的直出)
  - REF greedy:        next = logits.argmax(dim=-1).int32
  - REF stochastic:    softmax(logits/T) → top_k filter → top_p filter
                       → multinomial(generator)
  - Zeus:              sgl_kernel_zeus.sampling_from_logits(
                         logits, temperatures, top_k, top_p, ...,
                         generator=None | Generator,
                       ) → int32 [T]

参考代码:
  - reference glm5_next.py:1055-1057  (logits_processor → next_token_ids 这条
    out-of-band sampler 路径)
  - sgl-kernel-zeus 的 ``sampling_from_logits`` 与 flashinfer 同语义:
    deterministic 控制 sort/reduce 顺序, 不影响是否 argmax/sample (sample 需要
    generator). 这里 greedy 走 top_k=1 是 SGLang 通用约定.

不做的事:
  - min_p / 复杂 penalty (留给上层 sampling_params 装配, 与 sampler kernel 解耦)
  - speculative MTP 重打分 (NextN 暂不上)

用法:
  python glm5next_modules/dev_sampler.py                  # 16b / both stages
  python glm5next_modules/dev_sampler.py --config next
  python glm5next_modules/dev_sampler.py --mode zeus
  python glm5next_modules/dev_sampler.py --stage greedy   # 只跑 greedy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

# 公共脚手架
import _common
from _common import (
    ZEUS_IMPORT_ERROR, sgl_kernel_zeus,
    config_path, zeus_chain_available,
    make_argparser, print_header, print_summary,
)


# ── Config ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class Glm5NextSamplerConfig:
    V: int
    name: str = "proxy"

    @classmethod
    def from_json(cls, path: Path, name: Optional[str] = None) -> "Glm5NextSamplerConfig":
        import json
        raw = json.loads(Path(path).read_text())
        return cls(V=int(raw["vocab_size"]), name=name or Path(path).stem)


def load_cfg(which: str) -> Glm5NextSamplerConfig:
    return Glm5NextSamplerConfig.from_json(config_path(which), name=which)


# ── REF math helpers ────────────────────────────────────────────
def _ref_greedy(logits: torch.Tensor) -> torch.Tensor:
    """REF greedy: argmax over V, 输出 int32."""
    return logits.argmax(dim=-1).to(torch.int32)


def _ref_topk_topp_mask(
    logits: torch.Tensor, top_k: int, top_p: float, temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """REF top-k ∩ top-p 候选集合.

    返回 (kept_mask[T, V] bool, probs[T, V] fp32) — kept_mask 标记最终允许采样的
    token, probs 是 mask 之后 re-norm 的 softmax 分布.

    Filter order 与 Zeus 默认 ``filter_apply_order='top_k_first'`` 一致 (先按
    top_k 截断, 再按 top_p 累积截断).
    """
    T, V = logits.shape
    # temperature
    scaled = logits.float() / max(temperature, 1e-8)

    # top-k mask: 保留 top_k 大的, 其余 -inf
    if top_k > 0 and top_k < V:
        topk_vals, _ = scaled.topk(top_k, dim=-1)
        thresh = topk_vals[:, -1:].expand_as(scaled)
        kept_k = scaled >= thresh
    else:
        kept_k = torch.ones_like(scaled, dtype=torch.bool)

    masked = scaled.masked_fill(~kept_k, float("-inf"))

    # top-p mask on the survivors
    if top_p < 1.0:
        sorted_vals, sorted_idx = masked.sort(dim=-1, descending=True)
        probs_sorted = torch.softmax(sorted_vals, dim=-1)
        cum = probs_sorted.cumsum(dim=-1)
        # 保留累积到首次超过 top_p 的 (含该位置), 其余踢掉
        keep_sorted = cum - probs_sorted < top_p
        # scatter back
        kept_p = torch.zeros_like(kept_k)
        kept_p.scatter_(dim=-1, index=sorted_idx, src=keep_sorted)
        kept = kept_k & kept_p
    else:
        kept = kept_k

    final_logits = scaled.masked_fill(~kept, float("-inf"))
    probs = torch.softmax(final_logits, dim=-1)
    return kept, probs


# ── Module ──────────────────────────────────────────────────────
class Glm5NextSampler:
    """GLM5-Next sampler (greedy + top-k/top-p/temperature).

    使用模式::

        sampler = Glm5NextSampler(cfg)
        # REF
        tok_g = sampler.forward_greedy(logits)
        tok_s = sampler.forward_stochastic(logits, top_k=K, top_p=P,
                                           temperature=temp, seed=42)
        # Zeus
        tok_g_z = sampler.forward_zeus_greedy(logits_z)
        tok_s_z = sampler.forward_zeus_stochastic(logits_z, top_k=K, top_p=P,
                                                  temperature=temp, seed=42)

    无 weight, 无 ``_pack_zeus`` 阶段 (sampler 是 pure compute).
    """

    def __init__(self, cfg: Glm5NextSamplerConfig):
        self.cfg = cfg

    # ── REF forward ─────────────────────────────────────────────
    def forward_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """REF greedy. ``logits: [T, V] fp32 (host)  -> next[T] int32``"""
        return _ref_greedy(logits)

    def forward_stochastic(
        self, logits: torch.Tensor, *,
        top_k: int, top_p: float, temperature: float, seed: int,
    ) -> torch.Tensor:
        """REF stochastic: top-k ∩ top-p filter → softmax → multinomial.

        ``logits: [T, V] fp32 (host)  -> next[T] int32``.

        与 Zeus 不能 bit-equal (RNG state 不通用), 但同 seed / 同 filter →
        同候选集合, 调用方用 ``forward_stochastic_mask`` 拿 kept mask 做
        membership check.
        """
        _, probs = _ref_topk_topp_mask(logits, top_k, top_p, temperature)
        g = torch.Generator().manual_seed(seed)
        next_tok = torch.multinomial(probs, num_samples=1, generator=g).squeeze(-1)
        return next_tok.to(torch.int32)

    def forward_stochastic_mask(
        self, logits: torch.Tensor, *, top_k: int, top_p: float, temperature: float,
    ) -> torch.Tensor:
        """返回 [T, V] bool kept mask, 给 membership 对拍用."""
        kept, _ = _ref_topk_topp_mask(logits, top_k, top_p, temperature)
        return kept

    # ── Zeus forward ────────────────────────────────────────────
    def forward_zeus_greedy(self, logits_z: torch.Tensor) -> torch.Tensor:
        """Zeus greedy: top_k=1 → bit-exact argmax. ``logits: [T, V] fp32 (zeus)``"""
        T = logits_z.shape[0]
        temps = torch.ones(T, dtype=torch.float32, device="zeus")
        return sgl_kernel_zeus.sampling_from_logits(
            logits_z, temps, top_k=1, top_p=1.0,
        )

    def forward_zeus_stochastic(
        self, logits_z: torch.Tensor, *,
        top_k: int, top_p: float, temperature: float, seed: int,
    ) -> torch.Tensor:
        """Zeus stochastic: 与 REF 同 filter 配置, 不同 RNG."""
        T = logits_z.shape[0]
        temps = torch.full((T,), temperature, dtype=torch.float32, device="zeus")
        g = torch.Generator(device="zeus").manual_seed(seed)
        return sgl_kernel_zeus.sampling_from_logits(
            logits_z, temps, top_k=top_k, top_p=top_p, generator=g,
        )


# ── Stage runners ───────────────────────────────────────────────
_ZEUS_OPS_REQUIRED = ("sampling_from_logits",)


def _make_logits(T: int, V: int, seed: int) -> torch.Tensor:
    """fp32 logits, 模拟 lm_head 出来的量级 (std~3)."""
    g = torch.Generator().manual_seed(seed + 7)
    return torch.randn(T, V, generator=g, dtype=torch.float32) * 3.0


def _stage_greedy(args, cfg: Glm5NextSamplerConfig) -> Optional[bool]:
    """Greedy stage: REF argmax == Zeus top_k=1, bit-exact."""
    print("\n" + "-" * 60)
    print(f"  Stage: greedy  (top_k=1)")
    print("-" * 60)
    sampler = Glm5NextSampler(cfg)
    logits = _make_logits(args.num_tokens, cfg.V, args.seed)

    ref_tok = None
    if args.mode in ("ref", "both"):
        ref_tok = sampler.forward_greedy(logits)
        print(f"    REF tok = {ref_tok.tolist()}")

    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"    ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
            return None
        try:
            z_tok = sampler.forward_zeus_greedy(logits.to("zeus")).cpu()
            print(f"    ZEUS tok = {z_tok.tolist()}")
            shape_ok = (z_tok.shape == (args.num_tokens,)
                        and z_tok.dtype == torch.int32)
            range_ok = ((z_tok >= 0).all().item()
                        and (z_tok < cfg.V).all().item())
            if ref_tok is not None:
                eq = torch.equal(ref_tok, z_tok)
                print(f"    [greedy] {'PASS' if eq else 'FAIL'} "
                      f"(bit-exact match against argmax)")
                return eq and shape_ok and range_ok
            return shape_ok and range_ok
        except Exception as e:
            import traceback
            print(f"    ZEUS EXCEPTION: {e!r}")
            traceback.print_exc()
            return False
    return True


def _stage_stochastic(args, cfg: Glm5NextSamplerConfig) -> Optional[bool]:
    """Stochastic stage: Zeus 输出 token 必须落在 REF top-k ∩ top-p 候选集合内."""
    print("\n" + "-" * 60)
    print(f"  Stage: stochastic  (top_k={args.top_k}, top_p={args.top_p}, "
          f"temp={args.temperature})")
    print("-" * 60)
    sampler = Glm5NextSampler(cfg)
    logits = _make_logits(args.num_tokens, cfg.V, args.seed)

    kept_mask = None  # [T, V] bool
    if args.mode in ("ref", "both"):
        # REF 输出 token (走 multinomial, 仅做 sanity), 主要 build kept_mask
        ref_tok = sampler.forward_stochastic(
            logits, top_k=args.top_k, top_p=args.top_p,
            temperature=args.temperature, seed=args.seed,
        )
        kept_mask = sampler.forward_stochastic_mask(
            logits, top_k=args.top_k, top_p=args.top_p,
            temperature=args.temperature,
        )
        kept_counts = kept_mask.sum(dim=-1).tolist()
        print(f"    REF tok = {ref_tok.tolist()}  (kept set sizes={kept_counts})")

    if args.mode in ("zeus", "both"):
        if not zeus_chain_available(*_ZEUS_OPS_REQUIRED):
            print(f"    ZEUS: SKIP (chain unavailable: {ZEUS_IMPORT_ERROR})")
            return None
        try:
            z_tok = sampler.forward_zeus_stochastic(
                logits.to("zeus"),
                top_k=args.top_k, top_p=args.top_p,
                temperature=args.temperature, seed=args.seed,
            ).cpu()
            print(f"    ZEUS tok = {z_tok.tolist()}")
            shape_ok = (z_tok.shape == (args.num_tokens,)
                        and z_tok.dtype == torch.int32)
            range_ok = ((z_tok >= 0).all().item()
                        and (z_tok < cfg.V).all().item())
            if kept_mask is not None:
                # membership: 每行 z_tok 必须在 kept_mask 标记的候选里
                rows = torch.arange(args.num_tokens)
                in_set = kept_mask[rows, z_tok.long()]
                membership_ok = in_set.all().item()
                print(f"    [stochastic] {'PASS' if membership_ok else 'FAIL'} "
                      f"(all sampled tokens ∈ REF top-k∩top-p set)")
                return membership_ok and shape_ok and range_ok
            return shape_ok and range_ok
        except Exception as e:
            import traceback
            print(f"    ZEUS EXCEPTION: {e!r}")
            traceback.print_exc()
            return False
    return True


def _run_stage(args) -> Optional[bool]:
    print("\n" + "=" * 60)
    print(f"Stage: {args.config} (real vocab)")
    print("=" * 60)
    cfg = load_cfg(args.config)
    print(f"  cfg: V={cfg.V}")

    stages = []
    if args.stage in ("greedy", "both"):
        stages.append(("greedy", _stage_greedy(args, cfg)))
    if args.stage in ("stochastic", "both"):
        stages.append(("stochastic", _stage_stochastic(args, cfg)))

    print()
    print(f"  sub-stages: {stages}")
    if any(r is False for _, r in stages):
        return False
    if all(r is True for _, r in stages):
        return True
    return None  # 全 SKIP


def _add_extra_args(parser):
    parser.add_argument(
        "--stage", choices=["greedy", "stochastic", "both"], default="both",
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)


def main():
    parser = make_argparser(
        "dev_sampler", description="GLM5-Next sampler dev test",
        extra_setup=_add_extra_args,
    )
    args = parser.parse_args()
    print_header("GLM5-Next sampler", args)
    ok = _run_stage(args)
    print_summary(f"glm5next_sampler ({args.config})", ok)


if __name__ == "__main__":
    main()
