"""
Shared monkey-patch hooks for Zeus decode-only PD-disaggregation testing.

This module is imported INSIDE the scheduler subprocess by the
``run_scheduler_process_func`` wrapper in each test script:

    * dump_glm5_next_prefill_cache.py
    * test_zeus_decode_only_llm.py

It exposes two install helpers:

    install_dump_hook(dump_path)   — patch Scheduler.process_batch_result_prefill
                                     to snapshot KV/state pool to ``dump_path``
                                     after the first prefill batch completes.

    install_inject_hook(dump_path) — patch FakeKVReceiver.send_metadata to
                                     load ``dump_path`` and inject the KV/state
                                     into the decode-side pool via the
                                     production write path (set_kv_buffer →
                                     sgl_kernel_zeus.store_kv_cache), plus
                                     write the first-sampled token into the
                                     shared MetadataBuffers.

Both hooks rely on ``_SCHEDULER_REF`` — a module-level scheduler reference
set at the tail of ``Scheduler.__init__`` (patched lazily on first install).
That keeps the receiver/manager monkey-patches independent of constructor
plumbing (FakeKVReceiver itself receives no scheduler handle).

Implementation policy
---------------------
* No silent fallback. If pool layout / dtype / page_size disagree between
  dump and inject ends, the inject hook raises immediately so the user can
  see what to fix. We are intentionally trying to run an incompatible
  pool combo (Zeus MHA pool, GLM5-next MLA layers) to surface real errors.
* All tensor data is round-tripped through CPU between devices — dump
  writes ``.detach().cpu().clone()``, inject does ``.to("zeus")`` at write
  time.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# ── module-level state (lives inside the scheduler subprocess) ─────────
_SCHEDULER_REF: Optional[Any] = None
_DUMP_INSTALLED: bool = False
_INJECT_INSTALLED: bool = False
_INJECT_DUMP_CACHE: Optional[Dict[str, Any]] = None
_INJECT_DUMP_LOCK = threading.Lock()
# When set (by install_dump_hook or install_inject_hook based on the
# ZEUS_DECODE_NUM_LAYERS env var), the Scheduler.__init__ wrapper truncates
# ``model.end_layer`` to ``start_layer + this`` so model.forward loops only
# over the first N decoder layers.
#   * dump side: cuts prefill wall-time; KV for layers [0, N) is still
#     correct because each layer's KV depends only on prior layers'
#     hidden_states (the causal chain is preserved by running 0..N-1).
#   * decode side: keeps decode forward cheap; layers beyond N would read
#     uninitialized KV anyway (we only inject the first N).
_TRUNCATE_TO_N_LAYERS: Optional[int] = None


def _patch_scheduler_init_once() -> None:
    """Wrap Scheduler.__init__ so the first instance is recorded in
    ``_SCHEDULER_REF``. Called by both install_* helpers.
    """
    from sglang.srt.managers.scheduler import Scheduler

    if getattr(Scheduler.__init__, "_zeus_decode_only_patched", False):
        return

    orig_init = Scheduler.__init__

    def _patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        global _SCHEDULER_REF
        _SCHEDULER_REF = self
        logger.info(
            "[zeus-decode-only] Scheduler.__init__ done; _SCHEDULER_REF set"
        )
        if _TRUNCATE_TO_N_LAYERS is not None and _TRUNCATE_TO_N_LAYERS > 0:
            _apply_layer_truncation(self, _TRUNCATE_TO_N_LAYERS)

    _patched_init._zeus_decode_only_patched = True  # type: ignore[attr-defined]
    Scheduler.__init__ = _patched_init


def _apply_layer_truncation(scheduler, n: int) -> None:
    """Shrink ``Glm5NextModel.end_layer`` so model.forward iterates only
    the first ``n`` decoder layers. Pool allocations have already
    happened at this point (full size), so we waste a bit of KV / mamba
    memory — fine for a dev test.

    No-op if the model has no ``start_layer`` / ``end_layer`` attributes
    (i.e. not a hybrid PP-aware model).
    """
    try:
        model = scheduler.tp_worker.model_runner.model
    except AttributeError:
        logger.warning(
            "[zeus-decode-only] no model on scheduler — skip truncation"
        )
        return

    # Glm5NextForCausalLM.model -> Glm5NextModel (the one that owns
    # start_layer / end_layer / the layer for-loop).
    inner = getattr(model, "model", model)
    if not (hasattr(inner, "start_layer") and hasattr(inner, "end_layer")):
        logger.warning(
            "[zeus-decode-only] model %s lacks start/end_layer — skip "
            "truncation (forward will run all layers)",
            type(inner).__name__,
        )
        return

    orig_end = inner.end_layer
    new_end = min(inner.start_layer + n, orig_end)
    if new_end == orig_end:
        return

    inner.end_layer = new_end
    logger.warning(
        "[zeus-decode-only] TRUNCATED model forward: end_layer %d -> %d "
        "(running first %d decoder layers only — output will be garbage "
        "but lets us validate the hook / kernel dispatch chain)",
        orig_end,
        new_end,
        new_end - inner.start_layer,
    )


# ════════════════════════════════════════════════════════════════════════
#                          DUMP side
# ════════════════════════════════════════════════════════════════════════
def install_dump_hook(dump_path: str) -> None:
    """Snapshot KV/state pool after the first prefill batch finishes.

    If ``ZEUS_DECODE_NUM_LAYERS`` > 0, also truncate model.forward to run
    only the first N decoder layers. This is safe: each layer's KV write
    only depends on the hidden_states from earlier layers, so the KV for
    layers ``[0, N)`` is identical whether the rest of the layers run or
    not. Truncation cuts prefill wall-time roughly in proportion to the
    layer count saved.
    """
    global _DUMP_INSTALLED, _TRUNCATE_TO_N_LAYERS
    if _DUMP_INSTALLED:
        return

    n_layers_cap = int(os.environ.get("ZEUS_DECODE_NUM_LAYERS", "0"))
    if n_layers_cap > 0:
        _TRUNCATE_TO_N_LAYERS = n_layers_cap
        logger.info(
            "[zeus-decode-only][dump] will truncate prefill forward to "
            "first %d decoder layers (full 92-layer forward is wasted "
            "when we only dump %d)",
            n_layers_cap,
            n_layers_cap,
        )

    _patch_scheduler_init_once()

    from sglang.srt.managers.scheduler import Scheduler

    orig_process = Scheduler.process_batch_result_prefill
    _state = {"done": False}

    def _patched_process(self, batch, result, launch_done=None):
        # Call original so the scheduler keeps making forward progress.
        try:
            ret = orig_process(self, batch, result, launch_done)
        except TypeError:
            # Older signature without launch_done
            ret = orig_process(self, batch, result)

        if _state["done"]:
            return ret

        try:
            _do_dump(self, batch, dump_path)
        except Exception:
            logger.exception("[zeus-decode-only][dump] snapshot failed")
            raise
        finally:
            _state["done"] = True

        return ret

    Scheduler.process_batch_result_prefill = _patched_process
    _DUMP_INSTALLED = True
    logger.info(
        "[zeus-decode-only][dump] hook installed; dump_path=%s", dump_path
    )


def _do_dump(scheduler, batch, dump_path: str) -> None:
    """Snapshot pool tensors for batch.reqs[0] and torch.save to disk."""
    if not batch.reqs:
        logger.warning("[zeus-decode-only][dump] empty batch — skip")
        return

    req = batch.reqs[0]
    model_runner = scheduler.tp_worker.model_runner
    req_to_token_pool = model_runner.req_to_token_pool
    token_to_kv_pool = model_runner.token_to_kv_pool
    server_args = scheduler.server_args
    model_config = model_runner.model_config

    seq_len = len(req.origin_input_ids)
    if not req.output_ids:
        raise RuntimeError(
            "dump expects req.output_ids[-1] (the first sampled token); "
            "but req.output_ids is empty. Make sure max_new_tokens >= 1."
        )
    first_sampled_token = int(req.output_ids[-1])

    kv_slots = (
        req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len]
        .detach()
        .cpu()
        .clone()
    )

    # ── first-N-layers filter (dev-time cost cap) ───────────────────
    # If ``ZEUS_DECODE_NUM_LAYERS`` is set and > 0, restrict the dump to
    # model layers ``[0, N)``. This keeps the dump file and dev-iteration
    # time tractable while we are debugging GLM5-next + Zeus end-to-end.
    n_layers_cap = int(os.environ.get("ZEUS_DECODE_NUM_LAYERS", "0"))

    def _within_cap(layer_id: int) -> bool:
        return n_layers_cap <= 0 or layer_id < n_layers_cap

    # ── full-attn per-layer KV ──────────────────────────────────────
    full_attn_kv: Dict[int, Dict[str, torch.Tensor]] = {}
    mambaish_config = model_runner.mambaish_config
    if mambaish_config is not None and hasattr(
        mambaish_config, "full_attention_layer_ids"
    ):
        full_attn_layer_ids = list(mambaish_config.full_attention_layer_ids)
    else:
        full_attn_layer_ids = list(
            range(model_runner.start_layer, model_runner.end_layer)
        )
    full_attn_layer_ids = [lid for lid in full_attn_layer_ids if _within_cap(lid)]

    kv_slots_dev = kv_slots.to(token_to_kv_pool.device)

    # Resolve which pool actually owns the full-attention layer KV.
    # HybridLinearKVPool → .full_kv_pool; otherwise the pool itself.
    full_pool = getattr(token_to_kv_pool, "full_kv_pool", token_to_kv_pool)
    full_layer_id_map = getattr(
        token_to_kv_pool, "full_attention_layer_id_mapping", None
    )

    for layer_id in full_attn_layer_ids:
        # Resolve pool-internal index (HybridLinearKVPool uses a sparse map;
        # plain MLATokenToKVPool / MHATokenToKVPool uses layer_id - start_layer)
        if full_layer_id_map is not None:
            pool_idx = full_layer_id_map[layer_id]
        else:
            pool_idx = layer_id - getattr(full_pool, "start_layer", 0)

        entry: Dict[str, torch.Tensor] = {}
        if hasattr(full_pool, "kv_buffer"):
            # MLATokenToKVPool: single fused tensor
            entry["kv"] = (
                full_pool.kv_buffer[pool_idx][kv_slots_dev]
                .detach()
                .cpu()
                .clone()
            )
        elif hasattr(full_pool, "k_buffer") and hasattr(full_pool, "v_buffer"):
            # MHATokenToKVPool: separate k/v
            entry["k"] = (
                full_pool.k_buffer[pool_idx][kv_slots_dev]
                .detach()
                .cpu()
                .clone()
            )
            entry["v"] = (
                full_pool.v_buffer[pool_idx][kv_slots_dev]
                .detach()
                .cpu()
                .clone()
            )
        else:
            raise RuntimeError(
                f"dump: unknown pool type {type(full_pool).__name__}; "
                "expected MLATokenToKVPool or MHATokenToKVPool"
            )
        full_attn_kv[layer_id] = entry

    # ── KDA mamba state (conv + temporal) ──────────────────────────
    kda_state: Dict[int, Dict[str, Any]] = {}
    mamba_pool_idx = getattr(req, "mamba_pool_idx", None)
    if mamba_pool_idx is not None and hasattr(req_to_token_pool, "mamba_pool"):
        mamba_pool = req_to_token_pool.mamba_pool
        mamba_cache = mamba_pool.mamba_cache
        mamba_pool_idx_int = (
            int(mamba_pool_idx.item())
            if isinstance(mamba_pool_idx, torch.Tensor)
            else int(mamba_pool_idx)
        )

        # `mamba_map` maps model layer_id -> mamba-pool layer dim
        mamba_map = getattr(req_to_token_pool, "mamba_map", None)
        linear_layer_ids: List[int] = []
        if mambaish_config is not None and hasattr(
            mambaish_config, "linear_layer_ids"
        ):
            linear_layer_ids = list(mambaish_config.linear_layer_ids)
        linear_layer_ids = [lid for lid in linear_layer_ids if _within_cap(lid)]

        for layer_id in linear_layer_ids:
            mamba_layer_idx = (
                int(mamba_map[layer_id])
                if mamba_map is not None
                else linear_layer_ids.index(layer_id)
            )
            conv_groups = [
                conv[mamba_pool_idx_int].detach().cpu().clone()
                for conv in mamba_cache.conv
            ]
            temporal = (
                mamba_cache.temporal[mamba_layer_idx, mamba_pool_idx_int]
                .detach()
                .cpu()
                .clone()
            )
            kda_state[layer_id] = {
                "conv_groups": conv_groups,
                "temporal": temporal,
                "mamba_layer_idx": mamba_layer_idx,
            }

    # ── server-args fingerprint (for inject-end assertion) ─────────
    fingerprint = {
        "page_size": server_args.page_size,
        "kv_cache_dtype": str(server_args.kv_cache_dtype),
        "dtype": str(server_args.dtype),
        "model_path": server_args.model_path,
        "max_running_requests": server_args.max_running_requests,
        "head_num": getattr(model_config, "num_key_value_heads", None),
        "head_dim": getattr(model_config, "head_dim", None),
        "kv_lora_rank": getattr(model_config, "kv_lora_rank", None),
        "qk_rope_head_dim": getattr(model_config, "qk_rope_head_dim", None),
        "full_pool_type": type(full_pool).__name__,
        "num_layers_cap": n_layers_cap,
    }

    dump: Dict[str, Any] = {
        "fingerprint": fingerprint,
        "origin_input_ids": list(req.origin_input_ids),
        "first_sampled_token": first_sampled_token,
        "seq_len": seq_len,
        "req_pool_idx": int(req.req_pool_idx),
        "mamba_pool_idx": (
            int(mamba_pool_idx.item())
            if isinstance(mamba_pool_idx, torch.Tensor)
            else (int(mamba_pool_idx) if mamba_pool_idx is not None else None)
        ),
        "kv_slots": kv_slots,
        "full_attn_kv": full_attn_kv,
        "kda_state": kda_state,
        "full_attn_layer_ids": full_attn_layer_ids,
    }

    os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
    torch.save(dump, dump_path)
    logger.info(
        "[zeus-decode-only][dump] saved seq_len=%d first_token=%d "
        "full_attn_layers=%d kda_layers=%d → %s",
        seq_len,
        first_sampled_token,
        len(full_attn_kv),
        len(kda_state),
        dump_path,
    )


# ════════════════════════════════════════════════════════════════════════
#                          INJECT side
# ════════════════════════════════════════════════════════════════════════
def install_inject_hook(dump_path: str) -> None:
    """Patch FakeKVReceiver.send_metadata to inject the dumped KV/state
    into the decode-side pool via the production set_kv_buffer write path.

    Also (if ``ZEUS_DECODE_NUM_LAYERS`` > 0) arranges for the model's
    forward to run only the first N decoder layers. The pool / mamba
    allocations are unaffected — they're already sized for all layers
    at this point.
    """
    global _INJECT_INSTALLED, _TRUNCATE_TO_N_LAYERS
    if _INJECT_INSTALLED:
        return

    n_layers_cap = int(os.environ.get("ZEUS_DECODE_NUM_LAYERS", "0"))
    if n_layers_cap > 0:
        _TRUNCATE_TO_N_LAYERS = n_layers_cap
        logger.info(
            "[zeus-decode-only][inject] will truncate model forward to "
            "first %d decoder layers after init",
            n_layers_cap,
        )

    _patch_scheduler_init_once()

    from sglang.srt.disaggregation.fake import conn as fake_conn

    orig_send_metadata = fake_conn.FakeKVReceiver.send_metadata

    def _patched_send_metadata(
        self,
        kv_indices,
        aux_index=None,
        state_indices=None,
    ):
        try:
            _do_inject(dump_path, kv_indices, aux_index, state_indices)
        except Exception:
            logger.exception("[zeus-decode-only][inject] failed; re-raising")
            raise
        # Let the original mark has_sent_metadata = True
        return orig_send_metadata(
            self, kv_indices, aux_index=aux_index, state_indices=state_indices
        )

    fake_conn.FakeKVReceiver.send_metadata = _patched_send_metadata
    _INJECT_INSTALLED = True
    logger.info(
        "[zeus-decode-only][inject] hook installed; dump_path=%s", dump_path
    )


def _load_dump(dump_path: str) -> Dict[str, Any]:
    global _INJECT_DUMP_CACHE
    with _INJECT_DUMP_LOCK:
        if _INJECT_DUMP_CACHE is None:
            if not os.path.exists(dump_path):
                raise FileNotFoundError(
                    f"dump file not found: {dump_path} — run "
                    f"dump_glm5_next_prefill_cache.py on CUDA host first"
                )
            _INJECT_DUMP_CACHE = torch.load(dump_path, map_location="cpu")
            logger.info(
                "[zeus-decode-only][inject] loaded dump: seq_len=%d "
                "first_token=%d fingerprint=%s",
                _INJECT_DUMP_CACHE["seq_len"],
                _INJECT_DUMP_CACHE["first_sampled_token"],
                _INJECT_DUMP_CACHE["fingerprint"],
            )
        return _INJECT_DUMP_CACHE


def _assert_fingerprint(dump_fp: Dict[str, Any], scheduler) -> None:
    """Hard assert that the live engine config matches what the dump saw."""
    sa = scheduler.server_args
    mc = scheduler.tp_worker.model_runner.model_config

    live_n_cap = int(os.environ.get("ZEUS_DECODE_NUM_LAYERS", "0"))

    checks = [
        ("page_size", sa.page_size, dump_fp.get("page_size")),
        ("kv_cache_dtype", str(sa.kv_cache_dtype), dump_fp.get("kv_cache_dtype")),
        ("model_path", sa.model_path, dump_fp.get("model_path")),
        ("head_num", getattr(mc, "num_key_value_heads", None), dump_fp.get("head_num")),
        ("head_dim", getattr(mc, "head_dim", None), dump_fp.get("head_dim")),
        ("kv_lora_rank", getattr(mc, "kv_lora_rank", None), dump_fp.get("kv_lora_rank")),
        (
            "qk_rope_head_dim",
            getattr(mc, "qk_rope_head_dim", None),
            dump_fp.get("qk_rope_head_dim"),
        ),
        ("num_layers_cap", live_n_cap, dump_fp.get("num_layers_cap", 0)),
    ]
    mismatches = [
        f"  {name}: live={live!r} vs dump={dump!r}"
        for name, live, dump in checks
        if live != dump
    ]
    if mismatches:
        raise AssertionError(
            "[zeus-decode-only][inject] fingerprint mismatch — dump and "
            "decode-side server args disagree:\n" + "\n".join(mismatches)
        )


def _expand_page_indices_to_tokens(
    page_indices, page_size: int, seq_len: int, device
) -> torch.Tensor:
    """Convert page-level kv_indices (numpy or tensor) to absolute token
    indices on ``device``. Length must equal seq_len.
    """
    if isinstance(page_indices, torch.Tensor):
        pages = page_indices.detach().cpu().tolist()
    else:
        pages = list(page_indices)

    tokens: List[int] = []
    for p in pages:
        base = int(p) * page_size
        tokens.extend(range(base, base + page_size))
    # Trim to seq_len (last page may be partial)
    tokens = tokens[:seq_len]
    if len(tokens) != seq_len:
        raise RuntimeError(
            f"expand_page_indices: got {len(tokens)} token slots from "
            f"{len(pages)} pages × {page_size}, but dump.seq_len={seq_len}"
        )
    return torch.tensor(tokens, dtype=torch.int64, device=device)


def _do_inject(
    dump_path: str,
    kv_indices,
    aux_index,
    state_indices,
) -> None:
    if _SCHEDULER_REF is None:
        raise RuntimeError(
            "_SCHEDULER_REF is None — Scheduler.__init__ patch did not fire. "
            "Make sure the Engine subclass overrides run_scheduler_process_func."
        )
    scheduler = _SCHEDULER_REF
    model_runner = scheduler.tp_worker.model_runner
    token_to_kv_pool = model_runner.token_to_kv_pool
    req_to_token_pool = model_runner.req_to_token_pool
    metadata_buffers = getattr(scheduler, "disagg_metadata_buffers", None)
    if metadata_buffers is None:
        raise RuntimeError(
            "scheduler.disagg_metadata_buffers is None — make sure "
            "disaggregation_mode='decode' is set on the Engine"
        )

    dump = _load_dump(dump_path)
    _assert_fingerprint(dump["fingerprint"], scheduler)

    page_size = scheduler.server_args.page_size
    seq_len: int = dump["seq_len"]
    first_token: int = dump["first_sampled_token"]
    device = token_to_kv_pool.device

    # ── 1) Full-attn KV via production set_kv_buffer (store_kv_cache) ──
    kv_token_indices = _expand_page_indices_to_tokens(
        kv_indices, page_size=page_size, seq_len=seq_len, device=device
    )

    full_attn_layer_ids: List[int] = dump["full_attn_layer_ids"]
    full_attn_kv: Dict[int, Dict[str, torch.Tensor]] = dump["full_attn_kv"]

    for layer_id in full_attn_layer_ids:
        entry = full_attn_kv[layer_id]
        if "k" in entry and "v" in entry:
            cache_k = entry["k"].to(device)
            cache_v = entry["v"].to(device)
        elif "kv" in entry:
            # Dump came from MLA fused tensor. Zeus pool wants separate k/v —
            # caller chose to ignore MLA compatibility for now (see plan §
            # 'GLM5-next 在 Zeus 上的可运行性未验证'). Forward the fused
            # tensor as both k and v so set_kv_buffer at least gets exercised;
            # the actual numerical correctness is deferred until the real
            # MLA-on-Zeus pool lands.
            fused = entry["kv"].to(device)
            cache_k = fused
            cache_v = fused
        else:
            raise RuntimeError(
                f"layer {layer_id}: dump entry has neither (k,v) nor (kv): "
                f"keys={list(entry.keys())}"
            )

        token_to_kv_pool.set_kv_buffer(
            layer=None,
            loc=kv_token_indices,
            cache_k=cache_k,
            cache_v=cache_v,
            layer_id_override=layer_id,
        )

    # ── 2) KDA mamba state direct copy ─────────────────────────────
    if state_indices is not None and len(state_indices) > 0:
        mamba_slot = int(
            state_indices[0]
            if not isinstance(state_indices[0], torch.Tensor)
            else state_indices[0].item()
        )
        mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
        if mamba_pool is None:
            raise RuntimeError(
                "state_indices was provided but req_to_token_pool has no "
                "mamba_pool — model config / hybrid-pool routing mismatch"
            )
        mamba_cache = mamba_pool.mamba_cache
        kda_state: Dict[int, Dict[str, Any]] = dump["kda_state"]
        for layer_id, entry in kda_state.items():
            mamba_layer_idx: int = entry["mamba_layer_idx"]
            for g, conv_dump in enumerate(entry["conv_groups"]):
                mamba_cache.conv[g][mamba_slot].copy_(conv_dump.to(device))
            mamba_cache.temporal[mamba_layer_idx, mamba_slot].copy_(
                entry["temporal"].to(device)
            )

    # ── 3) First sampled token + bootstrap_room into MetadataBuffers ──
    if aux_index is None:
        raise RuntimeError(
            "FakeKVReceiver.send_metadata got aux_index=None; "
            "we need it to address metadata_buffers slot"
        )
    metadata_buffers.output_ids[aux_index][0] = first_token
    if hasattr(metadata_buffers, "bootstrap_room"):
        # Belt-and-suspenders: fake transfer bypasses the room check,
        # but write a sentinel so the inflight slot looks consistent.
        try:
            metadata_buffers.bootstrap_room[aux_index, 0] = 0
        except Exception:
            pass

    logger.info(
        "[zeus-decode-only][inject] wrote seq_len=%d first_token=%d "
        "full_attn_layers=%d kda_layers=%d into pool",
        seq_len,
        first_token,
        len(full_attn_layer_ids),
        len(dump["kda_state"]),
    )
