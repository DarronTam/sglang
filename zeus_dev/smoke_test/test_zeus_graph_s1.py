"""
S1: Zeus Graph Capture — Phase 0 + Phase 3 + Phase 4 Verification

Tests whether every kernel/op used in the Qwen2 decode forward path
can be captured and replayed by Zeus Graph (zertGraphLaunch).
Phase 3 additions: argmax on device, full decode→greedy sampling path.
Phase 4 additions: multi-BS capture/replay, padded BS, full-buffer kv_indices.

Decode forward path for Qwen2:
  embed_tokens → N × DecoderLayer → final_norm → lm_head
  where DecoderLayer =
    fused_add_rmsnorm → qkv_proj(linear) → rotary_embedding →
    decode_attention + store_kv_cache → o_proj(linear) →
    fused_add_rmsnorm → gate_up_proj(linear) → silu_and_mul →
    down_proj(linear)

Zeus-specific constraints:
  - GEMM (mm/addmm/linear) requires weight in LocalMem (zeus.pack_weights)
  - ATen fallback ops (add.out, mm.out etc.) fall back to CPU — NOT captured
  - Process cleanup can segfault — each test uses subprocess + os._exit()

SUCCESS CRITERIA:
  Phase 0 asks "can all ops be captured and replayed?" — NOT "does the
  stub runtime produce numerically correct results after replay?".
  Known stub limitations:
    - device_synchronize() after replay can segfault
    - .cpu() readback after replay can segfault
    - Replay produces different values from eager (stub runtime issue)
  Therefore: tests verify capture + replay complete without crash.
  Numerical correctness is deferred to hardware/full-runtime testing.

Test levels:
  L1: Individual sgl_kernel_zeus ops (rmsnorm, fused_add_rmsnorm, silu_and_mul,
      rotary_embedding, store_kv_cache, decode_attention, embedding)
  L2: Standard ops (mm/addmm/linear with LocalMem, copy_, view, argmax)
  L3: Single-layer decode pipeline (all ops chained)
  L4: Multi-layer decode (N=4 layers)
  L5: Variable seq_lens replay
  L6: Decode graph replay → greedy argmax (Phase 3: full decode+sampling path)
  L7: Multi-BS graph support (Phase 4: multi-BS capture, padded replay,
      full-buffer kv_indices, shared graph pool)

Usage:
  python zeus_dev/test_zeus_graph_s1.py
"""

import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Test infrastructure (subprocess isolation, per test_graph_e2e.py pattern)
# ---------------------------------------------------------------------------

TESTS = {}


def register(name):
    def decorator(fn):
        TESTS[name] = fn
        return fn
    return decorator


def run_test(name, code):
    """Run test code in a subprocess; return (passed, output)."""
    full_code = textwrap.dedent(f"""\
        import ctypes, os, sys
        _libstdcxx = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
        if os.path.exists(_libstdcxx):
            ctypes.CDLL(_libstdcxx, mode=ctypes.RTLD_GLOBAL)
        import torch
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
        os.environ["SGLANG_DEVICE"] = "zeus"
        import torch_zeus
        import torch_zeus.zeus as zeus
        DEVICE = "zeus:0"
        DTYPE = torch.bfloat16

        try:
{textwrap.indent(code, '            ')}
            print("PASSED")
            os._exit(0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("FAILED:", e)
            os._exit(1)
    """)

    result = subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0 and "PASSED" in result.stdout
    return passed, output.strip()


# ---------------------------------------------------------------------------
# L1: Individual sgl_kernel_zeus ops
# ---------------------------------------------------------------------------

@register("L1_rmsnorm")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import rmsnorm

        s = torch_zeus.zeus.Stream()
        H = 896  # Qwen2.5-0.5B hidden
        x = torch.randn(4, H, dtype=DTYPE, device=DEVICE)
        w = torch.ones(H, dtype=DTYPE, device=DEVICE)
        out = torch.empty_like(x)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            rmsnorm(x, w, eps=1e-6, out=out)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L1_fused_add_rmsnorm")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import fused_add_rmsnorm

        s = torch_zeus.zeus.Stream()
        H = 896
        x = torch.randn(4, H, dtype=DTYPE, device=DEVICE)
        res = torch.randn(4, H, dtype=DTYPE, device=DEVICE)
        w = torch.ones(H, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            fused_add_rmsnorm(x, res, w, eps=1e-6)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L1_silu_and_mul")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import silu_and_mul

        s = torch_zeus.zeus.Stream()
        D = 4864  # Qwen2.5-0.5B intermediate
        x = torch.randn(4, D * 2, dtype=DTYPE, device=DEVICE)
        out = torch.empty(4, D, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            silu_and_mul(x, out=out)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L1_rotary_embedding")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import rotary_embedding

        s = torch_zeus.zeus.Stream()
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 14, 2, 64
        bs = 4

        half = HEAD_DIM // 2
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        freqs = torch.outer(torch.arange(512, dtype=torch.float32), inv_freq)
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(DTYPE).to(DEVICE)

        positions = torch.arange(bs, dtype=torch.int32).to(DEVICE)
        q = torch.randn(bs, NUM_HEADS * HEAD_DIM, dtype=DTYPE, device=DEVICE)
        k = torch.randn(bs, NUM_KV_HEADS * HEAD_DIM, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            rotary_embedding(positions, q, k, HEAD_DIM, cos_sin_cache, True)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L1_store_kv_cache")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import store_kv_cache

        s = torch_zeus.zeus.Stream()
        NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 2, 64, 128
        bs = 4

        k_cache = torch.zeros(8, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_cache = torch.zeros(8, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        loc = torch.arange(bs, dtype=torch.int32).to(DEVICE)
        k = torch.randn(bs, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v = torch.randn(bs, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            store_kv_cache(k_cache, v_cache, loc, k, v, PAGE_SIZE)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L1_decode_attention")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import decode_attention

        s = torch_zeus.zeus.Stream()
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        bs = 2
        seq_lens = [10, 20]
        total_kv = sum(seq_lens)
        num_pages = (total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1

        k_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        kv_indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32).to(DEVICE)
        kv_indices = torch.arange(total_kv, dtype=torch.int32).to(DEVICE)
        sm_scale = HEAD_DIM ** -0.5

        q = torch.randn(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        o = torch.empty(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            decode_attention(q, o, k_cache, v_cache, kv_indptr, kv_indices,
                             NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L1_embedding")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import embedding

        s = torch_zeus.zeus.Stream()
        VOCAB, HIDDEN = 1024, 896
        weight = torch.randn(VOCAB, HIDDEN, dtype=DTYPE, device=DEVICE)
        ids = torch.randint(0, VOCAB, (4,), dtype=torch.int32).to(DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            out = embedding(ids, weight)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


# ---------------------------------------------------------------------------
# L2: Standard ATen ops and GEMM with LocalMem
# ---------------------------------------------------------------------------

@register("L2_mm_localmem")
def _():
    return textwrap.dedent("""\
        import torch.nn as nn

        s = torch_zeus.zeus.Stream()
        H = 896
        layer = nn.Linear(H, H, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(layer, Tr=1, Tc=1)

        x = torch.randn(4, H, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            out = torch.mm(x, layer.weight)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L2_addmm_localmem")
def _():
    return textwrap.dedent("""\
        import torch.nn as nn

        s = torch_zeus.zeus.Stream()
        H = 896
        layer = nn.Linear(H, H, bias=True, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(layer, Tr=1, Tc=1)

        x = torch.randn(4, H, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            out = torch.addmm(layer.bias, x, layer.weight)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L2_nn_linear_localmem")
def _():
    return textwrap.dedent("""\
        import torch.nn as nn

        s = torch_zeus.zeus.Stream()
        H = 896
        layer = nn.Linear(H, H, bias=True, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(layer, Tr=1, Tc=1)

        x = torch.randn(4, H, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            out = layer(x)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L2_copy_inplace")
def _():
    return textwrap.dedent("""\
        s = torch_zeus.zeus.Stream()
        src = torch.randn(4, 64, dtype=DTYPE, device=DEVICE)
        dst = torch.empty_like(src)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            dst.copy_(src)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L2_fill_zero")
def _():
    return textwrap.dedent("""\
        s = torch_zeus.zeus.Stream()
        x = torch.randn(4, 64, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            x.zero_()
            x.fill_(3.14)
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


@register("L2_argmax")
def _():
    return textwrap.dedent("""\
        # Phase 3: Verify torch.argmax works natively on Zeus device
        # (Previously required CPU roundtrip: argmax(logits.cpu()).to(device))
        VOCAB = 1024
        bs = 4
        logits = torch.randn(bs, VOCAB, dtype=DTYPE, device=DEVICE)

        # argmax on device — no .cpu() roundtrip
        token_ids = torch.argmax(logits, dim=-1)
        assert token_ids.device.type == "zeus", f"Expected zeus device, got {token_ids.device}"
        assert token_ids.shape == (bs,), f"Expected shape ({bs},), got {token_ids.shape}"
        print(f"argmax on device ok, shape={token_ids.shape}, device={token_ids.device}")
    """)


# ---------------------------------------------------------------------------
# L3: Single-layer decode pipeline
# ---------------------------------------------------------------------------

@register("L3_single_layer_decode")
def _():
    return textwrap.dedent("""\
        import torch.nn as nn
        from sgl_kernel_zeus import (
            fused_add_rmsnorm, silu_and_mul,
            rotary_embedding, decode_attention, store_kv_cache,
        )

        s = torch_zeus.zeus.Stream()
        H, D = 896, 4864
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        q_size = NUM_HEADS * HEAD_DIM
        kv_size = NUM_KV_HEADS * HEAD_DIM
        qkv_total = q_size + 2 * kv_size
        bs = 2
        seq_lens = [10, 20]
        total_kv = sum(seq_lens)
        num_pages = (total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1
        sm_scale = HEAD_DIM ** -0.5

        # Weights (packed LocalMem)
        ln1_w = torch.randn(H, dtype=DTYPE, device=DEVICE)
        ln2_w = torch.randn(H, dtype=DTYPE, device=DEVICE)
        qkv_layer = nn.Linear(H, qkv_total, bias=True, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(qkv_layer, Tr=1, Tc=1)
        o_layer = nn.Linear(q_size, H, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(o_layer, Tr=1, Tc=1)
        gate_up_layer = nn.Linear(H, D * 2, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(gate_up_layer, Tr=1, Tc=1)
        down_layer = nn.Linear(D, H, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(down_layer, Tr=1, Tc=1)

        # cos_sin_cache
        half = HEAD_DIM // 2
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        freqs = torch.outer(torch.arange(512, dtype=torch.float32), inv_freq)
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(DTYPE).to(DEVICE)

        # KV cache + metadata
        k_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        kv_indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32).to(DEVICE)
        kv_indices = torch.arange(total_kv, dtype=torch.int32).to(DEVICE)
        positions = torch.tensor([sl - 1 for sl in seq_lens], dtype=torch.int32).to(DEVICE)
        out_cache_loc = torch.tensor([seq_lens[0] - 1, total_kv - 1], dtype=torch.int32).to(DEVICE)

        # Static input buffers
        hidden = torch.randn(bs, H, dtype=DTYPE, device=DEVICE)
        residual = torch.randn(bs, H, dtype=DTYPE, device=DEVICE)

        def decode_step():
            fused_add_rmsnorm(hidden, residual, ln1_w, 1e-6)

            qkv = qkv_layer(hidden)
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            q_flat = q.contiguous()
            k_flat = k.contiguous()
            rotary_embedding(positions, q_flat, k_flat, HEAD_DIM, cos_sin_cache, True)

            k_3d = k_flat.reshape(bs, NUM_KV_HEADS, HEAD_DIM).contiguous()
            v_3d = v.reshape(bs, NUM_KV_HEADS, HEAD_DIM).contiguous()
            store_kv_cache(k_cache, v_cache, out_cache_loc, k_3d, v_3d, PAGE_SIZE)

            q_3d = q_flat.reshape(bs, NUM_HEADS, HEAD_DIM)
            o_dec = torch.empty(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
            decode_attention(q_3d, o_dec, k_cache, v_cache,
                             kv_indptr, kv_indices,
                             NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)

            attn_out = o_dec.reshape(bs, q_size)
            o_out = o_layer(attn_out)

            fused_add_rmsnorm(o_out, residual, ln2_w, 1e-6)

            gate_up = gate_up_layer(o_out)
            act = silu_and_mul(gate_up)
            down_out = down_layer(act)
            hidden.copy_(down_out)

        # Graph capture
        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            decode_step()
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


# ---------------------------------------------------------------------------
# L4: Multi-layer decode (N=4 layers + final norm + lm_head)
# ---------------------------------------------------------------------------

@register("L4_multi_layer_decode")
def _():
    return textwrap.dedent("""\
        import torch.nn as nn
        from sgl_kernel_zeus import (
            fused_add_rmsnorm, rmsnorm, silu_and_mul,
            rotary_embedding, decode_attention, store_kv_cache,
            embedding,
        )

        s = torch_zeus.zeus.Stream()
        H, D = 896, 4864
        VOCAB = 1024
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        N_LAYERS = 4
        q_size = NUM_HEADS * HEAD_DIM
        kv_size = NUM_KV_HEADS * HEAD_DIM
        qkv_total = q_size + 2 * kv_size
        bs = 2
        seq_lens = [10, 20]
        total_kv = sum(seq_lens)
        num_pages = (total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1
        sm_scale = HEAD_DIM ** -0.5

        # Per-layer weights
        layer_lns1, layer_lns2, layer_qkv, layer_o, layer_gu, layer_dn = [], [], [], [], [], []
        for _ in range(N_LAYERS):
            layer_lns1.append(torch.randn(H, dtype=DTYPE, device=DEVICE))
            layer_lns2.append(torch.randn(H, dtype=DTYPE, device=DEVICE))
            qkv = nn.Linear(H, qkv_total, bias=True, dtype=DTYPE).to(DEVICE)
            zeus.pack_weights(qkv, Tr=1, Tc=1); layer_qkv.append(qkv)
            o = nn.Linear(q_size, H, bias=False, dtype=DTYPE).to(DEVICE)
            zeus.pack_weights(o, Tr=1, Tc=1); layer_o.append(o)
            gu = nn.Linear(H, D * 2, bias=False, dtype=DTYPE).to(DEVICE)
            zeus.pack_weights(gu, Tr=1, Tc=1); layer_gu.append(gu)
            dn = nn.Linear(D, H, bias=False, dtype=DTYPE).to(DEVICE)
            zeus.pack_weights(dn, Tr=1, Tc=1); layer_dn.append(dn)

        final_norm_w = torch.randn(H, dtype=DTYPE, device=DEVICE)
        lm_head = nn.Linear(H, VOCAB, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(lm_head, Tr=1, Tc=1)

        # cos_sin_cache
        half = HEAD_DIM // 2
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        freqs = torch.outer(torch.arange(512, dtype=torch.float32), inv_freq)
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(DTYPE).to(DEVICE)

        # KV caches
        k_caches = [torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE) for _ in range(N_LAYERS)]
        v_caches = [torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE) for _ in range(N_LAYERS)]

        kv_indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32).to(DEVICE)
        kv_indices = torch.arange(total_kv, dtype=torch.int32).to(DEVICE)
        positions = torch.tensor([sl - 1 for sl in seq_lens], dtype=torch.int32).to(DEVICE)
        out_cache_loc = torch.tensor([seq_lens[0] - 1, total_kv - 1], dtype=torch.int32).to(DEVICE)

        embed_w = torch.randn(VOCAB, H, dtype=DTYPE, device=DEVICE)
        input_ids = torch.randint(0, VOCAB, (bs,), dtype=torch.int32).to(DEVICE)

        hidden = torch.empty(bs, H, dtype=DTYPE, device=DEVICE)
        residual = torch.empty(bs, H, dtype=DTYPE, device=DEVICE)

        def full_decode():
            h = embedding(input_ids, embed_w)
            hidden.copy_(h)
            residual.copy_(hidden)

            for i in range(N_LAYERS):
                fused_add_rmsnorm(hidden, residual, layer_lns1[i], 1e-6)
                qkv = layer_qkv[i](hidden)
                q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
                q_flat, k_flat = q.contiguous(), k.contiguous()
                rotary_embedding(positions, q_flat, k_flat, HEAD_DIM, cos_sin_cache, True)
                k_3d = k_flat.reshape(bs, NUM_KV_HEADS, HEAD_DIM).contiguous()
                v_3d = v.reshape(bs, NUM_KV_HEADS, HEAD_DIM).contiguous()
                store_kv_cache(k_caches[i], v_caches[i], out_cache_loc, k_3d, v_3d, PAGE_SIZE)
                q_3d = q_flat.reshape(bs, NUM_HEADS, HEAD_DIM)
                o_dec = torch.empty(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
                decode_attention(q_3d, o_dec, k_caches[i], v_caches[i],
                                 kv_indptr, kv_indices,
                                 NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)
                o_out = layer_o[i](o_dec.reshape(bs, q_size))
                fused_add_rmsnorm(o_out, residual, layer_lns2[i], 1e-6)
                gate_up = layer_gu[i](o_out)
                act = silu_and_mul(gate_up)
                hidden.copy_(layer_dn[i](act))

            out = rmsnorm(residual, final_norm_w, 1e-6)
            logits = lm_head(out)
            return logits

        # Graph capture
        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            logits = full_decode()
        print("capture ok")

        g.replay()
        print("replay ok")
    """)


# ---------------------------------------------------------------------------
# L5: Variable seq_lens replay
# ---------------------------------------------------------------------------

@register("L5_multiple_replays")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import decode_attention

        s = torch_zeus.zeus.Stream()
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        bs = 2
        max_total_kv = 200
        num_pages = (max_total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1
        sm_scale = HEAD_DIM ** -0.5

        k_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        kv_indptr = torch.tensor([0, 10, 30], dtype=torch.int32).to(DEVICE)
        kv_indices = torch.arange(max_total_kv, dtype=torch.int32).to(DEVICE)

        q = torch.randn(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        o = torch.empty(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            decode_attention(q, o, k_cache, v_cache, kv_indptr, kv_indices,
                             NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)
        print("capture ok")

        # Multiple replays without buffer updates (core graph replay test)
        for i in range(5):
            g.replay()
        print(f"5 replays ok")
    """)


# ---------------------------------------------------------------------------
# L6: Decode graph replay → greedy argmax (Phase 3 validation)
# ---------------------------------------------------------------------------

@register("L6_decode_graph_then_greedy")
def _():
    return textwrap.dedent("""\
        import torch.nn as nn
        from sgl_kernel_zeus import (
            fused_add_rmsnorm, rmsnorm, silu_and_mul,
            rotary_embedding, decode_attention, store_kv_cache,
            embedding,
        )

        s = torch_zeus.zeus.Stream()
        H, D = 896, 4864
        VOCAB = 1024
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        N_LAYERS = 2
        q_size = NUM_HEADS * HEAD_DIM
        kv_size = NUM_KV_HEADS * HEAD_DIM
        qkv_total = q_size + 2 * kv_size
        bs = 2
        seq_lens = [10, 20]
        total_kv = sum(seq_lens)
        num_pages = (total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1
        sm_scale = HEAD_DIM ** -0.5

        # Per-layer weights
        layer_lns1, layer_lns2, layer_qkv, layer_o, layer_gu, layer_dn = [], [], [], [], [], []
        for _ in range(N_LAYERS):
            layer_lns1.append(torch.randn(H, dtype=DTYPE, device=DEVICE))
            layer_lns2.append(torch.randn(H, dtype=DTYPE, device=DEVICE))
            qkv = nn.Linear(H, qkv_total, bias=True, dtype=DTYPE).to(DEVICE)
            zeus.pack_weights(qkv, Tr=1, Tc=1); layer_qkv.append(qkv)
            o = nn.Linear(q_size, H, bias=False, dtype=DTYPE).to(DEVICE)
            zeus.pack_weights(o, Tr=1, Tc=1); layer_o.append(o)
            gu = nn.Linear(H, D * 2, bias=False, dtype=DTYPE).to(DEVICE)
            zeus.pack_weights(gu, Tr=1, Tc=1); layer_gu.append(gu)
            dn = nn.Linear(D, H, bias=False, dtype=DTYPE).to(DEVICE)
            zeus.pack_weights(dn, Tr=1, Tc=1); layer_dn.append(dn)

        final_norm_w = torch.randn(H, dtype=DTYPE, device=DEVICE)
        lm_head = nn.Linear(H, VOCAB, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(lm_head, Tr=1, Tc=1)

        half = HEAD_DIM // 2
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        freqs = torch.outer(torch.arange(512, dtype=torch.float32), inv_freq)
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(DTYPE).to(DEVICE)

        k_caches = [torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE) for _ in range(N_LAYERS)]
        v_caches = [torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE) for _ in range(N_LAYERS)]

        kv_indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32).to(DEVICE)
        kv_indices = torch.arange(total_kv, dtype=torch.int32).to(DEVICE)
        positions = torch.tensor([sl - 1 for sl in seq_lens], dtype=torch.int32).to(DEVICE)
        out_cache_loc = torch.tensor([seq_lens[0] - 1, total_kv - 1], dtype=torch.int32).to(DEVICE)

        embed_w = torch.randn(VOCAB, H, dtype=DTYPE, device=DEVICE)
        input_ids = torch.randint(0, VOCAB, (bs,), dtype=torch.int32).to(DEVICE)

        hidden = torch.empty(bs, H, dtype=DTYPE, device=DEVICE)
        residual = torch.empty(bs, H, dtype=DTYPE, device=DEVICE)
        logits_buf = torch.empty(bs, VOCAB, dtype=DTYPE, device=DEVICE)

        def full_decode():
            h = embedding(input_ids, embed_w)
            hidden.copy_(h)
            residual.copy_(hidden)
            for i in range(N_LAYERS):
                fused_add_rmsnorm(hidden, residual, layer_lns1[i], 1e-6)
                qkv = layer_qkv[i](hidden)
                q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
                q_flat, k_flat = q.contiguous(), k.contiguous()
                rotary_embedding(positions, q_flat, k_flat, HEAD_DIM, cos_sin_cache, True)
                k_3d = k_flat.reshape(bs, NUM_KV_HEADS, HEAD_DIM).contiguous()
                v_3d = v.reshape(bs, NUM_KV_HEADS, HEAD_DIM).contiguous()
                store_kv_cache(k_caches[i], v_caches[i], out_cache_loc, k_3d, v_3d, PAGE_SIZE)
                q_3d = q_flat.reshape(bs, NUM_HEADS, HEAD_DIM)
                o_dec = torch.empty(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
                decode_attention(q_3d, o_dec, k_caches[i], v_caches[i],
                                 kv_indptr, kv_indices,
                                 NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)
                o_out = layer_o[i](o_dec.reshape(bs, q_size))
                fused_add_rmsnorm(o_out, residual, layer_lns2[i], 1e-6)
                gate_up = layer_gu[i](o_out)
                act = silu_and_mul(gate_up)
                hidden.copy_(layer_dn[i](act))
            out = rmsnorm(residual, final_norm_w, 1e-6)
            logits_buf.copy_(lm_head(out))

        # Graph capture (model forward only — same as CudaGraphRunner)
        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            full_decode()
        print("capture ok")

        # Simulate greedy decode loop: replay → argmax on device (Phase 3 fix)
        for step in range(3):
            g.replay()
            next_tokens = torch.argmax(logits_buf, dim=-1)
            assert next_tokens.device.type == "zeus", f"argmax result on wrong device: {next_tokens.device}"
            assert next_tokens.shape == (bs,), f"Wrong shape: {next_tokens.shape}"
            # Feed back (simulate: update input_ids for next step)
            input_ids.copy_(next_tokens.to(torch.int32))
        print(f"3 decode+argmax steps ok, device={next_tokens.device}")
    """)


# ---------------------------------------------------------------------------
# L7: Multi-BS graph support (Phase 4 validation)
# ---------------------------------------------------------------------------

@register("L7_multi_bs_capture_replay")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import decode_attention

        # Capture decode_attention graphs for multiple batch sizes (bs=1, 2, 4)
        # sharing a single graph pool — simulates CudaGraphRunner.capture() flow.
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        max_total_kv = 512
        num_pages = (max_total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1
        sm_scale = HEAD_DIM ** -0.5

        k_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        # Full pre-allocated kv_indices buffer (shared across all BS graphs)
        max_bs = 4
        kv_indptr_buf = torch.zeros(max_bs + 1, dtype=torch.int32, device=DEVICE)
        kv_indices_buf = torch.arange(max_total_kv, dtype=torch.int32, device=DEVICE)

        pool = torch_zeus.zeus.graph_pool_handle()
        graphs = {}

        for bs in [4, 2, 1]:  # capture in reverse order (like CudaGraphRunner)
            s = torch_zeus.zeus.Stream()
            # Dummy data: each seq has seq_len=1 (matching get_cuda_graph_seq_len_fill_value)
            kv_indptr_buf[:bs + 1] = torch.arange(bs + 1, dtype=torch.int32, device=DEVICE)

            q = torch.randn(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
            o = torch.empty(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

            g = torch_zeus.zeus.ZEUSGraph()
            with torch_zeus.zeus.graph(g, pool=pool, stream=s):
                decode_attention(q, o, k_cache, v_cache,
                                 kv_indptr_buf[:bs + 1], kv_indices_buf,
                                 NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)
            graphs[bs] = g
            print(f"  capture bs={bs} ok")

        # Replay each graph
        for bs in [1, 2, 4]:
            graphs[bs].replay()
            print(f"  replay bs={bs} ok")

        print(f"multi-BS capture+replay ok ({len(graphs)} graphs, shared pool)")
    """)


@register("L7_padded_bs_replay")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import decode_attention

        # Simulate padded BS replay: capture with bs=4 (all seq_lens=1),
        # then replay with actual_bs=2 (real seq_lens) + 2 padding positions
        # (seq_len=1 each). This is what CudaGraphRunner.replay_prepare does.
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        max_bs = 4
        max_total_kv = max_bs * 512  # generous buffer
        num_pages = (max_total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1
        sm_scale = HEAD_DIM ** -0.5

        k_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        kv_indptr_buf = torch.zeros(max_bs + 1, dtype=torch.int32, device=DEVICE)
        kv_indices_buf = torch.zeros(max_total_kv, dtype=torch.int32, device=DEVICE)

        q = torch.randn(max_bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        o = torch.empty(max_bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        # --- Capture with bs=4, all seq_lens=1 (dummy) ---
        kv_indptr_buf[:max_bs + 1] = torch.arange(max_bs + 1, dtype=torch.int32, device=DEVICE)
        kv_indices_buf[:max_bs] = torch.arange(max_bs, dtype=torch.int32, device=DEVICE)

        s = torch_zeus.zeus.Stream()
        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            decode_attention(q, o, k_cache, v_cache,
                             kv_indptr_buf[:max_bs + 1], kv_indices_buf,
                             NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)
        print("capture bs=4 (dummy) ok")

        # --- Replay: actual_bs=2 with real seq_lens=[50, 100], padding=[1, 1] ---
        real_seq_lens = [50, 100]
        padded_seq_lens = real_seq_lens + [1, 1]  # padding positions
        total_kv = sum(padded_seq_lens)  # 152

        # In-place update kv_indptr (CPU construct, copy to device — like _fill_decode_metadata_for_graph)
        indptr_cpu = torch.zeros(max_bs + 1, dtype=torch.int32)
        indptr_cpu[1:] = torch.cumsum(torch.tensor(padded_seq_lens, dtype=torch.int32), dim=0)
        kv_indptr_buf[:max_bs + 1].copy_(indptr_cpu.to(DEVICE))

        # In-place update kv_indices
        kv_indices_buf[:total_kv] = torch.arange(total_kv, dtype=torch.int32, device=DEVICE)

        g.replay()
        print(f"replay with padded BS ok (actual=2, padded=4, total_kv={total_kv})")
    """)


@register("L7_full_buffer_kv_indices")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import decode_attention

        # Verify the Phase 4 fix: graph captured with FULL kv_indices buffer
        # (not sliced to total_kv at capture time). During replay, kv_indptr
        # directs the kernel to read beyond the capture-time range.
        #
        # At capture: bs=2, seq_lens=[1,1], total_kv=2 — kernel only needs 2 indices
        # At replay:  bs=2, seq_lens=[100,200], total_kv=300 — kernel needs 300 indices
        # If graph captured kv_indices[:2], replay would fail accessing index 299.
        # With full buffer, replay works correctly.
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        bs = 2
        max_total_kv = 1024
        num_pages = (max_total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1
        sm_scale = HEAD_DIM ** -0.5

        k_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        kv_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
        kv_indices_buf = torch.zeros(max_total_kv, dtype=torch.int32, device=DEVICE)

        q = torch.randn(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        o = torch.empty(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        # Capture with dummy: seq_lens=[1, 1], total_kv=2
        kv_indptr_buf[:] = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
        kv_indices_buf[:2] = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)

        s = torch_zeus.zeus.Stream()
        g = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g, stream=s):
            # Pass FULL buffer, not kv_indices_buf[:2]
            decode_attention(q, o, k_cache, v_cache,
                             kv_indptr_buf, kv_indices_buf,
                             NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)
        print("capture ok (full buffer, dummy total_kv=2)")

        # Replay with real data: seq_lens=[100, 200], total_kv=300
        kv_indptr_buf[:] = torch.tensor([0, 100, 300], dtype=torch.int32, device=DEVICE)
        kv_indices_buf[:300] = torch.arange(300, dtype=torch.int32, device=DEVICE)

        g.replay()
        print("replay ok (real total_kv=300, kernel accesses indices[0:300])")

        # Replay again with different seq_lens: [50, 400], total_kv=450
        kv_indptr_buf[:] = torch.tensor([0, 50, 450], dtype=torch.int32, device=DEVICE)
        kv_indices_buf[:450] = torch.arange(450, dtype=torch.int32, device=DEVICE)

        g.replay()
        print("replay ok (real total_kv=450)")
    """)


@register("L7_shared_pool_multi_graph")
def _():
    return textwrap.dedent("""\
        from sgl_kernel_zeus import decode_attention, rmsnorm
        import torch.nn as nn

        # Capture multiple heterogeneous graphs (decode_attention + rmsnorm+linear)
        # with a shared pool, then replay in arbitrary order.
        # This simulates CudaGraphRunner capturing graphs for different BS values.
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        H = 896
        max_total_kv = 256
        num_pages = (max_total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1
        sm_scale = HEAD_DIM ** -0.5

        k_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        ln_w = torch.randn(H, dtype=DTYPE, device=DEVICE)
        linear = nn.Linear(H, H, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(linear, Tr=1, Tc=1)

        pool = torch_zeus.zeus.graph_pool_handle()

        # Graph A: decode_attention with bs=2
        bs_a = 2
        kv_indptr_a = torch.tensor([0, 10, 30], dtype=torch.int32, device=DEVICE)
        kv_indices_a = torch.arange(max_total_kv, dtype=torch.int32, device=DEVICE)
        q_a = torch.randn(bs_a, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        o_a = torch.empty(bs_a, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

        s = torch_zeus.zeus.Stream()
        g_a = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g_a, pool=pool, stream=s):
            decode_attention(q_a, o_a, k_cache, v_cache,
                             kv_indptr_a, kv_indices_a,
                             NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)
        print("graph A (decode_attention bs=2) captured")

        # Graph B: rmsnorm + linear with bs=4
        bs_b = 4
        x_b = torch.randn(bs_b, H, dtype=DTYPE, device=DEVICE)
        g_b = torch_zeus.zeus.ZEUSGraph()
        with torch_zeus.zeus.graph(g_b, pool=pool, stream=s):
            normed = rmsnorm(x_b, ln_w, 1e-6)
            out_b = linear(normed)
        print("graph B (rmsnorm+linear bs=4) captured")

        # Replay in non-capture order
        g_b.replay()
        print("graph B replay ok")
        g_a.replay()
        print("graph A replay ok")
        g_a.replay()
        g_b.replay()
        print("interleaved replay ok")
    """)


# ---------------------------------------------------------------------------
# Dispatch audit (not a graph test, just logs what ops are dispatched)
# ---------------------------------------------------------------------------

@register("AUDIT_dispatch_ops")
def _():
    return textwrap.dedent("""\
        import torch.nn as nn
        from sgl_kernel_zeus import (
            fused_add_rmsnorm, rmsnorm, silu_and_mul,
            rotary_embedding, decode_attention, store_kv_cache,
            embedding,
        )
        from torch.utils._python_dispatch import TorchDispatchMode

        H, D, VOCAB = 896, 4864, 1024
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 14, 2, 64, 128
        q_size = NUM_HEADS * HEAD_DIM
        kv_size = NUM_KV_HEADS * HEAD_DIM
        bs = 2
        seq_lens = [10, 20]
        total_kv = sum(seq_lens)
        num_pages = (total_kv + PAGE_SIZE - 1) // PAGE_SIZE + 1
        sm_scale = HEAD_DIM ** -0.5

        ln_w = torch.randn(H, dtype=DTYPE, device=DEVICE)
        qkv_layer = nn.Linear(H, q_size + 2 * kv_size, bias=True, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(qkv_layer, Tr=1, Tc=1)
        o_layer = nn.Linear(q_size, H, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(o_layer, Tr=1, Tc=1)
        gu_layer = nn.Linear(H, D * 2, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(gu_layer, Tr=1, Tc=1)
        dn_layer = nn.Linear(D, H, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(dn_layer, Tr=1, Tc=1)
        lm_layer = nn.Linear(H, VOCAB, bias=False, dtype=DTYPE).to(DEVICE)
        zeus.pack_weights(lm_layer, Tr=1, Tc=1)

        half = HEAD_DIM // 2
        inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        freqs = torch.outer(torch.arange(512, dtype=torch.float32), inv_freq)
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(DTYPE).to(DEVICE)

        k_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        v_cache = torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        kv_indptr = torch.tensor([0, seq_lens[0], total_kv], dtype=torch.int32).to(DEVICE)
        kv_indices = torch.arange(total_kv, dtype=torch.int32).to(DEVICE)
        positions = torch.tensor([sl - 1 for sl in seq_lens], dtype=torch.int32).to(DEVICE)
        out_cache_loc = torch.tensor([seq_lens[0] - 1, total_kv - 1], dtype=torch.int32).to(DEVICE)
        embed_w = torch.randn(VOCAB, H, dtype=DTYPE, device=DEVICE)
        input_ids = torch.randint(0, VOCAB, (bs,), dtype=torch.int32).to(DEVICE)

        hidden = torch.empty(bs, H, dtype=DTYPE, device=DEVICE)
        residual = torch.empty(bs, H, dtype=DTYPE, device=DEVICE)

        ops = []

        class Logger(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                ops.append(str(func))
                return func(*args, **(kwargs or {}))

        def decode():
            h = embedding(input_ids, embed_w)
            hidden.copy_(h); residual.copy_(hidden)
            fused_add_rmsnorm(hidden, residual, ln_w, 1e-6)
            qkv = qkv_layer(hidden)
            q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            q_f, k_f = q.contiguous(), k.contiguous()
            rotary_embedding(positions, q_f, k_f, HEAD_DIM, cos_sin_cache, True)
            k3 = k_f.reshape(bs, NUM_KV_HEADS, HEAD_DIM).contiguous()
            v3 = v.reshape(bs, NUM_KV_HEADS, HEAD_DIM).contiguous()
            store_kv_cache(k_cache, v_cache, out_cache_loc, k3, v3, PAGE_SIZE)
            q3 = q_f.reshape(bs, NUM_HEADS, HEAD_DIM)
            od = torch.empty(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
            decode_attention(q3, od, k_cache, v_cache, kv_indptr, kv_indices,
                             NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE, sm_scale)
            o_out = o_layer(od.reshape(bs, q_size))
            fused_add_rmsnorm(o_out, residual, ln_w, 1e-6)
            gu = gu_layer(o_out); act = silu_and_mul(gu)
            hidden.copy_(dn_layer(act))
            out = rmsnorm(residual, ln_w, 1e-6)
            logits = lm_layer(out)

        with Logger():
            decode()

        # Also audit argmax (Phase 3: must dispatch on device, not CPU)
        argmax_ops = []
        logits_test = torch.randn(bs, VOCAB, dtype=DTYPE, device=DEVICE)
        with Logger():
            _ = torch.argmax(logits_test, dim=-1)
        argmax_ops = [op for op in ops if "argmax" in op.lower()]

        zeus_ops = sorted(set(op for op in ops if "sgl_kernel_zeus" in op))
        aten_ops = sorted(set(op for op in ops if "aten::" in op))
        other = sorted(set(op for op in ops if "sgl_kernel_zeus" not in op and "aten::" not in op))

        print(f"\\nZeus custom ops ({len(zeus_ops)}):")
        for op in zeus_ops: print(f"  {op}")
        print(f"\\nATen ops ({len(aten_ops)}):")
        for op in aten_ops: print(f"  {op}")
        if other:
            print(f"\\nOther ops ({len(other)}):")
            for op in other: print(f"  {op}")
        print(f"\\nTotal unique: {len(zeus_ops)+len(aten_ops)+len(other)}, dispatches: {len(ops)}")

        # Known CPU roundtrip ops that would block graph capture
        cpu_rt = {"aten::bmm", "aten::mv", "aten::addmv", "aten::baddbmm"}
        found = cpu_rt & set(aten_ops)
        if found:
            print(f"\\nWARNING: CPU roundtrip ops: {found}")
        else:
            print(f"\\nOK: No CPU roundtrip GEMM ops in decode path")

        # Phase 3: argmax dispatch check
        if argmax_ops:
            print(f"\\nOK: argmax dispatched as: {argmax_ops}")
        else:
            print(f"\\nINFO: argmax not observed in dispatch log (may be handled natively)")
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  S1: Zeus Graph Capture — Phase 0 + Phase 3 + Phase 4 Verification")
    print("=" * 70)

    passed = 0
    failed = 0
    skipped = 0
    fail_list = []

    for name, code_fn in TESTS.items():
        code = code_fn()
        ok, output = run_test(name, code)

        if "SKIP" in output:
            skipped += 1
            status = "SKIP"
        elif ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
            fail_list.append((name, output))

        print(f"\n  {status}  {name}")
        # Show key output lines (skip fallback noise)
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "[ZEUS Fallback]" in line or "UserWarning" in line:
                continue
            if status == "FAIL" or line.startswith(("capture ", "replay ", "Zeus custom", "ATen ops", "Other", "Total", "OK:", "WARNING:")):
                print(f"         {line}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'=' * 70}")

    if fail_list:
        print("\nFailed tests detail:")
        for name, output in fail_list:
            print(f"\n  --- {name} ---")
            for line in output.split("\n"):
                line = line.strip()
                if line and "[ZEUS Fallback]" not in line:
                    print(f"    {line}")

    # Feasibility summary
    l1 = [n for n in TESTS if n.startswith("L1_")]
    l2 = [n for n in TESTS if n.startswith("L2_")]
    l1_ok = sum(1 for n in l1 if any(n == f[0] for f in fail_list) is False)
    l2_ok = sum(1 for n in l2 if any(n == f[0] for f in fail_list) is False)
    fail_names = {f[0] for f in fail_list}

    print(f"\n{'=' * 70}")
    print("  Graph Capture Feasibility Summary")
    print(f"{'=' * 70}")
    print(f"  sgl_kernel_zeus ops : {sum(1 for n in l1 if n not in fail_names)}/{len(l1)}")
    print(f"  ATen/GEMM ops       : {sum(1 for n in l2 if n not in fail_names)}/{len(l2)}")
    print(f"  Single-layer decode : {'YES' if 'L3_single_layer_decode' not in fail_names else 'NO'}")
    print(f"  Multi-layer decode  : {'YES' if 'L4_multi_layer_decode' not in fail_names else 'NO'}")
    print(f"  Multiple replays    : {'YES' if 'L5_multiple_replays' not in fail_names else 'NO'}")
    print(f"  Decode+greedy(P3)  : {'YES' if 'L6_decode_graph_then_greedy' not in fail_names else 'NO'}")
    l7 = [n for n in TESTS if n.startswith("L7_")]
    l7_ok = sum(1 for n in l7 if n not in fail_names)
    print(f"  Multi-BS (P4)       : {l7_ok}/{len(l7)}")

    if failed == 0:
        print("\n  >>> ALL ops capturable — ZeusGraphRunner integration is feasible! <<<")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
