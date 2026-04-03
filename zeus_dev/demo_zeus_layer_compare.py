"""
Qwen2.5-0.5B layer-by-layer CUDA vs Zeus comparison

Progressive porting approach:
  1. Load real model weights from HuggingFace (CPU)
  2. Run component on CUDA -> reference output
  3. Run same component on Zeus -> compare

Currently implemented stages:
  - embedding: nn.Embedding with real Qwen2.5-0.5B weights

Usage:
  python demo_zeus_layer_compare.py                    # run all implemented stages
  python demo_zeus_layer_compare.py --stage embedding  # only embedding
"""

import argparse

import torch
import torch_zeus  # noqa: F401 - registers zeus backend

# Mock server args for SGLang components
from unittest.mock import Mock
import sglang.srt.server_args
dummy_args = Mock()
dummy_args.rl_on_policy_target = None

def mock_get_global_server_args(*args, **kwargs):
    return dummy_args

sglang.srt.server_args.get_global_server_args = mock_get_global_server_args

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def compare_tensors(name, cuda_out, zeus_out, atol=5e-3, rtol=5e-3):
    """Compare CUDA vs Zeus tensor outputs."""
    a = cuda_out.detach().float().cpu()
    b = zeus_out.detach().float().cpu()

    if a.shape != b.shape:
        print(f"  [{name}] SHAPE MISMATCH: cuda={a.shape} zeus={b.shape}")
        return False

    abs_diff = (a - b).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    close = torch.allclose(a, b, atol=atol, rtol=rtol)

    status = "PASS" if close else "DIFF"
    print(
        f"  [{name}] {status} | "
        f"max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  "
        f"shape={list(a.shape)}"
    )
    return close


def load_hf_model():
    """Load HuggingFace model on CPU."""
    print("Loading HuggingFace model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def skip_stage_without_cuda(stage_name, results):
    print()
    print("=" * 60)
    print(f"Stage: {stage_name} (SKIPPED)")
    print("=" * 60)
    print("  CUDA not available; skipping CUDA vs Zeus comparison.")
    results[stage_name] = None


# ── Stage: Embedding ──────────────────────────────────────────
def test_embedding(model, tokenizer):
    print()
    print("=" * 60)
    print("Stage: Embedding (model.embed_tokens)")
    print("=" * 60)

    embed = model.model.embed_tokens  # nn.Embedding on CPU
    text = "Hello, this is a test for Zeus device comparison."
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    print(f"  input text     : '{text}'")
    print(f"  input_ids      : {input_ids.shape} -> {input_ids[0].tolist()}")
    print(f"  embed weight   : {embed.weight.shape} ({embed.weight.dtype})")

    # ── CUDA reference ──
    embed_cuda = embed.to("cuda")
    ids_cuda = input_ids.to("cuda")
    with torch.no_grad():
        out_cuda = embed_cuda(ids_cuda)
    print(f"  CUDA output    : {out_cuda.shape}, device={out_cuda.device}")

    # ── Zeus ──
    embed.cpu()
    embed_zeus = embed.to("zeus")
    ids_zeus = input_ids.to("zeus")
    with torch.no_grad():
        out_zeus = embed_zeus(ids_zeus)
    print(f"  Zeus output    : {out_zeus.shape}, device={out_zeus.device}")

    # ── Compare ──
    ok = compare_tensors("embed_tokens", out_cuda, out_zeus)

    # Spot-check: verify a few embedding vectors by index
    print("  -- Spot check (first 3 tokens) --")
    for i in range(min(3, input_ids.shape[1])):
        tok_id = input_ids[0, i].item()
        cuda_vec = out_cuda[0, i].cpu()
        zeus_vec = out_zeus[0, i].cpu()
        weight_vec = embed.weight.data[tok_id].cpu()  # direct weight lookup
        # Embedding is just a table lookup, so all three should be identical
        match_cz = torch.equal(cuda_vec.half(), zeus_vec.half())
        match_cw = torch.equal(cuda_vec.half(), weight_vec.half())
        print(f"    token[{i}] id={tok_id:6d} | cuda==zeus: {match_cz}  cuda==weight: {match_cw}")

    embed.cpu()
    return ok, out_cuda



# ── Stage: RMSNorm ────────────────────────────────────────────
def test_rmsnorm(model, tokenizer):
    print()
    print("=" * 60)
    print("Stage: RMSNorm (model.model.norm)")
    print("=" * 60)

    # Qwen2.5 final layernorm
    norm = model.model.norm
    hidden_size = model.config.hidden_size
    
    torch.manual_seed(42)
    # sgl_kernel rmsnorm expects 2D (tokens, hidden_size)
    x = torch.randn(11, hidden_size, dtype=torch.bfloat16)

    print(f"  input x        : {x.shape} ({x.dtype})")
    print(f"  norm weight    : {norm.weight.shape} ({norm.weight.dtype})")

    # ── CUDA reference (using SGLang's RMSNorm for fair comparison) ──
    from sglang.srt.layers.layernorm import RMSNorm as SGLangRMSNorm

    sgl_norm_cuda = SGLangRMSNorm(hidden_size=hidden_size, eps=model.config.rms_norm_eps)
    sgl_norm_cuda.weight.data.copy_(norm.weight.data)
    sgl_norm_cuda = sgl_norm_cuda.to("cuda")
    x_cuda = x.to("cuda")
    with torch.no_grad():
        out_cuda = sgl_norm_cuda.forward_cuda(x_cuda)
    print(f"  CUDA output    : {out_cuda.shape}, device={out_cuda.device}")

    # ── Zeus (via SGLang CustomOp) ──
    norm.cpu()
    sgl_norm_zeus = SGLangRMSNorm(hidden_size=hidden_size, eps=model.config.rms_norm_eps)
    sgl_norm_zeus.weight.data.copy_(norm.weight.data)
    sgl_norm_zeus = sgl_norm_zeus.to("zeus")
    x_zeus = x.to("zeus")
    with torch.no_grad():
        out_zeus = sgl_norm_zeus.forward_zeus(x_zeus)
    print(f"  Zeus output    : {out_zeus.shape}, device={out_zeus.device}")

    # ── Compare ──
    ok = compare_tensors("rmsnorm", out_cuda, out_zeus)

    norm.cpu()
    return ok, out_cuda


# ── Stage: SiluAndMul ─────────────────────────────────────────
def test_silu_and_mul(model, tokenizer):
    print()
    print("=" * 60)
    print("Stage: SiluAndMul (SwiGLU Activation)")
    print("=" * 60)

    from sglang.srt.layers.activation import SiluAndMul
    
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    
    torch.manual_seed(42)
    x = torch.randn(1, 11, 2 * intermediate_size, dtype=torch.bfloat16)
    print(f"  input x        : {x.shape} ({x.dtype})")

    # ── CUDA reference ──
    # Instantiate specifically for CUDA
    act_cuda_layer = SiluAndMul().to("cuda")
    x_cuda = x.to("cuda")
    with torch.no_grad():
        out_cuda = act_cuda_layer.forward_cuda(x_cuda)
    print(f"  CUDA output    : {out_cuda.shape}, device={out_cuda.device}")

    # ── Zeus ──
    # Instantiate specifically for Zeus
    act_zeus_layer = SiluAndMul().to("zeus")
    x_zeus = x.to("zeus")
    with torch.no_grad():
        out_zeus = act_zeus_layer.forward_zeus(x_zeus)
    print(f"  Zeus output    : {out_zeus.shape}, device={out_zeus.device}")

    # ── Compare ──
    ok = compare_tensors("silu_and_mul", out_cuda, out_zeus)

    return ok, out_cuda


# ── Stage: RoPE (Rotary Embedding) ────────────────────────────
def test_rope(model, tokenizer):
    print()
    print("=" * 60)
    print("Stage: RoPE (Rotary Embedding)")
    print("=" * 60)

    from sglang.srt.layers.rotary_embedding import get_rope

    config = model.config
    head_size = config.hidden_size // config.num_attention_heads
    rotary_dim = int(head_size * getattr(config, "partial_rotary_factor", 1.0))
    max_position = config.max_position_embeddings
    base = getattr(config, "rope_theta", 10000.0)
    
    # Qwen uses neox style (True)
    is_neox_style = True

    # Instantiate RoPE layer
    rope = get_rope(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling=getattr(config, "rope_scaling", None),
        dtype=torch.bfloat16,
    )

    batch_size = 1
    seq_len = 11
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    
    torch.manual_seed(42)
    positions = torch.arange(seq_len, dtype=torch.int64)  # [num_tokens]
    
    # query: [num_tokens, num_heads * head_size]
    query = torch.randn(batch_size * seq_len, num_heads * head_size, dtype=torch.bfloat16)
    key = torch.randn(batch_size * seq_len, num_kv_heads * head_size, dtype=torch.bfloat16)
    
    print(f"  positions      : {positions.shape}")
    print(f"  query          : {query.shape} ({query.dtype})")
    print(f"  key            : {key.shape} ({key.dtype})")
    print(f"  head_size      : {head_size}")

    # ── CUDA reference ──
    rope_cuda = rope.to("cuda")
    q_cuda, k_cuda = query.clone().to("cuda"), key.clone().to("cuda")
    pos_cuda = positions.to("cuda")
    
    with torch.no_grad():
        q_out_cuda, k_out_cuda = rope_cuda.forward_cuda(pos_cuda, q_cuda, k_cuda)
    print(f"  CUDA output    : q={q_out_cuda.shape}, k={k_out_cuda.shape}")

    # ── Zeus ──
    rope.cpu()
    rope_zeus = rope.to("zeus")
    q_zeus, k_zeus = query.clone().to("zeus"), key.clone().to("zeus")
    pos_zeus = positions.to("zeus")
    
    with torch.no_grad():
        q_out_zeus, k_out_zeus = rope_zeus.forward_zeus(pos_zeus, q_zeus, k_zeus)
    print(f"  Zeus output    : q={q_out_zeus.shape}, k={k_out_zeus.shape}")

    # ── Compare ──
    ok_q = compare_tensors("rope_query", q_out_cuda, q_out_zeus, atol=2e-2, rtol=1e-2)
    ok_k = compare_tensors("rope_key", k_out_cuda, k_out_zeus, atol=2e-2, rtol=1e-2)

    return ok_q and ok_k, (q_out_cuda, k_out_cuda)


# ── Stage: QKV Proj ───────────────────────────────────────────
def test_qkv_proj(model, tokenizer):
    print()
    print("=" * 60)
    print("Stage: QKV Proj (model.model.layers[0].self_attn.qkv_proj)")
    print("=" * 60)

    config = model.config
    head_size = config.hidden_size // config.num_attention_heads
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    hidden_size = config.hidden_size
    
    # Copy weights from huggingface model
    hf_q_proj = model.model.layers[0].self_attn.q_proj
    hf_k_proj = model.model.layers[0].self_attn.k_proj
    hf_v_proj = model.model.layers[0].self_attn.v_proj
    
    # Concatenate weights
    qkv_weight = torch.cat([hf_q_proj.weight.data, hf_k_proj.weight.data, hf_v_proj.weight.data], dim=0)
    
    has_bias = getattr(config, "attention_bias", False)
    qkv_proj = torch.nn.Linear(hidden_size, qkv_weight.shape[0], bias=has_bias, dtype=torch.bfloat16)
    qkv_proj.weight.data.copy_(qkv_weight)
    
    if has_bias:
        qkv_bias = torch.cat([hf_q_proj.bias.data, hf_k_proj.bias.data, hf_v_proj.bias.data], dim=0)
        qkv_proj.bias.data.copy_(qkv_bias)
        
    batch_size = 1
    seq_len = 11
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    print(f"  input x        : {x.shape} ({x.dtype})")
    print(f"  qkv weight     : {qkv_proj.weight.shape} ({qkv_proj.weight.dtype})")

    # ── CUDA reference ──
    qkv_cuda = qkv_proj.to("cuda")
    x_cuda = x.to("cuda")
    with torch.no_grad():
        out_cuda = qkv_cuda(x_cuda)
    print(f"  CUDA output    : {out_cuda.shape}, device={out_cuda.device}")

    # ── Zeus (with LocalMem weight) ──
    from torch_zeus.zeus.local_memory import to_local_mem
    from torch_zeus.zeus.dispatch import wrap_as_dispatch_tensor

    qkv_proj.cpu()
    qkv_zeus = qkv_proj.to("zeus")

    # to_local_mem with kind='weight' auto-transposes (N,K) → (K,N) for ZENL GEMM
    local_weight = to_local_mem(qkv_zeus.weight.data.t().contiguous(), kind='weight', aligned_size=0)
    print(f"  weight: {qkv_zeus.weight.data.shape} (N,K) -> {local_weight.shape} (K,N) in LocalMem")
    qkv_zeus.weight = torch.nn.Parameter(
        wrap_as_dispatch_tensor(local_weight), requires_grad=False
    )
    print(f"  weight LocalMem: Tr={local_weight.Tr}, Tc={local_weight.Tc}")

    # Bypass nn.Linear.forward which decomposes aten::linear → aten::t + aten::addmm.
    # aten::t would convert ZeusLocalMemTensor to GDG and break the ZENL path.
    # Instead, call addmm directly: output = input @ weight(K,N) + bias
    # (weight is already (K,N) in LocalMem, no transpose needed)
    x_zeus = x.to("zeus")
    x_zeus_2d = x_zeus.reshape(-1, hidden_size)  # (batch*seq, K)
    with torch.no_grad():
        if qkv_zeus.bias is not None:
            bias_zeus = qkv_zeus.bias.data
            out_zeus_2d = torch.addmm(bias_zeus, x_zeus_2d, qkv_zeus.weight)
        else:
            out_zeus_2d = torch.mm(x_zeus_2d, qkv_zeus.weight)
    out_zeus = out_zeus_2d.reshape(batch_size, seq_len, -1)
    print(f"  Zeus output    : {out_zeus.shape}, device={out_zeus.device}")

    # ── Compare ──
    ok = compare_tensors("qkv_proj", out_cuda, out_zeus)

    return ok, out_cuda

# ── Stage: O Proj ─────────────────────────────────────────────
def test_o_proj(model, tokenizer):
    print()
    print("=" * 60)
    print("Stage: O Proj (model.model.layers[0].self_attn.o_proj)")
    print("=" * 60)

    config = model.config
    hidden_size = config.hidden_size
    
    # Copy weights from huggingface model
    hf_o_proj = model.model.layers[0].self_attn.o_proj
    
    has_bias = getattr(config, "attention_bias", False)
    o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=has_bias, dtype=torch.bfloat16)
    o_proj.weight.data.copy_(hf_o_proj.weight.data)
    
    if has_bias:
        o_proj.bias.data.copy_(hf_o_proj.bias.data)
        
    batch_size = 1
    seq_len = 11
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    print(f"  input x        : {x.shape} ({x.dtype})")
    print(f"  o_proj weight  : {o_proj.weight.shape} ({o_proj.weight.dtype})")

    # ── CUDA reference ──
    o_cuda = o_proj.to("cuda")
    x_cuda = x.to("cuda")
    with torch.no_grad():
        out_cuda = o_cuda(x_cuda)
    print(f"  CUDA output    : {out_cuda.shape}, device={out_cuda.device}")

    # ── Zeus (with LocalMem weight) ──
    from torch_zeus.zeus.local_memory import to_local_mem
    from torch_zeus.zeus.dispatch import wrap_as_dispatch_tensor

    o_proj.cpu()
    o_zeus = o_proj.to("zeus")

    local_weight = to_local_mem(o_zeus.weight.data.t().contiguous(), kind='weight', aligned_size=0)
    print(f"  weight: {o_zeus.weight.data.shape} (N,K) -> {local_weight.shape} (K,N) in LocalMem")
    o_zeus.weight = torch.nn.Parameter(
        wrap_as_dispatch_tensor(local_weight), requires_grad=False
    )
    
    x_zeus = x.to("zeus")
    x_zeus_2d = x_zeus.reshape(-1, hidden_size)
    with torch.no_grad():
        if o_zeus.bias is not None:
            bias_zeus = o_zeus.bias.data
            out_zeus_2d = torch.addmm(bias_zeus, x_zeus_2d, o_zeus.weight)
        else:
            out_zeus_2d = torch.mm(x_zeus_2d, o_zeus.weight)
    out_zeus = out_zeus_2d.reshape(batch_size, seq_len, -1)
    print(f"  Zeus output    : {out_zeus.shape}, device={out_zeus.device}")

    # ── Compare ──
    ok = compare_tensors("o_proj", out_cuda, out_zeus)

    return ok, out_cuda

# ── Stage: MLP ────────────────────────────────────────────────
def test_mlp(model, tokenizer):
    print()
    print("=" * 60)
    print("Stage: MLP (model.model.layers[0].mlp)")
    print("=" * 60)

    from sglang.srt.layers.activation import SiluAndMul
    from torch_zeus.zeus.local_memory import to_local_mem
    from torch_zeus.zeus.dispatch import wrap_as_dispatch_tensor

    config = model.config
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    
    # Copy weights from huggingface model
    hf_gate_proj = model.model.layers[0].mlp.gate_proj
    hf_up_proj = model.model.layers[0].mlp.up_proj
    hf_down_proj = model.model.layers[0].mlp.down_proj
    
    # SGLang concats gate and up proj
    gate_up_weight = torch.cat([hf_gate_proj.weight.data, hf_up_proj.weight.data], dim=0)
    
    # We create torch.nn.Linear for gate_up and down, and use SiluAndMul
    gate_up_proj = torch.nn.Linear(hidden_size, gate_up_weight.shape[0], bias=False, dtype=torch.bfloat16)
    gate_up_proj.weight.data.copy_(gate_up_weight)
    
    down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.bfloat16)
    down_proj.weight.data.copy_(hf_down_proj.weight.data)
    
    act_fn = SiluAndMul()
    
    batch_size = 1
    seq_len = 11
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    print(f"  input x        : {x.shape} ({x.dtype})")
    print(f"  gate_up weight : {gate_up_proj.weight.shape} ({gate_up_proj.weight.dtype})")
    print(f"  down weight    : {down_proj.weight.shape} ({down_proj.weight.dtype})")

    # ── CUDA reference ──
    gate_up_cuda = gate_up_proj.to("cuda")
    down_cuda = down_proj.to("cuda")
    act_cuda = act_fn.to("cuda")
    x_cuda = x.to("cuda")
    
    with torch.no_grad():
        gate_up_out_cuda = gate_up_cuda(x_cuda)
        # SGLang SiluAndMul expects the concatenated output
        act_out_cuda = act_cuda.forward_cuda(gate_up_out_cuda)
        out_cuda = down_cuda(act_out_cuda)
    print(f"  CUDA output    : {out_cuda.shape}, device={out_cuda.device}")

    # ── Zeus (with LocalMem weight) ──
    gate_up_proj.cpu()
    down_proj.cpu()
    gate_up_zeus = gate_up_proj.to("zeus")
    down_zeus = down_proj.to("zeus")
    act_zeus = act_fn.to("zeus")

    # LocalMem for gate_up
    gu_local_weight = to_local_mem(gate_up_zeus.weight.data.t().contiguous(), kind='weight', aligned_size=0)
    gate_up_zeus.weight = torch.nn.Parameter(wrap_as_dispatch_tensor(gu_local_weight), requires_grad=False)
    
    # LocalMem for down
    d_local_weight = to_local_mem(down_zeus.weight.data.t().contiguous(), kind='weight', aligned_size=0)
    down_zeus.weight = torch.nn.Parameter(wrap_as_dispatch_tensor(d_local_weight), requires_grad=False)
    
    x_zeus = x.to("zeus")
    
    with torch.no_grad():
        # gate_up
        x_zeus_2d = x_zeus.reshape(-1, hidden_size)
        gate_up_out_zeus_2d = torch.mm(x_zeus_2d, gate_up_zeus.weight)
        gate_up_out_zeus = gate_up_out_zeus_2d.reshape(batch_size, seq_len, -1)
        
        # act
        act_out_zeus = act_zeus.forward_zeus(gate_up_out_zeus)
        
        # down
        act_out_zeus_2d = act_out_zeus.reshape(-1, intermediate_size)
        out_zeus_2d = torch.mm(act_out_zeus_2d, down_zeus.weight)
        out_zeus = out_zeus_2d.reshape(batch_size, seq_len, -1)

    print(f"  Zeus output    : {out_zeus.shape}, device={out_zeus.device}")

    # ── Compare ──
    ok = compare_tensors("mlp", out_cuda, out_zeus)

    return ok, out_cuda


# ── Stage: store_kv_cache (KV Cache Layout) ──────────────────
def test_store_kv_cache(model, tokenizer):
    """Test that Zeus store_kv_cache kernel produces the same tiled layout
    as the to_local_mem Python API.

    Approach:
      CUDA path  : scatter into flat cache -> gather per page -> reshape to
                   [batch, page_num, num_kv_heads, page_size, head_dim] ->
                   to_local_mem(kind='kv_key'/'kv_value') -> read raw tiled bytes
      Zeus path  : store_kv_cache kernel writes tiled bytes directly into the
                   cache buffer -> read raw bytes
      Compare    : raw tiled bytes should be identical
    """
    print()
    print("=" * 60)
    print("Stage: store_kv_cache (KV Cache Layout)")
    print("=" * 60)

    import ctypes
    from torch_zeus.zeus.local_memory import to_local_mem
    from sgl_kernel_zeus import store_kv_cache as zeus_store_kv_cache

    config = model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = 128  # override to test larger dim (128 cols = 2 LocalMem blocks wide)
    page_size = 128
    num_tokens = 1000  # multi-block: ceil(1000/128) = 8 blocks
    num_blocks = (num_tokens + page_size - 1) // page_size  # 8
    total_slots = num_blocks * page_size  # 1024

    # Tc = head_dim / 64 for BF16 (one block col = 64 BF16 elements = 128 bytes)
    Tc = head_dim // 64  # 2

    torch.manual_seed(42)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.bfloat16)
    # loc: token i -> slot i (sequential fill)
    loc = torch.arange(num_tokens, dtype=torch.int64)

    print(f"  num_kv_heads   : {num_kv_heads}")
    print(f"  head_dim       : {head_dim}")
    print(f"  page_size      : {page_size}")
    print(f"  num_tokens     : {num_tokens}")
    print(f"  num_blocks     : {num_blocks}")
    print(f"  Tc             : {Tc}")
    print(f"  k              : {k.shape} ({k.dtype})")
    print(f"  v              : {v.shape} ({v.dtype})")

    # ── CUDA reference: scatter -> gather -> reshape ──
    # Allocate [total_slots, num_kv_heads, head_dim] to hold all blocks
    k_cache_cuda = torch.zeros(total_slots, num_kv_heads, head_dim,
                               dtype=torch.bfloat16, device="cuda")
    v_cache_cuda = torch.zeros(total_slots, num_kv_heads, head_dim,
                               dtype=torch.bfloat16, device="cuda")
    loc_cuda = loc.to("cuda")
    k_cuda = k.to("cuda")
    v_cuda = v.to("cuda")

    # CUDA store_kv_cache is a simple scatter: cache[loc] = data
    try:
        torch.ops.sgl_kernel.store_kv_cache(
            k_cache_cuda, v_cache_cuda, loc_cuda, k_cuda, v_cuda)
    except Exception:
        k_cache_cuda[loc_cuda] = k_cuda
        v_cache_cuda[loc_cuda] = v_cuda

    # Reshape to [1, num_blocks, num_kv_heads, page_size, head_dim]
    # cache is [total_slots, num_kv_heads, head_dim]
    # -> permute(1,0,2) -> [num_kv_heads, total_slots, head_dim]
    # -> reshape [1, num_blocks, num_kv_heads, page_size, head_dim] needs different approach
    # Instead: reshape [num_blocks, page_size, num_kv_heads, head_dim]
    #        -> permute(0, 2, 1, 3) -> [num_blocks, num_kv_heads, page_size, head_dim]
    #        -> unsqueeze(0) -> [1, num_blocks, num_kv_heads, page_size, head_dim]
    # .contiguous() is critical: the Zeus backend's .to() doesn't handle non-contiguous correctly.
    k_ref = k_cache_cuda.cpu().reshape(
        num_blocks, page_size, num_kv_heads, head_dim
    ).permute(0, 2, 1, 3).contiguous().unsqueeze(0)
    v_ref = v_cache_cuda.cpu().reshape(
        num_blocks, page_size, num_kv_heads, head_dim
    ).permute(0, 2, 1, 3).contiguous().unsqueeze(0)

    print(f"  CUDA k gathered: {k_ref.shape}")
    print(f"  CUDA v gathered: {v_ref.shape}")

    # Convert CUDA result to LocalMem tiled layout via to_local_mem API
    # Each [page_size=128, head_dim=128] BF16 slice = 1 block tall x 2 blocks wide (Tr=1, Tc=2)
    k_ref_zeus = k_ref.to("zeus")
    v_ref_zeus = v_ref.to("zeus")

    local_k_ref = to_local_mem(k_ref_zeus, Tr=1, Tc=Tc, kind="kv_key", aligned_size=0)
    local_v_ref = to_local_mem(v_ref_zeus, Tr=1, Tc=Tc, kind="kv_value", aligned_size=0)

    print(f"  LocalMem K ref : {local_k_ref}")
    print(f"  LocalMem V ref : {local_v_ref}")

    # ── Zeus kernel: store_kv_cache writes tiled layout directly ──
    # K cache: [num_blocks, num_kv_heads, page_size, head_dim] (row-major, matches LocalMem KV_KEY)
    # V cache: same total size, 16-byte column-group interleaved (matches LocalMem KV_VALUE)
    k_cache_zeus = torch.zeros(num_blocks, num_kv_heads, page_size, head_dim,
                               dtype=torch.bfloat16, device="zeus")
    v_cache_zeus = torch.zeros(num_blocks, num_kv_heads, page_size, head_dim,
                               dtype=torch.bfloat16, device="zeus")

    zeus_store_kv_cache(k_cache_zeus, v_cache_zeus,
                        loc.to("zeus"), k.to("zeus"), v.to("zeus"), page_size)

    print(f"  Zeus k_cache   : {k_cache_zeus.shape}, device={k_cache_zeus.device}")
    print(f"  Zeus v_cache   : {v_cache_zeus.shape}, device={v_cache_zeus.device}")

    # ── Compare raw tiled bytes per (block, head) ──
    nbytes = page_size * head_dim * 2  # BF16 = 2 bytes per element

    ok_k = True
    ok_v = True
    num_compared = 0

    for b in range(num_blocks):
        for h in range(num_kv_heads):
            mat_idx = b * num_kv_heads + h

            # --- Reference: read raw tiled bytes from LocalMem ---
            k_ref_ptr = local_k_ref.data_ptrs[mat_idx]
            k_ref_raw = bytearray((ctypes.c_uint8 * nbytes).from_address(k_ref_ptr))
            k_ref_flat = torch.frombuffer(k_ref_raw, dtype=torch.bfloat16).clone()

            v_ref_ptr = local_v_ref.data_ptrs[mat_idx]
            v_ref_raw = bytearray((ctypes.c_uint8 * nbytes).from_address(v_ref_ptr))
            v_ref_flat = torch.frombuffer(v_ref_raw, dtype=torch.bfloat16).clone()

            # --- Zeus kernel: raw bytes from cache tensor ---
            # k_cache[b, h] is [page_size, head_dim] contiguous
            k_zeus_flat = k_cache_zeus[b, h].contiguous().view(-1).cpu()
            v_zeus_flat = v_cache_zeus[b, h].contiguous().view(-1).cpu()

            # --- Compare ---
            k_match = torch.equal(k_ref_flat, k_zeus_flat)
            v_match = torch.equal(v_ref_flat, v_zeus_flat)

            if not k_match:
                ok_k = False
            if not v_match:
                ok_v = False
            num_compared += 1

    status_k = "PASS" if ok_k else "DIFF"
    status_v = "PASS" if ok_v else "DIFF"
    print(f"  [store_kv_k] {status_k} | raw tiled bytes match={ok_k}  "
          f"blocks={num_blocks}  heads={num_kv_heads}  matrices={num_compared}  bytes_per_head={nbytes}")
    print(f"  [store_kv_v] {status_v} | raw tiled bytes match={ok_v}  "
          f"blocks={num_blocks}  heads={num_kv_heads}  matrices={num_compared}  bytes_per_head={nbytes}")

    return ok_k and ok_v, None


def test_extend_attention(model, tokenizer):
    """Test Zeus extend_attention kernel against PyTorch reference.

    CUDA golden : standard attention (Q @ K^T * sm_scale + causal_mask → softmax → @ V)
    Zeus path   : store_kv_cache into tiled cache → extend_attention reads tiled KV
    Compare     : output tensors (standard layout) directly
    """
    print()
    print("=" * 60)
    print("Stage: extend_attention (Prefill Attention)")
    print("=" * 60)

    import torch.nn.functional as F
    from sgl_kernel_zeus import store_kv_cache as zeus_store_kv_cache
    from sgl_kernel_zeus import extend_attention as zeus_extend_attention

    config = model.config
    num_q_heads = config.num_attention_heads   # 14 for Qwen2.5-0.5B
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)  # 2
    head_dim = config.hidden_size // num_q_heads   # 64
    page_size = 128
    sm_scale = 1.0 / (head_dim ** 0.5)

    # Two sequences with different prefix/extend lengths
    prefix_lens = [200, 300]
    extend_lens = [50, 80]
    batch_size = len(prefix_lens)

    torch.manual_seed(42)

    # Generate random Q, K, V for all tokens
    all_kv_tokens = []     # (K, V) per sequence, prefix + extend concatenated
    all_q_tokens = []      # Q per sequence, extend only
    for i in range(batch_size):
        total_seq = prefix_lens[i] + extend_lens[i]
        k_seq = torch.randn(total_seq, num_kv_heads, head_dim, dtype=torch.bfloat16)
        v_seq = torch.randn(total_seq, num_kv_heads, head_dim, dtype=torch.bfloat16)
        q_seq = torch.randn(extend_lens[i], num_q_heads, head_dim, dtype=torch.bfloat16)
        all_kv_tokens.append((k_seq, v_seq))
        all_q_tokens.append(q_seq)

    total_q = sum(extend_lens)
    total_kv = sum(prefix_lens[i] + extend_lens[i] for i in range(batch_size))
    num_pages = (total_kv + page_size - 1) // page_size

    print(f"  batch_size     : {batch_size}")
    print(f"  prefix_lens    : {prefix_lens}")
    print(f"  extend_lens    : {extend_lens}")
    print(f"  num_q_heads    : {num_q_heads}")
    print(f"  num_kv_heads   : {num_kv_heads}")
    print(f"  head_dim       : {head_dim}")
    print(f"  page_size      : {page_size}")
    print(f"  total_q        : {total_q}")
    print(f"  total_kv       : {total_kv}")
    print(f"  num_pages      : {num_pages}")

    # ── CUDA golden: standard attention with causal mask ──
    cuda_outputs = []
    for i in range(batch_size):
        k_seq, v_seq = all_kv_tokens[i]
        q_seq = all_q_tokens[i]
        plen = prefix_lens[i]
        elen = extend_lens[i]
        total_seq = plen + elen

        # Expand KV for GQA: [total_seq, num_kv_heads, head_dim] → [total_seq, num_q_heads, head_dim]
        kv_group = num_q_heads // num_kv_heads
        k_exp = k_seq.unsqueeze(2).expand(-1, -1, kv_group, -1).reshape(total_seq, num_q_heads, head_dim)
        v_exp = v_seq.unsqueeze(2).expand(-1, -1, kv_group, -1).reshape(total_seq, num_q_heads, head_dim)

        # Transpose to [num_q_heads, seq, head_dim] for batched matmul
        q_t = q_seq.permute(1, 0, 2).float()       # [num_q_heads, elen, head_dim]
        k_t = k_exp.permute(1, 0, 2).float()       # [num_q_heads, total_seq, head_dim]
        v_t = v_exp.permute(1, 0, 2).float()       # [num_q_heads, total_seq, head_dim]

        # QK^T: [num_q_heads, elen, total_seq]
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale

        # Causal mask: prefix fully visible, extend causal
        mask = torch.zeros(elen, total_seq, dtype=torch.bool)
        for qi in range(elen):
            # prefix: all visible
            mask[qi, :plen] = True
            # extend: causal (qi >= kv_pos - plen)
            for n in range(elen):
                if qi >= n:
                    mask[qi, plen + n] = True
        mask = mask.unsqueeze(0).expand(num_q_heads, -1, -1)  # [num_q_heads, elen, total_seq]
        scores = scores.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t)  # [num_q_heads, elen, head_dim]
        out = out.permute(1, 0, 2).bfloat16()  # [elen, num_q_heads, head_dim]
        cuda_outputs.append(out)

    cuda_output = torch.cat(cuda_outputs, dim=0)  # [total_q, num_q_heads, head_dim]
    print(f"  CUDA output    : {cuda_output.shape}")

    # ── Zeus path: store_kv_cache → extend_attention ──
    # Allocate tiled KV cache (enough pages for all tokens)
    total_slots = num_pages * page_size
    k_cache_zeus = torch.zeros(num_pages, num_kv_heads, page_size, head_dim,
                               dtype=torch.bfloat16, device="zeus")
    v_cache_zeus = torch.zeros(num_pages, num_kv_heads, page_size, head_dim,
                               dtype=torch.bfloat16, device="zeus")

    # Store all KV into tiled cache and build index structures
    # kv_indices: sequential slot assignment (seq0 gets slots 0..len0-1, seq1 gets len0..len0+len1-1, etc.)
    kv_indices_list = []
    slot_offset = 0
    for i in range(batch_size):
        k_seq, v_seq = all_kv_tokens[i]
        total_seq = prefix_lens[i] + extend_lens[i]

        loc = torch.arange(slot_offset, slot_offset + total_seq, dtype=torch.int64)
        zeus_store_kv_cache(k_cache_zeus, v_cache_zeus,
                            loc.to("zeus"), k_seq.to("zeus"), v_seq.to("zeus"),
                            page_size)

        kv_indices_list.append(torch.arange(slot_offset, slot_offset + total_seq, dtype=torch.int32))
        slot_offset += total_seq

    # Build indptr arrays
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    for i in range(batch_size):
        qo_indptr[i + 1] = qo_indptr[i] + extend_lens[i]
        kv_indptr[i + 1] = kv_indptr[i] + prefix_lens[i] + extend_lens[i]
    kv_indices = torch.cat(kv_indices_list)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32)

    # Q: concatenate all extend queries
    q_zeus = torch.cat(all_q_tokens, dim=0)  # [total_q, num_q_heads, head_dim]

    # Output buffer
    o_zeus = torch.zeros_like(q_zeus, device="zeus")

    print(f"  qo_indptr      : {qo_indptr.tolist()}")
    print(f"  kv_indptr      : {kv_indptr.tolist()}")
    print(f"  prefix_lens    : {prefix_lens_t.tolist()}")

    zeus_extend_attention(
        q_zeus.to("zeus"), o_zeus,
        k_cache_zeus, v_cache_zeus,
        qo_indptr.to("zeus"), kv_indptr.to("zeus"),
        kv_indices.to("zeus"), prefix_lens_t.to("zeus"),
        num_q_heads, num_kv_heads, head_dim,
        page_size, sm_scale, True)

    zeus_output = o_zeus.cpu()  # [total_q, num_q_heads, head_dim]
    print(f"  Zeus output    : {zeus_output.shape}")

    # ── Compare ──
    diff = (cuda_output.float() - zeus_output.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # BF16 attention accumulates many multiply-adds; allow small tolerance
    ok = max_diff < 0.05
    status = "PASS" if ok else "DIFF"
    print(f"  [{status}] max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")

    return ok, None


def test_decode_attention(model, tokenizer):
    """Test Zeus decode_attention kernel against PyTorch reference.

    Each sequence has 1 query token attending to all historical KV (no causal mask).
    CUDA golden : standard attention Q @ K^T * sm_scale -> softmax -> @ V
    Zeus path   : store_kv_cache into tiled cache -> decode_attention
    """
    print()
    print("=" * 60)
    print("Stage: decode_attention (Decode Attention)")
    print("=" * 60)

    from sgl_kernel_zeus import store_kv_cache as zeus_store_kv_cache
    from sgl_kernel_zeus import decode_attention as zeus_decode_attention

    config = model.config
    num_q_heads = config.num_attention_heads   # 14
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)  # 2
    head_dim = config.hidden_size // num_q_heads   # 64
    page_size = 128
    sm_scale = 1.0 / (head_dim ** 0.5)

    # 3 sequences with different KV lengths
    seq_lens = [512, 1024, 256]
    batch_size = len(seq_lens)

    torch.manual_seed(42)

    # Generate random Q (1 per sequence) and KV (seq_len per sequence)
    all_kv = []
    all_q = []
    for i in range(batch_size):
        k_seq = torch.randn(seq_lens[i], num_kv_heads, head_dim, dtype=torch.bfloat16)
        v_seq = torch.randn(seq_lens[i], num_kv_heads, head_dim, dtype=torch.bfloat16)
        q_seq = torch.randn(1, num_q_heads, head_dim, dtype=torch.bfloat16)
        all_kv.append((k_seq, v_seq))
        all_q.append(q_seq)

    total_kv = sum(seq_lens)
    num_pages = (total_kv + page_size - 1) // page_size

    print(f"  batch_size     : {batch_size}")
    print(f"  seq_lens       : {seq_lens}")
    print(f"  num_q_heads    : {num_q_heads}")
    print(f"  num_kv_heads   : {num_kv_heads}")
    print(f"  head_dim       : {head_dim}")
    print(f"  page_size      : {page_size}")
    print(f"  total_kv       : {total_kv}")
    print(f"  num_pages      : {num_pages}")

    # ── CUDA golden: standard attention (no causal mask) ──
    cuda_outputs = []
    kv_group = num_q_heads // num_kv_heads
    for i in range(batch_size):
        k_seq, v_seq = all_kv[i]
        q_seq = all_q[i]

        # Expand KV for GQA
        k_exp = k_seq.unsqueeze(2).expand(-1, -1, kv_group, -1).reshape(
            seq_lens[i], num_q_heads, head_dim)
        v_exp = v_seq.unsqueeze(2).expand(-1, -1, kv_group, -1).reshape(
            seq_lens[i], num_q_heads, head_dim)

        # [num_q_heads, 1, head_dim] @ [num_q_heads, head_dim, seq_len]
        q_t = q_seq.permute(1, 0, 2).float()       # [num_q_heads, 1, head_dim]
        k_t = k_exp.permute(1, 0, 2).float()       # [num_q_heads, seq_len, head_dim]
        v_t = v_exp.permute(1, 0, 2).float()       # [num_q_heads, seq_len, head_dim]

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t)  # [num_q_heads, 1, head_dim]
        out = out.permute(1, 0, 2).bfloat16()  # [1, num_q_heads, head_dim]
        cuda_outputs.append(out)

    cuda_output = torch.cat(cuda_outputs, dim=0)  # [batch_size, num_q_heads, head_dim]
    print(f"  CUDA output    : {cuda_output.shape}")

    # ── Zeus path: store_kv_cache -> decode_attention ──
    k_cache_zeus = torch.zeros(num_pages, num_kv_heads, page_size, head_dim,
                               dtype=torch.bfloat16, device="zeus")
    v_cache_zeus = torch.zeros(num_pages, num_kv_heads, page_size, head_dim,
                               dtype=torch.bfloat16, device="zeus")

    kv_indices_list = []
    slot_offset = 0
    for i in range(batch_size):
        k_seq, v_seq = all_kv[i]
        loc = torch.arange(slot_offset, slot_offset + seq_lens[i], dtype=torch.int64)
        zeus_store_kv_cache(k_cache_zeus, v_cache_zeus,
                            loc.to("zeus"), k_seq.to("zeus"), v_seq.to("zeus"),
                            page_size)
        kv_indices_list.append(torch.arange(slot_offset, slot_offset + seq_lens[i],
                                            dtype=torch.int32))
        slot_offset += seq_lens[i]

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    for i in range(batch_size):
        kv_indptr[i + 1] = kv_indptr[i] + seq_lens[i]
    kv_indices = torch.cat(kv_indices_list)

    q_zeus = torch.cat(all_q, dim=0)  # [batch_size, num_q_heads, head_dim]
    o_zeus = torch.zeros_like(q_zeus, device="zeus")

    print(f"  kv_indptr      : {kv_indptr.tolist()}")

    zeus_decode_attention(
        q_zeus.to("zeus"), o_zeus,
        k_cache_zeus, v_cache_zeus,
        kv_indptr.to("zeus"), kv_indices.to("zeus"),
        num_q_heads, num_kv_heads, head_dim,
        page_size, sm_scale, 0.0)

    zeus_output = o_zeus.cpu()
    print(f"  Zeus output    : {zeus_output.shape}")

    # ── Compare ──
    diff = (cuda_output.float() - zeus_output.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    ok = max_diff < 0.05
    status = "PASS" if ok else "DIFF"
    print(f"  [{status}] max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")

    return ok, None


def test_transformer_block(model, tokenizer):
    """Test a complete Qwen2.5 transformer block (layer 0).

    CUDA golden : HuggingFace Qwen2DecoderLayer forward
    Zeus path   : Compose all Zeus ops manually:
                  rmsnorm → qkv_proj → rope → store_kv → extend_attn →
                  o_proj → residual → rmsnorm → gate_up → silu_mul → down → residual
    """
    print()
    print("=" * 60)
    print("Stage: transformer_block (Full Decoder Layer 0)")
    print("=" * 60)

    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.layers.activation import SiluAndMul
    from sgl_kernel_zeus import (rmsnorm as zeus_rmsnorm,
                                 silu_and_mul as zeus_silu_and_mul,
                                 rotary_embedding as zeus_rotary_embedding,
                                 store_kv_cache as zeus_store_kv_cache,
                                 extend_attention as zeus_extend_attention)
    from torch_zeus.zeus.local_memory import to_local_mem
    from torch_zeus.zeus.dispatch import wrap_as_dispatch_tensor

    config = model.config
    hidden_size = config.hidden_size        # 896
    num_q_heads = config.num_attention_heads # 14
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)  # 2
    head_dim = hidden_size // num_q_heads   # 64
    intermediate_size = config.intermediate_size  # 4864
    eps = config.rms_norm_eps
    page_size = 128
    sm_scale = 1.0 / (head_dim ** 0.5)

    # ── Input: embedding output for a real sentence ──
    text = "Hello, this is a test for Zeus device comparison."
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    bs = 1
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0)

    with torch.no_grad():
        x = model.model.embed_tokens(input_ids)  # [1, seq_len, hidden_size] BF16

    layer = model.model.layers[0]

    print(f"  hidden_size    : {hidden_size}")
    print(f"  num_q_heads    : {num_q_heads}")
    print(f"  num_kv_heads   : {num_kv_heads}")
    print(f"  head_dim       : {head_dim}")
    print(f"  seq_len        : {seq_len}")
    print(f"  input x        : {x.shape} ({x.dtype})")

    # ══════════════════════════════════════════════════════════════
    # CUDA golden: HF Qwen2DecoderLayer forward
    # ══════════════════════════════════════════════════════════════
    layer_cuda = layer.cuda()
    x_cuda = x.cuda()
    pos_cuda = positions.cuda()

    # HF Qwen2 needs position_embeddings=(cos, sin) from model.rotary_emb
    rotary_emb = model.model.rotary_emb.cuda()
    cos_cuda, sin_cuda = rotary_emb(x_cuda, pos_cuda)

    with torch.no_grad():
        out_cuda = layer_cuda(
            x_cuda,
            position_embeddings=(cos_cuda, sin_cuda),
            use_cache=False,
        )
    # Newer HF returns tensor directly; older returns tuple
    if isinstance(out_cuda, tuple):
        out_cuda = out_cuda[0]
    print(f"  CUDA output    : {out_cuda.shape}")

    rotary_emb.cpu()
    layer.cpu()

    # ══════════════════════════════════════════════════════════════
    # Zeus path: manually compose all ops
    # ══════════════════════════════════════════════════════════════
    x_zeus = x.to("zeus")

    # ── Helper: prepare LocalMem weight for GEMM ──
    def make_local_weight(weight_data):
        """Convert (N, K) weight to LocalMem (K, N) dispatch tensor."""
        w_zeus = weight_data.to("zeus")
        local_w = to_local_mem(w_zeus.t().contiguous(), kind='weight', aligned_size=0)
        return wrap_as_dispatch_tensor(local_w)

    def linear_zeus(x_2d, weight_dispatch, bias=None):
        """x_2d @ weight(K,N) + bias"""
        if bias is not None:
            return torch.addmm(bias.to("zeus"), x_2d, weight_dispatch)
        else:
            return torch.mm(x_2d, weight_dispatch)

    # ── 1. Input LayerNorm ──
    ln1_weight = layer.input_layernorm.weight.data.to("zeus")
    normed = zeus_rmsnorm(x_zeus.view(-1, hidden_size), ln1_weight, eps)
    normed = normed.view(bs, seq_len, hidden_size)
    # ── 2. QKV Projection ──
    # Concat Q/K/V weights and biases (same as existing qkv_proj test)
    qkv_weight = torch.cat([
        layer.self_attn.q_proj.weight.data,
        layer.self_attn.k_proj.weight.data,
        layer.self_attn.v_proj.weight.data,
    ], dim=0)  # [1152, 896]
    qkv_w_dispatch = make_local_weight(qkv_weight)

    # Detect bias from actual layer weights (config attribute name varies)
    has_bias = layer.self_attn.q_proj.bias is not None
    qkv_bias = None
    if has_bias:
        qkv_bias = torch.cat([
            layer.self_attn.q_proj.bias.data,
            layer.self_attn.k_proj.bias.data,
            layer.self_attn.v_proj.bias.data,
        ], dim=0)

    q_dim = num_q_heads * head_dim     # 896
    kv_dim = num_kv_heads * head_dim   # 128

    normed_2d = normed.view(-1, hidden_size)
    qkv = linear_zeus(normed_2d, qkv_w_dispatch, qkv_bias)
    # qkv: [seq_len, 1152]
    q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
    # q: [seq_len, 896], k: [seq_len, 128], v: [seq_len, 128]

    # ── 3. RoPE ──
    rope_layer = get_rope(
        head_dim,
        int(head_dim * getattr(config, "partial_rotary_factor", 1.0)),
        config.max_position_embeddings,
        getattr(config, "rope_theta", 10000.0),
        True,  # is_neox_style
        rope_scaling=getattr(config, "rope_scaling", None),
        dtype=torch.bfloat16,
    )
    rope_zeus = rope_layer.to("zeus")
    # SGLang RoPE expects 1D positions [total_tokens], not [batch, seq_len]
    pos_zeus = torch.arange(seq_len, dtype=torch.int64).to("zeus")
    # q, k, v from split() are non-contiguous — make contiguous for RoPE and attention
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_rope = q.clone()
    k_rope = k.clone()

    q_rope, k_rope = rope_zeus.forward_zeus(pos_zeus, q_rope, k_rope)

    # Reshape for attention: [seq_len, num_heads, head_dim]
    q_3d = q_rope.view(seq_len, num_q_heads, head_dim)
    k_3d = k_rope.view(seq_len, num_kv_heads, head_dim)
    v_3d = v.view(seq_len, num_kv_heads, head_dim)

    # ── 4. Store KV Cache ──
    num_pages = (seq_len + page_size - 1) // page_size
    k_cache = torch.zeros(num_pages, num_kv_heads, page_size, head_dim,
                          dtype=torch.bfloat16, device="zeus")
    v_cache = torch.zeros(num_pages, num_kv_heads, page_size, head_dim,
                          dtype=torch.bfloat16, device="zeus")
    loc = torch.arange(seq_len, dtype=torch.int64).to("zeus")
    zeus_store_kv_cache(k_cache, v_cache, loc, k_3d, v_3d, page_size)

    # ── 5. Extend Attention (prefill: all tokens are extend, no prefix) ──
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="zeus")
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="zeus")
    kv_indices = torch.arange(seq_len, dtype=torch.int32, device="zeus")
    prefix_lens = torch.tensor([0], dtype=torch.int32, device="zeus")

    attn_out = torch.zeros(seq_len, num_q_heads, head_dim,
                           dtype=torch.bfloat16, device="zeus")
    zeus_extend_attention(
        q_3d, attn_out, k_cache, v_cache,
        qo_indptr, kv_indptr, kv_indices, prefix_lens,
        num_q_heads, num_kv_heads, head_dim,
        page_size, sm_scale, True)
    # ── 6. O Projection ──
    o_weight = layer.self_attn.o_proj.weight.data  # [896, 896]
    o_w_dispatch = make_local_weight(o_weight)

    o_bias = None
    if hasattr(layer.self_attn.o_proj, 'bias') and layer.self_attn.o_proj.bias is not None:
        o_bias = layer.self_attn.o_proj.bias.data

    attn_out_2d = attn_out.reshape(-1, hidden_size)
    o_proj_out = linear_zeus(attn_out_2d, o_w_dispatch, o_bias)

    # ── 7. Residual Add ──
    hidden = x_zeus.view(-1, hidden_size) + o_proj_out
    # hidden: [seq_len, hidden_size]
    # ── 8. Post-Attention LayerNorm ──
    ln2_weight = layer.post_attention_layernorm.weight.data.to("zeus")
    normed2 = zeus_rmsnorm(hidden, ln2_weight, eps)

    # ── 9. Gate+Up Projection ──
    gate_up_weight = torch.cat([
        layer.mlp.gate_proj.weight.data,
        layer.mlp.up_proj.weight.data,
    ], dim=0)  # [9728, 896]
    gu_w_dispatch = make_local_weight(gate_up_weight)
    gate_up_out = linear_zeus(normed2, gu_w_dispatch)
    # gate_up_out: [seq_len, 9728]

    # ── 10. SiLU & Mul ──
    act_out = zeus_silu_and_mul(gate_up_out)

    # ── 11. Down Projection ──
    down_weight = layer.mlp.down_proj.weight.data  # [896, 4864]
    down_w_dispatch = make_local_weight(down_weight)
    mlp_out = linear_zeus(act_out, down_w_dispatch)

    # ── 12. Residual Add ──
    out_zeus = hidden + mlp_out
    out_zeus = out_zeus.view(bs, seq_len, hidden_size)

    print(f"  Zeus output    : {out_zeus.shape}")

    # ── Compare ──
    diff = (out_cuda.float().cpu() - out_zeus.float().cpu()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Transformer block accumulates errors from multiple ops; use relaxed tolerance
    ok = max_diff < 0.1
    status = "PASS" if ok else "DIFF"
    print(f"  [{status}] max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}")

    return ok, None


def test_lm_head(model, tokenizer):
    """Test LM Head large GEMM: [seq_len, 896] @ [896, 151936] -> [seq_len, 151936]."""
    print()
    print("=" * 60)
    print("Stage: LM Head (model.lm_head)")
    print("=" * 60)

    from torch_zeus.zeus.local_memory import to_local_mem
    from torch_zeus.zeus.dispatch import wrap_as_dispatch_tensor

    config = model.config
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    # Simulate hidden states (output of final norm)
    batch_size = 1
    seq_len = 11
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    print(f"  input x        : {x.shape} ({x.dtype})")
    print(f"  lm_head weight : {model.lm_head.weight.shape} ({model.lm_head.weight.dtype})")
    print(f"  vocab_size     : {vocab_size}")

    # ── CUDA reference ──
    lm_head_cuda = model.lm_head.to("cuda")
    x_cuda = x.to("cuda")
    with torch.no_grad():
        out_cuda = lm_head_cuda(x_cuda)
    print(f"  CUDA output    : {out_cuda.shape}, device={out_cuda.device}")
    model.lm_head.cpu()

    # ── Zeus (LocalMem weight) ──
    lm_head_weight = model.lm_head.weight.data  # [vocab_size, hidden_size] = [151936, 896]
    w_zeus = lm_head_weight.to("zeus")
    local_w = to_local_mem(w_zeus.t().contiguous(), kind='weight', aligned_size=0)
    w_dispatch = wrap_as_dispatch_tensor(local_w)
    print(f"  weight: {lm_head_weight.shape} (N,K) -> {local_w.shape} (K,N) in LocalMem")
    print(f"  weight LocalMem: Tr={local_w.Tr}, Tc={local_w.Tc}")

    x_zeus = x.to("zeus")
    x_zeus_2d = x_zeus.reshape(-1, hidden_size)
    with torch.no_grad():
        out_zeus_2d = torch.mm(x_zeus_2d, w_dispatch)
    out_zeus = out_zeus_2d.reshape(batch_size, seq_len, -1)
    print(f"  Zeus output    : {out_zeus.shape}, device={out_zeus.device}")

    # ── Compare ──
    ok = compare_tensors("lm_head", out_cuda, out_zeus, atol=0.05, rtol=0.05)

    # Also compare greedy token
    cuda_tokens = torch.argmax(out_cuda[0], dim=-1).cpu()
    zeus_tokens = torch.argmax(out_zeus[0].cpu().float(), dim=-1)
    token_match = torch.equal(cuda_tokens, zeus_tokens)
    print(f"  Greedy tokens match: {token_match}")
    if not token_match:
        for t in range(seq_len):
            ct, zt = cuda_tokens[t].item(), zeus_tokens[t].item()
            match = "OK" if ct == zt else "MISMATCH"
            print(f"    pos {t}: cuda={ct} zeus={zt} {match}")

    return ok, out_cuda


def test_full_model(model, tokenizer):
    """Test full Qwen2.5-0.5B forward: embedding -> 24 layers -> final_norm -> lm_head.

    CUDA golden : HuggingFace model.generate with greedy decoding
    Zeus path   : Manually compose all ops with fused_add_rmsnorm residual stream
    Compare     : Greedy token match
    """
    print()
    print("=" * 60)
    print("Stage: full_model (All 24 Layers + LM Head)")
    print("=" * 60)

    from sglang.srt.layers.rotary_embedding import get_rope
    from sgl_kernel_zeus import (rmsnorm as zeus_rmsnorm,
                                 fused_add_rmsnorm as zeus_fused_add_rmsnorm,
                                 silu_and_mul as zeus_silu_and_mul,
                                 rotary_embedding as zeus_rotary_embedding,
                                 store_kv_cache as zeus_store_kv_cache,
                                 extend_attention as zeus_extend_attention)
    from torch_zeus.zeus.local_memory import to_local_mem
    from torch_zeus.zeus.dispatch import wrap_as_dispatch_tensor

    config = model.config
    hidden_size = config.hidden_size        # 896
    num_q_heads = config.num_attention_heads # 14
    num_kv_heads = getattr(config, "num_key_value_heads", num_q_heads)  # 2
    head_dim = hidden_size // num_q_heads   # 64
    intermediate_size = config.intermediate_size  # 4864
    num_layers = config.num_hidden_layers   # 24
    eps = config.rms_norm_eps
    page_size = 128
    sm_scale = 1.0 / (head_dim ** 0.5)
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    text = "Hello, this is a test for Zeus device comparison."
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    seq_len = input_ids.shape[1]

    print(f"  num_layers     : {num_layers}")
    print(f"  hidden_size    : {hidden_size}")
    print(f"  seq_len        : {seq_len}")
    print(f"  text           : '{text}'")

    # ══════════════════════════════════════════════════════════════
    # CUDA golden: HuggingFace full model forward
    # ══════════════════════════════════════════════════════════════
    model_cuda = model.cuda()
    ids_cuda = input_ids.cuda()
    with torch.no_grad():
        out_cuda = model_cuda(ids_cuda)
    logits_cuda = out_cuda.logits  # [1, seq_len, vocab_size]
    greedy_cuda = torch.argmax(logits_cuda[0, -1], dim=-1).item()
    print(f"  CUDA logits    : {logits_cuda.shape}")
    print(f"  CUDA greedy    : {greedy_cuda} -> '{tokenizer.decode([greedy_cuda])}'")
    model.cpu()

    # ══════════════════════════════════════════════════════════════
    # Zeus path: manually compose all ops
    # ══════════════════════════════════════════════════════════════

    # ── Helper ──
    def make_local_weight(weight_data):
        w_zeus = weight_data.to("zeus")
        local_w = to_local_mem(w_zeus.t().contiguous(), kind='weight', aligned_size=0)
        return wrap_as_dispatch_tensor(local_w)

    def linear_zeus(x_2d, weight_dispatch, bias=None):
        if bias is not None:
            return torch.addmm(bias.to("zeus"), x_2d, weight_dispatch)
        else:
            return torch.mm(x_2d, weight_dispatch)

    # ── Preload all weights into LocalMem ──
    print("  Preloading weights into LocalMem...")
    layer_weights = []
    for i in range(num_layers):
        layer = model.model.layers[i]

        # QKV
        qkv_w = torch.cat([
            layer.self_attn.q_proj.weight.data,
            layer.self_attn.k_proj.weight.data,
            layer.self_attn.v_proj.weight.data,
        ], dim=0)
        has_bias = layer.self_attn.q_proj.bias is not None
        qkv_b = None
        if has_bias:
            qkv_b = torch.cat([
                layer.self_attn.q_proj.bias.data,
                layer.self_attn.k_proj.bias.data,
                layer.self_attn.v_proj.bias.data,
            ], dim=0).to("zeus")

        # O proj
        o_b = None
        if hasattr(layer.self_attn.o_proj, 'bias') and layer.self_attn.o_proj.bias is not None:
            o_b = layer.self_attn.o_proj.bias.data.to("zeus")

        # Gate+Up
        gu_w = torch.cat([
            layer.mlp.gate_proj.weight.data,
            layer.mlp.up_proj.weight.data,
        ], dim=0)

        lw = {
            'ln1_w': layer.input_layernorm.weight.data.to("zeus"),
            'ln2_w': layer.post_attention_layernorm.weight.data.to("zeus"),
            'qkv_w': make_local_weight(qkv_w),
            'qkv_b': qkv_b,
            'o_w': make_local_weight(layer.self_attn.o_proj.weight.data),
            'o_b': o_b,
            'gu_w': make_local_weight(gu_w),
            'down_w': make_local_weight(layer.mlp.down_proj.weight.data),
        }
        layer_weights.append(lw)
        if (i + 1) % 8 == 0:
            print(f"    layers {i-6}..{i} loaded")

    final_norm_w = model.model.norm.weight.data.to("zeus")
    lm_head_w = make_local_weight(model.lm_head.weight.data)
    print("  All weights loaded.")

    # ── RoPE ──
    rope_layer = get_rope(
        head_dim,
        int(head_dim * getattr(config, "partial_rotary_factor", 1.0)),
        config.max_position_embeddings,
        getattr(config, "rope_theta", 10000.0),
        True,
        rope_scaling=getattr(config, "rope_scaling", None),
        dtype=torch.bfloat16,
    )
    rope_zeus = rope_layer.to("zeus")
    pos_zeus = torch.arange(seq_len, dtype=torch.int64).to("zeus")

    # ── Embedding ──
    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids).to("zeus")
    hidden = hidden.view(-1, hidden_size)  # [seq_len, hidden_size]
    residual = None

    # ── Layer loop ──
    print("  Running 24 layers...")
    for i in range(num_layers):
        lw = layer_weights[i]

        # ── Input LayerNorm ──
        if residual is None:
            residual = hidden.clone()
            normed = zeus_rmsnorm(hidden, lw['ln1_w'], eps)
        else:
            zeus_fused_add_rmsnorm(hidden, residual, lw['ln1_w'], eps)
            normed = hidden

        # ── QKV Projection ──
        qkv = linear_zeus(normed, lw['qkv_w'], lw['qkv_b'])
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # ── RoPE ──
        q_rope = q.clone()
        k_rope = k.clone()
        q_rope, k_rope = rope_zeus.forward_zeus(pos_zeus, q_rope, k_rope)

        # ── Reshape for attention ──
        q_3d = q_rope.view(seq_len, num_q_heads, head_dim)
        k_3d = k_rope.view(seq_len, num_kv_heads, head_dim)
        v_3d = v.view(seq_len, num_kv_heads, head_dim)

        # ── Store KV Cache ──
        num_pages = (seq_len + page_size - 1) // page_size
        k_cache = torch.zeros(num_pages, num_kv_heads, page_size, head_dim,
                              dtype=torch.bfloat16, device="zeus")
        v_cache = torch.zeros(num_pages, num_kv_heads, page_size, head_dim,
                              dtype=torch.bfloat16, device="zeus")
        loc = torch.arange(seq_len, dtype=torch.int64).to("zeus")
        zeus_store_kv_cache(k_cache, v_cache, loc, k_3d, v_3d, page_size)

        # ── Extend Attention ──
        qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="zeus")
        kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="zeus")
        kv_indices = torch.arange(seq_len, dtype=torch.int32, device="zeus")
        prefix_lens = torch.tensor([0], dtype=torch.int32, device="zeus")

        attn_out = torch.zeros(seq_len, num_q_heads, head_dim,
                               dtype=torch.bfloat16, device="zeus")
        zeus_extend_attention(
            q_3d, attn_out, k_cache, v_cache,
            qo_indptr, kv_indptr, kv_indices, prefix_lens,
            num_q_heads, num_kv_heads, head_dim,
            page_size, sm_scale, True)

        # ── O Projection ──
        attn_out_2d = attn_out.reshape(-1, hidden_size)
        hidden = linear_zeus(attn_out_2d, lw['o_w'], lw['o_b'])

        # ── Post-Attention LayerNorm ──
        zeus_fused_add_rmsnorm(hidden, residual, lw['ln2_w'], eps)
        normed2 = hidden

        # ── MLP ──
        gate_up = linear_zeus(normed2, lw['gu_w'])
        act_out = zeus_silu_and_mul(gate_up)
        hidden = linear_zeus(act_out, lw['down_w'])

        if (i + 1) % 8 == 0:
            print(f"    layer {i} done")

    # ── Final Norm ──
    zeus_fused_add_rmsnorm(hidden, residual, final_norm_w, eps)

    # ── LM Head ──
    logits_zeus = torch.mm(hidden, lm_head_w)  # [seq_len, vocab_size]
    print(f"  Zeus logits    : {logits_zeus.shape}")

    greedy_zeus = torch.argmax(logits_zeus[-1].cpu().float(), dim=-1).item()
    print(f"  Zeus greedy    : {greedy_zeus} -> '{tokenizer.decode([greedy_zeus])}'")

    # ── Compare ──
    token_match = (greedy_cuda == greedy_zeus)
    print(f"  Greedy match   : {token_match} (cuda={greedy_cuda}, zeus={greedy_zeus})")

    # Compare logits for last position
    last_cuda = logits_cuda[0, -1].cpu().float()
    last_zeus = logits_zeus[-1].cpu().float()

    # Check top-5 overlap
    top5_cuda = torch.topk(last_cuda, 5).indices.tolist()
    top5_zeus = torch.topk(last_zeus, 5).indices.tolist()
    overlap = len(set(top5_cuda) & set(top5_zeus))
    print(f"  Top-5 overlap  : {overlap}/5 (cuda={top5_cuda}, zeus={top5_zeus})")

    ok = token_match
    status = "PASS" if ok else "DIFF"
    print(f"  [{status}] greedy_token_match={ok}")

    return ok, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["embedding", "rmsnorm", "silu_and_mul", "rope", "qkv_proj",
                 "o_proj", "mlp", "store_kv_cache", "extend_attention",
                 "decode_attention", "transformer_block",
                 "lm_head", "full_model", "all"],
        default="all",
    )
    args = parser.parse_args()

    model, tokenizer = load_hf_model()
    print(f"Model: {MODEL_NAME}")
    print(f"  hidden_size  = {model.config.hidden_size}")
    print(f"  num_layers   = {model.config.num_hidden_layers}")
    print(f"  vocab_size   = {model.config.vocab_size}")

    results = {}
    has_cuda = torch.cuda.is_available()
    stage_fns = [
        ("embedding", test_embedding),
        ("rmsnorm", test_rmsnorm),
        ("silu_and_mul", test_silu_and_mul),
        ("rope", test_rope),
        ("qkv_proj", test_qkv_proj),
        ("o_proj", test_o_proj),
        ("mlp", test_mlp),
        ("store_kv_cache", test_store_kv_cache),
        ("extend_attention", test_extend_attention),
        ("decode_attention", test_decode_attention),
        ("transformer_block", test_transformer_block),
        ("lm_head", test_lm_head),
        ("full_model", test_full_model),
    ]

    for stage_name, stage_fn in stage_fns:
        if args.stage not in (stage_name, "all"):
            continue
        if not has_cuda:
            skip_stage_without_cuda(stage_name, results)
            continue
        ok, _ = stage_fn(model, tokenizer)
        results[stage_name] = ok

    # -- Summary --
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed is True else "FAIL" if passed is False else "SKIP"
        print(f"  {name:20s} : {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
