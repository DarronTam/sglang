"""
Tests for Zeus GEMM operators: mm, addmm, linear, bmm, mv.

mm / addmm / linear
  Weights MUST be in LocalMem (packed with zeus.pack_weights).
  Non-LocalMem weight raises RuntimeError — no CPU roundtrip.

bmm / mv
  No ZENL kernel yet; CPU roundtrip fallback with a one-time warning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from conftest import DEVICE, to_zeus, assert_close

import torch_zeus.zeus as zeus

# FP8 may not be available on older PyTorch
_has_fp8 = hasattr(torch, 'float8_e4m3fn')
fp8_skip = pytest.mark.skipif(not _has_fp8, reason="float8_e4m3fn not available")


# ============================================================================
# Helpers
# ============================================================================

def make_linear(in_f: int, out_f: int, bias: bool = True,
                dtype=torch.bfloat16, Tr: int = 1, Tc: int = 1,
                aligned_size: int = -1) -> nn.Linear:
    """Create an nn.Linear on Zeus with pack_weights applied."""
    layer = nn.Linear(in_f, out_f, bias=bias, dtype=dtype).to(DEVICE)
    zeus.pack_weights(layer, Tr=Tr, Tc=Tc, aligned_size=aligned_size)
    return layer


def make_linear_fp8(in_f: int, out_f: int, bias: bool = True,
                    Tr: int = 1, Tc: int = 1,
                    aligned_size: int = -1) -> nn.Linear:
    """Create nn.Linear with FP8 E4M3 weight (and bias), packed to LocalMem.

    FP8 doesn't support standard weight init, so we create in float32 first,
    scale to FP8-representable range, cast, then pack.
    """
    layer = nn.Linear(in_f, out_f, bias=bias, dtype=torch.float32, device='cpu')
    with torch.no_grad():
        layer.weight.data.mul_(0.1)  # keep values in FP8 representable range
        layer.weight.data = layer.weight.data.to(torch.float8_e4m3fn)
        if bias:
            layer.bias.data.mul_(0.1)
            layer.bias.data = layer.bias.data.to(torch.float8_e4m3fn)
    layer = layer.to(DEVICE)
    zeus.pack_weights(layer, Tr=Tr, Tc=Tc, aligned_size=aligned_size)
    return layer


def assert_packed_weight_config(layer: nn.Linear, *, Tr: int, Tc: int,
                                aligned_size: int) -> None:
    """Assert LocalMem metadata matches the requested pack configuration."""
    local_weight = layer._zeus_local_mems['weight']
    assert local_weight.Tr == Tr
    assert local_weight.Tc == Tc
    assert local_weight.aligned_size == aligned_size


def assert_close_fp8(actual_zeus, expected_cpu, rtol=0.15, atol=0.1):
    """Compare FP8 results via float32 (FP8 has only 3 mantissa bits)."""
    torch.testing.assert_close(
        actual_zeus.cpu().to(torch.float32),
        expected_cpu.to(torch.float32),
        rtol=rtol, atol=atol,
    )


ALIGNED_4X4_AS1024 = dict(K=1024, N=768, M=4, Tr=4, Tc=4, aligned_size=1024)
ALIGNED_8X8_AS1024 = dict(K=1100, N=520, M=3, Tr=8, Tc=8, aligned_size=1024)
ALIGNED_4X4_AS1024_LARGE = dict(
    K=1792, N=1408, M=2, Tr=4, Tc=4, aligned_size=1024
)
ALIGNED_8X8_AS1024_LARGE = dict(
    K=2176, N=1152, M=2, Tr=8, Tc=8, aligned_size=1024
)

ALIGNED_AS1024_CONFIGS = [
    ALIGNED_4X4_AS1024,
    ALIGNED_8X8_AS1024,
    ALIGNED_4X4_AS1024_LARGE,
    ALIGNED_8X8_AS1024_LARGE,
]


# ============================================================================
# torch.mm  — LocalMem weight required
# ============================================================================

class TestMM:
    def test_bfloat16(self):
        K, N = 128, 64
        layer = make_linear(K, N, bias=False)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()  # (K,N)

        x = torch.randn(4, K, dtype=torch.bfloat16, device=DEVICE)
        ref = torch.mm(x.cpu(), weight_cpu)
        out = torch.mm(x, layer.weight)  # layer.weight has LocalMem data_ptr
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_single_row(self):
        K, N = 128, 64
        layer = make_linear(K, N, bias=False)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()

        x = torch.randn(1, K, dtype=torch.bfloat16, device=DEVICE)
        ref = torch.mm(x.cpu(), weight_cpu)
        out = torch.mm(x, layer.weight)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_requires_localmem_weight(self):
        """Non-LocalMem mat2 must raise RuntimeError — no CPU fallback."""
        a = torch.randn(4, 8, dtype=torch.bfloat16, device=DEVICE)
        b = torch.randn(8, 6, dtype=torch.bfloat16, device=DEVICE)
        with pytest.raises(RuntimeError, match="LocalMem"):
            torch.mm(a, b)

    @pytest.mark.parametrize("cfg", ALIGNED_AS1024_CONFIGS)
    def test_bfloat16_aligned_as1024(self, cfg):
        layer = make_linear(cfg["K"], cfg["N"], bias=False, Tr=cfg["Tr"],
                            Tc=cfg["Tc"], aligned_size=cfg["aligned_size"])
        assert_packed_weight_config(
            layer, Tr=cfg["Tr"], Tc=cfg["Tc"],
            aligned_size=cfg["aligned_size"]
        )
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()

        x = torch.randn(cfg["M"], cfg["K"], dtype=torch.bfloat16, device=DEVICE)
        ref = torch.mm(x.cpu(), weight_cpu)
        out = torch.mm(x, layer.weight)
        assert_close(out, ref, rtol=0.05, atol=0.05)


# ============================================================================
# torch.addmm  — LocalMem weight required
# ============================================================================

class TestAddMM:
    def test_bfloat16(self):
        K, N = 128, 64
        layer = make_linear(K, N, bias=True)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()  # (K,N)
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(4, K, dtype=torch.bfloat16, device=DEVICE)
        ref = torch.addmm(bias_cpu, x.cpu(), weight_cpu)
        out = torch.addmm(layer.bias, x, layer.weight)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_alpha_beta(self):
        K, N = 128, 64
        layer = make_linear(K, N, bias=True)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(4, K, dtype=torch.bfloat16, device=DEVICE)
        ref = torch.addmm(bias_cpu, x.cpu(), weight_cpu, alpha=0.5, beta=2.0)
        out = torch.addmm(layer.bias, x, layer.weight, alpha=0.5, beta=2.0)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_requires_localmem_weight(self):
        """Non-LocalMem mat2 must raise RuntimeError."""
        bias = torch.randn(6, dtype=torch.bfloat16, device=DEVICE)
        a = torch.randn(4, 8, dtype=torch.bfloat16, device=DEVICE)
        b = torch.randn(8, 6, dtype=torch.bfloat16, device=DEVICE)
        with pytest.raises(RuntimeError, match="LocalMem"):
            torch.addmm(bias, a, b)

    @pytest.mark.parametrize("cfg", ALIGNED_AS1024_CONFIGS)
    def test_bfloat16_aligned_as1024(self, cfg):
        layer = make_linear(cfg["K"], cfg["N"], bias=True, Tr=cfg["Tr"],
                            Tc=cfg["Tc"], aligned_size=cfg["aligned_size"])
        assert_packed_weight_config(
            layer, Tr=cfg["Tr"], Tc=cfg["Tc"],
            aligned_size=cfg["aligned_size"]
        )
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(cfg["M"], cfg["K"], dtype=torch.bfloat16, device=DEVICE)
        ref = torch.addmm(bias_cpu, x.cpu(), weight_cpu)
        out = torch.addmm(layer.bias, x, layer.weight)
        assert_close(out, ref, rtol=0.05, atol=0.05)


# ============================================================================
# nn.Linear / F.linear  — LocalMem weight via pack_weights
# ============================================================================

class TestLinear:
    def test_basic(self):
        layer = make_linear(128, 64)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()  # (K,N)
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(4, 128, dtype=torch.bfloat16, device=DEVICE)
        ref = torch.addmm(bias_cpu, x.cpu(), weight_cpu)
        out = layer(x)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_no_bias(self):
        layer = make_linear(128, 64, bias=False)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()

        x = torch.randn(4, 128, dtype=torch.bfloat16, device=DEVICE)
        ref = torch.mm(x.cpu(), weight_cpu)
        out = layer(x)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_3d_input(self):
        """Batched input (batch, seq, features)."""
        layer = make_linear(128, 64)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()  # (K,N)
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(2, 8, 128, dtype=torch.bfloat16, device=DEVICE)
        # Reference uses (K,N) weight directly (matching pack_weights orientation)
        x_2d = x.cpu().reshape(-1, 128)
        ref_2d = torch.addmm(bias_cpu, x_2d, weight_cpu)
        ref = ref_2d.reshape(2, 8, 64)
        out = layer(x)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_requires_localmem_weight(self):
        """Non-LocalMem weight must raise RuntimeError."""
        layer = nn.Linear(64, 32, dtype=torch.bfloat16).to(DEVICE)
        x = torch.randn(4, 64, dtype=torch.bfloat16, device=DEVICE)
        with pytest.raises(RuntimeError, match="LocalMem"):
            layer(x)

    @pytest.mark.parametrize("cfg", ALIGNED_AS1024_CONFIGS)
    def test_aligned_as1024(self, cfg):
        layer = make_linear(cfg["K"], cfg["N"], Tr=cfg["Tr"], Tc=cfg["Tc"],
                            aligned_size=cfg["aligned_size"])
        assert_packed_weight_config(
            layer, Tr=cfg["Tr"], Tc=cfg["Tc"],
            aligned_size=cfg["aligned_size"]
        )
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(cfg["M"], cfg["K"], dtype=torch.bfloat16, device=DEVICE)
        ref = torch.addmm(bias_cpu, x.cpu(), weight_cpu)
        out = layer(x)
        assert_close(out, ref, rtol=0.05, atol=0.05)


# ============================================================================
# torch.mm  — FP8 E4M3
# ============================================================================

@fp8_skip
class TestMMFP8:
    def test_fp8e4m3(self):
        K, N = 128, 128
        layer = make_linear_fp8(K, N, bias=False)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()  # (K,N) FP8

        x = torch.randn(4, K, dtype=torch.float32).mul(0.1).to(torch.float8_e4m3fn).to(DEVICE)
        ref = torch.mm(x.cpu().float(), weight_cpu.float()).to(torch.float8_e4m3fn)
        out = torch.mm(x, layer.weight)
        assert_close_fp8(out, ref)

    def test_single_row(self):
        K, N = 128, 128
        layer = make_linear_fp8(K, N, bias=False)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()

        x = torch.randn(1, K, dtype=torch.float32).mul(0.1).to(torch.float8_e4m3fn).to(DEVICE)
        ref = torch.mm(x.cpu().float(), weight_cpu.float()).to(torch.float8_e4m3fn)
        out = torch.mm(x, layer.weight)
        assert_close_fp8(out, ref)

    def test_non_aligned_dims(self):
        K, N = 200, 100
        layer = make_linear_fp8(K, N, bias=False)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()

        x = torch.randn(5, K, dtype=torch.float32).mul(0.1).to(torch.float8_e4m3fn).to(DEVICE)
        ref = torch.mm(x.cpu().float(), weight_cpu.float()).to(torch.float8_e4m3fn)
        out = torch.mm(x, layer.weight)
        assert_close_fp8(out, ref)

    @pytest.mark.parametrize("cfg", ALIGNED_AS1024_CONFIGS)
    def test_fp8e4m3_aligned_as1024(self, cfg):
        layer = make_linear_fp8(cfg["K"], cfg["N"], bias=False, Tr=cfg["Tr"],
                                Tc=cfg["Tc"], aligned_size=cfg["aligned_size"])
        assert_packed_weight_config(
            layer, Tr=cfg["Tr"], Tc=cfg["Tc"],
            aligned_size=cfg["aligned_size"]
        )
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()

        x = torch.randn(cfg["M"], cfg["K"], dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn).to(DEVICE)
        ref = torch.mm(x.cpu().float(), weight_cpu.float()).to(torch.float8_e4m3fn)
        out = torch.mm(x, layer.weight)
        assert_close_fp8(out, ref)


# ============================================================================
# torch.addmm  — FP8 E4M3
# ============================================================================

@fp8_skip
class TestAddMMFP8:
    def test_fp8e4m3(self):
        K, N = 128, 128
        layer = make_linear_fp8(K, N, bias=True)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(4, K, dtype=torch.float32).mul(0.1).to(torch.float8_e4m3fn).to(DEVICE)
        ref = torch.addmm(bias_cpu.float(), x.cpu().float(), weight_cpu.float()) \
            .to(torch.float8_e4m3fn)
        out = torch.addmm(layer.bias, x, layer.weight)
        assert_close_fp8(out, ref)

    def test_alpha_beta(self):
        K, N = 128, 128
        layer = make_linear_fp8(K, N, bias=True)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(4, K, dtype=torch.float32).mul(0.1).to(torch.float8_e4m3fn).to(DEVICE)
        ref = torch.addmm(bias_cpu.float(), x.cpu().float(), weight_cpu.float(),
                           alpha=0.5, beta=2.0).to(torch.float8_e4m3fn)
        out = torch.addmm(layer.bias, x, layer.weight, alpha=0.5, beta=2.0)
        assert_close_fp8(out, ref)

    @pytest.mark.parametrize("cfg", ALIGNED_AS1024_CONFIGS)
    def test_fp8e4m3_aligned_as1024(self, cfg):
        layer = make_linear_fp8(cfg["K"], cfg["N"], bias=True, Tr=cfg["Tr"],
                                Tc=cfg["Tc"], aligned_size=cfg["aligned_size"])
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(cfg["M"], cfg["K"], dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn).to(DEVICE)
        ref = torch.addmm(bias_cpu.float(), x.cpu().float(), weight_cpu.float()) \
            .to(torch.float8_e4m3fn)
        out = torch.addmm(layer.bias, x, layer.weight)
        assert_close_fp8(out, ref)


# ============================================================================
# nn.Linear / F.linear  — FP8 E4M3
# ============================================================================

@fp8_skip
class TestLinearFP8:
    def test_basic(self):
        layer = make_linear_fp8(128, 64)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(4, 128, dtype=torch.float32).mul(0.1).to(torch.float8_e4m3fn).to(DEVICE)
        ref = torch.addmm(bias_cpu.float(), x.cpu().float(), weight_cpu.float()) \
            .to(torch.float8_e4m3fn)
        out = layer(x)
        assert_close_fp8(out, ref)

    def test_no_bias(self):
        layer = make_linear_fp8(128, 64, bias=False)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()

        x = torch.randn(4, 128, dtype=torch.float32).mul(0.1).to(torch.float8_e4m3fn).to(DEVICE)
        ref = torch.mm(x.cpu().float(), weight_cpu.float()).to(torch.float8_e4m3fn)
        out = layer(x)
        assert_close_fp8(out, ref)

    def test_3d_input(self):
        layer = make_linear_fp8(128, 64)
        weight_cpu = layer._zeus_local_mems['weight'].to_linear().cpu()
        bias_cpu = layer.bias.detach().cpu()

        x = torch.randn(2, 8, 128, dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn).to(DEVICE)
        x_2d = x.cpu().float().reshape(-1, 128)
        ref_2d = torch.addmm(bias_cpu.float(), x_2d, weight_cpu.float())
        ref = ref_2d.reshape(2, 8, 64).to(torch.float8_e4m3fn)
        out = layer(x)
        assert_close_fp8(out, ref)


# ============================================================================
# torch.bmm  (CPU roundtrip — no ZENL batched GEMM kernel yet)
# ============================================================================

class TestBMM:
    def test_basic(self):
        a = torch.randn(3, 4, 8)
        b = torch.randn(3, 8, 6)
        ref = torch.bmm(a, b)
        out = torch.bmm(to_zeus(a), to_zeus(b))
        assert_close(out, ref, rtol=1e-4, atol=1e-4)


# ============================================================================
# torch.mv  (CPU roundtrip — no ZENL matrix-vector kernel yet)
# ============================================================================

class TestMV:
    def test_basic(self):
        mat = torch.randn(4, 8)
        vec = torch.randn(8)
        ref = torch.mv(mat, vec)
        out = torch.mv(to_zeus(mat), to_zeus(vec))
        assert_close(out, ref, rtol=1e-4, atol=1e-4)


# ============================================================================
# GEMM WUS (Weight-with-Uscale dequantization)
#   FP8 weight + BF16 uscale → BF16 output
# ============================================================================

@fp8_skip
class TestGemmWUS:
    """Tests for gemm_wus: GEMM with packed weight+uscale dequantization."""

    @staticmethod
    def _reference_gemm_wus(input_bf16, weight_fp8, uscale_bf16,
                             group_size, alpha=1.0, beta=1.0, bias=None):
        """CPU reference: C = alpha * A @ (W * uscale_expanded).T + beta * bias."""
        A = input_bf16.float()
        W = weight_fp8.float()
        N, K = W.shape
        # Expand uscale from (N, K//gs) to (N, K)
        uscale_exp = uscale_bf16.float().repeat_interleave(group_size, dim=1)
        uscale_exp = uscale_exp[:, :K]
        W_dequant = W * uscale_exp
        C = alpha * torch.mm(A, W_dequant.T)
        if bias is not None:
            C = C + beta * bias.float()
        return C.to(torch.bfloat16)

    def test_basic_nopad(self):
        """Basic WUS GEMM with PATCH-aligned dimensions, no padding."""
        from torch_zeus.zeus import local_memory as local_mem

        M, N, K = 4, 128, 128
        Tr, Tc = 1, 1
        group_size = 128

        weight_fp8 = torch.randn(N, K, dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn)
        uscale = torch.randn(N, K // group_size, dtype=torch.bfloat16)
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16)

        # Pack and upload
        packed = local_mem.pack_weight_with_uscale(
            weight_fp8.to(DEVICE), uscale.to(DEVICE),
            Tr, Tc, group_size, aligned_size=0)

        # Run WUS GEMM
        out = local_mem.gemm_wus(input_bf16.to(DEVICE), packed)

        # CPU reference
        ref = self._reference_gemm_wus(input_bf16, weight_fp8, uscale,
                                        group_size)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_multi_patch(self):
        """WUS GEMM spanning multiple patches."""
        from torch_zeus.zeus import local_memory as local_mem

        M, N, K = 4, 512, 256
        Tr, Tc = 2, 1
        group_size = 128

        weight_fp8 = torch.randn(N, K, dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn)
        uscale = torch.randn(N, K // group_size, dtype=torch.bfloat16)
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16)

        packed = local_mem.pack_weight_with_uscale(
            weight_fp8.to(DEVICE), uscale.to(DEVICE),
            Tr, Tc, group_size, aligned_size=0)

        out = local_mem.gemm_wus(input_bf16.to(DEVICE), packed)
        ref = self._reference_gemm_wus(input_bf16, weight_fp8, uscale,
                                        group_size)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_non_aligned_dims(self):
        """WUS GEMM with dimensions that require padding."""
        from torch_zeus.zeus import local_memory as local_mem

        M, N, K = 3, 300, 200
        Tr, Tc = 1, 1
        group_size = 8  # K=200 is divisible by 8

        weight_fp8 = torch.randn(N, K, dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn)
        uscale = torch.randn(N, K // group_size, dtype=torch.bfloat16)
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16)

        packed = local_mem.pack_weight_with_uscale(
            weight_fp8.to(DEVICE), uscale.to(DEVICE),
            Tr, Tc, group_size, aligned_size=0)

        out = local_mem.gemm_wus(input_bf16.to(DEVICE), packed)
        ref = self._reference_gemm_wus(input_bf16, weight_fp8, uscale,
                                        group_size)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_with_bias(self):
        """WUS GEMM with bias."""
        from torch_zeus.zeus import local_memory as local_mem

        M, N, K = 4, 128, 128
        Tr, Tc = 1, 1
        group_size = 128

        weight_fp8 = torch.randn(N, K, dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn)
        uscale = torch.randn(N, K // group_size, dtype=torch.bfloat16)
        bias = torch.randn(N, dtype=torch.bfloat16)
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16)

        packed = local_mem.pack_weight_with_uscale(
            weight_fp8.to(DEVICE), uscale.to(DEVICE),
            Tr, Tc, group_size, aligned_size=0)

        out = local_mem.gemm_wus(input_bf16.to(DEVICE), packed,
                                  bias=bias.to(DEVICE))
        ref = self._reference_gemm_wus(input_bf16, weight_fp8, uscale,
                                        group_size, bias=bias)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_alpha_beta(self):
        """WUS GEMM with custom alpha and beta."""
        from torch_zeus.zeus import local_memory as local_mem

        M, N, K = 4, 128, 128
        Tr, Tc = 1, 1
        group_size = 128

        weight_fp8 = torch.randn(N, K, dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn)
        uscale = torch.randn(N, K // group_size, dtype=torch.bfloat16)
        bias = torch.randn(N, dtype=torch.bfloat16)
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16)

        packed = local_mem.pack_weight_with_uscale(
            weight_fp8.to(DEVICE), uscale.to(DEVICE),
            Tr, Tc, group_size, aligned_size=0)

        out = local_mem.gemm_wus(input_bf16.to(DEVICE), packed,
                                  bias=bias.to(DEVICE),
                                  alpha=0.5, beta=2.0)
        ref = self._reference_gemm_wus(input_bf16, weight_fp8, uscale,
                                        group_size, alpha=0.5, beta=2.0,
                                        bias=bias)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_aligned_space(self):
        """WUS GEMM with aligned spaces (aligned_size > 0)."""
        from torch_zeus.zeus import local_memory as local_mem

        M, N, K = 4, 1024, 512
        Tr, Tc = 4, 2
        group_size = 128
        aligned_size = 256

        weight_fp8 = torch.randn(N, K, dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn)
        uscale = torch.randn(N, K // group_size, dtype=torch.bfloat16)
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16)

        packed = local_mem.pack_weight_with_uscale(
            weight_fp8.to(DEVICE), uscale.to(DEVICE),
            Tr, Tc, group_size, aligned_size=aligned_size)

        out = local_mem.gemm_wus(input_bf16.to(DEVICE), packed)
        ref = self._reference_gemm_wus(input_bf16, weight_fp8, uscale,
                                        group_size)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_single_row(self):
        """WUS GEMM with M=1 (single-token inference)."""
        from torch_zeus.zeus import local_memory as local_mem

        M, N, K = 1, 256, 256
        Tr, Tc = 1, 2
        group_size = 128

        weight_fp8 = torch.randn(N, K, dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn)
        uscale = torch.randn(N, K // group_size, dtype=torch.bfloat16)
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16)

        packed = local_mem.pack_weight_with_uscale(
            weight_fp8.to(DEVICE), uscale.to(DEVICE),
            Tr, Tc, group_size, aligned_size=0)

        out = local_mem.gemm_wus(input_bf16.to(DEVICE), packed)
        ref = self._reference_gemm_wus(input_bf16, weight_fp8, uscale,
                                        group_size)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_small_group_size(self):
        """WUS GEMM with small group_size (many uscale columns)."""
        from torch_zeus.zeus import local_memory as local_mem

        M, N, K = 2, 128, 128
        Tr, Tc = 1, 1
        group_size = 8  # Ku_per_patch = 128/8 = 16

        weight_fp8 = torch.randn(N, K, dtype=torch.float32).mul(0.1) \
            .to(torch.float8_e4m3fn)
        uscale = torch.randn(N, K // group_size, dtype=torch.bfloat16)
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16)

        packed = local_mem.pack_weight_with_uscale(
            weight_fp8.to(DEVICE), uscale.to(DEVICE),
            Tr, Tc, group_size, aligned_size=0)

        out = local_mem.gemm_wus(input_bf16.to(DEVICE), packed)
        ref = self._reference_gemm_wus(input_bf16, weight_fp8, uscale,
                                        group_size)
        assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_requires_wus_info(self):
        """packed_wus without _wus_info raises ValueError."""
        from torch_zeus.zeus import local_memory as local_mem

        input_bf16 = torch.randn(4, 128, dtype=torch.bfloat16, device=DEVICE)
        weight = torch.randn(128, 128, dtype=torch.bfloat16, device=DEVICE)
        local_w = local_mem.to_local_mem(weight, Tr=1, Tc=1, kind='weight')

        with pytest.raises(ValueError, match="pack_weight_with_uscale"):
            local_mem.gemm_wus(input_bf16, local_w)
