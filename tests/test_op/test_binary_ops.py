"""
Tests for element-wise binary operators: add, sub, mul.

Dispatched via REGISTER_PRIVATEUSE1_DISPATCH (add_stub, sub_stub, mul_stub)
→ ZENL kernels (zenlAdd, zenlSub, zenlMul).

fp8e4m3 and int8-saturation tests run Triton kernels directly on CUDA GPU
as golden reference (PyTorch TensorIterator doesn't dispatch Float8_e4m3fn
through add_stub, and its int8 path uses wrapping instead of saturation).
"""

import sys
import os
import shutil
import torch
import pytest
from conftest import DEVICE, to_zeus, assert_close

# Make zenl/src/triton importable for GPU golden tests
_TRITON_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "zenl", "src", "triton"
)
sys.path.insert(0, os.path.normpath(_TRITON_DIR))

def _has_cuda_headers():
    include_dirs = []
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_name)
        if root:
            include_dirs.append(os.path.join(root, "include"))
            include_dirs.append(root)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        include_dirs.append(
            os.path.join(conda_prefix, "targets", "x86_64-linux", "include")
        )
        include_dirs.append(os.path.join(conda_prefix, "include"))

    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
        include_dirs.append(
            os.path.join(cuda_root, "include")
        )

    include_dirs.extend(("/usr/local/cuda/include", "/usr/include"))

    return any(os.path.exists(os.path.join(path, "cuda.h")) for path in include_dirs)


def _cuda_arch():
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


_HAS_CUDA = torch.cuda.is_available()
_HAS_CUDA_HEADERS = _has_cuda_headers()
_HAS_CUDA_DEV = _HAS_CUDA and _HAS_CUDA_HEADERS
_HAS_CUDA_FP8E4NV = _HAS_CUDA_DEV and _cuda_arch() >= 89

if not _HAS_CUDA:
    _CUDA_SKIP_REASON = (
        "PyTorch CUDA runtime is not available for Triton GPU golden tests"
    )
elif not _HAS_CUDA_HEADERS:
    _CUDA_SKIP_REASON = "cuda.h is required for Triton GPU golden tests"
else:
    _CUDA_SKIP_REASON = ""

requires_cuda = pytest.mark.skipif(
    not _HAS_CUDA_DEV,
    reason=_CUDA_SKIP_REASON,
)
requires_cuda_fp8e4nv = pytest.mark.skipif(
    not _HAS_CUDA_FP8E4NV,
    reason="CUDA sm89+ with cuda.h is required for Triton fp8e4nv golden tests",
)

if _HAS_CUDA_DEV:
    from add_fp8e4m3 import add_fp8e4m3_kernel
    from add_int8 import add_int8_kernel
    from sub_fp8e4m3 import sub_fp8e4m3_kernel
    from sub_int8 import sub_int8_kernel
    from mul_fp8e4m3 import mul_fp8e4m3_kernel
    from mul_int8 import mul_int8_kernel

_BLOCK = 128


def _gpu_add(a, b, alpha=1.0):
    out = torch.empty_like(a)
    n = a.numel()
    kern = add_fp8e4m3_kernel if a.dtype == torch.float8_e4m3fn else add_int8_kernel
    kern[(1,)](out, a, b, n, alpha, BLOCK=_BLOCK, CORE_NUM=1)
    return out


def _gpu_sub(a, b, alpha=1.0):
    out = torch.empty_like(a)
    n = a.numel()
    kern = sub_fp8e4m3_kernel if a.dtype == torch.float8_e4m3fn else sub_int8_kernel
    kern[(1,)](out, a, b, n, alpha, BLOCK=_BLOCK, CORE_NUM=1)
    return out


def _gpu_mul(a, b):
    out = torch.empty_like(a)
    n = a.numel()
    kern = mul_fp8e4m3_kernel if a.dtype == torch.float8_e4m3fn else mul_int8_kernel
    kern[(1,)](out, a, b, n, BLOCK=_BLOCK, CORE_NUM=1)
    return out


# ============================================================================
# torch.add
# ============================================================================

class TestAdd:
    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([10.0, 20.0, 30.0, 40.0])
        assert_close(to_zeus(a) + to_zeus(b), a + b)

    def test_alpha(self):
        a, b = torch.randn(64), torch.randn(64)
        assert_close(torch.add(to_zeus(a), to_zeus(b), alpha=0.5),
                     torch.add(a, b, alpha=0.5))

    def test_negative_alpha(self):
        a, b = torch.randn(64), torch.randn(64)
        assert_close(torch.add(to_zeus(a), to_zeus(b), alpha=-1.0),
                     torch.add(a, b, alpha=-1.0))

    def test_2d(self):
        a, b = torch.randn(32, 64), torch.randn(32, 64)
        assert_close(to_zeus(a) + to_zeus(b), a + b)

    def test_inplace(self):
        a, b = torch.randn(128), torch.randn(128)
        az = to_zeus(a.clone())
        az.add_(to_zeus(b))
        assert_close(az, a + b)

    def test_large(self):
        """numel >= 4096 triggers the 2-core kernel path."""
        a, b = torch.randn(8192), torch.randn(8192)
        assert_close(to_zeus(a) + to_zeus(b), a + b)

    def test_bfloat16(self):
        a = torch.randn(256, dtype=torch.bfloat16)
        b = torch.randn(256, dtype=torch.bfloat16)
        assert_close(to_zeus(a) + to_zeus(b), a + b)

    def test_empty(self):
        a = torch.empty(0, device=DEVICE)
        b = torch.empty(0, device=DEVICE)
        assert (a + b).numel() == 0

    def test_scalar(self):
        a = torch.tensor([3.14])
        b = torch.tensor([2.72])
        assert_close(to_zeus(a) + to_zeus(b), a + b)

    def test_int8(self):
        a = torch.tensor([10, 20, -5, 0], dtype=torch.int8)
        b = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
        result = (to_zeus(a) + to_zeus(b)).cpu()
        expected = torch.tensor([11, 22, -2, 4], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_int8_overflow(self):
        """int8 overflow saturates on Zeus (ZENL add kernel clamps to [-128, 127])."""
        a = torch.tensor([120, -120], dtype=torch.int8)
        b = torch.tensor([10, -10], dtype=torch.int8)
        result = (to_zeus(a) + to_zeus(b)).cpu()
        expected = torch.tensor([127, -128], dtype=torch.int8)
        assert torch.equal(result, expected)

    @requires_cuda_fp8e4nv
    def test_fp8e4m3(self):
        """GPU golden: Triton kernel on CUDA (ATen has no fp8 add_stub)."""
        a = torch.tensor([1.0, 2.0, 0.5, -1.0], device="cuda").to(torch.float8_e4m3fn)
        b = torch.tensor([0.5, 1.0, 0.25, 0.5], device="cuda").to(torch.float8_e4m3fn)
        result = _gpu_add(a, b).float()
        expected = a.float() + b.float()
        torch.testing.assert_close(result, expected, atol=0.125, rtol=0)

    @requires_cuda_fp8e4nv
    def test_fp8e4m3_alpha(self):
        a = torch.tensor([1.0, 2.0, 4.0], device="cuda").to(torch.float8_e4m3fn)
        b = torch.tensor([2.0, 4.0, 8.0], device="cuda").to(torch.float8_e4m3fn)
        result = _gpu_add(a, b, alpha=0.5).float()
        expected = a.float() + 0.5 * b.float()
        torch.testing.assert_close(result, expected, atol=0.25, rtol=0)

    @requires_cuda_fp8e4nv
    def test_fp8e4m3_large(self):
        a = torch.randn(4096, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(4096, device="cuda").to(torch.float8_e4m3fn)
        out = torch.empty_like(a)
        add_fp8e4m3_kernel[(2,)](out, a, b, 4096, 1.0, BLOCK=_BLOCK, CORE_NUM=2)
        torch.testing.assert_close(out.float(), a.float() + b.float(), atol=0.25, rtol=0)

    @requires_cuda
    def test_int8_wrap(self):
        """GPU golden: Triton kernel wraps on int8 overflow (120+10 → -126)."""
        a = torch.tensor([120, -120], dtype=torch.int8, device="cuda")
        b = torch.tensor([10, -10], dtype=torch.int8, device="cuda")
        result = _gpu_add(a, b)
        expected = torch.tensor([-126, 126], dtype=torch.int8, device="cuda")
        assert torch.equal(result, expected)


# ============================================================================
# torch.sub
# ============================================================================

class TestSub:
    def test_basic(self):
        a = torch.tensor([10.0, 20.0, 30.0, 40.0])
        b = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert_close(to_zeus(a) - to_zeus(b), a - b)

    def test_alpha(self):
        a, b = torch.randn(64), torch.randn(64)
        assert_close(torch.sub(to_zeus(a), to_zeus(b), alpha=2.0),
                     torch.sub(a, b, alpha=2.0))

    def test_negative_alpha(self):
        a, b = torch.randn(64), torch.randn(64)
        assert_close(torch.sub(to_zeus(a), to_zeus(b), alpha=-0.5),
                     torch.sub(a, b, alpha=-0.5))

    def test_inplace(self):
        a, b = torch.randn(128), torch.randn(128)
        az = to_zeus(a.clone())
        az.sub_(to_zeus(b))
        assert_close(az, a - b)

    def test_large(self):
        a, b = torch.randn(8192), torch.randn(8192)
        assert_close(to_zeus(a) - to_zeus(b), a - b)

    def test_bfloat16(self):
        a = torch.randn(256, dtype=torch.bfloat16)
        b = torch.randn(256, dtype=torch.bfloat16)
        assert_close(to_zeus(a) - to_zeus(b), a - b)

    def test_int8(self):
        a = torch.tensor([50, 100, -10], dtype=torch.int8)
        b = torch.tensor([10, 20, 5], dtype=torch.int8)
        result = (to_zeus(a) - to_zeus(b)).cpu()
        expected = torch.tensor([40, 80, -15], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_int8_overflow(self):
        """int8 overflow saturates on Zeus (ZENL sub kernel clamps to [-128, 127])."""
        a = torch.tensor([-120, 120], dtype=torch.int8)
        b = torch.tensor([10, -10], dtype=torch.int8)
        result = (to_zeus(a) - to_zeus(b)).cpu()
        expected = torch.tensor([-128, 127], dtype=torch.int8)
        assert torch.equal(result, expected)

    @requires_cuda_fp8e4nv
    def test_fp8e4m3(self):
        a = torch.tensor([2.0, 1.0, 0.5, -1.0], device="cuda").to(torch.float8_e4m3fn)
        b = torch.tensor([0.5, 0.25, 0.25, 0.5], device="cuda").to(torch.float8_e4m3fn)
        result = _gpu_sub(a, b).float()
        expected = a.float() - b.float()
        torch.testing.assert_close(result, expected, atol=0.125, rtol=0)

    @requires_cuda_fp8e4nv
    def test_fp8e4m3_alpha(self):
        a = torch.tensor([4.0, 8.0], device="cuda").to(torch.float8_e4m3fn)
        b = torch.tensor([1.0, 2.0], device="cuda").to(torch.float8_e4m3fn)
        result = _gpu_sub(a, b, alpha=2.0).float()
        expected = a.float() - 2.0 * b.float()
        torch.testing.assert_close(result, expected, atol=0.25, rtol=0)

    @requires_cuda
    def test_int8_wrap(self):
        """GPU golden: Triton kernel wraps on int8 overflow (-120-10 → 126)."""
        a = torch.tensor([-120, 120], dtype=torch.int8, device="cuda")
        b = torch.tensor([10, -10], dtype=torch.int8, device="cuda")
        result = _gpu_sub(a, b)
        expected = torch.tensor([126, -126], dtype=torch.int8, device="cuda")
        assert torch.equal(result, expected)


# ============================================================================
# torch.mul
# ============================================================================

class TestMul:
    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([10.0, 20.0, 30.0, 40.0])
        assert_close(to_zeus(a) * to_zeus(b), a * b)

    def test_2d(self):
        a, b = torch.randn(32, 64), torch.randn(32, 64)
        assert_close(to_zeus(a) * to_zeus(b), a * b)

    def test_inplace(self):
        a, b = torch.randn(128), torch.randn(128)
        az = to_zeus(a.clone())
        az.mul_(to_zeus(b))
        assert_close(az, a * b)

    def test_large(self):
        a, b = torch.randn(8192), torch.randn(8192)
        assert_close(to_zeus(a) * to_zeus(b), a * b)

    def test_bfloat16(self):
        a = torch.randn(256, dtype=torch.bfloat16)
        b = torch.randn(256, dtype=torch.bfloat16)
        assert_close(to_zeus(a) * to_zeus(b), a * b)

    def test_int8(self):
        a = torch.tensor([2, 3, -4, 5], dtype=torch.int8)
        b = torch.tensor([5, 6, 7, -8], dtype=torch.int8)
        result = (to_zeus(a) * to_zeus(b)).cpu()
        expected = torch.tensor([10, 18, -28, -40], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_int8_overflow(self):
        """int8 overflow saturates on Zeus (ZENL mul kernel clamps to [-128, 127])."""
        a = torch.tensor([100, -100], dtype=torch.int8)
        b = torch.tensor([2, 2], dtype=torch.int8)
        result = (to_zeus(a) * to_zeus(b)).cpu()
        expected = torch.tensor([127, -128], dtype=torch.int8)
        assert torch.equal(result, expected)

    @requires_cuda_fp8e4nv
    def test_fp8e4m3(self):
        a = torch.tensor([2.0, 0.5, -1.0, 4.0], device="cuda").to(torch.float8_e4m3fn)
        b = torch.tensor([0.5, 2.0, 0.25, 0.5], device="cuda").to(torch.float8_e4m3fn)
        result = _gpu_mul(a, b).float()
        expected = a.float() * b.float()
        torch.testing.assert_close(result, expected, atol=0.125, rtol=0)

    @requires_cuda_fp8e4nv
    def test_fp8e4m3_large(self):
        a = torch.randn(4096, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(4096, device="cuda").to(torch.float8_e4m3fn)
        out = torch.empty_like(a)
        mul_fp8e4m3_kernel[(2,)](out, a, b, 4096, BLOCK=_BLOCK, CORE_NUM=2)
        torch.testing.assert_close(out.float(), a.float() * b.float(), atol=0.5, rtol=0)

    @requires_cuda
    def test_int8_wrap(self):
        """GPU golden: Triton kernel wraps on int8 overflow (100*2 → -56)."""
        a = torch.tensor([100, -100], dtype=torch.int8, device="cuda")
        b = torch.tensor([2, 2], dtype=torch.int8, device="cuda")
        result = _gpu_mul(a, b)
        expected = torch.tensor([-56, 56], dtype=torch.int8, device="cuda")
        assert torch.equal(result, expected)
