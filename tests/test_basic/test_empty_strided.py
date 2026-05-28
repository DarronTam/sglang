"""
Test empty_strided operator for torch_zeus

empty_strided creates an uninitialized tensor with explicit stride specification.
This allows creation of tensors with custom memory layouts (e.g., transposed, column-major).
"""

import torch
import torch_zeus


def test_basic_empty_strided():
    """Test basic empty_strided with contiguous strides"""
    print("Test 1: Basic empty_strided with contiguous strides")

    size = [2, 3, 4]
    stride = [12, 4, 1]  # Standard row-major (C-order) strides

    x = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"Size: {x.shape}")
    print(f"Stride: {x.stride()}")
    print(f"Is contiguous: {x.is_contiguous()}")

    assert x.shape == torch.Size(size), f"Expected shape {size}, got {x.shape}"
    assert x.stride() == tuple(stride), f"Expected stride {stride}, got {x.stride()}"
    assert x.is_contiguous(), "Should be contiguous"
    assert x.device.type == 'zeus', f"Expected device zeus, got {x.device.type}"

    print("✓ Test 1 passed\n")


def test_transposed_strides():
    """Test empty_strided with transposed (non-contiguous) strides"""
    print("Test 2: Transposed strides (non-contiguous)")

    size = [3, 4]
    stride = [1, 3]  # Column-major (Fortran order)

    x = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"Size: {x.shape}")
    print(f"Stride: {x.stride()}")
    print(f"Is contiguous: {x.is_contiguous()}")

    assert x.shape == torch.Size(size)
    assert x.stride() == tuple(stride)
    assert not x.is_contiguous(), "Should NOT be contiguous (column-major)"

    # Check it's Fortran contiguous
    print(f"Is Fortran contiguous: {x.is_contiguous(memory_format=torch.contiguous_format)}")

    print("✓ Test 2 passed\n")


def test_custom_strides():
    """Test empty_strided with custom stride patterns"""
    print("Test 3: Custom stride patterns")

    # Example: overlapping memory (broadcast-like strides)
    size = [2, 3]
    stride = [0, 1]  # First dimension has stride 0 (broadcast behavior)

    x = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"Broadcast-like size: {x.shape}, stride: {x.stride()}")
    assert x.shape == torch.Size(size)
    assert x.stride() == tuple(stride)

    # Example: Skipping elements (stride > element_size)
    size = [2, 3]
    stride = [6, 2]  # Skip elements

    y = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"Skipping elements size: {y.shape}, stride: {y.stride()}")
    assert y.shape == torch.Size(size)
    assert y.stride() == tuple(stride)

    print("✓ Test 3 passed\n")


def test_different_dtypes():
    """Test empty_strided with different data types"""
    print("Test 4: Different data types")

    size = [2, 3]
    stride = [3, 1]

    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]

    for dtype in dtypes:
        x = torch.empty_strided(size, stride, device='zeus', dtype=dtype)

        assert x.dtype == dtype, f"Expected dtype {dtype}, got {x.dtype}"
        assert x.shape == torch.Size(size)
        assert x.stride() == tuple(stride)
        print(f"dtype {dtype}: ✓")

    print("✓ Test 4 passed\n")


def test_1d_tensor():
    """Test empty_strided with 1D tensor"""
    print("Test 5: 1D tensor")

    size = [10]
    stride = [1]

    x = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"1D tensor size: {x.shape}, stride: {x.stride()}")
    assert x.shape == torch.Size(size)
    assert x.stride() == tuple(stride)
    assert x.dim() == 1

    # Non-unit stride
    stride = [2]
    y = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)
    print(f"1D with stride 2: {y.shape}, stride: {y.stride()}")
    assert y.stride() == tuple(stride)

    print("✓ Test 5 passed\n")


def test_high_dimensional():
    """Test empty_strided with high-dimensional tensors"""
    print("Test 6: High-dimensional tensors")

    # 4D tensor (batch, channels, height, width)
    size = [2, 3, 4, 5]
    stride = [60, 20, 5, 1]  # Standard NCHW order

    x = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"4D tensor size: {x.shape}, stride: {x.stride()}")
    assert x.shape == torch.Size(size)
    assert x.stride() == tuple(stride)
    assert x.dim() == 4

    # 5D tensor
    size = [2, 3, 4, 5, 6]
    stride = [360, 120, 30, 6, 1]

    y = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"5D tensor size: {y.shape}, stride: {y.stride()}")
    assert y.shape == torch.Size(size)
    assert y.dim() == 5

    print("✓ Test 6 passed\n")


def test_scalar_tensor():
    """Test empty_strided with scalar (0D tensor)"""
    print("Test 7: Scalar tensor")

    size = []
    stride = []

    x = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"Scalar tensor size: {x.shape}, stride: {x.stride()}")
    assert x.shape == torch.Size([])
    assert x.dim() == 0
    assert x.numel() == 1

    print("✓ Test 7 passed\n")


def test_zero_size_dimension():
    """Test empty_strided with zero-size dimension"""
    print("Test 8: Zero-size dimension")

    size = [0, 3]
    stride = [3, 1]

    x = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"Zero-size tensor: {x.shape}, stride: {x.stride()}")
    assert x.shape == torch.Size(size)
    assert x.numel() == 0

    # Empty tensor
    size = [0]
    stride = [1]

    y = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)
    print(f"Empty 1D tensor: {y.shape}")
    assert y.numel() == 0

    print("✓ Test 8 passed\n")


def test_channels_last_layout():
    """Test empty_strided with channels-last memory format"""
    print("Test 9: Channels-last memory format")

    # NCHW -> NHWC (channels last)
    size = [2, 3, 4, 5]  # N=2, C=3, H=4, W=5
    # Channels-last strides: N=60, H=15, W=3, C=1
    stride = [60, 1, 15, 3]

    x = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"Channels-last size: {x.shape}, stride: {x.stride()}")
    assert x.shape == torch.Size(size)
    assert x.stride() == tuple(stride)

    # Check if recognized as channels-last
    is_cl = x.is_contiguous(memory_format=torch.channels_last)
    print(f"Is channels-last contiguous: {is_cl}")

    print("✓ Test 9 passed\n")


def test_stride_consistency():
    """Test that strides determine memory layout correctly"""
    print("Test 10: Stride consistency and memory layout")

    size = [2, 3]

    # Row-major (C order)
    stride_c = [3, 1]
    x_c = torch.empty_strided(size, stride_c, device='zeus', dtype=torch.float32)
    print(f"Row-major: stride {x_c.stride()}, contiguous: {x_c.is_contiguous()}")
    assert x_c.is_contiguous()

    # Column-major (Fortran order)
    stride_f = [1, 2]
    x_f = torch.empty_strided(size, stride_f, device='zeus', dtype=torch.float32)
    print(f"Column-major: stride {x_f.stride()}, contiguous: {x_f.is_contiguous()}")
    assert not x_f.is_contiguous()

    # Verify they have different memory layouts
    assert x_c.stride() != x_f.stride(), "Different layouts should have different strides"

    print("✓ Test 10 passed\n")


def test_comparison_with_cpu():
    """Test that zeus empty_strided behaves like CPU version"""
    print("Test 11: Comparison with CPU implementation")

    size = [3, 4, 5]
    stride = [20, 5, 1]

    cpu_tensor = torch.empty_strided(size, stride, device='cpu', dtype=torch.float32)
    zeus_tensor = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    print(f"CPU: size={cpu_tensor.shape}, stride={cpu_tensor.stride()}")
    print(f"Zeus: size={zeus_tensor.shape}, stride={zeus_tensor.stride()}")

    # Check that properties match
    assert cpu_tensor.shape == zeus_tensor.shape
    assert cpu_tensor.stride() == zeus_tensor.stride()
    assert cpu_tensor.dtype == zeus_tensor.dtype
    assert cpu_tensor.is_contiguous() == zeus_tensor.is_contiguous()

    print("✓ Test 11 passed\n")


def test_memory_allocation():
    """Test that memory is actually allocated"""
    print("Test 12: Memory allocation verification")

    size = [100, 100]
    stride = [100, 1]

    x = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)

    # Check that we can get a valid data pointer
    ptr = x.data_ptr()
    print(f"Data pointer: {ptr}")
    assert ptr != 0, "Data pointer should not be null"

    # Create another tensor and verify different pointer
    y = torch.empty_strided(size, stride, device='zeus', dtype=torch.float32)
    assert y.data_ptr() != ptr, "Different tensors should have different pointers"

    print("✓ Test 12 passed\n")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running empty_strided operator tests")
    print("=" * 60 + "\n")

    try:
        test_basic_empty_strided()
        test_transposed_strides()
        test_custom_strides()
        test_different_dtypes()
        test_1d_tensor()
        test_high_dimensional()
        test_scalar_tensor()
        test_zero_size_dimension()
        test_channels_last_layout()
        test_stride_consistency()
        test_comparison_with_cpu()
        test_memory_allocation()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
