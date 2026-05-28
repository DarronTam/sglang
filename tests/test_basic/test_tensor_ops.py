#!/usr/bin/env python3
"""Test Zeus tensor operations (zeros, ones, full, fill_, copy_, to, cpu)"""

import sys
import torch
import torch_zeus

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def print_tensor_info(name, tensor):
    """Print tensor information"""
    print(f"{name}:")
    print(f"  shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    
    # Only print data for CPU tensors or small Zeus tensors (converted to CPU)
    if tensor.device.type == 'cpu':
        if tensor.numel() > 0 and tensor.numel() <= 20:
            print(f"  data={tensor}")
        elif tensor.numel() > 0:
            print(f"  data (first 5)={tensor.flatten()[:5]}")
    elif tensor.numel() > 0 and tensor.numel() <= 100:
        # For Zeus tensors, transfer to CPU to print
        cpu_tensor = tensor.cpu()
        if cpu_tensor.numel() <= 20:
            print(f"  data={cpu_tensor}")
        else:
            print(f"  data (first 5)={cpu_tensor.flatten()[:5]}")

def main():
    print("Zeus Tensor Operations Test")
    
    # Check device
    count = torch.zeus.device_count()
    print(f"Device count: {count}")
    if count == 0:
        print("No Zeus devices available")
        return 1
    
    torch.zeus.set_device(0)
    print(f"Current device: {torch.zeus.current_device()}")
    
    #==========================================================================
    # Test 1: zeros
    #==========================================================================
    print_section("Test 1: torch.zeros")
    try:
        x = torch.zeros(3, 4, device='zeus:0')
        print_tensor_info("zeros(3, 4)", x)
        
        # Verify all zeros
        x_cpu = x.cpu()
        assert x_cpu.sum().item() == 0, "zeros should create all-zero tensor"
        print("✓ All values are zero")
        
        # Test different dtypes
        y = torch.zeros(2, 3, dtype=torch.int64, device='zeus:0')
        print_tensor_info("zeros(2, 3, dtype=int64)", y)
        print("✓ zeros with dtype works")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    #==========================================================================
    # Test 2: ones
    #==========================================================================
    print_section("Test 2: torch.ones")
    try:
        x = torch.ones(2, 5, device='zeus:0')
        print_tensor_info("ones(2, 5)", x)
        
        # Verify all ones
        x_cpu = x.cpu()
        assert x_cpu.sum().item() == 10, "ones should create all-one tensor"
        print("✓ All values are one")
        
        # Test different dtypes
        y = torch.ones(3, dtype=torch.float32, device='zeus:0')
        print_tensor_info("ones(3, dtype=float32)", y)
        print("✓ ones with dtype works")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    #==========================================================================
    # Test 3: full
    #==========================================================================
    print_section("Test 3: torch.full")
    try:
        x = torch.full((3, 3), 3.14, device='zeus:0')
        print_tensor_info("full((3, 3), 3.14)", x)
        
        # Verify all values are 3.14
        x_cpu = x.cpu()
        assert abs(x_cpu[0, 0].item() - 3.14) < 0.01, "full should create tensor with specified value"
        print("✓ All values are 3.14")
        
        # Test with integer
        y = torch.full((2, 4), 42, dtype=torch.int8, device='zeus:0')
        print_tensor_info("full((2, 4), 42, dtype=int8)", y)
        print("✓ full with integer works")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    #==========================================================================
    # Test 4: fill_
    #==========================================================================
    print_section("Test 4: tensor.fill_")
    try:
        x = torch.empty(2, 3, device='zeus:0')
        print(f"Before fill_: shape={x.shape}, device={x.device}")
        
        x.fill_(7.0)
        print_tensor_info("After fill_(7.0)", x)
        
        # Verify all values are 7.0
        x_cpu = x.cpu()
        assert abs(x_cpu[0, 0].item() - 7.0) < 0.01, "fill_ should fill tensor with specified value"
        print("✓ fill_ works correctly")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    #==========================================================================
    # Test 5: copy_ (Zeus -> Zeus)
    #==========================================================================
    print_section("Test 5: copy_ (Zeus -> Zeus)")
    try:
        x = torch.ones(3, 4, device='zeus:0')
        y = torch.zeros(3, 4, device='zeus:0')
        
        print(f"Source (ones): shape={x.shape}, device={x.device}")
        print(f"Dest before copy (zeros): shape={y.shape}, device={y.device}")
        
        y.copy_(x)
        print_tensor_info("Dest after copy_", y)
        
        # Verify copy worked
        y_cpu = y.cpu()
        assert y_cpu.sum().item() == 12, "copy_ should copy all values"
        print("✓ Zeus -> Zeus copy works")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    #==========================================================================
    # Test 6: to('cpu') and copy_ (Zeus -> CPU)
    #==========================================================================
    print_section("Test 6: Zeus -> CPU")
    try:
        # Create tensor on Zeus with known values
        x = torch.full((2, 3), 5.0, device='zeus:0')
        print(f"Zeus tensor: shape={x.shape}, device={x.device}")
        
        # Transfer to CPU
        y = x.to('cpu')
        print_tensor_info("CPU tensor (via to)", y)
        
        # Verify values
        assert y.sum().item() == 30, "to('cpu') should preserve values"
        assert y.device.type == 'cpu', "Tensor should be on CPU"
        print("✓ Zeus -> CPU transfer works")
        
        # Test copy_ from Zeus to CPU
        z = torch.zeros(2, 3)  # CPU tensor
        z.copy_(x)
        print_tensor_info("CPU tensor (via copy_)", z)
        assert z.sum().item() == 30, "copy_ Zeus->CPU should preserve values"
        print("✓ copy_ Zeus -> CPU works")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    #==========================================================================
    # Test 7: to('zeus') and copy_ (CPU -> Zeus)
    #==========================================================================
    print_section("Test 7: CPU -> Zeus")
    try:
        # Create tensor on CPU with known values
        x = torch.full((3, 2), 8.0)
        print_tensor_info("CPU tensor", x)
        
        # Transfer to Zeus
        y = x.to('zeus:0')
        print(f"Zeus tensor (via to): shape={y.shape}, device={y.device}")
        
        # Verify values (transfer back to CPU to check)
        y_cpu = y.to('cpu')
        assert y_cpu.sum().item() == 48, "to('zeus:0') should preserve values"
        assert y.device.type == 'zeus', "Tensor should be on Zeus"
        print("✓ CPU -> Zeus transfer works")
        
        # Test copy_ from CPU to Zeus
        z = torch.zeros(3, 2, device='zeus:0')
        z.copy_(x)
        z_cpu = z.to('cpu')
        print_tensor_info("Zeus tensor (via copy_)", z_cpu)
        assert z_cpu.sum().item() == 48, "copy_ CPU->Zeus should preserve values"
        print("✓ copy_ CPU -> Zeus works")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    #==========================================================================
    # Test 8: cpu() method
    #==========================================================================
    print_section("Test 8: tensor.cpu() method")
    try:
        x = torch.ones(2, 2, device='zeus:0')
        print(f"Zeus tensor: shape={x.shape}, device={x.device}")
        
        y = x.cpu()
        print_tensor_info("After .cpu()", y)
        
        assert y.device.type == 'cpu', "cpu() should return CPU tensor"
        assert y.sum().item() == 4, "cpu() should preserve values"
        print("✓ .cpu() method works")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    #==========================================================================
    # Test 9: Round-trip test
    #==========================================================================
    print_section("Test 9: Round-trip (CPU -> Zeus -> CPU)")
    try:
        # Create tensor with specific pattern
        original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        print_tensor_info("Original CPU tensor", original)
        
        # Transfer to Zeus
        on_zeus = original.to('zeus:0')
        print(f"On Zeus: shape={on_zeus.shape}, device={on_zeus.device}")
        
        # Transfer back to CPU
        back_to_cpu = on_zeus.to('cpu')
        print_tensor_info("Back to CPU", back_to_cpu)
        
        # Verify values preserved
        assert torch.allclose(original, back_to_cpu), "Round-trip should preserve values"
        print("✓ Round-trip preserves all values correctly")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    #==========================================================================
    # Test 10: Memory stats
    #==========================================================================
    print_section("Test 10: Memory Statistics")
    try:
        print(f"Allocated: {torch.zeus.memory_allocated():,} bytes")
        print(f"Reserved: {torch.zeus.memory_reserved():,} bytes")
        
        # Create some tensors
        tensors = []
        for i in range(5):
            tensors.append(torch.ones(100, 100, device='zeus:0'))
        
        print(f"After creating 5 tensors:")
        print(f"  Allocated: {torch.zeus.memory_allocated():,} bytes")
        print(f"  Reserved: {torch.zeus.memory_reserved():,} bytes")
        
        # Clean up
        del tensors
        torch.zeus.empty_cache()
        
        print(f"After cleanup:")
        print(f"  Allocated: {torch.zeus.memory_allocated():,} bytes")
        print(f"  Reserved: {torch.zeus.memory_reserved():,} bytes")
        
        print("✓ Memory tracking works")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    try:
        result = main()
        # Force cleanup before exit to avoid destructor ordering issues
        import gc
        gc.collect()
        sys.exit(result)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
