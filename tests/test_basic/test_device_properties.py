#!/usr/bin/env python3
"""Test device properties implementation"""

import torch_zeus

def test_get_device_properties():
    """Test getting device properties"""
    print("Testing get_device_properties()...")

    # Get device properties for device 0
    props = torch_zeus._C.get_device_properties(0)

    print(f"\nDevice Properties:")
    print(f"  Name: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  Multiprocessors: {props.multi_core_count}")
    print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"  Clock Rate: {props.clock_rate / 1000:.0f} MHz")
    print(f"  Memory Clock Rate: {props.memory_clock_rate / 1000:.0f} MHz")
    print(f"  Memory Bus Width: {props.memory_bus_width} bits")
    print(f"  L2 Cache Size: {props.l2_cache_size / (1024**2):.2f} MB")
    print(f"  Max Threads Per Block: {props.max_threads_per_block}")
    print(f"  Max Threads Per SM: {props.max_threads_per_multiprocessor}")
    print(f"  Supports BF16: {props.supports_bf16}")
    print(f"  Supports FP16: {props.supports_fp16}")
    print(f"  Supports Linear Memory: {props.supports_linear_memory}")

    # Verify properties are valid
    assert props.name != "", "Device name should not be empty"
    assert props.major >= 0, "Major version should be non-negative"
    assert props.minor >= 0, "Minor version should be non-negative"
    assert props.multi_core_count > 0, "Should have at least one multiprocessor"
    assert props.total_memory > 0, "Should have some memory"
    assert props.clock_rate > 0, "Clock rate should be positive"

    print("\n  ✓ Device properties are valid")

def test_get_device_properties_default():
    """Test getting device properties with default device (-1)"""
    print("\nTesting get_device_properties() with default device...")

    props = torch_zeus._C.get_device_properties(-1)
    print(f"  Default device properties: {props.name}")

    assert props.name != "", "Device name should not be empty"
    print("  ✓ Default device properties work")

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Device Properties Implementation")
    print("=" * 70)

    try:
        test_get_device_properties()
        test_get_device_properties_default()

        print("\n" + "=" * 70)
        print("All device properties tests passed! ✓")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
