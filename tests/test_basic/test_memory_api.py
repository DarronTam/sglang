"""
Test Memory API Functions

Tests for the advanced memory management API:
- memory_summary()
- memory_snapshot()
- mem_get_info()
- memory_stats_as_nested_dict()
"""

import pytest
import torch
import torch_zeus
import sys


def _flatten_stats(stats):
    """Flatten nested allocator stats using the same dotted-key format as memory_stats()."""
    result = {}

    def recurse(prefix, obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                recurse(next_prefix, value)
        else:
            result[prefix] = obj

    recurse("", stats)
    return result


def test_memory_stats_as_nested_dict():
    """Test memory_stats_as_nested_dict() function"""
    print("\n" + "=" * 60)
    print("TEST: memory_stats_as_nested_dict()")
    print("=" * 60)

    # Get nested stats
    nested_stats = torch.zeus.memory_stats_as_nested_dict()

    # Check that it's a dict
    assert isinstance(nested_stats, dict), \
        f"Should return dict, got {type(nested_stats)}"
    print("✓ Returns a dictionary")

    # Check nested structure
    assert 'allocated_bytes' in nested_stats, "Should have 'allocated_bytes' key"
    assert 'all' in nested_stats['allocated_bytes'], "Should have nested 'all' key"
    assert 'current' in nested_stats['allocated_bytes']['all'], "Should have nested 'current' key"

    print(f"✓ Has nested structure")
    print(f"  Sample: allocated_bytes.all.current = {nested_stats['allocated_bytes']['all']['current']}")

    # Verify the nested snapshot flattens to the same dotted-key structure.
    # Do not compare values against a second memory_stats() call: in a full
    # suite, live Zeus tensors from earlier tests may be released between two
    # independent allocator snapshots.
    flat_from_nested = _flatten_stats(nested_stats)
    assert flat_from_nested['allocated_bytes.all.current'] == nested_stats['allocated_bytes']['all']['current'], \
        "Flattened nested stat should match the source snapshot"

    flat_stats = torch.zeus.memory_stats()
    assert 'allocated_bytes.all.current' in flat_stats, \
        "Flat stats should include allocated_bytes.all.current"
    assert set(flat_from_nested) == set(flat_stats), \
        "Nested and flat stats should expose the same keys"
    print("✓ Matches flat stats structure\n")


def test_memory_summary():
    """Test memory_summary() function"""
    print("=" * 60)
    print("TEST: memory_summary()")
    print("=" * 60)

    # Allocate some memory to make summary interesting
    x = torch.randn(1000, 1000, device='zeus')

    # Get memory summary
    summary = torch.zeus.memory_summary()

    # Check that it's a string
    assert isinstance(summary, str), \
        f"Should return str, got {type(summary)}"
    print("✓ Returns a string")

    # Check that it contains expected content
    assert "PyTorch Zeus memory summary" in summary, \
        "Should contain title"
    assert "Allocated memory" in summary, \
        "Should contain memory metrics"
    assert "Zeus OOMs" in summary, \
        "Should contain OOM info"

    print("✓ Contains expected content")
    print("\nSample output:")
    print(summary[:500])  # Print first 500 chars

    # Test abbreviated version
    summary_abbrev = torch.zeus.memory_summary(abbreviated=True)
    assert isinstance(summary_abbrev, str), \
        "Abbreviated version should also return str"
    print("✓ Abbreviated version works")

    # Clean up
    del x
    torch.zeus.empty_cache()
    print("✓ Test completed\n")


def test_memory_snapshot():
    """Test memory_snapshot() function"""
    print("=" * 60)
    print("TEST: memory_snapshot()")
    print("=" * 60)

    # Allocate memory so the snapshot has segments to report
    x = torch.randn(1000, 1000, device='zeus')

    # Get snapshot
    snapshot = torch.zeus.memory_snapshot()

    # Check that it's a list
    assert isinstance(snapshot, list), \
        f"Should return list, got {type(snapshot)}"
    print(f"✓ Returns a list with {len(snapshot)} segments")

    # With an active allocation there must be at least one segment
    assert len(snapshot) > 0, "Should have at least 1 segment after allocation"

    segment = snapshot[0]
    assert isinstance(segment, dict), "Segments should be dictionaries"

    # Check required keys
    required_keys = ['device', 'total_size', 'allocated_size', 'active_size']
    for key in required_keys:
        assert key in segment, f"Segment should have '{key}' key"

    print("✓ Segments have correct structure")
    print(f"  Sample segment:")
    for key in required_keys:
        print(f"    {key}: {segment[key]}")

    # Check blocks sub-structure
    assert 'blocks' in segment, "Segment should have 'blocks' key"
    assert len(segment['blocks']) > 0, "Segment should have at least one block"
    block = segment['blocks'][0]
    assert 'size' in block and 'state' in block, \
        "Block should have 'size' and 'state' keys"
    print(f"✓ Block structure valid ({len(segment['blocks'])} blocks)")

    # Clean up
    del x
    torch.zeus.empty_cache()
    print("✓ Snapshot function works\n")


def test_mem_get_info():
    """Test mem_get_info() function"""
    print("=" * 60)
    print("TEST: mem_get_info()")
    print("=" * 60)

    # Get memory info
    free, total = torch.zeus.mem_get_info()

    # Check return types
    assert isinstance(free, int), f"Free should be int, got {type(free)}"
    assert isinstance(total, int), f"Total should be int, got {type(total)}"
    print("✓ Returns (free, total) as integers")

    # Check values make sense
    assert free >= 0, "Free memory should be non-negative"
    assert total > 0, "Total memory should be positive"
    assert free <= total, "Free memory should not exceed total"

    print(f"✓ Values are sensible")
    print(f"  Free:  {free / 1024**3:.2f} GB")
    print(f"  Total: {total / 1024**3:.2f} GB")
    print(f"  Used:  {(total - free) / 1024**3:.2f} GB")

    # Test with device argument
    free2, total2 = torch.zeus.mem_get_info(0)
    assert free2 == free and total2 == total, \
        "Should return same values for device 0"
    print("✓ Device argument works\n")


def test_integration():
    """Test integration of all memory APIs"""
    print("=" * 60)
    print("TEST: Integration Test")
    print("=" * 60)

    # Allocate some tensors
    print("Allocating tensors...")
    tensors = []
    for i in range(5):
        t = torch.randn(500, 500, device='zeus')
        tensors.append(t)

    # Check that memory stats reflect allocations
    stats = torch.zeus.memory_stats_as_nested_dict()
    allocated = stats['allocated_bytes']['all']['current']
    print(f"✓ Allocated {allocated / 1024**2:.2f} MB")

    # Get memory info
    free, total = torch.zeus.mem_get_info()
    print(f"✓ Memory info: {free / 1024**3:.2f} GB free / {total / 1024**3:.2f} GB total")

    # Get summary
    summary = torch.zeus.memory_summary(abbreviated=True)
    assert "Allocated memory" in summary
    print("✓ Memory summary generated")

    # Get snapshot
    snapshot = torch.zeus.memory_snapshot()
    assert len(snapshot) > 0
    print(f"✓ Snapshot contains {len(snapshot)} segments")

    # Free memory
    tensors.clear()
    torch.zeus.empty_cache()

    # Check that memory was freed
    stats_after = torch.zeus.memory_stats_as_nested_dict()
    allocated_after = stats_after['allocated_bytes']['all']['current']
    assert allocated_after < allocated, "Memory should be freed"
    print(f"✓ Memory freed: {allocated_after / 1024**2:.2f} MB remaining")

    print("✓ Integration test passed\n")


def test_device_argument_variants():
    """Test different forms of device arguments"""
    print("=" * 60)
    print("TEST: Device Argument Variants")
    print("=" * 60)

    # Test different ways to specify device
    results = []

    # Int
    free1, total1 = torch.zeus.mem_get_info(0)
    results.append((free1, total1))
    print("✓ mem_get_info(0) works")

    # Default (None)
    free2, total2 = torch.zeus.mem_get_info()
    results.append((free2, total2))
    print("✓ mem_get_info() works")

    # torch.device
    free3, total3 = torch.zeus.mem_get_info(torch.device('zeus', 0))
    results.append((free3, total3))
    print("✓ mem_get_info(torch.device('zeus', 0)) works")

    # String (if supported)
    try:
        free4, total4 = torch.zeus.mem_get_info('zeus:0')
        results.append((free4, total4))
        print("✓ mem_get_info('zeus:0') works")
    except Exception as e:
        print(f"⚠ String device not supported: {e}")

    # All should return similar values (may vary slightly due to timing)
    for i, (f, t) in enumerate(results[1:], 1):
        assert abs(f - results[0][0]) < 1024**3, f"Result {i} free differs significantly"
        assert t == results[0][1], f"Result {i} total differs"

    print("✓ All device argument forms work\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MEMORY API TEST SUITE")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Zeus devices available: {torch.zeus.device_count()}")
    print(f"Zeus available: {torch.zeus.is_available()}")
    print("=" * 60)

    try:
        test_memory_stats_as_nested_dict()
        test_memory_summary()
        test_memory_snapshot()
        test_mem_get_info()
        test_integration()
        test_device_argument_variants()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60 + "\n")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60 + "\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
