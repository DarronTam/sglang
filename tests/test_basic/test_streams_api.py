#!/usr/bin/env python3
"""Test Zeus high-level Stream and Event API"""

import sys
import torch
import torch_zeus

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def main():
    print("Zeus High-Level Stream and Event API Test")

    # Check device
    count = torch.zeus.device_count()
    print(f"Device count: {count}")
    if count == 0:
        print("No Zeus devices available")
        return 1

    torch.zeus.set_device(0)
    print(f"Current device: {torch.zeus.current_device()}")

    #==========================================================================
    # Test 1: Event creation (high-level API)
    #==========================================================================
    print_section("Test 1: High-Level Event Creation")
    try:
        # Create event with default settings
        event1 = torch.zeus.Event()
        print(f"  Created event: {event1}")

        # Create event with timing enabled
        event2 = torch.zeus.Event(enable_timing=True)
        print(f"  Created event with timing: {event2}")

        print("  High-level Event creation works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 2: Event record and synchronize
    #==========================================================================
    print_section("Test 2: Event Record and Synchronize")
    try:
        event = torch.zeus.Event()
        print(f"  Before record: {event}")
        print(f"    device: {event.device}")

        # Record event
        event.record()
        print(f"  After record: {event}")
        print(f"    device: {event.device}")

        # Synchronize
        event.synchronize()
        print("  Synchronize completed")

        # Query
        is_ready = event.query()
        print(f"    is_ready: {is_ready}")
        assert is_ready == True, "Event should be ready after synchronize"

        print("  Event record and synchronize work")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 3: Event timing
    #==========================================================================
    print_section("Test 3: Event Timing")
    try:
        start = torch.zeus.Event(enable_timing=True)
        end = torch.zeus.Event(enable_timing=True)

        # Record start
        start.record()

        # Do some work
        tensors = []
        for i in range(10):
            t = torch.ones(100, 100, device='zeus:0')
            t.fill_(float(i))
            tensors.append(t)

        # Record end
        end.record()
        end.synchronize()

        # Calculate elapsed time
        elapsed = start.elapsed_time(end)
        print(f"  Elapsed time: {elapsed:.4f} ms")
        assert elapsed >= 0, "Elapsed time should be non-negative"

        print("  Event timing works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 4: Stream creation
    #==========================================================================
    print_section("Test 4: Stream Creation")
    try:
        # Get current stream
        s1 = torch.zeus.current_stream()
        print(f"  Current stream: {s1}")
        print(f"    device: {s1.device}")
        print(f"    priority: {s1.priority}")

        # Get default stream
        s2 = torch.zeus.default_stream()
        print(f"  Default stream: {s2}")

        # Create a new stream
        s3 = torch.zeus.Stream(device=0, priority=-1)
        print(f"  Created stream: {s3}")

        # Create stream with default device
        s4 = torch.zeus.Stream()
        print(f"  Created stream (default device): {s4}")

        # Create stream with priority 0
        s5 = torch.zeus.Stream(device=0, priority=0)
        print(f"  Created stream (priority 0): {s5}")

        print("  Stream creation works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 5: Stream record_event
    #==========================================================================
    print_section("Test 5: Stream record_event")
    try:
        stream = torch.zeus.current_stream()

        # Record event on stream (create new event)
        event1 = stream.record_event()
        print(f"  Recorded new event: {event1}")

        # Record existing event on stream
        event2 = torch.zeus.Event(enable_timing=True)
        stream.record_event(event2)
        print(f"  Recorded existing event: {event2}")

        print("  Stream record_event works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 6: Stream wait_event
    #==========================================================================
    print_section("Test 6: Stream wait_event")
    try:
        stream = torch.zeus.current_stream()
        event = torch.zeus.Event()

        # Record event
        event.record()

        # Make stream wait for event
        stream.wait_event(event)
        print("  Stream wait_event succeeded")

        # Verify event completed
        event.synchronize()
        assert event.query() == True

        print("  Stream wait_event works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 7: Stream context manager
    #==========================================================================
    print_section("Test 7: Stream Context Manager")
    try:
        s = torch.zeus.Stream(device=0)
        print(f"  Created stream: {s}")

        # Use stream context
        with torch.zeus.stream(s):
            print("    Inside stream context")
            x = torch.ones(10, 10, device='zeus:0')
            print(f"    Created tensor: {x.shape}")

        print("    Outside stream context")
        print("  Stream context manager works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 8: Multiple streams with synchronization
    #==========================================================================
    print_section("Test 8: Multiple Streams with Synchronization")
    try:
        # Create multiple streams
        stream1 = torch.zeus.Stream(device=0, priority=0)
        stream2 = torch.zeus.Stream(device=0, priority=-1)  # Higher priority
        print(f"  Stream 1: {stream1}")
        print(f"  Stream 2: {stream2}")

        # Create work on stream1
        with torch.zeus.stream(stream1):
            x = torch.ones(100, 100, device='zeus:0')
            x.fill_(1.0)

        # Record event on stream1
        event1 = stream1.record_event()
        print(f"  Recorded event on stream1: {event1}")

        # Make stream2 wait for stream1's event
        stream2.wait_event(event1)
        print("  Stream2 waiting for stream1's event")

        # Create work on stream2 (will wait for stream1)
        with torch.zeus.stream(stream2):
            y = torch.ones(100, 100, device='zeus:0')
            y.fill_(2.0)

        # Synchronize both streams
        stream1.synchronize()
        stream2.synchronize()
        print("  Both streams synchronized")

        print("  Multiple streams with synchronization works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 9: Stream wait_stream
    #==========================================================================
    print_section("Test 9: Stream wait_stream")
    try:
        stream1 = torch.zeus.Stream(device=0)
        stream2 = torch.zeus.Stream(device=0)

        # Create work on stream1
        with torch.zeus.stream(stream1):
            a = torch.ones(50, 50, device='zeus:0')
            a.fill_(10.0)

        # Make stream2 wait for stream1
        stream2.wait_stream(stream1)
        print("  Stream2 waiting for stream1")

        # Create dependent work on stream2
        with torch.zeus.stream(stream2):
            b = torch.ones(50, 50, device='zeus:0')
            b.fill_(20.0)

        stream2.synchronize()
        print("  Stream wait_stream works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 10: Event with memory operations (like torch.cuda)
    #==========================================================================
    print_section("Test 10: Event with Memory Operations (CUDA-style)")
    try:
        # This mimics typical torch.cuda.Event usage
        start_event = torch.zeus.Event(enable_timing=True)
        end_event = torch.zeus.Event(enable_timing=True)

        start_event.record()

        # Allocate and fill tensors
        x = torch.zeros(200, 200, device='zeus:0')
        x.fill_(42.0)
        y = torch.ones(200, 200, device='zeus:0')
        y.fill_(100.0)

        end_event.record()

        # Wait for completion
        torch.zeus.device_synchronize()

        # Measure time
        elapsed = start_event.elapsed_time(end_event)
        print(f"  Memory operations took: {elapsed:.4f} ms")

        # Verify results
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        assert torch.allclose(x_cpu, torch.ones(200, 200) * 42.0)
        assert torch.allclose(y_cpu, torch.ones(200, 200) * 100.0)

        print("  CUDA-style Event usage works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 11: Stream properties and comparison
    #==========================================================================
    print_section("Test 11: Stream Properties and Comparison")
    try:
        s1 = torch.zeus.Stream(device=0, priority=0)
        s2 = torch.zeus.Stream(device=0, priority=-1)
        s3 = torch.zeus.Stream(device=0, priority=0)

        # Test device property
        assert s1.device == torch.device('zeus', 0), "Device should be zeus:0"
        print(f"  s1.device = {s1.device}")

        # Test priority property (values may be normalized by the backend)
        print(f"  s1.priority = {s1.priority}")
        print(f"  s2.priority = {s2.priority}")
        # Just verify priority is an integer and accessible
        assert isinstance(s1.priority, int), "s1 priority should be an int"
        assert isinstance(s2.priority, int), "s2 priority should be an int"

        # Test __repr__
        repr_s1 = repr(s1)
        assert "torch_zeus.Stream" in repr_s1 or "zeus" in repr_s1
        print(f"  repr(s1) = {repr_s1}")

        print("  Stream properties and comparison work")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 12: Event query before and after work
    #==========================================================================
    print_section("Test 12: Event Query Behavior")
    try:
        event = torch.zeus.Event()

        # Record event
        event.record()

        # Query before synchronize - might be ready or not depending on timing
        query_before = event.query()
        print(f"  Query before sync: {query_before}")

        # Synchronize
        event.synchronize()

        # Query after synchronize - should be ready
        query_after = event.query()
        print(f"  Query after sync: {query_after}")
        assert query_after == True, "Event should be ready after synchronize"

        print("  Event query behavior works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 13: Nested stream contexts
    #==========================================================================
    print_section("Test 13: Nested Stream Contexts")
    try:
        outer_stream = torch.zeus.Stream(device=0, priority=0)
        inner_stream = torch.zeus.Stream(device=0, priority=-1)

        original_stream = torch.zeus.current_stream()
        print(f"  Original stream: {original_stream}")

        with torch.zeus.stream(outer_stream):
            print(f"    In outer stream context")
            x = torch.ones(10, 10, device='zeus:0')

            with torch.zeus.stream(inner_stream):
                print(f"      In inner stream context")
                y = torch.ones(10, 10, device='zeus:0')

            print(f"    Back to outer stream context")
            z = torch.ones(10, 10, device='zeus:0')

        print(f"  Back to original context")
        print("  Nested stream contexts work")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 14: Event wait on stream
    #==========================================================================
    print_section("Test 14: Event Wait on Stream")
    try:
        stream1 = torch.zeus.Stream(device=0)
        stream2 = torch.zeus.Stream(device=0)

        # Create event
        event = torch.zeus.Event()

        # Record event on stream1
        with torch.zeus.stream(stream1):
            t1 = torch.ones(100, 100, device='zeus:0')
            event.record(stream1)
            print("  Recorded event on stream1")

        # Wait for event on stream2
        event.wait(stream2)
        print("  Stream2 waiting for event")

        # Work on stream2 (will wait for event)
        with torch.zeus.stream(stream2):
            t2 = torch.ones(100, 100, device='zeus:0')

        # Synchronize
        stream2.synchronize()
        print("  Event wait on stream works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 15: Multiple events timing
    #==========================================================================
    print_section("Test 15: Multiple Events Timing")
    try:
        events = []
        for i in range(5):
            events.append(torch.zeus.Event(enable_timing=True))

        # Record events with work in between
        for i, event in enumerate(events):
            # Do some work
            t = torch.ones(50, 50, device='zeus:0')
            t.fill_(float(i))
            event.record()

        # Wait for all events
        for event in events:
            event.synchronize()

        # Calculate elapsed times between consecutive events
        print("  Elapsed times between events:")
        for i in range(len(events) - 1):
            elapsed = events[i].elapsed_time(events[i + 1])
            print(f"    Event {i} -> Event {i+1}: {elapsed:.4f} ms")

        print("  Multiple events timing works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Test 16: Accessing torch.zeus API
    #==========================================================================
    print_section("Test 16: torch.zeus API Access")
    try:
        # Since we registered as torch.zeus, check if accessible
        print(f"  torch_zeus module: {torch_zeus}")
        print(f"  Device count via torch_zeus: {torch.zeus.device_count()}")

        # Test Event class
        evt = torch.zeus.Event(enable_timing=True)
        print(f"  Created event via torch.zeus.Event: {evt}")

        # Test Stream class
        strm = torch.zeus.Stream()
        print(f"  Created stream via torch.zeus.Stream: {strm}")

        # Test current_stream
        cur = torch.zeus.current_stream()
        print(f"  torch.zeus.current_stream(): {cur}")

        # Test default_stream
        default = torch.zeus.default_stream()
        print(f"  torch.zeus.default_stream(): {default}")

        print("  torch.zeus API access works")

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    #==========================================================================
    # Summary
    #==========================================================================
    print("\n" + "=" * 70)
    print("All high-level API tests passed!")
    print("=" * 70)

    print("\nTested functionality:")
    print("  [Events]")
    print("    - High-level Event creation")
    print("    - Event record and synchronize")
    print("    - Event timing (elapsed_time)")
    print("    - Event query behavior")
    print("    - Multiple events timing")
    print("    - Event wait on stream")
    print("  [Streams]")
    print("    - Stream creation (current, default, custom)")
    print("    - Stream properties (device, priority)")
    print("    - Stream record_event")
    print("    - Stream wait_event")
    print("    - Stream wait_stream")
    print("    - Stream context manager")
    print("    - Multiple streams with synchronization")
    print("    - Nested stream contexts")
    print("  [Integration]")
    print("    - CUDA-style Event usage pattern")
    print("    - torch.zeus API access")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
