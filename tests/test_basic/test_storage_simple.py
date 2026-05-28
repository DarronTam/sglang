"""Simple storage test"""
import torch
import torch_zeus

print("=" * 60)
print("SIMPLE STORAGE TEST")
print("=" * 60)

print(f"\ntorch_zeus file: {torch_zeus.__file__}")
print(f"torch_zeus._has_cpp_ext: {torch_zeus._has_cpp_ext if hasattr(torch_zeus, '_has_cpp_ext') else 'N/A'}")
print(f"\n1. Check if UntypedStorage exists: {hasattr(torch.zeus, 'UntypedStorage')}")
if hasattr(torch.zeus, 'UntypedStorage'):
    print(f"2. UntypedStorage type: {type(torch.zeus.UntypedStorage)}")
else:
    print("2. UntypedStorage NOT FOUND")
    print("   Available storage classes:")
    for attr in sorted(dir(torch.zeus)):
        if 'Storage' in attr:
            print(f"   - {attr}")

try:
    storage = torch.zeus.UntypedStorage(1024, device=torch.device('zeus'))
    print(f"3. Created storage: {storage}")
    print(f"4. Storage size: {storage.size()}")
    print(f"5. Storage device: {storage.device}")
    print(f"6. Storage is_zeus: {storage.is_zeus}")
    print("\n✓ ALL TESTS PASSED!")
except Exception as e:
    print(f"\n✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
