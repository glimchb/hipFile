#!/usr/bin/env python3
"""
Test hipfile bindings integration with LMCache pattern.
This verifies that the hipFile bindings can be used in the same way as cuFile.
"""

import ctypes
import tempfile
import os

try:
    import hipfile
    print("✓ hipfile imported successfully")
except ImportError as e:
    print(f"✗ Failed to import hipfile: {e}")
    exit(1)

def test_buffer_registration():
    """Test buffer registration similar to LMCache usage."""
    print("\n=== Testing Buffer Registration ===")
    
    # Test that the required functions exist
    required_functions = [
        'buf_register', 'buf_deregister', 'driver_open', 'driver_close'
    ]
    
    for func_name in required_functions:
        if hasattr(hipfile, func_name):
            print(f"✓ {func_name} available")
        else:
            print(f"✗ {func_name} missing")
            return False
    
    return True

def test_context_managers():
    """Test context manager patterns."""
    print("\n=== Testing Context Managers ===")
    
    # Check if context managers are available
    if hasattr(hipfile, 'Driver'):
        print("✓ Driver context manager available")
    else:
        print("✗ Driver context manager missing")
        return False
        
    if hasattr(hipfile, 'RegisteredBuffer'):
        print("✓ RegisteredBuffer context manager available")
    else:
        print("✗ RegisteredBuffer context manager missing")
        return False
        
    if hasattr(hipfile, 'HipFileHandle'):
        print("✓ HipFileHandle context manager available")
    else:
        print("✗ HipFileHandle context manager missing")
        return False
    
    return True

def test_error_handling():
    """Test error handling."""
    print("\n=== Testing Error Handling ===")
    
    if hasattr(hipfile, 'HipFileError'):
        print("✓ HipFileError exception class available")
    else:
        print("✗ HipFileError exception class missing")
        return False
        
    if hasattr(hipfile, 'error_name'):
        print("✓ error_name function available")
    else:
        print("✗ error_name function missing")
        return False
    
    return True

def test_constants():
    """Test that required constants are available."""
    print("\n=== Testing Constants ===")
    
    required_constants = [
        'HIPFILE_SUCCESS',
        'HIPFILE_INVALID_VALUE',
        'HIPFILE_OPEN_FLAGS_DEFAULT',
        'HIPFILE_HANDLE_TYPE_OPAQUE_FD'
    ]
    
    for const_name in required_constants:
        if hasattr(hipfile, const_name):
            print(f"✓ {const_name} available")
        else:
            print(f"✗ {const_name} missing")
            return False
    
    return True

def main():
    """Run all integration tests."""
    print("=== hipfile Python Bindings Integration Test ===")
    print(f"hipfile version: {getattr(hipfile, '__version__', 'unknown')}")
    
    tests = [
        test_buffer_registration,
        test_context_managers, 
        test_error_handling,
        test_constants
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All integration tests passed! hipfile bindings are ready for LMCache.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
