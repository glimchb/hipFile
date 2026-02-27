"""
Tests for hipfile Python bindings.

These tests use a mock shared library so they run on any machine without
real AMD GPU hardware or libhipfile.so installed.
"""

import ctypes
import ctypes.util
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Shared mock library helper
# ---------------------------------------------------------------------------

def _make_mock_lib():
    """Return a MagicMock that looks enough like the ctypes CDLL for our API."""
    lib = MagicMock()

    from hipfile._hipfile import hipFileStatus_t, hipFileDriverProps_t

    # Default: everything succeeds
    ok = hipFileStatus_t()
    ok.err    = 0
    ok.cu_err = 0

    lib.hipFileDriverOpen.return_value = ok
    lib.hipFileDriverClose.return_value = None
    lib.hipFileDriverGetProperties.return_value = ok
    lib.hipFileDriverSetMaxDirectIOSize.return_value = ok
    lib.hipFileDriverSetMaxCacheSize.return_value = ok
    lib.hipFileDriverSetMaxPinnedMemSize.return_value = ok
    lib.hipFileHandleRegister.return_value = ok
    lib.hipFileHandleDeregister.return_value = None
    lib.hipFileBufRegister.return_value = ok
    lib.hipFileBufDeregister.return_value = ok
    lib.hipFileRead.return_value  = 1024
    lib.hipFileWrite.return_value = 1024
    lib.hipFileReadAsync.return_value  = ok
    lib.hipFileWriteAsync.return_value = ok
    return lib


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDriverLifecycle(unittest.TestCase):

    def setUp(self):
        import hipfile._hipfile as low
        import hipfile.hipfile  as high
        self.mock_lib = _make_mock_lib()
        # Patch _get_lib so no real .so is needed
        self.patcher = patch("hipfile.hipfile._get_lib", return_value=self.mock_lib)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        import hipfile._hipfile
        hipfile._hipfile._lib = None  # reset singleton

    def test_driver_open_close(self):
        import hipfile
        hipfile.driver_open()
        self.mock_lib.hipFileDriverOpen.assert_called_once()
        hipfile.driver_close()
        self.mock_lib.hipFileDriverClose.assert_called_once()

    def test_driver_context_manager(self):
        import hipfile
        with hipfile.Driver():
            pass
        self.mock_lib.hipFileDriverOpen.assert_called_once()
        self.mock_lib.hipFileDriverClose.assert_called_once()

    def test_driver_open_error_raises(self):
        import hipfile
        from hipfile._hipfile import hipFileStatus_t, HIPFILE_PLATFORM_NOT_SUPPORTED
        bad = hipFileStatus_t()
        bad.err = HIPFILE_PLATFORM_NOT_SUPPORTED
        self.mock_lib.hipFileDriverOpen.return_value = bad
        with self.assertRaises(hipfile.HipFileError) as cm:
            hipfile.driver_open()
        self.assertEqual(cm.exception.code, HIPFILE_PLATFORM_NOT_SUPPORTED)

    def test_driver_set_limits(self):
        import hipfile
        hipfile.driver_set_max_direct_io_size(128)
        hipfile.driver_set_max_cache_size(512)
        hipfile.driver_set_max_pinned_mem_size(256)
        self.mock_lib.hipFileDriverSetMaxDirectIOSize.assert_called_once()
        self.mock_lib.hipFileDriverSetMaxCacheSize.assert_called_once()
        self.mock_lib.hipFileDriverSetMaxPinnedMemSize.assert_called_once()


class TestBufRegistration(unittest.TestCase):

    def setUp(self):
        self.mock_lib = _make_mock_lib()
        self.patcher = patch("hipfile.hipfile._get_lib", return_value=self.mock_lib)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_buf_register_deregister(self):
        import hipfile
        FAKE_PTR = 0xDEADBEEF
        hipfile.buf_register(FAKE_PTR, 4096)
        self.mock_lib.hipFileBufRegister.assert_called_once()
        hipfile.buf_deregister(FAKE_PTR)
        self.mock_lib.hipFileBufDeregister.assert_called_once()

    def test_registered_buffer_context_manager(self):
        import hipfile
        FAKE_PTR = 0xCAFEBABE
        with hipfile.RegisteredBuffer(FAKE_PTR, 8192):
            self.mock_lib.hipFileBufRegister.assert_called_once()
        self.mock_lib.hipFileBufDeregister.assert_called_once()

    def test_buf_register_error(self):
        import hipfile
        from hipfile._hipfile import hipFileStatus_t, HIPFILE_INVALID_PTR
        bad = hipFileStatus_t(); bad.err = HIPFILE_INVALID_PTR
        self.mock_lib.hipFileBufRegister.return_value = bad
        with self.assertRaises(hipfile.HipFileError):
            hipfile.buf_register(0, 4096)


class TestHipFileHandle(unittest.TestCase):

    def setUp(self):
        self.mock_lib = _make_mock_lib()
        self.patcher = patch("hipfile.hipfile._get_lib", return_value=self.mock_lib)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_handle_register_deregister(self):
        import hipfile
        with hipfile.HipFileHandle(fd=3) as hf:
            self.mock_lib.hipFileHandleRegister.assert_called_once()
        self.mock_lib.hipFileHandleDeregister.assert_called_once()

    def test_read_returns_byte_count(self):
        import hipfile
        with hipfile.HipFileHandle(fd=3) as hf:
            n = hf.read(device_ptr=0xABCDEF, count=1024, file_offset=0)
        self.assertEqual(n, 1024)

    def test_write_returns_byte_count(self):
        import hipfile
        with hipfile.HipFileHandle(fd=3) as hf:
            n = hf.write(device_ptr=0xABCDEF, count=1024, file_offset=0)
        self.assertEqual(n, 1024)

    def test_read_io_error(self):
        import hipfile
        self.mock_lib.hipFileRead.return_value = -9
        with hipfile.HipFileHandle(fd=3) as hf:
            with self.assertRaises(hipfile.HipFileError):
                hf.read(device_ptr=0x1, count=512, file_offset=0)

    def test_read_async(self):
        import hipfile
        with hipfile.HipFileHandle(fd=3) as hf:
            # Should not raise
            hf.read_async(device_ptr=0x1, count=512, file_offset=0, stream=None)
        self.mock_lib.hipFileReadAsync.assert_called_once()

    def test_write_async(self):
        import hipfile
        with hipfile.HipFileHandle(fd=3) as hf:
            hf.write_async(device_ptr=0x1, count=512, file_offset=0, stream=None)
        self.mock_lib.hipFileWriteAsync.assert_called_once()

    def test_handle_register_error(self):
        import hipfile
        from hipfile._hipfile import hipFileStatus_t, HIPFILE_INVALID_FILE_DESCRIPTOR
        bad = hipFileStatus_t(); bad.err = HIPFILE_INVALID_FILE_DESCRIPTOR
        self.mock_lib.hipFileHandleRegister.return_value = bad
        with self.assertRaises(hipfile.HipFileError) as cm:
            hipfile.HipFileHandle(fd=999)
        self.assertEqual(cm.exception.code, HIPFILE_INVALID_FILE_DESCRIPTOR)

    def test_repr(self):
        import hipfile
        with hipfile.HipFileHandle(fd=3) as hf:
            self.assertIn("HipFileHandle", repr(hf))


class TestErrorNames(unittest.TestCase):

    def test_known_error(self):
        from hipfile._hipfile import error_name, HIPFILE_PERMISSION_DENIED
        self.assertEqual(error_name(HIPFILE_PERMISSION_DENIED), "HIPFILE_PERMISSION_DENIED")

    def test_unknown_error(self):
        from hipfile._hipfile import error_name
        name = error_name(9999)
        self.assertIn("9999", name)


class TestPublicAPI(unittest.TestCase):
    """Smoke-test that the public __init__ exports are all importable."""

    def test_all_exports_present(self):
        import hipfile
        for name in hipfile.__all__:
            self.assertTrue(hasattr(hipfile, name), f"Missing export: {name}")


if __name__ == "__main__":
    unittest.main()
