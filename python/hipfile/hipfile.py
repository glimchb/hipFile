"""
hipfile.py – High-level Pythonic interface to AMD hipFile (GPU-direct storage).

Typical usage::

    import hipfile

    hipfile.driver_open()

    with open("data.bin", "r+b") as f:
        with hipfile.HipFileHandle(f.fileno()) as hf:
            # device_ptr is a raw int GPU pointer (e.g. from hip.malloc or torch)
            n = hf.read(device_ptr, count=4096, file_offset=0)

    hipfile.driver_close()

Or using the context-manager form for the driver itself::

    with hipfile.Driver():
        with open("data.bin", "r+b") as f:
            with hipfile.HipFileHandle(f.fileno()) as hf:
                hf.write(device_ptr, count=4096, file_offset=0)
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

from ._hipfile import (
    _load_library,
    _setup_functions,
    hipFileDescr_t,
    hipFileDriverProps_t,
    hipFileHandle_t,
    hipFileStatus_t,
    HIPFILE_SUCCESS,
    HIPFILE_HANDLE_TYPE_OPAQUE_FD,
    HIPFILE_OPEN_FLAGS_DEFAULT,
    error_name,
)

# ---------------------------------------------------------------------------
# Module-level library handle (lazy initialisation)
# ---------------------------------------------------------------------------

_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_library()
        _setup_functions(_lib)
    return _lib


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HipFileError(Exception):
    """Raised when a hipFile API call returns a non-success status."""

    def __init__(self, code: int, hip_err: int = 0, msg: str = ""):
        self.code    = code
        self.hip_err = hip_err
        name = error_name(code)
        detail = f" – {msg}" if msg else ""
        super().__init__(f"hipFile error {name} ({code}), HIP error {hip_err}{detail}")


def _check(status: hipFileStatus_t, msg: str = "") -> None:
    if status.err != HIPFILE_SUCCESS:
        raise HipFileError(status.err, status.cu_err, msg)


def _check_io(retval, op: str) -> int:
    """Check the ssize_t return of hipFileRead / hipFileWrite."""
    # ctypes may return a c_long/c_ssize_t wrapper; normalise to int
    val = int(retval)
    if val < 0:
        raise HipFileError(int(-val), msg=f"{op} returned {val}")
    return val


# ---------------------------------------------------------------------------
# Driver lifecycle
# ---------------------------------------------------------------------------

def driver_open() -> None:
    """Initialise the hipFile driver.  Must be called before any other API."""
    status = _get_lib().hipFileDriverOpen()
    _check(status, "hipFileDriverOpen")


def driver_close() -> None:
    """Tear down the hipFile driver."""
    _get_lib().hipFileDriverClose()


def driver_get_properties() -> hipFileDriverProps_t:
    """Return a :class:`hipFileDriverProps_t` struct with driver limits."""
    props = hipFileDriverProps_t()
    status = _get_lib().hipFileDriverGetProperties(ctypes.byref(props))
    _check(status, "hipFileDriverGetProperties")
    return props


def driver_set_max_direct_io_size(max_size_kb: int) -> None:
    status = _get_lib().hipFileDriverSetMaxDirectIOSize(max_size_kb)
    _check(status, "hipFileDriverSetMaxDirectIOSize")


def driver_set_max_cache_size(max_size_kb: int) -> None:
    status = _get_lib().hipFileDriverSetMaxCacheSize(max_size_kb)
    _check(status, "hipFileDriverSetMaxCacheSize")


def driver_set_max_pinned_mem_size(max_size_kb: int) -> None:
    status = _get_lib().hipFileDriverSetMaxPinnedMemSize(max_size_kb)
    _check(status, "hipFileDriverSetMaxPinnedMemSize")


class Driver:
    """Context manager that calls :func:`driver_open` / :func:`driver_close`."""

    def __enter__(self) -> "Driver":
        driver_open()
        return self

    def __exit__(self, *_) -> None:
        driver_close()


# ---------------------------------------------------------------------------
# Buffer registration
# ---------------------------------------------------------------------------

def buf_register(device_ptr: int, length: int, flags: int = 0) -> None:
    """Pin a GPU buffer for use with hipFile I/O.

    Parameters
    ----------
    device_ptr:
        Raw integer GPU device pointer (e.g. ``tensor.data_ptr()``).
    length:
        Size in bytes of the buffer to pin.
    flags:
        Reserved; pass 0.
    """
    status = _get_lib().hipFileBufRegister(
        ctypes.c_void_p(device_ptr), ctypes.c_size_t(length), ctypes.c_int(flags)
    )
    _check(status, "hipFileBufRegister")


def buf_deregister(device_ptr: int) -> None:
    """Unpin a previously registered GPU buffer."""
    status = _get_lib().hipFileBufDeregister(ctypes.c_void_p(device_ptr))
    _check(status, "hipFileBufDeregister")


class RegisteredBuffer:
    """Context manager that registers / deregisters a GPU buffer."""

    def __init__(self, device_ptr: int, length: int, flags: int = 0) -> None:
        self.device_ptr = device_ptr
        self.length     = length
        self.flags      = flags

    def __enter__(self) -> "RegisteredBuffer":
        buf_register(self.device_ptr, self.length, self.flags)
        return self

    def __exit__(self, *_) -> None:
        try:
            buf_deregister(self.device_ptr)
        except HipFileError:
            pass  # best-effort on cleanup


# ---------------------------------------------------------------------------
# File handle
# ---------------------------------------------------------------------------

class HipFileHandle:
    """Wraps a registered file descriptor for GPU-direct I/O.

    Parameters
    ----------
    fd:
        An open file descriptor (``int``).  The file should typically be
        opened with ``os.O_RDONLY``, ``os.O_WRONLY``, or ``os.O_RDWR``.
        For best performance use ``os.O_DIRECT`` as well.
    flags:
        ``HIPFILE_OPEN_FLAGS_DEFAULT`` (0) or ``HIPFILE_OPEN_FLAGS_DIRECT``.

    Example
    -------
    ::

        fd = os.open("big_tensor.bin", os.O_RDONLY | os.O_DIRECT)
        with HipFileHandle(fd) as hf:
            hf.read(device_ptr, count=len_bytes, file_offset=0)
        os.close(fd)
    """

    def __init__(self, fd: int, flags: int = HIPFILE_OPEN_FLAGS_DEFAULT) -> None:
        self._handle = None
        descr         = hipFileDescr_t()
        descr.type    = HIPFILE_HANDLE_TYPE_OPAQUE_FD
        descr.handle  = fd

        _h = hipFileHandle_t(None)
        status = _get_lib().hipFileHandleRegister(
            ctypes.byref(_h), ctypes.byref(descr)
        )
        _check(status, f"hipFileHandleRegister(fd={fd})")
        self._handle = _h

    def deregister(self) -> None:
        """Explicitly deregister the file handle."""
        if self._handle is not None:
            _get_lib().hipFileHandleDeregister(self._handle)
            self._handle = None

    def read(
        self,
        device_ptr: int,
        count: int,
        file_offset: int = 0,
        buf_offset:  int = 0,
    ) -> int:
        """Read *count* bytes from the file directly into GPU memory.

        Parameters
        ----------
        device_ptr:
            Raw integer GPU pointer to the destination buffer.
        count:
            Number of bytes to read.
        file_offset:
            Byte offset within the file.
        buf_offset:
            Byte offset within the device buffer (usually 0).

        Returns
        -------
        int
            Number of bytes actually read.
        """
        ret = _get_lib().hipFileRead(
            self._handle,
            ctypes.c_void_p(device_ptr),
            ctypes.c_size_t(count),
            ctypes.c_int64(file_offset),
            ctypes.c_int64(buf_offset),
        )
        return _check_io(ret, "hipFileRead")

    def write(
        self,
        device_ptr: int,
        count: int,
        file_offset: int = 0,
        buf_offset:  int = 0,
    ) -> int:
        """Write *count* bytes from GPU memory directly to the file.

        Parameters
        ----------
        device_ptr:
            Raw integer GPU pointer to the source buffer.
        count:
            Number of bytes to write.
        file_offset:
            Byte offset within the file.
        buf_offset:
            Byte offset within the device buffer (usually 0).

        Returns
        -------
        int
            Number of bytes actually written.
        """
        ret = _get_lib().hipFileWrite(
            self._handle,
            ctypes.c_void_p(device_ptr),
            ctypes.c_size_t(count),
            ctypes.c_int64(file_offset),
            ctypes.c_int64(buf_offset),
        )
        return _check_io(ret, "hipFileWrite")

    def read_async(
        self,
        device_ptr: int,
        count: int,
        file_offset: int,
        stream,   # hipStream_t (int / ctypes.c_void_p)
        buf_offset: int = 0,
    ) -> None:
        """Stream-ordered async read (requires hipFile async support)."""
        c_count      = ctypes.c_size_t(count)
        c_file_off   = ctypes.c_int64(file_offset)
        c_buf_off    = ctypes.c_int64(buf_offset)
        stream_ptr   = ctypes.c_void_p(int(stream)) if stream else ctypes.c_void_p(None)

        status = _get_lib().hipFileReadAsync(
            self._handle,
            ctypes.c_void_p(device_ptr),
            ctypes.byref(c_count),
            ctypes.byref(c_file_off),
            ctypes.byref(c_buf_off),
            stream_ptr,
        )
        _check(status, "hipFileReadAsync")

    def write_async(
        self,
        device_ptr: int,
        count: int,
        file_offset: int,
        stream,
        buf_offset: int = 0,
    ) -> None:
        """Stream-ordered async write."""
        c_count    = ctypes.c_size_t(count)
        c_file_off = ctypes.c_int64(file_offset)
        c_buf_off  = ctypes.c_int64(buf_offset)
        stream_ptr = ctypes.c_void_p(int(stream)) if stream else ctypes.c_void_p(None)

        status = _get_lib().hipFileWriteAsync(
            self._handle,
            ctypes.c_void_p(device_ptr),
            ctypes.byref(c_count),
            ctypes.byref(c_file_off),
            ctypes.byref(c_buf_off),
            stream_ptr,
        )
        _check(status, "hipFileWriteAsync")

    # Context-manager support
    def __enter__(self) -> "HipFileHandle":
        return self

    def __exit__(self, *_) -> None:
        self.deregister()

    def __repr__(self) -> str:
        return f"<HipFileHandle handle={self._handle}>"
