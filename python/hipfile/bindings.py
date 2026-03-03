"""
Low-level ctypes bindings for libhipfile.so

Structured as a direct mirror of cufile/bindings.py so that the two modules
are interchangeable. Variable and function names are prefixed hip/Hip instead
of cu/Cu, but signatures, field layouts, and calling conventions are identical.

Reference: /usr/local/lib/python3.12/dist-packages/cufile/bindings.py
"""

import ctypes
import os

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_library() -> ctypes.CDLL:
    search_paths = [
        os.environ.get("HIPFILE_LIB_PATH", ""),
        "/opt/rocm/lib/libhipfile.so",
        "/opt/rocm/lib64/libhipfile.so",
        "/usr/lib/libhipfile.so",
        "/usr/local/lib/libhipfile.so",
        "libhipfile.so",
    ]
    for path in search_paths:
        if not path:
            continue
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    raise OSError(
        "Could not find libhipfile.so. Make sure hipFile is installed "
        "(see https://github.com/ROCm/hipFile) and its library is on "
        "LD_LIBRARY_PATH, or set the HIPFILE_LIB_PATH environment variable."
    )


# ---------------------------------------------------------------------------
# Structs  (mirror cufile exactly, renamed hip*)
# ---------------------------------------------------------------------------

class hipFileError(ctypes.Structure):
    """Maps to CUfileError / hipFileStatus_t."""
    _fields_ = [
        ("err",    ctypes.c_int),   # hipFile error code
        ("cu_err", ctypes.c_int),   # underlying HIP runtime error
    ]


hipFileHandle_t = ctypes.c_void_p   # opaque handle (mirrors CUfileHandle_t)


class DescrUnion(ctypes.Union):
    """Union holding either a file descriptor or an opaque handle."""
    _fields_ = [
        ("fd",     ctypes.c_int),
        ("handle", ctypes.c_void_p),
    ]


class hipFileDescr(ctypes.Structure):
    """File descriptor passed to hipFileHandleRegister (mirrors CUfileDescr)."""
    _fields_ = [
        ("type",   ctypes.c_int),       # handle type: 1 = opaque fd
        ("handle", DescrUnion),
        ("fs_ops", ctypes.c_void_p),    # reserved / filesystem ops ptr
    ]


# ---------------------------------------------------------------------------
# Function signatures
# ---------------------------------------------------------------------------

def _setup_functions(lib: ctypes.CDLL) -> None:
    """Attach argtypes / restype — mirrors cufile/bindings.py layout."""

    lib.hipFileDriverOpen.restype  = hipFileError
    lib.hipFileDriverClose.restype = hipFileError

    lib.hipFileHandleRegister.restype  = hipFileError
    lib.hipFileHandleRegister.argtypes = [
        ctypes.POINTER(hipFileHandle_t),
        ctypes.POINTER(hipFileDescr),
    ]

    lib.hipFileHandleDeregister.argtypes = [hipFileHandle_t]

    lib.hipFileBufRegister.restype  = hipFileError
    lib.hipFileBufRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    lib.hipFileBufDeregister.restype  = hipFileError
    lib.hipFileBufDeregister.argtypes = [ctypes.c_void_p]

    # Read/Write return c_size_t (mirrors cuFileRead / cuFileWrite)
    lib.hipFileRead.restype  = ctypes.c_size_t
    lib.hipFileRead.argtypes = [
        hipFileHandle_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_longlong,   # file_offset
        ctypes.c_longlong,   # dev_offset
    ]

    lib.hipFileWrite.restype  = ctypes.c_size_t
    lib.hipFileWrite.argtypes = [
        hipFileHandle_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]


# ---------------------------------------------------------------------------
# Convenience functions  (mirror cufile/bindings.py function for function)
# ---------------------------------------------------------------------------

_lib: ctypes.CDLL = None   # injected by hipfile.py after load


def _ck(status: hipFileError, name: str) -> None:
    if status.err != 0:
        raise RuntimeError(
            f"{name} failed (hipFile err={status.err}, hip_err={status.cu_err})"
        )


def hipFileDriverOpen() -> None:
    _ck(_lib.hipFileDriverOpen(), "hipFileDriverOpen")


def hipFileDriverClose() -> None:
    _ck(_lib.hipFileDriverClose(), "hipFileDriverClose")


def hipFileHandleRegister(fd: int) -> hipFileHandle_t:
    descr        = hipFileDescr()
    descr.type   = 1                  # opaque fd  (matches cufile type=1)
    descr.handle = DescrUnion(fd=fd)
    handle       = hipFileHandle_t()
    _ck(
        _lib.hipFileHandleRegister(
            ctypes.byref(handle), ctypes.byref(descr)
        ),
        "hipFileHandleRegister",
    )
    return handle


def hipFileHandleDeregister(handle: hipFileHandle_t) -> None:
    _lib.hipFileHandleDeregister(handle)


def hipFileBufRegister(buf: ctypes.c_void_p, size: int, flags: int) -> None:
    _ck(_lib.hipFileBufRegister(buf, size, flags), "hipFileBufRegister")


def hipFileBufDeregister(buf: ctypes.c_void_p) -> None:
    _ck(_lib.hipFileBufDeregister(buf), "hipFileBufDeregister")


def hipFileRead(
    handle: hipFileHandle_t,
    buf: ctypes.c_void_p,
    size: int,
    file_offset: int,
    dev_offset: int,
) -> int:
    return _lib.hipFileRead(handle, buf, size, file_offset, dev_offset)


def hipFileWrite(
    handle: hipFileHandle_t,
    buf: ctypes.c_void_p,
    size: int,
    file_offset: int,
    dev_offset: int,
) -> int:
    return _lib.hipFileWrite(handle, buf, size, file_offset, dev_offset)
