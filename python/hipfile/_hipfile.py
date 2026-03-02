"""
Low-level ctypes bindings for libhipfile.so

These bindings mirror the hipFile C API, which is designed as an AMD equivalent
to NVIDIA's cuFile. The API surface is based on the cuFile -> hipFile mapping
maintained at:
https://github.com/ROCm/HIPIFY/blob/amd-develop/docs/reference/tables/cuFile_API_supported_by_HIP.md
"""

import ctypes
import ctypes.util
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _load_library() -> ctypes.CDLL:
    # Search order: env override, then standard ROCm install paths
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
# Enumerations and constants  (mirrors cuFile / hipFile enums)
# ---------------------------------------------------------------------------

# hipFileError_t / CUfileError_t status codes
HIPFILE_SUCCESS                  = 0
HIPFILE_DRIVER_NOT_INITIALIZED   = 1
HIPFILE_DRIVER_INVALID_PROPS     = 2
HIPFILE_DRIVER_UNSUPPORTED_LIMIT = 3
HIPFILE_DRIVER_VERSION_MISMATCH  = 4
HIPFILE_DRIVER_VERSION_READ_ERROR= 5
HIPFILE_DRIVER_CLOSING           = 6
HIPFILE_PLATFORM_NOT_SUPPORTED   = 7
HIPFILE_IO_NOT_SUPPORTED         = 8
HIPFILE_INVALID_VALUE            = 9
HIPFILE_INVALID_FILE_OPEN_FLAG   = 10
HIPFILE_INVALID_FILE_TYPE        = 11
HIPFILE_FILE_NOT_OPEN            = 12
HIPFILE_INVALID_FILE_DESCRIPTOR  = 13
HIPFILE_INVALID_BUFFER_SIZE      = 14
HIPFILE_INVALID_OFFSET           = 15
HIPFILE_INVALID_PTR              = 16
HIPFILE_INVALID_HANDLE           = 17
HIPFILE_PERMISSION_DENIED        = 18
HIPFILE_TIMEOUT                  = 19
HIPFILE_IO_DISABLED              = 20
HIPFILE_OP_NOT_SUPPORTED         = 21
HIPFILE_MEMORY_ALLOCATION_FAILED = 22

_ERROR_NAMES = {v: k for k, v in globals().items() if k.startswith("HIPFILE_")}

def error_name(code: int) -> str:
    return _ERROR_NAMES.get(code, f"HIPFILE_UNKNOWN_ERROR({code})")


# hipFileHandleType_t
HIPFILE_HANDLE_TYPE_OPAQUE_FD     = 0  # file descriptor (int)
HIPFILE_HANDLE_TYPE_OPAQUE_WIN32  = 1  # Windows HANDLE (not used on Linux)

# hipFileDescr_t flags (passed to hipFileRegister)
HIPFILE_OPEN_FLAGS_DEFAULT = 0x0
HIPFILE_OPEN_FLAGS_DIRECT  = 0x1   # hint: use O_DIRECT semantics

# hipFileBatchMode_t
HIPFILE_BATCH_IO_MAX_OPS_PER_BATCH = 128


# ---------------------------------------------------------------------------
# C structs
# ---------------------------------------------------------------------------

class hipFileStatus_t(ctypes.Structure):
    """Maps to CUfileStatus_t / hipFileStatus_t"""
    _fields_ = [
        ("err",  ctypes.c_int),   # hipFileError_t (see constants above)
        ("cu_err", ctypes.c_int), # underlying HIP/CUDA error
    ]


class hipFileDescr_t(ctypes.Structure):
    """File descriptor wrapper passed to hipFileRegister"""
    _fields_ = [
        ("type",   ctypes.c_int),          # hipFileHandleType_t
        ("handle", ctypes.c_int),          # fd (for OPAQUE_FD type)
    ]


class hipFileDriverProps_t(ctypes.Structure):
    """Driver property limits returned by hipFileDriverGetProperties"""
    _fields_ = [
        ("major_version",            ctypes.c_int),
        ("minor_version",            ctypes.c_int),
        ("poll_mode_max_size_kb",    ctypes.c_size_t),
        ("per_buffer_cache_size_kb", ctypes.c_size_t),
        ("max_pinned_memory_size_kb",ctypes.c_size_t),
        ("max_batch_io_size",        ctypes.c_size_t),
        ("max_direct_io_size_kb",    ctypes.c_size_t),
        ("device_ordinal",           ctypes.c_int),
    ]


# Opaque handle returned by hipFileRegister
hipFileHandle_t = ctypes.c_void_p

# Opaque batch handle
hipFileBatchHandle_t = ctypes.c_void_p


# ---------------------------------------------------------------------------
# Function signatures
# ---------------------------------------------------------------------------

def _setup_functions(lib: ctypes.CDLL) -> None:
    """Attach argtypes / restype to each API function."""

    # hipFileDriverOpen() / hipFileDriverClose()
    lib.hipFileDriverOpen.restype  = hipFileStatus_t
    lib.hipFileDriverOpen.argtypes = []

    lib.hipFileDriverClose.restype  = ctypes.c_void_p  # void
    lib.hipFileDriverClose.argtypes = []

    # hipFileDriverGetProperties(hipFileDriverProps_t *props)
    lib.hipFileDriverGetProperties.restype  = hipFileStatus_t
    lib.hipFileDriverGetProperties.argtypes = [ctypes.POINTER(hipFileDriverProps_t)]

    # hipFileDriverSetMaxDirectIOSize(size_t max_size_kb)
    lib.hipFileDriverSetMaxDirectIOSize.restype  = hipFileStatus_t
    lib.hipFileDriverSetMaxDirectIOSize.argtypes = [ctypes.c_size_t]

    # hipFileDriverSetMaxCacheSize(size_t max_size_kb)
    lib.hipFileDriverSetMaxCacheSize.restype  = hipFileStatus_t
    lib.hipFileDriverSetMaxCacheSize.argtypes = [ctypes.c_size_t]

    # hipFileDriverSetMaxPinnedMemSize(size_t max_size_kb)
    lib.hipFileDriverSetMaxPinnedMemSize.restype  = hipFileStatus_t
    lib.hipFileDriverSetMaxPinnedMemSize.argtypes = [ctypes.c_size_t]

    # hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr)
    lib.hipFileHandleRegister.restype  = hipFileStatus_t
    lib.hipFileHandleRegister.argtypes = [
        ctypes.POINTER(hipFileHandle_t),
        ctypes.POINTER(hipFileDescr_t),
    ]

    # hipFileHandleDeregister(hipFileHandle_t fh)
    lib.hipFileHandleDeregister.restype  = ctypes.c_void_p
    lib.hipFileHandleDeregister.argtypes = [hipFileHandle_t]

    # hipFileBufRegister(void *buf, size_t len, int flags)
    lib.hipFileBufRegister.restype  = hipFileStatus_t
    lib.hipFileBufRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    # hipFileBufDeregister(void *buf)
    lib.hipFileBufDeregister.restype  = hipFileStatus_t
    lib.hipFileBufDeregister.argtypes = [ctypes.c_void_p]

    # hipFileRead(hipFileHandle_t fh, void *buf, size_t count,
    #             off_t file_offset, off_t buf_offset) -> ssize_t
    lib.hipFileRead.restype  = ctypes.c_ssize_t
    lib.hipFileRead.argtypes = [
        hipFileHandle_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int64,
        ctypes.c_int64,
    ]

    # hipFileWrite(hipFileHandle_t fh, const void *buf, size_t count,
    #              off_t file_offset, off_t buf_offset) -> ssize_t
    lib.hipFileWrite.restype  = ctypes.c_ssize_t
    lib.hipFileWrite.argtypes = [
        hipFileHandle_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int64,
        ctypes.c_int64,
    ]

    # Batch API
    # hipFileBatchIOSetUp is the unified function for both read and write
    lib.hipFileBatchIOSetUp.restype  = hipFileStatus_t
    lib.hipFileBatchIOSetUp.argtypes = [
        ctypes.POINTER(hipFileBatchHandle_t),
        ctypes.c_uint,  # max_nr
    ]

    lib.hipFileBatchIOSubmit.restype  = hipFileStatus_t
    lib.hipFileBatchIOSubmit.argtypes = [
        hipFileBatchHandle_t,
        ctypes.c_uint,          # nr
        ctypes.c_void_p,        # iocbp (array of hipFileBatchOp_t, opaque here)
        ctypes.c_uint,          # flags
    ]

    lib.hipFileBatchIOGetStatus.restype  = hipFileStatus_t
    lib.hipFileBatchIOGetStatus.argtypes = [
        hipFileBatchHandle_t,
        ctypes.c_uint,     # min_nr
        ctypes.POINTER(ctypes.c_uint),   # nr (in/out)
        ctypes.c_void_p,   # iocbp
        ctypes.c_void_p,   # timeout (struct timespec *)
    ]

    lib.hipFileBatchIODestroy.restype  = ctypes.c_void_p
    lib.hipFileBatchIODestroy.argtypes = [hipFileBatchHandle_t]

    # Read/Write async variants (stream-ordered)
    lib.hipFileReadAsync.restype  = hipFileStatus_t
    lib.hipFileReadAsync.argtypes = [
        hipFileHandle_t,
        ctypes.c_void_p,   # buf (device ptr)
        ctypes.POINTER(ctypes.c_size_t),  # size (in/out)
        ctypes.POINTER(ctypes.c_int64),   # file_offset (in/out)
        ctypes.POINTER(ctypes.c_int64),   # buf_offset (in/out)
        ctypes.c_void_p,                  # hipStream_t
    ]

    lib.hipFileWriteAsync.restype  = hipFileStatus_t
    lib.hipFileWriteAsync.argtypes = [
        hipFileHandle_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
