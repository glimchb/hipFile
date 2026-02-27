"""
hipfile – Python bindings for AMD hipFile (GPU-direct storage).

AMD hipFile is the ROCm equivalent of NVIDIA cuFile, enabling data movement
directly between NVMe/filesystem storage and GPU memory without CPU staging.

Project: https://github.com/ROCm/hipFile
Bindings issue: https://github.com/ROCm/hipFile/issues/201

Quick start::

    import hipfile

    with hipfile.Driver():
        fd = os.open("data.bin", os.O_RDONLY | os.O_DIRECT)
        try:
            with hipfile.HipFileHandle(fd) as hf:
                with hipfile.RegisteredBuffer(gpu_ptr, size):
                    bytes_read = hf.read(gpu_ptr, count=size, file_offset=0)
        finally:
            os.close(fd)
"""

from .hipfile import (
    # Driver lifecycle
    Driver,
    driver_open,
    driver_close,
    driver_get_properties,
    driver_set_max_direct_io_size,
    driver_set_max_cache_size,
    driver_set_max_pinned_mem_size,

    # Buffer registration
    RegisteredBuffer,
    buf_register,
    buf_deregister,

    # File handle
    HipFileHandle,

    # Exceptions
    HipFileError,
)

from ._hipfile import (
    # Enumerations / constants
    HIPFILE_SUCCESS,
    HIPFILE_INVALID_VALUE,
    HIPFILE_INVALID_FILE_DESCRIPTOR,
    HIPFILE_PERMISSION_DENIED,
    HIPFILE_IO_NOT_SUPPORTED,
    HIPFILE_PLATFORM_NOT_SUPPORTED,
    HIPFILE_OPEN_FLAGS_DEFAULT,
    HIPFILE_OPEN_FLAGS_DIRECT,
    HIPFILE_HANDLE_TYPE_OPAQUE_FD,

    # Structs (useful for introspection)
    hipFileDriverProps_t,

    error_name,
)

__version__ = "0.1.0"
__all__ = [
    "Driver",
    "driver_open",
    "driver_close",
    "driver_get_properties",
    "driver_set_max_direct_io_size",
    "driver_set_max_cache_size",
    "driver_set_max_pinned_mem_size",
    "RegisteredBuffer",
    "buf_register",
    "buf_deregister",
    "HipFileHandle",
    "HipFileError",
    "hipFileDriverProps_t",
    "error_name",
    "HIPFILE_SUCCESS",
    "HIPFILE_OPEN_FLAGS_DEFAULT",
    "HIPFILE_OPEN_FLAGS_DIRECT",
    "HIPFILE_HANDLE_TYPE_OPAQUE_FD",
]
