# hipfile – Python bindings for AMD hipFile

Python `ctypes`-based bindings for [AMD hipFile](https://github.com/ROCm/hipFile),
the ROCm equivalent of NVIDIA's cuFile, enabling **GPU-direct storage** – data
movement directly between NVMe/filesystem storage and GPU memory, bypassing CPU
staging buffers.

> **Status:** Early-stage community bindings, tracking
> [ROCm/hipFile#201](https://github.com/ROCm/hipFile/issues/201).

---

## Requirements

- Linux (x86_64 or aarch64)
- ROCm installed (tested with ROCm 6.x)
- hipFile library built and installed from [ROCm/hipFile](https://github.com/ROCm/hipFile)
- Python 3.8+

---

## Installation

```bash
# From source
git clone https://github.com/your-fork/hipfile-python
cd hipfile-python
pip install -e .
```

Make sure `libhipfile.so` is on your `LD_LIBRARY_PATH`, or set:

```bash
export HIPFILE_LIB_PATH=/opt/rocm/lib/libhipfile.so
```

---

## Quick start

```python
import os
import hipfile

# --- Open the driver ---
with hipfile.Driver():

    # Open a file with O_DIRECT for best performance
    fd = os.open("data.bin", os.O_RDONLY | os.O_DIRECT)
    try:
        byte_size = 4 * 1024 * 1024  # 4 MB

        # Allocate GPU memory (example using PyTorch)
        import torch
        tensor = torch.empty(1024 * 1024, dtype=torch.float32, device="cuda")
        gpu_ptr = tensor.data_ptr()

        # Register the GPU buffer, then do the I/O
        with hipfile.RegisteredBuffer(gpu_ptr, byte_size):
            with hipfile.HipFileHandle(fd) as hf:
                n = hf.read(gpu_ptr, count=byte_size, file_offset=0)
                print(f"Read {n} bytes directly into GPU memory")

    finally:
        os.close(fd)
```

---

## API overview

### Driver lifecycle

```python
hipfile.driver_open()            # initialise the hipFile driver
hipfile.driver_close()           # tear down

props = hipfile.driver_get_properties()  # returns hipFileDriverProps_t
print(props.major_version, props.minor_version)

hipfile.driver_set_max_direct_io_size(128)   # KB
hipfile.driver_set_max_cache_size(512)       # KB
hipfile.driver_set_max_pinned_mem_size(256)  # KB
```

### Context managers

```python
with hipfile.Driver():                          # open / close driver
    with hipfile.RegisteredBuffer(ptr, size):  # pin / unpin GPU buffer
        with hipfile.HipFileHandle(fd) as hf:  # register / deregister fd
            hf.read(ptr, count=size, file_offset=0)
            hf.write(ptr, count=size, file_offset=0)
```

### Async (stream-ordered) I/O

```python
with hipfile.HipFileHandle(fd) as hf:
    hf.read_async(ptr, count=size, file_offset=0, stream=hip_stream)
    hf.write_async(ptr, count=size, file_offset=0, stream=hip_stream)
    # synchronise the stream before using the data
```

### Error handling

```python
try:
    hipfile.driver_open()
except hipfile.HipFileError as e:
    print(f"Error code: {e.code}, HIP error: {e.hip_err}")
    print(hipfile.error_name(e.code))
```

---

## Running the tests

The test suite uses mocks and runs without real hardware:

```bash
pip install pytest
pytest tests/ -v
```

---

## PyTorch example

```bash
python examples/pytorch_example.py --create --count 1048576
```

---

## How it works

hipFile provides a C API for GPU-direct I/O on AMD ROCm hardware. These Python
bindings use `ctypes` to call `libhipfile.so` directly, with no C compilation
needed. The binding layer:

1. Loads `libhipfile.so` at import time (lazy, configurable via `HIPFILE_LIB_PATH`).
2. Declares `argtypes` / `restype` for each API function.
3. Wraps the C types in Pythonic classes with context-manager support.
4. Translates error status codes to `HipFileError` exceptions.

---

## Contributing

PRs welcome! The main tracking issue for official bindings is
[ROCm/hipFile#201](https://github.com/ROCm/hipFile/issues/201).
