# GPU Development

Runtime selection of libCEED backends allows users to use CPU backends for easier debugging.
Code that produces correct results with CPU backends will produce correct results on GPU backends, provided that JiT and memory access assumptions of the libCEED API are respected.

## JiT Compilation

The filepath to the user source code is passed in {c:func}`CeedQFunctionCreateInterior` as the `source` argument.
This filepath should typically be an absolute path to ensure the JiT compilation can locate the source file.
The filepath may also be relative to a root directory set with {c:func}`CeedAddJitSourceRoot`.

The entire contents of this file and all locally included files (`#include "foo.h"`) are used during JiT compilation for GPU backends.
Installed headers (`#include <bar.h>`) are omitted in the source code passed to JiT, but the compilation environment may supply common headers such as `<math.h>`.
These source file must only contain syntax constructs supported by C99 and all targeted backends (i.e. CUDA for `/gpu/cuda`, OpenCL/SYCL for `/gpu/sycl`, etc.).

All source files must be at the provided filepath at runtime for JiT to function.

## Memory Access

GPU backends require stricter adherence to memory access assumptions, but CPU backends may occasionally report correct results despite violations of memory access assumptions.
The `/cpu/self/memcheck` backends explicitly verify read-only and write-only memory access assumptions.

