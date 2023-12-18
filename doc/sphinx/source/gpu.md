# GPU Development

Runtime selection of libCEED backends allows users to use CPU backends for easier debugging.
Code that produces correct results with CPU backends will produce correct results on GPU backends, provided that JiT and memory access assumptions of the libCEED API are respected.

## JiT Compilation

The filepath to the user source code is passed in {c:func}`CeedQFunctionCreateInterior` as the `source` argument.
This filepath should typically be an absolute path to ensure the JiT compilation can locate the source file.
The filepath may also be relative to a root directory set with {c:func}`CeedAddJitSourceRoot`.
The {c:macro}`CEED_QFUNCTION` macro automatically creates a string with the absolute path stored in the variable `user_loc` for a {c:type}`CeedQFunctionUser` called `user`.

The entire contents of this file and all locally included files (`#include "foo.h"`) are used during JiT compilation for GPU backends.
Installed headers (`#include <bar.h>`) are omitted in the source code passed to JiT, but the compilation environment may supply common headers such as `<math.h>`.
These source file must only contain syntax constructs supported by C99 and all targeted backends (i.e. CUDA for `/gpu/cuda`, OpenCL/SYCL for `/gpu/sycl`, etc.).

All source files must be at the provided filepath at runtime for JiT to function.

## Memory Access

GPU backends require stricter adherence to memory access assumptions, but CPU backends may occasionally report correct results despite violations of memory access assumptions.
Both `CeedVector` and `CeedQFunctionContext` have read-only and read-write accessors, and `CeedVector` allow write-only access.
Read-only access of `CeedVector` and `CeedQFunctionContext` memory spaces must be respected for proper GPU behavior.
Write-only access of `CeedVector` memory spaces asserts that all data in the `CeedVector` is invalid until overwritten.

`CeedQFunction` assume that all input arrays are read-only and all output arrays are write-only and the {c:type}`CeedQFunctionUser` must adhere to these assumptions, only reading data in the input arrays and fully overwriting the output arrays.
Additionally, {c:type}`CeedQFunctionUser` have read-write access for `CeedQFunctionContext` data, unless {c:func}`CeedQFunctionSetContextWritable` was used to indicate that read-only access is sufficient.

The `/cpu/self/memcheck` backends explicitly verify read-only and write-only memory access assumptions.

