# GPU Development

Runtime selection of libCEED backends allows users to use CPU backends for easier debugging.
Code that produces correct results with CPU backends will produce correct results on GPU backends, provided that JiT and memory access assumptions of the libCEED API are respected.

## JiT Compilation

The filepath to the user source code is passed in {c:func}`CeedQFunctionCreateInterior` as the `source` argument.
This filepath should typically be an absolute path to ensure the JiT compilation can locate the source file.
The filepath may also be a relative path with respect to a root directory set with {c:func}`CeedAddJitSourceRoot`.
The {c:macro}`CEED_QFUNCTION` macro automatically creates a string with the absolute path, for example a {c:type}`CeedQFunctionUser` called `user` would have this string stored in the variable `user_loc`.

The entire contents of this source file and all included files are used during JiT compilation for GPU backends.
Include statements for system headers that are required for CPU compilation but are not available in GPU compilation environments should be guarded with `#ifdef CEED_RUNNING_JIT_PASS`.
Any function definitions in these system headers must still be available in the GPU compilation environments, such as the contents of `<math.h>`.
These source file must only contain syntax constructs supported by C99 and all targeted backends (i.e. CUDA for `/gpu/cuda`, OpenCL/SYCL for `/gpu/sycl`, etc.).

All source files must be at the provided filepath at runtime for JiT to function.

The CUDA and HIP backends reuse the binary of an identical earlier NVRTC or hipRTC JiT compilation instead of compiling it again.
Within a process, compilations with the same compiler options and source are treated as identical: included files are assumed not to change while a program runs.
Binaries are also kept for later runs in the `libCEED` directory of the cache base directory (`$CEED_CACHE_BASE_DIR`, `$XDG_CACHE_HOME`, or `$HOME/.cache`, see `CeedSetCacheBaseDir`), keyed in addition by the compiler and by the path and contents of every file that the source includes, recursively, so changes to source files between runs are detected.
Sources with include directives that may not be followed exactly, such as an `#include` of a macro, are only cached in memory.
The cache directory must belong to the user and not be writable by the group or others, otherwise, or if it cannot be created, only the in-memory cache is used; if it is read-only, no binaries are added to it.
For large parallel runs, a node-local or scratch cache base directory private to the user avoids many processes accessing a shared home directory.
Binaries on disk are never evicted: deleting the `libCEED` directory of the cache base directory clears them, with the other files that libCEED keeps there and any temporary files (`.tmp_*`) left by interrupted runs.

## Memory Access

GPU backends require stricter adherence to memory access assumptions, but CPU backends may occasionally report correct results despite violations of memory access assumptions.
Both `CeedVector` and `CeedQFunctionContext` have read-only and read-write accessors, and `CeedVector` allow write-only access.
Read-only access of `CeedVector` and `CeedQFunctionContext` memory spaces must be respected for to ensure proper GPU behavior.
Write-only access of `CeedVector` memory spaces asserts that all data in the `CeedVector` is invalid until overwritten.

`CeedQFunction` assume that all input arrays are read-only and all output arrays are write-only and the {c:type}`CeedQFunctionUser` must adhere to these assumptions, only reading data in the input arrays and fully overwriting all entries in the output arrays.
Additionally, {c:type}`CeedQFunctionUser` have read-write access for `CeedQFunctionContext` data, unless {c:func}`CeedQFunctionSetContextWritable` was used to indicate that read-only access is sufficient.

The `/cpu/self/memcheck` backends explicitly verify read-only and write-only memory access assumptions.

