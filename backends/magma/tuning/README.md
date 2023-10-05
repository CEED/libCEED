# MAGMA Backend Autotuning (Non-tensor Basis)

The `magma` backend uses specialized GPU kernels for a non-tensor basis with
`P`, `Q` less than a prescribed value, and above this cutoff uses a standard
library GEMM implementation. The specialized kernels have a single tunable
blocking factor parameter, `NB`, which varies with `P` and `Q` as well as the
size of the number of elements `N`. This folder contains the tuning data, in
header files called `<ARCH>_rtc.h`, where `<ARCH>` is the GPU name, as well as a
simple C++ program (`tuning.cpp`) and Python driver (`generate_tuning.py`) to
generate the optimal `NB` selections for a new target architecture.

## Generating Autotuning Data

A sample run to generate the tuning data for an A100 GPU, considering values of
`NB` from 1 to 32 and saved to `a100_rtc.h`, is:

```sh
python generate_tuning.py -arch a100 -max-nb 32 -build-cmd "make" -ceed "/gpu/cuda/magma"
```

The `-build-cmd` parameter specifies the command which should be used to compile
the libCEED library. For example, this may be a build script which calls `make`
internally with the desired parameters, or might just be `make` if a previous
call to `make configure` has configured the build. Finally, the `-ceed`
specifies the backend to use, typically one of `/gpu/cuda/magma` or
`/gpu/hip/magma`.

Alternatively, the `tuning` program can be built and run on its own to benchmark
the basis application for a given backend. Run `make tuning` from this directory
and call the program as:

```sh
./tuning "/gpu/cuda/magma"
````

Note that in order for the benchmarks to make sense for `magma` backends, the
`ceed_magma_queue_sync` in `ceed-magma.h` should be set to
`cudaDeviceSynchronize()` or `hipDeviceSynchronize()`.
