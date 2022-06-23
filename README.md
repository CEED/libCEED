# libCEED: Efficient Extensible Discretization

[![GitHub Actions][github-badge]][github-link]
[![GitLab-CI][gitlab-badge]][gitlab-link]
[![Code coverage][codecov-badge]][codecov-link]
[![BSD-2-Clause][license-badge]][license-link]
[![Documentation][doc-badge]][doc-link]
[![JOSS paper][joss-badge]][joss-link]
[![Binder][binder-badge]][binder-link]

## Summary and Purpose

libCEED provides fast algebra for element-based discretizations, designed for performance portability, run-time flexibility, and clean embedding in higher level libraries and applications.
It offers a C99 interface as well as bindings for Fortran, Python, Julia, and Rust.
While our focus is on high-order finite elements, the approach is mostly algebraic and thus applicable to other discretizations in factored form, as explained in the [user manual](https://libceed.org/en/latest/) and API implementation portion of the [documentation](https://libceed.org/en/latest/api/).

One of the challenges with high-order methods is that a global sparse matrix is no longer a good representation of a high-order linear operator, both with respect to the FLOPs needed for its evaluation, as well as the memory transfer needed for a matvec.
Thus, high-order methods require a new "format" that still represents a linear (or more generally non-linear) operator, but not through a sparse matrix.

The goal of libCEED is to propose such a format, as well as supporting implementations and data structures, that enable efficient operator evaluation on a variety of computational device types (CPUs, GPUs, etc.).
This new operator description is based on algebraically [factored form](https://libceed.org/en/latest/libCEEDapi/#finite-element-operator-decomposition), which is easy to incorporate in a wide variety of applications, without significant refactoring of their own discretization infrastructure.

The repository is part of the [CEED software suite](http://ceed.exascaleproject.org/software/), a collection of software benchmarks, miniapps, libraries and APIs for efficient exascale discretizations based on high-order finite element and spectral element methods.
See <http://github.com/ceed> for more information and source code availability.

The CEED research is supported by the [Exascale Computing Project](https://exascaleproject.org/exascale-computing-project) (17-SC-20-SC), a collaborative effort of two U.S. Department of Energy organizations (Office of Science and the National Nuclear Security Administration) responsible for the planning and preparation of a [capable exascale ecosystem](https://exascaleproject.org/what-is-exascale), including software, applications, hardware, advanced system engineering and early testbed platforms, in support of the nation’s exascale computing imperative.

For more details on the CEED API see the [user manual](https://libceed.org/en/latest/).

% gettingstarted-inclusion-marker

## Building

The CEED library, `libceed`, is a C99 library with no required dependencies, and with Fortran, Python, Julia, and Rust interfaces.
It can be built using:

```
make
```

or, with optimization flags:

```
make OPT='-O3 -march=skylake-avx512 -ffp-contract=fast'
```

These optimization flags are used by all languages (C, C++, Fortran) and this makefile variable can also be set for testing and examples (below).

The library attempts to automatically detect support for the AVX instruction set using gcc-style compiler options for the host.
Support may need to be manually specified via:

```
make AVX=1
```

or:

```
make AVX=0
```

if your compiler does not support gcc-style options, if you are cross compiling, etc.

To enable CUDA support, add `CUDA_DIR=/opt/cuda` or an appropriate directory to your `make` invocation.
To enable HIP support, add `HIP_DIR=/opt/rocm` or an appropriate directory.
To store these or other arguments as defaults for future invocations of `make`, use:

```
make configure CUDA_DIR=/usr/local/cuda HIP_DIR=/opt/rocm OPT='-O3 -march=znver2'
```

which stores these variables in `config.mk`.

## Additional Language Interfaces

The Fortran interface is built alongside the library automatically.

Python users can install using:

```
pip install libceed
```

or in a clone of the repository via `pip install .`.

Julia users can install using:

```
$ julia
julia> ]
pkg> add LibCEED
```

See the [LibCEED.jl documentation](http://ceed.exascaleproject.org/libCEED-julia-docs/dev/) for more information.

Rust users can include libCEED via `Cargo.toml`:

```toml
[dependencies]
libceed = { git = "https://github.com/CEED/libCEED", branch = "main" }
```

See the [Cargo documentation](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#specifying-dependencies-from-git-repositories) for details.

## Testing

The test suite produces [TAP](https://testanything.org) output and is run by:

```
make test
```

or, using the `prove` tool distributed with Perl (recommended):

```
make prove
```

## Backends

There are multiple supported backends, which can be selected at runtime in the examples:

| CEED resource              | Backend                                           | Deterministic Capable |
| :---                       | :---                                              | :---:                 |
||
| **CPU Native**             |
| `/cpu/self/ref/serial`     | Serial reference implementation                   | Yes                   |
| `/cpu/self/ref/blocked`    | Blocked reference implementation                  | Yes                   |
| `/cpu/self/opt/serial`     | Serial optimized C implementation                 | Yes                   |
| `/cpu/self/opt/blocked`    | Blocked optimized C implementation                | Yes                   |
| `/cpu/self/avx/serial`     | Serial AVX implementation                         | Yes                   |
| `/cpu/self/avx/blocked`    | Blocked AVX implementation                        | Yes                   |
||
| **CPU Valgrind**           |
| `/cpu/self/memcheck/*`     | Memcheck backends, undefined value checks         | Yes                   |
||
| **CPU LIBXSMM**            |
| `/cpu/self/xsmm/serial`    | Serial LIBXSMM implementation                     | Yes                   |
| `/cpu/self/xsmm/blocked`   | Blocked LIBXSMM implementation                    | Yes                   |
||
| **CUDA Native**            |
| `/gpu/cuda/ref`            | Reference pure CUDA kernels                       | Yes                   |
| `/gpu/cuda/shared`         | Optimized pure CUDA kernels using shared memory   | Yes                   |
| `/gpu/cuda/gen`            | Optimized pure CUDA kernels using code generation | No                    |
||
| **HIP Native**             |
| `/gpu/hip/ref`             | Reference pure HIP kernels                        | Yes                   |
| `/gpu/hip/shared`          | Optimized pure HIP kernels using shared memory    | Yes                   |
| `/gpu/hip/gen`             | Optimized pure HIP kernels using code generation  | No                    |
||
| **MAGMA**                  |
| `/gpu/cuda/magma`          | CUDA MAGMA kernels                                | No                    |
| `/gpu/cuda/magma/det`      | CUDA MAGMA kernels                                | Yes                   |
| `/gpu/hip/magma`           | HIP MAGMA kernels                                 | No                    |
| `/gpu/hip/magma/det`       | HIP MAGMA kernels                                 | Yes                   |

The `/cpu/self/*/serial` backends process one element at a time and are intended for meshes with a smaller number of high order elements.
The `/cpu/self/*/blocked` backends process blocked batches of eight interlaced elements and are intended for meshes with higher numbers of elements.

The `/cpu/self/ref/*` backends are written in pure C and provide basic functionality.

The `/cpu/self/opt/*` backends are written in pure C and use partial e-vectors to improve performance.

The `/cpu/self/avx/*` backends rely upon AVX instructions to provide vectorized CPU performance.

The `/cpu/self/memcheck/*` backends rely upon the [Valgrind](http://valgrind.org/) Memcheck tool to help verify that user QFunctions have no undefined values.
To use, run your code with Valgrind and the Memcheck backends, e.g. `valgrind ./build/ex1 -ceed /cpu/self/ref/memcheck`.
A 'development' or 'debugging' version of Valgrind with headers is required to use this backend.
This backend can be run in serial or blocked mode and defaults to running in the serial mode if `/cpu/self/memcheck` is selected at runtime.

The `/cpu/self/xsmm/*` backends rely upon the [LIBXSMM](http://github.com/hfp/libxsmm) package to provide vectorized CPU performance.
If linking MKL and LIBXSMM is desired but the Makefile is not detecting `MKLROOT`, linking libCEED against MKL can be forced by setting the environment variable `MKL=1`.

The `/gpu/cuda/*` backends provide GPU performance strictly using CUDA.

The `/gpu/hip/*` backends provide GPU performance strictly using HIP.
They are based on the `/gpu/cuda/*` backends.
ROCm version 4.2 or newer is required.

The `/gpu/*/magma/*` backends rely upon the [MAGMA](https://bitbucket.org/icl/magma) package.
To enable the MAGMA backends, the environment variable `MAGMA_DIR` must point to the top-level MAGMA directory, with the MAGMA library located in `$(MAGMA_DIR)/lib/`.
By default, `MAGMA_DIR` is set to `../magma`; to build the MAGMA backends with a MAGMA installation located elsewhere, create a link to `magma/` in libCEED's parent directory, or set `MAGMA_DIR` to the proper location.
MAGMA version 2.5.0 or newer is required.
Currently, each MAGMA library installation is only built for either CUDA or HIP.
The corresponding set of libCEED backends (`/gpu/cuda/magma/*` or `/gpu/hip/magma/*`) will automatically be built for the version of the MAGMA library found in `MAGMA_DIR`.

Users can specify a device for all CUDA, HIP, and MAGMA backends through adding `:device_id=#` after the resource name.
For example:

> - `/gpu/cuda/gen:device_id=1`

Bit-for-bit reproducibility is important in some applications.
However, some libCEED backends use non-deterministic operations, such as `atomicAdd` for increased performance.
The backends which are capable of generating reproducible results, with the proper compilation options, are highlighted in the list above.

## Examples

libCEED comes with several examples of its usage, ranging from standalone C codes in the `/examples/ceed` directory to examples based on external packages, such as MFEM, PETSc, and Nek5000.
Nek5000 v18.0 or greater is required.

To build the examples, set the `MFEM_DIR`, `PETSC_DIR`, and `NEK5K_DIR` variables and run:

```
cd examples/
```

% running-examples-inclusion-marker

```console
# libCEED examples on CPU and GPU
cd ceed/
make
./ex1-volume -ceed /cpu/self
./ex1-volume -ceed /gpu/cuda
./ex2-surface -ceed /cpu/self
./ex2-surface -ceed /gpu/cuda
cd ..

# MFEM+libCEED examples on CPU and GPU
cd mfem/
make
./bp1 -ceed /cpu/self -no-vis
./bp3 -ceed /gpu/cuda -no-vis
cd ..

# Nek5000+libCEED examples on CPU and GPU
cd nek/
make
./nek-examples.sh -e bp1 -ceed /cpu/self -b 3
./nek-examples.sh -e bp3 -ceed /gpu/cuda -b 3
cd ..

# PETSc+libCEED examples on CPU and GPU
cd petsc/
make
./bps -problem bp1 -ceed /cpu/self
./bps -problem bp2 -ceed /gpu/cuda
./bps -problem bp3 -ceed /cpu/self
./bps -problem bp4 -ceed /gpu/cuda
./bps -problem bp5 -ceed /cpu/self
./bps -problem bp6 -ceed /gpu/cuda
cd ..

cd petsc/
make
./bpsraw -problem bp1 -ceed /cpu/self
./bpsraw -problem bp2 -ceed /gpu/cuda
./bpsraw -problem bp3 -ceed /cpu/self
./bpsraw -problem bp4 -ceed /gpu/cuda
./bpsraw -problem bp5 -ceed /cpu/self
./bpsraw -problem bp6 -ceed /gpu/cuda
cd ..

cd petsc/
make
./bpssphere -problem bp1 -ceed /cpu/self
./bpssphere -problem bp2 -ceed /gpu/cuda
./bpssphere -problem bp3 -ceed /cpu/self
./bpssphere -problem bp4 -ceed /gpu/cuda
./bpssphere -problem bp5 -ceed /cpu/self
./bpssphere -problem bp6 -ceed /gpu/cuda
cd ..

cd petsc/
make
./area -problem cube -ceed /cpu/self -degree 3
./area -problem cube -ceed /gpu/cuda -degree 3
./area -problem sphere -ceed /cpu/self -degree 3 -dm_refine 2
./area -problem sphere -ceed /gpu/cuda -degree 3 -dm_refine 2

cd fluids/
make
./navierstokes -ceed /cpu/self -degree 1
./navierstokes -ceed /gpu/cuda -degree 1
cd ..

cd solids/
make
./elasticity -ceed /cpu/self -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -problem Linear -forcing mms
./elasticity -ceed /gpu/cuda -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -problem Linear -forcing mms
cd ..
```

For the last example shown, sample meshes to be used in place of `[.exo file]` can be found at <https://github.com/jeremylt/ceedSampleMeshes>

The above code assumes a GPU-capable machine with the CUDA backends enabled.
Depending on the available backends, other CEED resource specifiers can be provided with the `-ceed` option.
Other command line arguments can be found in [examples/petsc](https://github.com/CEED/libCEED/blob/main/examples/petsc/README.md).

% benchmarks-marker

## Benchmarks

A sequence of benchmarks for all enabled backends can be run using:

```
make benchmarks
```

The results from the benchmarks are stored inside the `benchmarks/` directory and they can be viewed using the commands (requires python with matplotlib):

```
cd benchmarks
python postprocess-plot.py petsc-bps-bp1-*-output.txt
python postprocess-plot.py petsc-bps-bp3-*-output.txt
```

Using the `benchmarks` target runs a comprehensive set of benchmarks which may take some time to run.
Subsets of the benchmarks can be run using the scripts in the `benchmarks` folder.

For more details about the benchmarks, see the `benchmarks/README.md` file.

## Install

To install libCEED, run:

```
make install prefix=/path/to/install/dir
```

or (e.g., if creating packages):

```
make install prefix=/usr DESTDIR=/packaging/path
```

To build and install in separate steps, run:

```
make for_install=1 prefix=/path/to/install/dir
make install prefix=/path/to/install/dir
```

The usual variables like `CC` and `CFLAGS` are used, and optimization flags for all languages can be set using the likes of `OPT='-O3 -march=native'`.
Use `STATIC=1` to build static libraries (`libceed.a`).

To install libCEED for Python, run:

```
pip install libceed
```

with the desired setuptools options, such as `--user`.

### pkg-config

In addition to library and header, libCEED provides a [pkg-config](https://en.wikipedia.org/wiki/Pkg-config) file that can be used to easily compile and link.
[For example](https://people.freedesktop.org/~dbn/pkg-config-guide.html#faq), if `$prefix` is a standard location or you set the environment variable `PKG_CONFIG_PATH`:

```
cc `pkg-config --cflags --libs ceed` -o myapp myapp.c
```

will build `myapp` with libCEED.
This can be used with the source or installed directories.
Most build systems have support for pkg-config.

## Contact

You can reach the libCEED team by emailing [ceed-users@llnl.gov](mailto:ceed-users@llnl.gov) or by leaving a comment in the [issue tracker](https://github.com/CEED/libCEED/issues).

## How to Cite

If you utilize libCEED please cite:

```
@article{libceed-joss-paper,
  author       = {Jed Brown and Ahmad Abdelfattah and Valeria Barra and Natalie Beams and Jean Sylvain Camier and Veselin Dobrev and Yohann Dudouit and Leila Ghaffari and Tzanio Kolev and David Medina and Will Pazner and Thilina Ratnayaka and Jeremy Thompson and Stan Tomov},
  title        = {{libCEED}: Fast algebra for high-order element-based discretizations},
  journal      = {Journal of Open Source Software},
  year         = {2021},
  publisher    = {The Open Journal},
  volume       = {6},
  number       = {63},
  pages        = {2945},
  doi          = {10.21105/joss.02945}
}

@misc{libceed-user-manual,
  author       = {Abdelfattah, Ahmad and
                  Barra, Valeria and
                  Beams, Natalie and
                  Brown, Jed and
                  Camier, Jean-Sylvain and
                  Dobrev, Veselin and
                  Dudouit, Yohann and
                  Ghaffari, Leila and
                  Kolev, Tzanio and
                  Medina, David and
                  Pazner, Will and
                  Ratnayaka, Thilina and
                  Thompson, Jeremy L and
                  Tomov, Stanimire},
  title        = {{libCEED} User Manual},
  month        = jul,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.9.0},
  doi          = {10.5281/zenodo.5077489}
}
```

For libCEED's Python interface please cite:

```
@InProceedings{libceed-paper-proc-scipy-2020,
  author    = {{V}aleria {B}arra and {J}ed {B}rown and {J}eremy {T}hompson and {Y}ohann {D}udouit},
  title     = {{H}igh-performance operator evaluations with ease of use: lib{C}{E}{E}{D}'s {P}ython interface},
  booktitle = {{P}roceedings of the 19th {P}ython in {S}cience {C}onference},
  pages     = {85 - 90},
  year      = {2020},
  editor    = {{M}eghann {A}garwal and {C}hris {C}alloway and {D}illon {N}iederhut and {D}avid {S}hupe},
  doi       = {10.25080/Majora-342d178e-00c}
}
```

The BiBTeX entries for these references can be found in the `doc/bib/references.bib` file.

## Copyright

The following copyright applies to each file in the CEED software suite, unless otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE for details.

[github-badge]: https://github.com/CEED/libCEED/workflows/C/Fortran/badge.svg
[github-link]: https://github.com/CEED/libCEED/actions
[gitlab-badge]: https://gitlab.com/libceed/libCEED/badges/main/pipeline.svg?key_text=GitLab-CI
[gitlab-link]: https://gitlab.com/libceed/libCEED/-/pipelines?page=1&scope=all&ref=main
[codecov-badge]: https://codecov.io/gh/CEED/libCEED/branch/main/graphs/badge.svg
[codecov-link]: https://codecov.io/gh/CEED/libCEED/
[license-badge]: https://img.shields.io/badge/License-BSD%202--Clause-orange.svg
[license-link]: https://opensource.org/licenses/BSD-2-Clause
[doc-badge]: https://readthedocs.org/projects/libceed/badge/?version=latest
[doc-link]: https://libceed.org/en/latest/?badge=latest
[joss-badge]: https://joss.theoj.org/papers/10.21105/joss.02945/status.svg
[joss-link]: https://doi.org/10.21105/joss.02945
[binder-badge]: http://mybinder.org/badge_logo.svg
[binder-link]: https://mybinder.org/v2/gh/CEED/libCEED/main?urlpath=lab/tree/examples/python/tutorial-0-ceed.ipynb
