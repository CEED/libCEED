# libCEED: the CEED API Library

[![Build Status](https://travis-ci.org/CEED/libCEED.svg?branch=master)](https://travis-ci.org/CEED/libCEED)
[![Code Coverage](https://codecov.io/gh/CEED/libCEED/branch/master/graphs/badge.svg)](https://codecov.io/gh/CEED/libCEED/)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Documentation Status](https://readthedocs.org/projects/libceed/badge/?version=latest)](https://libceed.readthedocs.io/en/latest/?badge=latest)
[![Doxygen](https://codedocs.xyz/CEED/libCEED.svg)](https://codedocs.xyz/CEED/libCEED/)

## Code for Efficient Extensible Discretization

This repository contains an initial low-level API library for the efficient
high-order discretization methods developed by the ECP co-design [Center for
Efficient Exascale Discretizations (CEED)](http://ceed.exascaleproject.org).
While our focus is on high-order finite elements, the approach is mostly
algebraic and thus applicable to other discretizations in factored form, as
explained in the [User manual](https://libceed.readthedocs.io/en/latest/) and API implementation portion of the [documentation](https://libceed.readthedocs.io/en/latest/libCEEDapi.html).

One of the challenges with high-order methods is that a global sparse matrix is
no longer a good representation of a high-order linear operator, both with
respect to the FLOPs needed for its evaluation, as well as the memory transfer
needed for a matvec.  Thus, high-order methods require a new "format" that still
represents a linear (or more generally non-linear) operator, but not through a
sparse matrix.

The goal of libCEED is to propose such a format, as well as supporting
implementations and data structures, that enable efficient operator evaluation
on a variety of computational device types (CPUs, GPUs, etc.). This new operator
description is based on algebraically [factored form](https://libceed.readthedocs.io/en/latest/libCEEDapi.html),
which is easy to incorporate in a wide variety of applications, without significant
refactoring of their own discretization infrastructure.

The repository is part of the [CEED software suite][ceed-soft], a collection of
software benchmarks, miniapps, libraries and APIs for efficient exascale
discretizations based on high-order finite element and spectral element methods.
See http://github.com/ceed for more information and source code availability.

The CEED research is supported by the [Exascale Computing Project][ecp]
(17-SC-20-SC), a collaborative effort of two U.S. Department of Energy
organizations (Office of Science and the National Nuclear Security
Administration) responsible for the planning and preparation of a [capable
exascale ecosystem](https://exascaleproject.org/what-is-exascale), including
software, applications, hardware, advanced system engineering and early testbed
platforms, in support of the nation’s exascale computing imperative.

For more details on the CEED API see http://ceed.exascaleproject.org/ceed-code/.

For detailed instructions on how to build libCEED and run benchmarks and examples, please see the dedicated [Getting Started](https://libceed.readthedocs.io/en/latest/gettingstarted.html) page in the [User manual](https://libceed.readthedocs.io/en/latest/). A short summary is provided here.

## Building

The CEED library, `libceed`, is a C99 library with no required dependencies, and
with Fortran and Python interfaces.  It can be built using

    make

or, with optimization flags

    make OPT='-O3 -march=skylake-avx512 -ffp-contract=fast'

These optimization flags are used by all languages (C, C++, Fortran) and this
makefile variable can also be set for testing and examples (below).
Python users can install using

    pip install libceed

or in a clone of the repository via `pip install .`.
The library attempts to automatically detect support for the AVX
instruction set using gcc-style compiler options for the host.
Support may need to be manually specified via

    make AVX=1

or

    make AVX=0

if your compiler does not support gcc-style options, if you are cross
compiling, etc.

## Testing

The test suite produces [TAP](https://testanything.org) output and is run by:

    make test

or, using the `prove` tool distributed with Perl (recommended)

    make prove

## Backends

There are multiple supported backends, which can be selected at runtime in the examples:

| CEED resource            | Backend                                           |
| :----------------------- | :------------------------------------------------ |
| `/cpu/self/ref/serial`   | Serial reference implementation                   |
| `/cpu/self/ref/blocked`  | Blocked refrence implementation                   |
| `/cpu/self/memcheck`     | Memcheck backend, undefined value checks          |
| `/cpu/self/opt/serial`   | Serial optimized C implementation                 |
| `/cpu/self/opt/blocked`  | Blocked optimized C implementation                |
| `/cpu/self/avx/serial`   | Serial AVX implementation                         |
| `/cpu/self/avx/blocked`  | Blocked AVX implementation                        |
| `/cpu/self/xsmm/serial`  | Serial LIBXSMM implementation                     |
| `/cpu/self/xsmm/blocked` | Blocked LIBXSMM implementation                    |
| `/cpu/occa`              | Serial OCCA kernels                               |
| `/gpu/occa`              | CUDA OCCA kernels                                 |
| `/omp/occa`              | OpenMP OCCA kernels                               |
| `/ocl/occa`              | OpenCL OCCA kernels                               |
| `/gpu/cuda/ref`          | Reference pure CUDA kernels                       |
| `/gpu/cuda/reg`          | Pure CUDA kernels using one thread per element    |
| `/gpu/cuda/shared`       | Optimized pure CUDA kernels using shared memory   |
| `/gpu/cuda/gen`          | Optimized pure CUDA kernels using code generation |
| `/gpu/magma`             | CUDA MAGMA kernels                                |

The `/cpu/self/*/serial` backends process one element at a time and are intended for meshes
with a smaller number of high order elements. The `/cpu/self/*/blocked` backends process
blocked batches of eight interlaced elements and are intended for meshes with higher numbers
of elements.

The `/cpu/self/ref/*` backends are written in pure C and provide basic functionality.

The `/cpu/self/opt/*` backends are written in pure C and use partial e-vectors to improve performance.

The `/cpu/self/avx/*` backends rely upon AVX instructions to provide vectorized CPU performance.

The `/cpu/self/xsmm/*` backends rely upon the [LIBXSMM](http://github.com/hfp/libxsmm) package
to provide vectorized CPU performance. If linking MKL and LIBXSMM is desired but
the Makefile is not detecting `MKLROOT`, linking libCEED against MKL can be
forced by setting the environment variable `MKL=1`.

The `/cpu/self/memcheck/*` backends rely upon the [Valgrind](http://valgrind.org/) Memcheck tool
to help verify that user QFunctions have no undefined values. To use, run your code with
Valgrind and the Memcheck backends, e.g. `valgrind ./build/ex1 -ceed /cpu/self/ref/memcheck`. A
'development' or 'debugging' version of Valgrind with headers is required to use this backend.
This backend can be run in serial or blocked mode and defaults to running in the serial mode
if `/cpu/self/memcheck` is selected at runtime.

The `/*/occa` backends rely upon the [OCCA](http://github.com/libocca/occa) package to provide
cross platform performance.

The `/gpu/cuda/*` backends provide GPU performance strictly using CUDA.

The `/gpu/magma` backend relies upon the [MAGMA](https://bitbucket.org/icl/magma) package.
