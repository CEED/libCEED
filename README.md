# libCEED: the CEED API Library

[![Build Status](https://travis-ci.org/CEED/libCEED.svg?branch=master)](https://travis-ci.org/CEED/libCEED)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Doxygen](https://codedocs.xyz/CEED/libCEED.svg)](https://codedocs.xyz/CEED/libCEED/)

## Code for Efficient Extensible Discretization

This repository contains an initial low-level API library for the efficient
high-order discretization methods developed by the ECP co-design [Center for
Efficient Exascale Discretizations (CEED)](http://ceed.exascaleproject.org).
While our focus is on high-order finite elements, the approach is mostly
algebraic and thus applicable to other discretizations in factored form, see the
[API documentation](doc/libCEED.md).

One of the challenges with high-order methods is that a global sparse matrix is
no longer a good representation of a high-order linear operator, both with
respect to the FLOPs needed for its evaluation, as well as the memory transfer
needed for a matvec.  Thus, high-order methods require a new "format" that still
represents a linear (or more generally non-linear) operator, but not through a
sparse matrix.

The goal of libCEED is to propose such a format, as well as supporting
implementations and data structures, that enable efficient operator evaluation
on a variety of computational device types (CPUs, GPUs, etc.). This new operator
description is based on algebraically [factored form](doc/libCEED.md), which is
easy to incorporate in a wide variety of applications, without significant
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

## Building

The CEED library, `libceed`, is a C99 library with no external dependencies.
It can be built using

    make

## Testing

The test suite produces [TAP](https://testanything.org) output and is run by:

    make test

or, using the `prove` tool distributed with Perl (recommended)

    make prove

## Examples

libCEED comes with several examples of its usage, ranging from standalone C
codes in the `/examples/ceed` directory to examples based on external packages,
such as MFEM, PETSc and Nek5000.

To build the examples, set the `MFEM_DIR`, `PETSC_DIR` and `NEK5K_DIR` variables
and run:

```console
# libCEED examples on CPU and GPU
cd examples/ceed
make
./ex1 -ceed /cpu/self
./ex1 -ceed /gpu/occa
cd ../..

# MFEM+libCEED examples on CPU and GPU
cd examples/mfem
make
./bp1 -ceed /cpu/self -no-vis
./bp1 -ceed /gpu/occa -no-vis
cd ../..

# PETSc+libCEED examples on CPU and GPU
cd examples/petsc
make
./bp1 -ceed /cpu/self
./bp1 -ceed /gpu/occa
cd ../..

# Nek+libCEED examples on CPU and GPU
cd examples/nek5000
./generate-boxes.sh 2 4
./make-nek-examples.sh
./run-nek-example.sh -ceed /cpu/self -b b3
./run-nek-example.sh -ceed /gpu/occa -b b3
cd ../..
```

The above code assumes a GPU-capable machine enabled in the OCCA
backend. Depending on the availabl backends, other Ceed resource specifiers can
be provided with the `-ceed` option, for example:

CEED resource (`-ceed`) | Backend
----------------------- | ---------------------------------
`/cpu/self`             | Serial reference implementation
`/cpu/occa`             | Serial OCCA kernels
`/gpu/occa`             | CUDA OCCA kernels
`/omp/occa`             | OpenMP OCCA kernels
`/ocl/occa`             | OpenCL OCCA kernels

## Install

To install libCEED, run

    make install prefix=/usr/local

or (e.g., if creating packages),

    make install prefix=/usr DESTDIR=/packaging/path

Note that along with the library, libCEED installs kernel sources, e.g. OCCA
kernels are installed in `$prefix/lib/okl`. This allows the OCCA backend to
build specialized kernels at run-time. In a normal setting, the kernel sources
will be found automatically (relative to the library file `libceed.so`).
However, if that fails (e.g. if `libceed.so` is moved), one can copy (cache) the
kernel sources inside the user OCCA directory, `~/.occa` using

    $(OCCA_DIR)/bin/occa cache ceed $(CEED_DIR)/lib/okl/*.okl

This will allow OCCA to find the sources regardless of the location of the CEED
library. One may occasionally need to clear the OCCA cache, which can be accomplished
by removing the `~/.occa` directory or by calling `$(OCCA_DIR)/bin/occa clear -a`.

### pkg-config

In addition to library and header, libCEED provides a [pkg-config][pkg-config1]
file that can be used to easily compile and link. [For example][pkg-config2], if
`$prefix` is a standard location or you set the environment variable
`PKG_CONFIG_PATH`,

    cc `pkg-config --cflags --libs ceed` -o myapp myapp.c

will build `myapp` with libCEED.  This can be used with the source or
installed directories.  Most build systems have support for pkg-config.

## Contact

You can reach the libCEED team by emailing [ceed-users@llnl.gov](mailto:ceed-users@llnl.gov)
or by leaving a comment in the [issue tracker](https://github.com/CEED/libCEED/issues).

## Copyright

The following copyright applies to each file in the CEED software suite, unless
otherwise stated in the file:

> Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
> Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE for details.

[ceed-soft]:   http://ceed.exascaleproject.org/software/
[ecp]:         https://exascaleproject.org/exascale-computing-project
[pkg-config1]: https://en.wikipedia.org/wiki/Pkg-config
[pkg-config2]: https://people.freedesktop.org/~dbn/pkg-config-guide.html#faq
