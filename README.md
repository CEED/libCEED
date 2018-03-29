# libCEED: the CEED API Library

[![Build Status](https://travis-ci.org/CEED/libCEED.svg?branch=master)](https://travis-ci.org/CEED/libCEED)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

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

The repository is part of the [CEED software
suite](http://ceed.exascaleproject.org/software/), a collection of software
benchmarks, miniapps, libraries and APIs for efficient exascale discretizations
based on high-order finite element and spectral element methods.  See
http://github.com/ceed for more information and source code availability.

The CEED research is supported by the [Exascale Computing Project](https://exascaleproject.org/exascale-computing-project)
(17-SC-20-SC), a collaborative effort of two U.S. Department of Energy
organizations (Office of Science and the National Nuclear Security
Administration) responsible for the planning and preparation of a
[capable exascale ecosystem](https://exascaleproject.org/what-is-exascale),
including software, applications, hardware, advanced system engineering and early
testbed platforms, in support of the nation’s exascale computing imperative.

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

## Install

To install libCEED, run

    make install prefix=/usr/local

or (e.g., if creating packages),

    make install prefix=/usr DESTDIR=/packaging/path

### pkg-config

In addition to library and header, libCEED provides a
[pkg-config](https://en.wikipedia.org/wiki/Pkg-config) file that can be
used to easily compile and link.
[For example](https://people.freedesktop.org/~dbn/pkg-config-guide.html#faq),
if `$prefix` is a standard location or you set the environment variable
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
