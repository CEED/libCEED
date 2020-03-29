libCEED: the CEED Library
============================================

|build-status| |codecov| |license| |doc| |doxygen| |binder|

.. |build-status| image:: https://travis-ci.org/CEED/libCEED.svg?branch=master
    :alt: Build Status
    :scale: 100%
    :target: https://travis-ci.org/CEED/libCEED

.. |codecov| image:: https://codecov.io/gh/CEED/libCEED/branch/master/graphs/badge.svg
    :alt: Code Coverage
    :scale: 100%
    :target: https://codecov.io/gh/CEED/libCEED/

.. |license| image:: https://img.shields.io/badge/License-BSD%202--Clause-orange.svg
    :alt: License
    :scale: 100%
    :target: https://opensource.org/licenses/BSD-2-Clause

.. |doc| image:: https://readthedocs.org/projects/libceed/badge/?version=latest
    :alt: License
    :scale: 100%
    :target: https://libceed.readthedocs.io/en/latest/?badge=latest

.. |doxygen| image:: https://codedocs.xyz/CEED/libCEED.svg
    :alt: License
    :scale: 100%
    :target: https://codedocs.xyz/CEED/libCEED/

.. |binder| image:: http://mybinder.org/badge_logo.svg
    :alt: Binder
    :scale: 100%
    :target: https://mybinder.org/v2/gh/CEED/libCEED/master?urlpath=lab/tree/examples/tutorials/tutorial1-python-ceed.ipynb

Code for Efficient Extensible Discretization
--------------------------------------------

This repository contains an initial low-level API library for the efficient
high-order discretization methods developed by the ECP co-design
`Center for Efficient Exascale Discretizations (CEED) <http://ceed.exascaleproject.org>`_.
While our focus is on high-order finite elements, the approach is mostly
algebraic and thus applicable to other discretizations in factored form, as
explained in the `User manual <https://libceed.readthedocs.io/en/latest/>`_ and
API implementation portion of the
`documentation <https://libceed.readthedocs.io/en/latest/libCEEDapi.html>`_.

One of the challenges with high-order methods is that a global sparse matrix is
no longer a good representation of a high-order linear operator, both with
respect to the FLOPs needed for its evaluation, as well as the memory transfer
needed for a matvec.  Thus, high-order methods require a new "format" that still
represents a linear (or more generally non-linear) operator, but not through a
sparse matrix.

The goal of libCEED is to propose such a format, as well as supporting
implementations and data structures, that enable efficient operator evaluation
on a variety of computational device types (CPUs, GPUs, etc.). This new operator
description is based on algebraically
`factored form <https://libceed.readthedocs.io/en/latest/libCEEDapi.html>`_,
which is easy to incorporate in a wide variety of applications, without significant
refactoring of their own discretization infrastructure.

The repository is part of the
`CEED software suite <http://ceed.exascaleproject.org/software/>`_, a collection of
software benchmarks, miniapps, libraries and APIs for efficient exascale
discretizations based on high-order finite element and spectral element methods.
See http://github.com/ceed for more information and source code availability.

The CEED research is supported by the
`Exascale Computing Project <https://exascaleproject.org/exascale-computing-project>`_
(17-SC-20-SC), a collaborative effort of two U.S. Department of Energy
organizations (Office of Science and the National Nuclear Security
Administration) responsible for the planning and preparation of a
`capable exascale ecosystem <https://exascaleproject.org/what-is-exascale>`_, including
software, applications, hardware, advanced system engineering and early testbed
platforms, in support of the nation’s exascale computing imperative.

For more details on the CEED API see the `user manual <https://libceed.readthedocs.io/en/latest/>`_.


.. gettingstarted-inclusion-marker

Building
----------------------------------------

The CEED library, ``libceed``, is a C99 library with no required dependencies, and
with Fortran and Python interfaces.  It can be built using::

    make

or, with optimization flags::

    make OPT='-O3 -march=skylake-avx512 -ffp-contract=fast'

These optimization flags are used by all languages (C, C++, Fortran) and this
makefile variable can also be set for testing and examples (below).
Python users can install using::

    pip install libceed

or in a clone of the repository via ``pip install .``.

The library attempts to automatically detect support for the AVX
instruction set using gcc-style compiler options for the host.
Support may need to be manually specified via::

    make AVX=1

or::

    make AVX=0

if your compiler does not support gcc-style options, if you are cross
compiling, etc.


Testing
----------------------------------------

The test suite produces `TAP <https://testanything.org>`_ output and is run by::

    make test

or, using the ``prove`` tool distributed with Perl (recommended)::

    make prove

Backends
----------------------------------------

There are multiple supported backends, which can be selected at runtime in the examples:

+----------------------------+---------------------------------------------------+
| CEED resource              | Backend                                           |
+----------------------------+---------------------------------------------------+
| ``/cpu/self/ref/serial``   | Serial reference implementation                   |
+----------------------------+---------------------------------------------------+
| ``/cpu/self/ref/blocked``  | Blocked reference implementation                  |
+----------------------------+---------------------------------------------------+
| ``/cpu/self/ref/memcheck`` | Memcheck backend, undefined value checks          |
+----------------------------+---------------------------------------------------+
| ``/cpu/self/opt/serial``   | Serial optimized C implementation                 |
+----------------------------+---------------------------------------------------+
| ``/cpu/self/opt/blocked``  | Blocked optimized C implementation                |
+----------------------------+---------------------------------------------------+
| ``/cpu/self/avx/serial``   | Serial AVX implementation                         |
+----------------------------+---------------------------------------------------+
| ``/cpu/self/avx/blocked``  | Blocked AVX implementation                        |
+----------------------------+---------------------------------------------------+
| ``/cpu/self/xsmm/serial``  | Serial LIBXSMM implementation                     |
+----------------------------+---------------------------------------------------+
| ``/cpu/self/xsmm/blocked`` | Blocked LIBXSMM implementation                    |
+----------------------------+---------------------------------------------------+
| ``/cpu/occa``              | Serial OCCA kernels                               |
+----------------------------+---------------------------------------------------+
| ``/gpu/occa``              | CUDA OCCA kernels                                 |
+----------------------------+---------------------------------------------------+
| ``/omp/occa``              | OpenMP OCCA kernels                               |
+----------------------------+---------------------------------------------------+
| ``/ocl/occa``              | OpenCL OCCA kernels                               |
+----------------------------+---------------------------------------------------+
| ``/gpu/cuda/ref``          | Reference pure CUDA kernels                       |
+----------------------------+---------------------------------------------------+
| ``/gpu/cuda/reg``          | Pure CUDA kernels using one thread per element    |
+----------------------------+---------------------------------------------------+
| ``/gpu/cuda/shared``       | Optimized pure CUDA kernels using shared memory   |
+----------------------------+---------------------------------------------------+
| ``/gpu/cuda/gen``          | Optimized pure CUDA kernels using code generation |
+----------------------------+---------------------------------------------------+
| ``/gpu/magma``             | CUDA MAGMA kernels                                |
+----------------------------+---------------------------------------------------+

The ``/cpu/self/*/serial`` backends process one element at a time and are intended for meshes
with a smaller number of high order elements. The ``/cpu/self/*/blocked`` backends process
blocked batches of eight interlaced elements and are intended for meshes with higher numbers
of elements.

The ``/cpu/self/ref/*`` backends are written in pure C and provide basic functionality.

The ``/cpu/self/opt/*`` backends are written in pure C and use partial e-vectors to improve performance.

The ``/cpu/self/avx/*`` backends rely upon AVX instructions to provide vectorized CPU performance.

The ``/cpu/self/xsmm/*`` backends rely upon the `LIBXSMM <http://github.com/hfp/libxsmm>`_ package
to provide vectorized CPU performance. If linking MKL and LIBXSMM is desired but
the Makefile is not detecting ``MKLROOT``, linking libCEED against MKL can be
forced by setting the environment variable ``MKL=1``.

The ``/cpu/self/memcheck/*`` backends rely upon the `Valgrind <http://valgrind.org/>`_ Memcheck tool
to help verify that user QFunctions have no undefined values. To use, run your code with
Valgrind and the Memcheck backends, e.g. ``valgrind ./build/ex1 -ceed /cpu/self/ref/memcheck``. A
'development' or 'debugging' version of Valgrind with headers is required to use this backend.
This backend can be run in serial or blocked mode and defaults to running in the serial mode
if ``/cpu/self/memcheck`` is selected at runtime.

The ``/*/occa`` backends rely upon the `OCCA <http://github.com/libocca/occa>`_ package to provide
cross platform performance.

The ``/gpu/cuda/*`` backends provide GPU performance strictly using CUDA.

The ``/gpu/magma`` backend relies upon the `MAGMA <https://bitbucket.org/icl/magma>`_ package.


Examples
----------------------------------------

libCEED comes with several examples of its usage, ranging from standalone C
codes in the ``/examples/ceed`` directory to examples based on external packages,
such as MFEM, PETSc, and Nek5000. Nek5000 v18.0 or greater is required.

To build the examples, set the ``MFEM_DIR``, ``PETSC_DIR``, and
``NEK5K_DIR`` variables and run::

   cd examples/

.. running-examples-inclusion-marker

.. code:: console

   # libCEED examples on CPU and GPU
   cd ceed/
   make
   ./ex1-volume -ceed /cpu/self
   ./ex1-volume -ceed /gpu/occa
   ./ex2-surface -ceed /cpu/self
   ./ex2-surface -ceed /gpu/occa
   cd ..

   # MFEM+libCEED examples on CPU and GPU
   cd mfem/
   make
   ./bp1 -ceed /cpu/self -no-vis
   ./bp3 -ceed /gpu/occa -no-vis
   cd ..

   # Nek5000+libCEED examples on CPU and GPU
   cd nek/
   make
   ./nek-examples.sh -e bp1 -ceed /cpu/self -b 3
   ./nek-examples.sh -e bp3 -ceed /gpu/occa -b 3
   cd ..

   # PETSc+libCEED examples on CPU and GPU
   cd petsc/
   make
   ./bps -problem bp1 -ceed /cpu/self
   ./bps -problem bp2 -ceed /gpu/occa
   ./bps -problem bp3 -ceed /cpu/self
   ./bps -problem bp4 -ceed /gpu/occa
   ./bps -problem bp5 -ceed /cpu/self
   ./bps -problem bp6 -ceed /gpu/occa
   cd ..

   cd petsc/
   make
   ./bpsraw -problem bp1 -ceed /cpu/self
   ./bpsraw -problem bp2 -ceed /gpu/occa
   ./bpsraw -problem bp3 -ceed /cpu/self
   ./bpsraw -problem bp4 -ceed /gpu/occa
   ./bpsraw -problem bp5 -ceed /cpu/self
   ./bpsraw -problem bp6 -ceed /gpu/occa
   cd ..

   cd petsc/
   make
   ./bpssphere -problem bp1 -ceed /cpu/self
   ./bpssphere -problem bp2 -ceed /gpu/occa
   ./bpssphere -problem bp3 -ceed /cpu/self
   ./bpssphere -problem bp4 -ceed /gpu/occa
   ./bpssphere -problem bp5 -ceed /cpu/self
   ./bpssphere -problem bp6 -ceed /gpu/occa
   cd ..

   cd petsc/
   make
   ./area -problem cube -ceed /cpu/self -petscspace_degree 3
   ./area -problem cube -ceed /gpu/occa -petscspace_degree 3
   ./area -problem sphere -ceed /cpu/self -petscspace_degree 3 -dm_refine 2
   ./area -problem sphere -ceed /gpu/occa -petscspace_degree 3 -dm_refine 2

   cd fluids/
   make
   ./navierstokes -ceed /cpu/self -petscspace_degree 1
   ./navierstokes -ceed /gpu/occa -petscspace_degree 1
   cd ..

   cd solids/
   make
   ./elasticity -ceed /cpu/self -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -problem linElas -forcing mms
   ./elasticity -ceed /gpu/occa -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -problem linElas -forcing mms
   cd ..

For the last example shown, sample meshes to be used in place of
``[.exo file]`` can be found at https://github.com/jeremylt/ceedSampleMeshes

The above code assumes a GPU-capable machine with the OCCA backend
enabled. Depending on the available backends, other CEED resource
specifiers can be provided with the ``-ceed`` option. Other command line
arguments can be found in the `petsc <./petsc/README.md>`_ folder.


.. benchmarks-marker

Benchmarks
----------------------------------------

A sequence of benchmarks for all enabled backends can be run using::

   make benchmarks

The results from the benchmarks are stored inside the ``benchmarks/`` directory
and they can be viewed using the commands (requires python with matplotlib)::

   cd benchmarks
   python postprocess-plot.py petsc-bps-bp1-*-output.txt
   python postprocess-plot.py petsc-bps-bp3-*-output.txt

Using the ``benchmarks`` target runs a comprehensive set of benchmarks which may
take some time to run. Subsets of the benchmarks can be run using the scripts in the ``benchmarks`` folder.

For more details about the benchmarks, see the ``benchmarks/README.md`` file.


Install
----------------------------------------

To install libCEED, run::

    make install prefix=/usr/local

or (e.g., if creating packages)::

    make install prefix=/usr DESTDIR=/packaging/path

Note that along with the library, libCEED installs kernel sources, e.g. OCCA
kernels are installed in ``$prefix/lib/okl``. This allows the OCCA backend to
build specialized kernels at run-time. In a normal setting, the kernel sources
will be found automatically (relative to the library file ``libceed.so``).
However, if that fails (e.g. if ``libceed.so`` is moved), one can copy (cache) the
kernel sources inside the user OCCA directory, ``~/.occa`` using::

    $(OCCA_DIR)/bin/occa cache ceed $(CEED_DIR)/lib/okl/*.okl

This will allow OCCA to find the sources regardless of the location of the CEED
library. One may occasionally need to clear the OCCA cache, which can be accomplished
by removing the ``~/.occa`` directory or by calling ``$(OCCA_DIR)/bin/occa clear -a``.

To install libCEED for Python, run::

    pip install libceed

with the desired setuptools options, such as `--user`.


pkg-config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to library and header, libCEED provides a `pkg-config <https://en.wikipedia.org/wiki/Pkg-config>`_
file that can be used to easily compile and link.
`For example <https://people.freedesktop.org/~dbn/pkg-config-guide.html#faq>`_, if
``$prefix`` is a standard location or you set the environment variable
``PKG_CONFIG_PATH``::

    cc `pkg-config --cflags --libs ceed` -o myapp myapp.c

will build ``myapp`` with libCEED.  This can be used with the source or
installed directories.  Most build systems have support for pkg-config.


Contact
----------------------------------------

You can reach the libCEED team by emailing `ceed-users@llnl.gov <mailto:ceed-users@llnl.gov>`_
or by leaving a comment in the `issue tracker <https://github.com/CEED/libCEED/issues>`_.


Copyright
----------------------------------------

The following copyright applies to each file in the CEED software suite, unless
otherwise stated in the file:

   Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at the
   Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights reserved.

See files LICENSE and NOTICE for details.
