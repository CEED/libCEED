Changes/Release Notes
========================================

On this page we provide a summary of the main API changes, new features and examples
for each release of libCEED.

.. _main:

Current Main
----------------------------------------

The current ``main`` (formerly called ``master``) branch contains bug fixes and additional features.

Interface changes
^^^^^^^^^^^^^^^^^
* Error handling improved to include enumerated error codes for C interface return values

New features
^^^^^^^^^^^^
* New HIP MAGMA backends for hipMAGMA library users: ``/gpu/hip/magma`` and ``/gpu/hip/magma/det``.
* Julia and Rust interfaces added, providing a nearly 1-1 correspondence with the C interface, plus some convenience features.
* New HIP backends for improved tensor basis performance: ``/gpu/hip/shared`` and ``/gpu/hip/gen``.
* Static libraries can be built with ``make STATIC=1`` and the pkg-config file is installed accordingly.
* Add :cpp:func:`CeedOperatorLinearAssembleSymbolic` and :cpp:func:`CeedOperatorLinearAssemble` to support full assembly of libCEED operators.

Performance improvements
^^^^^^^^^^^^^^^^^^^^^^^^

Examples
^^^^^^^^
* :ref:`example-petsc-elasticity` example updated with traction boundary conditions and improved Dirichlet boundary conditions.
* :ref:`example-petsc-elasticity` example updated with Neo-Hookean hyperelasticity in current configuration as well as improved Neo-Hookean hyperelasticity exploring storage vs computation tradeoffs.

.. _v0.7:

v0.7 (Sep 29, 2020)
-------------------

Interface changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Replace limited :code:`CeedInterlaceMode` with more flexible component stride :code:`compstride` in :code:`CeedElemRestriction` constructors.
  As a result, the :code:`indices` parameter has been replaced with :code:`offsets` and the :code:`nnodes` parameter has been replaced with :code:`lsize`.
  These changes improve support for mixed finite element methods.
* Replace various uses of :code:`Ceed*Get*Status` with :code:`Ceed*Is*` in the backend API to match common nomenclature.
* Replace :code:`CeedOperatorAssembleLinearDiagonal` with :cpp:func:`CeedOperatorLinearAssembleDiagonal` for clarity.
* Linear Operators can be assembled as point-block diagonal matrices with :cpp:func:`CeedOperatorLinearAssemblePointBlockDiagonal`, provided in row-major form in a :code:`ncomp` by :code:`ncomp` block per node.
* Diagonal assemble interface changed to accept a :ref:`CeedVector` instead of a pointer to a :ref:`CeedVector` to reduce memory movement when interfacing with calling code.
* Added :cpp:func:`CeedOperatorLinearAssembleAddDiagonal` and :cpp:func:`CeedOperatorLinearAssembleAddPointBlockDiagonal` for improved future integration with codes such as MFEM that compose the action of :ref:`CeedOperator`\s external to libCEED.
* Added :cpp:func:`CeedVectorTakeAray` to sync and remove libCEED read/write access to an allocated array and pass ownership of the array to the caller.
  This function is recommended over :cpp:func:`CeedVectorSyncArray` when the :code:`CeedVector` has an array owned by the caller that was set by :cpp:func:`CeedVectorSetArray`.
* Added :code:`CeedQFunctionContext` object to manage user QFunction context data and reduce copies between device and host memory.
* Added :cpp:func:`CeedOperatorMultigridLevelCreate`, :cpp:func:`CeedOperatorMultigridLevelCreateTensorH1`, and :cpp:func:`CeedOperatorMultigridLevelCreateH1` to facilitate creation of multigrid prolongation, restriction, and coarse grid operators using a common quadrature space.

New features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New HIP backend: ``/gpu/hip/ref``.
* CeedQFunction support for user ``CUfunction``\s in some backends

Performance improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* OCCA backend rebuilt to facilitate future performance enhancements.
* Petsc BPs suite improved to reduce noise due to multiple calls to :code:`mpiexec`.

Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* :ref:`example-petsc-elasticity` example updated with strain energy computation and more flexible boundary conditions.

Deprecated backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* The ``/gpu/cuda/reg`` backend has been removed, with its core features moved into ``/gpu/cuda/ref`` and ``/gpu/cuda/shared``.

.. _v0.6:

v0.6 (Mar 29, 2020)
----------------------------------------

libCEED v0.6 contains numerous new features and examples, as well as expanded
documentation in `this new website <https://libceed.readthedocs.io>`_.

New features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* New Python interface using `CFFI <https://cffi.readthedocs.io/>`_ provides a nearly
  1-1 correspondence with the C interface, plus some convenience features.  For instance,
  data stored in the :cpp:type:`CeedVector` structure are available without copy as
  :py:class:`numpy.ndarray`.  Short tutorials are provided in
  `Binder <https://mybinder.org/v2/gh/CEED/libCEED/main?urlpath=lab/tree/examples/tutorials/>`_.
* Linear QFunctions can be assembled as block-diagonal matrices (per quadrature point,
  :cpp:func:`CeedOperatorAssembleLinearQFunction`) or to evaluate the diagonal
  (:cpp:func:`CeedOperatorAssembleLinearDiagonal`).  These operations are useful for
  preconditioning ingredients and are used in the libCEED's multigrid examples.
* The inverse of separable operators can be obtained using
  :cpp:func:`CeedOperatorCreateFDMElementInverse` and applied with
  :cpp:func:`CeedOperatorApply`.  This is a useful preconditioning ingredient,
  especially for Laplacians and related operators.
* New functions: :cpp:func:`CeedVectorNorm`, :cpp:func:`CeedOperatorApplyAdd`,
  :cpp:func:`CeedQFunctionView`, :cpp:func:`CeedOperatorView`.
* Make public accessors for various attributes to facilitate writing composable code.
* New backend: ``/cpu/self/memcheck/serial``.
* QFunctions using variable-length array (VLA) pointer constructs can be used with CUDA
  backends.  (Single source is coming soon for OCCA backends.)
* Fix some missing edge cases in CUDA backend.

Performance Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* MAGMA backend performance optimization and non-tensor bases.
* No-copy optimization in :cpp:func:`CeedOperatorApply`.

Interface changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Replace :code:`CeedElemRestrictionCreateIdentity` and
  :code:`CeedElemRestrictionCreateBlocked` with more flexible
  :cpp:func:`CeedElemRestrictionCreateStrided` and
  :cpp:func:`CeedElemRestrictionCreateBlockedStrided`.
* Add arguments to :cpp:func:`CeedQFunctionCreateIdentity`.
* Replace ambiguous uses of :cpp:enum:`CeedTransposeMode` for L-vector identification
  with :cpp:enum:`CeedInterlaceMode`.  This is now an attribute of the
  :cpp:type:`CeedElemRestriction` (see :cpp:func:`CeedElemRestrictionCreate`) and no
  longer passed as ``lmode`` arguments to :cpp:func:`CeedOperatorSetField` and
  :cpp:func:`CeedElemRestrictionApply`.

Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

libCEED-0.6 contains greatly expanded examples with :ref:`new documentation <Examples>`.
Notable additions include:

* Standalone :ref:`ex2-surface` (:file:`examples/ceed/ex2-surface`): compute the area of
  a domain in 1, 2, and 3 dimensions by applying a Laplacian.
* PETSc :ref:`example-petsc-area` (:file:`examples/petsc/area.c`): computes surface area
  of domains (like the cube and sphere) by direct integration on a surface mesh;
  demonstrates geometric dimension different from topological dimension.
* PETSc :ref:`example-petsc-bps`:

  * :file:`examples/petsc/bpsraw.c` (formerly ``bps.c``): transparent CUDA support.
  * :file:`examples/petsc/bps.c` (formerly ``bpsdmplex.c``): performance improvements
    and transparent CUDA support.
  * :ref:`example-petsc-bps-sphere` (:file:`examples/petsc/bpssphere.c`):
    generalizations of all CEED BPs to the surface of the sphere; demonstrates geometric
    dimension different from topological dimension.

* :ref:`example-petsc-multigrid` (:file:`examples/petsc/multigrid.c`): new p-multigrid
  solver with algebraic multigrid coarse solve.
* :ref:`example-petsc-navier-stokes` (:file:`examples/fluids/navierstokes.c`; formerly
  ``examples/navier-stokes``): unstructured grid support (using PETSc's ``DMPlex``),
  implicit time integration, SU/SUPG stabilization, free-slip boundary conditions, and
  quasi-2D computational domain support.
* :ref:`example-petsc-elasticity` (:file:`examples/solids/elasticity.c`): new solver for
  linear elasticity, small-strain hyperelasticity, and globalized finite-strain
  hyperelasticity using p-multigrid with algebraic multigrid coarse solve.

.. _v0.5:

v0.5 (Sep 18, 2019)
----------------------------------------

For this release, several improvements were made. Two new CUDA backends were added to
the family of backends, of which, the new ``cuda-gen`` backend achieves state-of-the-art
performance using single-source :ref:`CeedQFunction`. From this release, users
can define Q-Functions in a single source code independently of the targeted backend
with the aid of a new macro ``CEED QFUNCTION`` to support JIT (Just-In-Time) and CPU
compilation of the user provided :ref:`CeedQFunction` code. To allow a unified
declaration, the :ref:`CeedQFunction` API has undergone a slight change:
the ``QFunctionField`` parameter ``ncomp`` has been changed to ``size``. This change
requires setting the previous value of ``ncomp`` to ``ncomp*dim`` when adding a
``QFunctionField`` with eval mode ``CEED EVAL GRAD``.

Additionally, new CPU backends
were included in this release, such as the ``/cpu/self/opt/*`` backends (which are
written in pure C and use partial **E-vectors** to improve performance) and the
``/cpu/self/ref/memcheck`` backend (which relies upon the
`Valgrind <http://valgrind.org/>`_ Memcheck tool to help verify that user
:ref:`CeedQFunction` have no undefined values).
This release also included various performance improvements, bug fixes, new examples,
and improved tests. Among these improvements, vectorized instructions for
:ref:`CeedQFunction` code compiled for CPU were enhanced by using ``CeedPragmaSIMD``
instead of ``CeedPragmaOMP``, implementation of a :ref:`CeedQFunction` gallery and
identity Q-Functions were introduced, and the PETSc benchmark problems were expanded
to include unstructured meshes handling were. For this expansion, the prior version of
the PETSc BPs, which only included data associated with structured geometries, were
renamed ``bpsraw``, and the new version of the BPs, which can handle data associated
with any unstructured geometry, were called ``bps``. Additionally, other benchmark
problems, namely BP2 and BP4 (the vector-valued versions of BP1 and BP3, respectively),
and BP5 and BP6 (the collocated versions---for which the quadrature points are the same
as the Gauss Lobatto nodes---of BP3 and BP4 respectively) were added to the PETSc
examples. Furthermoew, another standalone libCEED example, called ``ex2``, which
computes the surface area of a given mesh was added to this release.

Backends available in this release:

+----------------------------+-----------------------------------------------------+
| CEED resource (``-ceed``)  | Backend                                             |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/ref/serial``   | Serial reference implementation                     |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/ref/blocked``  | Blocked reference implementation                    |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/ref/memcheck`` | Memcheck backend, undefined value checks            |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/opt/serial``   | Serial optimized C implementation                   |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/opt/blocked``  | Blocked optimized C implementation                  |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/avx/serial``   | Serial AVX implementation                           |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/avx/blocked``  | Blocked AVX implementation                          |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/xsmm/serial``  | Serial LIBXSMM implementation                       |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/xsmm/blocked`` | Blocked LIBXSMM implementation                      |
+----------------------------+-----------------------------------------------------+
| ``/cpu/occa``              | Serial OCCA kernels                                 |
+----------------------------+-----------------------------------------------------+
| ``/gpu/occa``              | CUDA OCCA kernels                                   |
+----------------------------+-----------------------------------------------------+
| ``/omp/occa``              | OpenMP OCCA kernels                                 |
+----------------------------+-----------------------------------------------------+
| ``/ocl/occa``              | OpenCL OCCA kernels                                 |
+----------------------------+-----------------------------------------------------+
| ``/gpu/cuda/ref``          | Reference pure CUDA kernels                         |
+----------------------------+-----------------------------------------------------+
| ``/gpu/cuda/reg``          | Pure CUDA kernels using one thread per element      |
+----------------------------+-----------------------------------------------------+
| ``/gpu/cuda/shared``       | Optimized pure CUDA kernels using shared memory     |
+----------------------------+-----------------------------------------------------+
| ``/gpu/cuda/gen``          | Optimized pure CUDA kernels using code generation   |
+----------------------------+-----------------------------------------------------+
| ``/gpu/magma``             | CUDA MAGMA kernels                                  |
+----------------------------+-----------------------------------------------------+

Examples available in this release:

+-------------------------+--------------------------------------------+
| User code               | Example                                    |
+-------------------------+--------------------------------------------+
|                         | - ex1 (volume)                             |
| ``ceed``                | - ex2 (surface)                            |
+-------------------------+--------------------------------------------+
|                         | - BP1 (scalar mass operator)               |
| ``mfem``                | - BP3 (scalar Laplace operator)            |
+-------------------------+--------------------------------------------+
|                         | - BP1 (scalar mass operator)               |
|                         | - BP2 (vector mass operator)               |
|                         | - BP3 (scalar Laplace operator)            |
| ``petsc``               | - BP4 (vector Laplace operator)            |
|                         | - BP5 (collocated scalar Laplace operator) |
|                         | - BP6 (collocated vector Laplace operator) |
|                         | - Navier-Stokes                            |
+-------------------------+--------------------------------------------+
|                         | - BP1 (scalar mass operator)               |
| ``nek5000``             | - BP3 (scalar Laplace operator)            |
+-------------------------+--------------------------------------------+


.. _v0.4:

v0.4 (Apr 1, 2019)
----------------------------------------

libCEED v0.4 was made again publicly available in the second full CEED software
distribution, release CEED 2.0. This release contained notable features, such as
four new CPU backends, two new GPU backends, CPU backend optimizations, initial
support for operator composition, performance benchmarking, and a Navier-Stokes demo.
The new CPU backends in this release came in two families. The ``/cpu/self/*/serial``
backends process one element at a time and are intended for meshes with a smaller number
of high order elements. The ``/cpu/self/*/blocked`` backends process blocked batches of
eight interlaced elements and are intended for meshes with higher numbers of elements.
The ``/cpu/self/avx/*`` backends rely upon AVX instructions to provide vectorized CPU
performance. The ``/cpu/self/xsmm/*`` backends rely upon the
`LIBXSMM <http://github.com/hfp/libxsmm>`_ package to provide vectorized CPU
performance. The ``/gpu/cuda/*`` backends provide GPU performance strictly using CUDA.
The ``/gpu/cuda/ref`` backend is a reference CUDA backend, providing reasonable
performance for most problem configurations. The ``/gpu/cuda/reg`` backend uses a simple
parallelization approach, where each thread treats a finite element. Using just in time
compilation, provided by nvrtc (NVidia Runtime Compiler), and runtime parameters, this
backend unroll loops and map memory address to registers. The ``/gpu/cuda/reg`` backend
achieve good peak performance for 1D, 2D, and low order 3D problems, but performance
deteriorates very quickly when threads run out of registers.

A new explicit time-stepping Navier-Stokes solver was added to the family of libCEED
examples in the ``examples/petsc`` directory (see :ref:`example-petsc-navier-stokes`).
This example solves the time-dependent Navier-Stokes equations of compressible gas
dynamics in a static Eulerian three-dimensional frame, using structured high-order
finite/spectral element spatial discretizations and explicit high-order time-stepping
(available in PETSc). Moreover, the Navier-Stokes example was developed using PETSc,
so that the pointwise physics (defined at quadrature points) is separated from the
parallelization and meshing concerns.

Backends available in this release:

+----------------------------+-----------------------------------------------------+
| CEED resource (``-ceed``)  | Backend                                             |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/ref/serial``   | Serial reference implementation                     |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/ref/blocked``  | Blocked reference implementation                    |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/tmpl``         | Backend template, defaults to ``/cpu/self/blocked`` |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/avx/serial``   | Serial AVX implementation                           |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/avx/blocked``  | Blocked AVX implementation                          |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/xsmm/serial``  | Serial LIBXSMM implementation                       |
+----------------------------+-----------------------------------------------------+
| ``/cpu/self/xsmm/blocked`` | Blocked LIBXSMM implementation                      |
+----------------------------+-----------------------------------------------------+
| ``/cpu/occa``              | Serial OCCA kernels                                 |
+----------------------------+-----------------------------------------------------+
| ``/gpu/occa``              | CUDA OCCA kernels                                   |
+----------------------------+-----------------------------------------------------+
| ``/omp/occa``              | OpenMP OCCA kernels                                 |
+----------------------------+-----------------------------------------------------+
| ``/ocl/occa``              | OpenCL OCCA kernels                                 |
+----------------------------+-----------------------------------------------------+
| ``/gpu/cuda/ref``          | Reference pure CUDA kernels                         |
+----------------------------+-----------------------------------------------------+
| ``/gpu/cuda/reg``          | Pure CUDA kernels using one thread per element      |
+----------------------------+-----------------------------------------------------+
| ``/gpu/magma``             | CUDA MAGMA kernels                                  |
+----------------------------+-----------------------------------------------------+

Examples available in this release:

+-------------------------+---------------------------------+
| User code               | Example                         |
+-------------------------+---------------------------------+
| ``ceed``                | ex1 (volume)                    |
+-------------------------+---------------------------------+
|                         | - BP1 (scalar mass operator)    |
| ``mfem``                | - BP3 (scalar Laplace operator) |
+-------------------------+---------------------------------+
|                         | - BP1 (scalar mass operator)    |
| ``petsc``               | - BP3 (scalar Laplace operator) |
|                         | - Navier-Stokes                 |
+-------------------------+---------------------------------+
|                         | - BP1 (scalar mass operator)    |
| ``nek5000``             | - BP3 (scalar Laplace operator) |
+-------------------------+---------------------------------+


.. _v0.3:

v0.3 (Sep 30, 2018)
----------------------------------------

Notable features in this release include active/passive field interface, support for
non-tensor bases, backend optimization, and improved Fortran interface. This release
also focused on providing improved continuous integration, and many new tests with code
coverage reports of about 90%. This release also provided a significant change to the
public interface: a :ref:`CeedQFunction` can take any number of named input and output
arguments while :ref:`CeedOperator` connects them to the actual data, which may be
supplied explicitly to ``CeedOperatorApply()`` (active) or separately via
``CeedOperatorSetField()`` (passive). This interface change enables reusable libraries
of CeedQFunctions and composition of block solvers constructed using
:ref:`CeedOperator`. A concept of blocked restriction was added to this release and
used in an optimized CPU backend. Although this is typically not visible to the user,
it enables effective use of arbitrary-length SIMD while maintaining cache locality.
This CPU backend also implements an algebraic factorization of tensor product gradients
to perform fewer operations than standard application of interpolation and
differentiation from nodes to quadrature points. This algebraic formulation
automatically supports non-polynomial and non-interpolatory bases, thus is more general
than the more common derivation in terms of Lagrange polynomials on the quadrature points.

Backends available in this release:

+---------------------------+-----------------------------------------------------+
| CEED resource (``-ceed``) | Backend                                             |
+---------------------------+-----------------------------------------------------+
| ``/cpu/self/blocked``     | Blocked reference implementation                    |
+---------------------------+-----------------------------------------------------+
| ``/cpu/self/ref``         | Serial reference implementation                     |
+---------------------------+-----------------------------------------------------+
| ``/cpu/self/tmpl``        | Backend template, defaults to ``/cpu/self/blocked`` |
+---------------------------+-----------------------------------------------------+
| ``/cpu/occa``             | Serial OCCA kernels                                 |
+---------------------------+-----------------------------------------------------+
| ``/gpu/occa``             | CUDA OCCA kernels                                   |
+---------------------------+-----------------------------------------------------+
| ``/omp/occa``             | OpenMP OCCA kernels                                 |
+---------------------------+-----------------------------------------------------+
| ``/ocl/occa``             | OpenCL OCCA kernels                                 |
+---------------------------+-----------------------------------------------------+
| ``/gpu/magma``            | CUDA MAGMA kernels                                  |
+---------------------------+-----------------------------------------------------+

Examples available in this release:

+-------------------------+---------------------------------+
| User code               | Example                         |
+-------------------------+---------------------------------+
| ``ceed``                | ex1 (volume)                    |
+-------------------------+---------------------------------+
|                         | - BP1 (scalar mass operator)    |
| ``mfem``                | - BP3 (scalar Laplace operator) |
+-------------------------+---------------------------------+
|                         | - BP1 (scalar mass operator)    |
| ``petsc``               | - BP3 (scalar Laplace operator) |
+-------------------------+---------------------------------+
|                         | - BP1 (scalar mass operator)    |
| ``nek5000``             | - BP3 (scalar Laplace operator) |
+-------------------------+---------------------------------+


.. _v0.21:

v0.21 (Sep 30, 2018)
----------------------------------------

A MAGMA backend (which relies upon the
`MAGMA <https://bitbucket.org/icl/magma>`_ package) was integrated in libCEED for this
release. This initial integration set up the framework of using MAGMA and provided the
libCEED functionality through MAGMA kernels as one of libCEED’s computational backends.
As any other backend, the MAGMA backend provides extended basic data structures for
:ref:`CeedVector`, :ref:`CeedElemRestriction`, and :ref:`CeedOperator`, and implements
the fundamental CEED building blocks to work with the new data structures.
In general, the MAGMA-specific data structures keep the libCEED pointers to CPU data
but also add corresponding device (e.g., GPU) pointers to the data. Coherency is handled
internally, and thus seamlessly to the user, through the functions/methods that are
provided to support them.

Backends available in this release:

+---------------------------+---------------------------------+
| CEED resource (``-ceed``) | Backend                         |
+---------------------------+---------------------------------+
| ``/cpu/self``             | Serial reference implementation |
+---------------------------+---------------------------------+
| ``/cpu/occa``             | Serial OCCA kernels             |
+---------------------------+---------------------------------+
| ``/gpu/occa``             | CUDA OCCA kernels               |
+---------------------------+---------------------------------+
| ``/omp/occa``             | OpenMP OCCA kernels             |
+---------------------------+---------------------------------+
| ``/ocl/occa``             | OpenCL OCCA kernels             |
+---------------------------+---------------------------------+
| ``/gpu/magma``            | CUDA MAGMA kernels              |
+---------------------------+---------------------------------+

Examples available in this release:

+-------------------------+---------------------------------+
| User code               | Example                         |
+-------------------------+---------------------------------+
| ``ceed``                | ex1 (volume)                    |
+-------------------------+---------------------------------+
|                         | - BP1 (scalar mass operator)    |
| ``mfem``                | - BP3 (scalar Laplace operator) |
+-------------------------+---------------------------------+
| ``petsc``               | BP1 (scalar mass operator)      |
+-------------------------+---------------------------------+
| ``nek5000``             | BP1 (scalar mass operator)      |
+-------------------------+---------------------------------+


.. _v0.2:

v0.2 (Mar 30, 2018)
----------------------------------------

libCEED was made publicly available the first full CEED software distribution, release
CEED 1.0. The distribution was made available using the Spack package manager to provide
a common, easy-to-use build environment, where the user can build the CEED distribution
with all dependencies. This release included a new Fortran interface for the library.
This release also contained major improvements in the OCCA backend (including a new
``/ocl/occa`` backend) and new examples. The standalone libCEED example was modified to
compute the volume volume of a given mesh (in 1D, 2D, or 3D) and placed in an
``examples/ceed`` subfolder. A new ``mfem`` example to perform BP3 (with the application
of the Laplace operator) was also added to this release.

Backends available in this release:

+---------------------------+---------------------------------+
| CEED resource (``-ceed``) | Backend                         |
+---------------------------+---------------------------------+
| ``/cpu/self``             | Serial reference implementation |
+---------------------------+---------------------------------+
| ``/cpu/occa``             | Serial OCCA kernels             |
+---------------------------+---------------------------------+
| ``/gpu/occa``             | CUDA OCCA kernels               |
+---------------------------+---------------------------------+
| ``/omp/occa``             | OpenMP OCCA kernels             |
+---------------------------+---------------------------------+
| ``/ocl/occa``             | OpenCL OCCA kernels             |
+---------------------------+---------------------------------+

Examples available in this release:

+-------------------------+---------------------------------+
| User code               | Example                         |
+-------------------------+---------------------------------+
| ``ceed``                | ex1 (volume)                    |
+-------------------------+---------------------------------+
|                         | - BP1 (scalar mass operator)    |
| ``mfem``                | - BP3 (scalar Laplace operator) |
+-------------------------+---------------------------------+
| ``petsc``               | BP1 (scalar mass operator)      |
+-------------------------+---------------------------------+
| ``nek5000``             | BP1 (scalar mass operator)      |
+-------------------------+---------------------------------+


.. _v0.1:

v0.1 (Jan 3, 2018)
----------------------------------------

Initial low-level API of the CEED project. The low-level API provides a set of Finite
Elements kernels and components for writing new low-level kernels. Examples include:
vector and sparse linear algebra, element matrix assembly over a batch of elements,
partial assembly and action for efficient high-order operators like mass, diffusion,
advection, etc. The main goal of the low-level API is to establish the basis for the
high-level API. Also, identifying such low-level kernels and providing a reference
implementation for them serves as the basis for specialized backend implementations.
This release contained several backends: ``/cpu/self``, and backends which rely upon the
`OCCA <http://github.com/libocca/occa>`_ package, such as ``/cpu/occa``,
``/gpu/occa``, and ``/omp/occa``.
It also included several examples, in the ``examples`` folder:
A standalone code that shows the usage of libCEED (with no external
dependencies) to apply the Laplace operator, ``ex1``; an ``mfem`` example to perform BP1
(with the application of the mass operator); and a ``petsc`` example to perform BP1
(with the application of the mass operator).

Backends available in this release:

+---------------------------+---------------------------------+
| CEED resource (``-ceed``) | Backend                         |
+---------------------------+---------------------------------+
| ``/cpu/self``             | Serial reference implementation |
+---------------------------+---------------------------------+
| ``/cpu/occa``             | Serial OCCA kernels             |
+---------------------------+---------------------------------+
| ``/gpu/occa``             | CUDA OCCA kernels               |
+---------------------------+---------------------------------+
| ``/omp/occa``             | OpenMP OCCA kernels             |
+---------------------------+---------------------------------+

Examples available in this release:

+-------------------------+-----------------------------------+
| User code               | Example                           |
+-------------------------+-----------------------------------+
| ``ceed``                | ex1 (scalar Laplace operator)     |
+-------------------------+-----------------------------------+
| ``mfem``                | BP1 (scalar mass operator)        |
+-------------------------+-----------------------------------+
| ``petsc``               | BP1 (scalar mass operator)        |
+-------------------------+-----------------------------------+

