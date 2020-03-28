libCEED: Examples
==================================================
This page provides a brief description of the examples for the libCEED
library.


Basic libCEED Examples
--------------------------------------------------

Two examples are provided that rely only upon libCEED without any
external libraries.


Example 1: ex1-volume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses the mass matrix to compute the length, area, or volume
of a region, depending upon runtime parameters.


Example 2: ex2-surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses the diffusion matrix to compute the surface area of a
region, depending upon runtime parameters.


Bakeoff Problems
--------------------------------------------------

This section provides a brief description of the bakeoff problems, used
as examples for the libCEED library. These bakeoff problems are
high-order benchmarks designed to test and compare the performance of
high-order finite element codes.

For further documentation, readers may wish to consult the `CEED
documentation <http://ceed.exascaleproject.org/bps/>`__ of the bakeoff
problems.


Bakeoff Problem 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bakeoff problem 1 is the *L2* projection problem into the finite element
space.

The supplied examples solve *B u = f*, where *B* is the mass matrix.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature
points, *q* are Gauss-Legendre. There is one more quadrature point in
each dimension than nodal point, *q = p + 1*.


Bakeoff Problem 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bakeoff problem 2 is the *L2* projection problem into the finite element
space on a vector system.

The supplied examples solve *B u = f*, where *B* is the mass matrix.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature
points, *q* are Gauss-Legendre. There is one more quadrature point in
each dimension than nodal point, *q = p + 1*.


Bakeoff Problem 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bakeoff problem 3 is the Poisson problem.

The supplied examples solve *A u = f*, where *A* is the Poisson
operator.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature
points, *q* are Gauss-Legendre. There is one more quadrature point in
each dimension than nodal point, *q = p + 1*.


Bakeoff Problem 4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bakeoff problem 4 is the Poisson problem on a vector system.

The supplied examples solve *A u = f*, where *A* is the Laplace operator
for the Poisson equation.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature
points, *q* are Gauss-Legendre. There is one more quadrature point in
each dimension than nodal point, *q = p + 1*.


Bakeoff Problem 5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bakeoff problem 5 is the Poisson problem.

The supplied examples solve *A u = f*, where *A* is the Poisson
operator.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature
points, *q* are Gauss-Legendre-Lobatto. The nodal points and quadrature
points are collocated.


Bakeoff Problem 6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bakeoff problem 6 is the Poisson problem on a vector system.

The supplied examples solve *A u = f*, where *A* is the Laplace operator
for the Poisson equation.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature
points, *q* are Gauss-Legendre-Lobatto. The nodal points and quadrature
points are collocated.


PETSc+libCEED Navier-Stokes Solver
--------------------------------------------------

The Navier-Stokes problem solves the compressible Navier-Stokes
equations using an explicit time integration. A more detailed
description of the problem formulation can be found in the
`fluids <./fluids>`_ folder.


PETSc+libCEED Surface Area Examples
--------------------------------------------------

These examples use the mass operator to compute the surface area of a
cube or a discrete cubed-sphere, using PETSc.

These examples show in particular the constructions of geometric factors
to handle problems in which the elements topological dimension is
different from the geometrical dimension and for which the coordinate
transformation Jacobian from the 2D reference space to a manifold
embedded in 3D physical space is a non-square matrix.


PETSc+libCEED Bakeoff Problems on the Cubed-Sphere
--------------------------------------------------

These examples reproduce the Bakeoff Problems 1-6 on a discrete
cubed-sphere, using PETSc.


Running Examples
--------------------------------------------------

To build the examples, set the ``MFEM_DIR``, ``PETSC_DIR`` and
``NEK5K_DIR`` variables and, from the `examples/` directory, run

.. include:: ../README.rst
   :start-after: running-examples-inclusion-marker
   :end-before: benchmarks-marker
