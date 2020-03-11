.. _bps:

CEED Bakeoff Problems
========================================

The Center for Efficient Exascale Discretizations (CEED) uses Bakeoff Problems (BPs)
to test and compare the performance of high-order finite element implementations. The
definitions of the problems are given on the ceed
`website <https://ceed.exascaleproject.org/bps/>`_. Each of the following bakeoff
problems that use external discretization libraries (such as MFEM, PETSc, and Nek5000)
are located in the subdirectories :file:`examples/mfem`, :file:`examples/petsc`, and
:file:`examples/nek5000`, respectively.

Here we provide a short summary:

+-------------------------+----------------------------------------------------------------+
| User code               | BPs                                                            |
+-------------------------+----------------------------------------------------------------+
|                         | - BP1 (scalar mass operator), with :math:`Q=P+1`               |
| ``mfem``                | - BP3 (scalar Laplace operator), with :math:`Q=P+1`            |
+-------------------------+----------------------------------------------------------------+
|                         | - BP1 (scalar mass operator), with :math:`Q=P+1`               |
|                         | - BP2 (vector mass operator), with :math:`Q=P+1`               |
|                         | - BP3 (scalar Laplace operator), with :math:`Q=P+1`            |
| ``petsc``               | - BP4 (vector Laplace operator), with :math:`Q=P+1`            |
|                         | - BP5 (collocated scalar Laplace operator), with :math:`Q=P`   |
|                         | - BP6 (collocated vector Laplace operator), with :math:`Q=P`   |
+-------------------------+----------------------------------------------------------------+
|                         | - BP1 (scalar mass operator), with :math:`Q=P+1`               |
| ``nek5000``             | - BP3 (scalar Laplace operator), with :math:`Q=P+1`            |
+-------------------------+----------------------------------------------------------------+

These are all **T-vector**-to-**T-vector** and include parallel scatter, element
scatter, element evaluation kernel, element gather, and parallel gather (with the
parallel gathers/scatters done externally to libCEED).

BP1 and BP2 are :math:`L^2` projections, and thus have no boundary condition.
The rest of the BPs have homogeneous Dirichlet boundary conditions.

The BPs are parametrized by the number :math:`P` of Gauss-Legendre-Lobatto nodal points (with :math:`P=p+1`, and :math:`p` the degree of the basis polynomial) for the Lagrange polynomials, as well as the number of quadrature points, :math:`Q`.
A :math:`Q`-point Gauss-Legendre quadrature is used for all BPs except BP5 and BP6, which choose :math:`Q = P` and Gauss-Legendre-Lobatto quadrature to collocate with the interpolation nodes.
This latter choice is popular in applications that use spectral element methods because it produces a diagonal mass matrix (enabling easy explicit time integration) and significantly reduces the number of floating point operations to apply the operator.

