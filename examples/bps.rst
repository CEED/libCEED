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

The BPs are parametrized by the number :math:`P` of Gauss-Legendre-Lobatto nodal points
(with :math:`P=p+1`, and :math:`p` the degree of the basis polynomial) for the Lagrange
polynomials, as well as the number of quadrature points, :math:`Q`.
A :math:`Q`-point Gauss-Legendre quadrature is used for all BPs except BP5 and BP6,
which choose :math:`Q = P` and Gauss-Legendre-Lobatto quadrature to collocate with the
interpolation nodes. This latter choice is popular in applications that use spectral
element methods because it produces a diagonal mass matrix (enabling easy explicit
time integration) and significantly reduces the number of floating point operations
to apply the operator.


.. _Mass Operator:

Mass Operator
----------------------------------------

The Mass Operator used in BP1 and BP2 is defined via the :math:`L^2` projection
problem, posed as a weak form on a Hilbert space :math:`V^p \subset H^1`, i.e.,
find :math:`u \in V^p` such that for all :math:`v \in V^p`

.. math::
   :label: eq-general-weak-form

   \langle u,v \rangle = \langle f,v \rangle ,

where :math:`\langle u,v\rangle` and :math:`\langle f,v\rangle` express the continuous
bilinear and linear forms, respectively, defined on :math:`V^p`, and, for sufficiently
regular :math:`u`, :math:`v`, and :math:`f`, we have:

.. math::
   \begin{aligned}
   \langle v,u \rangle &:= \int_{\Omega} \, v \, u \, dV ,\\
   \langle v,f \rangle &:= \int_{\Omega} \, v \, f \, dV .
   \end{aligned}

Following the standard finite/spectral element approach, we formally
expand all functions in terms of basis functions, such as

.. math::
   :label: eq-nodal-values

   \begin{aligned}
   u(\bm x) &= \sum_{j=1}^n u_j \, \phi_j(\bm x) ,\\
   v(\bm x) &= \sum_{i=1}^n v_i \, \phi_i(\bm x) .
   \end{aligned}

The coefficients :math:`\{u_j\}` and :math:`\{v_i\}` are the nodal values of :math:`u`
and :math:`v`, respectively. Inserting the expressions :math:numref:`eq-nodal-values`
into :math:numref:`eq-general-weak-form`, we obtain the inner-products

.. math::
   :label: eq-inner-prods

   \langle v,u \rangle = \bm v^T M \bm u , \qquad  \langle f,v\rangle =  \bm v^T \bm b \,.

Here, we have introduced the mass matrix, :math:`M`, and the right-hand side,
:math:`\bm b`,

.. math::
   M_{ij} :=  (\phi_i,\phi_j), \;\; \qquad b_{i} :=  \langle f, \phi_i \rangle,

each defined for index sets :math:`i,j \; \in \; \{1,\dots,n\}`.


.. _Laplace Operator:

Laplace's Operator
----------------------------------------

The Laplace's operator used in BP3-BP6 is defined via the following variational
formulation, i.e., find :math:`u \in V^p` such that for all :math:`v \in V^p`

.. math::
   a(u,v) = \langle f,v \rangle , \,

where now :math:`a (u,v)` expresses the continuous bilinear form defined on
:math:`V^p` for sufficiently regular :math:`u`, :math:`v`, and :math:`f`, that is:

.. math::
   \begin{aligned}
   a(v,u) &:= \int_{\Omega}\nabla v \, \cdot \, \nabla u \, dV ,\\
   \langle v,f \rangle &:= \int_{\Omega} \, v \, f \, dV .
   \end{aligned}

After substituting the same formulations provided in :math:numref:`eq-nodal-values`,
we obtain

.. math::
   a(v,u) = \bm v^T K \bm u ,

in which we have introduced the stiffness (diffusion) matrix, :math:`K`, defined as

.. math::
   K_{ij} = a(\phi_i,\phi_j),

for index sets :math:`i,j \; \in \; \{1,\dots,n\}`.
