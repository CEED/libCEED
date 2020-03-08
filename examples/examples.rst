.. _Examples:

Examples
**************************************

This section contains a mathematical description of all examples provided with libCEED
in the :file:`examples/` directory.

For most of our examples, the spatial discretization
uses high-order finite elements/spectral elements, namely, the high-order Lagrange
polynomials defined over :math:`P` non-uniformly spaced nodes, the
Legendre-Gauss-Lobatto (LGL) points (roots of the :math:`p^{th}`-order Legendre
polynomial :math:`P_p`), and quadrature points :math:`\{q_i\}_{i=0}^Q`, with
corresponding weights :math:`\{w_i\}_{i=0}^Q` (typically the ones given by Gauss
or Gauss-Lobatto quadratures, that are built in the library).

We discretize the domain, :math:`\Omega \subset \mathbb{R}^d` (with :math:`d=1,2,3`,
typically) by letting :math:`\Omega = \bigcup_{e=1}^{N_e}\Omega_e`, with :math:`N_e`
disjoint elements. For most examples we use unstructured meshes for which the elements
are hexaedra (although this is not a requirement in libCEED).

The physical coordinates are denoted by :math:`\mathbf{x}=(x,y,z)\in\Omega_e`,
while the reference coordinates are represented as
:math:`\boldsymbol{X}=(X,Y,Z) \equiv (X_1,X_2,X_3) \in\mathbf{I}=[-1,1]^3`
(for :math:`d=3`).


Standalone libCEED
======================================

The following two examples have no dependencies, and are designed to be self-contained.
For additional examples that use external discretization libraries (MFEM, PETSc, Nek5000
etc.) see the subdirectories in :file:`examples/`.


.. _ex1-volume:

Ex1-Volume
--------------------------------------

This example is located in the subdirectory :file:`examples/ceed`. It illustrates a
simple usage of libCEED to compute the volume of a given body using a matrix-free
application of the mass operator. Arbitrary mesh and solution orders in 1D, 2D and 3D
are supported from the same code.

This example shows how to compute line/surface/volume integrals of a 1D, 2D, or 3D
domain :math:`\Omega` respectively, by applying the mass operator to a vector of
:math:`\mathbf{1}`\s. It computes:

.. math::
   I = \int_{\Omega} \mathbf{1} \, d V\, .
   :label: eq-ex1-volume

Using the same notation as in :ref:`Theoretical Framework`, we write here the vector
:math:`u(\mathbf{x})\equiv \mathbf{1}` in the Galerkin approximation,
and find the volume of :math:`\Omega` as

.. math::
   \sum_e \int_{\Omega_e} v(x) \cdot \mathbf{1} \, dV

with :math:`v(x) \in \mathcal{V}_p = \{ v \in H^{1}(\Omega_e) \,|\, v \in P_p(\boldsymbol{I}), e=1,\ldots,N_e \}`,
the test functions.


.. _ex2-surface:

Ex2-Surface
--------------------------------------

This example is located in the subdirectory :file:`examples/ceed`. It computes the
surface area of a given body using matrix-free application of a diffusion operator.
Arbitrary mesh and solution orders in 1D, 2D and 3D are supported from the same code.

Similarly to :ref:`Ex1-Volume`, it computes:

.. math::
   I = \int_{\partial \Omega} \mathbf{1} \, d S\, .
   :label: eq-ex2-surface

but this time by solving a Laplace's equation for a harmonic function
:math:`u(\mathbf{x})`. We write the Laplace's equation

.. math::
   \nabla \cdot \nabla u = 0, \textrm{ for  } \mathbf{x} \in \Omega.
   :label: eq-laplace

We can rewrite this via the bilinear form

.. math::
   B(u,v) = L(v)

where :math:`v` is the test function, and for which :math:`L(v)=0` in this case. We
obtain

.. math::
   B(u,v)  = \int_\Omega v \nabla \cdot \nabla u \, d V =   \int_{\partial \Omega} v \nabla u \cdot \mathbf{n}\, d S - \int_\Omega \nabla v \cdot \nabla u \, d V  = 0\, ,

where we have used integration by parts.

:math:`B(u,v) = 0` because we have chosen :math:`u(\mathbf{x})` to be harmonic, so we
can write

.. math::
   \int_{\partial \Omega} v \nabla u \cdot \mathbf{n}\, d S = \int_\Omega \nabla v \cdot \nabla u \, d V
   :label: eq-laplace-by-parts

and use the :ref:`CeedOperator` for Laplace's operator to compute the right-hand side of
equation :math:numref:`eq-laplace-by-parts`. This way, the left-hand side of equation
:math:numref:`eq-laplace-by-parts` (which gives :math:numref:`eq-ex2-surface` because
we have chosen :math:`u(\mathbf{x}) = (x + y + z)` such that  :math:`\nabla u \cdot \mathbf{n} = 1`)
is readily found.


PETSc
======================================

.. _example-petsc-area:

Area
--------------------------------------

This example is located in the subdirectory :file:`examples/petsc`. It
demonstrates a simple usage of libCEED with PETSc to calculate
the surface area of a closed surface. The code uses higher level
communication protocols for mesh handling in PETSc's DMPlex. This example has the
same mathematical formulation as :ref:`Ex1-Volume`, with the exception that the
physical coordinates for this problem are :math:`\mathbf{x}=(x,y,z)\in \mathbb{R}^3`,
while the coordinates of the reference element are
:math:`\boldsymbol{X}=(X,Y) \equiv (X_1,X_2) \in\mathbf{I}=[-1,1]^2`.


.. _example-petsc-area-cube:

Cube
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is one of the test cases of the computation of the :ref:`example-petsc-area` of a
2D manifold embedded in 3D. This problem can be run with::

   ./area -problem cube

This example uses the following coordinate transformations for the computation of the
geometric factors: from the physical coordinates on the cube, denoted by
:math:`\bar{\mathbf{x}}=(\bar{x},\bar{y},\bar{z})`,
and physical coordinates on the discrete surface, denoted by
:math:`\mathbf{{x}}=(x,y)`, to :math:`\mathbf{X}=(X,Y) \in\mathbf{I}=[-1,1]^2` on the
reference element, via the chain rule

.. math::
   \frac{\partial \mathbf{x}}{\partial \mathbf{X}}_{(2\times2)} = \frac{\partial {\mathbf{x}}}{\partial \bar{\mathbf{x}}}_{(2\times3)} \frac{\partial \bar{\mathbf{x}}}{\partial \mathbf{X}}_{(3\times2)} \, ,
   :label: eq-coordinate-transforms-cube

with Jacobian determinant given by

.. math::
   \left| J \right| = \left\|col_1\left(\frac{\partial \bar{\mathbf{x}}}{\partial \mathbf{X}}\right)\right\| \left\|col_2 \left(\frac{\partial \bar{\mathbf{x}}}{\partial \mathbf{X}}\right) \right\|
   :label: eq-jacobian-cube

We note that in equation :math:numref:`eq-coordinate-transforms-cube`, the right-most
Jacobian matrix :math:`{\partial\bar{\mathbf{x}}}/{\partial \mathbf{X}}_{(3\times2)}` is
provided by the library, while
:math:`{\partial{\mathbf{x}}}/{\partial \bar{ \mathbf{x}}}_{(2\times3)}` is
provided by the user as

.. math::
   \left[ col_1\left(\frac{\partial\bar{\mathbf{x}}}{\partial \mathbf{X}}\right) / \left\| col_1\left(\frac{\partial\bar{\mathbf{x}}}{\partial \mathbf{X}}\right)\right\| , col_2\left(\frac{\partial\bar{\mathbf{x}}}{\partial \mathbf{X}}\right) / \left\| col_2\left(\frac{\partial\bar{\mathbf{x}}}{\partial \mathbf{X}}\right)\right\| \right]^T_{(2\times 3)}.


.. _example-petsc-area-sphere:

Sphere
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This problem computes the surface :ref:`example-petsc-area` of a tensor-product
discrete sphere, obtained by projecting a cube inscribed in a sphere onto the surface
of the sphere (this discrete surface is sometimes referred to as a cubed-sphere).
This problem can be run with::

   ./area -problem sphere

This example uses the following coordinate transformations for the computation of the
geometric factors: from the physical coordinates on the sphere, denoted by
:math:`\overset{\circ}{\mathbf{x}}=(\overset{\circ}{x},\overset{\circ}{y},\overset{\circ}{z})`,
and physical coordinates on the discrete surface, denoted by
:math:`\mathbf{{x}}=(x,y,z)`, to :math:`\mathbf{X}=(X,Y) \in\mathbf{I}=[-1,1]^2` on the
reference element, via the chain rule

.. math::
   \frac{\partial \overset{\circ}{\mathbf{x}}}{\partial \mathbf{X}}_{(3\times2)} = \frac{\partial \overset{\circ}{\mathbf{x}}}{\partial \mathbf{x}}_{(3\times3)} \frac{\partial\mathbf{x}}{\partial \mathbf{X}}_{(3\times2)} \, ,
   :label: eq-coordinate-transforms-sphere

with Jacobian determinant given by

.. math::
   \left| J \right| = \left| col_1\left(\frac{\partial \overset{\circ}{\mathbf{x}}}{\partial \mathbf{X}}\right) \times col_2 \left(\frac{\partial \overset{\circ}{\mathbf{x}}}{\partial \mathbf{X}}\right)\right| .
   :label: eq-jacobian-sphere

We note that in equation :math:numref:`eq-coordinate-transforms-sphere`, the right-most
Jacobian matrix :math:`{\partial\mathbf{x}}/{\partial \mathbf{X}}_{(3\times2)}` is
provided by the library, while
:math:`{\partial \overset{\circ}{\mathbf{x}}}/{\partial \mathbf{x}}_{(3\times3)}` is
provided by the user with analytical derivatives.


.. _example-petsc-multigrid:

Multigrid
--------------------------------------

This example is located in the subdirectory :file:`examples/petsc`. It
investigates :math:`p`-multigrid for the Poisson problem, equation
:math:numref:`eq-variable-coeff-poisson`, using an unstructured high-order finite
element discretization. All of the operators associated with the geometric multigrid
are implemented in libCEED.

.. math::
   -\nabla\cdot \left( \kappa \left( x \right) \nabla x \right) = g \left( x \right)
   :label: eq-variable-coeff-poisson

The Poisson operator can be specified with the decomposition given by the equation in
figure :ref:`fig-operator-decomp`, and the restriction and prolongation operators given
by interpolation basis operations, :math:`\mathbf{B}`, and :math:`\mathbf{B}^T`,
respectively, act on the different grid levels with corresponding element restrictions,
:math:`\mathbf{G}`. These three operations can be exploited by existing matrix-free
multigrid software and smoothers. Preconditioning based on the libCEED finite element
operator decomposition is an ongoing area of research.


.. _example-petsc-navier-stokes:

Navier-Stokes
--------------------------------------

This example is located in the subdirectory :file:`examples/navier-stokes`. It solves
the time-dependent Navier-Stokes equations of compressible gas dynamics in a static
Eulerian three-dimensional frame using structured high-order finite element/spectral
element spatial discretizations and explicit high-order time-stepping (available in
PETSc). Moreover, the Navier-Stokes example has been developed using PETSc, so that the
pointwise physics (defined at quadrature points) is separated from the parallelization
and meshing concerns.

The mathematical formulation is given in what follows. The compressible Navier-Stokes
equations in conservative form are

.. math::
   :label: eq-ns

   \frac{\partial \rho}{\partial t} + \nabla \cdot \boldsymbol{U} &= 0

   \frac{\partial \boldsymbol{U}}{\partial t} + \nabla \cdot \left( \frac{\boldsymbol{U} \otimes \boldsymbol{U}}{\rho} + P \mathbf{I}_3 -\boldsymbol\sigma \right) &= -\rho g \boldsymbol{\hat k}

   \frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{(E + P)\boldsymbol{U}}{\rho} -\boldsymbol{u} \cdot \boldsymbol{\sigma} - k \nabla T \right) &= 0 \, ,

where :math:`\boldsymbol{\sigma} = \mu(\nabla \boldsymbol{u} + (\nabla \boldsymbol{u})^T + \lambda (\nabla \cdot \boldsymbol{u})\mathbf{I}_3)`
is the Cauchy (symmetric) stress tensor, with :math:`\mu` the dynamic viscosity
coefficient, and :math:`\lambda = - 2/3` the Stokes hypothesis constant. In equations
:math:numref:`eq-ns`, :math:`\rho` represents the volume mass density, :math:`U` the
momentum density (defined as :math:`\boldsymbol{U}=\rho \boldsymbol{u}`, where
:math:`\boldsymbol{u}` is the vector velocity field), :math:`E` the total energy
density (defined as :math:`E = \rho e`, where :math:`e` is the total energy),
:math:`\mathbf{I}_3` represents the :math:`3 \times 3` identity matrix, :math:`g`
the gravitational acceleration constant, :math:`\boldsymbol{\hat{k}}` the unit vector
in the :math:`z` direction, :math:`k` the thermal conductivity constant, :math:`T`
represents the temperature, and :math:`P` the pressure, given by the following equation
of state

.. math::
   P = \left( {c_p}/{c_v} -1\right) \left( E - {\boldsymbol{U}\cdot\boldsymbol{U}}/{(2 \rho)} - \rho g z \right) \, ,
   :label: eq-state

where :math:`c_p` is the specific heat at constant pressure and :math:`c_v` is the
specific heat at constant volume (that define :math:`\gamma = c_p / c_v`, the specific
heat ratio).

The system :math:numref:`eq-ns` can be rewritten in vector form

.. math::
   :label: eq-vector-ns

   \frac{\partial \boldsymbol{q}}{\partial t} + \nabla \cdot \boldsymbol{F}(\boldsymbol{q}) = S(\boldsymbol{q}) \, ,

for the state variables 5-dimensional vector

.. math::
    \boldsymbol{q} =
           \begin{pmatrix}
               \rho \\
               \boldsymbol{U} \equiv \rho \mathbf{ u }\\
               E \equiv \rho e
           \end{pmatrix}
           \begin{array}{l}
               \leftarrow\textrm{ volume mass density}\\
               \leftarrow\textrm{ momentum density}\\
               \leftarrow\textrm{ energy density}
           \end{array}

where the flux and the source terms, respectively, are given by

.. math::
    :nowrap:

    \begin{align*}
    \boldsymbol{F}(\boldsymbol{q}) &=
    \begin{pmatrix}
        \boldsymbol{U}\\
        {(\boldsymbol{U} \otimes \boldsymbol{U})}/{\rho} + P \mathbf{I}_3 -  \boldsymbol{\sigma} \\
        {(E + P)\boldsymbol{U}}/{\rho} - \boldsymbol{u}  \cdot \boldsymbol{\sigma} - k \nabla T
    \end{pmatrix} ,\\
    S(\boldsymbol{q}) &=
    - \begin{pmatrix}
        0\\
        \rho g \boldsymbol{\hat{k}}\\
        0
    \end{pmatrix}.
    \end{align*}

Let the discrete solution be

.. math::
   \mathbf{q}_N (\boldsymbol{x},t)^{(e)} = \sum_{k=1}^{P}\psi_k (\boldsymbol{x})\boldsymbol{q}_k^{(e)}

with :math:`P=p+1` the number of nodes in the element :math:`e`. We use tensor-product
bases :math:`\psi_{kji} = h_i(X_1)h_j(X_2)h_k(X_3)`.

For the time discretization, we use the follwoing explicit formulation solved with
the adaptive Runge-Kutta-Fehlberg (RKF4-5) method by default (any explicit time-stepping
scheme avaialble in PETSc can be chosen at runtime)

.. math::
   \boldsymbol{q}_N^{n+1} = \boldsymbol{q}_N^n + \Delta t \sum_{i=1}^{s} b_i k_i \, ,

where

.. math::
  :nowrap:

   \begin{align*}
      k_1 &= f(t^n, \boldsymbol{q}_N^n)\\
      k_2 &= f(t^n + c_2 \Delta t, \boldsymbol{q}_N^n + \Delta t (a_{21} k_1))\\
      k_3 &= f(t^n + c_3 \Delta t, \boldsymbol{q}_N^n + \Delta t (a_{31} k_1 + a_{32} k_2))\\
      \vdots&\\
      k_i &= f\left(t^n + c_i \Delta t, \boldsymbol{q}_N^n + \Delta t \sum_{j=1}^s a_{ij} k_j \right)\\
   \end{align*}

and with

.. math::
   f(t^n, \boldsymbol{q}_N^n) = - [\nabla \cdot \boldsymbol{F}(\boldsymbol{q}_N)]^n + [S(\boldsymbol{q}_N)]^n \, .

The strong form of :math:numref:`eq-vector-ns` is:

.. math::
   :label: eq-strong-vector-ns

   \int_{\Omega} v \left(\frac{\partial \boldsymbol{q}_N}{\partial t} + \nabla \cdot \boldsymbol{F}(\boldsymbol{q}_N) \right) \,dV = \int_\Omega v \mathbf{S}(\boldsymbol{q}_N) \, dV \, , \; \forall v \in \mathcal{V}_p

with :math:`\mathcal{V}_p = \{ v \in H^{1}(\Omega_e) \,|\, v \in P_p(\boldsymbol{I}), e=1,\ldots,N_e \}`.

And its weak form is:

.. math::
   :label: eq-weak-vector-ns
   :nowrap:

   \begin{multline}
    \int_{\Omega} v \frac{\partial \boldsymbol{q}_N}{\partial t}  \,dV + \int_{\Gamma} v \widehat{\mathbf{n}} \cdot \boldsymbol{F} (\boldsymbol{q}_N) \,dS - \int_{\Omega} \nabla v\cdot\boldsymbol{F}(\boldsymbol{q}_N)\,dV  =
        \int_\Omega v \mathbf{S}(\boldsymbol{q}_N) \, dV \, , \; \forall v \in \mathcal{V}_p
   \end{multline}

Currently, this demo provides two types of problems/physical models that can be selected
at run time via the option ``-problem``. One is the problem of transport of energy in a
uniform vector velocity field, called the :ref:`problem-advection` problem, and is the
so called :ref:`problem-density-current` problem.


.. _problem-advection:

Advection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simplified version of system :math:numref:`eq-ns`, only accounting for the transport
of total energy, is given by

.. math::
   \frac{\partial E}{\partial t} + \nabla \cdot (\boldsymbol{u} E ) = 0 \, ,
   :label: eq-advection

with $\boldsymbol{u}$ the vector velocity field. In this particular test case, a blob of
total energy (defined by a characteristic radius :math:`r_c`) is transported by a
uniform circular velocity field. We have solved :math:numref:`eq-advection` with no-slip
and non-penetration boundary conditions for :math:`\boldsymbol{u}`, and no-flux for
:math:`E`. This problem can be run with::

   ./navierstokes -problem advection


.. _problem-density-current:

Density Current
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this test problem, we solve the full Navier-Stokes equations :math:numref:`eq-ns`,
for which a cold air bubble (of radius :math:`r_c`) drops by convection in a neutrally
stratified atmosphere. Its initial condition is defined in terms of the Exner pressure,
:math:`\pi(\boldsymbol{x},t)`, and potential temperature,
:math:`\theta(\boldsymbol{x},t)`, that relate to the state variables via

.. math::
    \rho &= \frac{P_0}{( c_p - c_v)\theta(\boldsymbol{x},t)} \pi(\boldsymbol{x},t)^{\frac{c_v}{ c_p - c_v}} \, ,

    e &= c_v \theta(\boldsymbol{x},t) \pi(\boldsymbol{x},t) + \boldsymbol{u}\cdot \boldsymbol{u} /2 + g z \, ,

where :math:`P_0` is the atmospheric pressure. For this problem, we have used no-slip
and non-penetration boundary conditions for :math:`\boldsymbol{u}`, and no-flux
for mass and energy densities. This problem can be run with::

   ./navierstokes -problem density_current


.. _bps:

Bakeoff Problems
======================================

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

The boundary conditions for BP1 and BP2 are homogeneous Neumann. For the rest of the BPs
the boundary conditions are homogeneous Dirichlet.

The nodal points, denoted by :math:`P` (with :math:`P=p+1`, and :math:`p` the degree of the basis polynomial), are Gauss-Legendre-Lobatto (GLL) points, while
the quadrature points, denoted by :math:`Q`, are Gauss-Legendre (GL) for BP1-BP4 and GLL
for BP5-BP6.

