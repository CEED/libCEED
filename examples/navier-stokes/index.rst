.. _example-petsc-navier-stokes:

Compressible Navier-Stokes mini-app
======================================

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
--------------------------------------

A simplified version of system :math:numref:`eq-ns`, only accounting for the transport
of total energy, is given by

.. math::
   \frac{\partial E}{\partial t} + \nabla \cdot (\boldsymbol{u} E ) = 0 \, ,
   :label: eq-advection

with :math:`\boldsymbol{u}` the vector velocity field. In this particular test case, a blob of
total energy (defined by a characteristic radius :math:`r_c`) is transported by a
uniform circular velocity field. We have solved :math:numref:`eq-advection` with no-slip
and non-penetration boundary conditions for :math:`\boldsymbol{u}`, and no-flux for
:math:`E`. This problem can be run with::

   ./navierstokes -problem advection


.. _problem-density-current:

Density Current
--------------------------------------

For this test problem (from :cite:`straka1993numerical`), we solve the full Navier-Stokes equations :math:numref:`eq-ns`,
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
