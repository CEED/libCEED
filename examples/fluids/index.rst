.. _example-petsc-navier-stokes:

Compressible Navier-Stokes mini-app
========================================

This example is located in the subdirectory :file:`examples/fluids`. It solves
the time-dependent Navier-Stokes equations of compressible gas dynamics in a static
Eulerian three-dimensional frame using unstructured high-order finite/spectral
element spatial discretizations and explicit or implicit high-order time-stepping (available in
PETSc). Moreover, the Navier-Stokes example has been developed using PETSc, so that the
pointwise physics (defined at quadrature points) is separated from the parallelization
and meshing concerns.

The mathematical formulation (from :cite:`giraldoetal2010`, cf. SE3) is given in what
follows. The compressible Navier-Stokes equations in conservative form are

.. math::
   :label: eq-ns

   \begin{aligned}
   \frac{\partial \rho}{\partial t} + \nabla \cdot \bm{U} &= 0 \\
   \frac{\partial \bm{U}}{\partial t} + \nabla \cdot \left( \frac{\bm{U} \otimes \bm{U}}{\rho} + P \bm{I}_3 -\bm\sigma \right) + \rho g \bm{\hat k} &= 0 \\
   \frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{(E + P)\bm{U}}{\rho} -\bm{u} \cdot \bm{\sigma} - k \nabla T \right) &= 0 \, , \\
   \end{aligned}

where :math:`\bm{\sigma} = \mu(\nabla \bm{u} + (\nabla \bm{u})^T + \lambda (\nabla \cdot \bm{u})\bm{I}_3)`
is the Cauchy (symmetric) stress tensor, with :math:`\mu` the dynamic viscosity
coefficient, and :math:`\lambda = - 2/3` the Stokes hypothesis constant. In equations
:math:numref:`eq-ns`, :math:`\rho` represents the volume mass density, :math:`U` the
momentum density (defined as :math:`\bm{U}=\rho \bm{u}`, where
:math:`\bm{u}` is the vector velocity field), :math:`E` the total energy
density (defined as :math:`E = \rho e`, where :math:`e` is the total energy),
:math:`\bm{I}_3` represents the :math:`3 \times 3` identity matrix, :math:`g`
the gravitational acceleration constant, :math:`\bm{\hat{k}}` the unit vector
in the :math:`z` direction, :math:`k` the thermal conductivity constant, :math:`T`
represents the temperature, and :math:`P` the pressure, given by the following equation
of state

.. math::
   P = \left( {c_p}/{c_v} -1\right) \left( E - {\bm{U}\cdot\bm{U}}/{(2 \rho)} - \rho g z \right) \, ,
   :label: eq-state

where :math:`c_p` is the specific heat at constant pressure and :math:`c_v` is the
specific heat at constant volume (that define :math:`\gamma = c_p / c_v`, the specific
heat ratio).

The system :math:numref:`eq-ns` can be rewritten in vector form

.. math::
   :label: eq-vector-ns

   \frac{\partial \bm{q}}{\partial t} + \nabla \cdot \bm{F}(\bm{q}) -S(\bm{q}) = 0 \, ,

for the state variables 5-dimensional vector

.. math::
    \bm{q} =
           \begin{pmatrix}
               \rho \\
               \bm{U} \equiv \rho \bm{ u }\\
               E \equiv \rho e
           \end{pmatrix}
           \begin{array}{l}
               \leftarrow\textrm{ volume mass density}\\
               \leftarrow\textrm{ momentum density}\\
               \leftarrow\textrm{ energy density}
           \end{array}

where the flux and the source terms, respectively, are given by

.. math::

    \begin{aligned}
    \bm{F}(\bm{q}) &=
    \begin{pmatrix}
        \bm{U}\\
        {(\bm{U} \otimes \bm{U})}/{\rho} + P \bm{I}_3 -  \bm{\sigma} \\
        {(E + P)\bm{U}}/{\rho} - \bm{u}  \cdot \bm{\sigma} - k \nabla T
    \end{pmatrix} ,\\
    S(\bm{q}) &=
    - \begin{pmatrix}
        0\\
        \rho g \bm{\hat{k}}\\
        0
    \end{pmatrix}.
    \end{aligned}

Let the discrete solution be

.. math::
   \bm{q}_N (\bm{x},t)^{(e)} = \sum_{k=1}^{P}\psi_k (\bm{x})\bm{q}_k^{(e)}

with :math:`P=p+1` the number of nodes in the element :math:`e`. We use tensor-product
bases :math:`\psi_{kji} = h_i(X_0)h_j(X_1)h_k(X_2)`.

For the time discretization, we use two types of time stepping schemes.

- Explicit time-stepping method

    The following explicit formulation is solved with the adaptive Runge-Kutta-Fehlberg
    (RKF4-5) method by default (any explicit time-stepping
    scheme available in PETSc can be chosen at runtime)

    .. math::
       \bm{q}_N^{n+1} = \bm{q}_N^n + \Delta t \sum_{i=1}^{s} b_i k_i \, ,

    where

    .. math::

       \begin{aligned}
          k_1 &= f(t^n, \bm{q}_N^n)\\
          k_2 &= f(t^n + c_2 \Delta t, \bm{q}_N^n + \Delta t (a_{21} k_1))\\
          k_3 &= f(t^n + c_3 \Delta t, \bm{q}_N^n + \Delta t (a_{31} k_1 + a_{32} k_2))\\
          \vdots&\\
          k_i &= f\left(t^n + c_i \Delta t, \bm{q}_N^n + \Delta t \sum_{j=1}^s a_{ij} k_j \right)\\
       \end{aligned}

    and with

    .. math::
       f(t^n, \bm{q}_N^n) = - [\nabla \cdot \bm{F}(\bm{q}_N)]^n + [S(\bm{q}_N)]^n \, .

- Implicit time-stepping method

    This time stepping method which can be selected using the option ``-implicit`` is
    solved with Backward Differentiation Formula (BDF) method by default (similarly,
    any implicit time-stepping scheme available in PETSc can be chosen at runtime).
    The implicit formulation solves nonlinear systems for :math:`\bm q_N`:

    .. math::
       :label: eq-ts-implicit-ns

       \bm f(\bm q_N) \equiv \bm g(t^{n+1}, \bm{q}_N, \bm{\dot{q}}_N) = 0 \, ,

    where the time derivative :math:`\bm{\dot q}_N` is defined by

    .. math::
      \bm{\dot{q}}_N(\bm q_N) = \alpha \bm q_N + \bm z_N

    in terms of :math:`\bm z_N` from prior state and :math:`\alpha > 0`,
    both of which depend on the specific time integration scheme (backward difference
    formulas, generalized alpha, implicit Runge-Kutta, etc.).
    Each nonlinear system :math:numref:`eq-ts-implicit-ns` will correspond to a
    weak form, as explained below.
    In determining how difficult a given problem is to solve, we consider the
    Jacobian of :math:numref:`eq-ts-implicit-ns`,

    .. math::
       \frac{\partial \bm f}{\partial \bm q_N}
       = \frac{\partial \bm g}{\partial \bm q_N}
       + \alpha \frac{\partial \bm g}{\partial \bm{\dot q}_N}.

    The scalar "shift" :math:`\alpha` scales inversely with the time step
    :math:`\Delta t`, so small time steps result in the Jacobian being dominated
    by the second term, which is a sort of "mass matrix", and typically
    well-conditioned independent of grid resolution with a simple preconditioner
    (such as Jacobi).
    In contrast, the first term dominates for large time steps, with a condition
    number that grows with the diameter of the domain and polynomial degree of
    the approximation space.  Both terms are significant for time-accurate
    simulation and the setup costs of strong preconditioners must be balanced
    with the convergence rate of Krylov methods using weak preconditioners.

To obtain a finite element discretization, we first multiply the strong form
:math:numref:`eq-vector-ns` by a test function :math:`\bm v \in H^1(\Omega)`
and integrate,

.. math::
   \int_{\Omega} \bm v \cdot \left(\frac{\partial \bm{q}_N}{\partial t} + \nabla \cdot \bm{F}(\bm{q}_N) - \bm{S}(\bm{q}_N) \right) \,dV = 0 \, , \; \forall \bm v \in \mathcal{V}_p\,,

with :math:`\mathcal{V}_p = \{ \bm v(\bm x) \in H^{1}(\Omega_e) \,|\, \bm v(\bm x_e(\bm X)) \in P_p(\bm{I}), e=1,\ldots,N_e \}`
a mapped space of polynomials containing at least polynomials of degree :math:`p`
(with or without the higher mixed terms that appear in tensor product spaces).

Integrating by parts on the divergence term, we arrive at the weak form,

.. math::
   :label: eq-weak-vector-ns

   \begin{aligned}
   \int_{\Omega} \bm v \cdot \left( \frac{\partial \bm{q}_N}{\partial t} - \bm{S}(\bm{q}_N) \right)  \,dV
   - \int_{\Omega} \nabla \bm v \!:\! \bm{F}(\bm{q}_N)\,dV & \\
   + \int_{\partial \Omega} \bm v \cdot \bm{F}(\bm q_N) \cdot \widehat{\bm{n}} \,dS
     &= 0 \, , \; \forall \bm v \in \mathcal{V}_p \,,
   \end{aligned}

where :math:`\bm{F}(\bm q_N) \cdot \widehat{\bm{n}}` is typically
replaced with a boundary condition.

.. note::
  The notation :math:`\nabla \bm v \!:\! \bm F` represents contraction over both fields and spatial dimensions while a single dot represents contraction in just one, which should be clear from context, e.g., :math:`\bm v \cdot \bm S` contracts over fields while :math:`\bm F \cdot \widehat{\bm n}` contracts over spatial dimensions.

We solve :math:numref:`eq-weak-vector-ns` using a Galerkin discretization (default)
or a stabilized method, as is necessary for most real-world flows.

Galerkin methods produce oscillations for transport-dominated problems (any time
the cell Péclet number is larger than 1), and those tend to blow up for nonlinear
problems such as the Euler equations and (low-viscosity/poorly resolved) Navier-Stokes,
in which case stabilization is necessary. Our formulation follows :cite:`hughesetal2010`,
which offers a comprehensive review of stabilization and shock-capturing methods
for continuous finite element discretization of compressible flows.

- **SUPG** (streamline-upwind/Petrov-Galerkin)

    In this method, the weighted residual of the strong form
    :math:numref:`eq-vector-ns` is added to the Galerkin formulation
    :math:numref:`eq-weak-vector-ns`. The weak form for this method is given as

    .. math::
       :label: eq-weak-vector-ns-supg

       \begin{aligned}
       \int_{\Omega} \bm v \cdot \left( \frac{\partial \bm{q}_N}{\partial t} - \bm{S}(\bm{q}_N) \right)  \,dV
       - \int_{\Omega} \nabla \bm v \!:\! \bm{F}(\bm{q}_N)\,dV & \\
       + \int_{\partial \Omega} \bm v \cdot \bm{F}(\bm{q}_N) \cdot \widehat{\bm{n}} \,dS & \\
       + \int_{\Omega} \bm{P}(\bm v)^T \, \left( \frac{\partial \bm{q}_N}{\partial t} \, + \,
       \nabla \cdot \bm{F} \, (\bm{q}_N) - \bm{S}(\bm{q}_N) \right) \,dV &= 0
       \, , \; \forall \bm v \in \mathcal{V}_p
       \end{aligned}

    This stabilization technique can be selected using the option ``-stab supg``.


- **SU** (streamline-upwind)

    This method is a simplified version of *SUPG* :math:numref:`eq-weak-vector-ns-supg`
    which is developed for debugging/comparison purposes. The weak form for this method
    is

    .. math::
       :label: eq-weak-vector-ns-su

       \begin{aligned}
       \int_{\Omega} \bm v \cdot \left( \frac{\partial \bm{q}_N}{\partial t} - \bm{S}(\bm{q}_N) \right)  \,dV
       - \int_{\Omega} \nabla \bm v \!:\! \bm{F}(\bm{q}_N)\,dV & \\
       + \int_{\partial \Omega} \bm v \cdot \bm{F}(\bm{q}_N) \cdot \widehat{\bm{n}} \,dS & \\
       + \int_{\Omega} \bm{P}(\bm v)^T \, \nabla \cdot \bm{F} \, (\bm{q}_N) \,dV
       & = 0 \, , \; \forall \bm v \in \mathcal{V}_p
       \end{aligned}

    This stabilization technique can be selected using the option ``-stab su``.


In both :math:numref:`eq-weak-vector-ns-su` and :math:numref:`eq-weak-vector-ns-supg`,
:math:`\bm{P} \,` is called the *perturbation to the test-function space*,
since it modifies the original Galerkin method into *SUPG* or *SU* schemes. It is defined
as

.. math::
   \bm{P}(\bm v) \equiv \left(\bm{\tau} \cdot \frac{\partial \bm{F} \, (\bm{q}_N)}{\partial
   \bm{q}_N} \right)^T \, \nabla \bm v\,,

where parameter :math:`\bm{\tau} \in \mathbb R^{3\times 3}` is an intrinsic time/space scale matrix.

Currently, this demo provides two types of problems/physical models that can be selected
at run time via the option ``-problem``. One is the problem of transport of energy in a
uniform vector velocity field, called the :ref:`problem-advection` problem, and is the
so called :ref:`problem-density-current` problem.


.. _problem-advection:

Advection
----------------------------------------

A simplified version of system :math:numref:`eq-ns`, only accounting for the transport
of total energy, is given by

.. math::
   \frac{\partial E}{\partial t} + \nabla \cdot (\bm{u} E ) = 0 \, ,
   :label: eq-advection

with :math:`\bm{u}` the vector velocity field. In this particular test case, a
blob of total energy (defined by a characteristic radius :math:`r_c`) is transported by two
different wind types.

- **Rotation**

   In this case, a uniform circular velocity field transports the blob of total energy. We have
   solved :math:numref:`eq-advection` applying zero energy density :math:`E`, and no-flux for
   :math:`\bm{u}` on the boundaries.

   The :math:`3D` version of this test case can be run with::

      ./navierstokes -problem advection -problem_advection_wind rotation

   while the :math:`2D` version with::

      ./navierstokes -problem advection2d -problem_advection_wind rotation

- **Translation**

   In this case, a background wind with a constant rectilinear velocity field, enters the domain and transports
   the blob of total energy out of the domain.

   For the inflow boundary conditions, a prescribed :math:`E_{wind}` is applied weakly on the inflow boundaries
   such that the weak form boundary integral in :math:numref:`eq-weak-vector-ns` is defined as

   .. math::
      \int_{\partial \Omega_{inflow}} \bm v \cdot \bm{F}(\bm q_N) \cdot \widehat{\bm{n}} \,dS = \int_{\partial \Omega_{inflow}} \bm v \, E_{wind} \, \bm u \cdot \widehat{\bm{n}} \,dS  \, ,

   For the outflow boundary conditions, we have used the current values of :math:`E`, following
   :cite:`papanastasiou1992outflow` which extends the validity of the weak form of the governing
   equations to the outflow instead of replacing them with unknown essential or natural
   boundary conditions. The weak form boundary integral in :math:numref:`eq-weak-vector-ns` for
   outflow boundary conditions is defined as

   .. math::
      \int_{\partial \Omega_{outflow}} \bm v \cdot \bm{F}(\bm q_N) \cdot \widehat{\bm{n}} \,dS = \int_{\partial \Omega_{outflow}} \bm v \, E \, \bm u \cdot \widehat{\bm{n}} \,dS  \, ,

   The :math:`3D` version of this test case problem can be run with::

      ./navierstokes -problem advection -problem_advection_wind translation -problem_advection_wind translation .5,-1,0

   while the :math:`2D` version with::

      ./navierstokes -problem advection2d -problem_advection_wind translation -problem_advection_wind translation 1,-.5


.. _problem-euler-vortex:

Euler Traveling Vortex
----------------------------------------

Three-dimensional Euler equations, which are simplified version of system :math:numref:`eq-ns`
and account only for the convective fluxes, are given by

.. math::
   :label: eq-euler

   \begin{aligned}
   \frac{\partial \rho}{\partial t} + \nabla \cdot \bm{U} &= 0 \\
   \frac{\partial \bm{U}}{\partial t} + \nabla \cdot \left( \frac{\bm{U} \otimes \bm{U}}{\rho} + P \bm{I}_3 \right) &= 0 \\
   \frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{(E + P)\bm{U}}{\rho} \right) &= 0 \, , \\
   \end{aligned}

Following the setup given in :cite:`zhang2011verification`, the mean flow for this problem is
:math:`\rho=1`, :math:`P=1`, and :math:`\bm{u}=(1,1,0)` while the perturbation :math:`\delta \bm{u}`,
and :math:`\delta T` are defined as

.. math::
   \begin{aligned}
   (\delta u_1, \, \delta u_2) &= \frac{\epsilon}{2 \pi} \, e^{0.5(1-r^2)} \, (-\bar{y}, \, \bar{x}) \, , \\
   \delta T &= - \frac{(\gamma-1) \, \epsilon^2}{8 \, \gamma \, \pi^2} \, e^{1-r^2} \, , \\
   \end{aligned}

where :math:`(\bar{x}, \, \bar{y}) = (x-x_c, \, y-y_c)`, :math:`(x_c, \, y_c)` represents the center of the domain,
:math:`r^2=\bar{x}^2 + \bar{y}^2`, and :math:`\epsilon` is the vortex strength.

This problem can be run with::

   ./navierstokes -problem euler_vortex


.. _problem-density-current:

Density Current
----------------------------------------

For this test problem (from :cite:`straka1993numerical`), we solve the full
Navier-Stokes equations :math:numref:`eq-ns`, for which a cold air bubble
(of radius :math:`r_c`) drops by convection in a neutrally stratified atmosphere.
Its initial condition is defined in terms of the Exner pressure,
:math:`\pi(\bm{x},t)`, and potential temperature,
:math:`\theta(\bm{x},t)`, that relate to the state variables via

.. math::
   \begin{aligned}
   \rho &= \frac{P_0}{( c_p - c_v)\theta(\bm{x},t)} \pi(\bm{x},t)^{\frac{c_v}{ c_p - c_v}} \, , \\
   e &= c_v \theta(\bm{x},t) \pi(\bm{x},t) + \bm{u}\cdot \bm{u} /2 + g z \, ,
   \end{aligned}

where :math:`P_0` is the atmospheric pressure. For this problem, we have used no-slip
and non-penetration boundary conditions for :math:`\bm{u}`, and no-flux
for mass and energy densities. This problem can be run with::

   ./navierstokes -problem density_current

