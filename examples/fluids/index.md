(example-petsc-navier-stokes)=

# Compressible Navier-Stokes mini-app

This example is located in the subdirectory {file}`examples/fluids`.
It solves the time-dependent Navier-Stokes equations of compressible gas dynamics in a static Eulerian three-dimensional frame using unstructured high-order finite/spectral element spatial discretizations and explicit or implicit high-order time-stepping (available in PETSc).
Moreover, the Navier-Stokes example has been developed using PETSc, so that the pointwise physics (defined at quadrature points) is separated from the parallelization and meshing concerns.

## Running the mini-app

```{include} README.md
:start-after: inclusion-fluids-marker
```
## The Navier-Stokes equations

The mathematical formulation (from {cite}`giraldoetal2010`, cf. SE3) is given in what follows.
The compressible Navier-Stokes equations in conservative form are

$$
\begin{aligned}
\frac{\partial \rho}{\partial t} + \nabla \cdot \bm{U} &= 0 \\
\frac{\partial \bm{U}}{\partial t} + \nabla \cdot \left( \frac{\bm{U} \otimes \bm{U}}{\rho} + P \bm{I}_3 -\bm\sigma \right) + \rho g \bm{\hat k} &= 0 \\
\frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{(E + P)\bm{U}}{\rho} -\bm{u} \cdot \bm{\sigma} - k \nabla T \right) &= 0 \, , \\
\end{aligned}
$$ (eq-ns)

where $\bm{\sigma} = \mu(\nabla \bm{u} + (\nabla \bm{u})^T + \lambda (\nabla \cdot \bm{u})\bm{I}_3)$ is the Cauchy (symmetric) stress tensor, with $\mu$ the dynamic viscosity coefficient, and $\lambda = - 2/3$ the Stokes hypothesis constant.
In equations {eq}`eq-ns`, $\rho$ represents the volume mass density, $U$ the momentum density (defined as $\bm{U}=\rho \bm{u}$, where $\bm{u}$ is the vector velocity field), $E$ the total energy density (defined as $E = \rho e$, where $e$ is the total energy), $\bm{I}_3$ represents the $3 \times 3$ identity matrix, $g$ the gravitational acceleration constant, $\bm{\hat{k}}$ the unit vector in the $z$ direction, $k$ the thermal conductivity constant, $T$ represents the temperature, and $P$ the pressure, given by the following equation of state

$$
P = \left( {c_p}/{c_v} -1\right) \left( E - {\bm{U}\cdot\bm{U}}/{(2 \rho)} - \rho g z \right) \, ,
$$ (eq-state)

where $c_p$ is the specific heat at constant pressure and $c_v$ is the specific heat at constant volume (that define $\gamma = c_p / c_v$, the specific heat ratio).

The system {eq}`eq-ns` can be rewritten in vector form

$$
\frac{\partial \bm{q}}{\partial t} + \nabla \cdot \bm{F}(\bm{q}) -S(\bm{q}) = 0 \, ,
$$ (eq-vector-ns)

for the state variables 5-dimensional vector

$$
\bm{q} =        \begin{pmatrix}            \rho \\            \bm{U} \equiv \rho \bm{ u }\\            E \equiv \rho e        \end{pmatrix}        \begin{array}{l}            \leftarrow\textrm{ volume mass density}\\            \leftarrow\textrm{ momentum density}\\            \leftarrow\textrm{ energy density}        \end{array}
$$

where the flux and the source terms, respectively, are given by

$$
\begin{aligned}
\bm{F}(\bm{q}) &=
\underbrace{\begin{pmatrix}
    \bm{U}\\
    {(\bm{U} \otimes \bm{U})}/{\rho} + P \bm{I}_3 \\
    {(E + P)\bm{U}}/{\rho}
\end{pmatrix}}_{\bm F_{\text{adv}}} +
\underbrace{\begin{pmatrix}
0 \\
-  \bm{\sigma} \\
 - \bm{u}  \cdot \bm{\sigma} - k \nabla T
\end{pmatrix}}_{\bm F_{\text{diff}}},\\
S(\bm{q}) &=
- \begin{pmatrix}
    0\\
    \rho g \bm{\hat{k}}\\
    0
\end{pmatrix}.
\end{aligned}
$$ (eq-ns-flux)

Let the discrete solution be

$$
\bm{q}_N (\bm{x},t)^{(e)} = \sum_{k=1}^{P}\psi_k (\bm{x})\bm{q}_k^{(e)}
$$

with $P=p+1$ the number of nodes in the element $e$.
We use tensor-product bases $\psi_{kji} = h_i(X_0)h_j(X_1)h_k(X_2)$.

For the time discretization, we use two types of time stepping schemes.

- Explicit time-stepping method

  The following explicit formulation is solved with the adaptive Runge-Kutta-Fehlberg (RKF4-5) method by default (any explicit time-stepping scheme available in PETSc can be chosen at runtime)

  $$
  \bm{q}_N^{n+1} = \bm{q}_N^n + \Delta t \sum_{i=1}^{s} b_i k_i \, ,
  $$

  where

  $$
  \begin{aligned}
     k_1 &= f(t^n, \bm{q}_N^n)\\
     k_2 &= f(t^n + c_2 \Delta t, \bm{q}_N^n + \Delta t (a_{21} k_1))\\
     k_3 &= f(t^n + c_3 \Delta t, \bm{q}_N^n + \Delta t (a_{31} k_1 + a_{32} k_2))\\
     \vdots&\\
     k_i &= f\left(t^n + c_i \Delta t, \bm{q}_N^n + \Delta t \sum_{j=1}^s a_{ij} k_j \right)\\
  \end{aligned}
  $$

  and with

  $$
  f(t^n, \bm{q}_N^n) = - [\nabla \cdot \bm{F}(\bm{q}_N)]^n + [S(\bm{q}_N)]^n \, .
  $$

- Implicit time-stepping method

  This time stepping method which can be selected using the option `-implicit` is solved with Backward Differentiation Formula (BDF) method by default (similarly, any implicit time-stepping scheme available in PETSc can be chosen at runtime).
  The implicit formulation solves nonlinear systems for $\bm q_N$:

  $$
  \bm f(\bm q_N) \equiv \bm g(t^{n+1}, \bm{q}_N, \bm{\dot{q}}_N) = 0 \, ,
  $$ (eq-ts-implicit-ns)

  where the time derivative $\bm{\dot q}_N$ is defined by

  $$
  \bm{\dot{q}}_N(\bm q_N) = \alpha \bm q_N + \bm z_N
  $$

  in terms of $\bm z_N$ from prior state and $\alpha > 0$, both of which depend on the specific time integration scheme (backward difference formulas, generalized alpha, implicit Runge-Kutta, etc.).
  Each nonlinear system {eq}`eq-ts-implicit-ns` will correspond to a weak form, as explained below.
  In determining how difficult a given problem is to solve, we consider the Jacobian of {eq}`eq-ts-implicit-ns`,

  $$
  \frac{\partial \bm f}{\partial \bm q_N} = \frac{\partial \bm g}{\partial \bm q_N} + \alpha \frac{\partial \bm g}{\partial \bm{\dot q}_N}.
  $$

  The scalar "shift" $\alpha$ scales inversely with the time step $\Delta t$, so small time steps result in the Jacobian being dominated by the second term, which is a sort of "mass matrix", and typically well-conditioned independent of grid resolution with a simple preconditioner (such as Jacobi).
  In contrast, the first term dominates for large time steps, with a condition number that grows with the diameter of the domain and polynomial degree of the approximation space.
  Both terms are significant for time-accurate simulation and the setup costs of strong preconditioners must be balanced with the convergence rate of Krylov methods using weak preconditioners.

To obtain a finite element discretization, we first multiply the strong form {eq}`eq-vector-ns` by a test function $\bm v \in H^1(\Omega)$ and integrate,

$$
\int_{\Omega} \bm v \cdot \left(\frac{\partial \bm{q}_N}{\partial t} + \nabla \cdot \bm{F}(\bm{q}_N) - \bm{S}(\bm{q}_N) \right) \,dV = 0 \, , \; \forall \bm v \in \mathcal{V}_p\,,
$$

with $\mathcal{V}_p = \{ \bm v(\bm x) \in H^{1}(\Omega_e) \,|\, \bm v(\bm x_e(\bm X)) \in P_p(\bm{I}), e=1,\ldots,N_e \}$ a mapped space of polynomials containing at least polynomials of degree $p$ (with or without the higher mixed terms that appear in tensor product spaces).

Integrating by parts on the divergence term, we arrive at the weak form,

$$
\begin{aligned}
\int_{\Omega} \bm v \cdot \left( \frac{\partial \bm{q}_N}{\partial t} - \bm{S}(\bm{q}_N) \right)  \,dV
- \int_{\Omega} \nabla \bm v \!:\! \bm{F}(\bm{q}_N)\,dV & \\
+ \int_{\partial \Omega} \bm v \cdot \bm{F}(\bm q_N) \cdot \widehat{\bm{n}} \,dS
  &= 0 \, , \; \forall \bm v \in \mathcal{V}_p \,,
\end{aligned}
$$ (eq-weak-vector-ns)

where $\bm{F}(\bm q_N) \cdot \widehat{\bm{n}}$ is typically replaced with a boundary condition.

:::{note}
The notation $\nabla \bm v \!:\! \bm F$ represents contraction over both fields and spatial dimensions while a single dot represents contraction in just one, which should be clear from context, e.g., $\bm v \cdot \bm S$ contracts over fields while $\bm F \cdot \widehat{\bm n}$ contracts over spatial dimensions.
:::

We solve {eq}`eq-weak-vector-ns` using a Galerkin discretization (default) or a stabilized method, as is necessary for most real-world flows.

Galerkin methods produce oscillations for transport-dominated problems (any time the cell Péclet number is larger than 1), and those tend to blow up for nonlinear problems such as the Euler equations and (low-viscosity/poorly resolved) Navier-Stokes, in which case stabilization is necessary.
Our formulation follows {cite}`hughesetal2010`, which offers a comprehensive review of stabilization and shock-capturing methods for continuous finite element discretization of compressible flows.

- **SUPG** (streamline-upwind/Petrov-Galerkin)

  In this method, the weighted residual of the strong form {eq}`eq-vector-ns` is added to the Galerkin formulation {eq}`eq-weak-vector-ns`.
  The weak form for this method is given as

  $$
  \begin{aligned}
  \int_{\Omega} \bm v \cdot \left( \frac{\partial \bm{q}_N}{\partial t} - \bm{S}(\bm{q}_N) \right)  \,dV
  - \int_{\Omega} \nabla \bm v \!:\! \bm{F}(\bm{q}_N)\,dV & \\
  + \int_{\partial \Omega} \bm v \cdot \bm{F}(\bm{q}_N) \cdot \widehat{\bm{n}} \,dS & \\
  + \int_{\Omega} \bm{P}(\bm v)^T \, \left( \frac{\partial \bm{q}_N}{\partial t} \, + \,
  \nabla \cdot \bm{F} \, (\bm{q}_N) - \bm{S}(\bm{q}_N) \right) \,dV &= 0
  \, , \; \forall \bm v \in \mathcal{V}_p
  \end{aligned}
  $$ (eq-weak-vector-ns-supg)

  This stabilization technique can be selected using the option `-stab supg`.

- **SU** (streamline-upwind)

  This method is a simplified version of *SUPG* {eq}`eq-weak-vector-ns-supg` which is developed for debugging/comparison purposes. The weak form for this method is

  $$
  \begin{aligned}
  \int_{\Omega} \bm v \cdot \left( \frac{\partial \bm{q}_N}{\partial t} - \bm{S}(\bm{q}_N) \right)  \,dV
  - \int_{\Omega} \nabla \bm v \!:\! \bm{F}(\bm{q}_N)\,dV & \\
  + \int_{\partial \Omega} \bm v \cdot \bm{F}(\bm{q}_N) \cdot \widehat{\bm{n}} \,dS & \\
  + \int_{\Omega} \mathcal{P}(\bm v)^T \, \nabla \cdot \bm{F} \, (\bm{q}_N) \,dV
  & = 0 \, , \; \forall \bm v \in \mathcal{V}_p
  \end{aligned}
  $$ (eq-weak-vector-ns-su)

  This stabilization technique can be selected using the option `-stab su`.

In both {eq}`eq-weak-vector-ns-su` and {eq}`eq-weak-vector-ns-supg`, $\mathcal P$ is called the *perturbation to the test-function space*, since it modifies the original Galerkin method into *SUPG* or *SU* schemes.
It is defined as

$$
\mathcal P(\bm v) \equiv \left(\bm{\tau} \cdot \frac{\partial \bm{F}_{\text{adv}} (\bm{q}_N)}{\partial \bm{q}_N} \right)^T \, \nabla \bm v\,,
$$

where parameter $\bm{\tau} \in \mathbb R^{3\times 3}$ (spatial indices) or $\bm \tau \in \mathbb R^{5\times 5}$ (field indices) is an intrinsic time scale matrix.
This expression contains the flux Jacobian, which we express in variational notation by differentiating the advective flux $\bm F_{\text{adv}}$ of {eq}`eq-ns-flux`

$$
\begin{aligned}
\diff\bm F_{\text{adv}}(\diff\bm q; \bm q) &= \frac{\partial \bm F_{\text{adv}}}{\partial \bm q} \diff\bm q \\
&= \begin{pmatrix}
\diff\bm U \\
(\diff\bm U \otimes \bm U + \bm U \otimes \diff\bm U)/\rho - (\bm U \otimes \bm U)/\rho^2 \diff\rho + \diff P \bm I_3 \\
(E + P)\diff\bm U/\rho + (\diff E + \diff P)\bm U/\rho - (E + P) \bm U/\rho^2 \diff\rho
\end{pmatrix},
\end{aligned}
$$

where $\diff P$ is defined by differentiating {eq}`eq-state`.
In this notation, we may equivalently write the stabilization term as

$$
\mathcal P(\bm v)^T \bm r = \nabla \bm v \bm\tau \diff\bm F_{\text{adv}}(\bm r),
$$

where $\bm r$ is the strong form residual.
Note that both $\nabla \bm v$ and $\diff \bm F$ are $5\times 3$ matrices and that $\bm\tau$ can be defined with spatial indices, or field indices, leading to a stabilization term of $(\nabla \bm v)_{\alpha i} \tau_{ij} \diff \bm F_{\alpha j}$ for spatial or $(\nabla \bm v)_{\alpha i} \tau_{\alpha \beta} \diff \bm F_{\beta i}$ for field, where $\alpha,\beta$ are field indices and $i,j$ are spatial indices.

:::{dropdown} Stabilization scale $\bm\tau$
A velocity vector $\bm u$ can be pulled back to the reference element as $\bm u_{\bm X} = \nabla_{\bm x}\bm X \cdot \bm u$, with units of reference length (non-dimensional) per second.
To build intuition, consider a boundary layer element of dimension $(1, \epsilon)$, for which $\nabla_{\bm x} \bm X = \bigl(\begin{smallmatrix} 2 & \\ & 2/\epsilon \end{smallmatrix}\bigr)$.
So a small normal component of velocity will be amplified (by a factor of the aspect ratio $1/\epsilon$) in this transformation.
The ratio $\lVert \bm u \rVert / \lVert \bm u_{\bm X} \rVert$ is a covariant measure of (half) the element length in the direction of the velocity.
A contravariant measure of element length in the direction of a unit vector $\hat{\bm n}$ is given by $\lVert \bigl(\nabla_{\bm X} \bm x\bigr)^T \hat{\bm n} \rVert$.
While $\nabla_{\bm X} \bm x$ is readily computable, its inverse $\nabla_{\bm x} \bm X$ is needed directly in finite element methods and thus more convenient for our use.
If we consider a parallelogram, the covariant measure is larger than the contravariant measure for vectors pointing between acute corners and the opposite holds for vectors between oblique corners.

The cell Péclet number is classically defined by $\mathrm{Pe}_h = \lVert \bm u \rVert h / (2 \kappa)$ where $\kappa$ is the diffusivity (units of $m^2/s$).
This can be generalized to arbitrary grids by defining the local Péclet number

$$
\mathrm{Pe} = \frac{\lVert \bm u \rVert^2}{\lVert \bm u_{\bm X} \rVert \kappa}.
$$ (eq-peclet)

For scalar advection-diffusion, the stabilization is a scalar

$$
\tau = \frac{\xi(\mathrm{Pe})}{\lVert \bm u_{\bm X} \rVert},
$$ (eq-tau-advdiff)

where $\xi(\mathrm{Pe}) = \coth \mathrm{Pe} - 1/\mathrm{Pe}$ approaches 1 at large local Péclet number.
Note that $\tau$ has units of time and, in the transport-dominated limit, is proportional to element transit time in the direction of the propagating wave.
For advection-diffusion, $\bm F(q) = \bm u q$, and thus the perturbed test function is

$$
\mathcal P(v) = \tau \bm u \cdot \nabla v = \tau \bm u_{\bm X} \nabla_{\bm X} v.
$$ (eq-test-perturbation-advdiff)

See {cite}`hughesetal2010` equations 15-17 and 34-36 for further discussion of this formulation.

For the Navier-Stokes and Euler equations in primitive variables, {cite}`whiting2003hierarchical` defines a $5\times 5$ diagonal stabilization consisting of
1. continuity stabilization $\tau_c$
2. momentum stabilization $\tau_m$
3. energy stabilization $\tau_E$

However, since our equations are in conservative form, we follow {cite}`hughesetal2010` in defining a $3\times 3$ diagonal stabilization according to spatial criterion 2 (equation 27) as follows.

$$
\tau_{ii} = c_{\tau} \frac{2 \xi(\mathrm{Pe})}{(\lambda_{\max \text{abs}})_i \lVert \nabla_{x_i} \bm X \rVert}
$$ (eq-tau-conservative)

where $c_{\tau}$ is a multiplicative constant reported to be optimal at 0.5 for linear elements, $\hat{\bm n}_i$ is a unit vector in direction $i$, and $\nabla_{x_i} = \hat{\bm n}_i \cdot \nabla_{\bm x}$ is the derivative in direction $i$.
The flux Jacobian $\frac{\partial \bm F_{\text{adv}}}{\partial \bm q} \cdot \hat{\bm n}_i$ in each direction $i$ is a $5\times 5$ matrix with spectral radius $(\lambda_{\max \text{abs}})_i$ equal to the fastest wave speed.
The complete set of eigenvalues of the Euler flux Jacobian in direction $i$ are (e.g., {cite}`toro2009`)

$$
\Lambda_i = [u_i - a, u_i, u_i, u_i, u_i+a],
$$ (eq-eigval-advdiff)

where $u_i = \bm u \cdot \hat{\bm n}_i$ is the velocity component in direction $i$ and $a = \sqrt{\gamma P/\rho}$ is the sound speed for ideal gasses.
Note that the first and last eigenvalues represent nonlinear acoustic waves while the middle three are linearly degenerate, carrying a contact wave (temperature) and transverse components of momentum.
The fastest wave speed in direction $i$ is thus

$$
\lambda_{\max \text{abs}} \Bigl( \frac{\partial \bm F_{\text{adv}}}{\partial \bm q} \cdot \hat{\bm n}_i \Bigr) = |u_i| + a
$$ (eq-wavespeed)

Note that this wave speed is specific to ideal gases as $\gamma$ is an ideal gas parameter; other equations of state will yield a different acoustic wave speed.

:::

Currently, this demo provides three types of problems/physical models that can be selected at run time via the option `-problem`.
{ref}`problem-advection`, the problem of the transport of energy in a uniform vector velocity field, {ref}`problem-euler-vortex`, the exact solution to the Euler equations, and the so called {ref}`problem-density-current` problem.

(problem-advection)=

## Advection

A simplified version of system {eq}`eq-ns`, only accounting for the transport of total energy, is given by

$$
\frac{\partial E}{\partial t} + \nabla \cdot (\bm{u} E ) = 0 \, ,
$$ (eq-advection)

with $\bm{u}$ the vector velocity field. In this particular test case, a blob of total energy (defined by a characteristic radius $r_c$) is transported by two different wind types.

- **Rotation**

  In this case, a uniform circular velocity field transports the blob of total energy.
  We have solved {eq}`eq-advection` applying zero energy density $E$, and no-flux for $\bm{u}$ on the boundaries.

- **Translation**

  In this case, a background wind with a constant rectilinear velocity field, enters the domain and transports the blob of total energy out of the domain.

  For the inflow boundary conditions, a prescribed $E_{wind}$ is applied weakly on the inflow boundaries such that the weak form boundary integral in {eq}`eq-weak-vector-ns` is defined as

  $$
  \int_{\partial \Omega_{inflow}} \bm v \cdot \bm{F}(\bm q_N) \cdot \widehat{\bm{n}} \,dS = \int_{\partial \Omega_{inflow}} \bm v \, E_{wind} \, \bm u \cdot \widehat{\bm{n}} \,dS  \, ,
  $$

  For the outflow boundary conditions, we have used the current values of $E$, following {cite}`papanastasiou1992outflow` which extends the validity of the weak form of the governing equations to the outflow instead of replacing them with unknown essential or natural boundary conditions.
  The weak form boundary integral in {eq}`eq-weak-vector-ns` for outflow boundary conditions is defined as

  $$
  \int_{\partial \Omega_{outflow}} \bm v \cdot \bm{F}(\bm q_N) \cdot \widehat{\bm{n}} \,dS = \int_{\partial \Omega_{outflow}} \bm v \, E \, \bm u \cdot \widehat{\bm{n}} \,dS  \, ,
  $$

(problem-euler-vortex)=

## Isentropic Vortex

Three-dimensional Euler equations, which are simplified and nondimensionalized version of system {eq}`eq-ns` and account only for the convective fluxes, are given by

$$
\begin{aligned}
\frac{\partial \rho}{\partial t} + \nabla \cdot \bm{U} &= 0 \\
\frac{\partial \bm{U}}{\partial t} + \nabla \cdot \left( \frac{\bm{U} \otimes \bm{U}}{\rho} + P \bm{I}_3 \right) &= 0 \\
\frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{(E + P)\bm{U}}{\rho} \right) &= 0 \, , \\
\end{aligned}
$$ (eq-euler)

Following the setup given in {cite}`zhang2011verification`, the mean flow for this problem is $\rho=1$, $P=1$, $T=P/\rho= 1$ (Specific Gas Constant, $R$, is 1), and $\bm{u}=(u_1,u_2,0)$ while the perturbation $\delta \bm{u}$, and $\delta T$ are defined as

$$
\begin{aligned} (\delta u_1, \, \delta u_2) &= \frac{\epsilon}{2 \pi} \, e^{0.5(1-r^2)} \, (-\bar{y}, \, \bar{x}) \, , \\ \delta T &= - \frac{(\gamma-1) \, \epsilon^2}{8 \, \gamma \, \pi^2} \, e^{1-r^2} \, , \\ \end{aligned}
$$

where $(\bar{x}, \, \bar{y}) = (x-x_c, \, y-y_c)$, $(x_c, \, y_c)$ represents the center of the domain, $r^2=\bar{x}^2 + \bar{y}^2$, and $\epsilon$ is the vortex strength ($\epsilon$ < 10).
There is no perturbation in the entropy $S=P/\rho^\gamma$ ($\delta S=0)$.

(problem-shock-tube)=

## Shock Tube

This test problem is based on Sod's Shock Tube (from{cite}`sodshocktubewiki`), a canonical test case for discontinuity capturing in one dimension. For this problem, the three-dimensional Euler equations are formulated exactly as in the Isentropic Vortex problem. The default initial conditions are $P=1$, $\rho=1$ for the driver section and $P=0.1$, $\rho=0.125$ for the driven section. The initial velocity is zero in both sections. Slip boundary conditions are applied to the side walls and wall boundary conditions are applied at the end walls.

SU upwinding and discontinuity capturing have been implemented into the explicit timestepping operator for this problem. Discontinuity capturing is accomplished using a modified version of the YZB operator described in {cite}`tezduyar2007yzb`. This discontinuity capturing scheme involves the introduction of a dissipation term of the form

$$
\int_{\Omega} \nu_{SHOCK} \nabla \bm v \!:\! \nabla \bm q dV
$$

The shock capturing viscosity is implemented following the first formulation described in {cite} `tezduyar2007yzb`. The characteristic velocity $u_{cha}$ is taken to be the acoustic speed while the reference density $\rho_{ref}$ is just the local density. Shock capturing viscosity is defined by the following

$$
\nu_{SHOCK} = \tau_{SHOCK} u_{cha}^2
$$
where,
$$
\tau_{SHOCK} = \frac{h_{SHOCK}}{2u_{cha}} \left( \frac{ \,|\, \nabla \rho \,|\, h_{SHOCK}}{\rho_{ref}} \right)^{\beta}
$$

$\beta$ is a tuning parameter set between 1 (smoother shocks) and 2 (sharper shocks. The parameter $h_{SHOCK}$ is a length scale that is proportional to the element length in the direction of the density gradient unit vector. This density gradient unit vector is defined as $\hat{\bm j} = \frac{\nabla \rho}{|\nabla \rho|}. The original formulation of Tezduyar and Senga relies on the shape function gradient to define the element length scale, but this gradient is not available to qFunctions in libCEED. To avoid this problem, $h_{SHOCK}$ is defined in the current implementation as

$$
h_{SHOCK} = 2 \left( C_{YZB} \,|\, \bm p \,|\, \right)^{-1}
$$
where
$$
p_k = \hat{j}_i \frac{\partial \xi_i}{x_k}
$$

The constant $C_{YZB}$ is set to 0.1 for piecewise linear elements in the current implementation. Larger values approaching unity are expected with more robust stabilization and implicit timestepping.

(problem-density-current)=

## Density Current

For this test problem (from {cite}`straka1993numerical`), we solve the full Navier-Stokes equations {eq}`eq-ns`, for which a cold air bubble (of radius $r_c$) drops by convection in a neutrally stratified atmosphere.
Its initial condition is defined in terms of the Exner pressure, $\pi(\bm{x},t)$, and potential temperature, $\theta(\bm{x},t)$, that relate to the state variables via

$$
\begin{aligned} \rho &= \frac{P_0}{( c_p - c_v)\theta(\bm{x},t)} \pi(\bm{x},t)^{\frac{c_v}{ c_p - c_v}} \, , \\ e &= c_v \theta(\bm{x},t) \pi(\bm{x},t) + \bm{u}\cdot \bm{u} /2 + g z \, , \end{aligned}
$$

where $P_0$ is the atmospheric pressure.
For this problem, we have used no-slip and non-penetration boundary conditions for $\bm{u}$, and no-flux for mass and energy densities.
