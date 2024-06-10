(example-petsc-elasticity)=

# Solid mechanics mini-app

This example is located in the subdirectory {file}`examples/solids`.
It solves the steady-state static momentum balance equations using unstructured high-order finite/spectral element spatial discretizations.
As for the {ref}`example-petsc-navier-stokes` case, the solid mechanics elasticity example has been developed using PETSc, so that the pointwise physics (defined at quadrature points) is separated from the parallelization and meshing concerns.

In this mini-app, we consider three formulations used in solid mechanics applications: linear elasticity, Neo-Hookean hyperelasticity at small strain, and Neo-Hookean hyperelasticity at finite strain.
We provide the strong and weak forms of static balance of linear momentum in the small strain and finite strain regimes.
The stress-strain relationship (constitutive law) for each of the material models is provided.
Due to the nonlinearity of material models in Neo-Hookean hyperelasticity, the Newton linearization of the material models is provided.

:::{note}
Linear elasticity and small-strain hyperelasticity can both by obtained from the finite-strain hyperelastic formulation by linearization of geometric and constitutive nonlinearities.
The effect of these linearizations is sketched in the diagram below, where $\bm \sigma$ and $\bm \epsilon$ are stress and strain, respectively, in the small strain regime, while $\bm S$ and $\bm E$ are their finite-strain generalizations (second Piola-Kirchoff tensor and Green-Lagrange strain tensor, respectively) defined in the initial configuration, and $\mathsf C$ is a linearized constitutive model.

$$
\begin{CD}
  {\overbrace{\bm S(\bm E)}^{\text{Finite Strain Hyperelastic}}}
  @>{\text{constitutive}}>{\text{linearization}}>
  {\overbrace{\bm S = \mathsf C \bm E}^{\text{St. Venant-Kirchoff}}} \\
  @V{\text{geometric}}V{\begin{smallmatrix}\bm E \to \bm \epsilon \\ \bm S \to \bm \sigma \end{smallmatrix}}V
  @V{\begin{smallmatrix}\bm E \to \bm \epsilon \\ \bm S \to \bm \sigma \end{smallmatrix}}V{\text{geometric}}V \\
  {\underbrace{\bm \sigma(\bm \epsilon)}_\text{Small Strain Hyperelastic}}
  @>{\text{constitutive}}>\text{linearization}>
  {\underbrace{\bm \sigma = \mathsf C \bm \epsilon}_\text{Linear Elastic}}
\end{CD}
$$ (hyperelastic-cd)
:::

(running-elasticity)=

## Running the mini-app

```{include} README.md
:start-after: inclusion-solids-marker
```

(problem-linear-elasticity)=

## Linear Elasticity

The strong form of the static balance of linear momentum at small strain for the three-dimensional linear elasticity problem is given by {cite}`hughes2012finite`:

$$
\nabla \cdot \bm{\sigma} + \bm{g} = \bm{0}
$$ (lin-elas)

where $\bm{\sigma}$ and $\bm{g}$ are stress and forcing functions, respectively.
We multiply {eq}`lin-elas` by a test function $\bm v$ and integrate the divergence term by parts to arrive at the weak form: find $\bm u \in \mathcal V \subset H^1(\Omega)$ such that

$$
\int_{\Omega}{ \nabla \bm{v} \tcolon \bm{\sigma}} \, dV
- \int_{\partial \Omega}{\bm{v} \cdot \left(\bm{\sigma} \cdot \hat{\bm{n}}\right)} \, dS
- \int_{\Omega}{\bm{v} \cdot \bm{g}} \, dV
= 0, \quad \forall \bm v \in \mathcal V,
$$ (lin-elas-weak)

where $\bm{\sigma} \cdot \hat{\bm{n}}|_{\partial \Omega}$ is replaced by an applied force/traction boundary condition written in terms of the initial configuration.
When inhomogeneous Dirichlet boundary conditions are present, $\mathcal V$ is an affine space that satisfies those boundary conditions.

### Constitutive modeling

In their most general form, constitutive models define $\bm \sigma$ in terms of state variables.
In the model taken into consideration in the present mini-app, the state variables are constituted by the vector displacement field $\bm u$, and its gradient $\nabla \bm u$.
We begin by defining the symmetric (small/infintesimal) strain tensor as

$$
\bm{\epsilon} = \dfrac{1}{2}\left(\nabla \bm{u} + \nabla \bm{u}^T \right).
$$ (small-strain)

This constitutive model $\bm \sigma(\bm \epsilon)$ is a linear tensor-valued function of a tensor-valued input, but we will consider the more general nonlinear case in other models below.
In these cases, an arbitrary choice of such a function will generally not be invariant under orthogonal transformations and thus will not admissible as a physical model must not depend on the coordinate system chosen to express it.
In particular, given an orthogonal transformation $Q$, we desire

$$
Q \bm \sigma(\bm \epsilon) Q^T = \bm \sigma(Q \bm \epsilon Q^T),
$$ (elastic-invariance)

which means that we can change our reference frame before or after computing $\bm \sigma$, and get the same result either way.
Constitutive relations in which $\bm \sigma$ is uniquely determined by $\bm \epsilon$ while satisfying the invariance property {eq}`elastic-invariance` are known as Cauchy elastic materials.
Here, we define a strain energy density functional $\Phi(\bm \epsilon) \in \mathbb R$ and obtain the strain energy from its gradient,

$$
\bm \sigma(\bm \epsilon) = \frac{\partial \Phi}{\partial \bm \epsilon}.
$$ (strain-energy-grad)

:::{note}
The strain energy density functional cannot be an arbitrary function $\Phi(\bm \epsilon)$; it can only depend on *invariants*, scalar-valued functions $\gamma$ satisfying

$$
\gamma(\bm \epsilon) = \gamma(Q \bm \epsilon Q^T)
$$

for all orthogonal matrices $Q$.
:::

For the linear elasticity model, the strain energy density is given by

$$
\bm{\Phi} = \frac{\lambda}{2} (\operatorname{trace} \bm{\epsilon})^2 + \mu \bm{\epsilon} : \bm{\epsilon} .
$$

The constitutive law (stress-strain relationship) is therefore given by its gradient,

$$
\bm\sigma = \lambda (\operatorname{trace} \bm\epsilon) \bm I_3 + 2 \mu \bm\epsilon,
$$

where $\bm I_3$ is the $3 \times 3$ identity matrix, the colon represents a double contraction (over both indices of $\bm \epsilon$), and the Lamé parameters are given by

$$
\begin{aligned} \lambda &= \frac{E \nu}{(1 + \nu)(1 - 2 \nu)} \\ \mu &= \frac{E}{2(1 + \nu)} \end{aligned}.
$$

The constitutive law (stress-strain relationship) can also be written as

$$
\bm{\sigma} = \mathsf{C} \!:\! \bm{\epsilon}.
$$ (linear-stress-strain)

For notational convenience, we express the symmetric second order tensors $\bm \sigma$ and $\bm \epsilon$ as vectors of length 6 using the [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation).
Hence, the fourth order elasticity tensor $\mathsf C$ (also known as elastic moduli tensor or material stiffness tensor) can be represented as

$$
\mathsf C = \begin{pmatrix}
\lambda + 2\mu & \lambda & \lambda & & & \\
\lambda & \lambda + 2\mu & \lambda & & & \\
\lambda & \lambda & \lambda + 2\mu & & & \\
& & & \mu & & \\
& & & & \mu & \\
& & & & & \mu
\end{pmatrix}.
$$ (linear-elasticity-tensor)

Note that the incompressible limit $\nu \to \frac 1 2$ causes $\lambda \to \infty$, and thus $\mathsf C$ becomes singular.

(problem-hyper-small-strain)=

## Hyperelasticity at Small Strain

The strong and weak forms given above, in {eq}`lin-elas` and {eq}`lin-elas-weak`, are valid for Neo-Hookean hyperelasticity at small strain.
However, the strain energy density differs and is given by

$$
\bm{\Phi} = \lambda (1 + \operatorname{trace} \bm{\epsilon}) (\log(1 + \operatorname{trace} \bm\epsilon) - 1) + \mu \bm{\epsilon} : \bm{\epsilon} .
$$

As above, we have the corresponding constitutive law given by

$$
\bm{\sigma} = \lambda \log(1 + \operatorname{trace} \bm\epsilon) \bm{I}_3 + 2\mu \bm{\epsilon}
$$ (eq-neo-hookean-small-strain)

where $\bm{\epsilon}$ is defined as in {eq}`small-strain`.

### Newton linearization

Due to nonlinearity in the constitutive law, we require a Newton linearization of {eq}`eq-neo-hookean-small-strain`.
To derive the Newton linearization, we begin by expressing the derivative,

$$
\diff \bm{\sigma} = \dfrac{\partial \bm{\sigma}}{\partial \bm{\epsilon}} \tcolon \diff \bm{\epsilon}
$$

where

$$
\diff \bm{\epsilon} = \dfrac{1}{2}\left( \nabla \diff \bm{u} + \nabla \diff \bm{u}^T \right)
$$

and

$$
\diff \nabla \bm{u} = \nabla \diff \bm{u} .
$$

Therefore,

$$
\diff \bm{\sigma}  = \bar{\lambda} \cdot \operatorname{trace} \diff \bm{\epsilon} \cdot \bm{I}_3 + 2\mu \diff \bm{\epsilon}
$$ (derss)

where we have introduced the symbol

$$
\bar{\lambda} = \dfrac{\lambda}{1 + \epsilon_v }
$$

where volumetric strain is given by $\epsilon_v = \sum_i \epsilon_{ii}$.

Equation {eq}`derss` can be written in Voigt matrix notation as follows:

$$
\begin{pmatrix}
  \diff \sigma_{11} \\
  \diff \sigma_{22} \\
  \diff \sigma_{33} \\
  \diff \sigma_{23} \\
  \diff \sigma_{13} \\
  \diff \sigma_{12}
\end{pmatrix}  =
\begin{pmatrix}
  2 \mu +\bar{\lambda} & \bar{\lambda} & \bar{\lambda} & & & \\
  \bar{\lambda} & 2 \mu +\bar{\lambda} & \bar{\lambda} & & & \\
  \bar{\lambda} & \bar{\lambda} & 2 \mu +\bar{\lambda} & & & \\
  & & & \mu & & \\
  & & & & \mu & \\
  & & & & & \mu \\
\end{pmatrix}
\begin{pmatrix}
  \diff \epsilon_{11} \\
  \diff \epsilon_{22} \\
  \diff \epsilon_{33} \\
  2 \diff \epsilon_{23} \\
  2 \diff \epsilon_{13} \\
  2 \diff \epsilon_{12}
\end{pmatrix}.
$$ (mdss)

(problem-hyperelasticity-finite-strain)=

## Hyperelasticity at Finite Strain

In the *total Lagrangian* approach for the Neo-Hookean hyperelasticity problem, the discrete equations are formulated with respect to the initial configuration.
In this formulation, we solve for displacement $\bm u(\bm X)$ in the reference frame $\bm X$.
The notation for elasticity at finite strain is inspired by {cite}`holzapfel2000nonlinear` to distinguish between the current and initial configurations.
As explained in the {ref}`common-notation` section, we denote by capital letters the reference frame and by small letters the current one.

The strong form of the static balance of linear-momentum at *finite strain* (total Lagrangian) is given by:

$$
- \nabla_X \cdot \bm{P} - \rho_0 \bm{g} = \bm{0}
$$ (sblFinS)

where the $_X$ in $\nabla_X$ indicates that the gradient is calculated with respect to the initial configuration in the finite strain regime.
$\bm{P}$ and $\bm{g}$ are the *first Piola-Kirchhoff stress* tensor and the prescribed forcing function, respectively.
$\rho_0$ is known as the *initial* mass density.
The tensor $\bm P$ is not symmetric, living in the current configuration on the left and the initial configuration on the right.

$\bm{P}$ can be decomposed as

$$
\bm{P} = \bm{F} \, \bm{S},
$$ (1st2nd)

where $\bm S$ is the *second Piola-Kirchhoff stress* tensor, a symmetric tensor defined entirely in the initial configuration, and $\bm{F} = \bm I_3 + \nabla_X \bm u$ is the deformation gradient.
Different constitutive models can define $\bm S$.

### Constitutive modeling

For the constitutive modeling of hyperelasticity at finite strain, we begin by defining two symmetric tensors in the initial configuration, the right Cauchy-Green tensor

$$
\bm C = \bm F^T \bm F
$$

and the Green-Lagrange strain tensor

$$
\bm E = \frac 1 2 (\bm C - \bm I_3) = \frac 1 2 \Big( \nabla_X \bm u + (\nabla_X \bm u)^T + (\nabla_X \bm u)^T \nabla_X \bm u \Big),
$$ (eq-green-lagrange-strain)

the latter of which converges to the linear strain tensor $\bm \epsilon$ in the small-deformation limit.
The constitutive models considered, appropriate for large deformations, express $\bm S$ as a function of $\bm E$, similar to the linear case, shown in equation  {eq}`linear-stress-strain`, which  expresses the relationship between $\bm\sigma$ and $\bm\epsilon$.

Recall that the strain energy density functional can only depend upon invariants.
We will assume without loss of generality that $\bm E$ is diagonal and take its set of eigenvalues as the invariants.
It is clear that there can be only three invariants, and there are many alternate choices, such as $\operatorname{trace}(\bm E), \operatorname{trace}(\bm E^2), \lvert \bm E \rvert$, and combinations thereof.
It is common in the literature for invariants to be taken from $\bm C = \bm I_3 + 2 \bm E$ instead of $\bm E$.

For example, if we take the compressible Neo-Hookean model,

$$
\begin{aligned}
\Phi(\bm E) &= \frac{\lambda}{2}(\log J)^2 - \mu \log J + \frac \mu 2 (\operatorname{trace} \bm C - 3) \\
  &= \frac{\lambda}{2}(\log J)^2 - \mu \log J + \mu \operatorname{trace} \bm E,
\end{aligned}
$$ (neo-hookean-energy)

where $J = \lvert \bm F \rvert = \sqrt{\lvert \bm C \rvert}$ is the determinant of deformation (i.e., volume change) and $\lambda$ and $\mu$ are the Lamé parameters in the infinitesimal strain limit.

To evaluate {eq}`strain-energy-grad`, we make use of

$$
\frac{\partial J}{\partial \bm E} = \frac{\partial \sqrt{\lvert \bm C \rvert}}{\partial \bm E} = \lvert \bm C \rvert^{-1/2} \lvert \bm C \rvert \bm C^{-1} = J \bm C^{-1},
$$

where the factor of $\frac 1 2$ has been absorbed due to $\bm C = \bm I_3 + 2 \bm E.$
Carrying through the differentiation {eq}`strain-energy-grad` for the model {eq}`neo-hookean-energy`, we arrive at

$$
\bm S = \lambda \log J \bm C^{-1} + \mu (\bm I_3 - \bm C^{-1}).
$$ (neo-hookean-stress)

:::{tip}
An equivalent form of {eq}`neo-hookean-stress` is

$$
\bm S = \lambda \log J \bm C^{-1} + 2 \mu \bm C^{-1} \bm E,
$$ (neo-hookean-stress-stable)

which is more numerically stable for small $\bm E$, and thus preferred for computation.
Note that the product $\bm C^{-1} \bm E$ is also symmetric, and that $\bm E$ should be computed using {eq}`eq-green-lagrange-strain`.

Similarly, it is preferable to compute $\log J$ using `log1p`, especially in case of nearly incompressible materials.
To sketch this idea, suppose we have the $2\times 2$ non-symmetric matrix $\bm{F} = \left( \begin{smallmatrix} 1 + u_{0,0} & u_{0,1} \\ u_{1,0} & 1 + u_{1,1} \end{smallmatrix} \right)$.
Then we compute

$$
\log J = \mathtt{log1p}(u_{0,0} + u_{1,1} + u_{0,0} u_{1,1} - u_{0,1} u_{1,0}),
$$ (log1p)

which gives accurate results even in the limit when the entries $u_{i,j}$ are very small.
For example, if $u_{i,j} \sim 10^{-8}$, then naive computation of $\bm I_3 - \bm C^{-1}$ and $\log J$ will have a relative accuracy of order $10^{-8}$ in double precision and no correct digits in single precision.
When using the stable choices above, these quantities retain full $\varepsilon_{\text{machine}}$ relative accuracy.
:::

:::{dropdown} Mooney-Rivlin model
While the Neo-Hookean model depends on just two scalar invariants, $\mathbb I_1 = \trace \bm C = 3 + 2\trace \bm E$ and $J$, Mooney-Rivlin models depend on the additional invariant, $\mathbb I_2 = \frac 1 2 (\mathbb I_1^2 - \bm C \tcolon \bm C)$.
A coupled Mooney-Rivlin strain energy density (cf. Neo-Hookean {eq}`neo-hookean-energy`) is {cite}`holzapfel2000nonlinear`

$$
\Phi(\mathbb{I_1}, \mathbb{I_2}, J) = \frac{\lambda}{2}(\log J)^2 - (\mu_1 + 2\mu_2) \log J + \frac{\mu_1}{2}(\mathbb{I_1} - 3) + \frac{\mu_2}{2}(\mathbb{I_2} - 3).
$$ (mooney-rivlin-energy_coupled)

We differentiate $\Phi$ as in the Neo-Hookean case {eq}`neo-hookean-stress` to yield the second Piola-Kirchoff tensor,

$$
\begin{aligned}
\bm S &=  \lambda \log J \bm{C}^{-1} - (\mu_1 + 2\mu_2) \bm{C}^{-1} + \mu_1\bm I_3 + \mu_2(\mathbb{I_1} \bm I_3 - \bm C) \\
&= (\lambda \log J - \mu_1 - 2\mu_2) \bm C^{-1} + (\mu_1 + \mu_2 \mathbb I_1) \bm I_3 - \mu_2 \bm C,
\end{aligned}
$$ (mooney-rivlin-stress_coupled)

where we have used

$$
\begin{aligned}
\frac{\partial \mathbb{I_1}}{\partial \bm E} &= 2 \bm I_3, & \frac{\partial \mathbb{I_2}}{\partial \bm E} &= 2 \mathbb I_1 \bm I_3 - 2 \bm C, & \frac{\partial \log J}{\partial \bm E} &= \bm{C}^{-1}.
\end{aligned}
$$ (None)

This is a common model for vulcanized rubber, with a shear modulus (defined for the small-strain limit) of $\mu_1 + \mu_2$ that should be significantly smaller than the first Lamé parameter $\lambda$.
:::

:::{dropdown} Mooney-Rivlin strain energy comparison
We apply traction to a block and plot integrated strain energy $\Phi$ as a function of the loading paramater.

```{altair-plot}
:hide-code:

import altair as alt
import pandas as pd
def source_path(rel):
    import os
    return os.path.join(os.path.dirname(os.environ["DOCUTILSCONFIG"]), rel)

nh = pd.read_csv(source_path("examples/solids/tests-output/NH-strain.csv"))
nh["model"] = "Neo-Hookean"
nh["parameters"] = "E=2.8, nu=0.4"

mr = pd.read_csv(source_path("examples/solids/tests-output/MR-strain.csv"))
mr["model"] = "Mooney-Rivlin; Neo-Hookean equivalent"
mr["parameters"] = "mu_1=1, mu_2=0, nu=.4"

mr1 = pd.read_csv(source_path("examples/solids/tests-output/MR-strain1.csv"))
mr1["model"] = "Mooney-Rivlin"
mr1["parameters"] = "mu_1=0.5, mu_2=0.5, nu=.4"

df = pd.concat([nh, mr, mr1])
highlight = alt.selection_point(
   on = "mouseover",
   nearest = True,
   fields=["model", "parameters"],
)
base = alt.Chart(df).encode(
   alt.X("increment"),
   alt.Y("energy", scale=alt.Scale(type="sqrt")),
   alt.Color("model"),
   alt.Tooltip(("model", "parameters")),
   opacity=alt.condition(highlight, alt.value(1), alt.value(.5)),
   size=alt.condition(highlight, alt.value(2), alt.value(1)),
)
base.mark_point().add_params(highlight) + base.mark_line()
```
:::

:::{note}
One can linearize {eq}`neo-hookean-stress` around $\bm E = 0$, for which $\bm C = \bm I_3 + 2 \bm E \to \bm I_3$ and $J \to 1 + \operatorname{trace} \bm E$, therefore {eq}`neo-hookean-stress` reduces to

$$
\bm S = \lambda (\trace \bm E) \bm I_3 + 2 \mu \bm E,
$$ (eq-st-venant-kirchoff)

which is the St. Venant-Kirchoff model (constitutive linearization without geometric linearization; see {eq}`hyperelastic-cd`).

This model can be used for geometrically nonlinear mechanics (e.g., snap-through of thin structures), but is inappropriate for large strain.

Alternatively, one can drop geometric nonlinearities, $\bm E \to \bm \epsilon$ and $\bm C \to \bm I_3$, while retaining the nonlinear dependence on $J \to 1 + \operatorname{trace} \bm \epsilon$, thereby yielding {eq}`eq-neo-hookean-small-strain` (see {eq}`hyperelastic-cd`).
:::

### Weak form

We multiply {eq}`sblFinS` by a test function $\bm v$ and integrate by parts to obtain the weak form for finite-strain hyperelasticity:
find $\bm u \in \mathcal V \subset H^1(\Omega_0)$ such that

$$
\int_{\Omega_0}{\nabla_X \bm{v} \tcolon \bm{P}} \, dV
 - \int_{\Omega_0}{\bm{v} \cdot \rho_0 \bm{g}} \, dV
 - \int_{\partial \Omega_0}{\bm{v} \cdot (\bm{P} \cdot \hat{\bm{N}})} \, dS
 = 0, \quad \forall \bm v \in \mathcal V,
$$ (hyperelastic-weak-form-initial)

where $\bm{P} \cdot \hat{\bm{N}}|_{\partial\Omega}$ is replaced by any prescribed force/traction boundary condition written in terms of the initial configuration.
This equation contains material/constitutive nonlinearities in defining $\bm S(\bm E)$, as well as geometric nonlinearities through $\bm P = \bm F\, \bm S$, $\bm E(\bm F)$, and the body force $\bm g$, which must be pulled back from the current configuration to the initial configuration.
Discretization of {eq}`hyperelastic-weak-form-initial` produces a finite-dimensional system of nonlinear algebraic equations, which we solve using Newton-Raphson methods.
One attractive feature of Galerkin discretization is that we can arrive at the same linear system by discretizing the Newton linearization of the continuous form; that is, discretization and differentiation (Newton linearization) commute.

### Newton linearization

To derive a Newton linearization of {eq}`hyperelastic-weak-form-initial`, we begin by expressing the derivative of {eq}`1st2nd` in incremental form,

$$
\diff \bm P = \frac{\partial \bm P}{\partial \bm F} \!:\! \diff \bm F = \diff \bm F\, \bm S + \bm F \underbrace{\frac{\partial \bm S}{\partial \bm E} \!:\! \diff \bm E}_{\diff \bm S}
$$ (eq-diff-P)

where

$$
\diff \bm E = \frac{\partial \bm E}{\partial \bm F} \!:\! \diff \bm F = \frac 1 2 \Big( \diff \bm F^T \bm F + \bm F^T \diff \bm F \Big)
$$

and $\diff\bm F = \nabla_X\diff\bm u$.
The quantity ${\partial \bm S} / {\partial \bm E}$ is known as the incremental elasticity tensor, and is analogous to the linear elasticity tensor $\mathsf C$ of {eq}`linear-elasticity-tensor`.
We now evaluate $\diff \bm S$ for the Neo-Hookean model {eq}`neo-hookean-stress`,

$$
\diff\bm S = \frac{\partial \bm S}{\partial \bm E} \!:\! \diff \bm E
= \lambda (\bm C^{-1} \!:\! \diff\bm E) \bm C^{-1}
  + 2 (\mu - \lambda \log J) \bm C^{-1} \diff\bm E \, \bm C^{-1},
$$ (eq-neo-hookean-incremental-stress)

where we have used

$$
\diff \bm C^{-1} = \frac{\partial \bm C^{-1}}{\partial \bm E} \!:\! \diff\bm E = -2 \bm C^{-1} \diff \bm E \, \bm C^{-1} .
$$

:::{note}
In the small-strain limit, $\bm C \to \bm I_3$ and $\log J \to 0$, thereby reducing {eq}`eq-neo-hookean-incremental-stress` to the St. Venant-Kirchoff model {eq}`eq-st-venant-kirchoff`.
:::

:::{dropdown} Newton linearization of Mooney-Rivlin
Similar to {eq}`eq-neo-hookean-incremental-stress`, we differentiate {eq}`mooney-rivlin-stress_coupled` using variational notation,

$$
\begin{aligned}
\diff\bm S &= \lambda (\bm C^{-1} \tcolon \diff\bm E) \bm C^{-1} \\
&\quad + 2(\mu_1 + 2\mu_2 - \lambda \log J) \bm C^{-1} \diff\bm E \bm C^{-1} \\
&\quad + 2 \mu_2 \Big[ \trace (\diff\bm E) \bm I_3 - \diff\bm E\Big] .
\end{aligned}
$$ (mooney-rivlin-dS-coupled)

Note that this agrees with {eq}`eq-neo-hookean-incremental-stress` if $\mu_1 = \mu, \mu_2 = 0$.
Moving from Neo-Hookean to Mooney-Rivlin modifies the second term and adds the third.
:::

:::{dropdown} Cancellation vs symmetry
Some cancellation is possible (at the expense of symmetry) if we substitute {eq}`eq-neo-hookean-incremental-stress` into {eq}`eq-diff-P`,

$$
\begin{aligned}
\diff \bm P &= \diff \bm F\, \bm S
  + \lambda (\bm C^{-1} : \diff \bm E) \bm F^{-T} + 2(\mu - \lambda \log J) \bm F^{-T} \diff\bm E \, \bm C^{-1} \\
&= \diff \bm F\, \bm S
  + \lambda (\bm F^{-T} : \diff \bm F) \bm F^{-T} + (\mu - \lambda \log J) \bm F^{-T} (\bm F^T \diff \bm F + \diff \bm F^T \bm F) \bm C^{-1} \\
&= \diff \bm F\, \bm S
  + \lambda (\bm F^{-T} : \diff \bm F) \bm F^{-T} + (\mu - \lambda \log J) \Big( \diff \bm F\, \bm C^{-1} + \bm F^{-T} \diff \bm F^T \bm F^{-T} \Big),
\end{aligned}
$$ (eq-diff-P-dF)

where we have exploited $\bm F \bm C^{-1} = \bm F^{-T}$ and

$$
\begin{aligned} \bm C^{-1} \!:\! \diff \bm E = \bm C_{IJ}^{-1} \diff \bm E_{IJ} &= \frac 1 2 \bm F_{Ik}^{-1} \bm F_{Jk}^{-1} (\bm F_{\ell I} \diff \bm F_{\ell J} + \diff \bm F_{\ell I} \bm F_{\ell J}) \\ &= \frac 1 2 \Big( \delta_{\ell k} \bm F_{Jk}^{-1} \diff \bm F_{\ell J} + \delta_{\ell k} \bm F_{Ik}^{-1} \diff \bm F_{\ell I} \Big) \\ &= \bm F_{Ik}^{-1} \diff \bm F_{kI} = \bm F^{-T} \!:\! \diff \bm F. \end{aligned}
$$

We prefer to compute with {eq}`eq-neo-hookean-incremental-stress` because {eq}`eq-diff-P-dF` is more expensive, requiring access to (non-symmetric) $\bm F^{-1}$ in addition to (symmetric) $\bm C^{-1} = \bm F^{-1} \bm F^{-T}$, having fewer symmetries to exploit in contractions, and being less numerically stable.
:::

:::{dropdown} $\diff\bm S$ in index notation
It is sometimes useful to express {eq}`eq-neo-hookean-incremental-stress` in index notation,

$$
\begin{aligned}
\diff\bm S_{IJ} &= \frac{\partial \bm S_{IJ}}{\partial \bm E_{KL}} \diff \bm E_{KL} \\
  &= \lambda (\bm C^{-1}_{KL} \diff\bm E_{KL}) \bm C^{-1}_{IJ} + 2 (\mu - \lambda \log J) \bm C^{-1}_{IK} \diff\bm E_{KL} \bm C^{-1}_{LJ} \\
  &= \underbrace{\Big( \lambda \bm C^{-1}_{IJ} \bm C^{-1}_{KL} + 2 (\mu - \lambda \log J) \bm C^{-1}_{IK} \bm C^{-1}_{JL} \Big)}_{\mathsf C_{IJKL}} \diff \bm E_{KL} \,,
\end{aligned}
$$ (eq-neo-hookean-incremental-stress-index)

where we have identified the effective elasticity tensor $\mathsf C = \mathsf C_{IJKL}$.
It is generally not desirable to store $\mathsf C$, but rather to use the earlier expressions so that only $3\times 3$ tensors (most of which are symmetric) must be manipulated.
That is, given the linearization point $\bm F$ and solution increment $\diff \bm F = \nabla_X (\diff \bm u)$ (which we are solving for in the Newton step), we compute $\diff \bm P$ via

1. recover $\bm C^{-1}$ and $\log J$ (either stored at quadrature points or recomputed),
2. proceed with $3\times 3$ matrix products as in {eq}`eq-neo-hookean-incremental-stress` or the second line of {eq}`eq-neo-hookean-incremental-stress-index` to compute $\diff \bm S$ while avoiding computation or storage of higher order tensors, and
3. conclude by {eq}`eq-diff-P`, where $\bm S$ is either stored or recomputed from its definition exactly as in the nonlinear residual evaluation.
:::

Note that the Newton linearization of {eq}`hyperelastic-weak-form-initial` may be written as a weak form for linear operators: find $\diff\bm u \in \mathcal V_0$ such that

$$
\int_{\Omega_0} \nabla_X \bm v \!:\! \diff\bm P dV = \text{rhs}, \quad \forall \bm v \in \mathcal V_0,
$$

where $\diff \bm P$ is defined by {eq}`eq-diff-P` and {eq}`eq-neo-hookean-incremental-stress`, and $\mathcal V_0$ is the homogeneous space corresponding to $\mathcal V$.

:::{note}
The decision of whether to recompute or store functions of the current state $\bm F$ depends on a roofline analysis {cite}`williams2009roofline,brown2010` of the computation and the cost of the constitutive model.
For low-order elements where flops tend to be in surplus relative to memory bandwidth, recomputation is likely to be preferable, where as the opposite may be true for high-order elements.
Similarly, analysis with a simple constitutive model may see better performance while storing little or nothing while an expensive model such as Arruda-Boyce {cite}`arruda1993largestretch`, which contains many special functions, may be faster when using more storage to avoid recomputation.
In the case where complete linearization is preferred, note the symmetry $\mathsf C_{IJKL} = \mathsf C_{KLIJ}$ evident in {eq}`eq-neo-hookean-incremental-stress-index`, thus $\mathsf C$ can be stored as a symmetric $6\times 6$ matrix, which has 21 unique entries.
Along with 6 entries for $\bm S$, this totals 27 entries of overhead compared to computing everything from $\bm F$.
This compares with 13 entries of overhead for direct storage of $\{ \bm S, \bm C^{-1}, \log J \}$, which is sufficient for the Neo-Hookean model to avoid all but matrix products.
:::

(problem-hyperelasticity-finite-strain-current-configuration)=

## Hyperelasticity in current configuration

In the preceeding discussion, all equations have been formulated in the initial configuration.
This may feel convenient in that the computational domain is clearly independent of the solution, but there are some advantages to defining the equations in the current configuration.

1. Body forces (like gravity), traction, and contact are more easily defined in the current configuration.
2. Mesh quality in the initial configuration can be very bad for large deformation.
3. The required storage and numerical representation can be smaller in the current configuration.

Most of the benefit in case 3 can be attained solely by moving the Jacobian representation to the current configuration {cite}`davydov2020matrix`, though residual evaluation may also be slightly faster in current configuration.
There are multiple commuting paths from the nonlinear weak form in initial configuration {eq}`hyperelastic-weak-form-initial` to the Jacobian weak form in current configuration {eq}`jacobian-weak-form-current`.
One may push forward to the current configuration and then linearize or linearize in initial configuration and then push forward, as summarized below.

$$
\begin{CD}
  {\overbrace{\nabla_X \bm{v} \tcolon \bm{FS}}^{\text{Initial Residual}}}
  @>{\text{push forward}}>{}>
  {\overbrace{\nabla_x \bm{v} \tcolon \bm{\tau}}^{\text{Current Residual}}} \\
  @V{\text{linearize}}V{\begin{smallmatrix} \diff\bm F = \nabla_X\diff\bm u \\ \diff\bm S(\diff\bm E) \end{smallmatrix}}V
  @V{\begin{smallmatrix} \diff\nabla_x\bm v = -\nabla_x\bm v \nabla_x \diff\bm u \\ \diff\bm\tau(\diff\bm\epsilon) \end{smallmatrix}}V{\text{linearize}}V \\
  {\underbrace{\nabla_X\bm{v}\tcolon \Big(\diff\bm{F}\bm{S} + \bm{F}\diff\bm{S}\Big)}_\text{Initial Jacobian}}
  @>{\text{push forward}}>{}>
  {\underbrace{\nabla_x\bm{v}\tcolon \Big(\diff\bm{\tau} -\bm{\tau}(\nabla_x \diff\bm{u})^T \Big)}_\text{Current Jacobian}}
\end{CD}
$$ (initial-current-linearize)

We will follow both paths for consistency and because both intermediate representations may be useful for implementation.

### Push forward, then linearize

The first term of {eq}`hyperelastic-weak-form-initial` can be rewritten in terms of the symmetric Kirchhoff stress tensor
$\bm{\tau}=J\bm{\sigma}=\bm{P}\bm{F}^T = \bm F \bm S \bm F^T$ as

$$
\nabla_X \bm{v} \tcolon \bm{P} = \nabla_X \bm{v} \tcolon \bm{\tau}\bm{F}^{-T} = \nabla_X \bm{v}\bm{F}^{-1} \tcolon \bm{\tau} = \nabla_x \bm{v} \tcolon \bm{\tau}
$$

therefore, the weak form in terms of $\bm{\tau}$ and $\nabla_x$ with integral over $\Omega_0$ is

$$
\int_{\Omega_0}{\nabla_x \bm{v} \tcolon \bm{\tau}} \, dV
 - \int_{\Omega_0}{\bm{v} \cdot \rho_0 \bm{g}} \, dV
 - \int_{\partial \Omega_0}{\bm{v}\cdot(\bm{P}\cdot\hat{\bm{N}})} \, dS
 = 0, \quad \forall \bm v \in \mathcal V.
$$ (hyperelastic-weak-form-current)

#### Linearize in current configuration

To derive a Newton linearization of {eq}`hyperelastic-weak-form-current`, first we define

$$
\nabla_x \diff \bm{u} = \nabla_X \diff \bm{u} \  \bm{F}^{-1} = \diff \bm{F} \bm{F}^{-1}
$$ (nabla_xdu)

and $\bm{\tau}$ for Neo-Hookean materials as the push forward of {eq}`neo-hookean-stress`

$$
\bm{\tau} = \bm{F}\bm{S}\bm{F}^T = \mu (\bm{b} - \bm I_3) + \lambda \log J \bm{I}_3,
$$ (tau-neo-hookean)

where $\bm{b} = \bm{F} \bm{F}^T$, is the left Cauchy-Green tensor.
Then by expanding the directional derivative of $\nabla_x \bm{v} \tcolon \bm{\tau}$, we arrive at

$$
\diff \ (\nabla_x \bm{v} \tcolon \bm{\tau}) = \diff \ (\nabla_x \bm{v})\tcolon \bm{\tau} + \nabla_x \bm{v} \tcolon \diff \bm{\tau} .
$$ (hyperelastic-linearization-current1)

The first term of {eq}`hyperelastic-linearization-current1` can be written as

$$
\begin{aligned} \diff \ (\nabla_x \bm{v})\tcolon \bm{\tau} &= \diff \ (\nabla_X \bm{v} \bm{F}^{-1})\tcolon \bm{\tau} = \Big(\underbrace{\nabla_X (\diff \bm{v})}_{0}\bm{F}^{-1} +  \nabla_X \bm{v}\diff \bm{F}^{-1}\Big)\tcolon \bm{\tau}\\   &= \Big(-\nabla_X \bm{v} \bm{F}^{-1}\diff\bm{F}\bm{F}^{-1}\Big)\tcolon \bm{\tau}=\Big(-\nabla_x \bm{v} \diff\bm{F}\bm{F}^{-1}\Big)\tcolon \bm{\tau}\\   &= \Big(-\nabla_x \bm{v} \nabla_x \diff\bm{u} \Big)\tcolon \bm{\tau}= -\nabla_x \bm{v}\tcolon\bm{\tau}(\nabla_x \diff\bm{u})^T \,, \end{aligned}
$$

where we have used $\diff \bm{F}^{-1}=-\bm{F}^{-1} \diff \bm{F} \bm{F}^{-1}$ and {eq}`nabla_xdu`.
Using this and {eq}`hyperelastic-linearization-current1` in {eq}`hyperelastic-weak-form-current` yields the weak form in the current configuration

$$
\int_{\Omega_0} \nabla_x \bm v \tcolon \Big(\diff\bm\tau - \bm\tau (\nabla_x \diff\bm u)^T \Big) = \text{rhs}.
$$ (jacobian-weak-form-current)

In the following, we will sometimes make use of the incremental strain tensor in the current configuration,

$$
\diff\bm\epsilon \equiv \frac{1}{2}\Big(\nabla_x \diff\bm{u} + (\nabla_x \diff\bm{u})^T   \Big) .
$$

:::{dropdown} Deriving $\diff\bm\tau$ for Neo-Hookean material
To derive a useful expression of $\diff\bm\tau$ for Neo-Hookean materials, we will use the representations

$$
\begin{aligned}
\diff \bm{b} &= \diff \bm{F} \bm{F}^T + \bm{F} \diff \bm{F}^T \\
&= \nabla_x \diff \bm{u} \ \bm{b} + \bm{b} \ (\nabla_x \diff \bm{u})^T \\
&= (\nabla_x \diff\bm u)(\bm b - \bm I_3) + (\bm b - \bm I_3) (\nabla_x \diff\bm u)^T + 2 \diff\bm\epsilon
\end{aligned}
$$

and

$$
\begin{aligned} \diff\ (\log J) &= \frac{\partial \log J}{\partial \bm{b}}\tcolon \diff \bm{b} = \frac{\partial J}{J\partial \bm{b}}\tcolon \diff \bm{b}=\frac{1}{2}\bm{b}^{-1}\tcolon \diff \bm{b} \\ &= \frac 1 2 \bm b^{-1} \tcolon \Big(\nabla_x \diff\bm u \ \bm b + \bm b (\nabla_x \diff\bm u)^T \Big) \\ &= \trace (\nabla_x \diff\bm u) \\ &= \trace \diff\bm\epsilon . \end{aligned}
$$

Substituting into {eq}`tau-neo-hookean` gives

$$
\begin{aligned}
\diff \bm{\tau} &= \mu \diff \bm{b} + \lambda \trace (\diff\bm\epsilon) \bm I_3 \\
&= \underbrace{2 \mu \diff\bm\epsilon + \lambda \trace (\diff\bm\epsilon) \bm I_3 - 2\lambda \log J \diff\bm\epsilon}_{\bm F \diff\bm S \bm F^T} \\
&\quad + (\nabla_x \diff\bm u)\underbrace{\Big( \mu (\bm b - \bm I_3) + \lambda \log J \bm I_3 \Big)}_{\bm\tau} \\
&\quad + \underbrace{\Big( \mu (\bm b - \bm I_3) + \lambda \log J \bm I_3 \Big)}_{\bm\tau}  (\nabla_x \diff\bm u)^T ,
\end{aligned}
$$ (dtau-neo-hookean)

where the final expression has been identified according to

$$
\diff\bm\tau = \diff\ (\bm F \bm S \bm F^T) = (\nabla_x \diff\bm u) \bm\tau + \bm F \diff\bm S \bm F^T + \bm\tau(\nabla_x \diff\bm u)^T.
$$
:::

Collecting terms, we may thus opt to use either of the two forms

$$
\begin{aligned}
\diff \bm{\tau} -\bm{\tau}(\nabla_x \diff\bm{u})^T &= (\nabla_x \diff\bm u)\bm\tau + \bm F \diff\bm S \bm F^T \\
&= (\nabla_x \diff\bm u)\bm\tau + \lambda \trace(\diff\bm\epsilon) \bm I_3 + 2(\mu - \lambda \log J) \diff\bm\epsilon,
\end{aligned}
$$ (cur_simp_Jac)

with the last line showing the especially compact representation available for Neo-Hookean materials.

### Linearize, then push forward

We can move the derivatives to the current configuration via

$$
\nabla_X \bm v \!:\! \diff\bm P = (\nabla_X \bm v) \bm F^{-1} \!:\! \diff \bm P \bm F^T = \nabla_x \bm v \!:\! \diff\bm P \bm F^T
$$

and expand

$$
\begin{aligned}
\diff\bm P \bm F^T &= \diff\bm F \bm S \bm F^T + \bm F \diff\bm S \bm F^T \\
&= \underbrace{\diff\bm F \bm F^{-1}}_{\nabla_x \diff\bm u} \underbrace{\bm F \bm S \bm F^T}_{\bm\tau} + \bm F \diff\bm S \bm F^T .
\end{aligned}
$$

:::{dropdown} Representation of $\bm F \diff\bm S \bm F^T$ for Neo-Hookean materials
Now we push {eq}`eq-neo-hookean-incremental-stress` forward via

$$
\begin{aligned}
\bm F \diff\bm S \bm F^T &= \lambda (\bm C^{-1} \!:\! \diff\bm E) \bm F \bm C^{-1} \bm F^T
  + 2 (\mu - \lambda \log J) \bm F \bm C^{-1} \diff\bm E \, \bm C^{-1} \bm F^T \\
    &= \lambda (\bm C^{-1} \!:\! \diff\bm E) \bm I_3 + 2 (\mu - \lambda \log J) \bm F^{-T} \diff\bm E \, \bm F^{-1} \\
    &= \lambda \operatorname{trace}(\nabla_x \diff\bm u) \bm I_3 + 2 (\mu - \lambda \log J) \diff\bm \epsilon
\end{aligned}
$$

where we have used

$$
\begin{aligned}
\bm C^{-1} \!:\! \diff\bm E &= \bm F^{-1} \bm F^{-T} \!:\! \bm F^T \diff\bm F \\
&= \operatorname{trace}(\bm F^{-1} \bm F^{-T} \bm F^T \diff \bm F) \\
&= \operatorname{trace}(\bm F^{-1} \diff\bm F) \\
&= \operatorname{trace}(\diff \bm F \bm F^{-1}) \\
&= \operatorname{trace}(\nabla_x \diff\bm u)
\end{aligned}
$$

and

$$
\begin{aligned}
\bm F^{-T} \diff\bm E \, \bm F^{-1} &= \frac 1 2 \bm F^{-T} (\bm F^T \diff\bm F + \diff\bm F^T \bm F) \bm F^{-1} \\
&= \frac 1 2 (\diff \bm F \bm F^{-1} + \bm F^{-T} \diff\bm F^T) \\
&= \frac 1 2 \Big(\nabla_x \diff\bm u + (\nabla_x\diff\bm u)^T \Big) \equiv \diff\bm\epsilon.
\end{aligned}
$$
:::

Collecting terms, the weak form of the Newton linearization for Neo-Hookean materials in the current configuration is

$$
\int_{\Omega_0} \nabla_x \bm v \!:\! \Big( (\nabla_x \diff\bm u) \bm\tau + \lambda \operatorname{trace}(\diff\bm\epsilon)\bm I_3 + 2(\mu - \lambda\log J)\diff \bm\epsilon \Big) dV = \text{rhs},
$$ (jacobian-weak-form-current2)

which equivalent to Algorithm 2 of {cite}`davydov2020matrix` and requires only derivatives with respect to the current configuration. Note that {eq}`cur_simp_Jac` and {eq}`jacobian-weak-form-current2` have recovered the same representation
using different algebraic manipulations.

:::{tip}
We define a second order *Green-Euler* strain tensor (cf. Green-Lagrange strain {eq}`eq-green-lagrange-strain`) as

$$
\bm e = \frac 1 2 \Big(\bm{b} - \bm{I}_3 \Big) = \frac 1 2 \Big( \nabla_X \bm{u} + (\nabla_X \bm{u})^T + \nabla_X \bm{u} \, (\nabla_X \bm{u})^T \Big).
$$ (green-euler-strain)

Then, the Kirchhoff stress tensor {eq}`tau-neo-hookean` can be written as

$$
\bm \tau = \lambda \log J \bm I_{3} + 2\mu \bm e,
$$ (tau-neo-hookean-stable)

which is more numerically stable for small strain, and thus preferred for computation. Note that the $\log J$ is computed via `log1p` {eq}`log1p`, as we discussed in the previous tip.
:::
