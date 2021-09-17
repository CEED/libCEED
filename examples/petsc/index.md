# PETSc demos and BPs

(example-petsc-area)=

## Area

This example is located in the subdirectory {file}`examples/petsc`.
It demonstrates a simple usage of libCEED with PETSc to calculate the surface area of a closed surface.
The code uses higher level communication protocols for mesh handling in PETSc's DMPlex.
This example has the same mathematical formulation as {ref}`Ex1-Volume`, with the exception that the physical coordinates for this problem are $\bm{x}=(x,y,z)\in \mathbb{R}^3$, while the coordinates of the reference element are $\bm{X}=(X,Y) \equiv (X_0,X_1) \in \textrm{I} =[-1,1]^2$.

(example-petsc-area-cube)=

### Cube

This is one of the test cases of the computation of the {ref}`example-petsc-area` of a 2D manifold embedded in 3D.
This problem can be run with:

```
./area -problem cube
```

This example uses the following coordinate transformations for the computation of the geometric factors: from the physical coordinates on the cube, denoted by $\bar{\bm{x}}=(\bar{x},\bar{y},\bar{z})$, and physical coordinates on the discrete surface, denoted by $\bm{{x}}=(x,y)$, to $\bm{X}=(X,Y) \in \textrm{I}$ on the reference element, via the chain rule

$$
\frac{\partial \bm{x}}{\partial \bm{X}}_{(2\times2)} = \frac{\partial {\bm{x}}}{\partial \bar{\bm{x}}}_{(2\times3)} \frac{\partial \bar{\bm{x}}}{\partial \bm{X}}_{(3\times2)},
$$ (eq-coordinate-transforms-cube)

with Jacobian determinant given by

$$
\left| J \right| = \left\|col_1\left(\frac{\partial \bar{\bm{x}}}{\partial \bm{X}}\right)\right\| \left\|col_2 \left(\frac{\partial \bar{\bm{x}}}{\partial \bm{X}}\right) \right\|
$$ (eq-jacobian-cube)

We note that in equation {eq}`eq-coordinate-transforms-cube`, the right-most Jacobian matrix ${\partial\bar{\bm{x}}}/{\partial \bm{X}}_{(3\times2)}$ is provided by the library, while ${\partial{\bm{x}}}/{\partial \bar{ \bm{x}}}_{(2\times3)}$ is provided by the user as

$$
\left[ col_1\left(\frac{\partial\bar{\bm{x}}}{\partial \bm{X}}\right) / \left\| col_1\left(\frac{\partial\bar{\bm{x}}}{\partial \bm{X}}\right)\right\| , col_2\left(\frac{\partial\bar{\bm{x}}}{\partial \bm{X}}\right) / \left\| col_2\left(\frac{\partial\bar{\bm{x}}}{\partial \bm{X}}\right)\right\| \right]^T_{(2\times 3)}.
$$

(example-petsc-area-sphere)=

### Sphere

This problem computes the surface {ref}`example-petsc-area` of a tensor-product discrete sphere, obtained by projecting a cube inscribed in a sphere onto the surface of the sphere.
This discrete surface is sometimes referred to as a cubed-sphere (an example of such as a surface is given in figure {numref}`fig-cubed-sphere`).
This problem can be run with:

```
./area -problem sphere
```

(fig-cubed-sphere)=

:::{figure} ../../../../img/CubedSphere.svg
Example of a cubed-sphere, i.e., a tensor-product discrete sphere, obtained by
projecting a cube inscribed in a sphere onto the surface of the sphere.
:::

This example uses the following coordinate transformations for the computation of the geometric factors: from the physical coordinates on the sphere, denoted by $\overset{\circ}{\bm{x}}=(\overset{\circ}{x},\overset{\circ}{y},\overset{\circ}{z})$, and physical coordinates on the discrete surface, denoted by $\bm{{x}}=(x,y,z)$ (depicted, for simplicity, as coordinates on a circle and 1D linear element in figure {numref}`fig-sphere-coords`), to $\bm{X}=(X,Y) \in \textrm{I}$ on the reference element, via the chain rule

$$
\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}_{(3\times2)} = \frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{x}}_{(3\times3)} \frac{\partial\bm{x}}{\partial \bm{X}}_{(3\times2)} ,
$$ (eq-coordinate-transforms-sphere)

with Jacobian determinant given by

$$
\left| J \right| = \left| col_1\left(\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}\right) \times col_2 \left(\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}\right)\right| .
$$ (eq-jacobian-sphere)

(fig-sphere-coords)=

:::{figure} ../../../../img/SphereSketch.svg
Sketch of coordinates mapping between a 1D linear element and a circle.
In the case of a linear element the two nodes, $p_0$ and $p_1$, marked by red crosses, coincide with the endpoints of the element.
Two quadrature points, $q_0$ and $q_1$, marked by blue dots, with physical coordinates denoted by $\bm x(\bm X)$, are mapped to their corresponding radial projections on the circle, which have coordinates $\overset{\circ}{\bm{x}}(\bm x)$.
:::

We note that in equation {eq}`eq-coordinate-transforms-sphere`, the right-most Jacobian matrix ${\partial\bm{x}}/{\partial \bm{X}}_{(3\times2)}$ is provided by the library, while ${\partial \overset{\circ}{\bm{x}}}/{\partial \bm{x}}_{(3\times3)}$ is provided by the user with analytical derivatives.
In particular, for a sphere of radius 1, we have

$$
\overset{\circ}{\bm x}(\bm x) = \frac{1}{\lVert \bm x \rVert} \bm x_{(3\times 1)}
$$

and thus

$$
\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{x}} = \frac{1}{\lVert \bm x \rVert} \bm I_{(3\times 3)} - \frac{1}{\lVert \bm x \rVert^3} (\bm x \bm x^T)_{(3\times 3)} .
$$

(example-petsc-bps)=

## Bakeoff problems and generalizations

The PETSc examples in this directory include a full suite of parallel {ref}`bakeoff problems <bps>` (BPs) using a "raw" parallel decomposition (see `bpsraw.c`) and using PETSc's `DMPlex` for unstructured grid management (see `bps.c`).
A generalization of these BPs to the surface of the cubed-sphere are available in `bpssphere.c`.

(example-petsc-bps-sphere)=

### Bakeoff problems on the cubed-sphere

For the $L^2$ projection problems, BP1-BP2, that use the mass operator, the coordinate transformations and the corresponding Jacobian determinant, equation {eq}`eq-jacobian-sphere`, are the same as in the {ref}`example-petsc-area-sphere` example.
For the Poisson's problem, BP3-BP6, on the cubed-sphere, in addition to equation {eq}`eq-jacobian-sphere`, the pseudo-inverse of $\partial \overset{\circ}{\bm{x}} / \partial \bm{X}$ is used to derive the contravariant metric tensor (please see figure {numref}`fig-sphere-coords` for a reference of the notation used).
We begin by expressing the Moore-Penrose (left) pseudo-inverse:

$$
\frac{\partial \bm{X}}{\partial \overset{\circ}{\bm{x}}}_{(2\times 3)} \equiv \left(\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}\right)_{(2\times 3)}^{+} =  \left(\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}_{(2\times3)}^T \frac{\partial\overset{\circ}{\bm{x}}}{\partial \bm{X}}_{(3\times2)} \right)^{-1} \frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}_{(2\times3)}^T .
$$ (eq-dxcircdX-pseudo-inv)

This enables computation of gradients of an arbitrary function $u(\overset{\circ}{\bm x})$ in the embedding space as

$$
\frac{\partial u}{\partial \overset{\circ}{\bm x}}_{(1\times 3)} = \frac{\partial u}{\partial \bm X}_{(1\times 2)} \frac{\partial \bm X}{\partial \overset{\circ}{\bm x}}_{(2\times 3)}
$$

and thus the weak Laplacian may be expressed as

$$
\int_{\Omega} \frac{\partial v}{\partial \overset\circ{\bm x}} \left( \frac{\partial u}{\partial \overset\circ{\bm x}} \right)^T \, dS
    = \int_{\Omega} \frac{\partial v}{\partial \bm X} \underbrace{\frac{\partial \bm X}{\partial \overset\circ{\bm x}} \left( \frac{\partial \bm X}{\partial \overset\circ{\bm x}} \right)^T}_{\bm g_{(2\times 2)}}  \left(\frac{\partial u}{\partial \bm X} \right)^T \, dS
$$ (eq-weak-laplace-sphere)

where we have identified the $2\times 2$ contravariant metric tensor $\bm g$ (sometimes written $\bm g^{ij}$), and where now $\Omega$ represents the surface of the sphere, which is a two-dimensional closed surface embedded in the three-dimensional Euclidean space $\mathbb{R}^3$.
This expression can be simplified to avoid the explicit Moore-Penrose pseudo-inverse,

$$
\bm g = \left(\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}^T \frac{\partial\overset{\circ}{\bm{x}}}{\partial \bm{X}} \right)^{-1}_{(2\times 2)} \frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}_{(2\times3)}^T \frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}_{(3\times2)} \left(\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}^T \frac{\partial\overset{\circ}{\bm{x}}}{\partial \bm{X}} \right)^{-T}_{(2\times 2)} = \left(\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}^T \frac{\partial\overset{\circ}{\bm{x}}}{\partial \bm{X}} \right)^{-1}_{(2\times 2)}
$$

where we have dropped the transpose due to symmetry.
This allows us to simplify {eq}`eq-weak-laplace-sphere` as

$$
\int_{\Omega} \frac{\partial v}{\partial \overset\circ{\bm x}} \left( \frac{\partial u}{\partial \overset\circ{\bm x}} \right)^T \, dS     = \int_{\Omega} \frac{\partial v}{\partial \bm X} \underbrace{\left(\frac{\partial \overset{\circ}{\bm{x}}}{\partial \bm{X}}^T \frac{\partial\overset{\circ}{\bm{x}}}{\partial \bm{X}} \right)^{-1}}_{\bm g_{(2\times 2)}}  \left(\frac{\partial u}{\partial \bm X} \right)^T \, dS ,
$$

which is the form implemented in `qfunctions/bps/bp3sphere.h`.

(example-petsc-multigrid)=

## Multigrid

This example is located in the subdirectory {file}`examples/petsc`.
It investigates $p$-multigrid for the Poisson problem, equation {eq}`eq-variable-coeff-poisson`, using an unstructured high-order finite element discretization.
All of the operators associated with the geometric multigrid are implemented in libCEED.

$$
-\nabla\cdot \left( \kappa \left( x \right) \nabla x \right) = g \left( x \right)
$$ (eq-variable-coeff-poisson)

The Poisson operator can be specified with the decomposition given by the equation in figure {ref}`fig-operator-decomp`, and the restriction and prolongation operators given by interpolation basis operations, $\bm{B}$, and $\bm{B}^T$, respectively, act on the different grid levels with corresponding element restrictions, $\bm{G}$.
These three operations can be exploited by existing matrix-free multigrid software and smoothers.
Preconditioning based on the libCEED finite element operator decomposition is an ongoing area of research.
