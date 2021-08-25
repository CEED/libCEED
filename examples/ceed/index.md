# Standalone libCEED

The following two examples have no dependencies, and are designed to be self-contained.
For additional examples that use external discretization libraries (MFEM, PETSc, Nek5000
etc.) see the subdirectories in {file}`examples/`.

(ex1-volume)=

## Ex1-Volume

This example is located in the subdirectory {file}`examples/ceed`. It illustrates a
simple usage of libCEED to compute the volume of a given body using a matrix-free
application of the mass operator. Arbitrary mesh and solution orders in 1D, 2D, and 3D
are supported from the same code.

This example shows how to compute line/surface/volume integrals of a 1D, 2D, or 3D
domain $\Omega$ respectively, by applying the mass operator to a vector of
$1$s. It computes:

$$
I = \int_{\Omega} 1 \, dV .
$$ (eq-ex1-volume)

Using the same notation as in {ref}`theoretical-framework`, we write here the vector
$u(x)\equiv 1$ in the Galerkin approximation,
and find the volume of $\Omega$ as

$$
\sum_e \int_{\Omega_e} v(x) 1 \, dV
$$ (volume-sum)

with $v(x) \in \mathcal{V}_p = \{ v \in H^{1}(\Omega_e) \,|\, v \in P_p(\bm{I}), e=1,\ldots,N_e \}$,
the test functions.

(ex2-surface)=

## Ex2-Surface

This example is located in the subdirectory {file}`examples/ceed`. It computes the
surface area of a given body using matrix-free application of a diffusion operator.
Similar to {ref}`Ex1-Volume`, arbitrary mesh and solution orders in 1D, 2D, and 3D
are supported from the same code. It computes:

$$
I = \int_{\partial \Omega} 1 \, dS ,
$$ (eq-ex2-surface)

by applying the divergence theorem.
In particular, we select $u(\bm x) = x_0 + x_1 + x_2$, for which $\nabla u = [1, 1, 1]^T$, and thus $\nabla u \cdot \hat{\bm n} = 1$.

Given Laplace's equation,

$$
\nabla \cdot \nabla u = 0, \textrm{ for  } \bm{x} \in \Omega ,
$$

let us multiply by a test function $v$ and integrate by parts to obtain

$$
\int_\Omega \nabla v \cdot \nabla u \, dV - \int_{\partial \Omega} v \nabla u \cdot \hat{\bm n}\, dS = 0 .
$$

Since we have chosen $u$ such that $\nabla u \cdot \hat{\bm n} = 1$, the boundary integrand is $v 1 \equiv v$. Hence, similar to {eq}`volume-sum`, we can evaluate the surface integral by applying the volumetric Laplacian as follows

$$
\int_\Omega \nabla v \cdot \nabla u \, dV \approx \sum_e \int_{\partial \Omega_e} v(x) 1 \, dS .
$$
