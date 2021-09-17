# libCEED: Examples

This page provides a brief description of the examples for the libCEED
library.

## Basic libCEED Examples

Two examples that rely only upon libCEED without any external libraries are provided in the [ceed/](./ceed) folder. For more details, please see the dedicated [documentation section](https://libceed.readthedocs.io/en/latest/examples/ceed/index.html).

## Bakeoff Problems

% bps-inclusion-marker

The Center for Efficient Exascale Discretizations (CEED) uses Bakeoff Problems (BPs)
to test and compare the performance of high-order finite element implementations. The
definitions of the problems are given on the ceed
[website](https://ceed.exascaleproject.org/bps/). Each of the following bakeoff
problems that use external discretization libraries (such as MFEM, PETSc, and Nek5000)
are located in the subdirectories `mfem/`, `petsc/`, and
`nek5000/`, respectively.

Here we provide a short summary:

:::{list-table}
:header-rows: 1
:widths: auto
* - User code
  - Supported BPs
* - `mfem`
  - * BP1 (scalar mass operator) with $Q=P+1$
    * BP3 (scalar Laplace operator) with $Q=P+1$
* - `petsc`
  - * BP1 (scalar mass operator) with $Q=P+1$
    * BP2 (vector mass operator) with $Q=P+1$
    * BP3 (scalar Laplace operator) with $Q=P+1$
    * BP4 (vector Laplace operator) with $Q=P+1$
    * BP5 (collocated scalar Laplace operator) with $Q=P$
    * BP6 (collocated vector Laplace operator) with $Q=P$
* - `nek5000`
  - * BP1 (scalar mass operator) with $Q=P+1$
    * BP3 (scalar Laplace operator) with $Q=P+1$
:::

These are all **T-vector**-to-**T-vector** and include parallel scatter, element
scatter, element evaluation kernel, element gather, and parallel gather (with the
parallel gathers/scatters done externally to libCEED).

BP1 and BP2 are $L^2$ projections, and thus have no boundary condition.
The rest of the BPs have homogeneous Dirichlet boundary conditions.

The BPs are parametrized by the number $P$ of Gauss-Legendre-Lobatto nodal points
(with $P=p+1$, and $p$ the degree of the basis polynomial) for the Lagrange
polynomials, as well as the number of quadrature points, $Q$.
A $Q$-point Gauss-Legendre quadrature is used for all BPs except BP5 and BP6,
which choose $Q = P$ and Gauss-Legendre-Lobatto quadrature to collocate with the
interpolation nodes. This latter choice is popular in applications that use spectral
element methods because it produces a diagonal mass matrix (enabling easy explicit
time integration) and significantly reduces the number of floating point operations
to apply the operator.

% bps-exclusion-marker

For a more detailed description of the operators employed in the BPs, please see the dedicated [BPs documentation section](https://libceed.readthedocs.io/en/latest/examples/bps.html).

## PETSc+libCEED Navier-Stokes Solver

The Navier-Stokes problem solves the compressible Navier-Stokes
equations using an explicit or implicit time integration. A more detailed
description of the problem formulation can be found in the
[fluids/](./fluids) folder and the corresponding [fluids documentation page](https://libceed.readthedocs.io/en/latest/examples/fluids/index.html).

## PETSc+libCEED Solid mechanics elasticity mini-app

This example solves the steady-state static momentum balance equations using unstructured high-order finite/spectral element spatial discretizations. A more detailed
description of the problem formulation can be found in the
[solids/](./solids) folder and the corresponding [solids documentation page](https://libceed.readthedocs.io/en/latest/examples/solids/index.html).

## PETSc+libCEED Surface Area Examples

These examples, located in the [petsc/](./petsc) folder, use the mass operator to compute the surface area of a
cube or a discrete cubed-sphere, using PETSc. For a detailed description, please see the corresponding [area documentation page](https://libceed.readthedocs.io/en/latest/examples/petsc/index.html#area).

## PETSc+libCEED Bakeoff Problems on the Cubed-Sphere

These examples, located in the [petsc/](./petsc) folder, reproduce the Bakeoff Problems 1-6 on a discrete
cubed-sphere, using PETSc. For a detailed description, please see the corresponding [problems on the cubed-sphere documentation page](https://libceed.readthedocs.io/en/latest/examples/petsc/index.html#bakeoff-problems-on-the-cubed-sphere).

## Running Examples

To build the examples, set the `MFEM_DIR`, `PETSC_DIR`, and
`NEK5K_DIR` variables and, from the `examples/` directory, run

```{include} ../README.md
:start-after: running-examples-inclusion-marker
:end-before: benchmarks-marker
```
