# libCEED: Examples

This page provides a brief description of the examples for the libCEED library.

## Example 1

This example uses the mass matrix to compute the length, area, or volume of a
region, depending upon runtime parameters.

## Bakeoff Problems

This section provides a brief description of the bakeoff problems, used as examples
for the libCEED library. These bakeoff problems are high-order benchmarks designed
to test and compare the performance of high-order finite element codes.

For further documentation, readers may wish to consult the
[CEED documentation](http://ceed.exascaleproject.org/bps/) of the bakeoff problems.

### Bakeoff Problem 1

Bakeoff problem 1 is the *L<sup>2</sup>* projection problem into the finite element space.

The supplied examples solve *_B_ u = f*, where *_B_* is the mass matrix.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre. There is one more quadrature point in each dimension than nodal point, *q = p + 1*.

### Bakeoff Problem 2

Bakeoff problem 2 is the *L<sup>2</sup>* projection problem into the finite element space on a vector system.

The supplied examples solve *_B_ _u_ = f*, where *_B_* is the mass matrix.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre. There is one more quadrature point in each dimension than nodal point, *q = p + 1*.

### Bakeoff Problem 3

Bakeoff problem 3 is the Poisson problem.

The supplied examples solve *_A_ u = f*, where *_A_* is the Poisson operator.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre. There is one more quadrature point in each dimension than nodal point, *q = p + 1*.

### Bakeoff Problem 4

Bakeoff problem 4 is the Poisson problem on a vector system.

The supplied examples solve *_A_ _u_ = f*, where *_A_* is the Laplace operator for the Poisson equation.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre. There is one more quadrature point in each dimension than nodal point, *q = p + 1*.

### Bakeoff Problem 5

Bakeoff problem 5 is the Poisson problem.

The supplied examples solve *_A_ u = f*, where *_A_* is the Poisson operator.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre-Lobatto. The nodal points and quadrature points are collocated.

### Bakeoff Problem 6

Bakeoff problem 6 is the Poisson problem on a vector system.

The supplied examples solve *_A_ _u_ = f*, where *_A_* is the Laplace operator for the Poisson equation.

The nodal points, *p*, are Gauss-Legendre-Lobatto, and the quadrature points, *q* are Gauss-Legendre-Lobatto. The nodal points and quadrature points are collocated.

## Navier-Stokes Solver

The Navier-Stokes problem solves the compressible Navier-Stokes equations using an explicit time integration. A more detailed description of the problem formulation
can be found in the `navier-stokes` folder.

## Running Examples

To build the examples, set the `MFEM_DIR`, `PETSC_DIR` and `NEK5K_DIR` variables
and run:

```console
# libCEED examples on CPU and GPU
cd ceed
make
./ex1 -ceed /cpu/self
./ex1 -ceed /gpu/occa
cd ..

# MFEM+libCEED examples on CPU and GPU
cd mfem
make
./bp1 -ceed /cpu/self -no-vis
./bp3 -ceed /gpu/occa -no-vis
cd ..

# Nek5000+libCEED examples on CPU and GPU
cd nek
make
./nek-examples.sh -e bp1 -ceed /cpu/self -b 3
./nek-examples.sh -e bp3 -ceed /gpu/occa -b 3
cd ..

# PETSc+libCEED examples on CPU and GPU
cd petsc
make
./bps -problem bp1 -ceed /cpu/self
./bps -problem bp2 -ceed /gpu/occa
./bps -problem bp3 -ceed /cpu/self
./bps -problem bp4 -ceed /gpu/occa
./bps -problem bp5 -ceed /cpu/self
./bps -problem bp6 -ceed /gpu/occa
cd ..

cd navier-stokes
make
./navierstokes -ceed /cpu/self
./navierstokes -ceed /gpu/occa
cd ..
```

The above code assumes a GPU-capable machine with the OCCA backend 
enabled. Depending on the available backends, other CEED resource specifiers can
be provided with the `-ceed` option.
