# libCEED: Examples

libCEED comes with several examples of its usage, ranging from standalone C
codes to examples based on external packages,such as MFEM, PETSc and Nek5000.

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

### Bakeoff Problem 3

Bakeoff problem 1 is the Poisson problem.

The supplied examples solve *_A_ u = f*, where *_A_* is the Poisson operator.

### Navier-Stokes Solver

The Navier-Stokes problem solves the compressible Navier-Stokes equations using an explicit time integration.

## Running Examples

To build the examples, set the `MFEM_DIR`, `PETSC_DIR` and `NEK5K_DIR` variables
and run:

```console
# libCEED examples on CPU and GPU
cd ceed
make
./ex1 -ceed /cpu/self
./ex1 -ceed /gpu/occa
cd ../..

# MFEM+libCEED examples on CPU and GPU
cd mfem
make
./bp1 -ceed /cpu/self -no-vis
./bp1 -ceed /gpu/occa -no-vis
cd ../..

# PETSc+libCEED examples on CPU and GPU
cd petsc
make
./bp1 -ceed /cpu/self
./bp1 -ceed /gpu/occa
cd ../..

cd navier-stokes
make
./navierstokes -ceed /cpu/self
./navierstokes -ceed /gpu/occa
cd ../..

# Nek+libCEED examples on CPU and GPU
cd nek5000
./make-nek-examples.sh
./run-nek-example.sh -ceed /cpu/self -b 3
./run-nek-example.sh -ceed /gpu/occa -b 3
cd ../..
```

The above code assumes a GPU-capable machine with the OCCA backend 
enabled. Depending on the available backends, other Ceed resource specifiers can
be provided with the `-ceed` option.
