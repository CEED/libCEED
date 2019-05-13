# libCEED: Navier-Stokes Example

This page provides a description of the shallow-water example for the libCEED library, based on PETSc.

This solver computes the solution to the shallow-water equations on a cubed-sphere (i.e.,
a spherical surface tessellated by quadrilaterals, obtained by projecting the sides
of a circumscribed cube onto a spherical surface).

The main shallow-water solver for libCEED is defined in [`shallow-water.c`](shallow-water.c).

Build by using

`make`

and run with

`./shallow-water`

