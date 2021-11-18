// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// libCEED QFunctions for mass operator example using PETSc

#ifndef bp2_h
#define bp2_h

#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassRhs3)(void *ctx, const CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  const CeedScalar *x = in[0], *w = in[1];
  CeedScalar *true_soln = out[0], *rhs = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Component 1
    true_soln[i+0*Q] =  sqrt(x[i]*x[i] + x[i+Q]*x[i+Q] + x[i+2*Q]*x[i+2*Q]);
    // Component 2
    true_soln[i+1*Q] = 2 * true_soln[i+0*Q];
    // Component 3
    true_soln[i+2*Q] = 3 * true_soln[i+0*Q];

    // Component 1
    rhs[i+0*Q] = w[i] * true_soln[i+0*Q];
    // Component 2
    rhs[i+1*Q] = 2 * rhs[i+0*Q];
    // Component 3
    rhs[i+2*Q] = 3 * rhs[i+0*Q];
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the mass operator for a vector field of 3 components.
//
// Inputs:
//   u     - Input vector at quadrature points
//   q_data - Geometric factors
//
// Output:
//   v     - Output vector (test functions) at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(Mass3)(void *ctx, const CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *q_data = in[1];
  CeedScalar *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Component 1
    v[i+0*Q] = q_data[i] * u[i+0*Q];
    // Component 2
    v[i+1*Q] = q_data[i] * u[i+1*Q];
    // Component 3
    v[i+2*Q] = q_data[i] * u[i+2*Q];
  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif // bp2_h
