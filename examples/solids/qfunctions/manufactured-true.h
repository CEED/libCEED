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
/// Linear elasticity manufactured solution true solution for solid mechanics example using PETSc

#ifndef MANUFACTURED_TRUE_H
#define MANUFACTURED_TRUE_H

#include <math.h>

// -----------------------------------------------------------------------------
// True solution for linear elasticity manufactured solution
// -----------------------------------------------------------------------------
CEED_QFUNCTION(MMSTrueSoln)(void *ctx, const CeedInt Q,
                            const CeedScalar *const *in,
                            CeedScalar *const *out) {
  // Inputs
  const CeedScalar *coords = in[0];

  // Outputs
  CeedScalar *true_soln = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q], z = coords[i+2*Q];

    // True solution
    // -- Component 1
    true_soln[i+0*Q] = exp(2*x)*sin(3*y)*cos(4*z)/1e8;

    // -- Component 2
    true_soln[i+1*Q] = exp(3*y)*sin(4*z)*cos(2*x)/1e8;

    // -- Component 3
    true_soln[i+2*Q] = exp(4*z)*sin(2*x)*cos(3*y)/1e8;

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End MANUFACTURED_TRUE_H
