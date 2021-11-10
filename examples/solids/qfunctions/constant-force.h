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
/// Constant forcing term for solid mechanics example using PETSc

#ifndef CONSTANT_H
#define CONSTANT_H

#include <math.h>

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  CeedScalar   nu;      // Poisson's ratio
  CeedScalar   E;       // Young's Modulus
};
#endif

// -----------------------------------------------------------------------------
// Constant forcing term along specified vector
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupConstantForce)(void *ctx, const CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q_data = in[1];

  // Outputs
  CeedScalar *force = out[0];

  // Context
  const CeedScalar *forcing_vector = (CeedScalar(*))ctx;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    CeedScalar wdetJ = q_data[i];

    // Forcing function
    // -- Component 1
    force[i+0*Q] = forcing_vector[0]*wdetJ;

    // -- Component 2
    force[i+1*Q] = forcing_vector[1]*wdetJ;

    // -- Component 3
    force[i+2*Q] = forcing_vector[2]*wdetJ;

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End of CONSTANT_H
