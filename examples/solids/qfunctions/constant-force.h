// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Constant forcing term for solid mechanics example using PETSc

#ifndef CONSTANT_H
#define CONSTANT_H

#include <ceed.h>
#include <math.h>

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  CeedScalar nu;  // Poisson's ratio
  CeedScalar E;   // Young's Modulus
};
#endif

// -----------------------------------------------------------------------------
// Constant forcing term along specified vector
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupConstantForce)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q_data = in[1];

  // Outputs
  CeedScalar *force = out[0];

  // Context
  const CeedScalar *forcing_vector = (CeedScalar(*))ctx;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    CeedScalar wdetJ = q_data[i];

    // Forcing function
    // -- Component 1
    force[i + 0 * Q] = forcing_vector[0] * wdetJ;

    // -- Component 2
    force[i + 1 * Q] = forcing_vector[1] * wdetJ;

    // -- Component 3
    force[i + 2 * Q] = forcing_vector[2] * wdetJ;
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // End of CONSTANT_H
