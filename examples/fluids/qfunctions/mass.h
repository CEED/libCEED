// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Mass operator for Navier-Stokes example using PETSc

#ifndef mass_h
#define mass_h

#include <ceed.h>
#include <math.h>

// *****************************************************************************
// This QFunction applies the mass matrix to five interlaced fields.
//
// Inputs:
//   u     - Input vector at quadrature points
//   q_data - Quadrature weights
//
// Output:
//   v - Output vector at quadrature points
//
// *****************************************************************************
CEED_QFUNCTION(Mass)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*q_data) = in[1];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    v[0][i] = q_data[i] * u[0][i];
    v[1][i] = q_data[i] * u[1][i];
    v[2][i] = q_data[i] * u[2][i];
    v[3][i] = q_data[i] * u[3][i];
    v[4][i] = q_data[i] * u[4][i];
  }
  return 0;
}

// *****************************************************************************

#endif  // mass_h
