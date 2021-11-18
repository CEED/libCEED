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
/// Mass operator for Navier-Stokes example using PETSc

#ifndef mass_h
#define mass_h

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
CEED_QFUNCTION(Mass)(void *ctx, CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data) = in[1];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    v[0][i] = q_data[i] * u[0][i];
    v[1][i] = q_data[i] * u[1][i];
    v[2][i] = q_data[i] * u[2][i];
    v[3][i] = q_data[i] * u[3][i];
    v[4][i] = q_data[i] * u[4][i];
  }
  return 0;
}

// *****************************************************************************

#endif // mass_h
