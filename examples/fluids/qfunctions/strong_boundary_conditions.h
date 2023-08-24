// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef strong_boundary_conditions_h
#define strong_boundary_conditions_h

#include <ceed.h>

#include "setupgeo_helpers.h"

CEED_QFUNCTION(SetupStrongBC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*coords)[CEED_Q_VLA]       = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*dxdX_q)[3][CEED_Q_VLA]    = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  const CeedScalar(*multiplicity)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar(*coords_stored)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*scale_stored)              = out[1];
  CeedScalar(*dXdx_q)[CEED_Q_VLA]        = (CeedScalar(*)[CEED_Q_VLA])out[2];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    for (int j = 0; j < 3; j++) coords_stored[j][i] = coords[j][i];
    scale_stored[i] = 1.0 / multiplicity[0][i];
    CeedScalar dXdx[2][3];
    InvertBoundaryMappingJacobian_3D(Q, i, dxdX_q, dXdx);
    dXdx_q[0][i] = dXdx[0][0];
    dXdx_q[1][i] = dXdx[0][1];
    dXdx_q[2][i] = dXdx[0][2];
    dXdx_q[3][i] = dXdx[1][0];
    dXdx_q[4][i] = dXdx[1][1];
    dXdx_q[5][i] = dXdx[1][2];
  }
  return 0;
}

#endif  // strong_boundary_conditions_h
