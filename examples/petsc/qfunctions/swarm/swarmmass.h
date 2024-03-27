// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef SWARM_MASS_H
#define SWARM_MASS_H

#include <ceed.h>

CEED_QFUNCTION(SetupMass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];
  const CeedScalar *w                 = in[1];
  CeedScalar       *q_data            = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar B11 = J[1][1][i] * J[2][2][i] - J[1][2][i] * J[2][1][i];
    const CeedScalar B12 = J[0][2][i] * J[2][1][i] - J[0][1][i] * J[2][2][i];
    const CeedScalar B13 = J[0][1][i] * J[1][2][i] - J[0][2][i] * J[1][1][i];

    q_data[i] = w[i] * (J[0][0][i] * B11 + J[1][0][i] * B12 + J[2][0][i] * B13);
  }
  return 0;
}

CEED_QFUNCTION(Mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *rho            = (const CeedScalar *)in[0];
  const CeedScalar(*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const CeedInt num_comp = *(CeedInt *)ctx;
  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt j = 0; j < num_comp; j++) v[j][i] = rho[i] * u[j][i];
  }
  return 0;
}

#endif  // SWARM_MASS_H
