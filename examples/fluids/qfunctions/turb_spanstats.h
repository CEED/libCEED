// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

#include "newtonian_state.h"
#include "utils.h"

CEED_QFUNCTION_HELPER int ChildStatsCollection(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateFromQi_t StateFromQi,
                                               StateFromQi_fwd_t StateFromQi_fwd) {
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*x)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ = q_data[0][i];

    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const State      s      = StateFromQi(context, qi, x_i);

    v[0][i]  = wdetJ * s.U.density;
    v[1][i]  = wdetJ * s.Y.pressure;
    v[2][i]  = wdetJ * Square(s.Y.pressure);
    v[3][i]  = wdetJ * s.Y.pressure * s.Y.velocity[0];
    v[4][i]  = wdetJ * s.Y.pressure * s.Y.velocity[1];
    v[5][i]  = wdetJ * s.Y.pressure * s.Y.velocity[2];
    v[6][i]  = wdetJ * s.U.density * s.Y.temperature;
    v[7][i]  = wdetJ * s.U.density * s.Y.temperature * s.Y.velocity[0];
    v[8][i]  = wdetJ * s.U.density * s.Y.temperature * s.Y.velocity[1];
    v[9][i]  = wdetJ * s.U.density * s.Y.temperature * s.Y.velocity[2];
    v[10][i] = wdetJ * s.U.momentum[0];
    v[11][i] = wdetJ * s.U.momentum[1];
    v[12][i] = wdetJ * s.U.momentum[2];
    v[13][i] = wdetJ * s.U.momentum[0] * s.Y.velocity[0];
    v[14][i] = wdetJ * s.U.momentum[1] * s.Y.velocity[1];
    v[15][i] = wdetJ * s.U.momentum[2] * s.Y.velocity[2];
    v[16][i] = wdetJ * s.U.momentum[1] * s.Y.velocity[2];
    v[17][i] = wdetJ * s.U.momentum[0] * s.Y.velocity[2];
    v[18][i] = wdetJ * s.U.momentum[0] * s.Y.velocity[1];
    v[19][i] = wdetJ * s.Y.velocity[0];
    v[20][i] = wdetJ * s.Y.velocity[1];
    v[21][i] = wdetJ * s.Y.velocity[2];
  }
  return 0;
}

CEED_QFUNCTION(ChildStatsCollection_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(ChildStatsCollection_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

CEED_QFUNCTION(ChildStatsCollectionTest)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*x)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar         t       = context->time;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ  = q_data[0][i];
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};

    v[0][i] = wdetJ * ((x_i[0] + Square(x_i[1]) + t - 0.5) * 4 * Cube(x_i[2]));
  }
  return 0;
}
