// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

#include "newtonian_state.h"
#include "turb_stats_types.h"
#include "utils.h"

CEED_QFUNCTION_HELPER int ChildStatsCollection(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  Turbulence_SpanStatsContext context = (Turbulence_SpanStatsContext)ctx;
  NewtonianIdealGasContext    gas     = &context->gas;
  CeedScalar                  delta_t = context->solution_time - context->previous_time;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ = q_data[0][i] * delta_t;

    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s      = StateFromQ(gas, qi, state_var);

    v[TURB_MEAN_DENSITY][i]                    = wdetJ * s.U.density;
    v[TURB_MEAN_PRESSURE][i]                   = wdetJ * s.Y.pressure;
    v[TURB_MEAN_PRESSURE_SQUARED][i]           = wdetJ * Square(s.Y.pressure);
    v[TURB_MEAN_PRESSURE_VELOCITY_X][i]        = wdetJ * s.Y.pressure * s.Y.velocity[0];
    v[TURB_MEAN_PRESSURE_VELOCITY_Y][i]        = wdetJ * s.Y.pressure * s.Y.velocity[1];
    v[TURB_MEAN_PRESSURE_VELOCITY_Z][i]        = wdetJ * s.Y.pressure * s.Y.velocity[2];
    v[TURB_MEAN_DENSITY_TEMPERATURE][i]        = wdetJ * s.U.density * s.Y.temperature;
    v[TURB_MEAN_DENSITY_TEMPERATURE_FLUX_X][i] = wdetJ * s.U.density * s.Y.temperature * s.Y.velocity[0];
    v[TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Y][i] = wdetJ * s.U.density * s.Y.temperature * s.Y.velocity[1];
    v[TURB_MEAN_DENSITY_TEMPERATURE_FLUX_Z][i] = wdetJ * s.U.density * s.Y.temperature * s.Y.velocity[2];
    v[TURB_MEAN_MOMENTUM_X][i]                 = wdetJ * s.U.momentum[0];
    v[TURB_MEAN_MOMENTUM_Y][i]                 = wdetJ * s.U.momentum[1];
    v[TURB_MEAN_MOMENTUM_Z][i]                 = wdetJ * s.U.momentum[2];
    v[TURB_MEAN_MOMENTUMFLUX_XX][i]            = wdetJ * s.U.momentum[0] * s.Y.velocity[0];
    v[TURB_MEAN_MOMENTUMFLUX_YY][i]            = wdetJ * s.U.momentum[1] * s.Y.velocity[1];
    v[TURB_MEAN_MOMENTUMFLUX_ZZ][i]            = wdetJ * s.U.momentum[2] * s.Y.velocity[2];
    v[TURB_MEAN_MOMENTUMFLUX_YZ][i]            = wdetJ * s.U.momentum[1] * s.Y.velocity[2];
    v[TURB_MEAN_MOMENTUMFLUX_XZ][i]            = wdetJ * s.U.momentum[0] * s.Y.velocity[2];
    v[TURB_MEAN_MOMENTUMFLUX_XY][i]            = wdetJ * s.U.momentum[0] * s.Y.velocity[1];
    v[TURB_MEAN_VELOCITY_X][i]                 = wdetJ * s.Y.velocity[0];
    v[TURB_MEAN_VELOCITY_Y][i]                 = wdetJ * s.Y.velocity[1];
    v[TURB_MEAN_VELOCITY_Z][i]                 = wdetJ * s.Y.velocity[2];
  }
  return 0;
}

CEED_QFUNCTION(ChildStatsCollection_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(ChildStatsCollection_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ChildStatsCollection(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// QFunctions for testing
CEED_QFUNCTION_HELPER CeedScalar ChildStatsCollectionTest_Exact(const CeedScalar x_i[3]) { return x_i[0] + Square(x_i[1]); }

CEED_QFUNCTION(ChildStatsCollectionMMSTest)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*x)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar         t       = context->time;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ  = q_data[0][i];
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};

    // set spanwise domain to [0,1] and integrate from t \in [0,1] to recover exact solution
    v[0][i] = wdetJ * (ChildStatsCollectionTest_Exact(x_i) + t - 0.5) * 4 * Cube(x_i[2]);
    for (int j = 1; j < 22; j++) v[j][i] = 0;
  }
  return 0;
}

CEED_QFUNCTION(ChildStatsCollectionMMSTest_Error)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*x)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ  = q_data[0][i];
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};

    v[0][i] = wdetJ * Square(ChildStatsCollectionTest_Exact(x_i) - q[0][i]);
  }
  return 0;
}
