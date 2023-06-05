// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef velocity_gradient_projection_h
#define velocity_gradient_projection_h

#include <ceed.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

CEED_QFUNCTION_HELPER int VelocityGradientProjectionRHS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                        StateFromQi_t StateFromQi, StateFromQi_fwd_t StateFromQi_fwd) {
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1];
  const CeedScalar(*q_data)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*x)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  CeedScalar(*v)[CEED_Q_VLA]               = (CeedScalar(*)[CEED_Q_VLA])out[0];

  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5]      = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar x_i[3]     = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar wdetJ      = q_data[0][i];
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };

    const State s = StateFromQi(context, qi, x_i);
    State       grad_s[3];
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dx_i[3] = {0}, dqi[5];
      for (CeedInt k = 0; k < 5; k++) {
        dqi[k] = Grad_q[0][k][i] * dXdx[0][j] + Grad_q[1][k][i] * dXdx[1][j] + Grad_q[2][k][i] * dXdx[2][j];
      }
      dx_i[j]   = 1.;
      grad_s[j] = StateFromQi_fwd(context, s, dqi, x_i, dx_i);
    }

    CeedScalar grad_velocity[3][3];
    VelocityGradient(grad_s, grad_velocity);

    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        v[j * 3 + k][i] = wdetJ * grad_velocity[j][k];
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(VelocityGradientProjectionRHS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return VelocityGradientProjectionRHS(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(VelocityGradientProjectionRHS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return VelocityGradientProjectionRHS(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}
#endif  // velocity_gradient_projection_h
