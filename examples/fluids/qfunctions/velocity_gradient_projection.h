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
                                                        StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1];
  const CeedScalar(*q_data)                = in[2];
  CeedScalar(*v)[CEED_Q_VLA]               = (CeedScalar(*)[CEED_Q_VLA])out[0];

  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    CeedScalar       wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);

    const State s = StateFromQ(context, qi, state_var);
    State       grad_s[3];
    for (CeedInt k = 0; k < 3; k++) {
      CeedScalar dqi[5];
      for (CeedInt j = 0; j < 5; j++) {
        dqi[j] = Grad_q[0][j][i] * dXdx[0][k] + Grad_q[1][j][i] * dXdx[1][k] + Grad_q[2][j][i] * dXdx[2][k];
      }
      grad_s[k] = StateFromQ_fwd(context, s, dqi, state_var);
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
  return VelocityGradientProjectionRHS(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(VelocityGradientProjectionRHS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return VelocityGradientProjectionRHS(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}
#endif  // velocity_gradient_projection_h
