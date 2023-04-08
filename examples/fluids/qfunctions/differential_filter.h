// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
//
/// @file
/// Implementation of differential filtering

#include <ceed.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

enum DifferentialFilterComponent {
  DIFF_FILTER_PRESSURE,
  DIFF_FILTER_VELOCITY_X,
  DIFF_FILTER_VELOCITY_Y,
  DIFF_FILTER_VELOCITY_Z,
  DIFF_FILTER_TEMPERATURE,
  DIFF_FILTER_VELOCITY_SQUARED_XX,
  DIFF_FILTER_VELOCITY_SQUARED_YY,
  DIFF_FILTER_VELOCITY_SQUARED_ZZ,
  DIFF_FILTER_VELOCITY_SQUARED_YZ,
  DIFF_FILTER_VELOCITY_SQUARED_XZ,
  DIFF_FILTER_VELOCITY_SQUARED_XY,
  DIFF_FILTER_NUM_COMPONENTS,
};

typedef struct DifferentialFilterContext_ *DifferentialFilterContext;
struct DifferentialFilterContext_ {
  bool                             grid_based_width;
  CeedScalar                       width_scaling[3];
  CeedScalar                       kernel_scaling;
  struct NewtonianIdealGasContext_ gas;
};

CEED_QFUNCTION_HELPER int DifferentialFilter_RHS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateFromQi_t StateFromQi,
                                                 StateFromQi_fwd_t StateFromQi_fwd) {
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*x)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  DifferentialFilterContext context = (DifferentialFilterContext)ctx;
  NewtonianIdealGasContext  gas     = &context->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar wdetJ  = q_data[0][i];
    const State      s      = StateFromQi(gas, qi, x_i);

    v[DIFF_FILTER_PRESSURE][i]            = wdetJ * s.Y.pressure;
    v[DIFF_FILTER_VELOCITY_X][i]          = wdetJ * s.Y.velocity[0];
    v[DIFF_FILTER_VELOCITY_Y][i]          = wdetJ * s.Y.velocity[1];
    v[DIFF_FILTER_VELOCITY_Z][i]          = wdetJ * s.Y.velocity[2];
    v[DIFF_FILTER_TEMPERATURE][i]         = wdetJ * s.Y.temperature;
    v[DIFF_FILTER_VELOCITY_SQUARED_XX][i] = wdetJ * s.Y.velocity[0] * s.Y.velocity[0];
    v[DIFF_FILTER_VELOCITY_SQUARED_YY][i] = wdetJ * s.Y.velocity[1] * s.Y.velocity[1];
    v[DIFF_FILTER_VELOCITY_SQUARED_ZZ][i] = wdetJ * s.Y.velocity[2] * s.Y.velocity[2];
    v[DIFF_FILTER_VELOCITY_SQUARED_YZ][i] = wdetJ * s.Y.velocity[1] * s.Y.velocity[2];
    v[DIFF_FILTER_VELOCITY_SQUARED_XZ][i] = wdetJ * s.Y.velocity[0] * s.Y.velocity[2];
    v[DIFF_FILTER_VELOCITY_SQUARED_XY][i] = wdetJ * s.Y.velocity[0] * s.Y.velocity[1];
  }
  return 0;
}

CEED_QFUNCTION(DifferentialFilter_RHS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_RHS(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(DifferentialFilter_RHS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_RHS(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

CEED_QFUNCTION_HELPER int DifferentialFilter_LHS_N(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, const CeedInt N) {
  const CeedScalar(*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*A_ij_delta)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*q_data)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  CeedScalar(*v)[CEED_Q_VLA]                = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*Grad_v)[CEED_Q_VLA]           = (CeedScalar(*)[CEED_Q_VLA])out[1];

  DifferentialFilterContext context = (DifferentialFilterContext)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedPragmaSIMD for (CeedInt j = 0; j < N; j++) {
      const CeedScalar wdetJ      = q_data[0][i];
      const CeedScalar dXdx[3][3] = {
          {q_data[1][i], q_data[2][i], q_data[3][i]},
          {q_data[4][i], q_data[5][i], q_data[6][i]},
          {q_data[7][i], q_data[8][i], q_data[9][i]}
      };

      CeedScalar Delta_ij[3][3] = {{0.}};
      if (context->grid_based_width) {
        CeedScalar       km_A_ij[6] = {A_ij_delta[0][i], A_ij_delta[1][i], A_ij_delta[2][i], A_ij_delta[3][i], A_ij_delta[4][i], A_ij_delta[5][i]};
        const CeedScalar delta      = A_ij_delta[6][i];
        ScaleN(km_A_ij, delta, 6);  // Dimensionalize the normalized anisotropy tensor
        KMUnpack(km_A_ij, Delta_ij);
      } else {
        Delta_ij[0][0] = Delta_ij[1][1] = Delta_ij[2][2] = 1;
      }

      CeedScalar scaling_matrix[3][3] = {{0.}};
      scaling_matrix[0][0]            = context->width_scaling[0];
      scaling_matrix[1][1]            = context->width_scaling[1];
      scaling_matrix[2][2]            = context->width_scaling[2];

      CeedScalar scaled_Delta_ij[3][3] = {{0.}};
      MatMat3(scaling_matrix, Delta_ij, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, scaled_Delta_ij);
      CopyMat3(scaled_Delta_ij, Delta_ij);

      CeedScalar alpha_ij[3][3] = {{0.}};
      MatMat3(Delta_ij, Delta_ij, CEED_NOTRANSPOSE, CEED_NOTRANSPOSE, alpha_ij);
      ScaleN((CeedScalar *)alpha_ij, context->kernel_scaling, 9);

      v[j][i] = wdetJ * q[j][i];
      CeedScalar dq[3], dq_dXdx[3] = {0.}, dq_dXdx_a[3] = {0.};
      for (int k = 0; k < 3; k++) {
        dq[k] = Grad_q[0 * N + j][i] * dXdx[0][k] + Grad_q[1 * N + j][i] * dXdx[1][k] + Grad_q[2 * N + j][i] * dXdx[2][k];
      }
      MatVec3(dXdx, dq, CEED_NOTRANSPOSE, dq_dXdx);
      MatVec3(alpha_ij, dq_dXdx, CEED_NOTRANSPOSE, dq_dXdx_a);
      for (int k = 0; k < 3; k++) {
        Grad_v[k * N + j][i] = wdetJ * dq_dXdx_a[k];
      }
    }
  }
  return 0;
}

CEED_QFUNCTION(DifferentialFilter_LHS_1)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_LHS_N(ctx, Q, in, out, 1);
}

CEED_QFUNCTION(DifferentialFilter_LHS_5)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_LHS_N(ctx, Q, in, out, 5);
}

CEED_QFUNCTION(DifferentialFilter_LHS_11)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_LHS_N(ctx, Q, in, out, 11);
}
