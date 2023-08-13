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

enum DifferentialFilterStateComponent {
  DIFF_FILTER_PRESSURE,
  DIFF_FILTER_VELOCITY_X,
  DIFF_FILTER_VELOCITY_Y,
  DIFF_FILTER_VELOCITY_Z,
  DIFF_FILTER_TEMPERATURE,
  DIFF_FILTER_STATE_NUM,
};

enum DifferentialFilterVelocitySquared {
  DIFF_FILTER_VELOCITY_SQUARED_XX,
  DIFF_FILTER_VELOCITY_SQUARED_YY,
  DIFF_FILTER_VELOCITY_SQUARED_ZZ,
  DIFF_FILTER_VELOCITY_SQUARED_YZ,
  DIFF_FILTER_VELOCITY_SQUARED_XZ,
  DIFF_FILTER_VELOCITY_SQUARED_XY,
  DIFF_FILTER_VELOCITY_SQUARED_NUM,
};

enum DifferentialFilterDampingFunction { DIFF_FILTER_DAMP_NONE, DIFF_FILTER_DAMP_VAN_DRIEST, DIFF_FILTER_DAMP_MMS };

typedef struct DifferentialFilterContext_ *DifferentialFilterContext;
struct DifferentialFilterContext_ {
  bool                                   grid_based_width;
  CeedScalar                             width_scaling[3];
  CeedScalar                             kernel_scaling;
  CeedScalar                             friction_length;
  enum DifferentialFilterDampingFunction damping_function;
  CeedScalar                             damping_constant;
  struct NewtonianIdealGasContext_       gas;
};

CEED_QFUNCTION_HELPER int DifferentialFilter_RHS(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*x)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*v0)[CEED_Q_VLA]           = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*v1)[CEED_Q_VLA]           = (CeedScalar(*)[CEED_Q_VLA])out[1];

  DifferentialFilterContext context = (DifferentialFilterContext)ctx;
  NewtonianIdealGasContext  gas     = &context->gas;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar wdetJ  = q_data[0][i];
    const State      s      = StateFromQ(gas, qi, x_i, state_var);

    v0[DIFF_FILTER_PRESSURE][i]            = wdetJ * s.Y.pressure;
    v0[DIFF_FILTER_VELOCITY_X][i]          = wdetJ * s.Y.velocity[0];
    v0[DIFF_FILTER_VELOCITY_Y][i]          = wdetJ * s.Y.velocity[1];
    v0[DIFF_FILTER_VELOCITY_Z][i]          = wdetJ * s.Y.velocity[2];
    v0[DIFF_FILTER_TEMPERATURE][i]         = wdetJ * s.Y.temperature;
    v1[DIFF_FILTER_VELOCITY_SQUARED_XX][i] = wdetJ * s.Y.velocity[0] * s.Y.velocity[0];
    v1[DIFF_FILTER_VELOCITY_SQUARED_YY][i] = wdetJ * s.Y.velocity[1] * s.Y.velocity[1];
    v1[DIFF_FILTER_VELOCITY_SQUARED_ZZ][i] = wdetJ * s.Y.velocity[2] * s.Y.velocity[2];
    v1[DIFF_FILTER_VELOCITY_SQUARED_YZ][i] = wdetJ * s.Y.velocity[1] * s.Y.velocity[2];
    v1[DIFF_FILTER_VELOCITY_SQUARED_XZ][i] = wdetJ * s.Y.velocity[0] * s.Y.velocity[2];
    v1[DIFF_FILTER_VELOCITY_SQUARED_XY][i] = wdetJ * s.Y.velocity[0] * s.Y.velocity[1];
  }
  return 0;
}

CEED_QFUNCTION(DifferentialFilter_RHS_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_RHS(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(DifferentialFilter_RHS_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_RHS(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(DifferentialFilter_RHS_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_RHS(ctx, Q, in, out, STATEVAR_ENTROPY);
}

CEED_QFUNCTION_HELPER CeedScalar VanDriestWallDamping(const CeedScalar wall_dist_plus, const CeedScalar A_plus) {
  return -expm1(-wall_dist_plus / A_plus);
}

CEED_QFUNCTION_HELPER int DifferentialFilter_LHS_N(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, const CeedInt N) {
  const CeedScalar(*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*A_ij_delta)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*x)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  const CeedScalar(*q_data)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*v)[CEED_Q_VLA]                = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*Grad_v)[CEED_Q_VLA]           = (CeedScalar(*)[CEED_Q_VLA])out[1];

  DifferentialFilterContext context = (DifferentialFilterContext)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedPragmaSIMD for (CeedInt j = 0; j < N; j++) {
      const CeedScalar x_i[3]     = {x[0][i], x[1][i], x[2][i]};
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

      CeedScalar scaling_matrix[3][3] = {{0}};
      if (context->damping_function == DIFF_FILTER_DAMP_VAN_DRIEST) {
        const CeedScalar damping_coeff = VanDriestWallDamping(x_i[1] / context->friction_length, context->damping_constant);
        scaling_matrix[0][0]           = Max(1, damping_coeff * context->width_scaling[0]);
        scaling_matrix[1][1]           = damping_coeff * context->width_scaling[1];
        scaling_matrix[2][2]           = Max(1, damping_coeff * context->width_scaling[2]);
      } else if (context->damping_function == DIFF_FILTER_DAMP_NONE) {
        scaling_matrix[0][0] = context->width_scaling[0];
        scaling_matrix[1][1] = context->width_scaling[1];
        scaling_matrix[2][2] = context->width_scaling[2];
      } else if (context->damping_function == DIFF_FILTER_DAMP_MMS) {
        const CeedScalar damping_coeff = tanh(60 * x_i[1]);
        scaling_matrix[0][0]           = 1;
        scaling_matrix[1][1]           = damping_coeff;
        scaling_matrix[2][2]           = 1;
      }

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

CEED_QFUNCTION(DifferentialFilter_LHS_6)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_LHS_N(ctx, Q, in, out, 6);
}

CEED_QFUNCTION(DifferentialFilter_LHS_11)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return DifferentialFilter_LHS_N(ctx, Q, in, out, 11);
}

CEED_QFUNCTION_HELPER CeedScalar MMS_Solution(const CeedScalar x_i[3], const CeedScalar omega) {
  return sin(6 * omega * x_i[0]) + sin(6 * omega * x_i[1]);
}

CEED_QFUNCTION(DifferentialFilter_MMS_RHS)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wdetJ = q_data[0][i];
    v[0][i]                = wdetJ * q[0][i];
  }
  return 0;
}

// @brief Generate initial condition such that the solution of the differential filtering is given by MMS_Solution() above
//
// This requires a *very* specific grid, as the anisotropic filtering is grid dependent.
// It's for a 75x75x1 grid on a [0,0.5]x3 domain.
// The grid is evenly distributed in x, but distributed based on the analytical mesh distribution \Delta_y = .001 + .01*tanh(6*y).
// The MMS test can optionally include a wall damping function (must also be enabled for the differential filtering itself).
// It can be run via:
// ./navierstokes -options_file tests-output/blasius_test.yaml -diff_filter_enable -diff_filter_view cgns:filtered_solution.cgns -ts_max_steps 0
// -diff_filter_mms -diff_filter_kernel_scaling 1 -diff_filter_wall_damping_function mms -dm_plex_box_upper 0.5,0.5,0.5 -dm_plex_box_faces 75,75,1
// -platemesh_y_node_locs_path tests-output/diff_filter_mms_y_spacing.dat -platemesh_top_angle 0 -diff_filter_grid_based_width
CEED_QFUNCTION(DifferentialFilter_MMS_IC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  CeedScalar(*q0)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};

    const CeedScalar aniso_scale_factor = 1;  // Must match the one passed in by -diff_filter_aniso_scale
    const CeedScalar omega              = 2 * M_PI;
    const CeedScalar omega6             = 6 * omega;
    const CeedScalar phi_bar            = MMS_Solution(x_i, omega);
    const CeedScalar dx                 = 0.5 / 75;
    const CeedScalar dy_analytic        = .001 + .01 * tanh(6 * x_i[1]);
    const CeedScalar dy                 = dy_analytic;
    const CeedScalar d_dy_dy            = 0.06 * Square(1 / cosh(6 * x_i[1]));  // Change of \Delta_y w.r.t. y
    CeedScalar       alpha[2]           = {Square(dx) * aniso_scale_factor, Square(dy) * aniso_scale_factor};
    bool             damping            = true;
    CeedScalar       dalpha1dy;
    if (damping) {
      CeedScalar damping_coeff   = tanh(60 * x_i[1]);
      CeedScalar d_damping_coeff = 60 / Square(cosh(60 * x_i[1]));
      dalpha1dy                  = aniso_scale_factor * 2 * (damping_coeff * dy) * (dy * d_damping_coeff + damping_coeff * d_dy_dy);
      alpha[1] *= Square(damping_coeff);
    } else {
      dalpha1dy = aniso_scale_factor * 2 * dy * d_dy_dy;
    }

    CeedScalar phi = phi_bar + alpha[0] * Square(omega6) * sin(6 * omega * x_i[0]) + alpha[1] * Square(omega6) * sin(omega6 * x_i[1]);
    phi -= dalpha1dy * omega6 * cos(omega6 * x_i[1]);

    for (CeedInt j = 0; j < 5; j++) q0[j][i] = phi;
  }
  return 0;
}
