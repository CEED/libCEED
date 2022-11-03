// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc

#include "newtonian_types.h"
#include "newtonian_state.h"

typedef struct FreestreamContext_ *FreestreamContext;
struct FreestreamContext_ {
  struct NewtonianIdealGasContext_ newtonian_ctx;
  State S_infty;
};

typedef struct {
  CeedScalar left, right;
} RoeWeights;

CEED_QFUNCTION_HELPER RoeWeights RoeSetup(CeedScalar rho_left,
    CeedScalar rho_right) {
  CeedScalar sqrt_left = sqrt(rho_left), sqrt_right = sqrt(rho_right);
  return (RoeWeights) {sqrt_left / (sqrt_left + sqrt_right), sqrt_right / (sqrt_left + sqrt_right)};
}

CEED_QFUNCTION_HELPER CeedScalar RoeAverage(RoeWeights r, CeedScalar q_left,
    CeedScalar q_right) {
  return r.left * q_left + r.right * q_right;
}

// *****************************************************************************
// @brief Harten Lax VanLeer (HLL) approximate Riemann solver.
// Taking in two states (left, right) and returns RiemannFlux_HLL
// The left and right states are specified from the perspective of an
// outward-facing normal vector pointing left to right.
//
// @param gas    NewtonianIdealGasContext for the fluid
// @param left   Fluid state of the domain interior (the current solution)
// @param right  Fluid state of the domain exterior (free stream conditions)
// @param normal Normalized, outward facing boundary normal vector
// *****************************************************************************
CEED_QFUNCTION_HELPER StateConservative Harten_Lax_VanLeer_Flux(
  NewtonianIdealGasContext gas, State left, State right,
  const CeedScalar normal[3]) {

  const CeedScalar gamma = gas->cp / gas->cv;

  StateConservative flux_left  = FluxInviscidDotNormal(gas, left, normal);
  StateConservative flux_right = FluxInviscidDotNormal(gas, right, normal);
  StateConservative RiemannFlux_HLL;

  CeedScalar u_left = Dot3(left.Y.velocity, normal);
  CeedScalar u_right = Dot3(right.Y.velocity, normal);

  RoeWeights r = RoeSetup(left.U.density, right.U.density);
  // Speed estimate
  // Roe average eigenvalues for left and right non-linear waves
  // Stability requires that these speed estimates are *at least*
  // as fast as the physical wave speeds.
  CeedScalar u_roe = RoeAverage(r, u_left, u_right);

  CeedScalar H_left = (left.U.E_total + left.Y.pressure) / left.U.density;
  CeedScalar H_right = (right.U.E_total + right.Y.pressure) / right.U.density;
  CeedScalar H_roe = RoeAverage(r, H_left, H_right);
  CeedScalar a_roe = sqrt((gamma - 1) * (H_roe - 0.5 * Square(u_roe)));

  // Einfeldt (1988) justifies (and Toro's book repeats) that Roe speeds can be used here.
  CeedScalar s_left = u_roe - a_roe;
  CeedScalar s_right = u_roe + a_roe;

  // Compute HLL flux
  if (0 <= s_left) {
    RiemannFlux_HLL = flux_left;
  } else if (s_right <= 0) {
    RiemannFlux_HLL = flux_right;
  } else {
    RiemannFlux_HLL.density =
      (s_right * flux_left.density - s_left * flux_right.density +
       s_left * s_right * (right.U.density - left.U.density)) /
      (s_right - s_left);
    for (int i = 0; i < 3; i++)
      RiemannFlux_HLL.momentum[i] =
        (s_right * flux_left.momentum[i] - s_left * flux_right.momentum[i] +
         s_left * s_right * (right.U.momentum[i] - left.U.momentum[i])) /
        (s_right - s_left);
    RiemannFlux_HLL.E_total =
      (s_right * flux_left.E_total - s_left * flux_right.E_total +
       s_left * s_right * (right.U.E_total - left.U.E_total)) /
      (s_right - s_left);
  }
  // Return
  return RiemannFlux_HLL;
}

// *****************************************************************************
// Freestream Boundary Condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int Freestream(void *ctx, CeedInt Q,
                                     const CeedScalar *const *in, CeedScalar *const *out,
                                     StateFromQi_t StateFromQi, StateFromQi_fwd_t StateFromQi_fwd) {

  //*INDENT-OFF*
  const CeedScalar (*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*x)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[3];

  CeedScalar (*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA]) out[0];

  //*INDENT-ON*

  const FreestreamContext context = (FreestreamContext) ctx;
  const NewtonianIdealGasContext newt_ctx = &context->newtonian_ctx;
  const bool is_implicit  = newt_ctx->is_implicit;

  CeedPragmaSIMD
  for(CeedInt i=0; i<Q; i++) {
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State s = StateFromQi(newt_ctx, qi, x_i);

    const CeedScalar wdetJb  = (is_implicit ? -1. : 1.) * q_data_sur[0][i];
    // ---- Normal vector
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    StateConservative HLL_flux = Harten_Lax_VanLeer_Flux(newt_ctx, s,
                                 context->S_infty, norm);
    CeedScalar Flux[5];
    UnpackState_U(HLL_flux, Flux);
    for (CeedInt j=0; j<5; j++) v[j][i] = -wdetJb * Flux[j];
  }
  return 0;
}

CEED_QFUNCTION(Freestream_Conserv)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(Freestream_Prim)(void *ctx, CeedInt Q,
                                const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

