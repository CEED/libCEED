// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc

#include "freestream_bc_type.h"
#include "newtonian_state.h"
#include "newtonian_types.h"

typedef struct {
  CeedScalar left, right;
} RoeWeights;

CEED_QFUNCTION_HELPER RoeWeights RoeSetup(CeedScalar rho_left, CeedScalar rho_right) {
  CeedScalar sqrt_left = sqrt(rho_left), sqrt_right = sqrt(rho_right);
  return (RoeWeights){sqrt_left / (sqrt_left + sqrt_right), sqrt_right / (sqrt_left + sqrt_right)};
}

CEED_QFUNCTION_HELPER RoeWeights RoeSetup_fwd(CeedScalar rho_left, CeedScalar rho_right, CeedScalar drho_left, CeedScalar drho_right) {
  CeedScalar sqrt_left = sqrt(rho_left), sqrt_right = sqrt(rho_right);
  CeedScalar square_sum_root = Square(sqrt_left + sqrt_right);
  CeedScalar r_right = (sqrt_left / (2 * sqrt_right * square_sum_root)) * drho_right - (sqrt_right / (2 * sqrt_left * square_sum_root)) * drho_left;
  CeedScalar r_left  = (sqrt_right / (2 * sqrt_left * square_sum_root)) * drho_left - (sqrt_left / (2 * sqrt_right * square_sum_root)) * drho_right;
  return (RoeWeights){r_left, r_right};
}

CEED_QFUNCTION_HELPER CeedScalar RoeAverage(RoeWeights r, CeedScalar q_left, CeedScalar q_right) { return r.left * q_left + r.right * q_right; }

CEED_QFUNCTION_HELPER CeedScalar RoeAverage_fwd(RoeWeights r, RoeWeights dr, CeedScalar q_left, CeedScalar q_right, CeedScalar dq_left,
                                                CeedScalar dq_right) {
  return q_right * dr.right + q_left * dr.left + r.right * dq_right + r.left * dq_left;
}

CEED_QFUNCTION_HELPER StateConservative Flux_HLL(State left, State right, StateConservative flux_left, StateConservative flux_right,
                                                 CeedScalar s_left, CeedScalar s_right) {
  CeedScalar U_left[5], U_right[5], F_right[5], F_left[5], F_hll[5];
  UnpackState_U(left.U, U_left);
  UnpackState_U(right.U, U_right);
  UnpackState_U(flux_left, F_left);
  UnpackState_U(flux_right, F_right);
  for (int i = 0; i < 5; i++) {
    F_hll[i] = (s_right * F_left[i] - s_left * F_right[i] + s_left * s_right * (U_right[i] - U_left[i])) / (s_right - s_left);
  }
  return (StateConservative){
      F_hll[0], {F_hll[1], F_hll[2], F_hll[3]},
       F_hll[4]
  };
}

CEED_QFUNCTION_HELPER StateConservative Flux_HLL_fwd(State left, State right, State dleft, State dright, StateConservative flux_left,
                                                     StateConservative flux_right, StateConservative dflux_left, StateConservative dflux_right,
                                                     CeedScalar S_l, CeedScalar S_r, CeedScalar dS_l, CeedScalar dS_r) {
  CeedScalar U_l[5], U_r[5], F_r[5], F_l[5];
  UnpackState_U(left.U, U_l);
  UnpackState_U(right.U, U_r);
  UnpackState_U(flux_left, F_l);
  UnpackState_U(flux_right, F_r);

  CeedScalar dU_l[5], dU_r[5], dF_r[5], dF_l[5], dF_hll[5] = {0.};
  UnpackState_U(dleft.U, dU_l);
  UnpackState_U(dright.U, dU_r);
  UnpackState_U(dflux_left, dF_l);
  UnpackState_U(dflux_right, dF_r);
  for (int i = 0; i < 5; i++) {
    const CeedScalar U_diff      = U_r[i] - U_l[i];
    const CeedScalar S_diff      = S_r - S_l;
    const CeedScalar F_hll_denom = S_r * F_l[i] - S_l * F_r[i] + S_l * S_r * U_diff;

    dF_hll[i] += ((F_l[i] + S_r * U_diff) * S_diff - F_hll_denom) / Square(S_diff) * dS_r;
    dF_hll[i] += ((-F_r[i] + S_r * U_diff) * S_diff + F_hll_denom) / Square(S_diff) * dS_l;
    dF_hll[i] += (S_r * dF_l[i] - S_l * dF_r[i] + S_r * S_l * dU_r[i] - S_r * S_l * dU_l[i]) / S_diff;
  }
  return (StateConservative){
      dF_hll[0], {dF_hll[1], dF_hll[2], dF_hll[3]},
       dF_hll[4]
  };
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
CEED_QFUNCTION_HELPER StateConservative Harten_Lax_VanLeer_Flux(NewtonianIdealGasContext gas, State left, State right, const CeedScalar normal[3]) {
  const CeedScalar gamma = HeatCapacityRatio(gas);

  StateConservative flux_left  = FluxInviscidDotNormal(gas, left, normal);
  StateConservative flux_right = FluxInviscidDotNormal(gas, right, normal);
  StateConservative RiemannFlux_HLL;

  CeedScalar u_left  = Dot3(left.Y.velocity, normal);
  CeedScalar u_right = Dot3(right.Y.velocity, normal);

  RoeWeights r = RoeSetup(left.U.density, right.U.density);
  // Speed estimate
  // Roe average eigenvalues for left and right non-linear waves
  // Stability requires that these speed estimates are *at least*
  // as fast as the physical wave speeds.
  CeedScalar u_roe = RoeAverage(r, u_left, u_right);

  CeedScalar H_left  = TotalSpecificEnthalpy(gas, left);
  CeedScalar H_right = TotalSpecificEnthalpy(gas, right);
  CeedScalar H_roe   = RoeAverage(r, H_left, H_right);
  CeedScalar a_roe   = sqrt((gamma - 1) * (H_roe - 0.5 * Square(u_roe)));

  // Einfeldt (1988) justifies (and Toro's book repeats) that Roe speeds can be used here.
  CeedScalar s_left  = u_roe - a_roe;
  CeedScalar s_right = u_roe + a_roe;

  // Compute HLL flux
  if (0 <= s_left) {
    RiemannFlux_HLL = flux_left;
  } else if (s_right <= 0) {
    RiemannFlux_HLL = flux_right;
  } else {
    RiemannFlux_HLL = Flux_HLL(left, right, flux_left, flux_right, s_left, s_right);
  }
  return RiemannFlux_HLL;
}

// *****************************************************************************
// @brief Forward-mode Derivative of Harten Lax VanLeer (HLL) approximate Riemann solver.
//
// @param gas    NewtonianIdealGasContext for the fluid
// @param left   Fluid state of the domain interior (the current solution)
// @param right  Fluid state of the domain exterior (free stream conditions)
// @param dleft  Derivative of fluid state of the domain interior (the current solution)
// @param dright Derivative of fluid state of the domain exterior (free stream conditions)
// @param normal Normalized, outward facing boundary normal vector
// *****************************************************************************
CEED_QFUNCTION_HELPER StateConservative Harten_Lax_VanLeer_Flux_fwd(NewtonianIdealGasContext gas, State left, State right, State dleft, State dright,
                                                                    const CeedScalar normal[3]) {
  const CeedScalar gamma = HeatCapacityRatio(gas);

  StateConservative flux_left   = FluxInviscidDotNormal(gas, left, normal);
  StateConservative flux_right  = FluxInviscidDotNormal(gas, right, normal);
  StateConservative dflux_left  = FluxInviscidDotNormal_fwd(gas, left, dleft, normal);
  StateConservative dflux_right = FluxInviscidDotNormal_fwd(gas, right, dright, normal);

  CeedScalar u_left   = Dot3(left.Y.velocity, normal);
  CeedScalar u_right  = Dot3(right.Y.velocity, normal);
  CeedScalar du_left  = Dot3(dleft.Y.velocity, normal);
  CeedScalar du_right = Dot3(dright.Y.velocity, normal);

  RoeWeights r  = RoeSetup(left.U.density, right.U.density);
  RoeWeights dr = RoeSetup_fwd(left.U.density, right.U.density, dleft.U.density, dright.U.density);
  // Speed estimate
  // Roe average eigenvalues for left and right non-linear waves
  // Stability requires that these speed estimates are *at least*
  // as fast as the physical wave speeds.
  CeedScalar u_roe  = RoeAverage(r, u_left, u_right);
  CeedScalar du_roe = RoeAverage_fwd(r, dr, u_left, u_right, du_left, du_right);

  CeedScalar H_left   = TotalSpecificEnthalpy(gas, left);
  CeedScalar H_right  = TotalSpecificEnthalpy(gas, right);
  CeedScalar dH_left  = TotalSpecificEnthalpy_fwd(gas, left, dleft);
  CeedScalar dH_right = TotalSpecificEnthalpy_fwd(gas, right, dright);

  CeedScalar H_roe  = RoeAverage(r, H_left, H_right);
  CeedScalar dH_roe = RoeAverage_fwd(r, dr, H_left, H_right, dH_left, dH_right);
  CeedScalar a_roe  = sqrt((gamma - 1) * (H_roe - 0.5 * Square(u_roe)));
  CeedScalar da_roe = 0.5 * (gamma - 1) / sqrt(H_roe) * dH_roe - 0.5 * sqrt(gamma - 1) * u_roe / sqrt(H_roe - 0.5 * Square(u_roe)) * du_roe;

  // Einfeldt (1988) justifies (and Toro's book repeats) that Roe speeds can be used here.
  CeedScalar s_left   = u_roe - a_roe;
  CeedScalar s_right  = u_roe + a_roe;
  CeedScalar ds_left  = du_roe - da_roe;
  CeedScalar ds_right = du_roe + da_roe;

  // Compute HLL flux
  StateConservative dRiemannFlux_HLL;
  if (0 <= s_left) {
    dRiemannFlux_HLL = dflux_left;
  } else if (s_right <= 0) {
    dRiemannFlux_HLL = dflux_right;
  } else {
    dRiemannFlux_HLL = Flux_HLL_fwd(left, right, dleft, dright, flux_left, flux_right, dflux_left, dflux_right, s_left, s_right, ds_left, ds_right);
  }
  return dRiemannFlux_HLL;
}

// *****************************************************************************
// Freestream Boundary Condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int Freestream(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateFromQi_t StateFromQi,
                                     StateFromQi_fwd_t StateFromQi_fwd) {
  //*INDENT-OFF*
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
        (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*jac_data_sur)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];
  //*INDENT-ON*

  const FreestreamContext        context     = (FreestreamContext)ctx;
  const NewtonianIdealGasContext newt_ctx    = &context->newtonian_ctx;
  const bool                     is_implicit = newt_ctx->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s      = StateFromQi(newt_ctx, qi, x_i);

    const CeedScalar wdetJb = (is_implicit ? -1. : 1.) * q_data_sur[0][i];
    // ---- Normal vector
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};

    StateConservative HLL_flux = Harten_Lax_VanLeer_Flux(newt_ctx, s, context->S_infty, norm);
    CeedScalar        Flux[5];
    UnpackState_U(HLL_flux, Flux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    for (int j = 0; j < 5; j++) jac_data_sur[j][i] = qi[j];
  }
  return 0;
}

CEED_QFUNCTION(Freestream_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(Freestream_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

CEED_QFUNCTION_HELPER int Freestream_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateFromQi_t StateFromQi,
                                              StateFromQi_fwd_t StateFromQi_fwd) {
  //*INDENT-OFF*
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
        (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3], (*jac_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  //*INDENT-ON*

  const FreestreamContext        context     = (FreestreamContext)ctx;
  const NewtonianIdealGasContext newt_ctx    = &context->newtonian_ctx;
  const bool                     is_implicit = newt_ctx->is_implicit;
  const State                    dS_infty    = {{0}};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x_i[3]  = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar wdetJb  = (is_implicit ? -1. : 1.) * q_data_sur[0][i];
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};

    CeedScalar qi[5], dqi[5], dx_i[3] = {0.};
    for (int j = 0; j < 5; j++) qi[j] = jac_data_sur[j][i];
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];
    State s  = StateFromQi(newt_ctx, qi, x_i);
    State ds = StateFromQi_fwd(newt_ctx, s, dqi, x_i, dx_i);

    StateConservative dHLL_flux = Harten_Lax_VanLeer_Flux_fwd(newt_ctx, s, context->S_infty, ds, dS_infty, norm);
    CeedScalar        Flux[5];
    UnpackState_U(dHLL_flux, Flux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];
  }
  return 0;
}

CEED_QFUNCTION(Freestream_Jacobian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(Freestream_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}
