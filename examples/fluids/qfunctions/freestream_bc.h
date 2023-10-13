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

enum RiemannFluxType_ { RIEMANN_HLL, RIEMANN_HLLC };
typedef enum RiemannFluxType_ RiemannFluxType;

typedef struct {
  CeedScalar left, right;
} RoeWeights;

CEED_QFUNCTION_HELPER RoeWeights RoeSetup(CeedScalar rho_left, CeedScalar rho_right) {
  CeedScalar sqrt_left = sqrt(rho_left), sqrt_right = sqrt(rho_right);
  RoeWeights w = {sqrt_left / (sqrt_left + sqrt_right), sqrt_right / (sqrt_left + sqrt_right)};
  return w;
}

CEED_QFUNCTION_HELPER RoeWeights RoeSetup_fwd(CeedScalar rho_left, CeedScalar rho_right, CeedScalar drho_left, CeedScalar drho_right) {
  CeedScalar sqrt_left = sqrt(rho_left), sqrt_right = sqrt(rho_right);
  CeedScalar square_sum_root = Square(sqrt_left + sqrt_right);
  CeedScalar r_right = (sqrt_left / (2 * sqrt_right * square_sum_root)) * drho_right - (sqrt_right / (2 * sqrt_left * square_sum_root)) * drho_left;
  CeedScalar r_left  = (sqrt_right / (2 * sqrt_left * square_sum_root)) * drho_left - (sqrt_left / (2 * sqrt_right * square_sum_root)) * drho_right;
  RoeWeights dw      = {r_left, r_right};
  return dw;
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
  StateConservative F = {
      F_hll[0],
      {F_hll[1], F_hll[2], F_hll[3]},
      F_hll[4],
  };
  return F;
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
  StateConservative dF = {
      dF_hll[0],
      {dF_hll[1], dF_hll[2], dF_hll[3]},
      dF_hll[4],
  };
  return dF;
}

CEED_QFUNCTION_HELPER void ComputeHLLSpeeds_Roe(NewtonianIdealGasContext gas, State left, CeedScalar u_left, State right, CeedScalar u_right,
                                                CeedScalar *s_left, CeedScalar *s_right) {
  const CeedScalar gamma = HeatCapacityRatio(gas);

  RoeWeights r = RoeSetup(left.U.density, right.U.density);
  // Speed estimate
  // Roe average eigenvalues for left and right non-linear waves.
  // Stability requires that these speed estimates are *at least* as fast as the physical wave speeds.
  CeedScalar u_roe = RoeAverage(r, u_left, u_right);

  // TODO: revisit this for gravity
  CeedScalar H_left  = TotalSpecificEnthalpy(gas, left);
  CeedScalar H_right = TotalSpecificEnthalpy(gas, right);
  CeedScalar H_roe   = RoeAverage(r, H_left, H_right);
  CeedScalar a_roe   = sqrt((gamma - 1) * (H_roe - 0.5 * Square(u_roe)));

  // Einfeldt (1988) justifies (and Toro's book repeats) that Roe speeds can be used here.
  *s_left  = u_roe - a_roe;
  *s_right = u_roe + a_roe;
}

CEED_QFUNCTION_HELPER void ComputeHLLSpeeds_Roe_fwd(NewtonianIdealGasContext gas, State left, State dleft, CeedScalar u_left, CeedScalar du_left,
                                                    State right, State dright, CeedScalar u_right, CeedScalar du_right, CeedScalar *s_left,
                                                    CeedScalar *ds_left, CeedScalar *s_right, CeedScalar *ds_right) {
  const CeedScalar gamma = HeatCapacityRatio(gas);

  RoeWeights r  = RoeSetup(left.U.density, right.U.density);
  RoeWeights dr = RoeSetup_fwd(left.U.density, right.U.density, dleft.U.density, dright.U.density);
  // Speed estimate
  // Roe average eigenvalues for left and right non-linear waves.
  // Stability requires that these speed estimates are *at least* as fast as the physical wave speeds.
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

  *s_left   = u_roe - a_roe;
  *ds_left  = du_roe - da_roe;
  *s_right  = u_roe + a_roe;
  *ds_right = du_roe + da_roe;
}

// *****************************************************************************
// @brief Harten Lax VanLeer (HLL) approximate Riemann solver.
// Taking in two states (left, right) and returns RiemannFlux_HLL.
// The left and right states are specified from the perspective of an outward-facing normal vector pointing left to right.
//
// @param[in] gas    NewtonianIdealGasContext for the fluid
// @param[in] left   Fluid state of the domain interior (the current solution)
// @param[in] right  Fluid state of the domain exterior (free stream conditions)
// @param[in] normal Normalized, outward facing boundary normal vector
//
// @return StateConservative with HLL Riemann Flux
// *****************************************************************************
CEED_QFUNCTION_HELPER StateConservative RiemannFlux_HLL(NewtonianIdealGasContext gas, State left, State right, const CeedScalar normal[3]) {
  StateConservative flux_left  = FluxInviscidDotNormal(gas, left, normal);
  StateConservative flux_right = FluxInviscidDotNormal(gas, right, normal);

  CeedScalar u_left  = Dot3(left.Y.velocity, normal);
  CeedScalar u_right = Dot3(right.Y.velocity, normal);

  CeedScalar s_left, s_right;
  ComputeHLLSpeeds_Roe(gas, left, u_left, right, u_right, &s_left, &s_right);

  // Compute HLL flux
  if (0 <= s_left) {
    return flux_left;
  } else if (s_right <= 0) {
    return flux_right;
  } else {
    return Flux_HLL(left, right, flux_left, flux_right, s_left, s_right);
  }
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
//
// @return StateConservative with derivative of HLL Riemann Flux
// *****************************************************************************
CEED_QFUNCTION_HELPER StateConservative RiemannFlux_HLL_fwd(NewtonianIdealGasContext gas, State left, State dleft, State right, State dright,
                                                            const CeedScalar normal[3]) {
  StateConservative flux_left   = FluxInviscidDotNormal(gas, left, normal);
  StateConservative flux_right  = FluxInviscidDotNormal(gas, right, normal);
  StateConservative dflux_left  = FluxInviscidDotNormal_fwd(gas, left, dleft, normal);
  StateConservative dflux_right = FluxInviscidDotNormal_fwd(gas, right, dright, normal);

  CeedScalar u_left   = Dot3(left.Y.velocity, normal);
  CeedScalar u_right  = Dot3(right.Y.velocity, normal);
  CeedScalar du_left  = Dot3(dleft.Y.velocity, normal);
  CeedScalar du_right = Dot3(dright.Y.velocity, normal);

  CeedScalar s_left, ds_left, s_right, ds_right;
  ComputeHLLSpeeds_Roe_fwd(gas, left, dleft, u_left, du_left, right, dright, u_right, du_right, &s_left, &ds_left, &s_right, &ds_right);

  if (0 <= s_left) {
    return dflux_left;
  } else if (s_right <= 0) {
    return dflux_right;
  } else {
    return Flux_HLL_fwd(left, right, dleft, dright, flux_left, flux_right, dflux_left, dflux_right, s_left, s_right, ds_left, ds_right);
  }
}

CEED_QFUNCTION_HELPER StateConservative RiemannFlux_HLLC_Star(NewtonianIdealGasContext gas, State side, StateConservative F_side,
                                                              const CeedScalar normal[3], CeedScalar u_side, CeedScalar s_side, CeedScalar s_star) {
  CeedScalar fact  = side.U.density * (s_side - u_side) / (s_side - s_star);
  CeedScalar denom = side.U.density * (s_side - u_side);
  // U_* = fact * star
  StateConservative star = {
      1.0,
      {
        side.Y.velocity[0] + (s_star - u_side) * normal[0],
        side.Y.velocity[1] + (s_star - u_side) * normal[1],
        side.Y.velocity[2] + (s_star - u_side) * normal[2],
        },
      side.U.E_total / side.U.density + (s_star - u_side) * (s_star + side.Y.pressure / denom),
  };
  return StateConservativeAXPBYPCZ(1, F_side, s_side * fact, star, -s_side, side.U);
}

CEED_QFUNCTION_HELPER StateConservative RiemannFlux_HLLC_Star_fwd(NewtonianIdealGasContext gas, State side, State dside, StateConservative F_side,
                                                                  StateConservative dF_side, const CeedScalar normal[3], CeedScalar u_side,
                                                                  CeedScalar du_side, CeedScalar s_side, CeedScalar ds_side, CeedScalar s_star,
                                                                  CeedScalar ds_star) {
  CeedScalar fact  = side.U.density * (s_side - u_side) / (s_side - s_star);
  CeedScalar dfact = (side.U.density * (ds_side - du_side) + dside.U.density * (s_side - u_side)) / (s_side - s_star)  //
                     - fact / (s_side - s_star) * (ds_side - ds_star);
  CeedScalar denom  = side.U.density * (s_side - u_side);
  CeedScalar ddenom = side.U.density * (ds_side - du_side) + dside.U.density * (s_side - u_side);

  StateConservative star = {
      1.0,
      {
        side.Y.velocity[0] + (s_star - u_side) * normal[0],
        side.Y.velocity[1] + (s_star - u_side) * normal[1],
        side.Y.velocity[2] + (s_star - u_side) * normal[2],
        },
      side.U.E_total / side.U.density  //
          + (s_star - u_side) * (s_star + side.Y.pressure / denom),
  };
  StateConservative dstar = {
      0.,
      {
        dside.Y.velocity[0] + (ds_star - du_side) * normal[0],
        dside.Y.velocity[1] + (ds_star - du_side) * normal[1],
        dside.Y.velocity[2] + (ds_star - du_side) * normal[2],
        },
      dside.U.E_total / side.U.density - side.U.E_total / Square(side.U.density) * dside.U.density  //
          + (ds_star - du_side) * (s_star + side.Y.pressure / denom)  //
          + (s_star - u_side) * (ds_star + dside.Y.pressure / denom - side.Y.pressure / Square(denom) * ddenom)  //,
  };

  const CeedScalar        a[] = {1, ds_side * fact + s_side * dfact, s_side * fact, -ds_side, -s_side};
  const StateConservative U[] = {dF_side, star, dstar, side.U, dside.U};
  return StateConservativeMult(5, a, U);
}

// HLLC Riemann solver (from Toro's book)
CEED_QFUNCTION_HELPER StateConservative RiemannFlux_HLLC(NewtonianIdealGasContext gas, State left, State right, const CeedScalar normal[3]) {
  StateConservative flux_left  = FluxInviscidDotNormal(gas, left, normal);
  StateConservative flux_right = FluxInviscidDotNormal(gas, right, normal);

  CeedScalar u_left  = Dot3(left.Y.velocity, normal);
  CeedScalar u_right = Dot3(right.Y.velocity, normal);
  CeedScalar s_left, s_right;
  ComputeHLLSpeeds_Roe(gas, left, u_left, right, u_right, &s_left, &s_right);

  // Contact wave speed; Toro (10.37)
  CeedScalar rhou_left = left.U.density * u_left, rhou_right = right.U.density * u_right;
  CeedScalar numer  = right.Y.pressure - left.Y.pressure + rhou_left * (s_left - u_left) - rhou_right * (s_right - u_right);
  CeedScalar denom  = left.U.density * (s_left - u_left) - right.U.density * (s_right - u_right);
  CeedScalar s_star = numer / denom;

  // Compute HLLC flux
  if (0 <= s_left) {
    return flux_left;
  } else if (0 <= s_star) {
    return RiemannFlux_HLLC_Star(gas, left, flux_left, normal, u_left, s_left, s_star);
  } else if (0 <= s_right) {
    return RiemannFlux_HLLC_Star(gas, right, flux_right, normal, u_right, s_right, s_star);
  } else {
    return flux_right;
  }
}

CEED_QFUNCTION_HELPER StateConservative RiemannFlux_HLLC_fwd(NewtonianIdealGasContext gas, State left, State dleft, State right, State dright,
                                                             const CeedScalar normal[3]) {
  StateConservative flux_left   = FluxInviscidDotNormal(gas, left, normal);
  StateConservative flux_right  = FluxInviscidDotNormal(gas, right, normal);
  StateConservative dflux_left  = FluxInviscidDotNormal_fwd(gas, left, dleft, normal);
  StateConservative dflux_right = FluxInviscidDotNormal_fwd(gas, right, dright, normal);

  CeedScalar u_left   = Dot3(left.Y.velocity, normal);
  CeedScalar u_right  = Dot3(right.Y.velocity, normal);
  CeedScalar du_left  = Dot3(dleft.Y.velocity, normal);
  CeedScalar du_right = Dot3(dright.Y.velocity, normal);

  CeedScalar s_left, ds_left, s_right, ds_right;
  ComputeHLLSpeeds_Roe_fwd(gas, left, dleft, u_left, du_left, right, dright, u_right, du_right, &s_left, &ds_left, &s_right, &ds_right);

  // Contact wave speed; Toro (10.37)
  CeedScalar rhou_left = left.U.density * u_left, drhou_left = left.U.density * du_left + dleft.U.density * u_left;
  CeedScalar rhou_right = right.U.density * u_right, drhou_right = right.U.density * du_right + dright.U.density * u_right;
  CeedScalar numer = right.Y.pressure - left.Y.pressure  //
                     + rhou_left * (s_left - u_left)     //
                     - rhou_right * (s_right - u_right);
  CeedScalar dnumer = dright.Y.pressure - dleft.Y.pressure                                //
                      + rhou_left * (ds_left - du_left) + drhou_left * (s_left - u_left)  //
                      - rhou_right * (ds_right - du_right) - drhou_right * (s_right - u_right);
  CeedScalar denom  = left.U.density * (s_left - u_left) - right.U.density * (s_right - u_right);
  CeedScalar ddenom = left.U.density * (ds_left - du_left) + dleft.U.density * (s_left - u_left)  //
                      - right.U.density * (ds_right - du_right) - dright.U.density * (s_right - u_right);
  CeedScalar s_star  = numer / denom;
  CeedScalar ds_star = dnumer / denom - numer / Square(denom) * ddenom;

  // Compute HLLC flux
  if (0 <= s_left) {
    return dflux_left;
  } else if (0 <= s_star) {
    return RiemannFlux_HLLC_Star_fwd(gas, left, dleft, flux_left, dflux_left, normal, u_left, du_left, s_left, ds_left, s_star, ds_star);
  } else if (0 <= s_right) {
    return RiemannFlux_HLLC_Star_fwd(gas, right, dright, flux_right, dflux_right, normal, u_right, du_right, s_right, ds_right, s_star, ds_star);
  } else {
    return dflux_right;
  }
}

// *****************************************************************************
// Freestream Boundary Condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int Freestream(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var,
                                     RiemannFluxType flux_type) {
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)    = in[2];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)  = out[1];

  const FreestreamContext        context     = (FreestreamContext)ctx;
  const NewtonianIdealGasContext newt_ctx    = &context->newtonian_ctx;
  const bool                     is_implicit = newt_ctx->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s     = StateFromQ(newt_ctx, qi, state_var);

    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    StateConservative flux;
    switch (flux_type) {
      case RIEMANN_HLL:
        flux = RiemannFlux_HLL(newt_ctx, s, context->S_infty, norm);
        break;
      case RIEMANN_HLLC:
        flux = RiemannFlux_HLLC(newt_ctx, s, context->S_infty, norm);
        break;
    }
    CeedScalar Flux[5];
    UnpackState_U(flux, Flux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
  }
  return 0;
}

CEED_QFUNCTION(Freestream_Conserv_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Prim_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Conserv_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION(Freestream_Prim_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION_HELPER int Freestream_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var,
                                              RiemannFluxType flux_type) {
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const FreestreamContext        context     = (FreestreamContext)ctx;
  const NewtonianIdealGasContext newt_ctx    = &context->newtonian_ctx;
  const bool                     is_implicit = newt_ctx->is_implicit;
  const State                    dS_infty    = {0};

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    CeedScalar qi[5], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];
    State s  = StateFromQ(newt_ctx, qi, state_var);
    State ds = StateFromQ_fwd(newt_ctx, s, dqi, state_var);

    StateConservative dflux;
    switch (flux_type) {
      case RIEMANN_HLL:
        dflux = RiemannFlux_HLL_fwd(newt_ctx, s, ds, context->S_infty, dS_infty, norm);
        break;
      case RIEMANN_HLLC:
        dflux = RiemannFlux_HLLC_fwd(newt_ctx, s, ds, context->S_infty, dS_infty, norm);
        break;
    }
    CeedScalar dFlux[5];
    UnpackState_U(dflux, dFlux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }
  return 0;
}

CEED_QFUNCTION(Freestream_Jacobian_Conserv_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Jacobian_Prim_HLL)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLL);
}

CEED_QFUNCTION(Freestream_Jacobian_Conserv_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE, RIEMANN_HLLC);
}

CEED_QFUNCTION(Freestream_Jacobian_Prim_HLLC)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return Freestream_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE, RIEMANN_HLLC);
}

// Note the identity
//
// softplus(x) - x = log(1 + exp(x)) - x
//                 = log(1 + exp(x)) + log(exp(-x))
//                 = log((1 + exp(x)) * exp(-x))
//                 = log(exp(-x) + 1)
//                 = softplus(-x)
CEED_QFUNCTION_HELPER CeedScalar Softplus(CeedScalar x, CeedScalar width) {
  if (x > 40 * width) return x;
  return width * log1p(exp(x / width));
}

CEED_QFUNCTION_HELPER CeedScalar Softplus_fwd(CeedScalar x, CeedScalar dx, CeedScalar width) {
  if (x > 40 * width) return 1;
  const CeedScalar t = exp(x / width);
  return t / (1 + t);
}

// Viscous Outflow boundary condition, setting a constant exterior pressure and
// temperature as input for a Riemann solve. This condition is stable even in
// recirculating flow so long as the exterior temperature is sensible.
//
// The velocity in the exterior state has optional softplus regularization to
// keep it outflow. These parameters have been finnicky in practice and provide
// little or no benefit in the tests we've run thus far, thus we recommend
// skipping this feature and just allowing recirculation.
CEED_QFUNCTION_HELPER int RiemannOutflow(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data_sur)    = in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)  = out[1];

  const OutflowContext           outflow     = (OutflowContext)ctx;
  const NewtonianIdealGasContext gas         = &outflow->gas;
  const bool                     is_implicit = gas->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s_int = StateFromQ(gas, qi, state_var);

    StatePrimitive y_ext      = s_int.Y;
    y_ext.pressure            = outflow->pressure;
    y_ext.temperature         = outflow->temperature;
    const CeedScalar u_normal = Dot3(y_ext.velocity, norm);
    const CeedScalar proj     = (1 - outflow->recirc) * Softplus(-u_normal, outflow->softplus_velocity);
    for (CeedInt j = 0; j < 3; j++) {
      y_ext.velocity[j] += norm[j] * proj;  // (I - n n^T) projects into the plane tangent to the normal
    }
    State s_ext = StateFromPrimitive(gas, y_ext);

    State grad_s[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s_int, state_var, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(gas, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(gas, s_int.Y, grad_s, stress, Fe);

    StateConservative F_inviscid_normal = RiemannFlux_HLLC(gas, s_int, s_ext, norm);

    CeedScalar Flux[5];
    FluxTotal_RiemannBoundary(F_inviscid_normal, stress, Fe, norm, Flux);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    // Save values for Jacobian
    StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
    StoredValuesPack(Q, i, 5, 6, kmstress, jac_data_sur);
  }
  return 0;
}

CEED_QFUNCTION(RiemannOutflow_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return RiemannOutflow(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(RiemannOutflow_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return RiemannOutflow(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// *****************************************************************************
// Jacobian for Riemann pressure/temperature outflow boundary condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int RiemannOutflow_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                  StateVariable state_var) {
  // Inputs
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_dq)        = in[1];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const OutflowContext           outflow     = (OutflowContext)ctx;
  const NewtonianIdealGasContext gas         = &outflow->gas;
  const bool                     is_implicit = gas->is_implicit;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    CeedScalar qi[5], kmstress[6], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    StoredValuesUnpack(Q, i, 5, 6, jac_data_sur, kmstress);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];

    State          s_int  = StateFromQ(gas, qi, state_var);
    const State    ds_int = StateFromQ_fwd(gas, s_int, dqi, state_var);
    StatePrimitive y_ext = s_int.Y, dy_ext = ds_int.Y;
    y_ext.pressure             = outflow->pressure;
    y_ext.temperature          = outflow->temperature;
    dy_ext.pressure            = 0;
    dy_ext.temperature         = 0;
    const CeedScalar u_normal  = Dot3(s_int.Y.velocity, norm);
    const CeedScalar du_normal = Dot3(ds_int.Y.velocity, norm);
    const CeedScalar proj      = (1 - outflow->recirc) * Softplus(-u_normal, outflow->softplus_velocity);
    const CeedScalar dproj     = (1 - outflow->recirc) * Softplus_fwd(-u_normal, -du_normal, outflow->softplus_velocity);
    for (CeedInt j = 0; j < 3; j++) {
      y_ext.velocity[j] += norm[j] * proj;
      dy_ext.velocity[j] += norm[j] * dproj;
    }

    State s_ext  = StateFromPrimitive(gas, y_ext);
    State ds_ext = StateFromPrimitive_fwd(gas, s_ext, dy_ext);

    State grad_ds[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s_int, state_var, Grad_dq, dXdx, grad_ds);

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate_State(grad_ds, dstrain_rate);
    NewtonianStress(gas, dstrain_rate, dkmstress);
    KMUnpack(dkmstress, dstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux_fwd(gas, s_int.Y, ds_int.Y, grad_ds, stress, dstress, dFe);

    StateConservative dF_inviscid_normal = RiemannFlux_HLLC_fwd(gas, s_int, ds_int, s_ext, ds_ext, norm);

    CeedScalar dFlux[5];
    FluxTotal_RiemannBoundary(dF_inviscid_normal, dstress, dFe, norm, dFlux);

    for (int j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }  // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(RiemannOutflow_Jacobian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return RiemannOutflow_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(RiemannOutflow_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return RiemannOutflow_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// *****************************************************************************
// Outflow boundary condition, weakly setting a constant pressure. This is the
// classic outflow condition used by PHASTA-C and retained largely for
// comparison. In our experiments, it is never better than RiemannOutflow, and
// will crash if outflow ever becomes an inflow, as occurs with strong
// acoustics, vortices, etc.
// *****************************************************************************
CEED_QFUNCTION_HELPER int PressureOutflow(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data_sur)    = in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)  = out[1];

  const OutflowContext           outflow     = (OutflowContext)ctx;
  const NewtonianIdealGasContext gas         = &outflow->gas;
  const bool                     is_implicit = gas->is_implicit;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s     = StateFromQ(gas, qi, state_var);
    s.Y.pressure           = outflow->pressure;

    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    State grad_s[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s, state_var, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(gas, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(gas, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(gas, s, F_inviscid);

    CeedScalar Flux[5];
    FluxTotal_Boundary(F_inviscid, stress, Fe, norm, Flux);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    // Save values for Jacobian
    StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
    StoredValuesPack(Q, i, 5, 6, kmstress, jac_data_sur);
  }  // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(PressureOutflow_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(PressureOutflow_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// *****************************************************************************
// Jacobian for weak-pressure outflow boundary condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int PressureOutflow_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                   StateVariable state_var) {
  // Inputs
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_dq)        = in[1];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const OutflowContext           outflow     = (OutflowContext)ctx;
  const NewtonianIdealGasContext gas         = &outflow->gas;
  const bool                     is_implicit = gas->is_implicit;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    CeedScalar qi[5], kmstress[6], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    StoredValuesUnpack(Q, i, 5, 6, jac_data_sur, kmstress);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];

    State s       = StateFromQ(gas, qi, state_var);
    State ds      = StateFromQ_fwd(gas, s, dqi, state_var);
    s.Y.pressure  = outflow->pressure;
    ds.Y.pressure = 0.;

    State grad_ds[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, gas, s, state_var, Grad_dq, dXdx, grad_ds);

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate_State(grad_ds, dstrain_rate);
    NewtonianStress(gas, dstrain_rate, dkmstress);
    KMUnpack(dkmstress, dstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux_fwd(gas, s.Y, ds.Y, grad_ds, stress, dstress, dFe);

    StateConservative dF_inviscid[3];
    FluxInviscid_fwd(gas, s, ds, dF_inviscid);

    CeedScalar dFlux[5];
    FluxTotal_Boundary(dF_inviscid, dstress, dFe, norm, dFlux);

    for (int j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }  // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(PressureOutflow_Jacobian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(PressureOutflow_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}
