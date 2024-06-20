// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Helper functions for solving the Riemann problem.
// The left and right states are specified from the perspective of an outward-facing normal vector pointing left to right:
//
//     (domain)
//    /               (outward facing normal)
// |------------|   /
// |            |  /
// |    Left    |---->  Right
// | (Interior) |    (Exterior)
// |------------|
//
// The right state is exterior to the domain and the left state is the interior to the domain.
// Much of the work references Eleuterio F. Toro's "Riemann Solvers and Numerical Methods for Fluid Dynamics", 2009
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
    const CeedScalar S_diff = S_r - S_l;

    dF_hll[i] += (S_l * (-F_l[i] + F_r[i] + S_l * U_l[i] - S_l * U_r[i]) / Square(S_diff)) * dS_r;
    dF_hll[i] += (S_r * (F_l[i] - F_r[i] - S_r * U_l[i] + S_r * U_r[i]) / Square(S_diff)) * dS_l;
    dF_hll[i] += (S_r * dF_l[i] - S_l * dF_r[i] + S_r * S_l * (dU_r[i] - dU_l[i])) / S_diff;
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
  CeedScalar da_roe = 0.5 * sqrt((gamma - 1) / (H_roe - 0.5 * Square(u_roe))) * dH_roe;  // (da/dH) dH
  da_roe -= 0.5 * sqrt(gamma - 1) * u_roe / sqrt(H_roe - 0.5 * Square(u_roe)) * du_roe;  // (da/du) du

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
      side.U.E_total / side.U.density  //
          + (s_star - u_side) * (s_star + side.Y.pressure / denom)
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
          + (s_star - u_side) * (s_star + side.Y.pressure / denom)
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
          + (s_star - u_side) * (ds_star + dside.Y.pressure / denom - side.Y.pressure / Square(denom) * ddenom)  //
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
