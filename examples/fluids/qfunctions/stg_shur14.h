// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Implementation of the Synthetic Turbulence Generation (STG) algorithm
/// presented in Shur et al. 2014
//
/// SetupSTG_Rand reads in the input files and fills in STGShur14Context.
/// Then STGShur14_CalcQF is run over quadrature points.
/// Before the program exits, TearDownSTG is run to free the memory of the allocated arrays.
#include <ceed/types.h>
#ifndef CEED_RUNNING_JIT_PASS
#include <math.h>
#include <stdlib.h>
#endif

#include "newtonian_state.h"
#include "setupgeo_helpers.h"
#include "stg_shur14_type.h"
#include "utils.h"

#define STG_NMODES_MAX 1024

/*
 * @brief Interpolate quantities from input profile to given location
 *
 * Assumed that prof_wd[i+1] > prof_wd[i] and prof_wd[0] = 0
 * If wall_dist > prof_wd[-1], then the interpolation takes the values at prof_wd[-1]
 *
 * @param[in]  wall_dist Distance to the nearest wall
 * @param[out] ubar      Mean velocity at wall_dist
 * @param[out] cij       Cholesky decomposition at wall_dist
 * @param[out] eps       Turbulent dissipation at wall_dist
 * @param[out] lt        Turbulent length scale at wall_dist
 * @param[in]  stg_ctx   STGShur14Context for the problem
 */
CEED_QFUNCTION_HELPER void InterpolateProfile(const CeedScalar wall_dist, CeedScalar ubar[3], CeedScalar cij[6], CeedScalar *eps, CeedScalar *lt,
                                              const StgShur14Context stg_ctx) {
  const CeedInt     nprofs    = stg_ctx->nprofs;
  const CeedScalar *prof_wd   = &stg_ctx->data[stg_ctx->offsets.wall_dist];
  const CeedScalar *prof_eps  = &stg_ctx->data[stg_ctx->offsets.eps];
  const CeedScalar *prof_lt   = &stg_ctx->data[stg_ctx->offsets.lt];
  const CeedScalar *prof_ubar = &stg_ctx->data[stg_ctx->offsets.ubar];
  const CeedScalar *prof_cij  = &stg_ctx->data[stg_ctx->offsets.cij];
  CeedInt           idx       = -1;

  for (CeedInt i = 0; i < nprofs; i++) {
    if (wall_dist < prof_wd[i]) {
      idx = i;
      break;
    }
  }

  if (idx > 0) {  // y within the bounds of prof_wd
    CeedScalar coeff = (wall_dist - prof_wd[idx - 1]) / (prof_wd[idx] - prof_wd[idx - 1]);

    ubar[0] = prof_ubar[0 * nprofs + idx - 1] + coeff * (prof_ubar[0 * nprofs + idx] - prof_ubar[0 * nprofs + idx - 1]);
    ubar[1] = prof_ubar[1 * nprofs + idx - 1] + coeff * (prof_ubar[1 * nprofs + idx] - prof_ubar[1 * nprofs + idx - 1]);
    ubar[2] = prof_ubar[2 * nprofs + idx - 1] + coeff * (prof_ubar[2 * nprofs + idx] - prof_ubar[2 * nprofs + idx - 1]);
    cij[0]  = prof_cij[0 * nprofs + idx - 1] + coeff * (prof_cij[0 * nprofs + idx] - prof_cij[0 * nprofs + idx - 1]);
    cij[1]  = prof_cij[1 * nprofs + idx - 1] + coeff * (prof_cij[1 * nprofs + idx] - prof_cij[1 * nprofs + idx - 1]);
    cij[2]  = prof_cij[2 * nprofs + idx - 1] + coeff * (prof_cij[2 * nprofs + idx] - prof_cij[2 * nprofs + idx - 1]);
    cij[3]  = prof_cij[3 * nprofs + idx - 1] + coeff * (prof_cij[3 * nprofs + idx] - prof_cij[3 * nprofs + idx - 1]);
    cij[4]  = prof_cij[4 * nprofs + idx - 1] + coeff * (prof_cij[4 * nprofs + idx] - prof_cij[4 * nprofs + idx - 1]);
    cij[5]  = prof_cij[5 * nprofs + idx - 1] + coeff * (prof_cij[5 * nprofs + idx] - prof_cij[5 * nprofs + idx - 1]);
    *eps    = prof_eps[idx - 1] + coeff * (prof_eps[idx] - prof_eps[idx - 1]);
    *lt     = prof_lt[idx - 1] + coeff * (prof_lt[idx] - prof_lt[idx - 1]);
  } else {  // y outside bounds of prof_wd
    ubar[0] = prof_ubar[1 * nprofs - 1];
    ubar[1] = prof_ubar[2 * nprofs - 1];
    ubar[2] = prof_ubar[3 * nprofs - 1];
    cij[0]  = prof_cij[1 * nprofs - 1];
    cij[1]  = prof_cij[2 * nprofs - 1];
    cij[2]  = prof_cij[3 * nprofs - 1];
    cij[3]  = prof_cij[4 * nprofs - 1];
    cij[4]  = prof_cij[5 * nprofs - 1];
    cij[5]  = prof_cij[6 * nprofs - 1];
    *eps    = prof_eps[nprofs - 1];
    *lt     = prof_lt[nprofs - 1];
  }
}

/*
 * @brief Calculate spectrum coefficient, qn
 *
 * Calculates q_n at a given distance to the wall
 *
 * @param[in]  kappa     nth wavenumber
 * @param[in]  dkappa    Difference between wavenumbers
 * @param[in]  keta      Dissipation wavenumber
 * @param[in]  kcut      Mesh-induced cutoff wavenumber
 * @param[in]  ke        Energy-containing wavenumber
 * @param[in]  Ektot_inv Inverse of total turbulent kinetic energy of spectrum
 * @returns    qn        Spectrum coefficient
 */
CEED_QFUNCTION_HELPER CeedScalar Calc_qn(const CeedScalar kappa, const CeedScalar dkappa, const CeedScalar keta, const CeedScalar kcut,
                                         const CeedScalar ke, const CeedScalar Ektot_inv) {
  const CeedScalar feta_x_fcut = exp(-Square(12 * kappa / keta) - Cube(4 * Max(kappa - 0.9 * kcut, 0) / kcut));
  return pow(kappa / ke, 4.) * pow(1 + 2.4 * Square(kappa / ke), -17. / 6) * feta_x_fcut * dkappa * Ektot_inv;
}

// Calculate hmax, ke, keta, and kcut
CEED_QFUNCTION_HELPER void SpectrumConstants(const CeedScalar wall_dist, const CeedScalar eps, const CeedScalar lt, const CeedScalar hNodSep[3],
                                             const CeedScalar nu, CeedScalar *hmax, CeedScalar *ke, CeedScalar *keta, CeedScalar *kcut) {
  *hmax = Max(Max(hNodSep[0], hNodSep[1]), hNodSep[2]);
  *ke   = wall_dist == 0 ? 1e16 : 2 * M_PI / Min(2 * wall_dist, 3 * lt);
  *keta = 2 * M_PI * pow(Cube(nu) / eps, -0.25);
  *kcut = M_PI / Min(Max(Max(hNodSep[1], hNodSep[2]), 0.3 * (*hmax)) + 0.1 * wall_dist, *hmax);
}

/*
 * @brief Calculate spectrum coefficients for STG
 *
 * Calculates q_n at a given distance to the wall
 *
 * @param[in]  wall_dist  Distance to the nearest wall
 * @param[in]  eps        Turbulent dissipation w/rt wall_dist
 * @param[in]  lt         Turbulent length scale w/rt wall_dist
 * @param[in]  h_node_sep Element lengths in coordinate directions
 * @param[in]  nu         Dynamic Viscosity;
 * @param[in]  stg_ctx    STGShur14Context for the problem
 * @param[out] qn         Spectrum coefficients, [nmodes]
 */
CEED_QFUNCTION_HELPER void CalcSpectrum(const CeedScalar wall_dist, const CeedScalar eps, const CeedScalar lt, const CeedScalar h_node_sep[3],
                                        const CeedScalar nu, CeedScalar qn[], const StgShur14Context stg_ctx) {
  const CeedInt     nmodes = stg_ctx->nmodes;
  const CeedScalar *kappa  = &stg_ctx->data[stg_ctx->offsets.kappa];
  CeedScalar        hmax, ke, keta, kcut, Ektot = 0.0;

  SpectrumConstants(wall_dist, eps, lt, h_node_sep, nu, &hmax, &ke, &keta, &kcut);

  for (CeedInt n = 0; n < nmodes; n++) {
    const CeedScalar dkappa = n == 0 ? kappa[0] : kappa[n] - kappa[n - 1];
    qn[n]                   = Calc_qn(kappa[n], dkappa, keta, kcut, ke, 1.0);
    Ektot += qn[n];
  }

  if (Ektot == 0) return;
  for (CeedInt n = 0; n < nmodes; n++) qn[n] /= Ektot;
}

/******************************************************
 * @brief Calculate u(x,t) for STG inflow condition
 *
 * @param[in]  X       Location to evaluate u(X,t)
 * @param[in]  t       Time to evaluate u(X,t)
 * @param[in]  ubar    Mean velocity at X
 * @param[in]  cij     Cholesky decomposition at X
 * @param[in]  qn      Wavemode amplitudes at X, [nmodes]
 * @param[out] u       Velocity at X and t
 * @param[in]  stg_ctx STGShur14Context for the problem
 */
CEED_QFUNCTION_HELPER void StgShur14Calc(const CeedScalar X[3], const CeedScalar t, const CeedScalar ubar[3], const CeedScalar cij[6],
                                         const CeedScalar qn[], CeedScalar u[3], const StgShur14Context stg_ctx) {
  const CeedInt     nmodes = stg_ctx->nmodes;
  const CeedScalar *kappa  = &stg_ctx->data[stg_ctx->offsets.kappa];
  const CeedScalar *phi    = &stg_ctx->data[stg_ctx->offsets.phi];
  const CeedScalar *sigma  = &stg_ctx->data[stg_ctx->offsets.sigma];
  const CeedScalar *d      = &stg_ctx->data[stg_ctx->offsets.d];
  CeedScalar        xdotd, vp[3] = {0.};
  CeedScalar        xhat[] = {0., X[1], X[2]};

  CeedPragmaSIMD for (CeedInt n = 0; n < nmodes; n++) {
    xhat[0] = (X[0] - stg_ctx->u0 * t) * Max(2 * kappa[0] / kappa[n], 0.1);
    xdotd   = 0.;
    for (CeedInt i = 0; i < 3; i++) xdotd += d[i * nmodes + n] * xhat[i];
    const CeedScalar cos_kxdp = cos(kappa[n] * xdotd + phi[n]);
    vp[0] += sqrt(qn[n]) * sigma[0 * nmodes + n] * cos_kxdp;
    vp[1] += sqrt(qn[n]) * sigma[1 * nmodes + n] * cos_kxdp;
    vp[2] += sqrt(qn[n]) * sigma[2 * nmodes + n] * cos_kxdp;
  }
  for (CeedInt i = 0; i < 3; i++) vp[i] *= 2 * sqrt(1.5);

  u[0] = ubar[0] + cij[0] * vp[0];
  u[1] = ubar[1] + cij[3] * vp[0] + cij[1] * vp[1];
  u[2] = ubar[2] + cij[4] * vp[0] + cij[5] * vp[1] + cij[2] * vp[2];
}

/******************************************************
 * @brief Calculate u(x,t) for STG inflow condition
 *
 * @param[in]  X          Location to evaluate u(X,t)
 * @param[in]  t          Time to evaluate u(X,t)
 * @param[in]  ubar       Mean velocity at X
 * @param[in]  cij        Cholesky decomposition at X
 * @param[in]  Ektot      Total spectrum energy at this location
 * @param[in]  h_node_sep Element size in 3 directions
 * @param[in]  wall_dist  Distance to closest wall
 * @param[in]  eps        Turbulent dissipation
 * @param[in]  lt         Turbulent length scale
 * @param[out] u          Velocity at X and t
 * @param[in]  stg_ctx    STGShur14Context for the problem
 */
CEED_QFUNCTION_HELPER void StgShur14Calc_PrecompEktot(const CeedScalar X[3], const CeedScalar t, const CeedScalar ubar[3], const CeedScalar cij[6],
                                                      const CeedScalar Ektot, const CeedScalar h_node_sep[3], const CeedScalar wall_dist,
                                                      const CeedScalar eps, const CeedScalar lt, const CeedScalar nu, CeedScalar u[3],
                                                      const StgShur14Context stg_ctx) {
  const CeedInt     nmodes = stg_ctx->nmodes;
  const CeedScalar *kappa  = &stg_ctx->data[stg_ctx->offsets.kappa];
  const CeedScalar *phi    = &stg_ctx->data[stg_ctx->offsets.phi];
  const CeedScalar *sigma  = &stg_ctx->data[stg_ctx->offsets.sigma];
  const CeedScalar *d      = &stg_ctx->data[stg_ctx->offsets.d];
  CeedScalar        hmax, ke, keta, kcut;
  SpectrumConstants(wall_dist, eps, lt, h_node_sep, nu, &hmax, &ke, &keta, &kcut);
  CeedScalar xdotd, vp[3] = {0.};
  CeedScalar xhat[] = {0., X[1], X[2]};

  CeedPragmaSIMD for (CeedInt n = 0; n < nmodes; n++) {
    xhat[0] = (X[0] - stg_ctx->u0 * t) * Max(2 * kappa[0] / kappa[n], 0.1);
    xdotd   = 0.;
    for (CeedInt i = 0; i < 3; i++) xdotd += d[i * nmodes + n] * xhat[i];
    const CeedScalar cos_kxdp = cos(kappa[n] * xdotd + phi[n]);
    const CeedScalar dkappa   = n == 0 ? kappa[0] : kappa[n] - kappa[n - 1];
    const CeedScalar qn       = Calc_qn(kappa[n], dkappa, keta, kcut, ke, Ektot);
    vp[0] += sqrt(qn) * sigma[0 * nmodes + n] * cos_kxdp;
    vp[1] += sqrt(qn) * sigma[1 * nmodes + n] * cos_kxdp;
    vp[2] += sqrt(qn) * sigma[2 * nmodes + n] * cos_kxdp;
  }
  for (CeedInt i = 0; i < 3; i++) vp[i] *= 2 * sqrt(1.5);

  u[0] = ubar[0] + cij[0] * vp[0];
  u[1] = ubar[1] + cij[3] * vp[0] + cij[1] * vp[1];
  u[2] = ubar[2] + cij[4] * vp[0] + cij[5] * vp[1] + cij[2] * vp[2];
}

// Create preprocessed input for the stg calculation
//
// stg_data[0] = 1 / Ektot (inverse of total spectrum energy)
CEED_QFUNCTION(StgShur14Preprocess)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*dXdx_q)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];
  const CeedScalar(*x)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  CeedScalar(*stg_data) = (CeedScalar(*))out[0];

  CeedScalar             ubar[3], cij[6], eps, lt;
  const StgShur14Context stg_ctx = (StgShur14Context)ctx;
  const CeedScalar       dx      = stg_ctx->dx;
  const CeedScalar       mu      = stg_ctx->newtonian_ctx.mu;
  const CeedScalar       theta0  = stg_ctx->theta0;
  const CeedScalar       P0      = stg_ctx->P0;
  const CeedScalar       Rd      = GasConstant(&stg_ctx->newtonian_ctx);
  const CeedScalar       rho     = P0 / (Rd * theta0);
  const CeedScalar       nu      = mu / rho;

  const CeedInt     nmodes = stg_ctx->nmodes;
  const CeedScalar *kappa  = &stg_ctx->data[stg_ctx->offsets.kappa];
  CeedScalar        hmax, ke, keta, kcut;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar wall_dist  = x[1][i];
    const CeedScalar dXdx[2][3] = {
        {dXdx_q[0][0][i], dXdx_q[0][1][i], dXdx_q[0][2][i]},
        {dXdx_q[1][0][i], dXdx_q[1][1][i], dXdx_q[1][2][i]},
    };

    CeedScalar h_node_sep[3];
    h_node_sep[0] = dx;
    for (CeedInt j = 1; j < 3; j++) h_node_sep[j] = 2 / sqrt(dXdx[0][j] * dXdx[0][j] + dXdx[1][j] * dXdx[1][j]);
    ScaleN(h_node_sep, stg_ctx->h_scale_factor, 3);

    InterpolateProfile(wall_dist, ubar, cij, &eps, &lt, stg_ctx);
    SpectrumConstants(wall_dist, eps, lt, h_node_sep, nu, &hmax, &ke, &keta, &kcut);

    // Calculate total TKE per spectrum
    CeedScalar Ek_tot = 0;
    CeedPragmaSIMD for (CeedInt n = 0; n < nmodes; n++) {
      const CeedScalar dkappa = n == 0 ? kappa[0] : kappa[n] - kappa[n - 1];
      Ek_tot += Calc_qn(kappa[n], dkappa, keta, kcut, ke, 1.0);
    }
    // avoid underflowed and poorly defined spectrum coefficients
    stg_data[i] = Ek_tot != 0 ? 1 / Ek_tot : 0;
  }
  return 0;
}

// Extrude the STGInflow profile through out the domain for an initial condition
CEED_QFUNCTION(ICsStg)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*x)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  CeedScalar(*q0)[CEED_Q_VLA]         = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const StgShur14Context         stg_ctx = (StgShur14Context)ctx;
  const NewtonianIdealGasContext gas     = &stg_ctx->newtonian_ctx;
  CeedScalar                     qn[STG_NMODES_MAX], u[3], ubar[3], cij[6], eps, lt;
  const CeedScalar               dx     = stg_ctx->dx;
  const CeedScalar               time   = stg_ctx->time;
  const CeedScalar               theta0 = stg_ctx->theta0;
  const CeedScalar               P0     = stg_ctx->P0;
  const CeedScalar               rho    = P0 / (GasConstant(gas) * theta0);
  const CeedScalar               nu     = gas->mu / rho;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    CeedScalar       dXdx[3][3];
    InvertMappingJacobian_3D(Q, i, J, dXdx, NULL);
    CeedScalar h_node_sep[3];
    h_node_sep[0] = dx;
    for (CeedInt j = 1; j < 3; j++) h_node_sep[j] = 2 / sqrt(Square(dXdx[0][j]) + Square(dXdx[1][j]) + Square(dXdx[2][j]));
    ScaleN(h_node_sep, stg_ctx->h_scale_factor, 3);

    InterpolateProfile(x_i[1], ubar, cij, &eps, &lt, stg_ctx);
    if (stg_ctx->use_fluctuating_IC) {
      CalcSpectrum(x_i[1], eps, lt, h_node_sep, nu, qn, stg_ctx);
      StgShur14Calc(x_i, time, ubar, cij, qn, u, stg_ctx);
    } else {
      for (CeedInt j = 0; j < 3; j++) u[j] = ubar[j];
    }

    CeedScalar Y[5] = {P0, u[0], u[1], u[2], theta0}, q[5] = {0.};
    State      s = StateFromY(gas, Y);
    StateToQ(gas, s, q, gas->state_var);
    for (CeedInt j = 0; j < 5; j++) {
      q0[j][i] = q[j];
    }
  }
  return 0;
}

/********************************************************************
 * @brief QFunction to calculate the inflow boundary condition
 *
 * This will loop through quadrature points, calculate the wavemode amplitudes
 * at each location, then calculate the actual velocity.
 */
CEED_QFUNCTION(StgShur14Inflow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)    = in[2];
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)  = out[1];

  const StgShur14Context stg_ctx = (StgShur14Context)ctx;
  CeedScalar             qn[STG_NMODES_MAX], u[3], ubar[3], cij[6], eps, lt;
  const bool             is_implicit = stg_ctx->is_implicit;
  const bool             mean_only   = stg_ctx->mean_only;
  const bool             prescribe_T = stg_ctx->prescribe_T;
  const CeedScalar       dx          = stg_ctx->dx;
  const CeedScalar       mu          = stg_ctx->newtonian_ctx.mu;
  const CeedScalar       time        = stg_ctx->time;
  const CeedScalar       theta0      = stg_ctx->theta0;
  const CeedScalar       P0          = stg_ctx->P0;
  const CeedScalar       cv          = stg_ctx->newtonian_ctx.cv;
  const CeedScalar       Rd          = GasConstant(&stg_ctx->newtonian_ctx);
  const CeedScalar       gamma       = HeatCapacityRatio(&stg_ctx->newtonian_ctx);

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar rho = prescribe_T ? q[0][i] : P0 / (Rd * theta0);
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar       wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    CeedScalar h_node_sep[3];
    h_node_sep[0] = dx;
    for (CeedInt j = 1; j < 3; j++) h_node_sep[j] = 2 / sqrt(Square(dXdx[0][j]) + Square(dXdx[1][j]));
    ScaleN(h_node_sep, stg_ctx->h_scale_factor, 3);

    InterpolateProfile(X[1][i], ubar, cij, &eps, &lt, stg_ctx);
    if (!mean_only) {
      CalcSpectrum(X[1][i], eps, lt, h_node_sep, mu / rho, qn, stg_ctx);
      StgShur14Calc(x, time, ubar, cij, qn, u, stg_ctx);
    } else {
      for (CeedInt j = 0; j < 3; j++) u[j] = ubar[j];
    }

    const CeedScalar E_kinetic = .5 * rho * Dot3(u, u);
    CeedScalar       E_internal, P;
    if (prescribe_T) {
      // Temperature is being set weakly (theta0) and for constant cv this sets E_internal
      E_internal = rho * cv * theta0;
      // Find pressure using
      P = rho * Rd * theta0;  // interior rho with exterior T
    } else {
      E_internal = q[4][i] - E_kinetic;  // uses prescribed rho and u, E from solution
      P          = E_internal * (gamma - 1.);
    }

    const CeedScalar E = E_internal + E_kinetic;

    // Velocity normal to the boundary
    const CeedScalar u_normal = Dot3(norm, u);

    // The Physics
    // Zero v so all future terms can safely sum into it
    for (CeedInt j = 0; j < 5; j++) v[j][i] = 0.;

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal;

    // -- Momentum
    for (CeedInt j = 0; j < 3; j++) v[j + 1][i] -= wdetJb * (rho * u_normal * u[j] + norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);

    const CeedScalar U[] = {rho, u[0], u[1], u[2], E}, kmstress[6] = {0.};
    StoredValuesPack(Q, i, 0, 5, U, jac_data_sur);
    StoredValuesPack(Q, i, 5, 6, kmstress, jac_data_sur);
  }
  return 0;
}

CEED_QFUNCTION(StgShur14Inflow_Jacobian)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*dq)[CEED_Q_VLA]           = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*jac_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  CeedScalar(*v)[CEED_Q_VLA]                  = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const StgShur14Context stg_ctx  = (StgShur14Context)ctx;
  const bool             implicit = stg_ctx->is_implicit;
  const CeedScalar       cv       = stg_ctx->newtonian_ctx.cv;
  const CeedScalar       Rd       = GasConstant(&stg_ctx->newtonian_ctx);
  const CeedScalar       gamma    = HeatCapacityRatio(&stg_ctx->newtonian_ctx);

  const CeedScalar theta0      = stg_ctx->theta0;
  const bool       prescribe_T = stg_ctx->prescribe_T;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // Calculate inflow values
    CeedScalar velocity[3];
    for (CeedInt j = 0; j < 3; j++) velocity[j] = jac_data_sur[5 + j][i];
    // TODO This is almost certainly a bug. Velocity isn't stored here, only 0s.

    // enabling user to choose between weak T and weak rho inflow
    CeedScalar drho, dE, dP;
    if (prescribe_T) {
      // rho should be from the current solution
      drho                   = dq[0][i];
      CeedScalar dE_internal = drho * cv * theta0;
      CeedScalar dE_kinetic  = .5 * drho * Dot3(velocity, velocity);
      dE                     = dE_internal + dE_kinetic;
      dP                     = drho * Rd * theta0;  // interior rho with exterior T
    } else {                                        // rho specified, E_internal from solution
      drho = 0;
      dE   = dq[4][i];
      dP   = dE * (gamma - 1.);
    }
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};

    const CeedScalar u_normal = Dot3(norm, velocity);

    v[0][i] = -wdetJb * drho * u_normal;
    for (int j = 0; j < 3; j++) v[j + 1][i] = -wdetJb * (drho * u_normal * velocity[j] + norm[j] * dP);
    v[4][i] = -wdetJb * u_normal * (dE + dP);
  }
  return 0;
}

/********************************************************************
 * @brief QFunction to calculate the strongly enforce inflow BC
 *
 * This QF is for the strong application of STG via libCEED (rather than
 * through the native PETSc `DMAddBoundary` -> `bcFunc` method.
 */
CEED_QFUNCTION(StgShur14InflowStrongQF)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*dXdx_q)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];
  const CeedScalar(*coords)[CEED_Q_VLA]    = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*scale)                 = (const CeedScalar(*))in[2];
  const CeedScalar(*inv_Ektotal)           = (const CeedScalar(*))in[3];
  CeedScalar(*bcval)[CEED_Q_VLA]           = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const StgShur14Context         stg_ctx = (StgShur14Context)ctx;
  const NewtonianIdealGasContext gas     = &stg_ctx->newtonian_ctx;
  CeedScalar                     u[3], ubar[3], cij[6], eps, lt;
  const bool                     mean_only = stg_ctx->mean_only;
  const CeedScalar               dx        = stg_ctx->dx;
  const CeedScalar               time      = stg_ctx->time;
  const CeedScalar               theta0    = stg_ctx->theta0;
  const CeedScalar               P0        = stg_ctx->P0;
  const CeedScalar               rho       = P0 / (GasConstant(gas) * theta0);
  const CeedScalar               nu        = gas->mu / rho;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[]        = {coords[0][i], coords[1][i], coords[2][i]};
    const CeedScalar dXdx[2][3] = {
        {dXdx_q[0][0][i], dXdx_q[0][1][i], dXdx_q[0][2][i]},
        {dXdx_q[1][0][i], dXdx_q[1][1][i], dXdx_q[1][2][i]},
    };

    CeedScalar h_node_sep[3];
    h_node_sep[0] = dx;
    for (CeedInt j = 1; j < 3; j++) h_node_sep[j] = 2 / sqrt(Square(dXdx[0][j]) + Square(dXdx[1][j]));
    ScaleN(h_node_sep, stg_ctx->h_scale_factor, 3);

    InterpolateProfile(coords[1][i], ubar, cij, &eps, &lt, stg_ctx);
    if (!mean_only) {
      if (1) {
        StgShur14Calc_PrecompEktot(x, time, ubar, cij, inv_Ektotal[i], h_node_sep, x[1], eps, lt, nu, u, stg_ctx);
      } else {  // Original way
        CeedScalar qn[STG_NMODES_MAX];
        CalcSpectrum(coords[1][i], eps, lt, h_node_sep, nu, qn, stg_ctx);
        StgShur14Calc(x, time, ubar, cij, qn, u, stg_ctx);
      }
    } else {
      for (CeedInt j = 0; j < 3; j++) u[j] = ubar[j];
    }

    CeedScalar Y[5] = {P0, u[0], u[1], u[2], theta0}, q[5] = {0.};
    State      s = StateFromY(gas, Y);
    StateToQ(gas, s, q, gas->state_var);
    switch (gas->state_var) {
      case STATEVAR_CONSERVATIVE:
        q[4] = 0.;  // Don't set energy
        break;
      case STATEVAR_PRIMITIVE:
        q[0] = 0;  // Don't set pressure
        break;
      case STATEVAR_ENTROPY:
        q[0] = 0;  // Don't set V_density
        break;
    }
    for (CeedInt j = 0; j < 5; j++) {
      bcval[j][i] = scale[i] * q[j];
    }
  }
  return 0;
}
