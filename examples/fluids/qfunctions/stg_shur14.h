// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Implementation of the Synthetic Turbulence Generation (STG) algorithm
/// presented in Shur et al. 2014
//
/// SetupSTG_Rand reads in the input files and fills in STGShur14Context. Then
/// STGShur14_CalcQF is run over quadrature points. Before the program exits,
/// TearDownSTG is run to free the memory of the allocated arrays.

#ifndef stg_shur14_h
#define stg_shur14_h

#include <math.h>
#include <ceed.h>
#include <stdlib.h>
#include "stg_shur14_type.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#define STG_NMODES_MAX 1024

CEED_QFUNCTION_HELPER CeedScalar Max(CeedScalar a, CeedScalar b) { return a < b ? b : a; }
CEED_QFUNCTION_HELPER CeedScalar Min(CeedScalar a, CeedScalar b) { return a < b ? a : b; }

/*
 * @brief Interpolate quantities from input profile to given location
 *
 * Assumed that prof_dw[i+1] > prof_dw[i] and prof_dw[0] = 0
 * If dw > prof_dw[-1], then the interpolation takes the values at prof_dw[-1]
 *
 * @param[in]  dw      Distance to the nearest wall
 * @param[out] ubar    Mean velocity at dw
 * @param[out] cij     Cholesky decomposition at dw
 * @param[out] eps     Turbulent dissipation at dw
 * @param[out] lt      Turbulent length scale at dw
 * @param[in]  stg_ctx STGShur14Context for the problem
 */
CEED_QFUNCTION_HELPER void InterpolateProfile(const CeedScalar dw,
    CeedScalar ubar[3], CeedScalar cij[6], CeedScalar *eps, CeedScalar *lt,
    const STGShur14Context stg_ctx) {

  const CeedInt    nprofs    = stg_ctx->nprofs;
  const CeedScalar *prof_dw  = &stg_ctx->data[stg_ctx->offsets.prof_dw];
  const CeedScalar *prof_eps = &stg_ctx->data[stg_ctx->offsets.eps];
  const CeedScalar *prof_lt  = &stg_ctx->data[stg_ctx->offsets.lt];
  const CeedScalar *prof_ubar = &stg_ctx->data[stg_ctx->offsets.ubar];
  const CeedScalar *prof_cij  = &stg_ctx->data[stg_ctx->offsets.cij];
  CeedInt idx=-1;

  for(CeedInt i=0; i<nprofs; i++) {
    if (dw < prof_dw[i]) {
      idx = i;
      break;
    }
  }

  if (idx > 0) { // y within the bounds of prof_dw
    CeedScalar coeff = (dw - prof_dw[idx-1]) / (prof_dw[idx] - prof_dw[idx-1]);

    //*INDENT-OFF*
    ubar[0] = prof_ubar[0*nprofs+idx-1] + coeff*( prof_ubar[0*nprofs+idx] - prof_ubar[0*nprofs+idx-1] );
    ubar[1] = prof_ubar[1*nprofs+idx-1] + coeff*( prof_ubar[1*nprofs+idx] - prof_ubar[1*nprofs+idx-1] );
    ubar[2] = prof_ubar[2*nprofs+idx-1] + coeff*( prof_ubar[2*nprofs+idx] - prof_ubar[2*nprofs+idx-1] );
    cij[0]  = prof_cij[0*nprofs+idx-1]  + coeff*( prof_cij[0*nprofs+idx]  - prof_cij[0*nprofs+idx-1] );
    cij[1]  = prof_cij[1*nprofs+idx-1]  + coeff*( prof_cij[1*nprofs+idx]  - prof_cij[1*nprofs+idx-1] );
    cij[2]  = prof_cij[2*nprofs+idx-1]  + coeff*( prof_cij[2*nprofs+idx]  - prof_cij[2*nprofs+idx-1] );
    cij[3]  = prof_cij[3*nprofs+idx-1]  + coeff*( prof_cij[3*nprofs+idx]  - prof_cij[3*nprofs+idx-1] );
    cij[4]  = prof_cij[4*nprofs+idx-1]  + coeff*( prof_cij[4*nprofs+idx]  - prof_cij[4*nprofs+idx-1] );
    cij[5]  = prof_cij[5*nprofs+idx-1]  + coeff*( prof_cij[5*nprofs+idx]  - prof_cij[5*nprofs+idx-1] );
    *eps    = prof_eps[idx-1]           + coeff*( prof_eps[idx]           - prof_eps[idx-1] );
    *lt     = prof_lt[idx-1]            + coeff*( prof_lt[idx]            - prof_lt[idx-1] );
    //*INDENT-ON*
  } else { // y outside bounds of prof_dw
    ubar[0] = prof_ubar[1*nprofs-1];
    ubar[1] = prof_ubar[2*nprofs-1];
    ubar[2] = prof_ubar[3*nprofs-1];
    cij[0]  = prof_cij[1*nprofs-1];
    cij[1]  = prof_cij[2*nprofs-1];
    cij[2]  = prof_cij[3*nprofs-1];
    cij[3]  = prof_cij[4*nprofs-1];
    cij[4]  = prof_cij[5*nprofs-1];
    cij[5]  = prof_cij[6*nprofs-1];
    *eps    = prof_eps[nprofs-1];
    *lt     = prof_lt[nprofs-1];
  }
}

/*
 * @brief Calculate spectrum coefficients for STG
 *
 * Calculates q_n at a given distance to the wall
 *
 * @param[in]  dw      Distance to the nearest wall
 * @param[in]  eps     Turbulent dissipation w/rt dw
 * @param[in]  lt      Turbulent length scale w/rt dw
 * @param[in]  h       Element lengths in coordinate directions
 * @param[in]  nu      Dynamic Viscosity;
 * @param[in]  stg_ctx STGShur14Context for the problem
 * @param[out] qn      Spectrum coefficients, [nmodes]
 */
void CEED_QFUNCTION_HELPER(CalcSpectrum)(const CeedScalar dw,
    const CeedScalar eps, const CeedScalar lt, const CeedScalar h[3],
    const CeedScalar nu, CeedScalar qn[], const STGShur14Context stg_ctx) {

  const CeedInt    nmodes = stg_ctx->nmodes;
  const CeedScalar *kappa = &stg_ctx->data[stg_ctx->offsets.kappa];

  const CeedScalar hmax = Max( Max(h[0], h[1]), h[2]);
  const CeedScalar ke   = dw==0 ? 1e16 : 2*M_PI/Min(2*dw, 3*lt);
  const CeedScalar keta = 2*M_PI*pow(pow(nu,3.0)/eps, -0.25);
  const CeedScalar kcut =
    M_PI/ Min( Max(Max(h[1], h[2]), 0.3*hmax) + 0.1*dw, hmax );
  CeedScalar fcut, feta, Ektot=0.0;

  for(CeedInt n=0; n<nmodes; n++) {
    feta   = exp(-Square(12*kappa[n]/keta));
    fcut   = exp( -pow(4*Max(kappa[n] - 0.9*kcut, 0)/kcut, 3.) );
    qn[n]  = pow(kappa[n]/ke, 4.)
             * pow(1 + 2.4*Square(kappa[n]/ke),-17./6)*feta*fcut;
    qn[n] *= n==0 ? kappa[0] : kappa[n] - kappa[n-1];
    Ektot += qn[n];
  }

  if (Ektot == 0) return;
  for(CeedInt n=0; n<nmodes; n++) qn[n] /= Ektot;
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
void CEED_QFUNCTION_HELPER(STGShur14_Calc)(const CeedScalar X[3],
    const CeedScalar t, const CeedScalar ubar[3], const CeedScalar cij[6],
    const CeedScalar qn[], CeedScalar u[3],
    const STGShur14Context stg_ctx) {

  //*INDENT-OFF*
  const CeedInt    nmodes = stg_ctx->nmodes;
  const CeedScalar *kappa = &stg_ctx->data[stg_ctx->offsets.kappa];
  const CeedScalar *phi   = &stg_ctx->data[stg_ctx->offsets.phi];
  const CeedScalar *sigma = &stg_ctx->data[stg_ctx->offsets.sigma];
  const CeedScalar *d     = &stg_ctx->data[stg_ctx->offsets.d];
  //*INDENT-ON*
  CeedScalar xdotd, vp[3] = {0.};
  CeedScalar xhat[] = {0., X[1], X[2]};

  CeedPragmaSIMD
  for(CeedInt n=0; n<nmodes; n++) {
    xhat[0] = (X[0] - stg_ctx->u0*t)*Max(2*kappa[0]/kappa[n], 0.1);
    xdotd = 0.;
    for(CeedInt i=0; i<3; i++) xdotd += d[i*nmodes+n]*xhat[i];
    const CeedScalar cos_kxdp = cos(kappa[n]*xdotd + phi[n]);
    vp[0] += sqrt(qn[n])*sigma[0*nmodes+n] * cos_kxdp;
    vp[1] += sqrt(qn[n])*sigma[1*nmodes+n] * cos_kxdp;
    vp[2] += sqrt(qn[n])*sigma[2*nmodes+n] * cos_kxdp;
  }
  for(CeedInt i=0; i<3; i++) vp[i] *= 2*sqrt(1.5);

  u[0] = ubar[0] + cij[0]*vp[0];
  u[1] = ubar[1] + cij[3]*vp[0] + cij[1]*vp[1];
  u[2] = ubar[2] + cij[4]*vp[0] + cij[5]*vp[1] + cij[2]*vp[2];
}

// Extrude the STGInflow profile through out the domain for an initial condition
CEED_QFUNCTION(ICsSTG)(void *ctx, CeedInt Q,
                       const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const STGShur14Context stg_ctx = (STGShur14Context) ctx;
  CeedScalar u[3], cij[6], eps, lt;
  const CeedScalar theta0 = stg_ctx->theta0;
  const CeedScalar P0     = stg_ctx->P0;
  const CeedScalar cv     = stg_ctx->newtonian_ctx.cv;
  const CeedScalar cp     = stg_ctx->newtonian_ctx.cp;
  const CeedScalar Rd     = cp - cv;
  const CeedScalar rho = P0 / (Rd * theta0);

  CeedPragmaSIMD
  for(CeedInt i=0; i<Q; i++) {
    InterpolateProfile(X[1][i], u, cij, &eps, &lt, stg_ctx);

    q0[0][i] = rho;
    q0[1][i] = u[0] * rho;
    q0[2][i] = u[1] * rho;
    q0[3][i] = u[2] * rho;
    q0[4][i] = rho * (0.5 * Dot3(u, u) + cv * theta0);
  } // End of Quadrature Point Loop
  return 0;
}

/********************************************************************
 * @brief QFunction to calculate the inflow boundary condition
 *
 * This will loop through quadrature points, calculate the wavemode amplitudes
 * at each location, then calculate the actual velocity.
 */
CEED_QFUNCTION(STGShur14_Inflow)(void *ctx, CeedInt Q,
                                 const CeedScalar *const *in,
                                 CeedScalar *const *out) {

  //*INDENT-OFF*
  const CeedScalar (*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA]) in[0],
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA]) in[2],
                   (*X)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA]) in[3];

   CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA]) out[0];

  //*INDENT-ON*

  const STGShur14Context stg_ctx = (STGShur14Context) ctx;
  CeedScalar qn[STG_NMODES_MAX], u[3], ubar[3], cij[6], eps, lt;
  const bool is_implicit  = stg_ctx->is_implicit;
  const bool mean_only    = stg_ctx->mean_only;
  const bool prescribe_T  = stg_ctx->prescribe_T;
  const CeedScalar dx     = stg_ctx->dx;
  const CeedScalar mu     = stg_ctx->newtonian_ctx.mu;
  const CeedScalar time   = stg_ctx->time;
  const CeedScalar theta0 = stg_ctx->theta0;
  const CeedScalar P0     = stg_ctx->P0;
  const CeedScalar cv     = stg_ctx->newtonian_ctx.cv;
  const CeedScalar cp     = stg_ctx->newtonian_ctx.cp;
  const CeedScalar Rd     = cp - cv;
  const CeedScalar gamma  = cp/cv;

  CeedPragmaSIMD
  for(CeedInt i=0; i<Q; i++) {
    const CeedScalar rho = prescribe_T ? q[0][i] : P0 / (Rd * theta0);
    const CeedScalar x[] = { X[0][i], X[1][i], X[2][i] };
    const CeedScalar dXdx[2][3] = {
      {q_data_sur[4][i], q_data_sur[5][i], q_data_sur[6][i]},
      {q_data_sur[7][i], q_data_sur[8][i], q_data_sur[9][i]}
    };

    CeedScalar h[3];
    for (CeedInt j=0; j<3; j++)
      h[j] = 2/sqrt(dXdx[0][j]*dXdx[0][j] + dXdx[1][j]*dXdx[1][j]);
    h[0] = dx;

    InterpolateProfile(X[1][i], ubar, cij, &eps, &lt, stg_ctx);
    if (!mean_only) {
      CalcSpectrum(X[1][i], eps, lt, h, mu/rho, qn, stg_ctx);
      STGShur14_Calc(x, time, ubar, cij, qn, u, stg_ctx);
    } else {
      for (CeedInt j=0; j<3; j++) u[j] = ubar[j];
    }

    const CeedScalar E_kinetic = .5 * rho * (u[0]*u[0] +
                                 u[1]*u[1] +
                                 u[2]*u[2]);
    CeedScalar E_internal, P;
    if (prescribe_T) {
      // Temperature is being set weakly (theta0) and for constant cv this sets E_internal
      E_internal = rho * cv * theta0;
      // Find pressure using
      P = rho * Rd * theta0; // interior rho with exterior T
    } else {
      E_internal = q[4][i] - E_kinetic; // uses prescribed rho and u, E from solution
      P = E_internal * (gamma - 1.);
    }

    const CeedScalar wdetJb  = (is_implicit ? -1. : 1.) * q_data_sur[0][i];
    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    const CeedScalar E = E_internal + E_kinetic;

    // Velocity normal to the boundary
    const CeedScalar u_normal = norm[0]*u[0] +
                                norm[1]*u[1] +
                                norm[2]*u[2];
    // The Physics
    // Zero v so all future terms can safely sum into it
    for (CeedInt j=0; j<5; j++) v[j][i] = 0.;

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal;

    // -- Momentum
    for (CeedInt j=0; j<3; j++)
      v[j+1][i] -= wdetJb *(rho * u_normal * u[j] +
                            norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);
  }
  return 0;
}

#endif // stg_shur14_h
