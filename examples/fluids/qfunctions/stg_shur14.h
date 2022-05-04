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
#include "../navierstokes.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

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
void CEED_QFUNCTION_HELPER(InterpolateProfile)(const CeedScalar dw,
    CeedScalar ubar[3], CeedScalar cij[6], CeedScalar *eps, CeedScalar *lt,
    const STGShur14Context stg_ctx) {

  CeedScalar *prof_dw = &stg_ctx->data[stg_ctx->offsets.prof_dw];
  CeedScalar *prof_eps = &stg_ctx->data[stg_ctx->offsets.eps];
  CeedScalar *prof_lt = &stg_ctx->data[stg_ctx->offsets.lt];
  CeedScalar (*prof_ubar)[stg_ctx->nprofs] = (CeedScalar (
        *)[stg_ctx->nprofs]) &stg_ctx->data[stg_ctx->offsets.ubar];
  CeedScalar (*prof_cij)[stg_ctx->nprofs] = (CeedScalar (*)[stg_ctx->nprofs])
      &stg_ctx->data[stg_ctx->offsets.cij];
  CeedInt idx=-1;

  for(CeedInt i=0; i<stg_ctx->nprofs; i++) {
    if (dw < prof_dw[i]) {
      idx = i;
      break;
    }
  }

  if (idx > 0) { // y within the bounds of prof_dw
    CeedScalar coeff = (dw - prof_dw[idx-1]) / (prof_dw[idx] - prof_dw[idx-1]);

    //*INDENT-OFF*
    ubar[0] = prof_ubar[0][idx-1] + coeff*( prof_ubar[0][idx] - prof_ubar[0][idx-1] );
    ubar[1] = prof_ubar[1][idx-1] + coeff*( prof_ubar[1][idx] - prof_ubar[1][idx-1] );
    ubar[2] = prof_ubar[2][idx-1] + coeff*( prof_ubar[2][idx] - prof_ubar[2][idx-1] );
    cij[0]  = prof_cij[0][idx-1]  + coeff*( prof_cij[0][idx]  - prof_cij[0][idx-1] );
    cij[1]  = prof_cij[1][idx-1]  + coeff*( prof_cij[1][idx]  - prof_cij[1][idx-1] );
    cij[2]  = prof_cij[2][idx-1]  + coeff*( prof_cij[2][idx]  - prof_cij[2][idx-1] );
    cij[3]  = prof_cij[3][idx-1]  + coeff*( prof_cij[3][idx]  - prof_cij[3][idx-1] );
    cij[4]  = prof_cij[4][idx-1]  + coeff*( prof_cij[4][idx]  - prof_cij[4][idx-1] );
    cij[5]  = prof_cij[5][idx-1]  + coeff*( prof_cij[5][idx]  - prof_cij[5][idx-1] );
    *eps    = prof_eps[idx-1]     + coeff*( prof_eps[idx]     - prof_eps[idx-1] );
    *lt     = prof_lt[idx-1]      + coeff*( prof_lt[idx]      - prof_lt[idx-1] );
    //*INDENT-ON*
  } else { // y outside bounds of prof_dw
    ubar[0] = prof_ubar[0][stg_ctx->nprofs-1];
    ubar[1] = prof_ubar[1][stg_ctx->nprofs-1];
    ubar[2] = prof_ubar[2][stg_ctx->nprofs-1];
    cij[0]  = prof_cij[0][stg_ctx->nprofs-1];
    cij[1]  = prof_cij[1][stg_ctx->nprofs-1];
    cij[2]  = prof_cij[2][stg_ctx->nprofs-1];
    cij[3]  = prof_cij[3][stg_ctx->nprofs-1];
    cij[4]  = prof_cij[4][stg_ctx->nprofs-1];
    cij[5]  = prof_cij[5][stg_ctx->nprofs-1];
    *eps    = prof_eps[stg_ctx->nprofs-1];
    *lt     = prof_lt[stg_ctx->nprofs-1];
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

  CeedScalar *kappa = &stg_ctx->data[stg_ctx->offsets.kappa];
  CeedScalar ke, fcut, feta, kcut, keta, hmax, Ektot=0.0;

  hmax = PetscMax( PetscMax(h[0], h[1]), h[2]);
  ke   = PetscMax(2*dw, 3*lt);
  keta = 2*M_PI*pow(pow(nu,3.0)/eps, -0.25);
  kcut = M_PI/ PetscMin( PetscMax(PetscMax(h[1], h[2]), 0.3*hmax) + 0.1*dw,
                         hmax );

  for(CeedInt n=0; n<stg_ctx->nmodes; n++) {
    feta = exp(-pow(12*kappa[n]/keta, 2));
    fcut = exp( -pow(4*PetscMax(kappa[n] - 0.9*kcut, 0)/kcut, 3) );
    qn[n] = pow(kappa[n]/ke, 4)*pow(1 + 2.4*pow(kappa[n]/ke,2), -17./6)*feta*fcut;
    Ektot += qn[n];
  }

  for(CeedInt n=0; n<stg_ctx->nmodes; n++) qn[n] /= Ektot;
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
  const CeedScalar (*sigma)[nmodes] = (CeedScalar (*)[nmodes])
                                         &stg_ctx->data[stg_ctx->offsets.sigma];
  const CeedScalar (*d)[nmodes]     = (CeedScalar (*)[nmodes])
                                         &stg_ctx->data[stg_ctx->offsets.d];
  //*INDENT-ON*
  const CeedScalar tworoot1p5 = 2*sqrt(1.5);
  CeedScalar xdotd, vp[3] = {0.};
  CeedScalar xhat[] = {0., X[1], X[2]};

  CeedPragmaSIMD
  for(CeedInt n=0; n<nmodes; n++) {
    xhat[0] = (X[0] - stg_ctx->u0*t)*PetscMax(2*kappa[0]/kappa[n], 0.1);
    xdotd = 0.;
    for(CeedInt i=0; i<3; i++) xdotd += d[i][n]*xhat[i];
    vp[0] += tworoot1p5*sqrt(qn[n])*sigma[0][n] * cos(kappa[n]*xdotd + phi[n]);
    vp[1] += tworoot1p5*sqrt(qn[n])*sigma[1][n] * cos(kappa[n]*xdotd + phi[n]);
    vp[2] += tworoot1p5*sqrt(qn[n])*sigma[2][n] * cos(kappa[n]*xdotd + phi[n]);
  }

  u[0] = ubar[0] + cij[0]*vp[0];
  u[1] = ubar[1] + cij[3]*vp[0] + cij[1]*vp[1];
  u[2] = ubar[2] + cij[4]*vp[0] + cij[5]*vp[1] + cij[2]*vp[2];
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
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA]) in[1],
                   (*X)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA]) in[2];

   CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA]) out[0];

  //*INDENT-ON*

  const STGShur14Context stg_ctx = (STGShur14Context) ctx;
  CeedScalar qn[stg_ctx->nmodes], u[3], ubar[3], cij[6], eps, lt;
  const bool implicit     = stg_ctx->implicit;
  const bool mean_only    = stg_ctx->mean_only;
  const CeedScalar dx     = stg_ctx->dx;
  const CeedScalar mu     = stg_ctx->newtonian_ctx.mu;
  const CeedScalar time   = stg_ctx->time;
  const CeedScalar theta0 = stg_ctx->theta0;
  const CeedScalar cv     = stg_ctx->newtonian_ctx.cv;
  const CeedScalar cp     = stg_ctx->newtonian_ctx.cp;
  const CeedScalar Rd     = cp - cv;

  CeedPragmaSIMD
  for(CeedInt i=0; i<Q; i++) {
    const CeedScalar rho = q[0][i];
    const CeedScalar nu  = mu / rho;
    const CeedScalar x[] = { X[0][i], X[1][i], X[2][i] };
    const CeedScalar dXdx[2][3] = {
      {q_data_sur[4][i], q_data_sur[5][i], q_data_sur[6][i]},
      {q_data_sur[7][i], q_data_sur[8][i], q_data_sur[9][i]}
    };

    const CeedScalar P = rho * Rd * theta0;

    CeedScalar h[3];
    for(CeedInt j=0; j<3; j++)
      h[j] = 2/sqrt(dXdx[0][j]*dXdx[0][j] + dXdx[1][j]*dXdx[1][j]);
    h[0] = dx;

    InterpolateProfile(X[1][i], ubar, cij, &eps, &lt, stg_ctx);
    if (!mean_only) {
      CalcSpectrum(X[1][i], eps, lt, h, nu, qn, stg_ctx);
      STGShur14_Calc(x, time, ubar, cij, qn, u, stg_ctx);
    } else {
      for (int j=0; j<3; j++) u[j] = ubar[j];
    }

    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];
    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    const CeedScalar E_kinetic = .5 * rho * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    const CeedScalar E = rho * cv * theta0 + E_kinetic;

    // Velocity normal to the boundary
    const CeedScalar u_normal = norm[0]*u[0] +
                                norm[1]*u[1] +
                                norm[2]*u[2];
    // The Physics
    // Zero v so all future terms can safely sum into it
    for (int j=0; j<5; j++) v[j][i] = 0.;

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal;

    // -- Momentum
    for (int j=0; j<3; j++)
      v[j+1][i] -= wdetJb *(rho * u_normal * u[j] +
                            norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);
  }
  return 0;
}


#endif // stg_shur14_h
