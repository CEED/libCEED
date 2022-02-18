// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

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
#include "../src/setupstg_shur14.h"

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
    if (dw > prof_dw[i]) {
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
    ubar[0] = prof_ubar[0][stg_ctx->nprofs];
    ubar[1] = prof_ubar[1][stg_ctx->nprofs];
    ubar[2] = prof_ubar[2][stg_ctx->nprofs];
    cij[0]  = prof_cij[0][stg_ctx->nprofs];
    cij[1]  = prof_cij[1][stg_ctx->nprofs];
    cij[2]  = prof_cij[2][stg_ctx->nprofs];
    cij[3]  = prof_cij[3][stg_ctx->nprofs];
    cij[4]  = prof_cij[4][stg_ctx->nprofs];
    cij[5]  = prof_cij[5][stg_ctx->nprofs];
    *eps    = prof_eps[stg_ctx->nprofs];
    *lt     = prof_lt[stg_ctx->nprofs];
  }
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
  CeedScalar *kappa = &stg_ctx->data[stg_ctx->offsets.kappa];
  CeedScalar *phi   = &stg_ctx->data[stg_ctx->offsets.phi];
  CeedScalar (*sigma)[stg_ctx->nprofs] = (CeedScalar (*)[stg_ctx->nprofs])
                                         &stg_ctx->data[stg_ctx->offsets.sigma];
  CeedScalar (*d)[stg_ctx->nprofs]     = (CeedScalar (*)[stg_ctx->nprofs])
                                         &stg_ctx->data[stg_ctx->offsets.d];
  //*INDENT-ON*
  CeedScalar xdotd, vp[] = {0., 0., 0.};
  CeedScalar xhat[] = {0., X[1], X[2]};

  CeedPragmaSIMD
  for(CeedInt n=0; n<stg_ctx->nmodes; n++) {
    xhat[0] = (X[0] - stg_ctx->u0*t)*PetscMax(2*kappa[0]/kappa[n], 0.1);
    xdotd = 0.;
    for(CeedInt i=0; i<3; i++) xdotd += d[i][n]*xhat[i];
    vp[0] += qn[n]*sigma[0][n]*cos(kappa[n]*xdotd + phi[n]);
  }

  u[0] = ubar[0] + cij[0]*vp[0] + cij[3]*vp[0] + cij[4]*vp[0];
  u[1] = ubar[1] + cij[3]*vp[1] + cij[1]*vp[1] + cij[5]*vp[1];
  u[2] = ubar[2] + cij[4]*vp[2] + cij[5]*vp[2] + cij[2]*vp[2];
}

/********************************************************************
 * @brief QFunction to calculate the inflow boundary condition
 *
 * This will loop through quadrature points, calculate the wavemode amplitudes
 * at each location, then calculate the actual velocity.
 */
CEED_QFUNCTION(STGShur14_CalcQF)(void *ctx, CeedInt Q,
                                 const CeedScalar *const *in,
                                 CeedScalar *const *out) {
  // Calculate qn on the fly
  // Use STGShur14_Calc to actually calculate u
  return 0;
}


#endif // stg_shur14_h
