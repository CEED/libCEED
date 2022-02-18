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
