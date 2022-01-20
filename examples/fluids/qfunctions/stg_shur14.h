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

#ifndef stg_shur14_struct
#define stg_shur14_struct
typedef struct STGShur14Context_ *STGShur14Context;
struct STGShur14Context_ {
  CeedInt nmodes;              // !< Number of wavemodes
  CeedScalar *sigma, *d, *phi; // !< Random number set, [nmodes,3], [nmodes,3], [nmodes]
  CeedScalar *kappa;           // !< Wavemode frequencies, [nmodes]
  CeedScalar alpha;            // !< Geometric growth rate of kappa

  CeedInt nprof;       // !< Number of profile points in STGInflow.dat
  CeedScalar *prof_dw; // !< Distance to wall for Inflow Profie, [nprof]
  CeedScalar *ubar;    // !< Mean velocity, [nprof, 3]
  CeedScalar *cij;     // !< Cholesky decomposition [nprof, 6]
  CeedScalar *eps;     // !< Turbulent Disspation [nprof, 6]
  CeedScalar *lt;      // !< Tubulent Length Scale [nprof, 6]
};
#endif

int SetupSTG_Rand(STGShur14Context stg_ctx){

  //TODO will probably want to have the paths to the STGRand.dat and
  // STGInflow.dat as inputs for this function.

  stg_ctx->alpha = 1.01; // Get from CLI, yaml/toml, etc.

  //TODO Read STGRand.dat
  stg_ctx->nmodes = 2;
  stg_ctx->sigma = malloc(sizeof(CeedScalar)*stg_ctx->nmodes*3);
  stg_ctx->d     = malloc(sizeof(CeedScalar)*stg_ctx->nmodes*3);
  stg_ctx->phi   = malloc(sizeof(CeedScalar)*stg_ctx->nmodes);

  //TODO Set sigma, d, and phi from STGRand.dat

  //TODO Read STGInflow.dat
  stg_ctx->nprof = 5; //TODO Set from STGInflow.dat
  stg_ctx->prof_dw = malloc(sizeof(CeedScalar)*stg_ctx->nprof);
  stg_ctx->ubar    = malloc(sizeof(CeedScalar)*stg_ctx->nprof*3);
  stg_ctx->cij     = malloc(sizeof(CeedScalar)*stg_ctx->nprof*6);
  stg_ctx->eps     = malloc(sizeof(CeedScalar)*stg_ctx->nprof);
  stg_ctx->lt      = malloc(sizeof(CeedScalar)*stg_ctx->nprof);

  //TODO Set allocated arrays above to STGInflow.dat values


  CeedScalar kmin = 5; //TODO Calc kmin just based on prof_dw and lt

  // Calculate wavemode frequencies, kappa
  stg_ctx->kappa = malloc(sizeof(CeedScalar)*stg_ctx->nmodes);
  CeedPragmaSIMD
  for(int i=0; i<stg_ctx->nmodes; i++){
    stg_ctx->kappa[i] = kmin*pow(stg_ctx->alpha, i);
  }

  return 1;
}


/******************************************************
 * @brief Calculate u(x,t) for STG inflow condition
 *
 * @param[in] x Location to evaluate u(x,t), [3]
 * @param[in] t Time to evaluate u(x,t)
 * @param[in] qn Wavemode amplitudes at x, [nmodes]
 * @param[out] u Velocity at x and t
 */
void CEED_QFUNCTION_HELPER(STGShur14_Calc)(const CeedScalar x[3],
    const CeedScalar t, const CeedScalar *qn, CeedScalar *u[3], const STGShur14Context *stg_ctx){
  for (CeedInt i=0; i<3; i++){
    *u[i] = (CeedScalar)i;
  }
}

/********************************************************************
 * @brief QFunction to calculate the inflow boundary condition
 *
 * This will loop through quadrature points, calculate the wavemode amplitudes
 * at each location, then calculate the actual velocity.
 */
CEED_QFUNCTION(STGShur14_CalcQF)(void *ctx, CeedInt Q,
                                const CeedScalar *const *in,
                                CeedScalar *const *out){
  // Calculate qn on the fly
  // Use STGShur14_Calc to actually calculate u
  return 1;
}


int TearDownSTG(STGShur14Context stg_ctx){
  free(stg_ctx->sigma);
  free(stg_ctx->d);
  free(stg_ctx->phi);
  free(stg_ctx->kappa);
  free(stg_ctx->prof_dw);
  free(stg_ctx->ubar);
  free(stg_ctx->cij);
  free(stg_ctx->eps);
  free(stg_ctx->lt);

  return 1;
}

#endif // stg_shur14_h
