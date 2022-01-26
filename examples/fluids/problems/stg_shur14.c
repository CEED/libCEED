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

#include <stdlib.h>
#include <math.h>
#include "../qfunctions/stg_shur14.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#ifndef NOMINMAX

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#endif  /* NOMINMAX */

int SetupSTG_Rand(STGShur14Context stg_ctx){

  //TODO will probably want to have the paths to the STGRand.dat and
  // STGInflow.dat as inputs for this function.


  //TODO Read STGRand.dat to get nmodes
  int nmodes = 2;
  //TODO Read STGInflow.dat to get nprof
  int nprofs = 5;

  int total_num_scalars = nmodes*(3 + 3 + 1) + nprofs*(1 + 3 + 6 + 1 + 1) ;
  stg_ctx = malloc(sizeof(STGShur14Context) + total_num_scalars*sizeof(CeedScalar) );

  stg_ctx->nmodes = nmodes;
  stg_ctx->nprofs = nprofs;

  stg_ctx->offsets.sigma = 0;
  stg_ctx->offsets.d       = nmodes*3;
  stg_ctx->offsets.phi     = stg_ctx->offsets.d       + nmodes*3;
  stg_ctx->offsets.kappa   = stg_ctx->offsets.phi     + nmodes;
  stg_ctx->offsets.prof_dw = stg_ctx->offsets.kappa   + nmodes;
  stg_ctx->offsets.ubar    = stg_ctx->offsets.prof_dw + nprofs;
  stg_ctx->offsets.cij     = stg_ctx->offsets.ubar    + nprofs*3;
  stg_ctx->offsets.eps     = stg_ctx->offsets.ubar    + nprofs*6;
  stg_ctx->offsets.lt      = stg_ctx->offsets.eps     + nprofs;

  //TODO Set sigma, d, and phi from STGRand.dat
  CeedScalar (*sigma)[nmodes] = (CeedScalar (*)[nmodes])&stg_ctx->data[stg_ctx->offsets.sigma];
  // or just pass &stg_ctx->data[stg_ctx->offsets.sigma]; to function that reads it in

  //TODO Read rest of STGInflow.dat and assign to data

  stg_ctx->alpha = 1.01; //TODO Get from CLI, yaml/toml, etc.

  // Calculate kappa
  {
    CeedScalar *kappa = &stg_ctx->data[stg_ctx->offsets.kappa];
    CeedScalar *prof_dw = &stg_ctx->data[stg_ctx->offsets.prof_dw];
    CeedScalar *lt = &stg_ctx->data[stg_ctx->offsets.lt];
    CeedScalar le, le_max=0;

    CeedPragmaSIMD
    for(int i=0; i<stg_ctx->nprofs; i++){
      le = max(2*prof_dw[i], 3*lt[i]); //TODO safe guard against negative prof_dw or lt?
      if (le_max < le){
        le_max = le;
      }
    }
    CeedScalar kmin = M_PI/le_max;

    CeedPragmaSIMD
    for(int i=0; i<stg_ctx->nmodes; i++){
      kappa[i] = kmin*pow(stg_ctx->alpha, i);
    }
  } //end calculate kappa

  return 1;
}

int TearDownSTG(STGShur14Context stg_ctx){

  free(stg_ctx);
  return 1;
}
