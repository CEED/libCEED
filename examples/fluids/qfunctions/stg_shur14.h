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
 * @param[in] x Location to evaluate u(x,t), [3]
 * @param[in] t Time to evaluate u(x,t)
 * @param[in] qn Wavemode amplitudes at x, [nmodes]
 * @param[out] u Velocity at x and t
 */
void CEED_QFUNCTION_HELPER(STGShur14_Calc)(const CeedScalar x[3],
    const CeedScalar t, const CeedScalar qn, CeedScalar *u[3], const STGShur14Context stg_ctx){
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


#endif // stg_shur14_h
