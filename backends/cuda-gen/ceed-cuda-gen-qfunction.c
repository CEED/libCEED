// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "ceed-cuda-gen.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Cuda_gen(CeedQFunction qf, CeedInt Q,
                                       CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement QFunctionApply");
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Cuda_gen(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Cuda_gen *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);
  ierr = cudaFree(data->d_c); CeedChk_Cu(ceed, ierr);
  ierr = CeedFree(&data->qFunctionSource); CeedChkBackend(ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Cuda_gen *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedQFunctionSetData(qf, data); CeedChkBackend(ierr);

  // Read QFunction source
  ierr = CeedQFunctionGetKernelName(qf, &data->qFunctionName);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source -----\n");
  ierr = CeedQFunctionLoadSourceToBuffer(qf, &data->qFunctionSource);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source Complete! -----\n");
  if (!data->qFunctionSource)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "/gpu/cuda/gen backend requires QFunction source code file");
  // LCOV_EXCL_STOP

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Cuda_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Cuda_gen); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
