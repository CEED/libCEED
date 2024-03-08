// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <cuda_runtime.h>

#include "../cuda/ceed-cuda-common.h"
#include "ceed-cuda-gen.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Cuda_gen(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  return CeedError(CeedQFunctionReturnCeed(qf), CEED_ERROR_BACKEND, "Backend does not implement QFunctionApply");
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Cuda_gen(CeedQFunction qf) {
  CeedQFunction_Cuda_gen *data;

  CeedCallBackend(CeedQFunctionGetData(qf, &data));
  CeedCallCuda(CeedQFunctionReturnCeed(qf), cudaFree(data->d_c));
  CeedCallBackend(CeedFree(&data->qfunction_source));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf) {
  Ceed                    ceed;
  CeedQFunction_Cuda_gen *data;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedQFunctionSetData(qf, data));

  // Read QFunction source
  CeedCallBackend(CeedQFunctionGetKernelName(qf, &data->qfunction_name));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading QFunction User Source -----\n");
  CeedCallBackend(CeedQFunctionLoadSourceToBuffer(qf, &data->qfunction_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading QFunction User Source Complete! -----\n");
  CeedCheck(data->qfunction_source, ceed, CEED_ERROR_UNSUPPORTED, "/gpu/cuda/gen backend requires QFunction source code file");

  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Cuda_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Cuda_gen));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
