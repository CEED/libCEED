// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>

#include "../hip/ceed-hip-common.h"
#include "ceed-hip-gen.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Hip_gen(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  return CeedError(CeedQFunctionReturnCeed(qf), CEED_ERROR_BACKEND, "Backend does not implement QFunctionApply");
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Hip_gen(CeedQFunction qf) {
  CeedQFunction_Hip_gen *data;

  CeedCallBackend(CeedQFunctionGetData(qf, &data));
  CeedCallHip(CeedQFunctionReturnCeed(qf), hipFree(data->d_c));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Hip_gen(CeedQFunction qf) {
  Ceed                   ceed;
  CeedQFunction_Hip_gen *data;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedQFunctionSetData(qf, data));

  CeedCallBackend(CeedQFunctionGetKernelName(qf, &data->qfunction_name));

  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Hip_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Hip_gen));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
