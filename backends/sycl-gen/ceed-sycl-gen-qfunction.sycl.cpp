// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <sycl/sycl.hpp>
#include <stdio.h>
#include <string.h>

#include "ceed-sycl-gen.hpp"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Sycl_gen(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement QFunctionApply");
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Sycl_gen(CeedQFunction qf) {
  // CeedQFunction_Sycl_gen *data;
  // CeedCallBackend(CeedQFunctionGetData(qf, &data));
  Ceed ceed;
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  // CeedCallSycl(ceed, syclFree(data->d_c));
  // CeedCallBackend(CeedFree(&data->q_function_source));
  // CeedCallBackend(CeedFree(&data));
  // return CEED_ERROR_SUCCESS;
  return CeedError(ceed, CEED_ERROR_BACKEND, "CeedQFunctionDestroy_Sycl_gen not implemented");
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Sycl_gen(CeedQFunction qf) {
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Sycl_gen *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedQFunctionSetData(qf, data));

  // Read QFunction source
  // CeedCallBackend(CeedQFunctionGetKernelName(qf, &data->q_function_name));
  // CeedDebug256(ceed, 2, "----- Loading QFunction User Source -----\n");
  // CeedCallBackend(CeedQFunctionLoadSourceToBuffer(qf, &data->q_function_source));
  // CeedDebug256(ceed, 2, "----- Loading QFunction User Source Complete! -----\n");
  // if (!data->q_function_source) {
  //   // LCOV_EXCL_START
  //   return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "/gpu/sycl/gen backend requires QFunction source code file");
  //   // LCOV_EXCL_STOP
  // }

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Sycl_gen));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Sycl_gen));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
