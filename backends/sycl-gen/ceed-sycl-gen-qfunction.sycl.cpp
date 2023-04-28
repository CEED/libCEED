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
  Ceed ceed;
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedQFunction_Sycl_gen *impl;
  CeedCallBackend(CeedQFunctionGetData(qf, &impl));
  Ceed_Sycl *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());
  CeedCallSycl(ceed, sycl::free(impl->d_c, data->sycl_context));
  
  CeedCallBackend(CeedFree(&impl->q_function_source));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Sycl_gen(CeedQFunction qf) {
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Sycl_gen *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedQFunctionSetData(qf, impl));

  // Read QFunction source
  CeedCallBackend(CeedQFunctionGetKernelName(qf, &impl->q_function_name));
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source -----\n");
  CeedCallBackend(CeedQFunctionLoadSourceToBuffer(qf, &impl->q_function_source));
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source Complete! -----\n");
  if (!impl->q_function_source) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "/gpu/sycl/gen backend requires QFunction source code file");
    // LCOV_EXCL_STOP
  }

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Sycl_gen));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Sycl_gen));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
