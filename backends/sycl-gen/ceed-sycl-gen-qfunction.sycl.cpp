// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  return CeedError(CeedQFunctionReturnCeed(qf), CEED_ERROR_BACKEND, "Backend does not implement QFunctionApply");
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Sycl_gen(CeedQFunction qf) {
  Ceed                    ceed;
  Ceed_Sycl              *data;
  CeedQFunction_Sycl_gen *impl;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedQFunctionGetData(qf, &impl));
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());
  CeedCallSycl(ceed, sycl::free(impl->d_c, data->sycl_context));

  CeedCallBackend(CeedFree(&impl->qfunction_source));
  CeedCallBackend(CeedFree(&impl));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Sycl_gen(CeedQFunction qf) {
  Ceed                    ceed;
  CeedQFunction_Sycl_gen *impl;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedQFunctionSetData(qf, impl));

  // Read QFunction source
  CeedCallBackend(CeedQFunctionGetKernelName(qf, &impl->qfunction_name));
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source -----\n");
  CeedCallBackend(CeedQFunctionLoadSourceToBuffer(qf, &impl->qfunction_source));
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source Complete! -----\n");
  CeedCheck(impl->qfunction_source, ceed, CEED_ERROR_UNSUPPORTED, "/gpu/sycl/gen backend requires QFunction source code file");

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Sycl_gen));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Sycl_gen));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
