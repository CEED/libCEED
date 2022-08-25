// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-gen.h"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <string.h>

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Cuda_gen(const char *resource, Ceed ceed) {
  char *resource_root;
  CeedCallBackend(CeedCudaGetResourceRoot(ceed, resource, &resource_root));
  if (strcmp(resource_root, "/gpu/cuda") && strcmp(resource_root, "/gpu/cuda/gen")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Cuda backend cannot use resource: %s", resource);
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedFree(&resource_root));

  Ceed_Cuda *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedCudaInit(ceed, resource));

  Ceed ceedshared;
  CeedCall(CeedInit("/gpu/cuda/shared", &ceedshared));
  CeedCallBackend(CeedSetDelegate(ceed, ceedshared));

  const char fallbackresource[] = "/gpu/cuda/ref";
  CeedCallBackend(CeedSetOperatorFallbackResource(ceed, fallbackresource));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Cuda_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Cuda_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda_Gen(void) { return CeedRegister("/gpu/cuda/gen", CeedInit_Cuda_gen, 20); }
//------------------------------------------------------------------------------
