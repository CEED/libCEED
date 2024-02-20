// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-gen.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <string.h>

#include "../cuda/ceed-cuda-common.h"

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Cuda_gen(const char *resource, Ceed ceed) {
  char      *resource_root;
  const char fallback_resource[] = "/gpu/cuda/shared";
  Ceed       ceed_shared;
  Ceed_Cuda *data;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/cuda") || !strcmp(resource_root, "/gpu/cuda/gen"), ceed, CEED_ERROR_BACKEND,
            "Cuda backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Cuda(ceed, resource));

  CeedCall(CeedInit("/gpu/cuda/shared", &ceed_shared));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_shared));

  CeedCallBackend(CeedSetOperatorFallbackResource(ceed, fallback_resource));

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
