// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-hip-gen.h"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <string.h>

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Hip_gen(const char *resource, Ceed ceed) {
  char *resource_root;
  CeedCallBackend(CeedHipGetResourceRoot(ceed, resource, &resource_root));
  if (strcmp(resource_root, "/gpu/hip") && strcmp(resource_root, "/gpu/hip/gen")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Hip backend cannot use resource: %s", resource);
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedFree(&resource_root));

  Ceed_Hip *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedHipInit(ceed, resource));

  Ceed ceedshared;
  CeedCallBackend(CeedInit("/gpu/hip/shared", &ceedshared));
  CeedCallBackend(CeedSetDelegate(ceed, ceedshared));

  const char fallbackresource[] = "/gpu/hip/ref";
  CeedCallBackend(CeedSetOperatorFallbackResource(ceed, fallbackresource));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Hip_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Hip_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Hip_Gen(void) { return CeedRegister("/gpu/hip/gen", CeedInit_Hip_gen, 20); }
//------------------------------------------------------------------------------
