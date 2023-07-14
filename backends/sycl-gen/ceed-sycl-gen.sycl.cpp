// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-gen.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <string.h>
#include <string>
#include <string_view>

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Sycl_gen(const char *resource, Ceed ceed) {
  char *resource_root;
  CeedCallBackend(CeedGetResourceRoot_Sycl(ceed, resource, &resource_root));
  if (strcmp(resource_root, "/gpu/sycl") && strcmp(resource_root, "/gpu/sycl/gen")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Sycl backend cannot use resource: %s", resource);
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedFree(&resource_root));

  Ceed_Sycl *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Sycl(ceed, resource));

  Ceed ceed_shared;
  CeedCallBackend(CeedInit("/gpu/sycl/shared", &ceed_shared));

  Ceed_Sycl *shared_data;
  CeedCallBackend(CeedGetData(ceed_shared, &shared_data));
  // Need to use the same queue everywhere for correct synchronization
  shared_data->sycl_queue = data->sycl_queue;

  CeedCallBackend(CeedSetDelegate(ceed, ceed_shared));

  const char fallbackresource[] = "/gpu/sycl/ref";
  CeedCallBackend(CeedSetOperatorFallbackResource(ceed, fallbackresource));

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Sycl_gen));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Sycl_gen));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Sycl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Sycl_Gen(void) { return CeedRegister("/gpu/sycl/gen", CeedInit_Sycl_gen, 20); }
//------------------------------------------------------------------------------
