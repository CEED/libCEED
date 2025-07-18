// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-sycl-gen.hpp"

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <string>
#include <string_view>
#include <string.h>

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Sycl_gen(const char *resource, Ceed ceed) {
  Ceed       ceed_shared, ceed_ref;
  Ceed_Sycl *data;
  char      *resource_root;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":device_id=", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/sycl") || !strcmp(resource_root, "/gpu/sycl/gen"), ceed, CEED_ERROR_BACKEND,
            "Sycl backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Sycl(ceed, resource));

  CeedCallBackend(CeedInit("/gpu/sycl/shared", &ceed_shared));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_shared));
  CeedCallBackend(CeedSetStream_Sycl(ceed_shared, &(data->sycl_queue)));
  CeedCallBackend(CeedDestroy(&ceed_shared));

  CeedCallBackend(CeedInit("/gpu/sycl/ref", &ceed_ref));
  CeedCallBackend(CeedSetOperatorFallbackCeed(ceed, ceed_ref));
  CeedCallBackend(CeedSetStream_Sycl(ceed_ref, &(data->sycl_queue)));
  CeedCallBackend(CeedDestroy(&ceed_ref));

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
