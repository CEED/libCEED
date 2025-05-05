// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <string.h>

#include "../ceed-backend-init.h"
#include "../hip/ceed-hip-common.h"
#include "ceed-hip-gen.h"

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
CEED_INTERN int CeedInit_Hip_Gen(const char *resource, Ceed ceed) {
  char     *resource_root;
  Ceed      ceed_shared, ceed_ref;
  Ceed_Hip *data;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/hip") || !strcmp(resource_root, "/gpu/hip/gen"), ceed, CEED_ERROR_BACKEND,
            "Hip backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Hip(ceed, resource));

  CeedCallBackend(CeedInit("/gpu/hip/shared", &ceed_shared));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_shared));
  CeedCallBackend(CeedDestroy(&ceed_shared));

  CeedCallBackend(CeedInit("/gpu/hip/ref", &ceed_ref));
  CeedCallBackend(CeedSetOperatorFallbackCeed(ceed, ceed_ref));
  CeedCallBackend(CeedDestroy(&ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Hip_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Hip_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "CompositeOperatorCreate", CeedOperatorCreate_Hip_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreateAtPoints", CeedOperatorCreate_Hip_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
