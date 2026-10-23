// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-hip-hipblas.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Hip_hipblas(const char *resource, Ceed ceed) {
  Ceed      ceed_ref;
  Ceed_Hip *data;
  char     *resource_root;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/hip/hipblas") || !strcmp(resource_root, "/gpu/hip"), ceed, CEED_ERROR_BACKEND,
            "Hip backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));
  CeedCallBackend(CeedSetDeterministic(ceed, false));

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Hip(ceed, resource));

  CeedCallBackend(CeedInit("/gpu/hip/shared", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));
  CeedCallBackend(CeedDestroy(&ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1", CeedBasisCreateH1_Hip_hipblas));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHdiv", CeedBasisCreateHdiv_Hip_hipblas));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHcurl", CeedBasisCreateHcurl_Hip_hipblas));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Hip_hipblas(void) { return CeedRegister("/gpu/hip/hipblas", CeedInit_Hip_hipblas, 23); }

//------------------------------------------------------------------------------
