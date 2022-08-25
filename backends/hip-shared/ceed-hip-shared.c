// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-hip-shared.h"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>
#include <string.h>

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Hip_shared(const char *resource, Ceed ceed) {
  char *resource_root;
  CeedCallBackend(CeedHipGetResourceRoot(ceed, resource, &resource_root));
  if (strcmp(resource_root, "/gpu/hip/shared")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Hip backend cannot use resource: %s", resource);
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedFree(&resource_root));
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  Ceed_Hip *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedHipInit(ceed, resource));

  Ceed ceed_ref;
  CeedCallBackend(CeedInit("/gpu/hip/ref", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Hip_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Hip_Shared(void) { return CeedRegister("/gpu/hip/shared", CeedInit_Hip_shared, 25); }
//------------------------------------------------------------------------------
