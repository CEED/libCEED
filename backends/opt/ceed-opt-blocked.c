// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>
#include <string.h>

#include "ceed-opt.h"

//------------------------------------------------------------------------------
// Backend Destroy
//------------------------------------------------------------------------------
static int CeedDestroy_Opt(Ceed ceed) {
  Ceed_Opt *data;
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedFree(&data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Opt_Blocked(const char *resource, Ceed ceed) {
  if (strcmp(resource, "/cpu/self") && strcmp(resource, "/cpu/self/opt") && strcmp(resource, "/cpu/self/opt/blocked")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Opt backend cannot use resource: %s", resource);
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceed_ref;
  CeedCallBackend(CeedInit("/cpu/self/ref/serial", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  // Set fallback CEED resource for advanced operator functionality
  const char fallbackresource[] = "/cpu/self/ref/serial";
  CeedCallBackend(CeedSetOperatorFallbackResource(ceed, fallbackresource));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Opt));

  // Set blocksize
  Ceed_Opt *data;
  CeedCallBackend(CeedCalloc(1, &data));
  data->blk_size = 8;
  CeedCallBackend(CeedSetData(ceed, data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Opt_Blocked(void) { return CeedRegister("/cpu/self/opt/blocked", CeedInit_Opt_Blocked, 40); }
//------------------------------------------------------------------------------
