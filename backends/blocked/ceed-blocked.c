// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-blocked.h"

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>
#include <string.h>

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
CEED_INTERN int CeedInit_Blocked(const char *resource, Ceed ceed) {
  if (strcmp(resource, "/cpu/self") && strcmp(resource, "/cpu/self/ref/blocked")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Blocked backend cannot use resource: %s", resource);
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

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Blocked));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Ref_Blocked(void) { return CeedRegister("/cpu/self/ref/blocked", CeedInit_Blocked, 55); }
//------------------------------------------------------------------------------
