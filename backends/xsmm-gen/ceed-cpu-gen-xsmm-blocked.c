// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>

#include "../cpu-gen/ceed-cpu-gen.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Cpu_Gen_Blocked(const char *resource, Ceed ceed) {
  Ceed          ceed_ref;
  Ceed_Cpu_Gen *data;

  CeedCheck(!strcmp(resource, "/cpu/self") || !strcmp(resource, "/cpu/self/gen/xsmm") || !strcmp(resource, "/cpu/self/gen/xsmm/blocked"), ceed,
            CEED_ERROR_BACKEND, "CPU Gen backend cannot use resource: %s", resource);

  // Create reference Ceed that implementation will be dispatched through unless overridden
  CeedCallBackend(CeedInit("/cpu/self/xsmm/blocked", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));
  CeedCallBackend(CeedSetOperatorFallbackCeed(ceed, ceed_ref));
  CeedCallBackend(CeedDestroy(&ceed_ref));

  // Setup from ref version of cpu-gen
  CeedCallBackend(CeedInit_Cpu_Gen_Blocked_Core(resource, ceed));

  // Set XSMM flags
  CeedCallBackend(CeedGetData(ceed, &data));
  data->is_xsmm = true;

#define CEED_QUOTE(name) #name
#define CEED_STRINGIFY(macro) CEED_QUOTE(macro)
  const char *CeedXsmmFlags = "CEED_USE_XSMM=1 " CEED_STRINGIFY(CEED_XSMM_C_FLAGS);
#undef CEED_QUOTE
#undef CEED_STRINGIFY

  CeedCallBackend(CeedAddJitDefine(ceed, CeedXsmmFlags));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cpu_Gen_Xsmm_Blocked(void) { return CeedRegister("/cpu/self/gen/xsmm/blocked", CeedInit_Cpu_Gen_Blocked, 59); }

//------------------------------------------------------------------------------
