// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>

#include "ceed-avx.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Avx(const char *resource, Ceed ceed) {
  Ceed ceed_ref;

  CeedCheck(!strcmp(resource, "/cpu/self") || !strcmp(resource, "/cpu/self/avx") || !strcmp(resource, "/cpu/self/avx/blocked"), ceed,
            CEED_ERROR_BACKEND, "AVX backend cannot use resource: %s", resource);
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  // Create reference Ceed that implementation will be dispatched through unless overridden
  CeedCallBackend(CeedInit("/cpu/self/opt/blocked", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));
  CeedCallBackend(CeedDestroy(&ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate", CeedTensorContractCreate_Avx));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Avx_Blocked(void) { return CeedRegister("/cpu/self/avx/blocked", CeedInit_Avx, 30); }

//------------------------------------------------------------------------------
