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

#include "ceed-avx.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Avx(const char *resource, Ceed ceed) {
  if (strcmp(resource, "/cpu/self") && strcmp(resource, "/cpu/self/avx") && strcmp(resource, "/cpu/self/avx/blocked")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "AVX backend cannot use resource: %s", resource);
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceed_ref;
  CeedCallBackend(CeedInit("/cpu/self/opt/blocked", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP64) {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate", CeedTensorContractCreate_f64_Avx));
  } else {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate", CeedTensorContractCreate_f32_Avx));
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Avx_Blocked(void) { return CeedRegister("/cpu/self/avx/blocked", CeedInit_Avx, 30); }
//------------------------------------------------------------------------------
