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

#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Xsmm_Blocked(const char *resource, Ceed ceed) {
  if (strcmp(resource, "/cpu/self") && strcmp(resource, "/cpu/self/xsmm") && strcmp(resource, "/cpu/self/xsmm/blocked")) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "blocked libXSMM backend cannot use resource: %s", resource);
    // LCOV_EXCL_STOP
  }
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceed_ref;
  CeedCallBackend(CeedInit("/cpu/self/opt/blocked", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP64) {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate", CeedTensorContractCreate_f64_Xsmm));
  } else {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate", CeedTensorContractCreate_f32_Xsmm));
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Xsmm_Blocked(void) { return CeedRegister("/cpu/self/xsmm/blocked", CeedInit_Xsmm_Blocked, 20); }
//------------------------------------------------------------------------------
