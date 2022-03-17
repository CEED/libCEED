// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>
#include "ceed-avx.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Avx(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/cpu/self")
      && strcmp(resource, "/cpu/self/avx/serial"))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "AVX backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChkBackend(ierr);

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceed_ref;
  CeedInit("/cpu/self/opt/serial", &ceed_ref);
  ierr = CeedSetDelegate(ceed, ceed_ref); CeedChkBackend(ierr);

  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP64) {
    ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate",
                                  CeedTensorContractCreate_f64_Avx);
    CeedChkBackend(ierr);
  } else {
    ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate",
                                  CeedTensorContractCreate_f32_Avx);
    CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Avx_Serial(void) {
  return CeedRegister("/cpu/self/avx/serial", CeedInit_Avx, 35);
}
//------------------------------------------------------------------------------
