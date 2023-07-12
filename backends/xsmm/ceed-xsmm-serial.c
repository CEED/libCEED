// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>

#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Xsmm_Serial(const char *resource, Ceed ceed) {
  CeedCheck(!strcmp(resource, "/cpu/self") || !strcmp(resource, "/cpu/self/xsmm/serial"), ceed, CEED_ERROR_BACKEND,
            "serial libXSMM backend cannot use resource: %s", resource);
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  // Create reference Ceed that implementation will be dispatched through unless overridden
  Ceed ceed_ref;
  CeedCallBackend(CeedInit("/cpu/self/opt/serial", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate", CeedTensorContractCreate_Xsmm));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Xsmm_Serial(void) { return CeedRegister("/cpu/self/xsmm/serial", CeedInit_Xsmm_Serial, 25); }

//------------------------------------------------------------------------------
