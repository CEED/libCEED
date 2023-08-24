// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdlib.h>
#include <string.h>

#include "ceed-magma-common.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Magma_Det(const char *resource, Ceed ceed) {
  const int nrc = 18;  // number of characters in resource
  CeedCheck(!strncmp(resource, "/gpu/cuda/magma/det", nrc) || !strncmp(resource, "/gpu/hip/magma/det", nrc), ceed, CEED_ERROR_BACKEND,
            "Magma backend cannot use resource: %s", resource);
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  Ceed_Magma *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Magma_common(ceed, resource));

  // Create reference Ceed that implementation will be dispatched through
  Ceed ceed_ref;
#ifdef CEED_MAGMA_USE_HIP
  CeedCallBackend(CeedInit("/gpu/hip/magma", &ceed_ref));
#else
  CeedCallBackend(CeedInit("/gpu/cuda/magma", &ceed_ref));
#endif
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Magma));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Magma_Det(void) {
#ifdef CEED_MAGMA_USE_HIP
  return CeedRegister("/gpu/hip/magma/det", CeedInit_Magma_Det, 125);
#else
  return CeedRegister("/gpu/cuda/magma/det", CeedInit_Magma_Det, 125);
#endif
}

//------------------------------------------------------------------------------
