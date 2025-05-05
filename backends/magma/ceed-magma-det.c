// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdlib.h>
#include <string.h>

#include "../ceed-backend-init.h"
#include "ceed-magma-common.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
CEED_INTERN int CeedInit_Magma_Det(const char *resource, Ceed ceed) {
  Ceed        ceed_ref;
  Ceed_Magma *data;
  const int   nrc = 18;  // number of characters in resource

  CeedCheck(!strncmp(resource, "/gpu/cuda/magma/det", nrc) || !strncmp(resource, "/gpu/hip/magma/det", nrc), ceed, CEED_ERROR_BACKEND,
            "Magma backend cannot use resource: %s", resource);
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Magma_common(ceed, resource));

  // Create reference Ceed that implementation will be dispatched through
#ifdef CEED_MAGMA_USE_HIP
  CeedCallBackend(CeedInit("/gpu/hip/magma", &ceed_ref));
#else
  CeedCallBackend(CeedInit("/gpu/cuda/magma", &ceed_ref));
#endif
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));
  CeedCallBackend(CeedDestroy(&ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Magma));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
