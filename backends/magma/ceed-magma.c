// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-magma.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <stdlib.h>
#include <string.h>

#include "ceed-magma-common.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Magma(const char *resource, Ceed ceed) {
  Ceed        ceed_ref;
  Ceed_Magma *data;
  const int   nrc = 14;  // number of characters in resource

  CeedCheck(!strncmp(resource, "/gpu/cuda/magma", nrc) || !strncmp(resource, "/gpu/hip/magma", nrc), ceed, CEED_ERROR_BACKEND,
            "Magma backend cannot use resource: %s", resource);

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Magma_common(ceed, resource));

  // Create reference Ceed that implementation will be dispatched through unless overridden
#ifdef CEED_MAGMA_USE_HIP
  CeedCallBackend(CeedInit("/gpu/hip/ref", &ceed_ref));
#else
  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
#endif
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1", CeedBasisCreateH1_Magma));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Magma));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Magma(void) {
#ifdef CEED_MAGMA_USE_HIP
  return CeedRegister("/gpu/hip/magma", CeedInit_Magma, 120);
#else
  return CeedRegister("/gpu/cuda/magma", CeedInit_Magma, 120);
#endif
}

//------------------------------------------------------------------------------
