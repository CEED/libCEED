// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cuda-shared.h"

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>

#include "../cuda/ceed-cuda-common.h"

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Cuda_shared(const char *resource, Ceed ceed) {
  Ceed       ceed_ref;
  Ceed_Cuda *data;
  char      *resource_root;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/cuda/shared"), ceed, CEED_ERROR_BACKEND, "Cuda backend cannot use resource: %s", resource);
  CeedCallBackend(CeedSetDeterministic(ceed, true));

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedSetData(ceed, data));
  CeedCallBackend(CeedInit_Cuda(ceed, resource));

  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Cuda_shared));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda_Shared(void) { return CeedRegister("/gpu/cuda/shared", CeedInit_Cuda_shared, 25); }

//------------------------------------------------------------------------------
