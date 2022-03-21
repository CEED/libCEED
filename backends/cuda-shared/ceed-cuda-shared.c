// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-cuda-shared.h"

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Cuda_shared(const char *resource, Ceed ceed) {
  int ierr;

  if (strcmp(resource, "/gpu/cuda/shared"))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Cuda backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChk(ierr);

  Ceed_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  ierr = CeedSetData(ceed, data); CeedChk(ierr);
  ierr = CeedCudaInit(ceed, resource); CeedChk(ierr);

  Ceed ceed_ref;
  CeedInit("/gpu/cuda/ref", &ceed_ref);
  ierr = CeedSetDelegate(ceed, ceed_ref); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Cuda_shared);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Cuda); CeedChk(ierr);
  CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda_Shared(void) {
  return CeedRegister("/gpu/cuda/shared", CeedInit_Cuda_shared, 25);
}
//------------------------------------------------------------------------------
