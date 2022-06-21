// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-cuda-gen.h"

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Cuda_gen(const char *resource, Ceed ceed) {
  int ierr;

  char *resource_root;
  ierr = CeedCudaGetResourceRoot(ceed, resource, &resource_root);
  CeedChkBackend(ierr);
  if (strcmp(resource_root, "/gpu/cuda")
      && strcmp(resource_root, "/gpu/cuda/gen"))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Cuda backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedFree(&resource_root); CeedChkBackend(ierr);

  Ceed_Cuda *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedSetData(ceed, data); CeedChkBackend(ierr);
  ierr = CeedCudaInit(ceed, resource); CeedChkBackend(ierr);

  Ceed ceedshared;
  CeedInit("/gpu/cuda/shared", &ceedshared);
  ierr = CeedSetDelegate(ceed, ceedshared); CeedChkBackend(ierr);

  const char fallbackresource[] = "/gpu/cuda/ref";
  ierr = CeedSetOperatorFallbackResource(ceed, fallbackresource);
  CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Cuda_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Cuda_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Cuda); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Cuda_Gen(void) {
  return CeedRegister("/gpu/cuda/gen", CeedInit_Cuda_gen, 20);
}
//------------------------------------------------------------------------------
