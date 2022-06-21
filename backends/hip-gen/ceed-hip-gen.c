// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include "ceed-hip-gen.h"

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Hip_gen(const char *resource, Ceed ceed) {
  int ierr;

  char *resource_root;
  ierr = CeedHipGetResourceRoot(ceed, resource, &resource_root);
  CeedChkBackend(ierr);
  if (strcmp(resource_root, "/gpu/hip") && strcmp(resource_root, "/gpu/hip/gen"))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Hip backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedFree(&resource_root); CeedChkBackend(ierr);

  Ceed_Hip *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedSetData(ceed, data); CeedChkBackend(ierr);
  ierr = CeedHipInit(ceed, resource); CeedChkBackend(ierr);

  Ceed ceedshared;
  CeedInit("/gpu/hip/shared", &ceedshared);
  ierr = CeedSetDelegate(ceed, ceedshared); CeedChkBackend(ierr);

  const char fallbackresource[] = "/gpu/hip/ref";
  ierr = CeedSetOperatorFallbackResource(ceed, fallbackresource);
  CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Hip_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Hip_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Hip_Gen(void) {
  return CeedRegister("/gpu/hip/gen", CeedInit_Hip_gen, 20);
}
//------------------------------------------------------------------------------
