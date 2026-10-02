// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-cutlass.h"
#include <ceed.h>
#include <ceed/backend.h>
#include <string.h>

static int CeedInit_Cutlass(const char *resource, Ceed ceed) {
  char *resource_root;
  Ceed  ceed_ref;

  CeedCallBackend(CeedGetResourceRoot(ceed, resource, ":", &resource_root));
  CeedCheck(!strcmp(resource_root, "/gpu/cuda/cutlass"), ceed, CEED_ERROR_BACKEND, "Cutlass backend cannot use resource: %s", resource);
  CeedCallBackend(CeedFree(&resource_root));

  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
  CeedCallBackend(CeedSetDelegate(ceed, ceed_ref));
  CeedCallBackend(CeedDestroy(&ceed_ref));

  return CEED_ERROR_SUCCESS;
}

CEED_INTERN int CeedRegister_Cutlass(void) { return CeedRegister("/gpu/cuda/cutlass", CeedInit_Cutlass, 120); }