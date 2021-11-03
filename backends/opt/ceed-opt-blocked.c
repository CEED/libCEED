// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "ceed-opt.h"

//------------------------------------------------------------------------------
// Backend Destroy
//------------------------------------------------------------------------------
static int CeedDestroy_Opt(Ceed ceed) {
  int ierr;
  Ceed_Opt *data;
  ierr = CeedGetData(ceed, &data); CeedChkBackend(ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Opt_Blocked(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/cpu/self") && strcmp(resource, "/cpu/self/opt") &&
      strcmp(resource, "/cpu/self/opt/blocked"))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Opt backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChkBackend(ierr);

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceed_ref;
  CeedInit("/cpu/self/ref/serial", &ceed_ref);
  ierr = CeedSetDelegate(ceed, ceed_ref); CeedChkBackend(ierr);

  // Set fallback CEED resource for advanced operator functionality
  const char fallbackresource[] = "/cpu/self/ref/serial";
  ierr = CeedSetOperatorFallbackResource(ceed, fallbackresource);
  CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Opt); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Opt); CeedChkBackend(ierr);

  // Set blocksize
  Ceed_Opt *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  data->blk_size = 8;
  ierr = CeedSetData(ceed, data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Opt_Blocked(void) {
  const char prefix[] = "/cpu/self/opt/blocked";
  if (getenv("CEED_DEBUG")) fprintf(stderr, "Backend Register: %s\n", prefix);
  return CeedRegister(prefix, CeedInit_Opt_Blocked, 40);
}
//------------------------------------------------------------------------------
