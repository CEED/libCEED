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
#include "ceed-magma.h"

CEED_INTERN int CeedInit_Magma_Det(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/gpu/cuda/magma/det")
      && strcmp(resource, "/gpu/hip/magma/det"))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Magma backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChkBackend(ierr);

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceedref;
  #ifdef HAVE_HIP
  CeedInit("/gpu/hip/magma", &ceedref);
  #else
  CeedInit("/gpu/cuda/magma", &ceedref);
  #endif
  ierr = CeedSetDelegate(ceed, ceedref); CeedChkBackend(ierr);

  // Create reference CEED for restriction
  Ceed restrictionceedref;
  #ifdef HAVE_HIP
  CeedInit("/gpu/hip/ref", &restrictionceedref);
  #else
  CeedInit("/gpu/cuda/ref", &restrictionceedref);
  #endif
  ierr = CeedSetObjectDelegate(ceed, restrictionceedref, "ElemRestriction");
  CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

CEED_INTERN int CeedRegister_Magma_Det(void) {
  #ifdef HAVE_HIP
  return CeedRegister("/gpu/hip/magma/det", CeedInit_Magma_Det, 125);
  #else
  return CeedRegister("/gpu/cuda/magma/det", CeedInit_Magma_Det, 125);
  #endif
}
