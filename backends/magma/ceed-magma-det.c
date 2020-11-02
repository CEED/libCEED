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

#include "ceed-magma.h"

static int CeedInit_Magma_Det(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/gpu/cuda/magma/det")
      && strcmp(resource, "/gpu/hip/magma/det"))
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Magma backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChk(ierr);

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceedref;
  #ifdef HAVE_HIP
  CeedInit("/gpu/hip/magma", &ceedref);
  #else
  CeedInit("/gpu/cuda/magma", &ceedref);
  #endif
  ierr = CeedSetDelegate(ceed, ceedref); CeedChk(ierr);

  // Create reference CEED for restriction
  Ceed restrictionceedref;
  #ifdef HAVE_HIP
  CeedInit("/gpu/hip/ref", &restrictionceedref);
  #else
  CeedInit("/gpu/cuda/ref", &restrictionceedref);
  #endif
  ierr = CeedSetObjectDelegate(ceed, restrictionceedref, "ElemRestriction");
  CeedChk(ierr);

  return 0;
}

__attribute__((constructor))
static void Register(void) {
  #ifdef HAVE_HIP
  CeedRegister("/gpu/hip/magma/det", CeedInit_Magma_Det, 125);
  #else
  CeedRegister("/gpu/cuda/magma/det", CeedInit_Magma_Det, 125);
  #endif
}
