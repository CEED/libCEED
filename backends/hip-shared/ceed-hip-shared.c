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

#include <string.h>
#include <stdarg.h>
#include "ceed-hip-shared.h"

//------------------------------------------------------------------------------
// Backend init
//------------------------------------------------------------------------------
static int CeedInit_Hip_shared(const char *resource, Ceed ceed) {
  int ierr;
  const int nrc = 8; // number of characters in resource
  if (strncmp(resource, "/gpu/hip/shared", nrc))
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Hip backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChk(ierr);

  Ceed ceedref;
  CeedInit("/gpu/hip/ref", &ceedref);
  ierr = CeedSetDelegate(ceed, ceedref); CeedChk(ierr);

  Ceed_Hip_shared *data;
  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  ierr = CeedSetData(ceed, data); CeedChk(ierr);
  ierr = CeedHipInit(ceed, resource, nrc); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Hip_shared);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Hip); CeedChk(ierr);
  CeedChk(ierr);
  return 0;
}

//------------------------------------------------------------------------------
// Register backend
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Hip_Shared(void) {
  return CeedRegister("/gpu/hip/shared", CeedInit_Hip_shared, 25);
}
//------------------------------------------------------------------------------
