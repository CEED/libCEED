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
#include <string.h>
#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Xsmm_Serial(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/cpu/self")
      && strcmp(resource, "/cpu/self/xsmm/serial"))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "serial libXSMM backend cannot use resource: %s",
                     resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChkBackend(ierr);

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceedref;
  CeedInit("/cpu/self/opt/serial", &ceedref);
  ierr = CeedSetDelegate(ceed, ceedref); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate",
                                CeedTensorContractCreate_Xsmm); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Xsmm_Serial(void) {
  return CeedRegister("/cpu/self/xsmm/serial", CeedInit_Xsmm_Serial, 25);
}
//------------------------------------------------------------------------------
