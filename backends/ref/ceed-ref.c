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
#include <string.h>
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Backend Init
//------------------------------------------------------------------------------
static int CeedInit_Ref(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/cpu/self") && strcmp(resource, "/cpu/self/ref")
      && strcmp(resource, "/cpu/self/ref/serial"))
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Ref backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP
  ierr = CeedSetDeterministic(ceed, true); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate",
                                CeedVectorCreate_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHdiv",
                                CeedBasisCreateHdiv_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "TensorContractCreate",
                                CeedTensorContractCreate_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateOriented",
                                CeedElemRestrictionCreateOriented_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreate_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate",
                                CeedQFunctionContextCreate_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_Ref); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Backend Register
//------------------------------------------------------------------------------
CEED_INTERN int CeedRegister_Ref(void) {
  return
//! [Register]
    CeedRegister("/cpu/self/ref/serial", CeedInit_Ref, 50);
//! [Register]
}
//------------------------------------------------------------------------------
