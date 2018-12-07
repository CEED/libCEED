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
#include "ceed-libparanumal.h"

static int CeedInit_libparanumal(const char *resource, Ceed ceed) {
  int ierr;
  Ceed ceeddelegate;
  CeedInit("/gpu/occa", &ceeddelegate);
  ierr = CeedSetDelegate(ceed, &ceeddelegate);
  CeedChk(ierr);
  if (strcmp(resource, "/gpu/libparanumal"))
    return CeedError(ceed, 1, "LibParanumal backend cannot use resource: %s", resource);

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_libparanumal); CeedChk(ierr);
  // ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorDestroy",
  //                               CeedOperatorDestroy_libparanumal); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_libparanumal); CeedChk(ierr);
  // ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionDestroy",
  //                               CeedQFunctionDestroy_libparanumal); CeedChk(ierr);

  return 0;
}

__attribute__((constructor))
static void Register(void) {
//! [Register]
  CeedRegister("/gpu/libparanumal", CeedInit_libparanumal, 20);
//! [Register]
}
