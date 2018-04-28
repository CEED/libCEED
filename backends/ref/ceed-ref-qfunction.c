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

#include <ceed-impl.h>
#include <string.h>
#include "ceed-ref.h"

static int CeedQFunctionApply_Ref(CeedQFunction qf, void *qdata, CeedInt Q,
                                  const CeedScalar *const *u,
                                  CeedScalar *const *v) {
  int ierr;
  ierr = qf->function(qf->ctx, qdata, Q, u, v); CeedChk(ierr);
  return 0;
}

static int CeedQFunctionDestroy_Ref(CeedQFunction qf) {
  return 0;
}

int CeedQFunctionCreate_Ref(CeedQFunction qf) {
  qf->Apply = CeedQFunctionApply_Ref;
  qf->Destroy = CeedQFunctionDestroy_Ref;
  return 0;
}
