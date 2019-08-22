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

#include "ceed-memcheck.h"

static int CeedQFunctionApply_Memcheck(CeedQFunction qf, CeedInt Q,
                                       CeedVector *U, CeedVector *V) {
  int ierr;
  CeedQFunction_Memcheck *impl;
  ierr = CeedQFunctionGetData(qf, (void *)&impl); CeedChk(ierr);

  void *ctx;
  ierr = CeedQFunctionGetContext(qf, &ctx); CeedChk(ierr);

  int (*f)() = NULL;
  ierr = CeedQFunctionGetUserFunction(qf, (int (* *)())&f); CeedChk(ierr);

  CeedInt nIn, nOut;
  ierr = CeedQFunctionGetNumArgs(qf, &nIn, &nOut); CeedChk(ierr);

  for (int i = 0; i<nIn; i++) {
    ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_HOST, &impl->inputs[i]);
    CeedChk(ierr);
  }
  for (int i = 0; i<nOut; i++) {
    ierr = CeedVectorGetArray(V[i], CEED_MEM_HOST, &impl->outputs[i]);
    CeedChk(ierr);
    CeedInt len;
    ierr = CeedVectorGetLength(V[i], &len); CeedChk(ierr);
    VALGRIND_MAKE_MEM_UNDEFINED(impl->outputs[i], len);
  }

  ierr = f(ctx, Q, impl->inputs, impl->outputs); CeedChk(ierr);

  for (int i = 0; i<nIn; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], &impl->inputs[i]); CeedChk(ierr);
  }
  for (int i = 0; i<nOut; i++) {
    ierr = CeedVectorRestoreArray(V[i], &impl->outputs[i]); CeedChk(ierr);
  }

  return 0;
}

static int CeedQFunctionDestroy_Memcheck(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Memcheck *impl;
  ierr = CeedQFunctionGetData(qf, (void *)&impl); CeedChk(ierr);

  ierr = CeedFree(&impl->inputs); CeedChk(ierr);
  ierr = CeedFree(&impl->outputs); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);

  return 0;
}

int CeedQFunctionCreate_Memcheck(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);

  CeedQFunction_Memcheck *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->inputs); CeedChk(ierr);
  ierr = CeedCalloc(16, &impl->outputs); CeedChk(ierr);
  ierr = CeedQFunctionSetData(qf, (void *)&impl); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Memcheck); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Memcheck); CeedChk(ierr);

  return 0;
}
