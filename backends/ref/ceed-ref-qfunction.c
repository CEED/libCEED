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

#include "ceed-ref.h"

static int CeedQFunctionApply_Ref(CeedQFunction qf, CeedInt Q,
                                  CeedVector *U, CeedVector *V) {
  int ierr;
  void *ctx;
  ierr = CeedQFunctionGetContext(qf, &ctx); CeedChk(ierr);

  CeedQFunctionUser f = NULL;
  ierr = CeedQFunctionGetUserFunction(qf, &f); CeedChk(ierr);

  CeedInt nIn, nOut;
  ierr = CeedQFunctionGetNumArgs(qf, &nIn, &nOut); CeedChk(ierr);

  CeedQFunctionArguments args;
  for (int i = 0; i<nIn; i++) {
    ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_HOST, &args.in[i]);
    CeedChk(ierr);
  }
  for (int i = 0; i<nOut; i++) {
    ierr = CeedVectorGetArray(V[i], CEED_MEM_HOST, &args.out[i]);
    CeedChk(ierr);
  }

  ierr = f(ctx, Q, Q, args); CeedChk(ierr);

  for (int i = 0; i<nIn; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], &args.in[i]); CeedChk(ierr);
  }
  for (int i = 0; i<nOut; i++) {
    ierr = CeedVectorRestoreArray(V[i], &args.out[i]); CeedChk(ierr);
  }

  return 0;
}

static int CeedQFunctionDestroy_Ref(CeedQFunction qf) {
  return 0;
}

int CeedQFunctionCreate_Ref(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Ref); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Ref); CeedChk(ierr);

  return 0;
}
