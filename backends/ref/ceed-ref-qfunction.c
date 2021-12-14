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
#include <stddef.h>
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// QFunction Apply
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Ref(CeedQFunction qf, CeedInt Q,
                                  CeedVector *U, CeedVector *V) {
  int ierr;
  CeedQFunction_Ref *impl;
  ierr = CeedQFunctionGetData(qf, &impl); CeedChkBackend(ierr);

  CeedQFunctionContext ctx;
  ierr = CeedQFunctionGetContext(qf, &ctx); CeedChkBackend(ierr);
  void *ctxData = NULL;
  if (ctx) {
    ierr = CeedQFunctionContextGetData(ctx, CEED_MEM_HOST, &ctxData);
    CeedChkBackend(ierr);
  }

  CeedQFunctionUser f = NULL;
  ierr = CeedQFunctionGetUserFunction(qf, &f); CeedChkBackend(ierr);

  CeedInt num_in, num_out;
  ierr = CeedQFunctionGetNumArgs(qf, &num_in, &num_out); CeedChkBackend(ierr);

  for (int i = 0; i<num_in; i++) {
    ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_HOST, &impl->inputs[i]);
    CeedChkBackend(ierr);
  }
  for (int i = 0; i<num_out; i++) {
    ierr = CeedVectorGetArrayWrite(V[i], CEED_MEM_HOST, &impl->outputs[i]);
    CeedChkBackend(ierr);
  }

  ierr = f(ctxData, Q, impl->inputs, impl->outputs); CeedChkBackend(ierr);

  for (int i = 0; i<num_in; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], &impl->inputs[i]); CeedChkBackend(ierr);
  }
  for (int i = 0; i<num_out; i++) {
    ierr = CeedVectorRestoreArray(V[i], &impl->outputs[i]); CeedChkBackend(ierr);
  }
  if (ctx) {
    ierr = CeedQFunctionContextRestoreData(ctx, &ctxData); CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction Destroy
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Ref(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Ref *impl;
  ierr = CeedQFunctionGetData(qf, &impl); CeedChkBackend(ierr);

  ierr = CeedFree(&impl->inputs); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->outputs); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction Create
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Ref(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);

  CeedQFunction_Ref *impl;
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedCalloc(CEED_FIELD_MAX, &impl->inputs); CeedChkBackend(ierr);
  ierr = CeedCalloc(CEED_FIELD_MAX, &impl->outputs); CeedChkBackend(ierr);
  ierr = CeedQFunctionSetData(qf, impl); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Ref); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
