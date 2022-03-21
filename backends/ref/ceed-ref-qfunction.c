// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

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

  void *ctx_data = NULL;
  ierr = CeedQFunctionGetContextData(qf, CEED_MEM_HOST, &ctx_data);
  CeedChkBackend(ierr);

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

  ierr = f(ctx_data, Q, impl->inputs, impl->outputs); CeedChkBackend(ierr);

  for (int i = 0; i<num_in; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], &impl->inputs[i]); CeedChkBackend(ierr);
  }
  for (int i = 0; i<num_out; i++) {
    ierr = CeedVectorRestoreArray(V[i], &impl->outputs[i]); CeedChkBackend(ierr);
  }
  ierr = CeedQFunctionRestoreContextData(qf, &ctx_data); CeedChkBackend(ierr);

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
