// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stddef.h>

#include "ceed-ref.h"

//------------------------------------------------------------------------------
// QFunction Apply
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Ref(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  CeedQFunction_Ref *impl;
  CeedCallBackend(CeedQFunctionGetData(qf, &impl));

  void *ctx_data = NULL;
  CeedCallBackend(CeedQFunctionGetContextData(qf, CEED_MEM_HOST, &ctx_data));

  CeedQFunctionUser f = NULL;
  CeedCallBackend(CeedQFunctionGetUserFunction(qf, &f));

  CeedInt num_in, num_out;
  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_in, &num_out));

  for (CeedInt i = 0; i < num_in; i++) {
    CeedCallBackend(CeedVectorGetArrayRead(U[i], CEED_MEM_HOST, &impl->inputs[i]));
  }
  for (CeedInt i = 0; i < num_out; i++) {
    CeedCallBackend(CeedVectorGetArrayWrite(V[i], CEED_MEM_HOST, &impl->outputs[i]));
  }

  CeedCallBackend(f(ctx_data, Q, impl->inputs, impl->outputs));

  for (CeedInt i = 0; i < num_in; i++) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U[i], &impl->inputs[i]));
  }
  for (CeedInt i = 0; i < num_out; i++) {
    CeedCallBackend(CeedVectorRestoreArray(V[i], &impl->outputs[i]));
  }
  CeedCallBackend(CeedQFunctionRestoreContextData(qf, &ctx_data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction Destroy
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Ref(CeedQFunction qf) {
  CeedQFunction_Ref *impl;
  CeedCallBackend(CeedQFunctionGetData(qf, &impl));

  CeedCallBackend(CeedFree(&impl->inputs));
  CeedCallBackend(CeedFree(&impl->outputs));
  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction Create
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Ref(CeedQFunction qf) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));

  CeedQFunction_Ref *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->inputs));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->outputs));
  CeedCallBackend(CeedQFunctionSetData(qf, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Ref));

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
