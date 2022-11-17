// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <string.h>
#include <valgrind/memcheck.h>

#include "ceed-memcheck.h"

//------------------------------------------------------------------------------
// QFunction Apply
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Memcheck(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  CeedQFunction_Memcheck *impl;
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
  int mem_block_ids[num_out];
  for (CeedInt i = 0; i < num_out; i++) {
    CeedSize len;
    char     name[30] = "";

    CeedCallBackend(CeedVectorGetArrayWrite(V[i], CEED_MEM_HOST, &impl->outputs[i]));

    CeedCallBackend(CeedVectorGetLength(V[i], &len));
    VALGRIND_MAKE_MEM_UNDEFINED(impl->outputs[i], len);

    snprintf(name, 30, "'QFunction output %" CeedInt_FMT "'", i);
    mem_block_ids[i] = VALGRIND_CREATE_BLOCK(impl->outputs[i], len, name);
  }

  CeedCallBackend(f(ctx_data, Q, impl->inputs, impl->outputs));

  for (CeedInt i = 0; i < num_in; i++) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U[i], &impl->inputs[i]));
  }
  for (CeedInt i = 0; i < num_out; i++) {
    CeedCallBackend(CeedVectorRestoreArray(V[i], &impl->outputs[i]));
    VALGRIND_DISCARD(mem_block_ids[i]);
  }
  CeedCallBackend(CeedQFunctionRestoreContextData(qf, &ctx_data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction Destroy
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Memcheck(CeedQFunction qf) {
  CeedQFunction_Memcheck *impl;
  CeedCallBackend(CeedQFunctionGetData(qf, (void *)&impl));

  CeedCallBackend(CeedFree(&impl->inputs));
  CeedCallBackend(CeedFree(&impl->outputs));
  CeedCallBackend(CeedFree(&impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction Create
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Memcheck(CeedQFunction qf) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));

  CeedQFunction_Memcheck *impl;
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->inputs));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->outputs));
  CeedCallBackend(CeedQFunctionSetData(qf, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Memcheck));

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
