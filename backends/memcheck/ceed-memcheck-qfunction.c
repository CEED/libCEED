// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <string.h>
#include <valgrind/memcheck.h>
#include "ceed-memcheck.h"

//------------------------------------------------------------------------------
// QFunction Apply
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Memcheck(CeedQFunction qf, CeedInt Q,
                                       CeedVector *U, CeedVector *V) {
  int ierr;
  CeedQFunction_Memcheck *impl;
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

  for (CeedInt i = 0; i<num_in; i++) {
    ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_HOST, &impl->inputs[i]);
    CeedChkBackend(ierr);
  }
  int mem_block_ids[num_out];
  for (CeedInt i = 0; i<num_out; i++) {
    CeedSize len;
    char name[30] = "";

    ierr = CeedVectorGetArrayWrite(V[i], CEED_MEM_HOST, &impl->outputs[i]);
    CeedChkBackend(ierr);

    ierr = CeedVectorGetLength(V[i], &len); CeedChkBackend(ierr);
    VALGRIND_MAKE_MEM_UNDEFINED(impl->outputs[i], len);

    snprintf(name, 30, "'QFunction output %d'", i);
    mem_block_ids[i] = VALGRIND_CREATE_BLOCK(impl->outputs[i], len, name);
  }

  ierr = f(ctxData, Q, impl->inputs, impl->outputs); CeedChkBackend(ierr);

  for (CeedInt i = 0; i<num_in; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], &impl->inputs[i]); CeedChkBackend(ierr);
  }
  for (CeedInt i = 0; i<num_out; i++) {
    ierr = CeedVectorRestoreArray(V[i], &impl->outputs[i]); CeedChkBackend(ierr);
    VALGRIND_DISCARD(mem_block_ids[i]);
  }
  if (ctx) {
    ierr = CeedQFunctionContextRestoreData(ctx, &ctxData); CeedChkBackend(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction Destroy
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Memcheck(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Memcheck *impl;
  ierr = CeedQFunctionGetData(qf, (void *)&impl); CeedChkBackend(ierr);

  ierr = CeedFree(&impl->inputs); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->outputs); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction Create
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Memcheck(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);

  CeedQFunction_Memcheck *impl;
  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedCalloc(CEED_FIELD_MAX, &impl->inputs); CeedChkBackend(ierr);
  ierr = CeedCalloc(CEED_FIELD_MAX, &impl->outputs); CeedChkBackend(ierr);
  ierr = CeedQFunctionSetData(qf, impl); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Memcheck); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Memcheck); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
