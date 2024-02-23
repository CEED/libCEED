// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdio.h>
#include <valgrind/memcheck.h>

#include "ceed-memcheck.h"

//------------------------------------------------------------------------------
// QFunction Apply
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Memcheck(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  Ceed                    ceed;
  void                   *ctx_data = NULL;
  CeedInt                 num_in, num_out;
  CeedQFunctionUser       f = NULL;
  CeedQFunctionField     *output_fields;
  CeedQFunction_Memcheck *impl;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedQFunctionGetData(qf, &impl));
  CeedCallBackend(CeedQFunctionGetContextData(qf, CEED_MEM_HOST, &ctx_data));
  CeedCallBackend(CeedQFunctionGetUserFunction(qf, &f));
  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_in, &num_out));
  int mem_block_ids[num_out];

  // Get input/output arrays
  for (CeedInt i = 0; i < num_in; i++) {
    CeedCallBackend(CeedVectorGetArrayRead(U[i], CEED_MEM_HOST, &impl->inputs[i]));
  }
  for (CeedInt i = 0; i < num_out; i++) {
    CeedSize len;
    char     name[32] = "";

    CeedCallBackend(CeedVectorGetArrayWrite(V[i], CEED_MEM_HOST, &impl->outputs[i]));

    CeedCallBackend(CeedVectorGetLength(V[i], &len));
    VALGRIND_MAKE_MEM_UNDEFINED(impl->outputs[i], len);

    snprintf(name, 32, "'QFunction output %" CeedInt_FMT "'", i);
    mem_block_ids[i] = VALGRIND_CREATE_BLOCK(impl->outputs[i], len, name);
  }

  // Call user function
  CeedCallBackend(f(ctx_data, Q, impl->inputs, impl->outputs));

  // Restore input arrays
  for (CeedInt i = 0; i < num_in; i++) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U[i], &impl->inputs[i]));
  }
  // Check for unset output values
  {
    const char *kernel_name, *kernel_path;

    CeedCallBackend(CeedQFunctionGetSourcePath(qf, &kernel_path));
    CeedCallBackend(CeedQFunctionGetKernelName(qf, &kernel_name));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &output_fields));
    for (CeedInt i = 0; i < num_out; i++) {
      CeedInt field_size;

      // Note: need field size because vector may be longer than needed for output
      CeedCallBackend(CeedQFunctionFieldGetSize(output_fields[i], &field_size));
      for (CeedSize j = 0; j < field_size * (CeedSize)Q; j++) {
        CeedCheck(!isnan(impl->outputs[i][j]), ceed, CEED_ERROR_BACKEND,
                  "QFunction output %" CeedInt_FMT " entry %" CeedSize_FMT " is NaN after restoring write-only access: %s:%s ", i, j, kernel_path,
                  kernel_name);
      }
      CeedCallBackend(CeedVectorRestoreArray(V[i], &impl->outputs[i]));
      VALGRIND_DISCARD(mem_block_ids[i]);
    }
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
  Ceed                    ceed;
  CeedQFunction_Memcheck *impl;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->inputs));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->outputs));
  CeedCallBackend(CeedQFunctionSetData(qf, impl));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Memcheck));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Memcheck));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
