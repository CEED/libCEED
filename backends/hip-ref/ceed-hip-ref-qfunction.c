// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/hip/hip-types.h>
#include <hip/hip_runtime.h>

#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref-qfunction-load.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Hip(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  Ceed               ceed;
  Ceed_Hip          *ceed_Hip;
  CeedInt            num_input_fields, num_output_fields;
  CeedQFunction_Hip *data;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));

  // Build and compile kernel, if not done
  CeedCallBackend(CeedQFunctionBuildKernel_Hip_ref(qf));

  CeedCallBackend(CeedQFunctionGetData(qf, &data));
  CeedCallBackend(CeedGetData(ceed, &ceed_Hip));
  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields));
  const int block_size = ceed_Hip->opt_block_size;

  // Read vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedVectorGetArrayRead(U[i], CEED_MEM_DEVICE, &data->fields.inputs[i]));
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedVectorGetArrayWrite(V[i], CEED_MEM_DEVICE, &data->fields.outputs[i]));
  }

  // Get context data
  CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &data->d_c));

  // Run kernel
  void *args[] = {&data->d_c, (void *)&Q, &data->fields};

  CeedCallBackend(CeedRunKernel_Hip(ceed, data->QFunction, CeedDivUpInt(Q, block_size), block_size, args));

  // Restore vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U[i], &data->fields.inputs[i]));
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedVectorRestoreArray(V[i], &data->fields.outputs[i]));
  }

  // Restore context
  CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &data->d_c));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Hip(CeedQFunction qf) {
  CeedQFunction_Hip *data;

  CeedCallBackend(CeedQFunctionGetData(qf, &data));
  if (data->module) CeedCallHip(CeedQFunctionReturnCeed(qf), hipModuleUnload(data->module));
  CeedCallBackend(CeedFree(&data));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Hip(CeedQFunction qf) {
  Ceed               ceed;
  CeedInt            num_input_fields, num_output_fields;
  CeedQFunction_Hip *data;

  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedQFunctionSetData(qf, data));
  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields));

  CeedCallBackend(CeedQFunctionGetKernelName(qf, &data->qfunction_name));

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Hip));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
