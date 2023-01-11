// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>

#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref-qfunction-load.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Hip(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));

  // Build and compile kernel, if not done
  CeedCallBackend(CeedHipBuildQFunction(qf));

  CeedQFunction_Hip *data;
  CeedCallBackend(CeedQFunctionGetData(qf, &data));
  Ceed_Hip *ceed_Hip;
  CeedCallBackend(CeedGetData(ceed, &ceed_Hip));
  CeedInt num_input_fields, num_output_fields;
  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields));
  const int blocksize = ceed_Hip->opt_block_size;

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
  CeedCallBackend(CeedRunKernelHip(ceed, data->QFunction, CeedDivUpInt(Q, blocksize), blocksize, args));

  // Restore vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedVectorRestoreArrayRead(U[i], &data->fields.inputs[i]));
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedVectorRestoreArray(V[i], &data->fields.outputs[i]));
  }

  // Restore context
  CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &data->d_c));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Hip(CeedQFunction qf) {
  CeedQFunction_Hip *data;
  CeedCallBackend(CeedQFunctionGetData(qf, &data));
  Ceed ceed;
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  if (data->module) CeedCallHip(ceed, hipModuleUnload(data->module));
  CeedCallBackend(CeedFree(&data));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Hip(CeedQFunction qf) {
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Hip *data;
  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedQFunctionSetData(qf, data));
  CeedInt num_input_fields, num_output_fields;
  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields));

  // Read QFunction source
  CeedCallBackend(CeedQFunctionGetKernelName(qf, &data->qfunction_name));
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source -----\n");
  CeedCallBackend(CeedQFunctionLoadSourceToBuffer(qf, &data->qfunction_source));
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source Complete! -----\n");

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
