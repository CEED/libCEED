// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
#include "ceed-hip-ref.h"
#include "ceed-hip-ref-qfunction-load.h"
#include "../hip/ceed-hip-compile.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Hip(CeedQFunction qf, CeedInt Q,
                                  CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);

  // Build and compile kernel, if not done
  ierr = CeedHipBuildQFunction(qf); CeedChkBackend(ierr);

  CeedQFunction_Hip *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  Ceed_Hip *ceed_Hip;
  ierr = CeedGetData(ceed, &ceed_Hip); CeedChkBackend(ierr);
  CeedInt num_input_fields, num_output_fields;
  ierr = CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields);
  CeedChkBackend(ierr);
  const int blocksize = ceed_Hip->opt_block_size;

  // Read vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    ierr = CeedVectorGetArrayReadGeneric(U[i], CEED_MEM_DEVICE, CEED_SCALAR_TYPE,
                                         &data->fields.inputs[i]);
    CeedChkBackend(ierr);
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    ierr = CeedVectorGetArrayWriteGeneric(V[i], CEED_MEM_DEVICE, CEED_SCALAR_TYPE,
                                          &data->fields.outputs[i]);
    CeedChkBackend(ierr);
  }

  // Get context data
  ierr = CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &data->d_c);
  CeedChkBackend(ierr);

  // Run kernel
  void *args[] = {&data->d_c, (void *) &Q, &data->fields};
  ierr = CeedRunKernelHip(ceed, data->QFunction, CeedDivUpInt(Q, blocksize),
                          blocksize, args); CeedChkBackend(ierr);

  // Restore vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    ierr = CeedVectorRestoreArrayReadGeneric(U[i], &data->fields.inputs[i]);
    CeedChkBackend(ierr);
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    ierr = CeedVectorRestoreArrayGeneric(V[i], &data->fields.outputs[i]);
    CeedChkBackend(ierr);
  }

  // Restore context
  ierr = CeedQFunctionRestoreInnerContextData(qf, &data->d_c);
  CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Hip(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Hip *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);
  if  (data->module)
    CeedChk_Hip(ceed, hipModuleUnload(data->module));
  ierr = CeedFree(&data); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Hip(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Hip *data;
  ierr = CeedCalloc(1,&data); CeedChkBackend(ierr);
  ierr = CeedQFunctionSetData(qf, data); CeedChkBackend(ierr);
  CeedInt num_input_fields, num_output_fields;
  ierr = CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields);
  CeedChkBackend(ierr);

  // Read QFunction source
  ierr = CeedQFunctionGetKernelName(qf, &data->qfunction_name);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source -----\n");
  ierr = CeedQFunctionLoadSourceToBuffer(qf, &data->qfunction_source);
  CeedChkBackend(ierr);
  CeedDebug256(ceed, 2, "----- Loading QFunction User Source Complete! -----\n");

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
