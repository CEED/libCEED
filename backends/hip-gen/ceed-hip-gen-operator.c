// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stddef.h>

#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-gen-operator-build.h"
#include "ceed-hip-gen.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Hip_gen(CeedOperator op) {
  CeedOperator_Hip_gen *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Hip_gen(CeedOperator op, CeedVector invec, CeedVector outvec, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Hip_gen *data;
  CeedCallBackend(CeedOperatorGetData(op, &data));
  CeedQFunction          qf;
  CeedQFunction_Hip_gen *qf_data;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
  CeedInt nelem, numinputfields, numoutputfields;
  CeedCallBackend(CeedOperatorGetNumElements(op, &nelem));
  CeedOperatorField *opinputfields, *opoutputfields;
  CeedCallBackend(CeedOperatorGetFields(op, &numinputfields, &opinputfields, &numoutputfields, &opoutputfields));
  CeedQFunctionField *qfinputfields, *qfoutputfields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qfinputfields, NULL, &qfoutputfields));
  CeedEvalMode emode;
  CeedVector   vec, outvecs[16] = {};

  // Creation of the operator
  CeedCallBackend(CeedHipGenOperatorBuild(op));

  // Input vectors
  for (CeedInt i = 0; i < numinputfields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode));
    if (emode == CEED_EVAL_WEIGHT) {  // Skip
      data->fields.in[i] = NULL;
    } else {
      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(opinputfields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = invec;
      CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.in[i]));
    }
  }

  // Output vectors
  for (CeedInt i = 0; i < numoutputfields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode));
    if (emode == CEED_EVAL_WEIGHT) {  // Skip
      data->fields.out[i] = NULL;
    } else {
      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(opoutputfields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = outvec;
      outvecs[i]    = vec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == outvecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        CeedCallBackend(CeedVectorGetArray(vec, CEED_MEM_DEVICE, &data->fields.out[i]));
      } else {
        data->fields.out[i] = data->fields.out[index];
      }
    }
  }

  // Get context data
  CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &qf_data->d_c));

  // Apply operator
  void         *opargs[] = {(void *)&nelem, &qf_data->d_c, &data->indices, &data->fields, &data->B, &data->G, &data->W};
  const CeedInt dim      = data->dim;
  const CeedInt Q1d      = data->Q1d;
  const CeedInt P1d      = data->maxP1d;
  const CeedInt thread1d = CeedIntMax(Q1d, P1d);
  CeedInt       block_sizes[3];
  CeedCallBackend(BlockGridCalculate_Hip_gen(dim, nelem, P1d, Q1d, block_sizes));
  if (dim == 1) {
    CeedInt grid      = nelem / block_sizes[2] + ((nelem / block_sizes[2] * block_sizes[2] < nelem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * thread1d * sizeof(CeedScalar);
    CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  } else if (dim == 2) {
    CeedInt grid      = nelem / block_sizes[2] + ((nelem / block_sizes[2] * block_sizes[2] < nelem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * thread1d * thread1d * sizeof(CeedScalar);
    CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  } else if (dim == 3) {
    CeedInt grid      = nelem / block_sizes[2] + ((nelem / block_sizes[2] * block_sizes[2] < nelem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * thread1d * thread1d * sizeof(CeedScalar);
    CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  }

  // Restore input arrays
  for (CeedInt i = 0; i < numinputfields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfinputfields[i], &emode));
    if (emode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedCallBackend(CeedOperatorFieldGetVector(opinputfields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = invec;
      CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->fields.in[i]));
    }
  }

  // Restore output arrays
  for (CeedInt i = 0; i < numoutputfields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qfoutputfields[i], &emode));
    if (emode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedCallBackend(CeedOperatorFieldGetVector(opoutputfields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = outvec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == outvecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        CeedCallBackend(CeedVectorRestoreArray(vec, &data->fields.out[i]));
      }
    }
  }

  // Restore context data
  CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &qf_data->d_c));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Hip_gen(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Hip_gen *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Hip_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Hip_gen));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
