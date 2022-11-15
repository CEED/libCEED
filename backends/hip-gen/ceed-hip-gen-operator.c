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
static int CeedOperatorApplyAdd_Hip_gen(CeedOperator op, CeedVector input_vec, CeedVector output_vec, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Hip_gen *data;
  CeedCallBackend(CeedOperatorGetData(op, &data));
  CeedQFunction          qf;
  CeedQFunction_Hip_gen *qf_data;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
  CeedInt num_elem, num_input_fields, num_output_fields;
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedOperatorField *op_input_fields, *op_output_fields;
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedEvalMode eval_mode;
  CeedVector   vec, output_vecs[CEED_FIELD_MAX] = {};

  // Creation of the operator
  CeedCallBackend(CeedHipGenOperatorBuild(op));

  // Input vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      data->fields.inputs[i] = NULL;
    } else {
      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
      CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.inputs[i]));
    }
  }

  // Output vectors
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      data->fields.outputs[i] = NULL;
    } else {
      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = output_vec;
      output_vecs[i] = vec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == output_vecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        CeedCallBackend(CeedVectorGetArray(vec, CEED_MEM_DEVICE, &data->fields.outputs[i]));
      } else {
        data->fields.outputs[i] = data->fields.outputs[index];
      }
    }
  }

  // Get context data
  CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &qf_data->d_c));

  // Apply operator
  void         *opargs[]  = {(void *)&num_elem, &qf_data->d_c, &data->indices, &data->fields, &data->B, &data->G, &data->W};
  const CeedInt dim       = data->dim;
  const CeedInt Q_1d      = data->Q_1d;
  const CeedInt P_1d      = data->max_P_1d;
  const CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
  CeedInt       block_sizes[3];
  CeedCallBackend(BlockGridCalculate_Hip_gen(dim, num_elem, P_1d, Q_1d, block_sizes));
  if (dim == 1) {
    CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * thread_1d * sizeof(CeedScalar);
    CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  } else if (dim == 2) {
    CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * thread_1d * thread_1d * sizeof(CeedScalar);
    CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  } else if (dim == 3) {
    CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * thread_1d * thread_1d * sizeof(CeedScalar);
    CeedCallBackend(CeedRunKernelDimSharedHip(ceed, data->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  }

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
      CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->fields.inputs[i]));
    }
  }

  // Restore output arrays
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = output_vec;
      // Check for multiple output modes
      CeedInt index = -1;
      for (CeedInt j = 0; j < i; j++) {
        if (vec == output_vecs[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        CeedCallBackend(CeedVectorRestoreArray(vec, &data->fields.outputs[i]));
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
