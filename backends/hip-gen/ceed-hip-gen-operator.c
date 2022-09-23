// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stddef.h>
#include "ceed-hip-gen.h"
#include "ceed-hip-gen-operator-build.h"
#include "../hip/ceed-hip-compile.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Hip_gen(CeedOperator op) {
  int ierr;
  CeedOperator_Hip_gen *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Hip_gen(CeedOperator op, CeedVector input_vec,
                                        CeedVector output_vec, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Hip_gen *data;
  ierr = CeedOperatorGetData(op, &data); CeedChkBackend(ierr);
  CeedQFunction qf;
  CeedQFunction_Hip_gen *qf_data;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetData(qf, &qf_data); CeedChkBackend(ierr);
  CeedInt num_elem, num_input_fields, num_output_fields;
  ierr = CeedOperatorGetNumElements(op, &num_elem); CeedChkBackend(ierr);
  CeedOperatorField *op_input_fields, *op_output_fields;
  ierr = CeedOperatorGetFields(op, &num_input_fields, &op_input_fields,
                               &num_output_fields, &op_output_fields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL,
                                &qf_output_fields);
  CeedChkBackend(ierr);
  CeedEvalMode eval_mode;
  CeedVector vec, output_vecs[CEED_FIELD_MAX] = {};

  //Creation of the operator
  ierr = CeedHipGenOperatorBuild(op); CeedChkBackend(ierr);

  // Input vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    if (eval_mode == CEED_EVAL_WEIGHT) { // Skip
      data->fields.inputs[i] = NULL;
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(op_input_fields[i], &vec);
      CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
      ierr = CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.inputs[i]);
      CeedChkBackend(ierr);
    }
  }

  // Output vectors
  for (CeedInt i = 0; i < num_output_fields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    if (eval_mode == CEED_EVAL_WEIGHT) { // Skip
      data->fields.outputs[i] = NULL;
    } else {
      // Get output vector
      ierr = CeedOperatorFieldGetVector(op_output_fields[i], &vec);
      CeedChkBackend(ierr);
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
        ierr = CeedVectorGetArray(vec, CEED_MEM_DEVICE, &data->fields.outputs[i]);
        CeedChkBackend(ierr);
      } else {
        data->fields.outputs[i] = data->fields.outputs[index];
      }
    }
  }

  // Get context data
  ierr = CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &qf_data->d_c);
  CeedChkBackend(ierr);

  // Apply operator
  void *opargs[] = {(void *) &num_elem, &qf_data->d_c, &data->indices,
                    &data->fields, &data->B, &data->G, &data->W
                   };
  const CeedInt dim = data->dim;
  const CeedInt Q_1d = data->Q_1d;
  const CeedInt P_1d = data->max_P_1d;
  const CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
  CeedInt block_sizes[3];
  ierr = BlockGridCalculate_Hip_gen(dim, num_elem, P_1d, Q_1d, block_sizes);
  CeedChkBackend(ierr);
  if (dim==1) {
    CeedInt grid = num_elem/block_sizes[2] + ( (
                     num_elem/block_sizes[2]*block_sizes[2]<num_elem)
                   ? 1 : 0 );
    CeedInt sharedMem = block_sizes[2]*thread_1d*sizeof(CeedScalar);
    ierr = CeedRunKernelDimSharedHip(ceed, data->op, grid, block_sizes[0],
                                     block_sizes[1],
                                     block_sizes[2], sharedMem, opargs);
  } else if (dim==2) {
    CeedInt grid = num_elem/block_sizes[2] + ( (
                     num_elem/block_sizes[2]*block_sizes[2]<num_elem)
                   ? 1 : 0 );
    CeedInt sharedMem = block_sizes[2]*thread_1d*thread_1d*sizeof(CeedScalar);
    ierr = CeedRunKernelDimSharedHip(ceed, data->op, grid, block_sizes[0],
                                     block_sizes[1],
                                     block_sizes[2], sharedMem, opargs);
  } else if (dim==3) {
    CeedInt grid = num_elem/block_sizes[2] + ( (
                     num_elem/block_sizes[2]*block_sizes[2]<num_elem)
                   ? 1 : 0 );
    CeedInt sharedMem = block_sizes[2]*thread_1d*thread_1d*sizeof(CeedScalar);
    ierr = CeedRunKernelDimSharedHip(ceed, data->op, grid, block_sizes[0],
                                     block_sizes[1],
                                     block_sizes[2], sharedMem, opargs);
  }
  CeedChkBackend(ierr);

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    if (eval_mode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedOperatorFieldGetVector(op_input_fields[i], &vec);
      CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
      ierr = CeedVectorRestoreArrayRead(vec, &data->fields.inputs[i]);
      CeedChkBackend(ierr);
    }
  }

  // Restore output arrays
  for (CeedInt i = 0; i < num_output_fields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    if (eval_mode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedOperatorFieldGetVector(op_output_fields[i], &vec);
      CeedChkBackend(ierr);
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
        ierr = CeedVectorRestoreArray(vec, &data->fields.outputs[i]);
        CeedChkBackend(ierr);
      }
    }
  }

  // Restore context data
  ierr = CeedQFunctionRestoreInnerContextData(qf, &qf_data->d_c);
  CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Hip_gen(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Hip_gen *impl;

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedOperatorSetData(op, impl); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Hip_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Hip_gen); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
