// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stddef.h>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-gen-operator-build.hpp"
#include "ceed-sycl-gen.hpp"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Sycl_gen(CeedOperator op) {
  CeedOperator_Sycl_gen *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Sycl_gen(CeedOperator op, CeedVector input_vec, CeedVector output_vec, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Sycl_gen *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  // CeedQFunction          qf;
  // CeedQFunction_Sycl_gen *qf_impl;
  // CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  // CeedCallBackend(CeedQFunctionGetData(qf, &qf_impl));
  // CeedInt num_elem, num_input_fields, num_output_fields;
  // CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  // CeedOperatorField *op_input_fields, *op_output_fields;
  // CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  // CeedQFunctionField *qf_input_fields, *qf_output_fields;
  // CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  // CeedEvalMode eval_mode;
  // CeedVector   vec, output_vecs[CEED_FIELD_MAX] = {};

  // // Creation of the operator
  CeedCallBackend(CeedSyclGenOperatorBuild(op));

  // // Input vectors
  // for (CeedInt i = 0; i < num_input_fields; i++) {
  //   CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
  //   if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
  //     impl->fields.inputs[i] = NULL;
  //   } else {
  //     // Get input vector
  //     CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
  //     if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
  //     CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &impl->fields.inputs[i]));
  //   }
  // }

  // // Output vectors
  // for (CeedInt i = 0; i < num_output_fields; i++) {
  //   CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
  //   if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
  //     impl->fields.outputs[i] = NULL;
  //   } else {
  //     // Get output vector
  //     CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
  //     if (vec == CEED_VECTOR_ACTIVE) vec = output_vec;
  //     output_vecs[i] = vec;
  //     // Check for multiple output modes
  //     CeedInt index = -1;
  //     for (CeedInt j = 0; j < i; j++) {
  //       if (vec == output_vecs[j]) {
  //         index = j;
  //         break;
  //       }
  //     }
  //     if (index == -1) {
  //       CeedCallBackend(CeedVectorGetArray(vec, CEED_MEM_DEVICE, &impl->fields.outputs[i]));
  //     } else {
  //       impl->fields.outputs[i] = impl->fields.outputs[index];
  //     }
  //   }
  // }

  // // Get context data
  // CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &qf_impl->d_c));

  // // Apply operator
  // void         *opargs[]  = {(void *)&num_elem, &qf_impl->d_c, &impl->indices, &impl->fields, &impl->B, &impl->G, &impl->W};
  // const CeedInt dim       = impl->dim;
  // const CeedInt Q_1d      = impl->Q_1d;
  // const CeedInt P_1d      = impl->max_P_1d;
  // const CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);
  // CeedInt       block_sizes[3];
  // CeedCallBackend(BlockGridCalculate_Sycl_gen(dim, num_elem, P_1d, Q_1d, block_sizes));
  // if (dim == 1) {
  //   CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
  //   CeedInt sharedMem = block_sizes[2] * thread_1d * sizeof(CeedScalar);
  //   CeedCallBackend(CeedRunKernelDimSharedSycl(ceed, impl->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  // } else if (dim == 2) {
  //   CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
  //   CeedInt sharedMem = block_sizes[2] * thread_1d * thread_1d * sizeof(CeedScalar);
  //   CeedCallBackend(CeedRunKernelDimSharedSycl(ceed, impl->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  // } else if (dim == 3) {
  //   CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
  //   CeedInt sharedMem = block_sizes[2] * thread_1d * thread_1d * sizeof(CeedScalar);
  //   CeedCallBackend(CeedRunKernelDimSharedSycl(ceed, impl->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  // }

  // // Restore input arrays
  // for (CeedInt i = 0; i < num_input_fields; i++) {
  //   CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
  //   if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
  //   } else {
  //     CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
  //     if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
  //     CeedCallBackend(CeedVectorRestoreArrayRead(vec, &impl->fields.inputs[i]));
  //   }
  // }

  // // Restore output arrays
  // for (CeedInt i = 0; i < num_output_fields; i++) {
  //   CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
  //   if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
  //   } else {
  //     CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
  //     if (vec == CEED_VECTOR_ACTIVE) vec = output_vec;
  //     // Check for multiple output modes
  //     CeedInt index = -1;
  //     for (CeedInt j = 0; j < i; j++) {
  //       if (vec == output_vecs[j]) {
  //         index = j;
  //         break;
  //       }
  //     }
  //     if (index == -1) {
  //       CeedCallBackend(CeedVectorRestoreArray(vec, &impl->fields.outputs[i]));
  //     }
  //   }
  // }

  // // Restore context data
  // CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &qf_impl->d_c));

  // return CEED_ERROR_SUCCESS;
  return CeedError(ceed, CEED_ERROR_BACKEND, "CeedOperatorApplyAdd_Sycl_gen not implemented");

}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Sycl_gen(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Sycl_gen *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Sycl_gen));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Sycl_gen));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
