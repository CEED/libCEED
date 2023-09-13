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
  Ceed                    ceed;
  Ceed_Sycl              *ceed_Sycl;
  CeedInt                 num_elem, num_input_fields, num_output_fields;
  CeedEvalMode            eval_mode;
  CeedVector              output_vecs[CEED_FIELD_MAX] = {};
  CeedQFunctionField     *qf_input_fields, *qf_output_fields;
  CeedQFunction_Sycl_gen *qf_impl;
  CeedQFunction           qf;
  CeedOperatorField      *op_input_fields, *op_output_fields;
  CeedOperator_Sycl_gen  *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_Sycl));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_impl));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Creation of the operator
  CeedCallBackend(CeedOperatorBuildKernel_Sycl_gen(op));

  // Input vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      impl->fields->inputs[i] = NULL;
    } else {
      CeedVector vec;

      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
      CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &impl->fields->inputs[i]));
    }
  }

  // Output vectors
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      impl->fields->outputs[i] = NULL;
    } else {
      CeedVector vec;

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
        CeedCallBackend(CeedVectorGetArray(vec, CEED_MEM_DEVICE, &impl->fields->outputs[i]));
      } else {
        impl->fields->outputs[i] = impl->fields->outputs[index];
      }
    }
  }

  // Get context data
  CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &qf_impl->d_c));

  // Apply operator
  const CeedInt dim  = impl->dim;
  const CeedInt Q_1d = impl->Q_1d;
  const CeedInt P_1d = impl->max_P_1d;
  CeedInt       block_sizes[3], grid = 0;

  CeedCallBackend(BlockGridCalculate_Sycl_gen(dim, P_1d, Q_1d, block_sizes));
  if (dim == 1) {
    grid = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    // CeedCallBackend(CeedRunKernelDimSharedSycl(ceed, impl->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  } else if (dim == 2) {
    grid = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    // CeedCallBackend(CeedRunKernelDimSharedSycl(ceed, impl->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  } else if (dim == 3) {
    grid = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    // CeedCallBackend(CeedRunKernelDimSharedSycl(ceed, impl->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  }

  sycl::range<3>    local_range(block_sizes[2], block_sizes[1], block_sizes[0]);
  sycl::range<3>    global_range(grid * block_sizes[2], block_sizes[1], block_sizes[0]);
  sycl::nd_range<3> kernel_range(global_range, local_range);

  //-----------
  // Order queue
  sycl::event e = ceed_Sycl->sycl_queue.ext_oneapi_submit_barrier();

  CeedCallSycl(ceed, ceed_Sycl->sycl_queue.submit([&](sycl::handler &cgh) {
    cgh.depends_on(e);
    cgh.set_args(num_elem, qf_impl->d_c, impl->indices, impl->fields, impl->B, impl->G, impl->W);
    cgh.parallel_for(kernel_range, *(impl->op));
  }));
  CeedCallSycl(ceed, ceed_Sycl->sycl_queue.wait_and_throw());

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) vec = input_vec;
      CeedCallBackend(CeedVectorRestoreArrayRead(vec, &impl->fields->inputs[i]));
    }
  }

  // Restore output arrays
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedVector vec;

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
        CeedCallBackend(CeedVectorRestoreArray(vec, &impl->fields->outputs[i]));
      }
    }
  }

  // Restore context data
  CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &qf_impl->d_c));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Sycl_gen(CeedOperator op) {
  Ceed                   ceed;
  Ceed_Sycl             *sycl_data;
  CeedOperator_Sycl_gen *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  impl->indices = sycl::malloc_device<FieldsInt_Sycl>(1, sycl_data->sycl_device, sycl_data->sycl_context);
  impl->fields  = sycl::malloc_host<Fields_Sycl>(1, sycl_data->sycl_context);
  impl->B       = sycl::malloc_device<Fields_Sycl>(1, sycl_data->sycl_device, sycl_data->sycl_context);
  impl->G       = sycl::malloc_device<Fields_Sycl>(1, sycl_data->sycl_device, sycl_data->sycl_context);
  impl->W       = sycl::malloc_device<CeedScalar>(1, sycl_data->sycl_device, sycl_data->sycl_context);

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Sycl_gen));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Sycl_gen));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
