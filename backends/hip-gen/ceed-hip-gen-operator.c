// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-source/hip/hip-types.h>
#include <stddef.h>
#include <hip/hiprtc.h>

#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-gen-operator-build.h"
#include "ceed-hip-gen.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Hip_gen(CeedOperator op) {
  Ceed                  ceed;
  CeedOperator_Hip_gen *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  if (impl->points.num_per_elem) CeedCallHip(ceed, hipFree((void **)impl->points.num_per_elem));
  CeedCallBackend(CeedFree(&impl));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Hip_gen(CeedOperator op, CeedVector input_vec, CeedVector output_vec, CeedRequest *request) {
  bool                   is_at_points, is_tensor;
  Ceed                   ceed;
  CeedInt                num_elem, num_input_fields, num_output_fields;
  CeedEvalMode           eval_mode;
  CeedVector             output_vecs[CEED_FIELD_MAX] = {NULL};
  CeedQFunctionField    *qf_input_fields, *qf_output_fields;
  CeedQFunction_Hip_gen *qf_data;
  CeedQFunction          qf;
  CeedOperatorField     *op_input_fields, *op_output_fields;
  CeedOperator_Hip_gen  *data;

  // Check for shared bases
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  {
    bool has_shared_bases = true, is_all_tensor = true, is_all_nontensor = true;

    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedBasis basis;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
      if (basis != CEED_BASIS_NONE) {
        bool        is_tensor = true;
        const char *resource;
        char       *resource_root;
        Ceed        basis_ceed;

        CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
        is_all_tensor &= is_tensor;
        is_all_nontensor &= !is_tensor;
        CeedCallBackend(CeedBasisGetCeed(basis, &basis_ceed));
        CeedCallBackend(CeedGetResource(basis_ceed, &resource));
        CeedCallBackend(CeedGetResourceRoot(basis_ceed, resource, ":", &resource_root));
        has_shared_bases &= !strcmp(resource_root, "/gpu/hip/shared");
        CeedCallBackend(CeedFree(&resource_root));
        CeedCallBackend(CeedDestroy(&basis_ceed));
      }
      CeedCallBackend(CeedBasisDestroy(&basis));
    }

    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedBasis basis;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
      if (basis != CEED_BASIS_NONE) {
        bool        is_tensor = true;
        const char *resource;
        char       *resource_root;
        Ceed        basis_ceed;

        CeedCallBackend(CeedBasisIsTensor(basis, &is_tensor));
        is_all_tensor &= is_tensor;
        is_all_nontensor &= !is_tensor;

        CeedCallBackend(CeedBasisGetCeed(basis, &basis_ceed));
        CeedCallBackend(CeedGetResource(basis_ceed, &resource));
        CeedCallBackend(CeedGetResourceRoot(basis_ceed, resource, ":", &resource_root));
        has_shared_bases &= !strcmp(resource_root, "/gpu/hip/shared");
        CeedCallBackend(CeedFree(&resource_root));
        CeedCallBackend(CeedDestroy(&basis_ceed));
      }
      CeedCallBackend(CeedBasisDestroy(&basis));
    }
    // -- Fallback to ref if not all bases are shared
    if (!has_shared_bases || (!is_all_tensor && !is_all_nontensor)) {
      CeedOperator op_fallback;

      CeedDebug256(CeedOperatorReturnCeed(op), CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/hip/ref CeedOperator due to unsupported bases");
      CeedCallBackend(CeedOperatorGetFallback(op, &op_fallback));
      CeedCallBackend(CeedOperatorApplyAdd(op_fallback, input_vec, output_vec, request));
      return CEED_ERROR_SUCCESS;
    }
  }

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &data));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Creation of the operator
  CeedCallBackend(CeedOperatorBuildKernel_Hip_gen(op));

  // Input vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      data->fields.inputs[i] = NULL;
    } else {
      CeedVector vec;

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
        CeedCallBackend(CeedVectorGetArray(vec, CEED_MEM_DEVICE, &data->fields.outputs[i]));
      } else {
        data->fields.outputs[i] = data->fields.outputs[index];
      }
    }
  }

  // Point coordinates, if needed
  CeedCallBackend(CeedOperatorIsAtPoints(op, &is_at_points));
  if (is_at_points) {
    // Coords
    CeedVector vec;

    CeedCallBackend(CeedOperatorAtPointsGetPoints(op, NULL, &vec));
    CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->points.coords));
    CeedCallBackend(CeedVectorDestroy(&vec));

    // Points per elem
    if (num_elem != data->points.num_elem) {
      CeedInt            *points_per_elem;
      const CeedInt       num_bytes   = num_elem * sizeof(CeedInt);
      CeedElemRestriction rstr_points = NULL;

      data->points.num_elem = num_elem;
      CeedCallBackend(CeedOperatorAtPointsGetPoints(op, &rstr_points, NULL));
      CeedCallBackend(CeedCalloc(num_elem, &points_per_elem));
      for (CeedInt e = 0; e < num_elem; e++) {
        CeedInt num_points_elem;

        CeedCallBackend(CeedElemRestrictionGetNumPointsInElement(rstr_points, e, &num_points_elem));
        points_per_elem[e] = num_points_elem;
      }
      if (data->points.num_per_elem) CeedCallHip(ceed, hipFree((void **)data->points.num_per_elem));
      CeedCallHip(ceed, hipMalloc((void **)&data->points.num_per_elem, num_bytes));
      CeedCallHip(ceed, hipMemcpy((void *)data->points.num_per_elem, points_per_elem, num_bytes, hipMemcpyHostToDevice));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr_points));
      CeedCallBackend(CeedFree(&points_per_elem));
    }
  }

  // Get context data
  CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &qf_data->d_c));

  // Apply operator
  void         *opargs[]  = {(void *)&num_elem, &qf_data->d_c, &data->indices, &data->fields, &data->B, &data->G, &data->W, &data->points};
  const CeedInt dim       = data->dim;
  const CeedInt Q_1d      = data->Q_1d;
  const CeedInt P_1d      = data->max_P_1d;
  const CeedInt thread_1d = CeedIntMax(Q_1d, P_1d);

  CeedCallBackend(CeedOperatorHasTensorBases(op, &is_tensor));
  CeedInt block_sizes[3] = {thread_1d, ((!is_tensor || dim == 1) ? 1 : thread_1d), -1};

  if (is_tensor) {
    CeedCallBackend(BlockGridCalculate_Hip_gen(is_tensor ? dim : 1, num_elem, P_1d, Q_1d, block_sizes));
  } else {
    CeedInt elems_per_block = 64 * thread_1d > 256 ? 256 / thread_1d : 64;

    elems_per_block = elems_per_block > 0 ? elems_per_block : 1;
    block_sizes[2]  = elems_per_block;
  }
  if (dim == 1 || !is_tensor) {
    CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * thread_1d * sizeof(CeedScalar);

    CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  } else if (dim == 2) {
    CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * thread_1d * thread_1d * sizeof(CeedScalar);

    CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  } else if (dim == 3) {
    CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * thread_1d * thread_1d * sizeof(CeedScalar);

    CeedCallBackend(CeedRunKernelDimShared_Hip(ceed, data->op, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, opargs));
  }

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedVector vec;

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
        CeedCallBackend(CeedVectorRestoreArray(vec, &data->fields.outputs[i]));
      }
    }
  }

  // Restore point coordinates, if needed
  if (is_at_points) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorAtPointsGetPoints(op, NULL, &vec));
    CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->points.coords));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }

  // Restore context data
  CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &qf_data->d_c));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Hip_gen(CeedOperator op) {
  Ceed                  ceed;
  CeedOperator_Hip_gen *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Hip_gen));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Hip_gen));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
