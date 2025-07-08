// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  bool                  is_composite;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt num_suboperators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    for (CeedInt i = 0; i < num_suboperators; i++) {
      if (impl->streams[i]) CeedCallHip(ceed, hipStreamDestroy(impl->streams[i]));
      impl->streams[i] = NULL;
    }
  }
  if (impl->module) CeedCallHip(ceed, hipModuleUnload(impl->module));
  if (impl->module_assemble_full) CeedCallHip(ceed, hipModuleUnload(impl->module_assemble_full));
  if (impl->module_assemble_diagonal) CeedCallHip(ceed, hipModuleUnload(impl->module_assemble_diagonal));
  if (impl->module_assemble_qfunction) CeedCallHip(ceed, hipModuleUnload(impl->module_assemble_qfunction));
  if (impl->points.num_per_elem) CeedCallHip(ceed, hipFree((void **)impl->points.num_per_elem));
  CeedCallBackend(CeedFree(&impl));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAddCore_Hip_gen(CeedOperator op, hipStream_t stream, const CeedScalar *input_arr, CeedScalar *output_arr,
                                            bool *is_run_good, CeedRequest *request) {
  bool                   is_at_points, is_tensor;
  Ceed                   ceed;
  CeedInt                num_elem, num_input_fields, num_output_fields;
  CeedEvalMode           eval_mode;
  CeedQFunctionField    *qf_input_fields, *qf_output_fields;
  CeedQFunction_Hip_gen *qf_data;
  CeedQFunction          qf;
  CeedOperatorField     *op_input_fields, *op_output_fields;
  CeedOperator_Hip_gen  *data;

  // Creation of the operator
  CeedCallBackend(CeedOperatorBuildKernel_Hip_gen(op, is_run_good));
  if (!(*is_run_good)) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &data));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Input vectors
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      data->fields.inputs[i] = NULL;
    } else {
      bool       is_active;
      CeedVector vec;

      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      is_active = vec == CEED_VECTOR_ACTIVE;
      if (is_active) data->fields.inputs[i] = input_arr;
      else CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.inputs[i]));
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
  }

  // Output vectors
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      data->fields.outputs[i] = NULL;
    } else {
      bool       is_active;
      CeedVector vec;

      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      is_active = vec == CEED_VECTOR_ACTIVE;
      if (is_active) data->fields.outputs[i] = output_arr;
      else CeedCallBackend(CeedVectorGetArray(vec, CEED_MEM_DEVICE, &data->fields.outputs[i]));
      CeedCallBackend(CeedVectorDestroy(&vec));
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
  void *opargs[] = {(void *)&num_elem, &qf_data->d_c, &data->indices, &data->fields, &data->B, &data->G, &data->W, &data->points};

  CeedCallBackend(CeedOperatorHasTensorBases(op, &is_tensor));
  CeedInt block_sizes[3] = {data->thread_1d, ((!is_tensor || data->dim == 1) ? 1 : data->thread_1d), -1};

  if (is_tensor) {
    CeedCallBackend(BlockGridCalculate_Hip_gen(data->dim, num_elem, data->max_P_1d, data->Q_1d, block_sizes));
    if (is_at_points) block_sizes[2] = 1;
  } else {
    CeedInt elems_per_block = 64 * data->thread_1d > 256 ? 256 / data->thread_1d : 64;

    elems_per_block = elems_per_block > 0 ? elems_per_block : 1;
    block_sizes[2]  = elems_per_block;
  }
  if (data->dim == 1 || !is_tensor) {
    CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * data->thread_1d * sizeof(CeedScalar);

    CeedCallBackend(
        CeedTryRunKernelDimShared_Hip(ceed, data->op, stream, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, is_run_good, opargs));
  } else if (data->dim == 2) {
    CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * data->thread_1d * data->thread_1d * sizeof(CeedScalar);

    CeedCallBackend(
        CeedTryRunKernelDimShared_Hip(ceed, data->op, stream, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, is_run_good, opargs));
  } else if (data->dim == 3) {
    CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
    CeedInt sharedMem = block_sizes[2] * data->thread_1d * data->thread_1d * sizeof(CeedScalar);

    CeedCallBackend(
        CeedTryRunKernelDimShared_Hip(ceed, data->op, stream, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem, is_run_good, opargs));
  }

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      bool       is_active;
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      is_active = vec == CEED_VECTOR_ACTIVE;
      if (!is_active) CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->fields.inputs[i]));
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
  }

  // Restore output arrays
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      bool       is_active;
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      is_active = vec == CEED_VECTOR_ACTIVE;
      if (!is_active) CeedCallBackend(CeedVectorRestoreArray(vec, &data->fields.outputs[i]));
      CeedCallBackend(CeedVectorDestroy(&vec));
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

  // Cleanup
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  if (!(*is_run_good)) data->use_fallback = true;
  return CEED_ERROR_SUCCESS;
}

static int CeedOperatorApplyAdd_Hip_gen(CeedOperator op, CeedVector input_vec, CeedVector output_vec, CeedRequest *request) {
  bool              is_run_good = false;
  const CeedScalar *input_arr   = NULL;
  CeedScalar       *output_arr  = NULL;

  // Try to run kernel
  if (input_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(input_vec, CEED_MEM_DEVICE, &input_arr));
  if (output_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArray(output_vec, CEED_MEM_DEVICE, &output_arr));
  CeedCallBackend(CeedOperatorApplyAddCore_Hip_gen(op, NULL, input_arr, output_arr, &is_run_good, request));
  if (input_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorRestoreArrayRead(input_vec, &input_arr));
  if (output_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorRestoreArray(output_vec, &output_arr));

  // Fallback on unsuccessful run
  if (!is_run_good) {
    CeedOperator op_fallback;

    CeedDebug256(CeedOperatorReturnCeed(op), CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/hip/ref CeedOperator");
    CeedCallBackend(CeedOperatorGetFallback(op, &op_fallback));
    CeedCallBackend(CeedOperatorApplyAdd(op_fallback, input_vec, output_vec, request));
  }
  return CEED_ERROR_SUCCESS;
}

static int CeedOperatorApplyAddComposite_Hip_gen(CeedOperator op, CeedVector input_vec, CeedVector output_vec, CeedRequest *request) {
  bool                  is_run_good[CEED_COMPOSITE_MAX] = {false};
  CeedInt               num_suboperators;
  const CeedScalar     *input_arr  = NULL;
  CeedScalar           *output_arr = NULL;
  Ceed                  ceed;
  CeedOperator_Hip_gen *impl;
  CeedOperator         *sub_operators;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
  CeedCallBackend(CeedCompositeOperatorGetSubList(op, &sub_operators));
  if (input_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArrayRead(input_vec, CEED_MEM_DEVICE, &input_arr));
  if (output_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorGetArray(output_vec, CEED_MEM_DEVICE, &output_arr));
  for (CeedInt i = 0; i < num_suboperators; i++) {
    CeedInt num_elem = 0;

    CeedCallBackend(CeedOperatorGetNumElements(sub_operators[i], &num_elem));
    if (num_elem > 0) {
      if (!impl->streams[i]) CeedCallHip(ceed, hipStreamCreate(&impl->streams[i]));
      CeedCallBackend(CeedOperatorApplyAddCore_Hip_gen(sub_operators[i], impl->streams[i], input_arr, output_arr, &is_run_good[i], request));
    } else {
      is_run_good[i] = true;
    }
  }

  for (CeedInt i = 0; i < num_suboperators; i++) {
    if (impl->streams[i]) {
      if (is_run_good[i]) CeedCallHip(ceed, hipStreamSynchronize(impl->streams[i]));
    }
  }
  if (input_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorRestoreArrayRead(input_vec, &input_arr));
  if (output_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorRestoreArray(output_vec, &output_arr));
  CeedCallHip(ceed, hipDeviceSynchronize());

  // Fallback on unsuccessful run
  for (CeedInt i = 0; i < num_suboperators; i++) {
    if (!is_run_good[i]) {
      CeedOperator op_fallback;

      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/hip/ref CeedOperator");
      CeedCallBackend(CeedOperatorGetFallback(sub_operators[i], &op_fallback));
      CeedCallBackend(CeedOperatorApplyAdd(op_fallback, input_vec, output_vec, request));
    }
  }
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// QFunction assembly
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionCore_Hip_gen(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                           CeedRequest *request) {
  Ceed                  ceed;
  CeedOperator_Hip_gen *data;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &data));

  // Build the assembly kernel
  if (!data->assemble_qfunction && !data->use_assembly_fallback) {
    bool is_build_good = false;

    CeedCallBackend(CeedOperatorBuildKernel_Hip_gen(op, &is_build_good));
    if (is_build_good) CeedCallBackend(CeedOperatorBuildKernelLinearAssembleQFunction_Hip_gen(op, &is_build_good));
    if (!is_build_good) data->use_assembly_fallback = true;
  }

  // Try assembly
  if (!data->use_assembly_fallback) {
    bool                   is_run_good = true;
    Ceed_Hip              *hip_data;
    CeedInt                num_elem, num_input_fields, num_output_fields;
    CeedEvalMode           eval_mode;
    CeedScalar            *assembled_array;
    CeedQFunctionField    *qf_input_fields, *qf_output_fields;
    CeedQFunction_Hip_gen *qf_data;
    CeedQFunction          qf;
    CeedOperatorField     *op_input_fields, *op_output_fields;

    CeedCallBackend(CeedGetData(ceed, &hip_data));
    CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
    CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
    CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
    CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

    // Input vectors
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
        data->fields.inputs[i] = NULL;
      } else {
        bool       is_active;
        CeedVector vec;

        // Get input vector
        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        is_active = vec == CEED_VECTOR_ACTIVE;
        if (is_active) data->fields.inputs[i] = NULL;
        else CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.inputs[i]));
        CeedCallBackend(CeedVectorDestroy(&vec));
      }
    }

    // Get context data
    CeedCallBackend(CeedQFunctionGetInnerContextData(qf, CEED_MEM_DEVICE, &qf_data->d_c));

    // Build objects if needed
    if (build_objects) {
      CeedInt qf_size_in = 0, qf_size_out = 0, Q;

      // Count number of active input fields
      {
        for (CeedInt i = 0; i < num_input_fields; i++) {
          CeedInt    field_size;
          CeedVector vec;

          // Get input vector
          CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
          // Check if active input
          if (vec == CEED_VECTOR_ACTIVE) {
            CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
            qf_size_in += field_size;
          }
          CeedCallBackend(CeedVectorDestroy(&vec));
        }
        CeedCheck(qf_size_in > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
      }

      // Count number of active output fields
      {
        for (CeedInt i = 0; i < num_output_fields; i++) {
          CeedInt    field_size;
          CeedVector vec;

          // Get output vector
          CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
          // Check if active output
          if (vec == CEED_VECTOR_ACTIVE) {
            CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &field_size));
            qf_size_out += field_size;
          }
          CeedCallBackend(CeedVectorDestroy(&vec));
        }
        CeedCheck(qf_size_out > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
      }
      CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));

      // Actually build objects now
      const CeedSize l_size     = (CeedSize)num_elem * Q * qf_size_in * qf_size_out;
      CeedInt        strides[3] = {1, num_elem * Q, Q}; /* *NOPAD* */

      // Create output restriction
      CeedCallBackend(CeedElemRestrictionCreateStrided(ceed, num_elem, Q, qf_size_in * qf_size_out,
                                                       (CeedSize)qf_size_in * (CeedSize)qf_size_out * (CeedSize)num_elem * (CeedSize)Q, strides,
                                                       rstr));
      // Create assembled vector
      CeedCallBackend(CeedVectorCreate(ceed, l_size, assembled));
    }

    // Assembly array
    CeedCallBackend(CeedVectorGetArrayWrite(*assembled, CEED_MEM_DEVICE, &assembled_array));

    // Assemble QFunction
    bool  is_tensor = false;
    void *opargs[] = {(void *)&num_elem, &qf_data->d_c, &data->indices, &data->fields, &data->B, &data->G, &data->W, &data->points, &assembled_array};

    CeedCallBackend(CeedOperatorHasTensorBases(op, &is_tensor));
    CeedInt block_sizes[3] = {data->thread_1d, ((!is_tensor || data->dim == 1) ? 1 : data->thread_1d), -1};

    if (is_tensor) {
      CeedCallBackend(BlockGridCalculate_Hip_gen(data->dim, num_elem, data->max_P_1d, data->Q_1d, block_sizes));
    } else {
      CeedInt elems_per_block = 64 * data->thread_1d > 256 ? 256 / data->thread_1d : 64;

      elems_per_block = elems_per_block > 0 ? elems_per_block : 1;
      block_sizes[2]  = elems_per_block;
    }
    if (data->dim == 1 || !is_tensor) {
      CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
      CeedInt sharedMem = block_sizes[2] * data->thread_1d * sizeof(CeedScalar);

      CeedCallBackend(CeedTryRunKernelDimShared_Hip(ceed, data->assemble_qfunction, NULL, grid, block_sizes[0], block_sizes[1], block_sizes[2],
                                                    sharedMem, &is_run_good, opargs));
    } else if (data->dim == 2) {
      CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
      CeedInt sharedMem = block_sizes[2] * data->thread_1d * data->thread_1d * sizeof(CeedScalar);

      CeedCallBackend(CeedTryRunKernelDimShared_Hip(ceed, data->assemble_qfunction, NULL, grid, block_sizes[0], block_sizes[1], block_sizes[2],
                                                    sharedMem, &is_run_good, opargs));
    } else if (data->dim == 3) {
      CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
      CeedInt sharedMem = block_sizes[2] * data->thread_1d * data->thread_1d * sizeof(CeedScalar);

      CeedCallBackend(CeedTryRunKernelDimShared_Hip(ceed, data->assemble_qfunction, NULL, grid, block_sizes[0], block_sizes[1], block_sizes[2],
                                                    sharedMem, &is_run_good, opargs));
    }

    // Restore input arrays
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      } else {
        bool       is_active;
        CeedVector vec;

        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        is_active = vec == CEED_VECTOR_ACTIVE;
        if (!is_active) CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->fields.inputs[i]));
        CeedCallBackend(CeedVectorDestroy(&vec));
      }
    }

    // Restore context data
    CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &qf_data->d_c));

    // Restore assembly array
    CeedCallBackend(CeedVectorRestoreArray(*assembled, &assembled_array));

    // Cleanup
    CeedCallBackend(CeedQFunctionDestroy(&qf));
    if (!is_run_good) {
      data->use_assembly_fallback = true;
      if (build_objects) {
        CeedCallBackend(CeedVectorDestroy(assembled));
        CeedCallBackend(CeedElemRestrictionDestroy(rstr));
      }
    }
  }
  CeedCallBackend(CeedDestroy(&ceed));

  // Fallback, if needed
  if (data->use_assembly_fallback) {
    CeedOperator op_fallback;

    CeedDebug256(CeedOperatorReturnCeed(op), CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/hip/ref CeedOperator");
    CeedCallBackend(CeedOperatorGetFallback(op, &op_fallback));
    CeedCallBackend(CeedOperatorFallbackLinearAssembleQFunctionBuildOrUpdate(op_fallback, assembled, rstr, request));
    return CEED_ERROR_SUCCESS;
  }
  return CEED_ERROR_SUCCESS;
}

static int CeedOperatorLinearAssembleQFunction_Hip_gen(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip_gen(op, true, assembled, rstr, request);
}

static int CeedOperatorLinearAssembleQFunctionUpdate_Hip_gen(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip_gen(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// AtPoints diagonal assembly
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonalAtPoints_Hip_gen(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  Ceed                  ceed;
  CeedOperator_Hip_gen *data;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &data));

  // Build the assembly kernel
  if (!data->assemble_diagonal && !data->use_assembly_fallback) {
    bool                     is_build_good = false;
    CeedInt                  num_active_bases_in, num_active_bases_out;
    CeedOperatorAssemblyData assembly_data;

    CeedCallBackend(CeedOperatorGetOperatorAssemblyData(op, &assembly_data));
    CeedCallBackend(
        CeedOperatorAssemblyDataGetEvalModes(assembly_data, &num_active_bases_in, NULL, NULL, NULL, &num_active_bases_out, NULL, NULL, NULL, NULL));
    if (num_active_bases_in == num_active_bases_out) {
      CeedCallBackend(CeedOperatorBuildKernel_Hip_gen(op, &is_build_good));
      if (is_build_good) CeedCallBackend(CeedOperatorBuildKernelDiagonalAssemblyAtPoints_Hip_gen(op, &is_build_good));
    }
    if (!is_build_good) data->use_assembly_fallback = true;
  }

  // Try assembly
  if (!data->use_assembly_fallback) {
    bool                   is_run_good = true;
    Ceed_Hip              *hip_data;
    CeedInt                num_elem, num_input_fields, num_output_fields;
    CeedEvalMode           eval_mode;
    CeedScalar            *assembled_array;
    CeedQFunctionField    *qf_input_fields, *qf_output_fields;
    CeedQFunction_Hip_gen *qf_data;
    CeedQFunction          qf;
    CeedOperatorField     *op_input_fields, *op_output_fields;

    CeedCallBackend(CeedGetData(ceed, &hip_data));
    CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
    CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
    CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
    CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

    // Input vectors
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
        data->fields.inputs[i] = NULL;
      } else {
        bool       is_active;
        CeedVector vec;

        // Get input vector
        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        is_active = vec == CEED_VECTOR_ACTIVE;
        if (is_active) data->fields.inputs[i] = NULL;
        else CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.inputs[i]));
        CeedCallBackend(CeedVectorDestroy(&vec));
      }
    }

    // Point coordinates
    {
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

    // Assembly array
    CeedCallBackend(CeedVectorGetArray(assembled, CEED_MEM_DEVICE, &assembled_array));

    // Assemble diagonal
    void *opargs[] = {(void *)&num_elem, &qf_data->d_c, &data->indices, &data->fields, &data->B, &data->G, &data->W, &data->points, &assembled_array};

    CeedInt block_sizes[3] = {data->thread_1d, (data->dim == 1 ? 1 : data->thread_1d), -1};

    CeedCallBackend(BlockGridCalculate_Hip_gen(data->dim, num_elem, data->max_P_1d, data->Q_1d, block_sizes));
    block_sizes[2] = 1;
    if (data->dim == 1) {
      CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
      CeedInt sharedMem = block_sizes[2] * data->thread_1d * sizeof(CeedScalar);

      CeedCallBackend(CeedTryRunKernelDimShared_Hip(ceed, data->assemble_diagonal, NULL, grid, block_sizes[0], block_sizes[1], block_sizes[2],
                                                    sharedMem, &is_run_good, opargs));
    } else if (data->dim == 2) {
      CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
      CeedInt sharedMem = block_sizes[2] * data->thread_1d * data->thread_1d * sizeof(CeedScalar);

      CeedCallBackend(CeedTryRunKernelDimShared_Hip(ceed, data->assemble_diagonal, NULL, grid, block_sizes[0], block_sizes[1], block_sizes[2],
                                                    sharedMem, &is_run_good, opargs));
    } else if (data->dim == 3) {
      CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
      CeedInt sharedMem = block_sizes[2] * data->thread_1d * data->thread_1d * sizeof(CeedScalar);

      CeedCallBackend(CeedTryRunKernelDimShared_Hip(ceed, data->assemble_diagonal, NULL, grid, block_sizes[0], block_sizes[1], block_sizes[2],
                                                    sharedMem, &is_run_good, opargs));
    }
    CeedCallHip(ceed, hipDeviceSynchronize());

    // Restore input arrays
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      } else {
        bool       is_active;
        CeedVector vec;

        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        is_active = vec == CEED_VECTOR_ACTIVE;
        if (!is_active) CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->fields.inputs[i]));
        CeedCallBackend(CeedVectorDestroy(&vec));
      }
    }

    // Restore point coordinates
    {
      CeedVector vec;

      CeedCallBackend(CeedOperatorAtPointsGetPoints(op, NULL, &vec));
      CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->points.coords));
      CeedCallBackend(CeedVectorDestroy(&vec));
    }

    // Restore context data
    CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &qf_data->d_c));

    // Restore assembly array
    CeedCallBackend(CeedVectorRestoreArray(assembled, &assembled_array));

    // Cleanup
    CeedCallBackend(CeedQFunctionDestroy(&qf));
    if (!is_run_good) data->use_assembly_fallback = true;
  }
  CeedCallBackend(CeedDestroy(&ceed));

  // Fallback, if needed
  if (data->use_assembly_fallback) {
    CeedOperator op_fallback;

    CeedDebug256(CeedOperatorReturnCeed(op), CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/hip/ref CeedOperator");
    CeedCallBackend(CeedOperatorGetFallback(op, &op_fallback));
    CeedCallBackend(CeedOperatorLinearAssembleAddDiagonal(op_fallback, assembled, request));
    return CEED_ERROR_SUCCESS;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// AtPoints full assembly
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssembleAtPoints_Hip_gen(CeedOperator op, CeedInt offset, CeedVector assembled) {
  Ceed                  ceed;
  CeedOperator_Hip_gen *data;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &data));

  // Build the assembly kernel
  if (!data->assemble_full && !data->use_assembly_fallback) {
    bool                     is_build_good = false;
    CeedInt                  num_active_bases_in, num_active_bases_out;
    CeedOperatorAssemblyData assembly_data;

    CeedCallBackend(CeedOperatorGetOperatorAssemblyData(op, &assembly_data));
    CeedCallBackend(
        CeedOperatorAssemblyDataGetEvalModes(assembly_data, &num_active_bases_in, NULL, NULL, NULL, &num_active_bases_out, NULL, NULL, NULL, NULL));
    if (num_active_bases_in == num_active_bases_out) {
      CeedCallBackend(CeedOperatorBuildKernel_Hip_gen(op, &is_build_good));
      if (is_build_good) CeedCallBackend(CeedOperatorBuildKernelFullAssemblyAtPoints_Hip_gen(op, &is_build_good));
    }
    if (!is_build_good) {
      CeedDebug(ceed, "Single Operator Assemble at Points compile failed, using fallback\n");
      data->use_assembly_fallback = true;
    }
  }

  // Try assembly
  if (!data->use_assembly_fallback) {
    bool                   is_run_good = true;
    Ceed_Hip              *Hip_data;
    CeedInt                num_elem, num_input_fields, num_output_fields;
    CeedEvalMode           eval_mode;
    CeedScalar            *assembled_array;
    CeedQFunctionField    *qf_input_fields, *qf_output_fields;
    CeedQFunction_Hip_gen *qf_data;
    CeedQFunction          qf;
    CeedOperatorField     *op_input_fields, *op_output_fields;

    CeedCallBackend(CeedGetData(ceed, &Hip_data));
    CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
    CeedCallBackend(CeedQFunctionGetData(qf, &qf_data));
    CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
    CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
    CeedDebug(ceed, "Running single operator assemble for /gpu/hip/gen\n");

    // Input vectors
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
        data->fields.inputs[i] = NULL;
      } else {
        bool       is_active;
        CeedVector vec;

        // Get input vector
        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        is_active = vec == CEED_VECTOR_ACTIVE;
        if (is_active) data->fields.inputs[i] = NULL;
        else CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, &data->fields.inputs[i]));
        CeedCallBackend(CeedVectorDestroy(&vec));
      }
    }

    // Point coordinates
    {
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

    // Assembly array
    CeedCallBackend(CeedVectorGetArray(assembled, CEED_MEM_DEVICE, &assembled_array));
    CeedScalar *assembled_offset_array = &assembled_array[offset];

    // Assemble diagonal
    void *opargs[] = {(void *)&num_elem, &qf_data->d_c, &data->indices, &data->fields,          &data->B,
                      &data->G,          &data->W,      &data->points,  &assembled_offset_array};

    CeedInt block_sizes[3] = {data->thread_1d, (data->dim == 1 ? 1 : data->thread_1d), -1};

    CeedCallBackend(BlockGridCalculate_Hip_gen(data->dim, num_elem, data->max_P_1d, data->Q_1d, block_sizes));
    block_sizes[2] = 1;
    if (data->dim == 1) {
      CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
      CeedInt sharedMem = block_sizes[2] * data->thread_1d * sizeof(CeedScalar);

      CeedCallBackend(CeedTryRunKernelDimShared_Hip(ceed, data->assemble_full, NULL, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem,
                                                    &is_run_good, opargs));
    } else if (data->dim == 2) {
      CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
      CeedInt sharedMem = block_sizes[2] * data->thread_1d * data->thread_1d * sizeof(CeedScalar);

      CeedCallBackend(CeedTryRunKernelDimShared_Hip(ceed, data->assemble_full, NULL, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem,
                                                    &is_run_good, opargs));
    } else if (data->dim == 3) {
      CeedInt grid      = num_elem / block_sizes[2] + ((num_elem / block_sizes[2] * block_sizes[2] < num_elem) ? 1 : 0);
      CeedInt sharedMem = block_sizes[2] * data->thread_1d * data->thread_1d * sizeof(CeedScalar);

      CeedCallBackend(CeedTryRunKernelDimShared_Hip(ceed, data->assemble_full, NULL, grid, block_sizes[0], block_sizes[1], block_sizes[2], sharedMem,
                                                    &is_run_good, opargs));
    }
    CeedCallHip(ceed, hipDeviceSynchronize());

    // Restore input arrays
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
      } else {
        bool       is_active;
        CeedVector vec;

        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        is_active = vec == CEED_VECTOR_ACTIVE;
        if (!is_active) CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->fields.inputs[i]));
        CeedCallBackend(CeedVectorDestroy(&vec));
      }
    }

    // Restore point coordinates
    {
      CeedVector vec;

      CeedCallBackend(CeedOperatorAtPointsGetPoints(op, NULL, &vec));
      CeedCallBackend(CeedVectorRestoreArrayRead(vec, &data->points.coords));
      CeedCallBackend(CeedVectorDestroy(&vec));
    }

    // Restore context data
    CeedCallBackend(CeedQFunctionRestoreInnerContextData(qf, &qf_data->d_c));

    // Restore assembly array
    CeedCallBackend(CeedVectorRestoreArray(assembled, &assembled_array));

    // Cleanup
    CeedCallBackend(CeedQFunctionDestroy(&qf));
    if (!is_run_good) {
      CeedDebug(ceed, "Single Operator Assemble at Points run failed, using fallback\n");
      data->use_assembly_fallback = true;
    }
  }
  CeedCallBackend(CeedDestroy(&ceed));

  // Fallback, if needed
  if (data->use_assembly_fallback) {
    CeedOperator op_fallback;

    CeedDebug256(CeedOperatorReturnCeed(op), CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/hip/ref CeedOperator");
    CeedCallBackend(CeedOperatorGetFallback(op, &op_fallback));
    CeedCallBackend(CeedSingleOperatorAssemble(op_fallback, offset, assembled));
    return CEED_ERROR_SUCCESS;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Hip_gen(CeedOperator op) {
  bool                  is_composite, is_at_points;
  Ceed                  ceed;
  CeedOperator_Hip_gen *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAddComposite", CeedOperatorApplyAddComposite_Hip_gen));
  } else {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Hip_gen));
  }
  CeedCall(CeedOperatorIsAtPoints(op, &is_at_points));
  if (is_at_points) {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonalAtPoints_Hip_gen));
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleSingle", CeedSingleOperatorAssembleAtPoints_Hip_gen));
  }
  if (!is_at_points) {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Hip_gen));
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Hip_gen));
  }
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Hip_gen));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
