// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <hip/hip_runtime.h>

#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Hip(CeedOperator op) {
  CeedOperator_Hip *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Apply data
  for (CeedInt i = 0; i < impl->num_inputs + impl->num_outputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs[i]));
  }
  CeedCallBackend(CeedFree(&impl->e_vecs));

  for (CeedInt i = 0; i < impl->num_inputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_in[i]));
  }
  CeedCallBackend(CeedFree(&impl->q_vecs_in));

  for (CeedInt i = 0; i < impl->num_outputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_out[i]));
  }
  CeedCallBackend(CeedFree(&impl->q_vecs_out));

  // QFunction assembly data
  for (CeedInt i = 0; i < impl->num_active_in; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->qf_active_in[i]));
  }
  CeedCallBackend(CeedFree(&impl->qf_active_in));

  // Diag data
  if (impl->diag) {
    Ceed ceed;

    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    if (impl->diag->module) {
      CeedCallHip(ceed, hipModuleUnload(impl->diag->module));
    }
    if (impl->diag->module_point_block) {
      CeedCallHip(ceed, hipModuleUnload(impl->diag->module_point_block));
    }
    CeedCallHip(ceed, hipFree(impl->diag->d_eval_modes_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_eval_modes_out));
    CeedCallHip(ceed, hipFree(impl->diag->d_identity));
    CeedCallHip(ceed, hipFree(impl->diag->d_interp_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_interp_out));
    CeedCallHip(ceed, hipFree(impl->diag->d_grad_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_grad_out));
    CeedCallHip(ceed, hipFree(impl->diag->d_div_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_div_out));
    CeedCallHip(ceed, hipFree(impl->diag->d_curl_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_curl_out));
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->diag->diag_rstr));
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->diag->point_block_diag_rstr));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->elem_diag));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->point_block_elem_diag));
  }
  CeedCallBackend(CeedFree(&impl->diag));

  if (impl->asmb) {
    Ceed ceed;

    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    CeedCallHip(ceed, hipModuleUnload(impl->asmb->module));
    CeedCallHip(ceed, hipFree(impl->asmb->d_B_in));
    CeedCallHip(ceed, hipFree(impl->asmb->d_B_out));
  }
  CeedCallBackend(CeedFree(&impl->asmb));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Hip(CeedQFunction qf, CeedOperator op, bool is_input, CeedVector *e_vecs, CeedVector *q_vecs, CeedInt start_e,
                                       CeedInt num_fields, CeedInt Q, CeedInt num_elem) {
  Ceed                ceed;
  CeedQFunctionField *qf_fields;
  CeedOperatorField  *op_fields;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  if (is_input) {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  } else {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  }

  // Loop over fields
  for (CeedInt i = 0; i < num_fields; i++) {
    bool         is_strided = false, skip_restriction = false;
    CeedSize     q_size;
    CeedInt      size;
    CeedEvalMode eval_mode;
    CeedBasis    basis;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {
      CeedElemRestriction elem_rstr;

      // Check whether this field can skip the element restriction:
      // Must be passive input, with eval_mode NONE, and have a strided restriction with CEED_STRIDES_BACKEND.
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &elem_rstr));

      // First, check whether the field is input or output:
      if (is_input) {
        CeedVector vec;

        // Check for passive input
        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec));
        if (vec != CEED_VECTOR_ACTIVE) {
          // Check eval_mode
          if (eval_mode == CEED_EVAL_NONE) {
            // Check for strided restriction
            CeedCallBackend(CeedElemRestrictionIsStrided(elem_rstr, &is_strided));
            if (is_strided) {
              // Check if vector is already in preferred backend ordering
              CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &skip_restriction));
            }
          }
        }
      }
      if (skip_restriction) {
        // We do not need an E-Vector, but will use the input field vector's data directly in the operator application.
        e_vecs[i + start_e] = NULL;
      } else {
        CeedCallBackend(CeedElemRestrictionCreateVector(elem_rstr, NULL, &e_vecs[i + start_e]));
      }
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        q_size = (CeedSize)num_elem * Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        q_size = (CeedSize)num_elem * Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_WEIGHT:  // Only on input fields
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        q_size = (CeedSize)num_elem * Q;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_vecs[i]));
        break;
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// CeedOperator needs to connect all the named fields (be they active or passive) to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Hip(CeedOperator op) {
  Ceed                ceed;
  bool                is_setup_done;
  CeedInt             Q, num_elem, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Allocate
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->e_vecs));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_out));
  impl->num_inputs  = num_input_fields;
  impl->num_outputs = num_output_fields;

  // Set up infield and outfield e_vecs and q_vecs
  // Infields
  CeedCallBackend(CeedOperatorSetupFields_Hip(qf, op, true, impl->e_vecs, impl->q_vecs_in, 0, num_input_fields, Q, num_elem));
  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Hip(qf, op, false, impl->e_vecs, impl->q_vecs_out, num_input_fields, num_output_fields, Q, num_elem));

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Hip(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                              CeedVector in_vec, const bool skip_active, CeedScalar *e_data[2 * CEED_FIELD_MAX],
                                              CeedOperator_Hip *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode        eval_mode;
    CeedVector          vec;
    CeedElemRestriction elem_rstr;

    // Get input vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      if (skip_active) continue;
      else vec = in_vec;
    }

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      // Get input element restriction
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
      if (vec == CEED_VECTOR_ACTIVE) vec = in_vec;
      // Restrict, if necessary
      if (!impl->e_vecs[i]) {
        // No restriction for this field; read data directly from vec.
        CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, (const CeedScalar **)&e_data[i]));
      } else {
        CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_NOTRANSPOSE, vec, impl->e_vecs[i], request));
        // Get evec
        CeedCallBackend(CeedVectorGetArrayRead(impl->e_vecs[i], CEED_MEM_DEVICE, (const CeedScalar **)&e_data[i]));
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Hip(CeedInt num_elem, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                             CeedInt num_input_fields, const bool skip_active, CeedScalar *e_data[2 * CEED_FIELD_MAX],
                                             CeedOperator_Hip *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedInt             elem_size, size;
    CeedEvalMode        eval_mode;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Skip active input
    if (skip_active) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_DEVICE, CEED_USE_POINTER, e_data[i]));
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, eval_mode, impl->e_vecs[i], impl->q_vecs_in[i]));
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Hip(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                                const bool skip_active, CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Hip *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;
    CeedVector   vec;

    // Skip active input
    if (skip_active) {
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      if (!impl->e_vecs[i]) {  // This was a skip_restriction case
        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        CeedCallBackend(CeedVectorRestoreArrayRead(vec, (const CeedScalar **)&e_data[i]));
      } else {
        CeedCallBackend(CeedVectorRestoreArrayRead(impl->e_vecs[i], (const CeedScalar **)&e_data[i]));
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Hip(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedInt             Q, num_elem, elem_size, num_input_fields, num_output_fields, size;
  CeedScalar         *e_data[2 * CEED_FIELD_MAX] = {NULL};
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetup_Hip(op));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Hip(num_input_fields, qf_input_fields, op_input_fields, in_vec, false, e_data, impl, request));

  // Input basis apply if needed
  CeedCallBackend(CeedOperatorInputBasis_Hip(num_elem, qf_input_fields, op_input_fields, num_input_fields, false, e_data, impl));

  // Output pointers, as necessary
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE) {
      // Set the output Q-Vector to use the E-Vector data directly.
      CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs[i + impl->num_inputs], CEED_MEM_DEVICE, &e_data[i + num_input_fields]));
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_DEVICE, CEED_USE_POINTER, e_data[i + num_input_fields]));
    }
  }

  // Q function
  CeedCallBackend(CeedQFunctionApply(qf, num_elem * Q, impl->q_vecs_in, impl->q_vecs_out));

  // Output basis apply if needed
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedEvalMode        eval_mode;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &size));
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_TRANSPOSE, eval_mode, impl->q_vecs_out[i], impl->e_vecs[i + impl->num_inputs]));
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        Ceed ceed;

        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        // LCOV_EXCL_STOP
      }
    }
  }

  // Output restriction
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedEvalMode        eval_mode;
    CeedVector          vec;
    CeedElemRestriction elem_rstr;

    // Restore evec
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE) {
      CeedCallBackend(CeedVectorRestoreArray(impl->e_vecs[i + impl->num_inputs], &e_data[i + num_input_fields]));
    }
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    // Restrict
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
    // Active
    if (vec == CEED_VECTOR_ACTIVE) vec = out_vec;

    CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_TRANSPOSE, impl->e_vecs[i + impl->num_inputs], vec, request));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Hip(num_input_fields, qf_input_fields, op_input_fields, false, e_data, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Linear QFunction Assembly Core
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Hip(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                              CeedRequest *request) {
  Ceed                ceed, ceed_parent;
  CeedInt             num_active_in, num_active_out, Q, num_elem, num_input_fields, num_output_fields, size;
  CeedScalar         *assembled_array, *e_data[2 * CEED_FIELD_MAX] = {NULL};
  CeedVector         *active_inputs;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetFallbackParentCeed(op, &ceed_parent));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  active_inputs = impl->qf_active_in;
  num_active_in = impl->num_active_in, num_active_out = impl->num_active_out;

  // Setup
  CeedCallBackend(CeedOperatorSetup_Hip(op));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Hip(num_input_fields, qf_input_fields, op_input_fields, NULL, true, e_data, impl, request));

  // Count number of active input fields
  if (!num_active_in) {
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedScalar *q_vec_array;
      CeedVector  vec;

      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      // Check if active input
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
        CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
        CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_DEVICE, &q_vec_array));
        CeedCallBackend(CeedRealloc(num_active_in + size, &active_inputs));
        for (CeedInt field = 0; field < size; field++) {
          CeedSize q_size = (CeedSize)Q * num_elem;

          CeedCallBackend(CeedVectorCreate(ceed, q_size, &active_inputs[num_active_in + field]));
          CeedCallBackend(
              CeedVectorSetArray(active_inputs[num_active_in + field], CEED_MEM_DEVICE, CEED_USE_POINTER, &q_vec_array[field * Q * num_elem]));
        }
        num_active_in += size;
        CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &q_vec_array));
      }
    }
    impl->num_active_in = num_active_in;
    impl->qf_active_in  = active_inputs;
  }

  // Count number of active output fields
  if (!num_active_out) {
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedVector vec;

      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &size));
        num_active_out += size;
      }
    }
    impl->num_active_out = num_active_out;
  }

  // Check sizes
  CeedCheck(num_active_in > 0 && num_active_out > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");

  // Build objects if needed
  if (build_objects) {
    CeedSize l_size     = (CeedSize)num_elem * Q * num_active_in * num_active_out;
    CeedInt  strides[3] = {1, num_elem * Q, Q}; /* *NOPAD* */

    // Create output restriction
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed_parent, num_elem, Q, num_active_in * num_active_out,
                                                     num_active_in * num_active_out * num_elem * Q, strides, rstr));
    // Create assembled vector
    CeedCallBackend(CeedVectorCreate(ceed_parent, l_size, assembled));
  }
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  CeedCallBackend(CeedVectorGetArray(*assembled, CEED_MEM_DEVICE, &assembled_array));

  // Input basis apply
  CeedCallBackend(CeedOperatorInputBasis_Hip(num_elem, qf_input_fields, op_input_fields, num_input_fields, true, e_data, impl));

  // Assemble QFunction
  for (CeedInt in = 0; in < num_active_in; in++) {
    // Set Inputs
    CeedCallBackend(CeedVectorSetValue(active_inputs[in], 1.0));
    if (num_active_in > 1) {
      CeedCallBackend(CeedVectorSetValue(active_inputs[(in + num_active_in - 1) % num_active_in], 0.0));
    }
    // Set Outputs
    for (CeedInt out = 0; out < num_output_fields; out++) {
      CeedVector vec;

      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_DEVICE, CEED_USE_POINTER, assembled_array));
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &size));
        assembled_array += size * Q * num_elem;  // Advance the pointer by the size of the output
      }
    }
    // Apply QFunction
    CeedCallBackend(CeedQFunctionApply(qf, Q * num_elem, impl->q_vecs_in, impl->q_vecs_out));
  }

  // Un-set output q_vecs to prevent accidental overwrite of Assembled
  for (CeedInt out = 0; out < num_output_fields; out++) {
    CeedVector vec;

    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_DEVICE, NULL));
    }
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Hip(num_input_fields, qf_input_fields, op_input_fields, true, e_data, impl));

  // Restore output
  CeedCallBackend(CeedVectorRestoreArray(*assembled, &assembled_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Hip(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Hip(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Assemble Diagonal Setup
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetup_Hip(CeedOperator op) {
  Ceed                ceed;
  CeedInt             num_input_fields, num_output_fields, num_eval_modes_in = 0, num_eval_modes_out = 0;
  CeedInt             q_comp, num_nodes, num_qpts;
  CeedEvalMode       *eval_modes_in = NULL, *eval_modes_out = NULL;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields));

  // Determine active input basis
  CeedCallBackend(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedBasis    basis;
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
      CeedCheck(!basis_in || basis_in == basis, ceed, CEED_ERROR_BACKEND,
                "Backend does not implement operator diagonal assembly with multiple active bases");
      basis_in = basis;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_in, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF assembly
        CeedCallBackend(CeedRealloc(num_eval_modes_in + q_comp, &eval_modes_in));
        for (CeedInt d = 0; d < q_comp; d++) eval_modes_in[num_eval_modes_in + d] = eval_mode;
        num_eval_modes_in += q_comp;
      }
    }
  }

  // Determine active output basis
  CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedBasis    basis;
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
      CeedCheck(!basis_out || basis_out == basis, ceed, CEED_ERROR_BACKEND,
                "Backend does not implement operator diagonal assembly with multiple active bases");
      basis_out = basis;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_out, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF assembly
        CeedCallBackend(CeedRealloc(num_eval_modes_out + q_comp, &eval_modes_out));
        for (CeedInt d = 0; d < q_comp; d++) eval_modes_out[num_eval_modes_out + d] = eval_mode;
        num_eval_modes_out += q_comp;
      }
    }
  }

  // Operator data struct
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedCalloc(1, &impl->diag));
  CeedOperatorDiag_Hip *diag = impl->diag;

  // Basis matrices
  CeedCallBackend(CeedBasisGetNumNodes(basis_in, &num_nodes));
  if (basis_in == CEED_BASIS_NONE) num_qpts = num_nodes;
  else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts));
  const CeedInt interp_bytes     = num_nodes * num_qpts * sizeof(CeedScalar);
  const CeedInt eval_modes_bytes = sizeof(CeedEvalMode);
  bool          has_eval_none    = false;

  // CEED_EVAL_NONE
  for (CeedInt i = 0; i < num_eval_modes_in; i++) has_eval_none = has_eval_none || (eval_modes_in[i] == CEED_EVAL_NONE);
  for (CeedInt i = 0; i < num_eval_modes_out; i++) has_eval_none = has_eval_none || (eval_modes_out[i] == CEED_EVAL_NONE);
  if (has_eval_none) {
    CeedScalar *identity = NULL;

    CeedCallBackend(CeedCalloc(num_nodes * num_qpts, &identity));
    for (CeedInt i = 0; i < (num_nodes < num_qpts ? num_nodes : num_qpts); i++) identity[i * num_nodes + i] = 1.0;
    CeedCallHip(ceed, hipMalloc((void **)&diag->d_identity, interp_bytes));
    CeedCallHip(ceed, hipMemcpy(diag->d_identity, identity, interp_bytes, hipMemcpyHostToDevice));
    CeedCallBackend(CeedFree(&identity));
  }

  // CEED_EVAL_INTERP, CEED_EVAL_GRAD, CEED_EVAL_DIV, and CEED_EVAL_CURL
  for (CeedInt in = 0; in < 2; in++) {
    CeedFESpace fespace;
    CeedBasis   basis = in ? basis_in : basis_out;

    CeedCallBackend(CeedBasisGetFESpace(basis, &fespace));
    switch (fespace) {
      case CEED_FE_SPACE_H1: {
        CeedInt           q_comp_interp, q_comp_grad;
        const CeedScalar *interp, *grad;
        CeedScalar       *d_interp, *d_grad;

        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_grad));

        CeedCallBackend(CeedBasisGetInterp(basis, &interp));
        CeedCallHip(ceed, hipMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
        CeedCallHip(ceed, hipMemcpy(d_interp, interp, interp_bytes * q_comp_interp, hipMemcpyHostToDevice));
        CeedCallBackend(CeedBasisGetGrad(basis, &grad));
        CeedCallHip(ceed, hipMalloc((void **)&d_grad, interp_bytes * q_comp_grad));
        CeedCallHip(ceed, hipMemcpy(d_grad, grad, interp_bytes * q_comp_grad, hipMemcpyHostToDevice));
        if (in) {
          diag->d_interp_in = d_interp;
          diag->d_grad_in   = d_grad;
        } else {
          diag->d_interp_out = d_interp;
          diag->d_grad_out   = d_grad;
        }
      } break;
      case CEED_FE_SPACE_HDIV: {
        CeedInt           q_comp_interp, q_comp_div;
        const CeedScalar *interp, *div;
        CeedScalar       *d_interp, *d_div;

        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_DIV, &q_comp_div));

        CeedCallBackend(CeedBasisGetInterp(basis, &interp));
        CeedCallHip(ceed, hipMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
        CeedCallHip(ceed, hipMemcpy(d_interp, interp, interp_bytes * q_comp_interp, hipMemcpyHostToDevice));
        CeedCallBackend(CeedBasisGetDiv(basis, &div));
        CeedCallHip(ceed, hipMalloc((void **)&d_div, interp_bytes * q_comp_div));
        CeedCallHip(ceed, hipMemcpy(d_div, div, interp_bytes * q_comp_div, hipMemcpyHostToDevice));
        if (in) {
          diag->d_interp_in = d_interp;
          diag->d_div_in    = d_div;
        } else {
          diag->d_interp_out = d_interp;
          diag->d_div_out    = d_div;
        }
      } break;
      case CEED_FE_SPACE_HCURL: {
        CeedInt           q_comp_interp, q_comp_curl;
        const CeedScalar *interp, *curl;
        CeedScalar       *d_interp, *d_curl;

        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_CURL, &q_comp_curl));

        CeedCallBackend(CeedBasisGetInterp(basis, &interp));
        CeedCallHip(ceed, hipMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
        CeedCallHip(ceed, hipMemcpy(d_interp, interp, interp_bytes * q_comp_interp, hipMemcpyHostToDevice));
        CeedCallBackend(CeedBasisGetCurl(basis, &curl));
        CeedCallHip(ceed, hipMalloc((void **)&d_curl, interp_bytes * q_comp_curl));
        CeedCallHip(ceed, hipMemcpy(d_curl, curl, interp_bytes * q_comp_curl, hipMemcpyHostToDevice));
        if (in) {
          diag->d_interp_in = d_interp;
          diag->d_curl_in   = d_curl;
        } else {
          diag->d_interp_out = d_interp;
          diag->d_curl_out   = d_curl;
        }
      } break;
    }
  }

  // Arrays of eval_modes
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_eval_modes_in, num_eval_modes_in * eval_modes_bytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_eval_modes_in, eval_modes_in, num_eval_modes_in * eval_modes_bytes, hipMemcpyHostToDevice));
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_eval_modes_out, num_eval_modes_out * eval_modes_bytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_eval_modes_out, eval_modes_out, num_eval_modes_out * eval_modes_bytes, hipMemcpyHostToDevice));
  CeedCallBackend(CeedFree(&eval_modes_in));
  CeedCallBackend(CeedFree(&eval_modes_out));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Diagonal Setup (Compilation)
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetupCompile_Hip(CeedOperator op, CeedInt use_ceedsize_idx, const bool is_point_block) {
  Ceed                ceed;
  char               *diagonal_kernel_path, *diagonal_kernel_source;
  CeedInt             num_input_fields, num_output_fields, num_eval_modes_in = 0, num_eval_modes_out = 0;
  CeedInt             num_comp, q_comp, num_nodes, num_qpts;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields));

  // Determine active input basis
  CeedCallBackend(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis_in));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_in, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        num_eval_modes_in += q_comp;
      }
    }
  }

  // Determine active output basis
  CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis_out));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_out, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        num_eval_modes_out += q_comp;
      }
    }
  }

  // Operator data struct
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedOperatorDiag_Hip *diag = impl->diag;

  // Assemble kernel
  hipModule_t *module          = is_point_block ? &diag->module_point_block : &diag->module;
  CeedInt      elems_per_block = 1;
  CeedCallBackend(CeedBasisGetNumNodes(basis_in, &num_nodes));
  CeedCallBackend(CeedBasisGetNumComponents(basis_in, &num_comp));
  if (basis_in == CEED_BASIS_NONE) num_qpts = num_nodes;
  else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-operator-assemble-diagonal.h", &diagonal_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Diagonal Assembly Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, diagonal_kernel_path, &diagonal_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Diagonal Assembly Source Complete! -----\n");
  CeedCallHip(ceed, CeedCompile_Hip(ceed, diagonal_kernel_source, module, 8, "NUM_EVAL_MODES_IN", num_eval_modes_in, "NUM_EVAL_MODES_OUT",
                                    num_eval_modes_out, "NUM_COMP", num_comp, "NUM_NODES", num_nodes, "NUM_QPTS", num_qpts, "USE_CEEDSIZE",
                                    use_ceedsize_idx, "USE_POINT_BLOCK", is_point_block ? 1 : 0, "BLOCK_SIZE", num_nodes * elems_per_block));
  CeedCallHip(ceed, CeedGetKernel_Hip(ceed, *module, "LinearDiagonal", is_point_block ? &diag->LinearPointBlock : &diag->LinearDiagonal));
  CeedCallBackend(CeedFree(&diagonal_kernel_path));
  CeedCallBackend(CeedFree(&diagonal_kernel_source));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Diagonal Core
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalCore_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request, const bool is_point_block) {
  Ceed                ceed;
  CeedInt             num_elem, num_nodes;
  CeedScalar         *elem_diag_array;
  const CeedScalar   *assembled_qf_array;
  CeedVector          assembled_qf   = NULL, elem_diag;
  CeedElemRestriction assembled_rstr = NULL, rstr_in, rstr_out, diag_rstr;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Assemble QFunction
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &assembled_rstr, request));
  CeedCallBackend(CeedElemRestrictionDestroy(&assembled_rstr));
  CeedCallBackend(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_DEVICE, &assembled_qf_array));

  // Setup
  if (!impl->diag) CeedCallBackend(CeedOperatorAssembleDiagonalSetup_Hip(op));
  CeedOperatorDiag_Hip *diag = impl->diag;

  assert(diag != NULL);

  // Assemble kernel if needed
  if ((!is_point_block && !diag->LinearDiagonal) || (is_point_block && !diag->LinearPointBlock)) {
    CeedSize assembled_length, assembled_qf_length;
    CeedInt  use_ceedsize_idx = 0;
    CeedCallBackend(CeedVectorGetLength(assembled, &assembled_length));
    CeedCallBackend(CeedVectorGetLength(assembled_qf, &assembled_qf_length));
    if ((assembled_length > INT_MAX) || (assembled_qf_length > INT_MAX)) use_ceedsize_idx = 1;

    CeedCallBackend(CeedOperatorAssembleDiagonalSetupCompile_Hip(op, use_ceedsize_idx, is_point_block));
  }

  // Restriction and diagonal vector
  CeedCallBackend(CeedOperatorGetActiveElemRestrictions(op, &rstr_in, &rstr_out));
  CeedCheck(rstr_in == rstr_out, ceed, CEED_ERROR_BACKEND,
            "Cannot assemble operator diagonal with different input and output active element restrictions");
  if (!is_point_block && !diag->diag_rstr) {
    CeedCallBackend(CeedElemRestrictionCreateUnsignedCopy(rstr_out, &diag->diag_rstr));
    CeedCallBackend(CeedElemRestrictionCreateVector(diag->diag_rstr, NULL, &diag->elem_diag));
  } else if (is_point_block && !diag->point_block_diag_rstr) {
    CeedCallBackend(CeedOperatorCreateActivePointBlockRestriction(rstr_out, &diag->point_block_diag_rstr));
    CeedCallBackend(CeedElemRestrictionCreateVector(diag->point_block_diag_rstr, NULL, &diag->point_block_elem_diag));
  }
  diag_rstr = is_point_block ? diag->point_block_diag_rstr : diag->diag_rstr;
  elem_diag = is_point_block ? diag->point_block_elem_diag : diag->elem_diag;
  CeedCallBackend(CeedVectorSetValue(elem_diag, 0.0));

  // Only assemble diagonal if the basis has nodes, otherwise inputs are null pointers
  CeedCallBackend(CeedElemRestrictionGetElementSize(diag_rstr, &num_nodes));
  if (num_nodes > 0) {
    // Assemble element operator diagonals
    CeedCallBackend(CeedVectorGetArray(elem_diag, CEED_MEM_DEVICE, &elem_diag_array));
    CeedCallBackend(CeedElemRestrictionGetNumElements(diag_rstr, &num_elem));

    // Compute the diagonal of B^T D B
    CeedInt elems_per_block = 1;
    CeedInt grid            = CeedDivUpInt(num_elem, elems_per_block);
    void   *args[]          = {(void *)&num_elem,      &diag->d_identity,       &diag->d_interp_in,  &diag->d_grad_in, &diag->d_div_in,
                               &diag->d_curl_in,       &diag->d_interp_out,     &diag->d_grad_out,   &diag->d_div_out, &diag->d_curl_out,
                               &diag->d_eval_modes_in, &diag->d_eval_modes_out, &assembled_qf_array, &elem_diag_array};

    if (is_point_block) {
      CeedCallBackend(CeedRunKernelDim_Hip(ceed, diag->LinearPointBlock, grid, num_nodes, 1, elems_per_block, args));
    } else {
      CeedCallBackend(CeedRunKernelDim_Hip(ceed, diag->LinearDiagonal, grid, num_nodes, 1, elems_per_block, args));
    }

    // Restore arrays
    CeedCallBackend(CeedVectorRestoreArray(elem_diag, &elem_diag_array));
    CeedCallBackend(CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array));
  }

  // Assemble local operator diagonal
  CeedCallBackend(CeedElemRestrictionApply(diag_rstr, CEED_TRANSPOSE, elem_diag, assembled, request));

  // Cleanup
  CeedCallBackend(CeedVectorDestroy(&assembled_qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, false));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, true));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Single Operator Assembly Setup
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssembleSetup_Hip(CeedOperator op, CeedInt use_ceedsize_idx) {
  Ceed                ceed;
  char               *assembly_kernel_path, *assembly_kernel_source;
  CeedInt             num_input_fields, num_output_fields, num_eval_modes_in = 0, num_eval_modes_out = 0;
  CeedInt             elem_size_in, num_qpts_in, num_comp_in, elem_size_out, num_qpts_out, num_comp_out, q_comp;
  CeedEvalMode       *eval_modes_in = NULL, *eval_modes_out = NULL;
  CeedElemRestriction rstr_in = NULL, rstr_out = NULL;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *input_fields, *output_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Get intput and output fields
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &input_fields, &num_output_fields, &output_fields));

  // Determine active input basis eval mode
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedBasis    basis;
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(input_fields[i], &basis));
      CeedCheck(!basis_in || basis_in == basis, ceed, CEED_ERROR_BACKEND, "Backend does not implement operator assembly with multiple active bases");
      basis_in = basis;
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(input_fields[i], &rstr_in));
      CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_in, &elem_size_in));
      if (basis_in == CEED_BASIS_NONE) num_qpts_in = elem_size_in;
      else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts_in));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_in, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF Assembly
        CeedCallBackend(CeedRealloc(num_eval_modes_in + q_comp, &eval_modes_in));
        for (CeedInt d = 0; d < q_comp; d++) {
          eval_modes_in[num_eval_modes_in + d] = eval_mode;
        }
        num_eval_modes_in += q_comp;
      }
    }
  }

  // Determine active output basis; basis_out and rstr_out only used if same as input, TODO
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(output_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedBasis    basis;
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(output_fields[i], &basis));
      CeedCheck(!basis_out || basis_out == basis, ceed, CEED_ERROR_BACKEND,
                "Backend does not implement operator assembly with multiple active bases");
      basis_out = basis;
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(output_fields[i], &rstr_out));
      CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_out, &elem_size_out));
      if (basis_out == CEED_BASIS_NONE) num_qpts_out = elem_size_out;
      else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_out, &num_qpts_out));
      CeedCheck(num_qpts_in == num_qpts_out, ceed, CEED_ERROR_UNSUPPORTED,
                "Active input and output bases must have the same number of quadrature points");
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_out, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF Assembly
        CeedCallBackend(CeedRealloc(num_eval_modes_out + q_comp, &eval_modes_out));
        for (CeedInt d = 0; d < q_comp; d++) {
          eval_modes_out[num_eval_modes_out + d] = eval_mode;
        }
        num_eval_modes_out += q_comp;
      }
    }
  }
  CeedCheck(num_eval_modes_in > 0 && num_eval_modes_out > 0, ceed, CEED_ERROR_UNSUPPORTED, "Cannot assemble operator without inputs/outputs");

  CeedCallBackend(CeedCalloc(1, &impl->asmb));
  CeedOperatorAssemble_Hip *asmb = impl->asmb;
  asmb->elems_per_block          = 1;
  asmb->block_size_x             = elem_size_in;
  asmb->block_size_y             = elem_size_out;

  bool fallback = asmb->block_size_x * asmb->block_size_y * asmb->elems_per_block > 1024;

  if (fallback) {
    // Use fallback kernel with 1D threadblock
    asmb->block_size_y = 1;
  }

  // Compile kernels
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr_in, &num_comp_in));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr_out, &num_comp_out));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-operator-assemble.h", &assembly_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Assembly Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, assembly_kernel_path, &assembly_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Assembly Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Hip(ceed, assembly_kernel_source, &asmb->module, 10, "NUM_EVAL_MODES_IN", num_eval_modes_in, "NUM_EVAL_MODES_OUT",
                                  num_eval_modes_out, "NUM_COMP_IN", num_comp_in, "NUM_COMP_OUT", num_comp_out, "NUM_NODES_IN", elem_size_in,
                                  "NUM_NODES_OUT", elem_size_out, "NUM_QPTS", num_qpts_in, "BLOCK_SIZE",
                                  asmb->block_size_x * asmb->block_size_y * asmb->elems_per_block, "BLOCK_SIZE_Y", asmb->block_size_y, "USE_CEEDSIZE",
                                  use_ceedsize_idx));
  CeedCallBackend(CeedGetKernel_Hip(ceed, asmb->module, "LinearAssemble", &asmb->LinearAssemble));
  CeedCallBackend(CeedFree(&assembly_kernel_path));
  CeedCallBackend(CeedFree(&assembly_kernel_source));

  // Load into B_in, in order that they will be used in eval_modes_in
  {
    const CeedInt in_bytes           = elem_size_in * num_qpts_in * num_eval_modes_in * sizeof(CeedScalar);
    CeedInt       d_in               = 0;
    CeedEvalMode  eval_modes_in_prev = CEED_EVAL_NONE;
    bool          has_eval_none      = false;
    CeedScalar   *identity           = NULL;

    for (CeedInt i = 0; i < num_eval_modes_in; i++) {
      has_eval_none = has_eval_none || (eval_modes_in[i] == CEED_EVAL_NONE);
    }
    if (has_eval_none) {
      CeedCallBackend(CeedCalloc(elem_size_in * num_qpts_in, &identity));
      for (CeedInt i = 0; i < (elem_size_in < num_qpts_in ? elem_size_in : num_qpts_in); i++) identity[i * elem_size_in + i] = 1.0;
    }

    CeedCallHip(ceed, hipMalloc((void **)&asmb->d_B_in, in_bytes));
    for (CeedInt i = 0; i < num_eval_modes_in; i++) {
      const CeedScalar *h_B_in;

      CeedCallBackend(CeedOperatorGetBasisPointer(basis_in, eval_modes_in[i], identity, &h_B_in));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_in, eval_modes_in[i], &q_comp));
      if (q_comp > 1) {
        if (i == 0 || eval_modes_in[i] != eval_modes_in_prev) d_in = 0;
        else h_B_in = &h_B_in[(++d_in) * elem_size_in * num_qpts_in];
      }
      eval_modes_in_prev = eval_modes_in[i];

      CeedCallHip(ceed, hipMemcpy(&asmb->d_B_in[i * elem_size_in * num_qpts_in], h_B_in, elem_size_in * num_qpts_in * sizeof(CeedScalar),
                                  hipMemcpyHostToDevice));
    }

    if (identity) {
      CeedCallBackend(CeedFree(&identity));
    }
  }

  // Load into B_out, in order that they will be used in eval_modes_out
  {
    const CeedInt out_bytes           = elem_size_out * num_qpts_out * num_eval_modes_out * sizeof(CeedScalar);
    CeedInt       d_out               = 0;
    CeedEvalMode  eval_modes_out_prev = CEED_EVAL_NONE;
    bool          has_eval_none       = false;
    CeedScalar   *identity            = NULL;

    for (CeedInt i = 0; i < num_eval_modes_out; i++) {
      has_eval_none = has_eval_none || (eval_modes_out[i] == CEED_EVAL_NONE);
    }
    if (has_eval_none) {
      CeedCallBackend(CeedCalloc(elem_size_out * num_qpts_out, &identity));
      for (CeedInt i = 0; i < (elem_size_out < num_qpts_out ? elem_size_out : num_qpts_out); i++) identity[i * elem_size_out + i] = 1.0;
    }

    CeedCallHip(ceed, hipMalloc((void **)&asmb->d_B_out, out_bytes));
    for (CeedInt i = 0; i < num_eval_modes_out; i++) {
      const CeedScalar *h_B_out;

      CeedCallBackend(CeedOperatorGetBasisPointer(basis_out, eval_modes_out[i], identity, &h_B_out));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_out, eval_modes_out[i], &q_comp));
      if (q_comp > 1) {
        if (i == 0 || eval_modes_out[i] != eval_modes_out_prev) d_out = 0;
        else h_B_out = &h_B_out[(++d_out) * elem_size_out * num_qpts_out];
      }
      eval_modes_out_prev = eval_modes_out[i];

      CeedCallHip(ceed, hipMemcpy(&asmb->d_B_out[i * elem_size_out * num_qpts_out], h_B_out, elem_size_out * num_qpts_out * sizeof(CeedScalar),
                                  hipMemcpyHostToDevice));
    }

    if (identity) {
      CeedCallBackend(CeedFree(&identity));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble matrix data for COO matrix of assembled operator.
// The sparsity pattern is set by CeedOperatorLinearAssembleSymbolic.
//
// Note that this (and other assembly routines) currently assume only one active input restriction/basis per operator (could have multiple basis eval
// modes).
// TODO: allow multiple active input restrictions/basis objects
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssemble_Hip(CeedOperator op, CeedInt offset, CeedVector values) {
  Ceed                ceed;
  CeedSize            values_length = 0, assembled_qf_length = 0;
  CeedInt             use_ceedsize_idx = 0, num_elem_in, num_elem_out, elem_size_in, elem_size_out;
  CeedScalar         *values_array;
  const CeedScalar   *assembled_qf_array;
  CeedVector          assembled_qf   = NULL;
  CeedElemRestriction assembled_rstr = NULL, rstr_in, rstr_out;
  CeedRestrictionType rstr_type_in, rstr_type_out;
  const bool         *orients_in = NULL, *orients_out = NULL;
  const CeedInt8     *curl_orients_in = NULL, *curl_orients_out = NULL;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Assemble QFunction
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &assembled_rstr, CEED_REQUEST_IMMEDIATE));
  CeedCallBackend(CeedElemRestrictionDestroy(&assembled_rstr));
  CeedCallBackend(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_DEVICE, &assembled_qf_array));

  CeedCallBackend(CeedVectorGetLength(values, &values_length));
  CeedCallBackend(CeedVectorGetLength(assembled_qf, &assembled_qf_length));
  if ((values_length > INT_MAX) || (assembled_qf_length > INT_MAX)) use_ceedsize_idx = 1;

  // Setup
  if (!impl->asmb) CeedCallBackend(CeedSingleOperatorAssembleSetup_Hip(op, use_ceedsize_idx));
  CeedOperatorAssemble_Hip *asmb = impl->asmb;

  assert(asmb != NULL);

  // Assemble element operator
  CeedCallBackend(CeedVectorGetArray(values, CEED_MEM_DEVICE, &values_array));
  values_array += offset;

  CeedCallBackend(CeedOperatorGetActiveElemRestrictions(op, &rstr_in, &rstr_out));
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr_in, &num_elem_in));
  CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_in, &elem_size_in));

  CeedCallBackend(CeedElemRestrictionGetType(rstr_in, &rstr_type_in));
  if (rstr_type_in == CEED_RESTRICTION_ORIENTED) {
    CeedCallBackend(CeedElemRestrictionGetOrientations(rstr_in, CEED_MEM_DEVICE, &orients_in));
  } else if (rstr_type_in == CEED_RESTRICTION_CURL_ORIENTED) {
    CeedCallBackend(CeedElemRestrictionGetCurlOrientations(rstr_in, CEED_MEM_DEVICE, &curl_orients_in));
  }

  if (rstr_in != rstr_out) {
    CeedCallBackend(CeedElemRestrictionGetNumElements(rstr_out, &num_elem_out));
    CeedCheck(num_elem_in == num_elem_out, ceed, CEED_ERROR_UNSUPPORTED,
              "Active input and output operator restrictions must have the same number of elements");
    CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_out, &elem_size_out));

    CeedCallBackend(CeedElemRestrictionGetType(rstr_out, &rstr_type_out));
    if (rstr_type_out == CEED_RESTRICTION_ORIENTED) {
      CeedCallBackend(CeedElemRestrictionGetOrientations(rstr_out, CEED_MEM_DEVICE, &orients_out));
    } else if (rstr_type_out == CEED_RESTRICTION_CURL_ORIENTED) {
      CeedCallBackend(CeedElemRestrictionGetCurlOrientations(rstr_out, CEED_MEM_DEVICE, &curl_orients_out));
    }
  } else {
    elem_size_out    = elem_size_in;
    orients_out      = orients_in;
    curl_orients_out = curl_orients_in;
  }

  // Compute B^T D B
  CeedInt shared_mem =
      ((curl_orients_in || curl_orients_out ? elem_size_in * elem_size_out : 0) + (curl_orients_in ? elem_size_in * asmb->block_size_y : 0)) *
      sizeof(CeedScalar);
  CeedInt grid   = CeedDivUpInt(num_elem_in, asmb->elems_per_block);
  void   *args[] = {(void *)&num_elem_in, &asmb->d_B_in,     &asmb->d_B_out,      &orients_in,  &curl_orients_in,
                    &orients_out,         &curl_orients_out, &assembled_qf_array, &values_array};

  CeedCallBackend(
      CeedRunKernelDimShared_Hip(ceed, asmb->LinearAssemble, grid, asmb->block_size_x, asmb->block_size_y, asmb->elems_per_block, shared_mem, args));

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArray(values, &values_array));
  CeedCallBackend(CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array));

  // Cleanup
  CeedCallBackend(CeedVectorDestroy(&assembled_qf));
  if (rstr_type_in == CEED_RESTRICTION_ORIENTED) {
    CeedCallBackend(CeedElemRestrictionRestoreOrientations(rstr_in, &orients_in));
  } else if (rstr_type_in == CEED_RESTRICTION_CURL_ORIENTED) {
    CeedCallBackend(CeedElemRestrictionRestoreCurlOrientations(rstr_in, &curl_orients_in));
  }
  if (rstr_in != rstr_out) {
    if (rstr_type_out == CEED_RESTRICTION_ORIENTED) {
      CeedCallBackend(CeedElemRestrictionRestoreOrientations(rstr_out, &orients_out));
    } else if (rstr_type_out == CEED_RESTRICTION_CURL_ORIENTED) {
      CeedCallBackend(CeedElemRestrictionRestoreCurlOrientations(rstr_out, &curl_orients_out));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Hip(CeedOperator op) {
  Ceed              ceed;
  CeedOperator_Hip *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonal_Hip));
  CeedCallBackend(
      CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddPointBlockDiagonal", CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleSingle", CeedSingleOperatorAssemble_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
