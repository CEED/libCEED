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
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <string.h>

#include "../cuda/ceed-cuda-common.h"
#include "../cuda/ceed-cuda-compile.h"
#include "ceed-cuda-ref.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Cuda(CeedOperator op) {
  CeedOperator_Cuda *impl;

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
    CeedCallCuda(ceed, cuModuleUnload(impl->diag->module));
    CeedCallBackend(CeedFree(&impl->diag->h_e_mode_in));
    CeedCallBackend(CeedFree(&impl->diag->h_e_mode_out));
    CeedCallCuda(ceed, cudaFree(impl->diag->d_e_mode_in));
    CeedCallCuda(ceed, cudaFree(impl->diag->d_e_mode_out));
    CeedCallCuda(ceed, cudaFree(impl->diag->d_identity));
    CeedCallCuda(ceed, cudaFree(impl->diag->d_interp_in));
    CeedCallCuda(ceed, cudaFree(impl->diag->d_interp_out));
    CeedCallCuda(ceed, cudaFree(impl->diag->d_grad_in));
    CeedCallCuda(ceed, cudaFree(impl->diag->d_grad_out));
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->diag->point_block_diag_rstr));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->elem_diag));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->point_block_elem_diag));
  }
  CeedCallBackend(CeedFree(&impl->diag));

  if (impl->asmb) {
    Ceed ceed;

    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    CeedCallCuda(ceed, cuModuleUnload(impl->asmb->module));
    CeedCallCuda(ceed, cudaFree(impl->asmb->d_B_in));
    CeedCallCuda(ceed, cudaFree(impl->asmb->d_B_out));
  }
  CeedCallBackend(CeedFree(&impl->asmb));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Cuda(CeedQFunction qf, CeedOperator op, bool is_input, CeedVector *e_vecs, CeedVector *q_vecs, CeedInt e_start,
                                        CeedInt num_fields, CeedInt Q, CeedInt num_elem) {
  Ceed                ceed;
  bool                is_strided, skip_restriction;
  CeedSize            q_size;
  CeedInt             dim, size;
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
    CeedEvalMode e_mode;
    CeedBasis    basis;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &e_mode));

    is_strided       = false;
    skip_restriction = false;
    if (e_mode != CEED_EVAL_WEIGHT) {
      CeedElemRestriction elem_rstr;

      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &elem_rstr));

      // Check whether this field can skip the element restriction:
      // must be passive input, with e_mode NONE, and have a strided restriction with CEED_STRIDES_BACKEND.

      // First, check whether the field is input or output:
      if (is_input) {
        CeedVector vec;

        // Check for passive input:
        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec));
        if (vec != CEED_VECTOR_ACTIVE) {
          // Check e_mode
          if (e_mode == CEED_EVAL_NONE) {
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
        e_vecs[i + e_start] = NULL;
      } else {
        CeedCallBackend(CeedElemRestrictionCreateVector(elem_rstr, NULL, &e_vecs[i + e_start]));
      }
    }

    switch (e_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        q_size = (CeedSize)num_elem * Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        q_size = (CeedSize)num_elem * Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        CeedCallBackend(CeedBasisGetDimension(basis, &dim));
        q_size = (CeedSize)num_elem * Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_WEIGHT:  // Only on input fields
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        q_size = (CeedSize)num_elem * Q;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_vecs[i]));
        break;
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// CeedOperator needs to connect all the named fields (be they active or passive) to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Cuda(CeedOperator op) {
  Ceed                ceed;
  bool                is_setup_done;
  CeedInt             Q, num_elem, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Cuda  *impl;

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
  CeedCallBackend(CeedOperatorSetupFields_Cuda(qf, op, true, impl->e_vecs, impl->q_vecs_in, 0, num_input_fields, Q, num_elem));
  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Cuda(qf, op, false, impl->e_vecs, impl->q_vecs_out, num_input_fields, num_output_fields, Q, num_elem));

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Cuda(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                               CeedVector in_vec, const bool skip_active_in, CeedScalar *e_data[2 * CEED_FIELD_MAX],
                                               CeedOperator_Cuda *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode        e_mode;
    CeedVector          vec;
    CeedElemRestriction elem_rstr;

    // Get input vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      if (skip_active_in) continue;
      else vec = in_vec;
    }

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &e_mode));
    if (e_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
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
static inline int CeedOperatorInputBasis_Cuda(CeedInt num_elem, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                              CeedInt num_input_fields, const bool skip_active_in, CeedScalar *e_data[2 * CEED_FIELD_MAX],
                                              CeedOperator_Cuda *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedInt             elem_size, size;
    CeedEvalMode        e_mode;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Skip active input
    if (skip_active_in) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    // Get elem_size, e_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &e_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
    // Basis action
    switch (e_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_DEVICE, CEED_USE_POINTER, e_data[i]));
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, impl->e_vecs[i], impl->q_vecs_in[i]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, impl->e_vecs[i], impl->q_vecs_in[i]));
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Cuda(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                                 const bool skip_active_in, CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Cuda *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode e_mode;
    CeedVector   vec;

    // Skip active input
    if (skip_active_in) {
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &e_mode));
    if (e_mode == CEED_EVAL_WEIGHT) {  // Skip
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
static int CeedOperatorApplyAdd_Cuda(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedOperator_Cuda  *impl;
  CeedInt             Q, num_elem, elem_size, num_input_fields, num_output_fields, size;
  CeedEvalMode        e_mode;
  CeedScalar         *e_data[2 * CEED_FIELD_MAX] = {NULL};
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetup_Cuda(op));

  // Input e_vecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Cuda(num_input_fields, qf_input_fields, op_input_fields, in_vec, false, e_data, impl, request));

  // Input basis apply if needed
  CeedCallBackend(CeedOperatorInputBasis_Cuda(num_elem, qf_input_fields, op_input_fields, num_input_fields, false, e_data, impl));

  // Output pointers, as necessary
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &e_mode));
    if (e_mode == CEED_EVAL_NONE) {
      // Set the output Q-Vector to use the E-Vector data directly.
      CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs[i + impl->num_inputs], CEED_MEM_DEVICE, &e_data[i + num_input_fields]));
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_DEVICE, CEED_USE_POINTER, e_data[i + num_input_fields]));
    }
  }

  // Q function
  CeedCallBackend(CeedQFunctionApply(qf, num_elem * Q, impl->q_vecs_in, impl->q_vecs_out));

  // Output basis apply if needed
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Get elem_size, e_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &e_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &size));
    // Basis action
    switch (e_mode) {
      case CEED_EVAL_NONE:
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_TRANSPOSE, CEED_EVAL_INTERP, impl->q_vecs_out[i], impl->e_vecs[i + impl->num_inputs]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_TRANSPOSE, CEED_EVAL_GRAD, impl->q_vecs_out[i], impl->e_vecs[i + impl->num_inputs]));
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        Ceed ceed;
        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break;  // Should not occur
      }
      case CEED_EVAL_DIV:
        break;  // TODO: Not implemented
      case CEED_EVAL_CURL:
        break;  // TODO: Not implemented
                // LCOV_EXCL_STOP
    }
  }

  // Output restriction
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector          vec;
    CeedElemRestriction elem_rstr;

    // Restore evec
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &e_mode));
    if (e_mode == CEED_EVAL_NONE) {
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
  CeedCallBackend(CeedOperatorRestoreInputs_Cuda(num_input_fields, qf_input_fields, op_input_fields, false, e_data, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for assembling linear QFunction
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Cuda(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                               CeedRequest *request) {
  Ceed                ceed, ceed_parent;
  CeedInt             num_active_in, num_active_out, Q, num_elem, num_input_fields, num_output_fields, size;
  CeedSize            q_size;
  CeedScalar         *assembled_array, *e_data[2 * CEED_FIELD_MAX] = {NULL};
  CeedVector         *active_inputs;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Cuda  *impl;

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
  CeedCallBackend(CeedOperatorSetup_Cuda(op));

  // Input e_vecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Cuda(num_input_fields, qf_input_fields, op_input_fields, NULL, true, e_data, impl, request));

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
          q_size = (CeedSize)Q * num_elem;
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
    // Create output restriction
    CeedInt strides[3] = {1, num_elem * Q, Q}; /* *NOPAD* */
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed_parent, num_elem, Q, num_active_in * num_active_out,
                                                     num_active_in * num_active_out * num_elem * Q, strides, rstr));
    // Create assembled vector
    CeedSize l_size = (CeedSize)num_elem * Q * num_active_in * num_active_out;
    CeedCallBackend(CeedVectorCreate(ceed_parent, l_size, assembled));
  }
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  CeedCallBackend(CeedVectorGetArray(*assembled, CEED_MEM_DEVICE, &assembled_array));

  // Input basis apply
  CeedCallBackend(CeedOperatorInputBasis_Cuda(num_elem, qf_input_fields, op_input_fields, num_input_fields, true, e_data, impl));

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
  CeedCallBackend(CeedOperatorRestoreInputs_Cuda(num_input_fields, qf_input_fields, op_input_fields, true, e_data, impl));

  // Restore output
  CeedCallBackend(CeedVectorRestoreArray(*assembled, &assembled_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Cuda(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Cuda(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Cuda(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Cuda(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Assemble diagonal setup
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetup_Cuda(CeedOperator op, CeedInt use_ceedsize_idx) {
  Ceed                ceed;
  char               *diagonal_kernel_path, *diagonal_kernel_source;
  CeedInt             num_input_fields, num_output_fields, num_e_mode_in = 0, num_comp = 0, dim = 1, num_e_mode_out = 0, num_nodes, num_qpts;
  CeedEvalMode       *e_mode_in = NULL, *e_mode_out = NULL;
  CeedElemRestriction rstr_in = NULL, rstr_out = NULL;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_fields;
  CeedOperator_Cuda  *impl;

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
      CeedEvalMode        e_mode;
      CeedElemRestriction rstr;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis_in));
      CeedCallBackend(CeedBasisGetNumComponents(basis_in, &num_comp));
      CeedCallBackend(CeedBasisGetDimension(basis_in, &dim));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr));
      CeedCheck(!rstr_in || rstr_in == rstr, ceed, CEED_ERROR_BACKEND,
                "Backend does not implement multi-field non-composite operator diagonal assembly");
      rstr_in = rstr;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &e_mode));
      switch (e_mode) {
        case CEED_EVAL_NONE:
        case CEED_EVAL_INTERP:
          CeedCallBackend(CeedRealloc(num_e_mode_in + 1, &e_mode_in));
          e_mode_in[num_e_mode_in] = e_mode;
          num_e_mode_in += 1;
          break;
        case CEED_EVAL_GRAD:
          CeedCallBackend(CeedRealloc(num_e_mode_in + dim, &e_mode_in));
          for (CeedInt d = 0; d < dim; d++) e_mode_in[num_e_mode_in + d] = e_mode;
          num_e_mode_in += dim;
          break;
        case CEED_EVAL_WEIGHT:
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // Caught by QF Assembly
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
      CeedEvalMode        e_mode;
      CeedElemRestriction rstr;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis_out));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr));
      CeedCheck(!rstr_out || rstr_out == rstr, ceed, CEED_ERROR_BACKEND,
                "Backend does not implement multi-field non-composite operator diagonal assembly");
      rstr_out = rstr;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &e_mode));
      switch (e_mode) {
        case CEED_EVAL_NONE:
        case CEED_EVAL_INTERP:
          CeedCallBackend(CeedRealloc(num_e_mode_out + 1, &e_mode_out));
          e_mode_out[num_e_mode_out] = e_mode;
          num_e_mode_out += 1;
          break;
        case CEED_EVAL_GRAD:
          CeedCallBackend(CeedRealloc(num_e_mode_out + dim, &e_mode_out));
          for (CeedInt d = 0; d < dim; d++) e_mode_out[num_e_mode_out + d] = e_mode;
          num_e_mode_out += dim;
          break;
        case CEED_EVAL_WEIGHT:
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // Caught by QF Assembly
      }
    }
  }

  // Operator data struct
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedCalloc(1, &impl->diag));
  CeedOperatorDiag_Cuda *diag = impl->diag;

  diag->basis_in       = basis_in;
  diag->basis_out      = basis_out;
  diag->h_e_mode_in    = e_mode_in;
  diag->h_e_mode_out   = e_mode_out;
  diag->num_e_mode_in  = num_e_mode_in;
  diag->num_e_mode_out = num_e_mode_out;

  // Assemble kernel
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-operator-assemble-diagonal.h", &diagonal_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Diagonal Assembly Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, diagonal_kernel_path, &diagonal_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Diagonal Assembly Source Complete! -----\n");
  CeedCallBackend(CeedBasisGetNumNodes(basis_in, &num_nodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts));
  diag->num_nodes = num_nodes;
  CeedCallCuda(ceed,
               CeedCompile_Cuda(ceed, diagonal_kernel_source, &diag->module, 6, "NUM_E_MODE_IN", num_e_mode_in, "NUM_E_MODE_OUT", num_e_mode_out,
                                "NUM_NODES", num_nodes, "NUM_QPTS", num_qpts, "NUM_COMP", num_comp, "USE_CEEDSIZE", use_ceedsize_idx));
  CeedCallCuda(ceed, CeedGetKernel_Cuda(ceed, diag->module, "linearDiagonal", &diag->linearDiagonal));
  CeedCallCuda(ceed, CeedGetKernel_Cuda(ceed, diag->module, "linearPointBlockDiagonal", &diag->linearPointBlock));
  CeedCallBackend(CeedFree(&diagonal_kernel_path));
  CeedCallBackend(CeedFree(&diagonal_kernel_source));

  // Basis matrices
  const CeedInt     q_bytes      = num_qpts * sizeof(CeedScalar);
  const CeedInt     interp_bytes = q_bytes * num_nodes;
  const CeedInt     grad_bytes   = q_bytes * num_nodes * dim;
  const CeedInt     e_mode_bytes = sizeof(CeedEvalMode);
  const CeedScalar *interp_in, *interp_out, *grad_in, *grad_out;

  // CEED_EVAL_NONE
  CeedScalar *identity     = NULL;
  bool        is_eval_none = false;

  for (CeedInt i = 0; i < num_e_mode_in; i++) is_eval_none = is_eval_none || (e_mode_in[i] == CEED_EVAL_NONE);
  for (CeedInt i = 0; i < num_e_mode_out; i++) is_eval_none = is_eval_none || (e_mode_out[i] == CEED_EVAL_NONE);
  if (is_eval_none) {
    CeedCallBackend(CeedCalloc(num_qpts * num_nodes, &identity));
    for (CeedInt i = 0; i < (num_nodes < num_qpts ? num_nodes : num_qpts); i++) identity[i * num_nodes + i] = 1.0;
    CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_identity, interp_bytes));
    CeedCallCuda(ceed, cudaMemcpy(diag->d_identity, identity, interp_bytes, cudaMemcpyHostToDevice));
  }

  // CEED_EVAL_INTERP
  CeedCallBackend(CeedBasisGetInterp(basis_in, &interp_in));
  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_interp_in, interp_bytes));
  CeedCallCuda(ceed, cudaMemcpy(diag->d_interp_in, interp_in, interp_bytes, cudaMemcpyHostToDevice));
  CeedCallBackend(CeedBasisGetInterp(basis_out, &interp_out));
  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_interp_out, interp_bytes));
  CeedCallCuda(ceed, cudaMemcpy(diag->d_interp_out, interp_out, interp_bytes, cudaMemcpyHostToDevice));

  // CEED_EVAL_GRAD
  CeedCallBackend(CeedBasisGetGrad(basis_in, &grad_in));
  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_grad_in, grad_bytes));
  CeedCallCuda(ceed, cudaMemcpy(diag->d_grad_in, grad_in, grad_bytes, cudaMemcpyHostToDevice));
  CeedCallBackend(CeedBasisGetGrad(basis_out, &grad_out));
  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_grad_out, grad_bytes));
  CeedCallCuda(ceed, cudaMemcpy(diag->d_grad_out, grad_out, grad_bytes, cudaMemcpyHostToDevice));

  // Arrays of e_modes
  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_e_mode_in, num_e_mode_in * e_mode_bytes));
  CeedCallCuda(ceed, cudaMemcpy(diag->d_e_mode_in, e_mode_in, num_e_mode_in * e_mode_bytes, cudaMemcpyHostToDevice));
  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_e_mode_out, num_e_mode_out * e_mode_bytes));
  CeedCallCuda(ceed, cudaMemcpy(diag->d_e_mode_out, e_mode_out, num_e_mode_out * e_mode_bytes, cudaMemcpyHostToDevice));

  // Restriction
  diag->diag_rstr = rstr_out;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalCore_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request, const bool is_point_block) {
  Ceed                ceed;
  CeedSize            assembled_length = 0, assembled_qf_length = 0;
  CeedInt             use_ceedsize_idx = 0, num_elem;
  CeedScalar         *elem_diag_array;
  const CeedScalar   *assembled_qf_array;
  CeedVector          assembled_qf = NULL;
  CeedElemRestriction rstr         = NULL;
  CeedOperator_Cuda  *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Assemble QFunction
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &rstr, request));
  CeedCallBackend(CeedElemRestrictionDestroy(&rstr));

  CeedCallBackend(CeedVectorGetLength(assembled, &assembled_length));
  CeedCallBackend(CeedVectorGetLength(assembled_qf, &assembled_qf_length));
  if ((assembled_length > INT_MAX) || (assembled_qf_length > INT_MAX)) use_ceedsize_idx = 1;

  // Setup
  if (!impl->diag) CeedCallBackend(CeedOperatorAssembleDiagonalSetup_Cuda(op, use_ceedsize_idx));
  CeedOperatorDiag_Cuda *diag = impl->diag;

  assert(diag != NULL);

  // Restriction
  if (is_point_block && !diag->point_block_diag_rstr) {
    CeedCallBackend(CeedOperatorCreateActivePointBlockRestriction(diag->diag_rstr, &diag->point_block_diag_rstr));
  }
  CeedElemRestriction diag_rstr = is_point_block ? diag->point_block_diag_rstr : diag->diag_rstr;

  // Create diagonal vector
  CeedVector elem_diag = is_point_block ? diag->point_block_elem_diag : diag->elem_diag;

  if (!elem_diag) {
    CeedCallBackend(CeedElemRestrictionCreateVector(diag_rstr, NULL, &elem_diag));
    if (is_point_block) diag->point_block_elem_diag = elem_diag;
    else diag->elem_diag = elem_diag;
  }
  CeedCallBackend(CeedVectorSetValue(elem_diag, 0.0));

  // Assemble element operator diagonals
  CeedCallBackend(CeedVectorGetArray(elem_diag, CEED_MEM_DEVICE, &elem_diag_array));
  CeedCallBackend(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_DEVICE, &assembled_qf_array));
  CeedCallBackend(CeedElemRestrictionGetNumElements(diag_rstr, &num_elem));

  // Compute the diagonal of B^T D B
  int   elem_per_block = 1;
  int   grid           = num_elem / elem_per_block + ((num_elem / elem_per_block * elem_per_block < num_elem) ? 1 : 0);
  void *args[]         = {(void *)&num_elem, &diag->d_identity,  &diag->d_interp_in,  &diag->d_grad_in,    &diag->d_interp_out,
                          &diag->d_grad_out, &diag->d_e_mode_in, &diag->d_e_mode_out, &assembled_qf_array, &elem_diag_array};
  if (is_point_block) {
    CeedCallBackend(CeedRunKernelDim_Cuda(ceed, diag->linearPointBlock, grid, diag->num_nodes, 1, elem_per_block, args));
  } else {
    CeedCallBackend(CeedRunKernelDim_Cuda(ceed, diag->linearDiagonal, grid, diag->num_nodes, 1, elem_per_block, args));
  }

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArray(elem_diag, &elem_diag_array));
  CeedCallBackend(CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array));

  // Assemble local operator diagonal
  CeedCallBackend(CeedElemRestrictionApply(diag_rstr, CEED_TRANSPOSE, elem_diag, assembled, request));

  // Cleanup
  CeedCallBackend(CeedVectorDestroy(&assembled_qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Cuda(op, assembled, request, false));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Cuda(op, assembled, request, true));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Single operator assembly setup
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssembleSetup_Cuda(CeedOperator op, CeedInt use_ceedsize_idx) {
  Ceed    ceed;
  char   *assembly_kernel_path, *assembly_kernel_source;
  CeedInt num_input_fields, num_output_fields, num_e_mode_in = 0, dim = 1, num_B_in_mats_to_load = 0, size_B_in = 0, num_qpts = 0, elem_size = 0,
                                               num_e_mode_out = 0, num_B_out_mats_to_load = 0, size_B_out = 0, num_elem, num_comp;
  CeedEvalMode       *eval_mode_in = NULL, *eval_mode_out = NULL;
  CeedElemRestriction rstr_in = NULL, rstr_out = NULL;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *input_fields, *output_fields;
  CeedOperator_Cuda  *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Get intput and output fields
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &input_fields, &num_output_fields, &output_fields));

  // Determine active input basis eval mode
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  // Note that the kernel will treat each dimension of a gradient action separately;
  // i.e., when an active input has a CEED_EVAL_GRAD mode, num_e_mode_in will increment by dim.
  // However, for the purposes of loading the B matrices, it will be treated as one mode, and we will load/copy the entire gradient matrix at once, so
  // num_B_in_mats_to_load will be incremented by 1.
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(input_fields[i], &basis_in));
      CeedCallBackend(CeedBasisGetDimension(basis_in, &dim));
      CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(input_fields[i], &rstr_in));
      CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_in, &elem_size));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      if (eval_mode != CEED_EVAL_NONE) {
        CeedCallBackend(CeedRealloc(num_B_in_mats_to_load + 1, &eval_mode_in));
        eval_mode_in[num_B_in_mats_to_load] = eval_mode;
        num_B_in_mats_to_load += 1;
        if (eval_mode == CEED_EVAL_GRAD) {
          num_e_mode_in += dim;
          size_B_in += dim * elem_size * num_qpts;
        } else {
          num_e_mode_in += 1;
          size_B_in += elem_size * num_qpts;
        }
      }
    }
  }

  // Determine active output basis; basis_out and rstr_out only used if same as input, TODO
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(output_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(output_fields[i], &basis_out));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(output_fields[i], &rstr_out));
      CeedCheck(!rstr_out || rstr_out == rstr_in, ceed, CEED_ERROR_BACKEND, "Backend does not implement multi-field non-composite operator assembly");
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      if (eval_mode != CEED_EVAL_NONE) {
        CeedCallBackend(CeedRealloc(num_B_out_mats_to_load + 1, &eval_mode_out));
        eval_mode_out[num_B_out_mats_to_load] = eval_mode;
        num_B_out_mats_to_load += 1;
        if (eval_mode == CEED_EVAL_GRAD) {
          num_e_mode_out += dim;
          size_B_out += dim * elem_size * num_qpts;
        } else {
          num_e_mode_out += 1;
          size_B_out += elem_size * num_qpts;
        }
      }
    }
  }
  CeedCheck(num_e_mode_in > 0 && num_e_mode_out > 0, ceed, CEED_ERROR_UNSUPPORTED, "Cannot assemble operator without inputs/outputs");

  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr_in, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr_in, &num_comp));

  CeedCallBackend(CeedCalloc(1, &impl->asmb));
  CeedOperatorAssemble_Cuda *asmb = impl->asmb;
  asmb->num_elem                  = num_elem;

  // Compile kernels
  int elem_per_block    = 1;
  asmb->elem_per_block  = elem_per_block;
  CeedInt    block_size = elem_size * elem_size * elem_per_block;
  Ceed_Cuda *cuda_data;

  CeedCallBackend(CeedGetData(ceed, &cuda_data));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/cuda/cuda-ref-operator-assemble.h", &assembly_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Assembly Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, assembly_kernel_path, &assembly_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Assembly Source Complete! -----\n");
  bool fallback = block_size > cuda_data->device_prop.maxThreadsPerBlock;

  if (fallback) {
    // Use fallback kernel with 1D threadblock
    block_size         = elem_size * elem_per_block;
    asmb->block_size_x = elem_size;
    asmb->block_size_y = 1;
  } else {  // Use kernel with 2D threadblock
    asmb->block_size_x = elem_size;
    asmb->block_size_y = elem_size;
  }
  CeedCallBackend(CeedCompile_Cuda(ceed, assembly_kernel_source, &asmb->module, 8, "NUM_ELEM", num_elem, "NUM_E_MODE_IN", num_e_mode_in,
                                   "NUM_E_MODE_OUT", num_e_mode_out, "NUM_QPTS", num_qpts, "NUM_NODES", elem_size, "BLOCK_SIZE", block_size,
                                   "NUM_COMP", num_comp, "USE_CEEDSIZE", use_ceedsize_idx));
  CeedCallBackend(CeedGetKernel_Cuda(ceed, asmb->module, fallback ? "linearAssembleFallback" : "linearAssemble", &asmb->linearAssemble));
  CeedCallBackend(CeedFree(&assembly_kernel_path));
  CeedCallBackend(CeedFree(&assembly_kernel_source));

  // Build 'full' B matrices (not 1D arrays used for tensor-product matrices)
  const CeedScalar *interp_in, *grad_in;

  CeedCallBackend(CeedBasisGetInterp(basis_in, &interp_in));
  CeedCallBackend(CeedBasisGetGrad(basis_in, &grad_in));

  // Load into B_in, in order that they will be used in eval_mode
  const CeedInt inBytes   = size_B_in * sizeof(CeedScalar);
  CeedInt       mat_start = 0;

  CeedCallCuda(ceed, cudaMalloc((void **)&asmb->d_B_in, inBytes));
  for (int i = 0; i < num_B_in_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_in[i];

    if (eval_mode == CEED_EVAL_INTERP) {
      CeedCallCuda(ceed, cudaMemcpy(&asmb->d_B_in[mat_start], interp_in, elem_size * num_qpts * sizeof(CeedScalar), cudaMemcpyHostToDevice));
      mat_start += elem_size * num_qpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      CeedCallCuda(ceed, cudaMemcpy(&asmb->d_B_in[mat_start], grad_in, dim * elem_size * num_qpts * sizeof(CeedScalar), cudaMemcpyHostToDevice));
      mat_start += dim * elem_size * num_qpts;
    }
  }

  const CeedScalar *interp_out, *grad_out;

  // Note that this function currently assumes 1 basis, so this should always be true for now
  if (basis_out == basis_in) {
    interp_out = interp_in;
    grad_out   = grad_in;
  } else {
    CeedCallBackend(CeedBasisGetInterp(basis_out, &interp_out));
    CeedCallBackend(CeedBasisGetGrad(basis_out, &grad_out));
  }

  // Load into B_out, in order that they will be used in eval_mode
  const CeedInt outBytes = size_B_out * sizeof(CeedScalar);
  mat_start              = 0;

  CeedCallCuda(ceed, cudaMalloc((void **)&asmb->d_B_out, outBytes));
  for (int i = 0; i < num_B_out_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_out[i];

    if (eval_mode == CEED_EVAL_INTERP) {
      CeedCallCuda(ceed, cudaMemcpy(&asmb->d_B_out[mat_start], interp_out, elem_size * num_qpts * sizeof(CeedScalar), cudaMemcpyHostToDevice));
      mat_start += elem_size * num_qpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      CeedCallCuda(ceed, cudaMemcpy(&asmb->d_B_out[mat_start], grad_out, dim * elem_size * num_qpts * sizeof(CeedScalar), cudaMemcpyHostToDevice));
      mat_start += dim * elem_size * num_qpts;
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
static int CeedSingleOperatorAssemble_Cuda(CeedOperator op, CeedInt offset, CeedVector values) {
  Ceed                ceed;
  CeedSize            values_length = 0, assembled_qf_length = 0;
  CeedInt             use_ceedsize_idx = 0;
  CeedScalar         *values_array;
  const CeedScalar   *qf_array;
  CeedVector          assembled_qf = NULL;
  CeedElemRestriction rstr_q       = NULL;
  CeedOperator_Cuda  *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Assemble QFunction
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &rstr_q, CEED_REQUEST_IMMEDIATE));
  CeedCallBackend(CeedElemRestrictionDestroy(&rstr_q));
  CeedCallBackend(CeedVectorGetArray(values, CEED_MEM_DEVICE, &values_array));
  values_array += offset;
  CeedCallBackend(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_DEVICE, &qf_array));

  CeedCallBackend(CeedVectorGetLength(values, &values_length));
  CeedCallBackend(CeedVectorGetLength(assembled_qf, &assembled_qf_length));
  if ((values_length > INT_MAX) || (assembled_qf_length > INT_MAX)) use_ceedsize_idx = 1;
  // Setup
  if (!impl->asmb) {
    CeedCallBackend(CeedSingleOperatorAssembleSetup_Cuda(op, use_ceedsize_idx));
    assert(impl->asmb != NULL);
  }

  // Compute B^T D B
  const CeedInt num_elem       = impl->asmb->num_elem;
  const CeedInt elem_per_block = impl->asmb->elem_per_block;
  const CeedInt grid           = num_elem / elem_per_block + ((num_elem / elem_per_block * elem_per_block < num_elem) ? 1 : 0);
  void         *args[]         = {&impl->asmb->d_B_in, &impl->asmb->d_B_out, &qf_array, &values_array};

  CeedCallBackend(
      CeedRunKernelDim_Cuda(ceed, impl->asmb->linearAssemble, grid, impl->asmb->block_size_x, impl->asmb->block_size_y, elem_per_block, args));

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArray(values, &values_array));
  CeedCallBackend(CeedVectorRestoreArrayRead(assembled_qf, &qf_array));

  // Cleanup
  CeedCallBackend(CeedVectorDestroy(&assembled_qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Cuda(CeedOperator op) {
  Ceed               ceed;
  CeedOperator_Cuda *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonal_Cuda));
  CeedCallBackend(
      CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddPointBlockDiagonal", CeedOperatorLinearAssembleAddPointBlockDiagonal_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleSingle", CeedSingleOperatorAssemble_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Cuda));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cuda));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
