// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Ref(CeedQFunction qf, CeedOperator op, bool is_input, CeedVector *e_vecs_full, CeedVector *e_vecs,
                                       CeedVector *q_vecs, CeedInt start_e, CeedInt num_fields, CeedInt Q) {
  Ceed                ceed;
  CeedSize            e_size, q_size;
  CeedInt             num_comp, size, P;
  CeedQFunctionField *qf_fields;
  CeedOperatorField  *op_fields;

  {
    Ceed ceed_parent;

    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    CeedCallBackend(CeedGetParent(ceed, &ceed_parent));
    if (ceed_parent) ceed = ceed_parent;
  }
  if (is_input) {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  } else {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  }

  // Loop over fields
  for (CeedInt i = 0; i < num_fields; i++) {
    CeedEvalMode        eval_mode;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &elem_rstr));
      CeedCallBackend(CeedElemRestrictionCreateVector(elem_rstr, NULL, &e_vecs_full[i + start_e]));
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        q_size = (CeedSize)Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
        CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
        e_size = (CeedSize)P * num_comp;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        q_size = (CeedSize)Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_WEIGHT:  // Only on input fields
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        q_size = (CeedSize)Q;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        CeedCallBackend(CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_vecs[i]));
        break;
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------/*
static int CeedOperatorSetup_Ref(CeedOperator op) {
  bool                is_setup_done;
  CeedInt             Q, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Ref   *impl;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &impl->is_identity_qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Allocate
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->e_vecs_full));

  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->input_states));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_out));

  impl->num_inputs  = num_input_fields;
  impl->num_outputs = num_output_fields;

  // Set up infield and outfield e_vecs and q_vecs
  // Infields
  CeedCallBackend(CeedOperatorSetupFields_Ref(qf, op, true, impl->e_vecs_full, impl->e_vecs_in, impl->q_vecs_in, 0, num_input_fields, Q));
  // Outfields
  CeedCallBackend(
      CeedOperatorSetupFields_Ref(qf, op, false, impl->e_vecs_full, impl->e_vecs_out, impl->q_vecs_out, num_input_fields, num_output_fields, Q));

  // Identity QFunctions
  if (impl->is_identity_qf) {
    CeedEvalMode        in_mode, out_mode;
    CeedQFunctionField *in_fields, *out_fields;

    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &in_fields, NULL, &out_fields));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(in_fields[0], &in_mode));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(out_fields[0], &out_mode));

    if (in_mode == CEED_EVAL_NONE && out_mode == CEED_EVAL_NONE) {
      impl->is_identity_rstr_op = true;
    } else {
      CeedCallBackend(CeedVectorReferenceCopy(impl->q_vecs_in[0], &impl->q_vecs_out[0]));
    }
  }

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Ref(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                              CeedVector in_vec, const bool skip_active, CeedScalar *e_data_full[2 * CEED_FIELD_MAX],
                                              CeedOperator_Ref *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    uint64_t            state;
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
    // Restrict and Evec
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      // Restrict
      CeedCallBackend(CeedVectorGetState(vec, &state));
      // Skip restriction if input is unchanged
      if (state != impl->input_states[i] || vec == in_vec) {
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
        CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_NOTRANSPOSE, vec, impl->e_vecs_full[i], request));
        impl->input_states[i] = state;
      }
      // Get evec
      CeedCallBackend(CeedVectorGetArrayRead(impl->e_vecs_full[i], CEED_MEM_HOST, (const CeedScalar **)&e_data_full[i]));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Ref(CeedInt e, CeedInt Q, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                             CeedInt num_input_fields, const bool skip_active, CeedScalar *e_data_full[2 * CEED_FIELD_MAX],
                                             CeedOperator_Ref *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedInt             elem_size, size, num_comp;
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
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i][(CeedSize)e * Q * size]));
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
        CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i][(CeedSize)e * elem_size * num_comp]));
        CeedCallBackend(CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, eval_mode, impl->e_vecs_in[i], impl->q_vecs_in[i]));
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Output Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorOutputBasis_Ref(CeedInt e, CeedInt Q, CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
                                              CeedInt num_input_fields, CeedInt num_output_fields, CeedOperator op,
                                              CeedScalar *e_data_full[2 * CEED_FIELD_MAX], CeedOperator_Ref *impl) {
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedInt             elem_size, num_comp;
    CeedEvalMode        eval_mode;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Get elem_size, eval_mode
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
        CeedCallBackend(CeedVectorSetArray(impl->e_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER,
                                           &e_data_full[i + num_input_fields][(CeedSize)e * elem_size * num_comp]));
        CeedCallBackend(CeedBasisApply(basis, 1, CEED_TRANSPOSE, eval_mode, impl->q_vecs_out[i], impl->e_vecs_out[i]));
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        return CeedError(CeedOperatorReturnCeed(op), CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        // LCOV_EXCL_STOP
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Ref(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                                const bool skip_active, CeedScalar *e_data_full[2 * CEED_FIELD_MAX], CeedOperator_Ref *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;

    // Skip active inputs
    if (skip_active) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    // Restore input
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      CeedCallBackend(CeedVectorRestoreArrayRead(impl->e_vecs_full[i], (const CeedScalar **)&e_data_full[i]));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Ref(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedInt             Q, num_elem, num_input_fields, num_output_fields, size;
  CeedEvalMode        eval_mode;
  CeedScalar         *e_data_full[2 * CEED_FIELD_MAX] = {NULL};
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Ref   *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetup_Ref(op));

  // Restriction only operator
  if (impl->is_identity_rstr_op) {
    CeedElemRestriction elem_rstr;

    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[0], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_full[0], request));
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[0], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_TRANSPOSE, impl->e_vecs_full[0], out_vec, request));
    return CEED_ERROR_SUCCESS;
  }

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, in_vec, false, e_data_full, impl, request));

  // Output Evecs
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs_full[i + impl->num_inputs], CEED_MEM_HOST, &e_data_full[i + num_input_fields]));
  }

  // Loop through elements
  for (CeedInt e = 0; e < num_elem; e++) {
    // Output pointers
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_NONE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &size));
        CeedCallBackend(
            CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i + num_input_fields][(CeedSize)e * Q * size]));
      }
    }

    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Ref(e, Q, qf_input_fields, op_input_fields, num_input_fields, false, e_data_full, impl));

    // Q function
    if (!impl->is_identity_qf) {
      CeedCallBackend(CeedQFunctionApply(qf, Q, impl->q_vecs_in, impl->q_vecs_out));
    }

    // Output basis apply
    CeedCallBackend(
        CeedOperatorOutputBasis_Ref(e, Q, qf_output_fields, op_output_fields, num_input_fields, num_output_fields, op, e_data_full, impl));
  }

  // Output restriction
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector          vec;
    CeedElemRestriction elem_rstr;

    // Restore Evec
    CeedCallBackend(CeedVectorRestoreArray(impl->e_vecs_full[i + impl->num_inputs], &e_data_full[i + num_input_fields]));
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    // Active
    if (vec == CEED_VECTOR_ACTIVE) vec = out_vec;
    // Restrict
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_TRANSPOSE, impl->e_vecs_full[i + impl->num_inputs], vec, request));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, false, e_data_full, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for assembling linear QFunction
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Ref(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                              CeedRequest *request) {
  Ceed                ceed, ceed_parent;
  CeedSize            q_size;
  CeedInt             num_active_in, num_active_out, Q, num_elem, num_input_fields, num_output_fields, size;
  CeedScalar         *assembled_array, *e_data_full[2 * CEED_FIELD_MAX] = {NULL};
  CeedVector         *active_in;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Ref   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetFallbackParentCeed(op, &ceed_parent));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  active_in     = impl->qf_active_in;
  num_active_in = impl->num_active_in, num_active_out = impl->num_active_out;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetup_Ref(op));

  // Check for restriction only operator
  CeedCheck(!impl->is_identity_rstr_op, ceed, CEED_ERROR_BACKEND, "Assembling restriction only operators is not supported");

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, NULL, true, e_data_full, impl, request));

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
        CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &q_vec_array));
        CeedCallBackend(CeedRealloc(num_active_in + size, &active_in));
        for (CeedInt field = 0; field < size; field++) {
          q_size = (CeedSize)Q;
          CeedCallBackend(CeedVectorCreate(ceed_parent, q_size, &active_in[num_active_in + field]));
          CeedCallBackend(CeedVectorSetArray(active_in[num_active_in + field], CEED_MEM_HOST, CEED_USE_POINTER, &q_vec_array[field * Q]));
        }
        num_active_in += size;
        CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &q_vec_array));
      }
    }
    impl->num_active_in = num_active_in;
    impl->qf_active_in  = active_in;
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
    const CeedSize l_size     = (CeedSize)num_elem * Q * num_active_in * num_active_out;
    CeedInt        strides[3] = {1, Q, num_active_in * num_active_out * Q}; /* *NOPAD* */

    // Create output restriction
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed_parent, num_elem, Q, num_active_in * num_active_out,
                                                     num_active_in * num_active_out * num_elem * Q, strides, rstr));
    // Create assembled vector
    CeedCallBackend(CeedVectorCreate(ceed_parent, l_size, assembled));
  }
  // Clear output vector
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  CeedCallBackend(CeedVectorGetArray(*assembled, CEED_MEM_HOST, &assembled_array));

  // Loop through elements
  for (CeedInt e = 0; e < num_elem; e++) {
    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Ref(e, Q, qf_input_fields, op_input_fields, num_input_fields, true, e_data_full, impl));

    // Assemble QFunction
    for (CeedInt in = 0; in < num_active_in; in++) {
      // Set Inputs
      CeedCallBackend(CeedVectorSetValue(active_in[in], 1.0));
      if (num_active_in > 1) {
        CeedCallBackend(CeedVectorSetValue(active_in[(in + num_active_in - 1) % num_active_in], 0.0));
      }
      if (!impl->is_identity_qf) {
        // Set Outputs
        for (CeedInt out = 0; out < num_output_fields; out++) {
          CeedVector vec;

          // Get output vector
          CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
          // Check if active output
          if (vec == CEED_VECTOR_ACTIVE) {
            CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_USE_POINTER, assembled_array));
            CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &size));
            assembled_array += size * Q;  // Advance the pointer by the size of the output
          }
        }
        // Apply QFunction
        CeedCallBackend(CeedQFunctionApply(qf, Q, impl->q_vecs_in, impl->q_vecs_out));
      } else {
        const CeedScalar *q_vec_array;

        // Copy Identity Outputs
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[0], &size));
        CeedCallBackend(CeedVectorGetArrayRead(impl->q_vecs_out[0], CEED_MEM_HOST, &q_vec_array));
        for (CeedInt i = 0; i < size * Q; i++) assembled_array[i] = q_vec_array[i];
        CeedCallBackend(CeedVectorRestoreArrayRead(impl->q_vecs_out[0], &q_vec_array));
        assembled_array += size * Q;
      }
    }
  }

  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  if (!impl->is_identity_qf) {
    for (CeedInt out = 0; out < num_output_fields; out++) {
      CeedVector vec;

      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE && num_elem > 0) {
        CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL));
      }
    }
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, true, e_data_full, impl));

  // Restore output
  CeedCallBackend(CeedVectorRestoreArray(*assembled, &assembled_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Ref(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Ref(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Ref(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Ref(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFieldsAtPoints_Ref(CeedQFunction qf, CeedOperator op, bool is_input, CeedVector *e_vecs_full, CeedVector *e_vecs,
                                               CeedVector *q_vecs, CeedInt start_e, CeedInt num_fields, CeedInt Q) {
  Ceed                ceed;
  CeedSize            e_size, q_size;
  CeedInt             e_size_padding = 0, max_num_points, num_comp, size, P;
  CeedQFunctionField *qf_fields;
  CeedOperatorField  *op_fields;

  {
    Ceed ceed_parent;

    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    CeedCallBackend(CeedGetParent(ceed, &ceed_parent));
    if (ceed_parent) ceed = ceed_parent;
  }
  if (is_input) {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  } else {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  }

  // Get max number of points
  {
    CeedInt             dim;
    CeedElemRestriction rstr_points = NULL;
    CeedOperator_Ref   *impl;

    CeedCallBackend(CeedOperatorAtPointsGetPoints(op, &rstr_points, NULL));
    CeedCallBackend(CeedElemRestrictionGetMaxPointsInElement(rstr_points, &max_num_points));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr_points, &dim));
    CeedCallBackend(CeedElemRestrictionDestroy(&rstr_points));
    CeedCallBackend(CeedOperatorGetData(op, &impl));
    if (is_input) {
      CeedCallBackend(CeedVectorCreate(ceed, dim * max_num_points, &impl->point_coords_elem));
      CeedCallBackend(CeedVectorSetValue(impl->point_coords_elem, 0.0));
    }
  }

  // Loop over fields
  for (CeedInt i = 0; i < num_fields; i++) {
    CeedEvalMode eval_mode;
    CeedBasis    basis;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {
      CeedElemRestriction elem_rstr;
      CeedSize            e_size;
      bool                is_at_points;

      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &elem_rstr));
      CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
      CeedCallBackend(CeedElemRestrictionIsPoints(elem_rstr, &is_at_points));
      if (is_at_points) {
        CeedCallBackend(CeedElemRestrictionGetEVectorSize(elem_rstr, &e_size));
        if (e_size_padding == 0) {
          CeedInt num_points, num_elem;

          CeedCallBackend(CeedElemRestrictionGetNumElements(elem_rstr, &num_elem));
          CeedCallBackend(CeedElemRestrictionGetNumPointsInElement(elem_rstr, num_elem - 1, &num_points));
          e_size_padding = (max_num_points - num_points) * num_comp;
        }
        CeedCallBackend(CeedVectorCreate(ceed, e_size + e_size_padding, &e_vecs_full[i + start_e]));
        CeedCallBackend(CeedVectorSetValue(e_vecs_full[i + start_e], 0.0));
      } else {
        CeedCallBackend(CeedElemRestrictionCreateVector(elem_rstr, NULL, &e_vecs_full[i + start_e]));
      }
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE: {
        CeedVector vec;

        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        e_size = (CeedSize)max_num_points * size;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec));
        if (vec == CEED_VECTOR_ACTIVE || !is_input) {
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs[i], &q_vecs[i]));
        } else {
          q_size = (CeedSize)max_num_points * size;
          CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        }
        break;
      }
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
        CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
        e_size = (CeedSize)P * num_comp;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        q_size = (CeedSize)max_num_points * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_WEIGHT:  // Only on input fields
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        q_size = (CeedSize)max_num_points;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        CeedCallBackend(
            CeedBasisApplyAtPoints(basis, max_num_points, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, CEED_VECTOR_NONE, q_vecs[i]));
        break;
    }
    // Initialize full arrays for E-vectors and Q-vectors
    if (e_vecs[i]) CeedCallBackend(CeedVectorSetValue(e_vecs[i], 0.0));
    if (eval_mode != CEED_EVAL_WEIGHT) CeedCallBackend(CeedVectorSetValue(q_vecs[i], 0.0));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------
static int CeedOperatorSetupAtPoints_Ref(CeedOperator op) {
  bool                is_setup_done;
  CeedInt             Q, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Ref   *impl;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &impl->is_identity_qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Allocate
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->e_vecs_full));

  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->input_states));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_out));

  impl->num_inputs  = num_input_fields;
  impl->num_outputs = num_output_fields;

  // Set up infield and outfield pointer arrays
  // Infields
  CeedCallBackend(CeedOperatorSetupFieldsAtPoints_Ref(qf, op, true, impl->e_vecs_full, impl->e_vecs_in, impl->q_vecs_in, 0, num_input_fields, Q));
  // Outfields
  CeedCallBackend(CeedOperatorSetupFieldsAtPoints_Ref(qf, op, false, impl->e_vecs_full, impl->e_vecs_out, impl->q_vecs_out, num_input_fields,
                                                      num_output_fields, Q));

  // Identity QFunctions
  if (impl->is_identity_qf) {
    CeedCallBackend(CeedVectorReferenceCopy(impl->q_vecs_in[0], &impl->q_vecs_out[0]));
    CeedCallBackend(CeedVectorReferenceCopy(impl->q_vecs_in[0], &impl->e_vecs_out[0]));
  }

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasisAtPoints_Ref(CeedInt e, CeedInt num_points_offset, CeedInt num_points, CeedQFunctionField *qf_input_fields,
                                                     CeedOperatorField *op_input_fields, CeedInt num_input_fields, CeedVector in_vec,
                                                     CeedVector point_coords_elem, bool skip_active, CeedScalar *e_data[2 * CEED_FIELD_MAX],
                                                     CeedOperator_Ref *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    bool                is_active_input = false;
    CeedInt             elem_size, size, num_comp;
    CeedRestrictionType rstr_type;
    CeedEvalMode        eval_mode;
    CeedVector          vec;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    // Skip active input
    is_active_input = vec == CEED_VECTOR_ACTIVE;
    if (skip_active && is_active_input) continue;

    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetType(elem_rstr, &rstr_type));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
    // Restrict block active input
    if (is_active_input) {
      if (rstr_type == CEED_RESTRICTION_POINTS) {
        CeedCallBackend(CeedElemRestrictionApplyAtPointsInElement(elem_rstr, e, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[i], request));
      } else {
        CeedCallBackend(CeedElemRestrictionApplyBlock(elem_rstr, e, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[i], request));
      }
    }
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        if (!is_active_input) {
          CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data[i][num_points_offset * size]));
        }
        break;
      // Note - these basis eval modes require FEM fields
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        if (!is_active_input) {
          CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
          CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
          CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data[i][(CeedSize)e * elem_size * num_comp]));
        }
        CeedCallBackend(
            CeedBasisApplyAtPoints(basis, num_points, CEED_NOTRANSPOSE, eval_mode, point_coords_elem, impl->e_vecs_in[i], impl->q_vecs_in[i]));
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Output Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorOutputBasisAtPoints_Ref(CeedInt e, CeedInt num_points_offset, CeedInt num_points, CeedQFunctionField *qf_output_fields,
                                                      CeedOperatorField *op_output_fields, CeedInt num_input_fields, CeedInt num_output_fields,
                                                      CeedOperator op, CeedVector out_vec, CeedVector point_coords_elem, CeedOperator_Ref *impl,
                                                      CeedRequest *request) {
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedRestrictionType rstr_type;
    CeedEvalMode        eval_mode;
    CeedVector          vec;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(
            CeedBasisApplyAtPoints(basis, num_points, CEED_TRANSPOSE, eval_mode, point_coords_elem, impl->q_vecs_out[i], impl->e_vecs_out[i]));
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        return CeedError(CeedOperatorReturnCeed(op), CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        // LCOV_EXCL_STOP
      }
    }
    // Restrict output block
    // Get output vector
    CeedCallBackend(CeedElemRestrictionGetType(elem_rstr, &rstr_type));
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) vec = out_vec;
    // Restrict
    if (rstr_type == CEED_RESTRICTION_POINTS) {
      CeedCallBackend(CeedElemRestrictionApplyAtPointsInElement(elem_rstr, e, CEED_TRANSPOSE, impl->e_vecs_out[i], vec, request));
    } else {
      CeedCallBackend(CeedElemRestrictionApplyBlock(elem_rstr, e, CEED_TRANSPOSE, impl->e_vecs_out[i], vec, request));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyAddAtPoints_Ref(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedInt             num_points_offset          = 0, num_input_fields, num_output_fields, num_elem;
  CeedScalar         *e_data[2 * CEED_FIELD_MAX] = {0};
  CeedVector          point_coords               = NULL;
  CeedElemRestriction rstr_points                = NULL;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Ref   *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetupAtPoints_Ref(op));

  // Point coordinates
  CeedCallBackend(CeedOperatorAtPointsGetPoints(op, &rstr_points, &point_coords));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, NULL, true, e_data, impl, request));

  // Loop through elements
  for (CeedInt e = 0; e < num_elem; e++) {
    CeedInt num_points;

    // Setup points for element
    CeedCallBackend(CeedElemRestrictionApplyAtPointsInElement(rstr_points, e, CEED_NOTRANSPOSE, point_coords, impl->point_coords_elem, request));
    CeedCallBackend(CeedElemRestrictionGetNumPointsInElement(rstr_points, e, &num_points));

    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasisAtPoints_Ref(e, num_points_offset, num_points, qf_input_fields, op_input_fields, num_input_fields, in_vec,
                                                       impl->point_coords_elem, false, e_data, impl, request));

    // Q function
    if (!impl->is_identity_qf) {
      CeedCallBackend(CeedQFunctionApply(qf, num_points, impl->q_vecs_in, impl->q_vecs_out));
    }

    // Output basis apply and restriction
    CeedCallBackend(CeedOperatorOutputBasisAtPoints_Ref(e, num_points_offset, num_points, qf_output_fields, op_output_fields, num_input_fields,
                                                        num_output_fields, op, out_vec, impl->point_coords_elem, impl, request));

    num_points_offset += num_points;
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, true, e_data, impl));

  // Cleanup point coordinates
  CeedCallBackend(CeedVectorDestroy(&point_coords));
  CeedCallBackend(CeedElemRestrictionDestroy(&rstr_points));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for assembling linear QFunction
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionAtPointsCore_Ref(CeedOperator op, bool build_objects, CeedVector *assembled,
                                                                      CeedElemRestriction *rstr, CeedRequest *request) {
  Ceed                ceed;
  CeedSize            q_size;
  CeedInt             num_active_in, num_active_out, max_num_points, num_elem, num_input_fields, num_output_fields, num_points_offset = 0;
  CeedScalar         *assembled_array, *e_data_full[2 * CEED_FIELD_MAX] = {NULL};
  CeedVector         *active_in, point_coords                           = NULL;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Ref   *impl;
  CeedElemRestriction rstr_points = NULL;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  active_in     = impl->qf_active_in;
  num_active_in = impl->num_active_in, num_active_out = impl->num_active_out;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetupAtPoints_Ref(op));

  // Check for restriction only operator
  CeedCheck(!impl->is_identity_rstr_op, ceed, CEED_ERROR_BACKEND, "Assembling restriction only operators is not supported");

  // Point coordinates
  CeedCallBackend(CeedOperatorAtPointsGetPoints(op, &rstr_points, &point_coords));
  CeedCallBackend(CeedElemRestrictionGetMaxPointsInElement(rstr_points, &max_num_points));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, NULL, true, e_data_full, impl, request));

  // Count number of active input fields
  if (!num_active_in) {
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedScalar *q_vec_array;
      CeedInt     field_size;
      CeedVector  vec;

      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      // Check if active input
      if (vec == CEED_VECTOR_ACTIVE) {
        // Check that all active inputs are nodal fields
        {
          CeedElemRestriction elem_rstr;
          bool                is_at_points = false;

          CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
          CeedCallBackend(CeedElemRestrictionIsPoints(elem_rstr, &is_at_points));
          CeedCheck(!is_at_points, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction with active input at points");
        }
        // Get size of active input
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
        CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
        CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &q_vec_array));
        CeedCallBackend(CeedRealloc(num_active_in + field_size, &active_in));
        for (CeedInt field = 0; field < field_size; field++) {
          q_size = (CeedSize)max_num_points;
          CeedCallBackend(CeedVectorCreate(ceed, q_size, &active_in[num_active_in + field]));
          CeedCallBackend(CeedVectorSetArray(active_in[num_active_in + field], CEED_MEM_HOST, CEED_USE_POINTER, &q_vec_array[field * q_size]));
        }
        num_active_in += field_size;
        CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &q_vec_array));
      }
    }
    impl->num_active_in = num_active_in;
    impl->qf_active_in  = active_in;
  }

  // Count number of active output fields
  if (!num_active_out) {
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedVector vec;
      CeedInt    field_size;

      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE) {
        // Check that all active inputs are nodal fields
        {
          CeedElemRestriction elem_rstr;
          bool                is_at_points = false;

          CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_rstr));
          CeedCallBackend(CeedElemRestrictionIsPoints(elem_rstr, &is_at_points));
          CeedCheck(!is_at_points, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction with active input at points");
        }
        // Get size of active output
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &field_size));
        num_active_out += field_size;
      }
    }
    impl->num_active_out = num_active_out;
  }

  // Check sizes
  CeedCheck(num_active_in > 0 && num_active_out > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");

  // Build objects if needed
  if (build_objects) {
    CeedInt        num_points_total;
    const CeedInt *offsets;

    CeedCallBackend(CeedElemRestrictionGetNumPoints(rstr_points, &num_points_total));

    // Create output restriction (at points)
    CeedCallBackend(CeedElemRestrictionGetOffsets(rstr_points, CEED_MEM_HOST, &offsets));
    CeedCallBackend(CeedElemRestrictionCreateAtPoints(ceed, num_elem, num_points_total, num_active_in * num_active_out,
                                                      num_active_in * num_active_out * num_points_total, CEED_MEM_HOST, CEED_COPY_VALUES, offsets,
                                                      rstr));
    CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr_points, &offsets));

    // Create assembled vector
    CeedCallBackend(CeedElemRestrictionCreateVector(*rstr, assembled, NULL));
  }
  // Clear output vector
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  CeedCallBackend(CeedVectorGetArray(*assembled, CEED_MEM_HOST, &assembled_array));

  // Loop through elements
  for (CeedInt e = 0; e < num_elem; e++) {
    CeedInt num_points;

    // Setup points for element
    CeedCallBackend(CeedElemRestrictionApplyAtPointsInElement(rstr_points, e, CEED_NOTRANSPOSE, point_coords, impl->point_coords_elem, request));
    CeedCallBackend(CeedElemRestrictionGetNumPointsInElement(rstr_points, e, &num_points));

    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasisAtPoints_Ref(e, num_points_offset, num_points, qf_input_fields, op_input_fields, num_input_fields, NULL,
                                                       impl->point_coords_elem, true, e_data_full, impl, request));

    // Assemble QFunction
    for (CeedInt in = 0; in < num_active_in; in++) {
      // Set Inputs
      CeedCallBackend(CeedVectorSetValue(active_in[in], 1.0));
      if (num_active_in > 1) {
        CeedCallBackend(CeedVectorSetValue(active_in[(in + num_active_in - 1) % num_active_in], 0.0));
      }
      if (!impl->is_identity_qf) {
        // Set Outputs
        for (CeedInt out = 0; out < num_output_fields; out++) {
          CeedVector vec;
          CeedInt    field_size;

          // Get output vector
          CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
          // Check if active output
          if (vec == CEED_VECTOR_ACTIVE) {
            CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_USE_POINTER, assembled_array));
            CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &field_size));
            assembled_array += field_size * num_points;  // Advance the pointer by the size of the output
          }
        }
        // Apply QFunction
        CeedCallBackend(CeedQFunctionApply(qf, num_points, impl->q_vecs_in, impl->q_vecs_out));
      } else {
        const CeedScalar *q_vec_array;
        CeedInt           field_size;

        // Copy Identity Outputs
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[0], &field_size));
        CeedCallBackend(CeedVectorGetArrayRead(impl->q_vecs_out[0], CEED_MEM_HOST, &q_vec_array));
        for (CeedInt i = 0; i < field_size * num_points; i++) assembled_array[i] = q_vec_array[i];
        CeedCallBackend(CeedVectorRestoreArrayRead(impl->q_vecs_out[0], &q_vec_array));
        assembled_array += field_size * num_points;
      }
    }
    num_points_offset += num_points;
  }

  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  if (!impl->is_identity_qf) {
    for (CeedInt out = 0; out < num_output_fields; out++) {
      CeedVector vec;

      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
      // Check if active output
      if (vec == CEED_VECTOR_ACTIVE && num_elem > 0) {
        CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL));
      }
    }
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, true, e_data_full, impl));

  // Restore output
  CeedCallBackend(CeedVectorRestoreArray(*assembled, &assembled_array));

  // Cleanup
  CeedCallBackend(CeedVectorDestroy(&point_coords));
  CeedCallBackend(CeedElemRestrictionDestroy(&rstr_points));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionAtPoints_Ref(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionAtPointsCore_Ref(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionAtPointsUpdate_Ref(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr,
                                                                 CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionAtPointsCore_Ref(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Assemble Operator
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Operator Destroy
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Ref(CeedOperator op) {
  CeedOperator_Ref *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  for (CeedInt i = 0; i < impl->num_inputs + impl->num_outputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_full[i]));
  }
  CeedCallBackend(CeedFree(&impl->e_vecs_full));
  CeedCallBackend(CeedFree(&impl->input_states));

  for (CeedInt i = 0; i < impl->num_inputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_in[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_in[i]));
  }
  CeedCallBackend(CeedFree(&impl->e_vecs_in));
  CeedCallBackend(CeedFree(&impl->q_vecs_in));

  for (CeedInt i = 0; i < impl->num_outputs; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_out[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_out[i]));
  }
  CeedCallBackend(CeedFree(&impl->e_vecs_out));
  CeedCallBackend(CeedFree(&impl->q_vecs_out));
  CeedCallBackend(CeedVectorDestroy(&impl->point_coords_elem));

  // QFunction assembly
  for (CeedInt i = 0; i < impl->num_active_in; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->qf_active_in[i]));
  }
  CeedCallBackend(CeedFree(&impl->qf_active_in));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create
//------------------------------------------------------------------------------
int CeedOperatorCreate_Ref(CeedOperator op) {
  Ceed              ceed;
  CeedOperator_Ref *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Ref));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create At Points
//------------------------------------------------------------------------------
int CeedOperatorCreateAtPoints_Ref(CeedOperator op) {
  Ceed              ceed;
  CeedOperator_Ref *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunctionAtPoints_Ref));
  CeedCallBackend(
      CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionAtPointsUpdate_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAddAtPoints_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Ref));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
