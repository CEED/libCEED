// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Ref(CeedQFunction qf, CeedOperator op, bool is_input, CeedVector *e_vecs_full, CeedVector *e_vecs,
                                       CeedVector *q_vecs, CeedInt start_e, CeedInt num_fields, CeedInt Q) {
  CeedInt  num_comp, size, P;
  CeedSize e_size, q_size;
  Ceed     ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedBasis           basis;
  CeedElemRestriction elem_restr;
  CeedOperatorField  *op_fields;
  CeedQFunctionField *qf_fields;
  if (is_input) {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  } else {
    CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
    CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  }

  // Loop over fields
  for (CeedInt i = 0; i < num_fields; i++) {
    CeedEvalMode eval_mode;
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));

    if (eval_mode != CEED_EVAL_WEIGHT) {
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &elem_restr));
      CeedCallBackend(CeedElemRestrictionCreateVector(elem_restr, NULL, &e_vecs_full[i + start_e]));
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        q_size = (CeedSize)Q * size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
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
      case CEED_EVAL_DIV:
        break;  // Not implemented
      case CEED_EVAL_CURL:
        break;  // Not implemented
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------/*
static int CeedOperatorSetup_Ref(CeedOperator op) {
  bool is_setup_done;
  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Ref *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedInt Q, num_input_fields, num_output_fields;
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &impl->is_identity_qf));
  CeedOperatorField *op_input_fields, *op_output_fields;
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
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
      impl->is_identity_restr_op = true;
    } else {
      CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_out[0]));
      impl->q_vecs_out[0] = impl->q_vecs_in[0];
      CeedCallBackend(CeedVectorAddReference(impl->q_vecs_in[0]));
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
  CeedEvalMode        eval_mode;
  CeedVector          vec;
  CeedElemRestriction elem_restr;
  uint64_t            state;

  for (CeedInt i = 0; i < num_input_fields; i++) {
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
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_restr));
        CeedCallBackend(CeedElemRestrictionApply(elem_restr, CEED_NOTRANSPOSE, vec, impl->e_vecs_full[i], request));
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
  CeedInt             dim, elem_size, size;
  CeedElemRestriction elem_restr;
  CeedEvalMode        eval_mode;
  CeedBasis           basis;

  for (CeedInt i = 0; i < num_input_fields; i++) {
    // Skip active input
    if (skip_active) {
      CeedVector vec;
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_restr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_restr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i][e * Q * size]));
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i][e * elem_size * size]));
        CeedCallBackend(CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, impl->e_vecs_in[i], impl->q_vecs_in[i]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisGetDimension(basis, &dim));
        CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i][e * elem_size * size / dim]));
        CeedCallBackend(CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, impl->e_vecs_in[i], impl->q_vecs_in[i]));
        break;
      case CEED_EVAL_WEIGHT:
        break;  // No action
      // LCOV_EXCL_START
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL: {
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        Ceed ceed;
        CeedCallBackend(CeedBasisGetCeed(basis, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed evaluation mode not implemented");
        // LCOV_EXCL_STOP
      }
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
  CeedInt             dim, elem_size, size;
  CeedElemRestriction elem_restr;
  CeedEvalMode        eval_mode;
  CeedBasis           basis;

  for (CeedInt i = 0; i < num_output_fields; i++) {
    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_restr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_restr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &size));
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(
            CeedVectorSetArray(impl->e_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i + num_input_fields][e * elem_size * size]));
        CeedCallBackend(CeedBasisApply(basis, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, impl->q_vecs_out[i], impl->e_vecs_out[i]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisGetDimension(basis, &dim));
        CeedCallBackend(
            CeedVectorSetArray(impl->e_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i + num_input_fields][e * elem_size * size / dim]));
        CeedCallBackend(CeedBasisApply(basis, 1, CEED_TRANSPOSE, CEED_EVAL_GRAD, impl->q_vecs_out[i], impl->e_vecs_out[i]));
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        Ceed ceed;
        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "CEED_EVAL_WEIGHT cannot be an output "
                         "evaluation mode");
      }
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL: {
        Ceed ceed;
        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "Ceed evaluation mode not implemented");
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
  CeedEvalMode eval_mode;

  for (CeedInt i = 0; i < num_input_fields; i++) {
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
  CeedOperator_Ref *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedInt Q, num_elem, num_input_fields, num_output_fields, size;
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedOperatorField *op_input_fields, *op_output_fields;
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedEvalMode        eval_mode;
  CeedVector          vec;
  CeedElemRestriction elem_restr;
  CeedScalar         *e_data_full[2 * CEED_FIELD_MAX] = {0};

  // Setup
  CeedCallBackend(CeedOperatorSetup_Ref(op));

  // Restriction only operator
  if (impl->is_identity_restr_op) {
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[0], &elem_restr));
    CeedCallBackend(CeedElemRestrictionApply(elem_restr, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_full[0], request));
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[0], &elem_restr));
    CeedCallBackend(CeedElemRestrictionApply(elem_restr, CEED_TRANSPOSE, impl->e_vecs_full[0], out_vec, request));
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
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i + num_input_fields][e * Q * size]));
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
    // Restore Evec
    CeedCallBackend(CeedVectorRestoreArray(impl->e_vecs_full[i + impl->num_inputs], &e_data_full[i + num_input_fields]));
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    // Active
    if (vec == CEED_VECTOR_ACTIVE) vec = out_vec;
    // Restrict
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_restr));
    CeedCallBackend(CeedElemRestrictionApply(elem_restr, CEED_TRANSPOSE, impl->e_vecs_full[i + impl->num_inputs], vec, request));
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
  CeedOperator_Ref *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedInt  Q, num_elem, num_input_fields, num_output_fields, size;
  CeedSize q_size;
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedOperatorField *op_input_fields, *op_output_fields;
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedVector  vec;
  CeedInt     num_active_in = impl->num_active_in, num_active_out = impl->num_active_out;
  CeedVector *active_in = impl->qf_active_in;
  CeedScalar *a, *tmp;
  Ceed        ceed, ceed_parent;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetOperatorFallbackParentCeed(ceed, &ceed_parent));
  ceed_parent                                 = ceed_parent ? ceed_parent : ceed;
  CeedScalar *e_data_full[2 * CEED_FIELD_MAX] = {0};

  // Setup
  CeedCallBackend(CeedOperatorSetup_Ref(op));

  // Check for identity
  if (impl->is_identity_qf) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Assembling identity QFunctions not supported");
    // LCOV_EXCL_STOP
  }

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, NULL, true, e_data_full, impl, request));

  // Count number of active input fields
  if (!num_active_in) {
    for (CeedInt i = 0; i < num_input_fields; i++) {
      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      // Check if active input
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
        CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
        CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &tmp));
        CeedCallBackend(CeedRealloc(num_active_in + size, &active_in));
        for (CeedInt field = 0; field < size; field++) {
          q_size = (CeedSize)Q;
          CeedCallBackend(CeedVectorCreate(ceed, q_size, &active_in[num_active_in + field]));
          CeedCallBackend(CeedVectorSetArray(active_in[num_active_in + field], CEED_MEM_HOST, CEED_USE_POINTER, &tmp[field * Q]));
        }
        num_active_in += size;
        CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &tmp));
      }
    }
    impl->num_active_in = num_active_in;
    impl->qf_active_in  = active_in;
  }

  // Count number of active output fields
  if (!num_active_out) {
    for (CeedInt i = 0; i < num_output_fields; i++) {
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
  if (!num_active_in || !num_active_out) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
    // LCOV_EXCL_STOP
  }

  // Build objects if needed
  if (build_objects) {
    // Create output restriction
    CeedInt strides[3] = {1, Q, num_active_in * num_active_out * Q}; /* *NOPAD* */
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed_parent, num_elem, Q, num_active_in * num_active_out,
                                                     num_active_in * num_active_out * num_elem * Q, strides, rstr));
    // Create assembled vector
    CeedSize l_size = (CeedSize)num_elem * Q * num_active_in * num_active_out;
    CeedCallBackend(CeedVectorCreate(ceed_parent, l_size, assembled));
  }
  // Clear output vector
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  CeedCallBackend(CeedVectorGetArray(*assembled, CEED_MEM_HOST, &a));

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
      // Set Outputs
      for (CeedInt out = 0; out < num_output_fields; out++) {
        // Get output vector
        CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
        // Check if active output
        if (vec == CEED_VECTOR_ACTIVE) {
          CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_USE_POINTER, a));
          CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &size));
          a += size * Q;  // Advance the pointer by the size of the output
        }
      }
      // Apply QFunction
      CeedCallBackend(CeedQFunctionApply(qf, Q, impl->q_vecs_in, impl->q_vecs_out));
    }
  }

  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  for (CeedInt out = 0; out < num_output_fields; out++) {
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE && num_elem > 0) {
      CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL));
    }
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Ref(num_input_fields, qf_input_fields, op_input_fields, true, e_data_full, impl));

  // Restore output
  CeedCallBackend(CeedVectorRestoreArray(*assembled, &a));

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
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedOperator_Ref *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Ref));
  return CEED_ERROR_SUCCESS;
}
