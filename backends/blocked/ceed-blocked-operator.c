// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
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

#include "ceed-blocked.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Blocked(CeedQFunction qf, CeedOperator op, bool is_input, const CeedInt block_size,
                                           CeedElemRestriction *block_rstr, CeedVector *e_vecs_full, CeedVector *e_vecs, CeedVector *q_vecs,
                                           CeedInt start_e, CeedInt num_fields, CeedInt Q) {
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
    CeedEvalMode eval_mode;
    CeedBasis    basis;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
    if (eval_mode != CEED_EVAL_WEIGHT) {
      Ceed                ceed_rstr;
      CeedSize            l_size;
      CeedInt             num_elem, elem_size, comp_stride;
      CeedRestrictionType rstr_type;
      CeedElemRestriction rstr;

      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr));
      CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed_rstr));
      CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
      CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
      CeedCallBackend(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
      CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
      CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));

      CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
      switch (rstr_type) {
        case CEED_RESTRICTION_STANDARD: {
          const CeedInt *offsets = NULL;

          CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
          CeedCallBackend(CeedElemRestrictionCreateBlocked(ceed_rstr, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, CEED_MEM_HOST,
                                                           CEED_COPY_VALUES, offsets, &block_rstr[i + start_e]));
          CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
        } break;
        case CEED_RESTRICTION_ORIENTED: {
          const bool    *orients = NULL;
          const CeedInt *offsets = NULL;

          CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
          CeedCallBackend(CeedElemRestrictionGetOrientations(rstr, CEED_MEM_HOST, &orients));
          CeedCallBackend(CeedElemRestrictionCreateBlockedOriented(ceed_rstr, num_elem, elem_size, block_size, num_comp, comp_stride, l_size,
                                                                   CEED_MEM_HOST, CEED_COPY_VALUES, offsets, orients, &block_rstr[i + start_e]));
          CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
          CeedCallBackend(CeedElemRestrictionRestoreOrientations(rstr, &orients));
        } break;
        case CEED_RESTRICTION_CURL_ORIENTED: {
          const CeedInt8 *curl_orients = NULL;
          const CeedInt  *offsets      = NULL;

          CeedCallBackend(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
          CeedCallBackend(CeedElemRestrictionGetCurlOrientations(rstr, CEED_MEM_HOST, &curl_orients));
          CeedCallBackend(CeedElemRestrictionCreateBlockedCurlOriented(ceed_rstr, num_elem, elem_size, block_size, num_comp, comp_stride, l_size,
                                                                       CEED_MEM_HOST, CEED_COPY_VALUES, offsets, curl_orients,
                                                                       &block_rstr[i + start_e]));
          CeedCallBackend(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
          CeedCallBackend(CeedElemRestrictionRestoreCurlOrientations(rstr, &curl_orients));
        } break;
        case CEED_RESTRICTION_STRIDED: {
          CeedInt strides[3];

          CeedCallBackend(CeedElemRestrictionGetStrides(rstr, &strides));
          CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed_rstr, num_elem, elem_size, block_size, num_comp, l_size, strides,
                                                                  &block_rstr[i + start_e]));
        } break;
        case CEED_RESTRICTION_POINTS:
          // Empty case - won't occur
          break;
      }
      CeedCallBackend(CeedElemRestrictionCreateVector(block_rstr[i + start_e], NULL, &e_vecs_full[i + start_e]));
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        q_size = (CeedSize)Q * size * block_size;
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
        e_size = (CeedSize)P * num_comp * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        q_size = (CeedSize)Q * size * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_WEIGHT:  // Only on input fields
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        q_size = (CeedSize)Q * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        CeedCallBackend(CeedBasisApply(basis, block_size, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_vecs[i]));
        break;
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Blocked(CeedOperator op) {
  bool                  is_setup_done;
  Ceed                  ceed;
  CeedInt               Q, num_input_fields, num_output_fields;
  const CeedInt         block_size = 8;
  CeedQFunctionField   *qf_input_fields, *qf_output_fields;
  CeedQFunction         qf;
  CeedOperatorField    *op_input_fields, *op_output_fields;
  CeedOperator_Blocked *impl;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &impl->is_identity_qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Allocate
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->block_rstr));
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
  CeedCallBackend(CeedOperatorSetupFields_Blocked(qf, op, true, block_size, impl->block_rstr, impl->e_vecs_full, impl->e_vecs_in, impl->q_vecs_in, 0,
                                                  num_input_fields, Q));
  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Blocked(qf, op, false, block_size, impl->block_rstr, impl->e_vecs_full, impl->e_vecs_out, impl->q_vecs_out,
                                                  num_input_fields, num_output_fields, Q));

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
static inline int CeedOperatorSetupInputs_Blocked(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                                  CeedVector in_vec, bool skip_active, CeedScalar *e_data_full[2 * CEED_FIELD_MAX],
                                                  CeedOperator_Blocked *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    uint64_t     state;
    CeedEvalMode eval_mode;
    CeedVector   vec;

    // Get input vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      if (skip_active) continue;
      else vec = in_vec;
    }

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      // Restrict
      CeedCallBackend(CeedVectorGetState(vec, &state));
      if (state != impl->input_states[i] || vec == in_vec) {
        CeedCallBackend(CeedElemRestrictionApply(impl->block_rstr[i], CEED_NOTRANSPOSE, vec, impl->e_vecs_full[i], request));
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
static inline int CeedOperatorInputBasis_Blocked(CeedInt e, CeedInt Q, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                                 CeedInt num_input_fields, CeedInt block_size, bool skip_active,
                                                 CeedScalar *e_data_full[2 * CEED_FIELD_MAX], CeedOperator_Blocked *impl) {
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
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i][e * Q * size]));
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
        CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i][e * elem_size * num_comp]));
        CeedCallBackend(CeedBasisApply(basis, block_size, CEED_NOTRANSPOSE, eval_mode, impl->e_vecs_in[i], impl->q_vecs_in[i]));
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
static inline int CeedOperatorOutputBasis_Blocked(CeedInt e, CeedInt Q, CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
                                                  CeedInt block_size, CeedInt num_input_fields, CeedInt num_output_fields, CeedOperator op,
                                                  CeedScalar *e_data_full[2 * CEED_FIELD_MAX], CeedOperator_Blocked *impl) {
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedInt             elem_size, num_comp;
    CeedEvalMode        eval_mode;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Get elem_size, eval_mode, size
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
        CeedCallBackend(
            CeedVectorSetArray(impl->e_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i + num_input_fields][e * elem_size * num_comp]));
        CeedCallBackend(CeedBasisApply(basis, block_size, CEED_TRANSPOSE, eval_mode, impl->q_vecs_out[i], impl->e_vecs_out[i]));
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
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Blocked(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                                    bool skip_active, CeedScalar *e_data_full[2 * CEED_FIELD_MAX], CeedOperator_Blocked *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;

    // Skip active inputs
    if (skip_active) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
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
static int CeedOperatorApplyAdd_Blocked(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedInt               Q, num_input_fields, num_output_fields, num_elem, size;
  const CeedInt         block_size = 8;
  CeedEvalMode          eval_mode;
  CeedScalar           *e_data_full[2 * CEED_FIELD_MAX] = {0};
  CeedQFunctionField   *qf_input_fields, *qf_output_fields;
  CeedQFunction         qf;
  CeedOperatorField    *op_input_fields, *op_output_fields;
  CeedOperator_Blocked *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  const CeedInt num_blocks = (num_elem / block_size) + !!(num_elem % block_size);

  // Setup
  CeedCallBackend(CeedOperatorSetup_Blocked(op));

  // Restriction only operator
  if (impl->is_identity_rstr_op) {
    CeedCallBackend(CeedElemRestrictionApply(impl->block_rstr[0], CEED_NOTRANSPOSE, in_vec, impl->e_vecs_full[0], request));
    CeedCallBackend(CeedElemRestrictionApply(impl->block_rstr[1], CEED_TRANSPOSE, impl->e_vecs_full[0], out_vec, request));
    return CEED_ERROR_SUCCESS;
  }

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Blocked(num_input_fields, qf_input_fields, op_input_fields, in_vec, false, e_data_full, impl, request));

  // Output Evecs
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs_full[i + impl->num_inputs], CEED_MEM_HOST, &e_data_full[i + num_input_fields]));
  }

  // Loop through elements
  for (CeedInt e = 0; e < num_blocks * block_size; e += block_size) {
    // Output pointers
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
      if (eval_mode == CEED_EVAL_NONE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &size));
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data_full[i + num_input_fields][e * Q * size]));
      }
    }

    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Blocked(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, false, e_data_full, impl));

    // Q function
    if (!impl->is_identity_qf) {
      CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
    }

    // Output basis apply
    CeedCallBackend(CeedOperatorOutputBasis_Blocked(e, Q, qf_output_fields, op_output_fields, block_size, num_input_fields, num_output_fields, op,
                                                    e_data_full, impl));
  }

  // Output restriction
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector vec;

    // Restore evec
    CeedCallBackend(CeedVectorRestoreArray(impl->e_vecs_full[i + impl->num_inputs], &e_data_full[i + num_input_fields]));
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    // Active
    if (vec == CEED_VECTOR_ACTIVE) vec = out_vec;
    // Restrict
    CeedCallBackend(
        CeedElemRestrictionApply(impl->block_rstr[i + impl->num_inputs], CEED_TRANSPOSE, impl->e_vecs_full[i + impl->num_inputs], vec, request));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Blocked(num_input_fields, qf_input_fields, op_input_fields, false, e_data_full, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for assembling linear QFunction
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Blocked(CeedOperator op, bool build_objects, CeedVector *assembled,
                                                                  CeedElemRestriction *rstr, CeedRequest *request) {
  Ceed                  ceed;
  CeedSize              q_size;
  CeedInt               Q, num_input_fields, num_output_fields, num_elem, size;
  const CeedInt         block_size = 8;
  CeedScalar           *l_vec_array;
  CeedScalar           *e_data_full[2 * CEED_FIELD_MAX] = {0};
  CeedQFunctionField   *qf_input_fields, *qf_output_fields;
  CeedQFunction         qf;
  CeedOperatorField    *op_input_fields, *op_output_fields;
  CeedOperator_Blocked *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedInt             num_active_in = impl->num_active_in, num_active_out = impl->num_active_out;
  CeedVector         *active_in  = impl->qf_active_in;
  CeedVector          l_vec      = impl->qf_l_vec;
  CeedElemRestriction block_rstr = impl->qf_block_rstr;

  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  const CeedInt num_blocks = (num_elem / block_size) + !!(num_elem % block_size);

  // Setup
  CeedCallBackend(CeedOperatorSetup_Blocked(op));

  // Check for restriction only operator
  CeedCheck(!impl->is_identity_rstr_op, ceed, CEED_ERROR_BACKEND, "Assembling restriction only operators is not supported");

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Blocked(num_input_fields, qf_input_fields, op_input_fields, NULL, true, e_data_full, impl, request));

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
          q_size = (CeedSize)Q * block_size;
          CeedCallBackend(CeedVectorCreate(ceed, q_size, &active_in[num_active_in + field]));
          CeedCallBackend(
              CeedVectorSetArray(active_in[num_active_in + field], CEED_MEM_HOST, CEED_USE_POINTER, &q_vec_array[field * Q * block_size]));
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

  // Setup Lvec
  if (!l_vec) {
    const CeedSize l_size = (CeedSize)num_blocks * block_size * Q * num_active_in * num_active_out;

    CeedCallBackend(CeedVectorCreate(ceed, l_size, &l_vec));
    impl->qf_l_vec = l_vec;
  }
  CeedCallBackend(CeedVectorGetArrayWrite(l_vec, CEED_MEM_HOST, &l_vec_array));

  // Setup block restriction
  if (!block_rstr) {
    const CeedInt strides[3] = {1, Q, num_active_in * num_active_out * Q};

    CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, Q, block_size, num_active_in * num_active_out,
                                                            num_active_in * num_active_out * num_elem * Q, strides, &block_rstr));
    impl->qf_block_rstr = block_rstr;
  }

  // Build objects if needed
  if (build_objects) {
    const CeedSize l_size     = (CeedSize)num_elem * Q * num_active_in * num_active_out;
    const CeedInt  strides[3] = {1, Q, num_active_in * num_active_out * Q};

    // Create output restriction
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed, num_elem, Q, num_active_in * num_active_out, num_active_in * num_active_out * num_elem * Q,
                                                     strides, rstr));
    // Create assembled vector
    CeedCallBackend(CeedVectorCreate(ceed, l_size, assembled));
  }

  // Loop through elements
  for (CeedInt e = 0; e < num_blocks * block_size; e += block_size) {
    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Blocked(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, true, e_data_full, impl));

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
            CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_USE_POINTER, l_vec_array));
            CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &size));
            l_vec_array += size * Q * block_size;  // Advance the pointer by the size of the output
          }
        }
        // Apply QFunction
        CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
      } else {
        const CeedScalar *q_vec_array;

        // Copy Identity Outputs
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[0], &size));
        CeedCallBackend(CeedVectorGetArrayRead(impl->q_vecs_out[0], CEED_MEM_HOST, &q_vec_array));
        for (CeedInt i = 0; i < size * Q * block_size; i++) l_vec_array[i] = q_vec_array[i];
        CeedCallBackend(CeedVectorRestoreArrayRead(impl->q_vecs_out[0], &q_vec_array));
        l_vec_array += size * Q * block_size;
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
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL));
      }
    }
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Blocked(num_input_fields, qf_input_fields, op_input_fields, true, e_data_full, impl));

  // Output blocked restriction
  CeedCallBackend(CeedVectorRestoreArray(l_vec, &l_vec_array));
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  CeedCallBackend(CeedElemRestrictionApply(block_rstr, CEED_TRANSPOSE, l_vec, *assembled, request));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Blocked(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Blocked(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Blocked(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Blocked(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Operator Destroy
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Blocked(CeedOperator op) {
  CeedOperator_Blocked *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));

  for (CeedInt i = 0; i < impl->num_inputs + impl->num_outputs; i++) {
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->block_rstr[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_full[i]));
  }
  CeedCallBackend(CeedFree(&impl->block_rstr));
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

  // QFunction assembly data
  for (CeedInt i = 0; i < impl->num_active_in; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->qf_active_in[i]));
  }
  CeedCallBackend(CeedFree(&impl->qf_active_in));
  CeedCallBackend(CeedVectorDestroy(&impl->qf_l_vec));
  CeedCallBackend(CeedElemRestrictionDestroy(&impl->qf_block_rstr));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create
//------------------------------------------------------------------------------
int CeedOperatorCreate_Blocked(CeedOperator op) {
  Ceed                  ceed;
  CeedOperator_Blocked *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Blocked));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Blocked));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Blocked));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Blocked));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
