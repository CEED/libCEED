// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "ceed-opt.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Opt(CeedQFunction qf, CeedOperator op, bool is_input, bool *skip_rstr, bool *apply_add_basis,
                                       const CeedInt block_size, CeedElemRestriction *block_rstr, CeedVector *e_vecs_full, CeedVector *e_vecs,
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
    CeedCallBackend(CeedReferenceCopy(ceed_parent, &ceed));
    CeedCallBackend(CeedDestroy(&ceed_parent));
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

          CeedCallBackend(CeedElemRestrictionGetStrides(rstr, strides));
          CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed_rstr, num_elem, elem_size, block_size, num_comp, l_size, strides,
                                                                  &block_rstr[i + start_e]));
        } break;
        case CEED_RESTRICTION_POINTS:
          // Empty case - won't occur
          break;
      }
      CeedCallBackend(CeedDestroy(&ceed_rstr));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr));
      CeedCallBackend(CeedElemRestrictionCreateVector(block_rstr[i + start_e], NULL, &e_vecs_full[i + start_e]));
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        e_size = (CeedSize)Q * size * block_size;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
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
        CeedCallBackend(CeedBasisDestroy(&basis));
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
        CeedCallBackend(CeedBasisDestroy(&basis));
        break;
    }
    // Initialize E-vec arrays
    if (e_vecs[i]) CeedCallBackend(CeedVectorSetValue(e_vecs[i], 0.0));
  }
  // Drop duplicate restrictions
  if (is_input) {
    for (CeedInt i = 0; i < num_fields; i++) {
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec_i));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr_i));
      for (CeedInt j = i + 1; j < num_fields; j++) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[j], &rstr_j));
        if (vec_i == vec_j && rstr_i == rstr_j) {
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs[i], &e_vecs[j]));
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs_full[i + start_e], &e_vecs_full[j + start_e]));
          skip_rstr[j] = true;
        }
        CeedCallBackend(CeedVectorDestroy(&vec_j));
        CeedCallBackend(CeedElemRestrictionDestroy(&rstr_j));
      }
      CeedCallBackend(CeedVectorDestroy(&vec_i));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr_i));
    }
  } else {
    for (CeedInt i = num_fields - 1; i >= 0; i--) {
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec_i));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr_i));
      for (CeedInt j = i - 1; j >= 0; j--) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[j], &rstr_j));
        if (vec_i == vec_j && rstr_i == rstr_j) {
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs[i], &e_vecs[j]));
          CeedCallBackend(CeedVectorReferenceCopy(e_vecs_full[i + start_e], &e_vecs_full[j + start_e]));
          skip_rstr[j]       = true;
          apply_add_basis[i] = true;
        }
        CeedCallBackend(CeedVectorDestroy(&vec_j));
        CeedCallBackend(CeedElemRestrictionDestroy(&rstr_j));
      }
      CeedCallBackend(CeedVectorDestroy(&vec_i));
      CeedCallBackend(CeedElemRestrictionDestroy(&rstr_i));
    }
  }
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Opt(CeedOperator op) {
  bool                is_setup_done;
  Ceed                ceed;
  Ceed_Opt           *ceed_impl;
  CeedInt             Q, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Opt   *impl;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedQFunctionIsIdentity(qf, &impl->is_identity_qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  const CeedInt block_size = ceed_impl->block_size;

  // Allocate
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->block_rstr));
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->e_vecs_full));

  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->skip_rstr_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->skip_rstr_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->apply_add_basis_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->input_states));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->e_vecs_out));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_in));
  CeedCallBackend(CeedCalloc(CEED_FIELD_MAX, &impl->q_vecs_out));

  impl->num_inputs  = num_input_fields;
  impl->num_outputs = num_output_fields;

  // Set up infield and outfield pointer arrays
  // Infields
  CeedCallBackend(CeedOperatorSetupFields_Opt(qf, op, true, impl->skip_rstr_in, NULL, block_size, impl->block_rstr, impl->e_vecs_full,
                                              impl->e_vecs_in, impl->q_vecs_in, 0, num_input_fields, Q));
  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Opt(qf, op, false, impl->skip_rstr_out, impl->apply_add_basis_out, block_size, impl->block_rstr,
                                              impl->e_vecs_full, impl->e_vecs_out, impl->q_vecs_out, num_input_fields, num_output_fields, Q));

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
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Input Fields
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Opt(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                              CeedVector in_vec, CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl,
                                              CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      uint64_t   state;
      CeedVector vec;

      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec != CEED_VECTOR_ACTIVE) {
        // Restrict
        CeedCallBackend(CeedVectorGetState(vec, &state));
        if (state != impl->input_states[i] && impl->block_rstr[i] && !impl->skip_rstr_in[i]) {
          CeedCallBackend(CeedElemRestrictionApply(impl->block_rstr[i], CEED_NOTRANSPOSE, vec, impl->e_vecs_full[i], request));
        }
        impl->input_states[i] = state;
        // Get evec
        CeedCallBackend(CeedVectorGetArrayRead(impl->e_vecs_full[i], CEED_MEM_HOST, (const CeedScalar **)&e_data[i]));
      } else {
        // Set Qvec for CEED_EVAL_NONE
        if (eval_mode == CEED_EVAL_NONE) {
          CeedCallBackend(CeedVectorGetArrayRead(impl->e_vecs_in[i], CEED_MEM_HOST, (const CeedScalar **)&e_data[i]));
          CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, e_data[i]));
          CeedCallBackend(CeedVectorRestoreArrayRead(impl->e_vecs_in[i], (const CeedScalar **)&e_data[i]));
        }
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Opt(CeedInt e, CeedInt Q, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                             CeedInt num_input_fields, CeedInt block_size, CeedVector in_vec, bool skip_active,
                                             CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    bool                is_active;
    CeedInt             elem_size, size, num_comp;
    CeedEvalMode        eval_mode;
    CeedVector          vec;
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Skip active input
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    is_active = vec == CEED_VECTOR_ACTIVE;
    CeedCallBackend(CeedVectorDestroy(&vec));
    if (skip_active && is_active) continue;

    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    CeedCallBackend(CeedElemRestrictionDestroy(&elem_rstr));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
    // Restrict block active input
    if (is_active && impl->block_rstr[i]) {
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[i], e / block_size, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[i], request));
    }
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        if (!is_active) {
          CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data[i][(CeedSize)e * Q * size]));
        }
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        if (!is_active) {
          CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
          CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data[i][(CeedSize)e * elem_size * num_comp]));
        }
        CeedCallBackend(CeedBasisApply(basis, block_size, CEED_NOTRANSPOSE, eval_mode, impl->e_vecs_in[i], impl->q_vecs_in[i]));
        CeedCallBackend(CeedBasisDestroy(&basis));
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
static inline int CeedOperatorOutputBasis_Opt(CeedInt e, CeedInt Q, CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
                                              CeedInt block_size, CeedInt num_input_fields, CeedInt num_output_fields, bool *apply_add_basis,
                                              bool *skip_rstr, CeedOperator op, CeedVector out_vec, CeedOperator_Opt *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_output_fields; i++) {
    bool         is_active;
    CeedEvalMode eval_mode;
    CeedVector   vec;
    CeedBasis    basis;

    // Get eval_mode
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
        if (apply_add_basis[i]) {
          CeedCallBackend(CeedBasisApplyAdd(basis, block_size, CEED_TRANSPOSE, eval_mode, impl->q_vecs_out[i], impl->e_vecs_out[i]));
        } else {
          CeedCallBackend(CeedBasisApply(basis, block_size, CEED_TRANSPOSE, eval_mode, impl->q_vecs_out[i], impl->e_vecs_out[i]));
        }
        CeedCallBackend(CeedBasisDestroy(&basis));
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        return CeedError(CeedOperatorReturnCeed(op), CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        // LCOV_EXCL_STOP
      }
    }
    // Restrict output block
    if (skip_rstr[i]) continue;
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    is_active = vec == CEED_VECTOR_ACTIVE;
    if (is_active) vec = out_vec;
    // Restrict
    CeedCallBackend(
        CeedElemRestrictionApplyBlock(impl->block_rstr[i + impl->num_inputs], e / block_size, CEED_TRANSPOSE, impl->e_vecs_out[i], vec, request));
    if (!is_active) CeedCallBackend(CeedVectorDestroy(&vec));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Opt(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                                CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;
    CeedVector   vec;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (eval_mode != CEED_EVAL_WEIGHT && vec != CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedVectorRestoreArrayRead(impl->e_vecs_full[i], (const CeedScalar **)&e_data[i]));
    }
    CeedCallBackend(CeedVectorDestroy(&vec));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Opt(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  Ceed                ceed;
  Ceed_Opt           *ceed_impl;
  CeedInt             Q, num_input_fields, num_output_fields, num_elem;
  CeedEvalMode        eval_mode;
  CeedScalar         *e_data[2 * CEED_FIELD_MAX] = {0};
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Opt   *impl;

  // Setup
  CeedCallBackend(CeedOperatorSetup_Opt(op));

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  const CeedInt block_size = ceed_impl->block_size;
  const CeedInt num_blocks = (num_elem / block_size) + !!(num_elem % block_size);

  // Restriction only operator
  if (impl->is_identity_rstr_op) {
    for (CeedInt b = 0; b < num_blocks; b++) {
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[0], b, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[0], request));
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[1], b, CEED_TRANSPOSE, impl->e_vecs_in[0], out_vec, request));
    }
    return CEED_ERROR_SUCCESS;
  }

  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, in_vec, e_data, impl, request));

  // Output Lvecs, Evecs, and Qvecs
  for (CeedInt i = 0; i < num_output_fields; i++) {
    // Set Qvec if needed
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE) {
      // Set qvec to single block evec
      CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs_out[i], CEED_MEM_HOST, &e_data[i + num_input_fields]));
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_HOST, CEED_USE_POINTER, e_data[i + num_input_fields]));
      CeedCallBackend(CeedVectorRestoreArray(impl->e_vecs_out[i], &e_data[i + num_input_fields]));
    }
  }

  // Loop through elements
  for (CeedInt e = 0; e < num_blocks * block_size; e += block_size) {
    // Input basis apply
    CeedCallBackend(
        CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, in_vec, false, e_data, impl, request));

    // Q function
    if (!impl->is_identity_qf) {
      CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
    }

    // Output basis apply and restriction
    CeedCallBackend(CeedOperatorOutputBasis_Opt(e, Q, qf_output_fields, op_output_fields, block_size, num_input_fields, num_output_fields,
                                                impl->apply_add_basis_out, impl->skip_rstr_out, op, out_vec, impl, request));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, e_data, impl));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for linear QFunction assembly
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Opt(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                              CeedRequest *request) {
  Ceed                ceed;
  Ceed_Opt           *ceed_impl;
  CeedInt             qf_size_in, qf_size_out, Q, num_input_fields, num_output_fields, num_elem;
  CeedScalar         *l_vec_array, *e_data[2 * CEED_FIELD_MAX] = {0};
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Opt   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  qf_size_in  = impl->qf_size_in;
  qf_size_out = impl->qf_size_out;

  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  const CeedInt       block_size = ceed_impl->block_size;
  const CeedInt       num_blocks = (num_elem / block_size) + !!(num_elem % block_size);
  CeedVector          l_vec      = impl->qf_l_vec;
  CeedElemRestriction block_rstr = impl->qf_block_rstr;

  // Setup
  CeedCallBackend(CeedOperatorSetup_Opt(op));

  // Check for restriction only operator
  CeedCheck(!impl->is_identity_rstr_op, ceed, CEED_ERROR_BACKEND, "Assembling restriction only operators is not supported");

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, NULL, e_data, impl, request));

  // Count number of active input fields
  if (qf_size_in == 0) {
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedInt    field_size;
      CeedVector vec;

      // Check if active input
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
        CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
        qf_size_in += field_size;
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
    CeedCheck(qf_size_in > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
    impl->qf_size_in = qf_size_in;
  }

  // Count number of active output fields
  if (qf_size_out == 0) {
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedInt    field_size;
      CeedVector vec;

      // Check if active output
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &field_size));
        qf_size_out += field_size;
      }
      CeedCallBackend(CeedVectorDestroy(&vec));
    }
    CeedCheck(qf_size_out > 0, ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
    impl->qf_size_out = qf_size_out;
  }

  // Setup l_vec
  if (!l_vec) {
    const CeedSize l_size = (CeedSize)block_size * Q * qf_size_in * qf_size_out;

    CeedCallBackend(CeedVectorCreate(ceed, l_size, &l_vec));
    CeedCallBackend(CeedVectorSetValue(l_vec, 0.0));
    impl->qf_l_vec = l_vec;
  }

  // Output blocked restriction
  if (!block_rstr) {
    CeedInt strides[3] = {1, Q, qf_size_in * qf_size_out * Q};

    CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, Q, block_size, qf_size_in * qf_size_out,
                                                            qf_size_in * qf_size_out * num_elem * Q, strides, &block_rstr));
    impl->qf_block_rstr = block_rstr;
  }

  // Build objects if needed
  if (build_objects) {
    const CeedSize l_size     = (CeedSize)num_elem * Q * qf_size_in * qf_size_out;
    CeedInt        strides[3] = {1, Q, qf_size_in * qf_size_out * Q};

    // Create output restriction
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed, num_elem, Q, qf_size_in * qf_size_out,
                                                     (CeedSize)qf_size_in * (CeedSize)qf_size_out * (CeedSize)num_elem * (CeedSize)Q, strides, rstr));
    // Create assembled vector
    CeedCallBackend(CeedVectorCreate(ceed, l_size, assembled));
  }

  // Loop through elements
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  for (CeedInt e = 0; e < num_blocks * block_size; e += block_size) {
    CeedCallBackend(CeedVectorGetArray(l_vec, CEED_MEM_HOST, &l_vec_array));

    // Input basis apply
    CeedCallBackend(
        CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, NULL, true, e_data, impl, request));

    // Assemble QFunction
    for (CeedInt i = 0; i < num_input_fields; i++) {
      bool       is_active;
      CeedInt    field_size;
      CeedVector vec;

      // Check if active input
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      is_active = vec == CEED_VECTOR_ACTIVE;
      CeedCallBackend(CeedVectorDestroy(&vec));
      if (!is_active) continue;
      CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &field_size));
      for (CeedInt field = 0; field < field_size; field++) {
        // Set current portion of input to 1.0
        {
          CeedScalar *array;

          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 1.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
        }

        if (!impl->is_identity_qf) {
          // Set Outputs
          for (CeedInt out = 0; out < num_output_fields; out++) {
            CeedVector vec;

            // Check if active output
            CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
            if (vec == CEED_VECTOR_ACTIVE) {
              CeedInt field_size;

              CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_USE_POINTER, l_vec_array));
              CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &field_size));
              l_vec_array += field_size * Q * block_size;  // Advance the pointer by the size of the output
            }
            CeedCallBackend(CeedVectorDestroy(&vec));
          }
          // Apply QFunction
          CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
        } else {
          CeedInt           field_size;
          const CeedScalar *array;

          // Copy Identity Outputs
          CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[0], &field_size));
          CeedCallBackend(CeedVectorGetArrayRead(impl->q_vecs_out[0], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < field_size * Q * block_size; j++) l_vec_array[j] = array[j];
          CeedCallBackend(CeedVectorRestoreArrayRead(impl->q_vecs_out[0], &array));
          l_vec_array += field_size * Q * block_size;
        }
        // Reset input to 0.0
        {
          CeedScalar *array;

          CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &array));
          for (CeedInt j = 0; j < Q * block_size; j++) array[field * Q * block_size + j] = 0.0;
          CeedCallBackend(CeedVectorRestoreArray(impl->q_vecs_in[i], &array));
        }
      }
    }

    // Un-set output Qvecs to prevent accidental overwrite of Assembled
    if (!impl->is_identity_qf) {
      for (CeedInt out = 0; out < num_output_fields; out++) {
        CeedVector vec;

        // Check if active output
        CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
        if (vec == CEED_VECTOR_ACTIVE && num_elem > 0) {
          CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL));
        }
        CeedCallBackend(CeedVectorDestroy(&vec));
      }
    }

    // Assemble into assembled vector
    CeedCallBackend(CeedVectorRestoreArray(l_vec, &l_vec_array));
    CeedCallBackend(CeedElemRestrictionApplyBlock(block_rstr, e / block_size, CEED_TRANSPOSE, l_vec, *assembled, request));
  }

  // Reset output Qvecs
  for (CeedInt out = 0; out < num_output_fields; out++) {
    CeedVector vec;

    // Initialize array if active output
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
    if (vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorSetValue(impl->q_vecs_out[out], 0.0));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, e_data, impl));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Opt(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Opt(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Opt(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Opt(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Operator Destroy
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Opt(CeedOperator op) {
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  for (CeedInt i = 0; i < impl->num_inputs + impl->num_outputs; i++) {
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->block_rstr[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_full[i]));
  }
  CeedCallBackend(CeedFree(&impl->block_rstr));
  CeedCallBackend(CeedFree(&impl->e_vecs_full));
  CeedCallBackend(CeedFree(&impl->input_states));
  CeedCallBackend(CeedFree(&impl->skip_rstr_in));
  CeedCallBackend(CeedFree(&impl->skip_rstr_out));
  CeedCallBackend(CeedFree(&impl->apply_add_basis_out));

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
  CeedCallBackend(CeedVectorDestroy(&impl->qf_l_vec));
  CeedCallBackend(CeedElemRestrictionDestroy(&impl->qf_block_rstr));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create
//------------------------------------------------------------------------------
int CeedOperatorCreate_Opt(CeedOperator op) {
  Ceed              ceed;
  Ceed_Opt         *ceed_impl;
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  const CeedInt block_size = ceed_impl->block_size;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCheck(block_size == 1 || block_size == 8, ceed, CEED_ERROR_BACKEND, "Opt backend cannot use blocksize: %" CeedInt_FMT, block_size);

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Opt));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
