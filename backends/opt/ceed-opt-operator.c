// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
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
      CeedElemRestriction rstr;

      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr));
      CeedCallBackend(CeedElemRestrictionGetBlockedElemRestriction(rstr, block_size, &block_rstr[i + start_e]));
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
// Setup First Touch Masks
//------------------------------------------------------------------------------
static int CeedOperatorSetupFirstTouchMasks_Opt(CeedInt num_ops, CeedElemRestriction *block_rstr, CeedSize l_size, uint8_t **first_touch,
                                                CeedSize *num_untouched, CeedSize **untouched, bool *is_exact) {
  uint8_t *marks;

  // Find the first contribution to each entry, in the order of the operators and of their transpose restrictions
  // Bit 0 marks offsets seen, bit 1 marks entries touched
  *is_exact = true;
  CeedCallBackend(CeedCalloc(l_size, &marks));
  for (CeedInt op_i = 0; op_i < num_ops && *is_exact; op_i++) {
    CeedInt        num_elem, elem_size, num_comp, comp_stride, block_size, num_blocks;
    const CeedInt *offsets;

    if (!block_rstr[op_i]) continue;
    CeedCallBackend(CeedElemRestrictionGetNumElements(block_rstr[op_i], &num_elem));
    CeedCallBackend(CeedElemRestrictionGetElementSize(block_rstr[op_i], &elem_size));
    CeedCallBackend(CeedElemRestrictionGetNumComponents(block_rstr[op_i], &num_comp));
    CeedCallBackend(CeedElemRestrictionGetCompStride(block_rstr[op_i], &comp_stride));
    CeedCallBackend(CeedElemRestrictionGetBlockSize(block_rstr[op_i], &block_size));
    CeedCallBackend(CeedElemRestrictionGetNumBlocks(block_rstr[op_i], &num_blocks));
    CeedCallBackend(CeedCalloc((CeedSize)num_blocks * elem_size, &first_touch[op_i]));
    CeedCallBackend(CeedElemRestrictionGetOffsets(block_rstr[op_i], CEED_MEM_HOST, &offsets));
    for (CeedSize b = 0; b < num_blocks && *is_exact; b++) {
      for (CeedSize n = 0; n < elem_size && *is_exact; n++) {
        for (CeedInt j = 0; j < CeedIntMin(block_size, num_elem - b * block_size) && *is_exact; j++) {
          const CeedInt offset = offsets[(b * elem_size + n) * block_size + j];

          // A repeated offset must have all its entries touched, which suboperators with other component layouts may not give
          if (marks[offset] & 1) {
            for (CeedSize k = 0; k < num_comp && *is_exact; k++) *is_exact = marks[offset + k * comp_stride] & 2;
            continue;
          }
          marks[offset] |= 1;
          // One bit per lane, opt block sizes are 1 and 8
          first_touch[op_i][b * elem_size + n] |= 1 << j;
          // Entries reached from two offsets keep the zero and add path
          for (CeedSize k = 0; k < num_comp && *is_exact; k++) {
            *is_exact = !(marks[offset + k * comp_stride] & 2);
            marks[offset + k * comp_stride] |= 2;
          }
        }
      }
    }
    CeedCallBackend(CeedElemRestrictionRestoreOffsets(block_rstr[op_i], &offsets));
  }

  // Entries without contributions, zeroed by Apply
  if (*is_exact) {
    for (CeedSize i = 0; i < l_size; i++) *num_untouched += !(marks[i] & 2);
    CeedCallBackend(CeedCalloc(*num_untouched, untouched));
    for (CeedSize i = 0, j = 0; i < l_size; i++) {
      if (!(marks[i] & 2)) (*untouched)[j++] = i;
    }
  } else {
    for (CeedInt op_i = 0; op_i < num_ops; op_i++) CeedCallBackend(CeedFree(&first_touch[op_i]));
  }
  CeedCallBackend(CeedFree(&marks));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup First Touch Output
//------------------------------------------------------------------------------
static int CeedOperatorSetupFirstTouch_Opt(CeedOperator op, CeedOperator_Opt *impl) {
  bool                is_active;
  CeedSize            l_size;
  CeedRestrictionType rstr_type;
  CeedVector          vec;
  CeedElemRestriction block_rstr;
  CeedOperatorField  *op_output_fields;

  // Single active output with standard restriction
  if (impl->is_identity_rstr_op || impl->num_outputs != 1) return CEED_ERROR_SUCCESS;
  CeedCallBackend(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_output_fields));
  CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[0], &vec));
  is_active = vec == CEED_VECTOR_ACTIVE;
  CeedCallBackend(CeedVectorDestroy(&vec));
  if (!is_active) return CEED_ERROR_SUCCESS;
  block_rstr = impl->block_rstr[impl->num_inputs];
  CeedCallBackend(CeedElemRestrictionGetType(block_rstr, &rstr_type));
  if (rstr_type != CEED_RESTRICTION_STANDARD) return CEED_ERROR_SUCCESS;
  CeedCallBackend(CeedElemRestrictionGetLVectorSize(block_rstr, &l_size));
  CeedCallBackend(CeedOperatorSetupFirstTouchMasks_Opt(1, &block_rstr, l_size, &impl->first_touch, &impl->num_untouched, &impl->untouched,
                                                       &impl->use_first_touch));
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

  // First touch output
  CeedCallBackend(CeedOperatorSetupFirstTouch_Opt(op, impl));

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
    if (is_active && impl->block_rstr[i] && !impl->skip_rstr_in[i]) {
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
// Output Restriction First Touch
//------------------------------------------------------------------------------
static inline int CeedOperatorOutputRestrictFirstTouch_Opt(CeedInt e, CeedInt block_size, const CeedInt *offsets, CeedScalar *out_array,
                                                           const uint8_t *first_touch, CeedOperator_Opt *impl, CeedRequest *request) {
  CeedInt             num_elem, elem_size, num_comp, comp_stride;
  const CeedScalar   *e_array;
  CeedElemRestriction block_rstr = impl->block_rstr[impl->num_inputs];

  CeedCallBackend(CeedElemRestrictionGetNumElements(block_rstr, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(block_rstr, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(block_rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(block_rstr, &comp_stride));
  CeedCallBackend(CeedVectorGetArrayRead(impl->e_vecs_out[0], CEED_MEM_HOST, &e_array));
  const CeedInt num_lanes = CeedIntMin(block_size, num_elem - e);

  // Transpose restriction order, with the first contribution to each entry overwriting it
  for (CeedSize k = 0; k < num_comp; k++) {
    for (CeedInt n = 0; n < elem_size; n++) {
      const uint8_t first_lanes = first_touch[(CeedSize)(e / block_size) * elem_size + n];

      for (CeedInt j = 0; j < num_lanes; j++) {
        const CeedSize ind = offsets[(CeedSize)e * elem_size + n * block_size + j] + k * comp_stride;

        // 0.0 + e_array gives the same +0.0 as zeroing and adding when e_array is -0.0
        out_array[ind] = ((first_lanes >> j) & 1 ? (CeedScalar)0.0 : out_array[ind]) + e_array[(k * elem_size + n) * block_size + j];
      }
    }
  }
  CeedCallBackend(CeedVectorRestoreArrayRead(impl->e_vecs_out[0], &e_array));
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED) *request = NULL;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Output Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorOutputBasis_Opt(CeedInt e, CeedInt Q, CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
                                              CeedInt block_size, CeedInt num_input_fields, CeedInt num_output_fields, bool *apply_add_basis,
                                              bool *skip_rstr, CeedOperator op, CeedVector out_vec, const CeedInt *out_offsets, CeedScalar *out_array,
                                              const uint8_t *first_touch, CeedOperator_Opt *impl, CeedRequest *request) {
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
    // First touch Apply writes its single active output directly
    if (out_array) {
      CeedCallBackend(CeedOperatorOutputRestrictFirstTouch_Opt(e, block_size, out_offsets, out_array, first_touch, impl, request));
      continue;
    }
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    is_active = vec == CEED_VECTOR_ACTIVE;
    if (is_active) vec = out_vec;
    // Restrict
    CeedCallBackend(CeedElemRestrictionApplyBlock(impl->block_rstr[i + impl->num_inputs], e / block_size, CEED_TRANSPOSE, impl->e_vecs_out[i], vec,
                                                  request));
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
// Core code for operator apply
//------------------------------------------------------------------------------
static inline int CeedOperatorApplyCore_Opt(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedScalar *out_array, const uint8_t *first_touch,
                                            CeedRequest *request) {
  Ceed                ceed;
  Ceed_Opt           *ceed_impl;
  CeedInt             Q, num_input_fields, num_output_fields, num_elem;
  const CeedInt      *out_offsets = NULL;
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

  // First touch output
  if (out_array) CeedCallBackend(CeedElemRestrictionGetOffsets(impl->block_rstr[num_input_fields], CEED_MEM_HOST, &out_offsets));

  // Loop through elements
  for (CeedInt e = 0; e < num_blocks * block_size; e += block_size) {
    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, in_vec, false, e_data, impl,
                                               request));

    // Q function
    if (!impl->is_identity_qf) {
      CeedCallBackend(CeedQFunctionApply(qf, Q * block_size, impl->q_vecs_in, impl->q_vecs_out));
    }

    // Output basis apply and restriction
    CeedCallBackend(CeedOperatorOutputBasis_Opt(e, Q, qf_output_fields, op_output_fields, block_size, num_input_fields, num_output_fields,
                                                impl->apply_add_basis_out, impl->skip_rstr_out, op, out_vec, out_offsets, out_array, first_touch,
                                                impl, request));
  }

  // Restore output offsets and input arrays
  if (out_array) CeedCallBackend(CeedElemRestrictionRestoreOffsets(impl->block_rstr[num_input_fields], &out_offsets));
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, e_data, impl));
  CeedCallBackend(CeedQFunctionDestroy(&qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// First Touch Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyFirstTouch_Opt(CeedInt num_ops, CeedOperator *ops, uint8_t **first_touch, CeedSize num_untouched,
                                           const CeedSize *untouched, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedScalar *out_array;

  // One write access for all operators, which only read entries they wrote
  CeedCallBackend(CeedVectorGetArrayWrite(out_vec, CEED_MEM_HOST, &out_array));
  for (CeedSize i = 0; i < num_untouched; i++) out_array[untouched[i]] = 0.0;
  for (CeedInt i = 0; i < num_ops; i++) {
    if (first_touch[i]) CeedCallBackend(CeedOperatorApplyCore_Opt(ops[i], in_vec, out_vec, out_array, first_touch[i], request));
  }
  CeedCallBackend(CeedVectorRestoreArray(out_vec, &out_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApply_Opt(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  bool              is_first_touch;
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedOperatorSetup_Opt(op));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  is_first_touch = impl->use_first_touch;
  // Output vectors longer than the L-vector also need their tail zeroed
  if (is_first_touch) {
    CeedSize out_size, l_size;

    CeedCallBackend(CeedVectorGetLength(out_vec, &out_size));
    CeedCallBackend(CeedElemRestrictionGetLVectorSize(impl->block_rstr[impl->num_inputs], &l_size));
    is_first_touch = out_size == l_size;
  }
  if (is_first_touch) {
    CeedCallBackend(CeedOperatorApplyFirstTouch_Opt(1, &op, &impl->first_touch, impl->num_untouched, impl->untouched, in_vec, out_vec, request));
  } else {
    if (out_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorSetValue(out_vec, 0.0));
    CeedCallBackend(CeedOperatorApplyAddActive(op, in_vec, out_vec, request));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply Add
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Opt(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedCallBackend(CeedOperatorApplyCore_Opt(op, in_vec, out_vec, NULL, NULL, request));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Composite First Touch Output
//------------------------------------------------------------------------------
static int CeedOperatorSetupFirstTouchComposite_Opt(CeedOperator op, CeedOperator_Opt *impl) {
  bool                 is_setup_done, is_first_touch = true;
  Ceed                 ceed;
  CeedInt              num_sub;
  CeedSize             l_size = -1;
  CeedOperator        *sub_ops;
  CeedElemRestriction *block_rstr;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorCompositeGetNumSub(op, &num_sub));
  CeedCallBackend(CeedOperatorCompositeGetSubList(op, &sub_ops));
  CeedCallBackend(CeedCalloc(num_sub, &block_rstr));
  // Every suboperator with elements must be set up for first touch into the same L-vector
  for (CeedInt i = 0; i < num_sub && is_first_touch; i++) {
    Ceed              ceed_sub;
    CeedInt           num_elem;
    CeedSize          sub_l_size;
    CeedOperator_Opt *sub_impl;

    // Only suboperators of this Ceed are opt operators; operators at points come from the ref delegate, and /cpu/self/gen creates its own
    CeedCallBackend(CeedOperatorGetCeed(sub_ops[i], &ceed_sub));
    is_first_touch = ceed_sub == ceed;
    CeedCallBackend(CeedDestroy(&ceed_sub));
    if (!is_first_touch) break;
    // Suboperators without elements are checked too, as the default path also zeroes their passive outputs
    CeedCallBackend(CeedOperatorSetup_Opt(sub_ops[i]));
    CeedCallBackend(CeedOperatorGetData(sub_ops[i], &sub_impl));
    is_first_touch = sub_impl->use_first_touch;
    if (!is_first_touch) break;
    CeedCallBackend(CeedElemRestrictionGetLVectorSize(sub_impl->block_rstr[sub_impl->num_inputs], &sub_l_size));
    is_first_touch = l_size == -1 || sub_l_size == l_size;
    l_size         = sub_l_size;
    CeedCallBackend(CeedOperatorGetNumElements(sub_ops[i], &num_elem));
    if (num_elem > 0) block_rstr[i] = sub_impl->block_rstr[sub_impl->num_inputs];
  }
  // Suboperators take their lane masks from one walk, so an entry they share takes its first contribution from the first of them
  if (is_first_touch && l_size != -1) {
    CeedCallBackend(CeedCalloc(num_sub, &impl->sub_first_touch));
    CeedCallBackend(CeedOperatorSetupFirstTouchMasks_Opt(num_sub, block_rstr, l_size, impl->sub_first_touch, &impl->num_untouched, &impl->untouched,
                                                         &impl->use_first_touch));
  }
  CeedCallBackend(CeedFree(&block_rstr));
  CeedCallBackend(CeedDestroy(&ceed));
  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Composite Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyComposite_Opt(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  bool              is_first_touch;
  CeedInt           num_sub;
  CeedOperator     *sub_ops;
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorSetupFirstTouchComposite_Opt(op, impl));
  is_first_touch = impl->use_first_touch;
  // Output vectors longer than the L-vector also need their tail zeroed
  if (is_first_touch) {
    CeedSize out_size, l_size;

    CeedCallBackend(CeedVectorGetLength(out_vec, &out_size));
    CeedCallBackend(CeedOperatorGetActiveVectorLengths(op, NULL, &l_size));
    is_first_touch = out_size == l_size;
  }
  if (is_first_touch) {
    CeedCallBackend(CeedOperatorCompositeGetNumSub(op, &num_sub));
    CeedCallBackend(CeedOperatorCompositeGetSubList(op, &sub_ops));
    CeedCallBackend(CeedOperatorApplyFirstTouch_Opt(num_sub, sub_ops, impl->sub_first_touch, impl->num_untouched, impl->untouched, in_vec, out_vec,
                                                    request));
  } else {
    if (out_vec != CEED_VECTOR_NONE) CeedCallBackend(CeedVectorSetValue(out_vec, 0.0));
    CeedCallBackend(CeedOperatorApplyAddActive(op, in_vec, out_vec, request));
  }
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

  // Clear input QFunction buffers
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    // Clear if active input
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
    CeedCallBackend(CeedVectorDestroy(&vec));
  }

  // Clear output vector
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));

  // Loop through elements
  for (CeedInt e = 0; e < num_blocks * block_size; e += block_size) {
    CeedCallBackend(CeedVectorGetArray(l_vec, CEED_MEM_HOST, &l_vec_array));

    // Input basis apply
    CeedCallBackend(CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, block_size, NULL, true, e_data, impl,
                                               request));

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
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionCore_Opt(op, true, assembled, rstr, request));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Opt(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionCore_Opt(op, false, &assembled, &rstr, request));
  return CEED_ERROR_SUCCESS;
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
  CeedCallBackend(CeedFree(&impl->first_touch));
  CeedCallBackend(CeedFree(&impl->untouched));
  if (impl->sub_first_touch) {
    CeedInt num_sub;

    CeedCallBackend(CeedOperatorCompositeGetNumSub(op, &num_sub));
    for (CeedInt i = 0; i < num_sub; i++) CeedCallBackend(CeedFree(&impl->sub_first_touch[i]));
    CeedCallBackend(CeedFree(&impl->sub_first_touch));
  }

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
  bool              is_composite;
  Ceed              ceed;
  Ceed_Opt         *ceed_impl;
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  const CeedInt block_size = ceed_impl->block_size;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCheck(block_size == 1 || block_size == 8, ceed, CEED_ERROR_BACKEND, "Opt backend cannot use blocksize: %" CeedInt_FMT, block_size);

  CeedCallBackend(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyComposite", CeedOperatorApplyComposite_Opt));
  } else {
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Opt));
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Opt));
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Apply", CeedOperatorApply_Opt));
    CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Opt));
  }
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Opt));
  CeedCallBackend(CeedDestroy(&ceed));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
