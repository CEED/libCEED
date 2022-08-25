// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "ceed-opt.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Opt(CeedQFunction qf, CeedOperator op, bool is_input, const CeedInt blk_size, CeedElemRestriction *blk_restr,
                                       CeedVector *e_vecs_full, CeedVector *e_vecs, CeedVector *q_vecs, CeedInt start_e, CeedInt num_fields,
                                       CeedInt Q) {
  CeedInt  num_comp, size, P;
  CeedSize e_size, q_size;
  Ceed     ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedBasis           basis;
  CeedElemRestriction r;
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
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &r));
      Ceed ceed;
      CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
      CeedSize l_size;
      CeedInt  num_elem, elem_size, comp_stride;
      CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
      CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
      CeedCallBackend(CeedElemRestrictionGetLVectorSize(r, &l_size));
      CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));

      bool strided;
      CeedCallBackend(CeedElemRestrictionIsStrided(r, &strided));
      if (strided) {
        CeedInt strides[3];
        CeedCallBackend(CeedElemRestrictionGetStrides(r, &strides));
        CeedCallBackend(
            CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, elem_size, blk_size, num_comp, l_size, strides, &blk_restr[i + start_e]));
      } else {
        const CeedInt *offsets = NULL;
        CeedCallBackend(CeedElemRestrictionGetOffsets(r, CEED_MEM_HOST, &offsets));
        CeedCallBackend(CeedElemRestrictionGetCompStride(r, &comp_stride));
        CeedCallBackend(CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size, blk_size, num_comp, comp_stride, l_size, CEED_MEM_HOST,
                                                         CEED_COPY_VALUES, offsets, &blk_restr[i + start_e]));
        CeedCallBackend(CeedElemRestrictionRestoreOffsets(r, &offsets));
      }
      CeedCallBackend(CeedElemRestrictionCreateVector(blk_restr[i + start_e], NULL, &e_vecs_full[i + start_e]));
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        e_size = (CeedSize)Q * size * blk_size;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        q_size = (CeedSize)Q * size * blk_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        CeedCallBackend(CeedBasisGetNumNodes(basis, &P));
        CeedCallBackend(CeedBasisGetNumComponents(basis, &num_comp));
        e_size = (CeedSize)P * num_comp * blk_size;
        CeedCallBackend(CeedVectorCreate(ceed, e_size, &e_vecs[i]));
        q_size = (CeedSize)Q * size * blk_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_WEIGHT:  // Only on input fields
        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        q_size = (CeedSize)Q * blk_size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        CeedCallBackend(CeedBasisApply(basis, blk_size, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_vecs[i]));

        break;
      case CEED_EVAL_DIV:
        break;  // Not implemented
      case CEED_EVAL_CURL:
        break;  // Not implemented
    }
    if (is_input && !!e_vecs[i]) {
      CeedCallBackend(CeedVectorSetArray(e_vecs[i], CEED_MEM_HOST, CEED_COPY_VALUES, NULL));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Opt(CeedOperator op) {
  bool is_setup_done;
  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  Ceed_Opt *ceed_impl;
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  const CeedInt     blk_size = ceed_impl->blk_size;
  CeedOperator_Opt *impl;
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
  CeedCallBackend(CeedCalloc(num_input_fields + num_output_fields, &impl->blk_restr));
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
  CeedCallBackend(CeedOperatorSetupFields_Opt(qf, op, true, blk_size, impl->blk_restr, impl->e_vecs_full, impl->e_vecs_in, impl->q_vecs_in, 0,
                                              num_input_fields, Q));
  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Opt(qf, op, false, blk_size, impl->blk_restr, impl->e_vecs_full, impl->e_vecs_out, impl->q_vecs_out,
                                              num_input_fields, num_output_fields, Q));

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
// Setup Input Fields
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Opt(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                              CeedVector in_vec, CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl,
                                              CeedRequest *request) {
  CeedEvalMode eval_mode;
  CeedVector   vec;
  uint64_t     state;

  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec != CEED_VECTOR_ACTIVE) {
        // Restrict
        CeedCallBackend(CeedVectorGetState(vec, &state));
        if (state != impl->input_states[i]) {
          CeedCallBackend(CeedElemRestrictionApply(impl->blk_restr[i], CEED_NOTRANSPOSE, vec, impl->e_vecs_full[i], request));
          impl->input_states[i] = state;
        }
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
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Opt(CeedInt e, CeedInt Q, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                             CeedInt num_input_fields, CeedInt blk_size, CeedVector in_vec, bool skip_active,
                                             CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Opt *impl, CeedRequest *request) {
  CeedInt             dim, elem_size, size;
  CeedElemRestriction elem_restr;
  CeedEvalMode        eval_mode;
  CeedBasis           basis;
  CeedVector          vec;

  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    // Skip active input
    if (skip_active) {
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }

    CeedInt active_in = 0;
    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_restr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(elem_restr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
    // Restrict block active input
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->blk_restr[i], e / blk_size, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[i], request));
      active_in = 1;
    }
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        if (!active_in) {
          CeedCallBackend(CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data[i][e * Q * size]));
        }
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        if (!active_in) {
          CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data[i][e * elem_size * size]));
        }
        CeedCallBackend(CeedBasisApply(basis, blk_size, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, impl->e_vecs_in[i], impl->q_vecs_in[i]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        if (!active_in) {
          CeedCallBackend(CeedBasisGetDimension(basis, &dim));
          CeedCallBackend(CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST, CEED_USE_POINTER, &e_data[i][e * elem_size * size / dim]));
        }
        CeedCallBackend(CeedBasisApply(basis, blk_size, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, impl->e_vecs_in[i], impl->q_vecs_in[i]));
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
static inline int CeedOperatorOutputBasis_Opt(CeedInt e, CeedInt Q, CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
                                              CeedInt blk_size, CeedInt num_input_fields, CeedInt num_output_fields, CeedOperator op,
                                              CeedVector out_vec, CeedOperator_Opt *impl, CeedRequest *request) {
  CeedElemRestriction elem_restr;
  CeedEvalMode        eval_mode;
  CeedBasis           basis;
  CeedVector          vec;

  for (CeedInt i = 0; i < num_output_fields; i++) {
    // Get elem_size, eval_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_restr));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    // Basis action
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, blk_size, CEED_TRANSPOSE, CEED_EVAL_INTERP, impl->q_vecs_out[i], impl->e_vecs_out[i]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, blk_size, CEED_TRANSPOSE, CEED_EVAL_GRAD, impl->q_vecs_out[i], impl->e_vecs_out[i]));
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
    // Restrict output block
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) vec = out_vec;
    // Restrict
    CeedCallBackend(
        CeedElemRestrictionApplyBlock(impl->blk_restr[i + impl->num_inputs], e / blk_size, CEED_TRANSPOSE, impl->e_vecs_out[i], vec, request));
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
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Opt(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  Ceed_Opt *ceed_impl;
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  CeedInt           blk_size = ceed_impl->blk_size;
  CeedOperator_Opt *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedInt Q, num_input_fields, num_output_fields, num_elem;
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedInt       num_blks = (num_elem / blk_size) + !!(num_elem % blk_size);
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedOperatorField *op_input_fields, *op_output_fields;
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedEvalMode eval_mode;
  CeedScalar  *e_data[2 * CEED_FIELD_MAX] = {0};

  // Setup
  CeedCallBackend(CeedOperatorSetup_Opt(op));

  // Restriction only operator
  if (impl->is_identity_restr_op) {
    for (CeedInt b = 0; b < num_blks; b++) {
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->blk_restr[0], b, CEED_NOTRANSPOSE, in_vec, impl->e_vecs_in[0], request));
      CeedCallBackend(CeedElemRestrictionApplyBlock(impl->blk_restr[1], b, CEED_TRANSPOSE, impl->e_vecs_in[0], out_vec, request));
    }
    return CEED_ERROR_SUCCESS;
  }

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
  for (CeedInt e = 0; e < num_blks * blk_size; e += blk_size) {
    // Input basis apply
    CeedCallBackend(
        CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, blk_size, in_vec, false, e_data, impl, request));

    // Q function
    if (!impl->is_identity_qf) {
      CeedCallBackend(CeedQFunctionApply(qf, Q * blk_size, impl->q_vecs_in, impl->q_vecs_out));
    }

    // Output basis apply and restrict
    CeedCallBackend(CeedOperatorOutputBasis_Opt(e, Q, qf_output_fields, op_output_fields, blk_size, num_input_fields, num_output_fields, op, out_vec,
                                                impl, request));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, e_data, impl));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for linear QFunction assembly
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Opt(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                              CeedRequest *request) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  Ceed_Opt *ceed_impl;
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  const CeedInt     blk_size = ceed_impl->blk_size;
  CeedSize          q_size;
  CeedOperator_Opt *impl;
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedInt Q, num_input_fields, num_output_fields, num_elem, size;
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedInt       num_blks = (num_elem / blk_size) + !!(num_elem % blk_size);
  CeedQFunction qf;
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedOperatorField *op_input_fields, *op_output_fields;
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedVector  vec, l_vec = impl->qf_l_vec;
  CeedInt     num_active_in = impl->num_active_in, num_active_out = impl->num_active_out;
  CeedVector *active_in = impl->qf_active_in;
  CeedScalar *a, *tmp;
  CeedScalar *e_data[2 * CEED_FIELD_MAX] = {0};

  // Setup
  CeedCallBackend(CeedOperatorSetup_Opt(op));

  // Check for identity
  if (impl->is_identity_qf) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Assembling identity qfunctions not supported");
    // LCOV_EXCL_STOP
  }

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, NULL, e_data, impl, request));

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
          q_size = (CeedSize)Q * blk_size;
          CeedCallBackend(CeedVectorCreate(ceed, q_size, &active_in[num_active_in + field]));
          CeedCallBackend(CeedVectorSetArray(active_in[num_active_in + field], CEED_MEM_HOST, CEED_USE_POINTER, &tmp[field * Q * blk_size]));
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

  // Setup l_vec
  if (!l_vec) {
    CeedSize l_size = (CeedSize)blk_size * Q * num_active_in * num_active_out;
    CeedCallBackend(CeedVectorCreate(ceed, l_size, &l_vec));
    CeedCallBackend(CeedVectorSetValue(l_vec, 0.0));
    impl->qf_l_vec = l_vec;
  }

  // Build objects if needed
  CeedInt strides[3] = {1, Q, num_active_in * num_active_out * Q};
  if (build_objects) {
    // Create output restriction
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed, num_elem, Q, num_active_in * num_active_out, num_active_in * num_active_out * num_elem * Q,
                                                     strides, rstr));
    // Create assembled vector
    CeedSize l_size = (CeedSize)num_elem * Q * num_active_in * num_active_out;
    CeedCallBackend(CeedVectorCreate(ceed, l_size, assembled));
  }

  // Output blocked restriction
  CeedElemRestriction blk_rstr = impl->qf_blk_rstr;
  if (!blk_rstr) {
    CeedCallBackend(CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, Q, blk_size, num_active_in * num_active_out,
                                                            num_active_in * num_active_out * num_elem * Q, strides, &blk_rstr));
    impl->qf_blk_rstr = blk_rstr;
  }

  // Loop through elements
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  for (CeedInt e = 0; e < num_blks * blk_size; e += blk_size) {
    CeedCallBackend(CeedVectorGetArray(l_vec, CEED_MEM_HOST, &a));

    // Input basis apply
    CeedCallBackend(
        CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields, num_input_fields, blk_size, NULL, true, e_data, impl, request));

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
          a += size * Q * blk_size;  // Advance the pointer by the size of the output
        }
      }
      // Apply QFunction
      CeedCallBackend(CeedQFunctionApply(qf, Q * blk_size, impl->q_vecs_in, impl->q_vecs_out));
    }

    // Assemble into assembled vector
    CeedCallBackend(CeedVectorRestoreArray(l_vec, &a));
    CeedCallBackend(CeedElemRestrictionApplyBlock(blk_rstr, e / blk_size, CEED_TRANSPOSE, l_vec, *assembled, request));
  }

  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  for (CeedInt out = 0; out < num_output_fields; out++) {
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &vec));
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_COPY_VALUES, NULL));
    }
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields, op_input_fields, e_data, impl));

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
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->blk_restr[i]));
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs_full[i]));
  }
  CeedCallBackend(CeedFree(&impl->blk_restr));
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
  CeedCallBackend(CeedElemRestrictionDestroy(&impl->qf_blk_rstr));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create
//------------------------------------------------------------------------------
int CeedOperatorCreate_Opt(CeedOperator op) {
  Ceed ceed;
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  Ceed_Opt *ceed_impl;
  CeedCallBackend(CeedGetData(ceed, &ceed_impl));
  CeedInt           blk_size = ceed_impl->blk_size;
  CeedOperator_Opt *impl;

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  if (blk_size != 1 && blk_size != 8) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Opt backend cannot use blocksize: %" CeedInt_FMT, blk_size);
    // LCOV_EXCL_STOP
  }

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Opt));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Opt));
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
