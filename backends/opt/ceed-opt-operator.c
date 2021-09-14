// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "ceed-opt.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Opt(CeedQFunction qf, CeedOperator op,
                                       bool inOrOut, const CeedInt blk_size,
                                       CeedElemRestriction *blk_restr,
                                       CeedVector *full_evecs, CeedVector *e_vecs,
                                       CeedVector *q_vecs, CeedInt start_e,
                                       CeedInt num_fields, CeedInt Q) {
  CeedInt dim, ierr, num_comp, size, P;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedBasis basis;
  CeedElemRestriction r;
  CeedOperatorField *op_fields;
  CeedQFunctionField *qf_fields;
  if (inOrOut) {
    ierr = CeedOperatorGetFields(op, NULL, NULL, NULL,&op_fields);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields);
    CeedChkBackend(ierr);
  } else {
    ierr = CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL);
    CeedChkBackend(ierr);
  }

  // Loop over fields
  for (CeedInt i=0; i<num_fields; i++) {
    CeedEvalMode eval_mode;
    ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode);
    CeedChkBackend(ierr);

    if (eval_mode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(op_fields[i], &r);
      CeedChkBackend(ierr);
      Ceed ceed;
      ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChkBackend(ierr);
      CeedInt num_elem, elem_size, l_size, comp_stride;
      ierr = CeedElemRestrictionGetNumElements(r, &num_elem); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(r, &elem_size); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetLVectorSize(r, &l_size); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetNumComponents(r, &num_comp); CeedChkBackend(ierr);

      bool strided;
      ierr = CeedElemRestrictionIsStrided(r, &strided); CeedChkBackend(ierr);
      if (strided) {
        CeedInt strides[3];
        ierr = CeedElemRestrictionGetStrides(r, &strides); CeedChkBackend(ierr);
        ierr = CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, elem_size,
               blk_size, num_comp, l_size, strides, &blk_restr[i+start_e]);
        CeedChkBackend(ierr);
      } else {
        const CeedInt *offsets = NULL;
        ierr = CeedElemRestrictionGetOffsets(r, CEED_MEM_HOST, &offsets);
        CeedChkBackend(ierr);
        ierr = CeedElemRestrictionGetCompStride(r, &comp_stride); CeedChkBackend(ierr);
        ierr = CeedElemRestrictionCreateBlocked(ceed, num_elem, elem_size,
                                                blk_size, num_comp, comp_stride,
                                                l_size, CEED_MEM_HOST,
                                                CEED_COPY_VALUES, offsets,
                                                &blk_restr[i+start_e]);
        CeedChkBackend(ierr);
        ierr = CeedElemRestrictionRestoreOffsets(r, &offsets); CeedChkBackend(ierr);
      }
      ierr = CeedElemRestrictionCreateVector(blk_restr[i+start_e], NULL,
                                             &full_evecs[i+start_e]);
      CeedChkBackend(ierr);
    }

    switch(eval_mode) {
    case CEED_EVAL_NONE:
      ierr = CeedQFunctionFieldGetSize(qf_fields[i], &size); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blk_size, &e_vecs[i]);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blk_size, &q_vecs[i]);
      CeedChkBackend(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedQFunctionFieldGetSize(qf_fields[i], &size); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(r, &P);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, P*size*blk_size, &e_vecs[i]);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blk_size, &q_vecs[i]);
      CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetSize(qf_fields[i], &size); CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(r, &P);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, P*size/dim*blk_size, &e_vecs[i]);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size*blk_size, &q_vecs[i]);
      CeedChkBackend(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*blk_size, &q_vecs[i]); CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, blk_size, CEED_NOTRANSPOSE,
                            CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_vecs[i]);
      CeedChkBackend(ierr);

      break;
    case CEED_EVAL_DIV:
      break; // Not implemented
    case CEED_EVAL_CURL:
      break; // Not implemented
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Opt(CeedOperator op) {
  int ierr;
  bool setup_done;
  ierr = CeedOperatorIsSetupDone(op, &setup_done); CeedChkBackend(ierr);
  if (setup_done) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  Ceed_Opt *ceed_impl;
  ierr = CeedGetData(ceed, &ceed_impl); CeedChkBackend(ierr);
  const CeedInt blk_size = ceed_impl->blk_size;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, num_input_fields, num_output_fields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedQFunctionIsIdentity(qf, &impl->is_identity_qf); CeedChkBackend(ierr);
  CeedOperatorField *op_input_fields, *op_output_fields;
  ierr = CeedOperatorGetFields(op, &num_input_fields, &op_input_fields,
                               &num_output_fields, &op_output_fields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL,
                                &qf_output_fields);
  CeedChkBackend(ierr);

  // Allocate
  ierr = CeedCalloc(num_input_fields + num_output_fields, &impl->blk_restr);
  CeedChkBackend(ierr);
  ierr = CeedCalloc(num_input_fields + num_output_fields, &impl->e_vecs);
  CeedChkBackend(ierr);
  ierr = CeedCalloc(num_input_fields + num_output_fields, &impl->e_data);
  CeedChkBackend(ierr);

  ierr = CeedCalloc(16, &impl->input_state); CeedChkBackend(ierr);
  ierr = CeedCalloc(16, &impl->e_vecs_in); CeedChkBackend(ierr);
  ierr = CeedCalloc(16, &impl->e_vecs_out); CeedChkBackend(ierr);
  ierr = CeedCalloc(16, &impl->q_vecs_in); CeedChkBackend(ierr);
  ierr = CeedCalloc(16, &impl->q_vecs_out); CeedChkBackend(ierr);

  impl->num_e_vecs_in = num_input_fields;
  impl->num_e_vecs_out = num_output_fields;

  // Set up infield and outfield pointer arrays
  // Infields
  ierr = CeedOperatorSetupFields_Opt(qf, op, 0, blk_size, impl->blk_restr,
                                     impl->e_vecs, impl->e_vecs_in,
                                     impl->q_vecs_in, 0, num_input_fields, Q);
  CeedChkBackend(ierr);
  // Outfields
  ierr = CeedOperatorSetupFields_Opt(qf, op, 1, blk_size, impl->blk_restr,
                                     impl->e_vecs, impl->e_vecs_out,
                                     impl->q_vecs_out, num_input_fields,
                                     num_output_fields, Q);
  CeedChkBackend(ierr);

  // Identity QFunctions
  if (impl->is_identity_qf) {
    CeedEvalMode in_mode, out_mode;
    CeedQFunctionField *in_fields, *out_fields;
    ierr = CeedQFunctionGetFields(qf, NULL, &in_fields, NULL, &out_fields);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(in_fields[0], &in_mode);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(out_fields[0], &out_mode);
    CeedChkBackend(ierr);

    if (in_mode == CEED_EVAL_NONE && out_mode == CEED_EVAL_NONE) {
      impl->is_identity_restr_op = true;
    } else {
      ierr = CeedVectorDestroy(&impl->q_vecs_out[0]); CeedChkBackend(ierr);
      impl->q_vecs_out[0] = impl->q_vecs_in[0];
      ierr = CeedVectorAddReference(impl->q_vecs_in[0]); CeedChkBackend(ierr);
    }
  }

  ierr = CeedOperatorSetSetupDone(op); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Input Fields
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Opt(CeedInt num_input_fields,
    CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
    CeedVector in_vec, CeedOperator_Opt *impl, CeedRequest *request) {
  CeedInt ierr;
  CeedEvalMode eval_mode;
  CeedVector vec;
  uint64_t state;

  for (CeedInt i=0; i<num_input_fields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    if (eval_mode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Get input vector
      ierr = CeedOperatorFieldGetVector(op_input_fields[i], &vec);
      CeedChkBackend(ierr);
      if (vec != CEED_VECTOR_ACTIVE) {
        // Restrict
        ierr = CeedVectorGetState(vec, &state); CeedChkBackend(ierr);
        if (state != impl->input_state[i]) {
          ierr = CeedElemRestrictionApply(impl->blk_restr[i], CEED_NOTRANSPOSE,
                                          vec, impl->e_vecs[i], request);
          CeedChkBackend(ierr);
          impl->input_state[i] = state;
        }
      } else {
        // Set Qvec for CEED_EVAL_NONE
        if (eval_mode == CEED_EVAL_NONE) {
          ierr = CeedVectorGetArray(impl->e_vecs_in[i], CEED_MEM_HOST,
                                    &impl->e_data[i]); CeedChkBackend(ierr);
          ierr = CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST,
                                    CEED_USE_POINTER,
                                    impl->e_data[i]); CeedChkBackend(ierr);
          ierr = CeedVectorRestoreArray(impl->e_vecs_in[i],
                                        &impl->e_data[i]); CeedChkBackend(ierr);
        }
      }
      // Get evec
      ierr = CeedVectorGetArrayRead(impl->e_vecs[i], CEED_MEM_HOST,
                                    (const CeedScalar **) &impl->e_data[i]);
      CeedChkBackend(ierr);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Opt(CeedInt e, CeedInt Q,
    CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
    CeedInt num_input_fields, CeedInt blk_size, CeedVector in_vec, bool skip_active,
    CeedOperator_Opt *impl, CeedRequest *request) {
  CeedInt ierr;
  CeedInt dim, elem_size, size;
  CeedElemRestriction elem_restr;
  CeedEvalMode eval_mode;
  CeedBasis basis;
  CeedVector vec;

  for (CeedInt i=0; i<num_input_fields; i++) {
    ierr = CeedOperatorFieldGetVector(op_input_fields[i], &vec);
    CeedChkBackend(ierr);
    // Skip active input
    if (skip_active) {
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }

    CeedInt active_in = 0;
    // Get elem_size, eval_mode, size
    ierr = CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_restr);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(elem_restr, &elem_size);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetSize(qf_input_fields[i], &size);
    CeedChkBackend(ierr);
    // Restrict block active input
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedElemRestrictionApplyBlock(impl->blk_restr[i], e/blk_size,
                                           CEED_NOTRANSPOSE, in_vec,
                                           impl->e_vecs_in[i], request);
      CeedChkBackend(ierr);
      active_in = 1;
    }
    // Basis action
    switch(eval_mode) {
    case CEED_EVAL_NONE:
      if (!active_in) {
        ierr = CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &impl->e_data[i][e*Q*size]); CeedChkBackend(ierr);
      }
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(op_input_fields[i], &basis);
      CeedChkBackend(ierr);
      if (!active_in) {
        ierr = CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &impl->e_data[i][e*elem_size*size]);
        CeedChkBackend(ierr);
      }
      ierr = CeedBasisApply(basis, blk_size, CEED_NOTRANSPOSE,
                            CEED_EVAL_INTERP, impl->e_vecs_in[i],
                            impl->q_vecs_in[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(op_input_fields[i], &basis);
      CeedChkBackend(ierr);
      if (!active_in) {
        ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
        ierr = CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &impl->e_data[i][e*elem_size*size/dim]);
        CeedChkBackend(ierr);
      }
      ierr = CeedBasisApply(basis, blk_size, CEED_NOTRANSPOSE,
                            CEED_EVAL_GRAD, impl->e_vecs_in[i],
                            impl->q_vecs_in[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_WEIGHT:
      break;  // No action
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL: {
      ierr = CeedOperatorFieldGetBasis(op_input_fields[i], &basis);
      CeedChkBackend(ierr);
      Ceed ceed;
      ierr = CeedBasisGetCeed(basis, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Ceed evaluation mode not implemented");
      // LCOV_EXCL_STOP
    }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Output Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorOutputBasis_Opt(CeedInt e, CeedInt Q,
    CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
    CeedInt blk_size, CeedInt num_input_fields, CeedInt num_output_fields,
    CeedOperator op, CeedVector out_vec, CeedOperator_Opt *impl,
    CeedRequest *request) {
  CeedInt ierr;
  CeedElemRestriction elem_restr;
  CeedEvalMode eval_mode;
  CeedBasis basis;
  CeedVector vec;

  for (CeedInt i=0; i<num_output_fields; i++) {
    // Get elem_size, eval_mode, size
    ierr = CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_restr);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    // Basis action
    switch(eval_mode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(op_output_fields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, blk_size, CEED_TRANSPOSE,
                            CEED_EVAL_INTERP, impl->q_vecs_out[i],
                            impl->e_vecs_out[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(op_output_fields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, blk_size, CEED_TRANSPOSE,
                            CEED_EVAL_GRAD, impl->q_vecs_out[i],
                            impl->e_vecs_out[i]); CeedChkBackend(ierr);
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_WEIGHT: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "CEED_EVAL_WEIGHT cannot be an output "
                       "evaluation mode");
    }
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL: {
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Ceed evaluation mode not implemented");
      // LCOV_EXCL_STOP
    }
    }
    // Restrict output block
    // Get output vector
    ierr = CeedOperatorFieldGetVector(op_output_fields[i], &vec);
    CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE)
      vec = out_vec;
    // Restrict
    ierr = CeedElemRestrictionApplyBlock(impl->blk_restr[i+impl->num_e_vecs_in],
                                         e/blk_size, CEED_TRANSPOSE,
                                         impl->e_vecs_out[i], vec, request);
    CeedChkBackend(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Opt(CeedInt num_input_fields,
    CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
    CeedOperator_Opt *impl) {
  CeedInt ierr;
  CeedEvalMode eval_mode;

  for (CeedInt i=0; i<num_input_fields; i++) {
    ierr = CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    if (eval_mode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      ierr = CeedVectorRestoreArrayRead(impl->e_vecs[i],
                                        (const CeedScalar **) &impl->e_data[i]);
      CeedChkBackend(ierr);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Apply
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Opt(CeedOperator op, CeedVector in_vec,
                                    CeedVector out_vec, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  Ceed_Opt *ceed_impl;
  ierr = CeedGetData(ceed, &ceed_impl); CeedChkBackend(ierr);
  CeedInt blk_size = ceed_impl->blk_size;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedInt Q, num_input_fields, num_output_fields, num_elem;
  ierr = CeedOperatorGetNumElements(op, &num_elem); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  CeedInt num_blks = (num_elem/blk_size) + !!(num_elem%blk_size);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedOperatorField *op_input_fields, *op_output_fields;
  ierr = CeedOperatorGetFields(op, &num_input_fields, &op_input_fields,
                               &num_output_fields, &op_output_fields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL,
                                &qf_output_fields);
  CeedChkBackend(ierr);
  CeedEvalMode eval_mode;

  // Setup
  ierr = CeedOperatorSetup_Opt(op); CeedChkBackend(ierr);

  // Restriction only operator
  if (impl->is_identity_restr_op) {
    for (CeedInt b=0; b<num_blks; b++) {
      ierr = CeedElemRestrictionApplyBlock(impl->blk_restr[0], b, CEED_NOTRANSPOSE,
                                           in_vec, impl->e_vecs_in[0], request); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionApplyBlock(impl->blk_restr[1], b, CEED_TRANSPOSE,
                                           impl->e_vecs_in[0], out_vec, request); CeedChkBackend(ierr);
    }
    return CEED_ERROR_SUCCESS;
  }

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields,
                                     op_input_fields, in_vec, impl, request);
  CeedChkBackend(ierr);

  // Output Lvecs, Evecs, and Qvecs
  for (CeedInt i=0; i<num_output_fields; i++) {
    // Set Qvec if needed
    ierr = CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    if (eval_mode == CEED_EVAL_NONE) {
      // Set qvec to single block evec
      ierr = CeedVectorGetArray(impl->e_vecs_out[i], CEED_MEM_HOST,
                                &impl->e_data[i + num_input_fields]);
      CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_HOST,
                                CEED_USE_POINTER, impl->e_data[i + num_input_fields]); CeedChkBackend(ierr);
      ierr = CeedVectorRestoreArray(impl->e_vecs_out[i],
                                    &impl->e_data[i + num_input_fields]);
      CeedChkBackend(ierr);
    }
  }

  // Loop through elements
  for (CeedInt e=0; e<num_blks*blk_size; e+=blk_size) {
    // Input basis apply
    ierr = CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields,
                                      num_input_fields, blk_size, in_vec, false,
                                      impl, request); CeedChkBackend(ierr);

    // Q function
    if (!impl->is_identity_qf) {
      ierr = CeedQFunctionApply(qf, Q*blk_size, impl->q_vecs_in, impl->q_vecs_out);
      CeedChkBackend(ierr);
    }

    // Output basis apply and restrict
    ierr = CeedOperatorOutputBasis_Opt(e, Q, qf_output_fields, op_output_fields,
                                       blk_size, num_input_fields, num_output_fields,
                                       op, out_vec, impl, request);
    CeedChkBackend(ierr);
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields,
                                       op_input_fields, impl);
  CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for linear QFunction assembly
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Opt(CeedOperator op,
    bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
    CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  Ceed_Opt *ceed_impl;
  ierr = CeedGetData(ceed, &ceed_impl); CeedChkBackend(ierr);
  const CeedInt blk_size = ceed_impl->blk_size;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedInt Q, num_input_fields, num_output_fields, num_elem, size;
  ierr = CeedOperatorGetNumElements(op, &num_elem); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  CeedInt num_blks = (num_elem/blk_size) + !!(num_elem%blk_size);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedOperatorField *op_input_fields, *op_output_fields;
  ierr = CeedOperatorGetFields(op, &num_input_fields, &op_input_fields,
                               &num_output_fields, &op_output_fields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL,
                                &qf_output_fields);
  CeedChkBackend(ierr);
  CeedVector vec, lvec;
  CeedInt num_active_in = 0, num_active_out = 0;
  CeedVector *active_in = NULL;
  CeedScalar *a, *tmp;

  // Setup
  ierr = CeedOperatorSetup_Opt(op); CeedChkBackend(ierr);

  // Check for identity
  if (impl->is_identity_qf)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Assembling identity qfunctions not supported");
  // LCOV_EXCL_STOP

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Opt(num_input_fields, qf_input_fields,
                                     op_input_fields, NULL, impl, request);
  CeedChkBackend(ierr);

  // Count number of active input fields
  for (CeedInt i=0; i<num_input_fields; i++) {
    // Get input vector
    ierr = CeedOperatorFieldGetVector(op_input_fields[i], &vec);
    CeedChkBackend(ierr);
    // Check if active input
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedQFunctionFieldGetSize(qf_input_fields[i], &size);
      CeedChkBackend(ierr);
      ierr = CeedVectorSetValue(impl->q_vecs_in[i], 0.0); CeedChkBackend(ierr);
      ierr = CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_HOST, &tmp);
      CeedChkBackend(ierr);
      ierr = CeedRealloc(num_active_in + size, &active_in); CeedChkBackend(ierr);
      for (CeedInt field=0; field<size; field++) {
        ierr = CeedVectorCreate(ceed, Q*blk_size, &active_in[num_active_in+field]);
        CeedChkBackend(ierr);
        ierr = CeedVectorSetArray(active_in[num_active_in+field], CEED_MEM_HOST,
                                  CEED_USE_POINTER, &tmp[field*Q*blk_size]);
        CeedChkBackend(ierr);
      }
      num_active_in += size;
      ierr = CeedVectorRestoreArray(impl->q_vecs_in[i], &tmp); CeedChkBackend(ierr);
    }
  }

  // Count number of active output fields
  for (CeedInt i=0; i<num_output_fields; i++) {
    // Get output vector
    ierr = CeedOperatorFieldGetVector(op_output_fields[i], &vec);
    CeedChkBackend(ierr);
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedQFunctionFieldGetSize(qf_output_fields[i], &size);
      CeedChkBackend(ierr);
      num_active_out += size;
    }
  }

  // Check sizes
  if (!num_active_in || !num_active_out)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Cannot assemble QFunction without active inputs "
                     "and outputs");
  // LCOV_EXCL_STOP

  // Setup lvec
  ierr = CeedVectorCreate(ceed, num_blks*blk_size*Q*num_active_in*num_active_out,
                          &lvec); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(lvec, CEED_MEM_HOST, &a); CeedChkBackend(ierr);

  // Build objects if needed
  CeedInt strides[3] = {1, Q, num_active_in *num_active_out*Q};
  if (build_objects) {
    // Create output restriction
    ierr = CeedElemRestrictionCreateStrided(ceed, num_elem, Q,
                                            num_active_in*num_active_out,
                                            num_active_in*num_active_out*num_elem*Q,
                                            strides, rstr); CeedChkBackend(ierr);
    // Create assembled vector
    ierr = CeedVectorCreate(ceed, num_elem*Q*num_active_in*num_active_out,
                            assembled); CeedChkBackend(ierr);
  }

  // Loop through elements
  for (CeedInt e=0; e<num_blks*blk_size; e+=blk_size) {
    // Input basis apply
    ierr = CeedOperatorInputBasis_Opt(e, Q, qf_input_fields, op_input_fields,
                                      num_input_fields, blk_size, NULL, true,
                                      impl, request); CeedChkBackend(ierr);

    // Assemble QFunction
    for (CeedInt in=0; in<num_active_in; in++) {
      // Set Inputs
      ierr = CeedVectorSetValue(active_in[in], 1.0); CeedChkBackend(ierr);
      if (num_active_in > 1) {
        ierr = CeedVectorSetValue(active_in[(in+num_active_in-1)%num_active_in],
                                  0.0); CeedChkBackend(ierr);
      }
      // Set Outputs
      for (CeedInt out=0; out<num_output_fields; out++) {
        // Get output vector
        ierr = CeedOperatorFieldGetVector(op_output_fields[out], &vec);
        CeedChkBackend(ierr);
        // Check if active output
        if (vec == CEED_VECTOR_ACTIVE) {
          CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST,
                             CEED_USE_POINTER, a); CeedChkBackend(ierr);
          ierr = CeedQFunctionFieldGetSize(qf_output_fields[out], &size);
          CeedChkBackend(ierr);
          a += size*Q*blk_size; // Advance the pointer by the size of the output
        }
      }
      // Apply QFunction
      ierr = CeedQFunctionApply(qf, Q*blk_size, impl->q_vecs_in, impl->q_vecs_out);
      CeedChkBackend(ierr);
    }
  }

  // Un-set output Qvecs to prevent accidental overwrite of Assembled
  for (CeedInt out=0; out<num_output_fields; out++) {
    // Get output vector
    ierr = CeedOperatorFieldGetVector(op_output_fields[out], &vec);
    CeedChkBackend(ierr);
    // Check if active output
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_HOST, CEED_COPY_VALUES,
                         NULL); CeedChkBackend(ierr);
    }
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Opt(num_input_fields, qf_input_fields,
                                       op_input_fields, impl);
  CeedChkBackend(ierr);

  // Output blocked restriction
  ierr = CeedVectorRestoreArray(lvec, &a); CeedChkBackend(ierr);
  ierr = CeedVectorSetValue(*assembled, 0.0); CeedChkBackend(ierr);
  CeedElemRestriction blk_rstr;
  ierr = CeedElemRestrictionCreateBlockedStrided(ceed, num_elem, Q, blk_size,
         num_active_in*num_active_out, num_active_in*num_active_out*num_elem*Q,
         strides, &blk_rstr); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionApply(blk_rstr, CEED_TRANSPOSE, lvec, *assembled,
                                  request); CeedChkBackend(ierr);

  // Cleanup
  for (CeedInt i=0; i<num_active_in; i++) {
    ierr = CeedVectorDestroy(&active_in[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&active_in); CeedChkBackend(ierr);
  ierr = CeedVectorDestroy(&lvec); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&blk_rstr); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Opt(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Opt(op, true, assembled, rstr,
         request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Opt(CeedOperator op,
    CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Opt(op, false, &assembled,
         &rstr, request);
}

//------------------------------------------------------------------------------
// Operator Destroy
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Opt(CeedOperator op) {
  int ierr;
  CeedOperator_Opt *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);

  for (CeedInt i=0; i<impl->num_e_vecs_in+impl->num_e_vecs_out; i++) {
    ierr = CeedElemRestrictionDestroy(&impl->blk_restr[i]); CeedChkBackend(ierr);
    ierr = CeedVectorDestroy(&impl->e_vecs[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->blk_restr); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->e_vecs); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->e_data); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->input_state); CeedChkBackend(ierr);

  for (CeedInt i=0; i<impl->num_e_vecs_in; i++) {
    ierr = CeedVectorDestroy(&impl->e_vecs_in[i]); CeedChkBackend(ierr);
    ierr = CeedVectorDestroy(&impl->q_vecs_in[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->e_vecs_in); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->q_vecs_in); CeedChkBackend(ierr);

  for (CeedInt i=0; i<impl->num_e_vecs_out; i++) {
    ierr = CeedVectorDestroy(&impl->e_vecs_out[i]); CeedChkBackend(ierr);
    ierr = CeedVectorDestroy(&impl->q_vecs_out[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&impl->e_vecs_out); CeedChkBackend(ierr);
  ierr = CeedFree(&impl->q_vecs_out); CeedChkBackend(ierr);

  ierr = CeedFree(&impl); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Create
//------------------------------------------------------------------------------
int CeedOperatorCreate_Opt(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  Ceed_Opt *ceed_impl;
  ierr = CeedGetData(ceed, &ceed_impl); CeedChkBackend(ierr);
  CeedInt blk_size = ceed_impl->blk_size;
  CeedOperator_Opt *impl;

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedOperatorSetData(op, impl); CeedChkBackend(ierr);

  if (blk_size != 1 && blk_size != 8)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Opt backend cannot use blocksize: %d", blk_size);
  // LCOV_EXCL_STOP

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction",
                                CeedOperatorLinearAssembleQFunction_Opt);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleQFunctionUpdate",
                                CeedOperatorLinearAssembleQFunctionUpdate_Opt);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Opt); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Opt); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
