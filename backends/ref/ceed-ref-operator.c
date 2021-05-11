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
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Setup Input/Output Fields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Ref(CeedQFunction qf, CeedOperator op,
                                       bool inOrOut,
                                       CeedVector *full_evecs, CeedVector *e_vecs,
                                       CeedVector *q_vecs, CeedInt starte,
                                       CeedInt num_fields, CeedInt Q) {
  CeedInt dim, ierr, size, P;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedBasis basis;
  CeedElemRestriction elem_restr;
  CeedOperatorField *op_fields;
  CeedQFunctionField *qf_fields;
  if (inOrOut) {
    ierr = CeedOperatorGetFields(op, NULL, &op_fields); CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, NULL, &qf_fields); CeedChkBackend(ierr);
  } else {
    ierr = CeedOperatorGetFields(op, &op_fields, NULL); CeedChkBackend(ierr);
    ierr = CeedQFunctionGetFields(qf, &qf_fields, NULL); CeedChkBackend(ierr);
  }

  // Loop over fields
  for (CeedInt i=0; i<num_fields; i++) {
    CeedEvalMode eval_mode;
    ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode);
    CeedChkBackend(ierr);

    if (eval_mode != CEED_EVAL_WEIGHT) {
      ierr = CeedOperatorFieldGetElemRestriction(op_fields[i], &elem_restr);
      CeedChkBackend(ierr);
      ierr = CeedElemRestrictionCreateVector(elem_restr, NULL,
                                             &full_evecs[i+starte]);
      CeedChkBackend(ierr);
    }

    switch(eval_mode) {
    case CEED_EVAL_NONE:
      ierr = CeedQFunctionFieldGetSize(qf_fields[i], &size); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size, &q_vecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedQFunctionFieldGetSize(qf_fields[i], &size); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(elem_restr, &P);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, P*size, &e_vecs[i]); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size, &q_vecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedQFunctionFieldGetSize(qf_fields[i], &size); CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      ierr = CeedElemRestrictionGetElementSize(elem_restr, &P);
      CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, P*size/dim, &e_vecs[i]); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q*size, &q_vecs[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_WEIGHT: // Only on input fields
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedVectorCreate(ceed, Q, &q_vecs[i]); CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                            CEED_VECTOR_NONE, q_vecs[i]); CeedChkBackend(ierr);
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
//------------------------------------------------------------------------------/*
static int CeedOperatorSetup_Ref(CeedOperator op) {
  int ierr;
  bool setup_done;
  ierr = CeedOperatorIsSetupDone(op, &setup_done); CeedChkBackend(ierr);
  if (setup_done) return CEED_ERROR_SUCCESS;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Ref *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, num_input_fields, num_output_fields;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedQFunctionIsIdentity(qf, &impl->is_identity_qf); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields);
  CeedChkBackend(ierr);
  CeedOperatorField *op_input_fields, *op_output_fields;
  ierr = CeedOperatorGetFields(op, &op_input_fields, &op_output_fields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  ierr = CeedQFunctionGetFields(qf, &qf_input_fields, &qf_output_fields);
  CeedChkBackend(ierr);

  // Allocate
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

  // Set up infield and outfield e_vecs and q_vecs
  // Infields
  ierr = CeedOperatorSetupFields_Ref(qf, op, 0, impl->e_vecs,
                                     impl->e_vecs_in, impl->q_vecs_in, 0,
                                     num_input_fields, Q);
  CeedChkBackend(ierr);
  // Outfields
  ierr = CeedOperatorSetupFields_Ref(qf, op, 1, impl->e_vecs,
                                     impl->e_vecs_out, impl->q_vecs_out,
                                     num_input_fields, num_output_fields, Q);
  CeedChkBackend(ierr);

  // Identity QFunctions
  if (impl->is_identity_qf) {
    CeedEvalMode in_mode, out_mode;
    CeedQFunctionField *in_fields, *out_fields;
    ierr = CeedQFunctionGetFields(qf, &in_fields, &out_fields);
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
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Ref(CeedInt num_input_fields,
    CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
    CeedVector in_vec, const bool skip_active, CeedOperator_Ref *impl,
    CeedRequest *request) {
  CeedInt ierr;
  CeedEvalMode eval_mode;
  CeedVector vec;
  CeedElemRestriction elem_restr;
  uint64_t state;

  for (CeedInt i=0; i<num_input_fields; i++) {
    // Get input vector
    ierr = CeedOperatorFieldGetVector(op_input_fields[i], &vec);
    CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      if (skip_active)
        continue;
      else
        vec = in_vec;
    }

    ierr = CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    // Restrict and Evec
    if (eval_mode == CEED_EVAL_WEIGHT) { // Skip
    } else {
      // Restrict
      ierr = CeedVectorGetState(vec, &state); CeedChkBackend(ierr);
      // Skip restriction if input is unchanged
      if (state != impl->input_state[i] || vec == in_vec) {
        ierr = CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_restr);
        CeedChkBackend(ierr);
        ierr = CeedElemRestrictionApply(elem_restr, CEED_NOTRANSPOSE, vec,
                                        impl->e_vecs[i], request); CeedChkBackend(ierr);
        impl->input_state[i] = state;
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
static inline int CeedOperatorInputBasis_Ref(CeedInt e, CeedInt Q,
    CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
    CeedInt num_input_fields, const bool skip_active, CeedOperator_Ref *impl) {
  CeedInt ierr;
  CeedInt dim, elem_size, size;
  CeedElemRestriction elem_restr;
  CeedEvalMode eval_mode;
  CeedBasis basis;

  for (CeedInt i=0; i<num_input_fields; i++) {
    // Skip active input
    if (skip_active) {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(op_input_fields[i], &vec);
      CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }
    // Get elem_size, eval_mode, size
    ierr = CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_restr);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(elem_restr, &elem_size);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetSize(qf_input_fields[i], &size);
    CeedChkBackend(ierr);
    // Basis action
    switch(eval_mode) {
    case CEED_EVAL_NONE:
      ierr = CeedVectorSetArray(impl->q_vecs_in[i], CEED_MEM_HOST,
                                CEED_USE_POINTER, &impl->e_data[i][e*Q*size]);
      CeedChkBackend(ierr);
      break;
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(op_input_fields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST,
                                CEED_USE_POINTER, &impl->e_data[i][e*elem_size*size]);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP,
                            impl->e_vecs_in[i], impl->q_vecs_in[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(op_input_fields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->e_vecs_in[i], CEED_MEM_HOST,
                                CEED_USE_POINTER, &impl->e_data[i][e*elem_size*size/dim]);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE,
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
static inline int CeedOperatorOutputBasis_Ref(CeedInt e, CeedInt Q,
    CeedQFunctionField *qf_output_fields, CeedOperatorField *op_output_fields,
    CeedInt num_input_fields, CeedInt num_output_fields, CeedOperator op,
    CeedOperator_Ref *impl) {
  CeedInt ierr;
  CeedInt dim, elem_size, size;
  CeedElemRestriction elem_restr;
  CeedEvalMode eval_mode;
  CeedBasis basis;

  for (CeedInt i=0; i<num_output_fields; i++) {
    // Get elem_size, eval_mode, size
    ierr = CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_restr);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionGetElementSize(elem_restr, &elem_size);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode);
    CeedChkBackend(ierr);
    ierr = CeedQFunctionFieldGetSize(qf_output_fields[i], &size);
    CeedChkBackend(ierr);
    // Basis action
    switch(eval_mode) {
    case CEED_EVAL_NONE:
      break; // No action
    case CEED_EVAL_INTERP:
      ierr = CeedOperatorFieldGetBasis(op_output_fields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->e_vecs_out[i], CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                &impl->e_data[i + num_input_fields][e*elem_size*size]);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_TRANSPOSE,
                            CEED_EVAL_INTERP, impl->q_vecs_out[i],
                            impl->e_vecs_out[i]); CeedChkBackend(ierr);
      break;
    case CEED_EVAL_GRAD:
      ierr = CeedOperatorFieldGetBasis(op_output_fields[i], &basis);
      CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
      ierr = CeedVectorSetArray(impl->e_vecs_out[i], CEED_MEM_HOST,
                                CEED_USE_POINTER,
                                &impl->e_data[i + num_input_fields][e*elem_size*size/dim]);
      CeedChkBackend(ierr);
      ierr = CeedBasisApply(basis, 1, CEED_TRANSPOSE,
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
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorRestoreInputs_Ref(CeedInt num_input_fields,
    CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
    const bool skip_active, CeedOperator_Ref *impl) {
  CeedInt ierr;
  CeedEvalMode eval_mode;

  for (CeedInt i=0; i<num_input_fields; i++) {
    // Skip active inputs
    if (skip_active) {
      CeedVector vec;
      ierr = CeedOperatorFieldGetVector(op_input_fields[i], &vec);
      CeedChkBackend(ierr);
      if (vec == CEED_VECTOR_ACTIVE)
        continue;
    }
    // Restore input
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
static int CeedOperatorApplyAdd_Ref(CeedOperator op, CeedVector in_vec,
                                    CeedVector out_vec, CeedRequest *request) {
  int ierr;
  CeedOperator_Ref *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, num_elem, num_input_fields, num_output_fields, size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumElements(op, &num_elem); CeedChkBackend(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields);
  CeedChkBackend(ierr);
  CeedOperatorField *op_input_fields, *op_output_fields;
  ierr = CeedOperatorGetFields(op, &op_input_fields, &op_output_fields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  ierr = CeedQFunctionGetFields(qf, &qf_input_fields, &qf_output_fields);
  CeedChkBackend(ierr);
  CeedEvalMode eval_mode;
  CeedVector vec;
  CeedElemRestriction elem_restr;

  // Setup
  ierr = CeedOperatorSetup_Ref(op); CeedChkBackend(ierr);

  // Restriction only operator
  if (impl->is_identity_restr_op) {
    ierr = CeedOperatorFieldGetElemRestriction(op_input_fields[0], &elem_restr);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionApply(elem_restr, CEED_NOTRANSPOSE, in_vec,
                                    impl->e_vecs[0], request); CeedChkBackend(ierr);
    ierr = CeedOperatorFieldGetElemRestriction(op_output_fields[0], &elem_restr);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionApply(elem_restr, CEED_TRANSPOSE, impl->e_vecs[0],
                                    out_vec, request); CeedChkBackend(ierr);
    return CEED_ERROR_SUCCESS;
  }

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Ref(num_input_fields, qf_input_fields,
                                     op_input_fields, in_vec, false, impl,
                                     request); CeedChkBackend(ierr);

  // Output Evecs
  for (CeedInt i=0; i<num_output_fields; i++) {
    ierr = CeedVectorGetArray(impl->e_vecs[i+impl->num_e_vecs_in], CEED_MEM_HOST,
                              &impl->e_data[i + num_input_fields]); CeedChkBackend(ierr);
  }

  // Loop through elements
  for (CeedInt e=0; e<num_elem; e++) {
    // Output pointers
    for (CeedInt i=0; i<num_output_fields; i++) {
      ierr = CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode);
      CeedChkBackend(ierr);
      if (eval_mode == CEED_EVAL_NONE) {
        ierr = CeedQFunctionFieldGetSize(qf_output_fields[i], &size);
        CeedChkBackend(ierr);
        ierr = CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_HOST,
                                  CEED_USE_POINTER,
                                  &impl->e_data[i + num_input_fields][e*Q*size]);
        CeedChkBackend(ierr);
      }
    }

    // Input basis apply
    ierr = CeedOperatorInputBasis_Ref(e, Q, qf_input_fields, op_input_fields,
                                      num_input_fields, false, impl);
    CeedChkBackend(ierr);

    // Q function
    if (!impl->is_identity_qf) {
      ierr = CeedQFunctionApply(qf, Q, impl->q_vecs_in, impl->q_vecs_out);
      CeedChkBackend(ierr);
    }

    // Output basis apply
    ierr = CeedOperatorOutputBasis_Ref(e, Q, qf_output_fields, op_output_fields,
                                       num_input_fields, num_output_fields, op, impl);
    CeedChkBackend(ierr);
  }

  // Output restriction
  for (CeedInt i=0; i<num_output_fields; i++) {
    // Restore Evec
    ierr = CeedVectorRestoreArray(impl->e_vecs[i+impl->num_e_vecs_in],
                                  &impl->e_data[i + num_input_fields]);
    CeedChkBackend(ierr);
    // Get output vector
    ierr = CeedOperatorFieldGetVector(op_output_fields[i], &vec);
    CeedChkBackend(ierr);
    // Active
    if (vec == CEED_VECTOR_ACTIVE)
      vec = out_vec;
    // Restrict
    ierr = CeedOperatorFieldGetElemRestriction(op_output_fields[i], &elem_restr);
    CeedChkBackend(ierr);
    ierr = CeedElemRestrictionApply(elem_restr, CEED_TRANSPOSE,
                                    impl->e_vecs[i+impl->num_e_vecs_in], vec, request);
    CeedChkBackend(ierr);
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Ref(num_input_fields, qf_input_fields,
                                       op_input_fields, false, impl);
  CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Ref(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  int ierr;
  CeedOperator_Ref *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt Q, num_elem, num_input_fields, num_output_fields, size;
  ierr = CeedOperatorGetNumQuadraturePoints(op, &Q); CeedChkBackend(ierr);
  ierr = CeedOperatorGetNumElements(op, &num_elem); CeedChkBackend(ierr);
  ierr= CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields);
  CeedChkBackend(ierr);
  CeedOperatorField *op_input_fields, *op_output_fields;
  ierr = CeedOperatorGetFields(op, &op_input_fields, &op_output_fields);
  CeedChkBackend(ierr);
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  ierr = CeedQFunctionGetFields(qf, &qf_input_fields, &qf_output_fields);
  CeedChkBackend(ierr);
  CeedVector vec;
  CeedInt num_active_in = 0, num_active_out = 0;
  CeedVector *active_in = NULL;
  CeedScalar *a, *tmp;
  Ceed ceed, ceed_parent;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  ierr = CeedGetOperatorFallbackParentCeed(ceed, &ceed_parent);
  CeedChkBackend(ierr);
  ceed_parent = ceed_parent ? ceed_parent : ceed;

  // Setup
  ierr = CeedOperatorSetup_Ref(op); CeedChkBackend(ierr);

  // Check for identity
  if (impl->is_identity_qf)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "Assembling identity QFunctions not supported");
  // LCOV_EXCL_STOP

  // Input Evecs and Restriction
  ierr = CeedOperatorSetupInputs_Ref(num_input_fields, qf_input_fields,
                                     op_input_fields, NULL, true, impl, request);
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
        ierr = CeedVectorCreate(ceed, Q, &active_in[num_active_in+field]);
        CeedChkBackend(ierr);
        ierr = CeedVectorSetArray(active_in[num_active_in+field], CEED_MEM_HOST,
                                  CEED_USE_POINTER, &tmp[field*Q]);
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

  // Create output restriction
  CeedInt strides[3] = {1, Q, num_active_in*num_active_out*Q}; /* *NOPAD* */
  ierr = CeedElemRestrictionCreateStrided(ceed_parent, num_elem, Q,
                                          num_active_in*num_active_out,
                                          num_active_in*num_active_out*num_elem*Q,
                                          strides, rstr); CeedChkBackend(ierr);
  // Create assembled vector
  ierr = CeedVectorCreate(ceed_parent, num_elem*Q*num_active_in*num_active_out,
                          assembled); CeedChkBackend(ierr);
  ierr = CeedVectorSetValue(*assembled, 0.0); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(*assembled, CEED_MEM_HOST, &a); CeedChkBackend(ierr);

  // Loop through elements
  for (CeedInt e=0; e<num_elem; e++) {
    // Input basis apply
    ierr = CeedOperatorInputBasis_Ref(e, Q, qf_input_fields, op_input_fields,
                                      num_input_fields, true, impl);
    CeedChkBackend(ierr);

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
          a += size*Q; // Advance the pointer by the size of the output
        }
      }
      // Apply QFunction
      ierr = CeedQFunctionApply(qf, Q, impl->q_vecs_in, impl->q_vecs_out);
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
      CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_HOST, NULL);
      CeedChkBackend(ierr);
    }
  }

  // Restore input arrays
  ierr = CeedOperatorRestoreInputs_Ref(num_input_fields, qf_input_fields,
                                       op_input_fields, true, impl);
  CeedChkBackend(ierr);

  // Restore output
  ierr = CeedVectorRestoreArray(*assembled, &a); CeedChkBackend(ierr);

  // Cleanup
  for (CeedInt i=0; i<num_active_in; i++) {
    ierr = CeedVectorDestroy(&active_in[i]); CeedChkBackend(ierr);
  }
  ierr = CeedFree(&active_in); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get Basis Emode Pointer
//------------------------------------------------------------------------------
static inline void CeedOperatorGetBasisPointer_Ref(const CeedScalar **basis_ptr,
    CeedEvalMode eval_mode, const CeedScalar *identity, const CeedScalar *interp,
    const CeedScalar *grad) {
  switch (eval_mode) {
  case CEED_EVAL_NONE:
    *basis_ptr = identity;
    break;
  case CEED_EVAL_INTERP:
    *basis_ptr = interp;
    break;
  case CEED_EVAL_GRAD:
    *basis_ptr = grad;
    break;
  case CEED_EVAL_WEIGHT:
  case CEED_EVAL_DIV:
  case CEED_EVAL_CURL:
    break; // Caught by QF Assembly
  }
}

//------------------------------------------------------------------------------
// Create point block restriction
//------------------------------------------------------------------------------
static int CreatePBRestriction_Ref(CeedElemRestriction rstr,
                                   CeedElemRestriction *pb_rstr) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(rstr, &ceed); CeedChkBackend(ierr);
  const CeedInt *offsets;
  ierr = CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets);
  CeedChkBackend(ierr);

  // Expand offsets
  CeedInt num_elem, num_comp, elem_size, comp_stride, max = 1, *pbOffsets;
  ierr = CeedElemRestrictionGetNumElements(rstr, &num_elem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr, &num_comp);
  CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr, &elem_size);
  CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetCompStride(rstr, &comp_stride);
  CeedChkBackend(ierr);
  CeedInt shift = num_comp;
  if (comp_stride != 1)
    shift *= num_comp;
  ierr = CeedCalloc(num_elem*elem_size, &pbOffsets); CeedChkBackend(ierr);
  for (CeedInt i = 0; i < num_elem*elem_size; i++) {
    pbOffsets[i] = offsets[i]*shift;
    if (pbOffsets[i] > max)
      max = pbOffsets[i];
  }

  // Create new restriction
  ierr = CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp*num_comp,
                                   1,
                                   max + num_comp*num_comp, CEED_MEM_HOST,
                                   CEED_OWN_POINTER, pbOffsets, pb_rstr);
  CeedChkBackend(ierr);

  // Cleanup
  ierr = CeedElemRestrictionRestoreOffsets(rstr, &offsets); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleAddDiagonalCore_Ref(CeedOperator op,
    CeedVector assembled, CeedRequest *request, const bool is_pointblock) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);

  // Assemble QFunction
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);
  CeedInt num_input_fields, num_output_fields;
  ierr= CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields);
  CeedChkBackend(ierr);
  CeedVector assembled_qf;
  CeedElemRestriction rstr;
  ierr = CeedOperatorLinearAssembleQFunction(op,  &assembled_qf, &rstr, request);
  CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr); CeedChkBackend(ierr);
  CeedScalar max_norm = 0;
  ierr = CeedVectorNorm(assembled_qf, CEED_NORM_MAX, &max_norm);
  CeedChkBackend(ierr);

  // Determine active input basis
  CeedOperatorField *op_fields;
  CeedQFunctionField *qf_fields;
  ierr = CeedOperatorGetFields(op, &op_fields, NULL); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetFields(qf, &qf_fields, NULL); CeedChkBackend(ierr);
  CeedInt num_eval_mode_in = 0, num_comp, dim = 1;
  CeedEvalMode *eval_mode_in = NULL;
  CeedBasis basis_in = NULL;
  CeedElemRestriction rstr_in = NULL;
  for (CeedInt i=0; i<num_input_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(op_fields[i], &vec); CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis_in); CeedChkBackend(ierr);
      ierr = CeedBasisGetNumComponents(basis_in, &num_comp); CeedChkBackend(ierr);
      ierr = CeedBasisGetDimension(basis_in, &dim); CeedChkBackend(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr);
      CeedChkBackend(ierr);
      if (rstr_in && rstr_in != rstr)
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "Multi-field non-composite operator diagonal assembly not supported");
      // LCOV_EXCL_STOP
      rstr_in = rstr;
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode);
      CeedChkBackend(ierr);
      switch (eval_mode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(num_eval_mode_in + 1, &eval_mode_in); CeedChkBackend(ierr);
        eval_mode_in[num_eval_mode_in] = eval_mode;
        num_eval_mode_in += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(num_eval_mode_in + dim, &eval_mode_in); CeedChkBackend(ierr);
        for (CeedInt d=0; d<dim; d++)
          eval_mode_in[num_eval_mode_in+d] = eval_mode;
        num_eval_mode_in += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }
  }

  // Determine active output basis
  ierr = CeedOperatorGetFields(op, NULL, &op_fields); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_fields); CeedChkBackend(ierr);
  CeedInt num_eval_mode_out = 0;
  CeedEvalMode *eval_mode_out = NULL;
  CeedBasis basis_out = NULL;
  CeedElemRestriction rstr_out = NULL;
  for (CeedInt i=0; i<num_output_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(op_fields[i], &vec); CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis_out);
      CeedChkBackend(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr);
      CeedChkBackend(ierr);
      if (rstr_out && rstr_out != rstr)
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "Multi-field non-composite operator diagonal assembly not supported");
      // LCOV_EXCL_STOP
      rstr_out = rstr;
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode);
      CeedChkBackend(ierr);
      switch (eval_mode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(num_eval_mode_out + 1, &eval_mode_out); CeedChkBackend(ierr);
        eval_mode_out[num_eval_mode_out] = eval_mode;
        num_eval_mode_out += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(num_eval_mode_out + dim, &eval_mode_out);
        CeedChkBackend(ierr);
        for (CeedInt d=0; d<dim; d++)
          eval_mode_out[num_eval_mode_out+d] = eval_mode;
        num_eval_mode_out += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }
  }

  // Assemble point-block diagonal restriction, if needed
  CeedElemRestriction diag_rstr = rstr_out;
  if (is_pointblock) {
    ierr = CreatePBRestriction_Ref(rstr_out, &diag_rstr); CeedChkBackend(ierr);
  }

  // Create diagonal vector
  CeedVector elem_diag;
  ierr = CeedElemRestrictionCreateVector(diag_rstr, NULL, &elem_diag);
  CeedChkBackend(ierr);

  // Assemble element operator diagonals
  CeedScalar *elem_diag_array, *assembled_qf_array;
  ierr = CeedVectorSetValue(elem_diag, 0.0); CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(elem_diag, CEED_MEM_HOST, &elem_diag_array);
  CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(assembled_qf, CEED_MEM_HOST, &assembled_qf_array);
  CeedChkBackend(ierr);
  CeedInt num_elem, num_nodes, num_qpts;
  ierr = CeedElemRestrictionGetNumElements(diag_rstr, &num_elem);
  CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis_in, &num_nodes); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts);
  CeedChkBackend(ierr);
  // Basis matrices
  const CeedScalar *interp_in, *interp_out, *grad_in, *grad_out;
  CeedScalar *identity = NULL;
  bool evalNone = false;
  for (CeedInt i=0; i<num_eval_mode_in; i++)
    evalNone = evalNone || (eval_mode_in[i] == CEED_EVAL_NONE);
  for (CeedInt i=0; i<num_eval_mode_out; i++)
    evalNone = evalNone || (eval_mode_out[i] == CEED_EVAL_NONE);
  if (evalNone) {
    ierr = CeedCalloc(num_qpts*num_nodes, &identity); CeedChkBackend(ierr);
    for (CeedInt i=0; i<(num_nodes<num_qpts?num_nodes:num_qpts); i++)
      identity[i*num_nodes+i] = 1.0;
  }
  ierr = CeedBasisGetInterp(basis_in, &interp_in); CeedChkBackend(ierr);
  ierr = CeedBasisGetInterp(basis_out, &interp_out); CeedChkBackend(ierr);
  ierr = CeedBasisGetGrad(basis_in, &grad_in); CeedChkBackend(ierr);
  ierr = CeedBasisGetGrad(basis_out, &grad_out); CeedChkBackend(ierr);
  // Compute the diagonal of B^T D B
  // Each element
  const CeedScalar qf_value_bound = max_norm*(100*CEED_EPSILON);
  for (CeedInt e=0; e<num_elem; e++) {
    CeedInt d_out = -1;
    // Each basis eval mode pair
    for (CeedInt e_out=0; e_out<num_eval_mode_out; e_out++) {
      const CeedScalar *bt = NULL;
      if (eval_mode_out[e_out] == CEED_EVAL_GRAD)
        d_out += 1;
      CeedOperatorGetBasisPointer_Ref(&bt, eval_mode_out[e_out], identity, interp_out,
                                      &grad_out[d_out*num_qpts*num_nodes]);
      CeedInt d_in = -1;
      for (CeedInt e_in=0; e_in<num_eval_mode_in; e_in++) {
        const CeedScalar *b = NULL;
        if (eval_mode_in[e_in] == CEED_EVAL_GRAD)
          d_in += 1;
        CeedOperatorGetBasisPointer_Ref(&b, eval_mode_in[e_in], identity, interp_in,
                                        &grad_in[d_in*num_qpts*num_nodes]);
        // Each component
        for (CeedInt c_out=0; c_out<num_comp; c_out++)
          // Each qpoint/node pair
          for (CeedInt q=0; q<num_qpts; q++)
            if (is_pointblock) {
              // Point Block Diagonal
              for (CeedInt c_in=0; c_in<num_comp; c_in++) {
                const CeedScalar qf_value =
                  assembled_qf_array[((((e*num_eval_mode_in+e_in)*num_comp+c_in)*
                                       num_eval_mode_out+e_out)*num_comp+c_out)*num_qpts+q];
                if (fabs(qf_value) > qf_value_bound)
                  for (CeedInt n=0; n<num_nodes; n++)
                    elem_diag_array[((e*num_comp+c_out)*num_comp+c_in)*num_nodes+n] +=
                      bt[q*num_nodes+n] * qf_value * b[q*num_nodes+n];
              }
            } else {
              // Diagonal Only
              const CeedScalar qf_value =
                assembled_qf_array[((((e*num_eval_mode_in+e_in)*num_comp+c_out)*
                                     num_eval_mode_out+e_out)*num_comp+c_out)*num_qpts+q];
              if (fabs(qf_value) > qf_value_bound)
                for (CeedInt n=0; n<num_nodes; n++)
                  elem_diag_array[(e*num_comp+c_out)*num_nodes+n] +=
                    bt[q*num_nodes+n] * qf_value * b[q*num_nodes+n];
            }
      }
    }
  }
  ierr = CeedVectorRestoreArray(elem_diag, &elem_diag_array);
  CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArray(assembled_qf, &assembled_qf_array);
  CeedChkBackend(ierr);

  // Assemble local operator diagonal
  ierr = CeedElemRestrictionApply(diag_rstr, CEED_TRANSPOSE, elem_diag,
                                  assembled, request); CeedChkBackend(ierr);

  // Cleanup
  if (is_pointblock) {
    ierr = CeedElemRestrictionDestroy(&diag_rstr); CeedChkBackend(ierr);
  }
  ierr = CeedVectorDestroy(&assembled_qf); CeedChkBackend(ierr);
  ierr = CeedVectorDestroy(&elem_diag); CeedChkBackend(ierr);
  ierr = CeedFree(&eval_mode_in); CeedChkBackend(ierr);
  ierr = CeedFree(&eval_mode_out); CeedChkBackend(ierr);
  ierr = CeedFree(&identity); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble composite diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleAddDiagonalCompositeCore_Ref(
  CeedOperator op, CeedVector assembled, CeedRequest *request,
  const bool is_pointblock) {
  int ierr;
  CeedInt num_sub;
  CeedOperator *suboperators;
  ierr = CeedOperatorGetNumSub(op, &num_sub); CeedChkBackend(ierr);
  ierr = CeedOperatorGetSubList(op, &suboperators); CeedChkBackend(ierr);
  for (CeedInt i = 0; i < num_sub; i++) {
    ierr = CeedOperatorAssembleAddDiagonalCore_Ref(suboperators[i], assembled,
           request, is_pointblock); CeedChkBackend(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Ref(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  bool is_composite;
  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChkBackend(ierr);
  if (is_composite) {
    return CeedOperatorLinearAssembleAddDiagonalCompositeCore_Ref(op, assembled,
           request, false);
  } else {
    return CeedOperatorAssembleAddDiagonalCore_Ref(op, assembled, request, false);
  }
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Ref(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  bool is_composite;
  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChkBackend(ierr);
  if (is_composite) {
    return CeedOperatorLinearAssembleAddDiagonalCompositeCore_Ref(op, assembled,
           request, true);
  } else {
    return CeedOperatorAssembleAddDiagonalCore_Ref(op, assembled, request, true);
  }
}

//------------------------------------------------------------------------------
// Build Mass, Laplacian matrices
//------------------------------------------------------------------------------
static int CeedBuildMassLaplace(const CeedScalar *interp_1d,
                                const CeedScalar *grad_1d,
                                const CeedScalar *q_weight_1d, CeedInt P_1d,
                                CeedInt Q_1d, CeedInt dim,
                                CeedScalar *mass, CeedScalar *laplace) {

  for (CeedInt i=0; i<P_1d; i++)
    for (CeedInt j=0; j<P_1d; j++) {
      CeedScalar sum = 0.0;
      for (CeedInt k=0; k<Q_1d; k++)
        sum += interp_1d[k*P_1d+i]*q_weight_1d[k]*interp_1d[k*P_1d+j];
      mass[i+j*P_1d] = sum;
    }
  // -- Laplacian
  for (CeedInt i=0; i<P_1d; i++)
    for (CeedInt j=0; j<P_1d; j++) {
      CeedScalar sum = 0.0;
      for (CeedInt k=0; k<Q_1d; k++)
        sum += grad_1d[k*P_1d+i]*q_weight_1d[k]*grad_1d[k*P_1d+j];
      laplace[i+j*P_1d] = sum;
    }
  CeedScalar perturbation = dim>2 ? 1e-6 : 1e-4;
  for (CeedInt i=0; i<P_1d; i++)
    laplace[i+P_1d*i] += perturbation;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create FDM Element Inverse
//------------------------------------------------------------------------------
int CeedOperatorCreateFDMElementInverse_Ref(CeedOperator op,
    CeedOperator *fdm_inv, CeedRequest *request) {
  int ierr;
  Ceed ceed, ceed_parent;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  ierr = CeedGetOperatorFallbackParentCeed(ceed, &ceed_parent);
  CeedChkBackend(ierr);
  ceed_parent = ceed_parent ? ceed_parent : ceed;
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChkBackend(ierr);

  // Determine active input basis
  bool interp = false, grad = false;
  CeedBasis basis = NULL;
  CeedElemRestriction rstr = NULL;
  CeedOperatorField *op_fields;
  CeedQFunctionField *qf_fields;
  ierr = CeedOperatorGetFields(op, &op_fields, NULL); CeedChkBackend(ierr);
  ierr = CeedQFunctionGetFields(qf, &qf_fields, NULL); CeedChkBackend(ierr);
  CeedInt num_input_fields;
  ierr = CeedQFunctionGetNumArgs(qf, &num_input_fields, NULL);
  CeedChkBackend(ierr);
  for (CeedInt i=0; i<num_input_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(op_fields[i], &vec); CeedChkBackend(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode);
      CeedChkBackend(ierr);
      interp = interp || eval_mode == CEED_EVAL_INTERP;
      grad = grad || eval_mode == CEED_EVAL_GRAD;
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis); CeedChkBackend(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr);
      CeedChkBackend(ierr);
    }
  }
  if (!basis)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No active field set");
  // LCOV_EXCL_STOP
  CeedInt P_1d, Q_1d, elem_size, num_qpts, dim, num_comp = 1, num_elem = 1,
                                                l_size = 1;
  ierr = CeedBasisGetNumNodes1D(basis, &P_1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumNodes(basis, &elem_size); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &num_qpts); CeedChkBackend(ierr);
  ierr = CeedBasisGetDimension(basis, &dim); CeedChkBackend(ierr);
  ierr = CeedBasisGetNumComponents(basis, &num_comp); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetNumElements(rstr, &num_elem); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionGetLVectorSize(rstr, &l_size); CeedChkBackend(ierr);

  // Build and diagonalize 1D Mass and Laplacian
  bool tensor_basis;
  ierr = CeedBasisIsTensor(basis, &tensor_basis); CeedChkBackend(ierr);
  if (!tensor_basis)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "FDMElementInverse only supported for tensor "
                     "bases");
  // LCOV_EXCL_STOP
  CeedScalar *mass, *laplace, *x, *fdm_interp, *lambda;
  ierr = CeedCalloc(P_1d*P_1d, &mass); CeedChkBackend(ierr);
  ierr = CeedCalloc(P_1d*P_1d, &laplace); CeedChkBackend(ierr);
  ierr = CeedCalloc(P_1d*P_1d, &x); CeedChkBackend(ierr);
  ierr = CeedCalloc(P_1d*P_1d, &fdm_interp); CeedChkBackend(ierr);
  ierr = CeedCalloc(P_1d, &lambda); CeedChkBackend(ierr);
  // -- Build matrices
  const CeedScalar *interp_1d, *grad_1d, *q_weight_1d;
  ierr = CeedBasisGetInterp1D(basis, &interp_1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetGrad1D(basis, &grad_1d); CeedChkBackend(ierr);
  ierr = CeedBasisGetQWeights(basis, &q_weight_1d); CeedChkBackend(ierr);
  ierr = CeedBuildMassLaplace(interp_1d, grad_1d, q_weight_1d, P_1d, Q_1d, dim,
                              mass, laplace); CeedChkBackend(ierr);
  // -- Diagonalize
  ierr = CeedSimultaneousDiagonalization(ceed, laplace, mass, x, lambda, P_1d);
  CeedChkBackend(ierr);
  ierr = CeedFree(&mass); CeedChkBackend(ierr);
  ierr = CeedFree(&laplace); CeedChkBackend(ierr);
  for (CeedInt i=0; i<P_1d; i++)
    for (CeedInt j=0; j<P_1d; j++)
      fdm_interp[i+j*P_1d] = x[j+i*P_1d];
  ierr = CeedFree(&x); CeedChkBackend(ierr);

  // Assemble QFunction
  CeedVector assembled;
  CeedElemRestriction rstr_qf;
  ierr =  CeedOperatorLinearAssembleQFunction(op, &assembled, &rstr_qf,
          request); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_qf); CeedChkBackend(ierr);
  CeedScalar max_norm = 0;
  ierr = CeedVectorNorm(assembled, CEED_NORM_MAX, &max_norm);
  CeedChkBackend(ierr);

  // Calculate element averages
  CeedInt num_modes = (interp?1:0) + (grad?dim:0);
  CeedScalar *elem_avg;
  const CeedScalar *assembled_array, *q_weight_array;
  CeedVector q_weight;
  ierr = CeedVectorCreate(ceed_parent, num_qpts, &q_weight); CeedChkBackend(ierr);
  ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                        CEED_VECTOR_NONE, q_weight); CeedChkBackend(ierr);
  ierr = CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
  CeedChkBackend(ierr);
  ierr = CeedVectorGetArrayRead(q_weight, CEED_MEM_HOST, &q_weight_array);
  CeedChkBackend(ierr);
  ierr = CeedCalloc(num_elem, &elem_avg); CeedChkBackend(ierr);
  for (CeedInt e=0; e<num_elem; e++) {
    CeedInt count = 0;
    CeedInt elem_offset = e*num_qpts*num_comp*num_comp*num_modes*num_modes;
    for (CeedInt q=0; q<num_qpts; q++)
      for (CeedInt i=0; i<num_comp*num_comp*num_modes*num_modes; i++)
        if (fabs(assembled_array[elem_offset + i*num_qpts + q]) > max_norm*
            (100*CEED_EPSILON)) {
          elem_avg[e] += assembled_array[elem_offset + i*num_qpts + q] /
                         q_weight_array[q];
          count++;
        }
    if (count) {
      elem_avg[e] /= count;
    } else {
      elem_avg[e] = 1.0;
    }
  }
  ierr = CeedVectorRestoreArrayRead(assembled, &assembled_array);
  CeedChkBackend(ierr);
  ierr = CeedVectorDestroy(&assembled); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArrayRead(q_weight, &q_weight_array);
  CeedChkBackend(ierr);
  ierr = CeedVectorDestroy(&q_weight); CeedChkBackend(ierr);

  // Build FDM diagonal
  CeedVector q_data;
  CeedScalar *q_data_array, *fdm_diagonal;
  ierr = CeedCalloc(num_comp*elem_size, &fdm_diagonal); CeedChkBackend(ierr);
  for (CeedInt c=0; c<num_comp; c++)
    for (CeedInt n=0; n<elem_size; n++) {
      if (interp)
        fdm_diagonal[c*elem_size + n] = 1.0;
      if (grad)
        for (CeedInt d=0; d<dim; d++) {
          CeedInt i = (n / CeedIntPow(P_1d, d)) % P_1d;
          fdm_diagonal[c*elem_size + n] += lambda[i];
        }
      if (fabs(fdm_diagonal[c*elem_size + n]) < 1000*CEED_EPSILON)
        fdm_diagonal[c*elem_size + n] = 1000*CEED_EPSILON;
    }
  ierr = CeedVectorCreate(ceed_parent, num_elem*num_comp*elem_size, &q_data);
  CeedChkBackend(ierr);
  ierr = CeedVectorSetValue(q_data, 0.0);
  CeedChkBackend(ierr);
  ierr = CeedVectorGetArray(q_data, CEED_MEM_HOST, &q_data_array);
  CeedChkBackend(ierr);
  for (CeedInt e=0; e<num_elem; e++)
    for (CeedInt c=0; c<num_comp; c++)
      for (CeedInt n=0; n<elem_size; n++)
        q_data_array[(e*num_comp+c)*elem_size+n] = 1. / (elem_avg[e] *
            fdm_diagonal[c*elem_size + n]);
  ierr = CeedFree(&elem_avg); CeedChkBackend(ierr);
  ierr = CeedFree(&fdm_diagonal); CeedChkBackend(ierr);
  ierr = CeedVectorRestoreArray(q_data, &q_data_array); CeedChkBackend(ierr);

  // Setup FDM operator
  // -- Basis
  CeedBasis fdm_basis;
  CeedScalar *grad_dummy, *q_ref_dummy, *q_weight_dummy;
  ierr = CeedCalloc(P_1d*P_1d, &grad_dummy); CeedChkBackend(ierr);
  ierr = CeedCalloc(P_1d, &q_ref_dummy); CeedChkBackend(ierr);
  ierr = CeedCalloc(P_1d, &q_weight_dummy); CeedChkBackend(ierr);
  ierr = CeedBasisCreateTensorH1(ceed_parent, dim, num_comp, P_1d, P_1d,
                                 fdm_interp, grad_dummy, q_ref_dummy,
                                 q_weight_dummy, &fdm_basis); CeedChkBackend(ierr);
  ierr = CeedFree(&fdm_interp); CeedChkBackend(ierr);
  ierr = CeedFree(&grad_dummy); CeedChkBackend(ierr);
  ierr = CeedFree(&q_ref_dummy); CeedChkBackend(ierr);
  ierr = CeedFree(&q_weight_dummy); CeedChkBackend(ierr);
  ierr = CeedFree(&lambda); CeedChkBackend(ierr);

  // -- Restriction
  CeedElemRestriction rstr_qd_i;
  CeedInt strides[3] = {1, elem_size, elem_size*num_comp};
  ierr = CeedElemRestrictionCreateStrided(ceed_parent, num_elem, elem_size,
                                          num_comp, num_elem*num_comp*elem_size,
                                          strides, &rstr_qd_i);
  CeedChkBackend(ierr);
  // -- QFunction
  CeedQFunction qf_fdm;
  ierr = CeedQFunctionCreateInteriorByName(ceed_parent, "Scale", &qf_fdm);
  CeedChkBackend(ierr);
  ierr = CeedQFunctionAddInput(qf_fdm, "input", num_comp, CEED_EVAL_INTERP);
  CeedChkBackend(ierr);
  ierr = CeedQFunctionAddInput(qf_fdm, "scale", num_comp, CEED_EVAL_NONE);
  CeedChkBackend(ierr);
  ierr = CeedQFunctionAddOutput(qf_fdm, "output", num_comp, CEED_EVAL_INTERP);
  CeedChkBackend(ierr);
  // -- QFunction context
  CeedInt *num_comp_data;
  ierr = CeedCalloc(1, &num_comp_data); CeedChk(ierr);
  num_comp_data[0] = num_comp;
  CeedQFunctionContext ctx_fdm;
  ierr = CeedQFunctionContextCreate(ceed, &ctx_fdm); CeedChkBackend(ierr);
  ierr = CeedQFunctionContextSetData(ctx_fdm, CEED_MEM_HOST, CEED_OWN_POINTER,
                                     sizeof(*num_comp_data), num_comp_data);
  CeedChkBackend(ierr);
  ierr = CeedQFunctionSetContext(qf_fdm, ctx_fdm); CeedChkBackend(ierr);
  ierr = CeedQFunctionContextDestroy(&ctx_fdm); CeedChkBackend(ierr);
  // -- Operator
  ierr = CeedOperatorCreate(ceed_parent, qf_fdm, NULL, NULL, fdm_inv);
  CeedChkBackend(ierr);
  CeedOperatorSetField(*fdm_inv, "input", rstr, fdm_basis, CEED_VECTOR_ACTIVE);
  CeedChkBackend(ierr);
  CeedOperatorSetField(*fdm_inv, "scale", rstr_qd_i, CEED_BASIS_COLLOCATED,
                       q_data); CeedChkBackend(ierr);
  CeedOperatorSetField(*fdm_inv, "output", rstr, fdm_basis, CEED_VECTOR_ACTIVE);
  CeedChkBackend(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&q_data); CeedChkBackend(ierr);
  ierr = CeedBasisDestroy(&fdm_basis); CeedChkBackend(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_qd_i); CeedChkBackend(ierr);
  ierr = CeedQFunctionDestroy(&qf_fdm); CeedChkBackend(ierr);

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Operator Destroy
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Ref(CeedOperator op) {
  int ierr;
  CeedOperator_Ref *impl;
  ierr = CeedOperatorGetData(op, &impl); CeedChkBackend(ierr);

  for (CeedInt i=0; i<impl->num_e_vecs_in+impl->num_e_vecs_out; i++) {
    ierr = CeedVectorDestroy(&impl->e_vecs[i]); CeedChkBackend(ierr);
  }
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
int CeedOperatorCreate_Ref(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  CeedOperator_Ref *impl;

  ierr = CeedCalloc(1, &impl); CeedChkBackend(ierr);
  ierr = CeedOperatorSetData(op, impl); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction",
                                CeedOperatorLinearAssembleQFunction_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal",
                                CeedOperatorLinearAssembleAddDiagonal_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleAddPointBlockDiagonal",
                                CeedOperatorLinearAssembleAddPointBlockDiagonal_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "CreateFDMElementInverse",
                                CeedOperatorCreateFDMElementInverse_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd",
                                CeedOperatorApplyAdd_Ref); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "Destroy",
                                CeedOperatorDestroy_Ref); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Composite Operator Create
//------------------------------------------------------------------------------
int CeedCompositeOperatorCreate_Ref(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal",
                                CeedOperatorLinearAssembleAddDiagonal_Ref);
  CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "Operator", op,
                                "LinearAssembleAddPointBlockDiagonal",
                                CeedOperatorLinearAssembleAddPointBlockDiagonal_Ref);
  CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
