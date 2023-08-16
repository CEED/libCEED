// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <cassert>
#include <string>
#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

class CeedOperatorSyclLinearDiagonal;
class CeedOperatorSyclLinearAssemble;
class CeedOperatorSyclLinearAssembleFallback;

//------------------------------------------------------------------------------
//  Get Basis Emode Pointer
//------------------------------------------------------------------------------
void CeedOperatorGetBasisPointer_Sycl(const CeedScalar **basis_ptr, CeedEvalMode e_mode, const CeedScalar *identity, const CeedScalar *interp,
                                      const CeedScalar *grad) {
  switch (e_mode) {
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
      break;  // Caught by QF Assembly
  }
}

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Sycl(CeedOperator op) {
  Ceed               ceed;
  Ceed_Sycl         *sycl_data;
  CeedOperator_Sycl *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  // Apply data
  for (CeedInt i = 0; i < impl->num_e_in + impl->num_e_out; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->e_vecs[i]));
  }
  CeedCallBackend(CeedFree(&impl->e_vecs));

  for (CeedInt i = 0; i < impl->num_e_in; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->q_vecs_in[i]));
  }
  CeedCallBackend(CeedFree(&impl->q_vecs_in));

  for (CeedInt i = 0; i < impl->num_e_out; i++) {
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
    CeedCallBackend(CeedFree(&impl->diag->h_e_mode_in));
    CeedCallBackend(CeedFree(&impl->diag->h_e_mode_out));

    CeedCallSycl(ceed, sycl_data->sycl_queue.wait_and_throw());
    CeedCallSycl(ceed, sycl::free(impl->diag->d_e_mode_in, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_e_mode_out, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_identity, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_interp_in, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_interp_out, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_grad_in, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->diag->d_grad_out, sycl_data->sycl_context));
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->diag->point_block_diag_rstr));

    CeedCallBackend(CeedVectorDestroy(&impl->diag->elem_diag));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->point_block_elem_diag));
  }
  CeedCallBackend(CeedFree(&impl->diag));

  if (impl->asmb) {
    CeedCallSycl(ceed, sycl_data->sycl_queue.wait_and_throw());
    CeedCallSycl(ceed, sycl::free(impl->asmb->d_B_in, sycl_data->sycl_context));
    CeedCallSycl(ceed, sycl::free(impl->asmb->d_B_out, sycl_data->sycl_context));
  }
  CeedCallBackend(CeedFree(&impl->asmb));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Sycl(CeedQFunction qf, CeedOperator op, bool is_input, CeedVector *e_vecs, CeedVector *q_vecs, CeedInt start_e,
                                        CeedInt num_fields, CeedInt Q, CeedInt num_elem) {
  Ceed                ceed;
  CeedSize            q_size;
  bool                is_strided, skip_restriction;
  CeedInt             dim, size;
  CeedOperatorField  *op_fields;
  CeedQFunctionField *qf_fields;

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
    CeedEvalMode        e_mode;
    CeedVector          vec;
    CeedElemRestriction rstr;
    CeedBasis           basis;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &e_mode));

    is_strided       = false;
    skip_restriction = false;
    if (e_mode != CEED_EVAL_WEIGHT) {
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr));

      // Check whether this field can skip the element restriction:
      // must be passive input, with  e_mode NONE, and have a strided restriction with CEED_STRIDES_BACKEND.

      // First, check whether the field is input or output:
      if (is_input) {
        // Check for passive input:
        CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &vec));
        if (vec != CEED_VECTOR_ACTIVE) {
          // Check  e_mode
          if (e_mode == CEED_EVAL_NONE) {
            // Check for  is_strided restriction
            CeedCallBackend(CeedElemRestrictionIsStrided(rstr, &is_strided));
            if (is_strided) {
              // Check if vector is already in preferred backend ordering
              CeedCallBackend(CeedElemRestrictionHasBackendStrides(rstr, &skip_restriction));
            }
          }
        }
      }
      if (skip_restriction) {
        // We do not need an E-Vector, but will use the input field vector's data directly in the operator application
        e_vecs[i + start_e] = NULL;
      } else {
        CeedCallBackend(CeedElemRestrictionCreateVector(rstr, NULL, &e_vecs[i + start_e]));
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
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, NULL, q_vecs[i]));
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
// CeedOperator needs to connect all the named fields (be they active or
// passive) to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Sycl(CeedOperator op) {
  Ceed                ceed;
  bool                is_setup_done;
  CeedInt             Q, num_elem, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Sycl  *impl;

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

  impl->num_e_in  = num_input_fields;
  impl->num_e_out = num_output_fields;

  // Set up infield and outfield  e_vecs and  q_vecs
  // Infields
  CeedCallBackend(CeedOperatorSetupFields_Sycl(qf, op, true, impl->e_vecs, impl->q_vecs_in, 0, num_input_fields, Q, num_elem));
  // Outfields
  CeedCallBackend(CeedOperatorSetupFields_Sycl(qf, op, false, impl->e_vecs, impl->q_vecs_out, num_input_fields, num_output_fields, Q, num_elem));

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorSetupInputs_Sycl(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                               CeedVector in_vec, const bool skip_active, CeedScalar *e_data[2 * CEED_FIELD_MAX],
                                               CeedOperator_Sycl *impl, CeedRequest *request) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode        e_mode;
    CeedVector          vec;
    CeedElemRestriction rstr;

    // Get input vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      if (skip_active) continue;
      else vec = in_vec;
    }

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &e_mode));
    if (e_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      // Get input vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      // Get input element restriction
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &rstr));
      if (vec == CEED_VECTOR_ACTIVE) vec = in_vec;
      // Restrict, if necessary
      if (!impl->e_vecs[i]) {
        // No restriction for this field; read data directly from vec.
        CeedCallBackend(CeedVectorGetArrayRead(vec, CEED_MEM_DEVICE, (const CeedScalar **)&e_data[i]));
      } else {
        CeedCallBackend(CeedElemRestrictionApply(rstr, CEED_NOTRANSPOSE, vec, impl->e_vecs[i], request));
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
static inline int CeedOperatorInputBasis_Sycl(CeedInt num_elem, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                              CeedInt num_input_fields, const bool skip_active, CeedScalar *e_data[2 * CEED_FIELD_MAX],
                                              CeedOperator_Sycl *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedInt             elem_size, size;
    CeedElemRestriction rstr;
    CeedEvalMode        e_mode;
    CeedBasis           basis;

    // Skip active input
    if (skip_active) {
      CeedVector vec;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    // Get elem_size,  e_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
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
static inline int CeedOperatorRestoreInputs_Sycl(CeedInt num_input_fields, CeedQFunctionField *qf_input_fields, CeedOperatorField *op_input_fields,
                                                 const bool skip_active, CeedScalar *e_data[2 * CEED_FIELD_MAX], CeedOperator_Sycl *impl) {
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode e_mode;
    CeedVector   vec;

    // Skip active input
    if (skip_active) {
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) continue;
    }
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &e_mode));
    if (e_mode == CEED_EVAL_WEIGHT) {  // Skip
    } else {
      if (!impl->e_vecs[i]) {  // This was a  skip_restriction case
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
static int CeedOperatorApplyAdd_Sycl(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedInt             Q, num_elem, elem_size, num_input_fields, num_output_fields, size;
  CeedEvalMode        e_mode;
  CeedScalar         *e_data[2 * CEED_FIELD_MAX] = {0};
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Sycl  *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetup_Sycl(op));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Sycl(num_input_fields, qf_input_fields, op_input_fields, in_vec, false, e_data, impl, request));

  // Input basis apply if needed
  CeedCallBackend(CeedOperatorInputBasis_Sycl(num_elem, qf_input_fields, op_input_fields, num_input_fields, false, e_data, impl));

  // Output pointers, as necessary
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &e_mode));
    if (e_mode == CEED_EVAL_NONE) {
      // Set the output Q-Vector to use the E-Vector data directly
      CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs[i + impl->num_e_in], CEED_MEM_DEVICE, &e_data[i + num_input_fields]));
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_DEVICE, CEED_USE_POINTER, e_data[i + num_input_fields]));
    }
  }

  // Q function
  CeedCallBackend(CeedQFunctionApply(qf, num_elem * Q, impl->q_vecs_in, impl->q_vecs_out));

  // Output basis apply if needed
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedElemRestriction rstr;
    CeedBasis           basis;

    // Get elem_size,  e_mode, size
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &rstr));
    CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &e_mode));
    CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[i], &size));
    // Basis action
    switch (e_mode) {
      case CEED_EVAL_NONE:
        break;
      case CEED_EVAL_INTERP:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_TRANSPOSE, CEED_EVAL_INTERP, impl->q_vecs_out[i], impl->e_vecs[i + impl->num_e_in]));
        break;
      case CEED_EVAL_GRAD:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_TRANSPOSE, CEED_EVAL_GRAD, impl->q_vecs_out[i], impl->e_vecs[i + impl->num_e_in]));
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT:
        Ceed ceed;
        CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        break;  // Should not occur
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
    CeedElemRestriction rstr;

    // Restore evec
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &e_mode));
    if (e_mode == CEED_EVAL_NONE) {
      CeedCallBackend(CeedVectorRestoreArray(impl->e_vecs[i + impl->num_e_in], &e_data[i + num_input_fields]));
    }
    // Get output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
    // Restrict
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &rstr));
    // Active
    if (vec == CEED_VECTOR_ACTIVE) vec = out_vec;

    CeedCallBackend(CeedElemRestrictionApply(rstr, CEED_TRANSPOSE, impl->e_vecs[i + impl->num_e_in], vec, request));
  }

  // Restore input arrays
  CeedCallBackend(CeedOperatorRestoreInputs_Sycl(num_input_fields, qf_input_fields, op_input_fields, false, e_data, impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Core code for assembling linear QFunction
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Sycl(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                               CeedRequest *request) {
  Ceed                ceed, ceed_parent;
  CeedSize            q_size;
  CeedInt             num_active_in, num_active_out, Q, num_elem, num_input_fields, num_output_fields, size;
  CeedScalar         *assembled_array, *e_data[2 * CEED_FIELD_MAX] = {NULL};
  CeedVector         *active_in;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Sycl  *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetFallbackParentCeed(op, &ceed_parent));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  active_in     = impl->qf_active_in;
  num_active_in = impl->num_active_in, num_active_out = impl->num_active_out;

  // Setup
  CeedCallBackend(CeedOperatorSetup_Sycl(op));

  // Input Evecs and Restriction
  CeedCallBackend(CeedOperatorSetupInputs_Sycl(num_input_fields, qf_input_fields, op_input_fields, NULL, true, e_data, impl, request));

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
        CeedCallBackend(CeedRealloc(num_active_in + size, &active_in));
        for (CeedInt field = 0; field < size; field++) {
          q_size = (CeedSize)Q * num_elem;
          CeedCallBackend(CeedVectorCreate(ceed, q_size, &active_in[num_active_in + field]));
          CeedCallBackend(
              CeedVectorSetArray(active_in[num_active_in + field], CEED_MEM_DEVICE, CEED_USE_POINTER, &q_vec_array[field * Q * num_elem]));
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
  if (!num_active_in || !num_active_out) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Cannot assemble QFunction without active inputs and outputs");
    // LCOV_EXCL_STOP
  }

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
  CeedCallBackend(CeedOperatorInputBasis_Sycl(num_elem, qf_input_fields, op_input_fields, num_input_fields, true, e_data, impl));

  // Assemble QFunction
  for (CeedInt in = 0; in < num_active_in; in++) {
    // Set Inputs
    CeedCallBackend(CeedVectorSetValue(active_in[in], 1.0));
    if (num_active_in > 1) {
      CeedCallBackend(CeedVectorSetValue(active_in[(in + num_active_in - 1) % num_active_in], 0.0));
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

  // Un-set output Qvecs to prevent accidental overwrite of Assembled
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
  CeedCallBackend(CeedOperatorRestoreInputs_Sycl(num_input_fields, qf_input_fields, op_input_fields, true, e_data, impl));

  // Restore output
  CeedCallBackend(CeedVectorRestoreArray(*assembled, &assembled_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Sycl(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Sycl(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Sycl(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Sycl(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Assemble diagonal setup
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetup_Sycl(CeedOperator op) {
  Ceed                ceed;
  Ceed_Sycl          *sycl_data;
  CeedInt             num_input_fields, num_output_fields, num_e_mode_in = 0, num_comp = 0, dim = 1, num_e_mode_out = 0;
  CeedEvalMode       *e_mode_in = NULL, *e_mode_out = NULL;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedElemRestriction rstr_in = NULL, rstr_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_fields;
  CeedOperator_Sycl  *impl;

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
      if (rstr_in && rstr_in != rstr) {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement multi-field non-composite operator diagonal assembly");
        // LCOV_EXCL_STOP
      }
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
      if (rstr_out && rstr_out != rstr) {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement multi-field non-composite operator diagonal assembly");
        // LCOV_EXCL_STOP
      }
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
  CeedCallBackend(CeedGetData(ceed, &sycl_data));
  CeedCallBackend(CeedCalloc(1, &impl->diag));
  CeedOperatorDiag_Sycl *diag = impl->diag;

  diag->basis_in       = basis_in;
  diag->basis_out      = basis_out;
  diag->h_e_mode_in    = e_mode_in;
  diag->h_e_mode_out   = e_mode_out;
  diag->num_e_mode_in  = num_e_mode_in;
  diag->num_e_mode_out = num_e_mode_out;

  // Kernel parameters
  CeedInt num_nodes, num_qpts;
  CeedCallBackend(CeedBasisGetNumNodes(basis_in, &num_nodes));
  CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts));
  diag->num_nodes = num_nodes;
  diag->num_qpts  = num_qpts;
  diag->num_comp  = num_comp;

  // Basis matrices
  const CeedInt     i_len = num_qpts * num_nodes;
  const CeedInt     g_len = num_qpts * num_nodes * dim;
  const CeedScalar *interp_in, *interp_out, *grad_in, *grad_out;

  // CEED_EVAL_NONE
  CeedScalar *identity      = NULL;
  bool        has_eval_none = false;
  for (CeedInt i = 0; i < num_e_mode_in; i++) has_eval_none = has_eval_none || (e_mode_in[i] == CEED_EVAL_NONE);
  for (CeedInt i = 0; i < num_e_mode_out; i++) has_eval_none = has_eval_none || (e_mode_out[i] == CEED_EVAL_NONE);

  // Order queue
  sycl::event e = sycl_data->sycl_queue.ext_oneapi_submit_barrier();

  std::vector<sycl::event> copy_events;
  if (has_eval_none) {
    CeedCallBackend(CeedCalloc(num_qpts * num_nodes, &identity));
    for (CeedSize i = 0; i < (num_nodes < num_qpts ? num_nodes : num_qpts); i++) identity[i * num_nodes + i] = 1.0;
    CeedCallSycl(ceed, diag->d_identity = sycl::malloc_device<CeedScalar>(i_len, sycl_data->sycl_device, sycl_data->sycl_context));
    sycl::event identity_copy = sycl_data->sycl_queue.copy<CeedScalar>(identity, diag->d_identity, i_len, {e});
    copy_events.push_back(identity_copy);
  }

  // CEED_EVAL_INTERP
  CeedCallBackend(CeedBasisGetInterp(basis_in, &interp_in));
  CeedCallSycl(ceed, diag->d_interp_in = sycl::malloc_device<CeedScalar>(i_len, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event interp_in_copy = sycl_data->sycl_queue.copy<CeedScalar>(interp_in, diag->d_interp_in, i_len, {e});
  copy_events.push_back(interp_in_copy);

  CeedCallBackend(CeedBasisGetInterp(basis_out, &interp_out));
  CeedCallSycl(ceed, diag->d_interp_out = sycl::malloc_device<CeedScalar>(i_len, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event interp_out_copy = sycl_data->sycl_queue.copy<CeedScalar>(interp_out, diag->d_interp_out, i_len, {e});
  copy_events.push_back(interp_out_copy);

  // CEED_EVAL_GRAD
  CeedCallBackend(CeedBasisGetGrad(basis_in, &grad_in));
  CeedCallSycl(ceed, diag->d_grad_in = sycl::malloc_device<CeedScalar>(g_len, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event grad_in_copy = sycl_data->sycl_queue.copy<CeedScalar>(grad_in, diag->d_grad_in, g_len, {e});
  copy_events.push_back(grad_in_copy);

  CeedCallBackend(CeedBasisGetGrad(basis_out, &grad_out));
  CeedCallSycl(ceed, diag->d_grad_out = sycl::malloc_device<CeedScalar>(g_len, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event grad_out_copy = sycl_data->sycl_queue.copy<CeedScalar>(grad_out, diag->d_grad_out, g_len, {e});
  copy_events.push_back(grad_out_copy);

  // Arrays of  e_modes
  CeedCallSycl(ceed, diag->d_e_mode_in = sycl::malloc_device<CeedEvalMode>(num_e_mode_in, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event e_mode_in_copy = sycl_data->sycl_queue.copy<CeedEvalMode>(e_mode_in, diag->d_e_mode_in, num_e_mode_in, {e});
  copy_events.push_back(e_mode_in_copy);

  CeedCallSycl(ceed, diag->d_e_mode_out = sycl::malloc_device<CeedEvalMode>(num_e_mode_out, sycl_data->sycl_device, sycl_data->sycl_context));
  sycl::event e_mode_out_copy = sycl_data->sycl_queue.copy<CeedEvalMode>(e_mode_out, diag->d_e_mode_out, num_e_mode_out, {e});
  copy_events.push_back(e_mode_out_copy);

  // Restriction
  diag->diag_rstr = rstr_out;

  // Wait for all copies to complete and handle exceptions
  CeedCallSycl(ceed, sycl::event::wait_and_throw(copy_events));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
//  Kernel for diagonal assembly
//------------------------------------------------------------------------------
static int CeedOperatorLinearDiagonal_Sycl(sycl::queue &sycl_queue, const bool is_point_block, const CeedInt num_elem,
                                           const CeedOperatorDiag_Sycl *diag, const CeedScalar *assembled_qf_array, CeedScalar *elem_diag_array) {
  const CeedSize      num_nodes      = diag->num_nodes;
  const CeedSize      num_qpts       = diag->num_qpts;
  const CeedSize      num_comp       = diag->num_comp;
  const CeedSize      num_e_mode_in  = diag->num_e_mode_in;
  const CeedSize      num_e_mode_out = diag->num_e_mode_out;
  const CeedScalar   *identity       = diag->d_identity;
  const CeedScalar   *interp_in      = diag->d_interp_in;
  const CeedScalar   *grad_in        = diag->d_grad_in;
  const CeedScalar   *interp_out     = diag->d_interp_out;
  const CeedScalar   *grad_out       = diag->d_grad_out;
  const CeedEvalMode *e_mode_in      = diag->d_e_mode_in;
  const CeedEvalMode *e_mode_out     = diag->d_e_mode_out;

  sycl::range<1> kernel_range(num_elem * num_nodes);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedOperatorSyclLinearDiagonal>(kernel_range, {e}, [=](sycl::id<1> idx) {
    const CeedInt tid = idx % num_nodes;
    const CeedInt e   = idx / num_nodes;

    // Compute the diagonal of B^T D B
    // Each element
    CeedInt d_out = -1;
    // Each basis eval mode pair
    for (CeedSize e_out = 0; e_out < num_e_mode_out; e_out++) {
      const CeedScalar *bt = NULL;

      if (e_mode_out[e_out] == CEED_EVAL_GRAD) ++d_out;
      CeedOperatorGetBasisPointer_Sycl(&bt, e_mode_out[e_out], identity, interp_out, &grad_out[d_out * num_qpts * num_nodes]);
      CeedInt d_in = -1;

      for (CeedSize e_in = 0; e_in < num_e_mode_in; e_in++) {
        const CeedScalar *b = NULL;

        if (e_mode_in[e_in] == CEED_EVAL_GRAD) ++d_in;
        CeedOperatorGetBasisPointer_Sycl(&b, e_mode_in[e_in], identity, interp_in, &grad_in[d_in * num_qpts * num_nodes]);
        // Each component
        for (CeedSize comp_out = 0; comp_out < num_comp; comp_out++) {
          // Each qpoint/node pair
          if (is_point_block) {
            // Point Block Diagonal
            for (CeedInt comp_in = 0; comp_in < num_comp; comp_in++) {
              CeedScalar e_value = 0.0;

              for (CeedSize q = 0; q < num_qpts; q++) {
                const CeedScalar qf_value =
                    assembled_qf_array[((((e_in * num_comp + comp_in) * num_e_mode_out + e_out) * num_comp + comp_out) * num_elem + e) * num_qpts +
                                       q];

                e_value += bt[q * num_nodes + tid] * qf_value * b[q * num_nodes + tid];
              }
              elem_diag_array[((comp_out * num_comp + comp_in) * num_elem + e) * num_nodes + tid] += e_value;
            }
          } else {
            // Diagonal Only
            CeedScalar e_value = 0.0;

            for (CeedSize q = 0; q < num_qpts; q++) {
              const CeedScalar qf_value =
                  assembled_qf_array[((((e_in * num_comp + comp_out) * num_e_mode_out + e_out) * num_comp + comp_out) * num_elem + e) * num_qpts + q];
              e_value += bt[q * num_nodes + tid] * qf_value * b[q * num_nodes + tid];
            }
            elem_diag_array[(comp_out * num_elem + e) * num_nodes + tid] += e_value;
          }
        }
      }
    }
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble diagonal common code
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalCore_Sycl(CeedOperator op, CeedVector assembled, CeedRequest *request, const bool is_point_block) {
  Ceed                ceed;
  Ceed_Sycl          *sycl_data;
  CeedInt             num_elem;
  CeedScalar         *elem_diag_array;
  const CeedScalar   *assembled_qf_array;
  CeedVector          assembled_qf = NULL;
  CeedElemRestriction rstr         = NULL;
  CeedOperator_Sycl  *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  // Assemble QFunction
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &rstr, request));
  CeedCallBackend(CeedElemRestrictionDestroy(&rstr));

  // Setup
  if (!impl->diag) {
    CeedCallBackend(CeedOperatorAssembleDiagonalSetup_Sycl(op));
  }
  CeedOperatorDiag_Sycl *diag = impl->diag;

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
  CeedCallBackend(CeedOperatorLinearDiagonal_Sycl(sycl_data->sycl_queue, is_point_block, num_elem, diag, assembled_qf_array, elem_diag_array));

  // Wait for queue to complete and handle exceptions
  sycl_data->sycl_queue.wait_and_throw();

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
static int CeedOperatorLinearAssembleAddDiagonal_Sycl(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Sycl(op, assembled, request, false));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Sycl(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Sycl(op, assembled, request, true));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Single operator assembly setup
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssembleSetup_Sycl(CeedOperator op) {
  Ceed    ceed;
  CeedInt num_input_fields, num_output_fields, num_e_mode_in = 0, dim = 1, num_B_in_mats_to_load = 0, size_B_in = 0, num_e_mode_out = 0,
                                               num_B_out_mats_to_load = 0, size_B_out = 0, num_qpts = 0, elem_size = 0, num_elem, num_comp,
                                               mat_start = 0;
  CeedEvalMode       *eval_mode_in = NULL, *eval_mode_out = NULL;
  const CeedScalar   *interp_in, *grad_in;
  CeedElemRestriction rstr_in = NULL, rstr_out = NULL;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *input_fields, *output_fields;
  CeedOperator_Sycl  *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Get input and output fields
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &input_fields, &num_output_fields, &output_fields));

  // Determine active input basis eval mode
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  // Note that the kernel will treat each dimension of a gradient action separately;
  // i.e., when an active input has a CEED_EVAL_GRAD mode, num_ e_mode_in will increment by dim.
  // However, for the purposes of load_ing the B matrices, it will be treated as one mode, and we will load/copy the entire gradient matrix at once,
  // so num_B_in_mats_to_load will be incremented by 1.
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedEvalMode eval_mode;
    CeedVector   vec;

    CeedCallBackend(CeedOperatorFieldGetVector(input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
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
    CeedEvalMode eval_mode;
    CeedVector   vec;

    CeedCallBackend(CeedOperatorFieldGetVector(output_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedOperatorFieldGetBasis(output_fields[i], &basis_out));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(output_fields[i], &rstr_out));
      if (rstr_out && rstr_out != rstr_in) {
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND, "Backend does not implement multi-field non-composite operator assembly");
        // LCOV_EXCL_STOP
      }
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

  if (num_e_mode_in == 0 || num_e_mode_out == 0) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Cannot assemble operator without inputs/outputs");
    // LCOV_EXCL_STOP
  }

  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr_in, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr_in, &num_comp));

  CeedCallBackend(CeedCalloc(1, &impl->asmb));
  CeedOperatorAssemble_Sycl *asmb = impl->asmb;
  asmb->num_elem                  = num_elem;

  Ceed_Sycl *sycl_data;
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  // Kernel setup
  int elems_per_block   = 1;
  asmb->elems_per_block = elems_per_block;
  CeedInt block_size    = elem_size * elem_size * elems_per_block;

  /* CeedInt maxThreadsPerBlock = sycl_data->sycl_device.get_info<sycl::info::device::max_work_group_size>();
  bool    fallback           = block_size > maxThreadsPerBlock;
  asmb->fallback             = fallback;
  if (fallback) {
    // Use fallback kernel with 1D threadblock
    block_size         = elem_size * elems_per_block;
    asmb->block_size_x = elem_size;
    asmb->block_size_y = 1;
  } else {  // Use kernel with 2D threadblock
    asmb->block_size_x = elem_size;
    asmb->block_size_y = elem_size;
  }*/
  asmb->block_size_x   = elem_size;
  asmb->block_size_y   = elem_size;
  asmb->num_e_mode_in  = num_e_mode_in;
  asmb->num_e_mode_out = num_e_mode_out;
  asmb->num_qpts       = num_qpts;
  asmb->num_nodes      = elem_size;
  asmb->block_size     = block_size;
  asmb->num_comp       = num_comp;

  // Build 'full' B matrices (not 1D arrays used for tensor-product matrices
  CeedCallBackend(CeedBasisGetInterp(basis_in, &interp_in));
  CeedCallBackend(CeedBasisGetGrad(basis_in, &grad_in));

  // Load into B_in, in order that they will be used in eval_mode
  CeedCallSycl(ceed, asmb->d_B_in = sycl::malloc_device<CeedScalar>(size_B_in, sycl_data->sycl_device, sycl_data->sycl_context));
  for (int i = 0; i < num_B_in_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_in[i];

    if (eval_mode == CEED_EVAL_INTERP) {
      // Order queue
      sycl::event e = sycl_data->sycl_queue.ext_oneapi_submit_barrier();
      sycl_data->sycl_queue.copy<CeedScalar>(interp_in, &asmb->d_B_in[mat_start], elem_size * num_qpts, {e});
      mat_start += elem_size * num_qpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      // Order queue
      sycl::event e = sycl_data->sycl_queue.ext_oneapi_submit_barrier();
      sycl_data->sycl_queue.copy<CeedScalar>(grad_in, &asmb->d_B_in[mat_start], dim * elem_size * num_qpts, {e});
      mat_start += dim * elem_size * num_qpts;
    }
  }

  const CeedScalar *interp_out, *grad_out;
  // Note that this function currently assumes 1 basis, so this should always be true
  // for now
  if (basis_out == basis_in) {
    interp_out = interp_in;
    grad_out   = grad_in;
  } else {
    CeedCallBackend(CeedBasisGetInterp(basis_out, &interp_out));
    CeedCallBackend(CeedBasisGetGrad(basis_out, &grad_out));
  }

  // Load into B_out, in order that they will be used in eval_mode
  mat_start = 0;
  CeedCallSycl(ceed, asmb->d_B_out = sycl::malloc_device<CeedScalar>(size_B_out, sycl_data->sycl_device, sycl_data->sycl_context));
  for (int i = 0; i < num_B_out_mats_to_load; i++) {
    CeedEvalMode eval_mode = eval_mode_out[i];

    if (eval_mode == CEED_EVAL_INTERP) {
      // Order queue
      sycl::event e = sycl_data->sycl_queue.ext_oneapi_submit_barrier();
      sycl_data->sycl_queue.copy<CeedScalar>(interp_out, &asmb->d_B_out[mat_start], elem_size * num_qpts, {e});
      mat_start += elem_size * num_qpts;
    } else if (eval_mode == CEED_EVAL_GRAD) {
      // Order queue
      sycl::event e = sycl_data->sycl_queue.ext_oneapi_submit_barrier();
      sycl_data->sycl_queue.copy<CeedScalar>(grad_out, &asmb->d_B_out[mat_start], dim * elem_size * num_qpts, {e});
      mat_start += dim * elem_size * num_qpts;
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Matrix assembly kernel for low-order elements (3D thread block)
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssemble_Sycl(sycl::queue &sycl_queue, const CeedOperator_Sycl *impl, const CeedScalar *qf_array,
                                           CeedScalar *values_array) {
  // This kernels assumes B_in and B_out have the same number of quadrature points and basis points.
  // TODO: expand to more general cases
  CeedOperatorAssemble_Sycl *asmb           = impl->asmb;
  const CeedInt              num_elem       = asmb->num_elem;
  const CeedSize             num_nodes      = asmb->num_nodes;
  const CeedSize             num_comp       = asmb->num_comp;
  const CeedSize             num_qpts       = asmb->num_qpts;
  const CeedSize             num_e_mode_in  = asmb->num_e_mode_in;
  const CeedSize             num_e_mode_out = asmb->num_e_mode_out;

  // Strides for final output ordering, determined by the reference (inference) implementation of the symbolic assembly, slowest --> fastest: element,
  // comp_in, comp_out, node_row, node_col
  const CeedSize comp_out_stride = num_nodes * num_nodes;
  const CeedSize comp_in_stride  = comp_out_stride * num_comp;
  const CeedSize e_stride        = comp_in_stride * num_comp;
  // Strides for QF array, slowest --> fastest:  e_mode_in, comp_in,  e_mode_out, comp_out, elem, qpt
  const CeedSize q_e_stride          = num_qpts;
  const CeedSize q_comp_out_stride   = num_elem * q_e_stride;
  const CeedSize q_e_mode_out_stride = q_comp_out_stride * num_comp;
  const CeedSize q_comp_in_stride    = q_e_mode_out_stride * num_e_mode_out;
  const CeedSize q_e_mode_in_stride  = q_comp_in_stride * num_comp;

  CeedScalar *B_in, *B_out;
  B_in                       = asmb->d_B_in;
  B_out                      = asmb->d_B_out;
  const CeedInt block_size_x = asmb->block_size_x;
  const CeedInt block_size_y = asmb->block_size_y;

  sycl::range<3> kernel_range(num_elem, block_size_y, block_size_x);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedOperatorSyclLinearAssemble>(kernel_range, {e}, [=](sycl::id<3> idx) {
    const int e = idx.get(0);  // Element index
    const int l = idx.get(1);  // The output column index of each B^TDB operation
    const int i = idx.get(2);  // The output row index of each B^TDB operation
                               // such that we have (Bout^T)_ij D_jk Bin_kl = C_il
    for (CeedSize comp_in = 0; comp_in < num_comp; comp_in++) {
      for (CeedSize comp_out = 0; comp_out < num_comp; comp_out++) {
        CeedScalar result        = 0.0;
        CeedSize   qf_index_comp = q_comp_in_stride * comp_in + q_comp_out_stride * comp_out + q_e_stride * e;

        for (CeedSize e_mode_in = 0; e_mode_in < num_e_mode_in; e_mode_in++) {
          CeedSize b_in_index = e_mode_in * num_qpts * num_nodes;

          for (CeedSize e_mode_out = 0; e_mode_out < num_e_mode_out; e_mode_out++) {
            CeedSize b_out_index = e_mode_out * num_qpts * num_nodes;
            CeedSize qf_index    = qf_index_comp + q_e_mode_out_stride * e_mode_out + q_e_mode_in_stride * e_mode_in;

            // Perform the B^T D B operation for this 'chunk' of D (the qf_array)
            for (CeedSize j = 0; j < num_qpts; j++) {
              result += B_out[b_out_index + j * num_nodes + i] * qf_array[qf_index + j] * B_in[b_in_index + j * num_nodes + l];
            }
          }  // end of  e_mode_out
        }    // end of  e_mode_in
        CeedSize val_index = comp_in_stride * comp_in + comp_out_stride * comp_out + e_stride * e + num_nodes * i + l;

        values_array[val_index] = result;
      }  // end of out component
    }    // end of in component
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Fallback kernel for larger orders (1D thread block)
//------------------------------------------------------------------------------
/*
static int CeedOperatorLinearAssembleFallback_Sycl(sycl::queue &sycl_queue, const CeedOperator_Sycl *impl, const CeedScalar *qf_array,
                                                   CeedScalar *values_array) {
  // This kernel assumes B_in and B_out have the same number of quadrature points and basis points.
  // TODO: expand to more general cases
  CeedOperatorAssemble_Sycl *asmb        = impl->asmb;
  const CeedInt              num_elem       = asmb->num_elem;
  const CeedInt              num_nodes      = asmb->num_nodes;
  const CeedInt              num_comp       = asmb->num_comp;
  const CeedInt              num_qpts       = asmb->num_qpts;
  const CeedInt              num_e_mode_in  = asmb->num_e_mode_in;
  const CeedInt              num_e_mode_out = asmb->num_e_mode_out;

  // Strides for final output ordering, determined by the reference (interface) implementation of the symbolic assembly, slowest --> fastest: elememt,
  // comp_in, comp_out, node_row, node_col
  const CeedInt comp_out_stride = num_nodes * num_nodes;
  const CeedInt comp_in_stride  = comp_out_stride * num_comp;
  const CeedInt e_stride        = comp_in_stride * num_comp;
  // Strides for QF array, slowest --> fastest:  e_mode_in, comp_in,  e_mode_out, comp_out, elem, qpt
  const CeedInt q_e_stride         = num_qpts;
  const CeedInt q_comp_out_stride  = num_elem * q_e_stride;
  const CeedInt q_e_mode_out_stride = q_comp_out_stride * num_comp;
  const CeedInt q_comp_in_stride   = q_e_mode_out_stride * num_e_mode_out;
  const CeedInt q_e_mode_in_stride  = q_comp_in_stride * num_comp;

  CeedScalar *B_in, *B_out;
  B_in                        = asmb->d_B_in;
  B_out                       = asmb->d_B_out;
  const CeedInt elems_per_block = asmb->elems_per_block;
  const CeedInt block_size_x  = asmb->block_size_x;
  const CeedInt block_size_y  = asmb->block_size_y;  // This will be 1 for the fallback kernel

  const CeedInt     grid = num_elem / elems_per_block + ((num_elem / elems_per_block * elems_per_block < num_elem) ? 1 : 0);
  sycl::range<3>    local_range(block_size_x, block_size_y, elems_per_block);
  sycl::range<3>    global_range(grid * block_size_x, block_size_y, elems_per_block);
  sycl::nd_range<3> kernel_range(global_range, local_range);

  sycl_queue.parallel_for<CeedOperatorSyclLinearAssembleFallback>(kernel_range, [=](sycl::nd_item<3> work_item) {
    const CeedInt blockIdx  = work_item.get_group(0);
    const CeedInt gridDimx  = work_item.get_group_range(0);
    const CeedInt threadIdx = work_item.get_local_id(0);
    const CeedInt threadIdz = work_item.get_local_id(2);
    const CeedInt blockDimz = work_item.get_local_range(2);

    const int l = threadIdx;  // The output column index of each B^TDB operation
                              // such that we have (Bout^T)_ij D_jk Bin_kl = C_il
    for (CeedInt e = blockIdx * blockDimz + threadIdz; e < num_elem; e += gridDimx * blockDimz) {
      for (CeedInt comp_in = 0; comp_in < num_comp; comp_in++) {
        for (CeedInt comp_out = 0; comp_out < num_comp; comp_out++) {
          for (CeedInt i = 0; i < num_nodes; i++) {
            CeedScalar result        = 0.0;
            CeedInt    qf_index_comp = q_comp_in_stride * comp_in + q_comp_out_stride * comp_out + q_e_stride * e;
            for (CeedInt  e_mode_in = 0;  e_mode_in < num_e_mode_in;  e_mode_in++) {
              CeedInt b_in_index =  e_mode_in * num_qpts * num_nodes;
              for (CeedInt  e_mode_out = 0;  e_mode_out < num_e_mode_out;  e_mode_out++) {
                CeedInt b_out_index =  e_mode_out * num_qpts * num_nodes;
                CeedInt qf_index    = qf_index_comp + q_e_mode_out_stride *  e_mode_out + q_e_mode_in_stride *  e_mode_in;
                // Perform the B^T D B operation for this 'chunk' of D (the qf_array)
                for (CeedInt j = 0; j < num_qpts; j++) {
                  result += B_out[b_out_index + j * num_nodes + i] * qf_array[qf_index + j] * B_in[b_in_index + j * num_nodes + l];
                }
              }  // end of  e_mode_out
            }    // end of  e_mode_in
            CeedInt val_index       = comp_in_stride * comp_in + comp_out_stride * comp_out + e_stride * e + num_nodes * i + l;
            values_array[val_index] = result;
          }  // end of loop over element node index, i
        }    // end of out component
      }      // end of in component
    }        // end of element loop
  });
  return CEED_ERROR_SUCCESS;
}*/

//------------------------------------------------------------------------------
// Assemble matrix data for COO matrix of assembled operator.
// The sparsity pattern is set by CeedOperatorLinearAssembleSymbolic.
//
// Note that this (and other assembly routines) currently assume only one active
// input restriction/basis per operator (could have multiple basis eval modes).
// TODO: allow multiple active input restrictions/basis objects
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssemble_Sycl(CeedOperator op, CeedInt offset, CeedVector values) {
  Ceed                ceed;
  Ceed_Sycl          *sycl_data;
  CeedScalar         *values_array;
  const CeedScalar   *qf_array;
  CeedVector          assembled_qf = NULL;
  CeedElemRestriction rstr_q       = NULL;
  CeedOperator_Sycl  *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedGetData(ceed, &sycl_data));

  // Setup
  if (!impl->asmb) {
    CeedCallBackend(CeedSingleOperatorAssembleSetup_Sycl(op));
    assert(impl->asmb != NULL);
  }

  // Assemble QFunction
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &rstr_q, CEED_REQUEST_IMMEDIATE));
  CeedCallBackend(CeedElemRestrictionDestroy(&rstr_q));
  CeedCallBackend(CeedVectorGetArrayWrite(values, CEED_MEM_DEVICE, &values_array));
  values_array += offset;
  CeedCallBackend(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_DEVICE, &qf_array));

  // Compute B^T D B
  CeedCallBackend(CeedOperatorLinearAssemble_Sycl(sycl_data->sycl_queue, impl, qf_array, values_array));

  // Wait for kernels to be completed
  // Kris: Review if this is necessary -- enqueing an async barrier may be sufficient
  sycl_data->sycl_queue.wait_and_throw();

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
int CeedOperatorCreate_Sycl(CeedOperator op) {
  Ceed               ceed;
  CeedOperator_Sycl *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonal_Sycl));
  CeedCallBackend(
      CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleAddPointBlockDiagonal", CeedOperatorLinearAssembleAddPointBlockDiagonal_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "LinearAssembleSingle", CeedSingleOperatorAssemble_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Sycl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
