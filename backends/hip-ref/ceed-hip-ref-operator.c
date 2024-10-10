// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <ceed/jit-tools.h>
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <hip/hip_runtime.h>

#include "../hip/ceed-hip-common.h"
#include "../hip/ceed-hip-compile.h"
#include "ceed-hip-ref.h"

//------------------------------------------------------------------------------
// Destroy operator
//------------------------------------------------------------------------------
static int CeedOperatorDestroy_Hip(CeedOperator op) {
  CeedOperator_Hip *impl;

  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Apply data
  CeedCallBackend(CeedFree(&impl->num_points));
  CeedCallBackend(CeedFree(&impl->skip_rstr_in));
  CeedCallBackend(CeedFree(&impl->skip_rstr_out));
  CeedCallBackend(CeedFree(&impl->apply_add_basis_out));
  CeedCallBackend(CeedFree(&impl->input_field_order));
  CeedCallBackend(CeedFree(&impl->output_field_order));
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

  // QFunction assembly data
  for (CeedInt i = 0; i < impl->num_active_in; i++) {
    CeedCallBackend(CeedVectorDestroy(&impl->qf_active_in[i]));
  }
  CeedCallBackend(CeedFree(&impl->qf_active_in));

  // Diag data
  if (impl->diag) {
    Ceed ceed;

    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    if (impl->diag->module) {
      CeedCallHip(ceed, hipModuleUnload(impl->diag->module));
    }
    if (impl->diag->module_point_block) {
      CeedCallHip(ceed, hipModuleUnload(impl->diag->module_point_block));
    }
    CeedCallHip(ceed, hipFree(impl->diag->d_eval_modes_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_eval_modes_out));
    CeedCallHip(ceed, hipFree(impl->diag->d_identity));
    CeedCallHip(ceed, hipFree(impl->diag->d_interp_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_interp_out));
    CeedCallHip(ceed, hipFree(impl->diag->d_grad_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_grad_out));
    CeedCallHip(ceed, hipFree(impl->diag->d_div_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_div_out));
    CeedCallHip(ceed, hipFree(impl->diag->d_curl_in));
    CeedCallHip(ceed, hipFree(impl->diag->d_curl_out));
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->diag->diag_rstr));
    CeedCallBackend(CeedElemRestrictionDestroy(&impl->diag->point_block_diag_rstr));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->elem_diag));
    CeedCallBackend(CeedVectorDestroy(&impl->diag->point_block_elem_diag));
  }
  CeedCallBackend(CeedFree(&impl->diag));

  if (impl->asmb) {
    Ceed ceed;

    CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
    CeedCallHip(ceed, hipModuleUnload(impl->asmb->module));
    CeedCallHip(ceed, hipFree(impl->asmb->d_B_in));
    CeedCallHip(ceed, hipFree(impl->asmb->d_B_out));
  }
  CeedCallBackend(CeedFree(&impl->asmb));

  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Setup infields or outfields
//------------------------------------------------------------------------------
static int CeedOperatorSetupFields_Hip(CeedQFunction qf, CeedOperator op, bool is_input, bool is_at_points, bool *skip_rstr, bool *apply_add_basis,
                                       CeedVector *e_vecs, CeedVector *q_vecs, CeedInt num_fields, CeedInt Q, CeedInt num_elem) {
  Ceed                ceed;
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
    bool                is_active = false, is_strided = false, skip_e_vec = false;
    CeedSize            q_size;
    CeedInt             size;
    CeedEvalMode        eval_mode;
    CeedVector          l_vec;
    CeedElemRestriction elem_rstr;

    // Check whether this field can skip the element restriction:
    // Input CEED_VECTOR_ACTIVE
    // Output CEED_VECTOR_ACTIVE without CEED_EVAL_NONE
    // Input CEED_VECTOR_NONE with CEED_EVAL_WEIGHT
    // Input passive vectorr with CEED_EVAL_NONE and strided restriction with CEED_STRIDES_BACKEND
    CeedCallBackend(CeedOperatorFieldGetVector(op_fields[i], &l_vec));
    is_active = l_vec == CEED_VECTOR_ACTIVE;
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_fields[i], &elem_rstr));
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
    skip_e_vec = (is_input && is_active) || (is_active && eval_mode != CEED_EVAL_NONE) || (eval_mode == CEED_EVAL_WEIGHT);
    if (!skip_e_vec && is_input && !is_active && eval_mode == CEED_EVAL_NONE) {
      CeedCallBackend(CeedElemRestrictionIsStrided(elem_rstr, &is_strided));
      if (is_strided) CeedCallBackend(CeedElemRestrictionHasBackendStrides(elem_rstr, &skip_e_vec));
    }
    if (skip_e_vec) {
      e_vecs[i] = NULL;
    } else {
      CeedCallBackend(CeedElemRestrictionCreateVector(elem_rstr, NULL, &e_vecs[i]));
    }

    switch (eval_mode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_fields[i], &size));
        q_size = (CeedSize)num_elem * (CeedSize)Q * (CeedSize)size;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        break;
      case CEED_EVAL_WEIGHT: {
        CeedBasis basis;

        CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
        q_size = (CeedSize)num_elem * (CeedSize)Q;
        CeedCallBackend(CeedVectorCreate(ceed, q_size, &q_vecs[i]));
        if (is_at_points) {
          CeedInt num_points[num_elem];

          for (CeedInt i = 0; i < num_elem; i++) num_points[i] = Q;
          CeedCallBackend(
              CeedBasisApplyAtPoints(basis, num_elem, num_points, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, CEED_VECTOR_NONE, q_vecs[i]));
        } else {
          CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_vecs[i]));
        }
        break;
      }
    }
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
          if (e_vecs[i]) CeedCallBackend(CeedVectorReferenceCopy(e_vecs[i], &e_vecs[j]));
          skip_rstr[j] = true;
        }
      }
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
          if (e_vecs[i]) CeedCallBackend(CeedVectorReferenceCopy(e_vecs[i], &e_vecs[j]));
          skip_rstr[j]       = true;
          apply_add_basis[i] = true;
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// CeedOperator needs to connect all the named fields (be they active or passive) to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetup_Hip(CeedOperator op) {
  Ceed                ceed;
  bool                is_setup_done;
  CeedInt             Q, num_elem, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Hip   *impl;

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
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->e_vecs_in));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->e_vecs_out));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->skip_rstr_in));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->skip_rstr_out));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->apply_add_basis_out));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_field_order));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->output_field_order));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_states));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->q_vecs_in));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->q_vecs_out));
  impl->num_inputs  = num_input_fields;
  impl->num_outputs = num_output_fields;

  // Set up infield and outfield e-vecs and q-vecs
  CeedCallBackend(
      CeedOperatorSetupFields_Hip(qf, op, true, false, impl->skip_rstr_in, NULL, impl->e_vecs_in, impl->q_vecs_in, num_input_fields, Q, num_elem));
  CeedCallBackend(CeedOperatorSetupFields_Hip(qf, op, false, false, impl->skip_rstr_out, impl->apply_add_basis_out, impl->e_vecs_out,
                                              impl->q_vecs_out, num_output_fields, Q, num_elem));

  // Reorder fields to allow reuse of buffers
  impl->max_active_e_vec_len = 0;
  {
    bool    is_ordered[CEED_FIELD_MAX];
    CeedInt curr_index = 0;

    for (CeedInt i = 0; i < num_input_fields; i++) is_ordered[i] = false;
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedSize            e_vec_len_i;
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      if (is_ordered[i]) continue;
      is_ordered[i]                       = true;
      impl->input_field_order[curr_index] = i;
      curr_index++;
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec_i));
      if (vec_i == CEED_VECTOR_NONE) continue;  // CEED_EVAL_WEIGHT
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &rstr_i));
      CeedCallBackend(CeedElemRestrictionGetEVectorSize(rstr_i, &e_vec_len_i));
      impl->max_active_e_vec_len = e_vec_len_i > impl->max_active_e_vec_len ? e_vec_len_i : impl->max_active_e_vec_len;
      for (CeedInt j = i + 1; j < num_input_fields; j++) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[j], &rstr_j));
        if (rstr_i == rstr_j && vec_i == vec_j) {
          is_ordered[j]                       = true;
          impl->input_field_order[curr_index] = j;
          curr_index++;
        }
      }
    }
  }
  {
    bool    is_ordered[CEED_FIELD_MAX];
    CeedInt curr_index = 0;

    for (CeedInt i = 0; i < num_output_fields; i++) is_ordered[i] = false;
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedSize            e_vec_len_i;
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      if (is_ordered[i]) continue;
      is_ordered[i]                        = true;
      impl->output_field_order[curr_index] = i;
      curr_index++;
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec_i));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &rstr_i));
      CeedCallBackend(CeedElemRestrictionGetEVectorSize(rstr_i, &e_vec_len_i));
      impl->max_active_e_vec_len = e_vec_len_i > impl->max_active_e_vec_len ? e_vec_len_i : impl->max_active_e_vec_len;
      for (CeedInt j = i + 1; j < num_output_fields; j++) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[j], &rstr_j));
        if (rstr_i == rstr_j && vec_i == vec_j) {
          is_ordered[j]                        = true;
          impl->output_field_order[curr_index] = j;
          curr_index++;
        }
      }
    }
  }
  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restrict Operator Inputs
//------------------------------------------------------------------------------
static inline int CeedOperatorInputRestrict_Hip(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
                                                CeedVector in_vec, CeedVector active_e_vec, const bool skip_active, CeedOperator_Hip *impl,
                                                CeedRequest *request) {
  bool       is_active = false;
  CeedVector l_vec, e_vec = impl->e_vecs_in[input_field];

  // Get input vector
  CeedCallBackend(CeedOperatorFieldGetVector(op_input_field, &l_vec));
  is_active = l_vec == CEED_VECTOR_ACTIVE;
  if (is_active && skip_active) return CEED_ERROR_SUCCESS;
  if (is_active) {
    l_vec = in_vec;
    if (!e_vec) e_vec = active_e_vec;
  }

  // Restriction action
  if (e_vec) {
    // Restrict, if necessary
    if (!impl->skip_rstr_in[input_field]) {
      uint64_t state;

      CeedCallBackend(CeedVectorGetState(l_vec, &state));
      if (is_active || state != impl->input_states[input_field]) {
        CeedElemRestriction elem_rstr;

        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_field, &elem_rstr));
        CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_NOTRANSPOSE, l_vec, e_vec, request));
      }
      impl->input_states[input_field] = state;
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasis_Hip(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
                                             CeedVector in_vec, CeedVector active_e_vec, CeedInt num_elem, const bool skip_active,
                                             CeedOperator_Hip *impl) {
  bool         is_active = false;
  CeedEvalMode eval_mode;
  CeedVector   l_vec, e_vec = impl->e_vecs_in[input_field], q_vec = impl->q_vecs_in[input_field];

  // Skip active input
  CeedCallBackend(CeedOperatorFieldGetVector(op_input_field, &l_vec));
  is_active = l_vec == CEED_VECTOR_ACTIVE;
  if (is_active && skip_active) return CEED_ERROR_SUCCESS;
  if (is_active) {
    l_vec = in_vec;
    if (!e_vec) e_vec = active_e_vec;
  }

  // Basis action
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_field, &eval_mode));
  switch (eval_mode) {
    case CEED_EVAL_NONE: {
      const CeedScalar *e_vec_array;

      if (e_vec) {
        CeedCallBackend(CeedVectorGetArrayRead(e_vec, CEED_MEM_DEVICE, &e_vec_array));
      } else {
        CeedCallBackend(CeedVectorGetArrayRead(l_vec, CEED_MEM_DEVICE, &e_vec_array));
      }
      CeedCallBackend(CeedVectorSetArray(q_vec, CEED_MEM_DEVICE, CEED_USE_POINTER, (CeedScalar *)e_vec_array));
      break;
    }
    case CEED_EVAL_INTERP:
    case CEED_EVAL_GRAD:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL: {
      CeedBasis basis;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_input_field, &basis));
      CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_NOTRANSPOSE, eval_mode, e_vec, q_vec));
      break;
    }
    case CEED_EVAL_WEIGHT:
      break;  // No action
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restore Input Vectors
//------------------------------------------------------------------------------
static inline int CeedOperatorInputRestore_Hip(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
                                               CeedVector in_vec, CeedVector active_e_vec, const bool skip_active, CeedOperator_Hip *impl) {
  bool         is_active = false;
  CeedEvalMode eval_mode;
  CeedVector   l_vec, e_vec = impl->e_vecs_in[input_field];

  // Skip active input
  CeedCallBackend(CeedOperatorFieldGetVector(op_input_field, &l_vec));
  is_active = l_vec == CEED_VECTOR_ACTIVE;
  if (is_active && skip_active) return CEED_ERROR_SUCCESS;
  if (is_active) {
    l_vec = in_vec;
    if (!e_vec) e_vec = active_e_vec;
  }

  // Restore e-vec
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_field, &eval_mode));
  if (eval_mode == CEED_EVAL_NONE) {
    const CeedScalar *e_vec_array;

    CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_in[input_field], CEED_MEM_DEVICE, (CeedScalar **)&e_vec_array));
    if (e_vec) {
      CeedCallBackend(CeedVectorRestoreArrayRead(e_vec, &e_vec_array));
    } else {
      CeedCallBackend(CeedVectorRestoreArrayRead(l_vec, &e_vec_array));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output
//------------------------------------------------------------------------------
static int CeedOperatorApplyAdd_Hip(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedInt             Q, num_elem, num_input_fields, num_output_fields;
  Ceed                ceed;
  CeedVector          active_e_vec;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumQuadraturePoints(op, &Q));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetup_Hip(op));

  // Work vector
  CeedCallBackend(CeedGetWorkVector(ceed, impl->max_active_e_vec_len, &active_e_vec));

  // Process inputs
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedInt field = impl->input_field_order[i];

    CeedCallBackend(CeedOperatorInputRestrict_Hip(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, false, impl, request));
    CeedCallBackend(CeedOperatorInputBasis_Hip(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, num_elem, false, impl));
  }

  // Output pointers, as necessary
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE) {
      CeedScalar *e_vec_array;

      CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs_out[i], CEED_MEM_DEVICE, &e_vec_array));
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_DEVICE, CEED_USE_POINTER, e_vec_array));
    }
  }

  // Q function
  CeedCallBackend(CeedQFunctionApply(qf, num_elem * Q, impl->q_vecs_in, impl->q_vecs_out));

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorInputRestore_Hip(op_input_fields[i], qf_input_fields[i], i, in_vec, active_e_vec, false, impl));
  }

  // Output basis and restriction
  for (CeedInt i = 0; i < num_output_fields; i++) {
    bool                is_active = false;
    CeedInt             field     = impl->output_field_order[i];
    CeedEvalMode        eval_mode;
    CeedVector          l_vec, e_vec = impl->e_vecs_out[field], q_vec = impl->q_vecs_out[field];
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[field], &l_vec));
    is_active = l_vec == CEED_VECTOR_ACTIVE;
    if (is_active) {
      l_vec = out_vec;
      if (!e_vec) e_vec = active_e_vec;
    }

    // Basis action
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[field], &eval_mode));
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[field], &basis));
        if (impl->apply_add_basis_out[field]) {
          CeedCallBackend(CeedBasisApplyAdd(basis, num_elem, CEED_TRANSPOSE, eval_mode, q_vec, e_vec));
        } else {
          CeedCallBackend(CeedBasisApply(basis, num_elem, CEED_TRANSPOSE, eval_mode, q_vec, e_vec));
        }
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        // LCOV_EXCL_STOP
      }
    }

    // Restore evec
    if (eval_mode == CEED_EVAL_NONE) {
      CeedScalar *e_vec_array;

      CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[i], CEED_MEM_DEVICE, &e_vec_array));
      CeedCallBackend(CeedVectorRestoreArray(e_vec, &e_vec_array));
    }

    // Restrict
    if (impl->skip_rstr_out[field]) continue;
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[field], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_TRANSPOSE, e_vec, l_vec, request));
  }

  // Return work vector
  CeedCallBackend(CeedRestoreWorkVector(ceed, &active_e_vec));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// CeedOperator needs to connect all the named fields (be they active or passive) to the named inputs and outputs of its CeedQFunction.
//------------------------------------------------------------------------------
static int CeedOperatorSetupAtPoints_Hip(CeedOperator op) {
  Ceed                ceed;
  bool                is_setup_done;
  CeedInt             max_num_points = -1, num_elem, num_input_fields, num_output_fields;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorIsSetupDone(op, &is_setup_done));
  if (is_setup_done) return CEED_ERROR_SUCCESS;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));
  {
    CeedElemRestriction rstr_points = NULL;

    CeedCallBackend(CeedOperatorAtPointsGetPoints(op, &rstr_points, NULL));
    CeedCallBackend(CeedElemRestrictionGetMaxPointsInElement(rstr_points, &max_num_points));
    CeedCallBackend(CeedCalloc(num_elem, &impl->num_points));
    for (CeedInt e = 0; e < num_elem; e++) {
      CeedInt num_points_elem;

      CeedCallBackend(CeedElemRestrictionGetNumPointsInElement(rstr_points, e, &num_points_elem));
      impl->num_points[e] = num_points_elem;
    }
    CeedCallBackend(CeedElemRestrictionDestroy(&rstr_points));
  }
  impl->max_num_points = max_num_points;

  // Allocate
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->e_vecs_in));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->e_vecs_out));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->skip_rstr_in));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->skip_rstr_out));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->apply_add_basis_out));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_field_order));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->output_field_order));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->input_states));
  CeedCallBackend(CeedCalloc(num_input_fields, &impl->q_vecs_in));
  CeedCallBackend(CeedCalloc(num_output_fields, &impl->q_vecs_out));
  impl->num_inputs  = num_input_fields;
  impl->num_outputs = num_output_fields;

  // Set up infield and outfield e-vecs and q-vecs
  CeedCallBackend(CeedOperatorSetupFields_Hip(qf, op, true, true, impl->skip_rstr_in, NULL, impl->e_vecs_in, impl->q_vecs_in, num_input_fields,
                                              max_num_points, num_elem));
  CeedCallBackend(CeedOperatorSetupFields_Hip(qf, op, false, true, impl->skip_rstr_out, impl->apply_add_basis_out, impl->e_vecs_out, impl->q_vecs_out,
                                              num_output_fields, max_num_points, num_elem));

  // Reorder fields to allow reuse of buffers
  impl->max_active_e_vec_len = 0;
  {
    bool    is_ordered[CEED_FIELD_MAX];
    CeedInt curr_index = 0;

    for (CeedInt i = 0; i < num_input_fields; i++) is_ordered[i] = false;
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedSize            e_vec_len_i;
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      if (is_ordered[i]) continue;
      is_ordered[i]                       = true;
      impl->input_field_order[curr_index] = i;
      curr_index++;
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec_i));
      if (vec_i == CEED_VECTOR_NONE) continue;  // CEED_EVAL_WEIGHT
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &rstr_i));
      CeedCallBackend(CeedElemRestrictionGetEVectorSize(rstr_i, &e_vec_len_i));
      impl->max_active_e_vec_len = e_vec_len_i > impl->max_active_e_vec_len ? e_vec_len_i : impl->max_active_e_vec_len;
      for (CeedInt j = i + 1; j < num_input_fields; j++) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[j], &rstr_j));
        if (rstr_i == rstr_j && vec_i == vec_j) {
          is_ordered[j]                       = true;
          impl->input_field_order[curr_index] = j;
          curr_index++;
        }
      }
    }
  }
  {
    bool    is_ordered[CEED_FIELD_MAX];
    CeedInt curr_index = 0;

    for (CeedInt i = 0; i < num_output_fields; i++) is_ordered[i] = false;
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedSize            e_vec_len_i;
      CeedVector          vec_i;
      CeedElemRestriction rstr_i;

      if (is_ordered[i]) continue;
      is_ordered[i]                        = true;
      impl->output_field_order[curr_index] = i;
      curr_index++;
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &vec_i));
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &rstr_i));
      CeedCallBackend(CeedElemRestrictionGetEVectorSize(rstr_i, &e_vec_len_i));
      impl->max_active_e_vec_len = e_vec_len_i > impl->max_active_e_vec_len ? e_vec_len_i : impl->max_active_e_vec_len;
      for (CeedInt j = i + 1; j < num_output_fields; j++) {
        CeedVector          vec_j;
        CeedElemRestriction rstr_j;

        CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[j], &vec_j));
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[j], &rstr_j));
        if (rstr_i == rstr_j && vec_i == vec_j) {
          is_ordered[j]                        = true;
          impl->output_field_order[curr_index] = j;
          curr_index++;
        }
      }
    }
  }

  CeedCallBackend(CeedOperatorSetSetupDone(op));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Input Basis Action AtPoints
//------------------------------------------------------------------------------
static inline int CeedOperatorInputBasisAtPoints_Hip(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
                                                     CeedVector in_vec, CeedVector active_e_vec, CeedInt num_elem, const CeedInt *num_points,
                                                     const bool skip_active, CeedOperator_Hip *impl) {
  bool         is_active = false;
  CeedEvalMode eval_mode;
  CeedVector   l_vec, e_vec = impl->e_vecs_in[input_field], q_vec = impl->q_vecs_in[input_field];

  // Skip active input
  CeedCallBackend(CeedOperatorFieldGetVector(op_input_field, &l_vec));
  is_active = l_vec == CEED_VECTOR_ACTIVE;
  if (is_active && skip_active) return CEED_ERROR_SUCCESS;
  if (is_active) {
    l_vec = in_vec;
    if (!e_vec) e_vec = active_e_vec;
  }

  // Basis action
  CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_field, &eval_mode));
  switch (eval_mode) {
    case CEED_EVAL_NONE: {
      const CeedScalar *e_vec_array;

      if (e_vec) {
        CeedCallBackend(CeedVectorGetArrayRead(e_vec, CEED_MEM_DEVICE, &e_vec_array));
      } else {
        CeedCallBackend(CeedVectorGetArrayRead(l_vec, CEED_MEM_DEVICE, &e_vec_array));
      }
      CeedCallBackend(CeedVectorSetArray(q_vec, CEED_MEM_DEVICE, CEED_USE_POINTER, (CeedScalar *)e_vec_array));
      break;
    }
    case CEED_EVAL_INTERP:
    case CEED_EVAL_GRAD:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL: {
      CeedBasis basis;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_input_field, &basis));
      CeedCallBackend(CeedBasisApplyAtPoints(basis, num_elem, num_points, CEED_NOTRANSPOSE, eval_mode, impl->point_coords_elem, e_vec, q_vec));
      break;
    }
    case CEED_EVAL_WEIGHT:
      break;  // No action
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply and add to output AtPoints
//------------------------------------------------------------------------------
static int CeedOperatorApplyAddAtPoints_Hip(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
  CeedInt             max_num_points, *num_points, num_elem, num_input_fields, num_output_fields;
  Ceed                ceed;
  CeedVector          active_e_vec;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetupAtPoints_Hip(op));
  num_points     = impl->num_points;
  max_num_points = impl->max_num_points;

  // Work vector
  CeedCallBackend(CeedGetWorkVector(ceed, impl->max_active_e_vec_len, &active_e_vec));

  // Get point coordinates
  if (!impl->point_coords_elem) {
    CeedVector          point_coords = NULL;
    CeedElemRestriction rstr_points  = NULL;

    CeedCallBackend(CeedOperatorAtPointsGetPoints(op, &rstr_points, &point_coords));
    CeedCallBackend(CeedElemRestrictionCreateVector(rstr_points, NULL, &impl->point_coords_elem));
    CeedCallBackend(CeedElemRestrictionApply(rstr_points, CEED_NOTRANSPOSE, point_coords, impl->point_coords_elem, request));
    CeedCallBackend(CeedVectorDestroy(&point_coords));
    CeedCallBackend(CeedElemRestrictionDestroy(&rstr_points));
  }

  // Process inputs
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedInt field = impl->input_field_order[i];

    CeedCallBackend(CeedOperatorInputRestrict_Hip(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, false, impl, request));
    CeedCallBackend(CeedOperatorInputBasisAtPoints_Hip(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, num_elem,
                                                       num_points, false, impl));
  }

  // Output pointers, as necessary
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE) {
      CeedScalar *e_vec_array;

      CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs_out[i], CEED_MEM_DEVICE, &e_vec_array));
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_DEVICE, CEED_USE_POINTER, e_vec_array));
    }
  }

  // Q function
  CeedCallBackend(CeedQFunctionApply(qf, num_elem * max_num_points, impl->q_vecs_in, impl->q_vecs_out));

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorInputRestore_Hip(op_input_fields[i], qf_input_fields[i], i, in_vec, active_e_vec, false, impl));
  }

  // Output basis and restriction
  for (CeedInt i = 0; i < num_output_fields; i++) {
    bool                is_active = false;
    CeedInt             field     = impl->output_field_order[i];
    CeedEvalMode        eval_mode;
    CeedVector          l_vec, e_vec = impl->e_vecs_out[field], q_vec = impl->q_vecs_out[field];
    CeedElemRestriction elem_rstr;
    CeedBasis           basis;

    // Output vector
    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[field], &l_vec));
    is_active = l_vec == CEED_VECTOR_ACTIVE;
    if (is_active) {
      l_vec = out_vec;
      if (!e_vec) e_vec = active_e_vec;
    }

    // Basis action
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[field], &eval_mode));
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        break;  // No action
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[field], &basis));
        if (impl->apply_add_basis_out[field]) {
          CeedCallBackend(CeedBasisApplyAddAtPoints(basis, num_elem, num_points, CEED_TRANSPOSE, eval_mode, impl->point_coords_elem, q_vec, e_vec));
        } else {
          CeedCallBackend(CeedBasisApplyAtPoints(basis, num_elem, num_points, CEED_TRANSPOSE, eval_mode, impl->point_coords_elem, q_vec, e_vec));
        }
        break;
      // LCOV_EXCL_START
      case CEED_EVAL_WEIGHT: {
        return CeedError(ceed, CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
        // LCOV_EXCL_STOP
      }
    }

    // Restore evec
    if (eval_mode == CEED_EVAL_NONE) {
      CeedScalar *e_vec_array;

      CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[i], CEED_MEM_DEVICE, &e_vec_array));
      CeedCallBackend(CeedVectorRestoreArray(e_vec, &e_vec_array));
    }

    // Restrict
    if (impl->skip_rstr_out[field]) continue;
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[field], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_TRANSPOSE, e_vec, l_vec, request));
  }

  // Restore work vector
  CeedCallBackend(CeedRestoreWorkVector(ceed, &active_e_vec));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Linear QFunction Assembly Core
//------------------------------------------------------------------------------
static inline int CeedOperatorLinearAssembleQFunctionCore_Hip(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
                                                              CeedRequest *request) {
  Ceed                ceed, ceed_parent;
  CeedInt             num_active_in, num_active_out, Q, num_elem, num_input_fields, num_output_fields, size;
  CeedScalar         *assembled_array;
  CeedVector         *active_inputs;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Hip   *impl;

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
  CeedCallBackend(CeedOperatorSetup_Hip(op));

  // Process inputs
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorInputRestrict_Hip(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl, request));
    CeedCallBackend(CeedOperatorInputBasis_Hip(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, num_elem, true, impl));
  }

  // Count number of active input fields
  if (!num_active_in) {
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedScalar *q_vec_array;
      CeedVector  l_vec;

      // Check if active input
      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &l_vec));
      if (l_vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_input_fields[i], &size));
        CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
        CeedCallBackend(CeedVectorGetArray(impl->q_vecs_in[i], CEED_MEM_DEVICE, &q_vec_array));
        CeedCallBackend(CeedRealloc(num_active_in + size, &active_inputs));
        for (CeedInt field = 0; field < size; field++) {
          CeedSize q_size = (CeedSize)Q * num_elem;

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
      CeedVector l_vec;

      // Check if active output
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[i], &l_vec));
      if (l_vec == CEED_VECTOR_ACTIVE) {
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
    CeedSize l_size     = (CeedSize)num_elem * Q * num_active_in * num_active_out;
    CeedInt  strides[3] = {1, num_elem * Q, Q}; /* *NOPAD* */

    // Create output restriction
    CeedCallBackend(CeedElemRestrictionCreateStrided(ceed_parent, num_elem, Q, num_active_in * num_active_out,
                                                     (CeedSize)num_active_in * (CeedSize)num_active_out * (CeedSize)num_elem * (CeedSize)Q, strides,
                                                     rstr));
    // Create assembled vector
    CeedCallBackend(CeedVectorCreate(ceed_parent, l_size, assembled));
  }
  CeedCallBackend(CeedVectorSetValue(*assembled, 0.0));
  CeedCallBackend(CeedVectorGetArray(*assembled, CEED_MEM_DEVICE, &assembled_array));

  // Assemble QFunction
  for (CeedInt in = 0; in < num_active_in; in++) {
    // Set Inputs
    CeedCallBackend(CeedVectorSetValue(active_inputs[in], 1.0));
    if (num_active_in > 1) {
      CeedCallBackend(CeedVectorSetValue(active_inputs[(in + num_active_in - 1) % num_active_in], 0.0));
    }
    // Set Outputs
    for (CeedInt out = 0; out < num_output_fields; out++) {
      CeedVector l_vec;

      // Get output vector
      CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &l_vec));
      // Check if active output
      if (l_vec == CEED_VECTOR_ACTIVE) {
        CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[out], CEED_MEM_DEVICE, CEED_USE_POINTER, assembled_array));
        CeedCallBackend(CeedQFunctionFieldGetSize(qf_output_fields[out], &size));
        assembled_array += size * Q * num_elem;  // Advance the pointer by the size of the output
      }
    }
    // Apply QFunction
    CeedCallBackend(CeedQFunctionApply(qf, Q * num_elem, impl->q_vecs_in, impl->q_vecs_out));
  }

  // Un-set output q-vecs to prevent accidental overwrite of Assembled
  for (CeedInt out = 0; out < num_output_fields; out++) {
    CeedVector l_vec;

    CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[out], &l_vec));
    if (l_vec == CEED_VECTOR_ACTIVE) {
      CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_out[out], CEED_MEM_DEVICE, NULL));
    }
  }

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorInputRestore_Hip(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl));
  }

  // Restore output
  CeedCallBackend(CeedVectorRestoreArray(*assembled, &assembled_array));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunction_Hip(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip(op, true, assembled, rstr, request);
}

//------------------------------------------------------------------------------
// Update Assembled Linear QFunction
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionUpdate_Hip(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
  return CeedOperatorLinearAssembleQFunctionCore_Hip(op, false, &assembled, &rstr, request);
}

//------------------------------------------------------------------------------
// Assemble Diagonal Setup
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetup_Hip(CeedOperator op) {
  Ceed                ceed;
  CeedInt             num_input_fields, num_output_fields, num_eval_modes_in = 0, num_eval_modes_out = 0;
  CeedInt             q_comp, num_nodes, num_qpts;
  CeedEvalMode       *eval_modes_in = NULL, *eval_modes_out = NULL;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_fields;
  CeedOperator_Hip   *impl;

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
      CeedBasis    basis;
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
      CeedCheck(!basis_in || basis_in == basis, ceed, CEED_ERROR_BACKEND,
                "Backend does not implement operator diagonal assembly with multiple active bases");
      basis_in = basis;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_in, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF assembly
        CeedCallBackend(CeedRealloc(num_eval_modes_in + q_comp, &eval_modes_in));
        for (CeedInt d = 0; d < q_comp; d++) eval_modes_in[num_eval_modes_in + d] = eval_mode;
        num_eval_modes_in += q_comp;
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
      CeedBasis    basis;
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis));
      CeedCheck(!basis_out || basis_out == basis, ceed, CEED_ERROR_BACKEND,
                "Backend does not implement operator diagonal assembly with multiple active bases");
      basis_out = basis;
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_out, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF assembly
        CeedCallBackend(CeedRealloc(num_eval_modes_out + q_comp, &eval_modes_out));
        for (CeedInt d = 0; d < q_comp; d++) eval_modes_out[num_eval_modes_out + d] = eval_mode;
        num_eval_modes_out += q_comp;
      }
    }
  }

  // Operator data struct
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedCalloc(1, &impl->diag));
  CeedOperatorDiag_Hip *diag = impl->diag;

  // Basis matrices
  CeedCallBackend(CeedBasisGetNumNodes(basis_in, &num_nodes));
  if (basis_in == CEED_BASIS_NONE) num_qpts = num_nodes;
  else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts));
  const CeedInt interp_bytes     = num_nodes * num_qpts * sizeof(CeedScalar);
  const CeedInt eval_modes_bytes = sizeof(CeedEvalMode);
  bool          has_eval_none    = false;

  // CEED_EVAL_NONE
  for (CeedInt i = 0; i < num_eval_modes_in; i++) has_eval_none = has_eval_none || (eval_modes_in[i] == CEED_EVAL_NONE);
  for (CeedInt i = 0; i < num_eval_modes_out; i++) has_eval_none = has_eval_none || (eval_modes_out[i] == CEED_EVAL_NONE);
  if (has_eval_none) {
    CeedScalar *identity = NULL;

    CeedCallBackend(CeedCalloc(num_nodes * num_qpts, &identity));
    for (CeedInt i = 0; i < (num_nodes < num_qpts ? num_nodes : num_qpts); i++) identity[i * num_nodes + i] = 1.0;
    CeedCallHip(ceed, hipMalloc((void **)&diag->d_identity, interp_bytes));
    CeedCallHip(ceed, hipMemcpy(diag->d_identity, identity, interp_bytes, hipMemcpyHostToDevice));
    CeedCallBackend(CeedFree(&identity));
  }

  // CEED_EVAL_INTERP, CEED_EVAL_GRAD, CEED_EVAL_DIV, and CEED_EVAL_CURL
  for (CeedInt in = 0; in < 2; in++) {
    CeedFESpace fespace;
    CeedBasis   basis = in ? basis_in : basis_out;

    CeedCallBackend(CeedBasisGetFESpace(basis, &fespace));
    switch (fespace) {
      case CEED_FE_SPACE_H1: {
        CeedInt           q_comp_interp, q_comp_grad;
        const CeedScalar *interp, *grad;
        CeedScalar       *d_interp, *d_grad;

        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp_grad));

        CeedCallBackend(CeedBasisGetInterp(basis, &interp));
        CeedCallHip(ceed, hipMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
        CeedCallHip(ceed, hipMemcpy(d_interp, interp, interp_bytes * q_comp_interp, hipMemcpyHostToDevice));
        CeedCallBackend(CeedBasisGetGrad(basis, &grad));
        CeedCallHip(ceed, hipMalloc((void **)&d_grad, interp_bytes * q_comp_grad));
        CeedCallHip(ceed, hipMemcpy(d_grad, grad, interp_bytes * q_comp_grad, hipMemcpyHostToDevice));
        if (in) {
          diag->d_interp_in = d_interp;
          diag->d_grad_in   = d_grad;
        } else {
          diag->d_interp_out = d_interp;
          diag->d_grad_out   = d_grad;
        }
      } break;
      case CEED_FE_SPACE_HDIV: {
        CeedInt           q_comp_interp, q_comp_div;
        const CeedScalar *interp, *div;
        CeedScalar       *d_interp, *d_div;

        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_DIV, &q_comp_div));

        CeedCallBackend(CeedBasisGetInterp(basis, &interp));
        CeedCallHip(ceed, hipMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
        CeedCallHip(ceed, hipMemcpy(d_interp, interp, interp_bytes * q_comp_interp, hipMemcpyHostToDevice));
        CeedCallBackend(CeedBasisGetDiv(basis, &div));
        CeedCallHip(ceed, hipMalloc((void **)&d_div, interp_bytes * q_comp_div));
        CeedCallHip(ceed, hipMemcpy(d_div, div, interp_bytes * q_comp_div, hipMemcpyHostToDevice));
        if (in) {
          diag->d_interp_in = d_interp;
          diag->d_div_in    = d_div;
        } else {
          diag->d_interp_out = d_interp;
          diag->d_div_out    = d_div;
        }
      } break;
      case CEED_FE_SPACE_HCURL: {
        CeedInt           q_comp_interp, q_comp_curl;
        const CeedScalar *interp, *curl;
        CeedScalar       *d_interp, *d_curl;

        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp_interp));
        CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_CURL, &q_comp_curl));

        CeedCallBackend(CeedBasisGetInterp(basis, &interp));
        CeedCallHip(ceed, hipMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
        CeedCallHip(ceed, hipMemcpy(d_interp, interp, interp_bytes * q_comp_interp, hipMemcpyHostToDevice));
        CeedCallBackend(CeedBasisGetCurl(basis, &curl));
        CeedCallHip(ceed, hipMalloc((void **)&d_curl, interp_bytes * q_comp_curl));
        CeedCallHip(ceed, hipMemcpy(d_curl, curl, interp_bytes * q_comp_curl, hipMemcpyHostToDevice));
        if (in) {
          diag->d_interp_in = d_interp;
          diag->d_curl_in   = d_curl;
        } else {
          diag->d_interp_out = d_interp;
          diag->d_curl_out   = d_curl;
        }
      } break;
    }
  }

  // Arrays of eval_modes
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_eval_modes_in, num_eval_modes_in * eval_modes_bytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_eval_modes_in, eval_modes_in, num_eval_modes_in * eval_modes_bytes, hipMemcpyHostToDevice));
  CeedCallHip(ceed, hipMalloc((void **)&diag->d_eval_modes_out, num_eval_modes_out * eval_modes_bytes));
  CeedCallHip(ceed, hipMemcpy(diag->d_eval_modes_out, eval_modes_out, num_eval_modes_out * eval_modes_bytes, hipMemcpyHostToDevice));
  CeedCallBackend(CeedFree(&eval_modes_in));
  CeedCallBackend(CeedFree(&eval_modes_out));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Diagonal Setup (Compilation)
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalSetupCompile_Hip(CeedOperator op, CeedInt use_ceedsize_idx, const bool is_point_block) {
  Ceed                ceed;
  char               *diagonal_kernel_source;
  const char         *diagonal_kernel_path;
  CeedInt             num_input_fields, num_output_fields, num_eval_modes_in = 0, num_eval_modes_out = 0;
  CeedInt             num_comp, q_comp, num_nodes, num_qpts;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_fields;
  CeedOperator_Hip   *impl;

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
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis_in));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_in, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        num_eval_modes_in += q_comp;
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
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(op_fields[i], &basis_out));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_out, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        num_eval_modes_out += q_comp;
      }
    }
  }

  // Operator data struct
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedOperatorDiag_Hip *diag = impl->diag;

  // Assemble kernel
  hipModule_t *module          = is_point_block ? &diag->module_point_block : &diag->module;
  CeedInt      elems_per_block = 1;
  CeedCallBackend(CeedBasisGetNumNodes(basis_in, &num_nodes));
  CeedCallBackend(CeedBasisGetNumComponents(basis_in, &num_comp));
  if (basis_in == CEED_BASIS_NONE) num_qpts = num_nodes;
  else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-operator-assemble-diagonal.h", &diagonal_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Diagonal Assembly Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, diagonal_kernel_path, &diagonal_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Diagonal Assembly Source Complete! -----\n");
  CeedCallHip(ceed, CeedCompile_Hip(ceed, diagonal_kernel_source, module, 8, "NUM_EVAL_MODES_IN", num_eval_modes_in, "NUM_EVAL_MODES_OUT",
                                    num_eval_modes_out, "NUM_COMP", num_comp, "NUM_NODES", num_nodes, "NUM_QPTS", num_qpts, "USE_CEEDSIZE",
                                    use_ceedsize_idx, "USE_POINT_BLOCK", is_point_block ? 1 : 0, "BLOCK_SIZE", num_nodes * elems_per_block));
  CeedCallHip(ceed, CeedGetKernel_Hip(ceed, *module, "LinearDiagonal", is_point_block ? &diag->LinearPointBlock : &diag->LinearDiagonal));
  CeedCallBackend(CeedFree(&diagonal_kernel_path));
  CeedCallBackend(CeedFree(&diagonal_kernel_source));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Diagonal Core
//------------------------------------------------------------------------------
static inline int CeedOperatorAssembleDiagonalCore_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request, const bool is_point_block) {
  Ceed                ceed;
  CeedInt             num_elem, num_nodes;
  CeedScalar         *elem_diag_array;
  const CeedScalar   *assembled_qf_array;
  CeedVector          assembled_qf   = NULL, elem_diag;
  CeedElemRestriction assembled_rstr = NULL, rstr_in, rstr_out, diag_rstr;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Assemble QFunction
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &assembled_rstr, request));
  CeedCallBackend(CeedElemRestrictionDestroy(&assembled_rstr));
  CeedCallBackend(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_DEVICE, &assembled_qf_array));

  // Setup
  if (!impl->diag) CeedCallBackend(CeedOperatorAssembleDiagonalSetup_Hip(op));
  CeedOperatorDiag_Hip *diag = impl->diag;

  assert(diag != NULL);

  // Assemble kernel if needed
  if ((!is_point_block && !diag->LinearDiagonal) || (is_point_block && !diag->LinearPointBlock)) {
    CeedSize assembled_length, assembled_qf_length;
    CeedInt  use_ceedsize_idx = 0;
    CeedCallBackend(CeedVectorGetLength(assembled, &assembled_length));
    CeedCallBackend(CeedVectorGetLength(assembled_qf, &assembled_qf_length));
    if ((assembled_length > INT_MAX) || (assembled_qf_length > INT_MAX)) use_ceedsize_idx = 1;

    CeedCallBackend(CeedOperatorAssembleDiagonalSetupCompile_Hip(op, use_ceedsize_idx, is_point_block));
  }

  // Restriction and diagonal vector
  CeedCallBackend(CeedOperatorGetActiveElemRestrictions(op, &rstr_in, &rstr_out));
  CeedCheck(rstr_in == rstr_out, ceed, CEED_ERROR_BACKEND,
            "Cannot assemble operator diagonal with different input and output active element restrictions");
  if (!is_point_block && !diag->diag_rstr) {
    CeedCallBackend(CeedElemRestrictionCreateUnsignedCopy(rstr_out, &diag->diag_rstr));
    CeedCallBackend(CeedElemRestrictionCreateVector(diag->diag_rstr, NULL, &diag->elem_diag));
  } else if (is_point_block && !diag->point_block_diag_rstr) {
    CeedCallBackend(CeedOperatorCreateActivePointBlockRestriction(rstr_out, &diag->point_block_diag_rstr));
    CeedCallBackend(CeedElemRestrictionCreateVector(diag->point_block_diag_rstr, NULL, &diag->point_block_elem_diag));
  }
  diag_rstr = is_point_block ? diag->point_block_diag_rstr : diag->diag_rstr;
  elem_diag = is_point_block ? diag->point_block_elem_diag : diag->elem_diag;
  CeedCallBackend(CeedVectorSetValue(elem_diag, 0.0));

  // Only assemble diagonal if the basis has nodes, otherwise inputs are null pointers
  CeedCallBackend(CeedElemRestrictionGetElementSize(diag_rstr, &num_nodes));
  if (num_nodes > 0) {
    // Assemble element operator diagonals
    CeedCallBackend(CeedElemRestrictionGetNumElements(diag_rstr, &num_elem));
    CeedCallBackend(CeedVectorGetArray(elem_diag, CEED_MEM_DEVICE, &elem_diag_array));

    // Compute the diagonal of B^T D B
    CeedInt elems_per_block = 1;
    CeedInt grid            = CeedDivUpInt(num_elem, elems_per_block);
    void   *args[]          = {(void *)&num_elem,      &diag->d_identity,       &diag->d_interp_in,  &diag->d_grad_in, &diag->d_div_in,
                               &diag->d_curl_in,       &diag->d_interp_out,     &diag->d_grad_out,   &diag->d_div_out, &diag->d_curl_out,
                               &diag->d_eval_modes_in, &diag->d_eval_modes_out, &assembled_qf_array, &elem_diag_array};

    if (is_point_block) {
      CeedCallBackend(CeedRunKernelDim_Hip(ceed, diag->LinearPointBlock, grid, num_nodes, 1, elems_per_block, args));
    } else {
      CeedCallBackend(CeedRunKernelDim_Hip(ceed, diag->LinearDiagonal, grid, num_nodes, 1, elems_per_block, args));
    }

    // Restore arrays
    CeedCallBackend(CeedVectorRestoreArray(elem_diag, &elem_diag_array));
    CeedCallBackend(CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array));
  }

  // Assemble local operator diagonal
  CeedCallBackend(CeedElemRestrictionApply(diag_rstr, CEED_TRANSPOSE, elem_diag, assembled, request));

  // Cleanup
  CeedCallBackend(CeedVectorDestroy(&assembled_qf));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonal_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, false));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear Point Block Diagonal
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Hip(op, assembled, request, true));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Single Operator Assembly Setup
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssembleSetup_Hip(CeedOperator op, CeedInt use_ceedsize_idx) {
  Ceed                ceed;
  Ceed_Hip           *Hip_data;
  char               *assembly_kernel_source;
  const char         *assembly_kernel_path;
  CeedInt             num_input_fields, num_output_fields, num_eval_modes_in = 0, num_eval_modes_out = 0;
  CeedInt             elem_size_in, num_qpts_in = 0, num_comp_in, elem_size_out, num_qpts_out, num_comp_out, q_comp;
  CeedEvalMode       *eval_modes_in = NULL, *eval_modes_out = NULL;
  CeedElemRestriction rstr_in = NULL, rstr_out = NULL;
  CeedBasis           basis_in = NULL, basis_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *input_fields, *output_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Get intput and output fields
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &input_fields, &num_output_fields, &output_fields));

  // Determine active input basis eval mode
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(input_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedBasis    basis;
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(input_fields[i], &basis));
      CeedCheck(!basis_in || basis_in == basis, ceed, CEED_ERROR_BACKEND, "Backend does not implement operator assembly with multiple active bases");
      basis_in = basis;
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(input_fields[i], &rstr_in));
      CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_in, &elem_size_in));
      if (basis_in == CEED_BASIS_NONE) num_qpts_in = elem_size_in;
      else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts_in));
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_in, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF Assembly
        CeedCallBackend(CeedRealloc(num_eval_modes_in + q_comp, &eval_modes_in));
        for (CeedInt d = 0; d < q_comp; d++) {
          eval_modes_in[num_eval_modes_in + d] = eval_mode;
        }
        num_eval_modes_in += q_comp;
      }
    }
  }

  // Determine active output basis; basis_out and rstr_out only used if same as input, TODO
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields));
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(output_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedBasis    basis;
      CeedEvalMode eval_mode;

      CeedCallBackend(CeedOperatorFieldGetBasis(output_fields[i], &basis));
      CeedCheck(!basis_out || basis_out == basis, ceed, CEED_ERROR_BACKEND,
                "Backend does not implement operator assembly with multiple active bases");
      basis_out = basis;
      CeedCallBackend(CeedOperatorFieldGetElemRestriction(output_fields[i], &rstr_out));
      CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_out, &elem_size_out));
      if (basis_out == CEED_BASIS_NONE) num_qpts_out = elem_size_out;
      else CeedCallBackend(CeedBasisGetNumQuadraturePoints(basis_out, &num_qpts_out));
      CeedCheck(num_qpts_in == num_qpts_out, ceed, CEED_ERROR_UNSUPPORTED,
                "Active input and output bases must have the same number of quadrature points");
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_out, eval_mode, &q_comp));
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF Assembly
        CeedCallBackend(CeedRealloc(num_eval_modes_out + q_comp, &eval_modes_out));
        for (CeedInt d = 0; d < q_comp; d++) {
          eval_modes_out[num_eval_modes_out + d] = eval_mode;
        }
        num_eval_modes_out += q_comp;
      }
    }
  }
  CeedCheck(num_eval_modes_in > 0 && num_eval_modes_out > 0, ceed, CEED_ERROR_UNSUPPORTED, "Cannot assemble operator without inputs/outputs");

  CeedCallBackend(CeedCalloc(1, &impl->asmb));
  CeedOperatorAssemble_Hip *asmb = impl->asmb;
  asmb->elems_per_block          = 1;
  asmb->block_size_x             = elem_size_in;
  asmb->block_size_y             = elem_size_out;

  CeedCallBackend(CeedGetData(ceed, &Hip_data));
  bool fallback = asmb->block_size_x * asmb->block_size_y * asmb->elems_per_block > Hip_data->device_prop.maxThreadsPerBlock;

  if (fallback) {
    // Use fallback kernel with 1D threadblock
    asmb->block_size_y = 1;
  }

  // Compile kernels
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr_in, &num_comp_in));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr_out, &num_comp_out));
  CeedCallBackend(CeedGetJitAbsolutePath(ceed, "ceed/jit-source/hip/hip-ref-operator-assemble.h", &assembly_kernel_path));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Assembly Kernel Source -----\n");
  CeedCallBackend(CeedLoadSourceToBuffer(ceed, assembly_kernel_path, &assembly_kernel_source));
  CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "----- Loading Assembly Source Complete! -----\n");
  CeedCallBackend(CeedCompile_Hip(ceed, assembly_kernel_source, &asmb->module, 10, "NUM_EVAL_MODES_IN", num_eval_modes_in, "NUM_EVAL_MODES_OUT",
                                  num_eval_modes_out, "NUM_COMP_IN", num_comp_in, "NUM_COMP_OUT", num_comp_out, "NUM_NODES_IN", elem_size_in,
                                  "NUM_NODES_OUT", elem_size_out, "NUM_QPTS", num_qpts_in, "BLOCK_SIZE",
                                  asmb->block_size_x * asmb->block_size_y * asmb->elems_per_block, "BLOCK_SIZE_Y", asmb->block_size_y, "USE_CEEDSIZE",
                                  use_ceedsize_idx));
  CeedCallBackend(CeedGetKernel_Hip(ceed, asmb->module, "LinearAssemble", &asmb->LinearAssemble));
  CeedCallBackend(CeedFree(&assembly_kernel_path));
  CeedCallBackend(CeedFree(&assembly_kernel_source));

  // Load into B_in, in order that they will be used in eval_modes_in
  {
    const CeedInt in_bytes           = elem_size_in * num_qpts_in * num_eval_modes_in * sizeof(CeedScalar);
    CeedInt       d_in               = 0;
    CeedEvalMode  eval_modes_in_prev = CEED_EVAL_NONE;
    bool          has_eval_none      = false;
    CeedScalar   *identity           = NULL;

    for (CeedInt i = 0; i < num_eval_modes_in; i++) {
      has_eval_none = has_eval_none || (eval_modes_in[i] == CEED_EVAL_NONE);
    }
    if (has_eval_none) {
      CeedCallBackend(CeedCalloc(elem_size_in * num_qpts_in, &identity));
      for (CeedInt i = 0; i < (elem_size_in < num_qpts_in ? elem_size_in : num_qpts_in); i++) identity[i * elem_size_in + i] = 1.0;
    }

    CeedCallHip(ceed, hipMalloc((void **)&asmb->d_B_in, in_bytes));
    for (CeedInt i = 0; i < num_eval_modes_in; i++) {
      const CeedScalar *h_B_in;

      CeedCallBackend(CeedOperatorGetBasisPointer(basis_in, eval_modes_in[i], identity, &h_B_in));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_in, eval_modes_in[i], &q_comp));
      if (q_comp > 1) {
        if (i == 0 || eval_modes_in[i] != eval_modes_in_prev) d_in = 0;
        else h_B_in = &h_B_in[(++d_in) * elem_size_in * num_qpts_in];
      }
      eval_modes_in_prev = eval_modes_in[i];

      CeedCallHip(ceed, hipMemcpy(&asmb->d_B_in[i * elem_size_in * num_qpts_in], h_B_in, elem_size_in * num_qpts_in * sizeof(CeedScalar),
                                  hipMemcpyHostToDevice));
    }

    if (identity) {
      CeedCallBackend(CeedFree(&identity));
    }
  }
  CeedCallBackend(CeedFree(&eval_modes_in));

  // Load into B_out, in order that they will be used in eval_modes_out
  {
    const CeedInt out_bytes           = elem_size_out * num_qpts_out * num_eval_modes_out * sizeof(CeedScalar);
    CeedInt       d_out               = 0;
    CeedEvalMode  eval_modes_out_prev = CEED_EVAL_NONE;
    bool          has_eval_none       = false;
    CeedScalar   *identity            = NULL;

    for (CeedInt i = 0; i < num_eval_modes_out; i++) {
      has_eval_none = has_eval_none || (eval_modes_out[i] == CEED_EVAL_NONE);
    }
    if (has_eval_none) {
      CeedCallBackend(CeedCalloc(elem_size_out * num_qpts_out, &identity));
      for (CeedInt i = 0; i < (elem_size_out < num_qpts_out ? elem_size_out : num_qpts_out); i++) identity[i * elem_size_out + i] = 1.0;
    }

    CeedCallHip(ceed, hipMalloc((void **)&asmb->d_B_out, out_bytes));
    for (CeedInt i = 0; i < num_eval_modes_out; i++) {
      const CeedScalar *h_B_out;

      CeedCallBackend(CeedOperatorGetBasisPointer(basis_out, eval_modes_out[i], identity, &h_B_out));
      CeedCallBackend(CeedBasisGetNumQuadratureComponents(basis_out, eval_modes_out[i], &q_comp));
      if (q_comp > 1) {
        if (i == 0 || eval_modes_out[i] != eval_modes_out_prev) d_out = 0;
        else h_B_out = &h_B_out[(++d_out) * elem_size_out * num_qpts_out];
      }
      eval_modes_out_prev = eval_modes_out[i];

      CeedCallHip(ceed, hipMemcpy(&asmb->d_B_out[i * elem_size_out * num_qpts_out], h_B_out, elem_size_out * num_qpts_out * sizeof(CeedScalar),
                                  hipMemcpyHostToDevice));
    }

    if (identity) {
      CeedCallBackend(CeedFree(&identity));
    }
  }
  CeedCallBackend(CeedFree(&eval_modes_out));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble matrix data for COO matrix of assembled operator.
// The sparsity pattern is set by CeedOperatorLinearAssembleSymbolic.
//
// Note that this (and other assembly routines) currently assume only one active input restriction/basis per operator
// (could have multiple basis eval modes).
// TODO: allow multiple active input restrictions/basis objects
//------------------------------------------------------------------------------
static int CeedSingleOperatorAssemble_Hip(CeedOperator op, CeedInt offset, CeedVector values) {
  Ceed                ceed;
  CeedSize            values_length = 0, assembled_qf_length = 0;
  CeedInt             use_ceedsize_idx = 0, num_elem_in, num_elem_out, elem_size_in, elem_size_out;
  CeedScalar         *values_array;
  const CeedScalar   *assembled_qf_array;
  CeedVector          assembled_qf   = NULL;
  CeedElemRestriction assembled_rstr = NULL, rstr_in, rstr_out;
  CeedRestrictionType rstr_type_in, rstr_type_out;
  const bool         *orients_in = NULL, *orients_out = NULL;
  const CeedInt8     *curl_orients_in = NULL, *curl_orients_out = NULL;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));

  // Assemble QFunction
  CeedCallBackend(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &assembled_rstr, CEED_REQUEST_IMMEDIATE));
  CeedCallBackend(CeedElemRestrictionDestroy(&assembled_rstr));
  CeedCallBackend(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_DEVICE, &assembled_qf_array));

  CeedCallBackend(CeedVectorGetLength(values, &values_length));
  CeedCallBackend(CeedVectorGetLength(assembled_qf, &assembled_qf_length));
  if ((values_length > INT_MAX) || (assembled_qf_length > INT_MAX)) use_ceedsize_idx = 1;

  // Setup
  if (!impl->asmb) CeedCallBackend(CeedSingleOperatorAssembleSetup_Hip(op, use_ceedsize_idx));
  CeedOperatorAssemble_Hip *asmb = impl->asmb;

  assert(asmb != NULL);

  // Assemble element operator
  CeedCallBackend(CeedVectorGetArray(values, CEED_MEM_DEVICE, &values_array));
  values_array += offset;

  CeedCallBackend(CeedOperatorGetActiveElemRestrictions(op, &rstr_in, &rstr_out));
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr_in, &num_elem_in));
  CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_in, &elem_size_in));

  CeedCallBackend(CeedElemRestrictionGetType(rstr_in, &rstr_type_in));
  if (rstr_type_in == CEED_RESTRICTION_ORIENTED) {
    CeedCallBackend(CeedElemRestrictionGetOrientations(rstr_in, CEED_MEM_DEVICE, &orients_in));
  } else if (rstr_type_in == CEED_RESTRICTION_CURL_ORIENTED) {
    CeedCallBackend(CeedElemRestrictionGetCurlOrientations(rstr_in, CEED_MEM_DEVICE, &curl_orients_in));
  }

  if (rstr_in != rstr_out) {
    CeedCallBackend(CeedElemRestrictionGetNumElements(rstr_out, &num_elem_out));
    CeedCheck(num_elem_in == num_elem_out, ceed, CEED_ERROR_UNSUPPORTED,
              "Active input and output operator restrictions must have the same number of elements");
    CeedCallBackend(CeedElemRestrictionGetElementSize(rstr_out, &elem_size_out));

    CeedCallBackend(CeedElemRestrictionGetType(rstr_out, &rstr_type_out));
    if (rstr_type_out == CEED_RESTRICTION_ORIENTED) {
      CeedCallBackend(CeedElemRestrictionGetOrientations(rstr_out, CEED_MEM_DEVICE, &orients_out));
    } else if (rstr_type_out == CEED_RESTRICTION_CURL_ORIENTED) {
      CeedCallBackend(CeedElemRestrictionGetCurlOrientations(rstr_out, CEED_MEM_DEVICE, &curl_orients_out));
    }
  } else {
    elem_size_out    = elem_size_in;
    orients_out      = orients_in;
    curl_orients_out = curl_orients_in;
  }

  // Compute B^T D B
  CeedInt shared_mem =
      ((curl_orients_in || curl_orients_out ? elem_size_in * elem_size_out : 0) + (curl_orients_in ? elem_size_in * asmb->block_size_y : 0)) *
      sizeof(CeedScalar);
  CeedInt grid   = CeedDivUpInt(num_elem_in, asmb->elems_per_block);
  void   *args[] = {(void *)&num_elem_in, &asmb->d_B_in,     &asmb->d_B_out,      &orients_in,  &curl_orients_in,
                    &orients_out,         &curl_orients_out, &assembled_qf_array, &values_array};

  CeedCallBackend(
      CeedRunKernelDimShared_Hip(ceed, asmb->LinearAssemble, grid, asmb->block_size_x, asmb->block_size_y, asmb->elems_per_block, shared_mem, args));

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArray(values, &values_array));
  CeedCallBackend(CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array));

  // Cleanup
  CeedCallBackend(CeedVectorDestroy(&assembled_qf));
  if (rstr_type_in == CEED_RESTRICTION_ORIENTED) {
    CeedCallBackend(CeedElemRestrictionRestoreOrientations(rstr_in, &orients_in));
  } else if (rstr_type_in == CEED_RESTRICTION_CURL_ORIENTED) {
    CeedCallBackend(CeedElemRestrictionRestoreCurlOrientations(rstr_in, &curl_orients_in));
  }
  if (rstr_in != rstr_out) {
    if (rstr_type_out == CEED_RESTRICTION_ORIENTED) {
      CeedCallBackend(CeedElemRestrictionRestoreOrientations(rstr_out, &orients_out));
    } else if (rstr_type_out == CEED_RESTRICTION_CURL_ORIENTED) {
      CeedCallBackend(CeedElemRestrictionRestoreCurlOrientations(rstr_out, &curl_orients_out));
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Assemble Linear QFunction AtPoints
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleQFunctionAtPoints_Hip(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  return CeedError(CeedOperatorReturnCeed(op), CEED_ERROR_BACKEND, "Backend does not implement CeedOperatorLinearAssembleQFunction");
}

//------------------------------------------------------------------------------
// Assemble Linear Diagonal AtPoints
//------------------------------------------------------------------------------
static int CeedOperatorLinearAssembleAddDiagonalAtPoints_Hip(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedInt             max_num_points, *num_points, num_elem, num_input_fields, num_output_fields;
  Ceed                ceed;
  CeedVector          active_e_vec_in, active_e_vec_out;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_input_fields, *op_output_fields;
  CeedOperator_Hip   *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedOperatorGetData(op, &impl));
  CeedCallBackend(CeedOperatorGetQFunction(op, &qf));
  CeedCallBackend(CeedOperatorGetNumElements(op, &num_elem));
  CeedCallBackend(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCallBackend(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  // Setup
  CeedCallBackend(CeedOperatorSetupAtPoints_Hip(op));
  num_points     = impl->num_points;
  max_num_points = impl->max_num_points;

  // Work vector
  CeedCallBackend(CeedGetWorkVector(ceed, impl->max_active_e_vec_len, &active_e_vec_in));
  CeedCallBackend(CeedGetWorkVector(ceed, impl->max_active_e_vec_len, &active_e_vec_out));
  {
    CeedSize length_in, length_out;

    CeedCallBackend(CeedVectorGetLength(active_e_vec_in, &length_in));
    CeedCallBackend(CeedVectorGetLength(active_e_vec_out, &length_out));
    // Need input e_vec to be longer
    if (length_in < length_out) {
      CeedVector temp = active_e_vec_in;

      active_e_vec_in  = active_e_vec_out;
      active_e_vec_out = temp;
    }
  }

  // Get point coordinates
  if (!impl->point_coords_elem) {
    CeedVector          point_coords = NULL;
    CeedElemRestriction rstr_points  = NULL;

    CeedCallBackend(CeedOperatorAtPointsGetPoints(op, &rstr_points, &point_coords));
    CeedCallBackend(CeedElemRestrictionCreateVector(rstr_points, NULL, &impl->point_coords_elem));
    CeedCallBackend(CeedElemRestrictionApply(rstr_points, CEED_NOTRANSPOSE, point_coords, impl->point_coords_elem, request));
    CeedCallBackend(CeedVectorDestroy(&point_coords));
    CeedCallBackend(CeedElemRestrictionDestroy(&rstr_points));
  }

  // Process inputs
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorInputRestrict_Hip(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl, request));
    CeedCallBackend(CeedOperatorInputBasisAtPoints_Hip(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, num_elem, num_points, true, impl));
  }

  // Clear active input Qvecs
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
    if (vec != CEED_VECTOR_ACTIVE) continue;
    CeedCallBackend(CeedVectorSetValue(impl->q_vecs_in[i], 0.0));
  }

  // Output pointers, as necessary
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedEvalMode eval_mode;

    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE) {
      CeedScalar *e_vec_array;

      CeedCallBackend(CeedVectorGetArrayWrite(impl->e_vecs_out[i], CEED_MEM_DEVICE, &e_vec_array));
      CeedCallBackend(CeedVectorSetArray(impl->q_vecs_out[i], CEED_MEM_DEVICE, CEED_USE_POINTER, e_vec_array));
    }
  }

  // Loop over active fields
  for (CeedInt i = 0; i < num_input_fields; i++) {
    bool                is_active_at_points = true;
    CeedInt             elem_size = 1, num_comp_active = 1, e_vec_size = 0;
    CeedRestrictionType rstr_type;
    CeedVector          l_vec;
    CeedElemRestriction elem_rstr;

    CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &l_vec));
    // -- Skip non-active input
    if (l_vec != CEED_VECTOR_ACTIVE) continue;

    // -- Get active restriction type
    CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &elem_rstr));
    CeedCallBackend(CeedElemRestrictionGetType(elem_rstr, &rstr_type));
    is_active_at_points = rstr_type == CEED_RESTRICTION_POINTS;
    if (!is_active_at_points) CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
    else elem_size = max_num_points;
    CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp_active));

    e_vec_size = elem_size * num_comp_active;
    for (CeedInt s = 0; s < e_vec_size; s++) {
      bool         is_active_input = false;
      CeedEvalMode eval_mode;
      CeedVector   l_vec, q_vec = impl->q_vecs_in[i];
      CeedBasis    basis;

      CeedCallBackend(CeedOperatorFieldGetVector(op_input_fields[i], &l_vec));
      // Skip non-active input
      is_active_input = l_vec == CEED_VECTOR_ACTIVE;
      if (!is_active_input) continue;

      // Update unit vector
      if (s == 0) CeedCallBackend(CeedVectorSetValue(active_e_vec_in, 0.0));
      else CeedCallBackend(CeedVectorSetValueStrided(active_e_vec_in, s - 1, e_vec_size, 0.0));
      CeedCallBackend(CeedVectorSetValueStrided(active_e_vec_in, s, e_vec_size, 1.0));

      // Basis action
      CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
      switch (eval_mode) {
        case CEED_EVAL_NONE: {
          const CeedScalar *e_vec_array;

          CeedCallBackend(CeedVectorGetArrayRead(active_e_vec_in, CEED_MEM_DEVICE, &e_vec_array));
          CeedCallBackend(CeedVectorSetArray(q_vec, CEED_MEM_DEVICE, CEED_USE_POINTER, (CeedScalar *)e_vec_array));
          break;
        }
        case CEED_EVAL_INTERP:
        case CEED_EVAL_GRAD:
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          CeedCallBackend(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
          CeedCallBackend(
              CeedBasisApplyAtPoints(basis, num_elem, num_points, CEED_NOTRANSPOSE, eval_mode, impl->point_coords_elem, active_e_vec_in, q_vec));
          break;
        case CEED_EVAL_WEIGHT:
          break;  // No action
      }

      // Q function
      CeedCallBackend(CeedQFunctionApply(qf, num_elem * max_num_points, impl->q_vecs_in, impl->q_vecs_out));

      // Output basis apply if needed
      for (CeedInt j = 0; j < num_output_fields; j++) {
        bool                is_active_output = false;
        CeedInt             elem_size        = 0;
        CeedRestrictionType rstr_type;
        CeedEvalMode        eval_mode;
        CeedVector          l_vec, e_vec = impl->e_vecs_out[j], q_vec = impl->q_vecs_out[j];
        CeedElemRestriction elem_rstr;
        CeedBasis           basis;

        CeedCallBackend(CeedOperatorFieldGetVector(op_output_fields[j], &l_vec));
        // ---- Skip non-active output
        is_active_output = l_vec == CEED_VECTOR_ACTIVE;
        if (!is_active_output) continue;
        if (!e_vec) e_vec = active_e_vec_out;

        // ---- Check if elem size matches
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[j], &elem_rstr));
        CeedCallBackend(CeedElemRestrictionGetType(elem_rstr, &rstr_type));
        if (is_active_at_points && rstr_type != CEED_RESTRICTION_POINTS) continue;
        if (rstr_type == CEED_RESTRICTION_POINTS) {
          CeedCallBackend(CeedElemRestrictionGetMaxPointsInElement(elem_rstr, &elem_size));
        } else {
          CeedCallBackend(CeedElemRestrictionGetElementSize(elem_rstr, &elem_size));
        }
        {
          CeedInt num_comp = 0;

          CeedCallBackend(CeedElemRestrictionGetNumComponents(elem_rstr, &num_comp));
          if (e_vec_size != num_comp * elem_size) continue;
        }

        // Basis action
        CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[j], &eval_mode));
        switch (eval_mode) {
          case CEED_EVAL_NONE: {
            CeedScalar *e_vec_array;

            CeedCallBackend(CeedVectorTakeArray(q_vec, CEED_MEM_DEVICE, &e_vec_array));
            CeedCallBackend(CeedVectorRestoreArray(e_vec, &e_vec_array));
            break;
          }
          case CEED_EVAL_INTERP:
          case CEED_EVAL_GRAD:
          case CEED_EVAL_DIV:
          case CEED_EVAL_CURL:
            CeedCallBackend(CeedOperatorFieldGetBasis(op_output_fields[j], &basis));
            CeedCallBackend(CeedBasisApplyAtPoints(basis, num_elem, num_points, CEED_TRANSPOSE, eval_mode, impl->point_coords_elem, q_vec, e_vec));
            break;
          // LCOV_EXCL_START
          case CEED_EVAL_WEIGHT: {
            return CeedError(CeedOperatorReturnCeed(op), CEED_ERROR_BACKEND, "CEED_EVAL_WEIGHT cannot be an output evaluation mode");
            // LCOV_EXCL_STOP
          }
        }

        // Mask output e-vec
        CeedCallBackend(CeedVectorPointwiseMult(e_vec, active_e_vec_in, e_vec));

        // Restrict
        CeedCallBackend(CeedOperatorFieldGetElemRestriction(op_output_fields[j], &elem_rstr));
        CeedCallBackend(CeedElemRestrictionApply(elem_rstr, CEED_TRANSPOSE, e_vec, assembled, request));

        // Reset q_vec for
        if (eval_mode == CEED_EVAL_NONE) {
          CeedScalar *e_vec_array;

          CeedCallBackend(CeedVectorGetArrayWrite(e_vec, CEED_MEM_DEVICE, &e_vec_array));
          CeedCallBackend(CeedVectorSetArray(q_vec, CEED_MEM_DEVICE, CEED_USE_POINTER, e_vec_array));
        }
      }

      // Reset vec
      if (s == e_vec_size - 1 && i != num_input_fields - 1) CeedCallBackend(CeedVectorSetValue(q_vec, 0.0));
    }
  }

  // Restore CEED_EVAL_NONE
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedEvalMode eval_mode;

    // Get eval_mode
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));

    // Restore evec
    CeedCallBackend(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
    if (eval_mode == CEED_EVAL_NONE) {
      CeedScalar *e_vec_array;

      CeedCallBackend(CeedVectorTakeArray(impl->q_vecs_in[i], CEED_MEM_DEVICE, &e_vec_array));
      CeedCallBackend(CeedVectorRestoreArray(impl->e_vecs_in[i], &e_vec_array));
    }
  }

  // Restore input arrays
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCallBackend(CeedOperatorInputRestore_Hip(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl));
  }

  // Restore work vector
  CeedCallBackend(CeedRestoreWorkVector(ceed, &active_e_vec_in));
  CeedCallBackend(CeedRestoreWorkVector(ceed, &active_e_vec_out));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator
//------------------------------------------------------------------------------
int CeedOperatorCreate_Hip(CeedOperator op) {
  Ceed              ceed;
  CeedOperator_Hip *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonal_Hip));
  CeedCallBackend(
      CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddPointBlockDiagonal", CeedOperatorLinearAssembleAddPointBlockDiagonal_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleSingle", CeedSingleOperatorAssemble_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create operator AtPoints
//------------------------------------------------------------------------------
int CeedOperatorCreateAtPoints_Hip(CeedOperator op) {
  Ceed              ceed;
  CeedOperator_Hip *impl;

  CeedCallBackend(CeedOperatorGetCeed(op, &ceed));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedOperatorSetData(op, impl));

  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunctionAtPoints_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonalAtPoints_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAddAtPoints_Hip));
  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Hip));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
