// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed-impl.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/// @file
/// Implementation of CeedOperator preconditioning interfaces

/// ----------------------------------------------------------------------------
/// CeedOperator Library Internal Preconditioning Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedOperatorDeveloper
/// @{

/**
  @brief Duplicate a CeedOperator with a reference Ceed to fallback for advanced
         CeedOperator functionality

  @param op  CeedOperator to create fallback for

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedOperatorCreateFallback(CeedOperator op) {
  int ierr;

  // Fallback Ceed
  const char *resource, *fallback_resource;
  ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
  ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
  CeedChk(ierr);
  if (!strcmp(resource, fallback_resource))
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend %s cannot create an operator"
                     "fallback to resource %s", resource, fallback_resource);
  // LCOV_EXCL_STOP

  // Fallback Ceed
  Ceed ceed_ref;
  if (!op->ceed->op_fallback_ceed) {
    ierr = CeedInit(fallback_resource, &ceed_ref); CeedChk(ierr);
    ceed_ref->op_fallback_parent = op->ceed;
    ceed_ref->Error = op->ceed->Error;
    op->ceed->op_fallback_ceed = ceed_ref;
  }
  ceed_ref = op->ceed->op_fallback_ceed;

  // Clone Op
  CeedOperator op_ref;
  ierr = CeedCalloc(1, &op_ref); CeedChk(ierr);
  memcpy(op_ref, op, sizeof(*op_ref));
  op_ref->data = NULL;
  op_ref->is_interface_setup = false;
  op_ref->is_backend_setup = false;
  op_ref->ceed = ceed_ref;
  ierr = ceed_ref->OperatorCreate(op_ref); CeedChk(ierr);
  ierr = CeedQFunctionAssemblyDataReferenceCopy(op->qf_assembled,
         &op_ref->qf_assembled); CeedChk(ierr);
  op->op_fallback = op_ref;

  // Clone QF
  CeedQFunction qf_ref;
  ierr = CeedCalloc(1, &qf_ref); CeedChk(ierr);
  memcpy(qf_ref, (op->qf), sizeof(*qf_ref));
  qf_ref->data = NULL;
  qf_ref->ceed = ceed_ref;
  ierr = ceed_ref->QFunctionCreate(qf_ref); CeedChk(ierr);
  op_ref->qf = qf_ref;
  op->qf_fallback = qf_ref;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Select correct basis matrix pointer based on CeedEvalMode

  @param[in] eval_mode   Current basis evaluation mode
  @param[in] identity    Pointer to identity matrix
  @param[in] interp      Pointer to interpolation matrix
  @param[in] grad        Pointer to gradient matrix
  @param[out] basis_ptr  Basis pointer to set

  @ref Developer
**/
static inline void CeedOperatorGetBasisPointer(CeedEvalMode eval_mode,
    const CeedScalar *identity, const CeedScalar *interp,
    const CeedScalar *grad, const CeedScalar **basis_ptr) {
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

/**
  @brief Create point block restriction for active operator field

  @param[in] rstr              Original CeedElemRestriction for active field
  @param[out] pointblock_rstr  Address of the variable where the newly created
                                 CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCreateActivePointBlockRestriction(
  CeedElemRestriction rstr,
  CeedElemRestriction *pointblock_rstr) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(rstr, &ceed); CeedChk(ierr);
  const CeedInt *offsets;
  ierr = CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets);
  CeedChk(ierr);

  // Expand offsets
  CeedInt num_elem, num_comp, elem_size, comp_stride, max = 1,
                                                      *pointblock_offsets;
  ierr = CeedElemRestrictionGetNumElements(rstr, &num_elem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr, &num_comp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr, &elem_size); CeedChk(ierr);
  ierr = CeedElemRestrictionGetCompStride(rstr, &comp_stride); CeedChk(ierr);
  CeedInt shift = num_comp;
  if (comp_stride != 1)
    shift *= num_comp;
  ierr = CeedCalloc(num_elem*elem_size, &pointblock_offsets);
  CeedChk(ierr);
  for (CeedInt i = 0; i < num_elem*elem_size; i++) {
    pointblock_offsets[i] = offsets[i]*shift;
    if (pointblock_offsets[i] > max)
      max = pointblock_offsets[i];
  }

  // Create new restriction
  ierr = CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp*num_comp,
                                   1, max + num_comp*num_comp, CEED_MEM_HOST,
                                   CEED_OWN_POINTER, pointblock_offsets, pointblock_rstr);
  CeedChk(ierr);

  // Cleanup
  ierr = CeedElemRestrictionRestoreOffsets(rstr, &offsets); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Core logic for assembling operator diagonal or point block diagonal

  @param[in] op             CeedOperator to assemble point block diagonal
  @param[in] request        Address of CeedRequest for non-blocking completion, else
                              CEED_REQUEST_IMMEDIATE
  @param[in] is_pointblock  Boolean flag to assemble diagonal or point block diagonal
  @param[out] assembled     CeedVector to store assembled diagonal

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static inline int CeedSingleOperatorAssembleAddDiagonal(CeedOperator op,
    CeedRequest *request, const bool is_pointblock, CeedVector assembled) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

  // Assemble QFunction
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt num_input_fields, num_output_fields;
  ierr= CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields);
  CeedChk(ierr);
  CeedVector assembled_qf;
  CeedElemRestriction rstr;
  ierr = CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op,  &assembled_qf,
         &rstr, request); CeedChk(ierr);
  CeedInt layout[3];
  ierr = CeedElemRestrictionGetELayout(rstr, &layout); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr); CeedChk(ierr);
  CeedScalar max_norm = 0;
  ierr = CeedVectorNorm(assembled_qf, CEED_NORM_MAX, &max_norm); CeedChk(ierr);

  // Determine active input basis
  CeedOperatorField *op_fields;
  CeedQFunctionField *qf_fields;
  ierr = CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL); CeedChk(ierr);
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL); CeedChk(ierr);
  CeedInt num_eval_mode_in = 0, num_comp, dim = 1;
  CeedEvalMode *eval_mode_in = NULL;
  CeedBasis basis_in = NULL;
  CeedElemRestriction rstr_in = NULL;
  for (CeedInt i=0; i<num_input_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(op_fields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis_in); CeedChk(ierr);
      ierr = CeedBasisGetNumComponents(basis_in, &num_comp); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis_in, &dim); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr); CeedChk(ierr);
      if (rstr_in && rstr_in != rstr)
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "Multi-field non-composite operator diagonal assembly not supported");
      // LCOV_EXCL_STOP
      rstr_in = rstr;
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode); CeedChk(ierr);
      switch (eval_mode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(num_eval_mode_in + 1, &eval_mode_in); CeedChk(ierr);
        eval_mode_in[num_eval_mode_in] = eval_mode;
        num_eval_mode_in += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(num_eval_mode_in + dim, &eval_mode_in); CeedChk(ierr);
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
  ierr = CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields); CeedChk(ierr);
  ierr = CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields); CeedChk(ierr);
  CeedInt num_eval_mode_out = 0;
  CeedEvalMode *eval_mode_out = NULL;
  CeedBasis basis_out = NULL;
  CeedElemRestriction rstr_out = NULL;
  for (CeedInt i=0; i<num_output_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(op_fields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedElemRestriction rstr;
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis_out); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr); CeedChk(ierr);
      if (rstr_out && rstr_out != rstr)
        // LCOV_EXCL_START
        return CeedError(ceed, CEED_ERROR_BACKEND,
                         "Multi-field non-composite operator diagonal assembly not supported");
      // LCOV_EXCL_STOP
      rstr_out = rstr;
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode); CeedChk(ierr);
      switch (eval_mode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(num_eval_mode_out + 1, &eval_mode_out); CeedChk(ierr);
        eval_mode_out[num_eval_mode_out] = eval_mode;
        num_eval_mode_out += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(num_eval_mode_out + dim, &eval_mode_out); CeedChk(ierr);
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

  // Assemble point block diagonal restriction, if needed
  CeedElemRestriction diag_rstr = rstr_out;
  if (is_pointblock) {
    ierr = CeedOperatorCreateActivePointBlockRestriction(rstr_out, &diag_rstr);
    CeedChk(ierr);
  }

  // Create diagonal vector
  CeedVector elem_diag;
  ierr = CeedElemRestrictionCreateVector(diag_rstr, NULL, &elem_diag);
  CeedChk(ierr);

  // Assemble element operator diagonals
  CeedScalar *elem_diag_array;
  const CeedScalar *assembled_qf_array;
  ierr = CeedVectorSetValue(elem_diag, 0.0); CeedChk(ierr);
  ierr = CeedVectorGetArray(elem_diag, CEED_MEM_HOST, &elem_diag_array);
  CeedChk(ierr);
  ierr = CeedVectorGetArrayRead(assembled_qf, CEED_MEM_HOST, &assembled_qf_array);
  CeedChk(ierr);
  CeedInt num_elem, num_nodes, num_qpts;
  ierr = CeedElemRestrictionGetNumElements(diag_rstr, &num_elem); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basis_in, &num_nodes); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts); CeedChk(ierr);
  // Basis matrices
  const CeedScalar *interp_in, *interp_out, *grad_in, *grad_out;
  CeedScalar *identity = NULL;
  bool evalNone = false;
  for (CeedInt i=0; i<num_eval_mode_in; i++)
    evalNone = evalNone || (eval_mode_in[i] == CEED_EVAL_NONE);
  for (CeedInt i=0; i<num_eval_mode_out; i++)
    evalNone = evalNone || (eval_mode_out[i] == CEED_EVAL_NONE);
  if (evalNone) {
    ierr = CeedCalloc(num_qpts*num_nodes, &identity); CeedChk(ierr);
    for (CeedInt i=0; i<(num_nodes<num_qpts?num_nodes:num_qpts); i++)
      identity[i*num_nodes+i] = 1.0;
  }
  ierr = CeedBasisGetInterp(basis_in, &interp_in); CeedChk(ierr);
  ierr = CeedBasisGetInterp(basis_out, &interp_out); CeedChk(ierr);
  ierr = CeedBasisGetGrad(basis_in, &grad_in); CeedChk(ierr);
  ierr = CeedBasisGetGrad(basis_out, &grad_out); CeedChk(ierr);
  // Compute the diagonal of B^T D B
  // Each element
  const CeedScalar qf_value_bound = max_norm*100*CEED_EPSILON;
  for (CeedInt e=0; e<num_elem; e++) {
    CeedInt d_out = -1;
    // Each basis eval mode pair
    for (CeedInt e_out=0; e_out<num_eval_mode_out; e_out++) {
      const CeedScalar *bt = NULL;
      if (eval_mode_out[e_out] == CEED_EVAL_GRAD)
        d_out += 1;
      CeedOperatorGetBasisPointer(eval_mode_out[e_out], identity, interp_out,
                                  &grad_out[d_out*num_qpts*num_nodes], &bt);
      CeedInt d_in = -1;
      for (CeedInt e_in=0; e_in<num_eval_mode_in; e_in++) {
        const CeedScalar *b = NULL;
        if (eval_mode_in[e_in] == CEED_EVAL_GRAD)
          d_in += 1;
        CeedOperatorGetBasisPointer(eval_mode_in[e_in], identity, interp_in,
                                    &grad_in[d_in*num_qpts*num_nodes], &b);
        // Each component
        for (CeedInt c_out=0; c_out<num_comp; c_out++)
          // Each qpoint/node pair
          for (CeedInt q=0; q<num_qpts; q++)
            if (is_pointblock) {
              // Point Block Diagonal
              for (CeedInt c_in=0; c_in<num_comp; c_in++) {
                const CeedScalar qf_value =
                  assembled_qf_array[q*layout[0] + (((e_in*num_comp+c_in)*
                                                     num_eval_mode_out+e_out)*num_comp+c_out)*layout[1] + e*layout[2]];
                if (fabs(qf_value) > qf_value_bound)
                  for (CeedInt n=0; n<num_nodes; n++)
                    elem_diag_array[((e*num_comp+c_out)*num_comp+c_in)*num_nodes+n] +=
                      bt[q*num_nodes+n] * qf_value * b[q*num_nodes+n];
              }
            } else {
              // Diagonal Only
              const CeedScalar qf_value =
                assembled_qf_array[q*layout[0] + (((e_in*num_comp+c_out)*
                                                   num_eval_mode_out+e_out)*num_comp+c_out)*layout[1] + e*layout[2]];
              if (fabs(qf_value) > qf_value_bound)
                for (CeedInt n=0; n<num_nodes; n++)
                  elem_diag_array[(e*num_comp+c_out)*num_nodes+n] +=
                    bt[q*num_nodes+n] * qf_value * b[q*num_nodes+n];
            }
      }
    }
  }
  ierr = CeedVectorRestoreArray(elem_diag, &elem_diag_array); CeedChk(ierr);
  ierr = CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array);
  CeedChk(ierr);

  // Assemble local operator diagonal
  ierr = CeedElemRestrictionApply(diag_rstr, CEED_TRANSPOSE, elem_diag,
                                  assembled, request); CeedChk(ierr);

  // Cleanup
  if (is_pointblock) {
    ierr = CeedElemRestrictionDestroy(&diag_rstr); CeedChk(ierr);
  }
  ierr = CeedVectorDestroy(&assembled_qf); CeedChk(ierr);
  ierr = CeedVectorDestroy(&elem_diag); CeedChk(ierr);
  ierr = CeedFree(&eval_mode_in); CeedChk(ierr);
  ierr = CeedFree(&eval_mode_out); CeedChk(ierr);
  ierr = CeedFree(&identity); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Core logic for assembling composite operator diagonal

  @param[in] op             CeedOperator to assemble point block diagonal
  @param[in] request        Address of CeedRequest for non-blocking completion, else
                            CEED_REQUEST_IMMEDIATE
  @param[in] is_pointblock  Boolean flag to assemble diagonal or point block diagonal
  @param[out] assembled     CeedVector to store assembled diagonal

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static inline int CeedCompositeOperatorLinearAssembleAddDiagonal(
  CeedOperator op, CeedRequest *request, const bool is_pointblock,
  CeedVector assembled) {
  int ierr;
  CeedInt num_sub;
  CeedOperator *suboperators;
  ierr = CeedOperatorGetNumSub(op, &num_sub); CeedChk(ierr);
  ierr = CeedOperatorGetSubList(op, &suboperators); CeedChk(ierr);
  for (CeedInt i = 0; i < num_sub; i++) {
    ierr = CeedSingleOperatorAssembleAddDiagonal(suboperators[i], request,
           is_pointblock, assembled); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build nonzero pattern for non-composite operator

  Users should generally use CeedOperatorLinearAssembleSymbolic()

  @param[in] op      CeedOperator to assemble nonzero pattern
  @param[in] offset  Offset for number of entries
  @param[out] rows   Row number for each entry
  @param[out] cols   Column number for each entry

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedSingleOperatorAssembleSymbolic(CeedOperator op, CeedInt offset,
    CeedInt *rows, CeedInt *cols) {
  int ierr;
  Ceed ceed = op->ceed;
  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "Composite operator not supported");
  // LCOV_EXCL_STOP

  CeedSize num_nodes;
  ierr = CeedOperatorGetActiveVectorLengths(op, &num_nodes, NULL); CeedChk(ierr);
  CeedElemRestriction rstr_in;
  ierr = CeedOperatorGetActiveElemRestriction(op, &rstr_in); CeedChk(ierr);
  CeedInt num_elem, elem_size, num_comp;
  ierr = CeedElemRestrictionGetNumElements(rstr_in, &num_elem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr_in, &elem_size); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr_in, &num_comp); CeedChk(ierr);
  CeedInt layout_er[3];
  ierr = CeedElemRestrictionGetELayout(rstr_in, &layout_er); CeedChk(ierr);

  CeedInt local_num_entries = elem_size*num_comp * elem_size*num_comp * num_elem;

  // Determine elem_dof relation
  CeedVector index_vec;
  ierr = CeedVectorCreate(ceed, num_nodes, &index_vec); CeedChk(ierr);
  CeedScalar *array;
  ierr = CeedVectorGetArrayWrite(index_vec, CEED_MEM_HOST, &array); CeedChk(ierr);
  for (CeedInt i = 0; i < num_nodes; ++i) {
    array[i] = i;
  }
  ierr = CeedVectorRestoreArray(index_vec, &array); CeedChk(ierr);
  CeedVector elem_dof;
  ierr = CeedVectorCreate(ceed, num_elem * elem_size * num_comp, &elem_dof);
  CeedChk(ierr);
  ierr = CeedVectorSetValue(elem_dof, 0.0); CeedChk(ierr);
  CeedElemRestrictionApply(rstr_in, CEED_NOTRANSPOSE, index_vec,
                           elem_dof, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  const CeedScalar *elem_dof_a;
  ierr = CeedVectorGetArrayRead(elem_dof, CEED_MEM_HOST, &elem_dof_a);
  CeedChk(ierr);
  ierr = CeedVectorDestroy(&index_vec); CeedChk(ierr);

  // Determine i, j locations for element matrices
  CeedInt count = 0;
  for (int e = 0; e < num_elem; ++e) {
    for (int comp_in = 0; comp_in < num_comp; ++comp_in) {
      for (int comp_out = 0; comp_out < num_comp; ++comp_out) {
        for (int i = 0; i < elem_size; ++i) {
          for (int j = 0; j < elem_size; ++j) {
            const CeedInt elem_dof_index_row = (i)*layout_er[0] +
                                               (comp_out)*layout_er[1] + e*layout_er[2];
            const CeedInt elem_dof_index_col = (j)*layout_er[0] +
                                               (comp_in)*layout_er[1] + e*layout_er[2];

            const CeedInt row = elem_dof_a[elem_dof_index_row];
            const CeedInt col = elem_dof_a[elem_dof_index_col];

            rows[offset + count] = row;
            cols[offset + count] = col;
            count++;
          }
        }
      }
    }
  }
  if (count != local_num_entries)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MAJOR, "Error computing assembled entries");
  // LCOV_EXCL_STOP
  ierr = CeedVectorRestoreArrayRead(elem_dof, &elem_dof_a); CeedChk(ierr);
  ierr = CeedVectorDestroy(&elem_dof); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble nonzero entries for non-composite operator

  Users should generally use CeedOperatorLinearAssemble()

  @param[in] op       CeedOperator to assemble
  @param[out] offset  Offest for number of entries
  @param[out] values  Values to assemble into matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedSingleOperatorAssemble(CeedOperator op, CeedInt offset,
                                      CeedVector values) {
  int ierr;
  Ceed ceed = op->ceed;
  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "Composite operator not supported");
  // LCOV_EXCL_STOP

  // Assemble QFunction
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedVector assembled_qf;
  CeedElemRestriction rstr_q;
  ierr = CeedOperatorLinearAssembleQFunctionBuildOrUpdate(
           op, &assembled_qf, &rstr_q, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

  CeedSize qf_length;
  ierr = CeedVectorGetLength(assembled_qf, &qf_length); CeedChk(ierr);

  CeedInt num_input_fields, num_output_fields;
  CeedOperatorField *input_fields;
  CeedOperatorField *output_fields;
  ierr = CeedOperatorGetFields(op, &num_input_fields, &input_fields,
                               &num_output_fields, &output_fields); CeedChk(ierr);

  // Determine active input basis
  CeedQFunctionField *qf_fields;
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL); CeedChk(ierr);
  CeedInt num_eval_mode_in = 0, dim = 1;
  CeedEvalMode *eval_mode_in = NULL;
  CeedBasis basis_in = NULL;
  CeedElemRestriction rstr_in = NULL;
  for (CeedInt i=0; i<num_input_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(input_fields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(input_fields[i], &basis_in);
      CeedChk(ierr);
      ierr = CeedBasisGetDimension(basis_in, &dim); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(input_fields[i], &rstr_in);
      CeedChk(ierr);
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode);
      CeedChk(ierr);
      switch (eval_mode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(num_eval_mode_in + 1, &eval_mode_in); CeedChk(ierr);
        eval_mode_in[num_eval_mode_in] = eval_mode;
        num_eval_mode_in += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(num_eval_mode_in + dim, &eval_mode_in); CeedChk(ierr);
        for (CeedInt d=0; d<dim; d++) {
          eval_mode_in[num_eval_mode_in+d] = eval_mode;
        }
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
  ierr = CeedQFunctionGetFields(qf, NULL, NULL, NULL, &qf_fields); CeedChk(ierr);
  CeedInt num_eval_mode_out = 0;
  CeedEvalMode *eval_mode_out = NULL;
  CeedBasis basis_out = NULL;
  CeedElemRestriction rstr_out = NULL;
  for (CeedInt i=0; i<num_output_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(output_fields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(output_fields[i], &basis_out);
      CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(output_fields[i], &rstr_out);
      CeedChk(ierr);
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode);
      CeedChk(ierr);
      switch (eval_mode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(num_eval_mode_out + 1, &eval_mode_out); CeedChk(ierr);
        eval_mode_out[num_eval_mode_out] = eval_mode;
        num_eval_mode_out += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(num_eval_mode_out + dim, &eval_mode_out); CeedChk(ierr);
        for (CeedInt d=0; d<dim; d++) {
          eval_mode_out[num_eval_mode_out+d] = eval_mode;
        }
        num_eval_mode_out += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }
  }

  if (num_eval_mode_in == 0 || num_eval_mode_out == 0)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "Cannot assemble operator with out inputs/outputs");
  // LCOV_EXCL_STOP

  CeedInt num_elem, elem_size, num_qpts, num_comp;
  ierr = CeedElemRestrictionGetNumElements(rstr_in, &num_elem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr_in, &elem_size); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr_in, &num_comp); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts); CeedChk(ierr);

  CeedInt local_num_entries = elem_size*num_comp * elem_size*num_comp * num_elem;

  // loop over elements and put in data structure
  const CeedScalar *interp_in, *grad_in;
  ierr = CeedBasisGetInterp(basis_in, &interp_in); CeedChk(ierr);
  ierr = CeedBasisGetGrad(basis_in, &grad_in); CeedChk(ierr);

  const CeedScalar *assembled_qf_array;
  ierr = CeedVectorGetArrayRead(assembled_qf, CEED_MEM_HOST, &assembled_qf_array);
  CeedChk(ierr);

  CeedInt layout_qf[3];
  ierr = CeedElemRestrictionGetELayout(rstr_q, &layout_qf); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_q); CeedChk(ierr);

  // we store B_mat_in, B_mat_out, BTD, elem_mat in row-major order
  CeedScalar B_mat_in[(num_qpts * num_eval_mode_in) * elem_size];
  CeedScalar B_mat_out[(num_qpts * num_eval_mode_out) * elem_size];
  CeedScalar D_mat[num_eval_mode_out * num_eval_mode_in *
                                     num_qpts]; // logically 3-tensor
  CeedScalar BTD[elem_size * num_qpts*num_eval_mode_in];
  CeedScalar elem_mat[elem_size * elem_size];
  int count = 0;
  CeedScalar *vals;
  ierr = CeedVectorGetArrayWrite(values, CEED_MEM_HOST, &vals); CeedChk(ierr);
  for (int e = 0; e < num_elem; ++e) {
    for (int comp_in = 0; comp_in < num_comp; ++comp_in) {
      for (int comp_out = 0; comp_out < num_comp; ++comp_out) {
        for (int ell = 0; ell < (num_qpts * num_eval_mode_in) * elem_size; ++ell) {
          B_mat_in[ell] = 0.0;
        }
        for (int ell = 0; ell < (num_qpts * num_eval_mode_out) * elem_size; ++ell) {
          B_mat_out[ell] = 0.0;
        }
        // Store block-diagonal D matrix as collection of small dense blocks
        for (int ell = 0; ell < num_eval_mode_in*num_eval_mode_out*num_qpts; ++ell) {
          D_mat[ell] = 0.0;
        }
        // form element matrix itself (for each block component)
        for (int ell = 0; ell < elem_size*elem_size; ++ell) {
          elem_mat[ell] = 0.0;
        }
        for (int q = 0; q < num_qpts; ++q) {
          for (int n = 0; n < elem_size; ++n) {
            CeedInt d_in = -1;
            for (int e_in = 0; e_in < num_eval_mode_in; ++e_in) {
              const int qq = num_eval_mode_in*q;
              if (eval_mode_in[e_in] == CEED_EVAL_INTERP) {
                B_mat_in[(qq+e_in)*elem_size + n] += interp_in[q * elem_size + n];
              } else if (eval_mode_in[e_in] == CEED_EVAL_GRAD) {
                d_in += 1;
                B_mat_in[(qq+e_in)*elem_size + n] +=
                  grad_in[(d_in*num_qpts+q) * elem_size + n];
              } else {
                // LCOV_EXCL_START
                return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Not implemented!");
                // LCOV_EXCL_STOP
              }
            }
            CeedInt d_out = -1;
            for (int e_out = 0; e_out < num_eval_mode_out; ++e_out) {
              const int qq = num_eval_mode_out*q;
              if (eval_mode_out[e_out] == CEED_EVAL_INTERP) {
                B_mat_out[(qq+e_out)*elem_size + n] += interp_in[q * elem_size + n];
              } else if (eval_mode_out[e_out] == CEED_EVAL_GRAD) {
                d_out += 1;
                B_mat_out[(qq+e_out)*elem_size + n] +=
                  grad_in[(d_out*num_qpts+q) * elem_size + n];
              } else {
                // LCOV_EXCL_START
                return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Not implemented!");
                // LCOV_EXCL_STOP
              }
            }
          }
          for (int ei = 0; ei < num_eval_mode_out; ++ei) {
            for (int ej = 0; ej < num_eval_mode_in; ++ej) {
              const int eval_mode_index = ((ei*num_comp+comp_in)*num_eval_mode_in+ej)*num_comp
                                          +comp_out;
              const int index = q*layout_qf[0] + eval_mode_index*layout_qf[1] +
                                e*layout_qf[2];
              D_mat[(ei*num_eval_mode_in+ej)*num_qpts + q] += assembled_qf_array[index];
            }
          }
        }
        // Compute B^T*D
        for (int ell = 0; ell < elem_size*num_qpts*num_eval_mode_in; ++ell) {
          BTD[ell] = 0.0;
        }
        for (int j = 0; j<elem_size; ++j) {
          for (int q = 0; q<num_qpts; ++q) {
            int qq = num_eval_mode_out*q;
            for (int ei = 0; ei < num_eval_mode_in; ++ei) {
              for (int ej = 0; ej < num_eval_mode_out; ++ej) {
                BTD[j*(num_qpts*num_eval_mode_in) + (qq+ei)] +=
                  B_mat_out[(qq+ej)*elem_size + j] * D_mat[(ei*num_eval_mode_in+ej)*num_qpts + q];
              }
            }
          }
        }

        ierr = CeedMatrixMultiply(ceed, BTD, B_mat_in, elem_mat, elem_size,
                                  elem_size, num_qpts*num_eval_mode_in); CeedChk(ierr);

        // put element matrix in coordinate data structure
        for (int i = 0; i < elem_size; ++i) {
          for (int j = 0; j < elem_size; ++j) {
            vals[offset + count] = elem_mat[i*elem_size + j];
            count++;
          }
        }
      }
    }
  }
  if (count != local_num_entries)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MAJOR, "Error computing entries");
  // LCOV_EXCL_STOP
  ierr = CeedVectorRestoreArray(values, &vals); CeedChk(ierr);

  ierr = CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array);
  CeedChk(ierr);
  ierr = CeedVectorDestroy(&assembled_qf); CeedChk(ierr);
  ierr = CeedFree(&eval_mode_in); CeedChk(ierr);
  ierr = CeedFree(&eval_mode_out); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Count number of entries for assembled CeedOperator

  @param[in] op            CeedOperator to assemble
  @param[out] num_entries  Number of entries in assembled representation

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedSingleOperatorAssemblyCountEntries(CeedOperator op,
    CeedInt *num_entries) {
  int ierr;
  CeedElemRestriction rstr;
  CeedInt num_elem, elem_size, num_comp;

  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED,
                     "Composite operator not supported");
  // LCOV_EXCL_STOP
  ierr = CeedOperatorGetActiveElemRestriction(op, &rstr); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(rstr, &num_elem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr, &elem_size); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr, &num_comp); CeedChk(ierr);
  *num_entries = elem_size*num_comp * elem_size*num_comp * num_elem;

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Common code for creating a multigrid coarse operator and level
           transfer operators for a CeedOperator

  @param[in] op_fine       Fine grid operator
  @param[in] p_mult_fine   L-vector multiplicity in parallel gather/scatter
  @param[in] rstr_coarse   Coarse grid restriction
  @param[in] basis_coarse  Coarse grid active vector basis
  @param[in] basis_c_to_f  Basis for coarse to fine interpolation
  @param[out] op_coarse    Coarse grid operator
  @param[out] op_prolong   Coarse to fine operator
  @param[out] op_restrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedSingleOperatorMultigridLevel(CeedOperator op_fine,
    CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
    CeedBasis basis_c_to_f, CeedOperator *op_coarse, CeedOperator *op_prolong,
    CeedOperator *op_restrict) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op_fine, &ceed); CeedChk(ierr);

  // Check for composite operator
  bool is_composite;
  ierr = CeedOperatorIsComposite(op_fine, &is_composite); CeedChk(ierr);
  if (is_composite)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "Automatic multigrid setup for composite operators not supported");
  // LCOV_EXCL_STOP

  // Coarse Grid
  ierr = CeedOperatorCreate(ceed, op_fine->qf, op_fine->dqf, op_fine->dqfT,
                            op_coarse); CeedChk(ierr);
  CeedElemRestriction rstr_fine = NULL;
  // -- Clone input fields
  for (int i = 0; i < op_fine->qf->num_input_fields; i++) {
    if (op_fine->input_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      rstr_fine = op_fine->input_fields[i]->elem_restr;
      ierr = CeedOperatorSetField(*op_coarse, op_fine->input_fields[i]->field_name,
                                  rstr_coarse, basis_coarse, CEED_VECTOR_ACTIVE);
      CeedChk(ierr);
    } else {
      ierr = CeedOperatorSetField(*op_coarse, op_fine->input_fields[i]->field_name,
                                  op_fine->input_fields[i]->elem_restr,
                                  op_fine->input_fields[i]->basis,
                                  op_fine->input_fields[i]->vec); CeedChk(ierr);
    }
  }
  // -- Clone output fields
  for (int i = 0; i < op_fine->qf->num_output_fields; i++) {
    if (op_fine->output_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorSetField(*op_coarse, op_fine->output_fields[i]->field_name,
                                  rstr_coarse, basis_coarse, CEED_VECTOR_ACTIVE);
      CeedChk(ierr);
    } else {
      ierr = CeedOperatorSetField(*op_coarse, op_fine->output_fields[i]->field_name,
                                  op_fine->output_fields[i]->elem_restr,
                                  op_fine->output_fields[i]->basis,
                                  op_fine->output_fields[i]->vec); CeedChk(ierr);
    }
  }
  // -- Clone QFunctionAssemblyData
  ierr = CeedQFunctionAssemblyDataReferenceCopy(op_fine->qf_assembled,
         &(*op_coarse)->qf_assembled); CeedChk(ierr);

  // Multiplicity vector
  CeedVector mult_vec, mult_e_vec;
  ierr = CeedElemRestrictionCreateVector(rstr_fine, &mult_vec, &mult_e_vec);
  CeedChk(ierr);
  ierr = CeedVectorSetValue(mult_e_vec, 0.0); CeedChk(ierr);
  ierr = CeedElemRestrictionApply(rstr_fine, CEED_NOTRANSPOSE, p_mult_fine,
                                  mult_e_vec, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  ierr = CeedVectorSetValue(mult_vec, 0.0); CeedChk(ierr);
  ierr = CeedElemRestrictionApply(rstr_fine, CEED_TRANSPOSE, mult_e_vec, mult_vec,
                                  CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  ierr = CeedVectorDestroy(&mult_e_vec); CeedChk(ierr);
  ierr = CeedVectorReciprocal(mult_vec); CeedChk(ierr);

  // Restriction
  CeedInt num_comp;
  ierr = CeedBasisGetNumComponents(basis_coarse, &num_comp); CeedChk(ierr);
  CeedQFunction qf_restrict;
  ierr = CeedQFunctionCreateInteriorByName(ceed, "Scale", &qf_restrict);
  CeedChk(ierr);
  CeedInt *num_comp_r_data;
  ierr = CeedCalloc(1, &num_comp_r_data); CeedChk(ierr);
  num_comp_r_data[0] = num_comp;
  CeedQFunctionContext ctx_r;
  ierr = CeedQFunctionContextCreate(ceed, &ctx_r); CeedChk(ierr);
  ierr = CeedQFunctionContextSetData(ctx_r, CEED_MEM_HOST, CEED_OWN_POINTER,
                                     sizeof(*num_comp_r_data), num_comp_r_data);
  CeedChk(ierr);
  ierr = CeedQFunctionSetContext(qf_restrict, ctx_r); CeedChk(ierr);
  ierr = CeedQFunctionContextDestroy(&ctx_r); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf_restrict, "input", num_comp, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf_restrict, "scale", num_comp, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf_restrict, "output", num_comp,
                                CEED_EVAL_INTERP); CeedChk(ierr);
  ierr = CeedQFunctionSetUserFlopsEstimate(qf_restrict, num_comp); CeedChk(ierr);

  ierr = CeedOperatorCreate(ceed, qf_restrict, CEED_QFUNCTION_NONE,
                            CEED_QFUNCTION_NONE, op_restrict);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*op_restrict, "input", rstr_fine,
                              CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*op_restrict, "scale", rstr_fine,
                              CEED_BASIS_COLLOCATED, mult_vec);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*op_restrict, "output", rstr_coarse, basis_c_to_f,
                              CEED_VECTOR_ACTIVE); CeedChk(ierr);

  // Prolongation
  CeedQFunction qf_prolong;
  ierr = CeedQFunctionCreateInteriorByName(ceed, "Scale", &qf_prolong);
  CeedChk(ierr);
  CeedInt *num_comp_p_data;
  ierr = CeedCalloc(1, &num_comp_p_data); CeedChk(ierr);
  num_comp_p_data[0] = num_comp;
  CeedQFunctionContext ctx_p;
  ierr = CeedQFunctionContextCreate(ceed, &ctx_p); CeedChk(ierr);
  ierr = CeedQFunctionContextSetData(ctx_p, CEED_MEM_HOST, CEED_OWN_POINTER,
                                     sizeof(*num_comp_p_data), num_comp_p_data);
  CeedChk(ierr);
  ierr = CeedQFunctionSetContext(qf_prolong, ctx_p); CeedChk(ierr);
  ierr = CeedQFunctionContextDestroy(&ctx_p); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf_prolong, "input", num_comp, CEED_EVAL_INTERP);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf_prolong, "scale", num_comp, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf_prolong, "output", num_comp, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionSetUserFlopsEstimate(qf_prolong, num_comp); CeedChk(ierr);

  ierr = CeedOperatorCreate(ceed, qf_prolong, CEED_QFUNCTION_NONE,
                            CEED_QFUNCTION_NONE, op_prolong);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*op_prolong, "input", rstr_coarse, basis_c_to_f,
                              CEED_VECTOR_ACTIVE); CeedChk(ierr);
  ierr = CeedOperatorSetField(*op_prolong, "scale", rstr_fine,
                              CEED_BASIS_COLLOCATED, mult_vec);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*op_prolong, "output", rstr_fine,
                              CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedChk(ierr);

  // Clone name
  bool has_name = op_fine->name;
  size_t name_len = op_fine->name ? strlen(op_fine->name) : 0;
  ierr = CeedOperatorSetName(*op_coarse, op_fine->name); CeedChk(ierr);
  {
    char *prolongation_name;
    ierr = CeedCalloc(18 + name_len, &prolongation_name); CeedChk(ierr);
    sprintf(prolongation_name, "prolongation%s%s", has_name ? " for " : "",
            op_fine->name);
    ierr = CeedOperatorSetName(*op_prolong, prolongation_name); CeedChk(ierr);
    ierr = CeedFree(&prolongation_name); CeedChk(ierr);
  }
  {
    char *restriction_name;
    ierr = CeedCalloc(17 + name_len, &restriction_name); CeedChk(ierr);
    sprintf(restriction_name, "restriction%s%s", has_name ? " for " : "",
            op_fine->name);
    ierr = CeedOperatorSetName(*op_restrict, restriction_name); CeedChk(ierr);
    ierr = CeedFree(&restriction_name); CeedChk(ierr);
  }

  // Cleanup
  ierr = CeedVectorDestroy(&mult_vec); CeedChk(ierr);
  ierr = CeedBasisDestroy(&basis_c_to_f); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&qf_restrict); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&qf_prolong); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build 1D mass matrix and Laplacian with perturbation

  @param[in] interp_1d    Interpolation matrix in one dimension
  @param[in] grad_1d      Gradient matrix in one dimension
  @param[in] q_weight_1d  Quadrature weights in one dimension
  @param[in] P_1d         Number of basis nodes in one dimension
  @param[in] Q_1d         Number of quadrature points in one dimension
  @param[in] dim          Dimension of basis
  @param[out] mass        Assembled mass matrix in one dimension
  @param[out] laplace     Assembled perturbed Laplacian in one dimension

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
CeedPragmaOptimizeOff
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
CeedPragmaOptimizeOn

/// @}

/// ----------------------------------------------------------------------------
/// CeedOperator Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedOperatorBackend
/// @{

/**
  @brief Create object holding CeedQFunction assembly data for CeedOperator

  @param[in] ceed  A Ceed object where the CeedQFunctionAssemblyData will be created
  @param[out] data Address of the variable where the newly created
                     CeedQFunctionAssemblyData will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataCreate(Ceed ceed,
                                    CeedQFunctionAssemblyData *data) {
  int ierr;

  ierr = CeedCalloc(1, data); CeedChk(ierr);
  (*data)->ref_count = 1;
  (*data)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedQFunctionAssemblyData

  @param data  CeedQFunctionAssemblyData to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataReference(CeedQFunctionAssemblyData data) {
  data->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set re-use of CeedQFunctionAssemblyData

  @param data       CeedQFunctionAssemblyData to mark for reuse
  @param reuse_data Boolean flag indicating data re-use

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataSetReuse(CeedQFunctionAssemblyData data,
                                      bool reuse_data) {
  data->reuse_data = reuse_data;
  data->needs_data_update = true;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Mark QFunctionAssemblyData as stale

  @param data              CeedQFunctionAssemblyData to mark as stale
  @param needs_data_update Boolean flag indicating if update is needed or completed

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataSetUpdateNeeded(CeedQFunctionAssemblyData data,
    bool needs_data_update) {
  data->needs_data_update = needs_data_update;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Determine if QFunctionAssemblyData needs update

  @param[in] data              CeedQFunctionAssemblyData to mark as stale
  @param[out] is_update_needed Boolean flag indicating if re-assembly is required

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataIsUpdateNeeded(CeedQFunctionAssemblyData data,
    bool *is_update_needed) {
  *is_update_needed = !data->reuse_data || data->needs_data_update;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedQFunctionAssemblyData. Both pointers should
           be destroyed with `CeedCeedQFunctionAssemblyDataDestroy()`;
           Note: If `*data_copy` is non-NULL, then it is assumed that
           `*data_copy` is a pointer to a CeedQFunctionAssemblyData. This
           CeedQFunctionAssemblyData will be destroyed if `*data_copy` is
           the only reference to this CeedQFunctionAssemblyData.

  @param data            CeedQFunctionAssemblyData to copy reference to
  @param[out] data_copy  Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataReferenceCopy(CeedQFunctionAssemblyData data,
    CeedQFunctionAssemblyData *data_copy) {
  int ierr;

  ierr = CeedQFunctionAssemblyDataReference(data); CeedChk(ierr);
  ierr = CeedQFunctionAssemblyDataDestroy(data_copy); CeedChk(ierr);
  *data_copy = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get setup status for internal objects for CeedQFunctionAssemblyData

  @param[in] data      CeedQFunctionAssemblyData to retreive status
  @param[out] is_setup Boolean flag for setup status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataIsSetup(CeedQFunctionAssemblyData data,
                                     bool *is_setup) {
  *is_setup = data->is_setup;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set internal objects for CeedQFunctionAssemblyData

  @param[in] data  CeedQFunctionAssemblyData to set objects
  @param[in] vec   CeedVector to store assembled CeedQFunction at quadrature points
  @param[in] rstr  CeedElemRestriction for CeedVector containing assembled CeedQFunction

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataSetObjects(CeedQFunctionAssemblyData data,
                                        CeedVector vec, CeedElemRestriction rstr) {
  int ierr;

  ierr = CeedVectorReferenceCopy(vec, &data->vec); CeedChk(ierr);
  ierr = CeedElemRestrictionReferenceCopy(rstr, &data->rstr); CeedChk(ierr);

  data->is_setup = true;
  return CEED_ERROR_SUCCESS;
}

int CeedQFunctionAssemblyDataGetObjects(CeedQFunctionAssemblyData data,
                                        CeedVector *vec, CeedElemRestriction *rstr) {
  int ierr;

  if (!data->is_setup)
    // LCOV_EXCL_START
    return CeedError(data->ceed, CEED_ERROR_INCOMPLETE,
                     "Internal objects not set; must call CeedQFunctionAssemblyDataSetObjects first.");
  // LCOV_EXCL_STOP

  ierr = CeedVectorReferenceCopy(data->vec, vec); CeedChk(ierr);
  ierr = CeedElemRestrictionReferenceCopy(data->rstr, rstr); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy CeedQFunctionAssemblyData

  @param[out] data  CeedQFunctionAssemblyData to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataDestroy(CeedQFunctionAssemblyData *data) {
  int ierr;

  if (!*data || --(*data)->ref_count > 0) return CEED_ERROR_SUCCESS;

  ierr = CeedDestroy(&(*data)->ceed); CeedChk(ierr);
  ierr = CeedVectorDestroy(&(*data)->vec); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&(*data)->rstr); CeedChk(ierr);

  ierr = CeedFree(data); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedOperator Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedOperatorUser
/// @{

/**
  @brief Assemble a linear CeedQFunction associated with a CeedOperator

  This returns a CeedVector containing a matrix at each quadrature point
    providing the action of the CeedQFunction associated with the CeedOperator.
    The vector 'assembled' is of shape
      [num_elements, num_input_fields, num_output_fields, num_quad_points]
    and contains column-major matrices representing the action of the
    CeedQFunction for a corresponding quadrature point on an element. Inputs and
    outputs are in the order provided by the user when adding CeedOperator fields.
    For example, a CeedQFunction with inputs 'u' and 'gradu' and outputs 'gradv' and
    'v', provided in that order, would result in an assembled QFunction that
    consists of (1 + dim) x (dim + 1) matrices at each quadrature point acting
    on the input [u, du_0, du_1] and producing the output [dv_0, dv_1, v].

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param op              CeedOperator to assemble CeedQFunction
  @param[out] assembled  CeedVector to store assembled CeedQFunction at
                           quadrature points
  @param[out] rstr       CeedElemRestriction for CeedVector containing assembled
                           CeedQFunction
  @param request         Address of CeedRequest for non-blocking completion, else
                           @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssembleQFunction(CeedOperator op, CeedVector *assembled,
                                        CeedElemRestriction *rstr,
                                        CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Backend version
  if (op->LinearAssembleQFunction) {
    ierr = op->LinearAssembleQFunction(op, assembled, rstr, request);
    CeedChk(ierr);
  } else {
    // Fallback to reference Ceed
    if (!op->op_fallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    ierr = CeedOperatorLinearAssembleQFunction(op->op_fallback, assembled,
           rstr, request); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble CeedQFunction and store result internall. Return copied
           references of stored data to the caller. Caller is responsible for
           ownership and destruction of the copied references. See also
           @ref CeedOperatorLinearAssembleQFunction

  @param op              CeedOperator to assemble CeedQFunction
  @param assembled       CeedVector to store assembled CeedQFunction at
                           quadrature points
  @param rstr            CeedElemRestriction for CeedVector containing assembled
                           CeedQFunction
  @param request         Address of CeedRequest for non-blocking completion, else
                           @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssembleQFunctionBuildOrUpdate(CeedOperator op,
    CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Backend version
  if (op->LinearAssembleQFunctionUpdate) {
    bool qf_assembled_is_setup;
    CeedVector assembled_vec = NULL;
    CeedElemRestriction assembled_rstr = NULL;

    ierr = CeedQFunctionAssemblyDataIsSetup(op->qf_assembled,
                                            &qf_assembled_is_setup); CeedChk(ierr);
    if (qf_assembled_is_setup) {
      ierr = CeedQFunctionAssemblyDataGetObjects(op->qf_assembled, &assembled_vec,
             &assembled_rstr); CeedChk(ierr);

      bool update_needed;
      ierr = CeedQFunctionAssemblyDataIsUpdateNeeded(op->qf_assembled,
             &update_needed); CeedChk(ierr);
      if (update_needed) {
        ierr = op->LinearAssembleQFunctionUpdate(op, assembled_vec, assembled_rstr,
               request); CeedChk(ierr);
      }
    } else {
      ierr = op->LinearAssembleQFunction(op, &assembled_vec, &assembled_rstr,
                                         request); CeedChk(ierr);
      ierr = CeedQFunctionAssemblyDataSetObjects(op->qf_assembled, assembled_vec,
             assembled_rstr); CeedChk(ierr);
    }
    ierr = CeedQFunctionAssemblyDataSetUpdateNeeded(op->qf_assembled, false);
    CeedChk(ierr);

    // Copy reference to internally held copy
    *assembled = NULL;
    *rstr = NULL;
    ierr = CeedVectorReferenceCopy(assembled_vec, assembled); CeedChk(ierr);
    ierr = CeedVectorDestroy(&assembled_vec); CeedChk(ierr);
    ierr = CeedElemRestrictionReferenceCopy(assembled_rstr, rstr); CeedChk(ierr);
    ierr = CeedElemRestrictionDestroy(&assembled_rstr); CeedChk(ierr);
  } else {
    // Fallback to reference Ceed
    if (!op->op_fallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    ierr = CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op->op_fallback,
           assembled, rstr, request); CeedChk(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble the diagonal of a square linear CeedOperator

  This overwrites a CeedVector with the diagonal of a linear CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param op              CeedOperator to assemble CeedQFunction
  @param[out] assembled  CeedVector to store assembled CeedOperator diagonal
  @param request         Address of CeedRequest for non-blocking completion, else
                           @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssembleDiagonal(CeedOperator op, CeedVector assembled,
                                       CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  CeedSize input_size = 0, output_size = 0;
  ierr = CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size);
  CeedChk(ierr);
  if (input_size != output_size)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");
  // LCOV_EXCL_STOP

  // Use backend version, if available
  if (op->LinearAssembleDiagonal) {
    ierr = op->LinearAssembleDiagonal(op, assembled, request); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else if (op->LinearAssembleAddDiagonal) {
    ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
    ierr = op->LinearAssembleAddDiagonal(op, assembled, request); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else {
    // Check for valid fallback resource
    const char *resource, *fallback_resource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
    CeedChk(ierr);
    if (strcmp(fallback_resource, "") && strcmp(resource, fallback_resource)) {
      // Fallback to reference Ceed
      if (!op->op_fallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      ierr = CeedOperatorLinearAssembleDiagonal(op->op_fallback, assembled, request);
      CeedChk(ierr);
      return CEED_ERROR_SUCCESS;
    }
  }

  // Default interface implementation
  ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
  ierr = CeedOperatorLinearAssembleAddDiagonal(op, assembled, request);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble the diagonal of a square linear CeedOperator

  This sums into a CeedVector the diagonal of a linear CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param op              CeedOperator to assemble CeedQFunction
  @param[out] assembled  CeedVector to store assembled CeedOperator diagonal
  @param request         Address of CeedRequest for non-blocking completion, else
                           @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssembleAddDiagonal(CeedOperator op, CeedVector assembled,
    CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  CeedSize input_size = 0, output_size = 0;
  ierr = CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size);
  CeedChk(ierr);
  if (input_size != output_size)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");
  // LCOV_EXCL_STOP

  // Use backend version, if available
  if (op->LinearAssembleAddDiagonal) {
    ierr = op->LinearAssembleAddDiagonal(op, assembled, request); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else {
    // Check for valid fallback resource
    const char *resource, *fallback_resource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
    CeedChk(ierr);
    if (strcmp(fallback_resource, "") && strcmp(resource, fallback_resource)) {
      // Fallback to reference Ceed
      if (!op->op_fallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      ierr = CeedOperatorLinearAssembleAddDiagonal(op->op_fallback, assembled,
             request); CeedChk(ierr);
      return CEED_ERROR_SUCCESS;
    }
  }

  // Default interface implementation
  bool is_composite;
  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);
  if (is_composite) {
    ierr = CeedCompositeOperatorLinearAssembleAddDiagonal(op, request,
           false, assembled); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else {
    ierr = CeedSingleOperatorAssembleAddDiagonal(op, request, false, assembled);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }
}

/**
  @brief Assemble the point block diagonal of a square linear CeedOperator

  This overwrites a CeedVector with the point block diagonal of a linear
    CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param op              CeedOperator to assemble CeedQFunction
  @param[out] assembled  CeedVector to store assembled CeedOperator point block
                           diagonal, provided in row-major form with an
                           @a num_comp * @a num_comp block at each node. The dimensions
                           of this vector are derived from the active vector
                           for the CeedOperator. The array has shape
                           [nodes, component out, component in].
  @param request         Address of CeedRequest for non-blocking completion, else
                           CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssemblePointBlockDiagonal(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  CeedSize input_size = 0, output_size = 0;
  ierr = CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size);
  CeedChk(ierr);
  if (input_size != output_size)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");
  // LCOV_EXCL_STOP

  // Use backend version, if available
  if (op->LinearAssemblePointBlockDiagonal) {
    ierr = op->LinearAssemblePointBlockDiagonal(op, assembled, request);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else if (op->LinearAssembleAddPointBlockDiagonal) {
    ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
    ierr = CeedOperatorLinearAssembleAddPointBlockDiagonal(op, assembled,
           request); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else {
    // Check for valid fallback resource
    const char *resource, *fallback_resource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
    CeedChk(ierr);
    if (strcmp(fallback_resource, "") && strcmp(resource, fallback_resource)) {
      // Fallback to reference Ceed
      if (!op->op_fallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      ierr = CeedOperatorLinearAssemblePointBlockDiagonal(op->op_fallback,
             assembled, request); CeedChk(ierr);
      return CEED_ERROR_SUCCESS;
    }
  }

  // Default interface implementation
  ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
  ierr = CeedOperatorLinearAssembleAddPointBlockDiagonal(op, assembled, request);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble the point block diagonal of a square linear CeedOperator

  This sums into a CeedVector with the point block diagonal of a linear
    CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param op              CeedOperator to assemble CeedQFunction
  @param[out] assembled  CeedVector to store assembled CeedOperator point block
                           diagonal, provided in row-major form with an
                           @a num_comp * @a num_comp block at each node. The dimensions
                           of this vector are derived from the active vector
                           for the CeedOperator. The array has shape
                           [nodes, component out, component in].
  @param request         Address of CeedRequest for non-blocking completion, else
                           CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssembleAddPointBlockDiagonal(CeedOperator op,
    CeedVector assembled, CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  CeedSize input_size = 0, output_size = 0;
  ierr = CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size);
  CeedChk(ierr);
  if (input_size != output_size)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");
  // LCOV_EXCL_STOP

  // Use backend version, if available
  if (op->LinearAssembleAddPointBlockDiagonal) {
    ierr = op->LinearAssembleAddPointBlockDiagonal(op, assembled, request);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else {
    // Check for valid fallback resource
    const char *resource, *fallback_resource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
    CeedChk(ierr);
    if (strcmp(fallback_resource, "") && strcmp(resource, fallback_resource)) {
      // Fallback to reference Ceed
      if (!op->op_fallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      ierr = CeedOperatorLinearAssembleAddPointBlockDiagonal(op->op_fallback,
             assembled, request); CeedChk(ierr);
      return CEED_ERROR_SUCCESS;
    }
  }

  // Default interface implemenation
  bool is_composite;
  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);
  if (is_composite) {
    ierr = CeedCompositeOperatorLinearAssembleAddDiagonal(op, request,
           true, assembled); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else {
    ierr = CeedSingleOperatorAssembleAddDiagonal(op, request, true, assembled);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }
}

/**
   @brief Fully assemble the nonzero pattern of a linear operator.

   Expected to be used in conjunction with CeedOperatorLinearAssemble()

   The assembly routines use coordinate format, with num_entries tuples of the
   form (i, j, value) which indicate that value should be added to the matrix
   in entry (i, j). Note that the (i, j) pairs are not unique and may repeat.
   This function returns the number of entries and their (i, j) locations,
   while CeedOperatorLinearAssemble() provides the values in the same
   ordering.

   This will generally be slow unless your operator is low-order.

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

   @param[in]  op           CeedOperator to assemble
   @param[out] num_entries  Number of entries in coordinate nonzero pattern
   @param[out] rows         Row number for each entry
   @param[out] cols         Column number for each entry

   @ref User
**/
int CeedOperatorLinearAssembleSymbolic(CeedOperator op, CeedSize *num_entries,
                                       CeedInt **rows, CeedInt **cols) {
  int ierr;
  CeedInt num_suboperators, single_entries;
  CeedOperator *sub_operators;
  bool is_composite;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssembleSymbolic) {
    ierr = op->LinearAssembleSymbolic(op, num_entries, rows, cols); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else {
    // Check for valid fallback resource
    const char *resource, *fallback_resource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
    CeedChk(ierr);
    if (strcmp(fallback_resource, "") && strcmp(resource, fallback_resource)) {
      // Fallback to reference Ceed
      if (!op->op_fallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      ierr = CeedOperatorLinearAssembleSymbolic(op->op_fallback, num_entries, rows,
             cols); CeedChk(ierr);
      return CEED_ERROR_SUCCESS;
    }
  }

  // Default interface implementation

  // count entries and allocate rows, cols arrays
  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);
  *num_entries = 0;
  if (is_composite) {
    ierr = CeedOperatorGetNumSub(op, &num_suboperators); CeedChk(ierr);
    ierr = CeedOperatorGetSubList(op, &sub_operators); CeedChk(ierr);
    for (int k = 0; k < num_suboperators; ++k) {
      ierr = CeedSingleOperatorAssemblyCountEntries(sub_operators[k],
             &single_entries); CeedChk(ierr);
      *num_entries += single_entries;
    }
  } else {
    ierr = CeedSingleOperatorAssemblyCountEntries(op,
           &single_entries); CeedChk(ierr);
    *num_entries += single_entries;
  }
  ierr = CeedCalloc(*num_entries, rows); CeedChk(ierr);
  ierr = CeedCalloc(*num_entries, cols); CeedChk(ierr);

  // assemble nonzero locations
  CeedInt offset = 0;
  if (is_composite) {
    ierr = CeedOperatorGetNumSub(op, &num_suboperators); CeedChk(ierr);
    ierr = CeedOperatorGetSubList(op, &sub_operators); CeedChk(ierr);
    for (int k = 0; k < num_suboperators; ++k) {
      ierr = CeedSingleOperatorAssembleSymbolic(sub_operators[k], offset, *rows,
             *cols); CeedChk(ierr);
      ierr = CeedSingleOperatorAssemblyCountEntries(sub_operators[k],
             &single_entries);
      CeedChk(ierr);
      offset += single_entries;
    }
  } else {
    ierr = CeedSingleOperatorAssembleSymbolic(op, offset, *rows, *cols);
    CeedChk(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

/**
   @brief Fully assemble the nonzero entries of a linear operator.

   Expected to be used in conjunction with CeedOperatorLinearAssembleSymbolic()

   The assembly routines use coordinate format, with num_entries tuples of the
   form (i, j, value) which indicate that value should be added to the matrix
   in entry (i, j). Note that the (i, j) pairs are not unique and may repeat.
   This function returns the values of the nonzero entries to be added, their
   (i, j) locations are provided by CeedOperatorLinearAssembleSymbolic()

   This will generally be slow unless your operator is low-order.

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

   @param[in]  op      CeedOperator to assemble
   @param[out] values  Values to assemble into matrix

   @ref User
**/
int CeedOperatorLinearAssemble(CeedOperator op, CeedVector values) {
  int ierr;
  CeedInt num_suboperators, single_entries = 0;
  CeedOperator *sub_operators;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssemble) {
    ierr = op->LinearAssemble(op, values); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else {
    // Check for valid fallback resource
    const char *resource, *fallback_resource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
    CeedChk(ierr);
    if (strcmp(fallback_resource, "") && strcmp(resource, fallback_resource)) {
      // Fallback to reference Ceed
      if (!op->op_fallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      ierr = CeedOperatorLinearAssemble(op->op_fallback, values); CeedChk(ierr);
      return CEED_ERROR_SUCCESS;
    }
  }

  // Default interface implementation
  bool is_composite;
  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);

  CeedInt offset = 0;
  if (is_composite) {
    ierr = CeedOperatorGetNumSub(op, &num_suboperators); CeedChk(ierr);
    ierr = CeedOperatorGetSubList(op, &sub_operators); CeedChk(ierr);
    for (int k = 0; k < num_suboperators; ++k) {
      ierr = CeedSingleOperatorAssemble(sub_operators[k], offset, values);
      CeedChk(ierr);
      ierr = CeedSingleOperatorAssemblyCountEntries(sub_operators[k],
             &single_entries);
      CeedChk(ierr);
      offset += single_entries;
    }
  } else {
    ierr = CeedSingleOperatorAssemble(op, offset, values); CeedChk(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators
           for a CeedOperator, creating the prolongation basis from the
           fine and coarse grid interpolation

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param[in] op_fine       Fine grid operator
  @param[in] p_mult_fine   L-vector multiplicity in parallel gather/scatter
  @param[in] rstr_coarse   Coarse grid restriction
  @param[in] basis_coarse  Coarse grid active vector basis
  @param[out] op_coarse    Coarse grid operator
  @param[out] op_prolong   Coarse to fine operator
  @param[out] op_restrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreate(CeedOperator op_fine,
                                     CeedVector p_mult_fine,
                                     CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
                                     CeedOperator *op_coarse, CeedOperator *op_prolong,
                                     CeedOperator *op_restrict) {
  int ierr;
  ierr = CeedOperatorCheckReady(op_fine); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op_fine, &ceed); CeedChk(ierr);

  // Check for compatible quadrature spaces
  CeedBasis basis_fine;
  ierr = CeedOperatorGetActiveBasis(op_fine, &basis_fine); CeedChk(ierr);
  CeedInt Q_f, Q_c;
  ierr = CeedBasisGetNumQuadraturePoints(basis_fine, &Q_f); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis_coarse, &Q_c); CeedChk(ierr);
  if (Q_f != Q_c)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION,
                     "Bases must have compatible quadrature spaces");
  // LCOV_EXCL_STOP

  // Coarse to fine basis
  CeedInt P_f, P_c, Q = Q_f;
  bool is_tensor_f, is_tensor_c;
  ierr = CeedBasisIsTensor(basis_fine, &is_tensor_f); CeedChk(ierr);
  ierr = CeedBasisIsTensor(basis_coarse, &is_tensor_c); CeedChk(ierr);
  CeedScalar *interp_c, *interp_f, *interp_c_to_f, *tau;
  if (is_tensor_f && is_tensor_c) {
    ierr = CeedBasisGetNumNodes1D(basis_fine, &P_f); CeedChk(ierr);
    ierr = CeedBasisGetNumNodes1D(basis_coarse, &P_c); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis_coarse, &Q); CeedChk(ierr);
  } else if (!is_tensor_f && !is_tensor_c) {
    ierr = CeedBasisGetNumNodes(basis_fine, &P_f); CeedChk(ierr);
    ierr = CeedBasisGetNumNodes(basis_coarse, &P_c); CeedChk(ierr);
  } else {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MINOR,
                     "Bases must both be tensor or non-tensor");
    // LCOV_EXCL_STOP
  }

  ierr = CeedMalloc(Q*P_f, &interp_f); CeedChk(ierr);
  ierr = CeedMalloc(Q*P_c, &interp_c); CeedChk(ierr);
  ierr = CeedCalloc(P_c*P_f, &interp_c_to_f); CeedChk(ierr);
  ierr = CeedMalloc(Q, &tau); CeedChk(ierr);
  if (is_tensor_f) {
    memcpy(interp_f, basis_fine->interp_1d, Q*P_f*sizeof basis_fine->interp_1d[0]);
    memcpy(interp_c, basis_coarse->interp_1d,
           Q*P_c*sizeof basis_coarse->interp_1d[0]);
  } else {
    memcpy(interp_f, basis_fine->interp, Q*P_f*sizeof basis_fine->interp[0]);
    memcpy(interp_c, basis_coarse->interp, Q*P_c*sizeof basis_coarse->interp[0]);
  }

  // -- QR Factorization, interp_f = Q R
  ierr = CeedQRFactorization(ceed, interp_f, tau, Q, P_f); CeedChk(ierr);

  // -- Apply Qtranspose, interp_c = Qtranspose interp_c
  ierr = CeedHouseholderApplyQ(interp_c, interp_f, tau, CEED_TRANSPOSE,
                               Q, P_c, P_f, P_c, 1); CeedChk(ierr);

  // -- Apply Rinv, interp_c_to_f = Rinv interp_c
  for (CeedInt j=0; j<P_c; j++) { // Column j
    interp_c_to_f[j+P_c*(P_f-1)] = interp_c[j+P_c*(P_f-1)]/interp_f[P_f*P_f-1];
    for (CeedInt i=P_f-2; i>=0; i--) { // Row i
      interp_c_to_f[j+P_c*i] = interp_c[j+P_c*i];
      for (CeedInt k=i+1; k<P_f; k++)
        interp_c_to_f[j+P_c*i] -= interp_f[k+P_f*i]*interp_c_to_f[j+P_c*k];
      interp_c_to_f[j+P_c*i] /= interp_f[i+P_f*i];
    }
  }
  ierr = CeedFree(&tau); CeedChk(ierr);
  ierr = CeedFree(&interp_c); CeedChk(ierr);
  ierr = CeedFree(&interp_f); CeedChk(ierr);

  // Complete with interp_c_to_f versions of code
  if (is_tensor_f) {
    ierr = CeedOperatorMultigridLevelCreateTensorH1(op_fine, p_mult_fine,
           rstr_coarse, basis_coarse, interp_c_to_f, op_coarse, op_prolong, op_restrict);
    CeedChk(ierr);
  } else {
    ierr = CeedOperatorMultigridLevelCreateH1(op_fine, p_mult_fine,
           rstr_coarse, basis_coarse, interp_c_to_f, op_coarse, op_prolong, op_restrict);
    CeedChk(ierr);
  }

  // Cleanup
  ierr = CeedFree(&interp_c_to_f); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a tensor basis for the active basis

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param[in] op_fine        Fine grid operator
  @param[in] p_mult_fine    L-vector multiplicity in parallel gather/scatter
  @param[in] rstr_coarse    Coarse grid restriction
  @param[in] basis_coarse   Coarse grid active vector basis
  @param[in] interp_c_to_f  Matrix for coarse to fine interpolation
  @param[out] op_coarse     Coarse grid operator
  @param[out] op_prolong    Coarse to fine operator
  @param[out] op_restrict   Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreateTensorH1(CeedOperator op_fine,
    CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
    const CeedScalar *interp_c_to_f, CeedOperator *op_coarse,
    CeedOperator *op_prolong, CeedOperator *op_restrict) {
  int ierr;
  ierr = CeedOperatorCheckReady(op_fine); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op_fine, &ceed); CeedChk(ierr);

  // Check for compatible quadrature spaces
  CeedBasis basis_fine;
  ierr = CeedOperatorGetActiveBasis(op_fine, &basis_fine); CeedChk(ierr);
  CeedInt Q_f, Q_c;
  ierr = CeedBasisGetNumQuadraturePoints(basis_fine, &Q_f); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis_coarse, &Q_c); CeedChk(ierr);
  if (Q_f != Q_c)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION,
                     "Bases must have compatible quadrature spaces");
  // LCOV_EXCL_STOP

  // Coarse to fine basis
  CeedInt dim, num_comp, num_nodes_c, P_1d_f, P_1d_c;
  ierr = CeedBasisGetDimension(basis_fine, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis_fine, &num_comp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes1D(basis_fine, &P_1d_f); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr_coarse, &num_nodes_c);
  CeedChk(ierr);
  P_1d_c = dim == 1 ? num_nodes_c :
           dim == 2 ? sqrt(num_nodes_c) :
           cbrt(num_nodes_c);
  CeedScalar *q_ref, *q_weight, *grad;
  ierr = CeedCalloc(P_1d_f, &q_ref); CeedChk(ierr);
  ierr = CeedCalloc(P_1d_f, &q_weight); CeedChk(ierr);
  ierr = CeedCalloc(P_1d_f*P_1d_c*dim, &grad); CeedChk(ierr);
  CeedBasis basis_c_to_f;
  ierr = CeedBasisCreateTensorH1(ceed, dim, num_comp, P_1d_c, P_1d_f,
                                 interp_c_to_f, grad, q_ref, q_weight, &basis_c_to_f);
  CeedChk(ierr);
  ierr = CeedFree(&q_ref); CeedChk(ierr);
  ierr = CeedFree(&q_weight); CeedChk(ierr);
  ierr = CeedFree(&grad); CeedChk(ierr);

  // Core code
  ierr = CeedSingleOperatorMultigridLevel(op_fine, p_mult_fine, rstr_coarse,
                                          basis_coarse, basis_c_to_f, op_coarse,
                                          op_prolong, op_restrict);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a non-tensor basis for the active vector

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param[in] op_fine        Fine grid operator
  @param[in] p_mult_fine    L-vector multiplicity in parallel gather/scatter
  @param[in] rstr_coarse    Coarse grid restriction
  @param[in] basis_coarse   Coarse grid active vector basis
  @param[in] interp_c_to_f  Matrix for coarse to fine interpolation
  @param[out] op_coarse     Coarse grid operator
  @param[out] op_prolong    Coarse to fine operator
  @param[out] op_restrict   Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreateH1(CeedOperator op_fine,
                                       CeedVector p_mult_fine,
                                       CeedElemRestriction rstr_coarse,
                                       CeedBasis basis_coarse,
                                       const CeedScalar *interp_c_to_f,
                                       CeedOperator *op_coarse,
                                       CeedOperator *op_prolong,
                                       CeedOperator *op_restrict) {
  int ierr;
  ierr = CeedOperatorCheckReady(op_fine); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op_fine, &ceed); CeedChk(ierr);

  // Check for compatible quadrature spaces
  CeedBasis basis_fine;
  ierr = CeedOperatorGetActiveBasis(op_fine, &basis_fine); CeedChk(ierr);
  CeedInt Q_f, Q_c;
  ierr = CeedBasisGetNumQuadraturePoints(basis_fine, &Q_f); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis_coarse, &Q_c); CeedChk(ierr);
  if (Q_f != Q_c)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION,
                     "Bases must have compatible quadrature spaces");
  // LCOV_EXCL_STOP

  // Coarse to fine basis
  CeedElemTopology topo;
  ierr = CeedBasisGetTopology(basis_fine, &topo); CeedChk(ierr);
  CeedInt dim, num_comp, num_nodes_c, num_nodes_f;
  ierr = CeedBasisGetDimension(basis_fine, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis_fine, &num_comp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basis_fine, &num_nodes_f); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr_coarse, &num_nodes_c);
  CeedChk(ierr);
  CeedScalar *q_ref, *q_weight, *grad;
  ierr = CeedCalloc(num_nodes_f*dim, &q_ref); CeedChk(ierr);
  ierr = CeedCalloc(num_nodes_f, &q_weight); CeedChk(ierr);
  ierr = CeedCalloc(num_nodes_f*num_nodes_c*dim, &grad); CeedChk(ierr);
  CeedBasis basis_c_to_f;
  ierr = CeedBasisCreateH1(ceed, topo, num_comp, num_nodes_c, num_nodes_f,
                           interp_c_to_f, grad, q_ref, q_weight, &basis_c_to_f);
  CeedChk(ierr);
  ierr = CeedFree(&q_ref); CeedChk(ierr);
  ierr = CeedFree(&q_weight); CeedChk(ierr);
  ierr = CeedFree(&grad); CeedChk(ierr);

  // Core code
  ierr = CeedSingleOperatorMultigridLevel(op_fine, p_mult_fine, rstr_coarse,
                                          basis_coarse, basis_c_to_f, op_coarse,
                                          op_prolong, op_restrict);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build a FDM based approximate inverse for each element for a
           CeedOperator

  This returns a CeedOperator and CeedVector to apply a Fast Diagonalization
    Method based approximate inverse. This function obtains the simultaneous
    diagonalization for the 1D mass and Laplacian operators,
      M = V^T V, K = V^T S V.
    The assembled QFunction is used to modify the eigenvalues from simultaneous
    diagonalization and obtain an approximate inverse of the form
      V^T S^hat V. The CeedOperator must be linear and non-composite. The
    associated CeedQFunction must therefore also be linear.

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param op            CeedOperator to create element inverses
  @param[out] fdm_inv  CeedOperator to apply the action of a FDM based inverse
                         for each element
  @param request       Address of CeedRequest for non-blocking completion, else
                         @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorCreateFDMElementInverse(CeedOperator op, CeedOperator *fdm_inv,
                                        CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Use backend version, if available
  if (op->CreateFDMElementInverse) {
    ierr = op->CreateFDMElementInverse(op, fdm_inv, request); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  } else {
    // Check for valid fallback resource
    const char *resource, *fallback_resource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
    CeedChk(ierr);
    if (strcmp(fallback_resource, "") && strcmp(resource, fallback_resource)) {
      // Fallback to reference Ceed
      if (!op->op_fallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      ierr = CeedOperatorCreateFDMElementInverse(op->op_fallback, fdm_inv, request);
      CeedChk(ierr);
      return CEED_ERROR_SUCCESS;
    }
  }

  // Interface implementation
  Ceed ceed, ceed_parent;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
  ierr = CeedGetOperatorFallbackParentCeed(ceed, &ceed_parent); CeedChk(ierr);
  ceed_parent = ceed_parent ? ceed_parent : ceed;
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);

  // Determine active input basis
  bool interp = false, grad = false;
  CeedBasis basis = NULL;
  CeedElemRestriction rstr = NULL;
  CeedOperatorField *op_fields;
  CeedQFunctionField *qf_fields;
  CeedInt num_input_fields;
  ierr = CeedOperatorGetFields(op, &num_input_fields, &op_fields, NULL, NULL);
  CeedChk(ierr);
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL); CeedChk(ierr);
  for (CeedInt i=0; i<num_input_fields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(op_fields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedEvalMode eval_mode;
      ierr = CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode); CeedChk(ierr);
      interp = interp || eval_mode == CEED_EVAL_INTERP;
      grad = grad || eval_mode == CEED_EVAL_GRAD;
      ierr = CeedOperatorFieldGetBasis(op_fields[i], &basis); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr); CeedChk(ierr);
    }
  }
  if (!basis)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No active field set");
  // LCOV_EXCL_STOP
  CeedSize l_size = 1;
  CeedInt P_1d, Q_1d, elem_size, num_qpts, dim, num_comp = 1, num_elem = 1;
  ierr = CeedBasisGetNumNodes1D(basis, &P_1d); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basis, &elem_size); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &num_qpts); CeedChk(ierr);
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &num_comp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(rstr, &num_elem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetLVectorSize(rstr, &l_size); CeedChk(ierr);

  // Build and diagonalize 1D Mass and Laplacian
  bool tensor_basis;
  ierr = CeedBasisIsTensor(basis, &tensor_basis); CeedChk(ierr);
  if (!tensor_basis)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "FDMElementInverse only supported for tensor "
                     "bases");
  // LCOV_EXCL_STOP
  CeedScalar *mass, *laplace, *x, *fdm_interp, *lambda;
  ierr = CeedCalloc(P_1d*P_1d, &mass); CeedChk(ierr);
  ierr = CeedCalloc(P_1d*P_1d, &laplace); CeedChk(ierr);
  ierr = CeedCalloc(P_1d*P_1d, &x); CeedChk(ierr);
  ierr = CeedCalloc(P_1d*P_1d, &fdm_interp); CeedChk(ierr);
  ierr = CeedCalloc(P_1d, &lambda); CeedChk(ierr);
  // -- Build matrices
  const CeedScalar *interp_1d, *grad_1d, *q_weight_1d;
  ierr = CeedBasisGetInterp1D(basis, &interp_1d); CeedChk(ierr);
  ierr = CeedBasisGetGrad1D(basis, &grad_1d); CeedChk(ierr);
  ierr = CeedBasisGetQWeights(basis, &q_weight_1d); CeedChk(ierr);
  ierr = CeedBuildMassLaplace(interp_1d, grad_1d, q_weight_1d, P_1d, Q_1d, dim,
                              mass, laplace); CeedChk(ierr);

  // -- Diagonalize
  ierr = CeedSimultaneousDiagonalization(ceed, laplace, mass, x, lambda, P_1d);
  CeedChk(ierr);
  ierr = CeedFree(&mass); CeedChk(ierr);
  ierr = CeedFree(&laplace); CeedChk(ierr);
  for (CeedInt i=0; i<P_1d; i++)
    for (CeedInt j=0; j<P_1d; j++)
      fdm_interp[i+j*P_1d] = x[j+i*P_1d];
  ierr = CeedFree(&x); CeedChk(ierr);

  // Assemble QFunction
  CeedVector assembled;
  CeedElemRestriction rstr_qf;
  ierr =  CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled,
          &rstr_qf, request); CeedChk(ierr);
  CeedInt layout[3];
  ierr = CeedElemRestrictionGetELayout(rstr_qf, &layout); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_qf); CeedChk(ierr);
  CeedScalar max_norm = 0;
  ierr = CeedVectorNorm(assembled, CEED_NORM_MAX, &max_norm); CeedChk(ierr);

  // Calculate element averages
  CeedInt num_modes = (interp?1:0) + (grad?dim:0);
  CeedScalar *elem_avg;
  const CeedScalar *assembled_array, *q_weight_array;
  CeedVector q_weight;
  ierr = CeedVectorCreate(ceed_parent, num_qpts, &q_weight); CeedChk(ierr);
  ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                        CEED_VECTOR_NONE, q_weight); CeedChk(ierr);
  ierr = CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array);
  CeedChk(ierr);
  ierr = CeedVectorGetArrayRead(q_weight, CEED_MEM_HOST, &q_weight_array);
  CeedChk(ierr);
  ierr = CeedCalloc(num_elem, &elem_avg); CeedChk(ierr);
  const CeedScalar qf_value_bound = max_norm*100*CEED_EPSILON;
  for (CeedInt e=0; e<num_elem; e++) {
    CeedInt count = 0;
    for (CeedInt q=0; q<num_qpts; q++)
      for (CeedInt i=0; i<num_comp*num_comp*num_modes*num_modes; i++)
        if (fabs(assembled_array[q*layout[0] + i*layout[1] + e*layout[2]]) >
            qf_value_bound) {
          elem_avg[e] += assembled_array[q*layout[0] + i*layout[1] + e*layout[2]] /
                         q_weight_array[q];
          count++;
        }
    if (count) {
      elem_avg[e] /= count;
    } else {
      elem_avg[e] = 1.0;
    }
  }
  ierr = CeedVectorRestoreArrayRead(assembled, &assembled_array); CeedChk(ierr);
  ierr = CeedVectorDestroy(&assembled); CeedChk(ierr);
  ierr = CeedVectorRestoreArrayRead(q_weight, &q_weight_array); CeedChk(ierr);
  ierr = CeedVectorDestroy(&q_weight); CeedChk(ierr);

  // Build FDM diagonal
  CeedVector q_data;
  CeedScalar *q_data_array, *fdm_diagonal;
  ierr = CeedCalloc(num_comp*elem_size, &fdm_diagonal); CeedChk(ierr);
  const CeedScalar fdm_diagonal_bound = elem_size*CEED_EPSILON;
  for (CeedInt c=0; c<num_comp; c++)
    for (CeedInt n=0; n<elem_size; n++) {
      if (interp)
        fdm_diagonal[c*elem_size + n] = 1.0;
      if (grad)
        for (CeedInt d=0; d<dim; d++) {
          CeedInt i = (n / CeedIntPow(P_1d, d)) % P_1d;
          fdm_diagonal[c*elem_size + n] += lambda[i];
        }
      if (fabs(fdm_diagonal[c*elem_size + n]) < fdm_diagonal_bound)
        fdm_diagonal[c*elem_size + n] = fdm_diagonal_bound;
    }
  ierr = CeedVectorCreate(ceed_parent, num_elem*num_comp*elem_size, &q_data);
  CeedChk(ierr);
  ierr = CeedVectorSetValue(q_data, 0.0); CeedChk(ierr);
  ierr = CeedVectorGetArrayWrite(q_data, CEED_MEM_HOST, &q_data_array);
  CeedChk(ierr);
  for (CeedInt e=0; e<num_elem; e++)
    for (CeedInt c=0; c<num_comp; c++)
      for (CeedInt n=0; n<elem_size; n++)
        q_data_array[(e*num_comp+c)*elem_size+n] = 1. / (elem_avg[e] *
            fdm_diagonal[c*elem_size + n]);
  ierr = CeedFree(&elem_avg); CeedChk(ierr);
  ierr = CeedFree(&fdm_diagonal); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(q_data, &q_data_array); CeedChk(ierr);

  // Setup FDM operator
  // -- Basis
  CeedBasis fdm_basis;
  CeedScalar *grad_dummy, *q_ref_dummy, *q_weight_dummy;
  ierr = CeedCalloc(P_1d*P_1d, &grad_dummy); CeedChk(ierr);
  ierr = CeedCalloc(P_1d, &q_ref_dummy); CeedChk(ierr);
  ierr = CeedCalloc(P_1d, &q_weight_dummy); CeedChk(ierr);
  ierr = CeedBasisCreateTensorH1(ceed_parent, dim, num_comp, P_1d, P_1d,
                                 fdm_interp, grad_dummy, q_ref_dummy,
                                 q_weight_dummy, &fdm_basis); CeedChk(ierr);
  ierr = CeedFree(&fdm_interp); CeedChk(ierr);
  ierr = CeedFree(&grad_dummy); CeedChk(ierr);
  ierr = CeedFree(&q_ref_dummy); CeedChk(ierr);
  ierr = CeedFree(&q_weight_dummy); CeedChk(ierr);
  ierr = CeedFree(&lambda); CeedChk(ierr);

  // -- Restriction
  CeedElemRestriction rstr_qd_i;
  CeedInt strides[3] = {1, elem_size, elem_size*num_comp};
  ierr = CeedElemRestrictionCreateStrided(ceed_parent, num_elem, elem_size,
                                          num_comp, num_elem*num_comp*elem_size,
                                          strides, &rstr_qd_i); CeedChk(ierr);
  // -- QFunction
  CeedQFunction qf_fdm;
  ierr = CeedQFunctionCreateInteriorByName(ceed_parent, "Scale", &qf_fdm);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf_fdm, "input", num_comp, CEED_EVAL_INTERP);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qf_fdm, "scale", num_comp, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qf_fdm, "output", num_comp, CEED_EVAL_INTERP);
  CeedChk(ierr);
  ierr = CeedQFunctionSetUserFlopsEstimate(qf_fdm, num_comp); CeedChk(ierr);
  // -- QFunction context
  CeedInt *num_comp_data;
  ierr = CeedCalloc(1, &num_comp_data); CeedChk(ierr);
  num_comp_data[0] = num_comp;
  CeedQFunctionContext ctx_fdm;
  ierr = CeedQFunctionContextCreate(ceed, &ctx_fdm); CeedChk(ierr);
  ierr = CeedQFunctionContextSetData(ctx_fdm, CEED_MEM_HOST, CEED_OWN_POINTER,
                                     sizeof(*num_comp_data), num_comp_data);
  CeedChk(ierr);
  ierr = CeedQFunctionSetContext(qf_fdm, ctx_fdm); CeedChk(ierr);
  ierr = CeedQFunctionContextDestroy(&ctx_fdm); CeedChk(ierr);
  // -- Operator
  ierr = CeedOperatorCreate(ceed_parent, qf_fdm, NULL, NULL, fdm_inv);
  CeedChk(ierr);
  CeedOperatorSetField(*fdm_inv, "input", rstr, fdm_basis, CEED_VECTOR_ACTIVE);
  CeedChk(ierr);
  CeedOperatorSetField(*fdm_inv, "scale", rstr_qd_i, CEED_BASIS_COLLOCATED,
                       q_data); CeedChk(ierr);
  CeedOperatorSetField(*fdm_inv, "output", rstr, fdm_basis, CEED_VECTOR_ACTIVE);
  CeedChk(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&q_data); CeedChk(ierr);
  ierr = CeedBasisDestroy(&fdm_basis); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_qd_i); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&qf_fdm); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/// @}
