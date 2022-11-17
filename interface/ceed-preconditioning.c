// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <assert.h>
#include <ceed-impl.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>
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
  @brief Duplicate a CeedQFunction with a reference Ceed to fallback for advanced
         CeedOperator functionality

  @param[in] fallback_ceed Ceed on which to create fallback CeedQFunction
  @param[in] qf            CeedQFunction to create fallback for
  @param[out] qf_fallback  fallback CeedQFunction

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedQFunctionCreateFallback(Ceed fallback_ceed, CeedQFunction qf, CeedQFunction *qf_fallback) {
  // Check if NULL qf passed in
  if (!qf) return CEED_ERROR_SUCCESS;

  CeedDebug256(qf->ceed, 1, "---------- CeedOperator Fallback ----------\n");
  CeedDebug(qf->ceed, "Creating fallback CeedQFunction\n");

  char *source_path_with_name = "";
  if (qf->source_path) {
    size_t path_len = strlen(qf->source_path), name_len = strlen(qf->kernel_name);
    CeedCall(CeedCalloc(path_len + name_len + 2, &source_path_with_name));
    memcpy(source_path_with_name, qf->source_path, path_len);
    memcpy(&source_path_with_name[path_len], ":", 1);
    memcpy(&source_path_with_name[path_len + 1], qf->kernel_name, name_len);
  } else {
    CeedCall(CeedCalloc(1, &source_path_with_name));
  }

  CeedCall(CeedQFunctionCreateInterior(fallback_ceed, qf->vec_length, qf->function, source_path_with_name, qf_fallback));
  {
    CeedQFunctionContext ctx;

    CeedCall(CeedQFunctionGetContext(qf, &ctx));
    CeedCall(CeedQFunctionSetContext(*qf_fallback, ctx));
  }
  for (CeedInt i = 0; i < qf->num_input_fields; i++) {
    CeedCall(CeedQFunctionAddInput(*qf_fallback, qf->input_fields[i]->field_name, qf->input_fields[i]->size, qf->input_fields[i]->eval_mode));
  }
  for (CeedInt i = 0; i < qf->num_output_fields; i++) {
    CeedCall(CeedQFunctionAddOutput(*qf_fallback, qf->output_fields[i]->field_name, qf->output_fields[i]->size, qf->output_fields[i]->eval_mode));
  }
  CeedCall(CeedFree(&source_path_with_name));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Duplicate a CeedOperator with a reference Ceed to fallback for advanced
         CeedOperator functionality

  @param op  CeedOperator to create fallback for

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCreateFallback(CeedOperator op) {
  Ceed ceed_fallback;

  // Check not already created
  if (op->op_fallback) return CEED_ERROR_SUCCESS;

  // Fallback Ceed
  CeedCall(CeedGetOperatorFallbackCeed(op->ceed, &ceed_fallback));
  if (!ceed_fallback) return CEED_ERROR_SUCCESS;

  CeedDebug256(op->ceed, 1, "---------- CeedOperator Fallback ----------\n");
  CeedDebug(op->ceed, "Creating fallback CeedOperator\n");

  // Clone Op
  CeedOperator op_fallback;
  if (op->is_composite) {
    CeedCall(CeedCompositeOperatorCreate(ceed_fallback, &op_fallback));
    for (CeedInt i = 0; i < op->num_suboperators; i++) {
      CeedOperator op_sub_fallback;

      CeedCall(CeedOperatorGetFallback(op->sub_operators[i], &op_sub_fallback));
      CeedCall(CeedCompositeOperatorAddSub(op_fallback, op_sub_fallback));
    }
  } else {
    CeedQFunction qf_fallback = NULL, dqf_fallback = NULL, dqfT_fallback = NULL;
    CeedCall(CeedQFunctionCreateFallback(ceed_fallback, op->qf, &qf_fallback));
    CeedCall(CeedQFunctionCreateFallback(ceed_fallback, op->dqf, &dqf_fallback));
    CeedCall(CeedQFunctionCreateFallback(ceed_fallback, op->dqfT, &dqfT_fallback));
    CeedCall(CeedOperatorCreate(ceed_fallback, qf_fallback, dqf_fallback, dqfT_fallback, &op_fallback));
    for (CeedInt i = 0; i < op->qf->num_input_fields; i++) {
      CeedCall(CeedOperatorSetField(op_fallback, op->input_fields[i]->field_name, op->input_fields[i]->elem_restr, op->input_fields[i]->basis,
                                    op->input_fields[i]->vec));
    }
    for (CeedInt i = 0; i < op->qf->num_output_fields; i++) {
      CeedCall(CeedOperatorSetField(op_fallback, op->output_fields[i]->field_name, op->output_fields[i]->elem_restr, op->output_fields[i]->basis,
                                    op->output_fields[i]->vec));
    }
    CeedCall(CeedQFunctionAssemblyDataReferenceCopy(op->qf_assembled, &op_fallback->qf_assembled));
    if (op_fallback->num_qpts == 0) {
      CeedCall(CeedOperatorSetNumQuadraturePoints(op_fallback, op->num_qpts));
    }
    // Cleanup
    CeedCall(CeedQFunctionDestroy(&qf_fallback));
    CeedCall(CeedQFunctionDestroy(&dqf_fallback));
    CeedCall(CeedQFunctionDestroy(&dqfT_fallback));
  }
  CeedCall(CeedOperatorSetName(op_fallback, op->name));
  CeedCall(CeedOperatorCheckReady(op_fallback));
  op->op_fallback = op_fallback;

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retreive fallback CeedOperator with a reference Ceed for advanced CeedOperator functionality

  @param[in] op            CeedOperator to retrieve fallback for
  @param[out] op_fallback  Fallback CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedOperatorGetFallback(CeedOperator op, CeedOperator *op_fallback) {
  // Create if needed
  if (!op->op_fallback) {
    CeedCall(CeedOperatorCreateFallback(op));
  }
  if (op->op_fallback) {
    bool is_debug;

    CeedCall(CeedIsDebug(op->ceed, &is_debug));
    if (is_debug) {
      Ceed        ceed_fallback;
      const char *resource, *resource_fallback;

      CeedCall(CeedGetOperatorFallbackCeed(op->ceed, &ceed_fallback));
      CeedCall(CeedGetResource(op->ceed, &resource));
      CeedCall(CeedGetResource(ceed_fallback, &resource_fallback));

      CeedDebug256(op->ceed, 1, "---------- CeedOperator Fallback ----------\n");
      CeedDebug(op->ceed, "Falling back from %s operator at address %ld to %s operator at address %ld\n", resource, op, resource_fallback,
                op->op_fallback);
    }
  }
  *op_fallback = op->op_fallback;

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
static inline void CeedOperatorGetBasisPointer(CeedEvalMode eval_mode, const CeedScalar *identity, const CeedScalar *interp, const CeedScalar *grad,
                                               const CeedScalar **basis_ptr) {
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
      break;  // Caught by QF Assembly
  }
  assert(*basis_ptr != NULL);
}

/**
  @brief Create point block restriction for active operator field

  @param[in] rstr              Original CeedElemRestriction for active field
  @param[out] pointblock_rstr  Address of the variable where the newly created
                                 CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCreateActivePointBlockRestriction(CeedElemRestriction rstr, CeedElemRestriction *pointblock_rstr) {
  Ceed ceed;
  CeedCall(CeedElemRestrictionGetCeed(rstr, &ceed));
  const CeedInt *offsets;
  CeedCall(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));

  // Expand offsets
  CeedInt  num_elem, num_comp, elem_size, comp_stride, *pointblock_offsets;
  CeedSize l_size;
  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCall(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCall(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  CeedCall(CeedElemRestrictionGetCompStride(rstr, &comp_stride));
  CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
  CeedInt shift = num_comp;
  if (comp_stride != 1) shift *= num_comp;
  CeedCall(CeedCalloc(num_elem * elem_size, &pointblock_offsets));
  for (CeedInt i = 0; i < num_elem * elem_size; i++) {
    pointblock_offsets[i] = offsets[i] * shift;
  }

  // Create new restriction
  CeedCall(CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp * num_comp, 1, l_size * num_comp, CEED_MEM_HOST, CEED_OWN_POINTER,
                                     pointblock_offsets, pointblock_rstr));

  // Cleanup
  CeedCall(CeedElemRestrictionRestoreOffsets(rstr, &offsets));

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
static inline int CeedSingleOperatorAssembleAddDiagonal_Core(CeedOperator op, CeedRequest *request, const bool is_pointblock, CeedVector assembled) {
  Ceed ceed;
  CeedCall(CeedOperatorGetCeed(op, &ceed));

  // Assemble QFunction
  CeedQFunction qf;
  CeedCall(CeedOperatorGetQFunction(op, &qf));
  CeedInt num_input_fields, num_output_fields;
  CeedCall(CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields));
  CeedVector          assembled_qf;
  CeedElemRestriction rstr;
  CeedCall(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &rstr, request));
  CeedInt layout[3];
  CeedCall(CeedElemRestrictionGetELayout(rstr, &layout));
  CeedCall(CeedElemRestrictionDestroy(&rstr));

  // Get assembly data
  CeedOperatorAssemblyData data;
  CeedCall(CeedOperatorGetOperatorAssemblyData(op, &data));
  const CeedEvalMode *eval_mode_in, *eval_mode_out;
  CeedInt             num_eval_mode_in, num_eval_mode_out;
  CeedCall(CeedOperatorAssemblyDataGetEvalModes(data, &num_eval_mode_in, &eval_mode_in, &num_eval_mode_out, &eval_mode_out));
  CeedBasis basis_in, basis_out;
  CeedCall(CeedOperatorAssemblyDataGetBases(data, &basis_in, NULL, &basis_out, NULL));
  CeedInt num_comp;
  CeedCall(CeedBasisGetNumComponents(basis_in, &num_comp));

  // Assemble point block diagonal restriction, if needed
  CeedElemRestriction diag_rstr;
  CeedCall(CeedOperatorGetActiveElemRestriction(op, &diag_rstr));
  if (is_pointblock) {
    CeedElemRestriction point_block_rstr;
    CeedCall(CeedOperatorCreateActivePointBlockRestriction(diag_rstr, &point_block_rstr));
    diag_rstr = point_block_rstr;
  }

  // Create diagonal vector
  CeedVector elem_diag;
  CeedCall(CeedElemRestrictionCreateVector(diag_rstr, NULL, &elem_diag));

  // Assemble element operator diagonals
  CeedScalar       *elem_diag_array;
  const CeedScalar *assembled_qf_array;
  CeedCall(CeedVectorSetValue(elem_diag, 0.0));
  CeedCall(CeedVectorGetArray(elem_diag, CEED_MEM_HOST, &elem_diag_array));
  CeedCall(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_HOST, &assembled_qf_array));
  CeedInt num_elem, num_nodes, num_qpts;
  CeedCall(CeedElemRestrictionGetNumElements(diag_rstr, &num_elem));
  CeedCall(CeedBasisGetNumNodes(basis_in, &num_nodes));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts));

  // Basis matrices
  const CeedScalar *interp_in, *interp_out, *grad_in, *grad_out;
  CeedScalar       *identity      = NULL;
  bool              has_eval_none = false;
  for (CeedInt i = 0; i < num_eval_mode_in; i++) {
    has_eval_none = has_eval_none || (eval_mode_in[i] == CEED_EVAL_NONE);
  }
  for (CeedInt i = 0; i < num_eval_mode_out; i++) {
    has_eval_none = has_eval_none || (eval_mode_out[i] == CEED_EVAL_NONE);
  }
  if (has_eval_none) {
    CeedCall(CeedCalloc(num_qpts * num_nodes, &identity));
    for (CeedInt i = 0; i < (num_nodes < num_qpts ? num_nodes : num_qpts); i++) identity[i * num_nodes + i] = 1.0;
  }
  CeedCall(CeedBasisGetInterp(basis_in, &interp_in));
  CeedCall(CeedBasisGetInterp(basis_out, &interp_out));
  CeedCall(CeedBasisGetGrad(basis_in, &grad_in));
  CeedCall(CeedBasisGetGrad(basis_out, &grad_out));
  // Compute the diagonal of B^T D B
  // Each element
  for (CeedInt e = 0; e < num_elem; e++) {
    CeedInt d_out = -1;
    // Each basis eval mode pair
    for (CeedInt e_out = 0; e_out < num_eval_mode_out; e_out++) {
      const CeedScalar *bt = NULL;
      if (eval_mode_out[e_out] == CEED_EVAL_GRAD) d_out += 1;
      CeedOperatorGetBasisPointer(eval_mode_out[e_out], identity, interp_out, &grad_out[d_out * num_qpts * num_nodes], &bt);
      CeedInt d_in = -1;
      for (CeedInt e_in = 0; e_in < num_eval_mode_in; e_in++) {
        const CeedScalar *b = NULL;
        if (eval_mode_in[e_in] == CEED_EVAL_GRAD) d_in += 1;
        CeedOperatorGetBasisPointer(eval_mode_in[e_in], identity, interp_in, &grad_in[d_in * num_qpts * num_nodes], &b);
        // Each component
        for (CeedInt c_out = 0; c_out < num_comp; c_out++) {
          // Each qpoint/node pair
          for (CeedInt q = 0; q < num_qpts; q++) {
            if (is_pointblock) {
              // Point Block Diagonal
              for (CeedInt c_in = 0; c_in < num_comp; c_in++) {
                const CeedScalar qf_value =
                    assembled_qf_array[q * layout[0] + (((e_in * num_comp + c_in) * num_eval_mode_out + e_out) * num_comp + c_out) * layout[1] +
                                       e * layout[2]];
                for (CeedInt n = 0; n < num_nodes; n++) {
                  elem_diag_array[((e * num_comp + c_out) * num_comp + c_in) * num_nodes + n] +=
                      bt[q * num_nodes + n] * qf_value * b[q * num_nodes + n];
                }
              }
            } else {
              // Diagonal Only
              const CeedScalar qf_value =
                  assembled_qf_array[q * layout[0] + (((e_in * num_comp + c_out) * num_eval_mode_out + e_out) * num_comp + c_out) * layout[1] +
                                     e * layout[2]];
              for (CeedInt n = 0; n < num_nodes; n++) {
                elem_diag_array[(e * num_comp + c_out) * num_nodes + n] += bt[q * num_nodes + n] * qf_value * b[q * num_nodes + n];
              }
            }
          }
        }
      }
    }
  }
  CeedCall(CeedVectorRestoreArray(elem_diag, &elem_diag_array));
  CeedCall(CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array));

  // Assemble local operator diagonal
  CeedCall(CeedElemRestrictionApply(diag_rstr, CEED_TRANSPOSE, elem_diag, assembled, request));

  // Cleanup
  if (is_pointblock) {
    CeedCall(CeedElemRestrictionDestroy(&diag_rstr));
  }
  CeedCall(CeedVectorDestroy(&assembled_qf));
  CeedCall(CeedVectorDestroy(&elem_diag));
  CeedCall(CeedFree(&identity));

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
static inline int CeedCompositeOperatorLinearAssembleAddDiagonal(CeedOperator op, CeedRequest *request, const bool is_pointblock,
                                                                 CeedVector assembled) {
  CeedInt       num_sub;
  CeedOperator *suboperators;
  CeedCall(CeedOperatorGetNumSub(op, &num_sub));
  CeedCall(CeedOperatorGetSubList(op, &suboperators));
  for (CeedInt i = 0; i < num_sub; i++) {
    if (is_pointblock) {
      CeedCall(CeedOperatorLinearAssembleAddPointBlockDiagonal(suboperators[i], assembled, request));
    } else {
      CeedCall(CeedOperatorLinearAssembleAddDiagonal(suboperators[i], assembled, request));
    }
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
static int CeedSingleOperatorAssembleSymbolic(CeedOperator op, CeedInt offset, CeedInt *rows, CeedInt *cols) {
  Ceed ceed = op->ceed;
  if (op->is_composite) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Composite operator not supported");
    // LCOV_EXCL_STOP
  }

  CeedSize num_nodes;
  CeedCall(CeedOperatorGetActiveVectorLengths(op, &num_nodes, NULL));
  CeedElemRestriction rstr_in;
  CeedCall(CeedOperatorGetActiveElemRestriction(op, &rstr_in));
  CeedInt num_elem, elem_size, num_comp;
  CeedCall(CeedElemRestrictionGetNumElements(rstr_in, &num_elem));
  CeedCall(CeedElemRestrictionGetElementSize(rstr_in, &elem_size));
  CeedCall(CeedElemRestrictionGetNumComponents(rstr_in, &num_comp));
  CeedInt layout_er[3];
  CeedCall(CeedElemRestrictionGetELayout(rstr_in, &layout_er));

  CeedInt local_num_entries = elem_size * num_comp * elem_size * num_comp * num_elem;

  // Determine elem_dof relation
  CeedVector index_vec;
  CeedCall(CeedVectorCreate(ceed, num_nodes, &index_vec));
  CeedScalar *array;
  CeedCall(CeedVectorGetArrayWrite(index_vec, CEED_MEM_HOST, &array));
  for (CeedInt i = 0; i < num_nodes; i++) array[i] = i;
  CeedCall(CeedVectorRestoreArray(index_vec, &array));
  CeedVector elem_dof;
  CeedCall(CeedVectorCreate(ceed, num_elem * elem_size * num_comp, &elem_dof));
  CeedCall(CeedVectorSetValue(elem_dof, 0.0));
  CeedCall(CeedElemRestrictionApply(rstr_in, CEED_NOTRANSPOSE, index_vec, elem_dof, CEED_REQUEST_IMMEDIATE));
  const CeedScalar *elem_dof_a;
  CeedCall(CeedVectorGetArrayRead(elem_dof, CEED_MEM_HOST, &elem_dof_a));
  CeedCall(CeedVectorDestroy(&index_vec));

  // Determine i, j locations for element matrices
  CeedInt count = 0;
  for (CeedInt e = 0; e < num_elem; e++) {
    for (CeedInt comp_in = 0; comp_in < num_comp; comp_in++) {
      for (CeedInt comp_out = 0; comp_out < num_comp; comp_out++) {
        for (CeedInt i = 0; i < elem_size; i++) {
          for (CeedInt j = 0; j < elem_size; j++) {
            const CeedInt elem_dof_index_row = i * layout_er[0] + (comp_out)*layout_er[1] + e * layout_er[2];
            const CeedInt elem_dof_index_col = j * layout_er[0] + comp_in * layout_er[1] + e * layout_er[2];

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
  if (count != local_num_entries) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MAJOR, "Error computing assembled entries");
    // LCOV_EXCL_STOP
  }
  CeedCall(CeedVectorRestoreArrayRead(elem_dof, &elem_dof_a));
  CeedCall(CeedVectorDestroy(&elem_dof));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble nonzero entries for non-composite operator

  Users should generally use CeedOperatorLinearAssemble()

  @param[in] op       CeedOperator to assemble
  @param[in] offset   Offest for number of entries
  @param[out] values  Values to assemble into matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedSingleOperatorAssemble(CeedOperator op, CeedInt offset, CeedVector values) {
  Ceed ceed = op->ceed;
  if (op->is_composite) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Composite operator not supported");
    // LCOV_EXCL_STOP
  }
  if (op->num_elem == 0) return CEED_ERROR_SUCCESS;

  if (op->LinearAssembleSingle) {
    // Backend version
    CeedCall(op->LinearAssembleSingle(op, offset, values));
    return CEED_ERROR_SUCCESS;
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedSingleOperatorAssemble(op_fallback, offset, values));
      return CEED_ERROR_SUCCESS;
    }
  }

  // Assemble QFunction
  CeedQFunction qf;
  CeedCall(CeedOperatorGetQFunction(op, &qf));
  CeedVector          assembled_qf;
  CeedElemRestriction rstr_q;
  CeedCall(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &rstr_q, CEED_REQUEST_IMMEDIATE));
  CeedSize qf_length;
  CeedCall(CeedVectorGetLength(assembled_qf, &qf_length));

  CeedInt            num_input_fields, num_output_fields;
  CeedOperatorField *input_fields;
  CeedOperatorField *output_fields;
  CeedCall(CeedOperatorGetFields(op, &num_input_fields, &input_fields, &num_output_fields, &output_fields));

  // Get assembly data
  CeedOperatorAssemblyData data;
  CeedCall(CeedOperatorGetOperatorAssemblyData(op, &data));
  const CeedEvalMode *eval_mode_in, *eval_mode_out;
  CeedInt             num_eval_mode_in, num_eval_mode_out;
  CeedCall(CeedOperatorAssemblyDataGetEvalModes(data, &num_eval_mode_in, &eval_mode_in, &num_eval_mode_out, &eval_mode_out));
  CeedBasis basis_in, basis_out;
  CeedCall(CeedOperatorAssemblyDataGetBases(data, &basis_in, NULL, &basis_out, NULL));

  if (num_eval_mode_in == 0 || num_eval_mode_out == 0) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Cannot assemble operator with out inputs/outputs");
    // LCOV_EXCL_STOP
  }

  CeedElemRestriction active_rstr;
  CeedInt             num_elem, elem_size, num_qpts, num_comp;
  CeedCall(CeedOperatorGetActiveElemRestriction(op, &active_rstr));
  CeedCall(CeedElemRestrictionGetNumElements(active_rstr, &num_elem));
  CeedCall(CeedElemRestrictionGetElementSize(active_rstr, &elem_size));
  CeedCall(CeedElemRestrictionGetNumComponents(active_rstr, &num_comp));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts));

  CeedInt local_num_entries = elem_size * num_comp * elem_size * num_comp * num_elem;

  // loop over elements and put in data structure
  const CeedScalar *interp_in, *grad_in;
  CeedCall(CeedBasisGetInterp(basis_in, &interp_in));
  CeedCall(CeedBasisGetGrad(basis_in, &grad_in));

  const CeedScalar *assembled_qf_array;
  CeedCall(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_HOST, &assembled_qf_array));

  CeedInt layout_qf[3];
  CeedCall(CeedElemRestrictionGetELayout(rstr_q, &layout_qf));
  CeedCall(CeedElemRestrictionDestroy(&rstr_q));

  // we store B_mat_in, B_mat_out, BTD, elem_mat in row-major order
  const CeedScalar *B_mat_in, *B_mat_out;
  CeedCall(CeedOperatorAssemblyDataGetBases(data, NULL, &B_mat_in, NULL, &B_mat_out));
  CeedScalar  BTD_mat[elem_size * num_qpts * num_eval_mode_in];
  CeedScalar  elem_mat[elem_size * elem_size];
  CeedInt     count = 0;
  CeedScalar *vals;
  CeedCall(CeedVectorGetArrayWrite(values, CEED_MEM_HOST, &vals));
  for (CeedInt e = 0; e < num_elem; e++) {
    for (CeedInt comp_in = 0; comp_in < num_comp; comp_in++) {
      for (CeedInt comp_out = 0; comp_out < num_comp; comp_out++) {
        // Compute B^T*D
        for (CeedInt n = 0; n < elem_size; n++) {
          for (CeedInt q = 0; q < num_qpts; q++) {
            for (CeedInt e_in = 0; e_in < num_eval_mode_in; e_in++) {
              const CeedInt btd_index = n * (num_qpts * num_eval_mode_in) + (num_eval_mode_in * q + e_in);
              CeedScalar    sum       = 0.0;
              for (CeedInt e_out = 0; e_out < num_eval_mode_out; e_out++) {
                const CeedInt b_out_index     = (num_eval_mode_out * q + e_out) * elem_size + n;
                const CeedInt eval_mode_index = ((e_in * num_comp + comp_in) * num_eval_mode_out + e_out) * num_comp + comp_out;
                const CeedInt qf_index        = q * layout_qf[0] + eval_mode_index * layout_qf[1] + e * layout_qf[2];
                sum += B_mat_out[b_out_index] * assembled_qf_array[qf_index];
              }
              BTD_mat[btd_index] = sum;
            }
          }
        }
        // form element matrix itself (for each block component)
        CeedCall(CeedMatrixMatrixMultiply(ceed, BTD_mat, B_mat_in, elem_mat, elem_size, elem_size, num_qpts * num_eval_mode_in));

        // put element matrix in coordinate data structure
        for (CeedInt i = 0; i < elem_size; i++) {
          for (CeedInt j = 0; j < elem_size; j++) {
            vals[offset + count] = elem_mat[i * elem_size + j];
            count++;
          }
        }
      }
    }
  }
  if (count != local_num_entries) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MAJOR, "Error computing entries");
    // LCOV_EXCL_STOP
  }
  CeedCall(CeedVectorRestoreArray(values, &vals));

  CeedCall(CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array));
  CeedCall(CeedVectorDestroy(&assembled_qf));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Count number of entries for assembled CeedOperator

  @param[in] op            CeedOperator to assemble
  @param[out] num_entries  Number of entries in assembled representation

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedSingleOperatorAssemblyCountEntries(CeedOperator op, CeedInt *num_entries) {
  CeedElemRestriction rstr;
  CeedInt             num_elem, elem_size, num_comp;

  if (op->is_composite) {
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED, "Composite operator not supported");
    // LCOV_EXCL_STOP
  }
  CeedCall(CeedOperatorGetActiveElemRestriction(op, &rstr));
  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCall(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  CeedCall(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  *num_entries = elem_size * num_comp * elem_size * num_comp * num_elem;

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
static int CeedSingleOperatorMultigridLevel(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
                                            CeedBasis basis_c_to_f, CeedOperator *op_coarse, CeedOperator *op_prolong, CeedOperator *op_restrict) {
  Ceed ceed;
  CeedCall(CeedOperatorGetCeed(op_fine, &ceed));

  // Check for composite operator
  bool is_composite;
  CeedCall(CeedOperatorIsComposite(op_fine, &is_composite));
  if (is_composite) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Automatic multigrid setup for composite operators not supported");
    // LCOV_EXCL_STOP
  }

  // Coarse Grid
  CeedCall(CeedOperatorCreate(ceed, op_fine->qf, op_fine->dqf, op_fine->dqfT, op_coarse));
  CeedElemRestriction rstr_fine = NULL;
  // -- Clone input fields
  for (CeedInt i = 0; i < op_fine->qf->num_input_fields; i++) {
    if (op_fine->input_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      rstr_fine = op_fine->input_fields[i]->elem_restr;
      CeedCall(CeedOperatorSetField(*op_coarse, op_fine->input_fields[i]->field_name, rstr_coarse, basis_coarse, CEED_VECTOR_ACTIVE));
    } else {
      CeedCall(CeedOperatorSetField(*op_coarse, op_fine->input_fields[i]->field_name, op_fine->input_fields[i]->elem_restr,
                                    op_fine->input_fields[i]->basis, op_fine->input_fields[i]->vec));
    }
  }
  // -- Clone output fields
  for (CeedInt i = 0; i < op_fine->qf->num_output_fields; i++) {
    if (op_fine->output_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      CeedCall(CeedOperatorSetField(*op_coarse, op_fine->output_fields[i]->field_name, rstr_coarse, basis_coarse, CEED_VECTOR_ACTIVE));
    } else {
      CeedCall(CeedOperatorSetField(*op_coarse, op_fine->output_fields[i]->field_name, op_fine->output_fields[i]->elem_restr,
                                    op_fine->output_fields[i]->basis, op_fine->output_fields[i]->vec));
    }
  }
  // -- Clone QFunctionAssemblyData
  CeedCall(CeedQFunctionAssemblyDataReferenceCopy(op_fine->qf_assembled, &(*op_coarse)->qf_assembled));

  // Multiplicity vector
  CeedVector mult_vec, mult_e_vec;
  CeedCall(CeedElemRestrictionCreateVector(rstr_fine, &mult_vec, &mult_e_vec));
  CeedCall(CeedVectorSetValue(mult_e_vec, 0.0));
  CeedCall(CeedElemRestrictionApply(rstr_fine, CEED_NOTRANSPOSE, p_mult_fine, mult_e_vec, CEED_REQUEST_IMMEDIATE));
  CeedCall(CeedVectorSetValue(mult_vec, 0.0));
  CeedCall(CeedElemRestrictionApply(rstr_fine, CEED_TRANSPOSE, mult_e_vec, mult_vec, CEED_REQUEST_IMMEDIATE));
  CeedCall(CeedVectorDestroy(&mult_e_vec));
  CeedCall(CeedVectorReciprocal(mult_vec));

  // Restriction
  CeedInt num_comp;
  CeedCall(CeedBasisGetNumComponents(basis_coarse, &num_comp));
  CeedQFunction qf_restrict;
  CeedCall(CeedQFunctionCreateInteriorByName(ceed, "Scale", &qf_restrict));
  CeedInt *num_comp_r_data;
  CeedCall(CeedCalloc(1, &num_comp_r_data));
  num_comp_r_data[0] = num_comp;
  CeedQFunctionContext ctx_r;
  CeedCall(CeedQFunctionContextCreate(ceed, &ctx_r));
  CeedCall(CeedQFunctionContextSetData(ctx_r, CEED_MEM_HOST, CEED_OWN_POINTER, sizeof(*num_comp_r_data), num_comp_r_data));
  CeedCall(CeedQFunctionSetContext(qf_restrict, ctx_r));
  CeedCall(CeedQFunctionContextDestroy(&ctx_r));
  CeedCall(CeedQFunctionAddInput(qf_restrict, "input", num_comp, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddInput(qf_restrict, "scale", num_comp, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf_restrict, "output", num_comp, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf_restrict, num_comp));

  CeedCall(CeedOperatorCreate(ceed, qf_restrict, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op_restrict));
  CeedCall(CeedOperatorSetField(*op_restrict, "input", rstr_fine, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  CeedCall(CeedOperatorSetField(*op_restrict, "scale", rstr_fine, CEED_BASIS_COLLOCATED, mult_vec));
  CeedCall(CeedOperatorSetField(*op_restrict, "output", rstr_coarse, basis_c_to_f, CEED_VECTOR_ACTIVE));

  // Prolongation
  CeedQFunction qf_prolong;
  CeedCall(CeedQFunctionCreateInteriorByName(ceed, "Scale", &qf_prolong));
  CeedInt *num_comp_p_data;
  CeedCall(CeedCalloc(1, &num_comp_p_data));
  num_comp_p_data[0] = num_comp;
  CeedQFunctionContext ctx_p;
  CeedCall(CeedQFunctionContextCreate(ceed, &ctx_p));
  CeedCall(CeedQFunctionContextSetData(ctx_p, CEED_MEM_HOST, CEED_OWN_POINTER, sizeof(*num_comp_p_data), num_comp_p_data));
  CeedCall(CeedQFunctionSetContext(qf_prolong, ctx_p));
  CeedCall(CeedQFunctionContextDestroy(&ctx_p));
  CeedCall(CeedQFunctionAddInput(qf_prolong, "input", num_comp, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionAddInput(qf_prolong, "scale", num_comp, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf_prolong, "output", num_comp, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf_prolong, num_comp));

  CeedCall(CeedOperatorCreate(ceed, qf_prolong, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op_prolong));
  CeedCall(CeedOperatorSetField(*op_prolong, "input", rstr_coarse, basis_c_to_f, CEED_VECTOR_ACTIVE));
  CeedCall(CeedOperatorSetField(*op_prolong, "scale", rstr_fine, CEED_BASIS_COLLOCATED, mult_vec));
  CeedCall(CeedOperatorSetField(*op_prolong, "output", rstr_fine, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));

  // Clone name
  bool   has_name = op_fine->name;
  size_t name_len = op_fine->name ? strlen(op_fine->name) : 0;
  CeedCall(CeedOperatorSetName(*op_coarse, op_fine->name));
  {
    char *prolongation_name;
    CeedCall(CeedCalloc(18 + name_len, &prolongation_name));
    sprintf(prolongation_name, "prolongation%s%s", has_name ? " for " : "", has_name ? op_fine->name : "");
    CeedCall(CeedOperatorSetName(*op_prolong, prolongation_name));
    CeedCall(CeedFree(&prolongation_name));
  }
  {
    char *restriction_name;
    CeedCall(CeedCalloc(17 + name_len, &restriction_name));
    sprintf(restriction_name, "restriction%s%s", has_name ? " for " : "", has_name ? op_fine->name : "");
    CeedCall(CeedOperatorSetName(*op_restrict, restriction_name));
    CeedCall(CeedFree(&restriction_name));
  }

  // Cleanup
  CeedCall(CeedVectorDestroy(&mult_vec));
  CeedCall(CeedBasisDestroy(&basis_c_to_f));
  CeedCall(CeedQFunctionDestroy(&qf_restrict));
  CeedCall(CeedQFunctionDestroy(&qf_prolong));

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
CeedPragmaOptimizeOff static int CeedBuildMassLaplace(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *q_weight_1d,
                                                      CeedInt P_1d, CeedInt Q_1d, CeedInt dim, CeedScalar *mass, CeedScalar *laplace) {
  for (CeedInt i = 0; i < P_1d; i++) {
    for (CeedInt j = 0; j < P_1d; j++) {
      CeedScalar sum = 0.0;
      for (CeedInt k = 0; k < Q_1d; k++) sum += interp_1d[k * P_1d + i] * q_weight_1d[k] * interp_1d[k * P_1d + j];
      mass[i + j * P_1d] = sum;
    }
  }
  // -- Laplacian
  for (CeedInt i = 0; i < P_1d; i++) {
    for (CeedInt j = 0; j < P_1d; j++) {
      CeedScalar sum = 0.0;
      for (CeedInt k = 0; k < Q_1d; k++) sum += grad_1d[k * P_1d + i] * q_weight_1d[k] * grad_1d[k * P_1d + j];
      laplace[i + j * P_1d] = sum;
    }
  }
  CeedScalar perturbation = dim > 2 ? 1e-6 : 1e-4;
  for (CeedInt i = 0; i < P_1d; i++) laplace[i + P_1d * i] += perturbation;
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
    int
    CeedQFunctionAssemblyDataCreate(Ceed ceed, CeedQFunctionAssemblyData *data) {
  CeedCall(CeedCalloc(1, data));
  (*data)->ref_count = 1;
  (*data)->ceed      = ceed;
  CeedCall(CeedReference(ceed));

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
int CeedQFunctionAssemblyDataSetReuse(CeedQFunctionAssemblyData data, bool reuse_data) {
  data->reuse_data        = reuse_data;
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
int CeedQFunctionAssemblyDataSetUpdateNeeded(CeedQFunctionAssemblyData data, bool needs_data_update) {
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
int CeedQFunctionAssemblyDataIsUpdateNeeded(CeedQFunctionAssemblyData data, bool *is_update_needed) {
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
int CeedQFunctionAssemblyDataReferenceCopy(CeedQFunctionAssemblyData data, CeedQFunctionAssemblyData *data_copy) {
  CeedCall(CeedQFunctionAssemblyDataReference(data));
  CeedCall(CeedQFunctionAssemblyDataDestroy(data_copy));
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
int CeedQFunctionAssemblyDataIsSetup(CeedQFunctionAssemblyData data, bool *is_setup) {
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
int CeedQFunctionAssemblyDataSetObjects(CeedQFunctionAssemblyData data, CeedVector vec, CeedElemRestriction rstr) {
  CeedCall(CeedVectorReferenceCopy(vec, &data->vec));
  CeedCall(CeedElemRestrictionReferenceCopy(rstr, &data->rstr));

  data->is_setup = true;
  return CEED_ERROR_SUCCESS;
}

int CeedQFunctionAssemblyDataGetObjects(CeedQFunctionAssemblyData data, CeedVector *vec, CeedElemRestriction *rstr) {
  if (!data->is_setup) {
    // LCOV_EXCL_START
    return CeedError(data->ceed, CEED_ERROR_INCOMPLETE, "Internal objects not set; must call CeedQFunctionAssemblyDataSetObjects first.");
    // LCOV_EXCL_STOP
  }

  CeedCall(CeedVectorReferenceCopy(data->vec, vec));
  CeedCall(CeedElemRestrictionReferenceCopy(data->rstr, rstr));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy CeedQFunctionAssemblyData

  @param[out] data  CeedQFunctionAssemblyData to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataDestroy(CeedQFunctionAssemblyData *data) {
  if (!*data || --(*data)->ref_count > 0) return CEED_ERROR_SUCCESS;

  CeedCall(CeedDestroy(&(*data)->ceed));
  CeedCall(CeedVectorDestroy(&(*data)->vec));
  CeedCall(CeedElemRestrictionDestroy(&(*data)->rstr));

  CeedCall(CeedFree(data));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedOperatorAssemblyData

  @param[in] op     CeedOperator to assemble
  @param[out] data  CeedQFunctionAssemblyData

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorGetOperatorAssemblyData(CeedOperator op, CeedOperatorAssemblyData *data) {
  if (!op->op_assembled) {
    CeedOperatorAssemblyData data;

    CeedCall(CeedOperatorAssemblyDataCreate(op->ceed, op, &data));
    op->op_assembled = data;
  }
  *data = op->op_assembled;

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create object holding CeedOperator assembly data

  @param[in] ceed   A Ceed object where the CeedOperatorAssemblyData will be created
  @param[in] op     CeedOperator to be assembled
  @param[out] data  Address of the variable where the newly created
                      CeedOperatorAssemblyData will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorAssemblyDataCreate(Ceed ceed, CeedOperator op, CeedOperatorAssemblyData *data) {
  CeedCall(CeedCalloc(1, data));
  (*data)->ceed = ceed;
  CeedCall(CeedReference(ceed));

  // Build OperatorAssembly data
  CeedQFunction       qf;
  CeedQFunctionField *qf_fields;
  CeedOperatorField  *op_fields;
  CeedInt             num_input_fields;
  CeedCall(CeedOperatorGetQFunction(op, &qf));
  CeedCall(CeedQFunctionGetFields(qf, &num_input_fields, &qf_fields, NULL, NULL));
  CeedCall(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));

  // Determine active input basis
  CeedInt       num_eval_mode_in = 0, dim = 1;
  CeedEvalMode *eval_mode_in = NULL;
  CeedBasis     basis_in     = NULL;
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;
    CeedCall(CeedOperatorFieldGetVector(op_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCall(CeedOperatorFieldGetBasis(op_fields[i], &basis_in));
      CeedCall(CeedBasisGetDimension(basis_in, &dim));
      CeedEvalMode eval_mode;
      CeedCall(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      switch (eval_mode) {
        case CEED_EVAL_NONE:
        case CEED_EVAL_INTERP:
          CeedCall(CeedRealloc(num_eval_mode_in + 1, &eval_mode_in));
          eval_mode_in[num_eval_mode_in] = eval_mode;
          num_eval_mode_in += 1;
          break;
        case CEED_EVAL_GRAD:
          CeedCall(CeedRealloc(num_eval_mode_in + dim, &eval_mode_in));
          for (CeedInt d = 0; d < dim; d++) {
            eval_mode_in[num_eval_mode_in + d] = eval_mode;
          }
          num_eval_mode_in += dim;
          break;
        case CEED_EVAL_WEIGHT:
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // Caught by QF Assembly
      }
    }
  }
  (*data)->num_eval_mode_in = num_eval_mode_in;
  (*data)->eval_mode_in     = eval_mode_in;
  CeedCall(CeedBasisReferenceCopy(basis_in, &(*data)->basis_in));

  // Determine active output basis
  CeedInt num_output_fields;
  CeedCall(CeedQFunctionGetFields(qf, NULL, NULL, &num_output_fields, &qf_fields));
  CeedCall(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
  CeedInt       num_eval_mode_out = 0;
  CeedEvalMode *eval_mode_out     = NULL;
  CeedBasis     basis_out         = NULL;
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector vec;
    CeedCall(CeedOperatorFieldGetVector(op_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedCall(CeedOperatorFieldGetBasis(op_fields[i], &basis_out));
      CeedEvalMode eval_mode;
      CeedCall(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      switch (eval_mode) {
        case CEED_EVAL_NONE:
        case CEED_EVAL_INTERP:
          CeedCall(CeedRealloc(num_eval_mode_out + 1, &eval_mode_out));
          eval_mode_out[num_eval_mode_out] = eval_mode;
          num_eval_mode_out += 1;
          break;
        case CEED_EVAL_GRAD:
          CeedCall(CeedRealloc(num_eval_mode_out + dim, &eval_mode_out));
          for (CeedInt d = 0; d < dim; d++) {
            eval_mode_out[num_eval_mode_out + d] = eval_mode;
          }
          num_eval_mode_out += dim;
          break;
        case CEED_EVAL_WEIGHT:
        case CEED_EVAL_DIV:
        case CEED_EVAL_CURL:
          break;  // Caught by QF Assembly
      }
    }
  }
  (*data)->num_eval_mode_out = num_eval_mode_out;
  (*data)->eval_mode_out     = eval_mode_out;
  CeedCall(CeedBasisReferenceCopy(basis_out, &(*data)->basis_out));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedOperator CeedEvalModes for assembly

  @param[in] data                CeedOperatorAssemblyData
  @param[out] num_eval_mode_in   Pointer to hold number of input CeedEvalModes, or NULL
  @param[out] eval_mode_in       Pointer to hold input CeedEvalModes, or NULL
  @param[out] num_eval_mode_out  Pointer to hold number of output CeedEvalModes, or NULL
  @param[out] eval_mode_out      Pointer to hold output CeedEvalModes, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorAssemblyDataGetEvalModes(CeedOperatorAssemblyData data, CeedInt *num_eval_mode_in, const CeedEvalMode **eval_mode_in,
                                         CeedInt *num_eval_mode_out, const CeedEvalMode **eval_mode_out) {
  if (num_eval_mode_in) *num_eval_mode_in = data->num_eval_mode_in;
  if (eval_mode_in) *eval_mode_in = data->eval_mode_in;
  if (num_eval_mode_out) *num_eval_mode_out = data->num_eval_mode_out;
  if (eval_mode_out) *eval_mode_out = data->eval_mode_out;

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedOperator CeedBasis data for assembly

  @param[in] data        CeedOperatorAssemblyData
  @param[out] basis_in   Pointer to hold active input CeedBasis, or NULL
  @param[out] B_in       Pointer to hold assembled active input B, or NULL
  @param[out] basis_out  Pointer to hold active output CeedBasis, or NULL
  @param[out] B_out      Pointer to hold assembled active output B, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorAssemblyDataGetBases(CeedOperatorAssemblyData data, CeedBasis *basis_in, const CeedScalar **B_in, CeedBasis *basis_out,
                                     const CeedScalar **B_out) {
  // Assemble B_in, B_out if needed
  if (B_in && !data->B_in) {
    CeedInt           num_qpts, elem_size;
    CeedScalar       *B_in, *identity = NULL;
    const CeedScalar *interp_in, *grad_in;
    bool              has_eval_none = false;

    CeedCall(CeedBasisGetNumQuadraturePoints(data->basis_in, &num_qpts));
    CeedCall(CeedBasisGetNumNodes(data->basis_in, &elem_size));
    CeedCall(CeedCalloc(num_qpts * elem_size * data->num_eval_mode_in, &B_in));

    for (CeedInt i = 0; i < data->num_eval_mode_in; i++) {
      has_eval_none = has_eval_none || (data->eval_mode_in[i] == CEED_EVAL_NONE);
    }
    if (has_eval_none) {
      CeedCall(CeedCalloc(num_qpts * elem_size, &identity));
      for (CeedInt i = 0; i < (elem_size < num_qpts ? elem_size : num_qpts); i++) {
        identity[i * elem_size + i] = 1.0;
      }
    }
    CeedCall(CeedBasisGetInterp(data->basis_in, &interp_in));
    CeedCall(CeedBasisGetGrad(data->basis_in, &grad_in));

    for (CeedInt q = 0; q < num_qpts; q++) {
      for (CeedInt n = 0; n < elem_size; n++) {
        CeedInt d_in = -1;
        for (CeedInt e_in = 0; e_in < data->num_eval_mode_in; e_in++) {
          const CeedInt     qq = data->num_eval_mode_in * q;
          const CeedScalar *b  = NULL;

          if (data->eval_mode_in[e_in] == CEED_EVAL_GRAD) d_in++;
          CeedOperatorGetBasisPointer(data->eval_mode_in[e_in], identity, interp_in, &grad_in[d_in * num_qpts * elem_size], &b);
          B_in[(qq + e_in) * elem_size + n] = b[q * elem_size + n];
        }
      }
    }
    data->B_in = B_in;
  }

  if (B_out && !data->B_out) {
    CeedInt           num_qpts, elem_size;
    CeedScalar       *B_out, *identity = NULL;
    const CeedScalar *interp_out, *grad_out;
    bool              has_eval_none = false;

    CeedCall(CeedBasisGetNumQuadraturePoints(data->basis_out, &num_qpts));
    CeedCall(CeedBasisGetNumNodes(data->basis_out, &elem_size));
    CeedCall(CeedCalloc(num_qpts * elem_size * data->num_eval_mode_out, &B_out));

    for (CeedInt i = 0; i < data->num_eval_mode_out; i++) {
      has_eval_none = has_eval_none || (data->eval_mode_out[i] == CEED_EVAL_NONE);
    }
    if (has_eval_none) {
      CeedCall(CeedCalloc(num_qpts * elem_size, &identity));
      for (CeedInt i = 0; i < (elem_size < num_qpts ? elem_size : num_qpts); i++) {
        identity[i * elem_size + i] = 1.0;
      }
    }
    CeedCall(CeedBasisGetInterp(data->basis_out, &interp_out));
    CeedCall(CeedBasisGetGrad(data->basis_out, &grad_out));

    for (CeedInt q = 0; q < num_qpts; q++) {
      for (CeedInt n = 0; n < elem_size; n++) {
        CeedInt d_out = -1;
        for (CeedInt e_out = 0; e_out < data->num_eval_mode_out; e_out++) {
          const CeedInt     qq = data->num_eval_mode_out * q;
          const CeedScalar *b  = NULL;

          if (data->eval_mode_out[e_out] == CEED_EVAL_GRAD) d_out++;
          CeedOperatorGetBasisPointer(data->eval_mode_out[e_out], identity, interp_out, &grad_out[d_out * num_qpts * elem_size], &b);
          B_out[(qq + e_out) * elem_size + n] = b[q * elem_size + n];
        }
      }
    }
    data->B_out = B_out;
  }

  if (basis_in) *basis_in = data->basis_in;
  if (B_in) *B_in = data->B_in;
  if (basis_out) *basis_out = data->basis_out;
  if (B_out) *B_out = data->B_out;

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy CeedOperatorAssemblyData

  @param[out] data  CeedOperatorAssemblyData to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorAssemblyDataDestroy(CeedOperatorAssemblyData *data) {
  if (!*data) return CEED_ERROR_SUCCESS;

  CeedCall(CeedDestroy(&(*data)->ceed));
  CeedCall(CeedBasisDestroy(&(*data)->basis_in));
  CeedCall(CeedBasisDestroy(&(*data)->basis_out));
  CeedCall(CeedFree(&(*data)->eval_mode_in));
  CeedCall(CeedFree(&(*data)->eval_mode_out));
  CeedCall(CeedFree(&(*data)->B_in));
  CeedCall(CeedFree(&(*data)->B_out));

  CeedCall(CeedFree(data));
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
int CeedOperatorLinearAssembleQFunction(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  CeedCall(CeedOperatorCheckReady(op));

  if (op->LinearAssembleQFunction) {
    // Backend version
    CeedCall(op->LinearAssembleQFunction(op, assembled, rstr, request));
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedOperatorLinearAssembleQFunction(op_fallback, assembled, rstr, request));
    } else {
      // LCOV_EXCL_START
      return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support CeedOperatorLinearAssembleQFunction");
      // LCOV_EXCL_STOP
    }
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
int CeedOperatorLinearAssembleQFunctionBuildOrUpdate(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  CeedCall(CeedOperatorCheckReady(op));

  if (op->LinearAssembleQFunctionUpdate) {
    // Backend version
    bool                qf_assembled_is_setup;
    CeedVector          assembled_vec  = NULL;
    CeedElemRestriction assembled_rstr = NULL;

    CeedCall(CeedQFunctionAssemblyDataIsSetup(op->qf_assembled, &qf_assembled_is_setup));
    if (qf_assembled_is_setup) {
      bool update_needed;

      CeedCall(CeedQFunctionAssemblyDataGetObjects(op->qf_assembled, &assembled_vec, &assembled_rstr));
      CeedCall(CeedQFunctionAssemblyDataIsUpdateNeeded(op->qf_assembled, &update_needed));
      if (update_needed) {
        CeedCall(op->LinearAssembleQFunctionUpdate(op, assembled_vec, assembled_rstr, request));
      }
    } else {
      CeedCall(op->LinearAssembleQFunction(op, &assembled_vec, &assembled_rstr, request));
      CeedCall(CeedQFunctionAssemblyDataSetObjects(op->qf_assembled, assembled_vec, assembled_rstr));
    }
    CeedCall(CeedQFunctionAssemblyDataSetUpdateNeeded(op->qf_assembled, false));

    // Copy reference from internally held copy
    *assembled = NULL;
    *rstr      = NULL;
    CeedCall(CeedVectorReferenceCopy(assembled_vec, assembled));
    CeedCall(CeedVectorDestroy(&assembled_vec));
    CeedCall(CeedElemRestrictionReferenceCopy(assembled_rstr, rstr));
    CeedCall(CeedElemRestrictionDestroy(&assembled_rstr));
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op_fallback, assembled, rstr, request));
    } else {
      // LCOV_EXCL_START
      return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support CeedOperatorLinearAssembleQFunctionUpdate");
      // LCOV_EXCL_STOP
    }
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
int CeedOperatorLinearAssembleDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCall(CeedOperatorCheckReady(op));

  CeedSize input_size = 0, output_size = 0;
  CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  if (input_size != output_size) {
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");
    // LCOV_EXCL_STOP
  }

  if (op->LinearAssembleDiagonal) {
    // Backend version
    CeedCall(op->LinearAssembleDiagonal(op, assembled, request));
    return CEED_ERROR_SUCCESS;
  } else if (op->LinearAssembleAddDiagonal) {
    // Backend version with zeroing first
    CeedCall(CeedVectorSetValue(assembled, 0.0));
    CeedCall(op->LinearAssembleAddDiagonal(op, assembled, request));
    return CEED_ERROR_SUCCESS;
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedOperatorLinearAssembleDiagonal(op_fallback, assembled, request));
      return CEED_ERROR_SUCCESS;
    }
  }
  // Default interface implementation
  CeedCall(CeedVectorSetValue(assembled, 0.0));
  CeedCall(CeedOperatorLinearAssembleAddDiagonal(op, assembled, request));

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
int CeedOperatorLinearAssembleAddDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCall(CeedOperatorCheckReady(op));

  CeedSize input_size = 0, output_size = 0;
  CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  if (input_size != output_size) {
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");
    // LCOV_EXCL_STOP
  }

  if (op->LinearAssembleAddDiagonal) {
    // Backend version
    CeedCall(op->LinearAssembleAddDiagonal(op, assembled, request));
    return CEED_ERROR_SUCCESS;
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedOperatorLinearAssembleAddDiagonal(op_fallback, assembled, request));
      return CEED_ERROR_SUCCESS;
    }
  }
  // Default interface implementation
  bool is_composite;
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedCall(CeedCompositeOperatorLinearAssembleAddDiagonal(op, request, false, assembled));
  } else {
    CeedCall(CeedSingleOperatorAssembleAddDiagonal_Core(op, request, false, assembled));
  }

  return CEED_ERROR_SUCCESS;
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
int CeedOperatorLinearAssemblePointBlockDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCall(CeedOperatorCheckReady(op));

  CeedSize input_size = 0, output_size = 0;
  CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  if (input_size != output_size) {
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");
    // LCOV_EXCL_STOP
  }

  if (op->LinearAssemblePointBlockDiagonal) {
    // Backend version
    CeedCall(op->LinearAssemblePointBlockDiagonal(op, assembled, request));
    return CEED_ERROR_SUCCESS;
  } else if (op->LinearAssembleAddPointBlockDiagonal) {
    // Backend version with zeroing first
    CeedCall(CeedVectorSetValue(assembled, 0.0));
    CeedCall(CeedOperatorLinearAssembleAddPointBlockDiagonal(op, assembled, request));
    return CEED_ERROR_SUCCESS;
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedOperatorLinearAssemblePointBlockDiagonal(op_fallback, assembled, request));
      return CEED_ERROR_SUCCESS;
    }
  }
  // Default interface implementation
  CeedCall(CeedVectorSetValue(assembled, 0.0));
  CeedCall(CeedOperatorLinearAssembleAddPointBlockDiagonal(op, assembled, request));

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
int CeedOperatorLinearAssembleAddPointBlockDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  CeedCall(CeedOperatorCheckReady(op));

  CeedSize input_size = 0, output_size = 0;
  CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  if (input_size != output_size) {
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");
    // LCOV_EXCL_STOP
  }

  if (op->LinearAssembleAddPointBlockDiagonal) {
    // Backend version
    CeedCall(op->LinearAssembleAddPointBlockDiagonal(op, assembled, request));
    return CEED_ERROR_SUCCESS;
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedOperatorLinearAssembleAddPointBlockDiagonal(op_fallback, assembled, request));
      return CEED_ERROR_SUCCESS;
    }
  }
  // Default interface implemenation
  bool is_composite;
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedCall(CeedCompositeOperatorLinearAssembleAddDiagonal(op, request, true, assembled));
  } else {
    CeedCall(CeedSingleOperatorAssembleAddDiagonal_Core(op, request, true, assembled));
  }

  return CEED_ERROR_SUCCESS;
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
int CeedOperatorLinearAssembleSymbolic(CeedOperator op, CeedSize *num_entries, CeedInt **rows, CeedInt **cols) {
  CeedInt       num_suboperators, single_entries;
  CeedOperator *sub_operators;
  bool          is_composite;
  CeedCall(CeedOperatorCheckReady(op));

  if (op->LinearAssembleSymbolic) {
    // Backend version
    CeedCall(op->LinearAssembleSymbolic(op, num_entries, rows, cols));
    return CEED_ERROR_SUCCESS;
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedOperatorLinearAssembleSymbolic(op_fallback, num_entries, rows, cols));
      return CEED_ERROR_SUCCESS;
    }
  }

  // Default interface implementation

  // count entries and allocate rows, cols arrays
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  *num_entries = 0;
  if (is_composite) {
    CeedCall(CeedOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedOperatorGetSubList(op, &sub_operators));
    for (CeedInt k = 0; k < num_suboperators; ++k) {
      CeedCall(CeedSingleOperatorAssemblyCountEntries(sub_operators[k], &single_entries));
      *num_entries += single_entries;
    }
  } else {
    CeedCall(CeedSingleOperatorAssemblyCountEntries(op, &single_entries));
    *num_entries += single_entries;
  }
  CeedCall(CeedCalloc(*num_entries, rows));
  CeedCall(CeedCalloc(*num_entries, cols));

  // assemble nonzero locations
  CeedInt offset = 0;
  if (is_composite) {
    CeedCall(CeedOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedOperatorGetSubList(op, &sub_operators));
    for (CeedInt k = 0; k < num_suboperators; ++k) {
      CeedCall(CeedSingleOperatorAssembleSymbolic(sub_operators[k], offset, *rows, *cols));
      CeedCall(CeedSingleOperatorAssemblyCountEntries(sub_operators[k], &single_entries));
      offset += single_entries;
    }
  } else {
    CeedCall(CeedSingleOperatorAssembleSymbolic(op, offset, *rows, *cols));
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
  CeedInt       num_suboperators, single_entries = 0;
  CeedOperator *sub_operators;
  CeedCall(CeedOperatorCheckReady(op));

  if (op->LinearAssemble) {
    // Backend version
    CeedCall(op->LinearAssemble(op, values));
    return CEED_ERROR_SUCCESS;
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedOperatorLinearAssemble(op_fallback, values));
      return CEED_ERROR_SUCCESS;
    }
  }

  // Default interface implementation
  bool is_composite;
  CeedCall(CeedOperatorIsComposite(op, &is_composite));

  CeedInt offset = 0;
  if (is_composite) {
    CeedCall(CeedOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedOperatorGetSubList(op, &sub_operators));
    for (CeedInt k = 0; k < num_suboperators; k++) {
      CeedCall(CeedSingleOperatorAssemble(sub_operators[k], offset, values));
      CeedCall(CeedSingleOperatorAssemblyCountEntries(sub_operators[k], &single_entries));
      offset += single_entries;
    }
  } else {
    CeedCall(CeedSingleOperatorAssemble(op, offset, values));
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
int CeedOperatorMultigridLevelCreate(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
                                     CeedOperator *op_coarse, CeedOperator *op_prolong, CeedOperator *op_restrict) {
  CeedCall(CeedOperatorCheckReady(op_fine));

  // Build prolongation matrix
  CeedBasis basis_fine, basis_c_to_f;
  CeedCall(CeedOperatorGetActiveBasis(op_fine, &basis_fine));
  CeedCall(CeedBasisCreateProjection(basis_coarse, basis_fine, &basis_c_to_f));

  // Core code
  CeedCall(CeedSingleOperatorMultigridLevel(op_fine, p_mult_fine, rstr_coarse, basis_coarse, basis_c_to_f, op_coarse, op_prolong, op_restrict));

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
int CeedOperatorMultigridLevelCreateTensorH1(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
                                             const CeedScalar *interp_c_to_f, CeedOperator *op_coarse, CeedOperator *op_prolong,
                                             CeedOperator *op_restrict) {
  CeedCall(CeedOperatorCheckReady(op_fine));
  Ceed ceed;
  CeedCall(CeedOperatorGetCeed(op_fine, &ceed));

  // Check for compatible quadrature spaces
  CeedBasis basis_fine;
  CeedCall(CeedOperatorGetActiveBasis(op_fine, &basis_fine));
  CeedInt Q_f, Q_c;
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_fine, &Q_f));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_coarse, &Q_c));
  if (Q_f != Q_c) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Bases must have compatible quadrature spaces");
    // LCOV_EXCL_STOP
  }

  // Coarse to fine basis
  CeedInt dim, num_comp, num_nodes_c, P_1d_f, P_1d_c;
  CeedCall(CeedBasisGetDimension(basis_fine, &dim));
  CeedCall(CeedBasisGetNumComponents(basis_fine, &num_comp));
  CeedCall(CeedBasisGetNumNodes1D(basis_fine, &P_1d_f));
  CeedCall(CeedElemRestrictionGetElementSize(rstr_coarse, &num_nodes_c));
  P_1d_c = dim == 1 ? num_nodes_c : dim == 2 ? sqrt(num_nodes_c) : cbrt(num_nodes_c);
  CeedScalar *q_ref, *q_weight, *grad;
  CeedCall(CeedCalloc(P_1d_f, &q_ref));
  CeedCall(CeedCalloc(P_1d_f, &q_weight));
  CeedCall(CeedCalloc(P_1d_f * P_1d_c * dim, &grad));
  CeedBasis basis_c_to_f;
  CeedCall(CeedBasisCreateTensorH1(ceed, dim, num_comp, P_1d_c, P_1d_f, interp_c_to_f, grad, q_ref, q_weight, &basis_c_to_f));
  CeedCall(CeedFree(&q_ref));
  CeedCall(CeedFree(&q_weight));
  CeedCall(CeedFree(&grad));

  // Core code
  CeedCall(CeedSingleOperatorMultigridLevel(op_fine, p_mult_fine, rstr_coarse, basis_coarse, basis_c_to_f, op_coarse, op_prolong, op_restrict));
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
int CeedOperatorMultigridLevelCreateH1(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
                                       const CeedScalar *interp_c_to_f, CeedOperator *op_coarse, CeedOperator *op_prolong,
                                       CeedOperator *op_restrict) {
  CeedCall(CeedOperatorCheckReady(op_fine));
  Ceed ceed;
  CeedCall(CeedOperatorGetCeed(op_fine, &ceed));

  // Check for compatible quadrature spaces
  CeedBasis basis_fine;
  CeedCall(CeedOperatorGetActiveBasis(op_fine, &basis_fine));
  CeedInt Q_f, Q_c;
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_fine, &Q_f));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_coarse, &Q_c));
  if (Q_f != Q_c) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Bases must have compatible quadrature spaces");
    // LCOV_EXCL_STOP
  }

  // Coarse to fine basis
  CeedElemTopology topo;
  CeedCall(CeedBasisGetTopology(basis_fine, &topo));
  CeedInt dim, num_comp, num_nodes_c, num_nodes_f;
  CeedCall(CeedBasisGetDimension(basis_fine, &dim));
  CeedCall(CeedBasisGetNumComponents(basis_fine, &num_comp));
  CeedCall(CeedBasisGetNumNodes(basis_fine, &num_nodes_f));
  CeedCall(CeedElemRestrictionGetElementSize(rstr_coarse, &num_nodes_c));
  CeedScalar *q_ref, *q_weight, *grad;
  CeedCall(CeedCalloc(num_nodes_f * dim, &q_ref));
  CeedCall(CeedCalloc(num_nodes_f, &q_weight));
  CeedCall(CeedCalloc(num_nodes_f * num_nodes_c * dim, &grad));
  CeedBasis basis_c_to_f;
  CeedCall(CeedBasisCreateH1(ceed, topo, num_comp, num_nodes_c, num_nodes_f, interp_c_to_f, grad, q_ref, q_weight, &basis_c_to_f));
  CeedCall(CeedFree(&q_ref));
  CeedCall(CeedFree(&q_weight));
  CeedCall(CeedFree(&grad));

  // Core code
  CeedCall(CeedSingleOperatorMultigridLevel(op_fine, p_mult_fine, rstr_coarse, basis_coarse, basis_c_to_f, op_coarse, op_prolong, op_restrict));
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
int CeedOperatorCreateFDMElementInverse(CeedOperator op, CeedOperator *fdm_inv, CeedRequest *request) {
  CeedCall(CeedOperatorCheckReady(op));

  if (op->CreateFDMElementInverse) {
    // Backend version
    CeedCall(op->CreateFDMElementInverse(op, fdm_inv, request));
    return CEED_ERROR_SUCCESS;
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) {
      CeedCall(CeedOperatorCreateFDMElementInverse(op_fallback, fdm_inv, request));
      return CEED_ERROR_SUCCESS;
    }
  }

  // Default interface implementation
  Ceed ceed, ceed_parent;
  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedGetOperatorFallbackParentCeed(ceed, &ceed_parent));
  ceed_parent = ceed_parent ? ceed_parent : ceed;
  CeedQFunction qf;
  CeedCall(CeedOperatorGetQFunction(op, &qf));

  // Determine active input basis
  bool                interp = false, grad = false;
  CeedBasis           basis = NULL;
  CeedElemRestriction rstr  = NULL;
  CeedOperatorField  *op_fields;
  CeedQFunctionField *qf_fields;
  CeedInt             num_input_fields;
  CeedCall(CeedOperatorGetFields(op, &num_input_fields, &op_fields, NULL, NULL));
  CeedCall(CeedQFunctionGetFields(qf, NULL, &qf_fields, NULL, NULL));
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;
    CeedCall(CeedOperatorFieldGetVector(op_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedEvalMode eval_mode;
      CeedCall(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      interp = interp || eval_mode == CEED_EVAL_INTERP;
      grad   = grad || eval_mode == CEED_EVAL_GRAD;
      CeedCall(CeedOperatorFieldGetBasis(op_fields[i], &basis));
      CeedCall(CeedOperatorFieldGetElemRestriction(op_fields[i], &rstr));
    }
  }
  if (!basis) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "No active field set");
    // LCOV_EXCL_STOP
  }
  CeedSize l_size = 1;
  CeedInt  P_1d, Q_1d, elem_size, num_qpts, dim, num_comp = 1, num_elem = 1;
  CeedCall(CeedBasisGetNumNodes1D(basis, &P_1d));
  CeedCall(CeedBasisGetNumNodes(basis, &elem_size));
  CeedCall(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCall(CeedBasisGetDimension(basis, &dim));
  CeedCall(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &l_size));

  // Build and diagonalize 1D Mass and Laplacian
  bool tensor_basis;
  CeedCall(CeedBasisIsTensor(basis, &tensor_basis));
  if (!tensor_basis) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "FDMElementInverse only supported for tensor bases");
    // LCOV_EXCL_STOP
  }
  CeedScalar *mass, *laplace, *x, *fdm_interp, *lambda;
  CeedCall(CeedCalloc(P_1d * P_1d, &mass));
  CeedCall(CeedCalloc(P_1d * P_1d, &laplace));
  CeedCall(CeedCalloc(P_1d * P_1d, &x));
  CeedCall(CeedCalloc(P_1d * P_1d, &fdm_interp));
  CeedCall(CeedCalloc(P_1d, &lambda));
  // -- Build matrices
  const CeedScalar *interp_1d, *grad_1d, *q_weight_1d;
  CeedCall(CeedBasisGetInterp1D(basis, &interp_1d));
  CeedCall(CeedBasisGetGrad1D(basis, &grad_1d));
  CeedCall(CeedBasisGetQWeights(basis, &q_weight_1d));
  CeedCall(CeedBuildMassLaplace(interp_1d, grad_1d, q_weight_1d, P_1d, Q_1d, dim, mass, laplace));

  // -- Diagonalize
  CeedCall(CeedSimultaneousDiagonalization(ceed, laplace, mass, x, lambda, P_1d));
  CeedCall(CeedFree(&mass));
  CeedCall(CeedFree(&laplace));
  for (CeedInt i = 0; i < P_1d; i++) {
    for (CeedInt j = 0; j < P_1d; j++) fdm_interp[i + j * P_1d] = x[j + i * P_1d];
  }
  CeedCall(CeedFree(&x));

  // Assemble QFunction
  CeedVector          assembled;
  CeedElemRestriction rstr_qf;
  CeedCall(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled, &rstr_qf, request));
  CeedInt layout[3];
  CeedCall(CeedElemRestrictionGetELayout(rstr_qf, &layout));
  CeedCall(CeedElemRestrictionDestroy(&rstr_qf));
  CeedScalar max_norm = 0;
  CeedCall(CeedVectorNorm(assembled, CEED_NORM_MAX, &max_norm));

  // Calculate element averages
  CeedInt           num_modes = (interp ? 1 : 0) + (grad ? dim : 0);
  CeedScalar       *elem_avg;
  const CeedScalar *assembled_array, *q_weight_array;
  CeedVector        q_weight;
  CeedCall(CeedVectorCreate(ceed_parent, num_qpts, &q_weight));
  CeedCall(CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, q_weight));
  CeedCall(CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembled_array));
  CeedCall(CeedVectorGetArrayRead(q_weight, CEED_MEM_HOST, &q_weight_array));
  CeedCall(CeedCalloc(num_elem, &elem_avg));
  const CeedScalar qf_value_bound = max_norm * 100 * CEED_EPSILON;
  for (CeedInt e = 0; e < num_elem; e++) {
    CeedInt count = 0;
    for (CeedInt q = 0; q < num_qpts; q++) {
      for (CeedInt i = 0; i < num_comp * num_comp * num_modes * num_modes; i++) {
        if (fabs(assembled_array[q * layout[0] + i * layout[1] + e * layout[2]]) > qf_value_bound) {
          elem_avg[e] += assembled_array[q * layout[0] + i * layout[1] + e * layout[2]] / q_weight_array[q];
          count++;
        }
      }
    }
    if (count) {
      elem_avg[e] /= count;
    } else {
      elem_avg[e] = 1.0;
    }
  }
  CeedCall(CeedVectorRestoreArrayRead(assembled, &assembled_array));
  CeedCall(CeedVectorDestroy(&assembled));
  CeedCall(CeedVectorRestoreArrayRead(q_weight, &q_weight_array));
  CeedCall(CeedVectorDestroy(&q_weight));

  // Build FDM diagonal
  CeedVector  q_data;
  CeedScalar *q_data_array, *fdm_diagonal;
  CeedCall(CeedCalloc(num_comp * elem_size, &fdm_diagonal));
  const CeedScalar fdm_diagonal_bound = elem_size * CEED_EPSILON;
  for (CeedInt c = 0; c < num_comp; c++) {
    for (CeedInt n = 0; n < elem_size; n++) {
      if (interp) fdm_diagonal[c * elem_size + n] = 1.0;
      if (grad) {
        for (CeedInt d = 0; d < dim; d++) {
          CeedInt i = (n / CeedIntPow(P_1d, d)) % P_1d;
          fdm_diagonal[c * elem_size + n] += lambda[i];
        }
      }
      if (fabs(fdm_diagonal[c * elem_size + n]) < fdm_diagonal_bound) fdm_diagonal[c * elem_size + n] = fdm_diagonal_bound;
    }
  }
  CeedCall(CeedVectorCreate(ceed_parent, num_elem * num_comp * elem_size, &q_data));
  CeedCall(CeedVectorSetValue(q_data, 0.0));
  CeedCall(CeedVectorGetArrayWrite(q_data, CEED_MEM_HOST, &q_data_array));
  for (CeedInt e = 0; e < num_elem; e++) {
    for (CeedInt c = 0; c < num_comp; c++) {
      for (CeedInt n = 0; n < elem_size; n++) q_data_array[(e * num_comp + c) * elem_size + n] = 1. / (elem_avg[e] * fdm_diagonal[c * elem_size + n]);
    }
  }
  CeedCall(CeedFree(&elem_avg));
  CeedCall(CeedFree(&fdm_diagonal));
  CeedCall(CeedVectorRestoreArray(q_data, &q_data_array));

  // Setup FDM operator
  // -- Basis
  CeedBasis   fdm_basis;
  CeedScalar *grad_dummy, *q_ref_dummy, *q_weight_dummy;
  CeedCall(CeedCalloc(P_1d * P_1d, &grad_dummy));
  CeedCall(CeedCalloc(P_1d, &q_ref_dummy));
  CeedCall(CeedCalloc(P_1d, &q_weight_dummy));
  CeedCall(CeedBasisCreateTensorH1(ceed_parent, dim, num_comp, P_1d, P_1d, fdm_interp, grad_dummy, q_ref_dummy, q_weight_dummy, &fdm_basis));
  CeedCall(CeedFree(&fdm_interp));
  CeedCall(CeedFree(&grad_dummy));
  CeedCall(CeedFree(&q_ref_dummy));
  CeedCall(CeedFree(&q_weight_dummy));
  CeedCall(CeedFree(&lambda));

  // -- Restriction
  CeedElemRestriction rstr_qd_i;
  CeedInt             strides[3] = {1, elem_size, elem_size * num_comp};
  CeedCall(CeedElemRestrictionCreateStrided(ceed_parent, num_elem, elem_size, num_comp, num_elem * num_comp * elem_size, strides, &rstr_qd_i));
  // -- QFunction
  CeedQFunction qf_fdm;
  CeedCall(CeedQFunctionCreateInteriorByName(ceed_parent, "Scale", &qf_fdm));
  CeedCall(CeedQFunctionAddInput(qf_fdm, "input", num_comp, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionAddInput(qf_fdm, "scale", num_comp, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf_fdm, "output", num_comp, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf_fdm, num_comp));
  // -- QFunction context
  CeedInt *num_comp_data;
  CeedCall(CeedCalloc(1, &num_comp_data));
  num_comp_data[0] = num_comp;
  CeedQFunctionContext ctx_fdm;
  CeedCall(CeedQFunctionContextCreate(ceed, &ctx_fdm));
  CeedCall(CeedQFunctionContextSetData(ctx_fdm, CEED_MEM_HOST, CEED_OWN_POINTER, sizeof(*num_comp_data), num_comp_data));
  CeedCall(CeedQFunctionSetContext(qf_fdm, ctx_fdm));
  CeedCall(CeedQFunctionContextDestroy(&ctx_fdm));
  // -- Operator
  CeedCall(CeedOperatorCreate(ceed_parent, qf_fdm, NULL, NULL, fdm_inv));
  CeedCall(CeedOperatorSetField(*fdm_inv, "input", rstr, fdm_basis, CEED_VECTOR_ACTIVE));
  CeedCall(CeedOperatorSetField(*fdm_inv, "scale", rstr_qd_i, CEED_BASIS_COLLOCATED, q_data));
  CeedCall(CeedOperatorSetField(*fdm_inv, "output", rstr, fdm_basis, CEED_VECTOR_ACTIVE));

  // Cleanup
  CeedCall(CeedVectorDestroy(&q_data));
  CeedCall(CeedBasisDestroy(&fdm_basis));
  CeedCall(CeedElemRestrictionDestroy(&rstr_qd_i));
  CeedCall(CeedQFunctionDestroy(&qf_fdm));

  return CEED_ERROR_SUCCESS;
}

/// @}
