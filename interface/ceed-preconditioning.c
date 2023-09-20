// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <assert.h>
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
  @brief Duplicate a CeedQFunction with a reference Ceed to fallback for advanced CeedOperator functionality

  @param[in]  fallback_ceed Ceed on which to create fallback CeedQFunction
  @param[in]  qf            CeedQFunction to create fallback for
  @param[out] qf_fallback   fallback CeedQFunction

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedQFunctionCreateFallback(Ceed fallback_ceed, CeedQFunction qf, CeedQFunction *qf_fallback) {
  char *source_path_with_name = NULL;

  // Check if NULL qf passed in
  if (!qf) return CEED_ERROR_SUCCESS;

  CeedDebug256(qf->ceed, 1, "---------- CeedOperator Fallback ----------\n");
  CeedDebug(qf->ceed, "Creating fallback CeedQFunction\n");

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
  @brief Duplicate a CeedOperator with a reference Ceed to fallback for advanced CeedOperator functionality

  @param[in,out] op CeedOperator to create fallback for

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCreateFallback(CeedOperator op) {
  Ceed         ceed_fallback;
  bool         is_composite;
  CeedOperator op_fallback;

  // Check not already created
  if (op->op_fallback) return CEED_ERROR_SUCCESS;

  // Fallback Ceed
  CeedCall(CeedGetOperatorFallbackCeed(op->ceed, &ceed_fallback));
  if (!ceed_fallback) return CEED_ERROR_SUCCESS;

  CeedDebug256(op->ceed, 1, "---------- CeedOperator Fallback ----------\n");
  CeedDebug(op->ceed, "Creating fallback CeedOperator\n");

  // Clone Op
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt       num_suboperators;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorCreate(ceed_fallback, &op_fallback));
    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    for (CeedInt i = 0; i < num_suboperators; i++) {
      CeedOperator op_sub_fallback;

      CeedCall(CeedOperatorGetFallback(sub_operators[i], &op_sub_fallback));
      CeedCall(CeedCompositeOperatorAddSub(op_fallback, op_sub_fallback));
    }
  } else {
    CeedQFunction qf_fallback = NULL, dqf_fallback = NULL, dqfT_fallback = NULL;

    CeedCall(CeedQFunctionCreateFallback(ceed_fallback, op->qf, &qf_fallback));
    CeedCall(CeedQFunctionCreateFallback(ceed_fallback, op->dqf, &dqf_fallback));
    CeedCall(CeedQFunctionCreateFallback(ceed_fallback, op->dqfT, &dqfT_fallback));
    CeedCall(CeedOperatorCreate(ceed_fallback, qf_fallback, dqf_fallback, dqfT_fallback, &op_fallback));
    for (CeedInt i = 0; i < op->qf->num_input_fields; i++) {
      CeedCall(CeedOperatorSetField(op_fallback, op->input_fields[i]->field_name, op->input_fields[i]->elem_rstr, op->input_fields[i]->basis,
                                    op->input_fields[i]->vec));
    }
    for (CeedInt i = 0; i < op->qf->num_output_fields; i++) {
      CeedCall(CeedOperatorSetField(op_fallback, op->output_fields[i]->field_name, op->output_fields[i]->elem_rstr, op->output_fields[i]->basis,
                                    op->output_fields[i]->vec));
    }
    CeedCall(CeedQFunctionAssemblyDataReferenceCopy(op->qf_assembled, &op_fallback->qf_assembled));
    if (op_fallback->num_qpts == 0) CeedCall(CeedOperatorSetNumQuadraturePoints(op_fallback, op->num_qpts));
    // Cleanup
    CeedCall(CeedQFunctionDestroy(&qf_fallback));
    CeedCall(CeedQFunctionDestroy(&dqf_fallback));
    CeedCall(CeedQFunctionDestroy(&dqfT_fallback));
  }
  CeedCall(CeedOperatorSetName(op_fallback, op->name));
  CeedCall(CeedOperatorCheckReady(op_fallback));
  // Note: No ref-counting here so we don't get caught in a reference loop.
  //       The op holds the only reference to op_fallback and is responsible for deleting itself and op_fallback.
  op->op_fallback                 = op_fallback;
  op_fallback->op_fallback_parent = op;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Retrieve fallback CeedOperator with a reference Ceed for advanced CeedOperator functionality

  @param[in]  op          CeedOperator to retrieve fallback for
  @param[out] op_fallback Fallback CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedOperatorGetFallback(CeedOperator op, CeedOperator *op_fallback) {
  // Create if needed
  if (!op->op_fallback) CeedCall(CeedOperatorCreateFallback(op));
  if (op->op_fallback) {
    bool is_debug;

    CeedCall(CeedIsDebug(op->ceed, &is_debug));
    if (is_debug) {
      Ceed        ceed, ceed_fallback;
      const char *resource, *resource_fallback;

      CeedCall(CeedOperatorGetCeed(op, &ceed));
      CeedCall(CeedGetOperatorFallbackCeed(ceed, &ceed_fallback));
      CeedCall(CeedGetResource(ceed, &resource));
      CeedCall(CeedGetResource(ceed_fallback, &resource_fallback));

      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- CeedOperator Fallback ----------\n");
      CeedDebug(ceed, "Falling back from %s operator at address %ld to %s operator at address %ld\n", resource, op, resource_fallback,
                op->op_fallback);
    }
  }
  *op_fallback = op->op_fallback;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the parent CeedOperator for a fallback CeedOperator

  @param[in]  op     CeedOperator context
  @param[out] parent Variable to store parent CeedOperator context

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedOperatorGetFallbackParent(CeedOperator op, CeedOperator *parent) {
  *parent = op->op_fallback_parent ? op->op_fallback_parent : NULL;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the Ceed context of the parent CeedOperator for a fallback CeedOperator

  @param[in]  op     CeedOperator context
  @param[out] parent Variable to store parent Ceed context

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedOperatorGetFallbackParentCeed(CeedOperator op, Ceed *parent) {
  *parent = op->op_fallback_parent ? op->op_fallback_parent->ceed : op->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Select correct basis matrix pointer based on CeedEvalMode

  @param[in]  basis     CeedBasis from which to get the basis matrix
  @param[in]  eval_mode Current basis evaluation mode
  @param[in]  identity  Pointer to identity matrix
  @param[out] basis_ptr Basis pointer to set

  @ref Developer
**/
static inline int CeedOperatorGetBasisPointer(CeedBasis basis, CeedEvalMode eval_mode, const CeedScalar *identity, const CeedScalar **basis_ptr) {
  switch (eval_mode) {
    case CEED_EVAL_NONE:
      *basis_ptr = identity;
      break;
    case CEED_EVAL_INTERP:
      CeedCall(CeedBasisGetInterp(basis, basis_ptr));
      break;
    case CEED_EVAL_GRAD:
      CeedCall(CeedBasisGetGrad(basis, basis_ptr));
      break;
    case CEED_EVAL_DIV:
      CeedCall(CeedBasisGetDiv(basis, basis_ptr));
      break;
    case CEED_EVAL_CURL:
      CeedCall(CeedBasisGetCurl(basis, basis_ptr));
      break;
    case CEED_EVAL_WEIGHT:
      break;  // Caught by QF Assembly
  }
  assert(*basis_ptr != NULL);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Core logic for assembling operator diagonal or point block diagonal

  @param[in]  op             CeedOperator to assemble point block diagonal
  @param[in]  request        Address of CeedRequest for non-blocking completion, else CEED_REQUEST_IMMEDIATE
  @param[in]  is_point_block Boolean flag to assemble diagonal or point block diagonal
  @param[out] assembled      CeedVector to store assembled diagonal

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static inline int CeedSingleOperatorAssembleAddDiagonal_Core(CeedOperator op, CeedRequest *request, const bool is_point_block, CeedVector assembled) {
  Ceed ceed;
  bool is_composite;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, ceed, CEED_ERROR_UNSUPPORTED, "Composite operator not supported");

  // Assemble QFunction
  CeedInt             layout_qf[3];
  const CeedScalar   *assembled_qf_array;
  CeedVector          assembled_qf        = NULL;
  CeedElemRestriction assembled_elem_rstr = NULL;

  CeedCall(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &assembled_elem_rstr, request));
  CeedCall(CeedElemRestrictionGetELayout(assembled_elem_rstr, &layout_qf));
  CeedCall(CeedElemRestrictionDestroy(&assembled_elem_rstr));
  CeedCall(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_HOST, &assembled_qf_array));

  // Get assembly data
  const CeedEvalMode     **eval_modes_in, **eval_modes_out;
  CeedInt                  num_active_bases_in, *num_eval_modes_in, num_active_bases_out, *num_eval_modes_out;
  CeedSize               **eval_mode_offsets_in, **eval_mode_offsets_out, num_output_components;
  CeedBasis               *active_bases_in, *active_bases_out;
  CeedElemRestriction     *active_elem_rstrs_in, *active_elem_rstrs_out;
  CeedOperatorAssemblyData data;

  CeedCall(CeedOperatorGetOperatorAssemblyData(op, &data));
  CeedCall(CeedOperatorAssemblyDataGetEvalModes(data, &num_active_bases_in, &num_eval_modes_in, &eval_modes_in, &eval_mode_offsets_in,
                                                &num_active_bases_out, &num_eval_modes_out, &eval_modes_out, &eval_mode_offsets_out,
                                                &num_output_components));
  CeedCall(CeedOperatorAssemblyDataGetBases(data, NULL, &active_bases_in, NULL, NULL, &active_bases_out, NULL));
  CeedCall(CeedOperatorAssemblyDataGetElemRestrictions(data, NULL, &active_elem_rstrs_in, NULL, &active_elem_rstrs_out));

  CeedCheck(num_active_bases_in == num_active_bases_out, ceed, CEED_ERROR_UNSUPPORTED,
            "Cannot assemble operator diagonal with different numbers of input and output active bases");

  // Loop over all active bases
  for (CeedInt b = 0; b < num_active_bases_in; b++) {
    bool                has_eval_none = false;
    CeedInt             num_elem, num_nodes, num_qpts, num_comp;
    CeedScalar         *elem_diag_array, *identity = NULL;
    CeedVector          elem_diag;
    CeedElemRestriction diag_elem_rstr;

    CeedCheck(active_elem_rstrs_in[b] == active_elem_rstrs_out[b], ceed, CEED_ERROR_UNSUPPORTED,
              "Cannot assemble operator diagonal with different input and output active element restrictions");

    // Assemble point block diagonal restriction, if needed
    if (is_point_block) {
      CeedCall(CeedOperatorCreateActivePointBlockRestriction(active_elem_rstrs_in[b], &diag_elem_rstr));
    } else {
      CeedCall(CeedElemRestrictionCreateUnsignedCopy(active_elem_rstrs_in[b], &diag_elem_rstr));
    }

    // Create diagonal vector
    CeedCall(CeedElemRestrictionCreateVector(diag_elem_rstr, NULL, &elem_diag));

    // Assemble element operator diagonals
    CeedCall(CeedVectorSetValue(elem_diag, 0.0));
    CeedCall(CeedVectorGetArray(elem_diag, CEED_MEM_HOST, &elem_diag_array));
    CeedCall(CeedElemRestrictionGetNumElements(diag_elem_rstr, &num_elem));
    CeedCall(CeedBasisGetNumNodes(active_bases_in[b], &num_nodes));
    CeedCall(CeedBasisGetNumComponents(active_bases_in[b], &num_comp));
    if (active_bases_in[b] == CEED_BASIS_NONE) num_qpts = num_nodes;
    else CeedCall(CeedBasisGetNumQuadraturePoints(active_bases_in[b], &num_qpts));

    if (active_bases_in[b] != active_bases_out[b]) {
      CeedInt num_nodes_out, num_qpts_out, num_comp_out;

      CeedCall(CeedBasisGetNumNodes(active_bases_out[b], &num_nodes_out));
      CeedCheck(num_nodes == num_nodes_out, ceed, CEED_ERROR_UNSUPPORTED, "Active input and output bases must have the same number of nodes");
      CeedCall(CeedBasisGetNumComponents(active_bases_out[b], &num_comp_out));
      CeedCheck(num_comp == num_comp_out, ceed, CEED_ERROR_UNSUPPORTED, "Active input and output bases must have the same number of components");
      if (active_bases_out[b] == CEED_BASIS_NONE) num_qpts_out = num_nodes_out;
      else CeedCall(CeedBasisGetNumQuadraturePoints(active_bases_out[b], &num_qpts_out));
      CeedCheck(num_qpts == num_qpts_out, ceed, CEED_ERROR_UNSUPPORTED,
                "Active input and output bases must have the same number of quadrature points");
    }

    // Construct identity matrix for basis if required
    for (CeedInt i = 0; i < num_eval_modes_in[b]; i++) {
      has_eval_none = has_eval_none || (eval_modes_in[b][i] == CEED_EVAL_NONE);
    }
    for (CeedInt i = 0; i < num_eval_modes_out[b]; i++) {
      has_eval_none = has_eval_none || (eval_modes_out[b][i] == CEED_EVAL_NONE);
    }
    if (has_eval_none) {
      CeedCall(CeedCalloc(num_qpts * num_nodes, &identity));
      for (CeedInt i = 0; i < (num_nodes < num_qpts ? num_nodes : num_qpts); i++) identity[i * num_nodes + i] = 1.0;
    }

    // Compute the diagonal of B^T D B
    // Each element
    for (CeedSize e = 0; e < num_elem; e++) {
      // Each basis eval mode pair
      CeedInt      d_out              = 0, q_comp_out;
      CeedEvalMode eval_mode_out_prev = CEED_EVAL_NONE;

      for (CeedInt e_out = 0; e_out < num_eval_modes_out[b]; e_out++) {
        CeedInt           d_in              = 0, q_comp_in;
        const CeedScalar *B_t               = NULL;
        CeedEvalMode      eval_mode_in_prev = CEED_EVAL_NONE;

        CeedCall(CeedOperatorGetBasisPointer(active_bases_out[b], eval_modes_out[b][e_out], identity, &B_t));
        CeedCall(CeedBasisGetNumQuadratureComponents(active_bases_out[b], eval_modes_out[b][e_out], &q_comp_out));
        if (q_comp_out > 1) {
          if (e_out == 0 || eval_modes_out[b][e_out] != eval_mode_out_prev) d_out = 0;
          else B_t = &B_t[(++d_out) * num_qpts * num_nodes];
        }
        eval_mode_out_prev = eval_modes_out[b][e_out];

        for (CeedInt e_in = 0; e_in < num_eval_modes_in[b]; e_in++) {
          const CeedScalar *B = NULL;

          CeedCall(CeedOperatorGetBasisPointer(active_bases_in[b], eval_modes_in[b][e_in], identity, &B));
          CeedCall(CeedBasisGetNumQuadratureComponents(active_bases_in[b], eval_modes_in[b][e_in], &q_comp_in));
          if (q_comp_in > 1) {
            if (e_in == 0 || eval_modes_in[b][e_in] != eval_mode_in_prev) d_in = 0;
            else B = &B[(++d_in) * num_qpts * num_nodes];
          }
          eval_mode_in_prev = eval_modes_in[b][e_in];

          // Each component
          for (CeedInt c_out = 0; c_out < num_comp; c_out++) {
            // Each qpt/node pair
            for (CeedInt q = 0; q < num_qpts; q++) {
              if (is_point_block) {
                // Point Block Diagonal
                for (CeedInt c_in = 0; c_in < num_comp; c_in++) {
                  const CeedSize c_offset = (eval_mode_offsets_in[b][e_in] + c_in) * num_output_components + eval_mode_offsets_out[b][e_out] + c_out;
                  const CeedScalar qf_value = assembled_qf_array[q * layout_qf[0] + c_offset * layout_qf[1] + e * layout_qf[2]];

                  for (CeedInt n = 0; n < num_nodes; n++) {
                    elem_diag_array[((e * num_comp + c_out) * num_comp + c_in) * num_nodes + n] +=
                        B_t[q * num_nodes + n] * qf_value * B[q * num_nodes + n];
                  }
                }
              } else {
                // Diagonal Only
                const CeedInt    c_offset = (eval_mode_offsets_in[b][e_in] + c_out) * num_output_components + eval_mode_offsets_out[b][e_out] + c_out;
                const CeedScalar qf_value = assembled_qf_array[q * layout_qf[0] + c_offset * layout_qf[1] + e * layout_qf[2]];

                for (CeedInt n = 0; n < num_nodes; n++) {
                  elem_diag_array[(e * num_comp + c_out) * num_nodes + n] += B_t[q * num_nodes + n] * qf_value * B[q * num_nodes + n];
                }
              }
            }
          }
        }
      }
    }
    CeedCall(CeedVectorRestoreArray(elem_diag, &elem_diag_array));

    // Assemble local operator diagonal
    CeedCall(CeedElemRestrictionApply(diag_elem_rstr, CEED_TRANSPOSE, elem_diag, assembled, request));

    // Cleanup
    CeedCall(CeedElemRestrictionDestroy(&diag_elem_rstr));
    CeedCall(CeedVectorDestroy(&elem_diag));
    CeedCall(CeedFree(&identity));
  }
  CeedCall(CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array));
  CeedCall(CeedVectorDestroy(&assembled_qf));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Core logic for assembling composite operator diagonal

  @param[in]  op             CeedOperator to assemble point block diagonal
  @param[in]  request        Address of CeedRequest for non-blocking completion, else CEED_REQUEST_IMMEDIATE
  @param[in]  is_point_block Boolean flag to assemble diagonal or point block diagonal
  @param[out] assembled      CeedVector to store assembled diagonal

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static inline int CeedCompositeOperatorLinearAssembleAddDiagonal(CeedOperator op, CeedRequest *request, const bool is_point_block,
                                                                 CeedVector assembled) {
  CeedInt       num_sub;
  CeedOperator *suboperators;

  CeedCall(CeedCompositeOperatorGetNumSub(op, &num_sub));
  CeedCall(CeedCompositeOperatorGetSubList(op, &suboperators));
  for (CeedInt i = 0; i < num_sub; i++) {
    if (is_point_block) {
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

  @param[in]  op     CeedOperator to assemble nonzero pattern
  @param[in]  offset Offset for number of entries
  @param[out] rows   Row number for each entry
  @param[out] cols   Column number for each entry

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedSingleOperatorAssembleSymbolic(CeedOperator op, CeedInt offset, CeedInt *rows, CeedInt *cols) {
  Ceed                ceed;
  bool                is_composite;
  CeedSize            num_nodes_in, num_nodes_out, count = 0;
  CeedInt             num_elem_in, elem_size_in, num_comp_in, layout_er_in[3];
  CeedInt             num_elem_out, elem_size_out, num_comp_out, layout_er_out[3], local_num_entries;
  CeedScalar         *array;
  const CeedScalar   *elem_dof_a_in, *elem_dof_a_out;
  CeedVector          index_vec_in, index_vec_out, elem_dof_in, elem_dof_out;
  CeedElemRestriction elem_rstr_in, elem_rstr_out, index_elem_rstr_in, index_elem_rstr_out;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, ceed, CEED_ERROR_UNSUPPORTED, "Composite operator not supported");

  CeedCall(CeedOperatorGetActiveVectorLengths(op, &num_nodes_in, &num_nodes_out));
  CeedCall(CeedOperatorGetActiveElemRestrictions(op, &elem_rstr_in, &elem_rstr_out));
  CeedCall(CeedElemRestrictionGetNumElements(elem_rstr_in, &num_elem_in));
  CeedCall(CeedElemRestrictionGetElementSize(elem_rstr_in, &elem_size_in));
  CeedCall(CeedElemRestrictionGetNumComponents(elem_rstr_in, &num_comp_in));
  CeedCall(CeedElemRestrictionGetELayout(elem_rstr_in, &layout_er_in));

  // Determine elem_dof relation for input
  CeedCall(CeedVectorCreate(ceed, num_nodes_in, &index_vec_in));
  CeedCall(CeedVectorGetArrayWrite(index_vec_in, CEED_MEM_HOST, &array));
  for (CeedInt i = 0; i < num_nodes_in; i++) array[i] = i;
  CeedCall(CeedVectorRestoreArray(index_vec_in, &array));
  CeedCall(CeedVectorCreate(ceed, num_elem_in * elem_size_in * num_comp_in, &elem_dof_in));
  CeedCall(CeedVectorSetValue(elem_dof_in, 0.0));
  CeedCall(CeedElemRestrictionCreateUnorientedCopy(elem_rstr_in, &index_elem_rstr_in));
  CeedCall(CeedElemRestrictionApply(index_elem_rstr_in, CEED_NOTRANSPOSE, index_vec_in, elem_dof_in, CEED_REQUEST_IMMEDIATE));
  CeedCall(CeedVectorGetArrayRead(elem_dof_in, CEED_MEM_HOST, &elem_dof_a_in));
  CeedCall(CeedVectorDestroy(&index_vec_in));
  CeedCall(CeedElemRestrictionDestroy(&index_elem_rstr_in));

  if (elem_rstr_in != elem_rstr_out) {
    CeedCall(CeedElemRestrictionGetNumElements(elem_rstr_out, &num_elem_out));
    CeedCheck(num_elem_in == num_elem_out, ceed, CEED_ERROR_UNSUPPORTED,
              "Active input and output operator restrictions must have the same number of elements");
    CeedCall(CeedElemRestrictionGetElementSize(elem_rstr_out, &elem_size_out));
    CeedCall(CeedElemRestrictionGetNumComponents(elem_rstr_out, &num_comp_out));
    CeedCall(CeedElemRestrictionGetELayout(elem_rstr_out, &layout_er_out));

    // Determine elem_dof relation for output
    CeedCall(CeedVectorCreate(ceed, num_nodes_out, &index_vec_out));
    CeedCall(CeedVectorGetArrayWrite(index_vec_out, CEED_MEM_HOST, &array));
    for (CeedInt i = 0; i < num_nodes_out; i++) array[i] = i;
    CeedCall(CeedVectorRestoreArray(index_vec_out, &array));
    CeedCall(CeedVectorCreate(ceed, num_elem_out * elem_size_out * num_comp_out, &elem_dof_out));
    CeedCall(CeedVectorSetValue(elem_dof_out, 0.0));
    CeedCall(CeedElemRestrictionCreateUnorientedCopy(elem_rstr_out, &index_elem_rstr_out));
    CeedCall(CeedElemRestrictionApply(index_elem_rstr_out, CEED_NOTRANSPOSE, index_vec_out, elem_dof_out, CEED_REQUEST_IMMEDIATE));
    CeedCall(CeedVectorGetArrayRead(elem_dof_out, CEED_MEM_HOST, &elem_dof_a_out));
    CeedCall(CeedVectorDestroy(&index_vec_out));
    CeedCall(CeedElemRestrictionDestroy(&index_elem_rstr_out));
  } else {
    num_elem_out     = num_elem_in;
    elem_size_out    = elem_size_in;
    num_comp_out     = num_comp_in;
    layout_er_out[0] = layout_er_in[0];
    layout_er_out[1] = layout_er_in[1];
    layout_er_out[2] = layout_er_in[2];
    elem_dof_a_out   = elem_dof_a_in;
  }
  local_num_entries = elem_size_out * num_comp_out * elem_size_in * num_comp_in * num_elem_in;

  // Determine i, j locations for element matrices
  for (CeedInt e = 0; e < num_elem_in; e++) {
    for (CeedInt comp_in = 0; comp_in < num_comp_in; comp_in++) {
      for (CeedInt comp_out = 0; comp_out < num_comp_out; comp_out++) {
        for (CeedInt i = 0; i < elem_size_out; i++) {
          for (CeedInt j = 0; j < elem_size_in; j++) {
            const CeedInt elem_dof_index_row = i * layout_er_out[0] + comp_out * layout_er_out[1] + e * layout_er_out[2];
            const CeedInt elem_dof_index_col = j * layout_er_in[0] + comp_in * layout_er_in[1] + e * layout_er_in[2];
            const CeedInt row                = elem_dof_a_out[elem_dof_index_row];
            const CeedInt col                = elem_dof_a_in[elem_dof_index_col];

            rows[offset + count] = row;
            cols[offset + count] = col;
            count++;
          }
        }
      }
    }
  }
  CeedCheck(count == local_num_entries, ceed, CEED_ERROR_MAJOR, "Error computing assembled entries");
  CeedCall(CeedVectorRestoreArrayRead(elem_dof_in, &elem_dof_a_in));
  CeedCall(CeedVectorDestroy(&elem_dof_in));
  if (elem_rstr_in != elem_rstr_out) {
    CeedCall(CeedVectorRestoreArrayRead(elem_dof_out, &elem_dof_a_out));
    CeedCall(CeedVectorDestroy(&elem_dof_out));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble nonzero entries for non-composite operator

  Users should generally use CeedOperatorLinearAssemble()

  @param[in]  op     CeedOperator to assemble
  @param[in]  offset Offset for number of entries
  @param[out] values Values to assemble into matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedSingleOperatorAssemble(CeedOperator op, CeedInt offset, CeedVector values) {
  Ceed ceed;
  bool is_composite;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, ceed, CEED_ERROR_UNSUPPORTED, "Composite operator not supported");

  // Early exit for empty operator
  {
    CeedInt num_elem = 0;

    CeedCall(CeedOperatorGetNumElements(op, &num_elem));
    if (num_elem == 0) return CEED_ERROR_SUCCESS;
  }

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
  CeedInt             layout_qf[3];
  const CeedScalar   *assembled_qf_array;
  CeedVector          assembled_qf        = NULL;
  CeedElemRestriction assembled_elem_rstr = NULL;

  CeedCall(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled_qf, &assembled_elem_rstr, CEED_REQUEST_IMMEDIATE));
  CeedCall(CeedElemRestrictionGetELayout(assembled_elem_rstr, &layout_qf));
  CeedCall(CeedElemRestrictionDestroy(&assembled_elem_rstr));
  CeedCall(CeedVectorGetArrayRead(assembled_qf, CEED_MEM_HOST, &assembled_qf_array));

  // Get assembly data
  CeedInt                  num_elem_in, elem_size_in, num_comp_in, num_qpts_in;
  CeedInt                  num_elem_out, elem_size_out, num_comp_out, num_qpts_out, local_num_entries;
  const CeedEvalMode     **eval_modes_in, **eval_modes_out;
  CeedInt                  num_active_bases_in, *num_eval_modes_in, num_active_bases_out, *num_eval_modes_out;
  CeedBasis               *active_bases_in, *active_bases_out, basis_in, basis_out;
  const CeedScalar       **B_mats_in, **B_mats_out, *B_mat_in, *B_mat_out;
  CeedElemRestriction      elem_rstr_in, elem_rstr_out;
  CeedRestrictionType      elem_rstr_type_in, elem_rstr_type_out;
  const bool              *elem_rstr_orients_in = NULL, *elem_rstr_orients_out = NULL;
  const CeedInt8          *elem_rstr_curl_orients_in = NULL, *elem_rstr_curl_orients_out = NULL;
  CeedOperatorAssemblyData data;

  CeedCall(CeedOperatorGetOperatorAssemblyData(op, &data));
  CeedCall(CeedOperatorAssemblyDataGetEvalModes(data, &num_active_bases_in, &num_eval_modes_in, &eval_modes_in, NULL, &num_active_bases_out,
                                                &num_eval_modes_out, &eval_modes_out, NULL, NULL));

  CeedCheck(num_active_bases_in == num_active_bases_out && num_active_bases_in == 1, ceed, CEED_ERROR_UNSUPPORTED,
            "Cannot assemble operator with multiple active bases");
  CeedCheck(num_eval_modes_in[0] > 0 && num_eval_modes_out[0] > 0, ceed, CEED_ERROR_UNSUPPORTED, "Cannot assemble operator without inputs/outputs");

  CeedCall(CeedOperatorAssemblyDataGetBases(data, NULL, &active_bases_in, &B_mats_in, NULL, &active_bases_out, &B_mats_out));
  CeedCall(CeedOperatorGetActiveElemRestrictions(op, &elem_rstr_in, &elem_rstr_out));
  basis_in  = active_bases_in[0];
  basis_out = active_bases_out[0];
  B_mat_in  = B_mats_in[0];
  B_mat_out = B_mats_out[0];

  CeedCall(CeedElemRestrictionGetNumElements(elem_rstr_in, &num_elem_in));
  CeedCall(CeedElemRestrictionGetElementSize(elem_rstr_in, &elem_size_in));
  CeedCall(CeedElemRestrictionGetNumComponents(elem_rstr_in, &num_comp_in));
  if (basis_in == CEED_BASIS_NONE) num_qpts_in = elem_size_in;
  else CeedCall(CeedBasisGetNumQuadraturePoints(basis_in, &num_qpts_in));

  CeedCall(CeedElemRestrictionGetType(elem_rstr_in, &elem_rstr_type_in));
  if (elem_rstr_type_in == CEED_RESTRICTION_ORIENTED) {
    CeedCall(CeedElemRestrictionGetOrientations(elem_rstr_in, CEED_MEM_HOST, &elem_rstr_orients_in));
  } else if (elem_rstr_type_in == CEED_RESTRICTION_CURL_ORIENTED) {
    CeedCall(CeedElemRestrictionGetCurlOrientations(elem_rstr_in, CEED_MEM_HOST, &elem_rstr_curl_orients_in));
  }

  if (elem_rstr_in != elem_rstr_out) {
    CeedCall(CeedElemRestrictionGetNumElements(elem_rstr_out, &num_elem_out));
    CeedCheck(num_elem_in == num_elem_out, ceed, CEED_ERROR_UNSUPPORTED,
              "Active input and output operator restrictions must have the same number of elements");
    CeedCall(CeedElemRestrictionGetElementSize(elem_rstr_out, &elem_size_out));
    CeedCall(CeedElemRestrictionGetNumComponents(elem_rstr_out, &num_comp_out));
    if (basis_out == CEED_BASIS_NONE) num_qpts_out = elem_size_out;
    else CeedCall(CeedBasisGetNumQuadraturePoints(basis_out, &num_qpts_out));
    CeedCheck(num_qpts_in == num_qpts_out, ceed, CEED_ERROR_UNSUPPORTED,
              "Active input and output bases must have the same number of quadrature points");

    CeedCall(CeedElemRestrictionGetType(elem_rstr_out, &elem_rstr_type_out));
    if (elem_rstr_type_out == CEED_RESTRICTION_ORIENTED) {
      CeedCall(CeedElemRestrictionGetOrientations(elem_rstr_out, CEED_MEM_HOST, &elem_rstr_orients_out));
    } else if (elem_rstr_type_out == CEED_RESTRICTION_CURL_ORIENTED) {
      CeedCall(CeedElemRestrictionGetCurlOrientations(elem_rstr_out, CEED_MEM_HOST, &elem_rstr_curl_orients_out));
    }
  } else {
    num_elem_out  = num_elem_in;
    elem_size_out = elem_size_in;
    num_comp_out  = num_comp_in;
    num_qpts_out  = num_qpts_in;

    elem_rstr_orients_out      = elem_rstr_orients_in;
    elem_rstr_curl_orients_out = elem_rstr_curl_orients_in;
  }
  local_num_entries = elem_size_out * num_comp_out * elem_size_in * num_comp_in * num_elem_in;

  // Loop over elements and put in data structure
  // We store B_mat_in, B_mat_out, BTD, elem_mat in row-major order
  CeedSize    count = 0;
  CeedScalar *vals, BTD_mat[elem_size_out * num_qpts_in * num_eval_modes_in[0]], elem_mat[elem_size_out * elem_size_in], *elem_mat_b = NULL;

  if (elem_rstr_curl_orients_in || elem_rstr_curl_orients_out) CeedCall(CeedCalloc(elem_size_out * elem_size_in, &elem_mat_b));

  CeedCall(CeedVectorGetArray(values, CEED_MEM_HOST, &vals));
  for (CeedSize e = 0; e < num_elem_in; e++) {
    for (CeedInt comp_in = 0; comp_in < num_comp_in; comp_in++) {
      for (CeedInt comp_out = 0; comp_out < num_comp_out; comp_out++) {
        // Compute B^T*D
        for (CeedSize n = 0; n < elem_size_out; n++) {
          for (CeedSize q = 0; q < num_qpts_in; q++) {
            for (CeedInt e_in = 0; e_in < num_eval_modes_in[0]; e_in++) {
              const CeedSize btd_index = n * (num_qpts_in * num_eval_modes_in[0]) + q * num_eval_modes_in[0] + e_in;
              CeedScalar     sum       = 0.0;

              for (CeedInt e_out = 0; e_out < num_eval_modes_out[0]; e_out++) {
                const CeedSize b_out_index     = (q * num_eval_modes_out[0] + e_out) * elem_size_out + n;
                const CeedSize eval_mode_index = ((e_in * num_comp_in + comp_in) * num_eval_modes_out[0] + e_out) * num_comp_out + comp_out;
                const CeedSize qf_index        = q * layout_qf[0] + eval_mode_index * layout_qf[1] + e * layout_qf[2];

                sum += B_mat_out[b_out_index] * assembled_qf_array[qf_index];
              }
              BTD_mat[btd_index] = sum;
            }
          }
        }

        // Form element matrix itself (for each block component)
        CeedCall(CeedMatrixMatrixMultiply(ceed, BTD_mat, B_mat_in, elem_mat, elem_size_out, elem_size_in, num_qpts_in * num_eval_modes_in[0]));

        // Transform the element matrix if required
        if (elem_rstr_orients_out) {
          const bool *elem_orients = &elem_rstr_orients_out[e * elem_size_out];

          for (CeedInt i = 0; i < elem_size_out; i++) {
            const double orient = elem_orients[i] ? -1.0 : 1.0;

            for (CeedInt j = 0; j < elem_size_in; j++) {
              elem_mat[i * elem_size_in + j] *= orient;
            }
          }
        } else if (elem_rstr_curl_orients_out) {
          const CeedInt8 *elem_curl_orients = &elem_rstr_curl_orients_out[e * 3 * elem_size_out];

          // T^T*(B^T*D*B)
          memcpy(elem_mat_b, elem_mat, elem_size_out * elem_size_in * sizeof(CeedScalar));
          for (CeedInt i = 0; i < elem_size_out; i++) {
            for (CeedInt j = 0; j < elem_size_in; j++) {
              elem_mat[i * elem_size_in + j] = elem_mat_b[i * elem_size_in + j] * elem_curl_orients[3 * i + 1] +
                                               (i > 0 ? elem_mat_b[(i - 1) * elem_size_in + j] * elem_curl_orients[3 * i - 1] : 0.0) +
                                               (i < elem_size_out - 1 ? elem_mat_b[(i + 1) * elem_size_in + j] * elem_curl_orients[3 * i + 3] : 0.0);
            }
          }
        }
        if (elem_rstr_orients_in) {
          const bool *elem_orients = &elem_rstr_orients_in[e * elem_size_in];

          for (CeedInt i = 0; i < elem_size_out; i++) {
            for (CeedInt j = 0; j < elem_size_in; j++) {
              elem_mat[i * elem_size_in + j] *= elem_orients[j] ? -1.0 : 1.0;
            }
          }
        } else if (elem_rstr_curl_orients_in) {
          const CeedInt8 *elem_curl_orients = &elem_rstr_curl_orients_in[e * 3 * elem_size_in];

          // (B^T*D*B)*T
          memcpy(elem_mat_b, elem_mat, elem_size_out * elem_size_in * sizeof(CeedScalar));
          for (CeedInt i = 0; i < elem_size_out; i++) {
            for (CeedInt j = 0; j < elem_size_in; j++) {
              elem_mat[i * elem_size_in + j] = elem_mat_b[i * elem_size_in + j] * elem_curl_orients[3 * j + 1] +
                                               (j > 0 ? elem_mat_b[i * elem_size_in + j - 1] * elem_curl_orients[3 * j - 1] : 0.0) +
                                               (j < elem_size_in - 1 ? elem_mat_b[i * elem_size_in + j + 1] * elem_curl_orients[3 * j + 3] : 0.0);
            }
          }
        }

        // Put element matrix in coordinate data structure
        for (CeedInt i = 0; i < elem_size_out; i++) {
          for (CeedInt j = 0; j < elem_size_in; j++) {
            vals[offset + count] = elem_mat[i * elem_size_in + j];
            count++;
          }
        }
      }
    }
  }
  CeedCheck(count == local_num_entries, ceed, CEED_ERROR_MAJOR, "Error computing entries");
  CeedCall(CeedVectorRestoreArray(values, &vals));

  // Cleanup
  CeedCall(CeedFree(&elem_mat_b));
  if (elem_rstr_type_in == CEED_RESTRICTION_ORIENTED) {
    CeedCall(CeedElemRestrictionRestoreOrientations(elem_rstr_in, &elem_rstr_orients_in));
  } else if (elem_rstr_type_in == CEED_RESTRICTION_CURL_ORIENTED) {
    CeedCall(CeedElemRestrictionRestoreCurlOrientations(elem_rstr_in, &elem_rstr_curl_orients_in));
  }
  if (elem_rstr_in != elem_rstr_out) {
    if (elem_rstr_type_out == CEED_RESTRICTION_ORIENTED) {
      CeedCall(CeedElemRestrictionRestoreOrientations(elem_rstr_out, &elem_rstr_orients_out));
    } else if (elem_rstr_type_out == CEED_RESTRICTION_CURL_ORIENTED) {
      CeedCall(CeedElemRestrictionRestoreCurlOrientations(elem_rstr_out, &elem_rstr_curl_orients_out));
    }
  }
  CeedCall(CeedVectorRestoreArrayRead(assembled_qf, &assembled_qf_array));
  CeedCall(CeedVectorDestroy(&assembled_qf));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Count number of entries for assembled CeedOperator

  @param[in]  op          CeedOperator to assemble
  @param[out] num_entries Number of entries in assembled representation

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedSingleOperatorAssemblyCountEntries(CeedOperator op, CeedSize *num_entries) {
  bool                is_composite;
  CeedInt             num_elem_in, elem_size_in, num_comp_in, num_elem_out, elem_size_out, num_comp_out;
  CeedElemRestriction rstr_in, rstr_out;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, op->ceed, CEED_ERROR_UNSUPPORTED, "Composite operator not supported");

  CeedCall(CeedOperatorGetActiveElemRestrictions(op, &rstr_in, &rstr_out));
  CeedCall(CeedElemRestrictionGetNumElements(rstr_in, &num_elem_in));
  CeedCall(CeedElemRestrictionGetElementSize(rstr_in, &elem_size_in));
  CeedCall(CeedElemRestrictionGetNumComponents(rstr_in, &num_comp_in));
  if (rstr_in != rstr_out) {
    CeedCall(CeedElemRestrictionGetNumElements(rstr_out, &num_elem_out));
    CeedCheck(num_elem_in == num_elem_out, op->ceed, CEED_ERROR_UNSUPPORTED,
              "Active input and output operator restrictions must have the same number of elements");
    CeedCall(CeedElemRestrictionGetElementSize(rstr_out, &elem_size_out));
    CeedCall(CeedElemRestrictionGetNumComponents(rstr_out, &num_comp_out));
  } else {
    num_elem_out  = num_elem_in;
    elem_size_out = elem_size_in;
    num_comp_out  = num_comp_in;
  }
  *num_entries = (CeedSize)elem_size_in * num_comp_in * elem_size_out * num_comp_out * num_elem_in;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Common code for creating a multigrid coarse operator and level transfer operators for a CeedOperator

  @param[in]  op_fine      Fine grid operator
  @param[in]  p_mult_fine  L-vector multiplicity in parallel gather/scatter, or NULL if not creating prolongation/restriction operators
  @param[in]  rstr_coarse  Coarse grid restriction
  @param[in]  basis_coarse Coarse grid active vector basis
  @param[in]  basis_c_to_f Basis for coarse to fine interpolation, or NULL if not creating prolongation/restriction operators
  @param[out] op_coarse    Coarse grid operator
  @param[out] op_prolong   Coarse to fine operator, or NULL
  @param[out] op_restrict  Fine to coarse operator, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedSingleOperatorMultigridLevel(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
                                            CeedBasis basis_c_to_f, CeedOperator *op_coarse, CeedOperator *op_prolong, CeedOperator *op_restrict) {
  bool                is_composite;
  Ceed                ceed;
  CeedInt             num_comp;
  CeedVector          mult_vec         = NULL;
  CeedElemRestriction rstr_p_mult_fine = NULL, rstr_fine = NULL;

  CeedCall(CeedOperatorGetCeed(op_fine, &ceed));

  // Check for composite operator
  CeedCall(CeedOperatorIsComposite(op_fine, &is_composite));
  CeedCheck(!is_composite, ceed, CEED_ERROR_UNSUPPORTED, "Automatic multigrid setup for composite operators not supported");

  // Coarse Grid
  CeedCall(CeedOperatorCreate(ceed, op_fine->qf, op_fine->dqf, op_fine->dqfT, op_coarse));
  // -- Clone input fields
  for (CeedInt i = 0; i < op_fine->qf->num_input_fields; i++) {
    if (op_fine->input_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      rstr_fine = op_fine->input_fields[i]->elem_rstr;
      CeedCall(CeedOperatorSetField(*op_coarse, op_fine->input_fields[i]->field_name, rstr_coarse, basis_coarse, CEED_VECTOR_ACTIVE));
    } else {
      CeedCall(CeedOperatorSetField(*op_coarse, op_fine->input_fields[i]->field_name, op_fine->input_fields[i]->elem_rstr,
                                    op_fine->input_fields[i]->basis, op_fine->input_fields[i]->vec));
    }
  }
  // -- Clone output fields
  for (CeedInt i = 0; i < op_fine->qf->num_output_fields; i++) {
    if (op_fine->output_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      CeedCall(CeedOperatorSetField(*op_coarse, op_fine->output_fields[i]->field_name, rstr_coarse, basis_coarse, CEED_VECTOR_ACTIVE));
    } else {
      CeedCall(CeedOperatorSetField(*op_coarse, op_fine->output_fields[i]->field_name, op_fine->output_fields[i]->elem_rstr,
                                    op_fine->output_fields[i]->basis, op_fine->output_fields[i]->vec));
    }
  }
  // -- Clone QFunctionAssemblyData
  CeedCall(CeedQFunctionAssemblyDataReferenceCopy(op_fine->qf_assembled, &(*op_coarse)->qf_assembled));

  // Multiplicity vector
  if (op_restrict || op_prolong) {
    CeedVector          mult_e_vec;
    CeedRestrictionType rstr_type;

    CeedCall(CeedElemRestrictionGetType(rstr_fine, &rstr_type));
    CeedCheck(rstr_type != CEED_RESTRICTION_CURL_ORIENTED, ceed, CEED_ERROR_UNSUPPORTED,
              "Element restrictions created with CeedElemRestrictionCreateCurlOriented are not supported");
    CeedCheck(p_mult_fine, ceed, CEED_ERROR_INCOMPATIBLE, "Prolongation or restriction operator creation requires fine grid multiplicity vector");
    CeedCall(CeedElemRestrictionCreateUnsignedCopy(rstr_fine, &rstr_p_mult_fine));
    CeedCall(CeedElemRestrictionCreateVector(rstr_fine, &mult_vec, &mult_e_vec));
    CeedCall(CeedVectorSetValue(mult_e_vec, 0.0));
    CeedCall(CeedElemRestrictionApply(rstr_p_mult_fine, CEED_NOTRANSPOSE, p_mult_fine, mult_e_vec, CEED_REQUEST_IMMEDIATE));
    CeedCall(CeedVectorSetValue(mult_vec, 0.0));
    CeedCall(CeedElemRestrictionApply(rstr_p_mult_fine, CEED_TRANSPOSE, mult_e_vec, mult_vec, CEED_REQUEST_IMMEDIATE));
    CeedCall(CeedVectorDestroy(&mult_e_vec));
    CeedCall(CeedVectorReciprocal(mult_vec));
  }

  // Clone name
  bool   has_name = op_fine->name;
  size_t name_len = op_fine->name ? strlen(op_fine->name) : 0;
  CeedCall(CeedOperatorSetName(*op_coarse, op_fine->name));

  // Check that coarse to fine basis is provided if prolong/restrict operators are requested
  CeedCheck(basis_c_to_f || (!op_restrict && !op_prolong), ceed, CEED_ERROR_INCOMPATIBLE,
            "Prolongation or restriction operator creation requires coarse-to-fine basis");

  // Restriction/Prolongation Operators
  CeedCall(CeedBasisGetNumComponents(basis_coarse, &num_comp));

  // Restriction
  if (op_restrict) {
    CeedInt             *num_comp_r_data;
    CeedQFunctionContext ctx_r;
    CeedQFunction        qf_restrict;

    CeedCall(CeedQFunctionCreateInteriorByName(ceed, "Scale", &qf_restrict));
    CeedCall(CeedCalloc(1, &num_comp_r_data));
    num_comp_r_data[0] = num_comp;
    CeedCall(CeedQFunctionContextCreate(ceed, &ctx_r));
    CeedCall(CeedQFunctionContextSetData(ctx_r, CEED_MEM_HOST, CEED_OWN_POINTER, sizeof(*num_comp_r_data), num_comp_r_data));
    CeedCall(CeedQFunctionSetContext(qf_restrict, ctx_r));
    CeedCall(CeedQFunctionContextDestroy(&ctx_r));
    CeedCall(CeedQFunctionAddInput(qf_restrict, "input", num_comp, CEED_EVAL_NONE));
    CeedCall(CeedQFunctionAddInput(qf_restrict, "scale", num_comp, CEED_EVAL_NONE));
    CeedCall(CeedQFunctionAddOutput(qf_restrict, "output", num_comp, CEED_EVAL_INTERP));
    CeedCall(CeedQFunctionSetUserFlopsEstimate(qf_restrict, num_comp));

    CeedCall(CeedOperatorCreate(ceed, qf_restrict, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, op_restrict));
    CeedCall(CeedOperatorSetField(*op_restrict, "input", rstr_fine, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
    CeedCall(CeedOperatorSetField(*op_restrict, "scale", rstr_p_mult_fine, CEED_BASIS_NONE, mult_vec));
    CeedCall(CeedOperatorSetField(*op_restrict, "output", rstr_coarse, basis_c_to_f, CEED_VECTOR_ACTIVE));

    // Set name
    char *restriction_name;

    CeedCall(CeedCalloc(17 + name_len, &restriction_name));
    sprintf(restriction_name, "restriction%s%s", has_name ? " for " : "", has_name ? op_fine->name : "");
    CeedCall(CeedOperatorSetName(*op_restrict, restriction_name));
    CeedCall(CeedFree(&restriction_name));

    // Check
    CeedCall(CeedOperatorCheckReady(*op_restrict));

    // Cleanup
    CeedCall(CeedQFunctionDestroy(&qf_restrict));
  }

  // Prolongation
  if (op_prolong) {
    CeedInt             *num_comp_p_data;
    CeedQFunctionContext ctx_p;
    CeedQFunction        qf_prolong;

    CeedCall(CeedQFunctionCreateInteriorByName(ceed, "Scale", &qf_prolong));
    CeedCall(CeedCalloc(1, &num_comp_p_data));
    num_comp_p_data[0] = num_comp;
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
    CeedCall(CeedOperatorSetField(*op_prolong, "scale", rstr_p_mult_fine, CEED_BASIS_NONE, mult_vec));
    CeedCall(CeedOperatorSetField(*op_prolong, "output", rstr_fine, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

    // Set name
    char *prolongation_name;

    CeedCall(CeedCalloc(18 + name_len, &prolongation_name));
    sprintf(prolongation_name, "prolongation%s%s", has_name ? " for " : "", has_name ? op_fine->name : "");
    CeedCall(CeedOperatorSetName(*op_prolong, prolongation_name));
    CeedCall(CeedFree(&prolongation_name));

    // Check
    CeedCall(CeedOperatorCheckReady(*op_prolong));

    // Cleanup
    CeedCall(CeedQFunctionDestroy(&qf_prolong));
  }

  // Check
  CeedCall(CeedOperatorCheckReady(*op_coarse));

  // Cleanup
  CeedCall(CeedVectorDestroy(&mult_vec));
  CeedCall(CeedElemRestrictionDestroy(&rstr_p_mult_fine));
  CeedCall(CeedBasisDestroy(&basis_c_to_f));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build 1D mass matrix and Laplacian with perturbation

  @param[in]  interp_1d   Interpolation matrix in one dimension
  @param[in]  grad_1d     Gradient matrix in one dimension
  @param[in]  q_weight_1d Quadrature weights in one dimension
  @param[in]  P_1d        Number of basis nodes in one dimension
  @param[in]  Q_1d        Number of quadrature points in one dimension
  @param[in]  dim         Dimension of basis
  @param[out] mass        Assembled mass matrix in one dimension
  @param[out] laplace     Assembled perturbed Laplacian in one dimension

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
CeedPragmaOptimizeOff
static int CeedBuildMassLaplace(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *q_weight_1d, CeedInt P_1d, CeedInt Q_1d,
                                CeedInt dim, CeedScalar *mass, CeedScalar *laplace) {
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
  @brief Create point block restriction for active operator field

  @param[in]  rstr             Original CeedElemRestriction for active field
  @param[out] point_block_rstr Address of the variable where the newly created CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorCreateActivePointBlockRestriction(CeedElemRestriction rstr, CeedElemRestriction *point_block_rstr) {
  Ceed           ceed;
  CeedInt        num_elem, num_comp, shift, elem_size, comp_stride, *point_block_offsets;
  CeedSize       l_size;
  const CeedInt *offsets;

  CeedCall(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCall(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));

  // Expand offsets
  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCall(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCall(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  CeedCall(CeedElemRestrictionGetCompStride(rstr, &comp_stride));
  CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
  shift = num_comp;
  if (comp_stride != 1) shift *= num_comp;
  CeedCall(CeedCalloc(num_elem * elem_size, &point_block_offsets));
  for (CeedInt i = 0; i < num_elem * elem_size; i++) {
    point_block_offsets[i] = offsets[i] * shift;
  }

  // Create new restriction
  CeedCall(CeedElemRestrictionCreate(ceed, num_elem, elem_size, num_comp * num_comp, 1, l_size * num_comp, CEED_MEM_HOST, CEED_OWN_POINTER,
                                     point_block_offsets, point_block_rstr));

  // Cleanup
  CeedCall(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create object holding CeedQFunction assembly data for CeedOperator

  @param[in]  ceed A Ceed object where the CeedQFunctionAssemblyData will be created
  @param[out] data Address of the variable where the newly created CeedQFunctionAssemblyData will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataCreate(Ceed ceed, CeedQFunctionAssemblyData *data) {
  CeedCall(CeedCalloc(1, data));
  (*data)->ref_count = 1;
  (*data)->ceed      = ceed;
  CeedCall(CeedReference(ceed));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedQFunctionAssemblyData

  @param[in,out] data CeedQFunctionAssemblyData to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataReference(CeedQFunctionAssemblyData data) {
  data->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set re-use of CeedQFunctionAssemblyData

  @param[in,out] data       CeedQFunctionAssemblyData to mark for reuse
  @param[in]     reuse_data Boolean flag indicating data re-use

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

  @param[in,out] data              CeedQFunctionAssemblyData to mark as stale
  @param[in]     needs_data_update Boolean flag indicating if update is needed or completed

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataSetUpdateNeeded(CeedQFunctionAssemblyData data, bool needs_data_update) {
  data->needs_data_update = needs_data_update;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Determine if QFunctionAssemblyData needs update

  @param[in]  data             CeedQFunctionAssemblyData to mark as stale
  @param[out] is_update_needed Boolean flag indicating if re-assembly is required

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataIsUpdateNeeded(CeedQFunctionAssemblyData data, bool *is_update_needed) {
  *is_update_needed = !data->reuse_data || data->needs_data_update;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedQFunctionAssemblyData.

  Both pointers should be destroyed with `CeedCeedQFunctionAssemblyDataDestroy()`.

  Note: If the value of `data_copy` passed to this function is non-NULL, then it is assumed that `*data_copy` is a pointer to a
        CeedQFunctionAssemblyData. This CeedQFunctionAssemblyData will be destroyed if `data_copy` is the only reference to this
        CeedQFunctionAssemblyData.

  @param[in]     data      CeedQFunctionAssemblyData to copy reference to
  @param[in,out] data_copy Variable to store copied reference

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

  @param[in]  data     CeedQFunctionAssemblyData to retrieve status
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

  @param[in,out] data CeedQFunctionAssemblyData to set objects
  @param[in]     vec  CeedVector to store assembled CeedQFunction at quadrature points
  @param[in]     rstr CeedElemRestriction for CeedVector containing assembled CeedQFunction

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
  CeedCheck(data->is_setup, data->ceed, CEED_ERROR_INCOMPLETE, "Internal objects not set; must call CeedQFunctionAssemblyDataSetObjects first.");

  CeedCall(CeedVectorReferenceCopy(data->vec, vec));
  CeedCall(CeedElemRestrictionReferenceCopy(data->rstr, rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy CeedQFunctionAssemblyData

  @param[in,out] data  CeedQFunctionAssemblyData to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedQFunctionAssemblyDataDestroy(CeedQFunctionAssemblyData *data) {
  if (!*data || --(*data)->ref_count > 0) {
    *data = NULL;
    return CEED_ERROR_SUCCESS;
  }
  CeedCall(CeedDestroy(&(*data)->ceed));
  CeedCall(CeedVectorDestroy(&(*data)->vec));
  CeedCall(CeedElemRestrictionDestroy(&(*data)->rstr));

  CeedCall(CeedFree(data));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedOperatorAssemblyData

  @param[in]  op   CeedOperator to assemble
  @param[out] data CeedQFunctionAssemblyData

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
  @brief Create object holding CeedOperator assembly data.

  The CeedOperatorAssemblyData holds an array with references to every active CeedBasis used in the CeedOperator.
  An array with references to the corresponding active CeedElemRestrictions is also stored.
  For each active CeedBasis, the CeedOperatorAssemblyData holds an array of all input and output CeedEvalModes for this CeedBasis.
  The CeedOperatorAssemblyData holds an array of offsets for indexing into the assembled CeedQFunction arrays to the row representing each
CeedEvalMode.
  The number of input columns across all active bases for the assembled CeedQFunction is also stored.
  Lastly, the CeedOperatorAssembly data holds assembled matrices representing the full action of the CeedBasis for all CeedEvalModes.

  @param[in]  ceed Ceed object where the CeedOperatorAssemblyData will be created
  @param[in]  op   CeedOperator to be assembled
  @param[out] data Address of the variable where the newly created CeedOperatorAssemblyData will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorAssemblyDataCreate(Ceed ceed, CeedOperator op, CeedOperatorAssemblyData *data) {
  CeedInt             num_active_bases_in = 0, num_active_bases_out = 0, offset = 0;
  CeedInt             num_input_fields, *num_eval_modes_in = NULL, num_output_fields, *num_eval_modes_out = NULL;
  CeedSize          **eval_mode_offsets_in = NULL, **eval_mode_offsets_out = NULL;
  CeedEvalMode      **eval_modes_in = NULL, **eval_modes_out = NULL;
  CeedQFunctionField *qf_fields;
  CeedQFunction       qf;
  CeedOperatorField  *op_fields;
  bool                is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, ceed, CEED_ERROR_INCOMPATIBLE, "Can only create CeedOperator assembly data for non-composite operators.");

  // Allocate
  CeedCall(CeedCalloc(1, data));
  (*data)->ceed = ceed;
  CeedCall(CeedReference(ceed));

  // Build OperatorAssembly data
  CeedCall(CeedOperatorGetQFunction(op, &qf));
  CeedCall(CeedQFunctionGetFields(qf, &num_input_fields, &qf_fields, NULL, NULL));
  CeedCall(CeedOperatorGetFields(op, NULL, &op_fields, NULL, NULL));

  // Determine active input basis
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedVector vec;

    CeedCall(CeedOperatorFieldGetVector(op_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedInt      index = -1, num_comp, q_comp;
      CeedEvalMode eval_mode;
      CeedBasis    basis_in = NULL;

      CeedCall(CeedOperatorFieldGetBasis(op_fields[i], &basis_in));
      CeedCall(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCall(CeedBasisGetNumComponents(basis_in, &num_comp));
      CeedCall(CeedBasisGetNumQuadratureComponents(basis_in, eval_mode, &q_comp));
      for (CeedInt i = 0; i < num_active_bases_in; i++) {
        if ((*data)->active_bases_in[i] == basis_in) index = i;
      }
      if (index == -1) {
        CeedElemRestriction elem_rstr_in;

        index = num_active_bases_in;
        CeedCall(CeedRealloc(num_active_bases_in + 1, &(*data)->active_bases_in));
        (*data)->active_bases_in[num_active_bases_in] = NULL;
        CeedCall(CeedBasisReferenceCopy(basis_in, &(*data)->active_bases_in[num_active_bases_in]));
        CeedCall(CeedRealloc(num_active_bases_in + 1, &(*data)->active_elem_rstrs_in));
        (*data)->active_elem_rstrs_in[num_active_bases_in] = NULL;
        CeedCall(CeedOperatorFieldGetElemRestriction(op_fields[i], &elem_rstr_in));
        CeedCall(CeedElemRestrictionReferenceCopy(elem_rstr_in, &(*data)->active_elem_rstrs_in[num_active_bases_in]));
        CeedCall(CeedRealloc(num_active_bases_in + 1, &num_eval_modes_in));
        num_eval_modes_in[index] = 0;
        CeedCall(CeedRealloc(num_active_bases_in + 1, &eval_modes_in));
        eval_modes_in[index] = NULL;
        CeedCall(CeedRealloc(num_active_bases_in + 1, &eval_mode_offsets_in));
        eval_mode_offsets_in[index] = NULL;
        CeedCall(CeedRealloc(num_active_bases_in + 1, &(*data)->assembled_bases_in));
        (*data)->assembled_bases_in[index] = NULL;
        num_active_bases_in++;
      }
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF Assembly
        CeedCall(CeedRealloc(num_eval_modes_in[index] + q_comp, &eval_modes_in[index]));
        CeedCall(CeedRealloc(num_eval_modes_in[index] + q_comp, &eval_mode_offsets_in[index]));
        for (CeedInt d = 0; d < q_comp; d++) {
          eval_modes_in[index][num_eval_modes_in[index] + d]        = eval_mode;
          eval_mode_offsets_in[index][num_eval_modes_in[index] + d] = offset;
          offset += num_comp;
        }
        num_eval_modes_in[index] += q_comp;
      }
    }
  }

  // Determine active output basis
  CeedCall(CeedQFunctionGetFields(qf, NULL, NULL, &num_output_fields, &qf_fields));
  CeedCall(CeedOperatorGetFields(op, NULL, NULL, NULL, &op_fields));
  offset = 0;
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedVector vec;

    CeedCall(CeedOperatorFieldGetVector(op_fields[i], &vec));
    if (vec == CEED_VECTOR_ACTIVE) {
      CeedInt      index = -1, num_comp, q_comp;
      CeedEvalMode eval_mode;
      CeedBasis    basis_out = NULL;

      CeedCall(CeedOperatorFieldGetBasis(op_fields[i], &basis_out));
      CeedCall(CeedQFunctionFieldGetEvalMode(qf_fields[i], &eval_mode));
      CeedCall(CeedBasisGetNumComponents(basis_out, &num_comp));
      CeedCall(CeedBasisGetNumQuadratureComponents(basis_out, eval_mode, &q_comp));
      for (CeedInt i = 0; i < num_active_bases_out; i++) {
        if ((*data)->active_bases_out[i] == basis_out) index = i;
      }
      if (index == -1) {
        CeedElemRestriction elem_rstr_out;

        index = num_active_bases_out;
        CeedCall(CeedRealloc(num_active_bases_out + 1, &(*data)->active_bases_out));
        (*data)->active_bases_out[num_active_bases_out] = NULL;
        CeedCall(CeedBasisReferenceCopy(basis_out, &(*data)->active_bases_out[num_active_bases_out]));
        CeedCall(CeedRealloc(num_active_bases_out + 1, &(*data)->active_elem_rstrs_out));
        (*data)->active_elem_rstrs_out[num_active_bases_out] = NULL;
        CeedCall(CeedOperatorFieldGetElemRestriction(op_fields[i], &elem_rstr_out));
        CeedCall(CeedElemRestrictionReferenceCopy(elem_rstr_out, &(*data)->active_elem_rstrs_out[num_active_bases_out]));
        CeedCall(CeedRealloc(num_active_bases_out + 1, &num_eval_modes_out));
        num_eval_modes_out[index] = 0;
        CeedCall(CeedRealloc(num_active_bases_out + 1, &eval_modes_out));
        eval_modes_out[index] = NULL;
        CeedCall(CeedRealloc(num_active_bases_out + 1, &eval_mode_offsets_out));
        eval_mode_offsets_out[index] = NULL;
        CeedCall(CeedRealloc(num_active_bases_out + 1, &(*data)->assembled_bases_out));
        (*data)->assembled_bases_out[index] = NULL;
        num_active_bases_out++;
      }
      if (eval_mode != CEED_EVAL_WEIGHT) {
        // q_comp = 1 if CEED_EVAL_NONE, CEED_EVAL_WEIGHT caught by QF Assembly
        CeedCall(CeedRealloc(num_eval_modes_out[index] + q_comp, &eval_modes_out[index]));
        CeedCall(CeedRealloc(num_eval_modes_out[index] + q_comp, &eval_mode_offsets_out[index]));
        for (CeedInt d = 0; d < q_comp; d++) {
          eval_modes_out[index][num_eval_modes_out[index] + d]        = eval_mode;
          eval_mode_offsets_out[index][num_eval_modes_out[index] + d] = offset;
          offset += num_comp;
        }
        num_eval_modes_out[index] += q_comp;
      }
    }
  }
  (*data)->num_active_bases_in   = num_active_bases_in;
  (*data)->num_eval_modes_in     = num_eval_modes_in;
  (*data)->eval_modes_in         = eval_modes_in;
  (*data)->eval_mode_offsets_in  = eval_mode_offsets_in;
  (*data)->num_active_bases_out  = num_active_bases_out;
  (*data)->num_eval_modes_out    = num_eval_modes_out;
  (*data)->eval_modes_out        = eval_modes_out;
  (*data)->eval_mode_offsets_out = eval_mode_offsets_out;
  (*data)->num_output_components = offset;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedOperator CeedEvalModes for assembly.

  Note: See CeedOperatorAssemblyDataCreate for a full description of the data stored in this object.

  @param[in]  data                  CeedOperatorAssemblyData
  @param[out] num_active_bases_in   Total number of active bases for input
  @param[out] num_eval_modes_in     Pointer to hold array of numbers of input CeedEvalModes, or NULL.
                                      `eval_modes_in[0]` holds an array of eval modes for the first active basis.
  @param[out] eval_modes_in         Pointer to hold arrays of input CeedEvalModes, or NULL.
  @param[out] eval_mode_offsets_in  Pointer to hold arrays of input offsets at each quadrature point.
  @param[out] num_active_bases_out  Total number of active bases for output
  @param[out] num_eval_modes_out    Pointer to hold array of numbers of output CeedEvalModes, or NULL
  @param[out] eval_modes_out        Pointer to hold arrays of output CeedEvalModes, or NULL.
  @param[out] eval_mode_offsets_out Pointer to hold arrays of output offsets at each quadrature point
  @param[out] num_output_components The number of columns in the assembled CeedQFunction matrix for each quadrature point,
                                      including contributions of all active bases

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorAssemblyDataGetEvalModes(CeedOperatorAssemblyData data, CeedInt *num_active_bases_in, CeedInt **num_eval_modes_in,
                                         const CeedEvalMode ***eval_modes_in, CeedSize ***eval_mode_offsets_in, CeedInt *num_active_bases_out,
                                         CeedInt **num_eval_modes_out, const CeedEvalMode ***eval_modes_out, CeedSize ***eval_mode_offsets_out,
                                         CeedSize *num_output_components) {
  if (num_active_bases_in) *num_active_bases_in = data->num_active_bases_in;
  if (num_eval_modes_in) *num_eval_modes_in = data->num_eval_modes_in;
  if (eval_modes_in) *eval_modes_in = (const CeedEvalMode **)data->eval_modes_in;
  if (eval_mode_offsets_in) *eval_mode_offsets_in = data->eval_mode_offsets_in;
  if (num_active_bases_out) *num_active_bases_out = data->num_active_bases_out;
  if (num_eval_modes_out) *num_eval_modes_out = data->num_eval_modes_out;
  if (eval_modes_out) *eval_modes_out = (const CeedEvalMode **)data->eval_modes_out;
  if (eval_mode_offsets_out) *eval_mode_offsets_out = data->eval_mode_offsets_out;
  if (num_output_components) *num_output_components = data->num_output_components;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedOperator CeedBasis data for assembly.

  Note: See CeedOperatorAssemblyDataCreate for a full description of the data stored in this object.

  @param[in]  data                 CeedOperatorAssemblyData
  @param[out] num_active_bases_in  Number of active input bases, or NULL
  @param[out] active_bases_in      Pointer to hold active input CeedBasis, or NULL
  @param[out] assembled_bases_in   Pointer to hold assembled active input B, or NULL
  @param[out] num_active_bases_out Number of active output bases, or NULL
  @param[out] active_bases_out     Pointer to hold active output CeedBasis, or NULL
  @param[out] assembled_bases_out  Pointer to hold assembled active output B, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorAssemblyDataGetBases(CeedOperatorAssemblyData data, CeedInt *num_active_bases_in, CeedBasis **active_bases_in,
                                     const CeedScalar ***assembled_bases_in, CeedInt *num_active_bases_out, CeedBasis **active_bases_out,
                                     const CeedScalar ***assembled_bases_out) {
  // Assemble B_in, B_out if needed
  if (assembled_bases_in && !data->assembled_bases_in[0]) {
    CeedInt num_qpts;

    if (data->active_bases_in[0] == CEED_BASIS_NONE) CeedCall(CeedElemRestrictionGetElementSize(data->active_elem_rstrs_in[0], &num_qpts));
    else CeedCall(CeedBasisGetNumQuadraturePoints(data->active_bases_in[0], &num_qpts));
    for (CeedInt b = 0; b < data->num_active_bases_in; b++) {
      bool        has_eval_none = false;
      CeedInt     num_nodes;
      CeedScalar *B_in = NULL, *identity = NULL;

      CeedCall(CeedElemRestrictionGetElementSize(data->active_elem_rstrs_in[b], &num_nodes));
      CeedCall(CeedCalloc(num_qpts * num_nodes * data->num_eval_modes_in[b], &B_in));

      for (CeedInt i = 0; i < data->num_eval_modes_in[b]; i++) {
        has_eval_none = has_eval_none || (data->eval_modes_in[b][i] == CEED_EVAL_NONE);
      }
      if (has_eval_none) {
        CeedCall(CeedCalloc(num_qpts * num_nodes, &identity));
        for (CeedInt i = 0; i < (num_nodes < num_qpts ? num_nodes : num_qpts); i++) {
          identity[i * num_nodes + i] = 1.0;
        }
      }

      for (CeedInt q = 0; q < num_qpts; q++) {
        for (CeedInt n = 0; n < num_nodes; n++) {
          CeedInt      d_in              = 0, q_comp_in;
          CeedEvalMode eval_mode_in_prev = CEED_EVAL_NONE;

          for (CeedInt e_in = 0; e_in < data->num_eval_modes_in[b]; e_in++) {
            const CeedInt     qq = data->num_eval_modes_in[b] * q;
            const CeedScalar *B  = NULL;

            CeedCall(CeedOperatorGetBasisPointer(data->active_bases_in[b], data->eval_modes_in[b][e_in], identity, &B));
            CeedCall(CeedBasisGetNumQuadratureComponents(data->active_bases_in[b], data->eval_modes_in[b][e_in], &q_comp_in));
            if (q_comp_in > 1) {
              if (e_in == 0 || data->eval_modes_in[b][e_in] != eval_mode_in_prev) d_in = 0;
              else B = &B[(++d_in) * num_qpts * num_nodes];
            }
            eval_mode_in_prev                 = data->eval_modes_in[b][e_in];
            B_in[(qq + e_in) * num_nodes + n] = B[q * num_nodes + n];
          }
        }
      }
      if (identity) CeedCall(CeedFree(&identity));
      data->assembled_bases_in[b] = B_in;
    }
  }

  if (assembled_bases_out && !data->assembled_bases_out[0]) {
    CeedInt num_qpts;

    if (data->active_bases_out[0] == CEED_BASIS_NONE) CeedCall(CeedElemRestrictionGetElementSize(data->active_elem_rstrs_out[0], &num_qpts));
    else CeedCall(CeedBasisGetNumQuadraturePoints(data->active_bases_out[0], &num_qpts));
    for (CeedInt b = 0; b < data->num_active_bases_out; b++) {
      bool        has_eval_none = false;
      CeedInt     num_nodes;
      CeedScalar *B_out = NULL, *identity = NULL;

      CeedCall(CeedElemRestrictionGetElementSize(data->active_elem_rstrs_out[b], &num_nodes));
      CeedCall(CeedCalloc(num_qpts * num_nodes * data->num_eval_modes_out[b], &B_out));

      for (CeedInt i = 0; i < data->num_eval_modes_out[b]; i++) {
        has_eval_none = has_eval_none || (data->eval_modes_out[b][i] == CEED_EVAL_NONE);
      }
      if (has_eval_none) {
        CeedCall(CeedCalloc(num_qpts * num_nodes, &identity));
        for (CeedInt i = 0; i < (num_nodes < num_qpts ? num_nodes : num_qpts); i++) {
          identity[i * num_nodes + i] = 1.0;
        }
      }

      for (CeedInt q = 0; q < num_qpts; q++) {
        for (CeedInt n = 0; n < num_nodes; n++) {
          CeedInt      d_out              = 0, q_comp_out;
          CeedEvalMode eval_mode_out_prev = CEED_EVAL_NONE;

          for (CeedInt e_out = 0; e_out < data->num_eval_modes_out[b]; e_out++) {
            const CeedInt     qq = data->num_eval_modes_out[b] * q;
            const CeedScalar *B  = NULL;

            CeedCall(CeedOperatorGetBasisPointer(data->active_bases_out[b], data->eval_modes_out[b][e_out], identity, &B));
            CeedCall(CeedBasisGetNumQuadratureComponents(data->active_bases_out[b], data->eval_modes_out[b][e_out], &q_comp_out));
            if (q_comp_out > 1) {
              if (e_out == 0 || data->eval_modes_out[b][e_out] != eval_mode_out_prev) d_out = 0;
              else B = &B[(++d_out) * num_qpts * num_nodes];
            }
            eval_mode_out_prev                  = data->eval_modes_out[b][e_out];
            B_out[(qq + e_out) * num_nodes + n] = B[q * num_nodes + n];
          }
        }
      }
      if (identity) CeedCall(CeedFree(&identity));
      data->assembled_bases_out[b] = B_out;
    }
  }

  // Pass out assembled data
  if (num_active_bases_in) *num_active_bases_in = data->num_active_bases_in;
  if (active_bases_in) *active_bases_in = data->active_bases_in;
  if (assembled_bases_in) *assembled_bases_in = (const CeedScalar **)data->assembled_bases_in;
  if (num_active_bases_out) *num_active_bases_out = data->num_active_bases_out;
  if (active_bases_out) *active_bases_out = data->active_bases_out;
  if (assembled_bases_out) *assembled_bases_out = (const CeedScalar **)data->assembled_bases_out;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedOperator CeedBasis data for assembly.

  Note: See CeedOperatorAssemblyDataCreate for a full description of the data stored in this object.

  @param[in]  data                      CeedOperatorAssemblyData
  @param[out] num_active_elem_rstrs_in  Number of active input element restrictions, or NULL
  @param[out] active_elem_rstrs_in      Pointer to hold active input CeedElemRestrictions, or NULL
  @param[out] num_active_elem_rstrs_out Number of active output element restrictions, or NULL
  @param[out] active_elem_rstrs_out     Pointer to hold active output CeedElemRestrictions, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorAssemblyDataGetElemRestrictions(CeedOperatorAssemblyData data, CeedInt *num_active_elem_rstrs_in,
                                                CeedElemRestriction **active_elem_rstrs_in, CeedInt *num_active_elem_rstrs_out,
                                                CeedElemRestriction **active_elem_rstrs_out) {
  if (num_active_elem_rstrs_in) *num_active_elem_rstrs_in = data->num_active_bases_in;
  if (active_elem_rstrs_in) *active_elem_rstrs_in = data->active_elem_rstrs_in;
  if (num_active_elem_rstrs_out) *num_active_elem_rstrs_out = data->num_active_bases_out;
  if (active_elem_rstrs_out) *active_elem_rstrs_out = data->active_elem_rstrs_out;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy CeedOperatorAssemblyData

  @param[in,out] data CeedOperatorAssemblyData to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorAssemblyDataDestroy(CeedOperatorAssemblyData *data) {
  if (!*data) {
    *data = NULL;
    return CEED_ERROR_SUCCESS;
  }
  CeedCall(CeedDestroy(&(*data)->ceed));
  for (CeedInt b = 0; b < (*data)->num_active_bases_in; b++) {
    CeedCall(CeedBasisDestroy(&(*data)->active_bases_in[b]));
    CeedCall(CeedElemRestrictionDestroy(&(*data)->active_elem_rstrs_in[b]));
    CeedCall(CeedFree(&(*data)->eval_modes_in[b]));
    CeedCall(CeedFree(&(*data)->eval_mode_offsets_in[b]));
    CeedCall(CeedFree(&(*data)->assembled_bases_in[b]));
  }
  for (CeedInt b = 0; b < (*data)->num_active_bases_out; b++) {
    CeedCall(CeedBasisDestroy(&(*data)->active_bases_out[b]));
    CeedCall(CeedElemRestrictionDestroy(&(*data)->active_elem_rstrs_out[b]));
    CeedCall(CeedFree(&(*data)->eval_modes_out[b]));
    CeedCall(CeedFree(&(*data)->eval_mode_offsets_out[b]));
    CeedCall(CeedFree(&(*data)->assembled_bases_out[b]));
  }
  CeedCall(CeedFree(&(*data)->active_bases_in));
  CeedCall(CeedFree(&(*data)->active_bases_out));
  CeedCall(CeedFree(&(*data)->active_elem_rstrs_in));
  CeedCall(CeedFree(&(*data)->active_elem_rstrs_out));
  CeedCall(CeedFree(&(*data)->num_eval_modes_in));
  CeedCall(CeedFree(&(*data)->num_eval_modes_out));
  CeedCall(CeedFree(&(*data)->eval_modes_in));
  CeedCall(CeedFree(&(*data)->eval_modes_out));
  CeedCall(CeedFree(&(*data)->eval_mode_offsets_in));
  CeedCall(CeedFree(&(*data)->eval_mode_offsets_out));
  CeedCall(CeedFree(&(*data)->assembled_bases_in));
  CeedCall(CeedFree(&(*data)->assembled_bases_out));

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

  This returns a CeedVector containing a matrix at each quadrature point providing the action of the CeedQFunction associated with the CeedOperator.
  The vector `assembled` is of shape `[num_elements, num_input_fields, num_output_fields, num_quad_points]` and contains column-major matrices
representing the action of the CeedQFunction for a corresponding quadrature point on an element.

  Inputs and outputs are in the order provided by the user when adding CeedOperator fields.
  For example, a CeedQFunction with inputs 'u' and 'gradu' and outputs 'gradv' and 'v', provided in that order, would result in an assembled QFunction
that consists of (1 + dim) x (dim + 1) matrices at each quadrature point acting on the input [u, du_0, du_1] and producing the output [dv_0, dv_1, v].

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op        CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedQFunction at quadrature points
  @param[out] rstr      CeedElemRestriction for CeedVector containing assembled CeedQFunction
  @param[in]  request   Address of CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

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
    if (op_fallback) CeedCall(CeedOperatorLinearAssembleQFunction(op_fallback, assembled, rstr, request));
    else return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support CeedOperatorLinearAssembleQFunction");
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble CeedQFunction and store result internally.

  Return copied references of stored data to the caller.
  Caller is responsible for ownership and destruction of the copied references.
  See also @ref CeedOperatorLinearAssembleQFunction

  Note: If the value of `assembled` or `rstr` passed to this function are non-NULL, then it is assumed that they hold valid pointers.
        These objects will be destroyed if `*assembled` or `*rstr` is the only reference to the object.

  @param[in]  op        CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedQFunction at quadrature points
  @param[out] rstr      CeedElemRestriction for CeedVector containing assembledCeedQFunction
  @param[in]  request   Address of CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssembleQFunctionBuildOrUpdate(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
  int (*LinearAssembleQFunctionUpdate)(CeedOperator, CeedVector, CeedElemRestriction, CeedRequest *) = NULL;
  CeedOperator op_assemble                                                                           = NULL;
  CeedOperator op_fallback_parent                                                                    = NULL;

  CeedCall(CeedOperatorCheckReady(op));

  // Determine if fallback parent or operator has implementation
  CeedCall(CeedOperatorGetFallbackParent(op, &op_fallback_parent));
  if (op_fallback_parent && op_fallback_parent->LinearAssembleQFunctionUpdate) {
    // -- Backend version for op fallback parent is faster, if it exists
    LinearAssembleQFunctionUpdate = op_fallback_parent->LinearAssembleQFunctionUpdate;
    op_assemble                   = op_fallback_parent;
  } else if (op->LinearAssembleQFunctionUpdate) {
    // -- Backend version for op
    LinearAssembleQFunctionUpdate = op->LinearAssembleQFunctionUpdate;
    op_assemble                   = op;
  }

  // Assemble QFunction
  if (LinearAssembleQFunctionUpdate) {
    // Backend or fallback parent version
    bool                qf_assembled_is_setup;
    CeedVector          assembled_vec  = NULL;
    CeedElemRestriction assembled_rstr = NULL;

    CeedCall(CeedQFunctionAssemblyDataIsSetup(op->qf_assembled, &qf_assembled_is_setup));
    if (qf_assembled_is_setup) {
      bool update_needed;

      CeedCall(CeedQFunctionAssemblyDataGetObjects(op->qf_assembled, &assembled_vec, &assembled_rstr));
      CeedCall(CeedQFunctionAssemblyDataIsUpdateNeeded(op->qf_assembled, &update_needed));
      if (update_needed) CeedCall(LinearAssembleQFunctionUpdate(op_assemble, assembled_vec, assembled_rstr, request));
    } else {
      CeedCall(CeedOperatorLinearAssembleQFunction(op_assemble, &assembled_vec, &assembled_rstr, request));
      CeedCall(CeedQFunctionAssemblyDataSetObjects(op->qf_assembled, assembled_vec, assembled_rstr));
    }
    CeedCall(CeedQFunctionAssemblyDataSetUpdateNeeded(op->qf_assembled, false));

    // Copy reference from internally held copy
    CeedCall(CeedVectorReferenceCopy(assembled_vec, assembled));
    CeedCall(CeedElemRestrictionReferenceCopy(assembled_rstr, rstr));
    CeedCall(CeedVectorDestroy(&assembled_vec));
    CeedCall(CeedElemRestrictionDestroy(&assembled_rstr));
  } else {
    // Operator fallback
    CeedOperator op_fallback;

    CeedCall(CeedOperatorGetFallback(op, &op_fallback));
    if (op_fallback) CeedCall(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op_fallback, assembled, rstr, request));
    else return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support CeedOperatorLinearAssembleQFunctionUpdate");
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble the diagonal of a square linear CeedOperator

  This overwrites a CeedVector with the diagonal of a linear CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and composite CeedOperators with single field sub-operators are supported.

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op        CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedOperator diagonal
  @param[in]  request   Address of CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssembleDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  bool     is_composite;
  CeedSize input_size = 0, output_size = 0;

  CeedCall(CeedOperatorCheckReady(op));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));

  CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  CeedCheck(input_size == output_size, op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");

  // Early exit for empty operator
  if (!is_composite) {
    CeedInt num_elem = 0;

    CeedCall(CeedOperatorGetNumElements(op, &num_elem));
    if (num_elem == 0) return CEED_ERROR_SUCCESS;
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

  Note: Currently only non-composite CeedOperators with a single field and composite CeedOperators with single field sub-operators are supported.

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op        CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedOperator diagonal
  @param[in]  request   Address of CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssembleAddDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  bool     is_composite;
  CeedSize input_size = 0, output_size = 0;

  CeedCall(CeedOperatorCheckReady(op));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));

  CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  CeedCheck(input_size == output_size, op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");

  // Early exit for empty operator
  if (!is_composite) {
    CeedInt num_elem = 0;

    CeedCall(CeedOperatorGetNumElements(op, &num_elem));
    if (num_elem == 0) return CEED_ERROR_SUCCESS;
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
  if (is_composite) {
    CeedCall(CeedCompositeOperatorLinearAssembleAddDiagonal(op, request, false, assembled));
  } else {
    CeedCall(CeedSingleOperatorAssembleAddDiagonal_Core(op, request, false, assembled));
  }
  return CEED_ERROR_SUCCESS;
}

/**
   @brief Fully assemble the point-block diagonal pattern of a linear operator.

   Expected to be used in conjunction with CeedOperatorLinearAssemblePointBlockDiagonal().

   The assembly routines use coordinate format, with `num_entries` tuples of the form (i, j, value) which indicate that value should be added to the
matrix in entry (i, j).
  Note that the (i, j) pairs are unique.
  This function returns the number of entries and their (i, j) locations, while CeedOperatorLinearAssemblePointBlockDiagonal() provides the values in
the same ordering.

   This will generally be slow unless your operator is low-order.

   Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

   @param[in]  op          CeedOperator to assemble
   @param[out] num_entries Number of entries in coordinate nonzero pattern
   @param[out] rows        Row number for each entry
   @param[out] cols        Column number for each entry

   @ref User
**/
int CeedOperatorLinearAssemblePointBlockDiagonalSymbolic(CeedOperator op, CeedSize *num_entries, CeedInt **rows, CeedInt **cols) {
  Ceed          ceed;
  bool          is_composite;
  CeedInt       num_active_components, num_sub_operators;
  CeedOperator *sub_operators;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));

  CeedSize input_size = 0, output_size = 0;
  CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  CeedCheck(input_size == output_size, ceed, CEED_ERROR_DIMENSION, "Operator must be square");

  if (is_composite) {
    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_sub_operators));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
  } else {
    sub_operators     = &op;
    num_sub_operators = 1;
  }

  // Verify operator can be assembled correctly
  {
    CeedOperatorAssemblyData data;
    CeedInt                  num_active_elem_rstrs, comp_stride;
    CeedElemRestriction     *active_elem_rstrs;

    // Get initial values to check against
    CeedCall(CeedOperatorGetOperatorAssemblyData(sub_operators[0], &data));
    CeedCall(CeedOperatorAssemblyDataGetElemRestrictions(data, &num_active_elem_rstrs, &active_elem_rstrs, NULL, NULL));
    CeedCall(CeedElemRestrictionGetCompStride(active_elem_rstrs[0], &comp_stride));
    CeedCall(CeedElemRestrictionGetNumComponents(active_elem_rstrs[0], &num_active_components));

    // Verify that all active element restrictions have same component stride and number of components
    for (CeedInt k = 0; k < num_sub_operators; k++) {
      CeedCall(CeedOperatorGetOperatorAssemblyData(sub_operators[k], &data));
      CeedCall(CeedOperatorAssemblyDataGetElemRestrictions(data, &num_active_elem_rstrs, &active_elem_rstrs, NULL, NULL));
      for (CeedInt i = 0; i < num_active_elem_rstrs; i++) {
        CeedInt comp_stride_sub, num_active_components_sub;

        CeedCall(CeedElemRestrictionGetCompStride(active_elem_rstrs[i], &comp_stride_sub));
        CeedCheck(comp_stride == comp_stride_sub, ceed, CEED_ERROR_DIMENSION,
                  "Active element restrictions must have the same component stride: %d vs %d", comp_stride, comp_stride_sub);
        CeedCall(CeedElemRestrictionGetNumComponents(active_elem_rstrs[i], &num_active_components_sub));
        CeedCheck(num_active_components == num_active_components_sub, ceed, CEED_ERROR_INCOMPATIBLE,
                  "All suboperators must have the same number of output components");
      }
    }
  }
  *num_entries = input_size * num_active_components;
  CeedCall(CeedCalloc(*num_entries, rows));
  CeedCall(CeedCalloc(*num_entries, cols));

  for (CeedInt o = 0; o < num_sub_operators; o++) {
    CeedElemRestriction active_elem_rstr, point_block_active_elem_rstr;
    CeedInt             comp_stride, num_elem, elem_size;
    const CeedInt      *offsets, *point_block_offsets;

    CeedCall(CeedOperatorGetActiveElemRestriction(sub_operators[o], &active_elem_rstr));
    CeedCall(CeedElemRestrictionGetCompStride(active_elem_rstr, &comp_stride));
    CeedCall(CeedElemRestrictionGetNumElements(active_elem_rstr, &num_elem));
    CeedCall(CeedElemRestrictionGetElementSize(active_elem_rstr, &elem_size));
    CeedCall(CeedElemRestrictionGetOffsets(active_elem_rstr, CEED_MEM_HOST, &offsets));

    CeedCall(CeedOperatorCreateActivePointBlockRestriction(active_elem_rstr, &point_block_active_elem_rstr));
    CeedCall(CeedElemRestrictionGetOffsets(point_block_active_elem_rstr, CEED_MEM_HOST, &point_block_offsets));

    for (CeedSize i = 0; i < num_elem * elem_size; i++) {
      for (CeedInt c_out = 0; c_out < num_active_components; c_out++) {
        for (CeedInt c_in = 0; c_in < num_active_components; c_in++) {
          (*rows)[point_block_offsets[i] + c_out * num_active_components + c_in] = offsets[i] + c_out * comp_stride;
          (*cols)[point_block_offsets[i] + c_out * num_active_components + c_in] = offsets[i] + c_in * comp_stride;
        }
      }
    }

    CeedCall(CeedElemRestrictionRestoreOffsets(active_elem_rstr, &offsets));
    CeedCall(CeedElemRestrictionRestoreOffsets(point_block_active_elem_rstr, &point_block_offsets));
    CeedCall(CeedElemRestrictionDestroy(&point_block_active_elem_rstr));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Assemble the point block diagonal of a square linear CeedOperator

  This overwrites a CeedVector with the point block diagonal of a linear CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and composite CeedOperators with single field sub-operators are supported.

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op        CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedOperator point block diagonal, provided in row-major form with an @a num_comp * @a num_comp
block at each node. The dimensions of this vector are derived from the active vector for the CeedOperator. The array has shape [nodes, component out,
component in].
  @param[in]  request   Address of CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssemblePointBlockDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  bool     is_composite;
  CeedSize input_size = 0, output_size = 0;

  CeedCall(CeedOperatorCheckReady(op));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));

  CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  CeedCheck(input_size == output_size, op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");

  // Early exit for empty operator
  if (!is_composite) {
    CeedInt num_elem = 0;

    CeedCall(CeedOperatorGetNumElements(op, &num_elem));
    if (num_elem == 0) return CEED_ERROR_SUCCESS;
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

  This sums into a CeedVector with the point block diagonal of a linear CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and composite CeedOperators with single field sub-operators are supported.

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op        CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedOperator point block diagonal, provided in row-major form with an @a num_comp * @a num_comp
block at each node. The dimensions of this vector are derived from the active vector for the CeedOperator. The array has shape [nodes, component out,
component in].
  @param[in]  request Address of CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorLinearAssembleAddPointBlockDiagonal(CeedOperator op, CeedVector assembled, CeedRequest *request) {
  bool     is_composite;
  CeedSize input_size = 0, output_size = 0;

  CeedCall(CeedOperatorCheckReady(op));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));

  CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
  CeedCheck(input_size == output_size, op->ceed, CEED_ERROR_DIMENSION, "Operator must be square");

  // Early exit for empty operator
  if (!is_composite) {
    CeedInt num_elem = 0;

    CeedCall(CeedOperatorGetNumElements(op, &num_elem));
    if (num_elem == 0) return CEED_ERROR_SUCCESS;
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
  // Default interface implementation
  if (is_composite) {
    CeedCall(CeedCompositeOperatorLinearAssembleAddDiagonal(op, request, true, assembled));
  } else {
    CeedCall(CeedSingleOperatorAssembleAddDiagonal_Core(op, request, true, assembled));
  }
  return CEED_ERROR_SUCCESS;
}

/**
   @brief Fully assemble the nonzero pattern of a linear operator.

   Expected to be used in conjunction with CeedOperatorLinearAssemble().

   The assembly routines use coordinate format, with num_entries tuples of the form (i, j, value) which indicate that value should be added to the
matrix in entry (i, j).
  Note that the (i, j) pairs are not unique and may repeat.
  This function returns the number of entries and their (i, j) locations, while CeedOperatorLinearAssemble() provides the values in the same ordering.

   This will generally be slow unless your operator is low-order.

   Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

   @param[in]  op          CeedOperator to assemble
   @param[out] num_entries Number of entries in coordinate nonzero pattern
   @param[out] rows        Row number for each entry
   @param[out] cols        Column number for each entry

   @ref User
**/
int CeedOperatorLinearAssembleSymbolic(CeedOperator op, CeedSize *num_entries, CeedInt **rows, CeedInt **cols) {
  bool          is_composite;
  CeedInt       num_suboperators, offset = 0;
  CeedSize      single_entries;
  CeedOperator *sub_operators;

  CeedCall(CeedOperatorCheckReady(op));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));

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

  // Count entries and allocate rows, cols arrays
  *num_entries = 0;
  if (is_composite) {
    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
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

  // Assemble nonzero locations
  if (is_composite) {
    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
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

   Expected to be used in conjunction with CeedOperatorLinearAssembleSymbolic().

   The assembly routines use coordinate format, with num_entries tuples of the form (i, j, value) which indicate that value should be added to the
matrix in entry (i, j).
  Note that the (i, j) pairs are not unique and may repeat.
  This function returns the values of the nonzero entries to be added, their (i, j) locations are provided by CeedOperatorLinearAssembleSymbolic()

   This will generally be slow unless your operator is low-order.

   Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

   @param[in]  op     CeedOperator to assemble
   @param[out] values Values to assemble into matrix

   @ref User
**/
int CeedOperatorLinearAssemble(CeedOperator op, CeedVector values) {
  bool          is_composite;
  CeedInt       num_suboperators, offset = 0;
  CeedSize      single_entries = 0;
  CeedOperator *sub_operators;

  CeedCall(CeedOperatorCheckReady(op));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));

  // Early exit for empty operator
  if (!is_composite) {
    CeedInt num_elem = 0;

    CeedCall(CeedOperatorGetNumElements(op, &num_elem));
    if (num_elem == 0) return CEED_ERROR_SUCCESS;
  }

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
  CeedCall(CeedVectorSetValue(values, 0.0));
  if (is_composite) {
    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
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
  @brief Get the multiplicity of nodes across suboperators in a composite CeedOperator

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op               Composite CeedOperator
  @param[in]  num_skip_indices Number of suboperators to skip
  @param[in]  skip_indices     Array of indices of suboperators to skip
  @param[out] mult             Vector to store multiplicity (of size l_size)

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedCompositeOperatorGetMultiplicity(CeedOperator op, CeedInt num_skip_indices, CeedInt *skip_indices, CeedVector mult) {
  Ceed                ceed;
  CeedInt             num_suboperators;
  CeedSize            l_vec_len;
  CeedScalar         *mult_array;
  CeedVector          ones_l_vec;
  CeedElemRestriction elem_rstr, mult_elem_rstr;
  CeedOperator       *sub_operators;

  CeedCall(CeedOperatorCheckReady(op));

  CeedCall(CeedOperatorGetCeed(op, &ceed));

  // Zero mult vector
  CeedCall(CeedVectorSetValue(mult, 0.0));

  // Get suboperators
  CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
  CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
  if (num_suboperators == 0) return CEED_ERROR_SUCCESS;

  // Work vector
  CeedCall(CeedVectorGetLength(mult, &l_vec_len));
  CeedCall(CeedVectorCreate(ceed, l_vec_len, &ones_l_vec));
  CeedCall(CeedVectorSetValue(ones_l_vec, 1.0));
  CeedCall(CeedVectorGetArray(mult, CEED_MEM_HOST, &mult_array));

  // Compute multiplicity across suboperators
  for (CeedInt i = 0; i < num_suboperators; i++) {
    const CeedScalar *sub_mult_array;
    CeedVector        sub_mult_l_vec, ones_e_vec;

    // -- Check for suboperator to skip
    for (CeedInt j = 0; j < num_skip_indices; j++) {
      if (skip_indices[j] == i) continue;
    }

    // -- Sub operator multiplicity
    CeedCall(CeedOperatorGetActiveElemRestriction(sub_operators[i], &elem_rstr));
    CeedCall(CeedElemRestrictionCreateUnorientedCopy(elem_rstr, &mult_elem_rstr));
    CeedCall(CeedElemRestrictionCreateVector(mult_elem_rstr, &sub_mult_l_vec, &ones_e_vec));
    CeedCall(CeedVectorSetValue(sub_mult_l_vec, 0.0));
    CeedCall(CeedElemRestrictionApply(mult_elem_rstr, CEED_NOTRANSPOSE, ones_l_vec, ones_e_vec, CEED_REQUEST_IMMEDIATE));
    CeedCall(CeedElemRestrictionApply(mult_elem_rstr, CEED_TRANSPOSE, ones_e_vec, sub_mult_l_vec, CEED_REQUEST_IMMEDIATE));
    CeedCall(CeedVectorGetArrayRead(sub_mult_l_vec, CEED_MEM_HOST, &sub_mult_array));
    // ---- Flag every node present in the current suboperator
    for (CeedInt j = 0; j < l_vec_len; j++) {
      if (sub_mult_array[j] > 0.0) mult_array[j] += 1.0;
    }
    CeedCall(CeedVectorRestoreArrayRead(sub_mult_l_vec, &sub_mult_array));
    CeedCall(CeedVectorDestroy(&sub_mult_l_vec));
    CeedCall(CeedVectorDestroy(&ones_e_vec));
    CeedCall(CeedElemRestrictionDestroy(&mult_elem_rstr));
  }
  CeedCall(CeedVectorRestoreArray(mult, &mult_array));
  CeedCall(CeedVectorDestroy(&ones_l_vec));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators for a CeedOperator, creating the prolongation basis from the fine and coarse
grid interpolation

  Note: Calling this function asserts that setup is complete and sets all four CeedOperators as immutable.

  @param[in]  op_fine      Fine grid operator
  @param[in]  p_mult_fine  L-vector multiplicity in parallel gather/scatter, or NULL if not creating prolongation/restriction operators
  @param[in]  rstr_coarse  Coarse grid restriction
  @param[in]  basis_coarse Coarse grid active vector basis
  @param[out] op_coarse    Coarse grid operator
  @param[out] op_prolong   Coarse to fine operator, or NULL
  @param[out] op_restrict  Fine to coarse operator, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreate(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
                                     CeedOperator *op_coarse, CeedOperator *op_prolong, CeedOperator *op_restrict) {
  CeedBasis basis_c_to_f = NULL;

  CeedCall(CeedOperatorCheckReady(op_fine));

  // Build prolongation matrix, if required
  if (op_prolong || op_restrict) {
    CeedBasis basis_fine;

    CeedCall(CeedOperatorGetActiveBasis(op_fine, &basis_fine));
    CeedCall(CeedBasisCreateProjection(basis_coarse, basis_fine, &basis_c_to_f));
  }

  // Core code
  CeedCall(CeedSingleOperatorMultigridLevel(op_fine, p_mult_fine, rstr_coarse, basis_coarse, basis_c_to_f, op_coarse, op_prolong, op_restrict));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators for a CeedOperator with a tensor basis for the active basis

  Note: Calling this function asserts that setup is complete and sets all four CeedOperators as immutable.

  @param[in]  op_fine       Fine grid operator
  @param[in]  p_mult_fine   L-vector multiplicity in parallel gather/scatter, or NULL if not creating prolongation/restriction operators
  @param[in]  rstr_coarse   Coarse grid restriction
  @param[in]  basis_coarse  Coarse grid active vector basis
  @param[in]  interp_c_to_f Matrix for coarse to fine interpolation, or NULL if not creating prolongation/restriction operators
  @param[out] op_coarse     Coarse grid operator
  @param[out] op_prolong    Coarse to fine operator, or NULL
  @param[out] op_restrict   Fine to coarse operator, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreateTensorH1(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
                                             const CeedScalar *interp_c_to_f, CeedOperator *op_coarse, CeedOperator *op_prolong,
                                             CeedOperator *op_restrict) {
  Ceed      ceed;
  CeedInt   Q_f, Q_c;
  CeedBasis basis_fine, basis_c_to_f = NULL;

  CeedCall(CeedOperatorCheckReady(op_fine));
  CeedCall(CeedOperatorGetCeed(op_fine, &ceed));

  // Check for compatible quadrature spaces
  CeedCall(CeedOperatorGetActiveBasis(op_fine, &basis_fine));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_fine, &Q_f));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_coarse, &Q_c));
  CeedCheck(Q_f == Q_c, ceed, CEED_ERROR_DIMENSION, "Bases must have compatible quadrature spaces");

  // Create coarse to fine basis, if required
  if (op_prolong || op_restrict) {
    CeedInt     dim, num_comp, num_nodes_c, P_1d_f, P_1d_c;
    CeedScalar *q_ref, *q_weight, *grad;

    // Check if interpolation matrix is provided
    CeedCheck(interp_c_to_f, ceed, CEED_ERROR_INCOMPATIBLE,
              "Prolongation or restriction operator creation requires coarse-to-fine interpolation matrix");
    CeedCall(CeedBasisGetDimension(basis_fine, &dim));
    CeedCall(CeedBasisGetNumComponents(basis_fine, &num_comp));
    CeedCall(CeedBasisGetNumNodes1D(basis_fine, &P_1d_f));
    CeedCall(CeedElemRestrictionGetElementSize(rstr_coarse, &num_nodes_c));
    P_1d_c = dim == 1 ? num_nodes_c : dim == 2 ? sqrt(num_nodes_c) : cbrt(num_nodes_c);
    CeedCall(CeedCalloc(P_1d_f, &q_ref));
    CeedCall(CeedCalloc(P_1d_f, &q_weight));
    CeedCall(CeedCalloc(P_1d_f * P_1d_c * dim, &grad));
    CeedCall(CeedBasisCreateTensorH1(ceed, dim, num_comp, P_1d_c, P_1d_f, interp_c_to_f, grad, q_ref, q_weight, &basis_c_to_f));
    CeedCall(CeedFree(&q_ref));
    CeedCall(CeedFree(&q_weight));
    CeedCall(CeedFree(&grad));
  }

  // Core code
  CeedCall(CeedSingleOperatorMultigridLevel(op_fine, p_mult_fine, rstr_coarse, basis_coarse, basis_c_to_f, op_coarse, op_prolong, op_restrict));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators for a CeedOperator with a non-tensor basis for the active vector

  Note: Calling this function asserts that setup is complete and sets all four CeedOperators as immutable.

  @param[in]  op_fine       Fine grid operator
  @param[in]  p_mult_fine   L-vector multiplicity in parallel gather/scatter, or NULL if not creating prolongation/restriction operators
  @param[in]  rstr_coarse   Coarse grid restriction
  @param[in]  basis_coarse  Coarse grid active vector basis
  @param[in]  interp_c_to_f Matrix for coarse to fine interpolation, or NULL if not creating prolongation/restriction operators
  @param[out] op_coarse     Coarse grid operator
  @param[out] op_prolong    Coarse to fine operator, or NULL
  @param[out] op_restrict   Fine to coarse operator, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreateH1(CeedOperator op_fine, CeedVector p_mult_fine, CeedElemRestriction rstr_coarse, CeedBasis basis_coarse,
                                       const CeedScalar *interp_c_to_f, CeedOperator *op_coarse, CeedOperator *op_prolong,
                                       CeedOperator *op_restrict) {
  Ceed      ceed;
  CeedInt   Q_f, Q_c;
  CeedBasis basis_fine, basis_c_to_f = NULL;

  CeedCall(CeedOperatorCheckReady(op_fine));
  CeedCall(CeedOperatorGetCeed(op_fine, &ceed));

  // Check for compatible quadrature spaces
  CeedCall(CeedOperatorGetActiveBasis(op_fine, &basis_fine));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_fine, &Q_f));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_coarse, &Q_c));
  CeedCheck(Q_f == Q_c, ceed, CEED_ERROR_DIMENSION, "Bases must have compatible quadrature spaces");

  // Coarse to fine basis
  if (op_prolong || op_restrict) {
    CeedInt          dim, num_comp, num_nodes_c, num_nodes_f;
    CeedScalar      *q_ref, *q_weight, *grad;
    CeedElemTopology topo;

    // Check if interpolation matrix is provided
    CeedCheck(interp_c_to_f, ceed, CEED_ERROR_INCOMPATIBLE,
              "Prolongation or restriction operator creation requires coarse-to-fine interpolation matrix");
    CeedCall(CeedBasisGetTopology(basis_fine, &topo));
    CeedCall(CeedBasisGetDimension(basis_fine, &dim));
    CeedCall(CeedBasisGetNumComponents(basis_fine, &num_comp));
    CeedCall(CeedBasisGetNumNodes(basis_fine, &num_nodes_f));
    CeedCall(CeedElemRestrictionGetElementSize(rstr_coarse, &num_nodes_c));
    CeedCall(CeedCalloc(num_nodes_f * dim, &q_ref));
    CeedCall(CeedCalloc(num_nodes_f, &q_weight));
    CeedCall(CeedCalloc(num_nodes_f * num_nodes_c * dim, &grad));
    CeedCall(CeedBasisCreateH1(ceed, topo, num_comp, num_nodes_c, num_nodes_f, interp_c_to_f, grad, q_ref, q_weight, &basis_c_to_f));
    CeedCall(CeedFree(&q_ref));
    CeedCall(CeedFree(&q_weight));
    CeedCall(CeedFree(&grad));
  }

  // Core code
  CeedCall(CeedSingleOperatorMultigridLevel(op_fine, p_mult_fine, rstr_coarse, basis_coarse, basis_c_to_f, op_coarse, op_prolong, op_restrict));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Build a FDM based approximate inverse for each element for a CeedOperator

  This returns a CeedOperator and CeedVector to apply a Fast Diagonalization Method based approximate inverse.
  This function obtains the simultaneous diagonalization for the 1D mass and Laplacian operators, \f$M = V^T V, K = V^T S V\f$.
  The assembled QFunction is used to modify the eigenvalues from simultaneous diagonalization and obtain an approximate inverse of the form \f$V^T
\hat S V\f$.
  The CeedOperator must be linear and non-composite.
  The associated CeedQFunction must therefore also be linear.

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op      CeedOperator to create element inverses
  @param[out] fdm_inv CeedOperator to apply the action of a FDM based inverse for each element
  @param[in]  request Address of CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorCreateFDMElementInverse(CeedOperator op, CeedOperator *fdm_inv, CeedRequest *request) {
  Ceed                 ceed, ceed_parent;
  bool                 interp = false, grad = false, is_tensor_basis = true;
  CeedInt              num_input_fields, P_1d, Q_1d, num_nodes, num_qpts, dim, num_comp = 1, num_elem = 1;
  CeedSize             l_size = 1;
  CeedScalar          *mass, *laplace, *x, *fdm_interp, *lambda, *elem_avg;
  const CeedScalar    *interp_1d, *grad_1d, *q_weight_1d;
  CeedVector           q_data;
  CeedElemRestriction  rstr  = NULL, rstr_qd_i;
  CeedBasis            basis = NULL, fdm_basis;
  CeedQFunctionContext ctx_fdm;
  CeedQFunctionField  *qf_fields;
  CeedQFunction        qf, qf_fdm;
  CeedOperatorField   *op_fields;

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
  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorGetFallbackParentCeed(op, &ceed_parent));
  CeedCall(CeedOperatorGetQFunction(op, &qf));

  // Determine active input basis
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
  CeedCheck(basis, ceed, CEED_ERROR_BACKEND, "No active field set");
  CeedCall(CeedBasisGetNumNodes1D(basis, &P_1d));
  CeedCall(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCall(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCall(CeedBasisGetDimension(basis, &dim));
  CeedCall(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &l_size));

  // Build and diagonalize 1D Mass and Laplacian
  CeedCall(CeedBasisIsTensor(basis, &is_tensor_basis));
  CeedCheck(is_tensor_basis, ceed, CEED_ERROR_BACKEND, "FDMElementInverse only supported for tensor bases");
  CeedCall(CeedCalloc(P_1d * P_1d, &mass));
  CeedCall(CeedCalloc(P_1d * P_1d, &laplace));
  CeedCall(CeedCalloc(P_1d * P_1d, &x));
  CeedCall(CeedCalloc(P_1d * P_1d, &fdm_interp));
  CeedCall(CeedCalloc(P_1d, &lambda));
  // -- Build matrices
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

  {
    CeedInt             layout[3], num_modes = (interp ? 1 : 0) + (grad ? dim : 0);
    CeedScalar          max_norm = 0;
    const CeedScalar   *assembled_array, *q_weight_array;
    CeedVector          assembled = NULL, q_weight;
    CeedElemRestriction rstr_qf   = NULL;

    // Assemble QFunction
    CeedCall(CeedOperatorLinearAssembleQFunctionBuildOrUpdate(op, &assembled, &rstr_qf, request));
    CeedCall(CeedElemRestrictionGetELayout(rstr_qf, &layout));
    CeedCall(CeedElemRestrictionDestroy(&rstr_qf));
    CeedCall(CeedVectorNorm(assembled, CEED_NORM_MAX, &max_norm));

    // Calculate element averages
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
  }

  // Build FDM diagonal
  {
    CeedScalar *q_data_array, *fdm_diagonal;

    CeedCall(CeedCalloc(num_comp * num_nodes, &fdm_diagonal));
    const CeedScalar fdm_diagonal_bound = num_nodes * CEED_EPSILON;
    for (CeedInt c = 0; c < num_comp; c++) {
      for (CeedInt n = 0; n < num_nodes; n++) {
        if (interp) fdm_diagonal[c * num_nodes + n] = 1.0;
        if (grad) {
          for (CeedInt d = 0; d < dim; d++) {
            CeedInt i = (n / CeedIntPow(P_1d, d)) % P_1d;
            fdm_diagonal[c * num_nodes + n] += lambda[i];
          }
        }
        if (fabs(fdm_diagonal[c * num_nodes + n]) < fdm_diagonal_bound) fdm_diagonal[c * num_nodes + n] = fdm_diagonal_bound;
      }
    }
    CeedCall(CeedVectorCreate(ceed_parent, num_elem * num_comp * num_nodes, &q_data));
    CeedCall(CeedVectorSetValue(q_data, 0.0));
    CeedCall(CeedVectorGetArrayWrite(q_data, CEED_MEM_HOST, &q_data_array));
    for (CeedInt e = 0; e < num_elem; e++) {
      for (CeedInt c = 0; c < num_comp; c++) {
        for (CeedInt n = 0; n < num_nodes; n++)
          q_data_array[(e * num_comp + c) * num_nodes + n] = 1. / (elem_avg[e] * fdm_diagonal[c * num_nodes + n]);
      }
    }
    CeedCall(CeedFree(&elem_avg));
    CeedCall(CeedFree(&fdm_diagonal));
    CeedCall(CeedVectorRestoreArray(q_data, &q_data_array));
  }

  // Setup FDM operator
  // -- Basis
  {
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
  }

  // -- Restriction
  {
    CeedInt strides[3] = {1, num_nodes, num_nodes * num_comp};
    CeedCall(CeedElemRestrictionCreateStrided(ceed_parent, num_elem, num_nodes, num_comp, num_elem * num_comp * num_nodes, strides, &rstr_qd_i));
  }

  // -- QFunction
  CeedCall(CeedQFunctionCreateInteriorByName(ceed_parent, "Scale", &qf_fdm));
  CeedCall(CeedQFunctionAddInput(qf_fdm, "input", num_comp, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionAddInput(qf_fdm, "scale", num_comp, CEED_EVAL_NONE));
  CeedCall(CeedQFunctionAddOutput(qf_fdm, "output", num_comp, CEED_EVAL_INTERP));
  CeedCall(CeedQFunctionSetUserFlopsEstimate(qf_fdm, num_comp));

  // -- QFunction context
  {
    CeedInt *num_comp_data;

    CeedCall(CeedCalloc(1, &num_comp_data));
    num_comp_data[0] = num_comp;
    CeedCall(CeedQFunctionContextCreate(ceed, &ctx_fdm));
    CeedCall(CeedQFunctionContextSetData(ctx_fdm, CEED_MEM_HOST, CEED_OWN_POINTER, sizeof(*num_comp_data), num_comp_data));
  }
  CeedCall(CeedQFunctionSetContext(qf_fdm, ctx_fdm));
  CeedCall(CeedQFunctionContextDestroy(&ctx_fdm));

  // -- Operator
  CeedCall(CeedOperatorCreate(ceed_parent, qf_fdm, NULL, NULL, fdm_inv));
  CeedCall(CeedOperatorSetField(*fdm_inv, "input", rstr, fdm_basis, CEED_VECTOR_ACTIVE));
  CeedCall(CeedOperatorSetField(*fdm_inv, "scale", rstr_qd_i, CEED_BASIS_NONE, q_data));
  CeedCall(CeedOperatorSetField(*fdm_inv, "output", rstr, fdm_basis, CEED_VECTOR_ACTIVE));

  // Cleanup
  CeedCall(CeedVectorDestroy(&q_data));
  CeedCall(CeedBasisDestroy(&fdm_basis));
  CeedCall(CeedElemRestrictionDestroy(&rstr_qd_i));
  CeedCall(CeedQFunctionDestroy(&qf_fdm));
  return CEED_ERROR_SUCCESS;
}

/// @}
