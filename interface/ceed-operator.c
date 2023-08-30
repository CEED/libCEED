// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/// @file
/// Implementation of CeedOperator interfaces

/// ----------------------------------------------------------------------------
/// CeedOperator Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedOperatorDeveloper
/// @{

/**
  @brief Check if a CeedOperator Field matches the QFunction Field

  @param[in] ceed     Ceed object for error handling
  @param[in] qf_field QFunction Field matching Operator Field
  @param[in] r        Operator Field ElemRestriction
  @param[in] b        Operator Field Basis

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCheckField(Ceed ceed, CeedQFunctionField qf_field, CeedElemRestriction r, CeedBasis b) {
  CeedInt      dim = 1, num_comp = 1, q_comp = 1, restr_num_comp = 1, size = qf_field->size;
  CeedEvalMode eval_mode = qf_field->eval_mode;

  // Restriction
  CeedCheck((r == CEED_ELEMRESTRICTION_NONE) == (eval_mode == CEED_EVAL_WEIGHT), ceed, CEED_ERROR_INCOMPATIBLE,
            "CEED_ELEMRESTRICTION_NONE and CEED_EVAL_WEIGHT must be used together.");
  if (r != CEED_ELEMRESTRICTION_NONE) {
    CeedCall(CeedElemRestrictionGetNumComponents(r, &restr_num_comp));
  }
  // Basis
  CeedCheck((b == CEED_BASIS_COLLOCATED) == (eval_mode == CEED_EVAL_NONE), ceed, CEED_ERROR_INCOMPATIBLE,
            "CEED_BASIS_COLLOCATED and CEED_EVAL_NONE must be used together.");
  if (b != CEED_BASIS_COLLOCATED) {
    CeedCall(CeedBasisGetDimension(b, &dim));
    CeedCall(CeedBasisGetNumComponents(b, &num_comp));
    CeedCall(CeedBasisGetNumQuadratureComponents(b, eval_mode, &q_comp));
    CeedCheck(r == CEED_ELEMRESTRICTION_NONE || restr_num_comp == num_comp, ceed, CEED_ERROR_DIMENSION,
              "Field '%s' of size %" CeedInt_FMT " and EvalMode %s: ElemRestriction has %" CeedInt_FMT " components, but Basis has %" CeedInt_FMT
              " components",
              qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode], restr_num_comp, num_comp);
  }
  // Field size
  switch (eval_mode) {
    case CEED_EVAL_NONE:
      CeedCheck(size == restr_num_comp, ceed, CEED_ERROR_DIMENSION,
                "Field '%s' of size %" CeedInt_FMT " and EvalMode %s: ElemRestriction has %" CeedInt_FMT " components", qf_field->field_name,
                qf_field->size, CeedEvalModes[qf_field->eval_mode], restr_num_comp);
      break;
    case CEED_EVAL_INTERP:
    case CEED_EVAL_GRAD:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      CeedCheck(size == num_comp * q_comp, ceed, CEED_ERROR_DIMENSION,
                "Field '%s' of size %" CeedInt_FMT " and EvalMode %s: ElemRestriction/Basis has %" CeedInt_FMT " components", qf_field->field_name,
                qf_field->size, CeedEvalModes[qf_field->eval_mode], num_comp * q_comp);
      break;
    case CEED_EVAL_WEIGHT:
      // No additional checks required
      break;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a field of a CeedOperator

  @param[in] field        Operator field to view
  @param[in] qf_field     QFunction field (carries field name)
  @param[in] field_number Number of field being viewed
  @param[in] sub          true indicates sub-operator, which increases indentation; false for top-level operator
  @param[in] input        true for an input field; false for output field
  @param[in] stream       Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedOperatorFieldView(CeedOperatorField field, CeedQFunctionField qf_field, CeedInt field_number, bool sub, bool input, FILE *stream) {
  const char *pre    = sub ? "  " : "";
  const char *in_out = input ? "Input" : "Output";

  fprintf(stream,
          "%s    %s field %" CeedInt_FMT
          ":\n"
          "%s      Name: \"%s\"\n",
          pre, in_out, field_number, pre, qf_field->field_name);
  fprintf(stream, "%s      Size: %" CeedInt_FMT "\n", pre, qf_field->size);
  fprintf(stream, "%s      EvalMode: %s\n", pre, CeedEvalModes[qf_field->eval_mode]);
  if (field->basis == CEED_BASIS_COLLOCATED) fprintf(stream, "%s      Collocated basis\n", pre);
  if (field->vec == CEED_VECTOR_ACTIVE) fprintf(stream, "%s      Active vector\n", pre);
  else if (field->vec == CEED_VECTOR_NONE) fprintf(stream, "%s      No vector\n", pre);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a single CeedOperator

  @param[in] op     CeedOperator to view
  @param[in] sub    Boolean flag for sub-operator
  @param[in] stream Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedOperatorSingleView(CeedOperator op, bool sub, FILE *stream) {
  const char *pre = sub ? "  " : "";
  CeedInt     num_elem, num_qpts, total_fields = 0;

  CeedCall(CeedOperatorGetNumElements(op, &num_elem));
  CeedCall(CeedOperatorGetNumQuadraturePoints(op, &num_qpts));
  CeedCall(CeedOperatorGetNumArgs(op, &total_fields));

  fprintf(stream, "%s  %" CeedInt_FMT " elements with %" CeedInt_FMT " quadrature points each\n", pre, num_elem, num_qpts);
  fprintf(stream, "%s  %" CeedInt_FMT " field%s\n", pre, total_fields, total_fields > 1 ? "s" : "");
  fprintf(stream, "%s  %" CeedInt_FMT " input field%s:\n", pre, op->qf->num_input_fields, op->qf->num_input_fields > 1 ? "s" : "");
  for (CeedInt i = 0; i < op->qf->num_input_fields; i++) {
    CeedCall(CeedOperatorFieldView(op->input_fields[i], op->qf->input_fields[i], i, sub, 1, stream));
  }
  fprintf(stream, "%s  %" CeedInt_FMT " output field%s:\n", pre, op->qf->num_output_fields, op->qf->num_output_fields > 1 ? "s" : "");
  for (CeedInt i = 0; i < op->qf->num_output_fields; i++) {
    CeedCall(CeedOperatorFieldView(op->output_fields[i], op->qf->output_fields[i], i, sub, 0, stream));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active vector basis for a non-composite CeedOperator

  @param[in] op            CeedOperator to find active basis for
  @param[out] active_basis Basis for active input vector or NULL for composite operator

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedOperatorGetActiveBasis(CeedOperator op, CeedBasis *active_basis) {
  Ceed ceed;

  CeedCall(CeedOperatorGetCeed(op, &ceed));

  *active_basis = NULL;
  if (op->is_composite) return CEED_ERROR_SUCCESS;
  for (CeedInt i = 0; i < op->qf->num_input_fields; i++) {
    if (op->input_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      CeedCheck(!*active_basis || *active_basis == op->input_fields[i]->basis, ceed, CEED_ERROR_MINOR, "Multiple active CeedBases found");
      *active_basis = op->input_fields[i]->basis;
    }
  }

  CeedCheck(*active_basis, ceed, CEED_ERROR_INCOMPLETE, "No active CeedBasis found");
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active vector ElemRestriction for a non-composite CeedOperator

  @param[in] op           CeedOperator to find active basis for
  @param[out] active_rstr ElemRestriction for active input vector or NULL for composite operator

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedOperatorGetActiveElemRestriction(CeedOperator op, CeedElemRestriction *active_rstr) {
  Ceed ceed;

  CeedCall(CeedOperatorGetCeed(op, &ceed));

  *active_rstr = NULL;
  if (op->is_composite) return CEED_ERROR_SUCCESS;
  for (CeedInt i = 0; i < op->qf->num_input_fields; i++) {
    if (op->input_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      CeedCheck(!*active_rstr || *active_rstr == op->input_fields[i]->elem_rstr, ceed, CEED_ERROR_MINOR,
                "Multiple active CeedElemRestrictions found");
      *active_rstr = op->input_fields[i]->elem_rstr;
    }
  }

  CeedCheck(*active_rstr, ceed, CEED_ERROR_INCOMPLETE, "No active CeedElemRestriction found");
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set QFunctionContext field values of the specified type.

  For composite operators, the value is set in all sub-operator QFunctionContexts that have a matching `field_name`.
  A non-zero error code is returned for single operators that do not have a matching field of the same type or composite operators that do not have
any field of a matching type.

  @param[in,out] op          CeedOperator
  @param[in]     field_label Label of field to set
  @param[in]     field_type  Type of field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
static int CeedOperatorContextSetGeneric(CeedOperator op, CeedContextFieldLabel field_label, CeedContextFieldType field_type, void *values) {
  bool is_composite = false;

  CeedCheck(field_label, op->ceed, CEED_ERROR_UNSUPPORTED, "Invalid field label");

  // Check if field_label and op correspond
  if (field_label->from_op) {
    CeedInt index = -1;

    for (CeedInt i = 0; i < op->num_context_labels; i++) {
      if (op->context_labels[i] == field_label) index = i;
    }
    CeedCheck(index != -1, op->ceed, CEED_ERROR_UNSUPPORTED, "ContextFieldLabel does not correspond to the operator");
  }

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt       num_sub;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_sub));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    CeedCheck(num_sub == field_label->num_sub_labels, op->ceed, CEED_ERROR_UNSUPPORTED,
              "Composite operator modified after ContextFieldLabel created");

    for (CeedInt i = 0; i < num_sub; i++) {
      // Try every sub-operator, ok if some sub-operators do not have field
      if (field_label->sub_labels[i] && sub_operators[i]->qf->ctx) {
        CeedCall(CeedQFunctionContextSetGeneric(sub_operators[i]->qf->ctx, field_label->sub_labels[i], field_type, values));
      }
    }
  } else {
    CeedCheck(op->qf->ctx, op->ceed, CEED_ERROR_UNSUPPORTED, "QFunction does not have context data");
    CeedCall(CeedQFunctionContextSetGeneric(op->qf->ctx, field_label, field_type, values));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get QFunctionContext field values of the specified type, read-only.

  For composite operators, the values retrieved are for the first sub-operator QFunctionContext that have a matching `field_name`.
  A non-zero error code is returned for single operators that do not have a matching field of the same type or composite operators that do not have
any field of a matching type.

  @param[in,out] op          CeedOperator
  @param[in]     field_label Label of field to set
  @param[in]     field_type  Type of field to set
  @param[out]    num_values  Number of values of type `field_type` in array `values`
  @param[out]    values      Values in the label

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
static int CeedOperatorContextGetGenericRead(CeedOperator op, CeedContextFieldLabel field_label, CeedContextFieldType field_type, size_t *num_values,
                                             void *values) {
  bool is_composite = false;

  CeedCheck(field_label, op->ceed, CEED_ERROR_UNSUPPORTED, "Invalid field label");

  *(void **)values = NULL;
  *num_values      = 0;

  // Check if field_label and op correspond
  if (field_label->from_op) {
    CeedInt index = -1;

    for (CeedInt i = 0; i < op->num_context_labels; i++) {
      if (op->context_labels[i] == field_label) index = i;
    }
    CeedCheck(index != -1, op->ceed, CEED_ERROR_UNSUPPORTED, "ContextFieldLabel does not correspond to the operator");
  }

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt       num_sub;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_sub));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    CeedCheck(num_sub == field_label->num_sub_labels, op->ceed, CEED_ERROR_UNSUPPORTED,
              "Composite operator modified after ContextFieldLabel created");

    for (CeedInt i = 0; i < num_sub; i++) {
      // Try every sub-operator, ok if some sub-operators do not have field
      if (field_label->sub_labels[i] && sub_operators[i]->qf->ctx) {
        CeedCall(CeedQFunctionContextGetGenericRead(sub_operators[i]->qf->ctx, field_label->sub_labels[i], field_type, num_values, values));
        return CEED_ERROR_SUCCESS;
      }
    }
  } else {
    CeedCheck(op->qf->ctx, op->ceed, CEED_ERROR_UNSUPPORTED, "QFunction does not have context data");
    CeedCall(CeedQFunctionContextGetGenericRead(op->qf->ctx, field_label, field_type, num_values, values));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore QFunctionContext field values of the specified type, read-only.

  For composite operators, the values restored are for the first sub-operator QFunctionContext that have a matching `field_name`.
  A non-zero error code is returned for single operators that do not have a matching field of the same type or composite operators that do not have
any field of a matching type.

  @param[in,out] op          CeedOperator
  @param[in]     field_label Label of field to set
  @param[in]     field_type  Type of field to set
  @param[in]     values      Values array to restore

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
static int CeedOperatorContextRestoreGenericRead(CeedOperator op, CeedContextFieldLabel field_label, CeedContextFieldType field_type, void *values) {
  bool is_composite = false;

  CeedCheck(field_label, op->ceed, CEED_ERROR_UNSUPPORTED, "Invalid field label");

  // Check if field_label and op correspond
  if (field_label->from_op) {
    CeedInt index = -1;

    for (CeedInt i = 0; i < op->num_context_labels; i++) {
      if (op->context_labels[i] == field_label) index = i;
    }
    CeedCheck(index != -1, op->ceed, CEED_ERROR_UNSUPPORTED, "ContextFieldLabel does not correspond to the operator");
  }

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt       num_sub;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_sub));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    CeedCheck(num_sub == field_label->num_sub_labels, op->ceed, CEED_ERROR_UNSUPPORTED,
              "Composite operator modified after ContextFieldLabel created");

    for (CeedInt i = 0; i < num_sub; i++) {
      // Try every sub-operator, ok if some sub-operators do not have field
      if (field_label->sub_labels[i] && sub_operators[i]->qf->ctx) {
        CeedCall(CeedQFunctionContextRestoreGenericRead(sub_operators[i]->qf->ctx, field_label->sub_labels[i], field_type, values));
        return CEED_ERROR_SUCCESS;
      }
    }
  } else {
    CeedCheck(op->qf->ctx, op->ceed, CEED_ERROR_UNSUPPORTED, "QFunction does not have context data");
    CeedCall(CeedQFunctionContextRestoreGenericRead(op->qf->ctx, field_label, field_type, values));
  }
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedOperator Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedOperatorBackend
/// @{

/**
  @brief Get the number of arguments associated with a CeedOperator

  @param[in]  op        CeedOperator
  @param[out] num_args  Variable to store vector number of arguments

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorGetNumArgs(CeedOperator op, CeedInt *num_args) {
  CeedCheck(!op->is_composite, op->ceed, CEED_ERROR_MINOR, "Not defined for composite operators");
  *num_args = op->num_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the setup status of a CeedOperator

  @param[in]  op            CeedOperator
  @param[out] is_setup_done Variable to store setup status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorIsSetupDone(CeedOperator op, bool *is_setup_done) {
  *is_setup_done = op->is_backend_setup;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the QFunction associated with a CeedOperator

  @param[in]  op CeedOperator
  @param[out] qf Variable to store QFunction

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorGetQFunction(CeedOperator op, CeedQFunction *qf) {
  CeedCheck(!op->is_composite, op->ceed, CEED_ERROR_MINOR, "Not defined for composite operator");
  *qf = op->qf;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get a boolean value indicating if the CeedOperator is composite

  @param[in]  op           CeedOperator
  @param[out] is_composite Variable to store composite status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorIsComposite(CeedOperator op, bool *is_composite) {
  *is_composite = op->is_composite;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a CeedOperator

  @param[in]  op   CeedOperator
  @param[out] data Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorGetData(CeedOperator op, void *data) {
  *(void **)data = op->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a CeedOperator

  @param[in,out] op   CeedOperator
  @param[in]     data Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorSetData(CeedOperator op, void *data) {
  op->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedOperator

  @param[in,out] op CeedOperator to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorReference(CeedOperator op) {
  op->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the setup flag of a CeedOperator to True

  @param[in,out] op CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorSetSetupDone(CeedOperator op) {
  op->is_backend_setup = true;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedOperator Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedOperatorUser
/// @{

/**
  @brief Create a CeedOperator and associate a CeedQFunction.

  A CeedBasis and CeedElemRestriction can be associated with CeedQFunction fields with \ref CeedOperatorSetField.

  @param[in]  ceed Ceed object where the CeedOperator will be created
  @param[in]  qf   QFunction defining the action of the operator at quadrature points
  @param[in]  dqf  QFunction defining the action of the Jacobian of @a qf (or @ref CEED_QFUNCTION_NONE)
  @param[in]  dqfT QFunction defining the action of the transpose of the Jacobian of @a qf (or @ref CEED_QFUNCTION_NONE)
  @param[out] op   Address of the variable where the newly created CeedOperator will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedOperatorCreate(Ceed ceed, CeedQFunction qf, CeedQFunction dqf, CeedQFunction dqfT, CeedOperator *op) {
  if (!ceed->OperatorCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Operator"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support OperatorCreate");
    CeedCall(CeedOperatorCreate(delegate, qf, dqf, dqfT, op));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(qf && qf != CEED_QFUNCTION_NONE, ceed, CEED_ERROR_MINOR, "Operator must have a valid QFunction.");

  CeedCall(CeedCalloc(1, op));
  CeedCall(CeedReferenceCopy(ceed, &(*op)->ceed));
  (*op)->ref_count   = 1;
  (*op)->input_size  = -1;
  (*op)->output_size = -1;
  CeedCall(CeedQFunctionReferenceCopy(qf, &(*op)->qf));
  if (dqf && dqf != CEED_QFUNCTION_NONE) CeedCall(CeedQFunctionReferenceCopy(dqf, &(*op)->dqf));
  if (dqfT && dqfT != CEED_QFUNCTION_NONE) CeedCall(CeedQFunctionReferenceCopy(dqfT, &(*op)->dqfT));
  CeedCall(CeedQFunctionAssemblyDataCreate(ceed, &(*op)->qf_assembled));
  CeedCall(CeedCalloc(CEED_FIELD_MAX, &(*op)->input_fields));
  CeedCall(CeedCalloc(CEED_FIELD_MAX, &(*op)->output_fields));
  CeedCall(ceed->OperatorCreate(*op));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create an operator that composes the action of several operators

  @param[in]  ceed Ceed object where the CeedOperator will be created
  @param[out] op   Address of the variable where the newly created Composite CeedOperator will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedCompositeOperatorCreate(Ceed ceed, CeedOperator *op) {
  if (!ceed->CompositeOperatorCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Operator"));
    if (delegate) {
      CeedCall(CeedCompositeOperatorCreate(delegate, op));
      return CEED_ERROR_SUCCESS;
    }
  }

  CeedCall(CeedCalloc(1, op));
  CeedCall(CeedReferenceCopy(ceed, &(*op)->ceed));
  (*op)->ref_count    = 1;
  (*op)->is_composite = true;
  CeedCall(CeedCalloc(CEED_COMPOSITE_MAX, &(*op)->sub_operators));
  (*op)->input_size  = -1;
  (*op)->output_size = -1;

  if (ceed->CompositeOperatorCreate) CeedCall(ceed->CompositeOperatorCreate(*op));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedOperator.

  Both pointers should be destroyed with `CeedOperatorDestroy()`.

  Note: If the value of `op_copy` passed to this function is non-NULL, then it is assumed that `op_copy` is a pointer to a CeedOperator.
        This CeedOperator will be destroyed if `op_copy` is the only reference to this CeedOperator.

  @param[in]  op         CeedOperator to copy reference to
  @param[in,out] op_copy Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorReferenceCopy(CeedOperator op, CeedOperator *op_copy) {
  CeedCall(CeedOperatorReference(op));
  CeedCall(CeedOperatorDestroy(op_copy));
  *op_copy = op;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Provide a field to a CeedOperator for use by its CeedQFunction.

  This function is used to specify both active and passive fields to a CeedOperator.
  For passive fields, a vector @arg v must be provided.
  Passive fields can inputs or outputs (updated in-place when operator is applied).

  Active fields must be specified using this function, but their data (in a CeedVector) is passed in CeedOperatorApply().
  There can be at most one active input CeedVector and at most one active output CeedVector passed to CeedOperatorApply().

  The number of quadrature points must agree across all points.
  When using @ref CEED_BASIS_COLLOCATED, the number of quadrature points is determined by the element size of r.

  @param[in,out] op         CeedOperator on which to provide the field
  @param[in]     field_name Name of the field (to be matched with the name used by CeedQFunction)
  @param[in]     r          CeedElemRestriction
  @param[in]     b          CeedBasis in which the field resides or @ref CEED_BASIS_COLLOCATED if collocated with quadrature points
  @param[in]     v          CeedVector to be used by CeedOperator or @ref CEED_VECTOR_ACTIVE if field is active or @ref CEED_VECTOR_NONE
                              if using @ref CEED_EVAL_WEIGHT in the QFunction

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetField(CeedOperator op, const char *field_name, CeedElemRestriction r, CeedBasis b, CeedVector v) {
  bool               is_input = true;
  CeedInt            num_elem = 0, num_qpts = 0;
  CeedQFunctionField qf_field;
  CeedOperatorField *op_field;

  CeedCheck(!op->is_composite, op->ceed, CEED_ERROR_INCOMPATIBLE, "Cannot add field to composite operator.");
  CeedCheck(!op->is_immutable, op->ceed, CEED_ERROR_MAJOR, "Operator cannot be changed after set as immutable");
  CeedCheck(r, op->ceed, CEED_ERROR_INCOMPATIBLE, "ElemRestriction r for field \"%s\" must be non-NULL.", field_name);
  CeedCheck(b, op->ceed, CEED_ERROR_INCOMPATIBLE, "Basis b for field \"%s\" must be non-NULL.", field_name);
  CeedCheck(v, op->ceed, CEED_ERROR_INCOMPATIBLE, "Vector v for field \"%s\" must be non-NULL.", field_name);

  CeedCall(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCheck(r == CEED_ELEMRESTRICTION_NONE || !op->has_restriction || op->num_elem == num_elem, op->ceed, CEED_ERROR_DIMENSION,
            "ElemRestriction with %" CeedInt_FMT " elements incompatible with prior %" CeedInt_FMT " elements", num_elem, op->num_elem);

  if (b == CEED_BASIS_COLLOCATED) CeedCall(CeedElemRestrictionGetElementSize(r, &num_qpts));
  else CeedCall(CeedBasisGetNumQuadraturePoints(b, &num_qpts));
  CeedCheck(op->num_qpts == 0 || op->num_qpts == num_qpts, op->ceed, CEED_ERROR_DIMENSION,
            "%s must correspond to the same number of quadrature points as previously added Bases. Found %" CeedInt_FMT
            " quadrature points but expected %" CeedInt_FMT " quadrature points.",
            b == CEED_BASIS_COLLOCATED ? "ElemRestriction" : "Basis", num_qpts, op->num_qpts);
  for (CeedInt i = 0; i < op->qf->num_input_fields; i++) {
    if (!strcmp(field_name, (*op->qf->input_fields[i]).field_name)) {
      qf_field = op->qf->input_fields[i];
      op_field = &op->input_fields[i];
      goto found;
    }
  }
  is_input = false;
  for (CeedInt i = 0; i < op->qf->num_output_fields; i++) {
    if (!strcmp(field_name, (*op->qf->output_fields[i]).field_name)) {
      qf_field = op->qf->output_fields[i];
      op_field = &op->output_fields[i];
      goto found;
    }
  }
  // LCOV_EXCL_START
  return CeedError(op->ceed, CEED_ERROR_INCOMPLETE, "QFunction has no knowledge of field '%s'", field_name);
  // LCOV_EXCL_STOP
found:
  CeedCall(CeedOperatorCheckField(op->ceed, qf_field, r, b));
  CeedCall(CeedCalloc(1, op_field));

  if (v == CEED_VECTOR_ACTIVE) {
    CeedSize l_size;

    CeedCall(CeedElemRestrictionGetLVectorSize(r, &l_size));
    if (is_input) {
      if (op->input_size == -1) op->input_size = l_size;
      CeedCheck(l_size == op->input_size, op->ceed, CEED_ERROR_INCOMPATIBLE, "LVector size %td does not match previous size %td", l_size,
                op->input_size);
    } else {
      if (op->output_size == -1) op->output_size = l_size;
      CeedCheck(l_size == op->output_size, op->ceed, CEED_ERROR_INCOMPATIBLE, "LVector size %td does not match previous size %td", l_size,
                op->output_size);
    }
  }

  CeedCall(CeedVectorReferenceCopy(v, &(*op_field)->vec));
  CeedCall(CeedElemRestrictionReferenceCopy(r, &(*op_field)->elem_rstr));
  if (r != CEED_ELEMRESTRICTION_NONE) {
    op->num_elem        = num_elem;
    op->has_restriction = true;  // Restriction set, but num_elem may be 0
  }
  CeedCall(CeedBasisReferenceCopy(b, &(*op_field)->basis));
  if (op->num_qpts == 0) CeedCall(CeedOperatorSetNumQuadraturePoints(op, num_qpts));

  op->num_fields += 1;
  CeedCall(CeedStringAllocCopy(field_name, (char **)&(*op_field)->field_name));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedOperatorFields of a CeedOperator

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op                CeedOperator
  @param[out] num_input_fields  Variable to store number of input fields
  @param[out] input_fields      Variable to store input_fields
  @param[out] num_output_fields Variable to store number of output fields
  @param[out] output_fields     Variable to store output_fields

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetFields(CeedOperator op, CeedInt *num_input_fields, CeedOperatorField **input_fields, CeedInt *num_output_fields,
                          CeedOperatorField **output_fields) {
  CeedCheck(!op->is_composite, op->ceed, CEED_ERROR_MINOR, "Not defined for composite operator");
  CeedCall(CeedOperatorCheckReady(op));

  if (num_input_fields) *num_input_fields = op->qf->num_input_fields;
  if (input_fields) *input_fields = op->input_fields;
  if (num_output_fields) *num_output_fields = op->qf->num_output_fields;
  if (output_fields) *output_fields = op->output_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get a CeedOperatorField of an CeedOperator from its name

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op         CeedOperator
  @param[in]  field_name Name of desired CeedOperatorField
  @param[out] op_field   CeedOperatorField corresponding to the name

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetFieldByName(CeedOperator op, const char *field_name, CeedOperatorField *op_field) {
  char              *name;
  CeedInt            num_input_fields, num_output_fields;
  CeedOperatorField *input_fields, *output_fields;

  CeedCall(CeedOperatorGetFields(op, &num_input_fields, &input_fields, &num_output_fields, &output_fields));
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCall(CeedOperatorFieldGetName(input_fields[i], &name));
    if (!strcmp(name, field_name)) {
      *op_field = input_fields[i];
      return CEED_ERROR_SUCCESS;
    }
  }
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCall(CeedOperatorFieldGetName(output_fields[i], &name));
    if (!strcmp(name, field_name)) {
      *op_field = output_fields[i];
      return CEED_ERROR_SUCCESS;
    }
  }
  // LCOV_EXCL_START
  bool has_name = op->name;

  return CeedError(op->ceed, CEED_ERROR_MINOR, "The field \"%s\" not found in CeedOperator%s%s%s.\n", field_name, has_name ? " \"" : "",
                   has_name ? op->name : "", has_name ? "\"" : "");
  // LCOV_EXCL_STOP
}

/**
  @brief Get the name of a CeedOperatorField

  @param[in]  op_field    CeedOperatorField
  @param[out] field_name  Variable to store the field name

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetName(CeedOperatorField op_field, char **field_name) {
  *field_name = (char *)op_field->field_name;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedElemRestriction of a CeedOperatorField

  @param[in]  op_field CeedOperatorField
  @param[out] rstr     Variable to store CeedElemRestriction

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetElemRestriction(CeedOperatorField op_field, CeedElemRestriction *rstr) {
  *rstr = op_field->elem_rstr;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedBasis of a CeedOperatorField

  @param[in]  op_field CeedOperatorField
  @param[out] basis    Variable to store CeedBasis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetBasis(CeedOperatorField op_field, CeedBasis *basis) {
  *basis = op_field->basis;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedVector of a CeedOperatorField

  @param[in]  op_field CeedOperatorField
  @param[out] vec      Variable to store CeedVector

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetVector(CeedOperatorField op_field, CeedVector *vec) {
  *vec = op_field->vec;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add a sub-operator to a composite CeedOperator

  @param[in,out] composite_op Composite CeedOperator
  @param[in]     sub_op       Sub-operator CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedCompositeOperatorAddSub(CeedOperator composite_op, CeedOperator sub_op) {
  CeedCheck(composite_op->is_composite, composite_op->ceed, CEED_ERROR_MINOR, "CeedOperator is not a composite operator");
  CeedCheck(composite_op->num_suboperators < CEED_COMPOSITE_MAX, composite_op->ceed, CEED_ERROR_UNSUPPORTED, "Cannot add additional sub-operators");
  CeedCheck(!composite_op->is_immutable, composite_op->ceed, CEED_ERROR_MAJOR, "Operator cannot be changed after set as immutable");

  {
    CeedSize input_size, output_size;

    CeedCall(CeedOperatorGetActiveVectorLengths(sub_op, &input_size, &output_size));
    if (composite_op->input_size == -1) composite_op->input_size = input_size;
    if (composite_op->output_size == -1) composite_op->output_size = output_size;
    // Note, a size of -1 means no active vector restriction set, so no incompatibility
    CeedCheck((input_size == -1 || input_size == composite_op->input_size) && (output_size == -1 || output_size == composite_op->output_size),
              composite_op->ceed, CEED_ERROR_MAJOR,
              "Sub-operators must have compatible dimensions; composite operator of shape (%td, %td) not compatible with sub-operator of "
              "shape (%td, %td)",
              composite_op->input_size, composite_op->output_size, input_size, output_size);
  }

  composite_op->sub_operators[composite_op->num_suboperators] = sub_op;
  CeedCall(CeedOperatorReference(sub_op));
  composite_op->num_suboperators++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of sub_operators associated with a CeedOperator

  @param[in]  op               CeedOperator
  @param[out] num_suboperators Variable to store number of sub_operators

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedCompositeOperatorGetNumSub(CeedOperator op, CeedInt *num_suboperators) {
  CeedCheck(op->is_composite, op->ceed, CEED_ERROR_MINOR, "Only defined for a composite operator");
  *num_suboperators = op->num_suboperators;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the list of sub_operators associated with a CeedOperator

  @param op                  CeedOperator
  @param[out] sub_operators  Variable to store list of sub_operators

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedCompositeOperatorGetSubList(CeedOperator op, CeedOperator **sub_operators) {
  CeedCheck(op->is_composite, op->ceed, CEED_ERROR_MINOR, "Only defined for a composite operator");
  *sub_operators = op->sub_operators;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check if a CeedOperator is ready to be used.

  @param[in] op CeedOperator to check

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorCheckReady(CeedOperator op) {
  Ceed ceed;

  CeedCall(CeedOperatorGetCeed(op, &ceed));

  if (op->is_interface_setup) return CEED_ERROR_SUCCESS;

  CeedQFunction qf = op->qf;
  if (op->is_composite) {
    if (!op->num_suboperators) {
      // Empty operator setup
      op->input_size  = 0;
      op->output_size = 0;
    } else {
      for (CeedInt i = 0; i < op->num_suboperators; i++) {
        CeedCall(CeedOperatorCheckReady(op->sub_operators[i]));
      }
      // Sub-operators could be modified after adding to composite operator
      // Need to verify no lvec incompatibility from any changes
      CeedSize input_size, output_size;
      CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
    }
  } else {
    CeedCheck(op->num_fields > 0, ceed, CEED_ERROR_INCOMPLETE, "No operator fields set");
    CeedCheck(op->num_fields == qf->num_input_fields + qf->num_output_fields, ceed, CEED_ERROR_INCOMPLETE, "Not all operator fields set");
    CeedCheck(op->has_restriction, ceed, CEED_ERROR_INCOMPLETE, "At least one restriction required");
    CeedCheck(op->num_qpts > 0, ceed, CEED_ERROR_INCOMPLETE,
              "At least one non-collocated basis is required or the number of quadrature points must be set");
  }

  // Flag as immutable and ready
  op->is_interface_setup = true;
  if (op->qf && op->qf != CEED_QFUNCTION_NONE) op->qf->is_immutable = true;
  if (op->dqf && op->dqf != CEED_QFUNCTION_NONE) op->dqf->is_immutable = true;
  if (op->dqfT && op->dqfT != CEED_QFUNCTION_NONE) op->dqfT->is_immutable = true;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get vector lengths for the active input and/or output vectors of a CeedOperator.

  Note: Lengths of -1 indicate that the CeedOperator does not have an active input and/or output.

  @param[in]  op          CeedOperator
  @param[out] input_size  Variable to store active input vector length, or NULL
  @param[out] output_size Variable to store active output vector length, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorGetActiveVectorLengths(CeedOperator op, CeedSize *input_size, CeedSize *output_size) {
  bool is_composite;

  if (input_size) *input_size = op->input_size;
  if (output_size) *output_size = op->output_size;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite && (op->input_size == -1 || op->output_size == -1)) {
    for (CeedInt i = 0; i < op->num_suboperators; i++) {
      CeedSize sub_input_size, sub_output_size;

      CeedCall(CeedOperatorGetActiveVectorLengths(op->sub_operators[i], &sub_input_size, &sub_output_size));
      if (op->input_size == -1) op->input_size = sub_input_size;
      if (op->output_size == -1) op->output_size = sub_output_size;
      // Note, a size of -1 means no active vector restriction set, so no incompatibility
      CeedCheck((sub_input_size == -1 || sub_input_size == op->input_size) && (sub_output_size == -1 || sub_output_size == op->output_size), op->ceed,
                CEED_ERROR_MAJOR,
                "Sub-operators must have compatible dimensions; composite operator of shape (%td, %td) not compatible with sub-operator of "
                "shape (%td, %td)",
                op->input_size, op->output_size, input_size, output_size);
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set reuse of CeedQFunction data in CeedOperatorLinearAssemble* functions.

  When `reuse_assembly_data = false` (default), the CeedQFunction associated with this CeedOperator is re-assembled every time a
`CeedOperatorLinearAssemble*` function is called.
  When `reuse_assembly_data = true`, the CeedQFunction associated with this CeedOperator is reused between calls to
`CeedOperatorSetQFunctionAssemblyDataUpdated`.

  @param[in] op                  CeedOperator
  @param[in] reuse_assembly_data Boolean flag setting assembly data reuse

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorSetQFunctionAssemblyReuse(CeedOperator op, bool reuse_assembly_data) {
  bool is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    for (CeedInt i = 0; i < op->num_suboperators; i++) {
      CeedCall(CeedOperatorSetQFunctionAssemblyReuse(op->sub_operators[i], reuse_assembly_data));
    }
  } else {
    CeedCall(CeedQFunctionAssemblyDataSetReuse(op->qf_assembled, reuse_assembly_data));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Mark CeedQFunction data as updated and the CeedQFunction as requiring re-assembly.

  @param[in] op                CeedOperator
  @param[in] needs_data_update Boolean flag setting assembly data reuse

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(CeedOperator op, bool needs_data_update) {
  bool is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    for (CeedInt i = 0; i < op->num_suboperators; i++) {
      CeedCall(CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op->sub_operators[i], needs_data_update));
    }
  } else {
    CeedCall(CeedQFunctionAssemblyDataSetUpdateNeeded(op->qf_assembled, needs_data_update));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the number of quadrature points associated with a CeedOperator.

  This should be used when creating a CeedOperator where every field has a collocated basis.
  This function cannot be used for composite CeedOperators.

  @param[in,out] op       CeedOperator
  @param[in]     num_qpts Number of quadrature points to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorSetNumQuadraturePoints(CeedOperator op, CeedInt num_qpts) {
  CeedCheck(!op->is_composite, op->ceed, CEED_ERROR_MINOR, "Not defined for composite CeedOperator");
  CeedCheck(!op->is_immutable, op->ceed, CEED_ERROR_MAJOR, "Operator cannot be changed after set as immutable");
  if (op->num_qpts > 0) {
    CeedWarn(
        "CeedOperatorSetNumQuadraturePoints will be removed from the libCEED interface in the next release.\n"
        "This function is reduntant and you can safely remove any calls to this function without replacing them.");
    CeedCheck(num_qpts == op->num_qpts, op->ceed, CEED_ERROR_DIMENSION, "Different number of quadrature points already defined for the CeedOperator");
  }
  op->num_qpts = num_qpts;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set name of CeedOperator for CeedOperatorView output

  @param[in,out] op   CeedOperator
  @param[in]     name Name to set, or NULL to remove previously set name

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetName(CeedOperator op, const char *name) {
  char  *name_copy;
  size_t name_len = name ? strlen(name) : 0;

  CeedCall(CeedFree(&op->name));
  if (name_len > 0) {
    CeedCall(CeedCalloc(name_len + 1, &name_copy));
    memcpy(name_copy, name, name_len);
    op->name = name_copy;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedOperator

  @param[in] op     CeedOperator to view
  @param[in] stream Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorView(CeedOperator op, FILE *stream) {
  bool has_name = op->name;

  if (op->is_composite) {
    fprintf(stream, "Composite CeedOperator%s%s\n", has_name ? " - " : "", has_name ? op->name : "");

    for (CeedInt i = 0; i < op->num_suboperators; i++) {
      has_name = op->sub_operators[i]->name;
      fprintf(stream, "  SubOperator %" CeedInt_FMT "%s%s:\n", i, has_name ? " - " : "", has_name ? op->sub_operators[i]->name : "");
      CeedCall(CeedOperatorSingleView(op->sub_operators[i], 1, stream));
    }
  } else {
    fprintf(stream, "CeedOperator%s%s\n", has_name ? " - " : "", has_name ? op->name : "");
    CeedCall(CeedOperatorSingleView(op, 0, stream));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the Ceed associated with a CeedOperator

  @param[in]  op   CeedOperator
  @param[out] ceed Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetCeed(CeedOperator op, Ceed *ceed) {
  *ceed = op->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of elements associated with a CeedOperator

  @param[in]  op       CeedOperator
  @param[out] num_elem Variable to store number of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetNumElements(CeedOperator op, CeedInt *num_elem) {
  CeedCheck(!op->is_composite, op->ceed, CEED_ERROR_MINOR, "Not defined for composite operator");
  *num_elem = op->num_elem;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of quadrature points associated with a CeedOperator

  @param[in]  op       CeedOperator
  @param[out] num_qpts Variable to store vector number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetNumQuadraturePoints(CeedOperator op, CeedInt *num_qpts) {
  CeedCheck(!op->is_composite, op->ceed, CEED_ERROR_MINOR, "Not defined for composite operator");
  *num_qpts = op->num_qpts;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Estimate number of FLOPs required to apply CeedOperator on the active vector

  @param[in]  op    CeedOperator to estimate FLOPs for
  @param[out] flops Address of variable to hold FLOPs estimate

  @ref Backend
**/
int CeedOperatorGetFlopsEstimate(CeedOperator op, CeedSize *flops) {
  bool is_composite;

  CeedCall(CeedOperatorCheckReady(op));

  *flops = 0;
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt num_suboperators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    CeedOperator *sub_operators;
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));

    // FLOPs for each suboperator
    for (CeedInt i = 0; i < num_suboperators; i++) {
      CeedSize suboperator_flops;

      CeedCall(CeedOperatorGetFlopsEstimate(sub_operators[i], &suboperator_flops));
      *flops += suboperator_flops;
    }
  } else {
    CeedInt            num_input_fields, num_output_fields, num_elem = 0;
    CeedOperatorField *input_fields, *output_fields;

    CeedCall(CeedOperatorGetFields(op, &num_input_fields, &input_fields, &num_output_fields, &output_fields));
    CeedCall(CeedOperatorGetNumElements(op, &num_elem));

    // Input FLOPs
    for (CeedInt i = 0; i < num_input_fields; i++) {
      if (input_fields[i]->vec == CEED_VECTOR_ACTIVE) {
        CeedSize restr_flops, basis_flops;

        CeedCall(CeedElemRestrictionGetFlopsEstimate(input_fields[i]->elem_rstr, CEED_NOTRANSPOSE, &restr_flops));
        *flops += restr_flops;
        CeedCall(CeedBasisGetFlopsEstimate(input_fields[i]->basis, CEED_NOTRANSPOSE, op->qf->input_fields[i]->eval_mode, &basis_flops));
        *flops += basis_flops * num_elem;
      }
    }
    // QF FLOPs
    {
      CeedInt  num_qpts;
      CeedSize qf_flops;

      CeedCall(CeedOperatorGetNumQuadraturePoints(op, &num_qpts));
      CeedCall(CeedQFunctionGetFlopsEstimate(op->qf, &qf_flops));
      *flops += num_elem * num_qpts * qf_flops;
    }

    // Output FLOPs
    for (CeedInt i = 0; i < num_output_fields; i++) {
      if (output_fields[i]->vec == CEED_VECTOR_ACTIVE) {
        CeedSize restr_flops, basis_flops;

        CeedCall(CeedElemRestrictionGetFlopsEstimate(output_fields[i]->elem_rstr, CEED_TRANSPOSE, &restr_flops));
        *flops += restr_flops;
        CeedCall(CeedBasisGetFlopsEstimate(output_fields[i]->basis, CEED_TRANSPOSE, op->qf->output_fields[i]->eval_mode, &basis_flops));
        *flops += basis_flops * num_elem;
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedQFunction global context for a CeedOperator.

  The caller is responsible for destroying `ctx` returned from this function via `CeedQFunctionContextDestroy()`.

  Note: If the value of `ctx` passed into this function is non-NULL, then it is assumed that `ctx` is a pointer to a CeedQFunctionContext.
        This CeedQFunctionContext will be destroyed if `ctx` is the only reference to this CeedQFunctionContext.

  @param[in]  op  CeedOperator
  @param[out] ctx Variable to store CeedQFunctionContext

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetContext(CeedOperator op, CeedQFunctionContext *ctx) {
  CeedCheck(!op->is_composite, op->ceed, CEED_ERROR_INCOMPATIBLE, "Cannot retrieve QFunctionContext for composite operator");
  if (op->qf->ctx) CeedCall(CeedQFunctionContextReferenceCopy(op->qf->ctx, ctx));
  else *ctx = NULL;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get label for a registered QFunctionContext field, or `NULL` if no field has been registered with this `field_name`.

  Fields are registered via `CeedQFunctionContextRegister*()` functions (eg. `CeedQFunctionContextRegisterDouble()`).

  @param[in]  op          CeedOperator
  @param[in]  field_name  Name of field to retrieve label
  @param[out] field_label Variable to field label

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorGetContextFieldLabel(CeedOperator op, const char *field_name, CeedContextFieldLabel *field_label) {
  bool is_composite, field_found = false;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));

  if (is_composite) {
    // Composite operator
    // -- Check if composite label already created
    for (CeedInt i = 0; i < op->num_context_labels; i++) {
      if (!strcmp(op->context_labels[i]->name, field_name)) {
        *field_label = op->context_labels[i];
        return CEED_ERROR_SUCCESS;
      }
    }

    // -- Create composite label if needed
    CeedInt               num_sub;
    CeedOperator         *sub_operators;
    CeedContextFieldLabel new_field_label;

    CeedCall(CeedCalloc(1, &new_field_label));
    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_sub));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    CeedCall(CeedCalloc(num_sub, &new_field_label->sub_labels));
    new_field_label->num_sub_labels = num_sub;

    for (CeedInt i = 0; i < num_sub; i++) {
      if (sub_operators[i]->qf->ctx) {
        CeedContextFieldLabel new_field_label_i;

        CeedCall(CeedQFunctionContextGetFieldLabel(sub_operators[i]->qf->ctx, field_name, &new_field_label_i));
        if (new_field_label_i) {
          field_found                    = true;
          new_field_label->sub_labels[i] = new_field_label_i;
          new_field_label->name          = new_field_label_i->name;
          new_field_label->description   = new_field_label_i->description;
          if (new_field_label->type && new_field_label->type != new_field_label_i->type) {
            // LCOV_EXCL_START
            CeedCall(CeedFree(&new_field_label));
            return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE, "Incompatible field types on sub-operator contexts. %s != %s",
                             CeedContextFieldTypes[new_field_label->type], CeedContextFieldTypes[new_field_label_i->type]);
            // LCOV_EXCL_STOP
          } else {
            new_field_label->type = new_field_label_i->type;
          }
          if (new_field_label->num_values != 0 && new_field_label->num_values != new_field_label_i->num_values) {
            // LCOV_EXCL_START
            CeedCall(CeedFree(&new_field_label));
            return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE, "Incompatible field number of values on sub-operator contexts. %ld != %ld",
                             new_field_label->num_values, new_field_label_i->num_values);
            // LCOV_EXCL_STOP
          } else {
            new_field_label->num_values = new_field_label_i->num_values;
          }
        }
      }
    }
    // -- Cleanup if field was found
    if (field_found) {
      *field_label = new_field_label;
    } else {
      // LCOV_EXCL_START
      CeedCall(CeedFree(&new_field_label->sub_labels));
      CeedCall(CeedFree(&new_field_label));
      *field_label = NULL;
      // LCOV_EXCL_STOP
    }
  } else {
    // Single, non-composite operator
    if (op->qf->ctx) {
      CeedCall(CeedQFunctionContextGetFieldLabel(op->qf->ctx, field_name, field_label));
    } else {
      *field_label = NULL;
    }
  }

  // Set label in operator
  if (*field_label) {
    (*field_label)->from_op = true;

    // Move new composite label to operator
    if (op->num_context_labels == 0) {
      CeedCall(CeedCalloc(1, &op->context_labels));
      op->max_context_labels = 1;
    } else if (op->num_context_labels == op->max_context_labels) {
      CeedCall(CeedRealloc(2 * op->num_context_labels, &op->context_labels));
      op->max_context_labels *= 2;
    }
    op->context_labels[op->num_context_labels] = *field_label;
    op->num_context_labels++;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set QFunctionContext field holding double precision values.

  For composite operators, the values are set in all sub-operator QFunctionContexts that have a matching `field_name`.

  @param[in,out] op          CeedOperator
  @param[in]     field_label Label of field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetContextDouble(CeedOperator op, CeedContextFieldLabel field_label, double *values) {
  return CeedOperatorContextSetGeneric(op, field_label, CEED_CONTEXT_FIELD_DOUBLE, values);
}

/**
  @brief Get QFunctionContext field holding double precision values, read-only.

  For composite operators, the values correspond to the first sub-operator QFunctionContexts that has a matching `field_name`.

  @param[in]  op          CeedOperator
  @param[in]  field_label Label of field to get
  @param[out] num_values  Number of values in the field label
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorGetContextDoubleRead(CeedOperator op, CeedContextFieldLabel field_label, size_t *num_values, const double **values) {
  return CeedOperatorContextGetGenericRead(op, field_label, CEED_CONTEXT_FIELD_DOUBLE, num_values, values);
}

/**
  @brief Restore QFunctionContext field holding double precision values, read-only.

  @param[in]  op          CeedOperator
  @param[in]  field_label Label of field to restore
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorRestoreContextDoubleRead(CeedOperator op, CeedContextFieldLabel field_label, const double **values) {
  return CeedOperatorContextRestoreGenericRead(op, field_label, CEED_CONTEXT_FIELD_DOUBLE, values);
}

/**
  @brief Set QFunctionContext field holding int32 values.

  For composite operators, the values are set in all sub-operator QFunctionContexts that have a matching `field_name`.

  @param[in,out] op          CeedOperator
  @param[in]     field_label Label of field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetContextInt32(CeedOperator op, CeedContextFieldLabel field_label, int *values) {
  return CeedOperatorContextSetGeneric(op, field_label, CEED_CONTEXT_FIELD_INT32, values);
}

/**
  @brief Get QFunctionContext field holding int32 values, read-only.

  For composite operators, the values correspond to the first sub-operator QFunctionContexts that has a matching `field_name`.

  @param[in]  op          CeedOperator
  @param[in]  field_label Label of field to get
  @param[out] num_values  Number of int32 values in `values`
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorGetContextInt32Read(CeedOperator op, CeedContextFieldLabel field_label, size_t *num_values, const int **values) {
  return CeedOperatorContextGetGenericRead(op, field_label, CEED_CONTEXT_FIELD_INT32, num_values, values);
}

/**
  @brief Restore QFunctionContext field holding int32 values, read-only.

  @param[in]  op          CeedOperator
  @param[in]  field_label Label of field to get
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorRestoreContextInt32Read(CeedOperator op, CeedContextFieldLabel field_label, const int **values) {
  return CeedOperatorContextRestoreGenericRead(op, field_label, CEED_CONTEXT_FIELD_INT32, values);
}

/**
  @brief Apply CeedOperator to a vector

  This computes the action of the operator on the specified (active) input, yielding its (active) output.
  All inputs and outputs must be specified using CeedOperatorSetField().

  Note: Calling this function asserts that setup is complete and sets the CeedOperator as immutable.

  @param[in]  op      CeedOperator to apply
  @param[in]  in      CeedVector containing input state or @ref CEED_VECTOR_NONE if there are no active inputs
  @param[out] out     CeedVector to store result of applying operator (must be distinct from @a in) or @ref CEED_VECTOR_NONE if there are no
active outputs
  @param[in]  request Address of CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorApply(CeedOperator op, CeedVector in, CeedVector out, CeedRequest *request) {
  CeedCall(CeedOperatorCheckReady(op));

  if (op->is_composite) {
    // Composite Operator
    if (op->ApplyComposite) {
      CeedCall(op->ApplyComposite(op, in, out, request));
    } else {
      CeedInt       num_suboperators;
      CeedOperator *sub_operators;

      CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
      CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));

      // Zero all output vectors
      if (out != CEED_VECTOR_NONE) CeedCall(CeedVectorSetValue(out, 0.0));
      for (CeedInt i = 0; i < num_suboperators; i++) {
        for (CeedInt j = 0; j < sub_operators[i]->qf->num_output_fields; j++) {
          CeedVector vec = sub_operators[i]->output_fields[j]->vec;

          if (vec != CEED_VECTOR_ACTIVE && vec != CEED_VECTOR_NONE) {
            CeedCall(CeedVectorSetValue(vec, 0.0));
          }
        }
      }
      // Apply
      for (CeedInt i = 0; i < op->num_suboperators; i++) {
        CeedCall(CeedOperatorApplyAdd(op->sub_operators[i], in, out, request));
      }
    }
  } else {
    // Standard Operator
    if (op->Apply) {
      CeedCall(op->Apply(op, in, out, request));
    } else {
      // Zero all output vectors
      CeedQFunction qf = op->qf;

      for (CeedInt i = 0; i < qf->num_output_fields; i++) {
        CeedVector vec = op->output_fields[i]->vec;

        if (vec == CEED_VECTOR_ACTIVE) vec = out;
        if (vec != CEED_VECTOR_NONE) CeedCall(CeedVectorSetValue(vec, 0.0));
      }
      // Apply
      if (op->num_elem) CeedCall(op->ApplyAdd(op, in, out, request));
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply CeedOperator to a vector and add result to output vector

  This computes the action of the operator on the specified (active) input, yielding its (active) output.
  All inputs and outputs must be specified using CeedOperatorSetField().

  @param[in]  op      CeedOperator to apply
  @param[in]  in      CeedVector containing input state or NULL if there are no active inputs
  @param[out] out     CeedVector to sum in result of applying operator (must be distinct from @a in) or NULL if there are no active outputs
  @param[in]  request Address of CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorApplyAdd(CeedOperator op, CeedVector in, CeedVector out, CeedRequest *request) {
  CeedCall(CeedOperatorCheckReady(op));

  if (op->is_composite) {
    // Composite Operator
    if (op->ApplyAddComposite) {
      CeedCall(op->ApplyAddComposite(op, in, out, request));
    } else {
      CeedInt       num_suboperators;
      CeedOperator *sub_operators;

      CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
      CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
      for (CeedInt i = 0; i < num_suboperators; i++) {
        CeedCall(CeedOperatorApplyAdd(sub_operators[i], in, out, request));
      }
    }
  } else if (op->num_elem) {
    // Standard Operator
    CeedCall(op->ApplyAdd(op, in, out, request));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedOperator

  @param[in,out] op CeedOperator to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorDestroy(CeedOperator *op) {
  if (!*op || --(*op)->ref_count > 0) {
    *op = NULL;
    return CEED_ERROR_SUCCESS;
  }
  if ((*op)->Destroy) CeedCall((*op)->Destroy(*op));
  CeedCall(CeedDestroy(&(*op)->ceed));
  // Free fields
  for (CeedInt i = 0; i < (*op)->num_fields; i++) {
    if ((*op)->input_fields[i]) {
      if ((*op)->input_fields[i]->elem_rstr != CEED_ELEMRESTRICTION_NONE) {
        CeedCall(CeedElemRestrictionDestroy(&(*op)->input_fields[i]->elem_rstr));
      }
      if ((*op)->input_fields[i]->basis != CEED_BASIS_COLLOCATED) {
        CeedCall(CeedBasisDestroy(&(*op)->input_fields[i]->basis));
      }
      if ((*op)->input_fields[i]->vec != CEED_VECTOR_ACTIVE && (*op)->input_fields[i]->vec != CEED_VECTOR_NONE) {
        CeedCall(CeedVectorDestroy(&(*op)->input_fields[i]->vec));
      }
      CeedCall(CeedFree(&(*op)->input_fields[i]->field_name));
      CeedCall(CeedFree(&(*op)->input_fields[i]));
    }
  }
  for (CeedInt i = 0; i < (*op)->num_fields; i++) {
    if ((*op)->output_fields[i]) {
      CeedCall(CeedElemRestrictionDestroy(&(*op)->output_fields[i]->elem_rstr));
      if ((*op)->output_fields[i]->basis != CEED_BASIS_COLLOCATED) {
        CeedCall(CeedBasisDestroy(&(*op)->output_fields[i]->basis));
      }
      if ((*op)->output_fields[i]->vec != CEED_VECTOR_ACTIVE && (*op)->output_fields[i]->vec != CEED_VECTOR_NONE) {
        CeedCall(CeedVectorDestroy(&(*op)->output_fields[i]->vec));
      }
      CeedCall(CeedFree(&(*op)->output_fields[i]->field_name));
      CeedCall(CeedFree(&(*op)->output_fields[i]));
    }
  }
  // Destroy sub_operators
  for (CeedInt i = 0; i < (*op)->num_suboperators; i++) {
    if ((*op)->sub_operators[i]) {
      CeedCall(CeedOperatorDestroy(&(*op)->sub_operators[i]));
    }
  }
  CeedCall(CeedQFunctionDestroy(&(*op)->qf));
  CeedCall(CeedQFunctionDestroy(&(*op)->dqf));
  CeedCall(CeedQFunctionDestroy(&(*op)->dqfT));
  // Destroy any composite labels
  if ((*op)->is_composite) {
    for (CeedInt i = 0; i < (*op)->num_context_labels; i++) {
      CeedCall(CeedFree(&(*op)->context_labels[i]->sub_labels));
      CeedCall(CeedFree(&(*op)->context_labels[i]));
    }
  }
  CeedCall(CeedFree(&(*op)->context_labels));

  // Destroy fallback
  CeedCall(CeedOperatorDestroy(&(*op)->op_fallback));

  // Destroy assembly data
  CeedCall(CeedQFunctionAssemblyDataDestroy(&(*op)->qf_assembled));
  CeedCall(CeedOperatorAssemblyDataDestroy(&(*op)->op_assembled));

  CeedCall(CeedFree(&(*op)->input_fields));
  CeedCall(CeedFree(&(*op)->output_fields));
  CeedCall(CeedFree(&(*op)->sub_operators));
  CeedCall(CeedFree(&(*op)->name));
  CeedCall(CeedFree(op));
  return CEED_ERROR_SUCCESS;
}

/// @}
