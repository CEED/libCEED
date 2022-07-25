// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed-impl.h>
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

  @param[in] ceed      Ceed object for error handling
  @param[in] qf_field  QFunction Field matching Operator Field
  @param[in] r         Operator Field ElemRestriction
  @param[in] b         Operator Field Basis

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCheckField(Ceed ceed, CeedQFunctionField qf_field,
                                  CeedElemRestriction r, CeedBasis b) {
  int ierr;
  CeedEvalMode eval_mode = qf_field->eval_mode;
  CeedInt dim = 1, num_comp = 1, Q_comp = 1, restr_num_comp = 1,
          size = qf_field->size;
  // Restriction
  if (r != CEED_ELEMRESTRICTION_NONE) {
    if (eval_mode == CEED_EVAL_WEIGHT) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPATIBLE,
                       "CEED_ELEMRESTRICTION_NONE should be used "
                       "for a field with eval mode CEED_EVAL_WEIGHT");
      // LCOV_EXCL_STOP
    }
    ierr = CeedElemRestrictionGetNumComponents(r, &restr_num_comp);
    CeedChk(ierr);
  }
  if ((r == CEED_ELEMRESTRICTION_NONE) != (eval_mode == CEED_EVAL_WEIGHT)) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_INCOMPATIBLE,
                     "CEED_ELEMRESTRICTION_NONE and CEED_EVAL_WEIGHT "
                     "must be used together.");
    // LCOV_EXCL_STOP
  }
  // Basis
  if (b != CEED_BASIS_COLLOCATED) {
    if (eval_mode == CEED_EVAL_NONE)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPATIBLE,
                       "Field '%s' configured with CEED_EVAL_NONE must "
                       "be used with CEED_BASIS_COLLOCATED",
                       qf_field->field_name);
    // LCOV_EXCL_STOP
    ierr = CeedBasisGetDimension(b, &dim); CeedChk(ierr);
    ierr = CeedBasisGetNumComponents(b, &num_comp); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadratureComponents(b, &Q_comp); CeedChk(ierr);
    if (r != CEED_ELEMRESTRICTION_NONE && restr_num_comp != num_comp) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Field '%s' of size %" CeedInt_FMT " and EvalMode %s: ElemRestriction "
                       "has %" CeedInt_FMT " components, but Basis has %" CeedInt_FMT " components",
                       qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode],
                       restr_num_comp, num_comp);
      // LCOV_EXCL_STOP
    }
  } else if (eval_mode != CEED_EVAL_NONE) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_INCOMPATIBLE,
                     "Field '%s' configured with %s cannot "
                     "be used with CEED_BASIS_COLLOCATED",
                     qf_field->field_name, CeedEvalModes[eval_mode]);
    // LCOV_EXCL_STOP

  }
  // Field size
  switch(eval_mode) {
  case CEED_EVAL_NONE:
    if (size != restr_num_comp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Field '%s' of size %" CeedInt_FMT " and EvalMode %s: ElemRestriction has "
                       CeedInt_FMT " components",
                       qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode],
                       restr_num_comp);
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_INTERP:
    if (size != num_comp*Q_comp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Field '%s' of size %" CeedInt_FMT
                       " and EvalMode %s: ElemRestriction/Basis has "
                       CeedInt_FMT " components",
                       qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode],
                       num_comp*Q_comp);
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_GRAD:
    if (size != num_comp * dim)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Field '%s' of size %" CeedInt_FMT " and EvalMode %s in %" CeedInt_FMT
                       " dimensions: "
                       "ElemRestriction/Basis has %" CeedInt_FMT " components",
                       qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode], dim,
                       num_comp);
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_WEIGHT:
    // No additional checks required
    break;
  case CEED_EVAL_DIV:
    if (size != num_comp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Field '%s' of size %" CeedInt_FMT
                       " and EvalMode %s: ElemRestriction/Basis has "
                       CeedInt_FMT " components",
                       qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode],
                       num_comp);
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_CURL:
    // Not implemented
    break;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a field of a CeedOperator

  @param[in] field         Operator field to view
  @param[in] qf_field      QFunction field (carries field name)
  @param[in] field_number  Number of field being viewed
  @param[in] sub           true indicates sub-operator, which increases indentation; false for top-level operator
  @param[in] input         true for an input field; false for output field
  @param[in] stream        Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedOperatorFieldView(CeedOperatorField field,
                                 CeedQFunctionField qf_field,
                                 CeedInt field_number, bool sub, bool input,
                                 FILE *stream) {
  const char *pre = sub ? "  " : "";
  const char *in_out = input ? "Input" : "Output";

  fprintf(stream, "%s    %s field %" CeedInt_FMT ":\n"
          "%s      Name: \"%s\"\n",
          pre, in_out, field_number, pre, qf_field->field_name);

  fprintf(stream, "%s      Size: %" CeedInt_FMT "\n", pre, qf_field->size);

  fprintf(stream, "%s      EvalMode: %s\n", pre,
          CeedEvalModes[qf_field->eval_mode]);

  if (field->basis == CEED_BASIS_COLLOCATED)
    fprintf(stream, "%s      Collocated basis\n", pre);

  if (field->vec == CEED_VECTOR_ACTIVE)
    fprintf(stream, "%s      Active vector\n", pre);
  else if (field->vec == CEED_VECTOR_NONE)
    fprintf(stream, "%s      No vector\n", pre);

  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a single CeedOperator

  @param[in] op      CeedOperator to view
  @param[in] sub     Boolean flag for sub-operator
  @param[in] stream  Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedOperatorSingleView(CeedOperator op, bool sub, FILE *stream) {
  int ierr;
  const char *pre = sub ? "  " : "";

  CeedInt num_elem, num_qpts;
  ierr = CeedOperatorGetNumElements(op, &num_elem); CeedChk(ierr);
  ierr = CeedOperatorGetNumQuadraturePoints(op, &num_qpts); CeedChk(ierr);

  CeedInt total_fields = 0;
  ierr = CeedOperatorGetNumArgs(op, &total_fields); CeedChk(ierr);
  fprintf(stream, "%s  %" CeedInt_FMT " elements with %" CeedInt_FMT
          " quadrature points each\n",
          pre, num_elem, num_qpts);

  fprintf(stream, "%s  %" CeedInt_FMT " field%s\n", pre, total_fields,
          total_fields>1 ? "s" : "");

  fprintf(stream, "%s  %" CeedInt_FMT " input field%s:\n", pre,
          op->qf->num_input_fields,
          op->qf->num_input_fields>1 ? "s" : "");
  for (CeedInt i=0; i<op->qf->num_input_fields; i++) {
    ierr = CeedOperatorFieldView(op->input_fields[i], op->qf->input_fields[i],
                                 i, sub, 1, stream); CeedChk(ierr);
  }

  fprintf(stream, "%s  %" CeedInt_FMT " output field%s:\n", pre,
          op->qf->num_output_fields,
          op->qf->num_output_fields>1 ? "s" : "");
  for (CeedInt i=0; i<op->qf->num_output_fields; i++) {
    ierr = CeedOperatorFieldView(op->output_fields[i], op->qf->output_fields[i],
                                 i, sub, 0, stream); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active vector basis for a non-composite CeedOperator

  @param[in] op             CeedOperator to find active basis for
  @param[out] active_basis  Basis for active input vector or NULL for composite operator

  @return An error code: 0 - success, otherwise - failure

  @ ref Developer
**/
int CeedOperatorGetActiveBasis(CeedOperator op, CeedBasis *active_basis) {
  *active_basis = NULL;
  if (op->is_composite) return CEED_ERROR_SUCCESS;
  for (CeedInt i = 0; i < op->qf->num_input_fields; i++)
    if (op->input_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      *active_basis = op->input_fields[i]->basis;
      break;
    }

  if (!*active_basis) {
    // LCOV_EXCL_START
    int ierr;
    Ceed ceed;
    ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
    return CeedError(ceed, CEED_ERROR_MINOR,
                     "No active CeedBasis found");
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active vector ElemRestriction for a non-composite CeedOperator

  @param[in] op            CeedOperator to find active basis for
  @param[out] active_rstr  ElemRestriction for active input vector or NULL for composite operator

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedOperatorGetActiveElemRestriction(CeedOperator op,
    CeedElemRestriction *active_rstr) {
  *active_rstr = NULL;
  if (op->is_composite) return CEED_ERROR_SUCCESS;
  for (CeedInt i = 0; i < op->qf->num_input_fields; i++)
    if (op->input_fields[i]->vec == CEED_VECTOR_ACTIVE) {
      *active_rstr = op->input_fields[i]->elem_restr;
      break;
    }

  if (!*active_rstr) {
    // LCOV_EXCL_START
    int ierr;
    Ceed ceed;
    ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);
    return CeedError(ceed, CEED_ERROR_INCOMPLETE,
                     "No active CeedElemRestriction found");
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set QFunctionContext field value of the specified type.
           For composite operators, the value is set in all
           sub-operator QFunctionContexts that have a matching `field_name`.
           A non-zero error code is returned for single operators
           that do not have a matching field of the same type or composite
           operators that do not have any field of a matching type.

  @param op          CeedOperator
  @param field_label Label of field to set
  @param field_type  Type of field to set
  @param value       Value to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
static int CeedOperatorContextSetGeneric(CeedOperator op,
    CeedContextFieldLabel field_label, CeedContextFieldType field_type,
    void *value) {
  int ierr;

  if (!field_label)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED,
                     "Invalid field label");
  // LCOV_EXCL_STOP

  bool is_composite = false;
  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);
  if (is_composite) {
    CeedInt num_sub;
    CeedOperator *sub_operators;

    ierr = CeedOperatorGetNumSub(op, &num_sub); CeedChk(ierr);
    ierr = CeedOperatorGetSubList(op, &sub_operators); CeedChk(ierr);
    if (num_sub != field_label->num_sub_labels)
      // LCOV_EXCL_START
      return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED,
                       "ContextLabel does not correspond to composite operator.\n"
                       "Use CeedOperatorGetContextFieldLabel().");
    // LCOV_EXCL_STOP

    for (CeedInt i = 0; i < num_sub; i++) {
      // Try every sub-operator, ok if some sub-operators do not have field
      if (field_label->sub_labels[i] && sub_operators[i]->qf->ctx) {
        ierr = CeedQFunctionContextSetGeneric(sub_operators[i]->qf->ctx,
                                              field_label->sub_labels[i],
                                              field_type, value); CeedChk(ierr);
      }
    }
  } else {
    if (!op->qf->ctx)
      // LCOV_EXCL_START
      return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED,
                       "QFunction does not have context data");
    // LCOV_EXCL_STOP

    ierr = CeedQFunctionContextSetGeneric(op->qf->ctx, field_label,
                                          field_type, value); CeedChk(ierr);
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

  @param op             CeedOperator
  @param[out] num_args  Variable to store vector number of arguments

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetNumArgs(CeedOperator op, CeedInt *num_args) {
  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operators");
  // LCOV_EXCL_STOP

  *num_args = op->num_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the setup status of a CeedOperator

  @param op                  CeedOperator
  @param[out] is_setup_done  Variable to store setup status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorIsSetupDone(CeedOperator op, bool *is_setup_done) {
  *is_setup_done = op->is_backend_setup;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the QFunction associated with a CeedOperator

  @param op       CeedOperator
  @param[out] qf  Variable to store QFunction

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetQFunction(CeedOperator op, CeedQFunction *qf) {
  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP

  *qf = op->qf;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get a boolean value indicating if the CeedOperator is composite

  @param op                 CeedOperator
  @param[out] is_composite  Variable to store composite status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorIsComposite(CeedOperator op, bool *is_composite) {
  *is_composite = op->is_composite;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of sub_operators associated with a CeedOperator

  @param op                     CeedOperator
  @param[out] num_suboperators  Variable to store number of sub_operators

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetNumSub(CeedOperator op, CeedInt *num_suboperators) {
  if (!op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR, "Not a composite operator");
  // LCOV_EXCL_STOP

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

int CeedOperatorGetSubList(CeedOperator op, CeedOperator **sub_operators) {
  if (!op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR, "Not a composite operator");
  // LCOV_EXCL_STOP

  *sub_operators = op->sub_operators;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a CeedOperator

  @param op         CeedOperator
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetData(CeedOperator op, void *data) {
  *(void **)data = op->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a CeedOperator

  @param[out] op  CeedOperator
  @param data     Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorSetData(CeedOperator op, void *data) {
  op->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedOperator

  @param op  CeedOperator to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorReference(CeedOperator op) {
  op->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the setup flag of a CeedOperator to True

  @param op  CeedOperator

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
  @brief Create a CeedOperator and associate a CeedQFunction. A CeedBasis and
           CeedElemRestriction can be associated with CeedQFunction fields with
           \ref CeedOperatorSetField.

  @param ceed     A Ceed object where the CeedOperator will be created
  @param qf       QFunction defining the action of the operator at quadrature points
  @param dqf      QFunction defining the action of the Jacobian of @a qf (or
                    @ref CEED_QFUNCTION_NONE)
  @param dqfT     QFunction defining the action of the transpose of the Jacobian
                    of @a qf (or @ref CEED_QFUNCTION_NONE)
  @param[out] op  Address of the variable where the newly created
                    CeedOperator will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedOperatorCreate(Ceed ceed, CeedQFunction qf, CeedQFunction dqf,
                       CeedQFunction dqfT, CeedOperator *op) {
  int ierr;

  if (!ceed->OperatorCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Operator"); CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                       "Backend does not support OperatorCreate");
    // LCOV_EXCL_STOP

    ierr = CeedOperatorCreate(delegate, qf, dqf, dqfT, op); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  if (!qf || qf == CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MINOR,
                     "Operator must have a valid QFunction.");
  // LCOV_EXCL_STOP
  ierr = CeedCalloc(1, op); CeedChk(ierr);
  (*op)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);
  (*op)->ref_count = 1;
  (*op)->qf = qf;
  (*op)->input_size = -1;
  (*op)->output_size = -1;
  ierr = CeedQFunctionReference(qf); CeedChk(ierr);
  if (dqf && dqf != CEED_QFUNCTION_NONE) {
    (*op)->dqf = dqf;
    ierr = CeedQFunctionReference(dqf); CeedChk(ierr);
  }
  if (dqfT && dqfT != CEED_QFUNCTION_NONE) {
    (*op)->dqfT = dqfT;
    ierr = CeedQFunctionReference(dqfT); CeedChk(ierr);
  }
  ierr = CeedQFunctionAssemblyDataCreate(ceed, &(*op)->qf_assembled);
  CeedChk(ierr);
  ierr = CeedCalloc(CEED_FIELD_MAX, &(*op)->input_fields); CeedChk(ierr);
  ierr = CeedCalloc(CEED_FIELD_MAX, &(*op)->output_fields); CeedChk(ierr);
  ierr = ceed->OperatorCreate(*op); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create an operator that composes the action of several operators

  @param ceed     A Ceed object where the CeedOperator will be created
  @param[out] op  Address of the variable where the newly created
                    Composite CeedOperator will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedCompositeOperatorCreate(Ceed ceed, CeedOperator *op) {
  int ierr;

  if (!ceed->CompositeOperatorCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Operator"); CeedChk(ierr);

    if (delegate) {
      ierr = CeedCompositeOperatorCreate(delegate, op); CeedChk(ierr);
      return CEED_ERROR_SUCCESS;
    }
  }

  ierr = CeedCalloc(1, op); CeedChk(ierr);
  (*op)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);
  (*op)->ref_count = 1;
  (*op)->is_composite = true;
  ierr = CeedCalloc(CEED_COMPOSITE_MAX, &(*op)->sub_operators); CeedChk(ierr);
  (*op)->input_size = -1;
  (*op)->output_size = -1;

  if (ceed->CompositeOperatorCreate) {
    ierr = ceed->CompositeOperatorCreate(*op); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedOperator. Both pointers should
           be destroyed with `CeedOperatorDestroy()`;
           Note: If `*op_copy` is non-NULL, then it is assumed that
           `*op_copy` is a pointer to a CeedOperator. This
           CeedOperator will be destroyed if `*op_copy` is the only
           reference to this CeedOperator.

  @param op            CeedOperator to copy reference to
  @param[out] op_copy  Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorReferenceCopy(CeedOperator op, CeedOperator *op_copy) {
  int ierr;

  ierr = CeedOperatorReference(op); CeedChk(ierr);
  ierr = CeedOperatorDestroy(op_copy); CeedChk(ierr);
  *op_copy = op;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Provide a field to a CeedOperator for use by its CeedQFunction

  This function is used to specify both active and passive fields to a
  CeedOperator.  For passive fields, a vector @arg v must be provided.  Passive
  fields can inputs or outputs (updated in-place when operator is applied).

  Active fields must be specified using this function, but their data (in a
  CeedVector) is passed in CeedOperatorApply().  There can be at most one active
  input CeedVector and at most one active output CeedVector passed to
  CeedOperatorApply().

  @param op          CeedOperator on which to provide the field
  @param field_name  Name of the field (to be matched with the name used by
                       CeedQFunction)
  @param r           CeedElemRestriction
  @param b           CeedBasis in which the field resides or @ref CEED_BASIS_COLLOCATED
                       if collocated with quadrature points
  @param v           CeedVector to be used by CeedOperator or @ref CEED_VECTOR_ACTIVE
                       if field is active or @ref CEED_VECTOR_NONE if using
                       @ref CEED_EVAL_WEIGHT in the QFunction

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetField(CeedOperator op, const char *field_name,
                         CeedElemRestriction r, CeedBasis b, CeedVector v) {
  int ierr;
  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE,
                     "Cannot add field to composite operator.");
  // LCOV_EXCL_STOP
  if (op->is_immutable)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MAJOR,
                     "Operator cannot be changed after set as immutable");
  // LCOV_EXCL_STOP
  if (!r)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE,
                     "ElemRestriction r for field \"%s\" must be non-NULL.",
                     field_name);
  // LCOV_EXCL_STOP
  if (!b)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE,
                     "Basis b for field \"%s\" must be non-NULL.",
                     field_name);
  // LCOV_EXCL_STOP
  if (!v)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE,
                     "Vector v for field \"%s\" must be non-NULL.",
                     field_name);
  // LCOV_EXCL_STOP

  CeedInt num_elem;
  ierr = CeedElemRestrictionGetNumElements(r, &num_elem); CeedChk(ierr);
  if (r != CEED_ELEMRESTRICTION_NONE && op->has_restriction &&
      op->num_elem != num_elem)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION,
                     "ElemRestriction with %" CeedInt_FMT " elements incompatible with prior %"
                     CeedInt_FMT " elements", num_elem, op->num_elem);
  // LCOV_EXCL_STOP

  CeedInt num_qpts = 0;
  if (b != CEED_BASIS_COLLOCATED) {
    ierr = CeedBasisGetNumQuadraturePoints(b, &num_qpts); CeedChk(ierr);
    if (op->num_qpts && op->num_qpts != num_qpts)
      // LCOV_EXCL_START
      return CeedError(op->ceed, CEED_ERROR_DIMENSION,
                       "Basis with %" CeedInt_FMT " quadrature points "
                       "incompatible with prior %" CeedInt_FMT " points", num_qpts,
                       op->num_qpts);
    // LCOV_EXCL_STOP
  }
  CeedQFunctionField qf_field;
  CeedOperatorField *op_field;
  bool is_input = true;
  for (CeedInt i=0; i<op->qf->num_input_fields; i++) {
    if (!strcmp(field_name, (*op->qf->input_fields[i]).field_name)) {
      qf_field = op->qf->input_fields[i];
      op_field = &op->input_fields[i];
      goto found;
    }
  }
  is_input = false;
  for (CeedInt i=0; i<op->qf->num_output_fields; i++) {
    if (!strcmp(field_name, (*op->qf->output_fields[i]).field_name)) {
      qf_field = op->qf->output_fields[i];
      op_field = &op->output_fields[i];
      goto found;
    }
  }
  // LCOV_EXCL_START
  return CeedError(op->ceed, CEED_ERROR_INCOMPLETE,
                   "QFunction has no knowledge of field '%s'",
                   field_name);
  // LCOV_EXCL_STOP
found:
  ierr = CeedOperatorCheckField(op->ceed, qf_field, r, b); CeedChk(ierr);
  ierr = CeedCalloc(1, op_field); CeedChk(ierr);

  if (v == CEED_VECTOR_ACTIVE) {
    CeedSize l_size;
    ierr = CeedElemRestrictionGetLVectorSize(r, &l_size); CeedChk(ierr);
    if (is_input) {
      if (op->input_size == -1) op->input_size = l_size;
      if (l_size != op->input_size)
        // LCOV_EXCL_START
        return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE,
                         "LVector size %td does not match previous size %td",
                         l_size, op->input_size);
      // LCOV_EXCL_STOP
    } else {
      if (op->output_size == -1) op->output_size = l_size;
      if (l_size != op->output_size)
        // LCOV_EXCL_START
        return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE,
                         "LVector size %td does not match previous size %td",
                         l_size, op->output_size);
      // LCOV_EXCL_STOP
    }
  }

  (*op_field)->vec = v;
  if (v != CEED_VECTOR_ACTIVE && v != CEED_VECTOR_NONE) {
    ierr = CeedVectorReference(v); CeedChk(ierr);
  }

  (*op_field)->elem_restr = r;
  ierr = CeedElemRestrictionReference(r); CeedChk(ierr);
  if (r != CEED_ELEMRESTRICTION_NONE) {
    op->num_elem = num_elem;
    op->has_restriction = true; // Restriction set, but num_elem may be 0
  }

  (*op_field)->basis = b;
  if (b != CEED_BASIS_COLLOCATED) {
    if (!op->num_qpts) {
      ierr = CeedOperatorSetNumQuadraturePoints(op, num_qpts); CeedChk(ierr);
    }
    ierr = CeedBasisReference(b); CeedChk(ierr);
  }

  op->num_fields += 1;
  ierr = CeedStringAllocCopy(field_name, (char **)&(*op_field)->field_name);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedOperatorFields of a CeedOperator

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param op                      CeedOperator
  @param[out] num_input_fields   Variable to store number of input fields
  @param[out] input_fields       Variable to store input_fields
  @param[out] num_output_fields  Variable to store number of output fields
  @param[out] output_fields      Variable to store output_fields

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetFields(CeedOperator op, CeedInt *num_input_fields,
                          CeedOperatorField **input_fields,
                          CeedInt *num_output_fields,
                          CeedOperatorField **output_fields) {
  int ierr;

  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  if (num_input_fields) *num_input_fields = op->qf->num_input_fields;
  if (input_fields) *input_fields = op->input_fields;
  if (num_output_fields) *num_output_fields = op->qf->num_output_fields;
  if (output_fields) *output_fields = op->output_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the name of a CeedOperatorField

  @param op_field         CeedOperatorField
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

  @param op_field   CeedOperatorField
  @param[out] rstr  Variable to store CeedElemRestriction

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetElemRestriction(CeedOperatorField op_field,
                                        CeedElemRestriction *rstr) {
  *rstr = op_field->elem_restr;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedBasis of a CeedOperatorField

  @param op_field    CeedOperatorField
  @param[out] basis  Variable to store CeedBasis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetBasis(CeedOperatorField op_field, CeedBasis *basis) {
  *basis = op_field->basis;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedVector of a CeedOperatorField

  @param op_field  CeedOperatorField
  @param[out] vec  Variable to store CeedVector

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetVector(CeedOperatorField op_field, CeedVector *vec) {
  *vec = op_field->vec;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add a sub-operator to a composite CeedOperator

  @param[out] composite_op  Composite CeedOperator
  @param      sub_op        Sub-operator CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedCompositeOperatorAddSub(CeedOperator composite_op,
                                CeedOperator sub_op) {
  int ierr;

  if (!composite_op->is_composite)
    // LCOV_EXCL_START
    return CeedError(composite_op->ceed, CEED_ERROR_MINOR,
                     "CeedOperator is not a composite operator");
  // LCOV_EXCL_STOP

  if (composite_op->num_suboperators == CEED_COMPOSITE_MAX)
    // LCOV_EXCL_START
    return CeedError(composite_op->ceed, CEED_ERROR_UNSUPPORTED,
                     "Cannot add additional sub-operators");
  // LCOV_EXCL_STOP
  if (composite_op->is_immutable)
    // LCOV_EXCL_START
    return CeedError(composite_op->ceed, CEED_ERROR_MAJOR,
                     "Operator cannot be changed after set as immutable");
  // LCOV_EXCL_STOP

  {
    CeedSize input_size, output_size;
    ierr = CeedOperatorGetActiveVectorLengths(sub_op, &input_size, &output_size);
    CeedChk(ierr);
    if (composite_op->input_size == -1) composite_op->input_size = input_size;
    if (composite_op->output_size == -1) composite_op->output_size = output_size;
    // Note, a size of -1 means no active vector restriction set, so no incompatibility
    if ((input_size != -1 && input_size != composite_op->input_size) ||
        (output_size != -1 && output_size != composite_op->output_size))
      // LCOV_EXCL_START
      return CeedError(composite_op->ceed, CEED_ERROR_MAJOR,
                       "Sub-operators must have compatible dimensions; "
                       "composite operator of shape (%td, %td) not compatible with "
                       "sub-operator of shape (%td, %td)",
                       composite_op->input_size, composite_op->output_size, input_size, output_size);
    // LCOV_EXCL_STOP
  }

  composite_op->sub_operators[composite_op->num_suboperators] = sub_op;
  ierr = CeedOperatorReference(sub_op); CeedChk(ierr);
  composite_op->num_suboperators++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check if a CeedOperator is ready to be used.

  @param[in] op  CeedOperator to check

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorCheckReady(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

  if (op->is_interface_setup)
    return CEED_ERROR_SUCCESS;

  CeedQFunction qf = op->qf;
  if (op->is_composite) {
    if (!op->num_suboperators)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE, "No sub_operators set");
    // LCOV_EXCL_STOP
    for (CeedInt i = 0; i < op->num_suboperators; i++) {
      ierr = CeedOperatorCheckReady(op->sub_operators[i]); CeedChk(ierr);
    }
    // Sub-operators could be modified after adding to composite operator
    // Need to verify no lvec incompatibility from any changes
    CeedSize input_size, output_size;
    ierr = CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size);
    CeedChk(ierr);
  } else {
    if (op->num_fields == 0)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE, "No operator fields set");
    // LCOV_EXCL_STOP
    if (op->num_fields < qf->num_input_fields + qf->num_output_fields)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE, "Not all operator fields set");
    // LCOV_EXCL_STOP
    if (!op->has_restriction)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE,
                       "At least one restriction required");
    // LCOV_EXCL_STOP
    if (op->num_qpts == 0)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE,
                       "At least one non-collocated basis is required "
                       "or the number of quadrature points must be set");
    // LCOV_EXCL_STOP
  }

  // Flag as immutable and ready
  op->is_interface_setup = true;
  if (op->qf && op->qf != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    op->qf->is_immutable = true;
  // LCOV_EXCL_STOP
  if (op->dqf && op->dqf != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    op->dqf->is_immutable = true;
  // LCOV_EXCL_STOP
  if (op->dqfT && op->dqfT != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    op->dqfT->is_immutable = true;
  // LCOV_EXCL_STOP
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get vector lengths for the active input and/or output vectors of a CeedOperator.
           Note: Lengths of -1 indicate that the CeedOperator does not have an
           active input and/or output.

  @param[in] op           CeedOperator
  @param[out] input_size  Variable to store active input vector length, or NULL
  @param[out] output_size Variable to store active output vector length, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorGetActiveVectorLengths(CeedOperator op, CeedSize *input_size,
                                       CeedSize *output_size) {
  int ierr;
  bool is_composite;

  if (input_size) *input_size = op->input_size;
  if (output_size) *output_size = op->output_size;

  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);
  if (is_composite && (op->input_size == -1 || op->output_size == -1)) {
    for (CeedInt i = 0; i < op->num_suboperators; i++) {
      CeedSize sub_input_size, sub_output_size;
      ierr = CeedOperatorGetActiveVectorLengths(op->sub_operators[i],
             &sub_input_size, &sub_output_size); CeedChk(ierr);
      if (op->input_size == -1) op->input_size = sub_input_size;
      if (op->output_size == -1) op->output_size = sub_output_size;
      // Note, a size of -1 means no active vector restriction set, so no incompatibility
      if ((sub_input_size != -1 && sub_input_size != op->input_size) ||
          (sub_output_size != -1 && sub_output_size != op->output_size))
        // LCOV_EXCL_START
        return CeedError(op->ceed, CEED_ERROR_MAJOR,
                         "Sub-operators must have compatible dimensions; "
                         "composite operator of shape (%td, %td) not compatible with "
                         "sub-operator of shape (%td, %td)",
                         op->input_size, op->output_size, input_size, output_size);
      // LCOV_EXCL_STOP
    }
  }

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set reuse of CeedQFunction data in CeedOperatorLinearAssemble* functions.
           When `reuse_assembly_data = false` (default), the CeedQFunction associated
           with this CeedOperator is re-assembled every time a `CeedOperatorLinearAssemble*`
           function is called.
           When `reuse_assembly_data = true`, the CeedQFunction associated with
           this CeedOperator is reused between calls to
           `CeedOperatorSetQFunctionAssemblyDataUpdated`.

  @param[in] op                  CeedOperator
  @param[in] reuse_assembly_data Boolean flag setting assembly data reuse

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorSetQFunctionAssemblyReuse(CeedOperator op,
    bool reuse_assembly_data) {
  int ierr;
  bool is_composite;

  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);
  if (is_composite) {
    for (CeedInt i = 0; i < op->num_suboperators; i++) {
      ierr = CeedOperatorSetQFunctionAssemblyReuse(op->sub_operators[i],
             reuse_assembly_data); CeedChk(ierr);
    }
  } else {
    ierr = CeedQFunctionAssemblyDataSetReuse(op->qf_assembled, reuse_assembly_data);
    CeedChk(ierr);
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
int CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(CeedOperator op,
    bool needs_data_update) {
  int ierr;
  bool is_composite;

  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);
  if (is_composite) {
    for (CeedInt i = 0; i < op->num_suboperators; i++) {
      ierr = CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op->sub_operators[i],
             needs_data_update); CeedChk(ierr);
    }
  } else {
    ierr = CeedQFunctionAssemblyDataSetUpdateNeeded(op->qf_assembled,
           needs_data_update);
    CeedChk(ierr);
  }

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the number of quadrature points associated with a CeedOperator.
           This should be used when creating a CeedOperator where every
           field has a collocated basis. This function cannot be used for
           composite CeedOperators.

  @param op        CeedOperator
  @param num_qpts  Number of quadrature points to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorSetNumQuadraturePoints(CeedOperator op, CeedInt num_qpts) {
  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP
  if (op->num_qpts)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Number of quadrature points already defined");
  // LCOV_EXCL_STOP
  if (op->is_immutable)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MAJOR,
                     "Operator cannot be changed after set as immutable");
  // LCOV_EXCL_STOP

  op->num_qpts = num_qpts;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set name of CeedOperator for CeedOperatorView output

  @param op    CeedOperator
  @param name  Name to set, or NULL to remove previously set name

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetName(CeedOperator op, const char *name) {
  int ierr;
  char *name_copy;
  size_t name_len = name ? strlen(name) : 0;

  ierr = CeedFree(&op->name); CeedChk(ierr);
  if (name_len > 0) {
    ierr = CeedCalloc(name_len + 1, &name_copy); CeedChk(ierr);
    memcpy(name_copy, name, name_len);
    op->name = name_copy;
  }

  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedOperator

  @param[in] op      CeedOperator to view
  @param[in] stream  Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorView(CeedOperator op, FILE *stream) {
  int ierr;
  bool has_name = op->name;

  if (op->is_composite) {
    fprintf(stream, "Composite CeedOperator%s%s\n",
            has_name ? " - " : "", has_name ? op->name : "");

    for (CeedInt i=0; i<op->num_suboperators; i++) {
      has_name = op->sub_operators[i]->name;
      fprintf(stream, "  SubOperator %" CeedInt_FMT "%s%s:\n", i,
              has_name ? " - " : "",
              has_name ? op->sub_operators[i]->name : "");
      ierr = CeedOperatorSingleView(op->sub_operators[i], 1, stream);
      CeedChk(ierr);
    }
  } else {
    fprintf(stream, "CeedOperator%s%s\n",
            has_name ? " - " : "", has_name ? op->name : "");
    ierr = CeedOperatorSingleView(op, 0, stream); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the Ceed associated with a CeedOperator

  @param op         CeedOperator
  @param[out] ceed  Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetCeed(CeedOperator op, Ceed *ceed) {
  *ceed = op->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of elements associated with a CeedOperator

  @param op             CeedOperator
  @param[out] num_elem  Variable to store number of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetNumElements(CeedOperator op, CeedInt *num_elem) {
  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP

  *num_elem = op->num_elem;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of quadrature points associated with a CeedOperator

  @param op             CeedOperator
  @param[out] num_qpts  Variable to store vector number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetNumQuadraturePoints(CeedOperator op, CeedInt *num_qpts) {
  if (op->is_composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP

  *num_qpts = op->num_qpts;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Estimate number of FLOPs required to apply CeedOperator on the active vector

  @param op    Operator to estimate FLOPs for
  @param flops Address of variable to hold FLOPs estimate

  @ref Backend
**/
int CeedOperatorGetFlopsEstimate(CeedOperator op, CeedSize *flops) {
  int ierr;
  bool is_composite;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  *flops = 0;
  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);
  if (is_composite) {
    CeedInt num_suboperators;
    ierr = CeedOperatorGetNumSub(op, &num_suboperators); CeedChk(ierr);
    CeedOperator *sub_operators;
    ierr = CeedOperatorGetSubList(op, &sub_operators); CeedChk(ierr);

    // FLOPs for each suboperator
    for (CeedInt i = 0; i < num_suboperators; i++) {
      CeedSize suboperator_flops;
      ierr = CeedOperatorGetFlopsEstimate(sub_operators[i], &suboperator_flops);
      CeedChk(ierr);
      *flops += suboperator_flops;
    }
  } else {
    CeedInt num_input_fields, num_output_fields;
    CeedOperatorField *input_fields, *output_fields;

    ierr = CeedOperatorGetFields(op, &num_input_fields, &input_fields,
                                 &num_output_fields, &output_fields); CeedChk(ierr);

    CeedInt num_elem = 0;
    ierr = CeedOperatorGetNumElements(op, &num_elem); CeedChk(ierr);
    // Input FLOPs
    for (CeedInt i = 0; i < num_input_fields; i++) {
      if (input_fields[i]->vec == CEED_VECTOR_ACTIVE) {
        CeedSize restr_flops, basis_flops;

        ierr = CeedElemRestrictionGetFlopsEstimate(input_fields[i]->elem_restr,
               CEED_NOTRANSPOSE, &restr_flops); CeedChk(ierr);
        *flops += restr_flops;
        ierr = CeedBasisGetFlopsEstimate(input_fields[i]->basis, CEED_NOTRANSPOSE,
                                         op->qf->input_fields[i]->eval_mode, &basis_flops); CeedChk(ierr);
        *flops += basis_flops * num_elem;
      }
    }
    // QF FLOPs
    CeedInt num_qpts;
    CeedSize qf_flops;
    ierr = CeedOperatorGetNumQuadraturePoints(op, &num_qpts); CeedChk(ierr);
    ierr = CeedQFunctionGetFlopsEstimate(op->qf, &qf_flops); CeedChk(ierr);
    *flops += num_elem * num_qpts * qf_flops;
    // Output FLOPs
    for (CeedInt i = 0; i < num_output_fields; i++) {
      if (output_fields[i]->vec == CEED_VECTOR_ACTIVE) {
        CeedSize restr_flops, basis_flops;

        ierr = CeedElemRestrictionGetFlopsEstimate(output_fields[i]->elem_restr,
               CEED_TRANSPOSE, &restr_flops); CeedChk(ierr);
        *flops += restr_flops;
        ierr = CeedBasisGetFlopsEstimate(output_fields[i]->basis, CEED_TRANSPOSE,
                                         op->qf->output_fields[i]->eval_mode, &basis_flops); CeedChk(ierr);
        *flops += basis_flops * num_elem;
      }
    }
  }

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get label for a registered QFunctionContext field, or `NULL` if no
           field has been registered with this `field_name`.

  @param[in] op            CeedOperator
  @param[in] field_name    Name of field to retrieve label
  @param[out] field_label  Variable to field label

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorContextGetFieldLabel(CeedOperator op,
                                     const char *field_name,
                                     CeedContextFieldLabel *field_label) {
  int ierr;

  bool is_composite;
  ierr = CeedOperatorIsComposite(op, &is_composite); CeedChk(ierr);
  if (is_composite) {
    // Check if composite label already created
    for (CeedInt i=0; i<op->num_context_labels; i++) {
      if (!strcmp(op->context_labels[i]->name, field_name)) {
        *field_label = op->context_labels[i];
        return CEED_ERROR_SUCCESS;
      }
    }

    // Create composite label if needed
    CeedInt num_sub;
    CeedOperator *sub_operators;
    CeedContextFieldLabel new_field_label;

    ierr = CeedCalloc(1, &new_field_label); CeedChk(ierr);
    ierr = CeedOperatorGetNumSub(op, &num_sub); CeedChk(ierr);
    ierr = CeedOperatorGetSubList(op, &sub_operators); CeedChk(ierr);
    ierr = CeedCalloc(num_sub, &new_field_label->sub_labels); CeedChk(ierr);
    new_field_label->num_sub_labels = num_sub;

    bool label_found = false;
    for (CeedInt i=0; i<num_sub; i++) {
      if (sub_operators[i]->qf->ctx) {
        CeedContextFieldLabel new_field_label_i;
        ierr = CeedQFunctionContextGetFieldLabel(sub_operators[i]->qf->ctx, field_name,
               &new_field_label_i); CeedChk(ierr);
        if (new_field_label_i) {
          label_found = true;
          new_field_label->sub_labels[i] = new_field_label_i;
          new_field_label->name = new_field_label_i->name;
          new_field_label->description = new_field_label_i->description;
          if (new_field_label->type &&
              new_field_label->type != new_field_label_i->type) {
            // LCOV_EXCL_START
            ierr = CeedFree(&new_field_label); CeedChk(ierr);
            return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE,
                             "Incompatible field types on sub-operator contexts. "
                             "%s != %s",
                             CeedContextFieldTypes[new_field_label->type],
                             CeedContextFieldTypes[new_field_label_i->type]);
            // LCOV_EXCL_STOP
          } else {
            new_field_label->type = new_field_label_i->type;
          }
          if (new_field_label->num_values != 0 &&
              new_field_label->num_values != new_field_label_i->num_values) {
            // LCOV_EXCL_START
            ierr = CeedFree(&new_field_label); CeedChk(ierr);
            return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE,
                             "Incompatible field number of values on sub-operator"
                             " contexts. %ld != %ld",
                             new_field_label->num_values, new_field_label_i->num_values);
            // LCOV_EXCL_STOP
          } else {
            new_field_label->num_values = new_field_label_i->num_values;
          }
        }
      }
    }
    if (!label_found) {
      // LCOV_EXCL_START
      ierr = CeedFree(&new_field_label->sub_labels); CeedChk(ierr);
      ierr = CeedFree(&new_field_label); CeedChk(ierr);
      *field_label = NULL;
      // LCOV_EXCL_STOP
    } else {
      // Move new composite label to operator
      if (op->num_context_labels == 0) {
        ierr = CeedCalloc(1, &op->context_labels); CeedChk(ierr);
        op->max_context_labels = 1;
      } else if (op->num_context_labels == op->max_context_labels) {
        ierr = CeedRealloc(2*op->num_context_labels, &op->context_labels);
        CeedChk(ierr);
        op->max_context_labels *= 2;
      }
      op->context_labels[op->num_context_labels] = new_field_label;
      *field_label = new_field_label;
      op->num_context_labels++;
    }

    return CEED_ERROR_SUCCESS;
  } else {
    return CeedQFunctionContextGetFieldLabel(op->qf->ctx, field_name, field_label);
  }
}

/**
  @brief Set QFunctionContext field holding a double precision value.
           For composite operators, the value is set in all
           sub-operator QFunctionContexts that have a matching `field_name`.

  @param op          CeedOperator
  @param field_label Label of field to register
  @param values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorContextSetDouble(CeedOperator op,
                                 CeedContextFieldLabel field_label,
                                 double *values) {
  return CeedOperatorContextSetGeneric(op, field_label, CEED_CONTEXT_FIELD_DOUBLE,
                                       values);
}

/**
  @brief Set QFunctionContext field holding an int32 value.
           For composite operators, the value is set in all
           sub-operator QFunctionContexts that have a matching `field_name`.

  @param op          CeedOperator
  @param field_label Label of field to set
  @param values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorContextSetInt32(CeedOperator op,
                                CeedContextFieldLabel field_label,
                                int *values) {
  return CeedOperatorContextSetGeneric(op, field_label, CEED_CONTEXT_FIELD_INT32,
                                       values);
}

/**
  @brief Apply CeedOperator to a vector

  This computes the action of the operator on the specified (active) input,
  yielding its (active) output.  All inputs and outputs must be specified using
  CeedOperatorSetField().

  Note: Calling this function asserts that setup is complete
          and sets the CeedOperator as immutable.

  @param op        CeedOperator to apply
  @param[in] in    CeedVector containing input state or @ref CEED_VECTOR_NONE if
                     there are no active inputs
  @param[out] out  CeedVector to store result of applying operator (must be
                     distinct from @a in) or @ref CEED_VECTOR_NONE if there are no
                     active outputs
  @param request   Address of CeedRequest for non-blocking completion, else
                     @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorApply(CeedOperator op, CeedVector in, CeedVector out,
                      CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  if (op->num_elem)  {
    // Standard Operator
    if (op->Apply) {
      ierr = op->Apply(op, in, out, request); CeedChk(ierr);
    } else {
      // Zero all output vectors
      CeedQFunction qf = op->qf;
      for (CeedInt i=0; i<qf->num_output_fields; i++) {
        CeedVector vec = op->output_fields[i]->vec;
        if (vec == CEED_VECTOR_ACTIVE)
          vec = out;
        if (vec != CEED_VECTOR_NONE) {
          ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
        }
      }
      // Apply
      ierr = op->ApplyAdd(op, in, out, request); CeedChk(ierr);
    }
  } else if (op->is_composite) {
    // Composite Operator
    if (op->ApplyComposite) {
      ierr = op->ApplyComposite(op, in, out, request); CeedChk(ierr);
    } else {
      CeedInt num_suboperators;
      ierr = CeedOperatorGetNumSub(op, &num_suboperators); CeedChk(ierr);
      CeedOperator *sub_operators;
      ierr = CeedOperatorGetSubList(op, &sub_operators); CeedChk(ierr);

      // Zero all output vectors
      if (out != CEED_VECTOR_NONE) {
        ierr = CeedVectorSetValue(out, 0.0); CeedChk(ierr);
      }
      for (CeedInt i=0; i<num_suboperators; i++) {
        for (CeedInt j=0; j<sub_operators[i]->qf->num_output_fields; j++) {
          CeedVector vec = sub_operators[i]->output_fields[j]->vec;
          if (vec != CEED_VECTOR_ACTIVE && vec != CEED_VECTOR_NONE) {
            ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
          }
        }
      }
      // Apply
      for (CeedInt i=0; i<op->num_suboperators; i++) {
        ierr = CeedOperatorApplyAdd(op->sub_operators[i], in, out, request);
        CeedChk(ierr);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply CeedOperator to a vector and add result to output vector

  This computes the action of the operator on the specified (active) input,
  yielding its (active) output.  All inputs and outputs must be specified using
  CeedOperatorSetField().

  @param op        CeedOperator to apply
  @param[in] in    CeedVector containing input state or NULL if there are no
                     active inputs
  @param[out] out  CeedVector to sum in result of applying operator (must be
                     distinct from @a in) or NULL if there are no active outputs
  @param request   Address of CeedRequest for non-blocking completion, else
                     @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorApplyAdd(CeedOperator op, CeedVector in, CeedVector out,
                         CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  if (op->num_elem)  {
    // Standard Operator
    ierr = op->ApplyAdd(op, in, out, request); CeedChk(ierr);
  } else if (op->is_composite) {
    // Composite Operator
    if (op->ApplyAddComposite) {
      ierr = op->ApplyAddComposite(op, in, out, request); CeedChk(ierr);
    } else {
      CeedInt num_suboperators;
      ierr = CeedOperatorGetNumSub(op, &num_suboperators); CeedChk(ierr);
      CeedOperator *sub_operators;
      ierr = CeedOperatorGetSubList(op, &sub_operators); CeedChk(ierr);

      for (CeedInt i=0; i<num_suboperators; i++) {
        ierr = CeedOperatorApplyAdd(sub_operators[i], in, out, request);
        CeedChk(ierr);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedOperator

  @param op  CeedOperator to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorDestroy(CeedOperator *op) {
  int ierr;

  if (!*op || --(*op)->ref_count > 0) return CEED_ERROR_SUCCESS;
  if ((*op)->Destroy) {
    ierr = (*op)->Destroy(*op); CeedChk(ierr);
  }
  ierr = CeedDestroy(&(*op)->ceed); CeedChk(ierr);
  // Free fields
  for (CeedInt i=0; i<(*op)->num_fields; i++)
    if ((*op)->input_fields[i]) {
      if ((*op)->input_fields[i]->elem_restr != CEED_ELEMRESTRICTION_NONE) {
        ierr = CeedElemRestrictionDestroy(&(*op)->input_fields[i]->elem_restr);
        CeedChk(ierr);
      }
      if ((*op)->input_fields[i]->basis != CEED_BASIS_COLLOCATED) {
        ierr = CeedBasisDestroy(&(*op)->input_fields[i]->basis); CeedChk(ierr);
      }
      if ((*op)->input_fields[i]->vec != CEED_VECTOR_ACTIVE &&
          (*op)->input_fields[i]->vec != CEED_VECTOR_NONE ) {
        ierr = CeedVectorDestroy(&(*op)->input_fields[i]->vec); CeedChk(ierr);
      }
      ierr = CeedFree(&(*op)->input_fields[i]->field_name); CeedChk(ierr);
      ierr = CeedFree(&(*op)->input_fields[i]); CeedChk(ierr);
    }
  for (CeedInt i=0; i<(*op)->num_fields; i++)
    if ((*op)->output_fields[i]) {
      ierr = CeedElemRestrictionDestroy(&(*op)->output_fields[i]->elem_restr);
      CeedChk(ierr);
      if ((*op)->output_fields[i]->basis != CEED_BASIS_COLLOCATED) {
        ierr = CeedBasisDestroy(&(*op)->output_fields[i]->basis); CeedChk(ierr);
      }
      if ((*op)->output_fields[i]->vec != CEED_VECTOR_ACTIVE &&
          (*op)->output_fields[i]->vec != CEED_VECTOR_NONE ) {
        ierr = CeedVectorDestroy(&(*op)->output_fields[i]->vec); CeedChk(ierr);
      }
      ierr = CeedFree(&(*op)->output_fields[i]->field_name); CeedChk(ierr);
      ierr = CeedFree(&(*op)->output_fields[i]); CeedChk(ierr);
    }
  // Destroy sub_operators
  for (CeedInt i=0; i<(*op)->num_suboperators; i++)
    if ((*op)->sub_operators[i]) {
      ierr = CeedOperatorDestroy(&(*op)->sub_operators[i]); CeedChk(ierr);
    }
  ierr = CeedQFunctionDestroy(&(*op)->qf); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&(*op)->dqf); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&(*op)->dqfT); CeedChk(ierr);
  // Destroy any composite labels
  for (CeedInt i=0; i<(*op)->num_context_labels; i++) {
    ierr = CeedFree(&(*op)->context_labels[i]->sub_labels); CeedChk(ierr);
    ierr = CeedFree(&(*op)->context_labels[i]); CeedChk(ierr);
  }
  ierr = CeedFree(&(*op)->context_labels); CeedChk(ierr);

  // Destroy fallback
  ierr = CeedOperatorDestroy(&(*op)->op_fallback); CeedChk(ierr);

  // Destroy assembly data
  ierr = CeedQFunctionAssemblyDataDestroy(&(*op)->qf_assembled); CeedChk(ierr);
  ierr = CeedOperatorAssemblyDataDestroy(&(*op)->op_assembled); CeedChk(ierr);

  ierr = CeedFree(&(*op)->input_fields); CeedChk(ierr);
  ierr = CeedFree(&(*op)->output_fields); CeedChk(ierr);
  ierr = CeedFree(&(*op)->sub_operators); CeedChk(ierr);
  ierr = CeedFree(&(*op)->name); CeedChk(ierr);
  ierr = CeedFree(op); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}
