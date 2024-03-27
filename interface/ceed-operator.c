// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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
  @brief Check if a `CeedOperator` Field matches the `CeedQFunction` Field

  @param[in] ceed     `Ceed` object for error handling
  @param[in] qf_field `CeedQFunction` Field matching `CeedOperator` Field
  @param[in] rstr     `CeedOperator` Field `CeedElemRestriction`
  @param[in] basis    `CeedOperator` Field `CeedBasis`

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCheckField(Ceed ceed, CeedQFunctionField qf_field, CeedElemRestriction rstr, CeedBasis basis) {
  const char  *field_name;
  CeedInt      dim = 1, num_comp = 1, q_comp = 1, rstr_num_comp = 1, size;
  CeedEvalMode eval_mode;

  // Field data
  CeedCall(CeedQFunctionFieldGetData(qf_field, &field_name, &size, &eval_mode));

  // Restriction
  CeedCheck((rstr == CEED_ELEMRESTRICTION_NONE) == (eval_mode == CEED_EVAL_WEIGHT), ceed, CEED_ERROR_INCOMPATIBLE,
            "CEED_ELEMRESTRICTION_NONE and CEED_EVAL_WEIGHT must be used together.");
  if (rstr != CEED_ELEMRESTRICTION_NONE) {
    CeedCall(CeedElemRestrictionGetNumComponents(rstr, &rstr_num_comp));
  }
  // Basis
  CeedCheck((basis == CEED_BASIS_NONE) == (eval_mode == CEED_EVAL_NONE), ceed, CEED_ERROR_INCOMPATIBLE,
            "CEED_BASIS_NONE and CEED_EVAL_NONE must be used together.");
  if (basis != CEED_BASIS_NONE) {
    CeedCall(CeedBasisGetDimension(basis, &dim));
    CeedCall(CeedBasisGetNumComponents(basis, &num_comp));
    CeedCall(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &q_comp));
    CeedCheck(rstr == CEED_ELEMRESTRICTION_NONE || rstr_num_comp == num_comp, ceed, CEED_ERROR_DIMENSION,
              "Field '%s' of size %" CeedInt_FMT " and EvalMode %s: CeedElemRestriction has %" CeedInt_FMT
              " components, but CeedBasis has %" CeedInt_FMT " components",
              field_name, size, CeedEvalModes[eval_mode], rstr_num_comp, num_comp);
  }
  // Field size
  switch (eval_mode) {
    case CEED_EVAL_NONE:
      CeedCheck(size == rstr_num_comp, ceed, CEED_ERROR_DIMENSION,
                "Field '%s' of size %" CeedInt_FMT " and EvalMode %s: CeedElemRestriction has %" CeedInt_FMT " components", field_name, size,
                CeedEvalModes[eval_mode], rstr_num_comp);
      break;
    case CEED_EVAL_INTERP:
    case CEED_EVAL_GRAD:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      CeedCheck(size == num_comp * q_comp, ceed, CEED_ERROR_DIMENSION,
                "Field '%s' of size %" CeedInt_FMT " and EvalMode %s: CeedElemRestriction/Basis has %" CeedInt_FMT " components", field_name, size,
                CeedEvalModes[eval_mode], num_comp * q_comp);
      break;
    case CEED_EVAL_WEIGHT:
      // No additional checks required
      break;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a field of a `CeedOperator`

  @param[in] op_field     `CeedOperator` Field to view
  @param[in] qf_field     `CeedQFunction` Field (carries field name)
  @param[in] field_number Number of field being viewed
  @param[in] sub          true indicates sub-operator, which increases indentation; false for top-level operator
  @param[in] input        true for an input field; false for output field
  @param[in] stream       Stream to view to, e.g., `stdout`

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedOperatorFieldView(CeedOperatorField op_field, CeedQFunctionField qf_field, CeedInt field_number, bool sub, bool input, FILE *stream) {
  const char  *pre    = sub ? "  " : "";
  const char  *in_out = input ? "Input" : "Output";
  const char  *field_name;
  CeedInt      size;
  CeedEvalMode eval_mode;
  CeedVector   vec;
  CeedBasis    basis;

  // Field data
  CeedCall(CeedQFunctionFieldGetData(qf_field, &field_name, &size, &eval_mode));
  CeedCall(CeedOperatorFieldGetData(op_field, NULL, NULL, &basis, &vec));

  fprintf(stream,
          "%s    %s field %" CeedInt_FMT
          ":\n"
          "%s      Name: \"%s\"\n",
          pre, in_out, field_number, pre, field_name);
  fprintf(stream, "%s      Size: %" CeedInt_FMT "\n", pre, size);
  fprintf(stream, "%s      EvalMode: %s\n", pre, CeedEvalModes[eval_mode]);
  if (basis == CEED_BASIS_NONE) fprintf(stream, "%s      No basis\n", pre);
  if (vec == CEED_VECTOR_ACTIVE) fprintf(stream, "%s      Active vector\n", pre);
  else if (vec == CEED_VECTOR_NONE) fprintf(stream, "%s      No vector\n", pre);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a single `CeedOperator`

  @param[in] op     `CeedOperator` to view
  @param[in] sub    Boolean flag for sub-operator
  @param[in] stream Stream to write; typically `stdout` or a file

  @return Error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedOperatorSingleView(CeedOperator op, bool sub, FILE *stream) {
  const char         *pre = sub ? "  " : "";
  CeedInt             num_elem, num_qpts, total_fields = 0, num_input_fields, num_output_fields;
  CeedQFunction       qf;
  CeedQFunctionField *qf_input_fields, *qf_output_fields;
  CeedOperatorField  *op_input_fields, *op_output_fields;

  CeedCall(CeedOperatorGetNumElements(op, &num_elem));
  CeedCall(CeedOperatorGetNumQuadraturePoints(op, &num_qpts));
  CeedCall(CeedOperatorGetNumArgs(op, &total_fields));
  CeedCall(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));
  CeedCall(CeedOperatorGetQFunction(op, &qf));
  CeedCall(CeedQFunctionGetFields(qf, NULL, &qf_input_fields, NULL, &qf_output_fields));

  fprintf(stream, "%s  %" CeedInt_FMT " elements with %" CeedInt_FMT " quadrature points each\n", pre, num_elem, num_qpts);
  fprintf(stream, "%s  %" CeedInt_FMT " field%s\n", pre, total_fields, total_fields > 1 ? "s" : "");
  fprintf(stream, "%s  %" CeedInt_FMT " input field%s:\n", pre, num_input_fields, num_input_fields > 1 ? "s" : "");
  for (CeedInt i = 0; i < num_input_fields; i++) {
    CeedCall(CeedOperatorFieldView(op_input_fields[i], qf_input_fields[i], i, sub, 1, stream));
  }
  fprintf(stream, "%s  %" CeedInt_FMT " output field%s:\n", pre, num_output_fields, num_output_fields > 1 ? "s" : "");
  for (CeedInt i = 0; i < num_output_fields; i++) {
    CeedCall(CeedOperatorFieldView(op_output_fields[i], qf_output_fields[i], i, sub, 0, stream));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active input vector `CeedBasis` for a non-composite `CeedOperator`

  @param[in]  op           `CeedOperator` to find active `CeedBasis` for
  @param[out] active_basis `CeedBasis` for active input vector or `NULL` for composite operator

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedOperatorGetActiveBasis(CeedOperator op, CeedBasis *active_basis) {
  CeedCall(CeedOperatorGetActiveBases(op, active_basis, NULL));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active input and output vector `CeedBasis` for a non-composite `CeedOperator`

  @param[in]  op                  `CeedOperator` to find active `CeedBasis` for
  @param[out] active_input_basis  `CeedBasis` for active input vector or `NULL` for composite operator
  @param[out] active_output_basis `CeedBasis` for active output vector or `NULL` for composite operator

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedOperatorGetActiveBases(CeedOperator op, CeedBasis *active_input_basis, CeedBasis *active_output_basis) {
  bool               is_composite;
  CeedInt            num_input_fields, num_output_fields;
  Ceed               ceed;
  CeedOperatorField *op_input_fields, *op_output_fields;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCall(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));

  if (active_input_basis) {
    *active_input_basis = NULL;
    if (!is_composite) {
      for (CeedInt i = 0; i < num_input_fields; i++) {
        CeedVector vec;

        CeedCall(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        if (vec == CEED_VECTOR_ACTIVE) {
          CeedBasis basis;

          CeedCall(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
          CeedCheck(!*active_input_basis || *active_input_basis == basis, ceed, CEED_ERROR_MINOR, "Multiple active input CeedBases found");
          *active_input_basis = basis;
        }
      }
      CeedCheck(*active_input_basis, ceed, CEED_ERROR_INCOMPLETE, "No active input CeedBasis found");
    }
  }
  if (active_output_basis) {
    *active_output_basis = NULL;
    if (!is_composite) {
      for (CeedInt i = 0; i < num_output_fields; i++) {
        CeedVector vec;

        CeedCall(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
        if (vec == CEED_VECTOR_ACTIVE) {
          CeedBasis basis;

          CeedCall(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
          CeedCheck(!*active_output_basis || *active_output_basis == basis, ceed, CEED_ERROR_MINOR, "Multiple active output CeedBases found");
          *active_output_basis = basis;
        }
      }
      CeedCheck(*active_output_basis, ceed, CEED_ERROR_INCOMPLETE, "No active output CeedBasis found");
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active vector `CeedElemRestriction` for a non-composite `CeedOperator`

  @param[in]  op          `CeedOperator` to find active `CeedElemRestriction` for
  @param[out] active_rstr `CeedElemRestriction` for active input vector or NULL for composite operator

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedOperatorGetActiveElemRestriction(CeedOperator op, CeedElemRestriction *active_rstr) {
  CeedCall(CeedOperatorGetActiveElemRestrictions(op, active_rstr, NULL));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active input and output vector `CeedElemRestriction` for a non-composite `CeedOperator`

  @param[in]  op                 `CeedOperator` to find active `CeedElemRestriction` for
  @param[out] active_input_rstr  `CeedElemRestriction` for active input vector or NULL for composite operator
  @param[out] active_output_rstr `CeedElemRestriction` for active output vector or NULL for composite operator

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedOperatorGetActiveElemRestrictions(CeedOperator op, CeedElemRestriction *active_input_rstr, CeedElemRestriction *active_output_rstr) {
  bool               is_composite;
  CeedInt            num_input_fields, num_output_fields;
  Ceed               ceed;
  CeedOperatorField *op_input_fields, *op_output_fields;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCall(CeedOperatorGetFields(op, &num_input_fields, &op_input_fields, &num_output_fields, &op_output_fields));

  if (active_input_rstr) {
    *active_input_rstr = NULL;
    if (!is_composite) {
      for (CeedInt i = 0; i < num_input_fields; i++) {
        CeedVector vec;

        CeedCall(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
        if (vec == CEED_VECTOR_ACTIVE) {
          CeedElemRestriction rstr;

          CeedCall(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &rstr));
          CeedCheck(!*active_input_rstr || *active_input_rstr == rstr, ceed, CEED_ERROR_MINOR, "Multiple active input CeedElemRestrictions found");
          *active_input_rstr = rstr;
        }
      }
      CeedCheck(*active_input_rstr, ceed, CEED_ERROR_INCOMPLETE, "No active input CeedElemRestriction found");
    }
  }
  if (active_output_rstr) {
    *active_output_rstr = NULL;
    if (!is_composite) {
      for (CeedInt i = 0; i < num_output_fields; i++) {
        CeedVector vec;

        CeedCall(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
        if (vec == CEED_VECTOR_ACTIVE) {
          CeedElemRestriction rstr;

          CeedCall(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &rstr));
          CeedCheck(!*active_output_rstr || *active_output_rstr == rstr, ceed, CEED_ERROR_MINOR, "Multiple active output CeedElemRestrictions found");
          *active_output_rstr = rstr;
        }
      }
      CeedCheck(*active_output_rstr, ceed, CEED_ERROR_INCOMPLETE, "No active output CeedElemRestriction found");
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set `CeedQFunctionContext` field values of the specified type.

  For composite operators, the value is set in all sub-operator `CeedQFunctionContext` that have a matching `field_name`.
  A non-zero error code is returned for single operators that do not have a matching field of the same type or composite operators that do not have any field of a matching type.

  @param[in,out] op          `CeedOperator`
  @param[in]     field_label Label of field to set
  @param[in]     field_type  Type of field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
static int CeedOperatorContextSetGeneric(CeedOperator op, CeedContextFieldLabel field_label, CeedContextFieldType field_type, void *values) {
  bool is_composite = false;
  Ceed ceed;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCheck(field_label, ceed, CEED_ERROR_UNSUPPORTED, "Invalid field label");

  // Check if field_label and op correspond
  if (field_label->from_op) {
    CeedInt index = -1;

    for (CeedInt i = 0; i < op->num_context_labels; i++) {
      if (op->context_labels[i] == field_label) index = i;
    }
    CeedCheck(index != -1, ceed, CEED_ERROR_UNSUPPORTED, "ContextFieldLabel does not correspond to the operator");
  }

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt       num_sub;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_sub));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    CeedCheck(num_sub == field_label->num_sub_labels, ceed, CEED_ERROR_UNSUPPORTED, "Composite operator modified after ContextFieldLabel created");

    for (CeedInt i = 0; i < num_sub; i++) {
      CeedQFunction        qf;
      CeedQFunctionContext ctx;

      CeedCall(CeedOperatorGetQFunction(sub_operators[i], &qf));
      CeedCall(CeedQFunctionGetContext(qf, &ctx));
      // Try every sub-operator, ok if some sub-operators do not have field
      if (field_label->sub_labels[i] && ctx) {
        CeedCall(CeedQFunctionContextSetGeneric(ctx, field_label->sub_labels[i], field_type, values));
      }
    }
  } else {
    CeedQFunction        qf;
    CeedQFunctionContext ctx;

    CeedCall(CeedOperatorGetQFunction(op, &qf));
    CeedCall(CeedQFunctionGetContext(qf, &ctx));
    CeedCheck(ctx, ceed, CEED_ERROR_UNSUPPORTED, "QFunction does not have context data");
    CeedCall(CeedQFunctionContextSetGeneric(ctx, field_label, field_type, values));
  }
  CeedCall(CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(op, true));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get `CeedQFunctionContext` field values of the specified type, read-only.

  For composite operators, the values retrieved are for the first sub-operator `CeedQFunctionContext` that have a matching `field_name`.
  A non-zero error code is returned for single operators that do not have a matching field of the same type or composite operators that do not have any field of a matching type.

  @param[in,out] op          `CeedOperator`
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
  Ceed ceed;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCheck(field_label, ceed, CEED_ERROR_UNSUPPORTED, "Invalid field label");

  *(void **)values = NULL;
  *num_values      = 0;

  // Check if field_label and op correspond
  if (field_label->from_op) {
    CeedInt index = -1;

    for (CeedInt i = 0; i < op->num_context_labels; i++) {
      if (op->context_labels[i] == field_label) index = i;
    }
    CeedCheck(index != -1, ceed, CEED_ERROR_UNSUPPORTED, "ContextFieldLabel does not correspond to the operator");
  }

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt       num_sub;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_sub));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    CeedCheck(num_sub == field_label->num_sub_labels, ceed, CEED_ERROR_UNSUPPORTED, "Composite operator modified after ContextFieldLabel created");

    for (CeedInt i = 0; i < num_sub; i++) {
      CeedQFunction        qf;
      CeedQFunctionContext ctx;

      CeedCall(CeedOperatorGetQFunction(sub_operators[i], &qf));
      CeedCall(CeedQFunctionGetContext(qf, &ctx));
      // Try every sub-operator, ok if some sub-operators do not have field
      if (field_label->sub_labels[i] && ctx) {
        CeedCall(CeedQFunctionContextGetGenericRead(ctx, field_label->sub_labels[i], field_type, num_values, values));
        return CEED_ERROR_SUCCESS;
      }
    }
  } else {
    CeedQFunction        qf;
    CeedQFunctionContext ctx;

    CeedCall(CeedOperatorGetQFunction(op, &qf));
    CeedCall(CeedQFunctionGetContext(qf, &ctx));
    CeedCheck(ctx, ceed, CEED_ERROR_UNSUPPORTED, "QFunction does not have context data");
    CeedCall(CeedQFunctionContextGetGenericRead(ctx, field_label, field_type, num_values, values));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore `CeedQFunctionContext` field values of the specified type, read-only.

  For composite operators, the values restored are for the first sub-operator `CeedQFunctionContext` that have a matching `field_name`.
  A non-zero error code is returned for single operators that do not have a matching field of the same type or composite operators that do not have any field of a matching type.

  @param[in,out] op          `CeedOperator`
  @param[in]     field_label Label of field to set
  @param[in]     field_type  Type of field to set
  @param[in]     values      Values array to restore

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
static int CeedOperatorContextRestoreGenericRead(CeedOperator op, CeedContextFieldLabel field_label, CeedContextFieldType field_type, void *values) {
  bool is_composite = false;
  Ceed ceed;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCheck(field_label, ceed, CEED_ERROR_UNSUPPORTED, "Invalid field label");

  // Check if field_label and op correspond
  if (field_label->from_op) {
    CeedInt index = -1;

    for (CeedInt i = 0; i < op->num_context_labels; i++) {
      if (op->context_labels[i] == field_label) index = i;
    }
    CeedCheck(index != -1, ceed, CEED_ERROR_UNSUPPORTED, "ContextFieldLabel does not correspond to the operator");
  }

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt       num_sub;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_sub));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    CeedCheck(num_sub == field_label->num_sub_labels, ceed, CEED_ERROR_UNSUPPORTED, "Composite operator modified after ContextFieldLabel created");

    for (CeedInt i = 0; i < num_sub; i++) {
      CeedQFunction        qf;
      CeedQFunctionContext ctx;

      CeedCall(CeedOperatorGetQFunction(sub_operators[i], &qf));
      CeedCall(CeedQFunctionGetContext(qf, &ctx));
      // Try every sub-operator, ok if some sub-operators do not have field
      if (field_label->sub_labels[i] && ctx) {
        CeedCall(CeedQFunctionContextRestoreGenericRead(ctx, field_label->sub_labels[i], field_type, values));
        return CEED_ERROR_SUCCESS;
      }
    }
  } else {
    CeedQFunction        qf;
    CeedQFunctionContext ctx;

    CeedCall(CeedOperatorGetQFunction(op, &qf));
    CeedCall(CeedQFunctionGetContext(qf, &ctx));
    CeedCheck(ctx, ceed, CEED_ERROR_UNSUPPORTED, "QFunction does not have context data");
    CeedCall(CeedQFunctionContextRestoreGenericRead(ctx, field_label, field_type, values));
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
  @brief Get the number of arguments associated with a `CeedOperator`

  @param[in]  op        `CeedOperator`
  @param[out] num_args  Variable to store vector number of arguments

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorGetNumArgs(CeedOperator op, CeedInt *num_args) {
  bool is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, CeedOperatorReturnCeed(op), CEED_ERROR_MINOR, "Not defined for composite operators");
  *num_args = op->num_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the tensor product status of all bases for a `CeedOperator`.

  `has_tensor_bases` is only set to `true` if every field uses a tensor-product basis.

  @param[in]  op               `CeedOperator`
  @param[out] has_tensor_bases Variable to store tensor bases status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorHasTensorBases(CeedOperator op, bool *has_tensor_bases) {
  CeedInt            num_inputs, num_outputs;
  CeedOperatorField *input_fields, *output_fields;

  CeedCall(CeedOperatorGetFields(op, &num_inputs, &input_fields, &num_outputs, &output_fields));
  *has_tensor_bases = true;
  for (CeedInt i = 0; i < num_inputs; i++) {
    bool      is_tensor;
    CeedBasis basis;

    CeedCall(CeedOperatorFieldGetBasis(input_fields[i], &basis));
    if (basis != CEED_BASIS_NONE) {
      CeedCall(CeedBasisIsTensor(basis, &is_tensor));
      *has_tensor_bases &= is_tensor;
    }
  }
  for (CeedInt i = 0; i < num_outputs; i++) {
    bool      is_tensor;
    CeedBasis basis;

    CeedCall(CeedOperatorFieldGetBasis(output_fields[i], &basis));
    if (basis != CEED_BASIS_NONE) {
      CeedCall(CeedBasisIsTensor(basis, &is_tensor));
      *has_tensor_bases &= is_tensor;
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get a boolean value indicating if the `CeedOperator` is immutable

  @param[in]  op           `CeedOperator`
  @param[out] is_immutable Variable to store immutability status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorIsImmutable(CeedOperator op, bool *is_immutable) {
  *is_immutable = op->is_immutable;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the setup status of a `CeedOperator`

  @param[in]  op            `CeedOperator`
  @param[out] is_setup_done Variable to store setup status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorIsSetupDone(CeedOperator op, bool *is_setup_done) {
  *is_setup_done = op->is_backend_setup;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `CeedQFunction` associated with a `CeedOperator`

  @param[in]  op `CeedOperator`
  @param[out] qf Variable to store `CeedQFunction`

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorGetQFunction(CeedOperator op, CeedQFunction *qf) {
  bool is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, CeedOperatorReturnCeed(op), CEED_ERROR_MINOR, "Not defined for composite operator");
  *qf = op->qf;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get a boolean value indicating if the `CeedOperator` is composite

  @param[in]  op           `CeedOperator`
  @param[out] is_composite Variable to store composite status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorIsComposite(CeedOperator op, bool *is_composite) {
  *is_composite = op->is_composite;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a `CeedOperator`

  @param[in]  op   `CeedOperator`
  @param[out] data Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorGetData(CeedOperator op, void *data) {
  *(void **)data = op->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a `CeedOperator`

  @param[in,out] op   `CeedOperator`
  @param[in]     data Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorSetData(CeedOperator op, void *data) {
  op->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a `CeedOperator`

  @param[in,out] op `CeedOperator` to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorReference(CeedOperator op) {
  op->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the setup flag of a `CeedOperator` to `true`

  @param[in,out] op `CeedOperator`

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
  @brief Create a `CeedOperator` and associate a `CeedQFunction`.

  A `CeedBasis` and `CeedElemRestriction` can be associated with `CeedQFunction` fields with @ref CeedOperatorSetField().

  @param[in]  ceed `Ceed` object used to create the `CeedOperator`
  @param[in]  qf   `CeedQFunction` defining the action of the operator at quadrature points
  @param[in]  dqf  `CeedQFunction` defining the action of the Jacobian of `qf` (or @ref CEED_QFUNCTION_NONE)
  @param[in]  dqfT `CeedQFunction` defining the action of the transpose of the Jacobian of `qf` (or @ref CEED_QFUNCTION_NONE)
  @param[out] op   Address of the variable where the newly created `CeedOperator` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedOperatorCreate(Ceed ceed, CeedQFunction qf, CeedQFunction dqf, CeedQFunction dqfT, CeedOperator *op) {
  if (!ceed->OperatorCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Operator"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support CeedOperatorCreate");
    CeedCall(CeedOperatorCreate(delegate, qf, dqf, dqfT, op));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(qf && qf != CEED_QFUNCTION_NONE, ceed, CEED_ERROR_MINOR, "Operator must have a valid CeedQFunction.");

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
  @brief Create a `CeedOperator` for evaluation at evaluation at arbitrary points in each element.

  A `CeedBasis` and `CeedElemRestriction` can be associated with `CeedQFunction` fields with `CeedOperator` SetField.
  The locations of each point are set with @ref CeedOperatorAtPointsSetPoints().

  @param[in]  ceed `Ceed` object used to create the `CeedOperator`
  @param[in]  qf   `CeedQFunction` defining the action of the operator at quadrature points
  @param[in]  dqf  `CeedQFunction` defining the action of the Jacobian of @a qf (or @ref CEED_QFUNCTION_NONE)
  @param[in]  dqfT `CeedQFunction` defining the action of the transpose of the Jacobian of @a qf (or @ref CEED_QFUNCTION_NONE)
  @param[out] op   Address of the variable where the newly created CeedOperator will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedOperatorCreateAtPoints(Ceed ceed, CeedQFunction qf, CeedQFunction dqf, CeedQFunction dqfT, CeedOperator *op) {
  if (!ceed->OperatorCreateAtPoints) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Operator"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support CeedOperatorCreateAtPoints");
    CeedCall(CeedOperatorCreateAtPoints(delegate, qf, dqf, dqfT, op));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(qf && qf != CEED_QFUNCTION_NONE, ceed, CEED_ERROR_MINOR, "Operator must have a valid CeedQFunction.");

  CeedCall(CeedCalloc(1, op));
  CeedCall(CeedReferenceCopy(ceed, &(*op)->ceed));
  (*op)->ref_count    = 1;
  (*op)->is_at_points = true;
  (*op)->input_size   = -1;
  (*op)->output_size  = -1;
  CeedCall(CeedQFunctionReferenceCopy(qf, &(*op)->qf));
  if (dqf && dqf != CEED_QFUNCTION_NONE) CeedCall(CeedQFunctionReferenceCopy(dqf, &(*op)->dqf));
  if (dqfT && dqfT != CEED_QFUNCTION_NONE) CeedCall(CeedQFunctionReferenceCopy(dqfT, &(*op)->dqfT));
  CeedCall(CeedQFunctionAssemblyDataCreate(ceed, &(*op)->qf_assembled));
  CeedCall(CeedCalloc(CEED_FIELD_MAX, &(*op)->input_fields));
  CeedCall(CeedCalloc(CEED_FIELD_MAX, &(*op)->output_fields));
  CeedCall(ceed->OperatorCreateAtPoints(*op));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a composite `CeedOperator` that composes the action of several `CeedOperator`

  @param[in]  ceed `Ceed` object used to create the `CeedOperator`
  @param[out] op   Address of the variable where the newly created composite `CeedOperator` will be stored

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
  @brief Copy the pointer to a `CeedOperator`.

  Both pointers should be destroyed with @ref CeedOperatorDestroy().

  Note: If the value of `*op_copy` passed to this function is non-`NULL`, then it is assumed that `*op_copy` is a pointer to a `CeedOperator`.
        This `CeedOperator` will be destroyed if `*op_copy` is the only reference to this `CeedOperator`.

  @param[in]     op      `CeedOperator` to copy reference to
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
  @brief Provide a field to a `CeedOperator` for use by its `CeedQFunction`.

  This function is used to specify both active and passive fields to a `CeedOperator`.
  For passive fields, a `CeedVector` `vec` must be provided.
  Passive fields can inputs or outputs (updated in-place when operator is applied).

  Active fields must be specified using this function, but their data (in a `CeedVector`) is passed in @ref CeedOperatorApply().
  There can be at most one active input `CeedVector` and at most one active output@ref  CeedVector passed to @ref CeedOperatorApply().

  The number of quadrature points must agree across all points.
  When using @ref CEED_BASIS_NONE, the number of quadrature points is determined by the element size of `rstr`.

  @param[in,out] op         `CeedOperator` on which to provide the field
  @param[in]     field_name Name of the field (to be matched with the name used by `CeedQFunction`)
  @param[in]     rstr       `CeedElemRestriction`
  @param[in]     basis      `CeedBasis` in which the field resides or @ref CEED_BASIS_NONE if collocated with quadrature points
  @param[in]     vec        `CeedVector` to be used by CeedOperator or @ref CEED_VECTOR_ACTIVE if field is active or @ref CEED_VECTOR_NONE if using @ref CEED_EVAL_WEIGHT in the `CeedQFunction`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetField(CeedOperator op, const char *field_name, CeedElemRestriction rstr, CeedBasis basis, CeedVector vec) {
  bool               is_input = true, is_at_points, is_composite, is_immutable;
  CeedInt            num_elem = 0, num_qpts = 0, num_input_fields, num_output_fields;
  Ceed               ceed;
  CeedQFunction      qf;
  CeedQFunctionField qf_field, *qf_input_fields, *qf_output_fields;
  CeedOperatorField *op_field;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorIsAtPoints(op, &is_at_points));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCall(CeedOperatorIsImmutable(op, &is_immutable));
  CeedCheck(!is_composite, ceed, CEED_ERROR_INCOMPATIBLE, "Cannot add field to composite operator.");
  CeedCheck(!is_immutable, ceed, CEED_ERROR_MAJOR, "Operator cannot be changed after set as immutable");
  CeedCheck(rstr, ceed, CEED_ERROR_INCOMPATIBLE, "CeedElemRestriction rstr for field \"%s\" must be non-NULL.", field_name);
  CeedCheck(basis, ceed, CEED_ERROR_INCOMPATIBLE, "CeedBasis basis for field \"%s\" must be non-NULL.", field_name);
  CeedCheck(vec, ceed, CEED_ERROR_INCOMPATIBLE, "CeedVector vec for field \"%s\" must be non-NULL.", field_name);

  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCheck(rstr == CEED_ELEMRESTRICTION_NONE || !op->has_restriction || num_elem == op->num_elem, ceed, CEED_ERROR_DIMENSION,
            "CeedElemRestriction with %" CeedInt_FMT " elements incompatible with prior %" CeedInt_FMT " elements", num_elem, op->num_elem);
  {
    CeedRestrictionType rstr_type;

    CeedCall(CeedElemRestrictionGetType(rstr, &rstr_type));
    if (rstr_type == CEED_RESTRICTION_POINTS) {
      CeedCheck(is_at_points, ceed, CEED_ERROR_UNSUPPORTED, "CeedElemRestriction AtPoints not supported for standard operator fields");
      CeedCheck(basis == CEED_BASIS_NONE, ceed, CEED_ERROR_UNSUPPORTED, "CeedElemRestriction AtPoints must be used with CEED_BASIS_NONE");
      if (!op->first_points_rstr) {
        CeedCall(CeedElemRestrictionReferenceCopy(rstr, &op->first_points_rstr));
      } else {
        bool are_compatible;

        CeedCall(CeedElemRestrictionAtPointsAreCompatible(op->first_points_rstr, rstr, &are_compatible));
        CeedCheck(are_compatible, ceed, CEED_ERROR_INCOMPATIBLE,
                  "CeedElemRestriction must have compatible offsets with previously set CeedElemRestriction");
      }
    }
  }

  if (basis == CEED_BASIS_NONE) CeedCall(CeedElemRestrictionGetElementSize(rstr, &num_qpts));
  else CeedCall(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCheck(op->num_qpts == 0 || num_qpts == op->num_qpts, ceed, CEED_ERROR_DIMENSION,
            "%s must correspond to the same number of quadrature points as previously added CeedBases. Found %" CeedInt_FMT
            " quadrature points but expected %" CeedInt_FMT " quadrature points.",
            basis == CEED_BASIS_NONE ? "CeedElemRestriction" : "CeedBasis", num_qpts, op->num_qpts);

  CeedCall(CeedOperatorGetQFunction(op, &qf));
  CeedCall(CeedQFunctionGetFields(qf, &num_input_fields, &qf_input_fields, &num_output_fields, &qf_output_fields));
  for (CeedInt i = 0; i < num_input_fields; i++) {
    const char *qf_field_name;

    CeedCall(CeedQFunctionFieldGetName(qf_input_fields[i], &qf_field_name));
    if (!strcmp(field_name, qf_field_name)) {
      qf_field = qf_input_fields[i];
      op_field = &op->input_fields[i];
      goto found;
    }
  }
  is_input = false;
  for (CeedInt i = 0; i < num_output_fields; i++) {
    const char *qf_field_name;

    CeedCall(CeedQFunctionFieldGetName(qf_output_fields[i], &qf_field_name));
    if (!strcmp(field_name, qf_field_name)) {
      qf_field = qf_output_fields[i];
      op_field = &op->output_fields[i];
      goto found;
    }
  }
  // LCOV_EXCL_START
  return CeedError(ceed, CEED_ERROR_INCOMPLETE, "CeedQFunction has no knowledge of field '%s'", field_name);
  // LCOV_EXCL_STOP
found:
  CeedCall(CeedOperatorCheckField(ceed, qf_field, rstr, basis));
  CeedCall(CeedCalloc(1, op_field));

  if (vec == CEED_VECTOR_ACTIVE) {
    CeedSize l_size;

    CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
    if (is_input) {
      if (op->input_size == -1) op->input_size = l_size;
      CeedCheck(l_size == op->input_size, ceed, CEED_ERROR_INCOMPATIBLE,
                "LVector size %" CeedSize_FMT " does not match previous size %" CeedSize_FMT "", l_size, op->input_size);
    } else {
      if (op->output_size == -1) op->output_size = l_size;
      CeedCheck(l_size == op->output_size, ceed, CEED_ERROR_INCOMPATIBLE,
                "LVector size %" CeedSize_FMT " does not match previous size %" CeedSize_FMT "", l_size, op->output_size);
    }
  }

  CeedCall(CeedVectorReferenceCopy(vec, &(*op_field)->vec));
  CeedCall(CeedElemRestrictionReferenceCopy(rstr, &(*op_field)->elem_rstr));
  if (rstr != CEED_ELEMRESTRICTION_NONE && !op->has_restriction) {
    op->num_elem        = num_elem;
    op->has_restriction = true;  // Restriction set, but num_elem may be 0
  }
  CeedCall(CeedBasisReferenceCopy(basis, &(*op_field)->basis));
  if (op->num_qpts == 0 && !is_at_points) op->num_qpts = num_qpts;  // no consistent number of qpts for OperatorAtPoints
  op->num_fields += 1;
  CeedCall(CeedStringAllocCopy(field_name, (char **)&(*op_field)->field_name));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `CeedOperator` Field of a `CeedOperator`.

  Note: Calling this function asserts that setup is complete and sets the `CeedOperator` as immutable.

  @param[in]  op                `CeedOperator`
  @param[out] num_input_fields  Variable to store number of input fields
  @param[out] input_fields      Variable to store input fields
  @param[out] num_output_fields Variable to store number of output fields
  @param[out] output_fields     Variable to store output fields

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetFields(CeedOperator op, CeedInt *num_input_fields, CeedOperatorField **input_fields, CeedInt *num_output_fields,
                          CeedOperatorField **output_fields) {
  bool          is_composite;
  CeedQFunction qf;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, CeedOperatorReturnCeed(op), CEED_ERROR_MINOR, "Not defined for composite operator");
  CeedCall(CeedOperatorCheckReady(op));

  CeedCall(CeedOperatorGetQFunction(op, &qf));
  CeedCall(CeedQFunctionGetFields(qf, num_input_fields, NULL, num_output_fields, NULL));
  if (input_fields) *input_fields = op->input_fields;
  if (output_fields) *output_fields = op->output_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the arbitrary points in each element for a `CeedOperator` at points.

  Note: Calling this function asserts that setup is complete and sets the `CeedOperator` as immutable.

  @param[in,out] op           `CeedOperator` at points
  @param[in]     rstr_points  `CeedElemRestriction` for the coordinates of each point by element
  @param[in]     point_coords `CeedVector` holding coordinates of each point

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorAtPointsSetPoints(CeedOperator op, CeedElemRestriction rstr_points, CeedVector point_coords) {
  bool is_at_points, is_immutable;
  Ceed ceed;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorIsAtPoints(op, &is_at_points));
  CeedCall(CeedOperatorIsImmutable(op, &is_immutable));
  CeedCheck(is_at_points, ceed, CEED_ERROR_MINOR, "Only defined for operator at points");
  CeedCheck(!is_immutable, ceed, CEED_ERROR_MAJOR, "Operator cannot be changed after set as immutable");

  if (!op->first_points_rstr) {
    CeedCall(CeedElemRestrictionReferenceCopy(rstr_points, &op->first_points_rstr));
  } else {
    bool are_compatible;

    CeedCall(CeedElemRestrictionAtPointsAreCompatible(op->first_points_rstr, rstr_points, &are_compatible));
    CeedCheck(are_compatible, ceed, CEED_ERROR_INCOMPATIBLE,
              "CeedElemRestriction must have compatible offsets with previously set field CeedElemRestriction");
  }

  CeedCall(CeedElemRestrictionReferenceCopy(rstr_points, &op->rstr_points));
  CeedCall(CeedVectorReferenceCopy(point_coords, &op->point_coords));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get a boolean value indicating if the `CeedOperator` was created with `CeedOperatorCreateAtPoints`
    
  @param[in]  op           `CeedOperator`
  @param[out] is_at_points Variable to store at points status
  
  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorIsAtPoints(CeedOperator op, bool *is_at_points) {
  *is_at_points = op->is_at_points;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the arbitrary points in each element for a `CeedOperator` at points.

  Note: Calling this function asserts that setup is complete and sets the `CeedOperator` as immutable.

  @param[in]  op           `CeedOperator` at points
  @param[out] rstr_points  Variable to hold `CeedElemRestriction` for the coordinates of each point by element
  @param[out] point_coords Variable to hold `CeedVector` holding coordinates of each point

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorAtPointsGetPoints(CeedOperator op, CeedElemRestriction *rstr_points, CeedVector *point_coords) {
  bool is_at_points;

  CeedCall(CeedOperatorIsAtPoints(op, &is_at_points));
  CeedCheck(is_at_points, CeedOperatorReturnCeed(op), CEED_ERROR_MINOR, "Only defined for operator at points");
  CeedCall(CeedOperatorCheckReady(op));

  if (rstr_points) CeedCall(CeedElemRestrictionReferenceCopy(op->rstr_points, rstr_points));
  if (point_coords) CeedCall(CeedVectorReferenceCopy(op->point_coords, point_coords));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get a `CeedOperator` Field of a `CeedOperator` from its name.

  `op_field` is set to `NULL` if the field is not found.

  Note: Calling this function asserts that setup is complete and sets the `CeedOperator` as immutable.

  @param[in]  op         `CeedOperator`
  @param[in]  field_name Name of desired `CeedOperator` Field
  @param[out] op_field   `CeedOperator` Field corresponding to the name

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetFieldByName(CeedOperator op, const char *field_name, CeedOperatorField *op_field) {
  const char        *name;
  CeedInt            num_input_fields, num_output_fields;
  CeedOperatorField *input_fields, *output_fields;

  *op_field = NULL;
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
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the name of a `CeedOperator` Field

  @param[in]  op_field   `CeedOperator` Field
  @param[out] field_name Variable to store the field name

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetName(CeedOperatorField op_field, const char **field_name) {
  *field_name = op_field->field_name;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `CeedElemRestriction` of a `CeedOperator` Field

  @param[in]  op_field `CeedOperator` Field
  @param[out] rstr     Variable to store `CeedElemRestriction`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetElemRestriction(CeedOperatorField op_field, CeedElemRestriction *rstr) {
  *rstr = op_field->elem_rstr;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `CeedBasis` of a `CeedOperator` Field

  @param[in]  op_field `CeedOperator` Field
  @param[out] basis    Variable to store `CeedBasis`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetBasis(CeedOperatorField op_field, CeedBasis *basis) {
  *basis = op_field->basis;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `CeedVector` of a `CeedOperator` Field

  @param[in]  op_field `CeedOperator` Field
  @param[out] vec      Variable to store `CeedVector`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetVector(CeedOperatorField op_field, CeedVector *vec) {
  *vec = op_field->vec;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the data of a `CeedOperator` Field.

  Any arguments set as `NULL` are ignored.

  @param[in]  op_field   `CeedOperator` Field
  @param[out] field_name Variable to store the field name
  @param[out] rstr       Variable to store `CeedElemRestriction`
  @param[out] basis      Variable to store `CeedBasis`
  @param[out] vec        Variable to store `CeedVector`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorFieldGetData(CeedOperatorField op_field, const char **field_name, CeedElemRestriction *rstr, CeedBasis *basis, CeedVector *vec) {
  if (field_name) CeedCall(CeedOperatorFieldGetName(op_field, field_name));
  if (rstr) CeedCall(CeedOperatorFieldGetElemRestriction(op_field, rstr));
  if (basis) CeedCall(CeedOperatorFieldGetBasis(op_field, basis));
  if (vec) CeedCall(CeedOperatorFieldGetVector(op_field, vec));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add a sub-operator to a composite `CeedOperator`

  @param[in,out] composite_op Composite `CeedOperator`
  @param[in]     sub_op       Sub-operator `CeedOperator`

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedCompositeOperatorAddSub(CeedOperator composite_op, CeedOperator sub_op) {
  bool is_immutable;
  Ceed ceed;

  CeedCall(CeedOperatorGetCeed(composite_op, &ceed));
  CeedCheck(composite_op->is_composite, ceed, CEED_ERROR_MINOR, "CeedOperator is not a composite operator");
  CeedCheck(composite_op->num_suboperators < CEED_COMPOSITE_MAX, ceed, CEED_ERROR_UNSUPPORTED, "Cannot add additional sub-operators");
  CeedCall(CeedOperatorIsImmutable(composite_op, &is_immutable));
  CeedCheck(!is_immutable, ceed, CEED_ERROR_MAJOR, "Operator cannot be changed after set as immutable");

  {
    CeedSize input_size, output_size;

    CeedCall(CeedOperatorGetActiveVectorLengths(sub_op, &input_size, &output_size));
    if (composite_op->input_size == -1) composite_op->input_size = input_size;
    if (composite_op->output_size == -1) composite_op->output_size = output_size;
    // Note, a size of -1 means no active vector restriction set, so no incompatibility
    CeedCheck((input_size == -1 || input_size == composite_op->input_size) && (output_size == -1 || output_size == composite_op->output_size), ceed,
              CEED_ERROR_MAJOR,
              "Sub-operators must have compatible dimensions; composite operator of shape (%" CeedSize_FMT ", %" CeedSize_FMT
              ") not compatible with sub-operator of "
              "shape (%" CeedSize_FMT ", %" CeedSize_FMT ")",
              composite_op->input_size, composite_op->output_size, input_size, output_size);
  }

  composite_op->sub_operators[composite_op->num_suboperators] = sub_op;
  CeedCall(CeedOperatorReference(sub_op));
  composite_op->num_suboperators++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of sub-operators associated with a `CeedOperator`

  @param[in]  op               `CeedOperator`
  @param[out] num_suboperators Variable to store number of sub-operators

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedCompositeOperatorGetNumSub(CeedOperator op, CeedInt *num_suboperators) {
  bool is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(is_composite, CeedOperatorReturnCeed(op), CEED_ERROR_MINOR, "Only defined for a composite operator");
  *num_suboperators = op->num_suboperators;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the list of sub-operators associated with a `CeedOperator`

  @param[in]  op             `CeedOperator`
  @param[out] sub_operators  Variable to store list of sub-operators

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedCompositeOperatorGetSubList(CeedOperator op, CeedOperator **sub_operators) {
  bool is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(is_composite, CeedOperatorReturnCeed(op), CEED_ERROR_MINOR, "Only defined for a composite operator");
  *sub_operators = op->sub_operators;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check if a `CeedOperator` is ready to be used.

  @param[in] op `CeedOperator` to check

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorCheckReady(CeedOperator op) {
  bool          is_at_points, is_composite;
  Ceed          ceed;
  CeedQFunction qf = NULL;

  if (op->is_interface_setup) return CEED_ERROR_SUCCESS;

  CeedCall(CeedOperatorGetCeed(op, &ceed));
  CeedCall(CeedOperatorIsAtPoints(op, &is_at_points));
  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (!is_composite) CeedCall(CeedOperatorGetQFunction(op, &qf));
  if (is_composite) {
    CeedInt num_suboperators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    if (!num_suboperators) {
      // Empty operator setup
      op->input_size  = 0;
      op->output_size = 0;
    } else {
      CeedOperator *sub_operators;

      CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
      for (CeedInt i = 0; i < num_suboperators; i++) {
        CeedCall(CeedOperatorCheckReady(sub_operators[i]));
      }
      // Sub-operators could be modified after adding to composite operator
      // Need to verify no lvec incompatibility from any changes
      CeedSize input_size, output_size;
      CeedCall(CeedOperatorGetActiveVectorLengths(op, &input_size, &output_size));
    }
  } else {
    CeedInt num_input_fields, num_output_fields;

    CeedCheck(op->num_fields > 0, ceed, CEED_ERROR_INCOMPLETE, "No operator fields set");
    CeedCall(CeedQFunctionGetFields(qf, &num_input_fields, NULL, &num_output_fields, NULL));
    CeedCheck(op->num_fields == num_input_fields + num_output_fields, ceed, CEED_ERROR_INCOMPLETE, "Not all operator fields set");
    CeedCheck(op->has_restriction, ceed, CEED_ERROR_INCOMPLETE, "At least one restriction required");
    CeedCheck(op->num_qpts > 0 || is_at_points, ceed, CEED_ERROR_INCOMPLETE,
              "At least one non-collocated CeedBasis is required or the number of quadrature points must be set");
  }

  // Flag as immutable and ready
  op->is_interface_setup = true;
  if (qf && qf != CEED_QFUNCTION_NONE) CeedCall(CeedQFunctionSetImmutable(qf));
  if (op->dqf && op->dqf != CEED_QFUNCTION_NONE) CeedCall(CeedQFunctionSetImmutable(op->dqf));
  if (op->dqfT && op->dqfT != CEED_QFUNCTION_NONE) CeedCall(CeedQFunctionSetImmutable(op->dqfT));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get vector lengths for the active input and/or output `CeedVector` of a `CeedOperator`.

  Note: Lengths of `-1` indicate that the CeedOperator does not have an active input and/or output.

  @param[in]  op          `CeedOperator`
  @param[out] input_size  Variable to store active input vector length, or `NULL`
  @param[out] output_size Variable to store active output vector length, or `NULL`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorGetActiveVectorLengths(CeedOperator op, CeedSize *input_size, CeedSize *output_size) {
  bool is_composite;

  if (input_size) *input_size = op->input_size;
  if (output_size) *output_size = op->output_size;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite && (op->input_size == -1 || op->output_size == -1)) {
    CeedInt       num_suboperators;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    for (CeedInt i = 0; i < num_suboperators; i++) {
      CeedSize sub_input_size, sub_output_size;

      CeedCall(CeedOperatorGetActiveVectorLengths(sub_operators[i], &sub_input_size, &sub_output_size));
      if (op->input_size == -1) op->input_size = sub_input_size;
      if (op->output_size == -1) op->output_size = sub_output_size;
      // Note, a size of -1 means no active vector restriction set, so no incompatibility
      CeedCheck((sub_input_size == -1 || sub_input_size == op->input_size) && (sub_output_size == -1 || sub_output_size == op->output_size),
                CeedOperatorReturnCeed(op), CEED_ERROR_MAJOR,
                "Sub-operators must have compatible dimensions; composite operator of shape (%" CeedSize_FMT ", %" CeedSize_FMT
                ") not compatible with sub-operator of "
                "shape (%" CeedSize_FMT ", %" CeedSize_FMT ")",
                op->input_size, op->output_size, input_size, output_size);
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set reuse of `CeedQFunction` data in `CeedOperatorLinearAssemble*()` functions.

  When `reuse_assembly_data = false` (default), the `CeedQFunction` associated with this `CeedOperator` is re-assembled every time a `CeedOperatorLinearAssemble*()` function is called.
  When `reuse_assembly_data = true`, the `CeedQFunction` associated with this `CeedOperator` is reused between calls to @ref CeedOperatorSetQFunctionAssemblyDataUpdateNeeded().

  @param[in] op                  `CeedOperator`
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
  @brief Mark `CeedQFunction` data as updated and the `CeedQFunction` as requiring re-assembly.

  @param[in] op                `CeedOperator`
  @param[in] needs_data_update Boolean flag setting assembly data reuse

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(CeedOperator op, bool needs_data_update) {
  bool is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt       num_suboperators;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    for (CeedInt i = 0; i < num_suboperators; i++) {
      CeedCall(CeedOperatorSetQFunctionAssemblyDataUpdateNeeded(sub_operators[i], needs_data_update));
    }
  } else {
    CeedCall(CeedQFunctionAssemblyDataSetUpdateNeeded(op->qf_assembled, needs_data_update));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set name of `CeedOperator` for @ref CeedOperatorView() output

  @param[in,out] op   `CeedOperator`
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
  @brief View a `CeedOperator`

  @param[in] op     `CeedOperator` to view
  @param[in] stream Stream to write; typically `stdout` or a file

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorView(CeedOperator op, FILE *stream) {
  bool has_name = op->name, is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
    CeedInt       num_suboperators;
    CeedOperator *sub_operators;

    CeedCall(CeedCompositeOperatorGetNumSub(op, &num_suboperators));
    CeedCall(CeedCompositeOperatorGetSubList(op, &sub_operators));
    fprintf(stream, "Composite CeedOperator%s%s\n", has_name ? " - " : "", has_name ? op->name : "");

    for (CeedInt i = 0; i < num_suboperators; i++) {
      has_name = sub_operators[i]->name;
      fprintf(stream, "  SubOperator %" CeedInt_FMT "%s%s:\n", i, has_name ? " - " : "", has_name ? sub_operators[i]->name : "");
      CeedCall(CeedOperatorSingleView(sub_operators[i], 1, stream));
    }
  } else {
    fprintf(stream, "CeedOperator%s%s\n", has_name ? " - " : "", has_name ? op->name : "");
    CeedCall(CeedOperatorSingleView(op, 0, stream));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `Ceed` associated with a `CeedOperator`

  @param[in]  op   `CeedOperator`
  @param[out] ceed Variable to store `Ceed`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetCeed(CeedOperator op, Ceed *ceed) {
  *ceed = CeedOperatorReturnCeed(op);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return the `Ceed` associated with a `CeedOperator`

  @param[in]  op `CeedOperator`

  @return `Ceed` associated with the `op`

  @ref Advanced
**/
Ceed CeedOperatorReturnCeed(CeedOperator op) { return op->ceed; }

/**
  @brief Get the number of elements associated with a `CeedOperator`

  @param[in]  op       `CeedOperator`
  @param[out] num_elem Variable to store number of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetNumElements(CeedOperator op, CeedInt *num_elem) {
  bool is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, CeedOperatorReturnCeed(op), CEED_ERROR_MINOR, "Not defined for composite operator");
  *num_elem = op->num_elem;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of quadrature points associated with a `CeedOperator`

  @param[in]  op       `CeedOperator`
  @param[out] num_qpts Variable to store vector number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetNumQuadraturePoints(CeedOperator op, CeedInt *num_qpts) {
  bool is_composite;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, CeedOperatorReturnCeed(op), CEED_ERROR_MINOR, "Not defined for composite operator");
  *num_qpts = op->num_qpts;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Estimate number of FLOPs required to apply `CeedOperator` on the active `CeedVector`

  @param[in]  op    `CeedOperator` to estimate FLOPs for
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
    CeedInt             num_input_fields, num_output_fields, num_elem = 0;
    CeedQFunction       qf;
    CeedQFunctionField *qf_input_fields, *qf_output_fields;
    CeedOperatorField  *op_input_fields, *op_output_fields;

    CeedCall(CeedOperatorGetQFunction(op, &qf));
    CeedCall(CeedQFunctionGetFields(qf, &num_input_fields, &qf_input_fields, &num_output_fields, &qf_output_fields));
    CeedCall(CeedOperatorGetFields(op, NULL, &op_input_fields, NULL, &op_output_fields));
    CeedCall(CeedOperatorGetNumElements(op, &num_elem));

    // Input FLOPs
    for (CeedInt i = 0; i < num_input_fields; i++) {
      CeedVector vec;

      CeedCall(CeedOperatorFieldGetVector(op_input_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedEvalMode        eval_mode;
        CeedSize            rstr_flops, basis_flops;
        CeedElemRestriction rstr;
        CeedBasis           basis;

        CeedCall(CeedOperatorFieldGetElemRestriction(op_input_fields[i], &rstr));
        CeedCall(CeedElemRestrictionGetFlopsEstimate(rstr, CEED_NOTRANSPOSE, &rstr_flops));
        *flops += rstr_flops;
        CeedCall(CeedOperatorFieldGetBasis(op_input_fields[i], &basis));
        CeedCall(CeedQFunctionFieldGetEvalMode(qf_input_fields[i], &eval_mode));
        CeedCall(CeedBasisGetFlopsEstimate(basis, CEED_NOTRANSPOSE, eval_mode, &basis_flops));
        *flops += basis_flops * num_elem;
      }
    }
    // QF FLOPs
    {
      CeedInt       num_qpts;
      CeedSize      qf_flops;
      CeedQFunction qf;

      CeedCall(CeedOperatorGetNumQuadraturePoints(op, &num_qpts));
      CeedCall(CeedOperatorGetQFunction(op, &qf));
      CeedCall(CeedQFunctionGetFlopsEstimate(qf, &qf_flops));
      CeedCheck(qf_flops > -1, CeedOperatorReturnCeed(op), CEED_ERROR_INCOMPLETE,
                "Must set CeedQFunction FLOPs estimate with CeedQFunctionSetUserFlopsEstimate");
      *flops += num_elem * num_qpts * qf_flops;
    }

    // Output FLOPs
    for (CeedInt i = 0; i < num_output_fields; i++) {
      CeedVector vec;

      CeedCall(CeedOperatorFieldGetVector(op_output_fields[i], &vec));
      if (vec == CEED_VECTOR_ACTIVE) {
        CeedEvalMode        eval_mode;
        CeedSize            rstr_flops, basis_flops;
        CeedElemRestriction rstr;
        CeedBasis           basis;

        CeedCall(CeedOperatorFieldGetElemRestriction(op_output_fields[i], &rstr));
        CeedCall(CeedElemRestrictionGetFlopsEstimate(rstr, CEED_TRANSPOSE, &rstr_flops));
        *flops += rstr_flops;
        CeedCall(CeedOperatorFieldGetBasis(op_output_fields[i], &basis));
        CeedCall(CeedQFunctionFieldGetEvalMode(qf_output_fields[i], &eval_mode));
        CeedCall(CeedBasisGetFlopsEstimate(basis, CEED_TRANSPOSE, eval_mode, &basis_flops));
        *flops += basis_flops * num_elem;
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get `CeedQFunction` global context for a `CeedOperator`.

  The caller is responsible for destroying `ctx` returned from this function via @ref CeedQFunctionContextDestroy().

  Note: If the value of `ctx` passed into this function is non-`NULL`, then it is assumed that `ctx` is a pointer to a `CeedQFunctionContext`.
        This `CeedQFunctionContext` will be destroyed if `ctx` is the only reference to this `CeedQFunctionContext`.

  @param[in]  op  `CeedOperator`
  @param[out] ctx Variable to store `CeedQFunctionContext`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorGetContext(CeedOperator op, CeedQFunctionContext *ctx) {
  bool                 is_composite;
  CeedQFunction        qf;
  CeedQFunctionContext qf_ctx;

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  CeedCheck(!is_composite, CeedOperatorReturnCeed(op), CEED_ERROR_INCOMPATIBLE, "Cannot retrieve CeedQFunctionContext for composite operator");
  CeedCall(CeedOperatorGetQFunction(op, &qf));
  CeedCall(CeedQFunctionGetInnerContext(qf, &qf_ctx));
  if (qf_ctx) CeedCall(CeedQFunctionContextReferenceCopy(qf_ctx, ctx));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get label for a registered `CeedQFunctionContext` field, or `NULL` if no field has been registered with this `field_name`.

  Fields are registered via `CeedQFunctionContextRegister*()` functions (eg. @ref CeedQFunctionContextRegisterDouble()).

  @param[in]  op          `CeedOperator`
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
            return CeedError(CeedOperatorReturnCeed(op), CEED_ERROR_INCOMPATIBLE, "Incompatible field types on sub-operator contexts. %s != %s",
                             CeedContextFieldTypes[new_field_label->type], CeedContextFieldTypes[new_field_label_i->type]);
            // LCOV_EXCL_STOP
          } else {
            new_field_label->type = new_field_label_i->type;
          }
          if (new_field_label->num_values != 0 && new_field_label->num_values != new_field_label_i->num_values) {
            // LCOV_EXCL_START
            CeedCall(CeedFree(&new_field_label));
            return CeedError(CeedOperatorReturnCeed(op), CEED_ERROR_INCOMPATIBLE,
                             "Incompatible field number of values on sub-operator contexts. %zu != %zu", new_field_label->num_values,
                             new_field_label_i->num_values);
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
    CeedQFunction        qf;
    CeedQFunctionContext ctx;

    // Single, non-composite operator
    CeedCall(CeedOperatorGetQFunction(op, &qf));
    CeedCall(CeedQFunctionGetInnerContext(qf, &ctx));
    if (ctx) {
      CeedCall(CeedQFunctionContextGetFieldLabel(ctx, field_name, field_label));
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
  @brief Set `CeedQFunctionContext` field holding double precision values.

  For composite operators, the values are set in all sub-operator `CeedQFunctionContext` that have a matching `field_name`.

  @param[in,out] op          `CeedOperator`
  @param[in]     field_label Label of field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetContextDouble(CeedOperator op, CeedContextFieldLabel field_label, double *values) {
  return CeedOperatorContextSetGeneric(op, field_label, CEED_CONTEXT_FIELD_DOUBLE, values);
}

/**
  @brief Get `CeedQFunctionContext` field holding double precision values, read-only.

  For composite operators, the values correspond to the first sub-operator `CeedQFunctionContext` that has a matching `field_name`.

  @param[in]  op          `CeedOperator`
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
  @brief Restore `CeedQFunctionContext` field holding double precision values, read-only.

  @param[in]  op          `CeedOperator`
  @param[in]  field_label Label of field to restore
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorRestoreContextDoubleRead(CeedOperator op, CeedContextFieldLabel field_label, const double **values) {
  return CeedOperatorContextRestoreGenericRead(op, field_label, CEED_CONTEXT_FIELD_DOUBLE, values);
}

/**
  @brief Set `CeedQFunctionContext` field holding `int32` values.

  For composite operators, the values are set in all sub-operator `CeedQFunctionContext` that have a matching `field_name`.

  @param[in,out] op          `CeedOperator`
  @param[in]     field_label Label of field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetContextInt32(CeedOperator op, CeedContextFieldLabel field_label, int32_t *values) {
  return CeedOperatorContextSetGeneric(op, field_label, CEED_CONTEXT_FIELD_INT32, values);
}

/**
  @brief Get `CeedQFunctionContext` field holding `int32` values, read-only.

  For composite operators, the values correspond to the first sub-operator `CeedQFunctionContext` that has a matching `field_name`.

  @param[in]  op          `CeedOperator`
  @param[in]  field_label Label of field to get
  @param[out] num_values  Number of `int32` values in `values`
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorGetContextInt32Read(CeedOperator op, CeedContextFieldLabel field_label, size_t *num_values, const int32_t **values) {
  return CeedOperatorContextGetGenericRead(op, field_label, CEED_CONTEXT_FIELD_INT32, num_values, values);
}

/**
  @brief Restore `CeedQFunctionContext` field holding `int32` values, read-only.

  @param[in]  op          `CeedOperator`
  @param[in]  field_label Label of field to get
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorRestoreContextInt32Read(CeedOperator op, CeedContextFieldLabel field_label, const int32_t **values) {
  return CeedOperatorContextRestoreGenericRead(op, field_label, CEED_CONTEXT_FIELD_INT32, values);
}

/**
  @brief Set `CeedQFunctionContext` field holding boolean values.

  For composite operators, the values are set in all sub-operator `CeedQFunctionContext` that have a matching `field_name`.

  @param[in,out] op          `CeedOperator`
  @param[in]     field_label Label of field to set
  @param[in]     values      Values to set

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetContextBoolean(CeedOperator op, CeedContextFieldLabel field_label, bool *values) {
  return CeedOperatorContextSetGeneric(op, field_label, CEED_CONTEXT_FIELD_BOOL, values);
}

/**
  @brief Get `CeedQFunctionContext` field holding boolean values, read-only.

  For composite operators, the values correspond to the first sub-operator `CeedQFunctionContext` that has a matching `field_name`.

  @param[in]  op          `CeedOperator`
  @param[in]  field_label Label of field to get
  @param[out] num_values  Number of boolean values in `values`
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorGetContextBooleanRead(CeedOperator op, CeedContextFieldLabel field_label, size_t *num_values, const bool **values) {
  return CeedOperatorContextGetGenericRead(op, field_label, CEED_CONTEXT_FIELD_BOOL, num_values, values);
}

/**
  @brief Restore `CeedQFunctionContext` field holding boolean values, read-only.

  @param[in]  op          `CeedOperator`
  @param[in]  field_label Label of field to get
  @param[out] values      Pointer to context values

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorRestoreContextBooleanRead(CeedOperator op, CeedContextFieldLabel field_label, const bool **values) {
  return CeedOperatorContextRestoreGenericRead(op, field_label, CEED_CONTEXT_FIELD_BOOL, values);
}

/**
  @brief Apply `CeedOperator` to a `CeedVector`.

  This computes the action of the operator on the specified (active) input, yielding its (active) output.
  All inputs and outputs must be specified using @ref CeedOperatorSetField().

  Note: Calling this function asserts that setup is complete and sets the `CeedOperator` as immutable.

  @param[in]  op      `CeedOperator` to apply
  @param[in]  in      `CeedVector` containing input state or @ref CEED_VECTOR_NONE if there are no active inputs
  @param[out] out     `CeedVector` to store result of applying operator (must be distinct from `in`) or @ref CEED_VECTOR_NONE if there are no active outputs
  @param[in]  request Address of @ref CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorApply(CeedOperator op, CeedVector in, CeedVector out, CeedRequest *request) {
  bool is_composite;

  CeedCall(CeedOperatorCheckReady(op));

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
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
        CeedInt            num_output_fields;
        CeedOperatorField *output_fields;

        CeedCall(CeedOperatorGetFields(sub_operators[i], NULL, NULL, &num_output_fields, &output_fields));
        for (CeedInt j = 0; j < num_output_fields; j++) {
          CeedVector vec;

          CeedCall(CeedOperatorFieldGetVector(output_fields[j], &vec));
          if (vec != CEED_VECTOR_ACTIVE && vec != CEED_VECTOR_NONE) {
            CeedCall(CeedVectorSetValue(vec, 0.0));
          }
        }
      }
      // Apply
      for (CeedInt i = 0; i < num_suboperators; i++) {
        CeedCall(CeedOperatorApplyAdd(sub_operators[i], in, out, request));
      }
    }
  } else {
    // Standard Operator
    if (op->Apply) {
      CeedCall(op->Apply(op, in, out, request));
    } else {
      CeedInt            num_output_fields;
      CeedOperatorField *output_fields;

      CeedCall(CeedOperatorGetFields(op, NULL, NULL, &num_output_fields, &output_fields));
      // Zero all output vectors
      for (CeedInt i = 0; i < num_output_fields; i++) {
        CeedVector vec;

        CeedCall(CeedOperatorFieldGetVector(output_fields[i], &vec));
        if (vec == CEED_VECTOR_ACTIVE) vec = out;
        if (vec != CEED_VECTOR_NONE) CeedCall(CeedVectorSetValue(vec, 0.0));
      }
      // Apply
      if (op->num_elem > 0) CeedCall(op->ApplyAdd(op, in, out, request));
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply `CeedOperator` to a `CeedVector` and add result to output `CeedVector`.

  This computes the action of the operator on the specified (active) input, yielding its (active) output.
  All inputs and outputs must be specified using @ref CeedOperatorSetField().

  @param[in]  op      `CeedOperator` to apply
  @param[in]  in      `CeedVector` containing input state or @ref CEED_VECTOR_NONE if there are no active inputs
  @param[out] out     `CeedVector` to sum in result of applying operator (must be distinct from `in`) or @ref CEED_VECTOR_NONE if there are no active outputs
  @param[in]  request Address of @ref CeedRequest for non-blocking completion, else @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorApplyAdd(CeedOperator op, CeedVector in, CeedVector out, CeedRequest *request) {
  bool is_composite;

  CeedCall(CeedOperatorCheckReady(op));

  CeedCall(CeedOperatorIsComposite(op, &is_composite));
  if (is_composite) {
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
  } else if (op->num_elem > 0) {
    // Standard Operator
    CeedCall(op->ApplyAdd(op, in, out, request));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a `CeedOperator`

  @param[in,out] op `CeedOperator` to destroy

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
      if ((*op)->input_fields[i]->basis != CEED_BASIS_NONE) {
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
      if ((*op)->output_fields[i]->basis != CEED_BASIS_NONE) {
        CeedCall(CeedBasisDestroy(&(*op)->output_fields[i]->basis));
      }
      if ((*op)->output_fields[i]->vec != CEED_VECTOR_ACTIVE && (*op)->output_fields[i]->vec != CEED_VECTOR_NONE) {
        CeedCall(CeedVectorDestroy(&(*op)->output_fields[i]->vec));
      }
      CeedCall(CeedFree(&(*op)->output_fields[i]->field_name));
      CeedCall(CeedFree(&(*op)->output_fields[i]));
    }
  }
  // AtPoints data
  CeedCall(CeedVectorDestroy(&(*op)->point_coords));
  CeedCall(CeedElemRestrictionDestroy(&(*op)->rstr_points));
  CeedCall(CeedElemRestrictionDestroy(&(*op)->first_points_rstr));
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
