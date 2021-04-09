// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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
#include <ceed-impl.h>
#include <math.h>
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
  op_ref->interface_setup = false;
  op_ref->backend_setup = false;
  op_ref->ceed = ceed_ref;
  ierr = ceed_ref->OperatorCreate(op_ref); CeedChk(ierr);
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
  CeedInt dim = 1, num_comp = 1, restr_num_comp = 1, size = qf_field->size;
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
      return CeedError(ceed, CEED_ERROR_INCOMPATIBLE,
                       "Field '%s' configured with CEED_EVAL_NONE must "
                       "be used with CEED_BASIS_COLLOCATED",
                       qf_field->field_name);
    ierr = CeedBasisGetDimension(b, &dim); CeedChk(ierr);
    ierr = CeedBasisGetNumComponents(b, &num_comp); CeedChk(ierr);
    if (r != CEED_ELEMRESTRICTION_NONE && restr_num_comp != num_comp) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Field '%s' of size %d and EvalMode %s: ElemRestriction "
                       "has %d components, but Basis has %d components",
                       qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode],
                       restr_num_comp,
                       num_comp);
      // LCOV_EXCL_STOP
    }
  }
  // Field size
  switch(eval_mode) {
  case CEED_EVAL_NONE:
    if (size != restr_num_comp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Field '%s' of size %d and EvalMode %s: ElemRestriction has %d components",
                       qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode],
                       restr_num_comp);
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_INTERP:
    if (size != num_comp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Field '%s' of size %d and EvalMode %s: ElemRestriction/Basis has %d components",
                       qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode],
                       num_comp);
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_GRAD:
    if (size != num_comp * dim)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Field '%s' of size %d and EvalMode %s in %d dimensions: "
                       "ElemRestriction/Basis has %d components",
                       qf_field->field_name, qf_field->size, CeedEvalModes[qf_field->eval_mode], dim,
                       num_comp);
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_WEIGHT:
    // No additional checks required
    break;
  case CEED_EVAL_DIV:
    // Not implemented
    break;
  case CEED_EVAL_CURL:
    // Not implemented
    break;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check if a CeedOperator is ready to be used.

  @param[in] op  CeedOperator to check

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCheckReady(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

  if (op->interface_setup)
    return CEED_ERROR_SUCCESS;

  CeedQFunction qf = op->qf;
  if (op->composite) {
    if (!op->num_suboperators)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE, "No sub_operators set");
    // LCOV_EXCL_STOP
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
                       "At least one non-collocated basis required");
    // LCOV_EXCL_STOP
  }

  // Flag as immutable and ready
  op->interface_setup = true;
  if (op->qf && op->qf != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    op->qf->operators_set++;
  // LCOV_EXCL_STOP
  if (op->dqf && op->dqf != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    op->dqf->operators_set++;
  // LCOV_EXCL_STOP
  if (op->dqfT && op->dqfT != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    op->dqfT->operators_set++;
  // LCOV_EXCL_STOP
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

  fprintf(stream, "%s    %s Field [%d]:\n"
          "%s      Name: \"%s\"\n",
          pre, in_out, field_number, pre, qf_field->field_name);

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

  CeedInt totalfields;
  ierr = CeedOperatorGetNumArgs(op, &totalfields); CeedChk(ierr);

  fprintf(stream, "%s  %d Field%s\n", pre, totalfields, totalfields>1 ? "s" : "");

  fprintf(stream, "%s  %d Input Field%s:\n", pre, op->qf->num_input_fields,
          op->qf->num_input_fields>1 ? "s" : "");
  for (CeedInt i=0; i<op->qf->num_input_fields; i++) {
    ierr = CeedOperatorFieldView(op->input_fields[i], op->qf->input_fields[i],
                                 i, sub, 1, stream); CeedChk(ierr);
  }

  fprintf(stream, "%s  %d Output Field%s:\n", pre, op->qf->num_output_fields,
          op->qf->num_output_fields>1 ? "s" : "");
  for (CeedInt i=0; i<op->qf->num_output_fields; i++) {
    ierr = CeedOperatorFieldView(op->output_fields[i], op->qf->output_fields[i],
                                 i, sub, 0, stream); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active vector ElemRestriction for a CeedOperator

  @param[in] op            CeedOperator to find active basis for
  @param[out] active_rstr  ElemRestriction for active input vector

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedOperatorGetActiveElemRestriction(CeedOperator op,
    CeedElemRestriction *active_rstr) {
  *active_rstr = NULL;
  for (int i = 0; i < op->qf->num_input_fields; i++)
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
                     "No active ElemRestriction found!");
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active vector basis for a CeedOperator

  @param[in] op             CeedOperator to find active basis for
  @param[out] active_basis  Basis for active input vector

  @return An error code: 0 - success, otherwise - failure

  @ ref Developer
**/
static int CeedOperatorGetActiveBasis(CeedOperator op,
                                      CeedBasis *active_basis) {
  *active_basis = NULL;
  for (int i = 0; i < op->qf->num_input_fields; i++)
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
                     "No active basis found for automatic multigrid setup");
    // LCOV_EXCL_STOP
  }
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
static int CeedOperatorMultigridLevel_Core(CeedOperator op_fine,
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

  // Cleanup
  ierr = CeedVectorDestroy(&mult_vec); CeedChk(ierr);
  ierr = CeedBasisDestroy(&basis_c_to_f); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&qf_restrict); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&qf_prolong); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedOperator Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedOperatorBackend
/// @{

/**
  @brief Get the Ceed associated with a CeedOperator

  @param op         CeedOperator
  @param[out] ceed  Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
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

  @ref Backend
**/

int CeedOperatorGetNumElements(CeedOperator op, CeedInt *num_elem) {
  if (op->composite)
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

  @ref Backend
**/

int CeedOperatorGetNumQuadraturePoints(CeedOperator op, CeedInt *num_qpts) {
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP

  *num_qpts = op->num_qpts;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of arguments associated with a CeedOperator

  @param op             CeedOperator
  @param[out] num_args  Variable to store vector number of arguments

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetNumArgs(CeedOperator op, CeedInt *num_args) {
  if (op->composite)
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
  *is_setup_done = op->backend_setup;
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
  if (op->composite)
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
  *is_composite = op->composite;
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
  if (!op->composite)
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
  if (!op->composite)
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
  @brief Set the setup flag of a CeedOperator to True

  @param op  CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorSetSetupDone(CeedOperator op) {
  op->backend_setup = true;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedOperatorFields of a CeedOperator

  @param op                  CeedOperator
  @param[out] input_fields   Variable to store input_fields
  @param[out] output_fields  Variable to store output_fields

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetFields(CeedOperator op, CeedOperatorField **input_fields,
                          CeedOperatorField **output_fields) {
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP

  if (input_fields) *input_fields = op->input_fields;
  if (output_fields) *output_fields = op->output_fields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedElemRestriction of a CeedOperatorField

  @param op_field   CeedOperatorField
  @param[out] rstr  Variable to store CeedElemRestriction

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
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

  @ref Backend
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

  @ref Backend
**/

int CeedOperatorFieldGetVector(CeedOperatorField op_field, CeedVector *vec) {
  *vec = op_field->vec;
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
  ceed->ref_count++;
  (*op)->ref_count = 1;
  (*op)->qf = qf;
  qf->ref_count++;
  if (dqf && dqf != CEED_QFUNCTION_NONE) {
    (*op)->dqf = dqf;
    dqf->ref_count++;
  }
  if (dqfT && dqfT != CEED_QFUNCTION_NONE) {
    (*op)->dqfT = dqfT;
    dqfT->ref_count++;
  }
  ierr = CeedCalloc(16, &(*op)->input_fields); CeedChk(ierr);
  ierr = CeedCalloc(16, &(*op)->output_fields); CeedChk(ierr);
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
  ceed->ref_count++;
  (*op)->composite = true;
  ierr = CeedCalloc(16, &(*op)->sub_operators); CeedChk(ierr);

  if (ceed->CompositeOperatorCreate) {
    ierr = ceed->CompositeOperatorCreate(*op); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Provide a field to a CeedOperator for use by its CeedQFunction

  This function is used to specify both active and passive fields to a
  CeedOperator.  For passive fields, a vector @arg v must be provided.  Passive
  fields can inputs or outputs (updated in-place when operator is applied).

  Active fields must be specified using this function, but their data (in a
  CeedVector) is passed in CeedOperatorApply().  There can be at most one active
  input and at most one active output.

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
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_INCOMPATIBLE,
                     "Cannot add field to composite operator.");
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
                     "ElemRestriction with %d elements incompatible with prior "
                     "%d elements", num_elem, op->num_elem);
  // LCOV_EXCL_STOP

  CeedInt num_qpts;
  if (b != CEED_BASIS_COLLOCATED) {
    ierr = CeedBasisGetNumQuadraturePoints(b, &num_qpts); CeedChk(ierr);
    if (op->num_qpts && op->num_qpts != num_qpts)
      // LCOV_EXCL_START
      return CeedError(op->ceed, CEED_ERROR_DIMENSION,
                       "Basis with %d quadrature points "
                       "incompatible with prior %d points", num_qpts,
                       op->num_qpts);
    // LCOV_EXCL_STOP
  }
  CeedQFunctionField qf_field;
  CeedOperatorField *op_field;
  for (CeedInt i=0; i<op->qf->num_input_fields; i++) {
    if (!strcmp(field_name, (*op->qf->input_fields[i]).field_name)) {
      qf_field = op->qf->input_fields[i];
      op_field = &op->input_fields[i];
      goto found;
    }
  }
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

  (*op_field)->vec = v;
  if (v != CEED_VECTOR_ACTIVE && v != CEED_VECTOR_NONE) {
    v->ref_count += 1;
  }

  (*op_field)->elem_restr = r;
  r->ref_count += 1;
  if (r != CEED_ELEMRESTRICTION_NONE) {
    op->num_elem = num_elem;
    op->has_restriction = true; // Restriction set, but num_elem may be 0
  }

  (*op_field)->basis = b;
  if (b != CEED_BASIS_COLLOCATED) {
    op->num_qpts = num_qpts;
    b->ref_count += 1;
  }

  op->num_fields += 1;
  size_t len = strlen(field_name);
  char *tmp;
  ierr = CeedCalloc(len+1, &tmp); CeedChk(ierr);
  memcpy(tmp, field_name, len+1);
  (*op_field)->field_name = tmp;
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
  if (!composite_op->composite)
    // LCOV_EXCL_START
    return CeedError(composite_op->ceed, CEED_ERROR_MINOR,
                     "CeedOperator is not a composite operator");
  // LCOV_EXCL_STOP

  if (composite_op->num_suboperators == CEED_COMPOSITE_MAX)
    // LCOV_EXCL_START
    return CeedError(composite_op->ceed, CEED_ERROR_UNSUPPORTED,
                     "Cannot add additional sub_operators");
  // LCOV_EXCL_STOP

  composite_op->sub_operators[composite_op->num_suboperators] = sub_op;
  sub_op->ref_count++;
  composite_op->num_suboperators++;
  return CEED_ERROR_SUCCESS;
}

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
    return op->LinearAssembleQFunction(op, assembled, rstr, request);
  } else {
    // Fallback to reference Ceed
    if (!op->op_fallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssembleQFunction(op->op_fallback, assembled,
           rstr, request);
  }
}

/**
  @brief Assemble the diagonal of a square linear CeedOperator

  This overwrites a CeedVector with the diagonal of a linear CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

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

  // Use backend version, if available
  if (op->LinearAssembleDiagonal) {
    return op->LinearAssembleDiagonal(op, assembled, request);
  } else if (op->LinearAssembleAddDiagonal) {
    ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
    return CeedOperatorLinearAssembleAddDiagonal(op, assembled, request);
  } else {
    // Fallback to reference Ceed
    if (!op->op_fallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssembleDiagonal(op->op_fallback, assembled,
           request);
  }
}

/**
  @brief Assemble the diagonal of a square linear CeedOperator

  This sums into a CeedVector the diagonal of a linear CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

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

  // Use backend version, if available
  if (op->LinearAssembleAddDiagonal) {
    return op->LinearAssembleAddDiagonal(op, assembled, request);
  } else {
    // Fallback to reference Ceed
    if (!op->op_fallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssembleAddDiagonal(op->op_fallback, assembled,
           request);
  }
}

/**
  @brief Assemble the point block diagonal of a square linear CeedOperator

  This overwrites a CeedVector with the point block diagonal of a linear
    CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

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

  // Use backend version, if available
  if (op->LinearAssemblePointBlockDiagonal) {
    return op->LinearAssemblePointBlockDiagonal(op, assembled, request);
  } else if (op->LinearAssembleAddPointBlockDiagonal) {
    ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
    return CeedOperatorLinearAssembleAddPointBlockDiagonal(op, assembled,
           request);
  } else {
    // Fallback to reference Ceed
    if (!op->op_fallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssemblePointBlockDiagonal(op->op_fallback,
           assembled, request);
  }
}

/**
  @brief Assemble the point block diagonal of a square linear CeedOperator

  This sums into a CeedVector with the point block diagonal of a linear
    CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

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

  // Use backend version, if available
  if (op->LinearAssembleAddPointBlockDiagonal) {
    return op->LinearAssembleAddPointBlockDiagonal(op, assembled, request);
  } else {
    // Fallback to reference Ceed
    if (!op->op_fallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssembleAddPointBlockDiagonal(op->op_fallback,
           assembled, request);
  }
}

/**
   @brief Build nonzero pattern for non-composite operator.

   Users should generally use CeedOperatorLinearAssembleSymbolic()

   @ref Developer
**/
int CeedSingleOperatorAssembleSymbolic(CeedOperator op, CeedInt offset,
                                       CeedInt *rows, CeedInt *cols) {
  int ierr;
  Ceed ceed = op->ceed;
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "Composite operator not supported");
  // LCOV_EXCL_STOP

  CeedElemRestriction rstr_in;
  ierr = CeedOperatorGetActiveElemRestriction(op, &rstr_in); CeedChk(ierr);
  CeedInt num_elem, elem_size, num_nodes, num_comp;
  ierr = CeedElemRestrictionGetNumElements(rstr_in, &num_elem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr_in, &elem_size); CeedChk(ierr);
  ierr = CeedElemRestrictionGetLVectorSize(rstr_in, &num_nodes); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr_in, &num_comp); CeedChk(ierr);
  CeedInt layout_er[3];
  ierr = CeedElemRestrictionGetELayout(rstr_in, &layout_er); CeedChk(ierr);

  CeedInt local_num_entries = elem_size*num_comp * elem_size*num_comp * num_elem;

  // Determine elem_dof relation
  CeedVector index_vec;
  ierr = CeedVectorCreate(ceed, num_nodes, &index_vec); CeedChk(ierr);
  CeedScalar *array;
  ierr = CeedVectorGetArray(index_vec, CEED_MEM_HOST, &array); CeedChk(ierr);
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

   @ref Developer
**/
int CeedSingleOperatorAssemble(CeedOperator op, CeedInt offset,
                               CeedVector values) {
  int ierr;
  Ceed ceed = op->ceed;;
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "Composite operator not supported");
  // LCOV_EXCL_STOP

  // Assemble QFunction
  CeedQFunction qf;
  ierr = CeedOperatorGetQFunction(op, &qf); CeedChk(ierr);
  CeedInt num_input_fields, num_output_fields;
  ierr= CeedQFunctionGetNumArgs(qf, &num_input_fields, &num_output_fields);
  CeedChk(ierr);
  CeedVector assembled_qf;
  CeedElemRestriction rstr_q;
  ierr = CeedOperatorLinearAssembleQFunction(
           op, &assembled_qf, &rstr_q, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

  CeedInt qf_length;
  ierr = CeedVectorGetLength(assembled_qf, &qf_length); CeedChk(ierr);

  CeedOperatorField *input_fields;
  CeedOperatorField *output_fields;
  ierr = CeedOperatorGetFields(op, &input_fields, &output_fields); CeedChk(ierr);

  // Determine active input basis
  CeedQFunctionField *qf_fields;
  ierr = CeedQFunctionGetFields(qf, &qf_fields, NULL); CeedChk(ierr);
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
  ierr = CeedQFunctionGetFields(qf, NULL, &qf_fields); CeedChk(ierr);
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
  ierr = CeedVectorGetArray(values, CEED_MEM_HOST, &vals); CeedChk(ierr);
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
   @ref Utility
**/
int CeedSingleOperatorAssemblyCountEntries(CeedOperator op,
    CeedInt *num_entries) {
  int ierr;
  CeedElemRestriction rstr;
  CeedInt num_elem, elem_size, num_comp;

  if (op->composite)
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
   @brief Fully assemble the nonzero pattern of a linear operator.

   Expected to be used in conjunction with CeedOperatorLinearAssemble()

   The assembly routines use coordinate format, with num_entries tuples of the
   form (i, j, value) which indicate that value should be added to the matrix
   in entry (i, j). Note that the (i, j) pairs are not unique and may repeat.
   This function returns the number of entries and their (i, j) locations,
   while CeedOperatorLinearAssemble() provides the values in the same
   ordering.

   This will generally be slow unless your operator is low-order.

   @param[in]  op           CeedOperator to assemble
   @param[out] num_entries  Number of entries in coordinate nonzero pattern.
   @param[out] rows         Row number for each entry.
   @param[out] cols         Column number for each entry.

   @ref User
**/
int CeedOperatorLinearAssembleSymbolic(CeedOperator op,
                                       CeedInt *num_entries, CeedInt **rows, CeedInt **cols) {
  int ierr;
  CeedInt num_suboperators, single_entries;
  CeedOperator *sub_operators;
  bool is_composite;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssembleSymbolic) {
    return op->LinearAssembleSymbolic(op, num_entries, rows, cols);
  } else {
    // Check for valid fallback resource
    const char *resource, *fallback_resource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
    if (strcmp(fallback_resource, "") && strcmp(resource, fallback_resource)) {
      // Fallback to reference Ceed
      if (!op->op_fallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      return CeedOperatorLinearAssembleSymbolic(op->op_fallback, num_entries, rows,
             cols);
    }
  }

  // if neither backend nor fallback resource provides
  // LinearAssembleSymbolic, continue with interface-level implementation

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

   @param[in]  op      CeedOperator to assemble
   @param[out] values  Values to assemble into matrix

   @ref User
**/
int CeedOperatorLinearAssemble(CeedOperator op, CeedVector values) {
  int ierr;
  CeedInt num_suboperators, single_entries;
  CeedOperator *sub_operators;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssemble) {
    return op->LinearAssemble(op, values);
  } else {
    // Check for valid fallback resource
    const char *resource, *fallback_resource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallback_resource);
    if (strcmp(fallback_resource, "") && strcmp(resource, fallback_resource)) {
      // Fallback to reference Ceed
      if (!op->op_fallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      return CeedOperatorLinearAssemble(op->op_fallback, values);
    }
  }

  // if neither backend nor fallback resource provides
  // LinearAssemble, continue with interface-level implementation

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
  ierr = CeedOperatorMultigridLevel_Core(op_fine, p_mult_fine, rstr_coarse,
                                         basis_coarse, basis_c_to_f, op_coarse,
                                         op_prolong, op_restrict);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a non-tensor basis for the active vector

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
  ierr = CeedCalloc(num_nodes_f, &q_ref); CeedChk(ierr);
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
  ierr = CeedOperatorMultigridLevel_Core(op_fine, p_mult_fine, rstr_coarse,
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

  @param op            CeedOperator to create element inverses
  @param[out] fdm_inv  CeedOperator to apply the action of a FDM based inverse
                         for each element
  @param request       Address of CeedRequest for non-blocking completion, else
                         @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorCreateFDMElementInverse(CeedOperator op, CeedOperator *fdm_inv,
                                        CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Use backend version, if available
  if (op->CreateFDMElementInverse) {
    ierr = op->CreateFDMElementInverse(op, fdm_inv, request); CeedChk(ierr);
  } else {
    // Fallback to reference Ceed
    if (!op->op_fallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    ierr = op->op_fallback->CreateFDMElementInverse(op->op_fallback, fdm_inv,
           request); CeedChk(ierr);
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

  if (op->composite) {
    fprintf(stream, "Composite CeedOperator\n");

    for (CeedInt i=0; i<op->num_suboperators; i++) {
      fprintf(stream, "  SubOperator [%d]:\n", i);
      ierr = CeedOperatorSingleView(op->sub_operators[i], 1, stream);
      CeedChk(ierr);
    }
  } else {
    fprintf(stream, "CeedOperator\n");
    ierr = CeedOperatorSingleView(op, 0, stream); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply CeedOperator to a vector

  This computes the action of the operator on the specified (active) input,
  yielding its (active) output.  All inputs and outputs must be specified using
  CeedOperatorSetField().

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
  } else if (op->composite) {
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
  } else if (op->composite) {
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
  for (int i=0; i<(*op)->num_fields; i++)
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
  for (int i=0; i<(*op)->num_fields; i++)
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
  for (int i=0; i<(*op)->num_suboperators; i++)
    if ((*op)->sub_operators[i]) {
      ierr = CeedOperatorDestroy(&(*op)->sub_operators[i]); CeedChk(ierr);
    }
  if ((*op)->qf)
    // LCOV_EXCL_START
    (*op)->qf->operators_set--;
  // LCOV_EXCL_STOP
  ierr = CeedQFunctionDestroy(&(*op)->qf); CeedChk(ierr);
  if ((*op)->dqf && (*op)->dqf != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    (*op)->dqf->operators_set--;
  // LCOV_EXCL_STOP
  ierr = CeedQFunctionDestroy(&(*op)->dqf); CeedChk(ierr);
  if ((*op)->dqfT && (*op)->dqfT != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    (*op)->dqfT->operators_set--;
  // LCOV_EXCL_STOP
  ierr = CeedQFunctionDestroy(&(*op)->dqfT); CeedChk(ierr);

  // Destroy fallback
  if ((*op)->op_fallback) {
    ierr = (*op)->qf_fallback->Destroy((*op)->qf_fallback); CeedChk(ierr);
    ierr = CeedFree(&(*op)->qf_fallback); CeedChk(ierr);
    ierr = (*op)->op_fallback->Destroy((*op)->op_fallback); CeedChk(ierr);
    ierr = CeedFree(&(*op)->op_fallback); CeedChk(ierr);
  }

  ierr = CeedFree(&(*op)->input_fields); CeedChk(ierr);
  ierr = CeedFree(&(*op)->output_fields); CeedChk(ierr);
  ierr = CeedFree(&(*op)->sub_operators); CeedChk(ierr);
  ierr = CeedFree(op); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}
