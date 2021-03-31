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

#include <ceed.h>
#include <ceed-backend.h>
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

  @param op           CeedOperator to create fallback for

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedOperatorCreateFallback(CeedOperator op) {
  int ierr;

  // Fallback Ceed
  const char *resource, *fallbackresource;
  ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
  ierr = CeedGetOperatorFallbackResource(op->ceed, &fallbackresource);
  CeedChk(ierr);
  if (!strcmp(resource, fallbackresource))
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend %s cannot create an operator"
                     "fallback to resource %s", resource, fallbackresource);
  // LCOV_EXCL_STOP

  // Fallback Ceed
  Ceed ceedref;
  if (!op->ceed->opfallbackceed) {
    ierr = CeedInit(fallbackresource, &ceedref); CeedChk(ierr);
    ceedref->opfallbackparent = op->ceed;
    ceedref->Error = op->ceed->Error;
    op->ceed->opfallbackceed = ceedref;
  }
  ceedref = op->ceed->opfallbackceed;

  // Clone Op
  CeedOperator opref;
  ierr = CeedCalloc(1, &opref); CeedChk(ierr);
  memcpy(opref, op, sizeof(*opref));
  opref->data = NULL;
  opref->interfacesetup = false;
  opref->backendsetup = false;
  opref->ceed = ceedref;
  ierr = ceedref->OperatorCreate(opref); CeedChk(ierr);
  op->opfallback = opref;

  // Clone QF
  CeedQFunction qfref;
  ierr = CeedCalloc(1, &qfref); CeedChk(ierr);
  memcpy(qfref, (op->qf), sizeof(*qfref));
  qfref->data = NULL;
  qfref->ceed = ceedref;
  ierr = ceedref->QFunctionCreate(qfref); CeedChk(ierr);
  opref->qf = qfref;
  op->qffallback = qfref;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check if a CeedOperator Field matches the QFunction Field

  @param[in] ceed   Ceed object for error handling
  @param[in] qfield QFunction Field matching Operator Field
  @param[in] r      Operator Field ElemRestriction
  @param[in] b      Operator Field Basis

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCheckField(Ceed ceed, CeedQFunctionField qfield,
                                  CeedElemRestriction r, CeedBasis b) {
  int ierr;
  CeedEvalMode emode = qfield->emode;
  CeedInt dim = 1, ncomp = 1, restr_ncomp = 1, size = qfield->size;
  // Restriction
  if (r != CEED_ELEMRESTRICTION_NONE) {
    ierr = CeedElemRestrictionGetNumComponents(r, &restr_ncomp);
  } else if (emode != CEED_EVAL_WEIGHT) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MINOR,
                     "CEED_ELEMRESTRICTION_NONE can only be used "
                     "for a field with eval mode CEED_EVAL_WEIGHT");
    // LCOV_EXCL_STOP
  }
  // Basis
  if (b != CEED_BASIS_COLLOCATED) {
    ierr = CeedBasisGetDimension(b, &dim); CeedChk(ierr);
    ierr = CeedBasisGetNumComponents(b, &ncomp); CeedChk(ierr);
    if (r != CEED_ELEMRESTRICTION_NONE && restr_ncomp != ncomp) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "ElemRestriction and Basis have incompatible numbers of components");
      // LCOV_EXCL_STOP
    }
  }
  // Field size
  switch(emode) {
  case CEED_EVAL_NONE:
    if (size != restr_ncomp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Incompatible QFunction field size and Operator field "
                       "ElemRestriction number of components");
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_INTERP:
    if (size != ncomp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Incompatible QFunction field size and Operator field "
                       "Basis number of components");
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_GRAD:
    if (size != ncomp * dim)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_DIMENSION,
                       "Incompatible QFunction field size and Operator field "
                       "Basis dimension and number of components");
    // LCOV_EXCL_STOP
    break;
  case CEED_EVAL_WEIGHT:
    // No addional checks required
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

  @param[in] op   CeedOperator to check

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCheckReady(CeedOperator op) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

  if (op->interfacesetup)
    return CEED_ERROR_SUCCESS;

  CeedQFunction qf = op->qf;
  if (op->composite) {
    if (!op->numsub)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE, "No suboperators set");
    // LCOV_EXCL_STOP
  } else {
    if (op->nfields == 0)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE, "No operator fields set");
    // LCOV_EXCL_STOP
    if (op->nfields < qf->numinputfields + qf->numoutputfields)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE, "Not all operator fields set");
    // LCOV_EXCL_STOP
    if (!op->hasrestriction)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE,
                       "At least one restriction required");
    // LCOV_EXCL_STOP
    if (op->numqpoints == 0)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_INCOMPLETE,
                       "At least one non-collocated basis required");
    // LCOV_EXCL_STOP
  }

  // Flag as immutable and ready
  op->interfacesetup = true;
  if (op->qf && op->qf != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    op->qf->operatorsset++;
  // LCOV_EXCL_STOP
  if (op->dqf && op->dqf != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    op->dqf->operatorsset++;
  // LCOV_EXCL_STOP
  if (op->dqfT && op->dqfT != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    op->dqfT->operatorsset++;
  // LCOV_EXCL_STOP
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a field of a CeedOperator

  @param[in] field       Operator field to view
  @param[in] qffield     QFunction field (carries field name)
  @param[in] fieldnumber Number of field being viewed
  @param[in] sub         true indicates sub-operator, which increases indentation; false for top-level operator
  @param[in] in          true for an input field; false for output field
  @param[in] stream      Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedOperatorFieldView(CeedOperatorField field,
                                 CeedQFunctionField qffield,
                                 CeedInt fieldnumber, bool sub, bool in,
                                 FILE *stream) {
  const char *pre = sub ? "  " : "";
  const char *inout = in ? "Input" : "Output";

  fprintf(stream, "%s    %s Field [%d]:\n"
          "%s      Name: \"%s\"\n",
          pre, inout, fieldnumber, pre, qffield->fieldname);

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

  @param[in] op     CeedOperator to view
  @param[in] sub    Boolean flag for sub-operator
  @param[in] stream Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedOperatorSingleView(CeedOperator op, bool sub, FILE *stream) {
  int ierr;
  const char *pre = sub ? "  " : "";

  CeedInt totalfields;
  ierr = CeedOperatorGetNumArgs(op, &totalfields); CeedChk(ierr);

  fprintf(stream, "%s  %d Field%s\n", pre, totalfields, totalfields>1 ? "s" : "");

  fprintf(stream, "%s  %d Input Field%s:\n", pre, op->qf->numinputfields,
          op->qf->numinputfields>1 ? "s" : "");
  for (CeedInt i=0; i<op->qf->numinputfields; i++) {
    ierr = CeedOperatorFieldView(op->inputfields[i], op->qf->inputfields[i],
                                 i, sub, 1, stream); CeedChk(ierr);
  }

  fprintf(stream, "%s  %d Output Field%s:\n", pre, op->qf->numoutputfields,
          op->qf->numoutputfields>1 ? "s" : "");
  for (CeedInt i=0; i<op->qf->numoutputfields; i++) {
    ierr = CeedOperatorFieldView(op->outputfields[i], op->qf->outputfields[i],
                                 i, sub, 0, stream); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the active vector ElemRestriction for a CeedOperator

  @param[in] op            CeedOperator to find active basis for
  @param[out] activerstr   ElemRestriction for active input vector

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
static int CeedOperatorGetActiveElemRestriction(CeedOperator op,
    CeedElemRestriction *activerstr) {
  *activerstr = NULL;
  for (int i = 0; i < op->qf->numinputfields; i++)
    if (op->inputfields[i]->vec == CEED_VECTOR_ACTIVE) {
      *activerstr = op->inputfields[i]->Erestrict;
      break;
    }

  if (!*activerstr) {
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

  @param[in] op            CeedOperator to find active basis for
  @param[out] activeBasis  Basis for active input vector

  @return An error code: 0 - success, otherwise - failure

  @ ref Developer
**/
static int CeedOperatorGetActiveBasis(CeedOperator op,
                                      CeedBasis *activeBasis) {
  *activeBasis = NULL;
  for (int i = 0; i < op->qf->numinputfields; i++)
    if (op->inputfields[i]->vec == CEED_VECTOR_ACTIVE) {
      *activeBasis = op->inputfields[i]->basis;
      break;
    }

  if (!*activeBasis) {
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

  @param[in] opFine       Fine grid operator
  @param[in] PMultFine    L-vector multiplicity in parallel gather/scatter
  @param[in] rstrCoarse   Coarse grid restriction
  @param[in] basisCoarse  Coarse grid active vector basis
  @param[in] basisCtoF    Basis for coarse to fine interpolation
  @param[out] opCoarse    Coarse grid operator
  @param[out] opProlong   Coarse to fine operator
  @param[out] opRestrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorMultigridLevel_Core(CeedOperator opFine,
    CeedVector PMultFine, CeedElemRestriction rstrCoarse, CeedBasis basisCoarse,
    CeedBasis basisCtoF, CeedOperator *opCoarse, CeedOperator *opProlong,
    CeedOperator *opRestrict) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(opFine, &ceed); CeedChk(ierr);

  // Check for composite operator
  bool isComposite;
  ierr = CeedOperatorIsComposite(opFine, &isComposite); CeedChk(ierr);
  if (isComposite)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                     "Automatic multigrid setup for composite operators not supported");
  // LCOV_EXCL_STOP

  // Coarse Grid
  ierr = CeedOperatorCreate(ceed, opFine->qf, opFine->dqf, opFine->dqfT,
                            opCoarse); CeedChk(ierr);
  CeedElemRestriction rstrFine = NULL;
  // -- Clone input fields
  for (int i = 0; i < opFine->qf->numinputfields; i++) {
    if (opFine->inputfields[i]->vec == CEED_VECTOR_ACTIVE) {
      rstrFine = opFine->inputfields[i]->Erestrict;
      ierr = CeedOperatorSetField(*opCoarse, opFine->inputfields[i]->fieldname,
                                  rstrCoarse, basisCoarse, CEED_VECTOR_ACTIVE);
      CeedChk(ierr);
    } else {
      ierr = CeedOperatorSetField(*opCoarse, opFine->inputfields[i]->fieldname,
                                  opFine->inputfields[i]->Erestrict,
                                  opFine->inputfields[i]->basis,
                                  opFine->inputfields[i]->vec); CeedChk(ierr);
    }
  }
  // -- Clone output fields
  for (int i = 0; i < opFine->qf->numoutputfields; i++) {
    if (opFine->outputfields[i]->vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorSetField(*opCoarse, opFine->outputfields[i]->fieldname,
                                  rstrCoarse, basisCoarse, CEED_VECTOR_ACTIVE);
      CeedChk(ierr);
    } else {
      ierr = CeedOperatorSetField(*opCoarse, opFine->outputfields[i]->fieldname,
                                  opFine->outputfields[i]->Erestrict,
                                  opFine->outputfields[i]->basis,
                                  opFine->outputfields[i]->vec); CeedChk(ierr);
    }
  }

  // Multiplicity vector
  CeedVector multVec, multE;
  ierr = CeedElemRestrictionCreateVector(rstrFine, &multVec, &multE);
  CeedChk(ierr);
  ierr = CeedVectorSetValue(multE, 0.0); CeedChk(ierr);
  ierr = CeedElemRestrictionApply(rstrFine, CEED_NOTRANSPOSE, PMultFine, multE,
                                  CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  ierr = CeedVectorSetValue(multVec, 0.0); CeedChk(ierr);
  ierr = CeedElemRestrictionApply(rstrFine, CEED_TRANSPOSE, multE, multVec,
                                  CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  ierr = CeedVectorDestroy(&multE); CeedChk(ierr);
  ierr = CeedVectorReciprocal(multVec); CeedChk(ierr);

  // Restriction
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basisCoarse, &ncomp); CeedChk(ierr);
  CeedQFunction qfRestrict;
  ierr = CeedQFunctionCreateInteriorByName(ceed, "Scale", &qfRestrict);
  CeedChk(ierr);
  CeedInt *ncompRData;
  ierr = CeedCalloc(1, &ncompRData); CeedChk(ierr);
  ncompRData[0] = ncomp;
  CeedQFunctionContext ctxR;
  ierr = CeedQFunctionContextCreate(ceed, &ctxR); CeedChk(ierr);
  ierr = CeedQFunctionContextSetData(ctxR, CEED_MEM_HOST, CEED_OWN_POINTER,
                                     sizeof(*ncompRData), ncompRData);
  CeedChk(ierr);
  ierr = CeedQFunctionSetContext(qfRestrict, ctxR); CeedChk(ierr);
  ierr = CeedQFunctionContextDestroy(&ctxR); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qfRestrict, "input", ncomp, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qfRestrict, "scale", ncomp, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qfRestrict, "output", ncomp, CEED_EVAL_INTERP);
  CeedChk(ierr);

  ierr = CeedOperatorCreate(ceed, qfRestrict, CEED_QFUNCTION_NONE,
                            CEED_QFUNCTION_NONE, opRestrict);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*opRestrict, "input", rstrFine,
                              CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*opRestrict, "scale", rstrFine,
                              CEED_BASIS_COLLOCATED, multVec);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*opRestrict, "output", rstrCoarse, basisCtoF,
                              CEED_VECTOR_ACTIVE); CeedChk(ierr);

  // Prolongation
  CeedQFunction qfProlong;
  ierr = CeedQFunctionCreateInteriorByName(ceed, "Scale", &qfProlong);
  CeedChk(ierr);
  CeedInt *ncompPData;
  ierr = CeedCalloc(1, &ncompPData); CeedChk(ierr);
  ncompPData[0] = ncomp;
  CeedQFunctionContext ctxP;
  ierr = CeedQFunctionContextCreate(ceed, &ctxP); CeedChk(ierr);
  ierr = CeedQFunctionContextSetData(ctxP, CEED_MEM_HOST, CEED_OWN_POINTER,
                                     sizeof(*ncompPData), ncompPData);
  CeedChk(ierr);
  ierr = CeedQFunctionSetContext(qfProlong, ctxP); CeedChk(ierr);
  ierr = CeedQFunctionContextDestroy(&ctxP); CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qfProlong, "input", ncomp, CEED_EVAL_INTERP);
  CeedChk(ierr);
  ierr = CeedQFunctionAddInput(qfProlong, "scale", ncomp, CEED_EVAL_NONE);
  CeedChk(ierr);
  ierr = CeedQFunctionAddOutput(qfProlong, "output", ncomp, CEED_EVAL_NONE);
  CeedChk(ierr);

  ierr = CeedOperatorCreate(ceed, qfProlong, CEED_QFUNCTION_NONE,
                            CEED_QFUNCTION_NONE, opProlong);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*opProlong, "input", rstrCoarse, basisCtoF,
                              CEED_VECTOR_ACTIVE); CeedChk(ierr);
  ierr = CeedOperatorSetField(*opProlong, "scale", rstrFine,
                              CEED_BASIS_COLLOCATED, multVec);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*opProlong, "output", rstrFine,
                              CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedChk(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&multVec); CeedChk(ierr);
  ierr = CeedBasisDestroy(&basisCtoF); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&qfRestrict); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&qfProlong); CeedChk(ierr);
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

  @param op              CeedOperator
  @param[out] ceed       Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetCeed(CeedOperator op, Ceed *ceed) {
  *ceed = op->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of elements associated with a CeedOperator

  @param op              CeedOperator
  @param[out] numelem    Variable to store number of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetNumElements(CeedOperator op, CeedInt *numelem) {
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP

  *numelem = op->numelements;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of quadrature points associated with a CeedOperator

  @param op              CeedOperator
  @param[out] numqpts    Variable to store vector number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetNumQuadraturePoints(CeedOperator op, CeedInt *numqpts) {
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP

  *numqpts = op->numqpoints;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of arguments associated with a CeedOperator

  @param op              CeedOperator
  @param[out] numargs    Variable to store vector number of arguments

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetNumArgs(CeedOperator op, CeedInt *numargs) {
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operators");
  // LCOV_EXCL_STOP

  *numargs = op->nfields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the setup status of a CeedOperator

  @param op                CeedOperator
  @param[out] issetupdone  Variable to store setup status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorIsSetupDone(CeedOperator op, bool *issetupdone) {
  *issetupdone = op->backendsetup;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the QFunction associated with a CeedOperator

  @param op              CeedOperator
  @param[out] qf         Variable to store QFunction

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

  @param op                CeedOperator
  @param[out] iscomposite  Variable to store composite status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorIsComposite(CeedOperator op, bool *iscomposite) {
  *iscomposite = op->composite;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of suboperators associated with a CeedOperator

  @param op              CeedOperator
  @param[out] numsub     Variable to store number of suboperators

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetNumSub(CeedOperator op, CeedInt *numsub) {
  if (!op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR, "Not a composite operator");
  // LCOV_EXCL_STOP

  *numsub = op->numsub;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the list of suboperators associated with a CeedOperator

  @param op                CeedOperator
  @param[out] suboperators Variable to store list of suboperators

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetSubList(CeedOperator op, CeedOperator **suboperators) {
  if (!op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR, "Not a composite operator");
  // LCOV_EXCL_STOP

  *suboperators = op->suboperators;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a CeedOperator

  @param op              CeedOperator
  @param[out] data       Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetData(CeedOperator op, void *data) {
  *(void **)data = op->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a CeedOperator

  @param[out] op         CeedOperator
  @param data            Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorSetData(CeedOperator op, void *data) {
  op->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the setup flag of a CeedOperator to True

  @param op              CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorSetSetupDone(CeedOperator op) {
  op->backendsetup = true;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedOperatorFields of a CeedOperator

  @param op                 CeedOperator
  @param[out] inputfields   Variable to store inputfields
  @param[out] outputfields  Variable to store outputfields

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetFields(CeedOperator op, CeedOperatorField **inputfields,
                          CeedOperatorField **outputfields) {
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Not defined for composite operator");
  // LCOV_EXCL_STOP

  if (inputfields) *inputfields = op->inputfields;
  if (outputfields) *outputfields = op->outputfields;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedElemRestriction of a CeedOperatorField

  @param opfield         CeedOperatorField
  @param[out] rstr       Variable to store CeedElemRestriction

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorFieldGetElemRestriction(CeedOperatorField opfield,
                                        CeedElemRestriction *rstr) {
  *rstr = opfield->Erestrict;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedBasis of a CeedOperatorField

  @param opfield         CeedOperatorField
  @param[out] basis      Variable to store CeedBasis

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorFieldGetBasis(CeedOperatorField opfield, CeedBasis *basis) {
  *basis = opfield->basis;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the CeedVector of a CeedOperatorField

  @param opfield         CeedOperatorField
  @param[out] vec        Variable to store CeedVector

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorFieldGetVector(CeedOperatorField opfield, CeedVector *vec) {
  *vec = opfield->vec;
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

  @param ceed    A Ceed object where the CeedOperator will be created
  @param qf      QFunction defining the action of the operator at quadrature points
  @param dqf     QFunction defining the action of the Jacobian of @a qf (or
                   @ref CEED_QFUNCTION_NONE)
  @param dqfT    QFunction defining the action of the transpose of the Jacobian
                   of @a qf (or @ref CEED_QFUNCTION_NONE)
  @param[out] op Address of the variable where the newly created
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
  ceed->refcount++;
  (*op)->refcount = 1;
  (*op)->qf = qf;
  qf->refcount++;
  if (dqf && dqf != CEED_QFUNCTION_NONE) {
    (*op)->dqf = dqf;
    dqf->refcount++;
  }
  if (dqfT && dqfT != CEED_QFUNCTION_NONE) {
    (*op)->dqfT = dqfT;
    dqfT->refcount++;
  }
  ierr = CeedCalloc(16, &(*op)->inputfields); CeedChk(ierr);
  ierr = CeedCalloc(16, &(*op)->outputfields); CeedChk(ierr);
  ierr = ceed->OperatorCreate(*op); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create an operator that composes the action of several operators

  @param ceed    A Ceed object where the CeedOperator will be created
  @param[out] op Address of the variable where the newly created
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
  ceed->refcount++;
  (*op)->composite = true;
  ierr = CeedCalloc(16, &(*op)->suboperators); CeedChk(ierr);

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

  @param op         CeedOperator on which to provide the field
  @param fieldname  Name of the field (to be matched with the name used by
                      CeedQFunction)
  @param r          CeedElemRestriction
  @param b          CeedBasis in which the field resides or @ref CEED_BASIS_COLLOCATED
                      if collocated with quadrature points
  @param v          CeedVector to be used by CeedOperator or @ref CEED_VECTOR_ACTIVE
                      if field is active or @ref CEED_VECTOR_NONE if using
                      @ref CEED_EVAL_WEIGHT in the QFunction

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorSetField(CeedOperator op, const char *fieldname,
                         CeedElemRestriction r, CeedBasis b, CeedVector v) {
  int ierr;
  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Cannot add field to composite operator.");
  // LCOV_EXCL_STOP
  if (!r)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "ElemRestriction r for field \"%s\" must be non-NULL.",
                     fieldname);
  // LCOV_EXCL_STOP
  if (!b)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Basis b for field \"%s\" must be non-NULL.",
                     fieldname);
  // LCOV_EXCL_STOP
  if (!v)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_MINOR,
                     "Vector v for field \"%s\" must be non-NULL.",
                     fieldname);
  // LCOV_EXCL_STOP

  CeedInt numelements;
  ierr = CeedElemRestrictionGetNumElements(r, &numelements); CeedChk(ierr);
  if (r != CEED_ELEMRESTRICTION_NONE && op->hasrestriction &&
      op->numelements != numelements)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_DIMENSION,
                     "ElemRestriction with %d elements incompatible with prior "
                     "%d elements", numelements, op->numelements);
  // LCOV_EXCL_STOP

  CeedInt numqpoints;
  if (b != CEED_BASIS_COLLOCATED) {
    ierr = CeedBasisGetNumQuadraturePoints(b, &numqpoints); CeedChk(ierr);
    if (op->numqpoints && op->numqpoints != numqpoints)
      // LCOV_EXCL_START
      return CeedError(op->ceed, CEED_ERROR_DIMENSION,
                       "Basis with %d quadrature points "
                       "incompatible with prior %d points", numqpoints,
                       op->numqpoints);
    // LCOV_EXCL_STOP
  }
  CeedQFunctionField qfield;
  CeedOperatorField *ofield;
  for (CeedInt i=0; i<op->qf->numinputfields; i++) {
    if (!strcmp(fieldname, (*op->qf->inputfields[i]).fieldname)) {
      qfield = op->qf->inputfields[i];
      ofield = &op->inputfields[i];
      goto found;
    }
  }
  for (CeedInt i=0; i<op->qf->numoutputfields; i++) {
    if (!strcmp(fieldname, (*op->qf->outputfields[i]).fieldname)) {
      qfield = op->qf->outputfields[i];
      ofield = &op->outputfields[i];
      goto found;
    }
  }
  // LCOV_EXCL_START
  return CeedError(op->ceed, CEED_ERROR_INCOMPLETE,
                   "QFunction has no knowledge of field '%s'",
                   fieldname);
  // LCOV_EXCL_STOP
found:
  ierr = CeedOperatorCheckField(op->ceed, qfield, r, b); CeedChk(ierr);
  ierr = CeedCalloc(1, ofield); CeedChk(ierr);

  (*ofield)->vec = v;
  if (v != CEED_VECTOR_ACTIVE && v != CEED_VECTOR_NONE) {
    v->refcount += 1;
  }

  (*ofield)->Erestrict = r;
  r->refcount += 1;
  if (r != CEED_ELEMRESTRICTION_NONE) {
    op->numelements = numelements;
    op->hasrestriction = true; // Restriction set, but numelements may be 0
  }

  (*ofield)->basis = b;
  if (b != CEED_BASIS_COLLOCATED) {
    op->numqpoints = numqpoints;
    b->refcount += 1;
  }

  op->nfields += 1;
  size_t len = strlen(fieldname);
  char *tmp;
  ierr = CeedCalloc(len+1, &tmp); CeedChk(ierr);
  memcpy(tmp, fieldname, len+1);
  (*ofield)->fieldname = tmp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add a sub-operator to a composite CeedOperator

  @param[out] compositeop Composite CeedOperator
  @param      subop       Sub-operator CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref User
 */
int CeedCompositeOperatorAddSub(CeedOperator compositeop, CeedOperator subop) {
  if (!compositeop->composite)
    // LCOV_EXCL_START
    return CeedError(compositeop->ceed, CEED_ERROR_MINOR,
                     "CeedOperator is not a composite operator");
  // LCOV_EXCL_STOP

  if (compositeop->numsub == CEED_COMPOSITE_MAX)
    // LCOV_EXCL_START
    return CeedError(compositeop->ceed, CEED_ERROR_UNSUPPORTED,
                     "Cannot add additional suboperators");
  // LCOV_EXCL_STOP

  compositeop->suboperators[compositeop->numsub] = subop;
  subop->refcount++;
  compositeop->numsub++;
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

  @param op             CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedQFunction at
                          quadrature points
  @param[out] rstr      CeedElemRestriction for CeedVector containing assembled
                          CeedQFunction
  @param request        Address of CeedRequest for non-blocking completion, else
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
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssembleQFunction(op->opfallback, assembled,
           rstr, request);
  }
}

/**
  @brief Assemble the diagonal of a square linear CeedOperator

  This overwrites a CeedVector with the diagonal of a linear CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

  @param op             CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedOperator diagonal
  @param request        Address of CeedRequest for non-blocking completion, else
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
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssembleDiagonal(op->opfallback, assembled,
           request);
  }
}

/**
  @brief Assemble the diagonal of a square linear CeedOperator

  This sums into a CeedVector the diagonal of a linear CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

  @param op             CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedOperator diagonal
  @param request        Address of CeedRequest for non-blocking completion, else
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
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssembleAddDiagonal(op->opfallback, assembled,
           request);
  }
}

/**
  @brief Assemble the point block diagonal of a square linear CeedOperator

  This overwrites a CeedVector with the point block diagonal of a linear
    CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

  @param op             CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedOperator point block
                          diagonal, provided in row-major form with an
                          @a ncomp * @a ncomp block at each node. The dimensions
                          of this vector are derived from the active vector
                          for the CeedOperator. The array has shape
                          [nodes, component out, component in].
  @param request        Address of CeedRequest for non-blocking completion, else
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
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssemblePointBlockDiagonal(op->opfallback,
           assembled, request);
  }
}

/**
  @brief Assemble the point block diagonal of a square linear CeedOperator

  This sums into a CeedVector with the point block diagonal of a linear
    CeedOperator.

  Note: Currently only non-composite CeedOperators with a single field and
          composite CeedOperators with single field sub-operators are supported.

  @param op             CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedOperator point block
                          diagonal, provided in row-major form with an
                          @a ncomp * @a ncomp block at each node. The dimensions
                          of this vector are derived from the active vector
                          for the CeedOperator. The array has shape
                          [nodes, component out, component in].
  @param request        Address of CeedRequest for non-blocking completion, else
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
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    return CeedOperatorLinearAssembleAddPointBlockDiagonal(op->opfallback,
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

  CeedElemRestriction rstrin;
  ierr = CeedOperatorGetActiveElemRestriction(op, &rstrin); CeedChk(ierr);
  CeedInt nelem, elemsize, nnodes, ncomp;
  ierr = CeedElemRestrictionGetNumElements(rstrin, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstrin, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetLVectorSize(rstrin, &nnodes); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstrin, &ncomp); CeedChk(ierr);
  CeedInt layout_er[3];
  ierr = CeedElemRestrictionGetELayout(rstrin, &layout_er); CeedChk(ierr);

  CeedInt local_nentries = elemsize*ncomp * elemsize*ncomp * nelem;

  // Determine elem_dof relation
  CeedVector index_vec;
  ierr = CeedVectorCreate(ceed, nnodes, &index_vec); CeedChk(ierr);
  CeedScalar *array;
  ierr = CeedVectorGetArray(index_vec, CEED_MEM_HOST, &array); CeedChk(ierr);
  for (CeedInt i = 0; i < nnodes; ++i) {
    array[i] = i;
  }
  ierr = CeedVectorRestoreArray(index_vec, &array); CeedChk(ierr);
  CeedVector elem_dof;
  ierr = CeedVectorCreate(ceed, nelem * elemsize * ncomp, &elem_dof);
  CeedChk(ierr);
  ierr = CeedVectorSetValue(elem_dof, 0.0); CeedChk(ierr);
  CeedElemRestrictionApply(rstrin, CEED_NOTRANSPOSE, index_vec,
                           elem_dof, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  const CeedScalar *elem_dof_a;
  ierr = CeedVectorGetArrayRead(elem_dof, CEED_MEM_HOST, &elem_dof_a);
  CeedChk(ierr);
  ierr = CeedVectorDestroy(&index_vec); CeedChk(ierr);

  // Determine i, j locations for element matrices
  CeedInt count = 0;
  for (int e = 0; e < nelem; ++e) {
    for (int compin = 0; compin < ncomp; ++compin) {
      for (int compout = 0; compout < ncomp; ++compout) {
        for (int i = 0; i < elemsize; ++i) {
          for (int j = 0; j < elemsize; ++j) {
            const CeedInt edof_index_row = (i)*layout_er[0] +
                                           (compout)*layout_er[1] + e*layout_er[2];
            const CeedInt edof_index_col = (j)*layout_er[0] +
                                           (compin)*layout_er[1] + e*layout_er[2];

            const CeedInt row = elem_dof_a[edof_index_row];
            const CeedInt col = elem_dof_a[edof_index_col];

            rows[offset + count] = row;
            cols[offset + count] = col;
            count++;
          }
        }
      }
    }
  }
  if (count != local_nentries)
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
  CeedInt numinputfields, numoutputfields;
  ierr= CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);
  CeedVector assembledqf;
  CeedElemRestriction rstr_q;
  ierr = CeedOperatorLinearAssembleQFunction(
           op, &assembledqf, &rstr_q, CEED_REQUEST_IMMEDIATE); CeedChk(ierr);

  CeedInt qflength;
  ierr = CeedVectorGetLength(assembledqf, &qflength); CeedChk(ierr);

  CeedOperatorField *input_fields;
  CeedOperatorField *output_fields;
  ierr = CeedOperatorGetFields(op, &input_fields, &output_fields); CeedChk(ierr);

  // Determine active input basis
  CeedQFunctionField *qffields;
  ierr = CeedQFunctionGetFields(qf, &qffields, NULL); CeedChk(ierr);
  CeedInt numemodein = 0, dim = 1;
  CeedEvalMode *emodein = NULL;
  CeedBasis basisin = NULL;
  CeedElemRestriction rstrin = NULL;
  for (CeedInt i=0; i<numinputfields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(input_fields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(input_fields[i], &basisin);
      CeedChk(ierr);
      ierr = CeedBasisGetDimension(basisin, &dim); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(input_fields[i], &rstrin);
      CeedChk(ierr);
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
      CeedChk(ierr);
      switch (emode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(numemodein + 1, &emodein); CeedChk(ierr);
        emodein[numemodein] = emode;
        numemodein += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(numemodein + dim, &emodein); CeedChk(ierr);
        for (CeedInt d=0; d<dim; d++) {
          emodein[numemodein+d] = emode;
        }
        numemodein += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }
  }

  // Determine active output basis
  ierr = CeedQFunctionGetFields(qf, NULL, &qffields); CeedChk(ierr);
  CeedInt numemodeout = 0;
  CeedEvalMode *emodeout = NULL;
  CeedBasis basisout = NULL;
  CeedElemRestriction rstrout = NULL;
  for (CeedInt i=0; i<numoutputfields; i++) {
    CeedVector vec;
    ierr = CeedOperatorFieldGetVector(output_fields[i], &vec); CeedChk(ierr);
    if (vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(output_fields[i], &basisout);
      CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(output_fields[i], &rstrout);
      CeedChk(ierr);
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qffields[i], &emode);
      CeedChk(ierr);
      switch (emode) {
      case CEED_EVAL_NONE:
      case CEED_EVAL_INTERP:
        ierr = CeedRealloc(numemodeout + 1, &emodeout); CeedChk(ierr);
        emodeout[numemodeout] = emode;
        numemodeout += 1;
        break;
      case CEED_EVAL_GRAD:
        ierr = CeedRealloc(numemodeout + dim, &emodeout); CeedChk(ierr);
        for (CeedInt d=0; d<dim; d++) {
          emodeout[numemodeout+d] = emode;
        }
        numemodeout += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }
  }

  CeedInt nelem, elemsize, nqpts, ncomp;
  ierr = CeedElemRestrictionGetNumElements(rstrin, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstrin, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstrin, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisin, &nqpts); CeedChk(ierr);

  CeedInt local_nentries = elemsize*ncomp * elemsize*ncomp * nelem;

  // loop over elements and put in data structure
  const CeedScalar *interpin, *gradin;
  ierr = CeedBasisGetInterp(basisin, &interpin); CeedChk(ierr);
  ierr = CeedBasisGetGrad(basisin, &gradin); CeedChk(ierr);

  const CeedScalar *assembledqfarray;
  ierr = CeedVectorGetArrayRead(assembledqf, CEED_MEM_HOST, &assembledqfarray);
  CeedChk(ierr);

  CeedInt layout_qf[3];
  ierr = CeedElemRestrictionGetELayout(rstr_q, &layout_qf); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_q); CeedChk(ierr);

  // we store Bmat_in, Bmat_out, BTD, elem_mat in row-major order
  CeedScalar Bmat_in[(nqpts * numemodein) * elemsize];
  CeedScalar Bmat_out[(nqpts * numemodeout) * elemsize];
  CeedScalar Dmat[numemodeout * numemodein * nqpts]; // logically 3-tensor
  CeedScalar BTD[elemsize * nqpts*numemodein];
  CeedScalar elem_mat[elemsize * elemsize];
  int count = 0;
  CeedScalar *vals;
  ierr = CeedVectorGetArray(values, CEED_MEM_HOST, &vals); CeedChk(ierr);
  for (int e = 0; e < nelem; ++e) {
    for (int compin = 0; compin < ncomp; ++compin) {
      for (int compout = 0; compout < ncomp; ++compout) {
        for (int ell = 0; ell < (nqpts * numemodein) * elemsize; ++ell) {
          Bmat_in[ell] = 0.0;
        }
        for (int ell = 0; ell < (nqpts * numemodeout) * elemsize; ++ell) {
          Bmat_out[ell] = 0.0;
        }
        // Store block-diagonal D matrix as collection of small dense blocks
        for (int ell = 0; ell < numemodein*numemodeout*nqpts; ++ell) {
          Dmat[ell] = 0.0;
        }
        // form element matrix itself (for each block component)
        for (int ell = 0; ell < elemsize*elemsize; ++ell) {
          elem_mat[ell] = 0.0;
        }
        for (int q = 0; q < nqpts; ++q) {
          for (int n = 0; n < elemsize; ++n) {
            CeedInt din = -1;
            for (int ein = 0; ein < numemodein; ++ein) {
              const int qq = numemodein*q;
              if (emodein[ein] == CEED_EVAL_INTERP) {
                Bmat_in[(qq+ein)*elemsize + n] += interpin[q * elemsize + n];
              } else if (emodein[ein] == CEED_EVAL_GRAD) {
                din += 1;
                Bmat_in[(qq+ein)*elemsize + n] +=
                  gradin[(din*nqpts+q) * elemsize + n];
              } else {
                // LCOV_EXCL_START
                return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Not implemented!");
                // LCOV_EXCL_STOP
              }
            }
            CeedInt dout = -1;
            for (int eout = 0; eout < numemodeout; ++eout) {
              const int qq = numemodeout*q;
              if (emodeout[eout] == CEED_EVAL_INTERP) {
                Bmat_out[(qq+eout)*elemsize + n] += interpin[q * elemsize + n];
              } else if (emodeout[eout] == CEED_EVAL_GRAD) {
                dout += 1;
                Bmat_out[(qq+eout)*elemsize + n] +=
                  gradin[(dout*nqpts+q) * elemsize + n];
              } else {
                // LCOV_EXCL_START
                return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Not implemented!");
                // LCOV_EXCL_STOP
              }
            }
          }
          for (int ei = 0; ei < numemodeout; ++ei) {
            for (int ej = 0; ej < numemodein; ++ej) {
              const int emode_index = ((ei*ncomp+compin)*numemodein+ej)*ncomp+compout;
              const int index = q*layout_qf[0] + emode_index*layout_qf[1] + e*layout_qf[2];
              Dmat[(ei*numemodein+ej)*nqpts + q] += assembledqfarray[index];
            }
          }
        }
        // Compute B^T*D
        for (int ell = 0; ell < elemsize*nqpts*numemodein; ++ell) {
          BTD[ell] = 0.0;
        }
        for (int j = 0; j<elemsize; ++j) {
          for (int q = 0; q<nqpts; ++q) {
            int qq = numemodeout*q;
            for (int ei = 0; ei < numemodein; ++ei) {
              for (int ej = 0; ej < numemodeout; ++ej) {
                BTD[j*(nqpts*numemodein) + (qq+ei)] +=
                  Bmat_out[(qq+ej)*elemsize + j] * Dmat[(ei*numemodein+ej)*nqpts + q];
              }
            }
          }
        }

        ierr = CeedMatrixMultiply(ceed, BTD, Bmat_in, elem_mat, elemsize,
                                  elemsize, nqpts*numemodein); CeedChk(ierr);

        // put element matrix in coordinate data structure
        for (int i = 0; i < elemsize; ++i) {
          for (int j = 0; j < elemsize; ++j) {
            vals[offset + count] = elem_mat[i*elemsize + j];
            count++;
          }
        }
      }
    }
  }
  if (count != local_nentries)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MAJOR, "Error computing entries");
  // LCOV_EXCL_STOP
  ierr = CeedVectorRestoreArray(values, &vals); CeedChk(ierr);

  ierr = CeedVectorRestoreArrayRead(assembledqf, &assembledqfarray);
  CeedChk(ierr);
  ierr = CeedVectorDestroy(&assembledqf); CeedChk(ierr);
  ierr = CeedFree(&emodein); CeedChk(ierr);
  ierr = CeedFree(&emodeout); CeedChk(ierr);

  return CEED_ERROR_SUCCESS;
}

/**
   @ref Utility
**/
int CeedSingleOperatorAssemblyCountEntries(CeedOperator op, CeedInt *nentries) {
  int ierr;
  CeedElemRestriction rstr;
  CeedInt nelem, elemsize, ncomp;

  if (op->composite)
    // LCOV_EXCL_START
    return CeedError(op->ceed, CEED_ERROR_UNSUPPORTED,
                     "Composite operator not supported");
  // LCOV_EXCL_STOP
  ierr = CeedOperatorGetActiveElemRestriction(op, &rstr); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(rstr, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstr, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(rstr, &ncomp); CeedChk(ierr);
  *nentries = elemsize*ncomp * elemsize*ncomp * nelem;

  return CEED_ERROR_SUCCESS;
}

/**
   @brief Fully assemble the nonzero pattern of a linear operator.

   Expected to be used in conjunction with CeedOperatorLinearAssemble()

   The assembly routines use coordinate format, with nentries tuples of the
   form (i, j, value) which indicate that value should be added to the matrix
   in entry (i, j). Note that the (i, j) pairs are not unique and may repeat.
   This function returns the number of entries and their (i, j) locations,
   while CeedOperatorLinearAssemble() provides the values in the same
   ordering.

   This will generally be slow unless your operator is low-order.

   @param[in]  op       CeedOperator to assemble
   @param[out] nentries Number of entries in coordinate nonzero pattern.
   @param[out] rows     Row number for each entry.
   @param[out] cols     Column number for each entry.

   @ref User
**/
int CeedOperatorLinearAssembleSymbolic(CeedOperator op,
                                       CeedInt *nentries, CeedInt **rows, CeedInt **cols) {
  int ierr;
  CeedInt numsub, single_entries;
  CeedOperator *suboperators;
  bool isComposite;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssembleSymbolic) {
    return op->LinearAssembleSymbolic(op, nentries, rows, cols);
  } else {
    // Check for valid fallback resource
    const char *resource, *fallbackresource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallbackresource);
    if (strcmp(fallbackresource, "") && strcmp(resource, fallbackresource)) {
      // Fallback to reference Ceed
      if (!op->opfallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      return CeedOperatorLinearAssembleSymbolic(op->opfallback, nentries, rows, cols);
    }
  }

  // if neither backend nor fallback resource provides
  // LinearAssembleSymbolic, continue with interface-level implementation

  // count entries and allocate rows, cols arrays
  ierr = CeedOperatorIsComposite(op, &isComposite); CeedChk(ierr);
  *nentries = 0;
  if (isComposite) {
    ierr = CeedOperatorGetNumSub(op, &numsub); CeedChk(ierr);
    ierr = CeedOperatorGetSubList(op, &suboperators); CeedChk(ierr);
    for (int k = 0; k < numsub; ++k) {
      ierr = CeedSingleOperatorAssemblyCountEntries(suboperators[k],
             &single_entries); CeedChk(ierr);
      *nentries += single_entries;
    }
  } else {
    ierr = CeedSingleOperatorAssemblyCountEntries(op,
           &single_entries); CeedChk(ierr);
    *nentries += single_entries;
  }
  ierr = CeedCalloc(*nentries, rows); CeedChk(ierr);
  ierr = CeedCalloc(*nentries, cols); CeedChk(ierr);

  // assemble nonzero locations
  CeedInt offset = 0;
  if (isComposite) {
    ierr = CeedOperatorGetNumSub(op, &numsub); CeedChk(ierr);
    ierr = CeedOperatorGetSubList(op, &suboperators); CeedChk(ierr);
    for (int k = 0; k < numsub; ++k) {
      ierr = CeedSingleOperatorAssembleSymbolic(suboperators[k], offset, *rows,
             *cols); CeedChk(ierr);
      ierr = CeedSingleOperatorAssemblyCountEntries(suboperators[k], &single_entries);
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

   The assembly routines use coordinate format, with nentries tuples of the
   form (i, j, value) which indicate that value should be added to the matrix
   in entry (i, j). Note that the (i, j) pairs are not unique and may repeat.
   This function returns the values of the nonzero entries to be added, their
   (i, j) locations are provided by CeedOperatorLinearAssembleSymbolic()

   This will generally be slow unless your operator is low-order.

   @param[in]  op       CeedOperator to assemble
   @param[out] values   Values to assemble into matrix

   @ref User
**/
int CeedOperatorLinearAssemble(CeedOperator op, CeedVector values) {
  int ierr;
  CeedInt numsub, single_entries;
  CeedOperator *suboperators;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssemble) {
    return op->LinearAssemble(op, values);
  } else {
    // Check for valid fallback resource
    const char *resource, *fallbackresource;
    ierr = CeedGetResource(op->ceed, &resource); CeedChk(ierr);
    ierr = CeedGetOperatorFallbackResource(op->ceed, &fallbackresource);
    if (strcmp(fallbackresource, "") && strcmp(resource, fallbackresource)) {
      // Fallback to reference Ceed
      if (!op->opfallback) {
        ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
      }
      // Assemble
      return CeedOperatorLinearAssemble(op->opfallback, values);
    }
  }

  // if neither backend nor fallback resource provides
  // LinearAssemble, continue with interface-level implementation

  bool isComposite;
  ierr = CeedOperatorIsComposite(op, &isComposite); CeedChk(ierr);

  CeedInt offset = 0;
  if (isComposite) {
    ierr = CeedOperatorGetNumSub(op, &numsub); CeedChk(ierr);
    ierr = CeedOperatorGetSubList(op, &suboperators); CeedChk(ierr);
    for (int k = 0; k < numsub; ++k) {
      ierr = CeedSingleOperatorAssemble(suboperators[k], offset, values);
      CeedChk(ierr);
      ierr = CeedSingleOperatorAssemblyCountEntries(suboperators[k], &single_entries);
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

  @param[in] opFine       Fine grid operator
  @param[in] PMultFine    L-vector multiplicity in parallel gather/scatter
  @param[in] rstrCoarse   Coarse grid restriction
  @param[in] basisCoarse  Coarse grid active vector basis
  @param[out] opCoarse    Coarse grid operator
  @param[out] opProlong   Coarse to fine operator
  @param[out] opRestrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreate(CeedOperator opFine, CeedVector PMultFine,
                                     CeedElemRestriction rstrCoarse, CeedBasis basisCoarse,
                                     CeedOperator *opCoarse, CeedOperator *opProlong, CeedOperator *opRestrict) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(opFine, &ceed); CeedChk(ierr);

  // Check for compatible quadrature spaces
  CeedBasis basisFine;
  ierr = CeedOperatorGetActiveBasis(opFine, &basisFine); CeedChk(ierr);
  CeedInt Qf, Qc;
  ierr = CeedBasisGetNumQuadraturePoints(basisFine, &Qf); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisCoarse, &Qc); CeedChk(ierr);
  if (Qf != Qc)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION,
                     "Bases must have compatible quadrature spaces");
  // LCOV_EXCL_STOP

  // Coarse to fine basis
  CeedInt Pf, Pc, Q = Qf;
  bool isTensorF, isTensorC;
  ierr = CeedBasisIsTensor(basisFine, &isTensorF); CeedChk(ierr);
  ierr = CeedBasisIsTensor(basisCoarse, &isTensorC); CeedChk(ierr);
  CeedScalar *interpC, *interpF, *interpCtoF, *tau;
  if (isTensorF && isTensorC) {
    ierr = CeedBasisGetNumNodes1D(basisFine, &Pf); CeedChk(ierr);
    ierr = CeedBasisGetNumNodes1D(basisCoarse, &Pc); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basisCoarse, &Q); CeedChk(ierr);
  } else if (!isTensorF && !isTensorC) {
    ierr = CeedBasisGetNumNodes(basisFine, &Pf); CeedChk(ierr);
    ierr = CeedBasisGetNumNodes(basisCoarse, &Pc); CeedChk(ierr);
  } else {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MINOR,
                     "Bases must both be tensor or non-tensor");
    // LCOV_EXCL_STOP
  }

  ierr = CeedMalloc(Q*Pf, &interpF); CeedChk(ierr);
  ierr = CeedMalloc(Q*Pc, &interpC); CeedChk(ierr);
  ierr = CeedCalloc(Pc*Pf, &interpCtoF); CeedChk(ierr);
  ierr = CeedMalloc(Q, &tau); CeedChk(ierr);
  if (isTensorF) {
    memcpy(interpF, basisFine->interp1d, Q*Pf*sizeof basisFine->interp1d[0]);
    memcpy(interpC, basisCoarse->interp1d, Q*Pc*sizeof basisCoarse->interp1d[0]);
  } else {
    memcpy(interpF, basisFine->interp, Q*Pf*sizeof basisFine->interp[0]);
    memcpy(interpC, basisCoarse->interp, Q*Pc*sizeof basisCoarse->interp[0]);
  }

  // -- QR Factorization, interpF = Q R
  ierr = CeedQRFactorization(ceed, interpF, tau, Q, Pf); CeedChk(ierr);

  // -- Apply Qtranspose, interpC = Qtranspose interpC
  CeedHouseholderApplyQ(interpC, interpF, tau, CEED_TRANSPOSE,
                        Q, Pc, Pf, Pc, 1);

  // -- Apply Rinv, interpCtoF = Rinv interpC
  for (CeedInt j=0; j<Pc; j++) { // Column j
    interpCtoF[j+Pc*(Pf-1)] = interpC[j+Pc*(Pf-1)]/interpF[Pf*Pf-1];
    for (CeedInt i=Pf-2; i>=0; i--) { // Row i
      interpCtoF[j+Pc*i] = interpC[j+Pc*i];
      for (CeedInt k=i+1; k<Pf; k++)
        interpCtoF[j+Pc*i] -= interpF[k+Pf*i]*interpCtoF[j+Pc*k];
      interpCtoF[j+Pc*i] /= interpF[i+Pf*i];
    }
  }
  ierr = CeedFree(&tau); CeedChk(ierr);
  ierr = CeedFree(&interpC); CeedChk(ierr);
  ierr = CeedFree(&interpF); CeedChk(ierr);

  // Complete with interpCtoF versions of code
  if (isTensorF) {
    ierr = CeedOperatorMultigridLevelCreateTensorH1(opFine, PMultFine,
           rstrCoarse, basisCoarse, interpCtoF, opCoarse, opProlong, opRestrict);
    CeedChk(ierr);
  } else {
    ierr = CeedOperatorMultigridLevelCreateH1(opFine, PMultFine,
           rstrCoarse, basisCoarse, interpCtoF, opCoarse, opProlong, opRestrict);
    CeedChk(ierr);
  }

  // Cleanup
  ierr = CeedFree(&interpCtoF); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a tensor basis for the active basis

  @param[in] opFine       Fine grid operator
  @param[in] PMultFine    L-vector multiplicity in parallel gather/scatter
  @param[in] rstrCoarse   Coarse grid restriction
  @param[in] basisCoarse  Coarse grid active vector basis
  @param[in] interpCtoF   Matrix for coarse to fine interpolation
  @param[out] opCoarse    Coarse grid operator
  @param[out] opProlong   Coarse to fine operator
  @param[out] opRestrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreateTensorH1(CeedOperator opFine,
    CeedVector PMultFine, CeedElemRestriction rstrCoarse, CeedBasis basisCoarse,
    const CeedScalar *interpCtoF, CeedOperator *opCoarse,
    CeedOperator *opProlong, CeedOperator *opRestrict) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(opFine, &ceed); CeedChk(ierr);

  // Check for compatible quadrature spaces
  CeedBasis basisFine;
  ierr = CeedOperatorGetActiveBasis(opFine, &basisFine); CeedChk(ierr);
  CeedInt Qf, Qc;
  ierr = CeedBasisGetNumQuadraturePoints(basisFine, &Qf); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisCoarse, &Qc); CeedChk(ierr);
  if (Qf != Qc)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION,
                     "Bases must have compatible quadrature spaces");
  // LCOV_EXCL_STOP

  // Coarse to fine basis
  CeedInt dim, ncomp, nnodesCoarse, P1dFine, P1dCoarse;
  ierr = CeedBasisGetDimension(basisFine, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basisFine, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes1D(basisFine, &P1dFine); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstrCoarse, &nnodesCoarse);
  CeedChk(ierr);
  P1dCoarse = dim == 1 ? nnodesCoarse :
              dim == 2 ? sqrt(nnodesCoarse) :
              cbrt(nnodesCoarse);
  CeedScalar *qref, *qweight, *grad;
  ierr = CeedCalloc(P1dFine, &qref); CeedChk(ierr);
  ierr = CeedCalloc(P1dFine, &qweight); CeedChk(ierr);
  ierr = CeedCalloc(P1dFine*P1dCoarse*dim, &grad); CeedChk(ierr);
  CeedBasis basisCtoF;
  ierr = CeedBasisCreateTensorH1(ceed, dim, ncomp, P1dCoarse, P1dFine,
                                 interpCtoF, grad, qref, qweight, &basisCtoF);
  CeedChk(ierr);
  ierr = CeedFree(&qref); CeedChk(ierr);
  ierr = CeedFree(&qweight); CeedChk(ierr);
  ierr = CeedFree(&grad); CeedChk(ierr);

  // Core code
  ierr = CeedOperatorMultigridLevel_Core(opFine, PMultFine, rstrCoarse,
                                         basisCoarse, basisCtoF, opCoarse,
                                         opProlong, opRestrict);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a non-tensor basis for the active vector

  @param[in] opFine       Fine grid operator
  @param[in] PMultFine    L-vector multiplicity in parallel gather/scatter
  @param[in] rstrCoarse   Coarse grid restriction
  @param[in] basisCoarse  Coarse grid active vector basis
  @param[in] interpCtoF   Matrix for coarse to fine interpolation
  @param[out] opCoarse    Coarse grid operator
  @param[out] opProlong   Coarse to fine operator
  @param[out] opRestrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreateH1(CeedOperator opFine,
                                       CeedVector PMultFine,
                                       CeedElemRestriction rstrCoarse,
                                       CeedBasis basisCoarse,
                                       const CeedScalar *interpCtoF,
                                       CeedOperator *opCoarse,
                                       CeedOperator *opProlong,
                                       CeedOperator *opRestrict) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(opFine, &ceed); CeedChk(ierr);

  // Check for compatible quadrature spaces
  CeedBasis basisFine;
  ierr = CeedOperatorGetActiveBasis(opFine, &basisFine); CeedChk(ierr);
  CeedInt Qf, Qc;
  ierr = CeedBasisGetNumQuadraturePoints(basisFine, &Qf); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisCoarse, &Qc); CeedChk(ierr);
  if (Qf != Qc)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION,
                     "Bases must have compatible quadrature spaces");
  // LCOV_EXCL_STOP

  // Coarse to fine basis
  CeedElemTopology topo;
  ierr = CeedBasisGetTopology(basisFine, &topo); CeedChk(ierr);
  CeedInt dim, ncomp, nnodesCoarse, nnodesFine;
  ierr = CeedBasisGetDimension(basisFine, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basisFine, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basisFine, &nnodesFine); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(rstrCoarse, &nnodesCoarse);
  CeedChk(ierr);
  CeedScalar *qref, *qweight, *grad;
  ierr = CeedCalloc(nnodesFine, &qref); CeedChk(ierr);
  ierr = CeedCalloc(nnodesFine, &qweight); CeedChk(ierr);
  ierr = CeedCalloc(nnodesFine*nnodesCoarse*dim, &grad); CeedChk(ierr);
  CeedBasis basisCtoF;
  ierr = CeedBasisCreateH1(ceed, topo, ncomp, nnodesCoarse, nnodesFine,
                           interpCtoF, grad, qref, qweight, &basisCtoF);
  CeedChk(ierr);
  ierr = CeedFree(&qref); CeedChk(ierr);
  ierr = CeedFree(&qweight); CeedChk(ierr);
  ierr = CeedFree(&grad); CeedChk(ierr);

  // Core code
  ierr = CeedOperatorMultigridLevel_Core(opFine, PMultFine, rstrCoarse,
                                         basisCoarse, basisCtoF, opCoarse,
                                         opProlong, opRestrict);
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

  @param op             CeedOperator to create element inverses
  @param[out] fdminv    CeedOperator to apply the action of a FDM based inverse
                          for each element
  @param request        Address of CeedRequest for non-blocking completion, else
                          @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedOperatorCreateFDMElementInverse(CeedOperator op, CeedOperator *fdminv,
                                        CeedRequest *request) {
  int ierr;
  ierr = CeedOperatorCheckReady(op); CeedChk(ierr);

  // Use backend version, if available
  if (op->CreateFDMElementInverse) {
    ierr = op->CreateFDMElementInverse(op, fdminv, request); CeedChk(ierr);
  } else {
    // Fallback to reference Ceed
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    ierr = op->opfallback->CreateFDMElementInverse(op->opfallback, fdminv,
           request); CeedChk(ierr);
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
  int ierr;

  if (op->composite) {
    fprintf(stream, "Composite CeedOperator\n");

    for (CeedInt i=0; i<op->numsub; i++) {
      fprintf(stream, "  SubOperator [%d]:\n", i);
      ierr = CeedOperatorSingleView(op->suboperators[i], 1, stream);
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

  if (op->numelements)  {
    // Standard Operator
    if (op->Apply) {
      ierr = op->Apply(op, in, out, request); CeedChk(ierr);
    } else {
      // Zero all output vectors
      CeedQFunction qf = op->qf;
      for (CeedInt i=0; i<qf->numoutputfields; i++) {
        CeedVector vec = op->outputfields[i]->vec;
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
      CeedInt numsub;
      ierr = CeedOperatorGetNumSub(op, &numsub); CeedChk(ierr);
      CeedOperator *suboperators;
      ierr = CeedOperatorGetSubList(op, &suboperators); CeedChk(ierr);

      // Zero all output vectors
      if (out != CEED_VECTOR_NONE) {
        ierr = CeedVectorSetValue(out, 0.0); CeedChk(ierr);
      }
      for (CeedInt i=0; i<numsub; i++) {
        for (CeedInt j=0; j<suboperators[i]->qf->numoutputfields; j++) {
          CeedVector vec = suboperators[i]->outputfields[j]->vec;
          if (vec != CEED_VECTOR_ACTIVE && vec != CEED_VECTOR_NONE) {
            ierr = CeedVectorSetValue(vec, 0.0); CeedChk(ierr);
          }
        }
      }
      // Apply
      for (CeedInt i=0; i<op->numsub; i++) {
        ierr = CeedOperatorApplyAdd(op->suboperators[i], in, out, request);
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

  if (op->numelements)  {
    // Standard Operator
    ierr = op->ApplyAdd(op, in, out, request); CeedChk(ierr);
  } else if (op->composite) {
    // Composite Operator
    if (op->ApplyAddComposite) {
      ierr = op->ApplyAddComposite(op, in, out, request); CeedChk(ierr);
    } else {
      CeedInt numsub;
      ierr = CeedOperatorGetNumSub(op, &numsub); CeedChk(ierr);
      CeedOperator *suboperators;
      ierr = CeedOperatorGetSubList(op, &suboperators); CeedChk(ierr);

      for (CeedInt i=0; i<numsub; i++) {
        ierr = CeedOperatorApplyAdd(suboperators[i], in, out, request);
        CeedChk(ierr);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedOperator

  @param op CeedOperator to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorDestroy(CeedOperator *op) {
  int ierr;

  if (!*op || --(*op)->refcount > 0) return CEED_ERROR_SUCCESS;
  if ((*op)->Destroy) {
    ierr = (*op)->Destroy(*op); CeedChk(ierr);
  }
  ierr = CeedDestroy(&(*op)->ceed); CeedChk(ierr);
  // Free fields
  for (int i=0; i<(*op)->nfields; i++)
    if ((*op)->inputfields[i]) {
      if ((*op)->inputfields[i]->Erestrict != CEED_ELEMRESTRICTION_NONE) {
        ierr = CeedElemRestrictionDestroy(&(*op)->inputfields[i]->Erestrict);
        CeedChk(ierr);
      }
      if ((*op)->inputfields[i]->basis != CEED_BASIS_COLLOCATED) {
        ierr = CeedBasisDestroy(&(*op)->inputfields[i]->basis); CeedChk(ierr);
      }
      if ((*op)->inputfields[i]->vec != CEED_VECTOR_ACTIVE &&
          (*op)->inputfields[i]->vec != CEED_VECTOR_NONE ) {
        ierr = CeedVectorDestroy(&(*op)->inputfields[i]->vec); CeedChk(ierr);
      }
      ierr = CeedFree(&(*op)->inputfields[i]->fieldname); CeedChk(ierr);
      ierr = CeedFree(&(*op)->inputfields[i]); CeedChk(ierr);
    }
  for (int i=0; i<(*op)->nfields; i++)
    if ((*op)->outputfields[i]) {
      ierr = CeedElemRestrictionDestroy(&(*op)->outputfields[i]->Erestrict);
      CeedChk(ierr);
      if ((*op)->outputfields[i]->basis != CEED_BASIS_COLLOCATED) {
        ierr = CeedBasisDestroy(&(*op)->outputfields[i]->basis); CeedChk(ierr);
      }
      if ((*op)->outputfields[i]->vec != CEED_VECTOR_ACTIVE &&
          (*op)->outputfields[i]->vec != CEED_VECTOR_NONE ) {
        ierr = CeedVectorDestroy(&(*op)->outputfields[i]->vec); CeedChk(ierr);
      }
      ierr = CeedFree(&(*op)->outputfields[i]->fieldname); CeedChk(ierr);
      ierr = CeedFree(&(*op)->outputfields[i]); CeedChk(ierr);
    }
  // Destroy suboperators
  for (int i=0; i<(*op)->numsub; i++)
    if ((*op)->suboperators[i]) {
      ierr = CeedOperatorDestroy(&(*op)->suboperators[i]); CeedChk(ierr);
    }
  if ((*op)->qf)
    // LCOV_EXCL_START
    (*op)->qf->operatorsset--;
  // LCOV_EXCL_STOP
  ierr = CeedQFunctionDestroy(&(*op)->qf); CeedChk(ierr);
  if ((*op)->dqf && (*op)->dqf != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    (*op)->dqf->operatorsset--;
  // LCOV_EXCL_STOP
  ierr = CeedQFunctionDestroy(&(*op)->dqf); CeedChk(ierr);
  if ((*op)->dqfT && (*op)->dqfT != CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    (*op)->dqfT->operatorsset--;
  // LCOV_EXCL_STOP
  ierr = CeedQFunctionDestroy(&(*op)->dqfT); CeedChk(ierr);

  // Destroy fallback
  if ((*op)->opfallback) {
    ierr = (*op)->qffallback->Destroy((*op)->qffallback); CeedChk(ierr);
    ierr = CeedFree(&(*op)->qffallback); CeedChk(ierr);
    ierr = (*op)->opfallback->Destroy((*op)->opfallback); CeedChk(ierr);
    ierr = CeedFree(&(*op)->opfallback); CeedChk(ierr);
  }

  ierr = CeedFree(&(*op)->inputfields); CeedChk(ierr);
  ierr = CeedFree(&(*op)->outputfields); CeedChk(ierr);
  ierr = CeedFree(&(*op)->suboperators); CeedChk(ierr);
  ierr = CeedFree(op); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}
