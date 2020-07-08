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

#include <ceed-impl.h>
#include <ceed-backend.h>
#include <string.h>
#include <math.h>

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
    return CeedError(op->ceed, 1, "Backend %s cannot create an operator"
                     "fallback to resource %s", resource, fallbackresource);
  // LCOV_EXCL_STOP

  // Fallback Ceed
  Ceed ceedref;
  if (!op->ceed->opfallbackceed) {
    ierr = CeedInit(fallbackresource, &ceedref); CeedChk(ierr);
    ceedref->opfallbackparent = op->ceed;
    op->ceed->opfallbackceed = ceedref;
  }
  ceedref = op->ceed->opfallbackceed;

  // Clone Op
  CeedOperator opref;
  ierr = CeedCalloc(1, &opref); CeedChk(ierr);
  memcpy(opref, op, sizeof(*opref)); CeedChk(ierr);
  opref->data = NULL;
  opref->setupdone = 0;
  opref->ceed = ceedref;
  ierr = ceedref->OperatorCreate(opref); CeedChk(ierr);
  op->opfallback = opref;

  // Clone QF
  CeedQFunction qfref;
  ierr = CeedCalloc(1, &qfref); CeedChk(ierr);
  memcpy(qfref, (op->qf), sizeof(*qfref)); CeedChk(ierr);
  qfref->data = NULL;
  qfref->ceed = ceedref;
  ierr = ceedref->QFunctionCreate(qfref); CeedChk(ierr);
  opref->qf = qfref;
  op->qffallback = qfref;

  return 0;
}

/**
  @brief Check if a CeedOperator is ready to be used.

  @param[in] ceed Ceed object for error handling
  @param[in] op   CeedOperator to check

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedOperatorCheckReady(Ceed ceed, CeedOperator op) {
  CeedQFunction qf = op->qf;

  if (op->composite) {
    if (!op->numsub)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "No suboperators set");
    // LCOV_EXCL_STOP
  } else {
    if (op->nfields == 0)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "No operator fields set");
    // LCOV_EXCL_STOP
    if (op->nfields < qf->numinputfields + qf->numoutputfields)
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Not all operator fields set");
    // LCOV_EXCL_STOP
    if (!op->hasrestriction)
      // LCOV_EXCL_START
      return CeedError(ceed, 1,"At least one restriction required");
    // LCOV_EXCL_STOP
    if (op->numqpoints == 0)
      // LCOV_EXCL_START
      return CeedError(ceed, 1,"At least one non-collocated basis required");
    // LCOV_EXCL_STOP
  }

  return 0;
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

  return 0;
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
    ierr = CeedOperatorFieldView(op->outputfields[i], op->qf->inputfields[i],
                                 i, sub, 0, stream); CeedChk(ierr);
  }

  return 0;
}

/**
  @brief Find the active vector basis for a CeedOperator

  @param[in] op            CeedOperator to find active basis for
  @param[out] activeBasis  Basis for active input vector

  @return An error code: 0 - success, otherwise - failure

  @ ref Developer
**/
static inline int CeedOperatorGetActiveBasis(CeedOperator op,
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
    return CeedError(ceed, 1,
                     "No active basis found for automatic multigrid setup");
    // LCOV_EXCL_STOP
  }
  return 0;
}


/**
  @brief Common code for creating a multigrid coarse operator and level
           transfer operators for a CeedOperator

  @param[in] rstrCoarse   Coarse grid restriction
  @param[in] basisCoarse  Coarse grid active vector basis
  @param[in] basisCtoF    Basis for coarse to fine interpolation
  @param[in] opFine       Fine grid operator
  @param[out] opCoarse    Coarse grid operator
  @param[out] opProlong   Coarse to fine operator
  @param[out] opRestrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static inline int CeedOperatorMultigridLevel_Core(CeedElemRestriction
    rstrCoarse, CeedBasis basisCoarse, CeedBasis basisCtoF,
    CeedOperator opFine, CeedOperator *opCoarse, CeedOperator *opProlong,
    CeedOperator *opRestrict) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(opFine, &ceed); CeedChk(ierr);

  // Check for composite operator
  bool isComposite;
  ierr = CeedOperatorIsComposite(opFine, &isComposite); CeedChk(ierr);
  if (isComposite)
    // LCOV_EXCL_START
    return CeedError(ceed, 1,
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
                                  rstrCoarse, basisCoarse, CEED_VECTOR_ACTIVE); CeedChk(ierr);
    } else {
      ierr = CeedOperatorSetField(*opCoarse, opFine->inputfields[i]->fieldname,
                                  opFine->inputfields[i]->Erestrict, opFine->inputfields[i]->basis,
                                  opFine->inputfields[i]->vec); CeedChk(ierr);
    }
  }
  // -- Clone output fields
  for (int i = 0; i < opFine->qf->numoutputfields; i++) {
    if (opFine->outputfields[i]->vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorSetField(*opCoarse, opFine->outputfields[i]->fieldname,
                                  rstrCoarse, basisCoarse, CEED_VECTOR_ACTIVE); CeedChk(ierr);
    } else {
      ierr = CeedOperatorSetField(*opCoarse, opFine->outputfields[i]->fieldname,
                                  opFine->outputfields[i]->Erestrict, opFine->outputfields[i]->basis,
                                  opFine->outputfields[i]->vec); CeedChk(ierr);
    }
  }

  // Restriction
  CeedInt ncomp;
  ierr = CeedBasisGetNumComponents(basisCoarse, &ncomp); CeedChk(ierr);
  CeedQFunction qfRestrict;
  ierr = CeedQFunctionCreateIdentity(ceed, ncomp, CEED_EVAL_NONE,
                                     CEED_EVAL_INTERP, &qfRestrict);
  CeedChk(ierr);
  ierr = CeedOperatorCreate(ceed, qfRestrict, CEED_QFUNCTION_NONE,
                            CEED_QFUNCTION_NONE, opRestrict);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*opRestrict, "input", rstrFine,
                              CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*opRestrict, "output", rstrCoarse, basisCtoF,
                              CEED_VECTOR_ACTIVE); CeedChk(ierr);

  // Prolongation
  CeedQFunction qfProlong;
  ierr = CeedQFunctionCreateIdentity(ceed, ncomp, CEED_EVAL_INTERP,
                                     CEED_EVAL_NONE, &qfProlong);
  CeedChk(ierr);
  ierr = CeedOperatorCreate(ceed, qfProlong, CEED_QFUNCTION_NONE,
                            CEED_QFUNCTION_NONE, opProlong);
  CeedChk(ierr);
  ierr = CeedOperatorSetField(*opProlong, "input", rstrCoarse, basisCtoF,
                              CEED_VECTOR_ACTIVE); CeedChk(ierr);
  ierr = CeedOperatorSetField(*opProlong, "output", rstrFine,
                              CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedChk(ierr);

  // Cleanup
  ierr = CeedBasisDestroy(&basisCtoF); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&qfRestrict); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&qfProlong); CeedChk(ierr);

  return 0;
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
  return 0;
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
    return CeedError(op->ceed, 1, "Not defined for composite operator");
  // LCOV_EXCL_STOP

  *numelem = op->numelements;
  return 0;
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
    return CeedError(op->ceed, 1, "Not defined for composite operator");
  // LCOV_EXCL_STOP

  *numqpts = op->numqpoints;
  return 0;
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
    return CeedError(op->ceed, 1, "Not defined for composite operators");
  // LCOV_EXCL_STOP

  *numargs = op->nfields;
  return 0;
}

/**
  @brief Get the setup status of a CeedOperator

  @param op                CeedOperator
  @param[out] issetupdone  Variable to store setup status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorIsSetupDone(CeedOperator op, bool *issetupdone) {
  *issetupdone = op->setupdone;
  return 0;
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
    return CeedError(op->ceed, 1, "Not defined for composite operator");
  // LCOV_EXCL_STOP

  *qf = op->qf;
  return 0;
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
  return 0;
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
    return CeedError(op->ceed, 1, "Not a composite operator");
  // LCOV_EXCL_STOP

  *numsub = op->numsub;
  return 0;
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
    return CeedError(op->ceed, 1, "Not a composite operator");
  // LCOV_EXCL_STOP

  *suboperators = op->suboperators;
  return 0;
}

/**
  @brief Get the backend data of a CeedOperator

  @param op              CeedOperator
  @param[out] data       Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorGetData(CeedOperator op, void **data) {
  *data = op->data;
  return 0;
}

/**
  @brief Set the backend data of a CeedOperator

  @param[out] op         CeedOperator
  @param data            Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorSetData(CeedOperator op, void **data) {
  op->data = *data;
  return 0;
}

/**
  @brief Set the setup flag of a CeedOperator to True

  @param op              CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/

int CeedOperatorSetSetupDone(CeedOperator op) {
  op->setupdone = 1;
  return 0;
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
    return CeedError(op->ceed, 1, "Not defined for composite operator");
  // LCOV_EXCL_STOP

  if (inputfields) *inputfields = op->inputfields;
  if (outputfields) *outputfields = op->outputfields;
  return 0;
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
  return 0;
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
  return 0;
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
  return 0;
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
      return CeedError(ceed, 1, "Backend does not support OperatorCreate");
    // LCOV_EXCL_STOP

    ierr = CeedOperatorCreate(delegate, qf, dqf, dqfT, op); CeedChk(ierr);
    return 0;
  }

  if (!qf || qf == CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Operator must have a valid QFunction.");
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
  return 0;
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
      return 0;
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
  return 0;
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
    return CeedError(op->ceed, 1, "Cannot add field to composite operator.");
  // LCOV_EXCL_STOP
  if (!r)
    // LCOV_EXCL_START
    return CeedError(op->ceed, 1,
                     "ElemRestriction r for field \"%s\" must be non-NULL.",
                     fieldname);
  // LCOV_EXCL_STOP
  if (!b)
    // LCOV_EXCL_START
    return CeedError(op->ceed, 1, "Basis b for field \"%s\" must be non-NULL.",
                     fieldname);
  // LCOV_EXCL_STOP
  if (!v)
    // LCOV_EXCL_START
    return CeedError(op->ceed, 1, "Vector v for field \"%s\" must be non-NULL.",
                     fieldname);
  // LCOV_EXCL_STOP

  CeedInt numelements;
  ierr = CeedElemRestrictionGetNumElements(r, &numelements); CeedChk(ierr);
  if (r != CEED_ELEMRESTRICTION_NONE && op->hasrestriction &&
      op->numelements != numelements)
    // LCOV_EXCL_START
    return CeedError(op->ceed, 1,
                     "ElemRestriction with %d elements incompatible with prior "
                     "%d elements", numelements, op->numelements);
  // LCOV_EXCL_STOP
  if (r != CEED_ELEMRESTRICTION_NONE) {
    op->numelements = numelements;
    op->hasrestriction = true; // Restriction set, but numelements may be 0
  }

  if (b != CEED_BASIS_COLLOCATED) {
    CeedInt numqpoints;
    ierr = CeedBasisGetNumQuadraturePoints(b, &numqpoints); CeedChk(ierr);
    if (op->numqpoints && op->numqpoints != numqpoints)
      // LCOV_EXCL_START
      return CeedError(op->ceed, 1, "Basis with %d quadrature points "
                       "incompatible with prior %d points", numqpoints,
                       op->numqpoints);
    // LCOV_EXCL_STOP
    op->numqpoints = numqpoints;
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
      qfield = op->qf->inputfields[i];
      ofield = &op->outputfields[i];
      goto found;
    }
  }
  // LCOV_EXCL_START
  return CeedError(op->ceed, 1, "QFunction has no knowledge of field '%s'",
                   fieldname);
  // LCOV_EXCL_STOP
found:
  if (r == CEED_ELEMRESTRICTION_NONE && qfield->emode != CEED_EVAL_WEIGHT)
    // LCOV_EXCL_START
    return CeedError(op->ceed, 1, "CEED_ELEMRESTRICTION_NONE can only be used "
                     "for a field with eval mode CEED_EVAL_WEIGHT");
  // LCOV_EXCL_STOP
  ierr = CeedCalloc(1, ofield); CeedChk(ierr);
  (*ofield)->Erestrict = r;
  r->refcount += 1;
  (*ofield)->basis = b;
  if (b != CEED_BASIS_COLLOCATED)
    b->refcount += 1;
  (*ofield)->vec = v;
  if (v != CEED_VECTOR_ACTIVE && v != CEED_VECTOR_NONE)
    v->refcount += 1;
  op->nfields += 1;

  size_t len = strlen(fieldname);
  char *tmp;
  ierr = CeedCalloc(len+1, &tmp); CeedChk(ierr);
  memcpy(tmp, fieldname, len+1);
  (*ofield)->fieldname = tmp;
  return 0;
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
    return CeedError(compositeop->ceed, 1, "CeedOperator is not a composite "
                     "operator");
  // LCOV_EXCL_STOP

  if (compositeop->numsub == CEED_COMPOSITE_MAX)
    // LCOV_EXCL_START
    return CeedError(compositeop->ceed, 1, "Cannot add additional suboperators");
  // LCOV_EXCL_STOP

  compositeop->suboperators[compositeop->numsub] = subop;
  subop->refcount++;
  compositeop->numsub++;
  return 0;
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
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

  // Backend version
  if (op->LinearAssembleQFunction) {
    ierr = op->LinearAssembleQFunction(op, assembled, rstr, request);
    CeedChk(ierr);
  } else {
    // Fallback to reference Ceed
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    ierr = op->opfallback->LinearAssembleQFunction(op->opfallback, assembled,
           rstr, request); CeedChk(ierr);
  }

  return 0;
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
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssembleDiagonal) {
    ierr = op->LinearAssembleDiagonal(op, assembled, request); CeedChk(ierr);
  } else if (op->LinearAssembleAddDiagonal) {
    ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
    return CeedOperatorLinearAssembleAddDiagonal(op, assembled, request);
  } else {
    // Fallback to reference Ceed
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    if (op->opfallback->LinearAssembleDiagonal) {
      ierr = op->opfallback->LinearAssembleDiagonal(op->opfallback, assembled,
             request); CeedChk(ierr);
    } else {
      ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
      return CeedOperatorLinearAssembleAddDiagonal(op, assembled, request);
    }
  }

  return 0;
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
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssembleAddDiagonal) {
    ierr = op->LinearAssembleAddDiagonal(op, assembled, request); CeedChk(ierr);
  } else {
    // Fallback to reference Ceed
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
    ierr = op->opfallback->LinearAssembleAddDiagonal(op->opfallback, assembled,
           request); CeedChk(ierr);
  }

  return 0;
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
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssemblePointBlockDiagonal) {
    ierr = op->LinearAssemblePointBlockDiagonal(op, assembled, request);
    CeedChk(ierr);
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
    if (op->opfallback->LinearAssemblePointBlockDiagonal) {
      ierr = op->opfallback->LinearAssemblePointBlockDiagonal(op->opfallback,
             assembled, request); CeedChk(ierr);
    } else {
      ierr = CeedVectorSetValue(assembled, 0.0); CeedChk(ierr);
      return CeedOperatorLinearAssembleAddPointBlockDiagonal(op, assembled,
             request);
    }
  }

  return 0;
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
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

  // Use backend version, if available
  if (op->LinearAssembleAddPointBlockDiagonal) {
    ierr = op->LinearAssembleAddPointBlockDiagonal(op, assembled, request);
    CeedChk(ierr);
  } else {
    // Fallback to reference Ceed
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    ierr = op->opfallback->LinearAssembleAddPointBlockDiagonal(op->opfallback,
           assembled, request); CeedChk(ierr);
  }

  return 0;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a Lagrange tensor basis for the active basis

  @param[in] rstrCoarse   Coarse grid restriction
  @param[in] basisCoarse  Coarse grid active vector basis
  @param[in] interpCtoF   Matrix for coarse to fine interpolation
  @param[in] opFine       Fine grid operator
  @param[out] opCoarse    Coarse grid operator
  @param[out] opProlong   Coarse to fine operator
  @param[out] opRestrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreateTensorH1Lagrange(CeedElemRestriction
    rstrCoarse, CeedInt degreeCoarse, CeedOperator opFine,
    CeedOperator *opCoarse, CeedOperator *opProlong,
    CeedOperator *opRestrict) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(opFine, &ceed); CeedChk(ierr);

  // Coarse to fine basis
  CeedBasis basisFine;
  ierr = CeedOperatorGetActiveBasis(opFine, &basisFine); CeedChk(ierr);
  CeedInt dim, ncomp, P1dFine;
  ierr = CeedBasisGetDimension(basisFine, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basisFine, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes1D(basisFine, &P1dFine); CeedChk(ierr);
  CeedBasis basisCtoF;
  ierr = CeedBasisCreateTensorH1Lagrange(ceed, dim, ncomp, degreeCoarse, P1dFine,
                                         CEED_GAUSS_LOBATTO, &basisCtoF);
  CeedChk(ierr);

  // Coarse basis
  CeedBasis basisCoarse;
  CeedInt Q1dFine;
  ierr = CeedBasisGetNumQuadraturePoints1D(basisFine, &Q1dFine); CeedChk(ierr);
  const CeedScalar *qref;
  ierr = CeedBasisGetQRef(basisFine, &qref); CeedChk(ierr);
  CeedQuadMode qmode = fabs(qref[0] + 1) > CEED_EPSILON ?
                       CEED_GAUSS_LOBATTO : CEED_GAUSS;
  ierr = CeedBasisCreateTensorH1Lagrange(ceed, dim, ncomp, degreeCoarse, Q1dFine,
                                         qmode, &basisCoarse);
  CeedChk(ierr);

  // Core code
  ierr = CeedOperatorMultigridLevel_Core(rstrCoarse, basisCoarse, basisCtoF,
                                         opFine, opCoarse, opProlong, opRestrict);
  CeedChk(ierr);
  return 0;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a tensor basis for the active basis

  @param[in] rstrCoarse   Coarse grid restriction
  @param[in] basisCoarse  Coarse grid active vector basis
  @param[in] interpCtoF   Matrix for coarse to fine interpolation
  @param[in] opFine       Fine grid operator
  @param[out] opCoarse    Coarse grid operator
  @param[out] opProlong   Coarse to fine operator
  @param[out] opRestrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreateTensorH1(CeedElemRestriction rstrCoarse,
    CeedBasis basisCoarse, const CeedScalar *interpCtoF, CeedOperator opFine,
    CeedOperator *opCoarse, CeedOperator *opProlong, CeedOperator *opRestrict) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(opFine, &ceed); CeedChk(ierr);

  // Coarse to fine basis
  CeedBasis basisFine;
  ierr = CeedOperatorGetActiveBasis(opFine, &basisFine); CeedChk(ierr);
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
  ierr = CeedOperatorMultigridLevel_Core(rstrCoarse, basisCoarse, basisCtoF,
                                         opFine, opCoarse, opProlong, opRestrict);
  CeedChk(ierr);
  return 0;
}

/**
  @brief Create a multigrid coarse operator and level transfer operators
           for a CeedOperator with a non-tensor basis for the active vector

  @param[in] rstrCoarse   Coarse grid restriction
  @param[in] basisCoarse  Coarse grid active vector basis
  @param[in] interpCtoF   Matrix for coarse to fine interpolation
  @param[in] opFine       Fine grid operator
  @param[out] opCoarse    Coarse grid operator
  @param[out] opProlong   Coarse to fine operator
  @param[out] opRestrict  Fine to coarse operator

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorMultigridLevelCreateH1(CeedElemRestriction rstrCoarse,
                                       CeedBasis basisCoarse,
                                       const CeedScalar *interpCtoF,
                                       CeedOperator opFine,
                                       CeedOperator *opCoarse,
                                       CeedOperator *opProlong,
                                       CeedOperator *opRestrict) {
  int ierr;
  Ceed ceed;
  ierr = CeedOperatorGetCeed(opFine, &ceed); CeedChk(ierr);

  // Coarse to fine basis
  CeedBasis basisFine;
  ierr = CeedOperatorGetActiveBasis(opFine, &basisFine); CeedChk(ierr);
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
  ierr = CeedOperatorMultigridLevel_Core(rstrCoarse, basisCoarse, basisCtoF,
                                         opFine, opCoarse, opProlong, opRestrict);
  CeedChk(ierr);
  return 0;
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
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

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

  return 0;
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

  return 0;
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
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

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

  return 0;
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
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

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

  return 0;
}

/**
  @brief Destroy a CeedOperator

  @param op CeedOperator to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedOperatorDestroy(CeedOperator *op) {
  int ierr;

  if (!*op || --(*op)->refcount > 0) return 0;
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
  ierr = CeedQFunctionDestroy(&(*op)->qf); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&(*op)->dqf); CeedChk(ierr);
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
  return 0;
}

/// @}
