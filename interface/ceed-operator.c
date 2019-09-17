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

/// @file
/// Implementation of public CeedOperator interfaces
///
/// @addtogroup CeedOperator
///   @{

/**
  @brief Create a CeedOperator and associate a CeedQFunction. A CeedBasis and
           CeedElemRestriction can be associated with CeedQFunction fields with
           \ref CeedOperatorSetField.

  @param ceed    A Ceed object where the CeedOperator will be created
  @param qf      QFunction defining the action of the operator at quadrature points
  @param dqf     QFunction defining the action of the Jacobian of @a qf (or NULL)
  @param dqfT    QFunction defining the action of the transpose of the Jacobian
                   of @a qf (or NULL)
  @param[out] op Address of the variable where the newly created
                     CeedOperator will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
 */
int CeedOperatorCreate(Ceed ceed, CeedQFunction qf, CeedQFunction dqf,
                       CeedQFunction dqfT, CeedOperator *op) {
  int ierr;

  if (!ceed->OperatorCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Operator"); CeedChk(ierr);

    if (!delegate)
      return CeedError(ceed, 1, "Backend does not support OperatorCreate");

    ierr = CeedOperatorCreate(delegate, qf, dqf, dqfT, op); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1,op); CeedChk(ierr);
  (*op)->ceed = ceed;
  ceed->refcount++;
  (*op)->refcount = 1;
  (*op)->qf = qf;
  qf->refcount++;
  (*op)->dqf = dqf;
  if (dqf) dqf->refcount++;
  (*op)->dqfT = dqfT;
  if (dqfT) dqfT->refcount++;
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

  @ref Basic
 */
int CeedCompositeOperatorCreate(Ceed ceed, CeedOperator *op) {
  int ierr;

  if (!ceed->CompositeOperatorCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "Operator"); CeedChk(ierr);

    if (!delegate)
      return CeedError(ceed, 1, "Backend does not support \
                                   CompositeOperatorCreate");

    ierr = CeedCompositeOperatorCreate(delegate, op); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1,op); CeedChk(ierr);
  (*op)->ceed = ceed;
  ceed->refcount++;
  (*op)->composite = true;
  ierr = CeedCalloc(16, &(*op)->suboperators); CeedChk(ierr);
  ierr = ceed->CompositeOperatorCreate(*op); CeedChk(ierr);
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
  @param lmode      CeedTransposeMode which specifies the ordering of the
                      components of the l-vector used by this CeedOperatorField,
                      CEED_NOTRANSPOSE indicates the component is the
                      outermost index and CEED_TRANSPOSE indicates the component
                      is the innermost index in ordering of the l-vector
  @param b          CeedBasis in which the field resides or CEED_BASIS_COLLOCATED
                      if collocated with quadrature points
  @param v          CeedVector to be used by CeedOperator or CEED_VECTOR_ACTIVE
                      if field is active or CEED_VECTOR_NONE if using
                      CEED_EVAL_WEIGHT in the QFunction

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedOperatorSetField(CeedOperator op, const char *fieldname,
                         CeedElemRestriction r, CeedTransposeMode lmode,
                         CeedBasis b, CeedVector v) {
  int ierr;
  if (op->composite)
    return CeedError(op->ceed, 1, "Cannot add field to composite operator.");
  if (!r)
    return CeedError(op->ceed, 1, "ElemRestriction r for field \"%s\" must be non-NULL.", fieldname);
  if (!b)
    return CeedError(op->ceed, 1, "Basis b for field \"%s\" must be non-NULL.", fieldname);
  if (!v)
    return CeedError(op->ceed, 1, "Vector v for field \"%s\" must be non-NULL.", fieldname);

  CeedInt numelements;
  ierr = CeedElemRestrictionGetNumElements(r, &numelements); CeedChk(ierr);
  if (op->numelements && op->numelements != numelements)
    return CeedError(op->ceed, 1,
                     "ElemRestriction with %d elements incompatible with prior %d elements",
                     numelements, op->numelements);
  op->numelements = numelements;

  if (b != CEED_BASIS_COLLOCATED) {
    CeedInt numqpoints;
    ierr = CeedBasisGetNumQuadraturePoints(b, &numqpoints); CeedChk(ierr);
    if (op->numqpoints && op->numqpoints != numqpoints)
      return CeedError(op->ceed, 1,
                       "Basis with %d quadrature points incompatible with prior %d points",
                       numqpoints, op->numqpoints);
    op->numqpoints = numqpoints;
  }
  CeedOperatorField *ofield;
  for (CeedInt i=0; i<op->qf->numinputfields; i++) {
    if (!strcmp(fieldname, (*op->qf->inputfields[i]).fieldname)) {
      ofield = &op->inputfields[i];
      goto found;
    }
  }
  for (CeedInt i=0; i<op->qf->numoutputfields; i++) {
    if (!strcmp(fieldname, (*op->qf->outputfields[i]).fieldname)) {
      ofield = &op->outputfields[i];
      goto found;
    }
  }
  return CeedError(op->ceed, 1, "QFunction has no knowledge of field '%s'",
                   fieldname);
found:
  ierr = CeedCalloc(1, ofield); CeedChk(ierr);
  (*ofield)->Erestrict = r;
  (*ofield)->lmode = lmode;
  (*ofield)->basis = b;
  (*ofield)->vec = v;
  op->nfields += 1;
  return 0;
}

/**
  @brief Add a sub-operator to a composite CeedOperator

  @param[out] compositeop Address of the composite CeedOperator
  @param      subop       Address of the sub-operator CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
 */
int CeedCompositeOperatorAddSub(CeedOperator compositeop,
                                CeedOperator subop) {
  if (!compositeop->composite)
    return CeedError(compositeop->ceed, 1,
                     "CeedOperator is not a composite operator");

  if (compositeop->numsub == CEED_COMPOSITE_MAX)
    return CeedError(compositeop->ceed, 1,
                     "Cannot add additional suboperators");

  compositeop->suboperators[compositeop->numsub] = subop;
  subop->refcount++;
  compositeop->numsub++;
  return 0;
}

/**
  @brief Apply CeedOperator to a vector

  This computes the action of the operator on the specified (active) input,
  yielding its (active) output.  All inputs and outputs must be specified using
  CeedOperatorSetField().

  @param op        CeedOperator to apply
  @param[in] in    CeedVector containing input state or NULL if there are no
                     active inputs
  @param[out] out  CeedVector to store result of applying operator (must be
                     distinct from @a in) or NULL if there are no active outputs
  @param request   Address of CeedRequest for non-blocking completion, else
                     CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedOperatorApply(CeedOperator op, CeedVector in,
                      CeedVector out, CeedRequest *request) {
  int ierr;
  Ceed ceed = op->ceed;
  CeedQFunction qf = op->qf;

  if (op->composite) {
    if (!op->numsub) return CeedError(ceed, 1, "No suboperators set");
  } else {
    if (op->nfields == 0) return CeedError(ceed, 1, "No operator fields set");
    if (op->nfields < qf->numinputfields + qf->numoutputfields) return CeedError(
            ceed, 1, "Not all operator fields set");
    if (op->numelements == 0) return CeedError(ceed, 1,
                                       "At least one restriction required");
    if (op->numqpoints == 0) return CeedError(ceed, 1,
                                      "At least one non-collocated basis required");
  }
  ierr = op->Apply(op, in, out, request); CeedChk(ierr);
  return 0;
}

/**
  @brief Get the Ceed associated with a CeedOperator

  @param op              CeedOperator
  @param[out] ceed       Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
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

  @ref Advanced
**/

int CeedOperatorGetNumElements(CeedOperator op, CeedInt *numelem) {
  if (op->composite)
    return CeedError(op->ceed, 1, "Not defined for composite operator");

  *numelem = op->numelements;
  return 0;
}

/**
  @brief Get the number of quadrature points associated with a CeedOperator

  @param op              CeedOperator
  @param[out] numqpts    Variable to store vector number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorGetNumQuadraturePoints(CeedOperator op, CeedInt *numqpts) {
  if (op->composite)
    return CeedError(op->ceed, 1, "Not defined for composite operator");

  *numqpts = op->numqpoints;
  return 0;
}

/**
  @brief Get the number of arguments associated with a CeedOperator

  @param op              CeedOperator
  @param[out] numargs    Variable to store vector number of arguments

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorGetNumArgs(CeedOperator op, CeedInt *numargs) {
  if (op->composite)
    return CeedError(op->ceed, 1, "Not defined for composite operators");
  *numargs = op->nfields;
  return 0;
}

/**
  @brief Get the setup status of a CeedOperator

  @param op             CeedOperator
  @param[out] setupdone Variable to store setup status

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorGetSetupStatus(CeedOperator op, bool *setupdone) {
  *setupdone = op->setupdone;
  return 0;
}

/**
  @brief Get the QFunction associated with a CeedOperator

  @param op              CeedOperator
  @param[out] qf         Variable to store QFunction

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorGetQFunction(CeedOperator op, CeedQFunction *qf) {
  if (op->composite)
    return CeedError(op->ceed, 1, "Not defined for composite operator");

  *qf = op->qf;
  return 0;
}

/**
  @brief Get the number of suboperators associated with a CeedOperator

  @param op              CeedOperator
  @param[out] numsub     Variable to store number of suboperators

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorGetNumSub(CeedOperator op, CeedInt *numsub) {
  Ceed ceed = op->ceed;
  if (!op->composite) return CeedError(ceed, 1, "Not a composite operator");

  *numsub = op->numsub;
  return 0;
}

/**
  @brief Get the list of suboperators associated with a CeedOperator

  @param op                CeedOperator
  @param[out] suboperators Variable to store list of suboperators

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorGetSubList(CeedOperator op, CeedOperator* *suboperators) {
  Ceed ceed = op->ceed;
  if (!op->composite) return CeedError(ceed, 1, "Not a composite operator");

  *suboperators = op->suboperators;
  return 0;
}

/**
  @brief Set the backend data of a CeedOperator

  @param[out] op         CeedOperator
  @param data            Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorSetData(CeedOperator op, void* *data) {
  op->data = *data;
  return 0;
}

/**
  @brief Get the backend data of a CeedOperator

  @param op              CeedOperator
  @param[out] data       Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorGetData(CeedOperator op, void* *data) {
  *data = op->data;
  return 0;
}

/**
  @brief Set the setup flag of a CeedOperator to True

  @param op              CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
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

  @ref Advanced
**/

int CeedOperatorGetFields(CeedOperator op,
                          CeedOperatorField* *inputfields,
                          CeedOperatorField* *outputfields) {
  if (op->composite)
    return CeedError(op->ceed, 1, "Not defined for composite operator");

  if (inputfields) *inputfields = op->inputfields;
  if (outputfields) *outputfields = op->outputfields;
  return 0;
}

/**
  @brief Get the L vector CeedTransposeMode of a CeedOperatorField

  @param opfield         CeedOperatorField
  @param[out] lmode      Variable to store CeedTransposeMode

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorFieldGetLMode(CeedOperatorField opfield,
                              CeedTransposeMode *lmode) {
  *lmode = opfield->lmode;
  return 0;
}/**
  @brief Get the CeedElemRestriction of a CeedOperatorField

  @param opfield         CeedOperatorField
  @param[out] rstr       Variable to store CeedElemRestriction

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
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

  @ref Advanced
**/

int CeedOperatorFieldGetBasis(CeedOperatorField opfield,
                              CeedBasis *basis) {
  *basis = opfield->basis;
  return 0;
}

/**
  @brief Get the CeedVector of a CeedOperatorField

  @param opfield         CeedOperatorField
  @param[out] vec        Variable to store CeedVector

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorFieldGetVector(CeedOperatorField opfield,
                               CeedVector *vec) {
  *vec = opfield->vec;
  return 0;
}

/**
  @brief Destroy a CeedOperator

  @param op CeedOperator to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedOperatorDestroy(CeedOperator *op) {
  int ierr;

  if (!*op || --(*op)->refcount > 0) return 0;
  if ((*op)->Destroy) {
    ierr = (*op)->Destroy(*op); CeedChk(ierr);
  }
  ierr = CeedDestroy(&(*op)->ceed); CeedChk(ierr);
  // Free fields
  for (int i=0; i<(*op)->nfields; i++) {
    if ((*op)->inputfields[i]) {
      ierr = CeedFree(&(*op)->inputfields[i]); CeedChk(ierr);
    }
  }
  for (int i=0; i<(*op)->nfields; i++) {
    if ((*op)->outputfields[i]) {
      ierr = CeedFree(&(*op)->outputfields[i]); CeedChk(ierr);
    }
  }
  // Destroy suboperators
  for (int i=0; i<(*op)->numsub; i++) {
    if ((*op)->suboperators[i]) {
      ierr = CeedOperatorDestroy(&(*op)->suboperators[i]); CeedChk(ierr);
    }
  }
  ierr = CeedQFunctionDestroy(&(*op)->qf); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&(*op)->dqf); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&(*op)->dqfT); CeedChk(ierr);

  ierr = CeedFree(&(*op)->inputfields); CeedChk(ierr);
  ierr = CeedFree(&(*op)->outputfields); CeedChk(ierr);
  ierr = CeedFree(&(*op)->suboperators); CeedChk(ierr);
  ierr = CeedFree(op); CeedChk(ierr);
  return 0;
}

/// @}
