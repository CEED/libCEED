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
  @param dqf     QFunction defining the action of the Jacobian of @a qf (or
                   CEED_QFUNCTION_NONE)
  @param dqfT    QFunction defining the action of the transpose of the Jacobian
                   of @a qf (or CEED_QFUNCTION_NONE)
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
      // LCOV_EXCL_START
      return CeedError(ceed, 1, "Backend does not support OperatorCreate");
    // LCOV_EXCL_STOP

    ierr = CeedOperatorCreate(delegate, qf, dqf, dqfT, op); CeedChk(ierr);
    return 0;
  }

  ierr = CeedCalloc(1, op); CeedChk(ierr);
  (*op)->ceed = ceed;
  ceed->refcount++;
  (*op)->refcount = 1;
  if (qf == CEED_QFUNCTION_NONE)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Operator must have a valid QFunction.");
  // LCOV_EXCL_STOP
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

  @ref Basic
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
  if (op->hasrestriction && op->numelements != numelements)
    // LCOV_EXCL_START
    return CeedError(op->ceed, 1,
                     "ElemRestriction with %d elements incompatible with prior "
                     "%d elements", numelements, op->numelements);
  // LCOV_EXCL_STOP
  op->numelements = numelements;
  op->hasrestriction = true; // Restriction set, but numelements may be 0

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
  // LCOV_EXCL_START
  return CeedError(op->ceed, 1, "QFunction has no knowledge of field '%s'",
                   fieldname);
  // LCOV_EXCL_STOP
found:
  ierr = CeedCalloc(1, ofield); CeedChk(ierr);
  (*ofield)->Erestrict = r;
  r->refcount += 1;
  (*ofield)->lmode = lmode;
  (*ofield)->basis = b;
  if (b != CEED_BASIS_COLLOCATED)
    b->refcount += 1;
  (*ofield)->vec = v;
  if (v != CEED_VECTOR_ACTIVE && v != CEED_VECTOR_NONE)
    v->refcount += 1;
  op->nfields += 1;
  return 0;
}

/**
  @brief Add a sub-operator to a composite CeedOperator

  @param[out] compositeop Composite CeedOperator
  @param      subop       Sub-operator CeedOperator

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
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

  Ceed ceedref;
  ierr = CeedInit(fallbackresource, &ceedref); CeedChk(ierr);
  ceedref->opfallbackparent = op->ceed;
  op->ceed->opfallbackceed = ceedref;

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

  @ref Basic
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
  @brief Assemble a linear CeedQFunction associated with a CeedOperator

  This returns a CeedVector containing a matrix at each quadrature point
    providing the action of the CeedQFunction associated with the CeedOperator.
    The vector 'assembled' is of shape
      [num_elements, num_input_fields, num_output_fields, num_quad_points]
    and contains column-major matrices representing the action of the
    CeedQFunction for a corresponding quadrature point on an element. Inputs and
    outputs are in the order provided by the user when adding CeedOperator fields.
    For example, a QFunction with inputs 'u' and 'gradu' and outputs 'gradv' and
    'v', provided in that order, would result in an assembled QFunction that
    consists of (1 + dim) x (dim + 1) matrices at each quadrature point acting
    on the input [u, du_0, du_1] and producing the output [dv_0, dv_1, v].

  @param op             CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedQFunction at
                          quadrature points
  @param[out] rstr      CeedElemRestriction for CeedVector containing assembled
                          CeedQFunction
  @param request        Address of CeedRequest for non-blocking completion, else
                          CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedOperatorAssembleLinearQFunction(CeedOperator op, CeedVector *assembled,
                                        CeedElemRestriction *rstr,
                                        CeedRequest *request) {
  int ierr;
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

  if (op->AssembleLinearQFunction) {
    ierr = op->AssembleLinearQFunction(op, assembled, rstr, request);
    CeedChk(ierr);
  } else {
    // Fallback to reference Ceed
    if (!op->opfallback) {
      ierr = CeedOperatorCreateFallback(op); CeedChk(ierr);
    }
    // Assemble
    ierr = op->opfallback->AssembleLinearQFunction(op->opfallback, assembled,
           rstr, request); CeedChk(ierr);
  }

  return 0;
}

/**
  @brief Assemble the diagonal of a square linear Operator

  This returns a CeedVector containing the diagonal of a linear CeedOperator.

  @param op             CeedOperator to assemble CeedQFunction
  @param[out] assembled CeedVector to store assembled CeedOperator diagonal
  @param request        Address of CeedRequest for non-blocking completion, else
                          CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
**/
int CeedOperatorAssembleLinearDiagonal(CeedOperator op, CeedVector *assembled,
                                       CeedRequest *request) {
  int ierr;
  Ceed ceed = op->ceed;
  ierr = CeedOperatorCheckReady(ceed, op); CeedChk(ierr);

  // Use backend version, if available
  if (op->AssembleLinearDiagonal) {
    ierr = op->AssembleLinearDiagonal(op, assembled, request); CeedChk(ierr);
    return 0;
  }

  // Assemble QFunction
  CeedQFunction qf = op->qf;
  CeedVector assembledqf;
  CeedElemRestriction rstr;
  ierr = CeedOperatorAssembleLinearQFunction(op,  &assembledqf, &rstr, request);
  CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr); CeedChk(ierr);

  // Determine active input basis
  CeedInt numemodein = 0, ncomp, dim = 1;
  CeedEvalMode *emodein = NULL;
  CeedBasis basisin = NULL;
  CeedElemRestriction rstrin = NULL;
  for (CeedInt i=0; i<qf->numinputfields; i++)
    if (op->inputfields[i]->vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(op->inputfields[i], &basisin);
      CeedChk(ierr);
      ierr = CeedBasisGetNumComponents(basisin, &ncomp); CeedChk(ierr);
      ierr = CeedBasisGetDimension(basisin, &dim); CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(op->inputfields[i], &rstrin);
      CeedChk(ierr);
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qf->inputfields[i], &emode);
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
        for (CeedInt d=0; d<dim; d++)
          emodein[numemodein+d] = emode;
        numemodein += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }

  // Determine active output basis
  CeedInt numemodeout = 0;
  CeedEvalMode *emodeout = NULL;
  CeedBasis basisout = NULL;
  CeedElemRestriction rstrout = NULL;
  CeedTransposeMode lmodeout = CEED_NOTRANSPOSE;
  for (CeedInt i=0; i<qf->numoutputfields; i++)
    if (op->outputfields[i]->vec == CEED_VECTOR_ACTIVE) {
      ierr = CeedOperatorFieldGetBasis(op->outputfields[i], &basisout);
      CeedChk(ierr);
      ierr = CeedOperatorFieldGetElemRestriction(op->outputfields[i], &rstrout);
      CeedChk(ierr);
      ierr = CeedOperatorFieldGetLMode(op->outputfields[i], &lmodeout);
      CeedChk(ierr);
      CeedEvalMode emode;
      ierr = CeedQFunctionFieldGetEvalMode(qf->outputfields[i], &emode);
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
        for (CeedInt d=0; d<dim; d++)
          emodeout[numemodeout+d] = emode;
        numemodeout += dim;
        break;
      case CEED_EVAL_WEIGHT:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        break; // Caught by QF Assembly
      }
    }

  // Create diagonal vector
  CeedVector elemdiag;
  ierr = CeedElemRestrictionCreateVector(rstrin, assembled, &elemdiag);
  CeedChk(ierr);

  // Assemble element operator diagonals
  CeedScalar *elemdiagarray, *assembledqfarray;
  ierr = CeedVectorSetValue(elemdiag, 0.0); CeedChk(ierr);
  ierr = CeedVectorGetArray(elemdiag, CEED_MEM_HOST, &elemdiagarray);
  CeedChk(ierr);
  ierr = CeedVectorGetArray(assembledqf, CEED_MEM_HOST, &assembledqfarray);
  CeedChk(ierr);
  CeedInt nelem, nnodes, nqpts;
  ierr = CeedElemRestrictionGetNumElements(rstrin, &nelem); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basisin, &nnodes); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basisin, &nqpts); CeedChk(ierr);
  // Compute the diagonal of B^T D B
  // Each node, qpt pair
  for (CeedInt n=0; n<nnodes; n++)
    for (CeedInt q=0; q<nqpts; q++) {
      CeedInt dout = -1;
      // Each basis eval mode pair
      for (CeedInt eout=0; eout<numemodeout; eout++) {
        CeedScalar bt = 1.0;
        if (emodeout[eout] == CEED_EVAL_GRAD)
          dout += 1;
        ierr = CeedBasisGetValue(basisout, emodeout[eout], q, n, dout, &bt);
        CeedChk(ierr);
        CeedInt din = -1;
        for (CeedInt ein=0; ein<numemodein; ein++) {
          CeedScalar b = 0.0;
          if (emodein[ein] == CEED_EVAL_GRAD)
            din += 1;
          ierr = CeedBasisGetValue(basisin, emodein[ein], q, n, din, &b);
          CeedChk(ierr);
          // Each element and component
          for (CeedInt e=0; e<nelem; e++)
            for (CeedInt cout=0; cout<ncomp; cout++) {
              CeedScalar db = 0.0;
              for (CeedInt cin=0; cin<ncomp; cin++)
                db += assembledqfarray[((((e*numemodein+ein)*ncomp+cin)*
                                         numemodeout+eout)*ncomp+cout)*nqpts+q]*b;
              elemdiagarray[(e*ncomp+cout)*nnodes+n] += bt * db;
            }
        }
      }
    }

  ierr = CeedVectorRestoreArray(elemdiag, &elemdiagarray); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(assembledqf, &assembledqfarray); CeedChk(ierr);

  // Assemble local operator diagonal
  ierr = CeedVectorSetValue(*assembled, 0.0); CeedChk(ierr);
  ierr = CeedElemRestrictionApply(rstrout, CEED_TRANSPOSE, lmodeout, elemdiag,
                                  *assembled, request); CeedChk(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&assembledqf); CeedChk(ierr);
  ierr = CeedVectorDestroy(&elemdiag); CeedChk(ierr);
  ierr = CeedFree(&emodein); CeedChk(ierr);
  ierr = CeedFree(&emodeout); CeedChk(ierr);

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
  @param[out] qdata     CeedVector to hold qdata for fdminv
  @param request        Address of CeedRequest for non-blocking completion, else
                          CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedOperatorCreateFDMElementInverse(CeedOperator op, CeedOperator *fdminv,
                                        CeedRequest *request) {
  int ierr;
  Ceed ceed = op->ceed;

  // Determine active input basis
  bool interp = false, grad = false;
  CeedBasis basis = NULL;
  CeedElemRestriction rstr = NULL;
  for (CeedInt i=0; i<op->qf->numinputfields; i++)
    if (op->inputfields[i] && op->inputfields[i]->vec == CEED_VECTOR_ACTIVE) {
      basis = op->inputfields[i]->basis;
      interp = interp || op->qf->inputfields[i]->emode == CEED_EVAL_INTERP;
      grad = grad || op->qf->inputfields[i]->emode == CEED_EVAL_GRAD;
      rstr = op->inputfields[i]->Erestrict;
    }
  if (!basis)
    return CeedError(ceed, 1, "No active field set");
  CeedInt P1d, Q1d, elemsize, nqpts, dim, ncomp = 1, nelem = 1, nnodes = 1;
  ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basis, &elemsize); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpts); CeedChk(ierr);
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(rstr, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumNodes(rstr, &nnodes); CeedChk(ierr);

  // Build and diagonalize 1D Mass and Laplacian
  if (!basis->tensorbasis)
    return CeedError(ceed, 1, "FDMElementInverse only supported for tensor "
                     "bases");
  CeedScalar *work, *mass, *laplace, *x, *x2, *lambda;
  ierr = CeedMalloc(Q1d*P1d, &work); CeedChk(ierr);
  ierr = CeedMalloc(P1d*P1d, &mass); CeedChk(ierr);
  ierr = CeedMalloc(P1d*P1d, &laplace); CeedChk(ierr);
  ierr = CeedMalloc(P1d*P1d, &x); CeedChk(ierr);
  ierr = CeedMalloc(P1d*P1d, &x2); CeedChk(ierr);
  ierr = CeedMalloc(P1d, &lambda); CeedChk(ierr);
  // -- Mass
  for (CeedInt i=0; i<Q1d; i++)
    for (CeedInt j=0; j<P1d; j++)
      work[i+j*Q1d] = basis->interp1d[i*P1d+j]*basis->qweight1d[i];
  ierr = CeedMatrixMultiply(ceed, work, basis->interp1d, mass, P1d, P1d, Q1d);
  CeedChk(ierr);
  // -- Laplacian
  for (CeedInt i=0; i<Q1d; i++)
    for (CeedInt j=0; j<P1d; j++)
      work[i+j*Q1d] = basis->grad1d[i*P1d+j]*basis->qweight1d[i];
  ierr = CeedMatrixMultiply(ceed, work, basis->grad1d, laplace, P1d, P1d, Q1d);
  CeedChk(ierr);
  // -- Diagonalize
  ierr = CeedSimultaneousDiagonalization(ceed, laplace, mass, x, lambda, P1d);
  CeedChk(ierr);
  ierr = CeedFree(&work); CeedChk(ierr);
  ierr = CeedFree(&mass); CeedChk(ierr);
  ierr = CeedFree(&laplace); CeedChk(ierr);
  for (CeedInt i=0; i<P1d; i++)
    for (CeedInt j=0; j<P1d; j++)
      x2[i+j*P1d] = x[j+i*P1d];
  ierr = CeedFree(&x); CeedChk(ierr);

  // Assemble QFunction
  CeedVector assembled;
  CeedElemRestriction rstr_qf;
  ierr =  CeedOperatorAssembleLinearQFunction(op, &assembled, &rstr_qf,
          request); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_qf); CeedChk(ierr);

  // Calculate element averages
  CeedInt nfields = ((interp?1:0) + (grad?dim:0))*((interp?1:0) + (grad?dim:0));
  CeedScalar *elemavg;
  const CeedScalar *assembledarray, *qweightsarray;
  CeedVector qweights;
  ierr = CeedVectorCreate(ceed, nqpts, &qweights); CeedChk(ierr);
  ierr = CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                        CEED_VECTOR_NONE, qweights); CeedChk(ierr);
  ierr = CeedVectorGetArrayRead(assembled, CEED_MEM_HOST, &assembledarray);
  CeedChk(ierr);
  ierr = CeedVectorGetArrayRead(qweights, CEED_MEM_HOST, &qweightsarray);
  CeedChk(ierr);
  ierr = CeedCalloc(nelem, &elemavg); CeedChk(ierr);
  for (CeedInt e=0; e<nelem; e++) {
    CeedInt count = 0;
    for (CeedInt q=0; q<nqpts; q++)
      for (CeedInt i=0; i<ncomp*ncomp*nfields; i++)
        if (fabs(assembledarray[e*nelem*nqpts*ncomp*ncomp*nfields +
                                                                  i*nqpts + q]) > CEED_EPSILON) {
          elemavg[e] += assembledarray[e*nelem*nqpts*ncomp*ncomp*nfields +
                                       i*nqpts + q] / qweightsarray[q];
          count++;
        }
    if (count)
      elemavg[e] /= count;
  }
  ierr = CeedVectorRestoreArrayRead(assembled, &assembledarray); CeedChk(ierr);
  ierr = CeedVectorDestroy(&assembled); CeedChk(ierr);
  ierr = CeedVectorRestoreArrayRead(qweights, &qweightsarray); CeedChk(ierr);
  ierr = CeedVectorDestroy(&qweights); CeedChk(ierr);

  // Build FDM diagonal
  CeedVector qdata;
  CeedScalar *qdataarray;
  ierr = CeedVectorCreate(ceed, nelem*ncomp*nnodes, &qdata); CeedChk(ierr);
  ierr = CeedVectorSetArray(qdata, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
  CeedChk(ierr);
  ierr = CeedVectorGetArray(qdata, CEED_MEM_HOST, &qdataarray); CeedChk(ierr);
  for (CeedInt e=0; e<nelem; e++)
    for (CeedInt c=0; c<ncomp; c++)
      for (CeedInt n=0; n<nnodes; n++) {
        if (interp)
          qdataarray[(e*ncomp+c)*nnodes+n] = 1;
        if (grad)
          for (CeedInt d=0; d<dim; d++) {
            CeedInt i = (n / CeedIntPow(P1d, d)) % P1d;
            qdataarray[(e*ncomp+c)*nnodes+n] += lambda[i];
          }
        qdataarray[(e*ncomp+c)*nnodes+n] = 1 / (elemavg[e] *
                                                qdataarray[(e*ncomp+c)*nnodes+n]);
      }
  ierr = CeedFree(&elemavg); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(qdata, &qdataarray); CeedChk(ierr);

  // Setup FDM operator
  // -- Basis
  CeedBasis fdm_basis;
  CeedScalar *graddummy, *qrefdummy, *qweightdummy;
  ierr = CeedCalloc(P1d*P1d, &graddummy); CeedChk(ierr);
  ierr = CeedCalloc(P1d, &qrefdummy); CeedChk(ierr);
  ierr = CeedCalloc(P1d, &qweightdummy); CeedChk(ierr);
  ierr = CeedBasisCreateTensorH1(ceed, dim, ncomp, P1d, P1d, x2, graddummy,
                                 qrefdummy, qweightdummy, &fdm_basis);
  CeedChk(ierr);
  ierr = CeedFree(&graddummy); CeedChk(ierr);
  ierr = CeedFree(&qrefdummy); CeedChk(ierr);
  ierr = CeedFree(&qweightdummy); CeedChk(ierr);
  ierr = CeedFree(&x2); CeedChk(ierr);
  ierr = CeedFree(&lambda); CeedChk(ierr);

  // -- Restriction
  CeedElemRestriction rstr_i;
  ierr = CeedElemRestrictionCreateIdentity(ceed, nelem, nnodes, nnodes*nelem,
         ncomp, &rstr_i); CeedChk(ierr);
  // -- QFunction
  CeedQFunction mass_qf;
  ierr = CeedQFunctionCreateInteriorByName(ceed, "MassApply", &mass_qf);
  CeedChk(ierr);
  // -- Operator
  ierr = CeedOperatorCreate(ceed, mass_qf, NULL, NULL, fdminv); CeedChk(ierr);
  CeedOperatorSetField(*fdminv, "u", rstr_i, CEED_NOTRANSPOSE,
                       fdm_basis, CEED_VECTOR_ACTIVE); CeedChk(ierr);
  CeedOperatorSetField(*fdminv, "qdata", rstr_i, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata); CeedChk(ierr);
  CeedOperatorSetField(*fdminv, "v", rstr_i, CEED_NOTRANSPOSE,
                       fdm_basis, CEED_VECTOR_ACTIVE); CeedChk(ierr);

  // Cleanup
  ierr = CeedVectorDestroy(&qdata); CeedChk(ierr);
  ierr = CeedBasisDestroy(&fdm_basis); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(&rstr_i); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&mass_qf); CeedChk(ierr);

  return 0;
}


/**
  @brief Apply CeedOperator to a vector

  This computes the action of the operator on the specified (active) input,
  yielding its (active) output.  All inputs and outputs must be specified using
  CeedOperatorSetField().

  @param op        CeedOperator to apply
  @param[in] in    CeedVector containing input state or CEED_VECTOR_NONE if
                  there are no active inputs
  @param[out] out  CeedVector to store result of applying operator (must be
                     distinct from @a in) or CEED_VECTOR_NONE if there are no
                     active outputs
  @param request   Address of CeedRequest for non-blocking completion, else
                     CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
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
                     CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Basic
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

  @ref Advanced
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

  @ref Advanced
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
    // LCOV_EXCL_START
    return CeedError(op->ceed, 1, "Not defined for composite operator");
  // LCOV_EXCL_STOP

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

  @ref Advanced
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
  @brief Set the backend data of a CeedOperator

  @param[out] op         CeedOperator
  @param data            Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/

int CeedOperatorSetData(CeedOperator op, void **data) {
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

int CeedOperatorGetData(CeedOperator op, void **data) {
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
  @brief View a field of a CeedOperator

  @param[in] field       Operator field to view
  @param[in] fieldnumber Number of field being viewed
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
          "%s      Name: \"%s\"\n"
          "%s      Lmode: \"%s\"\n",
          pre, inout, fieldnumber, pre, qffield->fieldname,
          pre, CeedTransposeModes[field->lmode]);

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
  @brief View a CeedOperator

  @param[in] op     CeedOperator to view
  @param[in] stream Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref Utility
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
  @brief Get the CeedOperatorFields of a CeedOperator

  @param op                 CeedOperator
  @param[out] inputfields   Variable to store inputfields
  @param[out] outputfields  Variable to store outputfields

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
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
}

/**
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

int CeedOperatorFieldGetBasis(CeedOperatorField opfield, CeedBasis *basis) {
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

int CeedOperatorFieldGetVector(CeedOperatorField opfield, CeedVector *vec) {
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
  for (int i=0; i<(*op)->nfields; i++)
    if ((*op)->inputfields[i]) {
      ierr = CeedElemRestrictionDestroy(&(*op)->inputfields[i]->Erestrict);
      CeedChk(ierr);
      if ((*op)->inputfields[i]->basis != CEED_BASIS_COLLOCATED) {
        ierr = CeedBasisDestroy(&(*op)->inputfields[i]->basis); CeedChk(ierr);
      }
      if ((*op)->inputfields[i]->vec != CEED_VECTOR_ACTIVE &&
          (*op)->inputfields[i]->vec != CEED_VECTOR_NONE ) {
        ierr = CeedVectorDestroy(&(*op)->inputfields[i]->vec); CeedChk(ierr);
      }
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
