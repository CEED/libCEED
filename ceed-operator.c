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
#include <string.h>

/**
  @file
  Implementation of public CeedOperator interfaces

  @defgroup CeedOperator CeedOperator: composed FE-type operations on vectors
  @{
 */

/**
  Create an operator from element restriction, basis, and QFunction

  @param ceed The Ceed library context on which to create the operator
  @param qf QFunction defining the action of the operator at quadrature points
  @param dqf QFunction defining the action of the Jacobian of @a qf (or NULL)
  @param dqfT QFunction defining the action of the transpose of the Jacobian
              of @a qf (or NULL)
  @param[out] op Newly created CeedOperator
  @return Error code, 0 on success
 */
int CeedOperatorCreate(Ceed ceed, CeedQFunction qf, CeedQFunction dqf,
                       CeedQFunction dqfT, CeedOperator *op) {
  int ierr;

  if (!ceed->OperatorCreate) return CeedError(ceed, 1,
                                      "Backend does not support OperatorCreate");
  ierr = CeedCalloc(1,op); CeedChk(ierr);
  (*op)->ceed = ceed;
  ceed->refcount++;
  (*op)->refcount = 1;
  (*op)->Erestrict = r;
  r->refcount++;
  (*op)->basis = b;
  b->refcount++;
  (*op)->qf = qf;
  qf->refcount++;
  (*op)->dqf = dqf;
  if (dqf) dqf->refcount++;
  (*op)->dqfT = dqfT;
  if (dqfT) dqfT->refcount++;
  ierr = ceed->OperatorCreate(*op); CeedChk(ierr);
  return 0;
}

/**
  Provide a field to a CeedOperator for use by its CeedQFunction

  This function is used to specify both active and passive fields to a
  CeedOperator.  For passive fields, a vector @arg v must be provided.  Passive
  fields can inputs or outputs (updated in-place when operator is applied).

  Active fields must be specified using this function, but their data (in a
  CeedVector) is passed in CeedOperatorApply().  There can be at most one active
  input and at most one active output.

  @param op the operator on which to provide the field
  @param fieldname name of the field (to be matched with the name used by CeedQFunction)
  @param r element restriction or NULL to use the identity
  @param b basis in which the field resides or NULL if collocated with quadrature points
  @param v vector to be used by CeedOperator or NULL if field is active
 */
int CeedOperatorSetField(CeedOperator op, const char *fieldname,
                         CeedElemRestriction r, CeedBasis b,
                         CeedVector v) {
  int ierr;
  if (r) {
    CeedInt numelements;
    ierr = CeedElemRestrictionGetNumElements(r, &numelements); CeedChk(ierr);
    if (op->numelements && op->numelements != numelements)
      return CeedError(op->ceed, 1,
                       "ElemRestriction with %d elements incompatible with prior %d elements",
                       numelements, op->numelements);
    op->numelements = numelements;
  }
  if (b) {
    CeedInt numqpoints;
    ierr = CeedBasisGetNumQuadraturePoints(b, &numqpoints); CeedChk(ierr);
    if (op->numqpoints && op->numqpoints != numqpoints)
      return CeedError(op->ceed, 1,
                       "Basis with %d quadrature points incompatible with prior %d points",
                       numqpoints, op->numqpoints);
    op->numqpoints = numqpoints;
  }
  struct CeedOperatorField *ofield;
  for (CeedInt i=0; i<op->qf->numinputfields; i++) {
    if (!strcmp(fieldname, op->qf->inputfields[i].fieldname)) {
      ofield = &op->inputfields[i];
      goto found;
    }
  }
  for (CeedInt i=0; i<op->qf->numoutputfields; i++) {
    if (!strcmp(fieldname, op->qf->outputfields[i].fieldname)) {
      ofield = &op->outputfields[i];
      goto found;
    }
  }
  return CeedError(op->ceed, 1, "QFunction has no knowledge of field '%s'",
                   fieldname);
found:
  ofield->Erestrict = r;
  ofield->basis = b;
  ofield->vec = v;
  return 0;
}

/**
  Apply CeedOperator to a vector

  This computes the action of the operator on the specified (active) input,
  yielding its (active) output.  All inputs and outputs must be specified using
  CeedOperatorSetField().

  @param op CeedOperator to apply
  @param in CeedVector containing input state or NULL if there are no active
            inputs
  @param out CeedVector to store result of applying operator (must be
                distinct from @a in) or NULL if there are no active outputs
  @param request Address of CeedRequest for non-blocking completion, else
                 CEED_REQUEST_IMMEDIATE
  @return Error code, 0 on success
 */
int CeedOperatorApply(CeedOperator op, CeedVector in,
                      CeedVector out, CeedRequest *request) {
  int ierr;

  ierr = op->Apply(op, in, out, request); CeedChk(ierr);
  return 0;
}

/**
  Destroy a CeedOperator

  @param op CeedOperator to destroy
  @return Error code, 0 on success
 */
int CeedOperatorDestroy(CeedOperator *op) {
  int ierr;

  if (!*op || --(*op)->refcount > 0) return 0;
  if ((*op)->Destroy) {
    ierr = (*op)->Destroy(*op); CeedChk(ierr);
  }
  ierr = CeedElemRestrictionDestroy(&(*op)->Erestrict); CeedChk(ierr);
  ierr = CeedBasisDestroy(&(*op)->basis); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&(*op)->qf); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&(*op)->dqf); CeedChk(ierr);
  ierr = CeedQFunctionDestroy(&(*op)->dqfT); CeedChk(ierr);
  ierr = CeedDestroy(&(*op)->ceed); CeedChk(ierr);
  ierr = CeedFree(op); CeedChk(ierr);
  return 0;
}
