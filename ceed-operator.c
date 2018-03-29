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

/**
  @file
  Implementation of public CeedOperator interfaces

  @defgroup CeedOperator CeedOperator: composed FE-type operations on vectors
  @{
 */

/**
  Create an operator from element restriction, basis, and QFunction

  @param ceed The Ceed library context on which to create the operator
  @param r Element restriction for the operator
  @param b Basis for elements restricted to by @a r
  @param qf QFunction defining the action of the operator at quadrature points
  @param dqf QFunction defining the action of the Jacobian of @a qf (or NULL)
  @param dqfT QFunction defining the action of the transpose of the Jacobian
              of @a qf (or NULL)
  @param[out] op Newly created CeedOperator
  @return Error code, 0 on success
 */
int CeedOperatorCreate(Ceed ceed, CeedElemRestriction r, CeedBasis b,
                       CeedQFunction qf, CeedQFunction dqf,
                       CeedQFunction dqfT, CeedOperator *op) {
  int ierr;

  if (!ceed->OperatorCreate) return CeedError(ceed, 1,
                                      "Backend does not support OperatorCreate");
  ierr = CeedCalloc(1,op); CeedChk(ierr);
  (*op)->ceed = ceed;
  ceed->refcount++;
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
  Apply CeedOperator to a vector

  @param op CeedOperator to apply
  @param qdata CeedVector containing any stored quadrature data (passive data)
               for the operator
  @param invec CeedVector containing input state
  @param outvec CeedVector to store result of applying operator (must be
                distinct from @a invec)
  @param request Address of CeedRequest for non-blocking completion, else
                 CEED_REQUEST_IMMEDIATE
  @return Error code, 0 on success
 */
int CeedOperatorApply(CeedOperator op, CeedVector qdata, CeedVector invec,
                      CeedVector outvec, CeedRequest *request) {
  int ierr;

  ierr = op->Apply(op, qdata, invec, outvec, request); CeedChk(ierr);
  return 0;
}

/**
  Get a suitably sized vector to hold passive fields (data at quadrature points)

  @param op CeedOperator for which to get
  @param[out] qdata Resulting CeedVector.  The implementation holds a reference
                    so the user should not call CeedVectorDestroy
  @return Error code, 0 on success
 */
int CeedOperatorGetQData(CeedOperator op, CeedVector *qdata) {
  int ierr;

  if (!op->GetQData)
    return CeedError(op->ceed, 1, "Backend does not support OperatorGetQData");
  ierr = op->GetQData(op, qdata); CeedChk(ierr);
  return 0;
}

/**
  Destroy a CeedOperator

  @param op CeedOperator to destroy
  @return Error code, 0 on success
 */
int CeedOperatorDestroy(CeedOperator *op) {
  int ierr;

  if (!*op) return 0;
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
