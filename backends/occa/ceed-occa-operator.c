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
#include "ceed-occa.h"

// *****************************************************************************
/// Destroy the CeedOperator_Occa
// *****************************************************************************
static int CeedOperatorDestroy_Occa(CeedOperator op) {
  CeedDebug("\033[37;1m[CeedOperator][Destroy]");
  CeedOperator_Occa *impl = op->data;
  int ierr  = CeedVectorDestroy(&impl->etmp); CeedChk(ierr);
  ierr = CeedVectorDestroy(&impl->qdata); CeedChk(ierr);
  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
/// Apply CeedOperator to a vector
// *****************************************************************************
static int CeedOperatorApply_Occa(CeedOperator op, CeedVector qdata,
                                  CeedVector ustate,
                                  CeedVector residual, CeedRequest *request) {
  CeedOperator_Occa *impl = op->data;
  const CeedInt nc = op->basis->ndof, dim = op->basis->dim;
  CeedVector etmp;
  CeedInt Q;
  CeedScalar *Eu;
  char *qd;
  int ierr;
  const CeedTransposeMode lmode = CEED_NOTRANSPOSE;
  // Fill CeedQFunction_Occa's structure with nc, dim & qdata
  CeedQFunction_Occa *QFdata = op->qf->data;
  QFdata->op = true;
  QFdata->nc = nc;
  QFdata->dim = dim;
  QFdata->d_qdata = ((CeedVector_Occa *)qdata->data)->d_array;

  if (!impl->etmp) {
    const int n = nc * op->Erestrict->nelem * op->Erestrict->elemsize;
    ierr = CeedVectorCreate(op->ceed,n,&impl->etmp); CeedChk(ierr);
    // etmp is allocated when CeedVectorGetArray is called below
  }
  etmp = impl->etmp;
  if (op->qf->inmode & ~CEED_EVAL_WEIGHT) {
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_NOTRANSPOSE,
                                    nc, lmode, ustate, etmp,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }
  ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
  ierr = CeedVectorGetArray(etmp, CEED_MEM_HOST, &Eu); CeedChk(ierr);
  // Fetching back data from device memory
  ierr = CeedVectorGetArray(qdata, CEED_MEM_HOST, (CeedScalar**)&qd);
  QFdata->qdata = qd; // Telling QF base address of qd
  CeedChk(ierr);
  // Local arrays, sizes & pointers
  CeedScalar BEu[Q*nc*(dim+2)], BEv[Q*nc*(dim+2)], *out[5] = {0,0,0,0,0};
  const CeedScalar *in[5] = {0,0,0,0,0};
  const size_t qbytes = op->qf->qdatasize;
  const size_t esize = op->Erestrict->elemsize;
  const CeedInt enelem = op->Erestrict->nelem;
  // TODO: quadrature weights can be computed just once
  for (CeedInt e=0; e<enelem; e++) {
    ierr = CeedBasisApply(op->basis, CEED_NOTRANSPOSE,
                          op->qf->inmode, &Eu[e*nc*esize], BEu);
    CeedChk(ierr);
    CeedScalar *u_ptr = BEu, *v_ptr = BEv;
    if (op->qf->inmode & CEED_EVAL_INTERP) { in[0] = u_ptr; u_ptr += Q*nc; }
    if (op->qf->inmode & CEED_EVAL_GRAD) { in[1] = u_ptr; u_ptr += Q*nc*dim; }
    if (op->qf->inmode & CEED_EVAL_WEIGHT) { in[4] = u_ptr; u_ptr += Q; }
    if (op->qf->outmode & CEED_EVAL_INTERP) { out[0] = v_ptr; v_ptr += Q*nc; }
    if (op->qf->outmode & CEED_EVAL_GRAD) { out[1] = v_ptr; v_ptr += Q*nc*dim; }
    ierr = CeedQFunctionApply(op->qf, &qd[e*Q*qbytes], Q, in, out);
    CeedChk(ierr);
    ierr = CeedBasisApply(op->basis, CEED_TRANSPOSE,
                          op->qf->outmode, BEv, &Eu[e*nc*esize]);
    CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(etmp, &Eu); CeedChk(ierr);
  if (residual) {
    CeedScalar *res;
    CeedVectorGetArray(residual, CEED_MEM_HOST, &res);
    for (int i = 0; i < residual->length; i++)
      res[i] = (CeedScalar)0;
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_TRANSPOSE,
                                    nc, lmode, etmp, residual,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}

// *****************************************************************************
/// Get a suitably sized vector to hold passive fields
// *****************************************************************************
static int CeedOperatorGetQData_Occa(CeedOperator op, CeedVector *qdata) {
  CeedDebug("\033[37;1m[CeedOperator][GetQData]");
  CeedOperator_Occa *impl = op->data;
  int ierr;

  if (!impl->qdata) {
    CeedDebug("\033[37;1m[CeedOperator][GetQData] New");
    CeedInt Q;
    ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
    const int n = op->Erestrict->nelem * Q * op->qf->qdatasize / sizeof(CeedScalar);
    ierr = CeedVectorCreate(op->ceed,n,&impl->qdata); CeedChk(ierr);
  }
  *qdata = impl->qdata;
  return 0;
}

// *****************************************************************************
/// Create an operator from element restriction, basis, and QFunction
// *****************************************************************************
int CeedOperatorCreate_Occa(CeedOperator op) {
  int ierr;
  CeedOperator_Occa *data;
  op->Destroy = CeedOperatorDestroy_Occa;
  op->Apply = CeedOperatorApply_Occa;
  op->GetQData = CeedOperatorGetQData_Occa;
  CeedDebug("\033[37;1m[CeedOperator][Create]");
  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  op->data = data;
  return 0;
}

