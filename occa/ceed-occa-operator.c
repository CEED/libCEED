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
// * OPERATORS: Create, Apply & Destroy
// *****************************************************************************
typedef struct {
  CeedVector etmp;
  CeedVector qdata;
} CeedOperatorOcca;

// *****************************************************************************
static int CeedOperatorApplyOcca(CeedOperator op, CeedVector qdata,
                                    CeedVector ustate,
                                    CeedVector residual, CeedRequest* request) {
  dbg("[CeedOperator][Apply][Occa]");
  CeedOperatorOcca* impl = op->data;
  CeedVector etmp;
  CeedInt Q;
  CeedScalar* Eu;
  char* qd;
  int ierr;

  if (!impl->etmp) {
    ierr = CeedVectorCreate(op->ceed,
                            op->Erestrict->nelem * op->Erestrict->elemsize,
                            &impl->etmp); CeedChk(ierr);
  }
  etmp = impl->etmp;
  if (op->qf->inmode != CEED_EVAL_NONE || op->qf->inmode != CEED_EVAL_WEIGHT) {
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_NOTRANSPOSE, ustate, etmp,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }
  ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
  ierr = CeedVectorGetArray(etmp, CEED_MEM_HOST, &Eu); CeedChk(ierr);
  ierr = CeedVectorGetArray(qdata, CEED_MEM_HOST, (CeedScalar**)&qd);
  CeedChk(ierr);
  for (CeedInt e=0; e<op->Erestrict->nelem; e++) {
    CeedScalar BEu[Q], BEv[Q], *out[1];
    const CeedScalar* in[1];
    ierr = CeedBasisApply(op->basis, CEED_NOTRANSPOSE, op->qf->inmode,
                          &Eu[e*op->Erestrict->elemsize], BEu); CeedChk(ierr);
    in[0] = BEu;
    out[0] = BEv;
    ierr = CeedQFunctionApply(op->qf, &qd[e*Q*op->qf->qdatasize], Q, in, out);
    CeedChk(ierr);
    ierr = CeedBasisApply(op->basis, CEED_TRANSPOSE, op->qf->outmode, BEv,
                          &Eu[e*op->Erestrict->elemsize]); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(etmp, &Eu); CeedChk(ierr);
  if (residual) {
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_TRANSPOSE, etmp, residual,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }
  if (request != CEED_REQUEST_IMMEDIATE) *request = NULL;
  return 0;
}

// *****************************************************************************
static int CeedOperatorGetQDataOcca(CeedOperator op, CeedVector* qdata) {
  dbg("[CeedOperator][GetQData][Occa]");
  CeedOperatorOcca* impl = op->data;
  int ierr;

  if (!impl->qdata) {
    CeedInt Q;
    ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,
                            op->Erestrict->nelem * Q * op->basis->ndof,
                            &impl->qdata); CeedChk(ierr);
  }
  *qdata = impl->qdata;
  return 0;
}

// *****************************************************************************
static int CeedOperatorDestroyOcca(CeedOperator op) {
  dbg("[CeedOperator][Destroy][Occa]");
  CeedOperatorOcca* impl = op->data;
  int ierr;

  ierr = CeedVectorDestroy(&impl->etmp); CeedChk(ierr);
  ierr = CeedVectorDestroy(&impl->qdata); CeedChk(ierr);
  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
int CeedOperatorCreateOcca(CeedOperator op) {
  CeedOperatorOcca* impl;
  int ierr;

  dbg("[CeedOperator][Create][Occa]");
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  op->data = impl;
  op->Destroy = CeedOperatorDestroyOcca;
  op->Apply = CeedOperatorApplyOcca;
  op->GetQData = CeedOperatorGetQDataOcca;
  return 0;
}

