// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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
#include "ceed-cuda.cuh"

static int CeedOperatorDestroy_Cuda(CeedOperator op) {
  CeedOperator_Cuda *impl = (CeedOperator_Cuda*) op->data;
  int ierr;

  if (impl->ready) {
    ierr = CeedVectorDestroy(&impl->etmp); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->qdata); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->BEu); CeedChk(ierr);
    ierr = CeedVectorDestroy(&impl->BEv); CeedChk(ierr);
    ierr = CeedFree(&op->data); CeedChk(ierr);
  }
  return 0;
}

static int CeedOperatorApply_Cuda(CeedOperator op, CeedVector qdata,
                                 CeedVector ustate,
                                 CeedVector residual, CeedRequest *request) {
  CeedOperator_Cuda *data = (CeedOperator_Cuda*) op->data;
  CeedInt Q;
  const CeedInt nc = op->basis->ndof, dim = op->basis->dim, nelem = op->Erestrict->nelem;
  int ierr;
  CeedTransposeMode lmode = CEED_NOTRANSPOSE;

  ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
  if (!data->ready) {
    data->ready = true;
    const CeedInt n = Q * (nc * (dim + 1) * nelem + 1);
    ierr = CeedVectorCreate(op->ceed,
                            nc * nelem * op->Erestrict->elemsize,
                            &data->etmp); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed, n, &data->BEu); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed, n, &data->BEv); CeedChk(ierr);
    // etmp is allocated when CeedVectorGetArray is called below
  }
  if (op->qf->inmode & ~CEED_EVAL_WEIGHT) {
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_NOTRANSPOSE,
                                    nc, lmode, ustate, data->etmp,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }

  CeedQFunction_Cuda *qfd = (CeedQFunction_Cuda *)op->qf->data;
  qfd->nc = op->basis->ndof;
  qfd->dim = op->basis->dim;
  qfd->nelem = nelem;

  ierr = CeedBasisApplyElems_Cuda(op->basis, nelem, CEED_NOTRANSPOSE, op->qf->inmode,
      data->etmp, data->BEu); CeedChk(ierr);

  ierr = CeedQFunctionApplyElems_Cuda(op->qf, qdata, Q, nelem, data->BEu, data->BEv); CeedChk(ierr);

  /*CeedScalar *out[5] = {0, 0, 0, 0, 0};
  const CeedScalar *in[5] = {0, 0, 0, 0, 0};

  if (op->qf->inmode & CEED_EVAL_WEIGHT) {
    in[4] = d_BEu + Q * nelem * nc * (dim + 1);
  }

  for (CeedInt e=0; e < nelem; e++) {
    if (op->qf->inmode & CEED_EVAL_INTERP) { in[0] = d_BEu + Q * nc * e; }
    if (op->qf->inmode & CEED_EVAL_GRAD) { in[1] = d_BEu + Q * nc * (nelem + dim * e); }
    if (op->qf->outmode & CEED_EVAL_INTERP) { out[0] = d_BEv + Q * nc * e; }
    if (op->qf->outmode & CEED_EVAL_GRAD) { out[1] = d_BEv + Q * nc * (nelem + dim * e); }
    ierr = CeedQFunctionApply(op->qf, &d_q[e*Q*op->qf->qdatasize], Q, in, out);
    CeedChk(ierr);
  }*/
  ierr = CeedBasisApplyElems_Cuda(op->basis, nelem, CEED_TRANSPOSE, op->qf->outmode, data->BEv, data->etmp);
  if (residual) {
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_TRANSPOSE,
                                    nc, lmode, data->etmp, residual,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}

static int CeedOperatorGetQData_Cuda(CeedOperator op, CeedVector *qdata) {
  CeedOperator_Cuda *impl = (CeedOperator_Cuda*)op->data;
  int ierr;

  if (!impl->qdata) {
    CeedInt Q;
    ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,
                            op->Erestrict->nelem * Q
                            * op->qf->qdatasize / sizeof(CeedScalar),
                            &impl->qdata); CeedChk(ierr);
  }
  *qdata = impl->qdata;
  return 0;
}

int CeedOperatorCreate_Cuda(CeedOperator op) {
  CeedOperator_Cuda *impl;
  int ierr;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  impl->ready = false;

  op->data = impl;
  op->Destroy = CeedOperatorDestroy_Cuda;
  op->Apply = CeedOperatorApply_Cuda;
  op->GetQData = CeedOperatorGetQData_Cuda;
  return 0;
}
