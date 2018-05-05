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
#define CEED_DEBUG_COLOR 198
#include "ceed-occa.h"

// *****************************************************************************
// * Destroy the CeedOperator_Occa
// *****************************************************************************
static int CeedOperatorDestroy_Occa(CeedOperator op) {
  const Ceed ceed = op->ceed;
  dbg("[CeedOperator][Destroy]");
  CeedOperator_Occa *data = op->data;
  int ierr = CeedVectorDestroy(&data->etmp); CeedChk(ierr);
  ierr = CeedVectorDestroy(&data->BEu); CeedChk(ierr);
  ierr = CeedVectorDestroy(&data->BEv); CeedChk(ierr);
  ierr = CeedVectorDestroy(&data->qdata); CeedChk(ierr);
  ierr = CeedFree(&op->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Apply CeedOperator to a vector
// *****************************************************************************
static int CeedOperatorApply_Occa(CeedOperator op, CeedVector qdata,
                                  CeedVector ustate,
                                  CeedVector residual, CeedRequest *request) {
  const Ceed ceed = op->ceed;
  dbg("[CeedOperator][Apply]");
  const CeedInt nc = op->basis->ndof, dim = op->basis->dim;
  const CeedTransposeMode lmode = CEED_NOTRANSPOSE;
  CeedOperator_Occa *data = op->data;
  CeedVector etmp;
  CeedScalar *Eu;
  CeedInt Q;
  char *qd;
  int ierr;
  const size_t elemsize = op->Erestrict->elemsize;
  const CeedInt nelem = op->Erestrict->nelem;
  // Get Q *********************************************************************
  ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
  // Fill CeedBasis_Occa's structure with CeedElemRestriction ******************
  CeedBasis_Occa *basis = op->basis->data;
  basis->er = op->Erestrict;
  // Fill CeedQFunction_Occa's structure with nc, dim & qdata ******************
  CeedQFunction_Occa *qfd = op->qf->data;
  qfd->op = true;
  qfd->nc = nc;
  qfd->dim = dim;
  qfd->nelem = nelem;
  qfd->elemsize = elemsize;
  qfd->d_q = ((CeedVector_Occa *)qdata->data)->d_array;
  // ***************************************************************************
  if (!data->etmp) {
    const int n = nc*nelem*elemsize;
    const int bn = Q*nc*(dim+2)*nelem;
    dbg("[CeedOperator][Apply] Setup, n=%d & bn=%d",n,bn);
    ierr = CeedVectorCreate(op->ceed,n,&data->etmp); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,bn,&data->BEu); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,bn,&data->BEv); CeedChk(ierr);
    // etmp is allocated when CeedVectorGetArray is called below
  }
  // Push the memory to the QFunction that will be used
  qfd->b_u = ((CeedVector_Occa *)data->BEu->data)->d_array;
  qfd->b_v = ((CeedVector_Occa *)data->BEv->data)->d_array;
  etmp = data->etmp;
  if (op->qf->inmode & ~CEED_EVAL_WEIGHT) {
    dbg("[CeedOperator][Apply] Apply Restriction");
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_NOTRANSPOSE,
                                    nc, lmode, ustate, etmp,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  }
  // We want to avoid Get/Restore
  ierr = CeedVectorGetArray(etmp, CEED_MEM_HOST, &Eu); CeedChk(ierr);
  // Fetching back data from device memory
  ierr = CeedVectorGetArray(qdata, CEED_MEM_HOST, (CeedScalar**)&qd);
  CeedChk(ierr);
  // Local arrays, sizes & pointers ********************************************
  CeedScalar BEu[Q*nc*(dim+2)], BEv[Q*nc*(dim+2)], *out[5] = {0,0,0,0,0};
  const CeedScalar *in[5] = {0,0,0,0,0};
  const size_t qbytes = op->qf->qdatasize;
  // ***************************************************************************
  ierr = CeedBasisApplyElems_Occa(op->basis,Q,CEED_NOTRANSPOSE,op->qf->inmode,
                                  data->etmp,data->BEu); CeedChk(ierr);
  // ***************************************************************************
  dbg("[CeedOperator][Apply] Q for-loop");
  for (CeedInt e=0; e<nelem; e++) {
    for(CeedInt k=0; k<(Q*nc*(dim+2)); k++) BEu[k]=0.0;
    ierr = CeedBasisApply(op->basis, CEED_NOTRANSPOSE,op->qf->inmode,
                          &Eu[e*nc*elemsize], BEu); CeedChk(ierr);
    CeedScalar *u_ptr = BEu, *v_ptr = BEv;
    if (op->qf->inmode & CEED_EVAL_INTERP) { in[0] = u_ptr; u_ptr += Q*nc; }
    if (op->qf->inmode & CEED_EVAL_GRAD) { in[1] = u_ptr; u_ptr += Q*nc*dim; }
    if (op->qf->inmode & CEED_EVAL_WEIGHT) { in[4] = u_ptr; u_ptr += Q; }
    if (op->qf->outmode & CEED_EVAL_INTERP) { out[0] = v_ptr; v_ptr += Q*nc; }
    if (op->qf->outmode & CEED_EVAL_GRAD) { out[1] = v_ptr; v_ptr += Q*nc*dim; }
    qfd->e = e;
    ierr = CeedQFunctionApply(op->qf, &qd[e*Q*qbytes], Q, in, out); CeedChk(ierr);
    ierr = CeedBasisApply(op->basis, CEED_TRANSPOSE,op->qf->outmode, BEv,
                          &Eu[e*nc*elemsize]); CeedChk(ierr);
  }
  // ***************************************************************************
  ierr = CeedBasisApplyElems_Occa(op->basis,Q,CEED_TRANSPOSE,op->qf->outmode,
                                  data->BEv,data->etmp); CeedChk(ierr);
  // *************************************************************************
  ierr = CeedVectorRestoreArray(etmp, &Eu); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(qdata, (CeedScalar**)&qd); CeedChk(ierr);
  // ***************************************************************************
  if (residual) {
    dbg("[CeedOperator][Apply] residual");
    ierr = CeedElemRestrictionApply(op->Erestrict, CEED_TRANSPOSE,
                                    nc, lmode, etmp, residual,
                                    CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
    // Restore used pointer if one was provided ********************************
    const CeedVector_Occa *data = residual->data;
    if (data->used_pointer)
      occaCopyMemToPtr(data->used_pointer,data->d_array,
                       residual->length*sizeof(CeedScalar),
                       NO_OFFSET, NO_PROPS);
  }
  // ***************************************************************************
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}

// *****************************************************************************
// * CeedOperatorGetQData_Occa
// *****************************************************************************
static int CeedOperatorGetQData_Occa(CeedOperator op, CeedVector *qdata) {
  const Ceed ceed = op->ceed;
  dbg("[CeedOperator][GetQData]");
  CeedOperator_Occa *data = op->data;
  int ierr;
  if (!data->qdata) {
    dbg("[CeedOperator][GetQData] New");
    CeedInt Q;
    ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
    dbg("[CeedOperator][GetQData] Q=%d",Q);
    dbg("[CeedOperator][GetQData] nelem=%d",op->Erestrict->nelem);
    dbg("[CeedOperator][GetQData] qdatasize=%d",op->qf->qdatasize);
    const int n = op->Erestrict->nelem * Q * op->qf->qdatasize / sizeof(CeedScalar);
    ierr = CeedVectorCreate(op->ceed,n,&data->qdata); CeedChk(ierr);
  }
  *qdata = data->qdata;
  return 0;
}

// *****************************************************************************
// * Create an operator from element restriction, basis, and QFunction
// *****************************************************************************
int CeedOperatorCreate_Occa(CeedOperator op) {
  int ierr;
  const Ceed ceed = op->ceed;
  CeedOperator_Occa *data;
  op->Destroy = CeedOperatorDestroy_Occa;
  op->Apply = CeedOperatorApply_Occa;
  op->GetQData = CeedOperatorGetQData_Occa;
  dbg("[CeedOperator][Create]");
  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  op->data = data;
  return 0;
}

