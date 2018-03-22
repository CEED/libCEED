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
#include "ceed-occa.h"

// *****************************************************************************
// * Destroy the CeedOperator_Occa
// *****************************************************************************
static int CeedOperatorDestroy_Occa(CeedOperator op) {
  CeedDebug("\033[37;1m[CeedOperator][Destroy]");
  CeedOperator_Occa *data = op->data;
  int ierr = CeedVectorDestroy(&data->etmp); CeedChk(ierr);
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
  CeedDebug("\033[37;1m[CeedOperator][Apply]");
  const CeedInt nc = op->basis->ndof, dim = op->basis->dim;
  const CeedTransposeMode lmode = CEED_NOTRANSPOSE;
  CeedOperator_Occa *data = op->data;
  CeedVector etmp;
  CeedScalar *Eu;
  CeedInt Q;
  char *qd;
  int ierr;
  const size_t esize = op->Erestrict->elemsize;
  const CeedInt enelem = op->Erestrict->nelem;
  // Get Q *********************************************************************
  ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
  // Fill CeedQFunction_Occa's structure with nc, dim & qdata ******************
  CeedQFunction_Occa *qfd = op->qf->data;
  qfd->op = true;
  qfd->nc = nc;
  qfd->dim = dim;
  qfd->d_q = ((CeedVector_Occa *)qdata->data)->d_array;
  // ***************************************************************************
  if (!data->etmp) {
    const int n = nc*enelem*esize;
    const int bn = Q*nc*(dim+2)*enelem;
    CeedDebug("\033[37;1m[CeedOperator][Apply] Setup, n=%d & bn=%d",n,bn);
    ierr = CeedVectorCreate(op->ceed,n,&data->etmp); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,bn,&data->BEu); CeedChk(ierr);
    ierr = CeedVectorCreate(op->ceed,bn,&data->BEv); CeedChk(ierr);
    // etmp is allocated when CeedVectorGetArray is called below
  }
  qfd->b_u = ((CeedVector_Occa *)data->BEu->data)->d_array;
  etmp = data->etmp;
  if (op->qf->inmode & ~CEED_EVAL_WEIGHT) {
    CeedDebug("\033[37;1m[CeedOperator][Apply] Apply Restriction");
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
  CeedDebug("\033[37;1m[CeedOperator][Apply] BEu");
  ierr = CeedBasisApplyElems_Occa(op->basis,CEED_NOTRANSPOSE,op->qf->inmode,
                                  data->etmp,data->BEu);CeedChk(ierr);
  CeedDebug("\033[37;1m[CeedOperator][Apply] etmp:");
  CeedVectorView(data->etmp,"%f",stdout);
  CeedDebug("\033[37;1m[CeedOperator][Apply] BEu:");
  CeedVectorView(data->BEu,"%f",stdout);
  // ***************************************************************************
  CeedDebug("\033[37;1m[CeedOperator][Apply] Q for-loop");
  for (CeedInt e=0; e<enelem; e++) {
    // t20 needs this
    ierr = CeedBasisApply(op->basis, CEED_NOTRANSPOSE,op->qf->inmode, &Eu[e*nc*esize], &BEu[0]);CeedChk(ierr);
    CeedDebug("\033[37;1m[CeedOperator][Apply] for-loop Eu:");
    for(CeedInt k=0;k<nc*esize;k++){
      printf("\t %f\n",Eu[e*nc*esize+k]);
    }
    CeedDebug("\033[37;1m[CeedOperator][Apply] for-loop BEu[e=%d,esize=%d,enelem=%d]:",e,esize,enelem);
    for(CeedInt k=0;k<(Q*nc*(dim+2));k++){
      printf("\t %f\n",BEu[k]);
    }
    /*
    CeedScalar *u_ptr = BEu, *v_ptr = BEv;
    if (op->qf->inmode & CEED_EVAL_INTERP) { in[0] = u_ptr; u_ptr += Q*nc; }
    if (op->qf->inmode & CEED_EVAL_GRAD) { in[1] = u_ptr; u_ptr += Q*nc*dim; }
    if (op->qf->inmode & CEED_EVAL_WEIGHT) { in[4] = u_ptr; u_ptr += Q; }
    if (op->qf->outmode & CEED_EVAL_INTERP) { out[0] = v_ptr; v_ptr += Q*nc; }
    if (op->qf->outmode & CEED_EVAL_GRAD) { out[1] = v_ptr; v_ptr += Q*nc*dim; }
    qfd->offset = e;
    ierr = CeedQFunctionApply(op->qf, &qd[e*Q*qbytes], Q, in, out);
    CeedChk(ierr);
    ierr = CeedBasisApply(op->basis, CEED_TRANSPOSE,op->qf->outmode, BEv, &Eu[e*nc*esize]); CeedChk(ierr);
    */
  }
  exit(0);
  // ***************************************************************************
  CeedDebug("\033[37;1m[CeedOperator][Apply] BEv");
  ierr = CeedBasisApplyElems_Occa(op->basis,CEED_TRANSPOSE,op->qf->outmode,
                                  data->BEv,etmp);CeedChk(ierr);
  // *************************************************************************
  ierr = CeedVectorRestoreArray(etmp, &Eu);
  CeedChk(ierr);
  // ***************************************************************************
  if (residual) {
    CeedDebug("\033[37;1m[CeedOperator][Apply] residual");
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
  CeedDebug("\033[37;1m[CeedOperator][GetQData]");
  CeedOperator_Occa *data = op->data;
  int ierr;
  if (!data->qdata) {
    CeedDebug("\033[37;1m[CeedOperator][GetQData] New");
    CeedInt Q;
    ierr = CeedBasisGetNumQuadraturePoints(op->basis, &Q); CeedChk(ierr);
    CeedDebug("\033[37;1m[CeedOperator][GetQData] Q=%d",Q);
    CeedDebug("\033[37;1m[CeedOperator][GetQData] nelem=%d",op->Erestrict->nelem);
    CeedDebug("\033[37;1m[CeedOperator][GetQData] qdatasize=%d",op->qf->qdatasize);
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
  CeedOperator_Occa *data;
  op->Destroy = CeedOperatorDestroy_Occa;
  op->Apply = CeedOperatorApply_Occa;
  op->GetQData = CeedOperatorGetQData_Occa;
  CeedDebug("\033[37;1m[CeedOperator][Create]");
  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  op->data = data;
  // Push to Ceed_Occa this operator
  ((Ceed_Occa*)op->ceed->data)->op = op;
  return 0;
}

