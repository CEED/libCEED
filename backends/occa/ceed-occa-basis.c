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
// * buildKernel
// *****************************************************************************
static int CeedBasisBuildKernel(CeedBasis basis) {
  const Ceed_Occa *ceed_data = basis->ceed->data;
  const occaDevice dev = ceed_data->device;
  CeedBasis_Occa *data = basis->data;
  // ***************************************************************************
  const int dim = basis->dim;
  const int P1d = basis->P1d;
  const int Q1d = basis->Q1d;
  const CeedInt ndof = basis->ndof;
  const CeedInt nqpt = ndof*CeedPowInt(Q1d,dim);
  const CeedInt vsize = ndof*CeedPowInt(P1d,dim);
  // ***************************************************************************
  const CeedElemRestriction er = data->er; assert(er);
  const CeedInt nelem = er->nelem;
  const CeedInt elemsize = er->elemsize;
  // ***************************************************************************
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] Building kernels");
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/dim", occaInt(dim));
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] dim=%d",dim);
  occaPropertiesSet(pKR, "defines/P1d", occaInt(P1d));
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] P1d=%d",P1d);
  occaPropertiesSet(pKR, "defines/Q1d", occaInt(Q1d));
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] Q1d=%d",Q1d);
  occaPropertiesSet(pKR, "defines/nc", occaInt(ndof));
  occaPropertiesSet(pKR, "defines/ndof", occaInt(ndof));
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] ndof=%d",ndof);
  occaPropertiesSet(pKR, "defines/nqpt", occaInt(nqpt));
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] nqpt=%d",nqpt);
  occaPropertiesSet(pKR, "defines/vsize", occaInt(vsize));
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] vsize=%d",vsize);
  // ***************************************************************************
  occaPropertiesSet(pKR, "defines/nelem", occaInt(nelem));
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] nelem=%d",nelem);
  occaPropertiesSet(pKR, "defines/elemsize", occaInt(elemsize));
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] elemsize=%d",elemsize);
  // ***************************************************************************
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(TILE_SIZE));
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] TILE_SIZE=%d",TILE_SIZE);
  // ***************************************************************************
  char oklPath[4096] = __FILE__;
  const size_t oklPathLen = strlen(oklPath);
  strcpy(&oklPath[oklPathLen-2],".okl");
  data->kZero   = occaDeviceBuildKernel(dev,oklPath,"kZero",pKR);
  data->kInterp = occaDeviceBuildKernel(dev,oklPath,"kInterp",pKR);
  data->kGrad   = occaDeviceBuildKernel(dev,oklPath,"kGrad",pKR);
  data->kWeight = occaDeviceBuildKernel(dev,oklPath,"kWeight",pKR);
  occaPropertiesFree(pKR);
  return 0;
}

// *****************************************************************************
// * TENSORS: Contracts on the middle index
// *          NOTRANSPOSE: V_ajc = T_jb U_abc
// *          TRANSPOSE:   V_ajc = T_bj U_abc
// * CeedScalars are used here, not CeedVectors: we don't touch it yet
// *****************************************************************************
static int CeedTensorContract_Occa(Ceed ceed,
                                   CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                   const CeedScalar *t, CeedTransposeMode tmode,
                                   const CeedInt Add,
                                   const CeedScalar *u, CeedScalar *v) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }
  //printf("ts0=%d, ts1=%d, %d %d %d %d %d",tstride0,tstride1,A,B,C,J,tmode);
  for (CeedInt a=0; a<A; a++) {
    for (CeedInt j=0; j<J; j++) {
      if (!Add) {
        for (CeedInt c=0; c<C; c++)
          v[(a*J+j)*C+c] = 0.0;
      }
      for (CeedInt b=0; b<B; b++) {
        for (CeedInt c=0; c<C; c++) {
          v[(a*J+j)*C+c] += t[j*tstride0 + b*tstride1] * u[(a*B+b)*C+c];
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// * CeedBasisApplyElems_Occa
// *****************************************************************************
int CeedBasisApplyElems_Occa(CeedBasis basis, CeedInt QnD,
                             CeedTransposeMode tmode, CeedEvalMode emode,
                             const CeedVector u, CeedVector v) {
  CeedBasis_Occa *data = basis->data;
  const CeedInt ready =  data->ready;
  // ***************************************************************************
  // We were waiting for the CeedElemRestriction to fill nelem and elemsize
  if (!ready){
    data->ready=true;
    CeedBasisBuildKernel(basis);
  }
  // ***************************************************************************
  const int Q1d = basis->Q1d;
  const CeedInt dim = basis->dim;
  const CeedInt ndof = basis->ndof;
  const CeedInt nqpt = ndof*CeedPowInt(Q1d,dim);
  //const CeedElemRestriction er = data->er; assert(er);
  //const CeedInt nelem = er->nelem;
  // ***************************************************************************
  const CeedInt transpose = (tmode == CEED_TRANSPOSE);
  CeedInt u_nqpt = 0;
  CeedInt v_nqpt = 0;
  // ***************************************************************************
  if (transpose) {
    CeedDebug("\033[31;1m[CeedBasis][ApplyElems] transpose\033[m");
    const CeedVector_Occa *v_data = v->data;
    const occaMemory d_v = v_data->d_array;
    int ierr = 0;
    occaKernelRun(data->kZero, d_v, occaPtr(&ierr));
    CeedChk(ierr);
  }
  // ***************************************************************************
  if (emode == CEED_EVAL_NONE) {
    CeedDebug("\033[31;1m[CeedBasis][Apply] CEED_EVAL_NONE");
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_INTERP) {
    CeedDebug("\033[31;1m[CeedBasis][ApplyElems] CEED_EVAL_INTERP\033[m");
    const occaMemory d_tmp0 = data->tmp0;
    const occaMemory d_tmp1 = data->tmp1;
    const occaMemory d_interp1d = data->interp1d;
    const CeedVector_Occa *u_data = u->data;assert(u_data);
    const CeedVector_Occa *v_data = v->data;assert(v_data);
    const occaMemory d_u = u_data->d_array;
    const occaMemory d_v = v_data->d_array;
    int ierr = 0;
    occaKernelRun(data->kInterp,occaInt(QnD),
                  occaInt(transpose),occaInt(tmode),
                  d_tmp0, d_tmp1, d_interp1d,
                  d_u, d_v,
                  occaPtr(&ierr));
    CeedChk(ierr);
    if (!transpose) v_nqpt += nqpt;
    else u_nqpt += nqpt;
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_GRAD) {
    CeedDebug("\033[31;1m[CeedBasis][ApplyElems] CEED_EVAL_GRAD\033[m");
    const occaMemory d_tmp0 = data->tmp0;
    const occaMemory d_tmp1 = data->tmp1;
    const occaMemory d_grad1d = data->grad1d;
    const occaMemory d_interp1d = data->interp1d;
    const CeedVector_Occa *u_data = u->data; assert(u_data);
    const CeedVector_Occa *v_data = v->data; assert(v_data);
    const occaMemory d_u = u_data->d_array;
    const occaMemory d_v = v_data->d_array;
    int ierr = 0;
    occaKernelRun(data->kGrad,occaInt(QnD),
                  occaInt(transpose),occaInt(tmode),
                  d_tmp0,d_tmp1,d_grad1d,d_interp1d,
                  d_u,occaInt(u_nqpt),
                  d_v,occaInt(v_nqpt),
                  occaPtr(&ierr));
    CeedChk(ierr);
    if (!transpose) v_nqpt += dim*nqpt;
    else u_nqpt += dim*nqpt;
    //CeedDebug("\033[31;1m[CeedBasis][ApplyElems] CEED_EVAL_GRAD v:\033[m");
    //CeedVectorView(v,"%f",stdout);    
    //CeedDebug("\033[31;1m[CeedBasis][ApplyElems] nqpt=%d\033[m",nqpt);
    //CeedDebug("\033[31;1m[CeedBasis][ApplyElems] u_nqpt=%d\033[m",u_nqpt);
    //CeedDebug("\033[31;1m[CeedBasis][ApplyElems] v_nqpt=%d\033[m",v_nqpt);
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_WEIGHT) {
    CeedDebug("\033[31;1m[CeedBasis][ApplyElems] CEED_EVAL_WEIGHT\033[m");
    if (transpose)
      return CeedError(basis->ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    const CeedInt Q = basis->Q1d;
    const occaMemory d_qw = data->qweight1d;
    const CeedVector_Occa *v_data = v->data;assert(v_data);
    const occaMemory d_v = v_data->d_array;
    //CeedDebug("\033[31;1m[CeedBasis][ApplyElems] CEED_EVAL_WEIGHT before v:\033[m");
    //CeedVectorView(v,"%f",stdout);
    int ierr = 0;
    occaKernelRun(data->kWeight,occaInt(QnD),occaInt(Q),d_qw,d_v,occaInt(v_nqpt),occaPtr(&ierr));
    CeedChk(ierr);
    //CeedDebug("\033[31;1m[CeedBasis][ApplyElems] CEED_EVAL_WEIGHT after v:\033[m");
    //CeedVectorView(v,"%f",stdout);    
    if (!transpose) v_nqpt += nqpt;
    else u_nqpt += nqpt;
  }
  return 0;
}

// *****************************************************************************
// * CeedBasisApply_Occa
// *****************************************************************************
static int CeedBasisApply_Occa(CeedBasis basis,
                               CeedTransposeMode tmode, CeedEvalMode emode,
                               const CeedScalar *u, CeedScalar *v) {
  int ierr;
  //const CeedScalar *bkp_v=v;
  const CeedInt dim = basis->dim;
  const CeedInt ndof = basis->ndof;
  const CeedInt nqpt = ndof*CeedPowInt(basis->Q1d, dim);
  const CeedInt transpose = (tmode == CEED_TRANSPOSE);
  // ***************************************************************************
  if (transpose) {
    const CeedInt vsize = ndof*CeedPowInt(basis->P1d, dim);
    CeedDebug("\033[38;5;249m[CeedBasis][Apply] transpose, vsize=%d",vsize);
    for (CeedInt i = 0; i < vsize; i++)
      v[i] = 0.0;
  }
  // ***************************************************************************
  if (emode == CEED_EVAL_NONE) {
    CeedDebug("\033[38;5;249m[CeedBasis][Apply] CEED_EVAL_NONE");
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_INTERP) {
    const CeedInt P = transpose?basis->Q1d:basis->P1d;
    const CeedInt Q = transpose?basis->P1d:basis->Q1d;
    CeedInt pre = ndof*CeedPowInt(P, dim-1), post = 1;
    CeedDebug("\033[38;5;249m[CeedBasis][Apply] CEED_EVAL_INTERP, P=%d, Q=%d, pre=%d, post=%d",P,Q,pre,post);
    CeedScalar tmp[2][ndof*Q*CeedPowInt(P>Q?P:Q, dim-1)];
    for (CeedInt d=0; d<dim; d++) {
      // CeedDebug("\033[38;5;249m[CeedBasis][Apply] d=%d, d==0: %s, d%%2=%d, d==dim-1: %s, (d+1)%%2=%d",d,d==0?"yes":"no",d%2,d==dim-1?"yes":"no",(d+1)%2);
      ierr = CeedTensorContract_Occa(basis->ceed,
                                     pre, P, post, Q,
                                     basis->interp1d,
                                     tmode, transpose&&(d==dim-1),
                                     d==0?u:tmp[d%2],
                                     d==dim-1?v:tmp[(d+1)%2]);
      CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
    if (!transpose) v += nqpt;
    else u += nqpt;
    //CeedDebug("\033[38;5;249m[CeedBasis][ApplyElems] CEED_EVAL_INTERP transpose:%s v:\033[m",transpose?"yes":"no");
    //for(int k=0;k<(Q*1*(dim+2));k++) printf("\t %f\n",v[k]);
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_GRAD) {
    const CeedInt P = transpose?basis->Q1d:basis->P1d;
    const CeedInt Q = transpose?basis->P1d:basis->Q1d;
    //CeedDebug("\033[38;5;249m[CeedBasis][Apply] CEED_EVAL_GRAD, P=%d, Q=%d",P,Q);
    CeedScalar tmp[2][ndof*Q*CeedPowInt(P>Q?P:Q, dim-1)];
    for (CeedInt p=0; p<dim; p++) {
      //CeedDebug("\033[38;5;249m\t[CeedBasis][Apply] p=%d",p);
      CeedInt pre = ndof*CeedPowInt(P, dim-1), post = 1;
      for (CeedInt d=0; d<dim; d++) {
        //CeedDebug("\033[38;5;249m\t\t[CeedBasis][Apply] d=%d",d);
        //printf(", pre=%d",pre);
        //printf(", post=%d",post);
        //printf(", d==0: %s",d==0?"yes":"no");
        //printf(", d%%2=%d ",d%2);
        //printf(", d==dim-1: %s",d==dim-1?"yes":"no");
        //printf(", (d+1)%%2=%d ",(d+1)%2);        
        ierr = CeedTensorContract_Occa(basis->ceed, pre, P, post, Q,
                                       (p==d)?basis->grad1d:basis->interp1d,
                                       tmode, transpose&&(d==dim-1),
                                       d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]);
        CeedChk(ierr);
        pre /= P;
        post *= Q;
      }
      if (!transpose) v += nqpt;
      else u += nqpt;      
    }
    //CeedDebug("\033[38;5;249m[CeedBasis][ApplyElems] CEED_EVAL_GRAD bkp_v:\033[m");
    //for(int k=0;k<(Q*ndof*(dim+2));k++) printf("\t %f\n",bkp_v[k]);
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_WEIGHT) {
    CeedDebug("\033[38;5;249m[CeedBasis][Apply] CEED_EVAL_WEIGHT");
    if (transpose)
      return CeedError(basis->ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    //CeedDebug("\033[38;5;249m[CeedBasis][Apply] CEED_EVAL_WEIGHT before v, but v-bkp_v=%d",v-bkp_v);
    //for(int k=0;k<(basis->Q1d*ndof*(dim+2));k++) printf("\t %f\n",v[k]);
    // *************************************************************************
    CeedInt Q = basis->Q1d;
    //CeedDebug("\033[38;5;249m[CeedBasis][Apply] Q=%d, v-bkp_v=%d",Q, v-bkp_v);
    for (CeedInt d=0; d<dim; d++) {
      const CeedInt pre = CeedPowInt(Q, dim-d-1), post = CeedPowInt(Q, d);
      //printf("\tpre=%d, post=%d",pre,post);
      for (CeedInt i=0; i<pre; i++) {
        for (CeedInt j=0; j<Q; j++) {
          for (CeedInt k=0; k<post; k++) {
            //printf(" d=%d, i=%d, j=%d, k=%d offset=%d",d,i,k,k,(i*Q+j)*post+k);
            v[(i*Q + j)*post + k] = basis->qweight1d[j] * (d == 0 ? 1 : v[(i*Q + j)*post + k]);
          }
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// * CeedBasisDestroy_Occa
// *****************************************************************************
static int CeedBasisDestroy_Occa(CeedBasis basis) {
  int ierr;
  CeedBasis_Occa *data = basis->data;
  CeedDebug("\033[38;5;249m[CeedBasis][Destroy]");
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedBasisCreateTensorH1_Occa
// *****************************************************************************
int CeedBasisCreateTensorH1_Occa(Ceed ceed,
                                 CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                 const CeedScalar *interp1d,
                                 const CeedScalar *grad1d,
                                 const CeedScalar *qref1d,
                                 const CeedScalar *qweight1d,
                                 CeedBasis basis) {
  int ierr;
  CeedBasis_Occa *data;
  const Ceed_Occa *ceed_data = ceed->data;
  const occaDevice dev = ceed_data->device;
  // ***************************************************************************
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  basis->data = data;
  // ***************************************************************************
  assert(qref1d);
  data->qref1d = occaDeviceMalloc(dev,Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  occaCopyPtrToMem(data->qref1d,qref1d,Q1d*sizeof(CeedScalar),NO_OFFSET,NO_PROPS);
  // ***************************************************************************
  assert(qweight1d);
  data->qweight1d = occaDeviceMalloc(dev,Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  occaCopyPtrToMem(data->qweight1d,qweight1d,Q1d*sizeof(CeedScalar),NO_OFFSET,NO_PROPS);
  // ***************************************************************************
  assert(interp1d);
  data->interp1d = occaDeviceMalloc(dev,P1d*Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  occaCopyPtrToMem(data->interp1d,interp1d,P1d*Q1d*sizeof(CeedScalar),NO_OFFSET,NO_PROPS);
  // ***************************************************************************
  assert(grad1d);
  data->grad1d = occaDeviceMalloc(dev,P1d*Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  occaCopyPtrToMem(data->grad1d,grad1d,P1d*Q1d*sizeof(CeedScalar),NO_OFFSET,NO_PROPS);
  // ***************************************************************************
  const CeedInt ndof = basis->ndof;
  const CeedInt M = (Q1d>P1d)?Q1d:P1d;
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] ndof=%d, P1d=%d, Q1d=%d, dim=%d",ndof,P1d,Q1d,dim);
  const CeedInt tmpSz = ndof*M*CeedPowInt(M,dim-1);
  CeedDebug("\033[38;5;249m[CeedBasis][CreateTensorH1] tmpSz=%d",tmpSz);
  data->tmp0 = occaDeviceMalloc(dev,tmpSz*sizeof(CeedScalar),NULL,NO_PROPS);
  data->tmp1 = occaDeviceMalloc(dev,tmpSz*sizeof(CeedScalar),NULL,NO_PROPS);
  // ***************************************************************************
  basis->Apply = CeedBasisApply_Occa;
  basis->Destroy = CeedBasisDestroy_Occa;
  return 0;
}
