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
#define CEED_DEBUG_COLOR 249
#include "ceed-occa.h"

// *****************************************************************************
// * buildKernel
// *****************************************************************************
static int CeedBasisBuildKernel(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  Ceed_Occa *ceed_data;
  ierr = CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  const occaDevice dev = ceed_data->device;
  CeedBasis_Occa *data;
  ierr = CeedBasisGetData(basis, (void *)&data); CeedChk(ierr);
  // ***************************************************************************
  CeedInt dim, P1d, Q1d, ncomp;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  const CeedInt nqpt = ncomp*CeedIntPow(Q1d,dim);
  const CeedInt vsize = ncomp*CeedIntPow(P1d,dim);
  // ***************************************************************************
  const CeedElemRestriction er = data->er; assert(er);
  CeedInt nelem, elemsize;
  ierr = CeedElemRestrictionGetNumElements(er, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(er, &elemsize); CeedChk(ierr);
  const bool ocl = ceed_data->ocl;
  // ***************************************************************************
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/dim", occaInt(dim));
  dbg("[CeedBasis][BK] dim=%d",dim);
  occaPropertiesSet(pKR, "defines/P1d", occaInt(P1d));
  dbg("[CeedBasis][BK] P1d=%d",P1d);
  occaPropertiesSet(pKR, "defines/Q1d", occaInt(Q1d));
  dbg("[CeedBasis][BK] Q1d=%d",Q1d);
  occaPropertiesSet(pKR, "defines/nc", occaInt(ncomp));
  occaPropertiesSet(pKR, "defines/ncomp", occaInt(ncomp));
  dbg("[CeedBasis][BK] ncomp=%d",ncomp);
  occaPropertiesSet(pKR, "defines/nqpt", occaInt(nqpt));
  dbg("[CeedBasis][BK] nqpt=%d",nqpt);
  occaPropertiesSet(pKR, "defines/vsize", occaInt(vsize));
  dbg("[CeedBasis][BK] vsize=%d",vsize);
  // ***************************************************************************
  occaPropertiesSet(pKR, "defines/nelem", occaInt(nelem));
  dbg("[CeedBasis][BK] nelem=%d",nelem);
  occaPropertiesSet(pKR, "defines/elemsize", occaInt(elemsize));
  dbg("[CeedBasis][BK] elemsize=%d",elemsize);
  // ***************************************************************************
  // OpenCL check for this requirement
  const CeedInt elem_tile_size = (nelem>TILE_SIZE)?TILE_SIZE:nelem;
  // OCCA+MacOS implementation needs that for now (if DeviceID targets a CPU)
  const CeedInt tile_size = ocl?1:elem_tile_size;
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(tile_size));
  dbg("[CeedBasis][BK] TILE_SIZE=%d",tile_size);
  // ***************************************************************************
  const CeedInt M1d = (Q1d>P1d)?Q1d:P1d;
  occaPropertiesSet(pKR, "defines/M1d", occaInt(M1d));
  const CeedInt MPow = CeedIntPow(M1d,dim-1);
  dbg("[CeedBasis][BK] nelem=%d, ncomp=%d, M1d=%d, MPow=%d",
      nelem,ncomp,M1d,MPow);
  const CeedInt tmpSz = ncomp*M1d*CeedIntPow(M1d,dim-1);
  occaPropertiesSet(pKR, "defines/tmpSz", occaInt(tmpSz));
  dbg("[CeedBasis][BK] dim=%d, ncomp=%d, P1d=%d, Q1d=%d, M1d=%d ",
      dim,ncomp,P1d,Q1d,M1d);
  const CeedInt elems_x_tmpSz = nelem*tmpSz;
  dbg("[CeedBasis][BK] elems_x_tmpSz=%d",elems_x_tmpSz);
  data->tmp0 = occaDeviceMalloc(dev,elems_x_tmpSz*sizeof(CeedScalar),NULL,
                                NO_PROPS);
  data->tmp1 = occaDeviceMalloc(dev,elems_x_tmpSz*sizeof(CeedScalar),NULL,
                                NO_PROPS);
  // ***************************************************************************
  char *oklPath;
  ierr = CeedOklPath_Occa(ceed,__FILE__, "ceed-occa-basis",&oklPath);
  CeedChk(ierr);
  // ***************************************************************************
  data->kZero   = occaDeviceBuildKernel(dev,oklPath,"kZero",pKR);
  data->kInterp = occaDeviceBuildKernel(dev,oklPath,"kInterp",pKR);
  data->kGrad   = occaDeviceBuildKernel(dev,oklPath,"kGrad",pKR);
  data->kWeight = occaDeviceBuildKernel(dev,oklPath,"kWeight",pKR);
  // free local usage **********************************************************
  ierr = CeedFree(&oklPath); CeedChk(ierr);
  occaFree(pKR);
  return 0;
}

// *****************************************************************************
// * TENSORS: Contracts on the middle index
// *          NOTRANSPOSE: V_ajc = T_jb U_abc
// *          TRANSPOSE:   V_ajc = T_bj U_abc
// * CeedScalars are used here, not CeedVectors: we don't touch it yet
// *****************************************************************************
static int CeedTensorContract_Occa(CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                   const CeedScalar *restrict t,
                                   CeedTransposeMode tmode, const CeedInt Add,
                                   const CeedScalar *restrict u,
                                   CeedScalar *restrict v) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

  if (!Add)
    for (CeedInt q=0; q<A*J*C; q++)
      v[q] = (CeedScalar) 0.0;

  for (CeedInt a=0; a<A; a++)
    for (CeedInt b=0; b<B; b++)
      for (CeedInt j=0; j<J; j++) {
        CeedScalar tq = t[j*tstride0 + b*tstride1];
        for (CeedInt c=0; c<C; c++)
          // NOLINTNEXTLINE: False positive for unitalized values
          v[(a*J+j)*C+c] += tq * u[(a*B+b)*C+c];
      }
  return 0;
}

// *****************************************************************************
// * CeedBasisApplyElems_Occa
// *****************************************************************************
int CeedBasisApplyElems_Occa(CeedBasis basis, CeedInt QnD,
                             CeedTransposeMode tmode, CeedEvalMode emode,
                             const CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedBasis_Occa *data;
  ierr = CeedBasisGetData(basis, (void *)&data); CeedChk(ierr);
  const CeedInt ready =  data->ready;
  // ***************************************************************************
  // We were waiting for the CeedElemRestriction to fill nelem and elemsize
  if (!ready) {
    data->ready=true;
    CeedBasisBuildKernel(basis);
  }
  // ***************************************************************************
  const CeedInt transpose = (tmode == CEED_TRANSPOSE);
  // ***************************************************************************
  if (transpose) {
    dbg("[CeedBasis][ApplyElems] transpose");
    CeedVector_Occa *v_data;
    ierr = CeedVectorGetData(v, (void *)&v_data); CeedChk(ierr);
    const occaMemory d_v = v_data->d_array;
    occaKernelRun(data->kZero, d_v);
  }
  // ***************************************************************************
  if (emode == CEED_EVAL_NONE) {
    dbg("[CeedBasis][Apply] CEED_EVAL_NONE");
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_INTERP) {
    dbg("[CeedBasis][ApplyElems] CEED_EVAL_INTERP");
    const occaMemory d_tmp0 = data->tmp0;
    const occaMemory d_tmp1 = data->tmp1;
    const occaMemory d_interp1d = data->interp1d;
    CeedVector_Occa *u_data;
    ierr = CeedVectorGetData(u, (void *)&u_data); CeedChk(ierr);
    assert(u_data);
    CeedVector_Occa *v_data;
    ierr = CeedVectorGetData(v, (void *)&v_data); CeedChk(ierr);
    assert(v_data);
    const occaMemory d_u = u_data->d_array;
    const occaMemory d_v = v_data->d_array;
    occaKernelRun(data->kInterp,occaInt(QnD),
                  occaInt(transpose),occaInt(tmode),
                  d_tmp0, d_tmp1, d_interp1d,
                  d_u, d_v);
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_GRAD) {
    dbg("[CeedBasis][ApplyElems] CEED_EVAL_GRAD");
    const occaMemory d_tmp0 = data->tmp0;
    const occaMemory d_tmp1 = data->tmp1;
    const occaMemory d_grad1d = data->grad1d;
    const occaMemory d_interp1d = data->interp1d;
    CeedVector_Occa *u_data;
    ierr = CeedVectorGetData(u, (void *)&u_data); CeedChk(ierr);
    assert(u_data);
    CeedVector_Occa *v_data;
    ierr = CeedVectorGetData(v, (void *)&v_data); CeedChk(ierr);
    assert(v_data);
    const occaMemory d_u = u_data->d_array;
    const occaMemory d_v = v_data->d_array;
    occaKernelRun(data->kGrad,occaInt(QnD),
                  occaInt(transpose),occaInt(tmode),
                  d_tmp0,d_tmp1,d_grad1d,d_interp1d,
                  d_u, d_v);
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_WEIGHT) {
    dbg("[CeedBasis][ApplyElems] CEED_EVAL_WEIGHT");
    if (transpose)
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    CeedInt Q1d;
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
    const occaMemory d_qw = data->qweight1d;
    CeedVector_Occa *v_data;
    ierr = CeedVectorGetData(v, (void *)&v_data); CeedChk(ierr);
    assert(v_data);
    const occaMemory d_v = v_data->d_array;
    occaKernelRun(data->kWeight,occaInt(QnD),occaInt(Q1d),d_qw,d_v);
  }
  return 0;
}

// *****************************************************************************
// * CeedBasisApply_Occa
// *****************************************************************************
static int CeedBasisApply_Occa(CeedBasis basis, CeedInt nelem,
                               CeedTransposeMode tmode, CeedEvalMode emode,
                               CeedVector U, CeedVector V) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedInt dim, P1d, Q1d, ncomp, nqpt;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChk(ierr);
  nqpt *= ncomp;
  const CeedInt transpose = (tmode == CEED_TRANSPOSE);
  const CeedScalar *u;
  CeedScalar *v;
  if (U) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u); CeedChk(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    return CeedError(ceed, 1,
                     "An input vector is required for this CeedEvalMode");
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_HOST, &v); CeedChk(ierr);

  if (nelem != 1)
    return CeedError(ceed, 1,
                     "This backend does not support BasisApply for multiple elements");
  // ***************************************************************************
  if (transpose) {
    const CeedInt vsize = ncomp*CeedIntPow(P1d, dim);
    //dbg("[CeedBasis][Apply] transpose");
    for (CeedInt i = 0; i < vsize; i++)
      v[i] = 0.0;
  }
  // ***************************************************************************
  if (emode == CEED_EVAL_NONE) {
    //dbg("[CeedBasis][Apply] CEED_EVAL_NONE");
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_INTERP) {
    const CeedInt P = transpose?Q1d:P1d;
    const CeedInt Q = transpose?P1d:Q1d;
    CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = 1;
    //dbg("[CeedBasis][Apply] CEED_EVAL_INTERP");
    CeedScalar tmp[2][ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
    CeedScalar *interp1d;
    ierr = CeedBasisGetInterp(basis, &interp1d); CeedChk(ierr);
    for (CeedInt d=0; d<dim; d++) {
      ierr = CeedTensorContract_Occa(pre, P, post, Q,
                                     interp1d,
                                     tmode, transpose&&(d==dim-1),
                                     d==0?u:tmp[d%2],
                                     d==dim-1?v:tmp[(d+1)%2]);
      CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_GRAD) {
    const CeedInt P = transpose?Q1d:P1d;
    const CeedInt Q = transpose?P1d:Q1d;
    //dbg("[CeedBasis][Apply] CEED_EVAL_GRAD, P=%d, Q=%d",P,Q);
    CeedScalar tmp[2][ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
    for (CeedInt p=0; p<dim; p++) {
      CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = 1;
      CeedScalar *interp1d, *grad1d;
      ierr = CeedBasisGetInterp(basis, &interp1d); CeedChk(ierr);
      ierr = CeedBasisGetGrad(basis, &grad1d); CeedChk(ierr);
      for (CeedInt d=0; d<dim; d++) {
        ierr = CeedTensorContract_Occa(pre, P, post, Q,
                                       (p==d)?grad1d:interp1d,
                                       tmode, transpose&&(d==dim-1),
                                       d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]);
        CeedChk(ierr);
        pre /= P;
        post *= Q;
      }
      if (!transpose) v += nqpt;
      else u += nqpt;
    }

    if (!transpose) v -= nqpt*dim;
    else u -= nqpt*dim;
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_WEIGHT) {
    //dbg("[CeedBasis][Apply] CEED_EVAL_WEIGHT");
    if (transpose)
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    // *************************************************************************
    CeedInt Q = Q1d;
    for (CeedInt d=0; d<dim; d++) {
      const CeedInt pre = CeedIntPow(Q, dim-d-1), post = CeedIntPow(Q, d);
      CeedScalar *qweight1d;
      ierr = CeedBasisGetQWeights(basis, &qweight1d); CeedChk(ierr);
      for (CeedInt i=0; i<pre; i++) {
        for (CeedInt j=0; j<Q; j++) {
          for (CeedInt k=0; k<post; k++) {
            v[(i*Q + j)*post + k] =
              qweight1d[j] * (d == 0 ? 1 : v[(i*Q + j)*post + k]);
          }
        }
      }
    }
  }
  if (U) {
    ierr = CeedVectorRestoreArrayRead(U, &u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &v); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedBasisDestroy_Occa
// *****************************************************************************
static int CeedBasisDestroy_Occa(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedBasis_Occa *data;
  ierr = CeedBasisGetData(basis, (void *)&data); CeedChk(ierr);
  dbg("[CeedBasis][Destroy]");
  occaFree(data->kZero);
  occaFree(data->kInterp);
  occaFree(data->kGrad);
  occaFree(data->kWeight);
  occaFree(data->qref1d);
  occaFree(data->qweight1d);
  occaFree(data->interp1d);
  occaFree(data->grad1d);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedBasisCreateTensorH1_Occa
// *****************************************************************************
int CeedBasisCreateTensorH1_Occa(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                 const CeedScalar *interp1d,
                                 const CeedScalar *grad1d,
                                 const CeedScalar *qref1d,
                                 const CeedScalar *qweight1d,
                                 CeedBasis basis) {
  int ierr;
  CeedBasis_Occa *data;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  Ceed_Occa *ceed_data;
  ierr = CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  const occaDevice dev = ceed_data->device;
  dbg("[CeedBasis][CreateTensorH1]");
  // ***************************************************************************
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  // ***************************************************************************
  assert(qref1d);
  data->qref1d = occaDeviceMalloc(dev,Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  occaCopyPtrToMem(data->qref1d,qref1d,Q1d*sizeof(CeedScalar),NO_OFFSET,NO_PROPS);
  // ***************************************************************************
  assert(qweight1d);
  data->qweight1d = occaDeviceMalloc(dev,Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  occaCopyPtrToMem(data->qweight1d,qweight1d,Q1d*sizeof(CeedScalar),NO_OFFSET,
                   NO_PROPS);
  // ***************************************************************************
  assert(interp1d);
  data->interp1d = occaDeviceMalloc(dev,P1d*Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  occaCopyPtrToMem(data->interp1d,interp1d,P1d*Q1d*sizeof(CeedScalar),NO_OFFSET,
                   NO_PROPS);
  // ***************************************************************************
  assert(grad1d);
  data->grad1d = occaDeviceMalloc(dev,P1d*Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  occaCopyPtrToMem(data->grad1d,grad1d,P1d*Q1d*sizeof(CeedScalar),NO_OFFSET,
                   NO_PROPS);
  ierr = CeedBasisSetData(basis, (void *)&data); CeedChk(ierr);
  // ***************************************************************************
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Occa); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedBasisCreateH1_Occa
// *****************************************************************************
int CeedBasisCreateH1_Occa(CeedElemTopology topo, CeedInt dim,
                           CeedInt nnodes, CeedInt nqpts,
                           const CeedScalar *interp,
                           const CeedScalar *grad,
                           const CeedScalar *qref,
                           const CeedScalar *qweight,
                           CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement non-tensor bases");
}
