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

#include "ceed-magma.h"

static int CeedBasisApply_Magma(CeedBasis basis, CeedInt nelem, 
                                CeedTransposeMode tmode, CeedEvalMode emode,
                                CeedVector U, CeedVector V) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedInt dim, ncomp, ndof, nqpt;
  ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
  ierr = CeedBasisGetNumComponents(basis, &ncomp); CeedChk(ierr);
  ierr = CeedBasisGetNumNodes(basis, &ndof); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints(basis, &nqpt); CeedChk(ierr);
  CeedTensorContract contract;
  ierr = CeedBasisGetTensorContract(basis, &contract); CeedChk(ierr);
  const CeedInt add = (tmode == CEED_TRANSPOSE);
  const CeedScalar *u;
  CeedScalar *v;
  if (U) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u); CeedChk(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    return CeedError(ceed, 1,
                     "An input vector is required for this CeedEvalMode");
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_HOST, &v); CeedChk(ierr);

  // If input scalar is on CPU, call CPU code
  if (magma_is_devptr(v)!=1)
      return CeedBasisApply_MagmaCPU(basis, tmode, emode, u, v);

  #define tmp(i) ( tmp + (i)*ldtmp)
 
  CeedBasis_Magma *impl;
  ierr = CeedBasisGetData(vec, (void*)&impl; CeedChk(ierr);
  #ifndef USE_MAGMA_BATCH4
  const CeedInt add = (tmode == CEED_TRANSPOSE);
  #endif

  CeedInt P1d, Q1d;
  ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
  ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);

  if (nelem != 1)
    return CeedError(ceed, 1,
                     "This backend does not support BasisApply for multiple elements");

  CeedDebug("\033[01m[CeedBasisApply_Magma] vsize=%d",
            ncomp*CeedIntPow(P1d, dim));

  if (tmode == CEED_TRANSPOSE) {
    #ifdef USE_MAGMA_BATCH3
        #ifndef USE_MAGMA_BATCH4
        const CeedInt vsize = ncomp*CeedIntPow(P1d, dim);
        magmablas_dlaset(MagmaFull, vsize, 1, 0., 0., v, vsize );
        #endif                             
    #else
    for (CeedInt i = 0; i < vsize; i++)
      v[i] = (CeedScalar) 0;
    #endif
  }
  if (emode & CEED_EVAL_INTERP) {
    CeedInt P = P1d, Q = Q1d;
    if (tmode == CEED_TRANSPOSE) {
      P = Q1d; Q = P1d;
    }

    #ifndef USE_MAGMA_BATCH4
    int ldtmp = ncomp*Q*CeedIntPow(P>Q?P:Q,dim-1);
    #endif

    #ifdef USE_MAGMA_BATCH3
        #ifndef USE_MAGMA_BATCH4
        CeedScalar *tmp;
        ierr = magma_malloc( (void**)&tmp, 2* ldtmp * sizeof(CeedScalar));
        CeedChk(ierr);
        #endif
    #else
    CeedScalar tmp[2*ldtmp];
    #endif

    #ifndef USE_MAGMA_BATCH4
    CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = 1;
    for (CeedInt d=0; d<dim; d++) {
      ierr = CeedTensorContract_Magma(ceed, pre, P, post, Q, 
                                      impl->dinterp1d,
                                      tmode, add&&(d==dim-1),
                                      d==0?u:tmp(d%2), d==dim-1?v:tmp((d+1)%2));
      CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
    #else
    ////////////////////////////////////////////////
    magmablas_dbasis_apply_batched_eval_interp(P, Q, dim, ncomp,
      impl->dinterp1d, tmode, u, 0, v, 0, 1);
    ////////////////////////////////////////////////
    #endif
    
    if (tmode == CEED_NOTRANSPOSE) {
      v += nqpt;
    } else {
      u += nqpt;
    }
    #ifdef USE_MAGMA_BATCH3
        #ifndef USE_MAGMA_BATCH4
        magma_free(tmp);
        #endif
    #endif
  }
  if (emode & CEED_EVAL_GRAD) {
    CeedInt P = P1d, Q = Q1d;
    // In CEED_NOTRANSPOSE mode:
    // u is (P^dim x nc), column-major layout (nc = ncomp)
    // v is (Q^dim x nc x dim), column-major layout (nc = ncomp)
    // In CEED_TRANSPOSE mode, the sizes of u and v are switched.
    if (tmode == CEED_TRANSPOSE) {
      P = Q1d, Q = P1d;
    }

    #ifndef USE_MAGMA_BATCH4
    int ldtmp = ncomp*Q*CeedIntPow(P>Q?P:Q,dim-1); 
    #endif

    #ifdef USE_MAGMA_BATCH3
        #ifndef USE_MAGMA_BATCH4
        CeedScalar *tmp;
        ierr = magma_malloc( (void**)&tmp, 2 * ldtmp * sizeof(CeedScalar));
        CeedChk(ierr);
        #endif
    #else
    CeedScalar tmp[2*ldtmp];
    #endif

    #ifndef USE_MAGMA_BATCH4
    for (CeedInt p = 0; p < dim; p++) {
      CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = 1;
      for (CeedInt d=0; d<dim; d++) {
        ierr = CeedTensorContract_Magma(ceed, pre, P, post, Q,
                                        (p==d)? impl->dgrad1d: impl->dinterp1d,
                                        tmode, add&&(d==dim-1),
                                        d==0?u:tmp(d%2), d==dim-1?v:tmp((d+1)%2));
        CeedChk(ierr);
        pre /= P;
        post *= Q;
      }
      if (tmode == CEED_NOTRANSPOSE) {
        v += nqpt;
      } else {
        u += nqpt;
      }
    }
    #else
    magmablas_dbasis_apply_batched_eval_grad(P, Q, dim, ncomp, nqpt,
      impl->dinterp1d, impl->dgrad1d, tmode, u, 0, v, 0, 1);
    #endif
    
    #ifdef USE_MAGMA_BATCH3
        #ifndef USE_MAGMA_BATCH4
        magma_free(tmp);
        #endif
    #endif
  }
  if (emode & CEED_EVAL_WEIGHT) {
    if (tmode == CEED_TRANSPOSE)
      return CeedError(ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    CeedInt Q = Q1d;
    
    #ifdef USE_MAGMA_BATCH4
    magmablas_dbasis_apply_batched_eval_weight(Q, dim, impl->dqweight1d, v, 0, 1);
    #else
    for (CeedInt d=0; d<dim; d++) {
      CeedScalar *qweight1d;
      ierr = CeedBasisGetQWeights(basis, &qweight1d); CeedChk(ierr);
      CeedInt pre = CeedIntPow(Q, dim-d-1), post = CeedIntPow(Q, d);

      #ifdef USE_MAGMA_BATCH3
      CeedScalar *dqweight1d = impl->dqweight1d;
      magma_template<<i=0:pre, j=0:Q, k=0:post>>
          (CeedScalar *v, int d, CeedScalar *dqweight1d) {  
            v[(i*jend + j)*kend + k] = dqweight1d[j] *
              (d == 0 ? 1: v[(i*jend + j)*kend + k]);
      }
      #else
      for (CeedInt i=0; i<pre; i++) {
        for (CeedInt j=0; j<Q; j++) {
          for (CeedInt k=0; k<post; k++) {
            v[(i*Q + j)*post + k] = qweight1d[j]
                                    * (d == 0 ? 1 : v[(i*Q + j)*post + k]);
          }
        }
      }
      #endif
    }
    #endif
  }
  return 0;
}

static int CeedBasisDestroy_Magma(CeedBasis basis) {
  int ierr;
  CeedBasis_Magma *impl;
  ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);
  
  #ifdef USE_MAGMA_BATCH3
  ierr = magma_free(impl->dqref1d); CeedChk(ierr);
  ierr = magma_free(impl->dinterp1d); CeedChk(ierr);
  ierr = magma_free(impl->dgrad1d); CeedChk(ierr);
  ierr = magma_free(impl->dqweight1d); CeedChk(ierr);
  #else
  ierr = CeedFree(&impl->dqref1d); CeedChk(ierr);
  ierr = CeedFree(&impl->dinterp1d); CeedChk(ierr);
  ierr = CeedFree(&impl->dgrad1d); CeedChk(ierr);
  ierr = CeedFree(&impl->dqweight1d); CeedChk(ierr);
  #endif

  ierr = CeedFree(&impl); CeedChk(ierr);

  return 0;
}

static int CeedBasisCreateTensorH1_Magma(CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis) {
  int ierr;
  CeedBasis_Magma *impl;

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_Magma); CeedChk(ierr);

  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  ierr = CeedBasisSetData(basis, (void *)&impl); CeedChk(ierr);

#ifdef USE_MAGMA_BATCH3
  // Copy qref1d to the GPU
  ierr = magma_malloc((void**)&impl->dqref1d, Q1d*sizeof(qref1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d, sizeof(qref1d[0]), qref1d, 1, impl->dqref1d, 1);
  
  // Copy interp1d to the GPU
  ierr = magma_malloc((void**)&impl->dinterp1d, Q1d*P1d*sizeof(interp1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d*P1d, sizeof(interp1d[0]), interp1d, 1, impl->dinterp1d, 1);

  // Copy grad1d to the GPU
  ierr = magma_malloc((void**)&impl->dgrad1d, Q1d*P1d*sizeof(grad1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d*P1d, sizeof(grad1d[0]), grad1d, 1, impl->dgrad1d, 1);

  // Copy qweight1d to the GPU
  ierr = magma_malloc((void**)&impl->dqweight1d, Q1d*sizeof(qweight1d[0]));
  CeedChk(ierr);
  magma_setvector(Q1d, sizeof(qweight1d[0]), qweight1d, 1, impl->dqweight1d, 1);
#else
  // Copy qref1d to the CPU
  ierr = CeedMalloc(Q1d, &impl->dqref1d); CeedChk(ierr);
  memcpy(impl->dqref1d, qref1d, Q1d*sizeof(qref1d[0]));

  // Copy interp1d to the CPU
  ierr = CeedMalloc(Q1d*P1d, &impl->dinterp1d); CeedChk(ierr);
  memcpy(impl->dinterp1d, interp1d, Q1d*P1d*sizeof(interp1d[0]));

  // Copy grad1d to the CPU
  ierr = CeedMalloc(Q1d*P1d, &impl->dgrad1d); CeedChk(ierr);
  memcpy(impl->dgrad1d, grad1d, Q1d*P1d*sizeof(grad1d[0]));

  // Copy qweight1d to the CPU
  ierr = CeedMalloc(Q1d, &impl->dqweight1d); CeedChk(ierr);
  memcpy(impl->dqweight1d, qweight1d,  Q1d*sizeof(qweight1d[0]));
#endif
  return 0;
}

static int CeedBasisCreateH1_Magma(CeedElemTopology topo, CeedInt dim,
                                   CeedInt ndof, CeedInt nqpts,
                                   const CeedScalar *interp,
                                   const CeedScalar *grad,
                                   const CeedScalar *qref,
                                   const CeedScalar *qweight,
                                   CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(op, &ceed); CeedChk(ierr);

  return CeedError(basis->ceed, 1, "Backend does not implement non-tensor bases");
}
