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

#include <string.h>
#include <immintrin.h>
#include "ceed-avx.h"

// Contracts on the middle index
// NOTRANSPOSE: V_ajc = T_jb U_abc
// TRANSPOSE:   V_ajc = T_bj U_abc
// If Add != 0, "=" is replaced by "+="
static int CeedTensorContract1_Avx(Ceed ceed, CeedInt A, CeedInt B,
                                   CeedInt C, CeedInt J,
                                   const CeedScalar *restrict t,
                                   CeedTransposeMode tmode, const CeedInt Add,
                                   const CeedScalar *restrict u, CeedScalar *restrict v) {
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
          v[(a*J+j)*C+c] += tq * u[(a*B+b)*C+c];
      }
  return 0;
}

static int CeedTensorContract8_Avx(Ceed ceed, CeedInt A, CeedInt B,
                                   CeedInt C, CeedInt J,
                                   const CeedScalar *restrict t,
                                   CeedTransposeMode tmode, const CeedInt Add,
                                   const CeedScalar *restrict u, CeedScalar *restrict v) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

  const int JJ = 4, CC=8;
  if (C % CC) return CeedError(ceed, 2, "Tensor [%d, %d, %d]: last dimension not divisible by %d", A, B, C, CC);

  if (!Add) {
    for (CeedInt q=0; q<A*J*C; q++) {
      v[q] = (CeedScalar) 0.0;
    }
  }

  for (CeedInt a=0; a<A; a++) {
    // Blocks of 4
    for (CeedInt j=0; j<(J/JJ)*JJ; j+=JJ) {
      for (CeedInt c=0; c<C; c+=CC) {
        __m256d vv[JJ][CC/4]; // Output tile to be held in registers
        for (CeedInt jj=0; jj<JJ; jj++)
          for (CeedInt cc=0; cc<CC/4; cc++)
            vv[jj][cc] = _mm256_loadu_pd(&v[(a*J+j+jj)*C+c+cc*4]);

        for (CeedInt b=0; b<B; b++) {
          for (CeedInt jj=0; jj<JJ; jj++) { // unroll
            __m256d tqv = _mm256_set1_pd(t[(j+jj)*tstride0 + b*tstride1]);
            for (CeedInt cc=0; cc<CC/4; cc++) { // unroll
              vv[jj][cc] += _mm256_mul_pd(tqv,
                              _mm256_loadu_pd(&u[(a*B+b)*C+c+cc*4]));
            }
          }
        }

        for (CeedInt jj=0; jj<JJ; jj++)
          for (CeedInt cc=0; cc<CC/4; cc++)
            _mm256_storeu_pd(&v[(a*J+j+jj)*C+c+cc*4], vv[jj][cc]);

      }
    }
    // Any remainder
    for (CeedInt j=(J/JJ)*JJ; j<J; j++) {
      for (CeedInt b=0; b<B; b++) {
        CeedScalar tq = t[j*tstride0 + b*tstride1];
        for (CeedInt c=0; c<C; c++)
          v[(a*J+j)*C+c] += tq * u[(a*B+b)*C+c];
      }
    }
  }
  return 0;
}

static int CeedBasisApply_Avx(CeedBasis basis, CeedInt nelem,
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
  const CeedInt add = (tmode == CEED_TRANSPOSE);
  const CeedInt blksize = 8;
  const CeedScalar *u;
  CeedScalar *v;
  if (U) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u); CeedChk(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    return CeedError(ceed, 1,
                     "An input vector is required for this CeedEvalMode");
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_HOST, &v); CeedChk(ierr);

  if ((nelem != 1) && (nelem != blksize))
    return CeedError(ceed, 1,
                     "This backend does not support BasisApply for %d elements", nelem);

  int (*CeedTensorContract_Avx)() = NULL;
  if (nelem == blksize)
    CeedTensorContract_Avx = CeedTensorContract8_Avx;
  else
    CeedTensorContract_Avx = CeedTensorContract1_Avx;

  if (tmode == CEED_TRANSPOSE) {
    const CeedInt vsize = nelem*ncomp*ndof;
    for (CeedInt i = 0; i < vsize; i++)
      v[i] = (CeedScalar) 0.0;
  }
  bool tensorbasis;
  ierr = CeedBasisGetTensorStatus(basis, &tensorbasis); CeedChk(ierr);
  if (tensorbasis) {
    CeedInt P1d, Q1d;
    ierr = CeedBasisGetNumNodes1D(basis, &P1d); CeedChk(ierr);
    ierr = CeedBasisGetNumQuadraturePoints1D(basis, &Q1d); CeedChk(ierr);

    switch (emode) {
    case CEED_EVAL_INTERP: {
      CeedInt P = P1d, Q = Q1d;
      if (tmode == CEED_TRANSPOSE) {
        P = Q1d; Q = P1d;
      }
      CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
      CeedScalar tmp[2][nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
      CeedScalar *interp1d;
      ierr = CeedBasisGetInterp(basis, &interp1d); CeedChk(ierr);
      for (CeedInt d=0; d<dim; d++) {
        ierr = CeedTensorContract_Avx(ceed, pre, P, post, Q,
                                      interp1d, tmode, add&&(d==dim-1),
                                      d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]);
        CeedChk(ierr);
        pre /= P;
        post *= Q;
      }
    } break;
    case CEED_EVAL_GRAD: {
      // In CEED_NOTRANSPOSE mode:
      // u has shape [dim, ncomp, P^dim, nelem], row-major layout
      // v has shape [dim, ncomp, Q^dim, nelem], row-major layout
      // In CEED_TRANSPOSE mode, the sizes of u and v are switched.
      CeedInt P = P1d, Q = Q1d;
      if (tmode == CEED_TRANSPOSE) {
        P = Q1d, Q = Q1d;
      }
      CeedBasis_Avx *impl;
      ierr = CeedBasisGetData(basis, (void*)&impl); CeedChk(ierr);
      CeedScalar interp[nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
      CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
      CeedScalar tmp[2][nelem*ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
      CeedScalar *interp1d;
      ierr = CeedBasisGetInterp(basis, &interp1d); CeedChk(ierr);
      // Interpolate to quadrature points (NoTranspose)
      //  or Grad to quadrature points (Transpose)
      for (CeedInt d=0; d<dim; d++) {
        ierr = CeedTensorContract_Avx(ceed, pre, P, post, Q,
                                      (tmode == CEED_NOTRANSPOSE
                                       ? interp1d
                                       : impl->colograd1d),
                                      tmode, add&&(d>0),
                                      (tmode == CEED_NOTRANSPOSE
                                       ? (d==0?u:tmp[d%2])
                                       : u + d*nqpt*ncomp*nelem),
                                      (tmode == CEED_NOTRANSPOSE
                                       ? (d==dim-1?interp:tmp[(d+1)%2])
                                       : interp));
        CeedChk(ierr);
        pre /= P;
        post *= Q;
      }
      // Grad to quadrature points (NoTranspose)
      //  or Interpolate to dofs (Transpose)
      P = Q1d, Q = Q1d;
      if (tmode == CEED_TRANSPOSE) {
        P = Q1d, Q = P1d;
      }
      pre = ncomp*CeedIntPow(P, dim-1), post = nelem;
      for (CeedInt d=0; d<dim; d++) {
        ierr = CeedTensorContract_Avx(ceed, pre, P, post, Q,
                                      (tmode == CEED_NOTRANSPOSE
                                       ? impl->colograd1d
                                       : interp1d),
                                      tmode, add&&(d==dim-1),
                                      (tmode == CEED_NOTRANSPOSE
                                       ? interp
                                       : (d==0?interp:tmp[d%2])),
                                      (tmode == CEED_NOTRANSPOSE
                                       ? v + d*nqpt*ncomp*nelem
                                       : (d==dim-1?v:tmp[(d+1)%2])));
        CeedChk(ierr);
        pre /= P;
        post *= Q;
      }
    } break;
    case CEED_EVAL_WEIGHT: {
      if (tmode == CEED_TRANSPOSE)
        return CeedError(ceed, 1,
                         "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      CeedInt Q = Q1d;
      CeedScalar *qweight1d;
      ierr = CeedBasisGetQWeights(basis, &qweight1d); CeedChk(ierr);
      for (CeedInt d=0; d<dim; d++) {
        CeedInt pre = CeedIntPow(Q, dim-d-1), post = CeedIntPow(Q, d);
        for (CeedInt i=0; i<pre; i++)
          for (CeedInt j=0; j<Q; j++)
            for (CeedInt k=0; k<post; k++) {
              CeedScalar w = qweight1d[j]
                             * (d == 0 ? 1 : v[((i*Q + j)*post + k)*nelem]);
              for (CeedInt e=0; e<nelem; e++)
                v[((i*Q + j)*post + k)*nelem + e] = w;
            }
      }
    } break;
    case CEED_EVAL_DIV:
      return CeedError(ceed, 1, "CEED_EVAL_DIV not supported");
    case CEED_EVAL_CURL:
      return CeedError(ceed, 1, "CEED_EVAL_CURL not supported");
    case CEED_EVAL_NONE:
      return CeedError(ceed, 1,
                       "CEED_EVAL_NONE does not make sense in this context");
    }
  } else {
    // Non-tensor basis
    switch (emode) {
    case CEED_EVAL_INTERP: {
      CeedInt P = ndof, Q = nqpt;
      CeedScalar *interp;
      ierr = CeedBasisGetInterp(basis, &interp); CeedChk(ierr);
      if (tmode == CEED_TRANSPOSE) {
        P = nqpt; Q = ndof;
      }
      ierr = CeedTensorContract_Avx(ceed, ncomp, P, nelem, Q,
                                    interp, tmode, add, u, v);
      CeedChk(ierr);
    }
    break;
    case CEED_EVAL_GRAD: {
      CeedInt P = ndof, Q = dim*nqpt;
      CeedScalar *grad;
      ierr = CeedBasisGetGrad(basis, &grad); CeedChk(ierr);
      if (tmode == CEED_TRANSPOSE) {
        P = dim*nqpt; Q = ndof;
      }
      ierr = CeedTensorContract_Avx(ceed, ncomp, P, nelem, Q,
                                    grad, tmode, add, u, v);
      CeedChk(ierr);
    }
    break;
    case CEED_EVAL_WEIGHT: {
      if (tmode == CEED_TRANSPOSE)
        return CeedError(ceed, 1,
                         "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
      CeedScalar *qweight;
      ierr = CeedBasisGetQWeights(basis, &qweight); CeedChk(ierr);
      for (CeedInt i=0; i<nqpt; i++)
        for (CeedInt e=0; e<nelem; e++)
          v[i*nelem + e] = qweight[i];
    } break;
    case CEED_EVAL_DIV:
      return CeedError(ceed, 1, "CEED_EVAL_DIV not supported");
    case CEED_EVAL_CURL:
      return CeedError(ceed, 1, "CEED_EVAL_CURL not supported");
    case CEED_EVAL_NONE:
      return CeedError(ceed, 1,
                       "CEED_EVAL_NONE does not make sense in this context");
    }
  }
  if (U) {
    ierr = CeedVectorRestoreArrayRead(U, &u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(V, &v); CeedChk(ierr);
  return 0;
}

static int CeedBasisDestroyNonTensor_Avx(CeedBasis basis) {
  return 0;
}

static int CeedBasisDestroyTensor_Avx(CeedBasis basis) {
  int ierr;
  CeedBasis_Avx *impl;
  ierr = CeedBasisGetData(basis, (void*)&impl); CeedChk(ierr);

  ierr = CeedFree(&impl->colograd1d); CeedChk(ierr);
  ierr = CeedFree(&impl); CeedChk(ierr);

  return 0;
}

int CeedBasisCreateTensorH1_Avx(CeedInt dim, CeedInt P1d,
                                CeedInt Q1d, const CeedScalar *interp1d,
                                const CeedScalar *grad1d,
                                const CeedScalar *qref1d,
                                const CeedScalar *qweight1d,
                                CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedBasis_Avx *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  ierr = CeedMalloc(Q1d*Q1d, &impl->colograd1d); CeedChk(ierr);
  ierr = CeedBasisGetCollocatedGrad(basis, impl->colograd1d); CeedChk(ierr);
  ierr = CeedBasisSetData(basis, (void*)&impl); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Avx); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyTensor_Avx); CeedChk(ierr);
  return 0;
}



int CeedBasisCreateH1_Avx(CeedElemTopology topo, CeedInt dim,
                          CeedInt ndof, CeedInt nqpts,
                          const CeedScalar *interp,
                          const CeedScalar *grad,
                          const CeedScalar *qref,
                          const CeedScalar *qweight,
                          CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_Avx); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroyNonTensor_Avx); CeedChk(ierr);
  return 0;
}
