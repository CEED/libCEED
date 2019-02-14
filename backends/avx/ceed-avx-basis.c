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

// Blocked Tensor Contact
static inline int CeedTensorContract_Avx_Blocked(Ceed ceed, CeedInt A,
                                          CeedInt B, CeedInt C, CeedInt J,
                                          const CeedScalar *restrict t,
                                          CeedTransposeMode tmode,
                                          const CeedInt Add,
                                          const CeedScalar *restrict u,
                                          CeedScalar *restrict v,
                                          const CeedInt JJ,
                                          const CeedInt CC) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

  if (!Add)
    for (CeedInt q=0; q<A*J*C; q++)
      v[q] = (CeedScalar) 0.0;

  for (CeedInt a=0; a<A; a++) {
    // Blocks of 4 rows
    for (CeedInt j=0; j<(J/JJ)*JJ; j+=JJ) {
      for (CeedInt c=0; c<(C/CC)*CC; c+=CC) {
        __m256d vv[JJ][CC/4]; // Output tile to be held in registers
        for (CeedInt jj=0; jj<JJ; jj++)
          for (CeedInt cc=0; cc<CC/4; cc++)
            vv[jj][cc] = _mm256_loadu_pd(&v[(a*J+j+jj)*C+c+cc*4]);

        for (CeedInt b=0; b<B; b++) {
          for (CeedInt jj=0; jj<JJ; jj++) { // unroll
            __m256d tqv = _mm256_set1_pd(t[(j+jj)*tstride0 + b*tstride1]);
            for (CeedInt cc=0; cc<CC/4; cc++) // unroll
              vv[jj][cc] += _mm256_mul_pd(tqv,
                              _mm256_loadu_pd(&u[(a*B+b)*C+c+cc*4]));
          }
        }
        for (CeedInt jj=0; jj<JJ; jj++)
          for (CeedInt cc=0; cc<CC/4; cc++)
            _mm256_storeu_pd(&v[(a*J+j+jj)*C+c+cc*4], vv[jj][cc]);
      }
    }
    // Remainder of rows
    CeedInt j=(J/JJ)*JJ;
    if (j < J) {
      for (CeedInt c=0; c<(C/CC)*CC; c+=CC) {
        __m256d vv[JJ][CC/4]; // Output tile to be held in registers
        for (CeedInt jj=0; jj<J-j; jj++)
          for (CeedInt cc=0; cc<CC/4; cc++)
            vv[jj][cc] = _mm256_loadu_pd(&v[(a*J+j+jj)*C+c+cc*4]);

        for (CeedInt b=0; b<B; b++) {
          for (CeedInt jj=0; jj<J-j; jj++) { // doesn't unroll
            __m256d tqv = _mm256_set1_pd(t[(j+jj)*tstride0 + b*tstride1]);
            for (CeedInt cc=0; cc<CC/4; cc++) // unroll
              vv[jj][cc] += _mm256_mul_pd(tqv,
                              _mm256_loadu_pd(&u[(a*B+b)*C+c+cc*4]));
          }
        }
        for (CeedInt jj=0; jj<J-j; jj++)
          for (CeedInt cc=0; cc<CC/4; cc++)
            _mm256_storeu_pd(&v[(a*J+j+jj)*C+c+cc*4], vv[jj][cc]);
      }
    }
  }
  return 0;
}

// Serial Tensor Contract Remainder
static inline int CeedTensorContract_Avx_Remainder(Ceed ceed, CeedInt A,
                                            CeedInt B, CeedInt C, CeedInt J,
                                            const CeedScalar *restrict t,
                                            CeedTransposeMode tmode,
                                            const CeedInt Add,
                                            const CeedScalar *restrict u,
                                            CeedScalar *restrict v,
                                            const CeedInt JJ,
                                            const CeedInt CC) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

  CeedInt Jbreak = J%JJ ? (J/JJ)*JJ : (J/JJ-1)*JJ;
  for (CeedInt a=0; a<A; a++) {
    // Blocks of 4 columns
    for (CeedInt c = (C/CC)*CC; c<C; c+=4) {
      // Blocks of 4 rows
      for (CeedInt j=0; j<Jbreak; j+=JJ) {
        __m256d vv[JJ]; // Output tile to be held in registers
        for (CeedInt jj=0; jj<JJ; jj++)
          vv[jj] = _mm256_loadu_pd(&v[(a*J+j+jj)*C+c]);

        for (CeedInt b=0; b<B; b++) {
          __m256d tqu;
          if (C-c == 1)
            tqu = _mm256_set_pd(0.0, 0.0, 0.0, u[(a*B+b)*C+c+0]);
          else if (C-c == 2)
            tqu = _mm256_set_pd(0.0, 0.0, u[(a*B+b)*C+c+1],
                    u[(a*B+b)*C+c+0]);
          else if (C-c == 3)
            tqu = _mm256_set_pd(0.0, u[(a*B+b)*C+c+2], u[(a*B+b)*C+c+1],
                    u[(a*B+b)*C+c+0]);
          else
            tqu = _mm256_loadu_pd(&u[(a*B+b)*C+c]);
          for (CeedInt jj=0; jj<JJ; jj++) // unroll
            vv[jj] += _mm256_mul_pd(tqu,
                        _mm256_set1_pd(t[(j+jj)*tstride0 + b*tstride1]));
        }
        for (CeedInt jj=0; jj<JJ; jj++)
          _mm256_storeu_pd(&v[(a*J+j+jj)*C+c], vv[jj]);
      }
    }
    // Remainder of rows, all columns
    for (CeedInt j=Jbreak; j<J; j++)
      for (CeedInt b=0; b<B; b++) {
        CeedScalar tq = t[j*tstride0 + b*tstride1];
        for (CeedInt c=(C/CC)*CC; c<C; c++)
          v[(a*J+j)*C+c] += tq * u[(a*B+b)*C+c];
      }
  }
  return 0;
}

// Serial Tensor Contract C=1 Case
static inline int CeedTensorContract_Avx_Single(Ceed ceed, CeedInt A,
                                         CeedInt B, CeedInt C, CeedInt J,
                                         const CeedScalar *restrict t,
                                         CeedTransposeMode tmode,
                                         const CeedInt Add,
                                         const CeedScalar *restrict u,
                                         CeedScalar *restrict v,
                                         const CeedInt AA,
                                         const CeedInt JJ) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

  if (!Add)
    for (CeedInt q=0; q<A*J*C; q++)
      v[q] = (CeedScalar) 0.0;

  // Blocks of 4 rows
  for (CeedInt a=0; a<(A/AA)*AA; a+=AA) {
    for (CeedInt j=0; j<(J/JJ)*JJ; j+=JJ) {
      __m256d vv[AA][JJ/4]; // Output tile to be held in registers
      for (CeedInt aa=0; aa<AA; aa++)
        for (CeedInt jj=0; jj<JJ/4; jj++)
          vv[aa][jj] = _mm256_loadu_pd(&v[(a+aa)*J+j+jj*4]);

      for (CeedInt b=0; b<B; b++) {
        for (CeedInt jj=0; jj<JJ/4; jj++) { // unroll
          __m256d tqv = _mm256_set_pd(t[(j+jj*4+3)*tstride0 + b*tstride1],
                                      t[(j+jj*4+2)*tstride0 + b*tstride1],
                                      t[(j+jj*4+1)*tstride0 + b*tstride1],
                                      t[(j+jj*4+0)*tstride0 + b*tstride1]);
          for (CeedInt aa=0; aa<AA; aa++) // unroll
            vv[aa][jj] += _mm256_mul_pd(tqv, _mm256_set1_pd(u[(a+aa)*B+b]));
        }
      }
      for (CeedInt aa=0; aa<AA; aa++)
        for (CeedInt jj=0; jj<JJ/4; jj++)
          _mm256_storeu_pd(&v[(a+aa)*J+j+jj*4], vv[aa][jj]);
    }
  }
  // Remainder of rows
  CeedInt a=(A/AA)*AA;
  for (CeedInt j=0; j<(J/JJ)*JJ; j+=JJ) {
    __m256d vv[AA][JJ/4]; // Output tile to be held in registers
    for (CeedInt aa=0; aa<A-a; aa++)
      for (CeedInt jj=0; jj<JJ/4; jj++)
        vv[aa][jj] = _mm256_loadu_pd(&v[(a+aa)*J+j+jj*4]);

    for (CeedInt b=0; b<B; b++) {
      for (CeedInt jj=0; jj<JJ/4; jj++) { // unroll
        __m256d tqv = _mm256_set_pd(t[(j+jj*4+3)*tstride0 + b*tstride1],
                                    t[(j+jj*4+2)*tstride0 + b*tstride1],
                                    t[(j+jj*4+1)*tstride0 + b*tstride1],
                                    t[(j+jj*4+0)*tstride0 + b*tstride1]);
        for (CeedInt aa=0; aa<A-a; aa++) // unroll
          vv[aa][jj] += _mm256_mul_pd(tqv, _mm256_set1_pd(u[(a+aa)*B+b]));
      }
    }
    for (CeedInt aa=0; aa<A-a; aa++)
      for (CeedInt jj=0; jj<JJ/4; jj++)
        _mm256_storeu_pd(&v[(a+aa)*J+j+jj*4], vv[aa][jj]);
  }
  // Column remainder
  CeedInt Abreak = A%AA ? (A/AA)*AA : (A/AA-1)*AA;
  // Blocks of 4 columns
  for (CeedInt j = (J/JJ)*JJ; j<J; j+=4) {
    // Blocks of 4 rows
    for (CeedInt a=0; a<Abreak; a+=AA) {
      __m256d vv[AA]; // Output tile to be held in registers
      for (CeedInt aa=0; aa<AA; aa++)
        vv[aa] = _mm256_loadu_pd(&v[(a+aa)*J+j]);

      for (CeedInt b=0; b<B; b++) {
        __m256d tqv;
        if (J-j == 1)
          tqv = _mm256_set_pd(0.0, 0.0, 0.0, t[(j+0)*tstride0 + b*tstride1]);
        else if (J-j == 2)
          tqv = _mm256_set_pd(0.0, 0.0, t[(j+1)*tstride0 + b*tstride1],
                  t[(j+0)*tstride0 + b*tstride1]);
        else if (J-3 == j)
          tqv = _mm256_set_pd(0.0, t[(j+2)*tstride0 + b*tstride1],
                  t[(j+1)*tstride0 + b*tstride1],
                  t[(j+0)*tstride0 + b*tstride1]);
        else
          tqv = _mm256_set_pd(t[(j+3)*tstride0 + b*tstride1],
                  t[(j+2)*tstride0 + b*tstride1],
                  t[(j+1)*tstride0 + b*tstride1],
                  t[(j+0)*tstride0 + b*tstride1]);
        for (CeedInt aa=0; aa<AA; aa++) // unroll
          vv[aa] += _mm256_mul_pd(tqv, _mm256_set1_pd(u[(a+aa)*B+b]));
      }
      for (CeedInt aa=0; aa<AA; aa++)
        _mm256_storeu_pd(&v[(a+aa)*J+j], vv[aa]);
    }
  }
  // Remainder of rows, all columns
  for (CeedInt b=0; b<B; b++) {
    for (CeedInt j=(J/JJ)*JJ; j<J; j++){ 
      CeedScalar tq = t[j*tstride0 + b*tstride1];
      for (CeedInt a=Abreak; a<A; a++)
        v[a*J+j] += tq * u[a*B+b];
    }
  }
  return 0;
}

// Specific Variants
static int CeedTensorContract_Avx_Blocked_4_8(Ceed ceed, CeedInt A, CeedInt B,
                                              CeedInt C, CeedInt J,
                                              const CeedScalar *restrict t,
                                              CeedTransposeMode tmode,
                                              const CeedInt Add,
                                              const CeedScalar *restrict u,
                                              CeedScalar *restrict v) {
  return CeedTensorContract_Avx_Blocked(ceed, A, B, C, J, t, tmode, Add, u, v,
                                        4, 8);
}
static int CeedTensorContract_Avx_Remainder_4_8(Ceed ceed, CeedInt A, CeedInt B,
                                                CeedInt C, CeedInt J,
                                                const CeedScalar *restrict t,
                                                CeedTransposeMode tmode,
                                                const CeedInt Add,
                                                const CeedScalar *restrict u,
                                                CeedScalar *restrict v) {
  return CeedTensorContract_Avx_Remainder(ceed, A, B, C, J, t, tmode, Add, u, v,
                                          4, 8);
}
static int CeedTensorContract_Avx_Single_4_8(Ceed ceed, CeedInt A, CeedInt B,
                                             CeedInt C, CeedInt J,
                                             const CeedScalar *restrict t,
                                             CeedTransposeMode tmode,
                                             const CeedInt Add,
                                             const CeedScalar *restrict u,
                                             CeedScalar *restrict v) {
  return CeedTensorContract_Avx_Single(ceed, A, B, C, J, t, tmode, Add, u, v,
                                       4, 8);
}

// Switch for Tensor Contract
static int CeedTensorContract_Avx(Ceed ceed, CeedInt A, CeedInt B,
                                  CeedInt C, CeedInt J,
                                  const CeedScalar *restrict t,
                                  CeedTransposeMode tmode,
                                  const CeedInt Add,
                                  const CeedScalar *restrict u,
                                  CeedScalar *restrict v) {
  const CeedInt blksize = 8;

  if (!Add)
    for (CeedInt q=0; q<A*J*C; q++)
      v[q] = (CeedScalar) 0.0;

  if (C == 1) {
    // Serial C=1 Case
    CeedTensorContract_Avx_Single_4_8(ceed, A, B, C, J, t, tmode, true, u, v);
  } else {
    // Blocks of 8 columns
    if (C >= blksize)
      CeedTensorContract_Avx_Blocked_4_8(ceed, A, B, C, J, t, tmode, true, u,
                                         v);
    // Remainder of columns
    if (C % blksize)
      CeedTensorContract_Avx_Remainder_4_8(ceed, A, B, C, J, t, tmode, true, u,
                                           v);
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
  const CeedScalar *u;
  CeedScalar *v;
  if (U) {
    ierr = CeedVectorGetArrayRead(U, CEED_MEM_HOST, &u); CeedChk(ierr);
  } else if (emode != CEED_EVAL_WEIGHT) {
    return CeedError(ceed, 1,
                     "An input vector is required for this CeedEvalMode");
  }
  ierr = CeedVectorGetArray(V, CEED_MEM_HOST, &v); CeedChk(ierr);

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
      ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);
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
  ierr = CeedBasisGetData(basis, (void *)&impl); CeedChk(ierr);

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
  ierr = CeedBasisSetData(basis, (void *)&impl); CeedChk(ierr);

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
