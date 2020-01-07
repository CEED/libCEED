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

#include "ceed-avx.h"

//------------------------------------------------------------------------------
// Blocked Tensor Contract
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_Blocked(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
    CeedTransposeMode tmode, const CeedInt Add, const CeedScalar *restrict u,
    CeedScalar *restrict v, const CeedInt JJ, const CeedInt CC) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

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

//------------------------------------------------------------------------------
// Serial Tensor Contract Remainder
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_Remainder(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
    CeedTransposeMode tmode, const CeedInt Add, const CeedScalar *restrict u,
    CeedScalar *restrict v, const CeedInt JJ, const CeedInt CC) {
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

//------------------------------------------------------------------------------
// Serial Tensor Contract C=1
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_Single(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
    CeedTransposeMode tmode, const CeedInt Add, const CeedScalar *restrict u,
    CeedScalar *restrict v, const CeedInt AA, const CeedInt JJ) {
  CeedInt tstride0 = B, tstride1 = 1;
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }

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
    for (CeedInt j=(J/JJ)*JJ; j<J; j++) {
      CeedScalar tq = t[j*tstride0 + b*tstride1];
      for (CeedInt a=Abreak; a<A; a++)
        v[a*J+j] += tq * u[a*B+b];
    }
  }
  return 0;
}

//------------------------------------------------------------------------------
// Tensor Contract - Common Sizes
//------------------------------------------------------------------------------
static int CeedTensorContract_Avx_Blocked_4_8(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
    CeedTransposeMode tmode, const CeedInt Add, const CeedScalar *restrict u,
    CeedScalar *restrict v) {
  return CeedTensorContract_Avx_Blocked(contract, A, B, C, J, t, tmode, Add, u,
                                        v, 4, 8);
}
static int CeedTensorContract_Avx_Remainder_8_8(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
    CeedTransposeMode tmode, const CeedInt Add, const CeedScalar *restrict u,
    CeedScalar *restrict v) {
  return CeedTensorContract_Avx_Remainder(contract, A, B, C, J, t, tmode, Add,
                                          u, v, 8, 8);
}
static int CeedTensorContract_Avx_Single_4_8(CeedTensorContract contract,
    CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
    CeedTransposeMode tmode, const CeedInt Add, const CeedScalar *restrict u,
    CeedScalar *restrict v) {
  return CeedTensorContract_Avx_Single(contract, A, B, C, J, t, tmode, Add, u,
                                       v, 4, 8);
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Avx(CeedTensorContract contract, CeedInt A,
                                       CeedInt B, CeedInt C, CeedInt J,
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
    CeedTensorContract_Avx_Single_4_8(contract, A, B, C, J, t, tmode, true, u,
                                      v);
  } else {
    // Blocks of 8 columns
    if (C >= blksize)
      CeedTensorContract_Avx_Blocked_4_8(contract, A, B, C, J, t, tmode, true,
                                         u, v);
    // Remainder of columns
    if (C % blksize)
      CeedTensorContract_Avx_Remainder_8_8(contract, A, B, C, J, t, tmode, true,
                                           u, v);
  }

  return 0;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Avx(CeedTensorContract contract) {
  return 0;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Avx(CeedBasis basis, CeedTensorContract contract) {
  int ierr;
  Ceed ceed;
  ierr = CeedTensorContractGetCeed(contract, &ceed); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply",
                                CeedTensorContractApply_Avx); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy",
                                CeedTensorContractDestroy_Avx); CeedChk(ierr);

  return 0;
}
//------------------------------------------------------------------------------
