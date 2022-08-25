// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <immintrin.h>
#include <stdbool.h>

#include "ceed-avx.h"

// c += a * b
#ifdef __FMA__
#define fmadd(c, a, b) (c) = _mm256_fmadd_pd((a), (b), (c))
#else
#define fmadd(c, a, b) (c) += _mm256_mul_pd((a), (b))
#endif

//------------------------------------------------------------------------------
// Blocked Tensor Contract
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_Blocked(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                                 CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v,
                                                 const CeedInt JJ, const CeedInt CC) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;
  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  for (CeedInt a = 0; a < A; a++) {
    // Blocks of 4 rows
    for (CeedInt j = 0; j < (J / JJ) * JJ; j += JJ) {
      for (CeedInt c = 0; c < (C / CC) * CC; c += CC) {
        __m256d vv[JJ][CC / 4];  // Output tile to be held in registers
        for (CeedInt jj = 0; jj < JJ; jj++) {
          for (CeedInt cc = 0; cc < CC / 4; cc++) vv[jj][cc] = _mm256_loadu_pd(&v[(a * J + j + jj) * C + c + cc * 4]);
        }

        for (CeedInt b = 0; b < B; b++) {
          for (CeedInt jj = 0; jj < JJ; jj++) {  // unroll
            __m256d tqv = _mm256_set1_pd(t[(j + jj) * t_stride_0 + b * t_stride_1]);
            for (CeedInt cc = 0; cc < CC / 4; cc++) {  // unroll
              fmadd(vv[jj][cc], tqv, _mm256_loadu_pd(&u[(a * B + b) * C + c + cc * 4]));
            }
          }
        }
        for (CeedInt jj = 0; jj < JJ; jj++) {
          for (CeedInt cc = 0; cc < CC / 4; cc++) _mm256_storeu_pd(&v[(a * J + j + jj) * C + c + cc * 4], vv[jj][cc]);
        }
      }
    }
    // Remainder of rows
    CeedInt j = (J / JJ) * JJ;
    if (j < J) {
      for (CeedInt c = 0; c < (C / CC) * CC; c += CC) {
        __m256d vv[JJ][CC / 4];  // Output tile to be held in registers
        for (CeedInt jj = 0; jj < J - j; jj++) {
          for (CeedInt cc = 0; cc < CC / 4; cc++) vv[jj][cc] = _mm256_loadu_pd(&v[(a * J + j + jj) * C + c + cc * 4]);
        }

        for (CeedInt b = 0; b < B; b++) {
          for (CeedInt jj = 0; jj < J - j; jj++) {  // doesn't unroll
            __m256d tqv = _mm256_set1_pd(t[(j + jj) * t_stride_0 + b * t_stride_1]);
            for (CeedInt cc = 0; cc < CC / 4; cc++) {  // unroll
              fmadd(vv[jj][cc], tqv, _mm256_loadu_pd(&u[(a * B + b) * C + c + cc * 4]));
            }
          }
        }
        for (CeedInt jj = 0; jj < J - j; jj++) {
          for (CeedInt cc = 0; cc < CC / 4; cc++) _mm256_storeu_pd(&v[(a * J + j + jj) * C + c + cc * 4], vv[jj][cc]);
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Serial Tensor Contract Remainder
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_Remainder(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                                   CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v,
                                                   const CeedInt JJ, const CeedInt CC) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;
  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  CeedInt J_break = J % JJ ? (J / JJ) * JJ : (J / JJ - 1) * JJ;
  for (CeedInt a = 0; a < A; a++) {
    // Blocks of 4 columns
    for (CeedInt c = (C / CC) * CC; c < C; c += 4) {
      // Blocks of 4 rows
      for (CeedInt j = 0; j < J_break; j += JJ) {
        __m256d vv[JJ];  // Output tile to be held in registers
        for (CeedInt jj = 0; jj < JJ; jj++) vv[jj] = _mm256_loadu_pd(&v[(a * J + j + jj) * C + c]);

        for (CeedInt b = 0; b < B; b++) {
          __m256d tqu;
          if (C - c == 1) tqu = _mm256_set_pd(0.0, 0.0, 0.0, u[(a * B + b) * C + c + 0]);
          else if (C - c == 2) tqu = _mm256_set_pd(0.0, 0.0, u[(a * B + b) * C + c + 1], u[(a * B + b) * C + c + 0]);
          else if (C - c == 3) tqu = _mm256_set_pd(0.0, u[(a * B + b) * C + c + 2], u[(a * B + b) * C + c + 1], u[(a * B + b) * C + c + 0]);
          else tqu = _mm256_loadu_pd(&u[(a * B + b) * C + c]);
          for (CeedInt jj = 0; jj < JJ; jj++) {  // unroll
            fmadd(vv[jj], tqu, _mm256_set1_pd(t[(j + jj) * t_stride_0 + b * t_stride_1]));
          }
        }
        for (CeedInt jj = 0; jj < JJ; jj++) _mm256_storeu_pd(&v[(a * J + j + jj) * C + c], vv[jj]);
      }
    }
    // Remainder of rows, all columns
    for (CeedInt j = J_break; j < J; j++) {
      for (CeedInt b = 0; b < B; b++) {
        double tq = t[j * t_stride_0 + b * t_stride_1];
        for (CeedInt c = (C / CC) * CC; c < C; c++) v[(a * J + j) * C + c] += tq * u[(a * B + b) * C + c];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Serial Tensor Contract C=1
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_Single(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                                CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v,
                                                const CeedInt AA, const CeedInt JJ) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;
  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  // Blocks of 4 rows
  for (CeedInt a = 0; a < (A / AA) * AA; a += AA) {
    for (CeedInt j = 0; j < (J / JJ) * JJ; j += JJ) {
      __m256d vv[AA][JJ / 4];  // Output tile to be held in registers
      for (CeedInt aa = 0; aa < AA; aa++) {
        for (CeedInt jj = 0; jj < JJ / 4; jj++) vv[aa][jj] = _mm256_loadu_pd(&v[(a + aa) * J + j + jj * 4]);
      }

      for (CeedInt b = 0; b < B; b++) {
        for (CeedInt jj = 0; jj < JJ / 4; jj++) {  // unroll
          __m256d tqv = _mm256_set_pd(t[(j + jj * 4 + 3) * t_stride_0 + b * t_stride_1], t[(j + jj * 4 + 2) * t_stride_0 + b * t_stride_1],
                                      t[(j + jj * 4 + 1) * t_stride_0 + b * t_stride_1], t[(j + jj * 4 + 0) * t_stride_0 + b * t_stride_1]);
          for (CeedInt aa = 0; aa < AA; aa++) {  // unroll
            fmadd(vv[aa][jj], tqv, _mm256_set1_pd(u[(a + aa) * B + b]));
          }
        }
      }
      for (CeedInt aa = 0; aa < AA; aa++) {
        for (CeedInt jj = 0; jj < JJ / 4; jj++) _mm256_storeu_pd(&v[(a + aa) * J + j + jj * 4], vv[aa][jj]);
      }
    }
  }
  // Remainder of rows
  CeedInt a = (A / AA) * AA;
  for (CeedInt j = 0; j < (J / JJ) * JJ; j += JJ) {
    __m256d vv[AA][JJ / 4];  // Output tile to be held in registers
    for (CeedInt aa = 0; aa < A - a; aa++) {
      for (CeedInt jj = 0; jj < JJ / 4; jj++) vv[aa][jj] = _mm256_loadu_pd(&v[(a + aa) * J + j + jj * 4]);
    }

    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt jj = 0; jj < JJ / 4; jj++) {  // unroll
        __m256d tqv = _mm256_set_pd(t[(j + jj * 4 + 3) * t_stride_0 + b * t_stride_1], t[(j + jj * 4 + 2) * t_stride_0 + b * t_stride_1],
                                    t[(j + jj * 4 + 1) * t_stride_0 + b * t_stride_1], t[(j + jj * 4 + 0) * t_stride_0 + b * t_stride_1]);
        for (CeedInt aa = 0; aa < A - a; aa++) {  // unroll
          fmadd(vv[aa][jj], tqv, _mm256_set1_pd(u[(a + aa) * B + b]));
        }
      }
    }
    for (CeedInt aa = 0; aa < A - a; aa++) {
      for (CeedInt jj = 0; jj < JJ / 4; jj++) _mm256_storeu_pd(&v[(a + aa) * J + j + jj * 4], vv[aa][jj]);
    }
  }
  // Column remainder
  CeedInt A_break = A % AA ? (A / AA) * AA : (A / AA - 1) * AA;
  // Blocks of 4 columns
  for (CeedInt j = (J / JJ) * JJ; j < J; j += 4) {
    // Blocks of 4 rows
    for (CeedInt a = 0; a < A_break; a += AA) {
      __m256d vv[AA];  // Output tile to be held in registers
      for (CeedInt aa = 0; aa < AA; aa++) vv[aa] = _mm256_loadu_pd(&v[(a + aa) * J + j]);

      for (CeedInt b = 0; b < B; b++) {
        __m256d tqv;
        if (J - j == 1) {
          tqv = _mm256_set_pd(0.0, 0.0, 0.0, t[(j + 0) * t_stride_0 + b * t_stride_1]);
        } else if (J - j == 2) {
          tqv = _mm256_set_pd(0.0, 0.0, t[(j + 1) * t_stride_0 + b * t_stride_1], t[(j + 0) * t_stride_0 + b * t_stride_1]);
        } else if (J - 3 == j) {
          tqv = _mm256_set_pd(0.0, t[(j + 2) * t_stride_0 + b * t_stride_1], t[(j + 1) * t_stride_0 + b * t_stride_1],
                              t[(j + 0) * t_stride_0 + b * t_stride_1]);
        } else {
          tqv = _mm256_set_pd(t[(j + 3) * t_stride_0 + b * t_stride_1], t[(j + 2) * t_stride_0 + b * t_stride_1],
                              t[(j + 1) * t_stride_0 + b * t_stride_1], t[(j + 0) * t_stride_0 + b * t_stride_1]);
        }
        for (CeedInt aa = 0; aa < AA; aa++) {  // unroll
          fmadd(vv[aa], tqv, _mm256_set1_pd(u[(a + aa) * B + b]));
        }
      }
      for (CeedInt aa = 0; aa < AA; aa++) _mm256_storeu_pd(&v[(a + aa) * J + j], vv[aa]);
    }
  }
  // Remainder of rows, all columns
  for (CeedInt b = 0; b < B; b++) {
    for (CeedInt j = (J / JJ) * JJ; j < J; j++) {
      double tq = t[j * t_stride_0 + b * t_stride_1];
      for (CeedInt a = A_break; a < A; a++) v[a * J + j] += tq * u[a * B + b];
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract - Common Sizes
//------------------------------------------------------------------------------
static int CeedTensorContract_Avx_Blocked_4_8(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                              CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  return CeedTensorContract_Avx_Blocked(contract, A, B, C, J, t, t_mode, add, u, v, 4, 8);
}
static int CeedTensorContract_Avx_Remainder_8_8(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                                CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  return CeedTensorContract_Avx_Remainder(contract, A, B, C, J, t, t_mode, add, u, v, 8, 8);
}
static int CeedTensorContract_Avx_Single_4_8(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                             CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  return CeedTensorContract_Avx_Single(contract, A, B, C, J, t, t_mode, add, u, v, 4, 8);
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Avx(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  const CeedInt blk_size = 8;

  if (!add) {
    for (CeedInt q = 0; q < A * J * C; q++) v[q] = (double)0.0;
  }

  if (C == 1) {
    // Serial C=1 Case
    CeedTensorContract_Avx_Single_4_8(contract, A, B, C, J, t, t_mode, true, u, v);
  } else {
    // Blocks of 8 columns
    if (C >= blk_size) CeedTensorContract_Avx_Blocked_4_8(contract, A, B, C, J, t, t_mode, true, u, v);
    // Remainder of columns
    if (C % blk_size) CeedTensorContract_Avx_Remainder_8_8(contract, A, B, C, J, t, t_mode, true, u, v);
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_f64_Avx(CeedBasis basis, CeedTensorContract contract) {
  Ceed ceed;
  CeedCallBackend(CeedTensorContractGetCeed(contract, &ceed));

  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply", CeedTensorContractApply_Avx));

  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
