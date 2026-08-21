// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "ceed-avx.h"

#ifdef CEED_SCALAR_IS_FP64
#define rtype __m256d
#define loadu _mm256_loadu_pd
#define storeu _mm256_storeu_pd
#define set _mm256_set_pd
#define set1 _mm256_set1_pd
// c += a * b
#ifdef __FMA__
#define fmadd(c, a, b) (c) = _mm256_fmadd_pd((a), (b), (c))
#else
#define fmadd(c, a, b) (c) += _mm256_mul_pd((a), (b))
#endif
#else
#define rtype __m128
#define loadu _mm_loadu_ps
#define storeu _mm_storeu_ps
#define set _mm_set_ps
#define set1 _mm_set1_ps
// c += a * b
#ifdef __FMA__
#define fmadd(c, a, b) (c) = _mm_fmadd_ps((a), (b), (c))
#else
#define fmadd(c, a, b) (c) += _mm_mul_ps((a), (b))
#endif
#endif

//------------------------------------------------------------------------------
// Blocked Tensor Contract
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_Blocked(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                                 const CeedScalar *restrict t, CeedTransposeMode t_mode, const CeedInt add,
                                                 const CeedScalar *restrict u, CeedScalar *restrict v, const CeedInt JJ, const CeedInt CC) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;

  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  for (CeedInt a = 0; a < A; a++) {
    // Blocks of 4 rows
    for (CeedInt j = 0; j < (J / JJ) * JJ; j += JJ) {
      for (CeedInt c = 0; c < (C / CC) * CC; c += CC) {
        rtype vv[JJ][CC / 4];  // Output tile to be held in registers
        for (CeedInt jj = 0; jj < JJ; jj++) {
          for (CeedInt cc = 0; cc < CC / 4; cc++) vv[jj][cc] = loadu(&v[(a * J + j + jj) * C + c + cc * 4]);
        }
        for (CeedInt b = 0; b < B; b++) {
          for (CeedInt jj = 0; jj < JJ; jj++) {  // unroll
            rtype tqv = set1(t[(j + jj) * t_stride_0 + b * t_stride_1]);
            for (CeedInt cc = 0; cc < CC / 4; cc++) {  // unroll
              fmadd(vv[jj][cc], tqv, loadu(&u[(a * B + b) * C + c + cc * 4]));
            }
          }
        }
        for (CeedInt jj = 0; jj < JJ; jj++) {
          for (CeedInt cc = 0; cc < CC / 4; cc++) storeu(&v[(a * J + j + jj) * C + c + cc * 4], vv[jj][cc]);
        }
      }
    }
    // Remainder of rows
    const CeedInt j = (J / JJ) * JJ;

    if (j < J) {
      for (CeedInt c = 0; c < (C / CC) * CC; c += CC) {
        rtype vv[JJ][CC / 4];  // Output tile to be held in registers

        for (CeedInt jj = 0; jj < J - j; jj++) {
          for (CeedInt cc = 0; cc < CC / 4; cc++) vv[jj][cc] = loadu(&v[(a * J + j + jj) * C + c + cc * 4]);
        }
        for (CeedInt b = 0; b < B; b++) {
          for (CeedInt jj = 0; jj < J - j; jj++) {  // doesn't unroll
            rtype tqv = set1(t[(j + jj) * t_stride_0 + b * t_stride_1]);

            for (CeedInt cc = 0; cc < CC / 4; cc++) {  // unroll
              fmadd(vv[jj][cc], tqv, loadu(&u[(a * B + b) * C + c + cc * 4]));
            }
          }
        }
        for (CeedInt jj = 0; jj < J - j; jj++) {
          for (CeedInt cc = 0; cc < CC / 4; cc++) storeu(&v[(a * J + j + jj) * C + c + cc * 4], vv[jj][cc]);
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Serial Tensor Contract Remainder
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_Remainder(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                                   const CeedScalar *restrict t, CeedTransposeMode t_mode, const CeedInt add,
                                                   const CeedScalar *restrict u, CeedScalar *restrict v, const CeedInt JJ, const CeedInt CC) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;

  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  const CeedInt J_break = J % JJ ? (J / JJ) * JJ : (J / JJ - 1) * JJ;

  for (CeedInt a = 0; a < A; a++) {
    // Blocks of 4 columns
    for (CeedInt c = (C / CC) * CC; c < C; c += 4) {
      // Blocks of 4 rows
      for (CeedInt j = 0; j < J_break; j += JJ) {
        rtype vv[JJ];  // Output tile to be held in registers

        for (CeedInt jj = 0; jj < JJ; jj++) vv[jj] = loadu(&v[(a * J + j + jj) * C + c]);
        for (CeedInt b = 0; b < B; b++) {
          rtype tqu;

          if (C - c == 1) {
            tqu = set(0.0, 0.0, 0.0, u[(a * B + b) * C + c + 0]);
          } else if (C - c == 2) {
            tqu = set(0.0, 0.0, u[(a * B + b) * C + c + 1], u[(a * B + b) * C + c + 0]);
          } else if (C - c == 3) {
            tqu = set(0.0, u[(a * B + b) * C + c + 2], u[(a * B + b) * C + c + 1], u[(a * B + b) * C + c + 0]);
          } else {
            tqu = loadu(&u[(a * B + b) * C + c]);
          }
          for (CeedInt jj = 0; jj < JJ; jj++) {  // unroll
            fmadd(vv[jj], tqu, set1(t[(j + jj) * t_stride_0 + b * t_stride_1]));
          }
        }
        for (CeedInt jj = 0; jj < JJ; jj++) storeu(&v[(a * J + j + jj) * C + c], vv[jj]);
      }
    }
    // Remainder of rows, all columns
    for (CeedInt j = J_break; j < J; j++) {
      for (CeedInt b = 0; b < B; b++) {
        const CeedScalar tq = t[j * t_stride_0 + b * t_stride_1];

        for (CeedInt c = (C / CC) * CC; c < C; c++) v[(a * J + j) * C + c] += tq * u[(a * B + b) * C + c];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Serial Tensor Contract C=1
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_Single(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                                CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v,
                                                const CeedInt AA, const CeedInt JJ) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;

  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  // Blocks of 4 rows
  for (CeedInt a = 0; a < (A / AA) * AA; a += AA) {
    for (CeedInt j = 0; j < (J / JJ) * JJ; j += JJ) {
      rtype vv[AA][JJ / 4];  // Output tile to be held in registers

      for (CeedInt aa = 0; aa < AA; aa++) {
        for (CeedInt jj = 0; jj < JJ / 4; jj++) vv[aa][jj] = loadu(&v[(a + aa) * J + j + jj * 4]);
      }
      for (CeedInt b = 0; b < B; b++) {
        for (CeedInt jj = 0; jj < JJ / 4; jj++) {  // unroll
          rtype tqv = set(t[(j + jj * 4 + 3) * t_stride_0 + b * t_stride_1], t[(j + jj * 4 + 2) * t_stride_0 + b * t_stride_1],
                          t[(j + jj * 4 + 1) * t_stride_0 + b * t_stride_1], t[(j + jj * 4 + 0) * t_stride_0 + b * t_stride_1]);

          for (CeedInt aa = 0; aa < AA; aa++) {  // unroll
            fmadd(vv[aa][jj], tqv, set1(u[(a + aa) * B + b]));
          }
        }
      }
      for (CeedInt aa = 0; aa < AA; aa++) {
        for (CeedInt jj = 0; jj < JJ / 4; jj++) storeu(&v[(a + aa) * J + j + jj * 4], vv[aa][jj]);
      }
    }
  }
  // Remainder of rows
  const CeedInt a = (A / AA) * AA;

  for (CeedInt j = 0; j < (J / JJ) * JJ; j += JJ) {
    rtype vv[AA][JJ / 4];  // Output tile to be held in registers

    for (CeedInt aa = 0; aa < A - a; aa++) {
      for (CeedInt jj = 0; jj < JJ / 4; jj++) vv[aa][jj] = loadu(&v[(a + aa) * J + j + jj * 4]);
    }
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt jj = 0; jj < JJ / 4; jj++) {  // unroll
        rtype tqv = set(t[(j + jj * 4 + 3) * t_stride_0 + b * t_stride_1], t[(j + jj * 4 + 2) * t_stride_0 + b * t_stride_1],
                        t[(j + jj * 4 + 1) * t_stride_0 + b * t_stride_1], t[(j + jj * 4 + 0) * t_stride_0 + b * t_stride_1]);

        for (CeedInt aa = 0; aa < A - a; aa++) {  // unroll
          fmadd(vv[aa][jj], tqv, set1(u[(a + aa) * B + b]));
        }
      }
    }
    for (CeedInt aa = 0; aa < A - a; aa++) {
      for (CeedInt jj = 0; jj < JJ / 4; jj++) storeu(&v[(a + aa) * J + j + jj * 4], vv[aa][jj]);
    }
  }
  // Column remainder
  const CeedInt A_break = A % AA ? (A / AA) * AA : (A / AA - 1) * AA;

  // Blocks of 4 columns
  for (CeedInt j = (J / JJ) * JJ; j < J; j += 4) {
    // Blocks of 4 rows
    for (CeedInt a = 0; a < A_break; a += AA) {
      rtype vv[AA];  // Output tile to be held in registers

      for (CeedInt aa = 0; aa < AA; aa++) vv[aa] = loadu(&v[(a + aa) * J + j]);
      for (CeedInt b = 0; b < B; b++) {
        rtype tqv;

        if (J - j == 1) {
          tqv = set(0.0, 0.0, 0.0, t[(j + 0) * t_stride_0 + b * t_stride_1]);
        } else if (J - j == 2) {
          tqv = set(0.0, 0.0, t[(j + 1) * t_stride_0 + b * t_stride_1], t[(j + 0) * t_stride_0 + b * t_stride_1]);
        } else if (J - 3 == j) {
          tqv =
              set(0.0, t[(j + 2) * t_stride_0 + b * t_stride_1], t[(j + 1) * t_stride_0 + b * t_stride_1], t[(j + 0) * t_stride_0 + b * t_stride_1]);
        } else {
          tqv = set(t[(j + 3) * t_stride_0 + b * t_stride_1], t[(j + 2) * t_stride_0 + b * t_stride_1], t[(j + 1) * t_stride_0 + b * t_stride_1],
                    t[(j + 0) * t_stride_0 + b * t_stride_1]);
        }
        for (CeedInt aa = 0; aa < AA; aa++) {  // unroll
          fmadd(vv[aa], tqv, set1(u[(a + aa) * B + b]));
        }
      }
      for (CeedInt aa = 0; aa < AA; aa++) storeu(&v[(a + aa) * J + j], vv[aa]);
    }
  }
  // Remainder of rows, all columns
  for (CeedInt b = 0; b < B; b++) {
    for (CeedInt j = (J / JJ) * JJ; j < J; j++) {
      const CeedScalar tq = t[j * t_stride_0 + b * t_stride_1];

      for (CeedInt a = A_break; a < A; a++) v[a * J + j] += tq * u[a * B + b];
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Even-Odd: Standard Dispatch (reused for half-contractions)
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Avx_StandardDispatch(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                                          const CeedScalar *restrict t, CeedTransposeMode t_mode, const CeedInt add,
                                                          const CeedScalar *restrict u, CeedScalar *restrict v) {
  const CeedInt blk_size = 8;

  if (C == 1) {
    CeedTensorContract_Avx_Single(contract, A, B, C, J, t, t_mode, add, u, v, 4, 8);
  } else {
    if (C >= blk_size) CeedTensorContract_Avx_Blocked(contract, A, B, C, J, t, t_mode, add, u, v, 4, 8);
    if (C % blk_size) CeedTensorContract_Avx_Remainder(contract, A, B, C, J, t, t_mode, add, u, v, 8, 8);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Even-Odd: Symmetry Detection
//------------------------------------------------------------------------------
static int CeedTensorContract_Avx_DetectSymmetry(CeedInt B, CeedInt J, const CeedScalar *t, CeedTransposeMode t_mode) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;

  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  const CeedScalar tol = 1e-14;
  int              sym = 0;

  for (CeedInt j = 0; j < (J + 1) / 2; j++) {
    for (CeedInt b = 0; b < (B + 1) / 2; b++) {
      CeedScalar val      = t[j * t_stride_0 + b * t_stride_1];
      CeedScalar mirror   = t[(J - 1 - j) * t_stride_0 + (B - 1 - b) * t_stride_1];
      CeedScalar scale    = fabs(val) + fabs(mirror);
      CeedScalar diff_sym = fabs(mirror - val);
      CeedScalar diff_ant = fabs(mirror + val);

      if (scale < tol) continue;
      if (sym == 0) {
        if (diff_sym / scale < tol) {
          sym = 1;
        } else if (diff_ant / scale < tol) {
          sym = -1;
        } else {
          return 0;
        }
      } else if (sym == 1) {
        if (diff_sym / scale >= tol) return 0;
      } else {
        if (diff_ant / scale >= tol) return 0;
      }
    }
  }
  return sym;
}

//------------------------------------------------------------------------------
// Even-Odd: Precompute Half-Matrices
//------------------------------------------------------------------------------
static int CeedTensorContract_Avx_ComputeHalfMatrices(CeedInt B, CeedInt J, const CeedScalar *t, CeedTransposeMode t_mode, CeedInt B_half,
                                                      CeedInt J_half, CeedScalar *t_even, CeedScalar *t_odd) {
  CeedInt t_stride_0 = B, t_stride_1 = 1;

  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  for (CeedInt j = 0; j < J_half; j++) {
    for (CeedInt b = 0; b < B / 2; b++) {
      CeedScalar val         = t[j * t_stride_0 + b * t_stride_1];
      CeedScalar mirror      = t[j * t_stride_0 + (B - 1 - b) * t_stride_1];
      t_even[j * B_half + b] = (val + mirror) * (CeedScalar)0.5;
      t_odd[j * B_half + b]  = (val - mirror) * (CeedScalar)0.5;
    }
    if (B % 2) {
      t_even[j * B_half + B / 2] = t[j * t_stride_0 + (B / 2) * t_stride_1];
      t_odd[j * B_half + B / 2]  = (CeedScalar)0.0;
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Even-Odd: Cache Lookup and Populate
//------------------------------------------------------------------------------
static CeedTensorContract_Avx_CacheEntry *CeedTensorContract_Avx_CacheLookup(CeedTensorContract_Avx *data, const CeedScalar *t,
                                                                             CeedTransposeMode t_mode, CeedInt B, CeedInt J) {
  for (CeedInt i = 0; i < data->n_cached; i++) {
    if (data->cache[i].t_ptr == t && data->cache[i].t_mode == t_mode && data->cache[i].B == B && data->cache[i].J == J) return &data->cache[i];
  }
  return NULL;
}

// The t matrix contents may change between applies at the same pointer (e.g. operator assembly rewrites a scratch buffer in place), so the entry
// stores a copy of t and must be refreshed whenever the contents no longer match.
static int CeedTensorContract_Avx_CacheEntryRefresh(CeedTensorContract_Avx_CacheEntry *e, const CeedScalar *t) {
  memcpy(e->t_copy, t, (size_t)e->B * e->J * sizeof(CeedScalar));
  e->symmetry = CeedTensorContract_Avx_DetectSymmetry(e->B, e->J, t, e->t_mode);

  if (e->symmetry != 0) {
    if (!e->t_even) {
      CeedCallBackend(CeedMalloc(e->J_half * e->B_half, &e->t_even));
      CeedCallBackend(CeedMalloc(e->J_half * e->B_half, &e->t_odd));
    }
    CeedTensorContract_Avx_ComputeHalfMatrices(e->B, e->J, t, e->t_mode, e->B_half, e->J_half, e->t_even, e->t_odd);
  } else {
    CeedCallBackend(CeedFree(&e->t_even));
    CeedCallBackend(CeedFree(&e->t_odd));
  }
  return CEED_ERROR_SUCCESS;
}

static int CeedTensorContract_Avx_CachePopulate(CeedTensorContract_Avx *data, const CeedScalar *t, CeedTransposeMode t_mode, CeedInt B, CeedInt J,
                                                CeedTensorContract_Avx_CacheEntry **entry) {
  *entry = NULL;
  if (data->n_cached >= CEED_AVX_EVEN_ODD_CACHE_MAX) return CEED_ERROR_SUCCESS;

  CeedTensorContract_Avx_CacheEntry *e = &data->cache[data->n_cached];

  e->t_ptr  = t;
  e->t_mode = t_mode;
  e->B      = B;
  e->J      = J;
  e->B_half = (B + 1) / 2;
  e->J_half = (J + 1) / 2;
  e->t_even = NULL;
  e->t_odd  = NULL;
  CeedCallBackend(CeedMalloc(B * J, &e->t_copy));
  CeedCallBackend(CeedTensorContract_Avx_CacheEntryRefresh(e, t));

  data->n_cached++;
  *entry = e;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Even-Odd: Apply
//------------------------------------------------------------------------------
static int CeedTensorContract_Avx_EvenOdd(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                          const CeedTensorContract_Avx_CacheEntry *entry, const CeedScalar *restrict u, CeedScalar *restrict v) {
  const CeedInt B_half = entry->B_half;
  const CeedInt J_half = entry->J_half;
  const int     sym    = entry->symmetry;

  CeedScalar u_even[A * B_half * C], u_odd[A * B_half * C];
  CeedScalar w_even[A * J_half * C], w_odd[A * J_half * C];

  memset(u_even, 0, sizeof(u_even));
  memset(u_odd, 0, sizeof(u_odd));

  // Fold input along b
  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B / 2; b++) {
      const CeedInt lo = (a * B + b) * C;
      const CeedInt hi = (a * B + (B - 1 - b)) * C;
      const CeedInt fo = (a * B_half + b) * C;

      for (CeedInt c = 0; c < C; c++) {
        u_even[fo + c] = u[lo + c] + u[hi + c];
        u_odd[fo + c]  = u[lo + c] - u[hi + c];
      }
    }
    if (B % 2) {
      const CeedInt mid_in  = (a * B + B / 2) * C;
      const CeedInt mid_out = (a * B_half + B / 2) * C;

      for (CeedInt c = 0; c < C; c++) {
        u_even[mid_out + c] = u[mid_in + c];
        u_odd[mid_out + c]  = (CeedScalar)0.0;
      }
    }
  }

  // Half-contractions: w_e = t_e * u_e, w_o = t_o * u_o
  memset(w_even, 0, A * J_half * C * sizeof(CeedScalar));
  memset(w_odd, 0, A * J_half * C * sizeof(CeedScalar));
  CeedTensorContract_Avx_StandardDispatch(contract, A, B_half, C, J_half, entry->t_even, CEED_NOTRANSPOSE, true, u_even, w_even);
  CeedTensorContract_Avx_StandardDispatch(contract, A, B_half, C, J_half, entry->t_odd, CEED_NOTRANSPOSE, true, u_odd, w_odd);

  // Unfold output along j
  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt j = 0; j < J / 2; j++) {
      const CeedInt eo   = (a * J_half + j) * C;
      const CeedInt v_lo = (a * J + j) * C;
      const CeedInt v_hi = (a * J + (J - 1 - j)) * C;

      if (sym == 1) {
        for (CeedInt c = 0; c < C; c++) {
          v[v_lo + c] += w_even[eo + c] + w_odd[eo + c];
          v[v_hi + c] += w_even[eo + c] - w_odd[eo + c];
        }
      } else {
        for (CeedInt c = 0; c < C; c++) {
          v[v_lo + c] += w_even[eo + c] + w_odd[eo + c];
          v[v_hi + c] += -w_even[eo + c] + w_odd[eo + c];
        }
      }
    }
    if (J % 2) {
      const CeedInt eo    = (a * J_half + J / 2) * C;
      const CeedInt v_mid = (a * J + J / 2) * C;

      for (CeedInt c = 0; c < C; c++) v[v_mid + c] += w_even[eo + c] + w_odd[eo + c];
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Avx(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  if (!add) {
    for (CeedInt q = 0; q < A * J * C; q++) v[q] = (CeedScalar)0.0;
  }

  // Try even-odd path
  CeedTensorContract_Avx *data;

  CeedCallBackend(CeedTensorContractGetData(contract, &data));
  if (data && B >= CEED_AVX_EVEN_ODD_MIN_DIM && J >= CEED_AVX_EVEN_ODD_MIN_DIM) {
    CeedTensorContract_Avx_CacheEntry *entry = CeedTensorContract_Avx_CacheLookup(data, t, t_mode, B, J);

    if (!entry) {
      CeedCallBackend(CeedTensorContract_Avx_CachePopulate(data, t, t_mode, B, J, &entry));
    } else if (memcmp(entry->t_copy, t, (size_t)B * J * sizeof(CeedScalar))) {
      CeedCallBackend(CeedTensorContract_Avx_CacheEntryRefresh(entry, t));
    }
    if (entry && entry->t_even) return CeedTensorContract_Avx_EvenOdd(contract, A, B, C, J, entry, u, v);
  }

  // Standard path
  CeedTensorContract_Avx_StandardDispatch(contract, A, B, C, J, t, t_mode, true, u, v);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Destroy
//------------------------------------------------------------------------------
static int CeedTensorContractDestroy_Avx(CeedTensorContract contract) {
  CeedTensorContract_Avx *data;

  CeedCallBackend(CeedTensorContractGetData(contract, &data));
  if (data) {
    for (CeedInt i = 0; i < data->n_cached; i++) {
      CeedCallBackend(CeedFree(&data->cache[i].t_copy));
      CeedCallBackend(CeedFree(&data->cache[i].t_even));
      CeedCallBackend(CeedFree(&data->cache[i].t_odd));
    }
    CeedCallBackend(CeedFree(&data));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Avx(CeedTensorContract contract) {
  Ceed                    ceed = CeedTensorContractReturnCeed(contract);
  CeedTensorContract_Avx *data;

  CeedCallBackend(CeedCalloc(1, &data));
  CeedCallBackend(CeedTensorContractSetData(contract, data));
  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply", CeedTensorContractApply_Avx));
  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Destroy", CeedTensorContractDestroy_Avx));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
