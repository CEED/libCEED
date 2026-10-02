// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <arm_sme.h>
#include <arm_sve.h>
#include <stdint.h>

#include "ceed-sme.h"

#ifdef CEED_SCALAR_IS_FP64
#define rtype svfloat64_t
#define vlength() ((CeedSize)svcntd())
#define whilelt(i, n) svwhilelt_b64((int64_t)(i), (int64_t)(n))
#define ptrue() svptrue_b64()
#define cntp(g, pg) svcntp_b64(g, pg)
#define load_vec(pg, src) svld1_f64(pg, src)
#define load_za_row(tile, row, pg_row, src) svld1_hor_za64(tile, row, pg_row, src)
#define store_za_row(tile, row, pg_row, dst) svst1_hor_za64(tile, row, pg_row, dst)
#define fmopa(tile, pg_col, pg_row, src_col, src_row) svmopa_za64_f64_m(tile, pg_col, pg_row, src_col, src_row)
#else
#define rtype svfloat32_t
#define vlength() ((CeedSize)svcntw())
#define whilelt(i, n) svwhilelt_b32((int64_t)(i), (int64_t)(n))
#define ptrue() svptrue_b32()
#define cntp(g, pg) svcntp_b32(g, pg)
#define load_vec(pg, src) svld1_f32(pg, src)
#define load_za_row(tile, row, pg_row, src) svld1_hor_za32(tile, row, pg_row, src)
#define store_za_row(tile, row, pg_row, dst) svst1_hor_za32(tile, row, pg_row, dst)
#define fmopa(tile, pg_col, pg_row, src_col, src_row) svmopa_za32_f32_m(tile, pg_col, pg_row, src_col, src_row)
#endif

//------------------------------------------------------------------------------
// Tensor Load Helper
//------------------------------------------------------------------------------
static inline __attribute__((always_inline)) rtype LoadStrided_Sme(const CeedScalar *base, CeedInt stride, svbool_t pg, CeedInt num_active,
                                                                   CeedScalar *scratch) __arm_streaming __arm_preserves("za") {
  if (stride == 1) return load_vec(pg, base);

  for (CeedInt i = 0; i < num_active; i++) scratch[i] = base[(CeedSize)i * stride];

  return load_vec(pg, scratch);
}

//------------------------------------------------------------------------------
// Tensor Contract Slice
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Sme_Slice(CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t, const CeedInt add,
                                               const CeedScalar *restrict u, CeedScalar *restrict v) __arm_streaming __arm_inout("za") {
  const CeedSize vl      = vlength();
  const svbool_t pg_full = ptrue();

  svbool_t pg_col;
  for (CeedSize j = 0; svptest_first(pg_full, pg_col = whilelt(j, J)); j += vl) {
    const CeedInt n_col = cntp(pg_full, pg_col);
    CeedSize      c     = 0;

    // 4 tiles
    for (; c + 3 * vl < C; c += 4 * vl) {
      svbool_t pg_3 = whilelt(c + 3 * vl, C);

      if (add) {
        for (CeedInt i = 0; i < n_col; i++) {
          load_za_row(0, i, pg_full, v + ((CeedSize)j + i) * C + c);
          load_za_row(1, i, pg_full, v + ((CeedSize)j + i) * C + c + vl);
          load_za_row(2, i, pg_full, v + ((CeedSize)j + i) * C + c + vl * 2);
          load_za_row(3, i, pg_3, v + ((CeedSize)j + i) * C + c + vl * 3);
        }
      } else {
        svzero_za();
      }

      for (CeedInt b = 0; b < B; b++) {
        rtype tt = load_vec(pg_col, t + (CeedSize)b * J + j);

        rtype uu0 = load_vec(pg_full, u + (CeedSize)b * C + c);
        rtype uu1 = load_vec(pg_full, u + (CeedSize)b * C + c + vl);
        rtype uu2 = load_vec(pg_full, u + (CeedSize)b * C + c + vl * 2);
        rtype uu3 = load_vec(pg_3, u + (CeedSize)b * C + c + vl * 3);

        fmopa(0, pg_col, pg_full, tt, uu0);
        fmopa(1, pg_col, pg_full, tt, uu1);
        fmopa(2, pg_col, pg_full, tt, uu2);
        fmopa(3, pg_col, pg_3, tt, uu3);
      }

      for (CeedInt i = 0; i < n_col; i++) {
        store_za_row(0, i, pg_full, v + ((CeedSize)j + i) * C + c);
        store_za_row(1, i, pg_full, v + ((CeedSize)j + i) * C + c + vl);
        store_za_row(2, i, pg_full, v + ((CeedSize)j + i) * C + c + vl * 2);
        store_za_row(3, i, pg_3, v + ((CeedSize)j + i) * C + c + vl * 3);
      }
    }

    // 2 tiles
    for (; c + vl < C; c += 2 * vl) {
      svbool_t pg_1 = whilelt(c + vl, C);

      if (add) {
        for (CeedInt i = 0; i < n_col; i++) {
          load_za_row(0, i, pg_full, v + ((CeedSize)j + i) * C + c);
          load_za_row(1, i, pg_1, v + ((CeedSize)j + i) * C + c + vl);
        }
      } else {
        svzero_za();
      }

      for (CeedInt b = 0; b < B; b++) {
        rtype tt = load_vec(pg_col, t + (CeedSize)b * J + j);

        rtype uu0 = load_vec(pg_full, u + (CeedSize)b * C + c);
        rtype uu1 = load_vec(pg_1, u + (CeedSize)b * C + c + vl);

        fmopa(0, pg_col, pg_full, tt, uu0);
        fmopa(1, pg_col, pg_1, tt, uu1);
      }

      for (CeedInt i = 0; i < n_col; i++) {
        store_za_row(0, i, pg_full, v + ((CeedSize)j + i) * C + c);
        store_za_row(1, i, pg_1, v + ((CeedSize)j + i) * C + c + vl);
      }
    }

    // 1 tiles
    for (; c < C; c += vl) {
      svbool_t pg = whilelt(c, C);

      if (add) {
        for (CeedInt i = 0; i < n_col; i++) {
          load_za_row(0, i, pg, v + ((CeedSize)j + i) * C + c);
        }
      } else {
        svzero_za();
      }

      for (CeedInt b = 0; b < B; b++) {
        rtype tt = load_vec(pg_col, t + (CeedSize)b * J + j);

        rtype uu0 = load_vec(pg, u + (CeedSize)b * C + c);

        fmopa(0, pg_col, pg, tt, uu0);
      }

      for (CeedInt i = 0; i < n_col; i++) {
        store_za_row(0, i, pg, v + ((CeedSize)j + i) * C + c);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract C=1
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Sme_Single(CeedInt A, CeedInt B, CeedInt J, const CeedScalar *restrict t, const CeedInt add,
                                                const CeedScalar *restrict u, CeedScalar *restrict v,
                                                CeedScalar *scratch) __arm_streaming __arm_inout("za") {
  const CeedSize vl      = vlength();
  const svbool_t pg_full = ptrue();

  svbool_t pg_col;
  for (CeedSize a = 0; svptest_first(pg_full, pg_col = whilelt(a, A)); a += vl) {
    const CeedInt n_col = cntp(pg_full, pg_col);
    CeedSize      j     = 0;

    // 4 tiles
    for (; j + 3 * vl < J; j += 4 * vl) {
      svbool_t pg_3 = whilelt(j + 3 * vl, J);

      if (add) {
        for (CeedInt i = 0; i < n_col; i++) {
          load_za_row(0, i, pg_full, v + ((CeedSize)a + i) * J + j);
          load_za_row(1, i, pg_full, v + ((CeedSize)a + i) * J + j + vl);
          load_za_row(2, i, pg_full, v + ((CeedSize)a + i) * J + j + vl * 2);
          load_za_row(3, i, pg_3, v + ((CeedSize)a + i) * J + j + vl * 3);
        }
      } else {
        svzero_za();
      }

      for (CeedInt b = 0; b < B; b++) {
        rtype uu = LoadStrided_Sme(u + a * B + (CeedSize)b, B, pg_col, n_col, scratch);

        rtype tT0 = load_vec(pg_full, t + (CeedSize)b * J + j);
        rtype tT1 = load_vec(pg_full, t + (CeedSize)b * J + j + vl);
        rtype tT2 = load_vec(pg_full, t + (CeedSize)b * J + j + vl * 2);
        rtype tT3 = load_vec(pg_3, t + (CeedSize)b * J + j + vl * 3);

        fmopa(0, pg_col, pg_full, uu, tT0);
        fmopa(1, pg_col, pg_full, uu, tT1);
        fmopa(2, pg_col, pg_full, uu, tT2);
        fmopa(3, pg_col, pg_3, uu, tT3);
      }

      for (CeedInt i = 0; i < n_col; i++) {
        store_za_row(0, i, pg_full, v + ((CeedSize)a + i) * J + j);
        store_za_row(1, i, pg_full, v + ((CeedSize)a + i) * J + j + vl);
        store_za_row(2, i, pg_full, v + ((CeedSize)a + i) * J + j + vl * 2);
        store_za_row(3, i, pg_3, v + ((CeedSize)a + i) * J + j + vl * 3);
      }
    }

    // 2 tiles
    for (; j + vl < J; j += 2 * vl) {
      svbool_t pg_2 = whilelt(j + vl, J);

      if (add) {
        for (CeedInt i = 0; i < n_col; i++) {
          load_za_row(0, i, pg_full, v + ((CeedSize)a + i) * J + j);
          load_za_row(1, i, pg_2, v + ((CeedSize)a + i) * J + j + vl);
        }
      } else {
        svzero_za();
      }

      for (CeedInt b = 0; b < B; b++) {
        rtype uu = LoadStrided_Sme(u + a * B + (CeedSize)b, B, pg_col, n_col, scratch);

        rtype tT0 = load_vec(pg_full, t + (CeedSize)b * J + j);
        rtype tT1 = load_vec(pg_2, t + (CeedSize)b * J + j + vl);

        fmopa(0, pg_col, pg_full, uu, tT0);
        fmopa(1, pg_col, pg_2, uu, tT1);
      }

      for (CeedInt i = 0; i < n_col; i++) {
        store_za_row(0, i, pg_full, v + ((CeedSize)a + i) * J + j);
        store_za_row(1, i, pg_2, v + ((CeedSize)a + i) * J + j + vl);
      }
    }

    // 1 tiles
    for (; j < J; j += vl) {
      svbool_t pg = whilelt(j, J);

      if (add) {
        for (CeedInt i = 0; i < n_col; i++) {
          load_za_row(0, i, pg, v + ((CeedSize)a + i) * J + j);
        }
      } else {
        svzero_za();
      }

      for (CeedInt b = 0; b < B; b++) {
        rtype uu = LoadStrided_Sme(u + a * B + (CeedSize)b, B, pg_col, n_col, scratch);

        rtype tT0 = load_vec(pg, t + (CeedSize)b * J + j);

        fmopa(0, pg_col, pg, uu, tT0);
      }

      for (CeedInt i = 0; i < n_col; i++) {
        store_za_row(0, i, pg, v + ((CeedSize)a + i) * J + j);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Batch
//------------------------------------------------------------------------------
__arm_new("za") static inline int CeedTensorContract_Sme_Batch(CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                                               CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u,
                                                               CeedScalar *restrict v) __arm_streaming {
  if (t_mode == CEED_NOTRANSPOSE) {
    CeedScalar tT[(CeedSize)B * J];

    for (CeedInt b = 0; b < B; b++)
      for (CeedInt j = 0; j < J; j++) tT[(CeedSize)b * J + j] = t[(CeedSize)j * B + b];

    if (C == 1) {
      const CeedSize vl = vlength();
      CeedScalar     scratch[vl];

      CeedCallBackend(CeedTensorContract_Sme_Single(A, B, J, tT, add, u, v, scratch));
    } else {
      for (CeedInt a = 0; a < A; a++) {
        CeedCallBackend(CeedTensorContract_Sme_Slice(B, C, J, tT, add, &u[(CeedSize)a * B * C], &v[(CeedSize)a * J * C]));
      }
    }
  } else {
    if (C == 1) {
      const CeedSize vl = vlength();
      CeedScalar     scratch[vl];

      CeedCallBackend(CeedTensorContract_Sme_Single(A, B, J, t, add, u, v, scratch));
    } else {
      for (CeedInt a = 0; a < A; a++) {
        CeedCallBackend(CeedTensorContract_Sme_Slice(B, C, J, t, add, &u[(CeedSize)a * B * C], &v[(CeedSize)a * J * C]));
      }
    }
  }

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Sme(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  CeedCallBackend(CeedTensorContract_Sme_Batch(A, B, C, J, t, t_mode, add, u, v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Sme(CeedTensorContract contract) {
  CeedCallBackend(CeedSetBackendFunction(CeedTensorContractReturnCeed(contract), "TensorContract", contract, "Apply", CeedTensorContractApply_Sme));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
