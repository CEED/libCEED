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
static inline __attribute__((always_inline)) rtype CeedTensorContract_Sme_LoadT(const CeedScalar *base, CeedInt stride, svbool_t pg,
                                                                                CeedInt n) __arm_streaming __arm_preserves("za") {
  if (stride == 1) return load_vec(pg, base);

  CeedScalar tmp[vlength()];
  for (CeedInt i = 0; i < n; i++) tmp[i] = base[(CeedSize)i * stride];

  return load_vec(pg, tmp);
}

//------------------------------------------------------------------------------
// Tensor Contract Slice
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Sme_Slice(CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t, CeedTransposeMode t_mode,
                                               const CeedInt add, const CeedScalar *restrict u,
                                               CeedScalar *restrict v) __arm_streaming __arm_inout("za") {
  CeedInt s0 = B, s1 = 1;

  if (t_mode == CEED_TRANSPOSE) {
    s0 = 1;
    s1 = J;
  }

  svbool_t pg_col;
  for (CeedSize j = 0; svptest_first(ptrue(), pg_col = whilelt(j, J)); j += vlength()) {
    const CeedSize vl = vlength();
    CeedSize       c  = 0;

    // 4 tiles
    for (; c + 3 * vl < C; c += 4 * vl) {
      const CeedInt n    = cntp(ptrue(), pg_col);
      svbool_t      pg   = ptrue();
      svbool_t      pg_3 = whilelt(c + 3 * vl, C);

      svzero_za();

      if (add)
        for (CeedInt i = 0; i < n; i++) {
          load_za_row(0, i, pg, v + ((CeedSize)j + i) * C + c);
          load_za_row(1, i, pg, v + ((CeedSize)j + i) * C + c + vl);
          load_za_row(2, i, pg, v + ((CeedSize)j + i) * C + c + vl * 2);
          load_za_row(3, i, pg_3, v + ((CeedSize)j + i) * C + c + vl * 3);
        }

      for (CeedInt b = 0; b < B; b++) {
        rtype tt = CeedTensorContract_Sme_LoadT(t + j * s0 + (CeedSize)b * s1, s0, pg_col, n);

        rtype uu0 = load_vec(pg, u + (CeedSize)b * C + c);
        rtype uu1 = load_vec(pg, u + (CeedSize)b * C + c + vl);
        rtype uu2 = load_vec(pg, u + (CeedSize)b * C + c + vl * 2);
        rtype uu3 = load_vec(pg_3, u + (CeedSize)b * C + c + vl * 3);

        fmopa(0, pg_col, pg, tt, uu0);
        fmopa(1, pg_col, pg, tt, uu1);
        fmopa(2, pg_col, pg, tt, uu2);
        fmopa(3, pg_col, pg_3, tt, uu3);
      }

      for (CeedInt i = 0; i < n; i++) {
        store_za_row(0, i, pg, v + ((CeedSize)j + i) * C + c);
        store_za_row(1, i, pg, v + ((CeedSize)j + i) * C + c + vl);
        store_za_row(2, i, pg, v + ((CeedSize)j + i) * C + c + vl * 2);
        store_za_row(3, i, pg_3, v + ((CeedSize)j + i) * C + c + vl * 3);
      }
    }

    // 2 tiles
    for (; c + vl < C; c += 2 * vl) {
      const CeedInt n    = cntp(ptrue(), pg_col);
      svbool_t      pg   = ptrue();
      svbool_t      pg_1 = whilelt(c + vl, C);

      svzero_za();

      if (add)
        for (CeedInt i = 0; i < n; i++) {
          load_za_row(0, i, pg, v + ((CeedSize)j + i) * C + c);
          load_za_row(1, i, pg_1, v + ((CeedSize)j + i) * C + c + vl);
        }

      for (CeedInt b = 0; b < B; b++) {
        rtype tt = CeedTensorContract_Sme_LoadT(t + j * s0 + (CeedSize)b * s1, s0, pg_col, n);

        rtype uu0 = load_vec(pg, u + (CeedSize)b * C + c);
        rtype uu1 = load_vec(pg_1, u + (CeedSize)b * C + c + vl);

        fmopa(0, pg_col, pg, tt, uu0);
        fmopa(1, pg_col, pg_1, tt, uu1);
      }

      for (CeedInt i = 0; i < n; i++) {
        store_za_row(0, i, pg, v + ((CeedSize)j + i) * C + c);
        store_za_row(1, i, pg_1, v + ((CeedSize)j + i) * C + c + vl);
      }
    }

    // 1 tiles
    for (; c < C; c += vl) {
      const CeedInt n  = cntp(ptrue(), pg_col);
      svbool_t      pg = whilelt(c, C);

      svzero_za();

      if (add)
        for (CeedInt i = 0; i < n; i++) {
          load_za_row(0, i, pg, v + ((CeedSize)j + i) * C + c);
        }

      for (CeedInt b = 0; b < B; b++) {
        rtype tt = CeedTensorContract_Sme_LoadT(t + j * s0 + (CeedSize)b * s1, s0, pg_col, n);

        rtype uu0 = load_vec(pg, u + (CeedSize)b * C + c);

        fmopa(0, pg_col, pg, tt, uu0);
      }

      for (CeedInt i = 0; i < n; i++) {
        store_za_row(0, i, pg, v + ((CeedSize)j + i) * C + c);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract C=1
//------------------------------------------------------------------------------
static inline int CeedTensorContract_Sme_Single(CeedInt A, CeedInt B, CeedInt J, const CeedScalar *restrict t, CeedTransposeMode t_mode,
                                                const CeedInt add, const CeedScalar *restrict u,
                                                CeedScalar *restrict v) __arm_streaming __arm_inout("za") {
  svbool_t pg_col;
  for (CeedSize a = 0; svptest_first(ptrue(), pg_col = whilelt(a, A)); a += vlength()) {
    const CeedSize vl = vlength();
    CeedSize       j  = 0;

    // 4 tiles
    for (; j + 3 * vl < J; j += 4 * vl) {
      const CeedInt n    = cntp(ptrue(), pg_col);
      svbool_t      pg   = ptrue();
      svbool_t      pg_3 = whilelt(j + 3 * vl, J);
      const CeedInt nn   = cntp(ptrue(), pg_3);

      svzero_za();

      if (add)
        for (CeedInt i = 0; i < n; i++) {
          load_za_row(0, i, pg, v + ((CeedSize)a + i) * J + j);
          load_za_row(1, i, pg, v + ((CeedSize)a + i) * J + j + vl);
          load_za_row(2, i, pg, v + ((CeedSize)a + i) * J + j + vl * 2);
          load_za_row(3, i, pg_3, v + ((CeedSize)a + i) * J + j + vl * 3);
        }

      for (CeedInt b = 0; b < B; b++) {
        rtype uu = CeedTensorContract_Sme_LoadT(u + a * B + (CeedSize)b, B, pg_col, n);

        rtype tT0;
        rtype tT1;
        rtype tT2;
        rtype tT3;

        if (t_mode == CEED_TRANSPOSE) {
          tT0 = load_vec(pg, t + (CeedSize)b * J + j);
          tT1 = load_vec(pg, t + (CeedSize)b * J + j + vl);
          tT2 = load_vec(pg, t + (CeedSize)b * J + j + vl * 2);
          tT3 = load_vec(pg_3, t + (CeedSize)b * J + j + vl * 3);
        } else {
          tT0 = CeedTensorContract_Sme_LoadT(t + j * B + (CeedSize)b, B, pg, vl);
          tT1 = CeedTensorContract_Sme_LoadT(t + (j + vl) * B + (CeedSize)b, B, pg, vl);
          tT2 = CeedTensorContract_Sme_LoadT(t + (j + vl * 2) * B + (CeedSize)b, B, pg, vl);
          tT3 = CeedTensorContract_Sme_LoadT(t + (j + vl * 3) * B + (CeedSize)b, B, pg_3, nn);
        }

        fmopa(0, pg_col, pg, uu, tT0);
        fmopa(1, pg_col, pg, uu, tT1);
        fmopa(2, pg_col, pg, uu, tT2);
        fmopa(3, pg_col, pg_3, uu, tT3);
      }

      for (CeedInt i = 0; i < n; i++) {
        store_za_row(0, i, pg, v + ((CeedSize)a + i) * J + j);
        store_za_row(1, i, pg, v + ((CeedSize)a + i) * J + j + vl);
        store_za_row(2, i, pg, v + ((CeedSize)a + i) * J + j + vl * 2);
        store_za_row(3, i, pg_3, v + ((CeedSize)a + i) * J + j + vl * 3);
      }
    }

    // 2 tiles
    for (; j + vl < J; j += 2 * vl) {
      const CeedInt n    = cntp(ptrue(), pg_col);
      svbool_t      pg   = ptrue();
      svbool_t      pg_2 = whilelt(j + vl, J);
      const CeedInt nn   = cntp(ptrue(), pg_2);

      svzero_za();

      if (add)
        for (CeedInt i = 0; i < n; i++) {
          load_za_row(0, i, pg, v + ((CeedSize)a + i) * J + j);
          load_za_row(1, i, pg_2, v + ((CeedSize)a + i) * J + j + vl);
        }

      for (CeedInt b = 0; b < B; b++) {
        rtype uu = CeedTensorContract_Sme_LoadT(u + a * B + (CeedSize)b, B, pg_col, n);

        rtype tT0;
        rtype tT1;

        if (t_mode == CEED_TRANSPOSE) {
          tT0 = load_vec(pg, t + (CeedSize)b * J + j);
          tT1 = load_vec(pg_2, t + (CeedSize)b * J + j + vl);
        } else {
          tT0 = CeedTensorContract_Sme_LoadT(t + j * B + (CeedSize)b, B, pg, vl);
          tT1 = CeedTensorContract_Sme_LoadT(t + (j + vl) * B + (CeedSize)b, B, pg_2, nn);
        }

        fmopa(0, pg_col, pg, uu, tT0);
        fmopa(1, pg_col, pg_2, uu, tT1);
      }

      for (CeedInt i = 0; i < n; i++) {
        store_za_row(0, i, pg, v + ((CeedSize)a + i) * J + j);
        store_za_row(1, i, pg_2, v + ((CeedSize)a + i) * J + j + vl);
      }
    }

    // 1 tiles
    for (; j < J; j += vl) {
      const CeedInt n  = cntp(ptrue(), pg_col);
      svbool_t      pg = whilelt(j, J);
      const CeedInt nn = cntp(ptrue(), pg);

      svzero_za();

      if (add)
        for (CeedInt i = 0; i < n; i++) {
          load_za_row(0, i, pg, v + ((CeedSize)a + i) * J + j);
        }

      for (CeedInt b = 0; b < B; b++) {
        rtype uu = CeedTensorContract_Sme_LoadT(u + a * B + (CeedSize)b, B, pg_col, n);

        rtype tT0;

        if (t_mode == CEED_TRANSPOSE) {
          tT0 = load_vec(pg, t + (CeedSize)b * J + j);
        } else {
          tT0 = CeedTensorContract_Sme_LoadT(t + j * B + (CeedSize)b, B, pg, nn);
        }

        fmopa(0, pg_col, pg, uu, tT0);
      }

      for (CeedInt i = 0; i < n; i++) {
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
  if (C == 1) {
    CeedCallBackend(CeedTensorContract_Sme_Single(A, B, J, t, t_mode, add, u, v));
  } else {
    for (CeedInt a = 0; a < A; a++) {
      CeedCallBackend(CeedTensorContract_Sme_Slice(B, C, J, t, t_mode, add, &u[(CeedSize)a * B * C], &v[(CeedSize)a * J * C]));
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
