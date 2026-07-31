// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <arm_sve.h>
#include <stdint.h>

#include "ceed-sve.h"

#ifdef CEED_SCALAR_IS_FP64
#define rtype svfloat64_t
#define vlength() ((CeedInt)svcntd())
#define whilelt(i, n) svwhilelt_b64((uint64_t)(i), (uint64_t)(n))
#define ptrue() svptrue_b64()
#define setzero() svdup_f64(0.0)
#define set1(a) svdup_f64(a)
#define load(pg, a) svld1_f64(pg, a)
#define store(pg, a, v) svst1_f64(pg, a, v)
#define fmadd(pg, c, a, b) (c) = svmla_f64_m(pg, c, a, b)
#else
#define rtype svfloat32_t
#define vlength() ((CeedInt)svcntw())
#define whilelt(i, n) svwhilelt_b32((uint64_t)(i), (uint64_t)(n))
#define ptrue() svptrue_b32()
#define setzero() svdup_f32(0.0f)
#define set1(a) svdup_f32(a)
#define load(pg, a) svld1_f32(pg, a)
#define store(pg, a, v) svst1_f32(pg, a, v)
#define fmadd(pg, c, a, b) (c) = svmla_f32_m(pg, c, a, b)
#endif

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Sve(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  const CeedInt vector_length = vlength();
  CeedInt       t_stride_0 = B, t_stride_1 = 1;

  if (t_mode == CEED_TRANSPOSE) {
    t_stride_0 = 1;
    t_stride_1 = J;
  }

  for (CeedInt a = 0; a < A; a++) {
    CeedSize          c   = 0;
    CeedScalar *const v_a = &v[(CeedSize)a * J * C];

    // Blocks of four vectors
    while (true) {
      const CeedSize c_1 = c + vector_length, c_2 = c_1 + vector_length, c_3 = c_2 + vector_length;

      if (c_3 >= C) break;
      const svbool_t pg = ptrue(), pg_3 = whilelt(c_3, C);
      CeedInt        j = 0;

      // Blocks of four rows
      for (; j + 3 < J; j += 4) {
        rtype v_00 = setzero(), v_01 = v_00, v_02 = v_00, v_03 = v_00;
        rtype v_10 = v_00, v_11 = v_00, v_12 = v_00, v_13 = v_00;
        rtype v_20 = v_00, v_21 = v_00, v_22 = v_00, v_23 = v_00;
        rtype v_30 = v_00, v_31 = v_00, v_32 = v_00, v_33 = v_00;

        if (add) {
          v_00 = load(pg, &v_a[(j + 0) * (CeedSize)C + c]);
          v_01 = load(pg, &v_a[(j + 0) * (CeedSize)C + c_1]);
          v_02 = load(pg, &v_a[(j + 0) * (CeedSize)C + c_2]);
          v_03 = load(pg_3, &v_a[(j + 0) * (CeedSize)C + c_3]);
          v_10 = load(pg, &v_a[(j + 1) * (CeedSize)C + c]);
          v_11 = load(pg, &v_a[(j + 1) * (CeedSize)C + c_1]);
          v_12 = load(pg, &v_a[(j + 1) * (CeedSize)C + c_2]);
          v_13 = load(pg_3, &v_a[(j + 1) * (CeedSize)C + c_3]);
          v_20 = load(pg, &v_a[(j + 2) * (CeedSize)C + c]);
          v_21 = load(pg, &v_a[(j + 2) * (CeedSize)C + c_1]);
          v_22 = load(pg, &v_a[(j + 2) * (CeedSize)C + c_2]);
          v_23 = load(pg_3, &v_a[(j + 2) * (CeedSize)C + c_3]);
          v_30 = load(pg, &v_a[(j + 3) * (CeedSize)C + c]);
          v_31 = load(pg, &v_a[(j + 3) * (CeedSize)C + c_1]);
          v_32 = load(pg, &v_a[(j + 3) * (CeedSize)C + c_2]);
          v_33 = load(pg_3, &v_a[(j + 3) * (CeedSize)C + c_3]);
        }
#ifdef CEED_SCALAR_IS_FP64
#pragma GCC unroll 2
#endif
        for (CeedInt b = 0; b < B; b++) {
          const CeedSize u_offset = ((CeedSize)a * B + b) * C;
          const rtype    u_0 = load(pg, &u[u_offset + c]), u_1 = load(pg, &u[u_offset + c_1]);
          const rtype    u_2 = load(pg, &u[u_offset + c_2]), u_3 = load(pg_3, &u[u_offset + c_3]);
          const CeedSize t_offset = (CeedSize)b * t_stride_1;
          const rtype    t_0      = set1(t[(CeedSize)(j + 0) * t_stride_0 + t_offset]);
          const rtype    t_1      = set1(t[(CeedSize)(j + 1) * t_stride_0 + t_offset]);
          const rtype    t_2      = set1(t[(CeedSize)(j + 2) * t_stride_0 + t_offset]);
          const rtype    t_3      = set1(t[(CeedSize)(j + 3) * t_stride_0 + t_offset]);

          fmadd(pg, v_00, t_0, u_0);
          fmadd(pg, v_10, t_1, u_0);
          fmadd(pg, v_20, t_2, u_0);
          fmadd(pg, v_30, t_3, u_0);
          fmadd(pg, v_01, t_0, u_1);
          fmadd(pg, v_11, t_1, u_1);
          fmadd(pg, v_21, t_2, u_1);
          fmadd(pg, v_31, t_3, u_1);
          fmadd(pg, v_02, t_0, u_2);
          fmadd(pg, v_12, t_1, u_2);
          fmadd(pg, v_22, t_2, u_2);
          fmadd(pg, v_32, t_3, u_2);
          fmadd(pg_3, v_03, t_0, u_3);
          fmadd(pg_3, v_13, t_1, u_3);
          fmadd(pg_3, v_23, t_2, u_3);
          fmadd(pg_3, v_33, t_3, u_3);
        }
        store(pg, &v_a[(j + 0) * (CeedSize)C + c], v_00);
        store(pg, &v_a[(j + 0) * (CeedSize)C + c_1], v_01);
        store(pg, &v_a[(j + 0) * (CeedSize)C + c_2], v_02);
        store(pg_3, &v_a[(j + 0) * (CeedSize)C + c_3], v_03);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c], v_10);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c_1], v_11);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c_2], v_12);
        store(pg_3, &v_a[(j + 1) * (CeedSize)C + c_3], v_13);
        store(pg, &v_a[(j + 2) * (CeedSize)C + c], v_20);
        store(pg, &v_a[(j + 2) * (CeedSize)C + c_1], v_21);
        store(pg, &v_a[(j + 2) * (CeedSize)C + c_2], v_22);
        store(pg_3, &v_a[(j + 2) * (CeedSize)C + c_3], v_23);
        store(pg, &v_a[(j + 3) * (CeedSize)C + c], v_30);
        store(pg, &v_a[(j + 3) * (CeedSize)C + c_1], v_31);
        store(pg, &v_a[(j + 3) * (CeedSize)C + c_2], v_32);
        store(pg_3, &v_a[(j + 3) * (CeedSize)C + c_3], v_33);
      }
      // Remainder of rows
      for (; j + 1 < J; j += 2) {
        rtype v_00 = setzero(), v_01 = v_00, v_02 = v_00, v_03 = v_00;
        rtype v_10 = v_00, v_11 = v_00, v_12 = v_00, v_13 = v_00;

        if (add) {
          v_00 = load(pg, &v_a[(j + 0) * (CeedSize)C + c]);
          v_01 = load(pg, &v_a[(j + 0) * (CeedSize)C + c_1]);
          v_02 = load(pg, &v_a[(j + 0) * (CeedSize)C + c_2]);
          v_03 = load(pg_3, &v_a[(j + 0) * (CeedSize)C + c_3]);
          v_10 = load(pg, &v_a[(j + 1) * (CeedSize)C + c]);
          v_11 = load(pg, &v_a[(j + 1) * (CeedSize)C + c_1]);
          v_12 = load(pg, &v_a[(j + 1) * (CeedSize)C + c_2]);
          v_13 = load(pg_3, &v_a[(j + 1) * (CeedSize)C + c_3]);
        }
#ifdef CEED_SCALAR_IS_FP64
#pragma GCC unroll 3
#endif
        for (CeedInt b = 0; b < B; b++) {
          const CeedSize u_offset = ((CeedSize)a * B + b) * C;
          const rtype    u_0 = load(pg, &u[u_offset + c]), u_1 = load(pg, &u[u_offset + c_1]);
          const rtype    u_2 = load(pg, &u[u_offset + c_2]), u_3 = load(pg_3, &u[u_offset + c_3]);
          const CeedSize t_offset = (CeedSize)b * t_stride_1;
          const rtype    t_0      = set1(t[(CeedSize)(j + 0) * t_stride_0 + t_offset]);
          const rtype    t_1      = set1(t[(CeedSize)(j + 1) * t_stride_0 + t_offset]);

          fmadd(pg, v_00, t_0, u_0);
          fmadd(pg, v_10, t_1, u_0);
          fmadd(pg, v_01, t_0, u_1);
          fmadd(pg, v_11, t_1, u_1);
          fmadd(pg, v_02, t_0, u_2);
          fmadd(pg, v_12, t_1, u_2);
          fmadd(pg_3, v_03, t_0, u_3);
          fmadd(pg_3, v_13, t_1, u_3);
        }
        store(pg, &v_a[(j + 0) * (CeedSize)C + c], v_00);
        store(pg, &v_a[(j + 0) * (CeedSize)C + c_1], v_01);
        store(pg, &v_a[(j + 0) * (CeedSize)C + c_2], v_02);
        store(pg_3, &v_a[(j + 0) * (CeedSize)C + c_3], v_03);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c], v_10);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c_1], v_11);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c_2], v_12);
        store(pg_3, &v_a[(j + 1) * (CeedSize)C + c_3], v_13);
      }
      if (j < J) {
        rtype v_0 = setzero(), v_1 = v_0, v_2 = v_0, v_3 = v_0;

        if (add) {
          v_0 = load(pg, &v_a[j * (CeedSize)C + c]);
          v_1 = load(pg, &v_a[j * (CeedSize)C + c_1]);
          v_2 = load(pg, &v_a[j * (CeedSize)C + c_2]);
          v_3 = load(pg_3, &v_a[j * (CeedSize)C + c_3]);
        }
        for (CeedInt b = 0; b < B; b++) {
          const CeedSize u_offset = ((CeedSize)a * B + b) * C, t_offset = (CeedSize)b * t_stride_1;
          const rtype    t_0 = set1(t[(CeedSize)j * t_stride_0 + t_offset]);

          fmadd(pg, v_0, t_0, load(pg, &u[u_offset + c]));
          fmadd(pg, v_1, t_0, load(pg, &u[u_offset + c_1]));
          fmadd(pg, v_2, t_0, load(pg, &u[u_offset + c_2]));
          fmadd(pg_3, v_3, t_0, load(pg_3, &u[u_offset + c_3]));
        }
        store(pg, &v_a[j * (CeedSize)C + c], v_0);
        store(pg, &v_a[j * (CeedSize)C + c_1], v_1);
        store(pg, &v_a[j * (CeedSize)C + c_2], v_2);
        store(pg_3, &v_a[j * (CeedSize)C + c_3], v_3);
      }
      c = c_3 + vector_length;
    }

    // Blocks of two vectors
    while (true) {
      const CeedSize c_1 = c + vector_length;

      if (c_1 >= C) break;
      const svbool_t pg = ptrue(), pg_1 = whilelt(c_1, C);
      CeedInt        j = 0;

      for (; j + 3 < J; j += 4) {
        rtype v_00 = setzero(), v_01 = v_00, v_10 = v_00, v_11 = v_00;
        rtype v_20 = v_00, v_21 = v_00, v_30 = v_00, v_31 = v_00;

        if (add) {
          v_00 = load(pg, &v_a[(j + 0) * (CeedSize)C + c]);
          v_01 = load(pg_1, &v_a[(j + 0) * (CeedSize)C + c_1]);
          v_10 = load(pg, &v_a[(j + 1) * (CeedSize)C + c]);
          v_11 = load(pg_1, &v_a[(j + 1) * (CeedSize)C + c_1]);
          v_20 = load(pg, &v_a[(j + 2) * (CeedSize)C + c]);
          v_21 = load(pg_1, &v_a[(j + 2) * (CeedSize)C + c_1]);
          v_30 = load(pg, &v_a[(j + 3) * (CeedSize)C + c]);
          v_31 = load(pg_1, &v_a[(j + 3) * (CeedSize)C + c_1]);
        }
#pragma GCC unroll 2
        for (CeedInt b = 0; b < B; b++) {
          const CeedSize u_offset = ((CeedSize)a * B + b) * C;
          const rtype    u_0 = load(pg, &u[u_offset + c]), u_1 = load(pg_1, &u[u_offset + c_1]);
          const CeedSize t_offset = (CeedSize)b * t_stride_1;
          const rtype    t_0      = set1(t[(CeedSize)(j + 0) * t_stride_0 + t_offset]);
          const rtype    t_1      = set1(t[(CeedSize)(j + 1) * t_stride_0 + t_offset]);

          fmadd(pg, v_00, t_0, u_0);
          fmadd(pg, v_10, t_1, u_0);
          fmadd(pg_1, v_01, t_0, u_1);
          fmadd(pg_1, v_11, t_1, u_1);
          const rtype t_2 = set1(t[(CeedSize)(j + 2) * t_stride_0 + t_offset]);
          const rtype t_3 = set1(t[(CeedSize)(j + 3) * t_stride_0 + t_offset]);

          fmadd(pg, v_20, t_2, u_0);
          fmadd(pg, v_30, t_3, u_0);
          fmadd(pg_1, v_21, t_2, u_1);
          fmadd(pg_1, v_31, t_3, u_1);
        }
        store(pg, &v_a[(j + 0) * (CeedSize)C + c], v_00);
        store(pg_1, &v_a[(j + 0) * (CeedSize)C + c_1], v_01);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c], v_10);
        store(pg_1, &v_a[(j + 1) * (CeedSize)C + c_1], v_11);
        store(pg, &v_a[(j + 2) * (CeedSize)C + c], v_20);
        store(pg_1, &v_a[(j + 2) * (CeedSize)C + c_1], v_21);
        store(pg, &v_a[(j + 3) * (CeedSize)C + c], v_30);
        store(pg_1, &v_a[(j + 3) * (CeedSize)C + c_1], v_31);
      }
      for (; j + 1 < J; j += 2) {
        rtype v_00 = setzero(), v_01 = v_00, v_10 = v_00, v_11 = v_00;

        if (add) {
          v_00 = load(pg, &v_a[(j + 0) * (CeedSize)C + c]);
          v_01 = load(pg_1, &v_a[(j + 0) * (CeedSize)C + c_1]);
          v_10 = load(pg, &v_a[(j + 1) * (CeedSize)C + c]);
          v_11 = load(pg_1, &v_a[(j + 1) * (CeedSize)C + c_1]);
        }
#pragma GCC unroll 2
        for (CeedInt b = 0; b < B; b++) {
          const CeedSize u_offset = ((CeedSize)a * B + b) * C;
          const rtype    u_0 = load(pg, &u[u_offset + c]), u_1 = load(pg_1, &u[u_offset + c_1]);
          const CeedSize t_offset = (CeedSize)b * t_stride_1;
          const rtype    t_0      = set1(t[(CeedSize)(j + 0) * t_stride_0 + t_offset]);
          const rtype    t_1      = set1(t[(CeedSize)(j + 1) * t_stride_0 + t_offset]);

          fmadd(pg, v_00, t_0, u_0);
          fmadd(pg, v_10, t_1, u_0);
          fmadd(pg_1, v_01, t_0, u_1);
          fmadd(pg_1, v_11, t_1, u_1);
        }
        store(pg, &v_a[(j + 0) * (CeedSize)C + c], v_00);
        store(pg_1, &v_a[(j + 0) * (CeedSize)C + c_1], v_01);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c], v_10);
        store(pg_1, &v_a[(j + 1) * (CeedSize)C + c_1], v_11);
      }
      if (j < J) {
        rtype v_0 = setzero(), v_1 = v_0;

        if (add) {
          v_0 = load(pg, &v_a[j * (CeedSize)C + c]);
          v_1 = load(pg_1, &v_a[j * (CeedSize)C + c_1]);
        }
        for (CeedInt b = 0; b < B; b++) {
          const CeedSize u_offset = ((CeedSize)a * B + b) * C;
          const rtype    t_0      = set1(t[(CeedSize)j * t_stride_0 + (CeedSize)b * t_stride_1]);

          fmadd(pg, v_0, t_0, load(pg, &u[u_offset + c]));
          fmadd(pg_1, v_1, t_0, load(pg_1, &u[u_offset + c_1]));
        }
        store(pg, &v_a[j * (CeedSize)C + c], v_0);
        store(pg_1, &v_a[j * (CeedSize)C + c_1], v_1);
      }
      c = c_1 + vector_length;
    }

    // Predicated remainder
    for (; c < C; c += vector_length) {
      const svbool_t pg = whilelt(c, C);
      CeedInt        j  = 0;

      for (; j + 3 < J; j += 4) {
        rtype v_0 = setzero(), v_1 = v_0, v_2 = v_0, v_3 = v_0;

        if (add) {
          v_0 = load(pg, &v_a[(j + 0) * (CeedSize)C + c]);
          v_1 = load(pg, &v_a[(j + 1) * (CeedSize)C + c]);
          v_2 = load(pg, &v_a[(j + 2) * (CeedSize)C + c]);
          v_3 = load(pg, &v_a[(j + 3) * (CeedSize)C + c]);
        }
#pragma GCC unroll 4
        for (CeedInt b = 0; b < B; b++) {
          const rtype    u_0      = load(pg, &u[((CeedSize)a * B + b) * C + c]);
          const CeedSize t_offset = (CeedSize)b * t_stride_1;

          fmadd(pg, v_0, set1(t[(CeedSize)(j + 0) * t_stride_0 + t_offset]), u_0);
          fmadd(pg, v_1, set1(t[(CeedSize)(j + 1) * t_stride_0 + t_offset]), u_0);
          fmadd(pg, v_2, set1(t[(CeedSize)(j + 2) * t_stride_0 + t_offset]), u_0);
          fmadd(pg, v_3, set1(t[(CeedSize)(j + 3) * t_stride_0 + t_offset]), u_0);
        }
        store(pg, &v_a[(j + 0) * (CeedSize)C + c], v_0);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c], v_1);
        store(pg, &v_a[(j + 2) * (CeedSize)C + c], v_2);
        store(pg, &v_a[(j + 3) * (CeedSize)C + c], v_3);
      }
      for (; j + 1 < J; j += 2) {
        rtype v_0 = setzero(), v_1 = v_0;

        if (add) {
          v_0 = load(pg, &v_a[(j + 0) * (CeedSize)C + c]);
          v_1 = load(pg, &v_a[(j + 1) * (CeedSize)C + c]);
        }
#pragma GCC unroll 4
        for (CeedInt b = 0; b < B; b++) {
          const rtype    u_0      = load(pg, &u[((CeedSize)a * B + b) * C + c]);
          const CeedSize t_offset = (CeedSize)b * t_stride_1;

          fmadd(pg, v_0, set1(t[(CeedSize)(j + 0) * t_stride_0 + t_offset]), u_0);
          fmadd(pg, v_1, set1(t[(CeedSize)(j + 1) * t_stride_0 + t_offset]), u_0);
        }
        store(pg, &v_a[(j + 0) * (CeedSize)C + c], v_0);
        store(pg, &v_a[(j + 1) * (CeedSize)C + c], v_1);
      }
      if (j < J) {
        rtype v_0 = setzero();

        if (add) v_0 = load(pg, &v_a[j * (CeedSize)C + c]);
        for (CeedInt b = 0; b < B; b++) {
          const rtype u_0 = load(pg, &u[((CeedSize)a * B + b) * C + c]);
          const rtype t_0 = set1(t[(CeedSize)j * t_stride_0 + (CeedSize)b * t_stride_1]);

          fmadd(pg, v_0, t_0, u_0);
        }
        store(pg, &v_a[j * (CeedSize)C + c], v_0);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Sve(CeedTensorContract contract) {
  CeedCallBackend(CeedSetBackendFunction(CeedTensorContractReturnCeed(contract), "TensorContract", contract, "Apply", CeedTensorContractApply_Sve));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
