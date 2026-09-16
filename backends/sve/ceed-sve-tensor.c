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
#define vlength() ((CeedSize)svcntd())
#define whilelt(i, n) svwhilelt_b64((int64_t)(i), (int64_t)(n))
#define ptrue() svptrue_b64()
#define setzero() svdup_f64(0.0)
#define set1(a) svdup_f64(a)
#define load(pg, a, n) svld1_vnum_f64(pg, a, n)
#define store(pg, a, n, v) svst1_vnum_f64(pg, a, n, v)
#define gather(pg, a, i) svld1_gather_s64index_f64(pg, a, i)
#define gather_index(step) svindex_s64(0, (int64_t)(step))
#define fmadd(pg, c, a, b) (c) = svmla_f64_m(pg, c, a, b)
#else
#define rtype svfloat32_t
#define vlength() ((CeedSize)svcntw())
#define whilelt(i, n) svwhilelt_b32((int64_t)(i), (int64_t)(n))
#define ptrue() svptrue_b32()
#define setzero() svdup_f32(0.0f)
#define set1(a) svdup_f32(a)
#define load(pg, a, n) svld1_vnum_f32(pg, a, n)
#define store(pg, a, n, v) svst1_vnum_f32(pg, a, n, v)
#define gather(pg, a, i) svld1_gather_s32index_f32(pg, a, i)
#define gather_index(step) svindex_s32(0, (int32_t)(step))
#define fmadd(pg, c, a, b) (c) = svmla_f32_m(pg, c, a, b)
#endif
// Unroll of the reduction loop in the multi-row tiles, by column width; measured per precision
#ifdef CEED_SCALAR_IS_FP64
#define UNROLL_4VEC _Pragma("GCC unroll 2")
#else
#define UNROLL_4VEC
#endif
#define UNROLL_2VEC _Pragma("GCC unroll 2")
#define UNROLL_1VEC _Pragma("GCC unroll 4")
// Hide a pointer from strength reduction; otherwise compilers create and spill
// one induction variable per address
#define opaque(p) __asm__ volatile("" : "+r"(p))

//------------------------------------------------------------------------------
// Tensor Contract Slice
//------------------------------------------------------------------------------
// v[j,c] (+)= sum_b t[j,b] u[b,c] for one a; vectorized over c, tiled over j.
static inline int CeedTensorContract_Sve_Slice(CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t, CeedTransposeMode t_mode,
                                               const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  const CeedSize vector_length = vlength();
  const CeedSize t_stride_0 = t_mode == CEED_TRANSPOSE ? 1 : B, t_stride_1 = t_mode == CEED_TRANSPOSE ? J : 1;
  const CeedSize row_block_4 = 4 * (CeedSize)C, t_block_4 = 4 * t_stride_0;
  const CeedSize row_block_2 = 2 * (CeedSize)C, t_block_2 = 2 * t_stride_0;
  CeedSize       c = 0;

  opaque(u);
  opaque(v);
  // Blocks of 4 vectors
  for (; c + 3 * vector_length < C; c += 4 * vector_length) {
    const svbool_t    pg = ptrue(), pg_3 = whilelt(c + 3 * vector_length, C);
    const CeedScalar *u_c = u + c;
    CeedScalar       *v_0 = v + c;
    const CeedScalar *t_0 = t;
    CeedInt           j   = 0;

    // Blocks of 4 rows
    for (; j + 3 < J; j += 4, v_0 += row_block_4, t_0 += t_block_4) {
      CeedScalar *const v_1 = v_0 + C, *const v_2 = v_1 + C, *const v_3 = v_2 + C;
      const CeedScalar *const t_1 = t_0 + t_stride_0, *const t_2 = t_1 + t_stride_0, *const t_3 = t_2 + t_stride_0;
      const CeedScalar *u_b = u_c;
      // Output tile held in registers
      rtype v_00 = add ? load(pg, v_0, 0) : setzero(), v_01 = add ? load(pg, v_0, 1) : setzero();
      rtype v_02 = add ? load(pg, v_0, 2) : setzero(), v_03 = add ? load(pg_3, v_0, 3) : setzero();
      rtype v_10 = add ? load(pg, v_1, 0) : setzero(), v_11 = add ? load(pg, v_1, 1) : setzero();
      rtype v_12 = add ? load(pg, v_1, 2) : setzero(), v_13 = add ? load(pg_3, v_1, 3) : setzero();
      rtype v_20 = add ? load(pg, v_2, 0) : setzero(), v_21 = add ? load(pg, v_2, 1) : setzero();
      rtype v_22 = add ? load(pg, v_2, 2) : setzero(), v_23 = add ? load(pg_3, v_2, 3) : setzero();
      rtype v_30 = add ? load(pg, v_3, 0) : setzero(), v_31 = add ? load(pg, v_3, 1) : setzero();
      rtype v_32 = add ? load(pg, v_3, 2) : setzero(), v_33 = add ? load(pg_3, v_3, 3) : setzero();

      UNROLL_4VEC
      for (CeedInt b = 0; b < B; b++, u_b += C) {
        const CeedSize t_b = (CeedSize)b * t_stride_1;
        const rtype    u_0 = load(pg, u_b, 0), u_1 = load(pg, u_b, 1), u_2 = load(pg, u_b, 2), u_3 = load(pg_3, u_b, 3);
        const rtype    s_0 = set1(t_0[t_b]), s_1 = set1(t_1[t_b]), s_2 = set1(t_2[t_b]), s_3 = set1(t_3[t_b]);

        fmadd(pg, v_00, s_0, u_0);
        fmadd(pg, v_10, s_1, u_0);
        fmadd(pg, v_20, s_2, u_0);
        fmadd(pg, v_30, s_3, u_0);
        fmadd(pg, v_01, s_0, u_1);
        fmadd(pg, v_11, s_1, u_1);
        fmadd(pg, v_21, s_2, u_1);
        fmadd(pg, v_31, s_3, u_1);
        fmadd(pg, v_02, s_0, u_2);
        fmadd(pg, v_12, s_1, u_2);
        fmadd(pg, v_22, s_2, u_2);
        fmadd(pg, v_32, s_3, u_2);
        fmadd(pg_3, v_03, s_0, u_3);
        fmadd(pg_3, v_13, s_1, u_3);
        fmadd(pg_3, v_23, s_2, u_3);
        fmadd(pg_3, v_33, s_3, u_3);
      }
      store(pg, v_0, 0, v_00), store(pg, v_0, 1, v_01), store(pg, v_0, 2, v_02), store(pg_3, v_0, 3, v_03);
      store(pg, v_1, 0, v_10), store(pg, v_1, 1, v_11), store(pg, v_1, 2, v_12), store(pg_3, v_1, 3, v_13);
      store(pg, v_2, 0, v_20), store(pg, v_2, 1, v_21), store(pg, v_2, 2, v_22), store(pg_3, v_2, 3, v_23);
      store(pg, v_3, 0, v_30), store(pg, v_3, 1, v_31), store(pg, v_3, 2, v_32), store(pg_3, v_3, 3, v_33);
    }
    // Blocks of 2 rows
    for (; j + 1 < J; j += 2, v_0 += row_block_2, t_0 += t_block_2) {
      CeedScalar *const       v_1  = v_0 + C;
      const CeedScalar *const t_1  = t_0 + t_stride_0;
      const CeedScalar       *u_b  = u_c;
      rtype                   v_00 = add ? load(pg, v_0, 0) : setzero(), v_01 = add ? load(pg, v_0, 1) : setzero();
      rtype                   v_02 = add ? load(pg, v_0, 2) : setzero(), v_03 = add ? load(pg_3, v_0, 3) : setzero();
      rtype                   v_10 = add ? load(pg, v_1, 0) : setzero(), v_11 = add ? load(pg, v_1, 1) : setzero();
      rtype                   v_12 = add ? load(pg, v_1, 2) : setzero(), v_13 = add ? load(pg_3, v_1, 3) : setzero();

      UNROLL_4VEC
      for (CeedInt b = 0; b < B; b++, u_b += C) {
        const CeedSize t_b = (CeedSize)b * t_stride_1;
        const rtype    u_0 = load(pg, u_b, 0), u_1 = load(pg, u_b, 1), u_2 = load(pg, u_b, 2), u_3 = load(pg_3, u_b, 3);
        const rtype    s_0 = set1(t_0[t_b]), s_1 = set1(t_1[t_b]);

        fmadd(pg, v_00, s_0, u_0);
        fmadd(pg, v_10, s_1, u_0);
        fmadd(pg, v_01, s_0, u_1);
        fmadd(pg, v_11, s_1, u_1);
        fmadd(pg, v_02, s_0, u_2);
        fmadd(pg, v_12, s_1, u_2);
        fmadd(pg_3, v_03, s_0, u_3);
        fmadd(pg_3, v_13, s_1, u_3);
      }
      store(pg, v_0, 0, v_00), store(pg, v_0, 1, v_01), store(pg, v_0, 2, v_02), store(pg_3, v_0, 3, v_03);
      store(pg, v_1, 0, v_10), store(pg, v_1, 1, v_11), store(pg, v_1, 2, v_12), store(pg_3, v_1, 3, v_13);
    }
    // Remainder of rows
    if (j < J) {
      const CeedScalar *u_b  = u_c;
      rtype             v_00 = add ? load(pg, v_0, 0) : setzero(), v_01 = add ? load(pg, v_0, 1) : setzero();
      rtype             v_02 = add ? load(pg, v_0, 2) : setzero(), v_03 = add ? load(pg_3, v_0, 3) : setzero();

      for (CeedInt b = 0; b < B; b++, u_b += C) {
        const rtype s_0 = set1(t_0[(CeedSize)b * t_stride_1]);

        fmadd(pg, v_00, s_0, load(pg, u_b, 0));
        fmadd(pg, v_01, s_0, load(pg, u_b, 1));
        fmadd(pg, v_02, s_0, load(pg, u_b, 2));
        fmadd(pg_3, v_03, s_0, load(pg_3, u_b, 3));
      }
      store(pg, v_0, 0, v_00), store(pg, v_0, 1, v_01), store(pg, v_0, 2, v_02), store(pg_3, v_0, 3, v_03);
    }
  }

  // Blocks of 2 vectors
  for (; c + vector_length < C; c += 2 * vector_length) {
    const svbool_t    pg = ptrue(), pg_1 = whilelt(c + vector_length, C);
    const CeedScalar *u_c = u + c;
    CeedScalar       *v_0 = v + c;
    const CeedScalar *t_0 = t;
    CeedInt           j   = 0;

    // Blocks of 4 rows
    for (; j + 3 < J; j += 4, v_0 += row_block_4, t_0 += t_block_4) {
      CeedScalar *const v_1 = v_0 + C, *const v_2 = v_1 + C, *const v_3 = v_2 + C;
      const CeedScalar *const t_1 = t_0 + t_stride_0, *const t_2 = t_1 + t_stride_0, *const t_3 = t_2 + t_stride_0;
      const CeedScalar *u_b  = u_c;
      rtype             v_00 = add ? load(pg, v_0, 0) : setzero(), v_01 = add ? load(pg_1, v_0, 1) : setzero();
      rtype             v_10 = add ? load(pg, v_1, 0) : setzero(), v_11 = add ? load(pg_1, v_1, 1) : setzero();
      rtype             v_20 = add ? load(pg, v_2, 0) : setzero(), v_21 = add ? load(pg_1, v_2, 1) : setzero();
      rtype             v_30 = add ? load(pg, v_3, 0) : setzero(), v_31 = add ? load(pg_1, v_3, 1) : setzero();

      UNROLL_2VEC
      for (CeedInt b = 0; b < B; b++, u_b += C) {
        const CeedSize t_b = (CeedSize)b * t_stride_1;
        const rtype    u_0 = load(pg, u_b, 0), u_1 = load(pg_1, u_b, 1);
        const rtype    s_0 = set1(t_0[t_b]), s_1 = set1(t_1[t_b]), s_2 = set1(t_2[t_b]), s_3 = set1(t_3[t_b]);

        fmadd(pg, v_00, s_0, u_0);
        fmadd(pg, v_10, s_1, u_0);
        fmadd(pg, v_20, s_2, u_0);
        fmadd(pg, v_30, s_3, u_0);
        fmadd(pg_1, v_01, s_0, u_1);
        fmadd(pg_1, v_11, s_1, u_1);
        fmadd(pg_1, v_21, s_2, u_1);
        fmadd(pg_1, v_31, s_3, u_1);
      }
      store(pg, v_0, 0, v_00), store(pg_1, v_0, 1, v_01);
      store(pg, v_1, 0, v_10), store(pg_1, v_1, 1, v_11);
      store(pg, v_2, 0, v_20), store(pg_1, v_2, 1, v_21);
      store(pg, v_3, 0, v_30), store(pg_1, v_3, 1, v_31);
    }
    // Blocks of 2 rows
    for (; j + 1 < J; j += 2, v_0 += row_block_2, t_0 += t_block_2) {
      CeedScalar *const       v_1  = v_0 + C;
      const CeedScalar *const t_1  = t_0 + t_stride_0;
      const CeedScalar       *u_b  = u_c;
      rtype                   v_00 = add ? load(pg, v_0, 0) : setzero(), v_01 = add ? load(pg_1, v_0, 1) : setzero();
      rtype                   v_10 = add ? load(pg, v_1, 0) : setzero(), v_11 = add ? load(pg_1, v_1, 1) : setzero();

      UNROLL_2VEC
      for (CeedInt b = 0; b < B; b++, u_b += C) {
        const CeedSize t_b = (CeedSize)b * t_stride_1;
        const rtype    u_0 = load(pg, u_b, 0), u_1 = load(pg_1, u_b, 1);
        const rtype    s_0 = set1(t_0[t_b]), s_1 = set1(t_1[t_b]);

        fmadd(pg, v_00, s_0, u_0);
        fmadd(pg, v_10, s_1, u_0);
        fmadd(pg_1, v_01, s_0, u_1);
        fmadd(pg_1, v_11, s_1, u_1);
      }
      store(pg, v_0, 0, v_00), store(pg_1, v_0, 1, v_01);
      store(pg, v_1, 0, v_10), store(pg_1, v_1, 1, v_11);
    }
    // Remainder of rows
    if (j < J) {
      const CeedScalar *u_b  = u_c;
      rtype             v_00 = add ? load(pg, v_0, 0) : setzero(), v_01 = add ? load(pg_1, v_0, 1) : setzero();

      for (CeedInt b = 0; b < B; b++, u_b += C) {
        const rtype s_0 = set1(t_0[(CeedSize)b * t_stride_1]);

        fmadd(pg, v_00, s_0, load(pg, u_b, 0));
        fmadd(pg_1, v_01, s_0, load(pg_1, u_b, 1));
      }
      store(pg, v_0, 0, v_00), store(pg_1, v_0, 1, v_01);
    }
  }

  // Remainder of columns, predicated
  for (; c < C; c += vector_length) {
    const svbool_t    pg  = whilelt(c, C);
    const CeedScalar *u_c = u + c;
    CeedScalar       *v_0 = v + c;
    const CeedScalar *t_0 = t;
    CeedInt           j   = 0;

    // Blocks of 4 rows
    for (; j + 3 < J; j += 4, v_0 += row_block_4, t_0 += t_block_4) {
      CeedScalar *const v_1 = v_0 + C, *const v_2 = v_1 + C, *const v_3 = v_2 + C;
      const CeedScalar *const t_1 = t_0 + t_stride_0, *const t_2 = t_1 + t_stride_0, *const t_3 = t_2 + t_stride_0;
      const CeedScalar *u_b  = u_c;
      rtype             v_00 = add ? load(pg, v_0, 0) : setzero(), v_10 = add ? load(pg, v_1, 0) : setzero();
      rtype             v_20 = add ? load(pg, v_2, 0) : setzero(), v_30 = add ? load(pg, v_3, 0) : setzero();

      UNROLL_1VEC
      for (CeedInt b = 0; b < B; b++, u_b += C) {
        const CeedSize t_b = (CeedSize)b * t_stride_1;
        const rtype    u_0 = load(pg, u_b, 0);

        fmadd(pg, v_00, set1(t_0[t_b]), u_0);
        fmadd(pg, v_10, set1(t_1[t_b]), u_0);
        fmadd(pg, v_20, set1(t_2[t_b]), u_0);
        fmadd(pg, v_30, set1(t_3[t_b]), u_0);
      }
      store(pg, v_0, 0, v_00), store(pg, v_1, 0, v_10), store(pg, v_2, 0, v_20), store(pg, v_3, 0, v_30);
    }
    // Blocks of 2 rows
    for (; j + 1 < J; j += 2, v_0 += row_block_2, t_0 += t_block_2) {
      CeedScalar *const       v_1  = v_0 + C;
      const CeedScalar *const t_1  = t_0 + t_stride_0;
      const CeedScalar       *u_b  = u_c;
      rtype                   v_00 = add ? load(pg, v_0, 0) : setzero(), v_10 = add ? load(pg, v_1, 0) : setzero();

      UNROLL_1VEC
      for (CeedInt b = 0; b < B; b++, u_b += C) {
        const CeedSize t_b = (CeedSize)b * t_stride_1;
        const rtype    u_0 = load(pg, u_b, 0);

        fmadd(pg, v_00, set1(t_0[t_b]), u_0);
        fmadd(pg, v_10, set1(t_1[t_b]), u_0);
      }
      store(pg, v_0, 0, v_00), store(pg, v_1, 0, v_10);
    }
    // Remainder of rows
    if (j < J) {
      const CeedScalar *u_b  = u_c;
      rtype             v_00 = add ? load(pg, v_0, 0) : setzero();

      for (CeedInt b = 0; b < B; b++, u_b += C) fmadd(pg, v_00, set1(t_0[(CeedSize)b * t_stride_1]), load(pg, u_b, 0));
      store(pg, v_0, 0, v_00);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract C=1
//------------------------------------------------------------------------------
// v[a,j] (+)= sum_b t[j,b] u[a,b]; vectorized over j, tiled over a.
// t[j,b] is contiguous in j when transposed and gathered with stride B
// otherwise.
// Kept out of line so the dispatcher stays small
static __attribute__((noinline)) int CeedTensorContract_Sve_Single(CeedInt A, CeedInt B, CeedInt J, const CeedScalar *restrict t,
                                                                   CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u,
                                                                   CeedScalar *restrict v) {
  const CeedSize vector_length = vlength();
  CeedInt        a             = 0;

  // Blocks of 4 rows
  for (; a + 3 < A; a += 4, u += 4 * (CeedSize)B, v += 4 * (CeedSize)J) {
    const CeedScalar *const u_1 = u + B, *const u_2 = u_1 + B, *const u_3 = u_2 + B;
    CeedScalar *const v_1 = v + J, *const v_2 = v_1 + J, *const v_3 = v_2 + J;

    for (CeedSize j = 0; j < J; j += vector_length) {
      const svbool_t pg   = whilelt(j, J);
      rtype          v_00 = add ? load(pg, v + j, 0) : setzero(), v_10 = add ? load(pg, v_1 + j, 0) : setzero();
      rtype          v_20 = add ? load(pg, v_2 + j, 0) : setzero(), v_30 = add ? load(pg, v_3 + j, 0) : setzero();

      if (t_mode == CEED_TRANSPOSE) {
        const CeedScalar *t_b = t + j;

        for (CeedInt b = 0; b < B; b++, t_b += J) {
          const rtype t_v = load(pg, t_b, 0);

          fmadd(pg, v_00, t_v, set1(u[b]));
          fmadd(pg, v_10, t_v, set1(u_1[b]));
          fmadd(pg, v_20, t_v, set1(u_2[b]));
          fmadd(pg, v_30, t_v, set1(u_3[b]));
        }
      } else {
        const CeedScalar *t_b = t + j * (CeedSize)B;

        for (CeedInt b = 0; b < B; b++, t_b++) {
          const rtype t_v = gather(pg, t_b, gather_index(B));

          fmadd(pg, v_00, t_v, set1(u[b]));
          fmadd(pg, v_10, t_v, set1(u_1[b]));
          fmadd(pg, v_20, t_v, set1(u_2[b]));
          fmadd(pg, v_30, t_v, set1(u_3[b]));
        }
      }
      store(pg, v + j, 0, v_00), store(pg, v_1 + j, 0, v_10), store(pg, v_2 + j, 0, v_20), store(pg, v_3 + j, 0, v_30);
    }
  }
  // Remainder of rows, 4 vectors of columns so the reduction keeps 4
  // independent accumulators
  for (; a < A; a++, u += B, v += J) {
    CeedSize j = 0;

    for (; j + 3 * vector_length < J; j += 4 * vector_length) {
      const svbool_t pg = ptrue(), pg_3 = whilelt(j + 3 * vector_length, J);
      CeedScalar    *v_j  = v + j;
      rtype          v_00 = add ? load(pg, v_j, 0) : setzero(), v_01 = add ? load(pg, v_j, 1) : setzero();
      rtype          v_02 = add ? load(pg, v_j, 2) : setzero(), v_03 = add ? load(pg_3, v_j, 3) : setzero();

      if (t_mode == CEED_TRANSPOSE) {
        const CeedScalar *t_b = t + j;

        for (CeedInt b = 0; b < B; b++, t_b += J) {
          const rtype u_b = set1(u[b]);

          fmadd(pg, v_00, load(pg, t_b, 0), u_b);
          fmadd(pg, v_01, load(pg, t_b, 1), u_b);
          fmadd(pg, v_02, load(pg, t_b, 2), u_b);
          fmadd(pg_3, v_03, load(pg_3, t_b, 3), u_b);
        }
      } else {
        const CeedScalar *t_b     = t + j * (CeedSize)B;
        const CeedSize    t_col_1 = vector_length * (CeedSize)B, t_col_2 = 2 * t_col_1, t_col_3 = 3 * t_col_1;

        for (CeedInt b = 0; b < B; b++, t_b++) {
          const rtype u_b = set1(u[b]);

          fmadd(pg, v_00, gather(pg, t_b, gather_index(B)), u_b);
          fmadd(pg, v_01, gather(pg, t_b + t_col_1, gather_index(B)), u_b);
          fmadd(pg, v_02, gather(pg, t_b + t_col_2, gather_index(B)), u_b);
          fmadd(pg_3, v_03, gather(pg_3, t_b + t_col_3, gather_index(B)), u_b);
        }
      }
      store(pg, v_j, 0, v_00), store(pg, v_j, 1, v_01), store(pg, v_j, 2, v_02), store(pg_3, v_j, 3, v_03);
    }
    for (; j < J; j += vector_length) {
      const svbool_t pg  = whilelt(j, J);
      rtype          v_0 = add ? load(pg, v + j, 0) : setzero();

      if (t_mode == CEED_TRANSPOSE) {
        const CeedScalar *t_b = t + j;

        for (CeedInt b = 0; b < B; b++, t_b += J) fmadd(pg, v_0, load(pg, t_b, 0), set1(u[b]));
      } else {
        const CeedScalar *t_b = t + j * (CeedSize)B;

        for (CeedInt b = 0; b < B; b++, t_b++) fmadd(pg, v_0, gather(pg, t_b, gather_index(B)), set1(u[b]));
      }
      store(pg, v + j, 0, v_0);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Sve(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  if (C == 1) {
    CeedCallBackend(CeedTensorContract_Sve_Single(A, B, J, t, t_mode, add, u, v));
  } else {
    for (CeedInt a = 0; a < A; a++) {
      CeedCallBackend(CeedTensorContract_Sve_Slice(B, C, J, t, t_mode, add, &u[(CeedSize)a * B * C], &v[(CeedSize)a * J * C]));
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
