// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CPU JiT backend Basis templates
#pragma once

#include <ceed/types.h>
#include "cpu-gen-utils.h"

//------------------------------------------------------------------------------
// At-Points Basis Helpers
//------------------------------------------------------------------------------

template <CeedInt Q_1D>
inline void ChebyshevPolynomialsAtPoint(const CeedScalar x, CeedScalar *chebyshev_x) {
  chebyshev_x[0] = 1.0;
  chebyshev_x[1] = x;
  for (CeedInt i = 2; i < Q_1D; i++) chebyshev_x[i] = 2 * x * chebyshev_x[i - 1] - chebyshev_x[i - 2];
}

template <CeedInt Q_1D>
inline void ChebyshevDerivativeAtPoint(const CeedScalar x, CeedScalar *chebyshev_dx) {
  CeedScalar chebyshev_x[2];

  chebyshev_x[0]  = 1.0;
  chebyshev_x[1]  = 2 * x;
  chebyshev_dx[0] = 0.0;
  chebyshev_dx[1] = 1.0;
  for (CeedInt i = 2; i < Q_1D; i++) {
    // dT_i/dx = i * dU_{i-1}/dx
    chebyshev_dx[i]    = i * chebyshev_x[(i + 1) % 2];
    chebyshev_x[i % 2] = 2 * x * chebyshev_x[(i + 1) % 2] - chebyshev_x[i % 2];
  }
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt DIM_POINTS, CeedInt Q_1D>
inline void CeedBasis_ChebyshevPolynomialEval(const CeedScalar *x, CeedScalar *chebyshev_x) {
  for (CeedInt d = 0; d < DIM_POINTS; d++) {
    for (CeedInt p = 0; p < BLOCK_SIZE * NUM_POINTS; p++) {
      ChebyshevPolynomialsAtPoint<Q_1D>(x[d * BLOCK_SIZE * NUM_POINTS + p], &chebyshev_x[(d * BLOCK_SIZE * NUM_POINTS + p) * Q_1D]);
    }
  }
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt DIM_POINTS, CeedInt Q_1D>
inline void CeedBasis_ChebyshevDerivativeEval(const CeedScalar *x, CeedScalar *chebyshev_dx) {
  for (CeedInt d = 0; d < DIM_POINTS; d++) {
    for (CeedInt p = 0; p < BLOCK_SIZE * NUM_POINTS; p++) {
      ChebyshevDerivativeAtPoint<Q_1D>(x[d * BLOCK_SIZE * NUM_POINTS + p], &chebyshev_dx[(d * BLOCK_SIZE * NUM_POINTS + p) * Q_1D]);
    }
  }
}

//------------------------------------------------------------------------------
// Tensor Contractions
//------------------------------------------------------------------------------

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_ApplyAdd_NoTranspose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  constexpr CeedInt t_stride_0 = B, t_stride_1 = 1;

  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt j = 0; j < J; j++) {
        const CeedScalar tq = t[j * t_stride_0 + b * t_stride_1];

        for (CeedInt c = 0; c < C; c++) v[(a * J + j) * C + c] += tq * u[(a * B + b) * C + c];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_Apply_NoTranspose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  for (CeedInt q = 0; q < A * J * C; q++) v[q] = (CeedScalar)0.0;

  TensorContract_ApplyAdd_NoTranspose<A, B, C, J>(t, u, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_ApplyAdd_Transpose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  constexpr CeedInt t_stride_0 = 1, t_stride_1 = J;

  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt j = 0; j < J; j++) {
        const CeedScalar tq = t[j * t_stride_0 + b * t_stride_1];

        for (CeedInt c = 0; c < C; c++) v[(a * J + j) * C + c] += tq * u[(a * B + b) * C + c];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_Apply_Transpose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  for (CeedInt q = 0; q < A * J * C; q++) v[q] = (CeedScalar)0.0;

  TensorContract_ApplyAdd_Transpose<A, B, C, J>(t, u, v);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// NonTensor General
//------------------------------------------------------------------------------

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt Q_COMP, CeedInt P, CeedInt Q>
static inline int CeedBasis_Apply_NoTranspose_NonTensor(const CeedScalar *mat, const CeedScalar *u, CeedScalar *v) {
  for (CeedInt d = 0; d < Q_COMP; d++) {
    TensorContract_Apply_NoTranspose<NUM_COMP, P, BLOCK_SIZE, Q>(mat + d * P * Q, u, v + d * NUM_COMP * BLOCK_SIZE * Q);
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt Q_COMP, CeedInt P, CeedInt Q, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_NonTensor(const CeedScalar *mat, const CeedScalar *u, CeedScalar *v) {
  if (!APPLY_ADD)
    for (CeedInt q = 0; q < P * NUM_COMP * BLOCK_SIZE; q++) v[q] = (CeedScalar)0.0;
  for (CeedInt d = 0; d < Q_COMP; d++) {
    TensorContract_ApplyAdd_Transpose<NUM_COMP, Q, BLOCK_SIZE, P>(mat + d * P * Q, u + d * NUM_COMP * BLOCK_SIZE * Q, v);
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt Q>
static inline int CeedBasis_Apply_Weight_NonTensor(const CeedScalar *weights, CeedScalar *v) {
  for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar w = weights[i];

    for (CeedInt b = 0; b < BLOCK_SIZE; b++) v[i * BLOCK_SIZE + b] = w;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// H1 Tensor
//------------------------------------------------------------------------------

// Weights

template <CeedInt BLOCK_SIZE, CeedInt MAX_POINTS>
static inline int CeedBasis_Apply_Weight_AtPoints(CeedScalar *v) {
  for (CeedInt i = 0; i < BLOCK_SIZE * MAX_POINTS; i++) v[i] = 1.0;
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt Q_1D>
static inline int CeedBasis_Apply_Weight_Tensor_1D(const CeedScalar *weights_1d, CeedScalar *v) {
  for (CeedInt i = 0; i < Q_1D; i++) {
    const CeedScalar w = weights_1d[i];

    for (CeedInt b = 0; b < BLOCK_SIZE; b++) v[i * BLOCK_SIZE + b] = w;
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt Q_1D>
static inline int CeedBasis_Apply_Weight_Tensor_2D(const CeedScalar *weights_1d, CeedScalar *v) {
  for (CeedInt i = 0; i < Q_1D; i++) {
    for (CeedInt j = 0; j < Q_1D; j++) {
      const CeedScalar w = weights_1d[i] * weights_1d[j];

      for (CeedInt b = 0; b < BLOCK_SIZE; b++) v[(i * Q_1D + j) * BLOCK_SIZE + b] = w;
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt Q_1D>
static inline int CeedBasis_Apply_Weight_Tensor_3D(const CeedScalar *weights_1d, CeedScalar *v) {
  for (CeedInt i = 0; i < Q_1D; i++) {
    for (CeedInt j = 0; j < Q_1D; j++) {
      const CeedScalar w = weights_1d[i] * weights_1d[j];

      for (CeedInt k = 0; k < Q_1D; k++) {
        const CeedScalar w_k = w * weights_1d[k];

        for (CeedInt b = 0; b < BLOCK_SIZE; b++) v[((i * Q_1D + j) * Q_1D + k) * BLOCK_SIZE + b] = w_k;
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

// Interp

// -- NoTranspose

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Interp_Tensor_1D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE, Q_1D>(interp_1d, u, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Interp_Tensor_2D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedScalar temp[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLOCK_SIZE, Q_1D>(interp_1d, u, temp);
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE * Q_1D, Q_1D>(interp_1d, temp, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Interp_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLOCK_SIZE, Q_1D>(interp_1d, u, temp_0);
  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLOCK_SIZE * Q_1D, Q_1D>(interp_1d, temp_0, temp_1);
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE * Q_1D * Q_1D, Q_1D>(interp_1d, temp_1, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Interp_AtPoints_Tensor_1D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                        const CeedScalar *u, CeedScalar *v_points) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D] = {0.};

  // CeedBasis_Apply_NoTranspose_Interp_Tensor_1D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D>(interp_cheby_1d, u, u_cheby);
  CeedBasis_Apply_NoTranspose_Interp_Tensor_1D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D>(interp_cheby_1d, u, u_cheby);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, 1, BLOCK_SIZE * NUM_POINTS>(x_ref_cheby, u_cheby, v_points);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Interp_AtPoints_Tensor_2D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                        const CeedScalar *u, CeedScalar *v_points) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D]    = {0.};
  CeedScalar temp[BLOCK_SIZE * NUM_COMP * Q_1D * NUM_POINTS] = {0.};

  CeedBasis_Apply_NoTranspose_Interp_Tensor_2D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D>(interp_cheby_1d, u, u_cheby);
  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D, Q_1D, 1, BLOCK_SIZE * NUM_POINTS>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], u_cheby,
                                                                                      temp);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, BLOCK_SIZE * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(&x_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                                                     temp, v_points);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Interp_AtPoints_Tensor_3D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                        const CeedScalar *u, CeedScalar *v_points) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]            = {0.};
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * NUM_POINTS]       = {0.};
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * Q_1D * NUM_POINTS * NUM_POINTS] = {0.};

  CeedBasis_Apply_NoTranspose_Interp_Tensor_3D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D>(interp_cheby_1d, u, u_cheby);
  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D * Q_1D, Q_1D, 1, BLOCK_SIZE * NUM_POINTS>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                                             u_cheby, temp_0);
  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(
      &x_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_0, temp_1);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, BLOCK_SIZE * NUM_POINTS * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(
      &x_ref_cheby[2 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_1, v_points);
  return CEED_ERROR_SUCCESS;
}

// -- Transpose

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Interp_Tensor_1D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  if (APPLY_ADD)
    TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE, P_1D>(interp_1d, u, v);
  else
    TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE, P_1D>(interp_1d, u, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Interp_Tensor_2D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedScalar temp[BLOCK_SIZE * NUM_COMP * P_1D * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE, P_1D>(interp_1d, u, temp);
  if (APPLY_ADD)
    TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D, P_1D>(interp_1d, temp, v);
  else
    TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D, P_1D>(interp_1d, temp, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Interp_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * P_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * P_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLOCK_SIZE, P_1D>(interp_1d, u, temp_0);
  TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * P_1D, P_1D>(interp_1d, temp_0, temp_1);
  if (APPLY_ADD)
    TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_1, v);
  else
    TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_1, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Interp_AtPoints_Tensor_1D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                      const CeedScalar *u_points, CeedScalar *v) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D] = {0.};

  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, 1, Q_1D>(x_ref_cheby, u_points, u_cheby);
  CeedBasis_Apply_Transpose_Interp_Tensor_1D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D, APPLY_ADD>(interp_cheby_1d, u_cheby, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Interp_AtPoints_Tensor_2D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                      const CeedScalar *u_points, CeedScalar *v) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D]    = {0.};
  CeedScalar temp[BLOCK_SIZE * NUM_COMP * Q_1D * NUM_POINTS] = {0.};

  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, 1, Q_1D>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], u_points, temp);
  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D, Q_1D>(&x_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp, u_cheby);
  CeedBasis_Apply_Transpose_Interp_Tensor_2D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D, APPLY_ADD>(interp_cheby_1d, u_cheby, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Interp_AtPoints_Tensor_3D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                      const CeedScalar *u_points, CeedScalar *v) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]            = {0.};
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * NUM_POINTS]       = {0.};
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * Q_1D * NUM_POINTS * NUM_POINTS] = {0.};

  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, 1, Q_1D>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], u_points, temp_0);
  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D, Q_1D>(&x_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_0, temp_1);
  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D * Q_1D, Q_1D>(&x_ref_cheby[2 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_1,
                                                                                       u_cheby);
  CeedBasis_Apply_Transpose_Interp_Tensor_2D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D, APPLY_ADD>(interp_cheby_1d, u_cheby, v);
  return CEED_ERROR_SUCCESS;
}

// Grad

// -- NoTranspose

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_Tensor_1D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                             CeedScalar *v) {
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE, Q_1D>(grad_1d, u, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_Tensor_2D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                             CeedScalar *v) {
  CeedScalar temp[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLOCK_SIZE, Q_1D>(grad_1d, u, temp);
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE * Q_1D, Q_1D>(interp_1d, temp, v);

  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLOCK_SIZE, Q_1D>(interp_1d, u, temp);
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE * Q_1D, Q_1D>(grad_1d, temp, &v[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D]);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                             CeedScalar *v) {
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLOCK_SIZE, Q_1D>(grad_1d, u, temp_0);
  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLOCK_SIZE * Q_1D, Q_1D>(interp_1d, temp_0, temp_1);
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE * Q_1D * Q_1D, Q_1D>(interp_1d, temp_1, v);

  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLOCK_SIZE, Q_1D>(interp_1d, u, temp_0);
  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLOCK_SIZE * Q_1D, Q_1D>(grad_1d, temp_0, temp_1);
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE * Q_1D * Q_1D, Q_1D>(interp_1d, temp_1,
                                                                                   &v[1 * BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]);

  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLOCK_SIZE, Q_1D>(interp_1d, u, temp_0);
  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLOCK_SIZE * Q_1D, Q_1D>(interp_1d, temp_0, temp_1);
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE * Q_1D * Q_1D, Q_1D>(grad_1d, temp_1,
                                                                                   &v[2 * BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_Collo_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *collo_grad_1d, const CeedScalar *u,
                                                                   CeedScalar *v) {
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLOCK_SIZE, Q_1D>(interp_1d, u, temp_0);
  TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLOCK_SIZE * Q_1D, Q_1D>(interp_1d, temp_0, temp_1);
  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE * Q_1D * Q_1D, Q_1D>(interp_1d, temp_1, temp_0);

  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLOCK_SIZE, Q_1D>(collo_grad_1d, temp_0, v);
  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * Q_1D, Q_1D>(collo_grad_1d, temp_0,
                                                                                   &v[1 * BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, BLOCK_SIZE * Q_1D * Q_1D, Q_1D>(collo_grad_1d, temp_0,
                                                                                   &v[2 * BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_AtPoints_Tensor_1D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                      const CeedScalar *dx_ref_cheby, const CeedScalar *u, CeedScalar *v_points) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D] = {0.};

  TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLOCK_SIZE, Q_1D>(interp_cheby_1d, u, u_cheby);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, 1, BLOCK_SIZE * NUM_POINTS>(dx_ref_cheby, u_cheby, v_points);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_AtPoints_Tensor_2D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                      const CeedScalar *dx_ref_cheby, const CeedScalar *u, CeedScalar *v_points) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D]    = {0.};
  CeedScalar temp[BLOCK_SIZE * NUM_COMP * Q_1D * NUM_POINTS] = {0.};

  CeedBasis_Apply_NoTranspose_Interp_Tensor_2D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D>(interp_cheby_1d, u, u_cheby);

  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D, Q_1D, 1, BLOCK_SIZE * NUM_POINTS>(&dx_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], u_cheby,
                                                                                      temp);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, BLOCK_SIZE * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(&x_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                                                     temp, v_points);

  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D, Q_1D, 1, BLOCK_SIZE * NUM_POINTS>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], u_cheby,
                                                                                      temp);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, BLOCK_SIZE * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(
      &dx_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp, &v_points[1 * BLOCK_SIZE * NUM_COMP * NUM_POINTS]);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_AtPoints_Tensor_3D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                      const CeedScalar *dx_ref_cheby, const CeedScalar *u, CeedScalar *v_points) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]            = {0.};
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * NUM_POINTS]       = {0.};
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * Q_1D * NUM_POINTS * NUM_POINTS] = {0.};

  CeedBasis_Apply_NoTranspose_Interp_Tensor_3D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D>(interp_cheby_1d, u, u_cheby);

  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D * Q_1D, Q_1D, 1, BLOCK_SIZE * NUM_POINTS>(&dx_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                                             u_cheby, temp_0);
  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(
      &x_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_0, temp_1);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, BLOCK_SIZE * NUM_POINTS * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(
      &x_ref_cheby[2 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_1, &v_points[0 * BLOCK_SIZE * NUM_COMP * NUM_POINTS]);

  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D * Q_1D, Q_1D, 1, BLOCK_SIZE * NUM_POINTS>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                                             u_cheby, temp_0);
  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(
      &dx_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_0, temp_1);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, BLOCK_SIZE * NUM_POINTS * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(
      &x_ref_cheby[2 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_1, &v_points[1 * BLOCK_SIZE * NUM_COMP * NUM_POINTS]);

  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D * Q_1D, Q_1D, 1, BLOCK_SIZE * NUM_POINTS>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                                             u_cheby, temp_0);
  TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(
      &x_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_0, temp_1);
  TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, BLOCK_SIZE * NUM_POINTS * NUM_POINTS, BLOCK_SIZE * NUM_POINTS>(
      &dx_ref_cheby[2 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_1, &v_points[2 * BLOCK_SIZE * NUM_COMP * NUM_POINTS]);
  return CEED_ERROR_SUCCESS;
}

// -- Transpose

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Grad_Tensor_1D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                           CeedScalar *v) {
  if (APPLY_ADD)
    TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE, P_1D>(grad_1d, u, v);
  else
    TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE, P_1D>(grad_1d, u, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Grad_Tensor_2D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                           CeedScalar *v) {
  CeedScalar temp[BLOCK_SIZE * NUM_COMP * P_1D * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE, P_1D>(grad_1d, u, temp);
  if (APPLY_ADD)
    TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D, P_1D>(interp_1d, temp, v);
  else
    TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D, P_1D>(interp_1d, temp, v);

  TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE, P_1D>(interp_1d, &u[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D], temp);
  TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D, P_1D>(grad_1d, temp, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Grad_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                           CeedScalar *v) {
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * P_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * P_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLOCK_SIZE, P_1D>(grad_1d, u, temp_0);
  TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * P_1D, P_1D>(interp_1d, temp_0, temp_1);
  if (APPLY_ADD)
    TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_1, v);
  else
    TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_1, v);

  TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLOCK_SIZE, P_1D>(interp_1d, &u[1 * BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D],
                                                                                 temp_0);
  TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * P_1D, P_1D>(grad_1d, temp_0, temp_1);
  TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_1, v);

  TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLOCK_SIZE, P_1D>(interp_1d, &u[2 * BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D],
                                                                                 temp_0);
  TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * P_1D, P_1D>(interp_1d, temp_0, temp_1);
  TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D * P_1D, P_1D>(grad_1d, temp_1, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Grad_Collo_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *collo_grad_1d, const CeedScalar *u,
                                                                 CeedScalar *v) {
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLOCK_SIZE, Q_1D>(collo_grad_1d, u, temp_0);
  TensorContract_ApplyAdd_Transpose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * Q_1D, Q_1D>(collo_grad_1d, &u[1 * BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D],
                                                                                    temp_0);
  TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * Q_1D * Q_1D, Q_1D>(collo_grad_1d, &u[2 * BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D],
                                                                                    temp_0);

  TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLOCK_SIZE, P_1D>(interp_1d, temp_0, temp_1);
  TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLOCK_SIZE * P_1D, P_1D>(interp_1d, temp_1, temp_0);
  if (APPLY_ADD)
    TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_0, v);
  else
    TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_0, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Grad_AtPoints_Tensor_1D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                    const CeedScalar *dx_ref_cheby, const CeedScalar *u_points, CeedScalar *v) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D] = {0.};

  TensorContract_Apply_Transpose<NUM_COMP, 1, BLOCK_SIZE * NUM_POINTS, Q_1D>(dx_ref_cheby, u_points, u_cheby);
  TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLOCK_SIZE, P_1D, APPLY_ADD>(interp_cheby_1d, u_cheby, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Grad_AtPoints_Tensor_2D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                    const CeedScalar *dx_ref_cheby, const CeedScalar *u_points, CeedScalar *v) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D]    = {0.};
  CeedScalar temp[BLOCK_SIZE * NUM_COMP * Q_1D * NUM_POINTS] = {0.};

  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, 1, Q_1D>(&dx_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                             &u_points[1 * BLOCK_SIZE * NUM_COMP * NUM_POINTS], temp);
  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D, Q_1D>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp, u_cheby);

  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, 1, Q_1D>(&x_ref_cheby[1 * BLOCK_SIZE * NUM_POINTS * Q_1D], u_points, temp);
  TensorContract_ApplyAdd_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D, Q_1D>(&dx_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp, u_cheby);

  CeedBasis_Apply_Transpose_Interp_Tensor_2D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D, APPLY_ADD>(interp_cheby_1d, u_cheby, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLOCK_SIZE, CeedInt NUM_POINTS, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D, bool APPLY_ADD>
static inline int CeedBasis_Apply_Transpose_Grad_AtPoints_Tensor_3D(const CeedScalar *interp_cheby_1d, const CeedScalar *x_ref_cheby,
                                                                    const CeedScalar *dx_ref_cheby, const CeedScalar *u_points, CeedScalar *v) {
  CeedScalar u_cheby[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]            = {0.};
  CeedScalar temp_0[BLOCK_SIZE * NUM_COMP * Q_1D * Q_1D * NUM_POINTS]       = {0.};
  CeedScalar temp_1[BLOCK_SIZE * NUM_COMP * Q_1D * NUM_POINTS * NUM_POINTS] = {0.};

  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, 1, Q_1D>(&dx_ref_cheby[2 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                             &u_points[2 * BLOCK_SIZE * NUM_COMP * NUM_POINTS], temp_1);
  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D, Q_1D>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_1, temp_0);
  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D * Q_1D, Q_1D>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_0,
                                                                                       u_cheby);

  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, 1, Q_1D>(&x_ref_cheby[2 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                             &u_points[1 * BLOCK_SIZE * NUM_COMP * NUM_POINTS], temp_1);
  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D, Q_1D>(&dx_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_1, temp_0);
  TensorContract_ApplyAdd_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D * Q_1D, Q_1D>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_0,
                                                                                          u_cheby);

  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, 1, Q_1D>(&x_ref_cheby[2 * BLOCK_SIZE * NUM_POINTS * Q_1D],
                                                                             &u_points[0 * BLOCK_SIZE * NUM_COMP * NUM_POINTS], temp_1);
  TensorContract_Apply_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D, Q_1D>(&x_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_1, temp_0);
  TensorContract_ApplyAdd_Transpose<NUM_COMP, BLOCK_SIZE * NUM_POINTS, Q_1D * Q_1D, Q_1D>(&dx_ref_cheby[0 * BLOCK_SIZE * NUM_POINTS * Q_1D], temp_0,
                                                                                          u_cheby);

  CeedBasis_Apply_Transpose_Interp_Tensor_3D<BLOCK_SIZE, NUM_COMP, P_1D, Q_1D, APPLY_ADD>(interp_cheby_1d, u_cheby, v);
  return CEED_ERROR_SUCCESS;
}
