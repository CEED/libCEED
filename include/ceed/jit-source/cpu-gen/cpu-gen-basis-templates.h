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
// Tensor Contractions
//------------------------------------------------------------------------------

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_Apply_NoTranspose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  const CeedInt t_stride_0 = B, t_stride_1 = 1;

  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt j = 0; j < J; j++) {
        const CeedScalar tq = t[j * t_stride_0 + b * t_stride_1];

        for (CeedInt c = 0; c < C; c++) v[(a * J + j) * C + c] = tq * u[(a * B + b) * C + c];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_Apply_Transpose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  const CeedInt t_stride_0 = 1, t_stride_1 = J;

  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt j = 0; j < J; j++) {
        const CeedScalar tq = t[j * t_stride_0 + b * t_stride_1];

        for (CeedInt c = 0; c < C; c++) v[(a * J + j) * C + c] = tq * u[(a * B + b) * C + c];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_ApplyAdd_Transpose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  const CeedInt t_stride_0 = 1, t_stride_1 = J;

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

//------------------------------------------------------------------------------
// NonTensor General
//------------------------------------------------------------------------------

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt Q_COMP, CeedInt P, CeedInt Q>
static inline int CeedBasis_Apply_NoTranspose_NonTensor(const CeedScalar *mat, const CeedScalar *u, CeedScalar *v) {
  for (CeedInt d = 0; d < Q_COMP; d++) {
    CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P, BLK_SIZE, Q>(mat + d * P * Q, u, v + d * NUM_COMP * BLK_SIZE * Q));
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt Q_COMP, CeedInt P, CeedInt Q>
static inline int CeedBasis_Apply_Transpose_NonTensor(const CeedScalar *mat, const CeedScalar *u, CeedScalar *v) {
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP, Q, BLK_SIZE, P>(mat, u, v));
  for (CeedInt d = 1; d < Q_COMP; d++) {
    CeedCall(TensorContract_ApplyAdd_Transpose<NUM_COMP, Q, BLK_SIZE, P>(mat + d * P * Q, u + d * NUM_COMP * BLK_SIZE * Q, v));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// H1 Tensor
//------------------------------------------------------------------------------

// Weights

template <CeedInt BLK_SIZE, CeedInt MAX_POINTS>
static inline int CeedBasis_Apply_Weight_AtPoints(CeedScalar *v) {
  for (CeedInt i = 0; i < BLK_SIZE * MAX_POINTS; i++) v[i] = 1.0;
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt Q_1D>
static inline int CeedBasis_Apply_Weight_Tensor_1D(const CeedScalar *weights_1d, CeedScalar *v) {
  for (CeedInt i = 0; i < Q_1D; i++) {
    const CeedScalar w = weights_1d[i];

    for (CeedInt b = 0; b < BLK_SIZE; b++) v[i * BLK_SIZE + b] = w;
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt Q_1D>
static inline int CeedBasis_Apply_Weight_Tensor_2D(const CeedScalar *weights_1d, CeedScalar *v) {
  for (CeedInt i = 0; i < Q_1D; i++) {
    for (CeedInt j = 0; j < Q_1D; j++) {
      const CeedScalar w = weights_1d[i] * weights_1d[j];

      for (CeedInt b = 0; b < BLK_SIZE; b++) v[(i * Q_1D + j) * BLK_SIZE + b] = w;
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt Q_1D>
static inline int CeedBasis_Apply_Weight_Tensor_3D(const CeedScalar *weights_1d, CeedScalar *v) {
  // Compute 2D weights
  const CeedInt last_slab = Q_1D - 1;

  for (CeedInt i = 0; i < Q_1D; i++) {
    for (CeedInt j = 0; j < Q_1D; j++) {
      v[((last_slab * Q_1D + j) * Q_1D + i) * BLK_SIZE] = weights_1d[i] * weights_1d[j];
    }
  }
  // Convert to 3D weights
  for (CeedInt i = 0; i < Q_1D - 1; i++) {
    for (CeedInt j = 0; j < Q_1D; j++) {
      for (CeedInt k = 0; k < Q_1D - 1; k++) {
        const CeedScalar w = v[((last_slab * Q_1D + j) * Q_1D + i) * BLK_SIZE] * weights_1d[k];

        for (CeedInt b = 0; b < BLK_SIZE; b++) v[((k * Q_1D + j) * Q_1D + i) * BLK_SIZE + b] = w;
      }
      const CeedScalar w = v[((last_slab * Q_1D + j) * Q_1D + i) * BLK_SIZE] * weights_1d[last_slab];

      for (CeedInt b = 0; b < BLK_SIZE; b++) v[((last_slab * Q_1D + j) * Q_1D + i) * BLK_SIZE] = w;
    }
  }
  return CEED_ERROR_SUCCESS;
}

// Interp

// -- NoTranspose

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Interp_Tensor_1D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE, Q_1D>(interp_1d, u, v));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Interp_Tensor_2D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedScalar temp[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLK_SIZE, Q_1D>(interp_1d, u, temp));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE * Q_1D, Q_1D>(interp_1d, temp, v));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Interp_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedScalar temp_0[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLK_SIZE, Q_1D>(interp_1d, u, temp_0));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLK_SIZE * Q_1D, Q_1D>(interp_1d, temp_0, temp_1));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE * Q_1D * Q_1D, Q_1D>(interp_1d, temp_1, v));
  return CEED_ERROR_SUCCESS;
}

// -- Transpose

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_Transpose_Interp_Tensor_1D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLK_SIZE, P_1D>(interp_1d, u, v));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_Transpose_Interp_Tensor_2D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedScalar temp[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE, P_1D>(interp_1d, u, temp));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLK_SIZE * P_1D, P_1D>(interp_1d, temp, v));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_Transpose_Interp_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *u, CeedScalar *v) {
  CeedScalar temp_0[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLK_SIZE, P_1D>(interp_1d, u, temp_0));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE * P_1D, P_1D>(interp_1d, temp_0, temp_1));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_1, v));
  return CEED_ERROR_SUCCESS;
}

// Grad

// -- NoTranspose

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_Tensor_1D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                             CeedScalar *v) {
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE, Q_1D>(grad_1d, u, v));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_Tensor_2D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                             CeedScalar *v) {
  CeedScalar temp[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLK_SIZE, Q_1D>(interp_1d, u, temp));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE * Q_1D, Q_1D>(grad_1d, temp, v));

  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLK_SIZE, Q_1D>(grad_1d, u, temp));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE * Q_1D, Q_1D>(interp_1d, temp, &v[BLK_SIZE * NUM_COMP * Q_1D * Q_1D]));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                             CeedScalar *v) {
  CeedScalar temp_0[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLK_SIZE, Q_1D>(interp_1d, u, temp_0));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLK_SIZE * Q_1D, Q_1D>(interp_1d, temp_0, temp_1));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE * Q_1D * Q_1D, Q_1D>(grad_1d, temp_1, v));

  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLK_SIZE, Q_1D>(interp_1d, u, temp_0));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLK_SIZE * Q_1D, Q_1D>(grad_1d, temp_0, temp_1));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE * Q_1D * Q_1D, Q_1D>(interp_1d, temp_1,
                                                                                          &v[1 * BLK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]));

  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLK_SIZE, Q_1D>(interp_1d, u, temp_0));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLK_SIZE * Q_1D, Q_1D>(interp_1d, temp_0, temp_1));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE * Q_1D * Q_1D, Q_1D>(grad_1d, temp_1,
                                                                                          &v[2 * BLK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_NoTranspose_Grad_Collo_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *collo_grad_1d, const CeedScalar *u,
                                                                   CeedScalar *v) {
  CeedScalar temp_0[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D * P_1D, P_1D, BLK_SIZE, Q_1D>(interp_1d, u, temp_0));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * P_1D, P_1D, BLK_SIZE * Q_1D, Q_1D>(interp_1d, temp_0, temp_1));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, P_1D, BLK_SIZE * Q_1D * Q_1D, Q_1D>(interp_1d, temp_1, temp_0));

  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLK_SIZE, Q_1D>(collo_grad_1d, temp_0, v));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE * Q_1D, Q_1D>(collo_grad_1d, temp_0,
                                                                                          &v[1 * BLK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]));
  CeedCall(TensorContract_Apply_NoTranspose<NUM_COMP, Q_1D, BLK_SIZE * Q_1D * Q_1D, Q_1D>(collo_grad_1d, temp_0,
                                                                                          &v[2 * BLK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D]));
  return CEED_ERROR_SUCCESS;
}

// -- Transpose

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_Transpose_Grad_Tensor_1D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                           CeedScalar *v) {
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLK_SIZE, P_1D>(grad_1d, u, v));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_Transpose_Grad_Tensor_2D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                           CeedScalar *v) {
  CeedScalar temp[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE, P_1D>(grad_1d, u, temp));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP, P_1D, BLK_SIZE * P_1D, P_1D>(interp_1d, temp, v));

  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE, P_1D>(interp_1d, u[BLK_SIZE * NUM_COMP * Q_1D * Q_1D], temp));
  CeedCall(TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLK_SIZE * P_1D, P_1D>(grad_1d, temp, v));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_Transpose_Grad_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *grad_1d, const CeedScalar *u,
                                                           CeedScalar *v) {
  CeedScalar temp_0[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLK_SIZE, P_1D>(grad_1d, u, temp_0));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE * P_1D, P_1D>(interp_1d, temp_0, temp_1));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_1, v));

  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLK_SIZE, P_1D>(interp_1d, &u[1 * BLK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D],
                                                                                        temp_0));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE * P_1D, P_1D>(grad_1d, temp_0, temp_1));
  CeedCall(TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_1, v));

  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLK_SIZE, P_1D>(interp_1d, &u[2 * BLK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D],
                                                                                        temp_0));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE * P_1D, P_1D>(interp_1d, temp_0, temp_1));
  CeedCall(TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLK_SIZE * P_1D * P_1D, P_1D>(grad_1d, temp_1, v));
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt P_1D, CeedInt Q_1D>
static inline int CeedBasis_Apply_Transpose_Grad_Collo_Tensor_3D(const CeedScalar *interp_1d, const CeedScalar *collo_grad_1d, const CeedScalar *u,
                                                                 CeedScalar *v) {
  CeedScalar temp_0[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];
  CeedScalar temp_1[BLK_SIZE * NUM_COMP * Q_1D * CeedIntMax(Q_1D, P_1D) * CeedIntMax(Q_1D, P_1D)];

  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLK_SIZE, Q_1D>(collo_grad_1d, u, temp_0));
  CeedCall(TensorContract_ApplyAdd_Transpose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE * Q_1D, Q_1D>(collo_grad_1d,
                                                                                           &u[1 * BLK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D], temp_0));
  CeedCall(TensorContract_ApplyAdd_Transpose<NUM_COMP, Q_1D, BLK_SIZE * Q_1D * Q_1D, Q_1D>(collo_grad_1d,
                                                                                           &u[2 * BLK_SIZE * NUM_COMP * Q_1D * Q_1D * Q_1D], temp_0));

  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D * Q_1D, Q_1D, BLK_SIZE, P_1D>(interp_1d, temp_0, temp_1));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP * Q_1D, Q_1D, BLK_SIZE * P_1D, P_1D>(interp_1d, temp_1, temp_0));
  CeedCall(TensorContract_Apply_Transpose<NUM_COMP, Q_1D, BLK_SIZE * P_1D * P_1D, P_1D>(interp_1d, temp_0, v));
  return CEED_ERROR_SUCCESS;
}
