// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CPU JiT backend TensorContract templates
#pragma once

#include <ceed/types.h>
#include "cpu-gen-utils.h"

//------------------------------------------------------------------------------
// Tensor Contractions
//------------------------------------------------------------------------------

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_ApplyAdd_NoTranspose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt j = 0; j < J; j++) {
        for (CeedInt c = 0; c < C; c++) v[(a * J + j) * C + c] += t[j * B + b] * u[(a * B + b) * C + c];
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

// v_Ap += t_pB u_ABp for p = 0,...,P-1
template <CeedInt A, CeedInt B, CeedInt P>
static inline int TensorContract_ApplyAdd_Points_NoTranspose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  constexpr CeedInt t_stride_0 = B, t_stride_1 = 1;

  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt p = 0; p < P; p++) {
        v[a * P + p] += t[p * B + b] * u[(a * B + b) * P + p];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

// v_Ap = t_pB u_ABp for p = 0,...,P-1
template <CeedInt A, CeedInt B, CeedInt P>
static inline int TensorContract_Apply_Points_NoTranspose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  for (CeedInt q = 0; q < A * P; q++) v[q] = (CeedScalar)0.0;

  TensorContract_ApplyAdd_Points_NoTranspose<A, B, P>(t, u, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_ApplyAdd_Transpose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt b = 0; b < B; b++) {
      for (CeedInt j = 0; j < J; j++) {
        for (CeedInt c = 0; c < C; c++) v[(a * J + j) * C + c] += t[b * J + j] * u[(a * B + b) * C + c];
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

// v_AJp += t_pJ u_Ap for p = 0,...,P-1
template <CeedInt A, CeedInt J, CeedInt P>
static inline int TensorContract_ApplyAdd_Points_Transpose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  for (CeedInt a = 0; a < A; a++) {
    for (CeedInt j = 0; j < J; j++) {
      for (CeedInt p = 0; p < P; p++) {
        v[(a * J + j) * P + p] += t[p * J + j] * u[a * P + p];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

// v_AJp = t_pJ u_Ap for p = 0,...,P-1
template <CeedInt A, CeedInt J, CeedInt P>
static inline int TensorContract_Apply_Points_Transpose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  for (CeedInt q = 0; q < A * J * P; q++) v[q] = (CeedScalar)0.0;

  TensorContract_ApplyAdd_Points_Transpose<A, J, P>(t, u, v);
  return CEED_ERROR_SUCCESS;
}
