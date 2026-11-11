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
#include <libxsmm.h>
#include "cpu-gen-utils.h"

//------------------------------------------------------------------------------
// Tensor Contract Generic Version
//------------------------------------------------------------------------------
template <CeedInt A, CeedInt B, CeedInt C, CeedInt J, bool ADD, bool TRANSPOSE>
static int CeedTensorContractApply_Gen_Xsmm(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  if (C == 1) {
    // Build or query the required kernel
    const int                  flags_t    = LIBXSMM_GEMM_FLAGS(!TRANSPOSE ? 'T' : 'N', 'N');
    const int                  flags_ab   = (!ADD) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE;
    const int                  flags      = (flags_t | flags_ab);
    const libxsmm_gemm_shape   gemm_shape = (CEED_SCALAR_TYPE == CEED_SCALAR_FP64)
                                                ? libxsmm_create_gemm_shape(J, A, B, !TRANSPOSE ? B : J, B, J, LIBXSMM_DATATYPE_F64,
                                                                            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64)
                                                : libxsmm_create_gemm_shape(J, A, B, !TRANSPOSE ? B : J, B, J, LIBXSMM_DATATYPE_F32,
                                                                            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    const libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm(gemm_shape, (libxsmm_bitfield)(flags), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
    libxsmm_gemm_param         gemm_param;

    // Run kernel
    gemm_param.a.primary = (CeedScalar *)&t[0];
    gemm_param.b.primary = (CeedScalar *)&u[0];
    gemm_param.c.primary = (CeedScalar *)&v[0];
    kernel(&gemm_param);
  } else {
    // Build or query the required kernel
    const int                  flags_t    = LIBXSMM_GEMM_FLAGS('N', TRANSPOSE ? 'T' : 'N');
    const int                  flags_ab   = (!ADD) ? LIBXSMM_GEMM_FLAG_BETA_0 : LIBXSMM_BASIC_GEMM_FLAG_NONE;
    const int                  flags      = (flags_t | flags_ab);
    const libxsmm_gemm_shape   gemm_shape = (CEED_SCALAR_TYPE == CEED_SCALAR_FP64)
                                                ? libxsmm_create_gemm_shape(C, J, B, C, !TRANSPOSE ? B : J, C, LIBXSMM_DATATYPE_F64,
                                                                            LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64)
                                                : libxsmm_create_gemm_shape(C, J, B, C, !TRANSPOSE ? B : J, C, LIBXSMM_DATATYPE_F32,
                                                                            LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    const libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm(gemm_shape, (libxsmm_bitfield)(flags), (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);
    libxsmm_gemm_param         gemm_param;

    // Run kernel
    gemm_param.b.primary = (CeedScalar *)&t[0];
    for (CeedInt a = 0; a < A; a++) {
      gemm_param.a.primary = (CeedScalar *)&u[a * B * C];
      gemm_param.c.primary = (CeedScalar *)&v[a * J * C];
      kernel(&gemm_param);
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contractions
//------------------------------------------------------------------------------

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_ApplyAdd_NoTranspose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  CeedTensorContractApply_Gen_Xsmm<A, B, C, J, true, false>(t, u, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_Apply_NoTranspose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  CeedTensorContractApply_Gen_Xsmm<A, B, C, J, false, false>(t, u, v);
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
  CeedTensorContractApply_Gen_Xsmm<A, B, C, J, true, true>(t, u, v);
  return CEED_ERROR_SUCCESS;
}

template <CeedInt A, CeedInt B, CeedInt C, CeedInt J>
static inline int TensorContract_Apply_Transpose(const CeedScalar *t, const CeedScalar *u, CeedScalar *v) {
  CeedTensorContractApply_Gen_Xsmm<A, B, C, J, false, true>(t, u, v);
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
