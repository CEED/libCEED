// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA backend common non-tensor basis definitions
#ifndef CEED_MAGMA_COMMON_NONTENSOR_H
#define CEED_MAGMA_COMMON_NONTENSOR_H

#include "magma-common-defs.h"

////////////////////////////////////////////////////////////////////////////////
// read A (no-trans) from global to reg.
// A is (P x Q)
// 2D thread config. with (P x BY) threads
// no sync at the end of the function
template <typename T, int P, int Q, int BY>
static __device__ __inline__ void read_A_notrans_g2r_1D_nosync(const int tx, const int ty, const T *dA, T *sA, T rA[Q]) {
  const int tid = ty * P + tx;
  int       i;

#pragma unroll
  for (i = 0; i < P * Q - P * BY; i += P * BY) {
    sA[i + tid] = dA[i + tid];
  }
  if (i + tid < P * Q) {
    sA[i + tid] = dA[i + tid];
  }
  __syncthreads();

#pragma unroll
  for (int j = 0; j < Q; j++) {
    rA[j] = sA[j * P + tx];
  }
}

////////////////////////////////////////////////////////////////////////////////
// read A (trans) from global to reg.
// A is (P x Q)
// 2D thread config. with (P x BY) threads
// no sync at the end of the function
template <typename T, int P, int Q, int BY>
static __device__ __inline__ void read_A_trans_g2r_1D_nosync(const int tx, const int ty, const T *dA, T *sA, T rA[Q]) {
  const int tid = ty * P + tx;
  int       i;

#pragma unroll
  for (i = 0; i < P * Q - P * BY; i += P * BY) {
    sA[i + tid] = dA[i + tid];
  }
  if (i + tid < P * Q) {
    sA[i + tid] = dA[i + tid];
  }
  __syncthreads();

#pragma unroll
  for (int j = 0; j < Q; j++) {
    rA[j] = sA[tx * Q + j];
  }
}

////////////////////////////////////////////////////////////////////////////////
// read B from global to shared
// B is (Q x NB)
// 1D thread config. with (P x 1) threads
// no sync at the end of the function
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void read_B_g2s_1D_nosync(const int tx, const int n, const T *dB, T *sB) {
  int i;

  if (n != NB) {
    for (i = 0; i < Q * n - P; i += P) {
      sB[i + tx] = dB[i + tx];
    }
  } else {
#pragma unroll
    for (i = 0; i < Q * NB - P; i += P) {
      sB[i + tx] = dB[i + tx];
    }
  }
  if (i + tx < Q * n) {
    sB[i + tx] = dB[i + tx];
  }
}

////////////////////////////////////////////////////////////////////////////////
// write C from reg. to global
// C is (P x NB)
// 1D thread config. with (P x 1) threads
// no sync at the end of the function
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void write_C_r2g_1D_nosync(const int tx, const int n, T rC[NB], T *dC) {
  if (n != NB) {
    for (int i = 0; i < n; i++) {
      dC[i * P + tx] = rC[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < NB; i++) {
      dC[i * P + tx] = rC[i];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// multiply C = A x B using 1D threads in P x 1 config
// A (P x Q)  in reg., one row per thread
// B (Q x NB) in shared memory
// C in registers -- one row per thread
// no sync at the end of the function
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void mul_rAsBrC_1D_nosync(T rA[Q], T *sB, T rC[NB]) {
  T rB[Q];

#pragma unroll
  for (int i = 0; i < NB; i++) {
#pragma unroll
    for (int j = 0; j < Q; j++) {
      rB[j] = sB[i * Q + j];
    }
    rC[i] = 0.0;
#pragma unroll
    for (int j = 0; j < Q; j++) {
      rC[i] += rA[j] * rB[j];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// multiply C += A x B using 1D threads in P x 1 config
// A (P x Q)  in reg., one row per thread
// B (Q x NB) in shared memory
// C in registers -- one row per thread
// no sync at the end of the function
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void addmul_rAsBrC_1D_nosync(T rA[Q], T *sB, T rC[NB]) {
  T rB[Q];

#pragma unroll
  for (int i = 0; i < NB; i++) {
#pragma unroll
    for (int j = 0; j < Q; j++) {
      rB[j] = sB[i * Q + j];
    }
#pragma unroll
    for (int j = 0; j < Q; j++) {
      rC[i] += rA[j] * rB[j];
    }
  }
}

#endif  // CEED_MAGMA_COMMON_NONTENSOR_H
