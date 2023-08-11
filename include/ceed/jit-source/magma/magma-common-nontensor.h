// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
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
// 1D thread config. with (P x 1) threads
// no sync at the end of the function
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void read_A_notrans_g2r_1D_nosync(const int tx, const T *dA, int ldda, T *sA, int slda, T rA[Q]) {
#pragma unroll
  for (int j = 0; j < Q; j++) {
    rA[j] = dA[j * ldda + tx];
  }
}

////////////////////////////////////////////////////////////////////////////////
// read A (trans) from global to reg.
// A is (P x Q)
// 1D thread config. with (P x 1) threads
// no sync at the end of the function
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void read_A_trans_g2r_1D_nosync(const int tx, const int ty, const T *dA, int ldda, T *sA, int slda, T rA[Q]) {
  const int nTH = MAGMA_BASIS_BOUNDS(P, MAGMA_MAXTHREADS_1D);
  const int tid = ty * blockDim.x + tx;
  int       i;

#pragma unroll
  for (i = 0; i < (Q * P) - nTH; i += nTH) {
    sA[i + tid] = dA[i + tid];
  }
  if (tid < ((Q * P) - i)) {
    sA[i + tid] = dA[i + tid];
  }
  __syncthreads();

#pragma unroll
  for (int j = 0; j < Q; j++) {
    rA[j] = sA[tx * slda + j];
  }
}

////////////////////////////////////////////////////////////////////////////////
// read B from global to shared
// B is (Q x NB)
// 1D thread config. with (P x 1) threads
// no sync at the end of the function
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void read_B_g2s_1D_nosync(const int tx, const int n, const T *dB, int lddb, T *sB, int sldb) {
  if (n != NB) {
    for (int i = 0; i < (Q * n) - P; i += P) {
      sB[i + tx] = dB[i + tx];
    }
  } else {
#pragma unroll
    for (int i = 0; i < (Q * NB) - P; i += P) {
      sB[i + tx] = dB[i + tx];
    }
  }

  // cleanup for B
  const int stride = MAGMA_ROUNDUP(Q * n - P, P);
  if (tx < (Q * n) - stride) {
    sB[stride + tx] = dB[stride + tx];
  }
}

////////////////////////////////////////////////////////////////////////////////
// write C from reg. to global
// C is (P x NB)
// 1D thread config. with (P x 1) threads
// no sync at the end of the function
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void write_C_r2g_1D_nosync(const int tx, const int n, T rC[NB], T *dC, int lddc) {
  if (n != NB) {
#pragma unroll
    for (int j = 0; j < NB; j++) {
      if (j < n) {
        dC[j * lddc + tx] = rC[j];
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < NB; j++) {
      dC[j * lddc + tx] = rC[j];
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
static __device__ __inline__ void mul_rAsBrC_1D_nosync(const int tx, T rA[Q], T *sB, int sldb, T rC[NB]) {
  T rB[Q];
#pragma unroll
  for (int i = 0; i < NB; i++) {
#pragma unroll
    for (int k = 0; k < Q; k++) {
      rB[k] = sB[i * sldb + k];
    }
    rC[i] = 0.0;
#pragma unroll
    for (int k = 0; k < Q; k++) {
      rC[i] += rA[k] * rB[k];
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
static __device__ __inline__ void addmul_rAsBrC_1D_nosync(const int tx, T rA[Q], T *sB, int sldb, T rC[NB]) {
  T rB[Q];
#pragma unroll
  for (int i = 0; i < NB; i++) {
#pragma unroll
    for (int k = 0; k < Q; k++) {
      rB[k] = sB[i * sldb + k];
    }
#pragma unroll
    for (int k = 0; k < Q; k++) {
      rC[i] += rA[k] * rB[k];
    }
  }
}

#endif  // CEED_MAGMA_COMMON_NONTENSOR_H
