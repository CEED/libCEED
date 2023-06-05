// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_COMMON_NONTENSOR_H
#define CEED_MAGMA_COMMON_NONTENSOR_H

#define NONTENSOR_MAX_THREADS (128)

#ifndef MAGMA_DEVICE_SHARED
#define MAGMA_DEVICE_SHARED
#ifdef CEED_MAGMA_USE_HIP
#define MAGMA_DEVICE_SHARED(type, name) HIP_DYNAMIC_SHARED(type, name)
#else
#define MAGMA_DEVICE_SHARED(type, name) extern __shared__ type name[];
#endif  // CEED_MAGMA_USE_HIP
#endif  // MAGMA_DEVICE_SHARED

#define MAGMA_NONTENSOR_BASIS_NTCOL(N) (MAGMA_MAX(1, (NONTENSOR_MAX_THREADS / (N))))

#define dA(i, j) dA[(j)*ldda + (i)]
#define sA(i, j) sA[(j)*slda + (i)]
#define dB(i, j) dB[(j)*lddb + (i)]
#define sB(i, j) sB[(j)*sldb + (i)]

////////////////////////////////////////////////////////////////////////////////
// read C from global to reg.
// C is (P_ x NB_)
// 1D thread config. with (Mx1) threads
// no sync at the end of the function
template <typename T, int P_, int NB_, int Q_>
static __device__ __inline__ void read_C_g2r_1D_nosync(const int tx, const int n, T* dC, int lddc, const T& beta, T rC[NB_]) {
  if (n != NB_) {
#pragma unroll
    for (int j = 0; j < NB_; j++) {
      rC[j] = (j < n) ? beta * dC[j * lddc + tx] : 0;
    }
  } else {
#pragma unroll
    for (int j = 0; j < NB_; j++) {
      rC[j] = beta * dC[j * lddc + tx];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// write C from reg. to global
// C is (P_ x NB_)
// 1D thread config. with (Mx1) threads
// no sync at the end of the function
template <typename T, int P_, int NB_, int Q_>
static __device__ __inline__ void write_C_r2g_1D_nosync(const int tx, const int n, T rC[NB_], T* dC, int lddc) {
  if (n != NB_) {
#pragma unroll
    for (int j = 0; j < NB_; j++) {
      if (j < n) {
        dC[j * lddc + tx] = rC[j];
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < NB_; j++) {
      dC[j * lddc + tx] = rC[j];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// read A (no-trans) from global to reg.
// A is (P_ x Q_)
// 1D thread config. with (Mx1) threads
// no sync at the end of the function
template <typename T, int P_, int NB_, int Q_>
static __device__ __inline__ void read_A_notrans_g2r_1D_nosync(const int tx, const T* dA, int ldda, T* sA, int slda, T rA[Q_]) {
#pragma unroll
  for (int j = 0; j < Q_; j++) {
    rA[j] = dA(tx, j);
  }
}

////////////////////////////////////////////////////////////////////////////////
// read A (no-trans) from global to reg.
// A is (P_ x Q_)
// 1D thread config. with (Mx1) threads
// no sync at the end of the function
template <typename T, int P_, int NB_, int Q_>
static __device__ __inline__ void read_A_trans_g2r_1D_nosync(const int tx, const int ty, const T* dA, int ldda, T* sA, int slda, T rA[Q_]) {
  int       ix  = 0;
  const int nTH = P_ * MAGMA_NONTENSOR_BASIS_NTCOL(P_);
  const int tid = ty * blockDim.x + tx;

#pragma unroll
  for (ix = 0; ix < (Q_ * P_) - nTH; ix += nTH) {
    sA[ix + tid] = dA[ix + tid];
  }

  if (tid < ((Q_ * P_) - ix)) {
    sA[ix + tid] = dA[ix + tid];
  }
  __syncthreads();

#pragma unroll
  for (int j = 0; j < Q_; j++) {
    rA[j] = sA[tx * slda + j];
  }
}

////////////////////////////////////////////////////////////////////////////////
// read B from global to shared
// B is (Q_ x NB_)
// 1D thread config. with (Mx1) threads
// no sync at the end of the function
template <typename T, int P_, int NB_, int Q_>
static __device__ __inline__ void read_B_g2s_1D_nosync(const int tx, int n, const T* dB, int lddb, T* sB, int sldb) {
  if (n != NB_) {
    for (int i = 0; i < (Q_ * n) - P_; i += P_) {
      sB[i + tx] = dB[i + tx];
    }
  } else {
#pragma unroll
    for (int i = 0; i < (Q_ * NB_) - P_; i += P_) {
      sB[i + tx] = dB[i + tx];
    }
  }

  // cleanup for B
  const int stride = MAGMA_ROUNDUP(Q_ * n - P_, P_);
  if (tx < (Q_ * n) - stride) {
    sB[stride + tx] = dB[stride + tx];
  }
}

////////////////////////////////////////////////////////////////////////////////
// multiply C = AxB using 1D threads in Mx1 config
// A (MxK)  in reg., one row per thread
// B (KxNB) in shared memory
// C in registers -- one row per thread
// no sync at the end of the function
template <typename T, int P_, int NB_, int Q_>
static __device__ __inline__ void mul_rAsBrC_1D_nosync(const int tx, const T& alpha, T rA[Q_], T* sB, int sldb, T rC[NB_]) {
  T rB[Q_] = {0};
#pragma unroll
  for (int i = 0; i < NB_; i++) {
#pragma unroll
    for (int k = 0; k < Q_; k++) {
      rB[k] = sB[i * sldb + k];
    }

    T rTmp = 0;
#pragma unroll
    for (int k = 0; k < Q_; k++) {
      rTmp += rA[k] * rB[k];
    }
    rC[i] += alpha * rTmp;
  }
}

#undef dA
#undef sA
#undef dB
#undef sB

#endif  // CEED_MAGMA_COMMON_NONTENSOR_H
