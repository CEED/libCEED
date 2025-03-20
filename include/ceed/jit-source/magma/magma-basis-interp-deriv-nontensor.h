// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA non-tensor basis interpolation
#include "magma-common-nontensor.h"

////////////////////////////////////////////////////////////////////////////////
template <typename T, int Q_COMP, int P, int Q, int NB>
static __device__ __inline__ void magma_basis_nontensor_device_n(const int n, CeedScalar const *dA, CeedScalar const *dB, CeedScalar *dC,
                                                                 CeedScalar *shared_data) {
  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = (n + NB - 1) / NB;
  const int myn     = min(NB, n - id * NB);

  dB += id * P * NB;
  dC += id * Q * NB;

  // A is P x Q
  CeedScalar *sB = shared_data + ty * P * NB;
  CeedScalar *sA = shared_data + blockDim.y * P * NB;

  // read B once for all C's
  if (id < nblocks) {
    read_B_g2s_1D_nosync<CeedScalar, Q, P, NB>(tx, myn, dB, sB);
  }

  // unrolling this loop yields dramatic performance drop using hipcc, so let the compiler decide (no pragma unroll)
  for (int d = 0; d < Q_COMP; d++) {
    // read A using all threads
    CeedScalar rA[P];
    read_A_trans_g2r_1D_nosync<CeedScalar, Q, P, MAGMA_BASIS_NTCOL(Q, MAGMA_MAXTHREADS_1D)>(tx, ty, dA, sA, rA);

    CeedScalar rC[NB];
    mul_rAsBrC_1D_nosync<CeedScalar, Q, P, NB>(rA, sB, rC);

    // write C
    if (id < nblocks) {
      write_C_r2g_1D_nosync<CeedScalar, Q, P, NB>(tx, myn, rC, dC);
    }

    dA += Q * P;
    dC += Q * n;

    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////
template <typename T, int Q_COMP, int P, int Q, int NB>
static __device__ __inline__ void magma_basis_nontensor_device_t(const int n, CeedScalar const *dA, CeedScalar const *dB, CeedScalar *dC,
                                                                 CeedScalar *shared_data) {
  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = (n + NB - 1) / NB;
  const int myn     = min(NB, n - id * NB);

  dB += id * Q * NB;
  dC += id * P * NB;

  // A is P x Q
  CeedScalar *sA = shared_data;
  CeedScalar *sB = shared_data + ty * Q * NB;

  CeedScalar rC[NB] = {0.0};

  // unrolling this loop yields dramatic performance drop using hipcc, so let the compiler decide (no pragma unroll)
  for (int d = 0; d < Q_COMP; d++) {
    // read A using all threads
    CeedScalar rA[Q];
    read_A_notrans_g2r_1D_nosync<CeedScalar, P, Q, MAGMA_BASIS_NTCOL(P, MAGMA_MAXTHREADS_1D)>(tx, ty, dA, sA, rA);
    __syncthreads();

    // read B
    if (id < nblocks) {
      read_B_g2s_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, dB, sB);
    }
    __syncthreads();

    addmul_rAsBrC_1D_nosync<CeedScalar, P, Q, NB>(rA, sB, rC);

    dA += P * Q;
    dB += Q * n;

    __syncthreads();
  }

  // write C
  if (id < nblocks) {
    write_C_r2g_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, rC, dC);
  }
}

////////////////////////////////////////////////////////////////////////////////
template <typename T, int Q_COMP, int P, int Q, int NB>
static __device__ __inline__ void magma_basis_nontensor_device_ta(const int n, const CeedScalar *dA, const CeedScalar *dB, CeedScalar *dC,
                                                                  CeedScalar *shared_data) {
  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = (n + NB - 1) / NB;
  const int myn     = min(NB, n - id * NB);

  dB += id * Q * NB;
  dC += id * P * NB;

  // A is P x Q
  CeedScalar *sA = shared_data;
  CeedScalar *sB = shared_data + ty * Q * NB;

  CeedScalar rC[NB] = {0.0};

  // unrolling this loop yields dramatic performance drop using hipcc, so let the compiler decide (no pragma unroll)
  for (int d = 0; d < Q_COMP; d++) {
    // read A using all threads
    CeedScalar rA[Q];
    read_A_notrans_g2r_1D_nosync<CeedScalar, P, Q, MAGMA_BASIS_NTCOL(P, MAGMA_MAXTHREADS_1D)>(tx, ty, dA, sA, rA);
    __syncthreads();

    // read B
    if (id < nblocks) {
      read_B_g2s_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, dB, sB);
    }
    __syncthreads();

    addmul_rAsBrC_1D_nosync<CeedScalar, P, Q, NB>(rA, sB, rC);

    dA += P * Q;
    dB += Q * n;

    __syncthreads();
  }

  // sum into C
  if (id < nblocks) {
    sum_C_r2g_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, rC, dC);
  }
}

////////////////////////////////////////////////////////////////////////////////
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void magma_basis_nontensor_device_n1(const int n, CeedScalar const *dA, CeedScalar const *dB, CeedScalar *dC,
                                                                  CeedScalar *shared_data) {
  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = (n + NB - 1) / NB;
  const int myn     = min(NB, n - id * NB);

  dB += id * P * NB;
  dC += id * Q * NB;

  // A is P x Q
  CeedScalar *sA = shared_data;
  CeedScalar *sB = shared_data + ty * P * NB;

  // read A using all threads
  CeedScalar rA[P];
  read_A_trans_g2r_1D_nosync<CeedScalar, Q, P, MAGMA_BASIS_NTCOL(Q, MAGMA_MAXTHREADS_1D)>(tx, ty, dA, sA, rA);
  __syncthreads();

  // terminate threads with no work
  if (id >= nblocks) return;

  // read B
  read_B_g2s_1D_nosync<CeedScalar, Q, P, NB>(tx, myn, dB, sB);
  __syncthreads();

  CeedScalar rC[NB];
  mul_rAsBrC_1D_nosync<CeedScalar, Q, P, NB>(rA, sB, rC);

  // write C
  write_C_r2g_1D_nosync<CeedScalar, Q, P, NB>(tx, myn, rC, dC);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void magma_basis_nontensor_device_t1(const int n, CeedScalar const *dA, CeedScalar const *dB, CeedScalar *dC,
                                                                  CeedScalar *shared_data) {
  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = (n + NB - 1) / NB;
  const int myn     = min(NB, n - id * NB);

  dB += id * Q * NB;
  dC += id * P * NB;

  // A is P x Q
  CeedScalar *sA = shared_data;
  CeedScalar *sB = shared_data + ty * Q * NB;

  // read A using all threads
  CeedScalar rA[Q];
  read_A_notrans_g2r_1D_nosync<CeedScalar, P, Q, MAGMA_BASIS_NTCOL(P, MAGMA_MAXTHREADS_1D)>(tx, ty, dA, sA, rA);
  __syncthreads();

  // terminate threads with no work
  if (id >= nblocks) return;

  // read B
  read_B_g2s_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, dB, sB);
  __syncthreads();

  CeedScalar rC[NB];
  mul_rAsBrC_1D_nosync<CeedScalar, P, Q, NB>(rA, sB, rC);

  // write C
  write_C_r2g_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, rC, dC);
}

////////////////////////////////////////////////////////////////////////////////
template <typename T, int P, int Q, int NB>
static __device__ __inline__ void magma_basis_nontensor_device_ta1(const int n, CeedScalar const *dA, CeedScalar const *dB, CeedScalar *dC,
                                                                   CeedScalar *shared_data) {
  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = (n + NB - 1) / NB;
  const int myn     = min(NB, n - id * NB);

  dB += id * Q * NB;
  dC += id * P * NB;

  // A is P x Q
  CeedScalar *sA = shared_data;
  CeedScalar *sB = shared_data + ty * Q * NB;

  // read A using all threads
  CeedScalar rA[Q];
  read_A_notrans_g2r_1D_nosync<CeedScalar, P, Q, MAGMA_BASIS_NTCOL(P, MAGMA_MAXTHREADS_1D)>(tx, ty, dA, sA, rA);
  __syncthreads();

  // terminate threads with no work
  if (id >= nblocks) return;

  // read B
  read_B_g2s_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, dB, sB);
  __syncthreads();

  CeedScalar rC[NB];
  mul_rAsBrC_1D_nosync<CeedScalar, P, Q, NB>(rA, sB, rC);

  // sum into C
  sum_C_r2g_1D_nosync<CeedScalar, P, Q, NB>(tx, myn, rC, dC);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_Q, MAGMA_MAXTHREADS_1D)) __global__
    void magma_interp_nontensor_n(const int n, CeedScalar const *__restrict__ dA, CeedScalar const *__restrict__ dB, CeedScalar *__restrict__ dC) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

#if BASIS_Q_COMP_INTERP == 1
  magma_basis_nontensor_device_n1<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_INTERP_N>(n, dA, dB, dC, (CeedScalar *)shared_data);
#else
  magma_basis_nontensor_device_n<CeedScalar, BASIS_Q_COMP_INTERP, BASIS_P, BASIS_Q, BASIS_NB_INTERP_N>(n, dA, dB, dC, (CeedScalar *)shared_data);
#endif
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_P, MAGMA_MAXTHREADS_1D)) __global__
    void magma_interp_nontensor_t(const int n, CeedScalar const *__restrict__ dA, CeedScalar const *__restrict__ dB, CeedScalar *__restrict__ dC) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

#if BASIS_Q_COMP_INTERP == 1
  magma_basis_nontensor_device_t1<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_INTERP_T>(n, dA, dB, dC, (CeedScalar *)shared_data);
#else
  magma_basis_nontensor_device_t<CeedScalar, BASIS_Q_COMP_INTERP, BASIS_P, BASIS_Q, BASIS_NB_INTERP_T>(n, dA, dB, dC, (CeedScalar *)shared_data);
#endif
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_P, MAGMA_MAXTHREADS_1D)) __global__
    void magma_interp_nontensor_ta(const int n, CeedScalar const *__restrict__ dA, CeedScalar const *__restrict__ dB, CeedScalar *__restrict__ dC) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

#if BASIS_Q_COMP_INTERP == 1
  magma_basis_nontensor_device_ta1<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_INTERP_T>(n, dA, dB, dC, (CeedScalar *)shared_data);
#else
  magma_basis_nontensor_device_ta<CeedScalar, BASIS_Q_COMP_INTERP, BASIS_P, BASIS_Q, BASIS_NB_INTERP_T>(n, dA, dB, dC, (CeedScalar *)shared_data);
#endif
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_Q, MAGMA_MAXTHREADS_1D)) __global__
    void magma_deriv_nontensor_n(const int n, CeedScalar const *__restrict__ dA, CeedScalar const *__restrict__ dB, CeedScalar *__restrict__ dC) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

#if BASIS_Q_COMP_DERIV == 1
  magma_basis_nontensor_device_n1<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_DERIV_N>(n, dA, dB, dC, (CeedScalar *)shared_data);
#else
  magma_basis_nontensor_device_n<CeedScalar, BASIS_Q_COMP_DERIV, BASIS_P, BASIS_Q, BASIS_NB_DERIV_N>(n, dA, dB, dC, (CeedScalar *)shared_data);
#endif
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_P, MAGMA_MAXTHREADS_1D)) __global__
    void magma_deriv_nontensor_t(const int n, CeedScalar const *__restrict__ dA, CeedScalar const *__restrict__ dB, CeedScalar *__restrict__ dC) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

#if BASIS_Q_COMP_DERIV == 1
  magma_basis_nontensor_device_t1<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_DERIV_T>(n, dA, dB, dC, (CeedScalar *)shared_data);
#else
  magma_basis_nontensor_device_t<CeedScalar, BASIS_Q_COMP_DERIV, BASIS_P, BASIS_Q, BASIS_NB_DERIV_T>(n, dA, dB, dC, (CeedScalar *)shared_data);
#endif
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_P, MAGMA_MAXTHREADS_1D)) __global__
    void magma_deriv_nontensor_ta(const int n, CeedScalar const *__restrict__ dA, CeedScalar const *__restrict__ dB, CeedScalar *__restrict__ dC) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

#if BASIS_Q_COMP_DERIV == 1
  magma_basis_nontensor_device_ta1<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_DERIV_T>(n, dA, dB, dC, (CeedScalar *)shared_data);
#else
  magma_basis_nontensor_device_ta<CeedScalar, BASIS_Q_COMP_DERIV, BASIS_P, BASIS_Q, BASIS_NB_DERIV_T>(n, dA, dB, dC, (CeedScalar *)shared_data);
#endif
}
