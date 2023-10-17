// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for Magma non-tensor basis gradient
#ifndef CEED_MAGMA_GRAD_NONTENSOR_H
#define CEED_MAGMA_GRAD_NONTENSOR_H

#include "magma-common-nontensor.h"

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_Q, MAGMA_MAXTHREADS_1D)) __global__
    void magma_grad_nontensor_n(int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, BASIS_NB_GRAD_N);
  const int myn     = min(BASIS_NB_GRAD_N, n - id * BASIS_NB_GRAD_N);

  dB += id * BASIS_NB_GRAD_N * lddb;
  dC += id * BASIS_NB_GRAD_N * lddc;

  // A is BASIS_P x BASIS_Q
  const int   slda = BASIS_P;
  const int   sldb = BASIS_P;
  CeedScalar *sA   = (CeedScalar *)shared_data;
  CeedScalar *sB   = sA + BASIS_Q * BASIS_P;
  sB += ty * sldb * BASIS_NB_GRAD_N;

  // read B once for all C's
  if (id < nblocks) {
    read_B_g2s_1D_nosync<CeedScalar, BASIS_Q, BASIS_P, BASIS_NB_GRAD_N>(tx, myn, dB, lddb, sB, sldb);
  }

  // unrolling this loop yields dramatic performance drop using hipcc, let the compiler decide (no pragma unroll)
  for (int d = 0; d < BASIS_DIM; d++) {
    // read A (BASIS_P x BASIS_Q) using all threads
    CeedScalar rA[BASIS_P] = {MAGMA_D_ZERO};
    __syncthreads();
    read_A_trans_g2r_1D_nosync<CeedScalar, BASIS_Q, BASIS_P, BASIS_NB_GRAD_N>(tx, ty, dA, ldda, sA, slda, rA);

    // init rC
    CeedScalar rC[BASIS_NB_GRAD_N] = {MAGMA_D_ZERO};
    if (id < nblocks) {
      mul_rAsBrC_1D_nosync<CeedScalar, BASIS_Q, BASIS_P, BASIS_NB_GRAD_N>(tx, rA, sB, sldb, rC);
    }

    // write C
    if (id < nblocks) {
      write_C_r2g_1D_nosync<CeedScalar, BASIS_Q, BASIS_P, BASIS_NB_GRAD_N>(tx, myn, rC, dC, lddc);
    }

    dA += BASIS_Q * BASIS_P;
    dC += BASIS_Q * n;
  }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_P, MAGMA_MAXTHREADS_1D)) __global__
    void magma_grad_nontensor_t(int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, BASIS_NB_GRAD_T);
  const int myn     = min(BASIS_NB_GRAD_T, n - id * BASIS_NB_GRAD_T);

  // terminate threads with no work
  if (id >= nblocks) return;

  dB += id * BASIS_NB_GRAD_T * lddb;
  dC += id * BASIS_NB_GRAD_T * lddc;

  // A is BASIS_P x BASIS_Q
  const int   sldb = BASIS_Q;
  CeedScalar *sB   = (CeedScalar *)shared_data;
  sB += ty * sldb * BASIS_NB_GRAD_T;

  // init rA, rC
  CeedScalar rA[BASIS_Q]         = {MAGMA_D_ZERO};
  CeedScalar rC[BASIS_NB_GRAD_T] = {MAGMA_D_ZERO};

  // unrolling this loop yields dramatic performance drop using hipcc, let the compiler decide (no pragma unroll)
  for (int d = 0; d < BASIS_DIM; d++) {
    // read A
    read_A_notrans_g2r_1D_nosync<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_GRAD_T>(tx, dA, ldda, NULL, 0, rA);

    // read B
    __syncthreads();
    read_B_g2s_1D_nosync<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_GRAD_T>(tx, myn, dB, lddb, sB, sldb);
    __syncthreads();

    addmul_rAsBrC_1D_nosync<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_GRAD_T>(tx, rA, sB, sldb, rC);

    dA += BASIS_P * BASIS_Q;
    dB += BASIS_Q * n;
  }

  // write C
  write_C_r2g_1D_nosync<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_GRAD_T>(tx, myn, rC, dC, lddc);
}

#endif  // CEED_MAGMA_GRAD_NONTENSOR_H
