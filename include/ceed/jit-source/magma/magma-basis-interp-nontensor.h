// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA non-tensor basis interpolation
#ifndef CEED_MAGMA_INTERP_NONTENSOR_H
#define CEED_MAGMA_INTERP_NONTENSOR_H

#include "magma-common-nontensor.h"

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_Q, MAGMA_MAXTHREADS_1D)) __global__
    void magma_interp_nontensor_n(int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, BASIS_NB_INTERP_N);
  const int myn     = min(BASIS_NB_INTERP_N, n - id * BASIS_NB_INTERP_N);

  dB += id * BASIS_NB_INTERP_N * lddb;
  dC += id * BASIS_NB_INTERP_N * lddc;

  // A is BASIS_P x BASIS_Q
  const int   slda = BASIS_P;
  const int   sldb = BASIS_P;
  CeedScalar *sA   = (CeedScalar *)shared_data;
  CeedScalar *sB   = sA;
  sB += ty * sldb * BASIS_NB_INTERP_N;

  // read A using all threads
  CeedScalar rA[BASIS_P] = {MAGMA_D_ZERO};
  read_A_trans_g2r_1D_nosync<CeedScalar, BASIS_Q, BASIS_P, BASIS_NB_INTERP_N>(tx, ty, dA, ldda, sA, slda, rA);
  __syncthreads();

  // terminate threads with no work
  if (id >= nblocks) return;

  // init rC
  CeedScalar rC[BASIS_NB_INTERP_N] = {MAGMA_D_ZERO};

  // read B
  read_B_g2s_1D_nosync<CeedScalar, BASIS_Q, BASIS_P, BASIS_NB_INTERP_N>(tx, myn, dB, lddb, sB, sldb);
  __syncthreads();

  mul_rAsBrC_1D_nosync<CeedScalar, BASIS_Q, BASIS_P, BASIS_NB_INTERP_N>(tx, rA, sB, sldb, rC);

  // write C
  write_C_r2g_1D_nosync<CeedScalar, BASIS_Q, BASIS_P, BASIS_NB_INTERP_N>(tx, myn, rC, dC, lddc);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_P, MAGMA_MAXTHREADS_1D)) __global__
    void magma_interp_nontensor_t(int n, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int id      = blockIdx.x * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, BASIS_NB_INTERP_T);
  const int myn     = min(BASIS_NB_INTERP_T, n - id * BASIS_NB_INTERP_T);

  // terminate threads with no work
  if (id >= nblocks) return;

  dB += id * BASIS_NB_INTERP_T * lddb;
  dC += id * BASIS_NB_INTERP_T * lddc;

  // A is BASIS_P x BASIS_Q
  const int   sldb = BASIS_Q;
  CeedScalar *sB   = (CeedScalar *)shared_data;
  sB += ty * sldb * BASIS_NB_INTERP_T;

  // init rC
  CeedScalar rC[BASIS_NB_INTERP_T] = {MAGMA_D_ZERO};

  // read A
  CeedScalar rA[BASIS_Q] = {MAGMA_D_ZERO};
  read_A_notrans_g2r_1D_nosync<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_INTERP_T>(tx, dA, ldda, NULL, 0, rA);

  // read B
  read_B_g2s_1D_nosync<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_INTERP_T>(tx, myn, dB, lddb, sB, sldb);
  __syncthreads();

  mul_rAsBrC_1D_nosync<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_INTERP_T>(tx, rA, sB, sldb, rC);

  // write C
  write_C_r2g_1D_nosync<CeedScalar, BASIS_P, BASIS_Q, BASIS_NB_INTERP_T>(tx, myn, rC, dC, lddc);
}

#endif  // CEED_MAGMA_INTERP_NONTENSOR_H
