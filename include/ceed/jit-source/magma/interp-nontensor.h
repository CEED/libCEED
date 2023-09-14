// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_INTERP_NONTENSOR_H
#define CEED_MAGMA_INTERP_NONTENSOR_H

////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ __launch_bounds__(Q *MAGMA_NONTENSOR_BASIS_NTCOL(Q)) void magma_interp_nontensor_n(
    magma_trans_t transA, magma_trans_t transB, int n, const CeedScalar alpha, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb,
    const CeedScalar beta, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int bx      = blockIdx.x;
  const int id      = bx * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, NB_INTERP_N);
  const int myn     = min(NB_INTERP_N, n - id * NB_INTERP_N);

  // const bool irrblock = ( myn != NB_INTERP_N );

  dB += id * NB_INTERP_N * lddb;
  dC += id * NB_INTERP_N * lddc;

  const int   slda = P;
  const int   sldb = P;
  CeedScalar *sA   = (CeedScalar *)(shared_data);
  CeedScalar *sB   = sA;
  sB += ty * sldb * NB_INTERP_N;

  // read A using all threads
  CeedScalar rA[P] = {MAGMA_D_ZERO};
  read_A_trans_g2r_1D_nosync<CeedScalar, Q, NB_INTERP_N, P>(tx, ty, dA, ldda, sA, slda, rA);
  __syncthreads();

  // terminate threads with no work
  if (id >= nblocks) return;

  // init rC
  CeedScalar rC[NB_INTERP_N] = {MAGMA_D_ZERO};
  read_C_g2r_1D_nosync<CeedScalar, Q, NB_INTERP_N, P>(tx, myn, dC, lddc, beta, rC);

  // read B
  read_B_g2s_1D_nosync<CeedScalar, Q, NB_INTERP_N, P>(tx, myn, dB, lddb, sB, sldb);
  __syncthreads();

  mul_rAsBrC_1D_nosync<CeedScalar, Q, NB_INTERP_N, P>(tx, alpha, rA, sB, sldb, rC);
  write_C_r2g_1D_nosync<CeedScalar, Q, NB_INTERP_N, P>(tx, myn, rC, dC, lddc);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ __launch_bounds__(P *MAGMA_NONTENSOR_BASIS_NTCOL(P)) void magma_interp_nontensor_t(
    magma_trans_t transA, magma_trans_t transB, int n, const CeedScalar alpha, CeedScalar const *dA, int ldda, CeedScalar const *dB, int lddb,
    const CeedScalar beta, CeedScalar *dC, int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int bx      = blockIdx.x;
  const int id      = bx * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, NB_INTERP_T);
  const int myn     = min(NB_INTERP_T, n - id * NB_INTERP_T);
  if (id >= nblocks) return;

  dB += id * NB_INTERP_T * lddb;
  dC += id * NB_INTERP_T * lddc;

  // A is P x Q
  const int   sldb = Q;
  CeedScalar *sB   = (CeedScalar *)(shared_data);
  sB += ty * sldb * NB_INTERP_T;

  // init rC
  CeedScalar rC[NB_INTERP_T] = {MAGMA_D_ZERO};
  if (beta != MAGMA_D_ZERO) {
    read_C_g2r_1D_nosync<CeedScalar, P, NB_INTERP_T, Q>(tx, myn, dC, lddc, beta, rC);
  }

  // read A
  CeedScalar rA[Q] = {MAGMA_D_ZERO};
  read_A_notrans_g2r_1D_nosync<CeedScalar, P, NB_INTERP_T, Q>(tx, dA, ldda, NULL, 0, rA);

  // read B
  read_B_g2s_1D_nosync<CeedScalar, P, NB_INTERP_T, Q>(tx, myn, dB, lddb, sB, sldb);
  __syncthreads();

  mul_rAsBrC_1D_nosync<CeedScalar, P, NB_INTERP_T, Q>(tx, alpha, rA, sB, sldb, rC);

  write_C_r2g_1D_nosync<CeedScalar, P, NB_INTERP_T, Q>(tx, myn, rC, dC, lddc);
}

#endif  // CEED_MAGMA_INTERP_NONTENSOR_H