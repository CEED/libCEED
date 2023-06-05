// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_GRAD_NONTENSOR_H
#define CEED_MAGMA_GRAD_NONTENSOR_H

////////////////////////////////////////////////////////////////////////////////
// Different A's and C's, same B
extern "C" __global__ __launch_bounds__(Q* MAGMA_NONTENSOR_BASIS_NTCOL(Q)) void magma_grad_nontensor_n(magma_trans_t transA, magma_trans_t transB,
                                                                                                       int n, CeedScalar const* dA, int ldda,
                                                                                                       CeedScalar const* dB, int lddb, CeedScalar* dC,
                                                                                                       int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int bx      = blockIdx.x;
  const int id      = bx * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, NB_GRAD_N);
  const int myn     = min(NB_GRAD_N, n - id * NB_GRAD_N);

  const double alpha = MAGMA_D_ONE;

  dB += id * NB_GRAD_N * lddb;
  dC += id * NB_GRAD_N * lddc;

  // A is P x Q
  const int   slda = P;
  const int   sldb = P;
  CeedScalar* sA   = (CeedScalar*)(shared_data);
  CeedScalar* sB   = sA + Q * P;
  sB += ty * sldb * NB_GRAD_N;

  // read B once for all C's
  if (id < nblocks) {
    read_B_g2s_1D_nosync<CeedScalar, Q, NB_GRAD_N, P>(tx, myn, dB, lddb, sB, sldb);
  }
  __syncthreads();

  // unrolling this loop yields dramatic performance drop using hipcc
  // let the compiler decide (no pragma unroll)
  for (int idim = 0; idim < DIM; idim++) {
    // read A (P x Q) using all threads
    CeedScalar rA[P] = {MAGMA_D_ZERO};
    read_A_trans_g2r_1D_nosync<CeedScalar, Q, NB_GRAD_N, P>(tx, ty, dA, ldda, sA, slda, rA);

    __syncthreads();

    // init rC
    CeedScalar rC[NB_GRAD_N] = {MAGMA_D_ZERO};
    if (id < nblocks) {
      mul_rAsBrC_1D_nosync<CeedScalar, Q, NB_GRAD_N, P>(tx, alpha, rA, sB, sldb, rC);
    }
    __syncthreads();

    if (id < nblocks) {
      write_C_r2g_1D_nosync<CeedScalar, Q, NB_GRAD_N, P>(tx, myn, rC, dC, lddc);
    }

    dA += Q * P;
    dC += Q * n;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Different A's and B's, same C
extern "C" __global__ __launch_bounds__(P* MAGMA_NONTENSOR_BASIS_NTCOL(P)) void magma_grad_nontensor_t(magma_trans_t transA, magma_trans_t transB,
                                                                                                       int n, CeedScalar const* dA, int ldda,
                                                                                                       CeedScalar const* dB, int lddb, CeedScalar* dC,
                                                                                                       int lddc) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int bx      = blockIdx.x;
  const int id      = bx * blockDim.y + ty;
  const int nblocks = MAGMA_CEILDIV(n, NB_GRAD_T);
  const int myn     = min(NB_GRAD_T, n - id * NB_GRAD_T);
  if (id >= nblocks) return;

  dB += id * NB_GRAD_T * lddb;
  dC += id * NB_GRAD_T * lddc;

  const double alpha = MAGMA_D_ONE;

  // A is P x Q
  const int   sldb = Q;
  CeedScalar* sB   = (CeedScalar*)(shared_data);
  sB += ty * sldb * NB_GRAD_T;

  // init rC
  CeedScalar rC[NB_GRAD_T] = {MAGMA_D_ZERO};

  CeedScalar rA[Q] = {MAGMA_D_ZERO};

  // unrolling this loop yields dramatic performance drop using hipcc
  // let the compiler decide (no pragma unroll)
  for (int idim = 0; idim < DIM; idim++) {
    __syncthreads();
    // read A
    read_A_notrans_g2r_1D_nosync<CeedScalar, P, NB_GRAD_T, Q>(tx, dA, ldda, NULL, 0, rA);

    // read B
    read_B_g2s_1D_nosync<CeedScalar, P, NB_GRAD_T, Q>(tx, myn, dB, lddb, sB, sldb);
    __syncthreads();

    mul_rAsBrC_1D_nosync<CeedScalar, P, NB_GRAD_T, Q>(tx, alpha, rA, sB, sldb, rC);

    // advance A and B
    dA += P * Q;
    dB += Q * n;
  }
  write_C_r2g_1D_nosync<CeedScalar, P, NB_GRAD_T, Q>(tx, myn, rC, dC, lddc);
}

#endif  // CEED_MAGMA_GRAD_NONTENSOR_H
