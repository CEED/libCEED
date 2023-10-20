// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA tensor basis gradient in 3D
#ifndef CEED_MAGMA_BASIS_GRAD_3D_H
#define CEED_MAGMA_BASIS_GRAD_3D_H

#include "magma-common-tensor.h"

// macros to abstract access of shared memory and reg. file
#define sT(i, j) sT[(j)*P + (i)]
#define sTmp(i, j, ldw) sTmp[(j) * (ldw) + (i)]
#define sTmp2(i, j, ldw) sTmp2[(j) * (ldw) + (i)]

////////////////////////////////////////////////////////////////////////////////
// grad basis action (3D)
// This function is called three times at a higher level for 3D
// DIM_U   -- for the size of rU[DIM_U * NUM_COMP * MAX_P_Q]
// DIM_V   -- for the size of rV[DIM_V * NUM_COMP * MAX_P_Q]
// i_DIM   -- the index of the outermost loop over dimensions in grad
// i_DIM_U -- which dim index of rU is accessed (always 0 for notrans, 0, 1, or 2 for trans)
// i_DIM_V -- which dim index of rV is accessed (0, 1, or 2 for notrans, always 0 for trans)
// the scalar beta is used to specify whether to accumulate to rV, or overwrite it
template <typename T, int DIM_U, int DIM_V, int NUM_COMP, int P, int Q, int rU_SIZE, int rV_SIZE, int i_DIM, int i_DIM_U, int i_DIM_V>
static __device__ __inline__ void magma_grad_3d_device(const T *sTinterp, const T *sTgrad, T rU[DIM_U][NUM_COMP][rU_SIZE],
                                                       T rV[DIM_V][NUM_COMP][rV_SIZE], T beta, const int tx, T rTmp, T *swork) {
  // Assumptions
  // 0. This device routine applies grad for one dim only (i_DIM), so it should be thrice for 3D
  // 1. 1D threads of size max(P,Q)^2
  // 2. input:  rU[DIM_U x NUM_COMP x rU_SIZE] in registers (per thread)
  // 3. output: rV[DIM_V x NUM_COMP x rV_SIZE] in registers (per thread)
  // 4. Three products per each (dim,component) pair
  //  4.1 Batch P^2 of (1xP) matrices times (PxQ) matrix => Batch P^2 of (1xQ) matrices
  //  4.2 Batch P   of (QxP) matrices times (PxQ) matrix => Batch P   of (QxQ) matrices
  //  4.3 Batch 1   of (Q^2xP_) matrix times (PxQ) matrix => (Q^2xQ_) matrix
  // 6. Each thread computes one row of the output of each product
  // 7. Sync is recommended before and after the call

  T *sW1 = swork;
  T *sW2 = sW1 + P * P * Q;
  for (int comp = 0; comp < NUM_COMP; comp++) {
    // Batch P^2 of (1xP) matrices [reg] times (PxQ) matrix [shmem] => Batch P^2 of (1xQ) matrices [shmem]
    if (tx < (P * P)) {
      const int batchid = tx;
      const int sld     = 1;
      const T  *sT      = (i_DIM == 0) ? sTgrad : sTinterp;
      T        *sTmp    = sW1 + batchid * (1 * Q);
      for (int j = 0; j < Q; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P; i++) {
          rTmp += rU[i_DIM_U][comp][i] * sT(i, j);
        }
        sTmp(0, j, sld) = rTmp;
      }
    }  // end of: if (tx < P*P)
    __syncthreads();

    // Batch P of (QxP) matrices [shmem] times (PxQ) matrix [shmem] => Batch P of (QxQ) matrices [reg]
    if (tx < (P * Q)) {
      const int batchid = tx / Q;
      const int tx_     = tx % Q;
      const int sld     = Q;
      const T  *sT      = (i_DIM == 1) ? sTgrad : sTinterp;
      T        *sTmp    = sW1 + batchid * (Q * P);  // sTmp is input
      T        *sTmp2   = sW2 + batchid * (Q * Q);  // sTmp2 is output
      for (int j = 0; j < Q; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P; i++) {
          rTmp += sTmp(tx_, i, sld) * sT(i, j);
        }
        sTmp2(tx_, j, sld) = rTmp;
      }
    }
    __syncthreads();

    // Batch 1 of (Q^2xP_) matrices [shmem] times (PxQ) matrix [shmem] => Batch 1 of (Q^2xQ_) matrices [reg]
    if (tx < (Q * Q)) {
      // No need to declare batchid = (tx  / Q^2) = always zero
      // No need to declare tx_     = (tx_ % Q^2) = always tx
      const int sld  = Q * Q;
      const T  *sT   = (i_DIM == 2) ? sTgrad : sTinterp;
      T        *sTmp = sW2;  // sTmp is input
      for (int j = 0; j < Q; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P; i++) {
          rTmp += sTmp(tx, i, sld) * sT(i, j);
        }
        rV[i_DIM_V][comp][j] *= beta;
        rV[i_DIM_V][comp][j] += rTmp;
      }
    }
    __syncthreads();
  }  // loop over NUM_COMP
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_MAX_P_Q *BASIS_MAX_P_Q, MAGMA_MAXTHREADS_3D)) __global__
    void magma_gradn_3d_kernel(const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, const CeedScalar *dU, const int estrdU, const int cstrdU,
                               const int dstrdU, CeedScalar *dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaNoTrans;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][BASIS_NUM_COMP][BASIS_P] = {0.0};  // here DIM_U = 1, but might be different for a fused operator
  CeedScalar rV[1][BASIS_NUM_COMP][BASIS_Q] = {0.0};  // here DIM_V = 1, but might be different for a fused operator
  CeedScalar rTmp                           = 0.0;

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sTinterp = (CeedScalar *)shared_data;
  CeedScalar *sTgrad   = sTinterp + BASIS_P * BASIS_Q;
  CeedScalar *sTmp     = sTgrad + BASIS_P * BASIS_Q;
  sTmp += ty * (max(BASIS_P * BASIS_P * BASIS_P, (BASIS_P * BASIS_P * BASIS_Q) + (BASIS_P * BASIS_Q * BASIS_Q)));

  // read T
  if (ty == 0) {
    read_T_notrans_gm2sm<BASIS_P, BASIS_Q>(tx, dinterp1d, sTinterp);
    read_T_notrans_gm2sm<BASIS_P, BASIS_Q>(tx, dgrad1d, sTgrad);
  }
  __syncthreads();

  // No need to read V ( required only in transposed grad )
  const CeedScalar beta = 0.0;

  /* read U (idim = 0 for dU, i_DIM = 0 for rU) --
     there is a sync at the end of this function */
  read_U_3d<CeedScalar, BASIS_P, 1, BASIS_NUM_COMP, BASIS_P, 0>(dU + (0 * dstrdU), cstrdU, rU, sTmp, tx);

  /* first call (i_DIM = 0, i_DIM_U = 0, i_DIM_V = 0) --
     output from rV[0][][] into dV (idim = 0) */
  magma_grad_3d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_P, BASIS_Q, BASIS_P, BASIS_Q, 0, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */
  write_V_3d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dV + (0 * dstrdV), cstrdV, rV, tx);

  /* second call (i_DIM = 1, i_DIM_U = 0, i_DIM_V = 0) --
     output from rV[0][][] into dV (idim = 1) */
  magma_grad_3d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_P, BASIS_Q, BASIS_P, BASIS_Q, 1, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */
  write_V_3d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dV + (1 * dstrdV), cstrdV, rV, tx);

  /* third call (i_DIM = 2, i_DIM_U = 0, i_DIM_V = 0) --
     output from rV[0][][] into dV (idim = 2) */
  magma_grad_3d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_P, BASIS_Q, BASIS_P, BASIS_Q, 2, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */
  write_V_3d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dV + (2 * dstrdV), cstrdV, rV, tx);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_MAX_P_Q *BASIS_MAX_P_Q, MAGMA_MAXTHREADS_3D)) __global__
    void magma_gradt_3d_kernel(const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, const CeedScalar *dU, const int estrdU, const int cstrdU,
                               const int dstrdU, CeedScalar *dV, const int estrdV, const int cstrdV, const int dstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaTrans;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][BASIS_NUM_COMP][BASIS_Q] = {0.0};  // here DIM_U = 1, but might be different for a fused operator
  CeedScalar rV[1][BASIS_NUM_COMP][BASIS_P] = {0.0};  // here DIM_V = 1, but might be different for a fused operator
  CeedScalar rTmp                           = 0.0;

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sTinterp = (CeedScalar *)shared_data;
  CeedScalar *sTgrad   = sTinterp + BASIS_Q * BASIS_P;
  CeedScalar *sTmp     = sTgrad + BASIS_Q * BASIS_P;
  sTmp += ty * (max(BASIS_Q * BASIS_Q * BASIS_Q, (BASIS_Q * BASIS_Q * BASIS_P) + (BASIS_Q * BASIS_P * BASIS_P)));

  // read T
  if (ty == 0) {
    read_T_trans_gm2sm<BASIS_Q, BASIS_P>(tx, dinterp1d, sTinterp);
    read_T_trans_gm2sm<BASIS_Q, BASIS_P>(tx, dgrad1d, sTgrad);
  }
  __syncthreads();

  // read V (since this is transposed mode)
  const CeedScalar beta = 1.0;
  read_V_3d<CeedScalar, BASIS_P, 1, BASIS_NUM_COMP, BASIS_P, 0>(dV + (0 * dstrdV), cstrdV, rV, tx);

  /* read U (idim = 0 for dU, i_DIM = 0 for rU) --
     there is a sync at the end of this function */
  read_U_3d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dU + (0 * dstrdU), cstrdU, rU, sTmp, tx);
  /* then first call (i_DIM = 0, i_DIM_U = 0, i_DIM_V = 0) */
  magma_grad_3d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_Q, BASIS_P, BASIS_Q, BASIS_P, 0, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */

  /* read U (idim = 1 for dU, i_DIM = 0 for rU) --
     there is a sync at the end of this function */
  read_U_3d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dU + (1 * dstrdU), cstrdU, rU, sTmp, tx);
  /* then second call (i_DIM = 1, i_DIM_U = 0, i_DIM_V = 0) */
  magma_grad_3d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_Q, BASIS_P, BASIS_Q, BASIS_P, 1, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */

  /* read U (idim = 2 for dU, i_DIM = 0 for rU) --
     there is a sync at the end of this function */
  read_U_3d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dU + (2 * dstrdU), cstrdU, rU, sTmp, tx);
  /* then third call (i_DIM = 2, i_DIM_U = 0, i_DIM_V = 0) */
  magma_grad_3d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_Q, BASIS_P, BASIS_Q, BASIS_P, 2, 0, 0>(sTinterp, sTgrad, rU, rV, beta, tx, rTmp, sTmp);
  /* there is a sync at the end of magma_grad_3d_device */

  // write V
  write_V_3d<CeedScalar, BASIS_P, 1, BASIS_NUM_COMP, BASIS_P, 0>(dV + (0 * dstrdV), cstrdV, rV, tx);
}

#endif  // CEED_MAGMA_BASIS_GRAD_3D_H
