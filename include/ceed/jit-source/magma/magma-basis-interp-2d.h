// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA tensor basis interpolation in 1D
#ifndef CEED_MAGMA_BASIS_INTERP_2D_H
#define CEED_MAGMA_BASIS_INTERP_2D_H

#include "magma-common-tensor.h"

// macros to abstract access of shared memory and reg. file
#define sT(i, j) sT[(j)*P + (i)]
#define sTmp(i, j, ldw) sTmp[(j) * (ldw) + (i)]

//////////////////////////////////////////////////////////////////////////////////////////
// interp basis action (2D)
template <typename T, int DIM_U, int DIM_V, int NUM_COMP, int P, int Q, int rU_SIZE, int rV_SIZE>
static __device__ __inline__ void magma_interp_2d_device(const T *sT, magma_trans_t transT, T rU[DIM_U][NUM_COMP][rU_SIZE],
                                                         T rV[DIM_V][NUM_COMP][rV_SIZE], const int tx, T rTmp, T *swork) {
  // Assumptions
  // 1. 1D threads of size max(P,Q)
  // 2. input:  rU[DIM_U x NUM_COMP x rU_SIZE] in registers (per thread)
  // 3. output: rV[DIM_V x NUM_COMP x rV_SIZE] in registers (per thread)
  // 4. Two products per component
  //  4.1 Batch P of (1xP) matrices times (PxQ) matrix => Batch P of (1xQ) matrices
  //  4.2 Batch 1 of (QxP) matrix   times (PxQ) matrix => (QxQ) matrix
  // 5. Each thread computes one row of the output of each product
  // 6. Sync is recommended before and after the call

  for (int comp = 0; comp < NUM_COMP; comp++) {
    // 1st product -- Batch P of (1xP) matrices [reg] x (PxQ) [shmem] => Batch P of (1xQ) matrices
    // the batch output P x (1xQ) is written on the fly to shmem
    if (tx < P) {
      const int batchid = tx;
      const int sld     = 1;
      T        *sTmp    = swork + batchid * (1 * Q);
      for (int j = 0; j < Q; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P; i++) {
          rTmp += rU[0][comp][i] * sT(i, j);
        }
        sTmp(0, j, sld) = rTmp;
      }
    }  // end of: if (tx < P)
    __syncthreads();

    // 2nd product -- Batch 1 of a (QxP) matrix [shmem] x (PxQ) [shmem] => (QxQ) matrix [reg]
    if (tx < Q) {
      const int batchid = 0;
      const int sld     = Q;
      T        *sTmp    = swork + batchid * (Q * P);
      for (int j = 0; j < Q; j++) {
        rTmp = 0.0;
        for (int i = 0; i < P; i++) {
          rTmp += sTmp(tx, i, sld) * sT(i, j);
        }
        rV[0][comp][j] += rTmp;
      }
    }
    __syncthreads();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_MAX_P_Q, MAGMA_MAXTHREADS_2D)) __global__
    void magma_interpn_2d_kernel(const CeedScalar *dT, const CeedScalar *dU, const int estrdU, const int cstrdU, CeedScalar *dV, const int estrdV,
                                 const int cstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaNoTrans;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][BASIS_NUM_COMP][BASIS_P] = {0.0};  // for a non-fused operator BASIS_DIM is always 1
  CeedScalar rV[1][BASIS_NUM_COMP][BASIS_Q] = {0.0};  // for a non-fused operator BASIS_DIM is always 1
  CeedScalar rTmp                           = 0.0;

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sT   = (CeedScalar *)shared_data;
  CeedScalar *sTmp = sT + BASIS_P * BASIS_Q;
  sTmp += ty * (BASIS_P * BASIS_MAX_P_Q);

  // read T
  if (ty == 0) {
    dread_T_gm2sm<BASIS_P, BASIS_Q>(tx, transT, dT, sT);
  }

  // read U -- there is a sync at the end of this function
  readU_2d<CeedScalar, BASIS_P, 1, BASIS_NUM_COMP, BASIS_P, 0>(dU, cstrdU, rU, sTmp, tx);

  // no sync needed here -- readU_2d already syncs at the end
  magma_interp_2d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_P, BASIS_Q, BASIS_P, BASIS_Q>(sT, transT, rU, rV, tx, rTmp, sTmp);
  __syncthreads();

  // write V
  writeV_2d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dV, cstrdV, rV, tx);
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_MAX_P_Q, MAGMA_MAXTHREADS_2D)) __global__
    void magma_interpt_2d_kernel(const CeedScalar *dT, const CeedScalar *dU, const int estrdU, const int cstrdU, CeedScalar *dV, const int estrdV,
                                 const int cstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaTrans;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][BASIS_NUM_COMP][BASIS_Q] = {0.0};  // for a non-fused operator BASIS_DIM is always 1
  CeedScalar rV[1][BASIS_NUM_COMP][BASIS_P] = {0.0};  // for a non-fused operator BASIS_DIM is always 1
  CeedScalar rTmp                           = 0.0;

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sT   = (CeedScalar *)shared_data;
  CeedScalar *sTmp = sT + BASIS_Q * BASIS_P;
  sTmp += ty * (BASIS_Q * BASIS_MAX_P_Q);

  // read T
  if (ty == 0) {
    dread_T_gm2sm<BASIS_Q, BASIS_P>(tx, transT, dT, sT);
  }

  // read V
  readV_2d<CeedScalar, BASIS_P, 1, BASIS_NUM_COMP, BASIS_P, 0>(dV, cstrdV, rV, tx);

  // read U -- there is a sync at the end of this function
  readU_2d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dU, cstrdU, rU, sTmp, tx);

  // no sync needed here -- readU_2d already syncs at the end
  magma_interp_2d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_Q, BASIS_P, BASIS_Q, BASIS_P>(sT, transT, rU, rV, tx, rTmp, sTmp);
  __syncthreads();

  // write V
  writeV_2d<CeedScalar, BASIS_P, 1, BASIS_NUM_COMP, BASIS_P, 0>(dV, cstrdV, rV, tx);
}

#endif  // CEED_MAGMA_BASIS_INTERP_2D_H
