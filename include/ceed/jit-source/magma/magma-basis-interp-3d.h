// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA tensor basis interpolation in 3D
#ifndef CEED_MAGMA_BASIS_INTERP_3D_H
#define CEED_MAGMA_BASIS_INTERP_3D_H

#include "magma-common-tensor.h"

// macros to abstract access of shared memory and reg. file
#define sT(i, j) sT[(j) * P + (i)]
#define sTmp(i, j, ldw) sTmp[(j) * (ldw) + (i)]

////////////////////////////////////////////////////////////////////////////////
// interp basis action (3D)
template <typename T, int DIM_U, int DIM_V, int NUM_COMP, int P, int Q, int rU_SIZE, int rV_SIZE>
static __device__ __inline__ void magma_interp_3d_device(const T *sT, T rU[DIM_U][NUM_COMP][rU_SIZE], T rV[DIM_V][NUM_COMP][rV_SIZE], const int tx,
                                                         T rTmp[Q], T *swork) {
  // Assumptions
  // 1. 1D threads of size max(P,Q)^2
  // 2. input:  rU[DIM_U x NUM_COMP x rU_SIZE] in registers (per thread)
  // 3. output: rV[DIM_V x NUM_COMP x rV_SIZE] in registers (per thread)
  // 4. Three products per component
  //  4.1 Batch P^2 of (1xP) matrices times (PxQ) matrix => Batch P^2 of (1xQ) matrices
  //  4.2 Batch P   of (QxP) matrices times (PxQ) matrix => Batch P   of (QxQ) matrices
  //  4.3 Batch 1   of (Q^2xP_) matrix times (PxQ) matrix => (Q^2xQ_) matrix
  // 5. Each thread computes one row of the output of each product
  // 6. Sync is recommended before and after the call

  for (int comp = 0; comp < NUM_COMP; comp++) {
    // Batch P^2 of (1xP) matrices [reg] times (PxQ) matrix [shmem] => Batch P^2 of (1xQ) matrices [shmem]
    if (tx < (P * P)) {
      const int batchid = tx;
      const int sld     = 1;
      T        *sTmp    = swork + batchid * (1 * Q);
      for (int j = 0; j < Q; j++) {
        rTmp[0] = 0.0;
        for (int i = 0; i < P; i++) {
          rTmp[0] += rU[0][comp][i] * sT(i, j);
        }
        sTmp(0, j, sld) = rTmp[0];
      }
    }  // end of: if (tx < P*P)
    __syncthreads();

    // Batch P of (QxP) matrices [shmem] times (PxQ) matrix [shmem] => Batch P of (QxQ) matrices [reg]
    if (tx < (P * Q)) {
      const int batchid = tx / Q;
      const int tx_     = tx % Q;
      const int sld     = Q;
      T        *sTmp    = swork + batchid * (Q * P);  // sTmp is input
      for (int j = 0; j < Q; j++) {
        rTmp[j] = 0.0;
        for (int i = 0; i < P; i++) {
          rTmp[j] += sTmp(tx_, i, sld) * sT(i, j);
        }
      }
    }
    __syncthreads();

    // write rTmp[] into shmem as batch P of QxQ matrices
    if (tx < (P * Q)) {
      const int batchid = tx / Q;
      const int tx_     = tx % Q;
      const int sld     = Q;
      T        *sTmp    = swork + batchid * (Q * Q);
      for (int j = 0; j < Q; j++) {
        sTmp(tx_, j, sld) = rTmp[j];
      }
    }
    __syncthreads();

    // Batch 1 of (Q^2xP_) matrices [shmem] times (PxQ) matrix [shmem] => Batch 1 of (Q^2xQ_) matrices [reg]
    if (tx < (Q * Q)) {
      // No need to declare batchid = (tx  / Q^2) = always zero
      // No need to declare tx_     = (tx_ % Q^2) = always tx
      const int sld  = Q * Q;
      T        *sTmp = swork;
      for (int j = 0; j < Q; j++) {
        rTmp[0] = 0.0;
        for (int i = 0; i < P; i++) {
          rTmp[0] += sTmp(tx, i, sld) * sT(i, j);
        }
        rV[0][comp][j] += rTmp[0];
      }
    }
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_MAX_P_Q *BASIS_MAX_P_Q, MAGMA_MAXTHREADS_3D)) __global__
    void magma_interpn_3d_kernel(const CeedScalar *dT, const CeedScalar *dU, const int estrdU, const int cstrdU, CeedScalar *dV, const int estrdV,
                                 const int cstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int elem_id = (blockIdx.x * blockDim.y) + ty;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][BASIS_NUM_COMP][BASIS_P] = {0.0};  // for a non-fused operator BASIS_DIM is always 1
  CeedScalar rV[1][BASIS_NUM_COMP][BASIS_Q] = {0.0};  // for a non-fused operator BASIS_DIM is always 1
  CeedScalar rTmp[BASIS_Q]                  = {0.0};

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sT   = (CeedScalar *)shared_data;
  CeedScalar *sTmp = sT + BASIS_P * BASIS_Q;
  sTmp += ty * (max(BASIS_P * BASIS_P * BASIS_MAX_P_Q, BASIS_P * BASIS_Q * BASIS_Q));

  // read T
  if (ty == 0) {
    read_T_notrans_gm2sm<BASIS_P, BASIS_Q>(tx, dT, sT);
  }

  // read U (idim = 0 for dU, i_DIM = 0 for rU, u_dimstride is always 0)
  read_U_3d<CeedScalar, BASIS_P, 1, BASIS_NUM_COMP, BASIS_P, 0>(dU, cstrdU, rU, sTmp, tx);
  // there is a sync at the end of this function

  magma_interp_3d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_P, BASIS_Q, BASIS_P, BASIS_Q>(sT, rU, rV, tx, rTmp, sTmp);
  __syncthreads();

  // write V
  write_V_3d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dV, cstrdV, rV, tx);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_MAX_P_Q *BASIS_MAX_P_Q, MAGMA_MAXTHREADS_3D)) __global__
    void magma_interpt_3d_kernel(const CeedScalar *dT, const CeedScalar *dU, const int estrdU, const int cstrdU, CeedScalar *dV, const int estrdV,
                                 const int cstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int elem_id = (blockIdx.x * blockDim.y) + ty;

  if (elem_id >= nelem) return;

  CeedScalar rU[1][BASIS_NUM_COMP][BASIS_Q] = {0.0};  // for a non-fused operator BASIS_DIM is always 1
  CeedScalar rV[1][BASIS_NUM_COMP][BASIS_P] = {0.0};  // for a non-fused operator BASIS_DIM is always 1
  CeedScalar rTmp[BASIS_P]                  = {0.0};

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sT   = (CeedScalar *)shared_data;
  CeedScalar *sTmp = sT + BASIS_Q * BASIS_P;
  sTmp += ty * (max(BASIS_Q * BASIS_Q * BASIS_MAX_P_Q, BASIS_Q * BASIS_P * BASIS_P));

  // read T
  if (ty == 0) {
    read_T_trans_gm2sm<BASIS_Q, BASIS_P>(tx, dT, sT);
  }

  // read U (idim = 0 for dU, i_DIM = 0 for rU, u_dimstride is always 0)
  read_U_3d<CeedScalar, BASIS_Q, 1, BASIS_NUM_COMP, BASIS_Q, 0>(dU, cstrdU, rU, sTmp, tx);
  // there is a sync at the end of this function

  magma_interp_3d_device<CeedScalar, 1, 1, BASIS_NUM_COMP, BASIS_Q, BASIS_P, BASIS_Q, BASIS_P>(sT, rU, rV, tx, rTmp, sTmp);
  __syncthreads();

  // write V
  write_V_3d<CeedScalar, BASIS_P, 1, BASIS_NUM_COMP, BASIS_P, 0>(dV, cstrdV, rV, tx);
}

#endif  // CEED_MAGMA_BASIS_INTERP_3D_H
