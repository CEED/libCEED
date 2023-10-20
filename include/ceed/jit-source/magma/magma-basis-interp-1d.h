// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA tensor basis interpolation in 1D
#ifndef CEED_MAGMA_BASIS_INTERP_1D_H
#define CEED_MAGMA_BASIS_INTERP_1D_H

#include "magma-common-tensor.h"

// macros to abstract access of shared memory and reg. file
#define sT(i, j) sT[(j)*P + (i)]

////////////////////////////////////////////////////////////////////////////////
// interp basis action (1D)
template <typename T, int DIM, int NUM_COMP, int P, int Q>
static __device__ __inline__ void magma_interp_1d_device(const T *sT, magma_trans_t transT, T *sU[NUM_COMP], T *sV[NUM_COMP], const int tx) {
  // Assumptions
  // 1. 1D threads of size max(P,Q)
  // 2. sU[i] is 1xP: in shared memory
  // 3. sV[i] is 1xQ: in shared memory
  // 4. P_roduct per component is one row (1xP) times T matrix (PxQ) => one row (1xQ)
  // 5. Each thread computes one entry in sV[i]
  // 6. Must sync before and after call
  // 7. Note that the layout for U and V is different from 2D/3D problem

  T rv;
  if (tx < Q) {
    for (int comp = 0; comp < NUM_COMP; comp++) {
      rv = (transT == MagmaTrans) ? sV[comp][tx] : 0.0;
      for (int i = 0; i < P; i++) {
        rv += sU[comp][i] * sT(i, tx);  // sT[tx * P + i];
      }
      sV[comp][tx] = rv;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_MAX_P_Q, MAGMA_MAXTHREADS_1D)) __global__
    void magma_interpn_1d_kernel(const CeedScalar *dT, const CeedScalar *dU, const int estrdU, const int cstrdU, CeedScalar *dV, const int estrdV,
                                 const int cstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaNoTrans;

  if (elem_id >= nelem) return;

  CeedScalar *sU[BASIS_NUM_COMP];
  CeedScalar *sV[BASIS_NUM_COMP];

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sT = (CeedScalar *)shared_data;
  CeedScalar *sW = sT + BASIS_P * BASIS_Q;
  sU[0]          = sW + ty * BASIS_NUM_COMP * (BASIS_P + BASIS_Q);
  sV[0]          = sU[0] + (BASIS_NUM_COMP * 1 * BASIS_P);
  for (int comp = 1; comp < BASIS_NUM_COMP; comp++) {
    sU[comp] = sU[comp - 1] + (1 * BASIS_P);
    sV[comp] = sV[comp - 1] + (1 * BASIS_Q);
  }

  // read T
  if (ty == 0) {
    read_T_notrans_gm2sm<BASIS_P, BASIS_Q>(tx, dT, sT);
  }

  // read U
  read_1d<CeedScalar, BASIS_P, BASIS_NUM_COMP>(dU, cstrdU, sU, tx);

  __syncthreads();
  magma_interp_1d_device<CeedScalar, BASIS_DIM, BASIS_NUM_COMP, BASIS_P, BASIS_Q>(sT, transT, sU, sV, tx);
  __syncthreads();

  // write V
  write_1d<CeedScalar, BASIS_Q, BASIS_NUM_COMP>(sV, dV, cstrdV, tx);
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_MAX_P_Q, MAGMA_MAXTHREADS_1D)) __global__
    void magma_interpt_1d_kernel(const CeedScalar *dT, const CeedScalar *dU, const int estrdU, const int cstrdU, CeedScalar *dV, const int estrdV,
                                 const int cstrdV, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int     tx      = threadIdx.x;
  const int     ty      = threadIdx.y;
  const int     elem_id = (blockIdx.x * blockDim.y) + ty;
  magma_trans_t transT  = MagmaTrans;

  if (elem_id >= nelem) return;

  CeedScalar *sU[BASIS_NUM_COMP];
  CeedScalar *sV[BASIS_NUM_COMP];

  // shift global memory pointers by elem stride
  dU += elem_id * estrdU;
  dV += elem_id * estrdV;

  // assign shared memory pointers
  CeedScalar *sT = (CeedScalar *)shared_data;
  CeedScalar *sW = sT + BASIS_Q * BASIS_P;
  sU[0]          = sW + ty * BASIS_NUM_COMP * (BASIS_Q + BASIS_P);
  sV[0]          = sU[0] + (BASIS_NUM_COMP * 1 * BASIS_Q);
  for (int comp = 1; comp < BASIS_NUM_COMP; comp++) {
    sU[comp] = sU[comp - 1] + (1 * BASIS_Q);
    sV[comp] = sV[comp - 1] + (1 * BASIS_P);
  }

  // read T
  if (ty == 0) {
    read_T_trans_gm2sm<BASIS_Q, BASIS_P>(tx, dT, sT);
  }

  // read U
  read_1d<CeedScalar, BASIS_Q, BASIS_NUM_COMP>(dU, cstrdU, sU, tx);

  // read V
  read_1d<CeedScalar, BASIS_P, BASIS_NUM_COMP>(dV, cstrdV, sV, tx);

  __syncthreads();
  magma_interp_1d_device<CeedScalar, BASIS_DIM, BASIS_NUM_COMP, BASIS_Q, BASIS_P>(sT, transT, sU, sV, tx);
  __syncthreads();

  // write V
  write_1d<CeedScalar, BASIS_P, BASIS_NUM_COMP>(sV, dV, cstrdV, tx);
}

#endif  // CEED_MAGMA_BASIS_INTERP_1D_H
