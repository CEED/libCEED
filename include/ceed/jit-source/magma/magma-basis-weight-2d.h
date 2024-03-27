// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA tensor basis weight in 2D

#include "magma-common-tensor.h"

////////////////////////////////////////////////////////////////////////////////
// weight basis action -- 2D
template <typename T, int DIM, int NUM_COMP, int Q, int i_DIM, int i_COMP>
static __device__ __inline__ void magma_weight_2d_device(const T *sTweight, T rV[DIM][NUM_COMP][Q], const int tx) {
  // Assumptions
  // 1. 1D thread configuration of size Q
  // 2. rV[][][] matches the storage used in other actions (interp, grad, ... etc)
  // 3. i_DIM and i_COMP specify which indexes to use in rV,
  //    since the output per thread is a register array of size Q
  // 4. Sync is recommended after the call (to make sure sTweight can be overwritten)

  if (tx < Q) {
    // x sTweight[j]  for first update
    // x sTweight[tx] for second update
    for (int j = 0; j < Q; j++) {
      rV[i_DIM][i_COMP][j] = sTweight[j] * sTweight[tx];
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_Q, MAGMA_MAXTHREADS_2D)) __global__
    void magma_weight_2d_kernel(const CeedScalar *dqweight1d, CeedScalar *dV, const int v_stride, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int elem_id = (blockIdx.x * blockDim.y) + ty;

  if (elem_id >= nelem) return;

  CeedScalar rV[1][1][BASIS_Q];  // allocate with BASIS_DIM=BASIS_NUM_COMP=1, but sizes may differ for a fused operator
  // global memory pointers
  dV += elem_id * v_stride;

  // shared memory pointers
  CeedScalar *sTweight = (CeedScalar *)shared_data;

  // read dqweight_1d
  if (ty == 0 && tx < BASIS_Q) {
    sTweight[tx] = dqweight1d[tx];
  }

  __syncthreads();
  magma_weight_2d_device<CeedScalar, 1, 1, BASIS_Q, 0, 0>(sTweight, rV, tx);

  // write V
  if (tx < BASIS_Q) {
    for (int j = 0; j < BASIS_Q; j++) {
      dV[j * BASIS_Q + tx] = rV[0][0][j];
    }
  }
}
