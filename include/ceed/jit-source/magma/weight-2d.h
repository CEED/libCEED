// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//////////////////////////////////////////////////////////////////////////////////////////
// weight basis action -- 2D
template <typename T, int DIM_, int NCOMP_, int Q_, int iDIM, int iCOMP>
__device__ __inline__ void magma_weight_2d_device(const T* sTweight, T rV[DIM_][NCOMP_][Q_], const int tx) {
  // Assumptions
  // 1. 1D thread configuration of size Q_
  // 2. rV[][][] matches the storage used in other actions (interp, grad, ... etc)
  // 3. iDIM and iCOMP specify which indexes to use in rV,
  //    since the output per thread is a register array of size Q_
  // 4. Sync is recommended after the call (to make sure sTweight can be overwritten)

  if (tx < Q_) {
    // x sTweight[j]  for first update
    // x sTweight[tx] for second update
    for (int j = 0; j < Q_; j++) {
      rV[iDIM][iCOMP][j] = sTweight[j] * sTweight[tx];
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(Q, MAGMA_MAXTHREADS_2D)) __global__
    void magma_weight_2d_kernel(const CeedScalar* dqweight1d, CeedScalar* dV, const int v_stride, const int nelem) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data)

  const int tx      = threadIdx.x;
  const int ty      = threadIdx.y;
  const int elem_id = (blockIdx.x * blockDim.y) + ty;

  if (elem_id >= nelem) return;

  CeedScalar rV[1][1][Q];  // allocate with DIM=NCOMP=1, but sizes may differ for a fused operator
  // global memory pointers
  dV += elem_id * v_stride;

  // shared memory pointers
  CeedScalar* sTweight = (CeedScalar*)shared_data;

  // read dqweight_1d
  if (ty == 0 && tx < Q) {
    sTweight[tx] = dqweight1d[tx];
  }

  __syncthreads();
  magma_weight_2d_device<CeedScalar, 1, 1, Q, 0, 0>(sTweight, rV, tx);

  // write V
  if (tx < Q) {
    for (int j = 0; j < Q; j++) {
      dV[j * Q + tx] = rV[0][0][j];
    }
  }
}
