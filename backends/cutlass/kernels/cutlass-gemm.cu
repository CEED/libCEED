// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <cuda.h>
#include <cutlass/gemm/device/gemm.h>

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor    = cutlass::layout::RowMajor;

using GemmNN = cutlass::gemm::device::Gemm<CeedScalar, ColumnMajor, CeedScalar, ColumnMajor, CeedScalar, ColumnMajor>;
using GemmTN = cutlass::gemm::device::Gemm<CeedScalar, RowMajor, CeedScalar, ColumnMajor, CeedScalar, ColumnMajor>;

extern "C" int CeedCutlassGemm_Cuda(bool trans_a, int m, int n, int k, CeedScalar alpha, const CeedScalar *A, int lda, const CeedScalar *B, int ldb,
                                    CeedScalar beta, CeedScalar *C, int ldc) {
  cutlass::Status status;
  if (trans_a) {
    GemmTN            gemm;
    GemmTN::Arguments args({m, n, k}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});

    status = gemm(args);

  } else {
    GemmNN            gemm;
    GemmNN::Arguments args({m, n, k}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});

    status = gemm(args);
  }
  if (status != cutlass::Status::kSuccess) {
    return 1;
  }
  return 0;
}

__global__ static void weightK(const CeedInt num_elem, const CeedInt Q, const CeedScalar *d_q_weight, CeedScalar *d_v) {
  CeedSize index     = (CeedSize)blockIdx.x * blockDim.x + threadIdx.x;
  CeedSize totalSize = (CeedSize)num_elem * Q;
  if (index < totalSize) {
    d_v[index] = d_q_weight[index % Q];
  }
}

extern "C" int CeedCutlassWeight_Cuda(const CeedInt num_elem, const CeedInt Q, const CeedScalar *d_q_weight, CeedScalar *d_v) {
  const int block_size = 512;
  CeedSize  grid_size  = ((CeedSize)num_elem * Q + block_size - 1) / block_size;
  weightK<<<grid_size, block_size>>>(num_elem, Q, d_q_weight, d_v);
  return 0;
}
