// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for MAGMA non-tensor basis weight
#ifndef CEED_MAGMA_BASIS_WEIGHT_NONTENSOR_H
#define CEED_MAGMA_BASIS_WEIGHT_NONTENSOR_H

#include "magma-common-nontensor.h"

////////////////////////////////////////////////////////////////////////////////
extern "C" __launch_bounds__(MAGMA_BASIS_BOUNDS(BASIS_Q, MAGMA_MAXTHREADS_1D)) __global__
    void magma_weight_nontensor(int n, const CeedScalar *__restrict__ dqweight, CeedScalar *__restrict__ dV) {
  MAGMA_DEVICE_SHARED(CeedScalar, shared_data);

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int id = blockIdx.x * blockDim.y + ty;

  // terminate threads with no work
  if (id >= n) return;

  dV += id * BASIS_Q;

  // shared memory pointers
  CeedScalar *sqweight = (CeedScalar *)shared_data;
  CeedScalar *sV       = sqweight + BASIS_Q;
  sV += ty * BASIS_Q;

  // read qweight
  if (ty == 0 && tx < BASIS_Q) {
    sqweight[tx] = dqweight[tx];
  }
  __syncthreads();

  if (tx < BASIS_Q) {
    sV[tx] = sqweight[tx];
  }

  // write V
  dV[tx] = sV[tx];
}

#endif  // CEED_MAGMA_BASIS_WEIGHT_NONTENSOR_H
