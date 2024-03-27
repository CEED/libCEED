// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA strided element restriction kernels

#include <ceed.h>

//------------------------------------------------------------------------------
// L-vector -> E-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void StridedNoTranspose(const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < RSTR_NUM_ELEM * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      v[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE] =
          u[loc_node * RSTR_STRIDE_NODES + comp * RSTR_STRIDE_COMP + elem * RSTR_STRIDE_ELEM];
    }
  }
}

//------------------------------------------------------------------------------
// E-vector -> L-vector, strided
//------------------------------------------------------------------------------
extern "C" __global__ void StridedTranspose(const CeedScalar *__restrict__ u, CeedScalar *__restrict__ v) {
  for (CeedInt node = blockIdx.x * blockDim.x + threadIdx.x; node < RSTR_NUM_ELEM * RSTR_ELEM_SIZE; node += blockDim.x * gridDim.x) {
    const CeedInt loc_node = node % RSTR_ELEM_SIZE;
    const CeedInt elem     = node / RSTR_ELEM_SIZE;

    for (CeedInt comp = 0; comp < RSTR_NUM_COMP; comp++) {
      v[loc_node * RSTR_STRIDE_NODES + comp * RSTR_STRIDE_COMP + elem * RSTR_STRIDE_ELEM] +=
          u[loc_node + comp * RSTR_ELEM_SIZE * RSTR_NUM_ELEM + elem * RSTR_ELEM_SIZE];
    }
  }
}
