// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <cuda.h>

const int               MAX_SIZE = 52, MAX_DIM = 3;
__constant__ CeedScalar c_B[MAX_SIZE * MAX_SIZE * MAX_DIM];

//------------------------------------------------------------------------------
// Interp device initialization
//------------------------------------------------------------------------------
extern "C" int CeedInit_CudaNonTensor(CeedScalar *d_B, CeedInt P, CeedInt Q, CeedInt dim, CeedScalar **c_B_ptr) {
  const int bytes = P * Q * dim * sizeof(CeedScalar);

  cudaMemcpyToSymbol(c_B, d_B, bytes, 0, cudaMemcpyDeviceToDevice);
  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
