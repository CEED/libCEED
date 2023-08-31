// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <cuda.h>

const int sizeMax = 16;
__constant__ CeedScalar c_B[sizeMax*sizeMax];
__constant__ CeedScalar c_G[sizeMax*sizeMax];

//------------------------------------------------------------------------------
// Interp device initalization
//------------------------------------------------------------------------------
extern "C" int CeedInit_CudaInterp(CeedScalar *d_B, CeedInt P_1d, CeedInt Q_1d,
                                  CeedScalar **c_B_ptr) {
  const int bytes = P_1d*Q_1d*sizeof(CeedScalar);

  cudaMemcpyToSymbol(c_B, d_B, bytes, 0, cudaMemcpyDeviceToDevice);
  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Grad device initalization
//------------------------------------------------------------------------------
extern "C" int CeedInit_CudaGrad(CeedScalar *d_B, CeedScalar *d_G,
    CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr) {
  const int bytes = P_1d*Q_1d*sizeof(CeedScalar);

  cudaMemcpyToSymbol(c_B, d_B, bytes, 0, cudaMemcpyDeviceToDevice);
  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
  cudaMemcpyToSymbol(c_G, d_G, bytes, 0, cudaMemcpyDeviceToDevice);
  cudaGetSymbolAddress((void **)c_G_ptr, c_G);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Collocated grad device initalization
//------------------------------------------------------------------------------
extern "C" int CeedInit_CudaCollocatedGrad(CeedScalar *d_B, CeedScalar *d_G,
    CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr) {
  const int bytes_interp = P_1d*Q_1d*sizeof(CeedScalar);
  const int bytes_grad = Q_1d*Q_1d*sizeof(CeedScalar);

  cudaMemcpyToSymbol(c_B, d_B, bytes_interp, 0, cudaMemcpyDeviceToDevice);
  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
  cudaMemcpyToSymbol(c_G, d_G, bytes_grad, 0, cudaMemcpyDeviceToDevice);
  cudaGetSymbolAddress((void **)c_G_ptr, c_G);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
