// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <cuda.h>

//------------------------------------------------------------------------------
// Kernel for set value on device
//------------------------------------------------------------------------------
__global__ static void setValueK(CeedScalar * __restrict__ vec, CeedSize size,
                                 CeedScalar val) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;
  if (index >= size)
    return;
  vec[index] = val;
}

//------------------------------------------------------------------------------
// Set value on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceSetValue_Cuda(CeedScalar* d_array, CeedSize length,
                                       CeedScalar val) {
  const int block_size = 512;
  const CeedSize vec_size = length;
  int grid_size = vec_size / block_size;

  if (block_size * grid_size < vec_size)
    grid_size += 1;
  setValueK<<<grid_size,block_size>>>(d_array, length, val);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for taking reciprocal
//------------------------------------------------------------------------------
__global__ static void rcpValueK(CeedScalar * __restrict__ vec, CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;
  if (index >= size)
    return;
  if (fabs(vec[index]) > 1E-16)
    vec[index] = 1./vec[index];
}

//------------------------------------------------------------------------------
// Take vector reciprocal in device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceReciprocal_Cuda(CeedScalar* d_array, CeedSize length) {
  const int block_size = 512;
  const CeedSize vec_size = length;
  int grid_size = vec_size / block_size;

  if (block_size * grid_size < vec_size)
    grid_size += 1;
  rcpValueK<<<grid_size,block_size>>>(d_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for scale
//------------------------------------------------------------------------------
__global__ static void scaleValueK(CeedScalar * __restrict__ x, CeedScalar alpha,
    CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;
  if (index >= size)
    return;
  x[index] *= alpha;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceScale_Cuda(CeedScalar *x_array, CeedScalar alpha,
    CeedSize length) {
  const int block_size = 512;
  const CeedSize vec_size = length;
  int grid_size = vec_size / block_size;

  if (block_size * grid_size < vec_size)
    grid_size += 1;
  scaleValueK<<<grid_size,block_size>>>(x_array, alpha, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpy
//------------------------------------------------------------------------------
__global__ static void axpyValueK(CeedScalar * __restrict__ y, CeedScalar alpha,
    CeedScalar * __restrict__ x, CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;
  if (index >= size)
    return;
  y[index] += alpha * x[index];
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha,
    CeedScalar *x_array, CeedSize length) {
  const int block_size = 512;
  const CeedSize vec_size = length;
  int grid_size = vec_size / block_size;

  if (block_size * grid_size < vec_size)
    grid_size += 1;
  axpyValueK<<<grid_size,block_size>>>(y_array, alpha, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpby
//------------------------------------------------------------------------------
__global__ static void axpbyValueK(CeedScalar * __restrict__ y, CeedScalar alpha, CeedScalar beta,
    CeedScalar * __restrict__ x, CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;
  if (index >= size)
    return;
  y[index] = beta * y[index];
  y[index] += alpha * x[index];
}

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta,
    CeedScalar *x_array, CeedSize length) {
  const int block_size = 512;
  const CeedSize vec_size = length;
  int grid_size = vec_size / block_size;

  if (block_size * grid_size < vec_size)
    grid_size += 1;
  axpbyValueK<<<grid_size,block_size>>>(y_array, alpha, beta, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for pointwise mult
//------------------------------------------------------------------------------
__global__ static void pointwiseMultValueK(CeedScalar * __restrict__ w,
    CeedScalar * x, CeedScalar * __restrict__ y, CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;
  if (index >= size)
    return;
  w[index] = x[index] * y[index];
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device
//------------------------------------------------------------------------------
extern "C" int CeedDevicePointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array,
    CeedScalar *y_array, CeedSize length) {
  const int block_size = 512;
  const CeedSize vec_size = length;
  int grid_size = vec_size / block_size;

  if (block_size * grid_size < vec_size)
    grid_size += 1;
  pointwiseMultValueK<<<grid_size,block_size>>>(w_array, x_array, y_array, length);
  return 0;
}

//------------------------------------------------------------------------------
