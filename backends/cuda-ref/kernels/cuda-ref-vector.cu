// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <cuda.h>

//------------------------------------------------------------------------------
// Kernel for copy strided on device
//------------------------------------------------------------------------------
__global__ static void copyStridedK(CeedScalar *__restrict__ vec, CeedSize start, CeedSize stop, CeedSize step, CeedScalar *__restrict__ vec_copy) {
  const CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index < stop - start) {
    if (index % step == 0) vec_copy[start + index] = vec[start + index];
  }
}

//------------------------------------------------------------------------------
// Copy strided on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceCopyStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize stop, CeedSize step, CeedScalar *d_copy_array) {
  const int      block_size = 512;
  const CeedSize copy_size  = stop - start;
  int            grid_size  = copy_size / block_size;

  if (block_size * grid_size < copy_size) grid_size += 1;
  copyStridedK<<<grid_size, block_size>>>(d_array, start, stop, step, d_copy_array);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for set value on device
//------------------------------------------------------------------------------
__global__ static void setValueK(CeedScalar *__restrict__ vec, CeedSize size, CeedScalar val) {
  const CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index < size) vec[index] = val;
}

//------------------------------------------------------------------------------
// Set value on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceSetValue_Cuda(CeedScalar *d_array, CeedSize length, CeedScalar val) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  setValueK<<<grid_size, block_size>>>(d_array, length, val);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for set value strided on device
//------------------------------------------------------------------------------
__global__ static void setValueStridedK(CeedScalar *__restrict__ vec, CeedSize start, CeedSize stop, CeedSize step, CeedScalar val) {
  const CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index < stop - start) {
    if (index % step == 0) vec[start + index] = val;
  }
}

//------------------------------------------------------------------------------
// Set value strided on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceSetValueStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize stop, CeedSize step, CeedScalar val) {
  const int      block_size = 512;
  const CeedSize set_size   = stop - start;
  int            grid_size  = set_size / block_size;

  if (block_size * grid_size < set_size) grid_size += 1;
  setValueStridedK<<<grid_size, block_size>>>(d_array, start, stop, step, val);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for taking reciprocal
//------------------------------------------------------------------------------
__global__ static void rcpValueK(CeedScalar *__restrict__ vec, CeedSize size) {
  const CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index < size) {
    if (fabs(vec[index]) > 1E-16) vec[index] = 1. / vec[index];
  }
}

//------------------------------------------------------------------------------
// Take vector reciprocal in device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceReciprocal_Cuda(CeedScalar *d_array, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  rcpValueK<<<grid_size, block_size>>>(d_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for scale
//------------------------------------------------------------------------------
__global__ static void scaleValueK(CeedScalar *__restrict__ x, CeedScalar alpha, CeedSize size) {
  const CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index < size) x[index] *= alpha;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceScale_Cuda(CeedScalar *x_array, CeedScalar alpha, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  scaleValueK<<<grid_size, block_size>>>(x_array, alpha, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpy
//------------------------------------------------------------------------------
__global__ static void axpyValueK(CeedScalar *__restrict__ y, CeedScalar alpha, CeedScalar *__restrict__ x, CeedSize size) {
  const CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index < size) y[index] += alpha * x[index];
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  axpyValueK<<<grid_size, block_size>>>(y_array, alpha, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpby
//------------------------------------------------------------------------------
__global__ static void axpbyValueK(CeedScalar *__restrict__ y, CeedScalar alpha, CeedScalar beta, CeedScalar *__restrict__ x, CeedSize size) {
  const CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index < size) {
    y[index] = beta * y[index];
    y[index] += alpha * x[index];
  }
}

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  axpbyValueK<<<grid_size, block_size>>>(y_array, alpha, beta, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for pointwise mult
//------------------------------------------------------------------------------
__global__ static void pointwiseMultValueK(CeedScalar *__restrict__ w, CeedScalar *x, CeedScalar *__restrict__ y, CeedSize size) {
  const CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index < size) w[index] = x[index] * y[index];
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device
//------------------------------------------------------------------------------
extern "C" int CeedDevicePointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  pointwiseMultValueK<<<grid_size, block_size>>>(w_array, x_array, y_array, length);
  return 0;
}

//------------------------------------------------------------------------------
