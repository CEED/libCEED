// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <hip/hip_runtime.h>

//------------------------------------------------------------------------------
// Kernel for copy strided on device
//------------------------------------------------------------------------------
__global__ static void copyStridedK(CeedScalar *__restrict__ vec, CeedSize start, CeedSize step, CeedSize size, CeedScalar *__restrict__ vec_copy) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index >= size) return;
  if ((index - start) % step == 0) vec_copy[index] = vec[index];
}

//------------------------------------------------------------------------------
// Copy strided on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceCopyStrided_Hip(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar *d_copy_array) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  hipLaunchKernelGGL(copyStridedK, dim3(grid_size), dim3(block_size), 0, 0, d_array, start, step, length, d_copy_array);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for set value on device
//------------------------------------------------------------------------------
__global__ static void setValueK(CeedScalar *__restrict__ vec, CeedSize size, CeedScalar val) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index >= size) return;
  vec[index] = val;
}

//------------------------------------------------------------------------------
// Set value on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceSetValue_Hip(CeedScalar *d_array, CeedSize length, CeedScalar val) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  hipLaunchKernelGGL(setValueK, dim3(grid_size), dim3(block_size), 0, 0, d_array, length, val);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for set value strided on device
//------------------------------------------------------------------------------
__global__ static void setValueStridedK(CeedScalar *__restrict__ vec, CeedSize start, CeedSize step, CeedSize size, CeedScalar val) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index >= size) return;
  if ((index - start) % step == 0) vec[index] = val;
}

//------------------------------------------------------------------------------
// Set value strided on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceSetValueStrided_Hip(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar val) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  hipLaunchKernelGGL(setValueStridedK, dim3(grid_size), dim3(block_size), 0, 0, d_array, start, step, length, val);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for taking reciprocal
//------------------------------------------------------------------------------
__global__ static void rcpValueK(CeedScalar *__restrict__ vec, CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index >= size) return;
  if (fabs(vec[index]) > 1E-16) vec[index] = 1. / vec[index];
}

//------------------------------------------------------------------------------
// Take vector reciprocal in device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceReciprocal_Hip(CeedScalar *d_array, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  hipLaunchKernelGGL(rcpValueK, dim3(grid_size), dim3(block_size), 0, 0, d_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for scale
//------------------------------------------------------------------------------
__global__ static void scaleValueK(CeedScalar *__restrict__ x, CeedScalar alpha, CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index >= size) return;
  x[index] *= alpha;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceScale_Hip(CeedScalar *x_array, CeedScalar alpha, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  hipLaunchKernelGGL(scaleValueK, dim3(grid_size), dim3(block_size), 0, 0, x_array, alpha, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpy
//------------------------------------------------------------------------------
__global__ static void axpyValueK(CeedScalar *__restrict__ y, CeedScalar alpha, CeedScalar *__restrict__ x, CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index >= size) return;
  y[index] += alpha * x[index];
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPY_Hip(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  hipLaunchKernelGGL(axpyValueK, dim3(grid_size), dim3(block_size), 0, 0, y_array, alpha, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpby
//------------------------------------------------------------------------------
__global__ static void axpbyValueK(CeedScalar *__restrict__ y, CeedScalar alpha, CeedScalar beta, CeedScalar *__restrict__ x, CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index >= size) return;
  y[index] = beta * y[index];
  y[index] += alpha * x[index];
}

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPBY_Hip(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  hipLaunchKernelGGL(axpbyValueK, dim3(grid_size), dim3(block_size), 0, 0, y_array, alpha, beta, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for pointwise mult
//------------------------------------------------------------------------------
__global__ static void pointwiseMultValueK(CeedScalar *__restrict__ w, CeedScalar *x, CeedScalar *__restrict__ y, CeedSize size) {
  CeedSize index = threadIdx.x + (CeedSize)blockDim.x * blockIdx.x;

  if (index >= size) return;
  w[index] = x[index] * y[index];
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device
//------------------------------------------------------------------------------
extern "C" int CeedDevicePointwiseMult_Hip(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length) {
  const int      block_size = 512;
  const CeedSize vec_size   = length;
  int            grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  hipLaunchKernelGGL(pointwiseMultValueK, dim3(grid_size), dim3(block_size), 0, 0, w_array, x_array, y_array, length);
  return 0;
}

//------------------------------------------------------------------------------
