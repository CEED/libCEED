// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <sycl/sycl.hpp>

//------------------------------------------------------------------------------
// Kernel for set value on device
//------------------------------------------------------------------------------
__global__ static void setValueK(CeedScalar *__restrict__ vec, CeedInt size, CeedScalar val) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  if (index >= size) return;
  vec[index] = val;
}

//------------------------------------------------------------------------------
// Set value on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceSetValue_Sycl(CeedScalar *d_array, CeedInt length, CeedScalar val) {
  const int block_size = 512;
  const int vec_size   = length;
  int       grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  setValueK<<<grid_size, block_size>>>(d_array, length, val);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for taking reciprocal
//------------------------------------------------------------------------------
__global__ static void rcpValueK(CeedScalar *__restrict__ vec, CeedInt size) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  if (index >= size) return;
  if (fabs(vec[index]) > 1E-16) vec[index] = 1. / vec[index];
}

//------------------------------------------------------------------------------
// Take vector reciprocal in device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceReciprocal_Sycl(CeedScalar *d_array, CeedInt length) {
  const int block_size = 512;
  const int vec_size   = length;
  int       grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  rcpValueK<<<grid_size, block_size>>>(d_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for scale
//------------------------------------------------------------------------------
__global__ static void scaleValueK(CeedScalar *__restrict__ x, CeedScalar alpha, CeedInt size) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  if (index >= size) return;
  x[index] *= alpha;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceScale_Sycl(CeedScalar *x_array, CeedScalar alpha, CeedInt length) {
  const int block_size = 512;
  const int vec_size   = length;
  int       grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  scaleValueK<<<grid_size, block_size>>>(x_array, alpha, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpy
//------------------------------------------------------------------------------
__global__ static void axpyValueK(CeedScalar *__restrict__ y, CeedScalar alpha, CeedScalar *__restrict__ x, CeedInt size) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= size) return;
  y[index] += alpha * x[index];
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPY_Sycl(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedInt length) {
  const int block_size = 512;
  const int vec_size   = length;
  int       grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  axpyValueK<<<grid_size, block_size>>>(y_array, alpha, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for pointwise mult
//------------------------------------------------------------------------------
__global__ static void pointwiseMultValueK(CeedScalar *__restrict__ w, CeedScalar *x, CeedScalar *__restrict__ y, CeedInt size) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  if (index >= size) return;
  w[index] = x[index] * y[index];
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device
//------------------------------------------------------------------------------
extern "C" int CeedDevicePointwiseMult_Sycl(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedInt length) {
  const int block_size = 512;
  const int vec_size   = length;
  int       grid_size  = vec_size / block_size;

  if (block_size * grid_size < vec_size) grid_size += 1;
  pointwiseMultValueK<<<grid_size, block_size>>>(w_array, x_array, y_array, length);
  return 0;
}

//------------------------------------------------------------------------------
