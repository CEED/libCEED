// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/ceed.h>
#include <hip/hip_runtime.h>

//------------------------------------------------------------------------------
// Kernel for set value on device
//------------------------------------------------------------------------------
__global__ static void setValueK(CeedScalar *__restrict__ vec, CeedInt size, CeedScalar val) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;
  vec[idx] = val;
}

//------------------------------------------------------------------------------
// Set value on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceSetValue_Hip(CeedScalar *d_array, CeedInt length, CeedScalar val) {
  const int bsize    = 512;
  const int vecsize  = length;
  int       gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize) gridsize += 1;
  hipLaunchKernelGGL(setValueK, dim3(gridsize), dim3(bsize), 0, 0, d_array, length, val);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for taking reciprocal
//------------------------------------------------------------------------------
__global__ static void rcpValueK(CeedScalar *__restrict__ vec, CeedInt size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;
  if (fabs(vec[idx]) > 1E-16) vec[idx] = 1. / vec[idx];
}

//------------------------------------------------------------------------------
// Take vector reciprocal in device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceReciprocal_Hip(CeedScalar *d_array, CeedInt length) {
  const int bsize    = 512;
  const int vecsize  = length;
  int       gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize) gridsize += 1;
  hipLaunchKernelGGL(rcpValueK, dim3(gridsize), dim3(bsize), 0, 0, d_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for scale
//------------------------------------------------------------------------------
__global__ static void scaleValueK(CeedScalar *__restrict__ x, CeedScalar alpha, CeedInt size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;
  x[idx] *= alpha;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceScale_Hip(CeedScalar *x_array, CeedScalar alpha, CeedInt length) {
  const int bsize    = 512;
  const int vecsize  = length;
  int       gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize) gridsize += 1;
  hipLaunchKernelGGL(scaleValueK, dim3(gridsize), dim3(bsize), 0, 0, x_array, alpha, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpy
//------------------------------------------------------------------------------
__global__ static void axpyValueK(CeedScalar *__restrict__ y, CeedScalar alpha, CeedScalar *__restrict__ x, CeedInt size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;
  y[idx] += alpha * x[idx];
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPY_Hip(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedInt length) {
  const int bsize    = 512;
  const int vecsize  = length;
  int       gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize) gridsize += 1;
  hipLaunchKernelGGL(axpyValueK, dim3(gridsize), dim3(bsize), 0, 0, y_array, alpha, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for pointwise mult
//------------------------------------------------------------------------------
__global__ static void pointwiseMultValueK(CeedScalar *__restrict__ w, CeedScalar *x, CeedScalar *__restrict__ y, CeedInt size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size) return;
  w[idx] = x[idx] * y[idx];
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device
//------------------------------------------------------------------------------
extern "C" int CeedDevicePointwiseMult_Hip(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedInt length) {
  const int bsize    = 512;
  const int vecsize  = length;
  int       gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize) gridsize += 1;
  hipLaunchKernelGGL(pointwiseMultValueK, dim3(gridsize), dim3(bsize), 0, 0, w_array, x_array, y_array, length);
  return 0;
}
