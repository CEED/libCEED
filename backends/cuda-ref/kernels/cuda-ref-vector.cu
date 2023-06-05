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
__global__ static void setValueK(CeedScalar * __restrict__ vec, CeedInt size,
                                 CeedScalar val) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size)
    return;
  vec[idx] = val;
}

//------------------------------------------------------------------------------
// Set value on device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceSetValue_Cuda(CeedScalar* d_array, CeedInt length,
                                       CeedScalar val) {
  const int bsize = 512;
  const int vecsize = length;
  int gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize)
    gridsize += 1;
  setValueK<<<gridsize,bsize>>>(d_array, length, val);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for taking reciprocal
//------------------------------------------------------------------------------
__global__ static void rcpValueK(CeedScalar * __restrict__ vec, CeedInt size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size)
    return;
  if (fabs(vec[idx]) > 1E-16)
    vec[idx] = 1./vec[idx];
}

//------------------------------------------------------------------------------
// Take vector reciprocal in device memory
//------------------------------------------------------------------------------
extern "C" int CeedDeviceReciprocal_Cuda(CeedScalar* d_array, CeedInt length) {
  const int bsize = 512;
  const int vecsize = length;
  int gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize)
    gridsize += 1;
  rcpValueK<<<gridsize,bsize>>>(d_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for scale
//------------------------------------------------------------------------------
__global__ static void scaleValueK(CeedScalar * __restrict__ x, CeedScalar alpha,
    CeedInt size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size)
    return;
  x[idx] *= alpha;
}

//------------------------------------------------------------------------------
// Compute x = alpha x on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceScale_Cuda(CeedScalar *x_array, CeedScalar alpha,
    CeedInt length) {
  const int bsize = 512;
  const int vecsize = length;
  int gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize)
    gridsize += 1;
  scaleValueK<<<gridsize,bsize>>>(x_array, alpha, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpy
//------------------------------------------------------------------------------
__global__ static void axpyValueK(CeedScalar * __restrict__ y, CeedScalar alpha,
    CeedScalar * __restrict__ x, CeedInt size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size)
    return;
  y[idx] += alpha * x[idx];
}

//------------------------------------------------------------------------------
// Compute y = alpha x + y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha,
    CeedScalar *x_array, CeedInt length) {
  const int bsize = 512;
  const int vecsize = length;
  int gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize)
    gridsize += 1;
  axpyValueK<<<gridsize,bsize>>>(y_array, alpha, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for axpby
//------------------------------------------------------------------------------
__global__ static void axpbyValueK(CeedScalar * __restrict__ y, CeedScalar alpha, CeedScalar beta,
    CeedScalar * __restrict__ x, CeedInt size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size)
    return;
  y[idx] = beta * y[idx];
  y[idx] += alpha * x[idx];
}

//------------------------------------------------------------------------------
// Compute y = alpha x + beta y on device
//------------------------------------------------------------------------------
extern "C" int CeedDeviceAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta,
    CeedScalar *x_array, CeedInt length) {
  const int bsize = 512;
  const int vecsize = length;
  int gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize)
    gridsize += 1;
  axpbyValueK<<<gridsize,bsize>>>(y_array, alpha, beta, x_array, length);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for pointwise mult
//------------------------------------------------------------------------------
__global__ static void pointwiseMultValueK(CeedScalar * __restrict__ w,
    CeedScalar * x, CeedScalar * __restrict__ y, CeedInt size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= size)
    return;
  w[idx] = x[idx] * y[idx];
}

//------------------------------------------------------------------------------
// Compute the pointwise multiplication w = x .* y on device
//------------------------------------------------------------------------------
extern "C" int CeedDevicePointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array,
    CeedScalar *y_array, CeedInt length) {
  const int bsize = 512;
  const int vecsize = length;
  int gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize)
    gridsize += 1;
  pointwiseMultValueK<<<gridsize,bsize>>>(w_array, x_array, y_array, length);
  return 0;
}
