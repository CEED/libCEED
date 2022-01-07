// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
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
// Kernel for converting double to single 
//------------------------------------------------------------------------------
__global__ static void convertFp64Fp32(CeedInt length, double* double_data,
                                       float* float_data) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= length)
    return;
  float_data[idx] = (float) double_data[idx];
}

//------------------------------------------------------------------------------
// Convert a double-precision array to single precision
//------------------------------------------------------------------------------
extern "C" int CeedDeviceConvertArray_Cuda_Fp64_Fp32(CeedInt length,
                                                     double* double_data,
                                                     float* float_data) {
  const int bsize = 512;
  const int vecsize = length;
  int gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize)
    gridsize += 1;
  convertFp64Fp32<<<gridsize,bsize>>>(length, double_data, float_data);
  return 0;
}

//------------------------------------------------------------------------------
// Kernel for converting single to double
//------------------------------------------------------------------------------
__global__ static void convertFp32Fp64(CeedInt length, float* float_data,
                                       double* double_data) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= length)
    return;
  double_data[idx] = (double) float_data[idx];
}

//------------------------------------------------------------------------------
// Convert a single-precision array to double precision
//------------------------------------------------------------------------------
extern "C" int CeedDeviceConvertArray_Cuda_Fp32_Fp64(CeedInt length,
                                                     float* float_data,
                                                     double* double_data) {
  const int bsize = 512;
  const int vecsize = length;
  int gridsize = vecsize / bsize;

  if (bsize * gridsize < vecsize)
    gridsize += 1;
  convertFp32Fp64<<<gridsize,bsize>>>(length, float_data, double_data);
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
