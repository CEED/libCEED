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
