// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_WEIGHT_H
#define CEED_MAGMA_WEIGHT_H

#include <ceed/ceed.h>

#include "magma_v2.h"

//////////////////////////////////////////////////////////////////////////////////////////
static __global__ void magma_weight_nontensor_kernel(const CeedInt nelem, const CeedInt Q, const CeedScalar *__restrict__ qweight,
                                                     CeedScalar *__restrict__ d_V) {
  const int tid = threadIdx.x;
  // TODO load qweight in shared memory if blockDim.z > 1?
  for (CeedInt elem = blockIdx.x * blockDim.z + threadIdx.z; elem < nelem; elem += gridDim.x * blockDim.z) {
    d_V[elem * Q + tid] = qweight[tid];
  }
}

#endif  // CEED_MAGMA_WEIGHT_H
