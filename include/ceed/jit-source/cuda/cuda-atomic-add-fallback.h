// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA atomic add fallback definition
#ifndef CEED_CUDA_ATOMIC_ADD_FALLBACK_H
#define CEED_CUDA_ATOMIC_ADD_FALLBACK_H

#include <ceed/types.h>

//------------------------------------------------------------------------------
// Atomic add, for older CUDA
//------------------------------------------------------------------------------
__device__ CeedScalar atomicAdd(CeedScalar *address, CeedScalar val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int  old            = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN
    // (since NaN != NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}

//------------------------------------------------------------------------------

#endif  // CEED_CUDA_ATOMIC_ADD_FALLBACK_H
