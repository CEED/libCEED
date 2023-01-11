// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA atomic add fallback definition
#ifndef _ceed_cuda_atomic_add_fallback_h
#define _ceed_cuda_atomic_add_fallback_h

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

#endif
