// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CPU JiT utilities
#include <ceed/types.h>

#define CeedCall(...)        \
  do {                       \
    int ierr_ = __VA_ARGS__; \
    if (ierr_) return ierr_; \
  } while (0)

static inline CeedInt CeedIntPow(CeedInt base, CeedInt power) {
  CeedInt result = 1;
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    base *= base;
  }
  return result;
}

static inline CeedInt CeedIntMin(CeedInt a, CeedInt b) { return a < b ? a : b; }

static inline CeedInt CeedIntMax(CeedInt a, CeedInt b) { return a > b ? a : b; }
