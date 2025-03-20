// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL backend QFunction read/write kernels
#include <ceed/types.h>

//------------------------------------------------------------------------------
// Read from quadrature points
//------------------------------------------------------------------------------
inline void readQuads(CeedInt N, CeedInt stride, CeedInt offset, const CeedScalar *src, CeedScalar *dest) {
  for (CeedInt i = 0; i < N; ++i) dest[i] = src[stride * i + offset];
}

//------------------------------------------------------------------------------
// Write at quadrature points
//------------------------------------------------------------------------------
inline void writeQuads(CeedInt N, CeedInt stride, CeedInt offset, const CeedScalar *src, CeedScalar *dest) {
  for (CeedInt i = 0; i < N; ++i) dest[stride * i + offset] = src[i];
}
