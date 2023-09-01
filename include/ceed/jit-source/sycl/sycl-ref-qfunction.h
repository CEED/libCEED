// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL backend QFunction read/write kernels
#ifndef CEED_SYCL_REF_QFUNCTION_H
#define CEED_SYCL_REF_QFUNCTION_H

#include <ceed.h>

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

//------------------------------------------------------------------------------

#endif  // CEED_SYCL_REF_QFUNCTION_H
