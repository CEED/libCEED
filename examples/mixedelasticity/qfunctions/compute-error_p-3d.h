// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for computing error of p field in 3D

#ifndef errorp3d_h
#define errorp3d_h

#include <ceed.h>

// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupError3Dp)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *p = in[0], *target = in[1], *q_data = in[2];
  CeedScalar       *error = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    error[i + 0 * Q] = (p[i + 0 * Q] - target[i + 3 * Q]) * (p[i + 0 * Q] - target[i + 3 * Q]) * q_data[i];
  }
  return 0;
}
// -----------------------------------------------------------------------------

#endif  // errorp3d_h
