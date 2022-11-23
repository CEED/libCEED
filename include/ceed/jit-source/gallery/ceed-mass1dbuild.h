// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief Ceed QFunction for building the geometric data for the 1D mass matrix
**/

#ifndef mass1dbuild_h
#define mass1dbuild_h

#include <ceed.h>

CEED_QFUNCTION(Mass1DBuild)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians, size (Q)
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *w = in[1];
  // out[0] is quadrature data, size (Q)
  CeedScalar *q_data = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { q_data[i] = J[i] * w[i]; }  // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif  // mass1dbuild_h
