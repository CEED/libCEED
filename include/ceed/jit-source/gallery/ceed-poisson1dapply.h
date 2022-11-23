// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief Ceed QFunction for applying the 1D Poisson operator
**/

#ifndef poisson1dapply_h
#define poisson1dapply_h

#include <ceed.h>

CEED_QFUNCTION(Poisson1DApply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is gradient u, size (Q)
  // in[1] is quadrature data, size (Q)
  const CeedScalar *ug = in[0], *q_data = in[1];

  // out[0] is output to multiply against gradient v, size (Q)
  CeedScalar *vg = out[0];

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) { vg[i] = ug[i] * q_data[i]; }  // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif  // poisson1dapply_h
