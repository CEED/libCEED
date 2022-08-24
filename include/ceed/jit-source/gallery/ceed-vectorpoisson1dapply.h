// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief Ceed QFunction for applying the 1D Poisson operator on a vector system with three components
**/

#ifndef vectorpoisson1dapply_h
#define vectorpoisson1dapply_h

#include <ceed.h>

CEED_QFUNCTION(Vector3Poisson1DApply)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is gradient u, shape [1, nc=3, Q]
  // in[1] is quadrature data, size (Q)
  typedef CeedScalar array_t[CEED_Q_VLA];
  const array_t* ug = (const array_t*) in[0];
  const CeedScalar * const q_data = in[1];
  // out[0] is output to multiply against gradient v, shape [1, nc=3, Q]
  array_t* vg = (array_t*) out[0];
  // *INDENT-ON*

  const CeedInt num_comp = 3;

  // Quadrature point loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    for (CeedInt c = 0; c < num_comp; c++) {
      vg[c][i] = ug[c][i] * q_data[i];
    }
  }  // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif  // vectorpoisson1dapply_h
