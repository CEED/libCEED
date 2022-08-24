// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/**
  @brief Ceed QFunction for applying the 2D Poisson operator
           on a vector system with three components
**/

#ifndef vectorpoisson2dapply_h
#define vectorpoisson2dapply_h

#include <ceed.h>

CEED_QFUNCTION(Vector3Poisson2DApply)(void *ctx, const CeedInt Q,
                                      const CeedScalar *const *in,
                                      CeedScalar *const *out) {
  // *INDENT-OFF*
  // in[0] is gradient u, shape [2, nc=3, Q]
  // in[1] is quadrature data, size (3*Q)
  typedef CeedScalar array_t[3][CEED_Q_VLA];
  const array_t* ug = (const array_t*) in[0];

  typedef CeedScalar vec_t[CEED_Q_VLA];
  const vec_t* q_data = (const vec_t*) in[1];

  // out[0] is output to multiply against gradient v, shape [2, nc=3, Q]
  array_t* vg = (array_t*) out[0];
  // *INDENT-ON*

  const CeedInt dim = 2, num_comp = 3;

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read qdata (dXdxdXdxT symmetric matrix)
    // Stored in Voigt convention
    // 0 2
    // 2 1
    // *INDENT-OFF*
    const CeedScalar dXdxdXdxT[2][2] = {{q_data[0][i],
                                         q_data[2][i]},
                                        {q_data[2][i],
                                         q_data[1][i]}
                                       };
    // *INDENT-ON*

    // Apply Poisson operator
    // j = direction of vg
    for (CeedInt j=0; j<dim; j++)
      for (CeedInt c=0; c<num_comp; c++)
        vg[j][c][i] = (ug[0][c][i] * dXdxdXdxT[0][j] +
                       ug[1][c][i] * dXdxdXdxT[1][j]);
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // vectorpoisson2dapply_h
