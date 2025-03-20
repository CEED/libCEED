// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

// H(div) basis for quadrilateral linear BDM element in 2D
// Local numbering is as follows (each edge has 2 vector DoF)
//     b4     b5
//    2---------3
//  b7|         |b3
//    |         |
//  b6|         |b2
//    0---------1
//     b0     b1
// Bx[0-->7] = b0_x-->b7_x, By[0-->7] = b0_y-->b7_y
// To see how the nodal basis is constructed visit:
// https://github.com/rezgarshakeri/H-div-Tests
int BuildNodalHdivQuadrilateral(CeedScalar *x, CeedScalar *Bx, CeedScalar *By) {
  CeedScalar x_hat = x[0];
  CeedScalar y_hat = x[1];
  Bx[0]            = -0.125 + 0.125 * x_hat * x_hat;
  By[0]            = -0.25 + 0.25 * x_hat + 0.25 * y_hat + -0.25 * x_hat * y_hat;
  Bx[1]            = 0.125 + -0.125 * x_hat * x_hat;
  By[1]            = -0.25 + -0.25 * x_hat + 0.25 * y_hat + 0.25 * x_hat * y_hat;
  Bx[2]            = 0.25 + 0.25 * x_hat + -0.25 * y_hat + -0.25 * x_hat * y_hat;
  By[2]            = -0.125 + 0.125 * y_hat * y_hat;
  Bx[3]            = 0.25 + 0.25 * x_hat + 0.25 * y_hat + 0.25 * x_hat * y_hat;
  By[3]            = 0.125 + -0.125 * y_hat * y_hat;
  Bx[4]            = -0.125 + 0.125 * x_hat * x_hat;
  By[4]            = 0.25 + -0.25 * x_hat + 0.25 * y_hat + -0.25 * x_hat * y_hat;
  Bx[5]            = 0.125 + -0.125 * x_hat * x_hat;
  By[5]            = 0.25 + 0.25 * x_hat + 0.25 * y_hat + 0.25 * x_hat * y_hat;
  Bx[6]            = -0.25 + 0.25 * x_hat + 0.25 * y_hat + -0.25 * x_hat * y_hat;
  By[6]            = -0.125 + 0.125 * y_hat * y_hat;
  Bx[7]            = -0.25 + 0.25 * x_hat + -0.25 * y_hat + 0.25 * x_hat * y_hat;
  By[7]            = 0.125 + -0.125 * y_hat * y_hat;
  return 0;
}

static void BuildHdivQuadrilateral(CeedInt q, CeedScalar *q_ref, CeedScalar *q_weights, CeedScalar *interp, CeedScalar *div, CeedQuadMode quad_mode) {
  // Get 1D quadrature on [-1,1]
  CeedScalar q_ref_1d[q], q_weight_1d[q];
  switch (quad_mode) {
    case CEED_GAUSS:
      CeedGaussQuadrature(q, q_ref_1d, q_weight_1d);
      break;
    // LCOV_EXCL_START
    case CEED_GAUSS_LOBATTO:
      CeedLobattoQuadrature(q, q_ref_1d, q_weight_1d);
      break;
  }
  // LCOV_EXCL_STOP

  // Divergence operator; Divergence of nodal basis for ref element
  CeedScalar D[8] = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
  CeedScalar Bx[8], By[8];
  CeedScalar X[2];

  // Loop over quadrature points
  for (CeedInt i = 0; i < q; i++) {
    for (CeedInt j = 0; j < q; j++) {
      CeedInt k1        = q * i + j;
      q_ref[k1]         = q_ref_1d[j];
      q_ref[k1 + q * q] = q_ref_1d[i];
      q_weights[k1]     = q_weight_1d[j] * q_weight_1d[i];
      X[0]              = q_ref_1d[j];
      X[1]              = q_ref_1d[i];
      BuildNodalHdivQuadrilateral(X, Bx, By);
      for (CeedInt k = 0; k < 8; k++) {
        interp[k1 * 8 + k]             = Bx[k];
        interp[k1 * 8 + k + 8 * q * q] = By[k];
        div[k1 * 8 + k]                = D[k];
      }
    }
  }
}
