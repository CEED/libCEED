// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for diffusion operator example using PETSc

#ifndef bp42d_h
#define bp42d_h

#include <ceed.h>
#include <math.h>

#include "utils.h"
// -----------------------------------------------------------------------------
// Strong form:
//  -nabla^2 (u)        = f   in \Omega
//
// Weak form: Find u \in V (V=H1) on \Omega
//  (grad(v), grad(u)) = (v, f)
//
// We set the true solution in a way that vanishes on the boundary.
// This QFunction setup the rhs and true solution of the above equation
// Inputs:
//   coords     : coordinate of the physical element
//   wdetJ      : updated weight of quadrature
//
// Output:
//   true_soln  :
//   rhs        : (v, f) = \int (v^T * f * wdetJ) dX
//                we save: rhs = f * wdetJ
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *coords = in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*true_soln)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*rhs)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar c[2] = {0., 1.};
    const CeedScalar k[2] = {1., 2.};
    CeedScalar       x = coords[i + 0 * Q], y = coords[i + 1 * Q];
    // Component 1
    true_soln[0][i] = sin(PI_DOUBLE * (c[0] + k[0] * x)) * sin(PI_DOUBLE * (c[1] + k[1] * y));
    // Component 2
    true_soln[1][i] = 2 * true_soln[0][i];

    // Component 1
    rhs[0][i] = q_data[0][i] * PI_DOUBLE * PI_DOUBLE * (k[0] * k[0] + k[1] * k[1]) * true_soln[0][i];
    // Component 2
    rhs[1][i] = 2 * rhs[0][i];
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------
// This QFunction setup the lhs of the above equation
// Inputs:
//   dudX       : derivative of basis with respect to ref element coordinate; du/dX
//   q_data     : updated weight of quadrature and inverse of the Jacobian J; [wdetJ, dXdx]
//
// Output:
//   dvdX       : (grad(v), grad(u)) = \int (dv/dX)^T (dX/dx^T * dX/dx) du/dX * wdetJ dX
//                we save: dvdX = (dX/dx^T * dX/dx) du/dX * wdetJ
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiff2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*ug)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*dvdX)[2][CEED_Q_VLA] = (CeedScalar(*)[2][CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar dudX[2][2] = {
        {ug[0][0][i], ug[1][0][i]},
        {ug[0][1][i], ug[1][1][i]}
    };
    CeedScalar dXdx_voigt[4];
    for (CeedInt j = 0; j < 4; j++) {
      dXdx_voigt[j] = q_data[j + 1][i];
    }
    CeedScalar dXdx[2][2];
    VoigtUnpackNonSymmetric2(dXdx_voigt, dXdx);
    // (dX/dx^T * dX/dx) * wdetJ
    CeedScalar dXdxT_dXdx[2][2];
    AlphaMatTransposeMatMult2(q_data[0][i], dXdx, dXdx, dXdxT_dXdx);

    CeedScalar dv[2][2];
    // save output: (dX/dx^T * dX/dx * wdetJ) * du/dX
    AlphaMatMatMult2(1.0, dXdxT_dXdx, dudX, dv);
    for (CeedInt j = 0; j < 2; j++) {
      for (CeedInt k = 0; k < 2; k++) {
        // we save the transpose, because of ordering in libCEED; See how we created dudX above
        dvdX[j][k][i] = dv[k][j];
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // bp42d_h
