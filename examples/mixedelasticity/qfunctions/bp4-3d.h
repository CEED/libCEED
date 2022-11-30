// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for diffusion operator example using PETSc

#ifndef bp43d_h
#define bp43d_h

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
CEED_QFUNCTION(SetupDiffRhs3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *coords = in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*true_soln)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*rhs)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar c[3] = {0, 1., 2.};
    const CeedScalar k[3] = {1., 2., 3.};
    CeedScalar       x = coords[i + 0 * Q], y = coords[i + 1 * Q], z = coords[i + 2 * Q];
    // Component 1
    true_soln[0][i] = sin(PI_DOUBLE * (c[0] + k[0] * x)) * sin(PI_DOUBLE * (c[1] + k[1] * y)) * sin(PI_DOUBLE * (c[2] + k[2] * z));
    // Component 2
    true_soln[1][i] = 2 * true_soln[0][i];
    // Component 3
    true_soln[2][i] = 3 * true_soln[0][i];

    // Component 1
    rhs[0][i] = q_data[0][i] * PI_DOUBLE * PI_DOUBLE * (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) * true_soln[0][i];
    // Component 2
    rhs[1][i] = 2 * rhs[0][i];
    // Component 3
    rhs[2][i] = 3 * rhs[0][i];
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
CEED_QFUNCTION(SetupDiff3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar dudX[3][3] = {
        {ug[0][0][i], ug[1][0][i], ug[2][0][i]},
        {ug[0][1][i], ug[1][1][i], ug[2][1][i]},
        {ug[0][2][i], ug[1][2][i], ug[2][2][i]}
    };
    CeedScalar dXdx_voigt[9];
    for (CeedInt j = 0; j < 9; j++) {
      dXdx_voigt[j] = q_data[j + 1][i];
    }
    CeedScalar dXdx[3][3];
    VoigtUnpackNonSymmetric3(dXdx_voigt, dXdx);
    // dXdxT_dXdx = (dX/dx^T * dX/dx)
    CeedScalar dXdxT_dXdx[3][3];
    AlphaMatTransposeMatMult3(1.0, dXdx, dXdx, dXdxT_dXdx);

    // save output:(dX/dx^T * dX/dx * wdetJ) * du/dX ==> du/dX^T * dXdxT_dXdx * wdetJ
    // we save the transpose, because of ordering in libCEED; See how we created dudX above
    AlphaMatTransposeMatMultAtQuadrature3(Q, i, q_data[0][i], dudX, dXdxT_dXdx, dvdX);
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // bp43d_h
