// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for diffusion operator example using PETSc

#ifndef bp4_h
#define bp4_h

#include <ceed.h>
#include <math.h>

#include "utils.h"

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
  const CeedScalar *coords = in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*true_soln)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*rhs)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar c[3] = {0, 1., 2.};
    const CeedScalar k[3] = {1., 2., 3.};
    CeedScalar       x = coords[i + 0 * Q], y = coords[i + 1 * Q], z = coords[i + 2 * Q];
    // Component 1
    true_soln[0][i] = sin(M_PI * (c[0] + k[0] * x)) * sin(M_PI * (c[1] + k[1] * y)) * sin(M_PI * (c[2] + k[2] * z));
    // Component 2
    true_soln[1][i] = 2 * true_soln[0][i];
    // Component 3
    true_soln[2][i] = 3 * true_soln[0][i];

    // Component 1
    rhs[0][i] = q_data[0][i] * M_PI * M_PI * (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) * true_soln[0][i];
    // Component 2
    rhs[1][i] = 2 * rhs[0][i];
    // Component 3
    rhs[2][i] = 3 * rhs[0][i];
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the diffusion operator for a vector field of 3 components.
// \int grad(v)^T grad(u) = \int (dv/dX)^T (dX/dx^T * dX/dx) du/dX * w*detJ
// Inputs:
//   ug      - Input vector du/dX
//   q_data  - Geometric factors
//
// Output:
//   dvdX    - Output vector (test functions) Jacobian at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiff)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
    VoigtUnpackNonSymmetric(dXdx_voigt, dXdx);
    // (dX/dx^T * dX/dx) * w * detJ
    CeedScalar dXdxT_dXdx[3][3];
    AlphaMatTransposeMatMult(q_data[0][i], dXdx, dXdx, dXdxT_dXdx);

    CeedScalar dv[3][3];
    // save output: (dX/dx^T * dX/dx) du/dX * w*detJ
    AlphaMatMatMult(1.0, dXdxT_dXdx, dudX, dv);
    for (CeedInt k = 0; k < 3; k++) {    // k = component
      for (CeedInt j = 0; j < 3; j++) {  // j = direction of vg
        // we save the transpose, because of ordering in libCEED; See how we created dudX above
        dvdX[j][k][i] = dv[k][j];
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // bp4_h
