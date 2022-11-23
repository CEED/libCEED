// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for diffusion operator example using PETSc

#ifndef bp3_h
#define bp3_h

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the geometric factors required to apply the
//   diffusion operator
//
// We require the product of the inverse of the Jacobian and its transpose to
//   properly compute integrals of the form: int( gradv gradu)
//
// Determinant of Jacobian:
//   detJ = J11*A11 + J21*A12 + J31*A13
//     Jij = Jacobian entry ij
//     Aij = Adjoint ij
//
// Inverse of Jacobian:
//   Bij = Aij / detJ
//
// Product of Inverse and Transpose:
//   BBij = sum( Bik Bkj )
//
// Stored: w B^T B detJ = w A^T A / detJ
//   Note: This matrix is symmetric, so we only store 6 distinct entries
//     qd: 1 4 7
//         2 5 8
//         3 6 9
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffGeo)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1];
  const CeedScalar(*w)                = in[2];  // Note: *X = in[0]
  // Outputs
  CeedScalar(*qd) = out[0];

  const CeedInt dim = 3;
  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    CeedScalar A[3][3];
    for (CeedInt j = 0; j < dim; j++) {
      for (CeedInt k = 0; k < dim; k++) {
        // Equivalent code with no mod operations:
        // A[k][j] = J[k+1][j+1]*J[k+2][j+2] - J[k+1][j+2]*J[k+2][j+1]
        A[k][j] = J[(k + 1) % dim][(j + 1) % dim][i] * J[(k + 2) % dim][(j + 2) % dim][i] -
                  J[(k + 1) % dim][(j + 2) % dim][i] * J[(k + 2) % dim][(j + 1) % dim][i];
      }
    }
    const CeedScalar detJ = J[0][0][i] * A[0][0] + J[0][1][i] * A[0][1] + J[0][2][i] * A[0][2];

    const CeedScalar qw = w[i] / detJ;
    qd[i + Q * 0]       = w[i] * detJ;
    qd[i + Q * 1]       = qw * (A[0][0] * A[0][0] + A[0][1] * A[0][1] + A[0][2] * A[0][2]);
    qd[i + Q * 2]       = qw * (A[0][0] * A[1][0] + A[0][1] * A[1][1] + A[0][2] * A[1][2]);
    qd[i + Q * 3]       = qw * (A[0][0] * A[2][0] + A[0][1] * A[2][1] + A[0][2] * A[2][2]);
    qd[i + Q * 4]       = qw * (A[1][0] * A[1][0] + A[1][1] * A[1][1] + A[1][2] * A[1][2]);
    qd[i + Q * 5]       = qw * (A[1][0] * A[2][0] + A[1][1] * A[2][1] + A[1][2] * A[2][2]);
    qd[i + Q * 6]       = qw * (A[2][0] * A[2][0] + A[2][1] * A[2][1] + A[2][2] * A[2][2]);
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
  const CeedScalar *x = in[0], *w = in[1];
  CeedScalar       *true_soln = out[0], *rhs = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar c[3] = {0, 1., 2.};
    const CeedScalar k[3] = {1., 2., 3.};

    true_soln[i] = sin(M_PI * (c[0] + k[0] * x[i + Q * 0])) * sin(M_PI * (c[1] + k[1] * x[i + Q * 1])) * sin(M_PI * (c[2] + k[2] * x[i + Q * 2]));

    rhs[i] = w[i + Q * 0] * M_PI * M_PI * (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) * true_soln[i];
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// This QFunction applies the diffusion operator for a scalar field.
//
// Inputs:
//   ug     - Input vector gradient at quadrature points
//   q_data  - Geometric factors
//
// Output:
//   vg     - Output vector (test functions) gradient at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(Diff)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *ug = in[0], *q_data = in[1];
  CeedScalar       *vg = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar du[3] = {ug[i + Q * 0], ug[i + Q * 1], ug[i + Q * 2]};
    // Read q_data (dXdxdXdx_T symmetric matrix)
    const CeedScalar dXdxdXdx_T[3][3] = {
        {q_data[i + 1 * Q], q_data[i + 2 * Q], q_data[i + 3 * Q]},
        {q_data[i + 2 * Q], q_data[i + 4 * Q], q_data[i + 5 * Q]},
        {q_data[i + 3 * Q], q_data[i + 5 * Q], q_data[i + 6 * Q]}
    };

    for (int j = 0; j < 3; j++) {  // j = direction of vg
      vg[i + j * Q] = (du[0] * dXdxdXdx_T[0][j] + du[1] * dXdxdXdx_T[1][j] + du[2] * dXdxdXdx_T[2][j]);
    }
  }  // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif  // bp3_h
