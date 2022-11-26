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

#ifndef LINEAR_CTX
#define LINEAR_CTX
typedef struct LINEARContext_ *LINEARContext;
struct LINEARContext_ {
  CeedScalar E;
  CeedScalar nu;
};
#endif
// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// mu * ui_jj + (mu + lambda) * uj_ji + fi = 0
// component i of elasticity equation; ui_ji = dui/d^2xj
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupLinearRhs)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
  const CeedScalar *coords = in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*true_soln)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*rhs)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

  // Context
  LINEARContext    context = (LINEARContext)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;
  const CeedScalar mu      = E / (2 * (1 + nu));
  const CeedScalar lambda  = 2 * nu * mu / (1 - 2 * nu);
  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar x = coords[i + 0 * Q], y = coords[i + 1 * Q], z = coords[i + 2 * Q];
    CeedScalar u1 = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z), u2 = 2 * u1, u3 = 3 * u1;
    // Component 1
    true_soln[0][i] = u1;
    // Component 2
    true_soln[1][i] = u2;
    // Component 3
    true_soln[2][i] = u3;

    // mu*(u1_11 + u1_22 + u1_33) + (mu+lambda)*(u1_11 + u2_21 + u3_31) + f1 = 0
    CeedScalar u1_11 = -M_PI * M_PI * u1, u1_22 = -M_PI * M_PI * u1, u1_33 = -M_PI * M_PI * u1;
    CeedScalar u2_21 = 2 * M_PI * M_PI * cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
    CeedScalar u3_31 = 3 * M_PI * M_PI * cos(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);
    CeedScalar f1    = -mu * (u1_11 + u1_22 + u1_33) - (mu + lambda) * (u1_11 + u2_21 + u3_31);
    // Component 1
    rhs[0][i] = q_data[0][i] * f1;
    // mu*(u2_11 + u2_22 + u2_33) + (mu+lambda)*(u1_12 + u2_22 + u3_32) + f2 = 0
    CeedScalar u2_11 = -2 * M_PI * M_PI * u1, u2_22 = -2 * M_PI * M_PI * u1, u2_33 = -2 * M_PI * M_PI * u1;
    CeedScalar u1_12 = M_PI * M_PI * cos(M_PI * x) * cos(M_PI * y) * sin(M_PI * z);
    CeedScalar u3_32 = 3 * M_PI * M_PI * sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
    CeedScalar f2    = -mu * (u2_11 + u2_22 + u2_33) - (mu + lambda) * (u1_12 + u2_22 + u3_32);
    // Component 2
    rhs[1][i] = q_data[0][i] * f2;
    // mu*(u3_11 + u3_22 + u3_33) + (mu+lambda)*(u1_13 + u2_23 + u3_33) + f2 = 0
    CeedScalar u3_11 = -3 * M_PI * M_PI * u1, u3_22 = -3 * M_PI * M_PI * u1, u3_33 = -3 * M_PI * M_PI * u1;
    CeedScalar u1_13 = M_PI * M_PI * cos(M_PI * x) * sin(M_PI * y) * cos(M_PI * z);
    CeedScalar u2_23 = 2 * M_PI * M_PI * sin(M_PI * x) * cos(M_PI * y) * cos(M_PI * z);
    CeedScalar f3    = -mu * (u3_11 + u3_22 + u3_33) - (mu + lambda) * (u1_13 + u2_23 + u3_33);
    // Component 3
    rhs[2][i] = q_data[0][i] * f3;
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// \int grad(v)^T sigma = \int (dv/dX)^T (dX/dx^T * sigma) * w*detJ
// Inputs:
//   ug      - Input vector du/dX
//   q_data  - Geometric factors
//
// Output:
//   dvdX    - Output vector (test functions) Jacobian at quadrature points
//
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupLinear)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];

  // Context
  LINEARContext    context = (LINEARContext)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;
  const CeedScalar mu      = E / (2 * (1 + nu));
  const CeedScalar lambda  = 2 * nu * mu / (1 - 2 * nu);
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
    // Compute grad_u = dX/dx * du/dX
    CeedScalar grad_u[3][3];
    AlphaMatMatMult(1.0, dXdx, dudX, grad_u);
    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)
    const CeedScalar e[3][3] = {
        {(grad_u[0][0] + grad_u[0][0]) / 2., (grad_u[0][1] + grad_u[1][0]) / 2., (grad_u[0][2] + grad_u[2][0]) / 2.},
        {(grad_u[1][0] + grad_u[0][1]) / 2., (grad_u[1][1] + grad_u[1][1]) / 2., (grad_u[1][2] + grad_u[2][1]) / 2.},
        {(grad_u[2][0] + grad_u[0][2]) / 2., (grad_u[2][1] + grad_u[1][2]) / 2., (grad_u[2][2] + grad_u[2][2]) / 2.}
    };
    // Compute Sigma : lambda*delta_ij*e_kk + 2*mu*e_ij
    CeedScalar       e_kk        = Trace3(e);
    const CeedScalar sigma[3][3] = {
        {lambda * e_kk + 2. * mu * e[0][0], 2. * mu * e[0][1],                 2. * mu * e[0][2]                },
        {2. * mu * e[1][0],                 lambda * e_kk + 2. * mu * e[1][1], 2. * mu * e[1][2]                },
        {2. * mu * e[2][0],                 2. * mu * e[2][1],                 lambda * e_kk + 2. * mu * e[2][2]}
    };
    CeedScalar dv[3][3];
    // save output: dX/dx^T * sigma * wdetJ
    AlphaMatTransposeMatMult(q_data[0][i], dXdx, sigma, dv);
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
