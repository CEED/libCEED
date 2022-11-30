// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for linear elasticity example using PETSc

#ifndef linear2d_h
#define linear2d_h

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
// Strong form:
//  -div(sigma)        = f   in \Omega
// in indicial notation
//    mu * ui_jj + (mu + lambda) * uj_ji + fi = 0
//
// Weak form: Find u \in V (V=H1) on \Omega
//  (grad(v), sigma) = (v, f)
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
CEED_QFUNCTION(SetupLinearRhs2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
    CeedScalar x = coords[i + 0 * Q], y = coords[i + 1 * Q];
    CeedScalar u1 = sin(PI_DOUBLE * x) * sin(PI_DOUBLE * y), u2 = 2 * u1;
    // Component 1
    true_soln[0][i] = u1;
    // Component 2
    true_soln[1][i] = u2;

    // mu*(u1_11 + u1_22) + (mu+lambda)*(u1_11 + u2_21) + f1 = 0
    CeedScalar u1_11 = -PI_DOUBLE * PI_DOUBLE * u1, u1_22 = -PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u2_21 = 2 * PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * cos(PI_DOUBLE * y);
    CeedScalar f1    = -mu * (u1_11 + u1_22) - (mu + lambda) * (u1_11 + u2_21);
    // Component 1
    rhs[0][i] = q_data[0][i] * f1;
    // mu*(u2_11 + u2_22) + (mu+lambda)*(u1_12 + u2_22) + f2 = 0
    CeedScalar u2_11 = -2 * PI_DOUBLE * PI_DOUBLE * u1, u2_22 = -2 * PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u1_12 = PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * cos(PI_DOUBLE * y);
    CeedScalar f2    = -mu * (u2_11 + u2_22) - (mu + lambda) * (u1_12 + u2_22);
    // Component 2
    rhs[1][i] = q_data[0][i] * f2;
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
//   dvdX       : (grad(v), sigma) = \int (dv/dX)^T (dX/dx^T * sigma) * wdetJ dX
//                we save: dvdX = (dX/dx^T * sigma) * wdetJ
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupLinear2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*ug)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*dvdX)[2][CEED_Q_VLA] = (CeedScalar(*)[2][CEED_Q_VLA])out[0];

  // Context
  LINEARContext    context = (LINEARContext)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;
  const CeedScalar mu      = E / (2 * (1 + nu));
  const CeedScalar lambda  = 2 * nu * mu / (1 - 2 * nu);
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
    // Compute grad_u = dX/dx * du/dX
    CeedScalar grad_u[2][2];
    AlphaMatMatMult2(1.0, dXdx, dudX, grad_u);
    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)
    const CeedScalar e[2][2] = {
        {(grad_u[0][0] + grad_u[0][0]) / 2., (grad_u[0][1] + grad_u[1][0]) / 2.},
        {(grad_u[1][0] + grad_u[0][1]) / 2., (grad_u[1][1] + grad_u[1][1]) / 2.}
    };
    // Compute Sigma = lambda*delta_ij*e_kk + 2*mu*e_ij
    CeedScalar       e_kk        = Trace2(e);
    const CeedScalar sigma[2][2] = {
        {lambda * e_kk + 2. * mu * e[0][0], 2. * mu * e[0][1]                },
        {2. * mu * e[1][0],                 lambda * e_kk + 2. * mu * e[1][1]}
    };
    // save output:dX/dx^T * sigma * wdetJ ==> sigma^T * dX/dx * wdetJ
    // we save the transpose, because of ordering in libCEED; See how we created dudX above
    AlphaMatTransposeMatMultAtQuadrature2(Q, i, q_data[0][i], sigma, dXdx, dvdX);
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // linear2d_h
