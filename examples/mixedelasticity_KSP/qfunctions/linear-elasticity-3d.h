// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for diffusion operator example using PETSc

#ifndef linear3d_h
#define linear3d_h

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
CEED_QFUNCTION(SetupLinearRhs3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
    CeedScalar u1 = sin(PI_DOUBLE * x) * sin(PI_DOUBLE * y) * sin(PI_DOUBLE * z), u2 = 2 * u1, u3 = 3 * u1;
    // Component 1
    true_soln[0][i] = u1;
    // Component 2
    true_soln[1][i] = u2;
    // Component 3
    true_soln[2][i] = u3;

    // mu*(u1_11 + u1_22 + u1_33) + (mu+lambda)*(u1_11 + u2_21 + u3_31) + f1 = 0
    CeedScalar u1_11 = -PI_DOUBLE * PI_DOUBLE * u1, u1_22 = -PI_DOUBLE * PI_DOUBLE * u1, u1_33 = -PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u2_21 = 2 * PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * cos(PI_DOUBLE * y) * sin(PI_DOUBLE * z);
    CeedScalar u3_31 = 3 * PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * sin(PI_DOUBLE * y) * cos(PI_DOUBLE * z);
    CeedScalar f1    = -mu * (u1_11 + u1_22 + u1_33) - (mu + lambda) * (u1_11 + u2_21 + u3_31);
    // Component 1
    rhs[0][i] = q_data[0][i] * f1;
    // mu*(u2_11 + u2_22 + u2_33) + (mu+lambda)*(u1_12 + u2_22 + u3_32) + f2 = 0
    CeedScalar u2_11 = -2 * PI_DOUBLE * PI_DOUBLE * u1, u2_22 = -2 * PI_DOUBLE * PI_DOUBLE * u1, u2_33 = -2 * PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u1_12 = PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * cos(PI_DOUBLE * y) * sin(PI_DOUBLE * z);
    CeedScalar u3_32 = 3 * PI_DOUBLE * PI_DOUBLE * sin(PI_DOUBLE * x) * cos(PI_DOUBLE * y) * cos(PI_DOUBLE * z);
    CeedScalar f2    = -mu * (u2_11 + u2_22 + u2_33) - (mu + lambda) * (u1_12 + u2_22 + u3_32);
    // Component 2
    rhs[1][i] = q_data[0][i] * f2;
    // mu*(u3_11 + u3_22 + u3_33) + (mu+lambda)*(u1_13 + u2_23 + u3_33) + f3 = 0
    CeedScalar u3_11 = -3 * PI_DOUBLE * PI_DOUBLE * u1, u3_22 = -3 * PI_DOUBLE * PI_DOUBLE * u1, u3_33 = -3 * PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u1_13 = PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * sin(PI_DOUBLE * y) * cos(PI_DOUBLE * z);
    CeedScalar u2_23 = 2 * PI_DOUBLE * PI_DOUBLE * sin(PI_DOUBLE * x) * cos(PI_DOUBLE * y) * cos(PI_DOUBLE * z);
    CeedScalar f3    = -mu * (u3_11 + u3_22 + u3_33) - (mu + lambda) * (u1_13 + u2_23 + u3_33);
    // Component 3
    rhs[2][i] = q_data[0][i] * f3;
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
CEED_QFUNCTION(SetupLinear3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
    VoigtUnpackNonSymmetric3(dXdx_voigt, dXdx);
    // Compute grad_u = dX/dx * du/dX
    CeedScalar grad_u[3][3];
    AlphaMatMatMult3(1.0, dXdx, dudX, grad_u);
    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)
    const CeedScalar e[3][3] = {
        {(grad_u[0][0] + grad_u[0][0]) / 2., (grad_u[0][1] + grad_u[1][0]) / 2., (grad_u[0][2] + grad_u[2][0]) / 2.},
        {(grad_u[1][0] + grad_u[0][1]) / 2., (grad_u[1][1] + grad_u[1][1]) / 2., (grad_u[1][2] + grad_u[2][1]) / 2.},
        {(grad_u[2][0] + grad_u[0][2]) / 2., (grad_u[2][1] + grad_u[1][2]) / 2., (grad_u[2][2] + grad_u[2][2]) / 2.}
    };
    // Compute Sigma = lambda*delta_ij*e_kk + 2*mu*e_ij
    CeedScalar       e_kk        = Trace3(e);
    const CeedScalar sigma[3][3] = {
        {lambda * e_kk + 2. * mu * e[0][0], 2. * mu * e[0][1],                 2. * mu * e[0][2]                },
        {2. * mu * e[1][0],                 lambda * e_kk + 2. * mu * e[1][1], 2. * mu * e[1][2]                },
        {2. * mu * e[2][0],                 2. * mu * e[2][1],                 lambda * e_kk + 2. * mu * e[2][2]}
    };
    // save output:dX/dx^T * sigma * wdetJ ==> sigma^T * dX/dx * wdetJ
    // we save the transpose, because of ordering in libCEED; See how we created dudX above
    AlphaMatTransposeMatMultAtQuadrature3(Q, i, q_data[0][i], sigma, dXdx, dvdX);
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------
#endif  // linear3d_h
