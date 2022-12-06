// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// libCEED QFunctions for mixed-linear elasticity example using PETSc

#ifndef mixed_linear2d_h
#define mixed_linear2d_h

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
//  div(sigma)  +  f   = 0   in \Omega
//  div(u)      -  p/k = 0   in \Omega
//
//  where k is bulk modulus, and sigma_ij = p * delta_ij + 2 * mu * ed_ij
//  ed = e - 1/3 trace(e) * I is the deviatoric strain and e = 0.5*(grad(u) + (grad(u))^T )
// in indicial notation
//   mu * ui_jj + (1/3 mu + k) * uj_ji + fi = 0
//   ui_i   - p/k                           = 0
//
// Weak form: Find (u,p) \in VxQ (V=H1, Q=L^2) on \Omega
//  (grad(v), sigma)      = (v, f)
//  (q, div(u)) - (q,p/k) = 0
// We set the true solution in a way that vanishes on the boundary.
// This QFunction setup the rhs and true solution of the above equation
// Inputs:
//   coords     : coordinate of the physical element
//   wdetJ      : updated weight of quadrature
//
// Output:
//   true_soln  : pe and ue
//   rhs        : (v, f) = \int (v^T * f * wdetJ) dX
//                we save: rhs_u = f * wdetJ and rhs_p = 0
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMixedLinearRhs2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *coords = in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  CeedScalar(*true_soln)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*rhs_u)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1],
  (*rhs_p)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];

  // Context
  LINEARContext    context = (LINEARContext)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;
  const CeedScalar mu      = E / (2. * (1 + nu));
  const CeedScalar kappa   = E / (3. * (1 - 2 * nu));
  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar x = coords[i + 0 * Q], y = coords[i + 1 * Q];
    CeedScalar u1 = sin(PI_DOUBLE * x) * sin(PI_DOUBLE * y), u2 = 2 * u1;
    CeedScalar u1_1 = PI_DOUBLE * cos(PI_DOUBLE * x) * sin(PI_DOUBLE * y), u2_2 = 2 * PI_DOUBLE * sin(PI_DOUBLE * x) * cos(PI_DOUBLE * y);
    // Component 1 of u
    true_soln[0][i] = u1;
    // Component 2 of u
    true_soln[1][i] = u2;
    // Pressure, p = kappa * div(u)
    true_soln[2][i] = kappa * (u1_1 + u2_2);

    // mu*(u1_11 + u1_22) + (1/3 * mu + kappa)*(u1_11 + u2_21) + f1 = 0
    CeedScalar u1_11 = -PI_DOUBLE * PI_DOUBLE * u1, u1_22 = -PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u2_21 = 2 * PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * cos(PI_DOUBLE * y);
    CeedScalar f1    = -mu * (u1_11 + u1_22) - ((1. / 3.) * mu + kappa) * (u1_11 + u2_21);
    // Component 1
    rhs_u[0][i] = q_data[0][i] * f1;
    // mu*(u2_11 + u2_22) + (1/3 * mu + kappa)*(u1_12 + u2_22) + f2 = 0
    CeedScalar u2_11 = -2 * PI_DOUBLE * PI_DOUBLE * u1, u2_22 = -2 * PI_DOUBLE * PI_DOUBLE * u1;
    CeedScalar u1_12 = PI_DOUBLE * PI_DOUBLE * cos(PI_DOUBLE * x) * cos(PI_DOUBLE * y);
    CeedScalar f2    = -mu * (u2_11 + u2_22) - ((1. / 3.) * mu + kappa) * (u1_12 + u2_22);
    // Component 2
    rhs_u[1][i] = q_data[0][i] * f2;
    rhs_p[0][i] = 0.0;
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------
// This QFunction setup the lhs of the above equation
// Inputs:
//   dudX       : derivative of basis with respect to ref element coordinate; du/dX
//   q_data     : updated weight of quadrature and inverse of the Jacobian J; [wdetJ, dXdx]
//   basis_p    : basis for pressure field
//
// Output:
//   dvdX       : (grad(v), sigma) = \int (dv/dX)^T (dX/dx^T * sigma) * wdetJ dX
//                we save:    dvdX = (dX/dx^T * sigma) * wdetJ
//   q          : (q, div(u)) - (q,p/k) = \int q^T [div(u) - p/k] * wdetJ dX
//                we save:            q = [div(u) - p/k] * wdetJ
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMixedLinear2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*ug)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1],
        (*p)[CEED_Q_VLA]           = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  CeedScalar(*dvdX)[2][CEED_Q_VLA] = (CeedScalar(*)[2][CEED_Q_VLA])out[0], (*q)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

  // Context
  LINEARContext    context = (LINEARContext)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;
  const CeedScalar mu      = E / (2 * (1 + nu));
  const CeedScalar kappa   = E / (3 * (1 - 2 * nu));
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
    CeedScalar e_kk = Trace2(e);
    // Compute Deviatoric Strain : ed (epsilon)
    // ed = e - 1/3 * trace(e) * I
    const CeedScalar ed[2][2] = {
        {e[0][0] - (1. / 3.) * e_kk, e[0][1]                   },
        {e[1][0],                    e[1][1] - (1. / 3.) * e_kk}
    };
    // Compute Sigma = p*delta_ij + 2*mu*ed_ij
    const CeedScalar sigma[2][2] = {
        {p[0][i] + 2. * mu * ed[0][0], 2. * mu * ed[0][1]          },
        {2. * mu * ed[1][0],           p[0][i] + 2. * mu * ed[1][1]}
    };
    // save output:dX/dx^T * sigma * wdetJ ==> sigma^T * dX/dx * wdetJ
    // we save the transpose, because of ordering in libCEED; See how we created dudX above
    AlphaMatTransposeMatMultAtQuadrature2(Q, i, q_data[0][i], sigma, dXdx, dvdX);
    // div(u) = trace(grad(u))
    CeedScalar div_u = Trace2(grad_u);
    // (q, div(u)) - (q,p/k) = q^T * (div(u) - p/k) * wdetJ
    q[0][i] = (div_u - p[0][i] / kappa) * q_data[0][i];
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // mixed_linear2d_h
