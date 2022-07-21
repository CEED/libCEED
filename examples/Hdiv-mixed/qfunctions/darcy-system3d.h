// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Darcy problem 3D (hex element) using PETSc

#ifndef DARCY_SYSTEM3D_H
#define DARCY_SYSTEM3D_H

#include <math.h>
#include <ceed.h>
#include "utils.h"

// -----------------------------------------------------------------------------
// See Matthew Farthing, Christopher Kees, Cass Miller (2003)
// https://www.sciencedirect.com/science/article/pii/S0309170802001872
// -----------------------------------------------------------------------------
// Strong form:
//  u        = -rho*k_r*K *[grad(\psi) - rho*g_u]   in \Omega
//  -\div(u) = -f                                   in \Omega
//  p        = p_b                                  on \Gamma_D
//  u.n      = u_b                                  on \Gamma_N
//
//  Where rho = rho_a/rho_a0, rho_a = rho_a0*exp(\beta * (p - p0)), p0 = 101325 Pa is atmospheric pressure
//  rho_a0 is the density at p_0, g_u = g/norm(g) where g is gravity.
//  k_r = b_a + alpha_a * (\psi - x2), where \psi = p / (rho_a0 * norm(g)) and x2 is vertical axis
//
// Weak form: Find (u, \psi) \in VxQ (V=H(div), Q=L^2) on \Omega
//  (v, K^{-1}/rho*k_r * u) -(v, rho*g_u) -(\div(v), \psi) = -<v, p_b*n>_{\Gamma_D}
// -(q, \div(u))  + (q, f)                                 = 0
//
// We solve MMS for  K = kappa*I and beta=0 ==> rho=1
//
// This QFunction setup the mixed form of the above equation
// Inputs:
//   w     : weight of quadrature
//   J     : dx/dX. x physical coordinate, X reference coordinate [-1,1]^dim
//   u     : basis_u at quadrature points
// div(u)  : divergence of basis_u at quadrature points
//   p     : basis_p at quadrature points
//   f     : force vector created in true qfunction
//
// Output:
//   v     : (v, K^{-1}/rho*k_r u) = \int (v^T * K^{-1}/rho*k_r*u detJ*w)dX ==> \int (v^T J^T * K^{-1}/rho*k_r *J*u*w/detJ)dX
//           -(v, rho*g_u)     = \int (v^T * rho*g_u detJ*w)dX ==> \int (v^T J^T * rho*g_u*w) dX
// div(v)  : -(\div(v), \psi) = -\int (div(v)^T * \psi *w) dX
//   q     : -(q, \div(u)) = -\int (q^T * div(u) * w) dX
//            (q, f)       = \int( q^T * f * w*detJ )dX
// -----------------------------------------------------------------------------
#ifndef DARCY_CTX
#define DARCY_CTX
typedef struct DARCYContext_ *DARCYContext;
struct DARCYContext_ {
  CeedScalar kappa;
  CeedScalar g;
  CeedScalar rho_a0;
  CeedScalar alpha_a, b_a;
};
#endif
// -----------------------------------------------------------------------------
// Residual evaluation for Darcy problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(DarcySystem3D)(void *ctx, CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_u) = (const CeedScalar(*))in[3],
                   (*p) = (const CeedScalar(*))in[4],
                   (*f) = in[5],
                   (*coords) = in[6];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_v) = (CeedScalar(*))out[1],
             (*q) = (CeedScalar(*))out[2];
  // Context
  DARCYContext  context = (DARCYContext)ctx;
  const CeedScalar kappa    = context->kappa;
  const CeedScalar rho_a0   = context->rho_a0;
  const CeedScalar g        = context->g;
  const CeedScalar alpha_a  = context->alpha_a;
  const CeedScalar b_a      = context->b_a;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup, J = dx/dX
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q], z = coords[i+2*Q];
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar det_J = MatDet3x3(J);

    // *INDENT-OFF*
    // k_r = b_a + alpha_a * (\psi - x2)
    CeedScalar k_r = b_a + alpha_a*(1-x*y*z);
    // rho = rho_a/rho_a0
    CeedScalar rho = 1.0;
    // (v, K^{-1}/rho*k_r u): v = J^T* (K^{-1}/rho*k_r) *J*u*w/detJ
    // 1) Compute K^{-1}, note K = kappa*I
    CeedScalar K[3][3] = {{kappa, 0., 0.},
                          {0., kappa, 0.},
                          {0., 0., kappa}
                         };
    const CeedScalar det_K = MatDet3x3(K);
    CeedScalar K_inv[3][3];
    MatInverse3x3(K, det_K, K_inv);

    // 2) (K^{-1}/rho*k_r) *J
    CeedScalar Kinv_J[3][3];
    AlphaMatMatMult3x3(1/(rho*k_r), K_inv, J, Kinv_J);

    // 3) Compute J^T* (K^{-1}/rho*k_r) *J
    CeedScalar JT_Kinv_J[3][3];
    AlphaMatTransposeMatMult3x3(1, J, Kinv_J, JT_Kinv_J);

    // 4) Compute v1 = J^T* (K^{-1}/rho*k_r) *J*u*w/detJ
    CeedScalar u1[3] = {u[0][i], u[1][i], u[2][i]}, v1[3];
    AlphaMatVecMult3x3(w[i]/det_J, JT_Kinv_J, u1, v1);

    // 5) -(v, rho*g_u): v2 = -J^T*rho*g_u*w, g_u = g/norm(g)
    CeedScalar g_u[3] = {0., 0., 1.}, v2[3];
    AlphaMatTransposeVecMult3x3(-rho*w[i], J, g_u, v2);

    // Output at quadrature points: (v, K^{-1}/rho*k_r u) -(v, rho*g_u)
    for (CeedInt k = 0; k < 3; k++) {
      v[k][i] = v1[k] + v2[k];
    }
    // Output at quadrature points: -(\div(v), \psi)
    CeedScalar psi = p[i] / (rho_a0 * g);
    div_v[i] = -psi * w[i];
    // Output at quadrature points:-(q, \div(u))  + (q, f)
    q[i] = -div_u[i] * w[i] + f[i+0*Q]*w[i]*det_J;
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for Darcy problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(JacobianDarcySystem3D)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in,
                                      CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*du)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_du) = (const CeedScalar(*))in[3],
                   (*dp) = (const CeedScalar(*))in[4],
                   (*coords) = in[5];

  // Outputs
  CeedScalar (*dv)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_dv) = (CeedScalar(*))out[1],
             (*dq) = (CeedScalar(*))out[2];
  // Context
  DARCYContext  context = (DARCYContext)ctx;
  const CeedScalar  kappa   = context->kappa;
  const CeedScalar rho_a0   = context->rho_a0;
  const CeedScalar g        = context->g;
  const CeedScalar alpha_a  = context->alpha_a;
  const CeedScalar b_a      = context->b_a;
  // *INDENT-ON*

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q], z = coords[i+2*Q];
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar det_J = MatDet3x3(J);

    // *INDENT-OFF*
    // k_r = b_a + alpha_a * (\psi - x2)
    CeedScalar k_r =  b_a + alpha_a*(1-x*y*z);
    // rho = rho_a/rho_a0
    CeedScalar rho = 1.0;
    // (dv, K^{-1}/rho*k_r du): dv = J^T* (K^{-1}/rho*k_r) *J*du*w/detJ
    // 1) Compute K^{-1}, note K = kappa*I
    CeedScalar K[3][3] = {{kappa, 0., 0.},
                          {0., kappa, 0.},
                          {0., 0., kappa}
                         };
    const CeedScalar det_K = MatDet3x3(K);
    CeedScalar K_inv[3][3];
    MatInverse3x3(K, det_K, K_inv);

    // 2) (K^{-1}/rho*k_r) *J
    CeedScalar Kinv_J[3][3];
    AlphaMatMatMult3x3(1/(rho*k_r), K_inv, J, Kinv_J);

    // 3) Compute J^T* (K^{-1}/rho*k_r) *J
    CeedScalar JT_Kinv_J[3][3];
    AlphaMatTransposeMatMult3x3(1, J, Kinv_J, JT_Kinv_J);

    // 4) Compute dv1 = J^T* (K^{-1}/rho*k_r) *J*du*w/detJ
    CeedScalar du1[3] = {du[0][i], du[1][i], du[2][i]}, dv1[3];
    AlphaMatVecMult3x3(w[i]/det_J, JT_Kinv_J, du1, dv1);

    // 5) -(dv, rho*g_u): dv2 = 0

    // Output at quadrature points: (dv, K^{-1}/rho*k_r u) -(dv, rho*g_u)
    for (CeedInt k = 0; k < 3; k++) {
      dv[k][i] = dv1[k];
    }
    // Output at quadrature points: -(\div(dv), d\psi)
    CeedScalar dpsi = dp[i] / (rho_a0 * g);
    div_dv[i] = -dpsi * w[i];
    // Output at quadrature points:-(dq, \div(du))
    dq[i] = -div_du[i] * w[i];
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------

#endif //End of DARCY_SYSTEM3D_H
