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
/// Richard problem 2D (quad element) using PETSc

#ifndef RICHARD_SYSTEM2D_H
#define RICHARD_SYSTEM2D_H

#include <math.h>
#include <ceed.h>
#include "ceed/ceed-f64.h"
#include "utils.h"

// See Matthew Farthing, Christopher Kees, Cass Miller (2003)
// https://www.sciencedirect.com/science/article/pii/S0309170802001872
// -----------------------------------------------------------------------------
// Strong form:
//  u        = -rho*k_r*K *[grad(\psi) - rho*g_u]   in \Omega x [0,T]
//  -\div(u) = -f  + d (rho*\theta)/dt              in \Omega x [0,T]
//  p        = p_b                                  on \Gamma_D x [0,T]
//  u.n      = u_b                                  on \Gamma_N x [0,T]
//  p        = p_0                                  in \Omega, t = 0
//
//  Where rho = rho_a/rho_a0, rho_a = rho_a0*exp(\beta * (p - p0)), p0 = 101325 Pa is atmospheric pressure
//  rho_a0 is the density at p_0, g_u = g/norm(g) where g is gravity.
//  k_r = b_a + alpha_a * (\psi - x2), where \psi = p / (rho_a0 * norm(g)) and x2 is vertical axis
//
// Weak form: Find (u, \psi) \in VxQ (V=H(div), Q=L^2) on \Omega
//  (v, K^{-1}/rho*k_r * u) -(v, rho*g_u) -(\div(v), \psi) = -<v, p_b*n>_{\Gamma_D}
// -(q, \div(u))  + (q, f) -(q, d (rho*\theta)/dt ) = 0
//
// We solve MMS for  K = kappa*I and beta=0 ==> rho=1 and \theta = alpha_a*\psi, so
// -(q, d (rho*\theta)/dt ) = -alpha_a*(q, d(\psi)/dt )
//
// This QFunction setup the mixed form of the above equation
// Inputs:
//   w     : weight of quadrature
//   J     : dx/dX. x physical coordinate, X reference coordinate [-1,1]^dim
//   u     : basis_u at quadrature points
// div(u)  : divergence of basis_u at quadrature points
//   p     : basis_p at quadrature points
//   U_t   : time derivative of U = [p, u]
//
// Output:
//   v     : (v, K^{-1}/rho*k_r u) = \int (v^T * K^{-1}/rho*k_r*u detJ*w)dX ==> \int (v^T J^T * K^{-1}/rho*k_r *J*u*w/detJ)dX
//           -(v, rho*g_u)     = \int (v^T * rho*g_u detJ*w)dX ==> \int (v^T J^T * rho*g_u*w) dX
// div(v)  : -(\div(v), \psi) = -\int (div(v)^T * \psi *w) dX
//   q     : -(q, \div(u)) = -\int (q^T * div(u) * w) dX
//            (q, f)       = \int( q^T * f * w*detJ )dX
//            -alpha_a*(q, d\psi/dt) = -alpha_a \int (q^T * \psi_t*w*detJ)dX
//
// -----------------------------------------------------------------------------
#ifndef RICHARD_CTX
#define RICHARD_CTX
typedef struct RICHARDContext_ *RICHARDContext;
struct RICHARDContext_ {
  CeedScalar kappa;
  CeedScalar g;
  CeedScalar rho_a0;
  CeedScalar alpha_a, b_a;
  CeedScalar beta, p0;
  CeedScalar t, t_final, dt;
  CeedScalar gamma;
  CeedScalar lx, ly;
};
#endif
// -----------------------------------------------------------------------------
// Residual evaluation for Richard problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardSystem2D)(void *ctx, CeedInt Q,
                                const CeedScalar *const *in,
                                CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_u) = (const CeedScalar(*))in[3],
                   (*p) = (const CeedScalar(*))in[4],
                   (*f) = in[5],
                   (*coords) = in[6],
                   (*p_t) = (const CeedScalar(*))in[7];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_v) = (CeedScalar(*))out[1],
             (*q) = (CeedScalar(*))out[2];
  // Context
  RICHARDContext  context = (RICHARDContext)ctx;
  const CeedScalar kappa    = context->kappa;
  const CeedScalar rho_a0   = context->rho_a0;
  const CeedScalar g        = context->g;
  const CeedScalar alpha_a  = context->alpha_a;
  const CeedScalar b_a      = context->b_a;
  //const CeedScalar beta     = context->beta;
  //const CeedScalar p0       = context->p0; // atmospheric pressure
  const CeedScalar gamma    = context->gamma;
  CeedScalar t              = context->t;
  //CeedScalar dt              = context->dt;

  // *INDENT-ON*
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q];
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);

    // *INDENT-ON*
    // \psi = p / (rho_a0 * norm(g))
    CeedScalar psi = p[i] / (rho_a0 * g);
    // k_r = b_a + alpha_a * (\psi - x2)
    CeedScalar k_r = b_a + alpha_a*(1 - x*y);
    // rho_a = rho_a0*exp(beta * (p - p0))
    //CeedScalar rho_a = rho_a0 * exp(beta * (p[i] - p0));
    // rho = rho_a/rho_a0
    CeedScalar rho = 1.;

    // (v, K^{-1}/rho*k_r u): v = J^T* (K^{-1}/rho*k_r) *J*u*w/detJ
    // 1) Compute K^{-1}, note K = kappa*I
    CeedScalar K[2][2] = {{kappa, 0.},{0., kappa}};
    const CeedScalar det_K = MatDet2x2(K);
    CeedScalar K_inv[2][2];
    MatInverse2x2(K, det_K, K_inv);

    // 2) (K^{-1}/rho*k_r) *J
    CeedScalar Kinv_J[2][2];
    AlphaMatMatMult2x2(1/(rho*k_r), K_inv, J, Kinv_J);

    // 3) Compute J^T* (K^{-1}/rho*k_r) *J
    CeedScalar JT_Kinv_J[2][2];
    AlphaMatTransposeMatMult2x2(1, J, Kinv_J, JT_Kinv_J);

    // 4) Compute v1 = J^T* (K^{-1}/rho*k_r) *J*u*w/detJ
    CeedScalar u1[2] = {u[0][i], u[1][i]}, v1[2];
    AlphaMatVecMult2x2(w[i]/det_J, JT_Kinv_J, u1, v1);

    // 5) -(v, rho*g_u): v2 = -J^T*rho*g_u*w
    //CeedScalar g_u[2] = {0., 1.}, v2[2];
    //AlphaMatTransposeVecMult2x2(-rho*w[i], J, g_u, v2);

    // Output at quadrature points: (v, k*K^{-1} * u) -(v, rho*g)
    for (CeedInt k = 0; k < 2; k++) {
      v[k][i] = v1[k];// + v2[k];
    }
    // Output at quadrature points: -(\div(v), \psi)
    div_v[i] = -psi * w[i];

    // Output at quadrature points:
    //-(q, \div(u))  + (q, f) - alpha_a * (q, d\psi/dt )
    CeedScalar dpsi_dt = p_t[i] / (rho_a0 * g);
    q[i] = -div_u[i]*w[i] + exp(-gamma*(t))*f[i+0*Q]*w[i]*det_J -
           alpha_a*dpsi_dt*w[i]*det_J;
  } // End of Quadrature Point Loop

  return 0;
}

/*
// -----------------------------------------------------------------------------
// Jacobian evaluation for Richard problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(JacobianRichardSystem2D)(void *ctx, CeedInt Q,
                                        const CeedScalar *const *in,
                                        CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*du)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*div_du) = (const CeedScalar(*))in[3],
                   (*dp) = (const CeedScalar(*))in[4],
                   (*coords) = in[5],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[6],
                   (*p) = (const CeedScalar(*))in[7];

  // Outputs
  CeedScalar (*dv)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*div_dv) = (CeedScalar(*))out[1],
             (*dq) = (CeedScalar(*))out[2];
  // Context
  RICHARDContext  context = (RICHARDContext)ctx;
  const CeedScalar kappa  = context->kappa;
  const CeedScalar alpha_a = context->alpha_a;
  const CeedScalar b_a     = context->b_a;
  const CeedScalar rho_a0   = context->rho_a0;
  const CeedScalar beta    = context->beta;
  const CeedScalar g       = context->g;
  const CeedScalar p0      = context->p0;// atmospheric pressure
  // *INDENT-ON*

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    CeedScalar y = coords[i+1*Q];
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);

    // *INDENT-ON*
    // psi = p / (rho_a0 * norm(g))
    CeedScalar psi = p[i] / (rho_a0 * g);
    // k_r = b_a + alpha_a * (psi - x2)
    CeedScalar k_r = b_a + alpha_a * (psi - y);
    // rho = rho_a0*exp(beta * (p - p0))
    CeedScalar rho = rho_a0 * exp(beta * (p[i] - p0));
    //k = rho_a0^2*norm(g)/(rho*k_r)
    CeedScalar k = rho_a0 * rho_a0 * g / (rho * k_r);

    // Piola map: J^T*k*K^{-1}*J*u*w/detJ
    // The jacobian term
    // dv = J^T* (k*K^{-1}) *J*du*w/detJ - [(rho*k_r),p*dp/(rho*k_r)]*J^T*(k*K^{-1}) *J*u*w/detJ
    //      -J^T * (beta*rho*g)*dp
    // 1) Compute K^{-1}, note K = kappa*I
    CeedScalar K[2][2] = {{kappa, 0.},{0., kappa}};
    const CeedScalar det_K = MatDet2x2(K);
    CeedScalar K_inv[2][2];
    MatInverse2x2(K, det_K, K_inv);

    // 2) Compute k*K^{-1}*J
    CeedScalar kKinv_J[2][2];
    AlphaMatMatMult2x2(k, K_inv, J, kKinv_J);

    // 3) Compute J^T * (k*K^{-1}*J)
    CeedScalar JT_kKinv_J[2][2];
    AlphaMatTransposeMatMult2x2(1, J, kKinv_J, JT_kKinv_J);

    // 4) Compute (J^T*k*K^{-1}*J) * du * w /detJ
    CeedScalar du1[2] = {du[0][i], du[1][i]}, dv1[2];
    AlphaMatVecMult2x2(w[i]/det_J, JT_kKinv_J, du1, dv1);

    // 5) Compute -(rho*k_r),p*dp/(rho*k_r))
    // (rho*k_r),p*dp = beta*rho*dp*k_r + rho*alpha*dp/(rho_a0*norm(g))
    CeedScalar d_rhokr_dp = -(beta + alpha_a/(rho_a0*g*k_r))*dp[i];

    // 6) -[(rho*k_r),p*dp/(rho*k_r)]*J^T*(k*K^{-1}) *J*u*w/detJ
    CeedScalar u1[2] = {u[0][i], u[1][i]}, dv2[2];
    AlphaMatVecMult2x2((d_rhokr_dp*w[i])/det_J, JT_kKinv_J, u1, dv2);

    // 7) -(v, rho*g): dv = -J^T * (beta*rho*g*dp)*w
    CeedScalar drho_g_dp[2] = {0., beta *rho *g *dp[i]}, dv3[2];
    AlphaMatTransposeVecMult2x2(-w[i], J, drho_g_dp, dv3);

    // Output at quadrature points
    for (CeedInt k = 0; k < 2; k++) {
      dv[k][i] = dv1[k] + dv2[k] + dv3[k];
    }

    div_dv[i] = -dp[i] * w[i];
    dq[i] = -div_du[i] * w[i];
  } // End of Quadrature Point Loop

  return 0;
}
*/
// -----------------------------------------------------------------------------

#endif //End of RICHARD_SYSTEM2D_H
