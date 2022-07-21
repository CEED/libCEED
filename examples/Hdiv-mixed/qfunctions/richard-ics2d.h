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
/// Force of Richard problem 2D (quad element) using PETSc

#ifndef RICHARD_ICS2D_H
#define RICHARD_ICS2D_H

#include <math.h>
#include <ceed.h>
#include "ceed/ceed-f64.h"
#include "utils.h"

// See Matthew Farthing, Christopher Kees, Cass Miller (2003)
// https://www.sciencedirect.com/science/article/pii/S0309170802001872
// -----------------------------------------------------------------------------
// Strong form:
//  u        = -rho*k_r*K *[grad(\psi) - rho*g_u]   in \Omega x [0,T]
//  -\div(u) = -f  + d (rho*theta)/dt              in \Omega x [0,T]
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
// This QFunction setup the true solution and forcing f of the above equation
// Inputs:
//   coords: physical coordinate
//
// Output:
//   true_force     : = div(u) + d (rho*theta)/dt
//   true_solution  : = [\psi, u] where \psi, u are the exact solution solution
// -----------------------------------------------------------------------------
// We have 3 experiment parameters as described in Table 1:P1, P2, P3
// Matthew Farthing, Christopher Kees, Cass Miller (2003)
// https://www.sciencedirect.com/science/article/pii/S0309170802001872
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
};
#endif

// -----------------------------------------------------------------------------
// We solve (v, u) = (v, ue) at t=0, to project ue to Hdiv space
// This QFunction create rhs_u0 = (v, ue)
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardRhsU02D)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*coords) = in[1],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*rhs_u0) = out[0];
  // Context
  RICHARDContext  context = (RICHARDContext)ctx;
  const CeedScalar kappa    = context->kappa;
  const CeedScalar alpha_a  = context->alpha_a;
  const CeedScalar b_a      = context->b_a;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q];
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    // psi = exp(-gamma*t)*sin(pi*x)*sin(pi*y)
    CeedScalar psi1_x = PI_DOUBLE*cos(PI_DOUBLE*x)*sin(PI_DOUBLE*y);
    CeedScalar psi1_y = PI_DOUBLE*sin(PI_DOUBLE*x)*cos(PI_DOUBLE*y);

    // k_r = b_a + alpha_a * (1 - x*y)
    CeedScalar k_r = b_a + alpha_a*(1-x*y);
    // rho = rho_a/rho_a0
    CeedScalar rho = 1.;
    // ue = -rho*k_r*K *[grad(\psi)]
    CeedScalar ue[2] = {-rho*k_r*kappa*psi1_x,
                        -rho*k_r*kappa*psi1_y};
    CeedScalar rhs1[2];
    // rhs = (v, ue) = J^T*ue*w
    AlphaMatTransposeVecMult2x2(w[i], J, ue, rhs1);
    // 
    rhs_u0[i+0*Q] = rhs1[0];
    rhs_u0[i+1*Q] = rhs1[1];
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// We solve (v, u) = (v, ue) at t=0, to project ue to Hdiv space
// This QFunction create mass matrix (v, u), then we solve using ksp to have 
// projected ue in Hdiv space and use it for initial conditions 
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardICsU2D)(void *ctx, const CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);

    // *INDENT-ON*
    // (v, u): v1 = J^T*J*u*w/detJ
    // 1) Compute J^T *J
    CeedScalar JT_J[2][2];
    AlphaMatTransposeMatMult2x2(1, J, J, JT_J);

    // 4) Compute v1 = J^T*J*u*w/detJ
    CeedScalar u1[2] = {u[0][i], u[1][i]}, v1[2];
    AlphaMatVecMult2x2(w[i]/det_J, JT_J, u1, v1);

    // Output at quadrature points: (v, K^{-1}/rho*k_r u)
    for (CeedInt k = 0; k < 2; k++) {
      v[k][i] = v1[k];
    }
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// We solve (q, p) = (q, pe) at t=0, to project pe to L2 space
// This QFunction create rhs_p0 = (q, pe)
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardRhsP02D)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*coords) = in[1],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*rhs_p0) = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q];
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);
    // psi = exp(-gamma*t)*sin(pi*x)*sin(pi*y)
    CeedScalar psi1 = sin(PI_DOUBLE*x)*sin(PI_DOUBLE*y);

    // rhs = (q, pe) = pe*w*det_J
    rhs_p0[i] = psi1*w[i]*det_J;
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// We solve (q, p) = (q, pe) at t=0, to project pe to L2 space
// This QFunction create mass matrix (q, p), then we solve using ksp to have 
// projected pe in L2 space and use it for initial conditions 
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardICsP2D)(void *ctx, const CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*p) = (const CeedScalar(*))in[2];

  // Outputs
  CeedScalar (*q) = (CeedScalar(*))out[0];
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);

    // Output at quadrature points: (q, p)
    q[i] = p[i]*w[i]*det_J;
  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif //End of RICHARD_ICS2D_H
