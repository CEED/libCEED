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

#ifndef RICHARD_TRUE3D_H
#define RICHARD_TRUE3D_H

#include <math.h>
#include <ceed.h>
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
  CeedScalar lx, ly, lz;
};
#endif

// -----------------------------------------------------------------------------
// True solution for Richard problem
// -----------------------------------------------------------------------------
CEED_QFUNCTION(RichardTrue3D)(void *ctx, const CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*coords) = in[0];
  // Outputs
  CeedScalar (*true_force) = out[0], (*true_solution) = out[1];
  // Context
  RICHARDContext  context = (RICHARDContext)ctx;
  const CeedScalar kappa    = context->kappa;
  const CeedScalar alpha_a  = context->alpha_a;
  const CeedScalar b_a      = context->b_a;
  const CeedScalar gamma    = context->gamma;
  CeedScalar t_final        = context->t_final;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q], z = coords[i+2*Q];
    // psi = exp(-gamma*t)*sin(pi*x)*sin(pi*y)
    // We factor exp() term
    CeedScalar psi    = sin(PI_DOUBLE*x)*sin(PI_DOUBLE*y)*sin(PI_DOUBLE*z);
    CeedScalar psi_x  = PI_DOUBLE*cos(PI_DOUBLE*x)*sin(PI_DOUBLE*y)*sin(PI_DOUBLE*z);
    CeedScalar psi_xx  = -PI_DOUBLE*PI_DOUBLE*psi;
    CeedScalar psi_y  = PI_DOUBLE*sin(PI_DOUBLE*x)*cos(PI_DOUBLE*y)*sin(PI_DOUBLE*z);
    CeedScalar psi_yy  = -PI_DOUBLE*PI_DOUBLE*psi;
    CeedScalar psi_z  = PI_DOUBLE*sin(PI_DOUBLE*x)*sin(PI_DOUBLE*y)*cos(PI_DOUBLE*z);
    CeedScalar psi_zz  = -PI_DOUBLE*PI_DOUBLE*psi;
    // k_r = b_a + alpha_a * (1 - x*y)
    CeedScalar k_r = b_a + alpha_a*(1-x*y*z);
    CeedScalar k_rx = -alpha_a*y*z;
    CeedScalar k_ry = -alpha_a*x*z;
    CeedScalar k_rz = -alpha_a*x*y;
    // rho = rho_a/rho_a0
    CeedScalar rho = 1.;
    // u = -rho*k_r*K *[grad(\psi)]
    CeedScalar u[3] = {-rho*kappa*exp(-gamma*t_final)*k_r*psi_x, 
                       -rho*kappa*exp(-gamma*t_final)*k_r*psi_y,
                       -rho*kappa*exp(-gamma*t_final)*k_r*psi_z};
    //CeedScalar div_u = -rho*kappa*exp(-gamma*t_final)*(k_rx*psi_x + k_r*psi_xx +
    //                                                     k_ry*psi_y + k_r*psi_yy);
    CeedScalar div_u = -rho*kappa*(k_rx*psi_x + k_r*psi_xx +
                                   k_ry*psi_y + k_r*psi_yy +
                                   k_rz*psi_z + k_r*psi_zz );
    // True Force: f = \div(u) + d (rho*theta)/dt
    // since the force is a function of time, and this qfunction get called once
    // and the t variable doesn't get updated, we factored exp() term and update it
    // in residual, thats why we have psi = exp() * psi1, ...
    true_force[i+0*Q] = div_u -alpha_a*gamma*psi;
    // True Solution
    true_solution[i+0*Q] = exp(-gamma*t_final)*psi;
    true_solution[i+1*Q] = u[0];
    true_solution[i+2*Q] = u[1];
    true_solution[i+3*Q] = u[2];
  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif //End of RICHARD_TRUE2D_H
