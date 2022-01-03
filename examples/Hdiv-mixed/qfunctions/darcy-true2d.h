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
/// Compute true solution of the H(div) example using PETSc

#ifndef DARCY_TRUE2D_H
#define DARCY_TRUE2D_H

#include <math.h>
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
// We solve MMS for  K = kappa*I and beta=0 ==> rho=1 and \theta = alpha_a*\psi, so
//
// This QFunction setup the true solution and forcing f of the above equation
// Inputs:
//   coords: physical coordinate
//
// Output:
//   true_force     : = div(u)
//   true_solution  : = [\psi, u] where \psi, u are the exact solution solution
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
CEED_QFUNCTION(DarcyTrue2D)(void *ctx, const CeedInt Q,
                            const CeedScalar *const *in,
                            CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*coords) = in[0]; 
  // Outputs
  CeedScalar (*true_force) = out[0], (*true_solution) = out[1];
  // Context
  DARCYContext  context = (DARCYContext)ctx;
  const CeedScalar kappa    = context->kappa;
  const CeedScalar alpha_a  = context->alpha_a;
  const CeedScalar b_a      = context->b_a;
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q];  
    CeedScalar psi    = sin(PI_DOUBLE*x)*sin(PI_DOUBLE*y);
    CeedScalar psi_x  = PI_DOUBLE*cos(PI_DOUBLE*x)*sin(PI_DOUBLE*y);
    CeedScalar psi_y  = PI_DOUBLE*sin(PI_DOUBLE*x)*cos(PI_DOUBLE*y);

    // k_r = b_a + alpha_a * (1 - x*y)
    CeedScalar k_r = b_a + alpha_a*(1-x*y);
    // rho = rho_a/rho_a0
    CeedScalar rho = 1.;
    // u = -rho*k_r*K *[grad(\psi) - rho*g_u]
    CeedScalar u[2] = {-rho*k_r*kappa*psi_x, -rho*k_r*kappa*(psi_y-1)};
    CeedScalar div_u = -rho*kappa*(-alpha_a*y*psi_x - k_r*PI_DOUBLE*PI_DOUBLE*psi
                                   -alpha_a*x*psi_y - k_r*PI_DOUBLE*PI_DOUBLE*psi);

    // True Force: f = \div(u)
    true_force[i+0*Q] = div_u;
    // True Solution
    true_solution[i+0*Q] = psi;
    true_solution[i+1*Q] = u[0];
    true_solution[i+2*Q] = u[1];
  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End DARCY_TRUE2D_H
