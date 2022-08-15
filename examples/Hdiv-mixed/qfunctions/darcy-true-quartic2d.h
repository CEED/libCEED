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

#ifndef DARCY_TRUE_QUARTIC2D_H
#define DARCY_TRUE_QUARTIC2D_H

#include <math.h>
#include <ceed.h>
#include "utils.h"

// -----------------------------------------------------------------------------
// See Matthew Farthing, Christopher Kees, Cass Miller (2003)
// https://www.sciencedirect.com/science/article/pii/S0309170802001872
// -----------------------------------------------------------------------------
// Strong form:
//  u       = -\grad(psi)  on \Omega
//  \div(u) = f              on \Omega
//  p = p0                   on \Gamma_D
//  u.n = g                  on \Gamma_N
// Weak form: Find (u,p) \in VxQ (V=H(div), Q=L^2) on \Omega
//  (v, u) - (\div(v), psi) = -<v, p0 n>_{\Gamma_D}
// -(q, \div(u))            = -(q, f)
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
  CeedScalar lx, ly;
};
#endif
CEED_QFUNCTION(DarcyTrueQuartic2D)(void *ctx, const CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*coords) = in[0]; 
  // Outputs
  CeedScalar (*true_force) = out[0], (*true_solution) = out[1];
  // Context
  DARCYContext  context = (DARCYContext)ctx;
  const CeedScalar lx   = context->lx;
  const CeedScalar ly   = context->ly;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q];  
    CeedScalar psi    = x*(lx-x)*y*(ly-y);
    CeedScalar psi_x  = (lx-2*x)*y*(ly-y);
    CeedScalar psi_xx  = -2*y*(ly-y);
    CeedScalar psi_y  = x*(lx-x)*(ly-2*y);
    CeedScalar psi_yy  = -2*x*(lx-x);

    // ue = -grad(\psi)
    CeedScalar ue[2] = {-psi_x, -psi_y};
    // f = \div(u)
    CeedScalar div_u = -psi_xx - psi_yy;
    // True Force: f = \div(u)
    true_force[i+0*Q] = div_u;
    // True Solution
    true_solution[i+0*Q] = psi;
    true_solution[i+1*Q] = ue[0];
    true_solution[i+2*Q] = ue[1];
  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End DARCY_TRUE_QUARTIC2D_H
