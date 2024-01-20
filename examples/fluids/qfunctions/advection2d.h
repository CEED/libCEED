// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Advection initial condition and operator for Navier-Stokes example using PETSc

#ifndef advection2d_h
#define advection2d_h

#include <ceed.h>
#include <math.h>

#include "advection_generic.h"
#include "advection_types.h"
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "stabilization_types.h"
#include "utils.h"

// *****************************************************************************
// This QFunction sets the initial conditions for 2D advection
// *****************************************************************************
CEED_QFUNCTION(ICsAdvection2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  CeedScalar(*q0)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SetupContextAdv context    = (SetupContextAdv)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[]  = {X[0][i], X[1][i]};
    CeedScalar       q[5] = {0.};

    Exact_AdvectionGeneric(2, context->time, x, 5, q, ctx);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of the advection equation
//
// This is 2D advection given in two formulations based upon the weak form.
//
// State Variables: q = ( rho, U1, U2, E )
//   rho - Mass Density
//   Ui  - Momentum Density    ,  Ui = rho ui
//   E   - Total Energy Density
//
// Advection Equation:
//   dE/dt + div( E u ) = 0
// *****************************************************************************
CEED_QFUNCTION(Advection2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*dq)[5][CEED_Q_VLA]  = (const CeedScalar(*)[5][CEED_Q_VLA])in[1];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar CtauS       = context->CtauS;
  const bool       strong_form = context->strong_form;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho  = q[0][i];
    const CeedScalar u[3] = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E    = q[4][i];
    // -- Grad in
    const CeedScalar drho[2] = {
        dq[0][0][i],
        dq[1][0][i],
    };
    const CeedScalar du[3][2] = {
        {(dq[0][1][i] - drho[0] * u[0]) / rho, (dq[1][1][i] - drho[1] * u[0]) / rho},
        {(dq[0][2][i] - drho[0] * u[1]) / rho, (dq[1][2][i] - drho[1] * u[1]) / rho},
        {(dq[0][3][i] - drho[0] * u[2]) / rho, (dq[1][3][i] - drho[1] * u[2]) / rho},
    };
    const CeedScalar dE[3] = {
        dq[0][4][i],
        dq[1][4][i],
    };
    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ = q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    const CeedScalar dXdx[2][2] = {
        {q_data[1][i], q_data[2][i]},
        {q_data[3][i], q_data[4][i]},
    };

    // The Physics

    // No Change in density or momentum
    for (CeedInt f = 0; f < 4; f++) {
      for (CeedInt j = 0; j < 2; j++) dv[j][f][i] = 0;
      v[f][i] = 0;
    }

    // -- Total Energy
    // Evaluate the strong form using div(E u) = u . grad(E) + E div(u)
    // or in index notation: (u_j E)_{,j} = u_j E_j + E u_{j,j}
    CeedScalar div_u = 0, u_dot_grad_E = 0;
    for (CeedInt j = 0; j < 2; j++) {
      CeedScalar dEdx_j = 0;
      for (CeedInt k = 0; k < 2; k++) {
        div_u += du[j][k] * dXdx[k][j];  // u_{j,j} = u_{j,K} X_{K,j}
        dEdx_j += dE[k] * dXdx[k][j];
      }
      u_dot_grad_E += u[j] * dEdx_j;
    }
    CeedScalar strong_conv = E * div_u + u_dot_grad_E;

    // Weak Galerkin convection term: dv \cdot (E u)
    for (CeedInt j = 0; j < 2; j++) dv[j][4][i] = (1 - strong_form) * wdetJ * E * (u[0] * dXdx[j][0] + u[1] * dXdx[j][1]);
    v[4][i] = 0;

    // Strong Galerkin convection term: - v div(E u)
    v[4][i] = -strong_form * wdetJ * strong_conv;

    // Stabilization requires a measure of element transit time in the velocity
    // field u.
    CeedScalar uX[2];
    for (CeedInt j = 0; j < 2; j++) uX[j] = dXdx[j][0] * u[0] + dXdx[j][1] * u[1];
    const CeedScalar TauS = CtauS / sqrt(uX[0] * uX[0] + uX[1] * uX[1]);
    for (CeedInt j = 0; j < 2; j++) dv[j][4][i] -= wdetJ * TauS * strong_conv * uX[j];
  }  // End Quadrature Point Loop

  return 0;
}

CEED_QFUNCTION(IFunction_Advection2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  IFunction_AdvectionGeneric(ctx, Q, in, out, 2);
  return 0;
}

// *****************************************************************************
// This QFunction implements consistent outflow and inflow BCs
//      for 2D advection
//
//  Inflow and outflow faces are determined based on sign(dot(wind, normal)):
//    sign(dot(wind, normal)) > 0 : outflow BCs
//    sign(dot(wind, normal)) < 0 : inflow BCs
//
//  Outflow BCs:
//    The validity of the weak form of the governing equations is extended to the outflow and the current values of E are applied.
//
//  Inflow BCs:
//    A prescribed Total Energy (E_wind) is applied weakly.
// *****************************************************************************
CEED_QFUNCTION(Advection2d_InOutFlow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]   = (CeedScalar(*)[CEED_Q_VLA])out[0];
  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar E_wind      = context->E_wind;
  const CeedScalar strong_form = context->strong_form;
  const bool       implicit    = context->implicit;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho  = q[0][i];
    const CeedScalar u[3] = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E    = q[4][i];

    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // ---- Normal vectors
    const CeedScalar norm[2] = {q_data_sur[1][i], q_data_sur[2][i]};
    // Normal velocity
    const CeedScalar u_normal = norm[0] * u[0] + norm[1] * u[1];

    // No Change in density or momentum
    for (CeedInt j = 0; j < 4; j++) {
      v[j][i] = 0;
    }

    // Implementing in/outflow BCs
    if (u_normal > 0) {  // outflow
      v[4][i] = -(1 - strong_form) * wdetJb * E * u_normal;
    } else {  // inflow
      v[4][i] = -(1 - strong_form) * wdetJb * E_wind * u_normal;
    }
  }  // End Quadrature Point Loop
  return 0;
}
// *****************************************************************************

#endif  // advection2d_h
