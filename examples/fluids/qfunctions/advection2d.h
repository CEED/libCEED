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

#include "utils.h"

typedef struct SetupContextAdv2D_ *SetupContextAdv2D;
struct SetupContextAdv2D_ {
  CeedScalar rc;
  CeedScalar lx;
  CeedScalar ly;
  CeedScalar wind[3];
  CeedScalar time;
  int        wind_type;  // See WindType: 0=ROTATION, 1=TRANSLATION
};

typedef struct AdvectionContext_ *AdvectionContext;
struct AdvectionContext_ {
  CeedScalar CtauS;
  CeedScalar strong_form;
  CeedScalar E_wind;
  bool       implicit;
  int        stabilization;  // See StabilizationType: 0=none, 1=SU, 2=SUPG
};

// *****************************************************************************
// This QFunction sets the initial conditions and the boundary conditions
//   for two test cases: ROTATION and TRANSLATION
//
// -- ROTATION (default)
//      Initial Conditions:
//        Mass Density:
//          Constant mass density of 1.0
//        Momentum Density:
//          Rotational field in x,y
//        Energy Density:
//          Maximum of 1. x0 decreasing linearly to 0. as radial distance
//            increases to (1.-r/rc), then 0. everywhere else
//
//      Boundary Conditions:
//        Mass Density:
//          0.0 flux
//        Momentum Density:
//          0.0
//        Energy Density:
//          0.0 flux
//
// -- TRANSLATION
//      Initial Conditions:
//        Mass Density:
//          Constant mass density of 1.0
//        Momentum Density:
//           Constant rectilinear field in x,y
//        Energy Density:
//          Maximum of 1. x0 decreasing linearly to 0. as radial distance
//            increases to (1.-r/rc), then 0. everywhere else
//
//      Boundary Conditions:
//        Mass Density:
//          0.0 flux
//        Momentum Density:
//          0.0
//        Energy Density:
//          Inflow BCs:
//            E = E_wind
//          Outflow BCs:
//            E = E(boundary)
//          Both In/Outflow BCs for E are applied weakly in the
//            QFunction "Advection2d_Sur"
//
// *****************************************************************************

// *****************************************************************************
// This helper function provides the exact, time-dependent solution
//   and IC formulation for 2D advection
// *****************************************************************************
CEED_QFUNCTION_HELPER CeedInt Exact_Advection2d(CeedInt dim, CeedScalar time, const CeedScalar X[], CeedInt Nf, CeedScalar q[], void *ctx) {
  const SetupContextAdv2D context = (SetupContextAdv2D)ctx;
  const CeedScalar        rc      = context->rc;
  const CeedScalar        lx      = context->lx;
  const CeedScalar        ly      = context->ly;
  const CeedScalar       *wind    = context->wind;

  // Setup
  const CeedScalar center[2] = {0.5 * lx, 0.5 * ly};
  const CeedScalar theta[]   = {M_PI, -M_PI / 3, M_PI / 3};
  const CeedScalar x0[2]     = {center[0] + .25 * lx * cos(theta[0] + time), center[1] + .25 * ly * sin(theta[0] + time)};
  const CeedScalar x1[2]     = {center[0] + .25 * lx * cos(theta[1] + time), center[1] + .25 * ly * sin(theta[1] + time)};
  const CeedScalar x2[2]     = {center[0] + .25 * lx * cos(theta[2] + time), center[1] + .25 * ly * sin(theta[2] + time)};

  const CeedScalar x = X[0], y = X[1];

  // Initial/Boundary Conditions
  switch (context->wind_type) {
    case 0:  // Rotation
      q[0] = 1.;
      q[1] = -(y - center[1]);
      q[2] = (x - center[0]);
      q[3] = 0;
      q[4] = 0;
      break;
    case 1:  // Translation
      q[0] = 1.;
      q[1] = wind[0];
      q[2] = wind[1];
      q[3] = 0;
      q[4] = 0;
      break;
    default:
      return 1;
  }

  CeedScalar r = sqrt(Square(x - x0[0]) + Square(y - x0[1]));
  CeedScalar E = 1 - r / rc;

  if (0) {  // non-smooth initial conditions
    if (q[4] < E) q[4] = E;
    r = sqrt(Square(x - x1[0]) + Square(y - x1[1]));
    if (r <= rc) q[4] = 1;
  }
  r = sqrt(Square(x - x2[0]) + Square(y - x2[1]));
  E = (r <= rc) ? .5 + .5 * cos(r * M_PI / rc) : 0;
  if (q[4] < E) q[4] = E;

  return 0;
}

// *****************************************************************************
// This QFunction sets the initial conditions for 2D advection
// *****************************************************************************
CEED_QFUNCTION(ICsAdvection2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  // Outputs
  CeedScalar(*q0)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SetupContextAdv2D context = (SetupContextAdv2D)ctx;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[]  = {X[0][i], X[1][i]};
    CeedScalar       q[5] = {0.};

    Exact_Advection2d(2, context->time, x, 5, q, ctx);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }  // End of Quadrature Point Loop

  // Return
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
//
// *****************************************************************************
CEED_QFUNCTION(Advection2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar CtauS       = context->CtauS;
  const CeedScalar strong_form = context->strong_form;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
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
    // *INDENT-OFF*
    const CeedScalar du[3][2] = {
        {(dq[0][1][i] - drho[0] * u[0]) / rho, (dq[1][1][i] - drho[1] * u[0]) / rho},
        {(dq[0][2][i] - drho[0] * u[1]) / rho, (dq[1][2][i] - drho[1] * u[1]) / rho},
        {(dq[0][3][i] - drho[0] * u[2]) / rho, (dq[1][3][i] - drho[1] * u[2]) / rho},
    };
    // *INDENT-ON*
    const CeedScalar dE[3] = {
        dq[0][4][i],
        dq[1][4][i],
    };
    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ = q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[2][2] = {
        {q_data[1][i], q_data[2][i]},
        {q_data[3][i], q_data[4][i]},
    };
    // *INDENT-ON*

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

// *****************************************************************************
// This QFunction implements 2D advection (mentioned above) with
//   implicit time stepping method
//
// *****************************************************************************
CEED_QFUNCTION(IFunction_Advection2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar CtauS       = context->CtauS;
  const CeedScalar strong_form = context->strong_form;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
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
    // *INDENT-OFF*
    const CeedScalar du[3][2] = {
        {(dq[0][1][i] - drho[0] * u[0]) / rho, (dq[1][1][i] - drho[1] * u[0]) / rho},
        {(dq[0][2][i] - drho[0] * u[1]) / rho, (dq[1][2][i] - drho[1] * u[1]) / rho},
        {(dq[0][3][i] - drho[0] * u[2]) / rho, (dq[1][3][i] - drho[1] * u[2]) / rho},
    };
    // *INDENT-ON*
    const CeedScalar dE[3] = {
        dq[0][4][i],
        dq[1][4][i],
    };
    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ = q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[2][2] = {
        {q_data[1][i], q_data[2][i]},
        {q_data[3][i], q_data[4][i]},
    };
    // *INDENT-ON*
    // The Physics
    // No Change in density or momentum
    for (CeedInt f = 0; f < 4; f++) {
      for (CeedInt j = 0; j < 2; j++) dv[j][f][i] = 0;
      v[f][i] = wdetJ * q_dot[f][i];
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
    CeedScalar strong_res  = q_dot[4][i] + strong_conv;

    v[4][i] = wdetJ * q_dot[4][i];  // transient part

    // Weak Galerkin convection term: -dv \cdot (E u)
    for (CeedInt j = 0; j < 2; j++) dv[j][4][i] = -wdetJ * (1 - strong_form) * E * (u[0] * dXdx[j][0] + u[1] * dXdx[j][1]);

    // Strong Galerkin convection term: v div(E u)
    v[4][i] += wdetJ * strong_form * strong_conv;

    // Stabilization requires a measure of element transit time in the velocity
    // field u.
    CeedScalar uX[2];
    for (CeedInt j = 0; j < 2; j++) uX[j] = dXdx[j][0] * u[0] + dXdx[j][1] * u[1];
    const CeedScalar TauS = CtauS / sqrt(uX[0] * uX[0] + uX[1] * uX[1]);

    for (CeedInt j = 0; j < 2; j++) switch (context->stabilization) {
        case 0:
          break;
        case 1:
          dv[j][4][i] += wdetJ * TauS * strong_conv * uX[j];
          break;
        case 2:
          dv[j][4][i] += wdetJ * TauS * strong_res * uX[j];
          break;
      }
  }  // End Quadrature Point Loop

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
//    The validity of the weak form of the governing equations is extended
//    to the outflow and the current values of E are applied.
//
//  Inflow BCs:
//    A prescribed Total Energy (E_wind) is applied weakly.
//
// *****************************************************************************
CEED_QFUNCTION(Advection2d_InOutFlow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar E_wind      = context->E_wind;
  const CeedScalar strong_form = context->strong_form;
  const bool       implicit    = context->implicit;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
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
