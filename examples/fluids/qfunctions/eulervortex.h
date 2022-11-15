// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Euler traveling vortex initial condition and operator for Navier-Stokes
/// example using PETSc

// Model from:
//   On the Order of Accuracy and Numerical Performance of Two Classes of
//   Finite Volume WENO Schemes, Zhang, Zhang, and Shu (2011).

#ifndef eulervortex_h
#define eulervortex_h

#include <ceed.h>
#include <math.h>

#include "utils.h"

typedef struct EulerContext_ *EulerContext;
struct EulerContext_ {
  CeedScalar center[3];
  CeedScalar curr_time;
  CeedScalar vortex_strength;
  CeedScalar c_tau;
  CeedScalar mean_velocity[3];
  bool       implicit;
  int        euler_test;
  int        stabilization;  // See StabilizationType: 0=none, 1=SU, 2=SUPG
};

// *****************************************************************************
// This function sets the initial conditions
//
//   Temperature:
//     T   = 1 - (gamma - 1) vortex_strength**2 exp(1 - r**2) / (8 gamma pi**2)
//   Density:
//     rho = (T/S_vortex)^(1 / (gamma - 1))
//   Pressure:
//     P   = rho * T
//   Velocity:
//     ui  = 1 + vortex_strength exp((1 - r**2)/2.) [yc - y, x - xc] / (2 pi)
//     r   = sqrt( (x - xc)**2 + (y - yc)**2 )
//   Velocity/Momentum Density:
//     Ui  = rho ui
//   Total Energy:
//     E   = P / (gamma - 1) + rho (u u)/2
//
// Constants:
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   vortex_strength ,  Strength of vortex
//   center          ,  Location of bubble center
//   gamma  = cp / cv,  Specific heat ratio
//
// *****************************************************************************

// *****************************************************************************
// This helper function provides support for the exact, time-dependent solution
//   (currently not implemented) and IC formulation for Euler traveling vortex
// *****************************************************************************
CEED_QFUNCTION_HELPER int Exact_Euler(CeedInt dim, CeedScalar time, const CeedScalar X[], CeedInt Nf, CeedScalar q[], void *ctx) {
  // Context
  const EulerContext context         = (EulerContext)ctx;
  const CeedScalar   vortex_strength = context->vortex_strength;
  const CeedScalar  *center          = context->center;  // Center of the domain
  const CeedScalar  *mean_velocity   = context->mean_velocity;

  // Setup
  const CeedScalar gamma = 1.4;
  const CeedScalar cv    = 2.5;
  const CeedScalar R     = 1.;
  const CeedScalar x = X[0], y = X[1];  // Coordinates
  // Vortex center
  const CeedScalar xc = center[0] + mean_velocity[0] * time;
  const CeedScalar yc = center[1] + mean_velocity[1] * time;

  const CeedScalar x0       = x - xc;
  const CeedScalar y0       = y - yc;
  const CeedScalar r        = sqrt(x0 * x0 + y0 * y0);
  const CeedScalar C        = vortex_strength * exp((1. - r * r) / 2.) / (2. * M_PI);
  const CeedScalar delta_T  = -(gamma - 1.) * vortex_strength * vortex_strength * exp(1 - r * r) / (8. * gamma * M_PI * M_PI);
  const CeedScalar S_vortex = 1;  // no perturbation in the entropy P / rho^gamma
  const CeedScalar S_bubble = (gamma - 1.) * vortex_strength * vortex_strength / (8. * gamma * M_PI * M_PI);
  CeedScalar       rho, P, T, E, u[3] = {0.};

  // Initial Conditions
  switch (context->euler_test) {
    case 0:  // Traveling vortex
      T = 1 + delta_T;
      // P = rho * T
      // P = S * rho^gamma
      // Solve for rho, then substitute for P
      rho  = pow(T / S_vortex, 1 / (gamma - 1.));
      P    = rho * T;
      u[0] = mean_velocity[0] - C * y0;
      u[1] = mean_velocity[1] + C * x0;

      // Assign exact solution
      q[0] = rho;
      q[1] = rho * u[0];
      q[2] = rho * u[1];
      q[3] = rho * u[2];
      q[4] = P / (gamma - 1.) + rho * (u[0] * u[0] + u[1] * u[1]) / 2.;
      break;
    case 1:  // Constant zero velocity, density constant, total energy constant
      rho = 1.;
      E   = 2.;

      // Assign exact solution
      q[0] = rho;
      q[1] = rho * u[0];
      q[2] = rho * u[1];
      q[3] = rho * u[2];
      q[4] = E;
      break;
    case 2:  // Constant nonzero velocity, density constant, total energy constant
      rho  = 1.;
      E    = 2.;
      u[0] = mean_velocity[0];
      u[1] = mean_velocity[1];

      // Assign exact solution
      q[0] = rho;
      q[1] = rho * u[0];
      q[2] = rho * u[1];
      q[3] = rho * u[2];
      q[4] = E;
      break;
    case 3:  // Velocity zero, pressure constant
      // (so density and internal energy will be non-constant),
      // but the velocity should stay zero and the bubble won't diffuse
      // (for Euler, where there is no thermal conductivity)
      P   = 1.;
      T   = 1. - S_bubble * exp(1. - r * r);
      rho = P / (R * T);

      // Assign exact solution
      q[0] = rho;
      q[1] = rho * u[0];
      q[2] = rho * u[1];
      q[3] = rho * u[2];
      q[4] = rho * (cv * T + (u[0] * u[0] + u[1] * u[1]) / 2.);
      break;
    case 4:  // Constant nonzero velocity, pressure constant
      // (so density and internal energy will be non-constant),
      // it should be transported across the domain, but velocity stays constant
      P    = 1.;
      T    = 1. - S_bubble * exp(1. - r * r);
      rho  = P / (R * T);
      u[0] = mean_velocity[0];
      u[1] = mean_velocity[1];

      // Assign exact solution
      q[0] = rho;
      q[1] = rho * u[0];
      q[2] = rho * u[1];
      q[3] = rho * u[2];
      q[4] = rho * (cv * T + (u[0] * u[0] + u[1] * u[1]) / 2.);
      break;
    case 5:  // non-smooth thermal bubble - cylinder
      P    = 1.;
      T    = 1. - (r < 1. ? S_bubble : 0.);
      rho  = P / (R * T);
      u[0] = mean_velocity[0];
      u[1] = mean_velocity[1];

      // Assign exact solution
      q[0] = rho;
      q[1] = rho * u[0];
      q[2] = rho * u[1];
      q[3] = rho * u[2];
      q[4] = rho * (cv * T + (u[0] * u[0] + u[1] * u[1]) / 2.);
      break;
  }
  // Return
  return 0;
}

// *****************************************************************************
// Helper function for computing flux Jacobian
// *****************************************************************************
CEED_QFUNCTION_HELPER void ConvectiveFluxJacobian_Euler(CeedScalar dF[3][5][5], const CeedScalar rho, const CeedScalar u[3], const CeedScalar E,
                                                        const CeedScalar gamma) {
  CeedScalar u_sq = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];  // Velocity square
  for (CeedInt i = 0; i < 3; i++) {                           // Jacobian matrices for 3 directions
    for (CeedInt j = 0; j < 3; j++) {                         // Rows of each Jacobian matrix
      dF[i][j + 1][0] = ((i == j) ? ((gamma - 1.) * (u_sq / 2.)) : 0.) - u[i] * u[j];
      for (CeedInt k = 0; k < 3; k++) {  // Columns of each Jacobian matrix
        dF[i][0][k + 1]     = ((i == k) ? 1. : 0.);
        dF[i][j + 1][k + 1] = ((j == k) ? u[i] : 0.) + ((i == k) ? u[j] : 0.) - ((i == j) ? u[k] : 0.) * (gamma - 1.);
        dF[i][4][k + 1]     = ((i == k) ? (E * gamma / rho - (gamma - 1.) * u_sq / 2.) : 0.) - (gamma - 1.) * u[i] * u[k];
      }
      dF[i][j + 1][4] = ((i == j) ? (gamma - 1.) : 0.);
    }
    dF[i][4][0] = u[i] * ((gamma - 1.) * u_sq - E * gamma / rho);
    dF[i][4][4] = u[i] * gamma;
  }
}

// *****************************************************************************
// Helper function for computing Tau elements (stabilization constant)
//   Model from:
//     Stabilized Methods for Compressible Flows, Hughes et al 2010
//
//   Spatial criterion #2 - Tau is a 3x3 diagonal matrix
//   Tau[i] = c_tau h[i] Xi(Pe) / rho(A[i]) (no sum)
//
// Where
//   c_tau     = stabilization constant (0.5 is reported as "optimal")
//   h[i]      = 2 length(dxdX[i])
//   Pe        = Peclet number ( Pe = sqrt(u u) / dot(dXdx,u) diffusivity )
//   Xi(Pe)    = coth Pe - 1. / Pe (1. at large local Peclet number )
//   rho(A[i]) = spectral radius of the convective flux Jacobian i,
//               wave speed in direction i
// *****************************************************************************
CEED_QFUNCTION_HELPER void Tau_spatial(CeedScalar Tau_x[3], const CeedScalar dXdx[3][3], const CeedScalar u[3], const CeedScalar sound_speed,
                                       const CeedScalar c_tau) {
  for (CeedInt i = 0; i < 3; i++) {
    // length of element in direction i
    CeedScalar h = 2 / sqrt(dXdx[0][i] * dXdx[0][i] + dXdx[1][i] * dXdx[1][i] + dXdx[2][i] * dXdx[2][i]);
    // fastest wave in direction i
    CeedScalar fastest_wave = fabs(u[i]) + sound_speed;
    Tau_x[i]                = c_tau * h / fastest_wave;
  }
}

// *****************************************************************************
// This QFunction sets the initial conditions for Euler traveling vortex
// *****************************************************************************
CEED_QFUNCTION(ICsEuler)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const EulerContext context  = (EulerContext)ctx;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[]  = {X[0][i], X[1][i], X[2][i]};
    CeedScalar       q[5] = {0.};

    Exact_Euler(3, context->curr_time, x, 5, q, ctx);

    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }  // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of Euler equations
//   with explicit time stepping method
//
// This is 3D Euler for compressible gas dynamics in conservation
//   form with state variables of density, momentum density, and total
//   energy density.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density,      Ui = rho ui
//   E   - Total Energy Density,  E  = P / (gamma - 1) + rho (u u)/2
//
// Euler Equations:
//   drho/dt + div( U )                   = 0
//   dU/dt   + div( rho (u x u) + P I3 )  = 0
//   dE/dt   + div( (E + P) u )           = 0
//
// Equation of State:
//   P = (gamma - 1) (E - rho (u u) / 2)
//
// Constants:
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   g               ,  Gravity
//   gamma  = cp / cv,  Specific heat ratio
// *****************************************************************************
CEED_QFUNCTION(Euler)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  EulerContext     context = (EulerContext)ctx;
  const CeedScalar c_tau   = context->c_tau;
  const CeedScalar gamma   = 1.4;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    // *INDENT-OFF*
    // Setup
    // -- Interp in
    const CeedScalar rho      = q[0][i];
    const CeedScalar u[3]     = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E        = q[4][i];
    const CeedScalar drho[3]  = {dq[0][0][i], dq[1][0][i], dq[2][0][i]};
    const CeedScalar dU[3][3] = {
        {dq[0][1][i], dq[1][1][i], dq[2][1][i]},
        {dq[0][2][i], dq[1][2][i], dq[2][2][i]},
        {dq[0][3][i], dq[1][3][i], dq[2][3][i]}
    };
    const CeedScalar dE[3] = {dq[0][4][i], dq[1][4][i], dq[2][4][i]};
    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ = q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*
    // dU/dx
    CeedScalar drhodx[3]       = {0.};
    CeedScalar dEdx[3]         = {0.};
    CeedScalar dUdx[3][3]      = {{0.}};
    CeedScalar dXdxdXdxT[3][3] = {{0.}};
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        drhodx[j] += drho[k] * dXdx[k][j];
        dEdx[j] += dE[k] * dXdx[k][j];
        for (CeedInt l = 0; l < 3; l++) {
          dUdx[j][k] += dU[j][l] * dXdx[l][k];
          dXdxdXdxT[j][k] += dXdx[j][l] * dXdx[k][l];  // dXdx_j,k * dXdx_k,j
        }
      }
    }
    // Pressure
    const CeedScalar E_kinetic = 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), E_internal = E - E_kinetic,
                     P = E_internal * (gamma - 1.);  // P = pressure

    // The Physics
    // Zero v and dv so all future terms can safely sum into it
    for (CeedInt j = 0; j < 5; j++) {
      v[j][i] = 0.;
      for (CeedInt k = 0; k < 3; k++) dv[k][j][i] = 0.;
    }

    // -- Density
    // ---- u rho
    for (CeedInt j = 0; j < 3; j++) dv[j][0][i] += wdetJ * (rho * u[0] * dXdx[j][0] + rho * u[1] * dXdx[j][1] + rho * u[2] * dXdx[j][2]);
    // -- Momentum
    // ---- rho (u x u) + P I3
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        dv[k][j + 1][i] += wdetJ * ((rho * u[j] * u[0] + (j == 0 ? P : 0.)) * dXdx[k][0] + (rho * u[j] * u[1] + (j == 1 ? P : 0.)) * dXdx[k][1] +
                                    (rho * u[j] * u[2] + (j == 2 ? P : 0.)) * dXdx[k][2]);
      }
    }
    // -- Total Energy Density
    // ---- (E + P) u
    for (CeedInt j = 0; j < 3; j++) dv[j][4][i] += wdetJ * (E + P) * (u[0] * dXdx[j][0] + u[1] * dXdx[j][1] + u[2] * dXdx[j][2]);

    // --Stabilization terms
    // ---- jacob_F_conv[3][5][5] = dF(convective)/dq at each direction
    CeedScalar jacob_F_conv[3][5][5] = {{{0.}}};
    ConvectiveFluxJacobian_Euler(jacob_F_conv, rho, u, E, gamma);

    // ---- dqdx collects drhodx, dUdx and dEdx in one vector
    CeedScalar dqdx[5][3];
    for (CeedInt j = 0; j < 3; j++) {
      dqdx[0][j] = drhodx[j];
      dqdx[4][j] = dEdx[j];
      for (CeedInt k = 0; k < 3; k++) dqdx[k + 1][j] = dUdx[k][j];
    }

    // ---- strong_conv = dF/dq * dq/dx    (Strong convection)
    CeedScalar strong_conv[5] = {0.};
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 5; k++) {
        for (CeedInt l = 0; l < 5; l++) strong_conv[k] += jacob_F_conv[j][k][l] * dqdx[l][j];
      }
    }

    // Stabilization
    // -- Tau elements
    const CeedScalar sound_speed = sqrt(gamma * P / rho);
    CeedScalar       Tau_x[3]    = {0.};
    Tau_spatial(Tau_x, dXdx, u, sound_speed, c_tau);

    // -- Stabilization method: none or SU
    CeedScalar stab[5][3] = {{0.}};
    switch (context->stabilization) {
      case 0:  // Galerkin
        break;
      case 1:  // SU
        for (CeedInt j = 0; j < 3; j++) {
          for (CeedInt k = 0; k < 5; k++) {
            for (CeedInt l = 0; l < 5; l++) stab[k][j] += jacob_F_conv[j][k][l] * Tau_x[j] * strong_conv[l];
          }
        }

        for (CeedInt j = 0; j < 5; j++) {
          for (CeedInt k = 0; k < 3; k++) dv[k][j][i] -= wdetJ * (stab[j][0] * dXdx[k][0] + stab[j][1] * dXdx[k][1] + stab[j][2] * dXdx[k][2]);
        }
        break;
      case 2:  // SUPG is not implemented for explicit scheme
        break;
    }

  }  // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the Euler equations with (mentioned above)
//   with implicit time stepping method
//
// *****************************************************************************
CEED_QFUNCTION(IFunction_Euler)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  EulerContext     context = (EulerContext)ctx;
  const CeedScalar c_tau   = context->c_tau;
  const CeedScalar gamma   = 1.4;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    // *INDENT-OFF*
    // Setup
    // -- Interp in
    const CeedScalar rho      = q[0][i];
    const CeedScalar u[3]     = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E        = q[4][i];
    const CeedScalar drho[3]  = {dq[0][0][i], dq[1][0][i], dq[2][0][i]};
    const CeedScalar dU[3][3] = {
        {dq[0][1][i], dq[1][1][i], dq[2][1][i]},
        {dq[0][2][i], dq[1][2][i], dq[2][2][i]},
        {dq[0][3][i], dq[1][3][i], dq[2][3][i]}
    };
    const CeedScalar dE[3] = {dq[0][4][i], dq[1][4][i], dq[2][4][i]};
    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ = q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*
    // dU/dx
    CeedScalar drhodx[3]       = {0.};
    CeedScalar dEdx[3]         = {0.};
    CeedScalar dUdx[3][3]      = {{0.}};
    CeedScalar dXdxdXdxT[3][3] = {{0.}};
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        drhodx[j] += drho[k] * dXdx[k][j];
        dEdx[j] += dE[k] * dXdx[k][j];
        for (CeedInt l = 0; l < 3; l++) {
          dUdx[j][k] += dU[j][l] * dXdx[l][k];
          dXdxdXdxT[j][k] += dXdx[j][l] * dXdx[k][l];  // dXdx_j,k * dXdx_k,j
        }
      }
    }
    const CeedScalar E_kinetic = 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), E_internal = E - E_kinetic,
                     P = E_internal * (gamma - 1.);  // P = pressure

    // The Physics
    // Zero v and dv so all future terms can safely sum into it
    for (CeedInt j = 0; j < 5; j++) {
      v[j][i] = 0.;
      for (CeedInt k = 0; k < 3; k++) dv[k][j][i] = 0.;
    }
    //-----mass matrix
    for (CeedInt j = 0; j < 5; j++) v[j][i] += wdetJ * q_dot[j][i];

    // -- Density
    // ---- u rho
    for (CeedInt j = 0; j < 3; j++) dv[j][0][i] -= wdetJ * (rho * u[0] * dXdx[j][0] + rho * u[1] * dXdx[j][1] + rho * u[2] * dXdx[j][2]);
    // -- Momentum
    // ---- rho (u x u) + P I3
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        dv[k][j + 1][i] -= wdetJ * ((rho * u[j] * u[0] + (j == 0 ? P : 0.)) * dXdx[k][0] + (rho * u[j] * u[1] + (j == 1 ? P : 0.)) * dXdx[k][1] +
                                    (rho * u[j] * u[2] + (j == 2 ? P : 0.)) * dXdx[k][2]);
      }
    }
    // -- Total Energy Density
    // ---- (E + P) u
    for (CeedInt j = 0; j < 3; j++) dv[j][4][i] -= wdetJ * (E + P) * (u[0] * dXdx[j][0] + u[1] * dXdx[j][1] + u[2] * dXdx[j][2]);

    // -- Stabilization terms
    // ---- jacob_F_conv[3][5][5] = dF(convective)/dq at each direction
    CeedScalar jacob_F_conv[3][5][5] = {{{0.}}};
    ConvectiveFluxJacobian_Euler(jacob_F_conv, rho, u, E, gamma);

    // ---- dqdx collects drhodx, dUdx and dEdx in one vector
    CeedScalar dqdx[5][3];
    for (CeedInt j = 0; j < 3; j++) {
      dqdx[0][j] = drhodx[j];
      dqdx[4][j] = dEdx[j];
      for (CeedInt k = 0; k < 3; k++) dqdx[k + 1][j] = dUdx[k][j];
    }

    // ---- strong_conv = dF/dq * dq/dx    (Strong convection)
    CeedScalar strong_conv[5] = {0.};
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 5; k++) {
        for (CeedInt l = 0; l < 5; l++) strong_conv[k] += jacob_F_conv[j][k][l] * dqdx[l][j];
      }
    }

    // ---- Strong residual
    CeedScalar strong_res[5];
    for (CeedInt j = 0; j < 5; j++) strong_res[j] = q_dot[j][i] + strong_conv[j];

    // Stabilization
    // -- Tau elements
    const CeedScalar sound_speed = sqrt(gamma * P / rho);
    CeedScalar       Tau_x[3]    = {0.};
    Tau_spatial(Tau_x, dXdx, u, sound_speed, c_tau);

    // -- Stabilization method: none, SU, or SUPG
    CeedScalar stab[5][3] = {{0.}};
    switch (context->stabilization) {
      case 0:  // Galerkin
        break;
      case 1:  // SU
        for (CeedInt j = 0; j < 3; j++) {
          for (CeedInt k = 0; k < 5; k++) {
            for (CeedInt l = 0; l < 5; l++) stab[k][j] += jacob_F_conv[j][k][l] * Tau_x[j] * strong_conv[l];
          }
        }

        for (CeedInt j = 0; j < 5; j++) {
          for (CeedInt k = 0; k < 3; k++) dv[k][j][i] += wdetJ * (stab[j][0] * dXdx[k][0] + stab[j][1] * dXdx[k][1] + stab[j][2] * dXdx[k][2]);
        }
        break;
      case 2:  // SUPG
        for (CeedInt j = 0; j < 3; j++) {
          for (CeedInt k = 0; k < 5; k++) {
            for (CeedInt l = 0; l < 5; l++) stab[k][j] = jacob_F_conv[j][k][l] * Tau_x[j] * strong_res[l];
          }
        }

        for (CeedInt j = 0; j < 5; j++) {
          for (CeedInt k = 0; k < 3; k++) dv[k][j][i] += wdetJ * (stab[j][0] * dXdx[k][0] + stab[j][1] * dXdx[k][1] + stab[j][2] * dXdx[k][2]);
        }
        break;
    }
  }  // End Quadrature Point Loop

  // Return
  return 0;
}
// *****************************************************************************
// This QFunction sets the inflow boundary conditions for
//   the traveling vortex problem.
//
//  Prescribed T_inlet and P_inlet are converted to conservative variables
//      and applied weakly.
//
// *****************************************************************************
CEED_QFUNCTION(TravelingVortex_Inflow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  EulerContext     context       = (EulerContext)ctx;
  const int        euler_test    = context->euler_test;
  const bool       implicit      = context->implicit;
  CeedScalar      *mean_velocity = context->mean_velocity;
  const CeedScalar cv            = 2.5;
  const CeedScalar R             = 1.;
  CeedScalar       T_inlet;
  CeedScalar       P_inlet;

  // For test cases 1 and 3 the background velocity is zero
  if (euler_test == 1 || euler_test == 3) {
    for (CeedInt i = 0; i < 3; i++) mean_velocity[i] = 0.;
  }

  // For test cases 1 and 2, T_inlet = T_inlet = 0.4
  if (euler_test == 1 || euler_test == 2) T_inlet = P_inlet = .4;
  else T_inlet = P_inlet = 1.;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb = (implicit ? -1. : 1.) * q_data_sur[0][i];
    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};

    // face_normal = Normal vector of the face
    const CeedScalar face_normal = norm[0] * mean_velocity[0] + norm[1] * mean_velocity[1] + norm[2] * mean_velocity[2];
    // The Physics
    // Zero v so all future terms can safely sum into it
    for (CeedInt j = 0; j < 5; j++) v[j][i] = 0.;

    // Implementing in/outflow BCs
    if (face_normal > 0) {
    } else {  // inflow
      const CeedScalar rho_inlet       = P_inlet / (R * T_inlet);
      const CeedScalar E_kinetic_inlet = (mean_velocity[0] * mean_velocity[0] + mean_velocity[1] * mean_velocity[1]) / 2.;
      // incoming total energy
      const CeedScalar E_inlet = rho_inlet * (cv * T_inlet + E_kinetic_inlet);

      // The Physics
      // -- Density
      v[0][i] -= wdetJb * rho_inlet * face_normal;

      // -- Momentum
      for (CeedInt j = 0; j < 3; j++) v[j + 1][i] -= wdetJb * (rho_inlet * face_normal * mean_velocity[j] + norm[j] * P_inlet);

      // -- Total Energy Density
      v[4][i] -= wdetJb * face_normal * (E_inlet + P_inlet);
    }

  }  // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************
// This QFunction sets the outflow boundary conditions for
//   the Euler solver.
//
//  Outflow BCs:
//    The validity of the weak form of the governing equations is
//      extended to the outflow.
//
// *****************************************************************************
CEED_QFUNCTION(Euler_Outflow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  EulerContext context       = (EulerContext)ctx;
  const bool   implicit      = context->implicit;
  CeedScalar  *mean_velocity = context->mean_velocity;

  const CeedScalar gamma = 1.4;

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
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};

    // face_normal = Normal vector of the face
    const CeedScalar face_normal = norm[0] * mean_velocity[0] + norm[1] * mean_velocity[1] + norm[2] * mean_velocity[2];
    // The Physics
    // Zero v so all future terms can safely sum into it
    for (CeedInt j = 0; j < 5; j++) v[j][i] = 0;

    // Implementing in/outflow BCs
    if (face_normal > 0) {  // outflow
      const CeedScalar E_kinetic = (u[0] * u[0] + u[1] * u[1]) / 2.;
      const CeedScalar P         = (E - E_kinetic * rho) * (gamma - 1.);              // pressure
      const CeedScalar u_normal  = norm[0] * u[0] + norm[1] * u[1] + norm[2] * u[2];  // Normal velocity
      // The Physics
      // -- Density
      v[0][i] -= wdetJb * rho * u_normal;

      // -- Momentum
      for (CeedInt j = 0; j < 3; j++) v[j + 1][i] -= wdetJb * (rho * u_normal * u[j] + norm[j] * P);

      // -- Total Energy Density
      v[4][i] -= wdetJb * u_normal * (E + P);
    }
  }  // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************

#endif  // eulervortex_h
