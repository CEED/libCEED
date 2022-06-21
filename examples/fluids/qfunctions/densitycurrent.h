// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Density current initial condition and operator for Navier-Stokes example using PETSc

// Model from:
//   Semi-Implicit Formulations of the Navier-Stokes Equations: Application to
//   Nonhydrostatic Atmospheric Modeling, Giraldo, Restelli, and Lauter (2010).

#ifndef densitycurrent_h
#define densitycurrent_h

#include <math.h>
#include <ceed.h>
#include "newtonian_types.h"
#include "utils.h"

// *****************************************************************************
// This function sets the initial conditions and the boundary conditions
//
// These initial conditions are given in terms of potential temperature and
//   Exner pressure and then converted to density and total energy.
//   Initial momentum density is zero.
//
// Initial Conditions:
//   Potential Temperature:
//     theta = thetabar + delta_theta
//       thetabar   = theta0 exp( N**2 z / g )
//       delta_theta = r <= rc : thetaC(1 + cos(pi r/rc)) / 2
//                     r > rc : 0
//         r        = sqrt( (x - xc)**2 + (y - yc)**2 + (z - zc)**2 )
//         with (xc,yc,zc) center of domain, rc characteristic radius of thermal bubble
//   Exner Pressure:
//     Pi = Pibar + deltaPi
//       Pibar      = 1. + g**2 (exp( - N**2 z / g ) - 1) / (cp theta0 N**2)
//       deltaPi    = 0 (hydrostatic balance)
//   Velocity/Momentum Density:
//     Ui = ui = 0
//
// Conversion to Conserved Variables:
//   rho = P0 Pi**(cv/Rd) / (Rd theta)
//   E   = rho (cv T + (u u)/2 + g z)
//
//  Boundary Conditions:
//    Mass Density:
//      0.0 flux
//    Momentum Density:
//      0.0
//    Energy Density:
//      0.0 flux
//
// Constants:
//   theta0          ,  Potential temperature constant
//   thetaC          ,  Potential temperature perturbation
//   P0              ,  Pressure at the surface
//   N               ,  Brunt-Vaisala frequency
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   Rd     = cp - cv,  Specific heat difference
//   g               ,  Gravity
//   rc              ,  Characteristic radius of thermal bubble
//   center          ,  Location of bubble center
//   dc_axis         ,  Axis of density current cylindrical anomaly, or {0,0,0} for spherically symmetric
// *****************************************************************************

// *****************************************************************************
// This helper function provides support for the exact, time-dependent solution
//   (currently not implemented) and IC formulation for density current
// *****************************************************************************
CEED_QFUNCTION_HELPER int Exact_DC(CeedInt dim, CeedScalar time,
                                   const CeedScalar X[], CeedInt Nf, CeedScalar q[],
                                   void *ctx) {
  // Context
  const SetupContext context = (SetupContext)ctx;
  const CeedScalar theta0   = context->theta0;
  const CeedScalar thetaC   = context->thetaC;
  const CeedScalar P0       = context->P0;
  const CeedScalar N        = context->N;
  const CeedScalar cv       = context->cv;
  const CeedScalar cp       = context->cp;
  const CeedScalar *g_vec   = context->g;
  const CeedScalar rc       = context->rc;
  const CeedScalar *center  = context->center;
  const CeedScalar *dc_axis = context->dc_axis;
  const CeedScalar Rd       = cp - cv;
  const CeedScalar g = -g_vec[2];

  // Setup
  // -- Coordinates
  const CeedScalar x = X[0];
  const CeedScalar y = X[1];
  const CeedScalar z = X[2];

  // -- Potential temperature, density current
  CeedScalar rr[3] = {x - center[0], y - center[1], z - center[2]};
  // (I - q q^T) r: distance from dc_axis (or from center if dc_axis is the zero vector)
  for (CeedInt i=0; i<3; i++)
    rr[i] -= dc_axis[i] *
             (dc_axis[0]*rr[0] + dc_axis[1]*rr[1] + dc_axis[2]*rr[2]);
  const CeedScalar r = sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]);
  const CeedScalar delta_theta = r <= rc ? thetaC*(1. + cos(M_PI*r/rc))/2. : 0.;
  const CeedScalar theta = theta0*exp(N*N*z/g) + delta_theta;

  // -- Exner pressure, hydrostatic balance
  const CeedScalar Pi = 1. + g*g*(exp(-N*N*z/g) - 1.) / (cp*theta0*N*N);

  // -- Density
  const CeedScalar rho = P0 * pow(Pi, cv/Rd) / (Rd*theta);

  // Initial Conditions
  q[0] = rho;
  q[1] = 0.0;
  q[2] = 0.0;
  q[3] = 0.0;
  q[4] = rho * (cv*theta*Pi + g*z);

  return 0;
}

// *****************************************************************************
// This QFunction sets the initial conditions for density current
// *****************************************************************************
CEED_QFUNCTION(ICsDC)(void *ctx, CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar q[5] = {0.};

    Exact_DC(3, 0., x, 5, q, ctx);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop

  return 0;
}

// *****************************************************************************
// This helper function provides support for the exact, time-dependent solution
//   (currently not implemented) and IC formulation for density current
//   with primitive variables
// *****************************************************************************
CEED_QFUNCTION_HELPER int Exact_DC_Prim(CeedInt dim, CeedScalar time,
                                        const CeedScalar X[], CeedInt Nf, CeedScalar q[],
                                        void *ctx) {
  // Context
  const SetupContext context = (SetupContext)ctx;
  const CeedScalar theta0   = context->theta0;
  const CeedScalar thetaC   = context->thetaC;
  const CeedScalar N        = context->N;
  const CeedScalar cp       = context->cp;
  const CeedScalar *g_vec   = context->g;
  const CeedScalar rc       = context->rc;
  const CeedScalar *center  = context->center;
  const CeedScalar *dc_axis = context->dc_axis;
  const CeedScalar g = -g_vec[2];

  // Setup
  // -- Coordinates
  const CeedScalar x = X[0];
  const CeedScalar y = X[1];
  const CeedScalar z = X[2];

  // -- Potential temperature, density current
  CeedScalar rr[3] = {x - center[0], y - center[1], z - center[2]};
  // (I - q q^T) r: distance from dc_axis (or from center if dc_axis is the zero vector)
  for (CeedInt i=0; i<3; i++)
    rr[i] -= dc_axis[i] *
             (dc_axis[0]*rr[0] + dc_axis[1]*rr[1] + dc_axis[2]*rr[2]);
  const CeedScalar r = sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]);
  const CeedScalar delta_theta = r <= rc ? thetaC*(1. + cos(M_PI*r/rc))/2. : 0.;
  const CeedScalar theta = theta0*exp(N*N*z/g) + delta_theta;

  // -- Exner pressure, hydrostatic balance
  const CeedScalar Pi = 1. + g*g*(exp(-N*N*z/g) - 1.) / (cp*theta0*N*N);

  // Initial Conditions
  q[0] = Pi;
  q[1] = 0.0;
  q[2] = 0.0;
  q[3] = 0.0;
  q[4] = theta;

  return 0;
}

// *****************************************************************************
// This QFunction sets the initial conditions for density current
//   with primitive variables
// *****************************************************************************
CEED_QFUNCTION(ICsDC_Prim)(void *ctx, CeedInt Q,
                           const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar q[5] = {0.};

    Exact_DC_Prim(3, 0., x, 5, q, ctx);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop

  return 0;
}
// *****************************************************************************

#endif // densitycurrent_h
