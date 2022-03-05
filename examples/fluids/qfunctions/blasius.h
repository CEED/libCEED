// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc


#ifndef blasius_h
#define blasius_h

#include <math.h>
#include <ceed.h>
#include "../navierstokes.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

// *****************************************************************************
// This QFunction sets a "still" initial condition for generic Newtonian IG problems
// *****************************************************************************
CEED_QFUNCTION(ICsBlasius)(void *ctx, CeedInt Q,
                           const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar q[5] = {0.};

    // Context
    /* const SetupContext context = (SetupContext)ctx; */
    const NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
    /* const CeedScalar theta0    = context->theta0; */
    /* const CeedScalar P0        = context->P0; */
    /* const CeedScalar N         = context->N; */
    const CeedScalar theta0    = 300;
    const CeedScalar P0        = 1.e5;
    const CeedScalar N         = 0.01;
    const CeedScalar cv        = context->cv;
    const CeedScalar cp        = context->cp;
    const CeedScalar g         = context->g;
    const CeedScalar Rd        = cp - cv;

    // Setup

    // -- Exner pressure, hydrostatic balance
    const CeedScalar Pi = 1. + g*g*(exp(-N*N*x[2]/g) - 1.) / (cp*theta0*N*N);

    // -- Density
    const CeedScalar rho = P0 * pow(Pi, cv/Rd) / (Rd*theta0);

    const CeedScalar bl_height = 1.e-2;
    // Initial Conditions
    q[0] = rho;
    if (x[1] > bl_height) {
      q[1] = rho*1.0;
    } else {
      q[1] = (x[1]/bl_height)*rho*1.0;
    }
    q[2] = rho*0.0;
    q[3] = rho*0.0;
    q[4] = rho * (cv*theta0*Pi + g*x[2]);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Blasius_Inflow)(void *ctx, CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const bool implicit     = false;
  CeedScalar velocity[]   = {1., 0., 0.};
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar g      = context->g;
  const CeedScalar Rd     = cp - cv;
  const CeedScalar theta0 = 300;
  const CeedScalar P0     = 1.e5;
  const CeedScalar N      = 0.01;
  const CeedScalar z      = 0.;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // -- Exner pressure, hydrostatic balance
    const CeedScalar Pi = 1. + g*g*(exp(-N*N*z/g) - 1.) / (cp*theta0*N*N);

    // -- Density
    const CeedScalar rho = P0 * pow(Pi, cv/Rd) / (Rd*theta0);
    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    // The Physics
    // Zero v so all future terms can safely sum into it
    for (int j=0; j<5; j++) v[j][i] = 0.;

    // Implementing inflow condition
    const CeedScalar E_kinetic = (velocity[0]*velocity[0] +
                                  velocity[1]*velocity[1] +
                                  velocity[2]*velocity[2]) / 2.;
    // incoming total energy
    const CeedScalar E = rho * (cv * theta0 + E_kinetic);

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho;

    // -- Momentum
    for (int j=0; j<3; j++)
      v[j+1][i] -= wdetJb *(rho *  velocity[j] +
                            norm[j] * P0);

    // -- Total Energy Density
    v[4][i] -= wdetJb * (E + P0);

  } // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Blasius_Outflow)(void *ctx, CeedInt Q,
                                const CeedScalar *const *in,
                                CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const bool implicit     = false;
  CeedScalar velocity[]   = {1., 0., 0.};
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar g      = context->g;
  const CeedScalar Rd     = cp - cv;
  const CeedScalar gamma  = cp/cv;
  const CeedScalar theta0 = 300;
  const CeedScalar P0     = 1.e5;
  const CeedScalar N      = 0.01;
  const CeedScalar z      = 0.;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho      =  q[0][i];
    const CeedScalar u[3]     = {q[1][i] / rho,
                                 q[2][i] / rho,
                                 q[3][i] / rho
                                };
    const CeedScalar E        =  q[4][i];

    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    // The Physics
    // Zero v so all future terms can safely sum into it
    for (int j=0; j<5; j++) v[j][i] = 0.;

    // Implementing outflow condition
    const CeedScalar E_kinetic = (u[0]*u[0] + u[1]*u[1]) / 2.;
    const CeedScalar P         = (E - E_kinetic * rho) * (gamma - 1.); // pressure
    const CeedScalar u_normal  = norm[0]*u[0] + norm[1]*u[1] +
                                 norm[2]*u[2]; // Normal velocity
    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal;

    // -- Momentum
    for (int j=0; j<3; j++)
      v[j+1][i] -= wdetJb *(rho * u_normal * u[j] + norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);

  } // End Quadrature Point Loop
  return 0;
}
#endif // blasius_h
