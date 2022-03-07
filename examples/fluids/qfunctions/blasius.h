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


CEED_QFUNCTION_HELPER int Exact_Channel(CeedInt dim, CeedScalar time,
                                        const CeedScalar X[], CeedInt Nf, CeedScalar q[], void *ctx) {

  const NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar theta0 = 300;
  const CeedScalar P0     = 1.e5;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar Rd     = cp - cv;
  const CeedScalar mu     = context->mu;
  const CeedScalar k      = context->k;

  const CeedScalar x=X[0], y=X[1], z=X[2];

  const CeedScalar meter  = 1e-2;
  const CeedScalar umax   = 10.;
  const CeedScalar center = 0.5*meter;

  const CeedScalar Pr    = mu / (cp*k);
  const CeedScalar Ec    = (umax*umax) / (cp*theta0);
  const CeedScalar theta = theta0*( 1 + (Pr*Ec/3)*(1 - pow((y-center)/center,4)));

  const CeedScalar ReH = umax*center/mu
                         ; //Deliberately not including density (it's canceled out)
  const CeedScalar p   = P0 - (2*umax*umax*x) / (ReH*center);
  const CeedScalar rho = p / (Rd*theta);

  q[0] = rho;
  q[1] = rho * umax*(1 - pow((y-center)/center,2));
  q[2] = 0;
  q[3] = 0;
  q[4] = rho * (cv*theta) + .5 * (q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) / rho;

  return 0;
}

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
    /* const CeedScalar Pi = 1. + g*g*(exp(-N*N*x[2]/g) - 1.) / (cp*theta0*N*N); */
    const CeedScalar Pi = 1.;

    // -- Density
    const CeedScalar rho = P0 * pow(Pi, cv/Rd) / (Rd*theta0);

    const CeedScalar meter = 1.e-2;
    const CeedScalar radius = 0.5*meter;
    const CeedScalar center[] = {0.5*meter, 0.5*meter};
    const CeedScalar xr[] = {x[0], x[1]-center[0], x[2]-center[1]};
    const CeedScalar r = sqrt(xr[1]*xr[1] + xr[2]*xr[2]);
    // Initial Conditions
    q[0] = rho;
    if (r > radius) {
      q[1] = rho*0.0;
    } else {
      q[1] = (1 - r/radius)*rho*1.0;
    }
    /* q[1] = rho*1.0; */
    q[2] = rho*0.0;
    q[3] = rho*0.0;
    /* q[4] = rho * (cv*theta0*Pi + g*x[2]); */
    q[4] = rho * (cv*theta0*Pi) + .5 * (q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) / rho;

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
  const CeedScalar (*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const bool implicit     = false;
  CeedScalar velocity[]   = {10., 0., 0.};
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
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // Find pressure using state inside the domain
    const CeedScalar rho = q[0][i];
    const CeedScalar u[3] = {q[1][i]/rho, q[2][i]/rho, q[3][i]/rho};
    const CeedScalar E_internal = q[4][i] - .5 * rho * (u[0]*u[0] + u[1]*u[1] +
                                  u[2]*u[2]);
    const CeedScalar P = E_internal * (gamma - 1.);

    // Find inflow state using calculated P and prescribed velocity, theta0
    const CeedScalar e_internal = cv * theta0;
    const CeedScalar rho_in = P / ((gamma - 1) * e_internal);
    const CeedScalar E_kinetic = .5 * rho_in * (velocity[0]*velocity[0] +
                                 velocity[1]*velocity[1] +
                                 velocity[2]*velocity[2]);
    const CeedScalar E = rho_in * e_internal + E_kinetic;
    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    // The Physics
    // Zero v so all future terms can safely sum into it
    for (int j=0; j<5; j++) v[j][i] = 0.;

    const CeedScalar u_normal = norm[0]*velocity[0] +
                                norm[1]*velocity[1] +
                                norm[2]*velocity[2];

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho_in * u_normal;

    // -- Momentum
    for (int j=0; j<3; j++)
      v[j+1][i] -= wdetJb * (rho_in * u_normal * velocity[j] +
                             norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);

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
