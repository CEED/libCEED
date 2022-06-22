// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc


#ifndef channel_h
#define channel_h

#include <math.h>
#include <ceed/ceed.h>
#include "newtonian_types.h"

typedef struct ChannelContext_ *ChannelContext;
struct ChannelContext_ {
  bool       implicit; // !< Using implicit timesteping or not
  CeedScalar theta0;   // !< Reference temperature
  CeedScalar P0;       // !< Reference Pressure
  CeedScalar umax;     // !< Centerline velocity
  CeedScalar center;   // !< Y Coordinate for center of channel
  CeedScalar H;        // !< Channel half-height
  CeedScalar B;        // !< Body-force driving the flow
  struct NewtonianIdealGasContext_ newtonian_ctx;
};

CEED_QFUNCTION_HELPER CeedInt Exact_Channel(CeedInt dim, CeedScalar time,
    const CeedScalar X[], CeedInt Nf, CeedScalar q[], void *ctx) {

  const ChannelContext context = (ChannelContext)ctx;
  const CeedScalar theta0 = context->theta0;
  const CeedScalar P0     = context->P0;
  const CeedScalar umax   = context->umax;
  const CeedScalar center = context->center;
  const CeedScalar H      = context->H;
  const CeedScalar cv     = context->newtonian_ctx.cv;
  const CeedScalar cp     = context->newtonian_ctx.cp;
  const CeedScalar Rd     = cp - cv;
  const CeedScalar mu     = context->newtonian_ctx.mu;
  const CeedScalar k      = context->newtonian_ctx.k;

  const CeedScalar y=X[1];

  const CeedScalar Pr    = mu / (cp*k);
  const CeedScalar Ec    = (umax*umax) / (cp*theta0);
  const CeedScalar theta = theta0*(1 + (Pr*Ec/3)
                                   * (1 - Square(Square((y-center)/H))));

  const CeedScalar p = P0;

  const CeedScalar rho = p / (Rd*theta);

  q[0] = rho;
  q[1] = rho * umax*(1 - Square((y-center)/H));
  q[2] = 0;
  q[3] = 0;
  q[4] = rho * (cv*theta) + .5 * (q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) / rho;

  return 0;
}

// *****************************************************************************
// This QFunction sets the initial condition
// *****************************************************************************
CEED_QFUNCTION(ICsChannel)(void *ctx, CeedInt Q,
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
    Exact_Channel(3, 0., x, 5, q, ctx);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Channel_Inflow)(void *ctx, CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*X)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[3];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  const ChannelContext context = (ChannelContext)ctx;
  const bool implicit     = context->implicit;
  const CeedScalar cv     = context->newtonian_ctx.cv;
  const CeedScalar cp     = context->newtonian_ctx.cp;
  const CeedScalar gamma  = cp/cv;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // Calcualte prescribed inflow values
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar q_exact[5] = {0.};
    Exact_Channel(3, 0., x, 5, q_exact, ctx);
    const CeedScalar E_kinetic_exact = 0.5*(q_exact[1]*q_exact[1] +
                                            q_exact[2]*q_exact[2] +
                                            q_exact[3]*q_exact[3]) / q_exact[0];
    const CeedScalar velocity[3] = {q_exact[1]/q_exact[0],
                                    q_exact[2]/q_exact[0],
                                    q_exact[3]/q_exact[0]
                                   };
    const CeedScalar theta = (q_exact[4] - E_kinetic_exact) / (q_exact[0]*cv);

    // Find pressure using state inside the domain
    const CeedScalar rho = q[0][i];
    const CeedScalar u[3] = {q[1][i]/rho, q[2][i]/rho, q[3][i]/rho};
    const CeedScalar E_internal = q[4][i] - .5 * rho * (u[0]*u[0] + u[1]*u[1] +
                                  u[2]*u[2]);
    const CeedScalar P = E_internal * (gamma - 1.);

    // Find inflow state using calculated P and prescribed velocity, theta0
    const CeedScalar e_internal = cv * theta;
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
    for (CeedInt j=0; j<5; j++) v[j][i] = 0.;

    const CeedScalar u_normal = norm[0]*velocity[0] +
                                norm[1]*velocity[1] +
                                norm[2]*velocity[2];

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho_in * u_normal;

    // -- Momentum
    for (CeedInt j=0; j<3; j++)
      v[j+1][i] -= wdetJb * (rho_in * u_normal * velocity[j] +
                             norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);

  } // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Channel_Outflow)(void *ctx, CeedInt Q,
                                const CeedScalar *const *in,
                                CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const ChannelContext context = (ChannelContext)ctx;
  const bool implicit     = context->implicit;
  const CeedScalar P0     = context->P0;

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
    for (CeedInt j=0; j<5; j++) v[j][i] = 0.;

    // Implementing outflow condition
    const CeedScalar P         = P0; // pressure
    const CeedScalar u_normal  = norm[0]*u[0] + norm[1]*u[1] +
                                 norm[2]*u[2]; // Normal velocity
    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal;

    // -- Momentum
    for (CeedInt j=0; j<3; j++)
      v[j+1][i] -= wdetJb *(rho * u_normal * u[j] + norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);

  } // End Quadrature Point Loop
  return 0;
}
#endif // channel_h
