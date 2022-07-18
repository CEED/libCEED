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
#include "newtonian_state.h"
#include "utils.h"

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

CEED_QFUNCTION_HELPER State Exact_Channel(CeedInt dim, CeedScalar time,
    const CeedScalar X[], CeedInt Nf, void *ctx) {

  const ChannelContext context = (ChannelContext)ctx;
  const CeedScalar theta0      = context->theta0;
  const CeedScalar P0          = context->P0;
  const CeedScalar umax        = context->umax;
  const CeedScalar center      = context->center;
  const CeedScalar H           = context->H;
  NewtonianIdealGasContext gas = &context->newtonian_ctx;
  const CeedScalar cp          = gas->cp;
  const CeedScalar mu          = gas->mu;
  const CeedScalar k           = gas->k;
  // There is a gravity body force but it is excluded from
  //   the potential energy due to periodicity.
  gas->g[0] = 0.;
  gas->g[1] = 0.;
  gas->g[2] = 0.;

  const CeedScalar y     = X[1];
  const CeedScalar Pr    = mu / (cp*k);
  const CeedScalar Ec    = (umax*umax) / (cp*theta0);
  const CeedScalar theta = theta0*(1 + (Pr*Ec/3)
                                   * (1 - Square(Square((y-center)/H))));
  CeedScalar Y[5] = {0.};
  Y[0] = P0;
  Y[1] = umax*(1 - Square((y-center)/H));
  Y[2] = 0.;
  Y[3] = 0.;
  Y[4] = theta;

  return StateFromY(gas, Y, X);
}

// *****************************************************************************
// This QFunction set the initial condition
// *****************************************************************************
CEED_QFUNCTION(ICsChannel)(void *ctx, CeedInt Q,
                           const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Context
  const ChannelContext context = (ChannelContext)ctx;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    State s = Exact_Channel(3, 0., x, 5, ctx);
    if (context->newtonian_ctx.primitive) {
      q0[0][i] = s.Y.pressure;
      for (CeedInt j=0; j<3; j++)
        q0[j+1][i] = s.Y.velocity[j];
      q0[4][i] = s.Y.temperature;
    } else {
      q0[0][i] = s.U.density;
      for (CeedInt j=0; j<3; j++)
        q0[j+1][i] = s.U.momentum[j];
      q0[4][i] = s.U.E_total;
    }

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
    State s = Exact_Channel(3, 0., x, 5, ctx);
    CeedScalar q_exact[5] = {0.};
    q_exact[0] = s.U.density;
    for (CeedInt j=0; j<3; j++)
      q_exact[j+1] = s.U.momentum[j];
    q_exact[4] = s.U.E_total;
    const CeedScalar E_kinetic_exact = 0.5*Dot3(&q_exact[1], &q_exact[1])
                                       / q_exact[0];
    const CeedScalar velocity[3] = {q_exact[1]/q_exact[0],
                                    q_exact[2]/q_exact[0],
                                    q_exact[3]/q_exact[0]
                                   };
    const CeedScalar theta = (q_exact[4] - E_kinetic_exact) / (q_exact[0]*cv);

    // Find pressure using state inside the domain
    const CeedScalar rho = q[0][i];
    const CeedScalar u[3] = {q[1][i]/rho, q[2][i]/rho, q[3][i]/rho};
    const CeedScalar E_internal = q[4][i] - .5 * rho * Dot3(u,u);
    const CeedScalar P = E_internal * (gamma - 1.);

    // Find inflow state using calculated P and prescribed velocity, theta0
    const CeedScalar e_internal = cv * theta;
    const CeedScalar rho_in = P / ((gamma - 1) * e_internal);
    const CeedScalar E_kinetic = .5 * rho_in * Dot3(velocity, velocity);
    const CeedScalar E = rho_in * e_internal + E_kinetic;
    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    // The Physics
    // Zero v so all future terms can safely sum into it
    for (CeedInt j=0; j<5; j++) v[j][i] = 0.;

    const CeedScalar u_normal = Dot3(norm, velocity);

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
    const CeedScalar u_normal  = Dot3(norm, u); // Normal velocity
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

// *****************************************************************************
#endif // channel_h
