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

#include <ceed.h>
#include <math.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

struct ChannelContext_ {
  bool                             implicit;  // !< Using implicit timesteping or not
  CeedScalar                       theta0;    // !< Reference temperature
  CeedScalar                       P0;        // !< Reference Pressure
  CeedScalar                       umax;      // !< Centerline velocity
  CeedScalar                       center;    // !< Y Coordinate for center of channel
  CeedScalar                       H;         // !< Channel half-height
  CeedScalar                       B;         // !< Body-force driving the flow
  struct NewtonianIdealGasContext_ newtonian_ctx;
};
#define ChannelContext struct ChannelContext_*;

CEED_QFUNCTION_HELPER State Exact_Channel(CeedInt dim, CeedScalar time, const CeedScalar X[], CeedInt Nf, void *ctx) {
  struct ChannelContext_* context = (struct ChannelContext_*) ctx;
  const CeedScalar         theta0  = context->theta0;
  const CeedScalar         P0      = context->P0;
  const CeedScalar         umax    = context->umax;
  const CeedScalar         center  = context->center;
  const CeedScalar         H       = context->H;
  NewtonianIdealGasContext gas     = &context->newtonian_ctx;
  const CeedScalar         cp      = gas->cp;
  const CeedScalar         mu      = gas->mu;
  const CeedScalar         k       = gas->k;
  // There is a gravity body force but it is excluded from
  //   the potential energy due to periodicity.
  //     g = (g, 0, 0)
  //     x = (0, x_2, x_3)
  //     e_potential = dot(g, x) = 0
  const CeedScalar x[3] = {0, X[1], X[2]};

  const CeedScalar Pr    = mu / (cp * k);
  const CeedScalar Ec    = (umax * umax) / (cp * theta0);
  const CeedScalar theta = theta0 * (1 + (Pr * Ec / 3) * (1 - Square(Square((x[1] - center) / H))));
  CeedScalar       Y[5]  = {0.};
  Y[0]                   = P0;
  Y[1]                   = umax * (1 - Square((x[1] - center) / H));
  Y[2]                   = 0.;
  Y[3]                   = 0.;
  Y[4]                   = theta;

  return StateFromY(gas, Y, x);
}

// *****************************************************************************
// This QFunction set the initial condition
// *****************************************************************************
CEED_QFUNCTION(ICsChannel)(void *ctx, CeedInt Q,
                           const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs 
  typedef CeedScalar vec_t[CEED_Q_VLA];
  const vec_t* X = (const vec_t*) in[0];

  // Outputs
  vec_t* q0 = (vec_t*) out[0];

  // Context
  struct ChannelContext_* context = (struct ChannelContext_*) ctx;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[]  = {X[0][i], X[1][i], X[2][i]};
    State            s    = Exact_Channel(3, 0., x, 5, ctx);
    CeedScalar       q[5] = {0};
    switch (context->newtonian_ctx.state_var) {
      case STATEVAR_CONSERVATIVE:
        UnpackState_U(s.U, q);
        break;
      case STATEVAR_PRIMITIVE:
        UnpackState_Y(s.Y, q);
        break;
    }

    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];

  }  // End of Quadrature Point Loop
  return 0;
}

// *****************************************************************************
// This QFunction set the inflow boundary condition for conservative variables
// *****************************************************************************
CEED_QFUNCTION(Channel_Inflow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  typedef CeedScalar vec_t[CEED_Q_VLA];
  const vec_t* q = (const vec_t*) in[0];
  const vec_t* q_data_sur = (const vec_t*) in[2];
  const vec_t* X = (const vec_t*) in[3];

  // Outputs
  vec_t* v = (vec_t*) out[0];

  struct ChannelContext_* context = (struct ChannelContext_*) ctx;
  const bool               implicit = context->implicit;
  NewtonianIdealGasContext gas      = &context->newtonian_ctx;
  const CeedScalar         cv       = gas->cv;
  const CeedScalar         gamma    = HeatCapacityRatio(&context->newtonian_ctx);
  
  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // There is a gravity body force but it is excluded from
    //   the potential energy due to periodicity.
    //     g = (g, 0, 0)
    //     x = (0, x_2, x_3)
    //     e_potential = dot(g, x) = 0
    const CeedScalar x[3] = {0, X[1][i], X[2][i]};

    // Calcualte prescribed inflow values
    State      s_exact    = Exact_Channel(3, 0., x, 5, ctx);
    CeedScalar q_exact[5] = {0.};
    UnpackState_U(s_exact.U, q_exact);

    // Find pressure using state inside the domain
    CeedScalar q_inside[5] = {0};
    for (CeedInt j = 0; j < 5; j++) q_inside[j] = q[j][i];
    State            s_inside = StateFromU(gas, q_inside, x);
    const CeedScalar P        = s_inside.Y.pressure;

    // Find inflow state using calculated P and prescribed velocity, theta0
    const CeedScalar e_internal = cv * s_exact.Y.temperature;
    const CeedScalar rho_in     = P / ((gamma - 1) * e_internal);
    const CeedScalar E_kinetic  = .5 * rho_in * Dot3(s_exact.Y.velocity, s_exact.Y.velocity);
    const CeedScalar E          = rho_in * e_internal + E_kinetic;

    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};
    // The Physics
    // Zero v so all future terms can safely sum into it
    for (CeedInt j = 0; j < 5; j++) v[j][i] = 0.;

    const CeedScalar u_normal = Dot3(norm, s_exact.Y.velocity);

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho_in * u_normal;

    // -- Momentum
    for (CeedInt j = 0; j < 3; j++) v[j + 1][i] -= wdetJb * (rho_in * u_normal * s_exact.Y.velocity[j] + norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);

  }  // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************
// This QFunction set the outflow boundary condition for conservative variables
// *****************************************************************************
CEED_QFUNCTION(Channel_Outflow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  typedef CeedScalar vec_t[CEED_Q_VLA];
  const vec_t* q = (const vec_t*) in[0];
  const vec_t* q_data_sur = (const vec_t*) in[2];

  // Outputs
  vec_t* v = (vec_t*) out[0];
  
  struct ChannelContext_* context = (struct ChannelContext_*) ctx;
  const bool           implicit = context->implicit;
  const CeedScalar     P0       = context->P0;

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

    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};
    // The Physics
    // Zero v so all future terms can safely sum into it
    for (CeedInt j = 0; j < 5; j++) v[j][i] = 0.;

    // Implementing outflow condition
    const CeedScalar P        = P0;             // pressure
    const CeedScalar u_normal = Dot3(norm, u);  // Normal velocity
    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal;

    // -- Momentum
    for (CeedInt j = 0; j < 3; j++) v[j + 1][i] -= wdetJb * (rho * u_normal * u[j] + norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);

  }  // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************
#endif  // channel_h
