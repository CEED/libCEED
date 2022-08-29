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

#include <ceed.h>
#include <math.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

typedef struct BlasiusContext_ *BlasiusContext;
struct BlasiusContext_ {
  bool                             implicit;  // !< Using implicit timesteping or not
  bool                             weakT;     // !< flag to set Temperature weakly at inflow
  CeedScalar                       delta0;    // !< Boundary layer height at inflow
  CeedScalar                       Uinf;      // !< Velocity at boundary layer edge
  CeedScalar                       Tinf;      // !< Temperature at boundary layer edge
  CeedScalar                       T_wall;    // !< Temperature at the wall
  CeedScalar                       P0;        // !< Pressure at outflow
  CeedScalar                       theta0;    // !< Temperature at inflow
  CeedScalar                       x_inflow;  // !< Location of inflow in x
  CeedScalar                       n_cheb;    // !< Number of Chebyshev terms
  CeedScalar                      *X;         // !< Chebyshev polynomial coordinate vector
  CeedScalar                       eta_max;   // !< Maximum eta in the domain
  CeedScalar                      *Tf_cheb;   // !< Chebyshev coefficient for f
  CeedScalar                      *Th_cheb;   // !< Chebyshev coefficient for h
  struct NewtonianIdealGasContext_ newtonian_ctx;
};

// *****************************************************************************
// This helper function evaluates Chebyshev polynomials with a set of
//  coefficients with all their derivatives represented as a recurrence table.
// *****************************************************************************
CEED_QFUNCTION_HELPER void ChebyshevEval(int N, const double *Tf, double x, double eta_max, double *f) {
  double dX_deta     = 2 / eta_max;
  double table[4][3] = {
  // Chebyshev polynomials T_0, T_1, T_2 of the first kind in (-1,1)
      {1, x, 2 * x * x - 1},
      {0, 1, 4 * x        },
      {0, 0, 4            },
      {0, 0, 0            }
  };
  for (int i = 0; i < 4; i++) {
    // i-th derivative of f
    f[i] = table[i][0] * Tf[0] + table[i][1] * Tf[1] + table[i][2] * Tf[2];
  }
  for (int i = 3; i < N; i++) {
    // T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
    table[0][i % 3] = 2 * x * table[0][(i - 1) % 3] - table[0][(i - 2) % 3];
    // Differentiate Chebyshev polynomials with the recurrence relation
    for (int j = 1; j < 4; j++) {
      // T'_{n}(x)/n = 2T_{n-1}(x) + T'_{n-2}(x)/n-2
      table[j][i % 3] = i * (2 * table[j - 1][(i - 1) % 3] + table[j][(i - 2) % 3] / (i - 2));
    }
    for (int j = 0; j < 4; j++) {
      f[j] += table[j][i % 3] * Tf[i];
    }
  }
  for (int i = 1; i < 4; i++) {
    // Transform derivatives from Chebyshev [-1, 1] to [0, eta_max].
    for (int j = 0; j < i; j++) f[i] *= dX_deta;
  }
}

// *****************************************************************************
// This helper function computes the Blasius boundary layer solution.
// *****************************************************************************
State CEED_QFUNCTION_HELPER(BlasiusSolution)(const BlasiusContext blasius, const CeedScalar x[3], const CeedScalar x0, const CeedScalar x_inflow,
                                             const CeedScalar rho, CeedScalar *t12) {
  CeedInt    N    = blasius->n_cheb;
  CeedScalar nu   = blasius->newtonian_ctx.mu / rho;
  CeedScalar eta  = x[1] * sqrt(blasius->Uinf / (nu * (x0 + x[0] - x_inflow)));
  CeedScalar X    = 2 * (eta / blasius->eta_max) - 1.;
  CeedScalar Uinf = blasius->Uinf;
  CeedScalar Rd   = GasConstant(&blasius->newtonian_ctx);

  CeedScalar f[4], h[4];
  ChebyshevEval(N, blasius->Tf_cheb, X, blasius->eta_max, f);
  ChebyshevEval(N - 1, blasius->Th_cheb, X, blasius->eta_max, h);

  *t12 = rho * nu * Uinf * f[2] * sqrt(Uinf / (nu * (x0 + x[0] - x_inflow)));

  CeedScalar Y[5];
  Y[1] = Uinf * f[1];
  Y[2] = 0.5 * sqrt(nu * Uinf / (x0 + x[0] - x_inflow)) * (eta * f[1] - f[0]);
  Y[3] = 0.;
  Y[4] = blasius->Tinf * h[0];
  Y[0] = rho * Rd * Y[4];
  return StateFromY(&blasius->newtonian_ctx, Y, x);
}

// *****************************************************************************
// This QFunction sets a Blasius boundary layer for the initial condition
// *****************************************************************************
CEED_QFUNCTION(ICsBlasius)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const BlasiusContext context    = (BlasiusContext)ctx;
  const CeedScalar     cv         = context->newtonian_ctx.cv;
  const CeedScalar     mu         = context->newtonian_ctx.mu;
  const CeedScalar     theta0     = context->theta0;
  const CeedScalar     P0         = context->P0;
  const CeedScalar     delta0     = context->delta0;
  const CeedScalar     Uinf       = context->Uinf;
  const CeedScalar     x_inflow   = context->x_inflow;
  const CeedScalar     gamma      = HeatCapacityRatio(&context->newtonian_ctx);
  const CeedScalar     e_internal = cv * theta0;
  const CeedScalar     rho        = P0 / ((gamma - 1) * e_internal);
  const CeedScalar     x0         = Uinf * rho / (mu * 25 / (delta0 * delta0));
  CeedScalar           t12;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[3] = {X[0][i], X[1][i], 0.};
    State            s    = BlasiusSolution(context, x, x0, x_inflow, rho, &t12);
    CeedScalar       q[5] = {0};
    UnpackState_U(s.U, q);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }  // End of Quadrature Point Loop
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Blasius_Inflow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
        (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]    = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  const BlasiusContext context  = (BlasiusContext)ctx;
  const bool           implicit = context->implicit;
  const CeedScalar     mu       = context->newtonian_ctx.mu;
  const CeedScalar     cv       = context->newtonian_ctx.cv;
  const CeedScalar     Rd       = GasConstant(&context->newtonian_ctx);
  const CeedScalar     gamma    = HeatCapacityRatio(&context->newtonian_ctx);
  const CeedScalar     theta0   = context->theta0;
  const CeedScalar     P0       = context->P0;
  const CeedScalar     delta0   = context->delta0;
  const CeedScalar     Uinf     = context->Uinf;
  const CeedScalar     x_inflow = context->x_inflow;
  const bool           weakT    = context->weakT;
  const CeedScalar     rho_0    = P0 / (Rd * theta0);
  const CeedScalar     x0       = Uinf * rho_0 / (mu * 25 / Square(delta0));

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // Calculate inflow values
    const CeedScalar x[3] = {X[0][i], X[1][i], 0.};
    CeedScalar       t12;
    State            s = BlasiusSolution(context, x, x0, x_inflow, rho_0, &t12);

    // enabling user to choose between weak T and weak rho inflow
    CeedScalar rho, E_internal, P, E_kinetic;
    if (weakT) {
      // rho should be from the current solution
      rho        = q[0][i];
      // Temperature is being set weakly (theta0) and for constant cv this sets E_internal
      E_internal = rho * cv * theta0;
      // Find pressure using
      P          = rho * Rd * theta0;  // interior rho with exterior T
      E_kinetic  = .5 * rho * Dot3(s.Y.velocity, s.Y.velocity);
    } else {
      //  Fixing rho weakly on the inflow to a value consistent with theta0 and P0
      rho        = rho_0;
      E_kinetic  = .5 * rho * Dot3(s.Y.velocity, s.Y.velocity);
      E_internal = q[4][i] - E_kinetic;  // uses set rho and u but E from solution
      P          = E_internal * (gamma - 1.);
    }
    const CeedScalar E       = E_internal + E_kinetic;
    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};

    // The Physics
    // Zero v so all future terms can safely sum into it
    for (CeedInt j = 0; j < 5; j++) v[j][i] = 0.;

    const CeedScalar u_normal        = Dot3(norm, s.Y.velocity);
    const CeedScalar viscous_flux[3] = {-t12 * norm[1], -t12 * norm[0], 0};

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal;  // interior rho

    // -- Momentum
    for (CeedInt j = 0; j < 3; j++) {
      v[j + 1][i] -= wdetJb * (rho * u_normal * s.Y.velocity[j]  // interior rho
                               + norm[j] * P                     // mixed P
                               + viscous_flux[j]);
    }

    // -- Total Energy Density
    v[4][i] -= wdetJb * (u_normal * (E + P) + Dot3(viscous_flux, s.Y.velocity));

  }  // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Blasius_Inflow_Jacobian)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
        (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]    = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  const BlasiusContext context  = (BlasiusContext)ctx;
  const bool           implicit = context->implicit;
  const CeedScalar     mu       = context->newtonian_ctx.mu;
  const CeedScalar     cv       = context->newtonian_ctx.cv;
  const CeedScalar     Rd       = GasConstant(&context->newtonian_ctx);
  const CeedScalar     gamma    = HeatCapacityRatio(&context->newtonian_ctx);
  const CeedScalar     theta0   = context->theta0;
  const CeedScalar     P0       = context->P0;
  const CeedScalar     delta0   = context->delta0;
  const CeedScalar     Uinf     = context->Uinf;
  const bool           weakT    = context->weakT;
  const CeedScalar     rho_0    = P0 / (Rd * theta0);
  const CeedScalar     x0       = Uinf * rho_0 / (mu * 25 / (delta0 * delta0));

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // Calculate inflow values
    const CeedScalar x[3] = {X[0][i], X[1][i], 0.};
    CeedScalar       t12;
    State            s = BlasiusSolution(context, x, x0, 0, rho_0, &t12);

    // enabling user to choose between weak T and weak rho inflow
    CeedScalar drho, dE, dP;
    if (weakT) {
      // rho should be from the current solution
      drho                   = dq[0][i];
      CeedScalar dE_internal = drho * cv * theta0;
      CeedScalar dE_kinetic  = .5 * drho * Dot3(s.Y.velocity, s.Y.velocity);
      dE                     = dE_internal + dE_kinetic;
      dP                     = drho * Rd * theta0;  // interior rho with exterior T
    } else {                                        // rho specified, E_internal from solution
      drho = 0;
      dE   = dq[4][i];
      dP   = dE * (gamma - 1.);
    }
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};

    const CeedScalar u_normal = Dot3(norm, s.Y.velocity);

    v[0][i] = -wdetJb * drho * u_normal;
    for (int j = 0; j < 3; j++) {
      v[j + 1][i] = -wdetJb * (drho * u_normal * s.Y.velocity[j] + norm[j] * dP);
    }
    v[4][i] = -wdetJb * u_normal * (dE + dP);
  }  // End Quadrature Point Loop
  return 0;
}

#endif  // blasius_h
