// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc
#include <ceed.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

#define BLASIUS_MAX_N_CHEBYSHEV 50

typedef struct BlasiusContext_ *BlasiusContext;
struct BlasiusContext_ {
  bool                             implicit;  // !< Using implicit timesteping or not
  bool                             weakT;     // !< flag to set Temperature weakly at inflow
  CeedScalar                       delta0;    // !< Boundary layer height at inflow
  State                            S_infty;
  CeedScalar                       T_wall;                                // !< Temperature at the wall
  CeedScalar                       x_inflow;                              // !< Location of inflow in x
  CeedScalar                       n_cheb;                                // !< Number of Chebyshev terms
  CeedScalar                      *X;                                     // !< Chebyshev polynomial coordinate vector (CPU only)
  CeedScalar                       eta_max;                               // !< Maximum eta in the domain
  CeedScalar                       Tf_cheb[BLASIUS_MAX_N_CHEBYSHEV];      // !< Chebyshev coefficient for f
  CeedScalar                       Th_cheb[BLASIUS_MAX_N_CHEBYSHEV - 1];  // !< Chebyshev coefficient for h
  struct NewtonianIdealGasContext_ newtonian_ctx;
};

// *****************************************************************************
// This helper function evaluates Chebyshev polynomials with a set of coefficients with all their derivatives represented as a recurrence table.
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
                                             const CeedScalar rho_infty, CeedScalar *t12) {
  CeedInt    N       = blasius->n_cheb;
  CeedScalar mu      = blasius->newtonian_ctx.mu;
  State      S_infty = blasius->S_infty;
  CeedScalar nu      = mu / rho_infty;
  CeedScalar U_infty = sqrt(Dot3(S_infty.Y.velocity, S_infty.Y.velocity));
  CeedScalar eta     = x[1] * sqrt(U_infty / (nu * (x0 + x[0] - x_inflow)));
  CeedScalar X       = 2 * (eta / blasius->eta_max) - 1.;
  CeedScalar Rd      = GasConstant(&blasius->newtonian_ctx);

  CeedScalar f[4], h[4];
  ChebyshevEval(N, blasius->Tf_cheb, X, blasius->eta_max, f);
  ChebyshevEval(N - 1, blasius->Th_cheb, X, blasius->eta_max, h);

  *t12 = mu * U_infty * f[2] * sqrt(U_infty / (nu * (x0 + x[0] - x_inflow)));

  CeedScalar Y[5];
  Y[1] = U_infty * f[1];
  Y[2] = 0.5 * sqrt(nu * U_infty / (x0 + x[0] - x_inflow)) * (eta * f[1] - f[0]);
  Y[3] = 0.;
  Y[4] = S_infty.Y.temperature * h[0];
  Y[0] = rho_infty / h[0] * Rd * Y[4];
  return StateFromY(&blasius->newtonian_ctx, Y);
}

// *****************************************************************************
// This QFunction sets a Blasius boundary layer for the initial condition
// *****************************************************************************
CEED_QFUNCTION(ICsBlasius)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  CeedScalar(*q0)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const BlasiusContext           context  = (BlasiusContext)ctx;
  const NewtonianIdealGasContext gas      = &context->newtonian_ctx;
  const CeedScalar               mu       = context->newtonian_ctx.mu;
  const CeedScalar               delta0   = context->delta0;
  const CeedScalar               x_inflow = context->x_inflow;
  CeedScalar                     t12;

  const State      S_infty = context->S_infty;
  const CeedScalar U_infty = sqrt(Dot3(S_infty.Y.velocity, S_infty.Y.velocity));

  const CeedScalar x0 = U_infty * S_infty.U.density / (mu * 25 / Square(delta0));

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};
    State            s    = BlasiusSolution(context, x, x0, x_inflow, S_infty.U.density, &t12);
    CeedScalar       q[5] = {0};

    switch (gas->state_var) {
      case STATEVAR_CONSERVATIVE:
        UnpackState_U(s.U, q);
        break;
      case STATEVAR_PRIMITIVE:
        UnpackState_Y(s.Y, q);
        break;
    }
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Blasius_Inflow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const BlasiusContext context     = (BlasiusContext)ctx;
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)    = in[2];
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  CeedScalar(*v)[CEED_Q_VLA]       = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)        = context->newtonian_ctx.is_implicit ? out[1] : NULL;

  const bool                     is_implicit = context->implicit;
  const NewtonianIdealGasContext gas         = &context->newtonian_ctx;
  State                          S_infty     = context->S_infty;
  const CeedScalar               rho_0       = S_infty.U.density;
  const CeedScalar               U_infty     = sqrt(Dot3(S_infty.Y.velocity, S_infty.Y.velocity));
  const CeedScalar               x0          = U_infty * rho_0 / (gas->mu * 25 / Square(context->delta0));
  const CeedScalar               zeros[11]   = {0.};

  CeedScalar(*jac_data_sur) = is_implicit ? out[1] : NULL;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    // Calculate inflow values
    const CeedScalar x[3] = {X[0][i], X[1][i], 0.};
    CeedScalar       t12;
    State            s = BlasiusSolution(context, x, x0, context->x_inflow, rho_0, &t12);
    CeedScalar       qi[5];
    for (CeedInt j = 0; j < 5; j++) qi[j] = q[j][i];
    State s_int = StateFromU(gas, qi);

    // enabling user to choose between weak T and weak rho inflow
    if (context->weakT) {  // density from the current solution
      s.U.density = s_int.U.density;
      s.Y         = StatePrimitiveFromConservative(gas, s.U);
    } else {  // Total energy from current solution
      s.U.E_total = s_int.U.E_total;
      s.Y         = StatePrimitiveFromConservative(gas, s.U);
    }

    StateConservative Flux_inviscid[3];
    FluxInviscid(&context->newtonian_ctx, s, Flux_inviscid);

    const CeedScalar stress[3][3] = {
        {0,   t12, 0},
        {t12, 0,   0},
        {0,   0,   0}
    };
    const CeedScalar Fe[3] = {0};  // TODO: viscous energy flux needs grad temperature
    CeedScalar       Flux[5];
    FluxTotal_Boundary(Flux_inviscid, stress, Fe, norm, Flux);
    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];
    if (is_implicit) StoredValuesPack(Q, i, 0, 11, zeros, jac_data_sur);
  }
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Blasius_Inflow_Jacobian)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*X)[CEED_Q_VLA]  = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  CeedScalar(*v)[CEED_Q_VLA]        = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const BlasiusContext           context     = (BlasiusContext)ctx;
  const NewtonianIdealGasContext gas         = &context->newtonian_ctx;
  const bool                     is_implicit = context->implicit;
  const CeedScalar               Rd          = GasConstant(gas);
  const CeedScalar               gamma       = HeatCapacityRatio(gas);
  const State                    S_infty     = context->S_infty;
  const CeedScalar               rho_0       = S_infty.U.density;
  const CeedScalar               U_infty     = sqrt(Dot3(S_infty.Y.velocity, S_infty.Y.velocity));
  const CeedScalar               x0          = U_infty * rho_0 / (gas->mu * 25 / Square(context->delta0));

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    // Calculate inflow values
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar       t12;
    State            s = BlasiusSolution(context, x, x0, 0, rho_0, &t12);

    // enabling user to choose between weak T and weak rho inflow
    CeedScalar drho, dE, dP;
    if (context->weakT) {
      // rho should be from the current solution
      drho                   = dq[0][i];
      CeedScalar dE_internal = drho * gas->cv * S_infty.Y.temperature;
      CeedScalar dE_kinetic  = .5 * drho * Dot3(s.Y.velocity, s.Y.velocity);
      dE                     = dE_internal + dE_kinetic;
      dP                     = drho * Rd * S_infty.Y.temperature;  // interior rho with exterior T
    } else {
      // rho specified, E_internal from solution
      drho = 0;
      dE   = dq[4][i];
      dP   = dE * (gamma - 1.);
    }

    const CeedScalar u_normal = Dot3(norm, s.Y.velocity);

    v[0][i] = -wdetJb * drho * u_normal;
    for (int j = 0; j < 3; j++) {
      v[j + 1][i] = -wdetJb * (drho * u_normal * s.Y.velocity[j] + norm[j] * dP);
    }
    v[4][i] = -wdetJb * u_normal * (dE + dP);
  }
  return 0;
}
