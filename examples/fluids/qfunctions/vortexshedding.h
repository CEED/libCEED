// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc


#ifndef vortexshedding_h
#define vortexshedding_h

#include <ceed.h>
#include <math.h>
#include "ceed/ceed-f64.h"
#include "ceed/types.h"
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

typedef struct VortexsheddingContext_ *VortexsheddingContext;
struct VortexsheddingContext_ {
  bool       implicit; // !< Using implicit timesteping or not
  bool       weakT;    // !< flag to set Temperature weakly at inflow
  CeedScalar U_infty;     // !< Velocity at freestream
  CeedScalar T_infty;     // !< Temperature at freestream
  CeedScalar P0;       // !< Pressure at freestream
  CeedScalar L;        // !< Length of the rectangular channel
  CeedScalar H;        // !< Height of the rectangular channel
  CeedScalar D;        // !< Cylinder diameter
  CeedScalar T;        // !< Shedding period
  CeedScalar center;   // !< Cylinder center
  CeedScalar radius;   // !< Cylinder radius
  State      S_infty;
  struct NewtonianIdealGasContext_ newtonian_ctx;
};

// *****************************************************************************
// This QFunction set the initial condition for the cylinder
// *****************************************************************************
CEED_QFUNCTION_HELPER int ICsVortexshedding(void *ctx, CeedInt Q,
                           const CeedScalar *const *in, CeedScalar *const *out,
                           StateToQi_t StateToQi) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // Context
  const VortexsheddingContext context = (VortexsheddingContext)ctx;
  const NewtonianIdealGasContext newtonian_ctx = &context->newtonian_ctx;
  //NewtonianIdealGasContext gas     = &context->newtonian_ctx;
  const CeedScalar T_infty     = context->T_infty;
  const CeedScalar P0          = context->P0;
  const CeedScalar U_infty     = context->U_infty;
  const CeedScalar H           = context->H;
  const CeedScalar L           = context->L;
  const CeedScalar D           = context->D;
  const CeedScalar T           = context->T;
  const CeedScalar center      = context->center;
  const CeedScalar radius      = context->radius;
  const State      S_infty     = context->S_infty;
  const CeedScalar cv          = context->newtonian_ctx.cv;
  const CeedScalar mu          = context->newtonian_ctx.mu;
  const CeedScalar gamma       = HeatCapacityRatio(&context->newtonian_ctx);
  const CeedScalar e_internal  = cv * T_infty;
  const CeedScalar rho         = P0 / ((gamma - 1) * e_internal);
  const CeedScalar Re          = (rho * U_infty * D) / mu;
  const CeedScalar Ma          = U_infty / SoundSpeed(newtonian_ctx, S_infty.Y.temperature);
  const CeedScalar St          = D / (U_infty * T);
  const CeedScalar e_kinetic = 0.5 * S_infty.U.density * Dot3(S_infty.Y.velocity, S_infty.Y.velocity);

  // Quadrature point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar U[5] = {0.};
    CeedScalar qi[5] = {0.};
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};

    U[0] = S_infty.U.density;
    U[1] = U[0] * tanh(Min(0, fabs(x[0] - center) - radius) / D);
    U[2] = U[0] * tanh(Min(0, fabs(x[1] - center) - radius) / D);
    U[3] = U[0] * tanh(Min(0, fabs(x[2] - center) - radius) / D);
    U[4] = S_infty.Y.pressure / (gamma - 1) + e_kinetic;

    State InitCond = StateFromU(newtonian_ctx, U, x);
    StateToQi(newtonian_ctx, InitCond, qi);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = qi[j];

  } // End of Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(ICsVortexshedding_Conserv)(void *ctx, CeedInt Q,
    const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsVortexshedding(ctx, Q, in, out, StateToU);
}

CEED_QFUNCTION(ICsVortexshedding_Prim)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsVortexshedding(ctx, Q, in, out, StateToY);
}

#endif // vortexshedding_h