// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Thermodynamic wave propogation for testing freestream/non-reflecting boundary conditions. Proposed in Mengaldo et. al. 2014

#include <ceed.h>
#include <math.h>

#include "newtonian_state.h"
#include "utils.h"

typedef struct GaussianWaveContext_ *GaussianWaveContext;
struct GaussianWaveContext_ {
  CeedScalar                       epicenter[3];  // Location of the perturbation
  CeedScalar                       width;         // Controls width of the perturbation
  CeedScalar                       amplitude;     // Amplitude of the perturbation
  State                            S_infty;       // Flow state at infinity
  struct NewtonianIdealGasContext_ newt_ctx;
};

CEED_QFUNCTION_HELPER int IC_GaussianWave(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const GaussianWaveContext      context  = (GaussianWaveContext)ctx;
  const NewtonianIdealGasContext newt_ctx = &context->newt_ctx;

  const CeedScalar amplitude = context->amplitude;
  const CeedScalar width     = context->width;
  const State      S_infty   = context->S_infty;
  const CeedScalar xc        = context->epicenter[0];
  const CeedScalar yc        = context->epicenter[1];

  const CeedScalar gamma = HeatCapacityRatio(newt_ctx);

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar       U[5] = {0.}, qi[5] = {0.};
    const CeedScalar x[3]      = {X[0][i], X[1][i], X[2][i]};
    const CeedScalar x0        = x[0] - xc;
    const CeedScalar y0        = x[1] - yc;
    const CeedScalar e_kinetic = 0.5 * S_infty.U.density * Dot3(S_infty.Y.velocity, S_infty.Y.velocity);

    const CeedScalar perturbation = 1 + amplitude * exp(-(Square(x0) + Square(y0)) / (2 * Square(width)));

    U[0] = S_infty.U.density * perturbation;
    U[1] = S_infty.Y.velocity[0] * U[0];
    U[2] = S_infty.Y.velocity[1] * U[0];
    U[3] = S_infty.Y.velocity[2] * U[0];
    U[4] = S_infty.Y.pressure / (gamma - 1) * perturbation + e_kinetic;

    State initCond = StateFromU(newt_ctx, U, x);
    StateToQ(newt_ctx, initCond, qi, state_var);

    for (CeedInt j = 0; j < 5; j++) q0[j][i] = qi[j];
  }

  return 0;
}

CEED_QFUNCTION(IC_GaussianWave_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IC_GaussianWave(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(IC_GaussianWave_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IC_GaussianWave(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

CEED_QFUNCTION(IC_GaussianWave_Entropy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IC_GaussianWave(ctx, Q, in, out, STATEVAR_ENTROPY);
}
