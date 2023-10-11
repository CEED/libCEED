// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
///

#ifndef taylorgreen_h
#define taylorgreen_h

#include <ceed.h>
#include <math.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "utils.h"

// @brief Set initial condition for Taylor-Green Vortex problem
CEED_QFUNCTION(ICsTaylorGreen)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const SetupContext                context   = (SetupContext)ctx;
  struct NewtonianIdealGasContext_ *gas       = &context->gas;
  CeedScalar                        R         = GasConstant(gas);
  StatePrimitive                    reference = context->reference;
  const CeedScalar                  V0        = sqrt(Dot3(reference.velocity, reference.velocity));
  const CeedScalar                  density0  = reference.pressure / (reference.temperature * R);

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar x[]  = {X[0][i], X[1][i], X[2][i]};
    CeedScalar q[5] = {0}, Y[5];
    ScaleN(x, 2 * M_PI / context->lx, 3);

    Y[0] = reference.pressure + (density0 * Square(V0) / 16) * (cos(2 * x[0]) + cos(2 * x[1])) * (cos(2 * x[2] + 2));
    Y[1] = V0 * sin(x[0]) * cos(x[1]) * cos(x[2]);
    Y[2] = -V0 * cos(x[0]) * sin(x[1]) * cos(x[2]);
    Y[3] = 0;
    Y[4] = reference.temperature;

    State s = StateFromY(gas, Y);
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

#endif  // taylorgreen_h
