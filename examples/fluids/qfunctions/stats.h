// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// QFunctions for statistics collection (mainly L^2 projections)

#ifndef reynolds_stress_h
#define reynolds_stress_h

#include <ceed.h>

#include "newtonian_state.h"
#include "newtonian_types.h"

// *****************************************************************************
// This QFunction calculates the velocity product to be L2 projected back onto
// the FE space. This is used to calculate the Reynolds stress via
// <u'_iu'_j> = <U_iU_j> - <U_i><U_j>
// *****************************************************************************

CEED_QFUNCTION_HELPER int VelocityProducts(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateFromQi_t StateFromQi) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  const CeedScalar(*x)[CEED_Q_VLA]      = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar(*U_prod)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Context
  const NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s      = StateFromQi(context, qi, x_i);

    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ = q_data[0][i];

    // Using Kelvin Mandel notation (array ordering)
    // U_prod[0][i] = s.Y.velocity[0] * s.Y.velocity[0] * wdetJ; // U*U
    // U_prod[1][i] = s.Y.velocity[1] * s.Y.velocity[1] * wdetJ; // V*V
    // U_prod[2][i] = s.Y.velocity[2] * s.Y.velocity[2] * wdetJ; // W*W
    // U_prod[3][i] = s.Y.velocity[1] * s.Y.velocity[2] * wdetJ; // V*W
    // U_prod[4][i] = s.Y.velocity[0] * s.Y.velocity[2] * wdetJ; // U*W
    // U_prod[5][i] = s.Y.velocity[0] * s.Y.velocity[1] * wdetJ; // U*V

    U_prod[0][i] = s.Y.pressure * s.Y.pressure * wdetJ;        // P*P
    U_prod[1][i] = s.Y.velocity[0] * s.Y.velocity[0] * wdetJ;  // U*U
    U_prod[2][i] = s.Y.velocity[0] * s.Y.velocity[1] * wdetJ;  // U*V
    U_prod[3][i] = s.Y.velocity[0] * s.Y.temperature * wdetJ;  // U*T
    U_prod[4][i] = s.Y.temperature * s.Y.temperature * wdetJ;  // T*T

  }  // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(VelocityProducts_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return VelocityProducts(ctx, Q, in, out, StateFromU);
}

CEED_QFUNCTION(VelocityProducts_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return VelocityProducts(ctx, Q, in, out, StateFromY);
}
#endif  // stats_h
