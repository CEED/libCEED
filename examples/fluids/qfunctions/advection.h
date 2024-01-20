// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Advection initial condition and operator for Navier-Stokes example using PETSc

#ifndef advection_h
#define advection_h

#include <ceed.h>
#include <math.h>

#include "advection_generic.h"
#include "advection_types.h"
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "stabilization_types.h"
#include "utils.h"

// *****************************************************************************
// This QFunction sets the initial conditions for 3D advection
// *****************************************************************************
CEED_QFUNCTION(ICsAdvection)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  CeedScalar(*q0)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[]  = {X[0][i], X[1][i], X[2][i]};
    CeedScalar       q[5] = {0.};

    Exact_AdvectionGeneric(3, 0., x, 5, q, ctx);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of the advection equation
//
// This is 3D advection given in two formulations based upon the weak form.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density    ,  Ui = rho ui
//   E   - Total Energy Density
//
// Advection Equation:
//   dE/dt + div( E u ) = 0
// *****************************************************************************
CEED_QFUNCTION(Advection)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  RHSFunction_AdvectionGeneric(ctx, Q, in, out, 3);
  return 0;
}

CEED_QFUNCTION(IFunction_Advection)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  IFunction_AdvectionGeneric(ctx, Q, in, out, 3);
  return 0;
}

CEED_QFUNCTION(Advection_InOutFlow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  Advection_InOutFlowGeneric(ctx, Q, in, out, 3);
  return 0;
}

#endif  // advection_h
