// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Advection initial condition and operator for Navier-Stokes example using PETSc

#ifndef advection2d_h
#define advection2d_h

#include <ceed.h>
#include <math.h>

#include "advection_generic.h"
#include "advection_types.h"
#include "newtonian_state.h"
#include "newtonian_types.h"
#include "stabilization_types.h"
#include "utils.h"

// *****************************************************************************
// This QFunction sets the initial conditions for 2D advection
// *****************************************************************************
CEED_QFUNCTION(ICsAdvection2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  CeedScalar(*q0)[CEED_Q_VLA]      = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const SetupContextAdv context    = (SetupContextAdv)ctx;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x[]  = {X[0][i], X[1][i]};
    CeedScalar       q[5] = {0.};

    Exact_AdvectionGeneric(2, context->time, x, 5, q, ctx);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of the advection equation
//
// This is 2D advection given in two formulations based upon the weak form.
//
// State Variables: q = ( rho, U1, U2, E )
//   rho - Mass Density
//   Ui  - Momentum Density    ,  Ui = rho ui
//   E   - Total Energy Density
//
// Advection Equation:
//   dE/dt + div( E u ) = 0
// *****************************************************************************
CEED_QFUNCTION(Advection2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  RHSFunction_AdvectionGeneric(ctx, Q, in, out, 2);
  return 0;
}

CEED_QFUNCTION(IFunction_Advection2d)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  IFunction_AdvectionGeneric(ctx, Q, in, out, 2);
  return 0;
}

// *****************************************************************************
// This QFunction implements consistent outflow and inflow BCs
//      for 2D advection
//
//  Inflow and outflow faces are determined based on sign(dot(wind, normal)):
//    sign(dot(wind, normal)) > 0 : outflow BCs
//    sign(dot(wind, normal)) < 0 : inflow BCs
//
//  Outflow BCs:
//    The validity of the weak form of the governing equations is extended to the outflow and the current values of E are applied.
//
//  Inflow BCs:
//    A prescribed Total Energy (E_wind) is applied weakly.
// *****************************************************************************
CEED_QFUNCTION(Advection2d_InOutFlow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]   = (CeedScalar(*)[CEED_Q_VLA])out[0];
  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar E_wind      = context->E_wind;
  const CeedScalar strong_form = context->strong_form;
  const bool       implicit    = context->implicit;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho  = q[0][i];
    const CeedScalar u[3] = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E    = q[4][i];

    CeedScalar wdetJb, norm[2];
    QdataBoundaryUnpack_2D(Q, i, (CeedScalar *)q_data_sur, &wdetJb, norm);
    wdetJb *= implicit ? -1. : 1.;

    // Normal velocity
    const CeedScalar u_normal = norm[0] * u[0] + norm[1] * u[1];

    // No Change in density or momentum
    for (CeedInt j = 0; j < 4; j++) {
      v[j][i] = 0;
    }

    // Implementing in/outflow BCs
    if (u_normal > 0) {  // outflow
      v[4][i] = -(1 - strong_form) * wdetJb * E * u_normal;
    } else {  // inflow
      v[4][i] = -(1 - strong_form) * wdetJb * E_wind * u_normal;
    }
  }  // End Quadrature Point Loop
  return 0;
}
// *****************************************************************************

#endif  // advection2d_h
