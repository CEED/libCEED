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
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1];
  const CeedScalar(*q_data)            = in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]     = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  // Context
  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar CtauS       = context->CtauS;
  const CeedScalar strong_form = context->strong_form;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho  = q[0][i];
    const CeedScalar u[3] = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E    = q[4][i];
    // -- Grad in
    const CeedScalar drho[3]  = {dq[0][0][i], dq[1][0][i], dq[2][0][i]};
    const CeedScalar du[3][3] = {
        {(dq[0][1][i] - drho[0] * u[0]) / rho, (dq[1][1][i] - drho[1] * u[0]) / rho, (dq[2][1][i] - drho[2] * u[0]) / rho},
        {(dq[0][2][i] - drho[0] * u[1]) / rho, (dq[1][2][i] - drho[1] * u[1]) / rho, (dq[2][2][i] - drho[2] * u[1]) / rho},
        {(dq[0][3][i] - drho[0] * u[2]) / rho, (dq[1][3][i] - drho[1] * u[2]) / rho, (dq[2][3][i] - drho[2] * u[2]) / rho}
    };
    const CeedScalar dE[3] = {dq[0][4][i], dq[1][4][i], dq[2][4][i]};
    CeedScalar       wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    // The Physics
    // Note with the order that du was filled and the order that dXdx was filled
    //   du[j][k]= du_j / dX_K    (note cap K to be clear this is u_{j,xi_k})
    //   dXdx[k][j] = dX_K / dx_j
    //   X_K=Kth reference element coordinate (note cap X and K instead of xi_k}
    //   x_j and u_j are jth  physical position and velocity components

    // No Change in density or momentum
    for (CeedInt f = 0; f < 4; f++) {
      for (CeedInt j = 0; j < 3; j++) dv[j][f][i] = 0;
      v[f][i] = 0;
    }

    // -- Total Energy
    // Evaluate the strong form using div(E u) = u . grad(E) + E div(u)
    // or in index notation: (u_j E)_{,j} = u_j E_j + E u_{j,j}
    CeedScalar div_u = 0, u_dot_grad_E = 0;
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dEdx_j = 0;
      for (CeedInt k = 0; k < 3; k++) {
        div_u += du[j][k] * dXdx[k][j];  // u_{j,j} = u_{j,K} X_{K,j}
        dEdx_j += dE[k] * dXdx[k][j];
      }
      u_dot_grad_E += u[j] * dEdx_j;
    }
    CeedScalar strong_conv = E * div_u + u_dot_grad_E;

    // Weak Galerkin convection term: dv \cdot (E u)
    for (CeedInt j = 0; j < 3; j++) dv[j][4][i] = (1 - strong_form) * wdetJ * E * (u[0] * dXdx[j][0] + u[1] * dXdx[j][1] + u[2] * dXdx[j][2]);
    v[4][i] = 0;

    // Strong Galerkin convection term: - v div(E u)
    v[4][i] = -strong_form * wdetJ * strong_conv;

    // Stabilization requires a measure of element transit time in the velocity
    //   field u.
    CeedScalar uX[3];
    for (CeedInt j = 0; j < 3; j++) uX[j] = dXdx[j][0] * u[0] + dXdx[j][1] * u[1] + dXdx[j][2] * u[2];
    const CeedScalar TauS = CtauS / sqrt(Dot3(uX, uX));
    for (CeedInt j = 0; j < 3; j++) dv[j][4][i] -= wdetJ * TauS * strong_conv * uX[j];
  }  // End Quadrature Point Loop

  return 0;
}

// *****************************************************************************
// This QFunction implements 3D (mentioned above) with implicit time stepping method
// *****************************************************************************
CEED_QFUNCTION(IFunction_Advection)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar(*q)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1];
  const CeedScalar(*q_dot)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*q_data)                = in[3];

  CeedScalar(*v)[CEED_Q_VLA]         = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*Grad_v)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  CeedScalar *jac_data               = out[2];

  AdvectionContext                 context     = (AdvectionContext)ctx;
  const CeedScalar                 CtauS       = context->CtauS;
  const CeedScalar                 strong_form = context->strong_form;
  const CeedScalar                 zeros[14]   = {0.};
  NewtonianIdealGasContext         gas;
  struct NewtonianIdealGasContext_ gas_struct = {0};
  gas                                         = &gas_struct;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const State      s     = StateFromU(gas, qi);

    CeedScalar wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    State grad_s[3];
    StatePhysicalGradientFromReference(Q, i, gas, s, STATEVAR_CONSERVATIVE, (CeedScalar *)Grad_q, dXdx, grad_s);

    const CeedScalar Grad_E[3] = {grad_s[0].U.E_total, grad_s[1].U.E_total, grad_s[2].U.E_total};

    for (CeedInt f = 0; f < 4; f++) {
      for (CeedInt j = 0; j < 3; j++) Grad_v[j][f][i] = 0;  // No Change in density or momentum
      v[f][i] = wdetJ * q_dot[f][i];                        // K Mass/transient term
    }

    CeedScalar div_u = 0;
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        div_u += grad_s[k].Y.velocity[j];
      }
    }
    CeedScalar strong_conv = s.U.E_total * div_u + Dot3(s.Y.velocity, Grad_E);
    CeedScalar strong_res  = q_dot[4][i] + strong_conv;

    v[4][i] = wdetJ * q_dot[4][i];  // transient part (ALWAYS)

    if (strong_form) {  // Strong Galerkin convection term: v div(E u)
      v[4][i] += wdetJ * strong_conv;
    } else {  // Weak Galerkin convection term: -dv \cdot (E u)
      for (CeedInt j = 0; j < 3; j++)
        Grad_v[j][4][i] = -wdetJ * s.U.E_total * (s.Y.velocity[0] * dXdx[j][0] + s.Y.velocity[1] * dXdx[j][1] + s.Y.velocity[2] * dXdx[j][2]);
    }

    // Stabilization requires a measure of element transit time in the velocity field u.
    CeedScalar uX[3] = {0.};
    MatVec3(dXdx, s.Y.velocity, CEED_NOTRANSPOSE, uX);
    const CeedScalar TauS = CtauS / sqrt(Dot3(uX, uX));

    for (CeedInt j = 0; j < 3; j++) switch (context->stabilization) {
        case STAB_NONE:
          break;
        case STAB_SU:
          Grad_v[j][4][i] += wdetJ * TauS * strong_conv * uX[j];
          break;
        case STAB_SUPG:
          Grad_v[j][4][i] += wdetJ * TauS * strong_res * uX[j];
          break;
      }
    StoredValuesPack(Q, i, 0, 14, zeros, jac_data);
  }
  return 0;
}

// *****************************************************************************
// This QFunction implements consistent outflow and inflow BCs
//      for 3D advection
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
CEED_QFUNCTION(Advection_InOutFlow)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*q_data_sur)    = in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]   = (CeedScalar(*)[CEED_Q_VLA])out[0];
  AdvectionContext context     = (AdvectionContext)ctx;
  const CeedScalar E_wind      = context->E_wind;
  const CeedScalar strong_form = context->strong_form;
  const bool       is_implicit = context->implicit;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho  = q[0][i];
    const CeedScalar u[3] = {q[1][i] / rho, q[2][i] / rho, q[3][i] / rho};
    const CeedScalar E    = q[4][i];

    CeedScalar wdetJb, norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, NULL, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    // Normal velocity
    const CeedScalar u_normal = norm[0] * u[0] + norm[1] * u[1] + norm[2] * u[2];

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

#endif  // advection_h
