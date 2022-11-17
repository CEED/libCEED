// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc

#ifndef newtonian_h
#define newtonian_h

#include <ceed.h>
#include <math.h>
#include <stdlib.h>

#include "newtonian_state.h"
#include "newtonian_types.h"
#include "stabilization.h"
#include "utils.h"

// *****************************************************************************
// This QFunction sets a "still" initial condition for generic Newtonian IG problems
// *****************************************************************************
CEED_QFUNCTION(ICsNewtonianIG)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Context
  const SetupContext context = (SetupContext)ctx;
  const CeedScalar   theta0  = context->theta0;
  const CeedScalar   P0      = context->P0;
  const CeedScalar   cv      = context->cv;
  const CeedScalar   cp      = context->cp;
  const CeedScalar  *g       = context->g;
  const CeedScalar   Rd      = cp - cv;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar q[5] = {0.};

    // Setup
    // -- Coordinates
    const CeedScalar x[3]        = {X[0][i], X[1][i], X[2][i]};
    const CeedScalar e_potential = -Dot3(g, x);

    // -- Density
    const CeedScalar rho = P0 / (Rd * theta0);

    // Initial Conditions
    q[0] = rho;
    q[1] = 0.0;
    q[2] = 0.0;
    q[3] = 0.0;
    q[4] = rho * (cv * theta0 + e_potential);

    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];

  }  // End of Quadrature Point Loop
  return 0;
}

// *****************************************************************************
// This QFunction sets a "still" initial condition for generic Newtonian IG
//   problems in primitive variables
// *****************************************************************************
CEED_QFUNCTION(ICsNewtonianIG_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Outputs
  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Context
  const SetupContext context = (SetupContext)ctx;
  const CeedScalar   theta0  = context->theta0;
  const CeedScalar   P0      = context->P0;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar q[5] = {0.};

    // Initial Conditions
    q[0] = P0;
    q[1] = 0.0;
    q[2] = 0.0;
    q[3] = 0.0;
    q[4] = theta0;

    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];

  }  // End of Quadrature Point Loop
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of Navier-Stokes with
//   explicit time stepping method
//
// This is 3D compressible Navier-Stokes in conservation form with state
//   variables of density, momentum density, and total energy density.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density,      Ui = rho ui
//   E   - Total Energy Density,  E  = rho (cv T + (u u)/2 + g z)
//
// Navier-Stokes Equations:
//   drho/dt + div( U )                               = 0
//   dU/dt   + div( rho (u x u) + P I3 ) + rho g khat = div( Fu )
//   dE/dt   + div( (E + P) u )                       = div( Fe )
//
// Viscous Stress:
//   Fu = mu (grad( u ) + grad( u )^T + lambda div ( u ) I3)
//
// Thermal Stress:
//   Fe = u Fu + k grad( T )
// Equation of State
//   P = (gamma - 1) (E - rho (u u) / 2 - rho g z)
//
// Stabilization:
//   Tau = diag(TauC, TauM, TauM, TauM, TauE)
//     f1 = rho  sqrt(ui uj gij)
//     gij = dXi/dX * dXi/dX
//     TauC = Cc f1 / (8 gii)
//     TauM = min( 1 , 1 / f1 )
//     TauE = TauM / (Ce cv)
//
//  SU   = Galerkin + grad(v) . ( Ai^T * Tau * (Aj q,j) )
//
// Constants:
//   lambda = - 2 / 3,  From Stokes hypothesis
//   mu              ,  Dynamic viscosity
//   k               ,  Thermal conductivity
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   g               ,  Gravity
//   gamma  = cp / cv,  Specific heat ratio
//
// We require the product of the inverse of the Jacobian (dXdx_j,k) and
// its transpose (dXdx_k,j) to properly compute integrals of the form:
// int( gradv gradu )
//
// *****************************************************************************
CEED_QFUNCTION(RHSFunction_Newtonian)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*Grad_q)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2], (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*Grad_v)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar        *g       = context->g;
  const CeedScalar         dt      = context->dt;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    CeedScalar U[5];
    for (int j = 0; j < 5; j++) U[j] = q[j][i];
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    State            s      = StateFromU(context, U, x_i);

    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ = q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*
    State grad_s[3];
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dx_i[3] = {0}, dU[5];
      for (CeedInt k = 0; k < 5; k++) dU[k] = Grad_q[0][k][i] * dXdx[0][j] + Grad_q[1][k][i] * dXdx[1][j] + Grad_q[2][k][i] * dXdx[2][j];
      dx_i[j]   = 1.;
      grad_s[j] = StateFromU_fwd(context, s, dU, x_i, dx_i);
    }

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate(grad_s, strain_rate);
    NewtonianStress(context, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(context, s, F_inviscid);

    // Total flux
    CeedScalar Flux[5][3];
    FluxTotal(F_inviscid, stress, Fe, Flux);

    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 5; k++) Grad_v[j][k][i] = wdetJ * (dXdx[j][0] * Flux[k][0] + dXdx[j][1] * Flux[k][1] + dXdx[j][2] * Flux[k][2]);
    }

    const CeedScalar body_force[5] = {0, s.U.density * g[0], s.U.density * g[1], s.U.density * g[2], 0};
    for (int j = 0; j < 5; j++) v[j][i] = wdetJ * body_force[j];

    // -- Stabilization method: none (Galerkin), SU, or SUPG
    CeedScalar Tau_d[3], stab[5][3], U_dot[5] = {0};
    Tau_diagPrim(context, s, dXdx, dt, Tau_d);
    Stabilization(context, s, Tau_d, grad_s, U_dot, body_force, x_i, stab);

    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) Grad_v[k][j][i] -= wdetJ * (stab[j][0] * dXdx[k][0] + stab[j][1] * dXdx[k][1] + stab[j][2] * dXdx[k][2]);
    }
  }  // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the Navier-Stokes equations (mentioned above) with
//   implicit time stepping method
//
//  SU   = Galerkin + grad(v) . ( Ai^T * Tau * (Aj q,j) )
//  SUPG = Galerkin + grad(v) . ( Ai^T * Tau * (q_dot + Aj q,j - body force) )
//                                       (diffussive terms will be added later)
//
// *****************************************************************************
CEED_QFUNCTION_HELPER int IFunction_Newtonian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateFromQi_t StateFromQi,
                                              StateFromQi_fwd_t StateFromQi_fwd) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*Grad_q)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3],
        (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*Grad_v)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1],
  (*jac_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];
  // *INDENT-ON*
  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar        *g       = context->g;
  const CeedScalar         dt      = context->dt;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    CeedScalar qi[5];
    for (CeedInt j = 0; j < 5; j++) qi[j] = q[j][i];
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    State            s      = StateFromQi(context, qi, x_i);

    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ = q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*
    State grad_s[3];
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dx_i[3] = {0}, dqi[5];
      for (CeedInt k = 0; k < 5; k++) dqi[k] = Grad_q[0][k][i] * dXdx[0][j] + Grad_q[1][k][i] * dXdx[1][j] + Grad_q[2][k][i] * dXdx[2][j];
      dx_i[j]   = 1.;
      grad_s[j] = StateFromQi_fwd(context, s, dqi, x_i, dx_i);
    }

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate(grad_s, strain_rate);
    NewtonianStress(context, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(context, s, F_inviscid);

    // Total flux
    CeedScalar Flux[5][3];
    FluxTotal(F_inviscid, stress, Fe, Flux);

    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 5; k++) Grad_v[j][k][i] = -wdetJ * (dXdx[j][0] * Flux[k][0] + dXdx[j][1] * Flux[k][1] + dXdx[j][2] * Flux[k][2]);
    }

    const CeedScalar body_force[5] = {0, s.U.density * g[0], s.U.density * g[1], s.U.density * g[2], 0};

    // -- Stabilization method: none (Galerkin), SU, or SUPG
    CeedScalar Tau_d[3], stab[5][3], U_dot[5] = {0}, qi_dot[5], dx0[3] = {0};
    for (int j = 0; j < 5; j++) qi_dot[j] = q_dot[j][i];
    State s_dot = StateFromQi_fwd(context, s, qi_dot, x_i, dx0);
    UnpackState_U(s_dot.U, U_dot);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = wdetJ * (U_dot[j] - body_force[j]);
    Tau_diagPrim(context, s, dXdx, dt, Tau_d);
    Stabilization(context, s, Tau_d, grad_s, U_dot, body_force, x_i, stab);

    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) Grad_v[k][j][i] += wdetJ * (stab[j][0] * dXdx[k][0] + stab[j][1] * dXdx[k][1] + stab[j][2] * dXdx[k][2]);
    }
    for (CeedInt j = 0; j < 5; j++) jac_data[j][i] = qi[j];
    for (CeedInt j = 0; j < 6; j++) jac_data[5 + j][i] = kmstress[j];
    for (CeedInt j = 0; j < 3; j++) jac_data[5 + 6 + j][i] = Tau_d[j];

  }  // End Quadrature Point Loop

  // Return
  return 0;
}

CEED_QFUNCTION(IFunction_Newtonian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_Newtonian(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(IFunction_Newtonian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_Newtonian(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

// *****************************************************************************
// This QFunction implements the jacobian of the Navier-Stokes equations
//   for implicit time stepping method.
// *****************************************************************************
CEED_QFUNCTION_HELPER int IJacobian_Newtonian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateFromQi_t StateFromQi,
                                              StateFromQi_fwd_t StateFromQi_fwd) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*Grad_dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2], (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3],
        (*jac_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*Grad_v)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar        *g       = context->g;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ = q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*

    CeedScalar qi[5], kmstress[6], Tau_d[3];
    for (int j = 0; j < 5; j++) qi[j] = jac_data[j][i];
    for (int j = 0; j < 6; j++) kmstress[j] = jac_data[5 + j][i];
    for (int j = 0; j < 3; j++) Tau_d[j] = jac_data[5 + 6 + j][i];
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    State            s      = StateFromQi(context, qi, x_i);

    CeedScalar dqi[5], dx0[3] = {0};
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];
    State ds = StateFromQi_fwd(context, s, dqi, x_i, dx0);

    State grad_ds[3];
    for (int j = 0; j < 3; j++) {
      CeedScalar dqi_j[5];
      for (int k = 0; k < 5; k++) dqi_j[k] = Grad_dq[0][k][i] * dXdx[0][j] + Grad_dq[1][k][i] * dXdx[1][j] + Grad_dq[2][k][i] * dXdx[2][j];
      grad_ds[j] = StateFromQi_fwd(context, s, dqi_j, x_i, dx0);
    }

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate(grad_ds, dstrain_rate);
    NewtonianStress(context, dstrain_rate, dkmstress);
    KMUnpack(dkmstress, dstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux_fwd(context, s.Y, ds.Y, grad_ds, stress, dstress, dFe);

    StateConservative dF_inviscid[3];
    FluxInviscid_fwd(context, s, ds, dF_inviscid);

    // Total flux
    CeedScalar dFlux[5][3];
    FluxTotal(dF_inviscid, dstress, dFe, dFlux);

    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 5; k++) Grad_v[j][k][i] = -wdetJ * (dXdx[j][0] * dFlux[k][0] + dXdx[j][1] * dFlux[k][1] + dXdx[j][2] * dFlux[k][2]);
    }

    const CeedScalar dbody_force[5] = {0, ds.U.density * g[0], ds.U.density * g[1], ds.U.density * g[2], 0};
    CeedScalar       dU[5]          = {0.};
    UnpackState_U(ds.U, dU);
    for (int j = 0; j < 5; j++) v[j][i] = wdetJ * (context->ijacobian_time_shift * dU[j] - dbody_force[j]);

    // -- Stabilization method: none (Galerkin), SU, or SUPG
    CeedScalar dstab[5][3], U_dot[5] = {0};
    for (CeedInt j = 0; j < 5; j++) U_dot[j] = context->ijacobian_time_shift * dU[j];
    Stabilization(context, s, Tau_d, grad_ds, U_dot, dbody_force, x_i, dstab);

    for (int j = 0; j < 5; j++) {
      for (int k = 0; k < 3; k++) Grad_v[k][j][i] += wdetJ * (dstab[j][0] * dXdx[k][0] + dstab[j][1] * dXdx[k][1] + dstab[j][2] * dXdx[k][2]);
    }
  }  // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(IJacobian_Newtonian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IJacobian_Newtonian(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(IJacobian_Newtonian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IJacobian_Newtonian(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

// *****************************************************************************
// Compute boundary integral (ie. for strongly set inflows)
// *****************************************************************************
CEED_QFUNCTION_HELPER int BoundaryIntegral(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateFromQi_t StateFromQi,
                                           StateFromQi_fwd_t StateFromQi_fwd) {
  //*INDENT-OFF*
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*Grad_q)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2], (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*jac_data_sur)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];

  //*INDENT-ON*

  const NewtonianIdealGasContext context     = (NewtonianIdealGasContext)ctx;
  const bool                     is_implicit = context->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s      = StateFromQi(context, qi, x_i);

    const CeedScalar wdetJb = (is_implicit ? -1. : 1.) * q_data_sur[0][i];
    // ---- Normal vector
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};

    const CeedScalar dXdx[2][3] = {
        {q_data_sur[4][i], q_data_sur[5][i], q_data_sur[6][i]},
        {q_data_sur[7][i], q_data_sur[8][i], q_data_sur[9][i]}
    };

    State grad_s[3];
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dx_i[3] = {0}, dqi[5];
      for (CeedInt k = 0; k < 5; k++) dqi[k] = Grad_q[0][k][i] * dXdx[0][j] + Grad_q[1][k][i] * dXdx[1][j];
      dx_i[j]   = 1.;
      grad_s[j] = StateFromQi_fwd(context, s, dqi, x_i, dx_i);
    }

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate(grad_s, strain_rate);
    NewtonianStress(context, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(context, s, F_inviscid);

    CeedScalar Flux[5];
    FluxTotal_Boundary(F_inviscid, stress, Fe, norm, Flux);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    for (int j = 0; j < 5; j++) jac_data_sur[j][i] = qi[j];
    for (int j = 0; j < 6; j++) jac_data_sur[5 + j][i] = kmstress[j];
  }
  return 0;
}

CEED_QFUNCTION(BoundaryIntegral_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(BoundaryIntegral_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

// *****************************************************************************
// Jacobian for "set nothing" boundary integral
// *****************************************************************************
CEED_QFUNCTION_HELPER int BoundaryIntegral_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                    StateFromQi_t StateFromQi, StateFromQi_fwd_t StateFromQi_fwd) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*Grad_dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2], (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3],
        (*jac_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const NewtonianIdealGasContext context  = (NewtonianIdealGasContext)ctx;
  const bool                     implicit = context->is_implicit;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x_i[3]     = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar wdetJb     = (implicit ? -1. : 1.) * q_data_sur[0][i];
    const CeedScalar norm[3]    = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};
    const CeedScalar dXdx[2][3] = {
        {q_data_sur[4][i], q_data_sur[5][i], q_data_sur[6][i]},
        {q_data_sur[7][i], q_data_sur[8][i], q_data_sur[9][i]}
    };

    CeedScalar qi[5], kmstress[6], dqi[5], dx_i[3] = {0.};
    for (int j = 0; j < 5; j++) qi[j] = jac_data_sur[j][i];
    for (int j = 0; j < 6; j++) kmstress[j] = jac_data_sur[5 + j][i];
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];

    State s  = StateFromQi(context, qi, x_i);
    State ds = StateFromQi_fwd(context, s, dqi, x_i, dx_i);

    State grad_ds[3];
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dx_i[3] = {0}, dqi_j[5];
      for (CeedInt k = 0; k < 5; k++) dqi_j[k] = Grad_dq[0][k][i] * dXdx[0][j] + Grad_dq[1][k][i] * dXdx[1][j];
      dx_i[j]    = 1.;
      grad_ds[j] = StateFromQi_fwd(context, s, dqi_j, x_i, dx_i);
    }

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate(grad_ds, dstrain_rate);
    NewtonianStress(context, dstrain_rate, dkmstress);
    KMUnpack(dkmstress, dstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux_fwd(context, s.Y, ds.Y, grad_ds, stress, dstress, dFe);

    StateConservative dF_inviscid[3];
    FluxInviscid_fwd(context, s, ds, dF_inviscid);

    CeedScalar dFlux[5];
    FluxTotal_Boundary(dF_inviscid, dstress, dFe, norm, dFlux);

    for (int j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }  // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(BoundaryIntegral_Jacobian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral_Jacobian(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(BoundaryIntegral_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral_Jacobian(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

// *****************************************************************************
// Outflow boundary condition, weakly setting a constant pressure
// *****************************************************************************
CEED_QFUNCTION_HELPER int PressureOutflow(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateFromQi_t StateFromQi,
                                          StateFromQi_fwd_t StateFromQi_fwd) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*Grad_q)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2], (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0], (*jac_data_sur)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];
  // *INDENT-ON*

  const NewtonianIdealGasContext context  = (NewtonianIdealGasContext)ctx;
  const bool                     implicit = context->is_implicit;
  const CeedScalar               P0       = context->P0;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s      = StateFromQi(context, qi, x_i);
    s.Y.pressure            = P0;

    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // ---- Normal vector
    const CeedScalar norm[3] = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};

    const CeedScalar dXdx[2][3] = {
        {q_data_sur[4][i], q_data_sur[5][i], q_data_sur[6][i]},
        {q_data_sur[7][i], q_data_sur[8][i], q_data_sur[9][i]}
    };

    State grad_s[3];
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dx_i[3] = {0}, dqi[5];
      for (CeedInt k = 0; k < 5; k++) dqi[k] = Grad_q[0][k][i] * dXdx[0][j] + Grad_q[1][k][i] * dXdx[1][j];
      dx_i[j]   = 1.;
      grad_s[j] = StateFromQi_fwd(context, s, dqi, x_i, dx_i);
    }

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate(grad_s, strain_rate);
    NewtonianStress(context, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(context, s, F_inviscid);

    CeedScalar Flux[5];
    FluxTotal_Boundary(F_inviscid, stress, Fe, norm, Flux);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    // Save values for Jacobian
    for (int j = 0; j < 5; j++) jac_data_sur[j][i] = qi[j];
    for (int j = 0; j < 6; j++) jac_data_sur[5 + j][i] = kmstress[j];
  }  // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(PressureOutflow_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(PressureOutflow_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

// *****************************************************************************
// Jacobian for weak-pressure outflow boundary condition
// *****************************************************************************
CEED_QFUNCTION_HELPER int PressureOutflow_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                   StateFromQi_t StateFromQi, StateFromQi_fwd_t StateFromQi_fwd) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*Grad_dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
        (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2], (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3],
        (*jac_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const NewtonianIdealGasContext context  = (NewtonianIdealGasContext)ctx;
  const bool                     implicit = context->is_implicit;

  CeedPragmaSIMD
      // Quadrature Point Loop
      for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar x_i[3]     = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar wdetJb     = (implicit ? -1. : 1.) * q_data_sur[0][i];
    const CeedScalar norm[3]    = {q_data_sur[1][i], q_data_sur[2][i], q_data_sur[3][i]};
    const CeedScalar dXdx[2][3] = {
        {q_data_sur[4][i], q_data_sur[5][i], q_data_sur[6][i]},
        {q_data_sur[7][i], q_data_sur[8][i], q_data_sur[9][i]}
    };

    CeedScalar qi[5], kmstress[6], dqi[5], dx_i[3] = {0.};
    for (int j = 0; j < 5; j++) qi[j] = jac_data_sur[j][i];
    for (int j = 0; j < 6; j++) kmstress[j] = jac_data_sur[5 + j][i];
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];

    State s       = StateFromQi(context, qi, x_i);
    State ds      = StateFromQi_fwd(context, s, dqi, x_i, dx_i);
    s.Y.pressure  = context->P0;
    ds.Y.pressure = 0.;

    State grad_ds[3];
    for (CeedInt j = 0; j < 3; j++) {
      CeedScalar dx_i[3] = {0}, dqi_j[5];
      for (CeedInt k = 0; k < 5; k++) dqi_j[k] = Grad_dq[0][k][i] * dXdx[0][j] + Grad_dq[1][k][i] * dXdx[1][j];
      dx_i[j]    = 1.;
      grad_ds[j] = StateFromQi_fwd(context, s, dqi_j, x_i, dx_i);
    }

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate(grad_ds, dstrain_rate);
    NewtonianStress(context, dstrain_rate, dkmstress);
    KMUnpack(dkmstress, dstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux_fwd(context, s.Y, ds.Y, grad_ds, stress, dstress, dFe);

    StateConservative dF_inviscid[3];
    FluxInviscid_fwd(context, s, ds, dF_inviscid);

    CeedScalar dFlux[5];
    FluxTotal_Boundary(dF_inviscid, dstress, dFe, norm, dFlux);

    for (int j = 0; j < 5; j++) v[j][i] = -wdetJb * dFlux[j];
  }  // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(PressureOutflow_Jacobian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow_Jacobian(ctx, Q, in, out, StateFromU, StateFromU_fwd);
}

CEED_QFUNCTION(PressureOutflow_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return PressureOutflow_Jacobian(ctx, Q, in, out, StateFromY, StateFromY_fwd);
}

#endif  // newtonian_h
