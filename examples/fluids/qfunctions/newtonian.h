// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
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

CEED_QFUNCTION_HELPER void InternalDampingLayer(const NewtonianIdealGasContext context, const State s, const CeedScalar sigma, CeedScalar damp_Y[5],
                                                CeedScalar damp_residual[5]) {
  ScaleN(damp_Y, sigma, 5);
  State damp_s = StateFromY_fwd(context, s, damp_Y);

  CeedScalar U[5];
  UnpackState_U(damp_s.U, U);
  for (int i = 0; i < 5; i++) damp_residual[i] += U[i];
}

// *****************************************************************************
// This QFunction sets a "still" initial condition for generic Newtonian IG problems
// *****************************************************************************
CEED_QFUNCTION_HELPER int ICsNewtonianIG(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  // Inputs

  // Outputs
  CeedScalar(*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Context
  const SetupContext context = (SetupContext)ctx;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar q[5] = {0.};
    State      s    = StateFromPrimitive(&context->gas, context->reference);
    StateToQ(&context->gas, s, q, state_var);
    for (CeedInt j = 0; j < 5; j++) q0[j][i] = q[j];
  }  // End of Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(ICsNewtonianIG_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsNewtonianIG(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}
CEED_QFUNCTION(ICsNewtonianIG_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return ICsNewtonianIG(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

// *****************************************************************************
// This QFunction implements the following formulation of Navier-Stokes with explicit time stepping method
//
// This is 3D compressible Navier-Stokes in conservation form with state variables of density, momentum density, and total energy density.
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
// We require the product of the inverse of the Jacobian (dXdx_j,k) and its transpose (dXdx_k,j) to properly compute integrals of the form: int( gradv
// gradu )
// *****************************************************************************
CEED_QFUNCTION(RHSFunction_Newtonian)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data)        = in[2];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]         = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*Grad_v)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar        *g       = context->g;
  const CeedScalar         dt      = context->dt;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar U[5], wdetJ, dXdx[3][3];
    for (int j = 0; j < 5; j++) U[j] = q[j][i];
    StoredValuesUnpack(Q, i, 0, 1, q_data, &wdetJ);
    StoredValuesUnpack(Q, i, 1, 9, q_data, (CeedScalar *)dXdx);
    State s = StateFromU(context, U);

    State grad_s[3];
    StatePhysicalGradientFromReference(Q, i, context, s, STATEVAR_CONSERVATIVE, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(context, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(context, s, F_inviscid);

    // Total flux
    CeedScalar Flux[5][3];
    FluxTotal(F_inviscid, stress, Fe, Flux);

    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) Grad_v[k][j][i] = wdetJ * (dXdx[k][0] * Flux[j][0] + dXdx[k][1] * Flux[j][1] + dXdx[k][2] * Flux[j][2]);
    }

    const CeedScalar body_force[5] = {0, s.U.density * g[0], s.U.density * g[1], s.U.density * g[2], Dot3(s.U.momentum, g)};
    for (int j = 0; j < 5; j++) v[j][i] = wdetJ * body_force[j];

    if (context->idl_enable) {
      const CeedScalar sigma = LinearRampCoefficient(context->idl_amplitude, context->idl_length, context->idl_start, x_i[0]);
      CeedScalar damp_state[5] = {s.Y.pressure - P0, 0, 0, 0, 0}, idl_residual[5] = {0.};
      InternalDampingLayer(context, s, sigma, damp_state, idl_residual);
      for (int j = 0; j < 5; j++) v[j][i] -= wdetJ * idl_residual[j];
    }

    // -- Stabilization method: none (Galerkin), SU, or SUPG
    CeedScalar Tau_d[3], stab[5][3], U_dot[5] = {0};
    Tau_diagPrim(context, s, dXdx, dt, Tau_d);
    Stabilization(context, s, Tau_d, grad_s, U_dot, body_force, stab);

    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) Grad_v[k][j][i] -= wdetJ * (stab[j][0] * dXdx[k][0] + stab[j][1] * dXdx[k][1] + stab[j][2] * dXdx[k][2]);
    }
  }  // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the Navier-Stokes equations (mentioned above) with implicit time stepping method
//
//  SU   = Galerkin + grad(v) . ( Ai^T * Tau * (Aj q,j) )
//  SUPG = Galerkin + grad(v) . ( Ai^T * Tau * (q_dot + Aj q,j - body force) )
//                                       (diffusive terms will be added later)
// *****************************************************************************
CEED_QFUNCTION_HELPER int IFunction_Newtonian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  // Inputs
  const CeedScalar(*q)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)            = in[1];
  const CeedScalar(*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  const CeedScalar(*q_data)            = in[3];
  const CeedScalar(*x)[CEED_Q_VLA]     = (const CeedScalar(*)[CEED_Q_VLA])in[4];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]         = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*Grad_v)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  CeedScalar(*jac_data)              = out[2];

  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar        *g       = context->g;
  const CeedScalar         dt      = context->dt;
  const CeedScalar         P0      = context->P0;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5]  = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    const State      s      = StateFromQ(context, qi, state_var);

    CeedScalar wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);
    State grad_s[3];
    StatePhysicalGradientFromReference(Q, i, context, s, state_var, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(context, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(context, s, F_inviscid);

    // Total flux
    CeedScalar Flux[5][3];
    FluxTotal(F_inviscid, stress, Fe, Flux);

    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        Grad_v[k][j][i] = -wdetJ * (dXdx[k][0] * Flux[j][0] + dXdx[k][1] * Flux[j][1] + dXdx[k][2] * Flux[j][2]);
      }
    }
// worked for exponential    const CeedScalar Amag=100.0;
// worked for cubic    const CeedScalar Amag=200000.0;
    const CeedScalar Amag=16000.0;
//    const CeedScalar amsig = -1.0*(Amag - LinearRampCoefficient(Amag, 0.1, -0.1, x_i[0]))*Max(0.0,exp(-100.0*Min(0,qi[1]))-1.0);
    const CeedScalar ux=qi[1];
//cubic    const CeedScalar amsig = (Amag - LinearRampCoefficient(Amag, 0.1, -0.1, x_i[0]))*Min(0.0,ux*ux*ux);
    const CeedScalar amsig = (Amag - LinearRampCoefficient(Amag, 0.1, -0.1, x_i[0]))*Min(0.0,ux);
//    const CeedScalar amsig = 0.0;

    const CeedScalar body_force[5] = {0, s.U.density * (g[0]-amsig), s.U.density * g[1], s.U.density * g[2], Dot3(s.U.momentum, g)};

    // -- Stabilization method: none (Galerkin), SU, or SUPG
    CeedScalar Tau_d[3], stab[5][3], U_dot[5] = {0}, qi_dot[5];
    for (int j = 0; j < 5; j++) qi_dot[j] = q_dot[j][i];
    State s_dot = StateFromQ_fwd(context, s, qi_dot, state_var);
    UnpackState_U(s_dot.U, U_dot);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = wdetJ * (U_dot[j] - body_force[j]);
    if (context->idl_enable) {
      const CeedScalar sigma = LinearRampCoefficient(context->idl_amplitude, context->idl_length, context->idl_start, x_i[0]);
      StoredValuesPack(Q, i, 14, 1, &sigma, jac_data);
      CeedScalar damp_state[5] = {s.Y.pressure - P0, 0, 0, 0, 0}, idl_residual[5] = {0.};
      InternalDampingLayer(context, s, sigma, damp_state, idl_residual);
      for (int j = 0; j < 5; j++) v[j][i] += wdetJ * idl_residual[j];
    }

    Tau_diagPrim(context, s, dXdx, dt, Tau_d);
    Stabilization(context, s, Tau_d, grad_s, U_dot, body_force, stab);

    for (CeedInt j = 0; j < 5; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        Grad_v[k][j][i] += wdetJ * (stab[j][0] * dXdx[k][0] + stab[j][1] * dXdx[k][1] + stab[j][2] * dXdx[k][2]);
      }
    }
    StoredValuesPack(Q, i, 0, 5, qi, jac_data);
    StoredValuesPack(Q, i, 5, 6, kmstress, jac_data);
    StoredValuesPack(Q, i, 11, 3, Tau_d, jac_data);

  }  // End Quadrature Point Loop

  // Return
  return 0;
}

CEED_QFUNCTION(IFunction_Newtonian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_Newtonian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(IFunction_Newtonian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IFunction_Newtonian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// *****************************************************************************
// This QFunction implements the jacobian of the Navier-Stokes equations for implicit time stepping method.
// *****************************************************************************
CEED_QFUNCTION_HELPER int IJacobian_Newtonian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  // Inputs
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_dq)        = in[1];
  const CeedScalar(*q_data)         = in[2];
  const CeedScalar(*jac_data)       = in[3];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA]         = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*Grad_v)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar        *g       = context->g;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJ, dXdx[3][3];
    QdataUnpack_3D(Q, i, q_data, &wdetJ, dXdx);

    CeedScalar qi[5], kmstress[6], Tau_d[3];
    StoredValuesUnpack(Q, i, 0, 5, jac_data, qi);
    StoredValuesUnpack(Q, i, 5, 6, jac_data, kmstress);
    StoredValuesUnpack(Q, i, 11, 3, jac_data, Tau_d);
    State s = StateFromQ(context, qi, state_var);

    CeedScalar dqi[5];
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];
    State ds = StateFromQ_fwd(context, s, dqi, state_var);

    State grad_ds[3];
    StatePhysicalGradientFromReference(Q, i, context, s, state_var, Grad_dq, dXdx, grad_ds);

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate_State(grad_ds, dstrain_rate);
    NewtonianStress(context, dstrain_rate, dkmstress);
    KMUnpack(dkmstress, dstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux_fwd(context, s.Y, ds.Y, grad_ds, stress, dstress, dFe);

    StateConservative dF_inviscid[3];
    FluxInviscid_fwd(context, s, ds, dF_inviscid);

    // Total flux
    CeedScalar dFlux[5][3];
    FluxTotal(dF_inviscid, dstress, dFe, dFlux);

    for (int j = 0; j < 5; j++) {
      for (int k = 0; k < 3; k++) Grad_v[k][j][i] = -wdetJ * (dXdx[k][0] * dFlux[j][0] + dXdx[k][1] * dFlux[j][1] + dXdx[k][2] * dFlux[j][2]);
    }

    const CeedScalar dbody_force[5] = {0, ds.U.density * g[0], ds.U.density * g[1], ds.U.density * g[2], Dot3(ds.U.momentum, g)};
    CeedScalar       dU[5]          = {0.};
    UnpackState_U(ds.U, dU);
    for (int j = 0; j < 5; j++) v[j][i] = wdetJ * (context->ijacobian_time_shift * dU[j] - dbody_force[j]);

    if (context->idl_enable) {
      const CeedScalar sigma         = jac_data[14 * Q + i];
      CeedScalar       damp_state[5] = {ds.Y.pressure, 0, 0, 0, 0}, idl_residual[5] = {0.};
      // This is a Picard-type linearization of the damping and could be replaced by an InternalDampingLayer_fwd that uses s and ds.
      InternalDampingLayer(context, s, sigma, damp_state, idl_residual);
      for (int j = 0; j < 5; j++) v[j][i] += wdetJ * idl_residual[j];
    }

    // -- Stabilization method: none (Galerkin), SU, or SUPG
    CeedScalar dstab[5][3], U_dot[5] = {0};
    for (CeedInt j = 0; j < 5; j++) U_dot[j] = context->ijacobian_time_shift * dU[j];
    Stabilization(context, s, Tau_d, grad_ds, U_dot, dbody_force, dstab);

    for (int j = 0; j < 5; j++) {
      for (int k = 0; k < 3; k++) Grad_v[k][j][i] += wdetJ * (dstab[j][0] * dXdx[k][0] + dstab[j][1] * dXdx[k][1] + dstab[j][2] * dXdx[k][2]);
    }
  }  // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(IJacobian_Newtonian_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IJacobian_Newtonian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(IJacobian_Newtonian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return IJacobian_Newtonian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// *****************************************************************************
// Compute boundary integral (ie. for strongly set inflows)
// *****************************************************************************
CEED_QFUNCTION_HELPER int BoundaryIntegral(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out, StateVariable state_var) {
  const CeedScalar(*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_q)        = in[1];
  const CeedScalar(*q_data_sur)    = in[2];

  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar(*jac_data_sur)  = out[1];

  const NewtonianIdealGasContext context     = (NewtonianIdealGasContext)ctx;
  const bool                     is_implicit = context->is_implicit;

  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    const CeedScalar qi[5] = {q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]};
    State            s     = StateFromQ(context, qi, state_var);

    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    State grad_s[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, context, s, state_var, Grad_q, dXdx, grad_s);

    CeedScalar strain_rate[6], kmstress[6], stress[3][3], Fe[3];
    KMStrainRate_State(grad_s, strain_rate);
    NewtonianStress(context, strain_rate, kmstress);
    KMUnpack(kmstress, stress);
    ViscousEnergyFlux(context, s.Y, grad_s, stress, Fe);

    StateConservative F_inviscid[3];
    FluxInviscid(context, s, F_inviscid);

    CeedScalar Flux[5];
    FluxTotal_Boundary(F_inviscid, stress, Fe, norm, Flux);

    for (CeedInt j = 0; j < 5; j++) v[j][i] = -wdetJb * Flux[j];

    StoredValuesPack(Q, i, 0, 5, qi, jac_data_sur);
    StoredValuesPack(Q, i, 5, 6, kmstress, jac_data_sur);
  }
  return 0;
}

CEED_QFUNCTION(BoundaryIntegral_Conserv)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(BoundaryIntegral_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

// *****************************************************************************
// Jacobian for "set nothing" boundary integral
// *****************************************************************************
CEED_QFUNCTION_HELPER int BoundaryIntegral_Jacobian(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out,
                                                    StateVariable state_var) {
  // Inputs
  const CeedScalar(*dq)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];
  const CeedScalar(*Grad_dq)        = in[1];
  const CeedScalar(*q_data_sur)     = in[2];
  const CeedScalar(*jac_data_sur)   = in[4];

  // Outputs
  CeedScalar(*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const NewtonianIdealGasContext context     = (NewtonianIdealGasContext)ctx;
  const bool                     is_implicit = context->is_implicit;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    CeedScalar wdetJb, dXdx[2][3], norm[3];
    QdataBoundaryUnpack_3D(Q, i, q_data_sur, &wdetJb, dXdx, norm);
    wdetJb *= is_implicit ? -1. : 1.;

    CeedScalar qi[5], kmstress[6], dqi[5];
    StoredValuesUnpack(Q, i, 0, 5, jac_data_sur, qi);
    StoredValuesUnpack(Q, i, 5, 6, jac_data_sur, kmstress);
    for (int j = 0; j < 5; j++) dqi[j] = dq[j][i];

    State s  = StateFromQ(context, qi, state_var);
    State ds = StateFromQ_fwd(context, s, dqi, state_var);

    State grad_ds[3];
    StatePhysicalGradientFromReference_Boundary(Q, i, context, s, state_var, Grad_dq, dXdx, grad_ds);

    CeedScalar dstrain_rate[6], dkmstress[6], stress[3][3], dstress[3][3], dFe[3];
    KMStrainRate_State(grad_ds, dstrain_rate);
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
  return BoundaryIntegral_Jacobian(ctx, Q, in, out, STATEVAR_CONSERVATIVE);
}

CEED_QFUNCTION(BoundaryIntegral_Jacobian_Prim)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  return BoundaryIntegral_Jacobian(ctx, Q, in, out, STATEVAR_PRIMITIVE);
}

#endif  // newtonian_h
