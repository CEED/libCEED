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

#include <math.h>
#include <ceed.h>
#include "newtonian_types.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

typedef struct {
  CeedScalar pressure;
  CeedScalar velocity[3];
  CeedScalar temperature;
} StatePrimitive;

typedef struct {
  CeedScalar density;
  CeedScalar momentum[3];
  CeedScalar E_total;
} StateConservative;

typedef struct {
  StateConservative U;
  StatePrimitive Y;
} State;

CEED_QFUNCTION_HELPER CeedScalar Dot3(const CeedScalar u[3],
                                      const CeedScalar v[3]) {
  return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromConservative(
  NewtonianIdealGasContext gas, StateConservative U, const CeedScalar x[3]) {
  StatePrimitive Y;
  for (int i=0; i<3; i++) Y.velocity[i] = U.momentum[i] / U.density;
  CeedScalar e_kinetic = .5 * Dot3(Y.velocity, Y.velocity);
  CeedScalar e_potential = -Dot3(gas->g, x);
  CeedScalar e_total = U.E_total / U.density;
  CeedScalar e_internal = e_total - e_kinetic - e_potential;
  Y.temperature = e_internal / gas->cv;
  Y.pressure = (gas->cp / gas->cv - 1) * U.density * e_internal;
  return Y;
}

CEED_QFUNCTION_HELPER StatePrimitive StatePrimitiveFromConservative_fwd(
  NewtonianIdealGasContext gas, State s, StateConservative dU,
  const CeedScalar x[3], const CeedScalar dx[3]) {
  StatePrimitive dY;
  for (int i=0; i<3; i++) {
    dY.velocity[i] = (dU.momentum[i] - s.Y.velocity[i] * dU.density) / s.U.density;
  }
  CeedScalar e_kinetic = .5 * Dot3(s.Y.velocity, s.Y.velocity);
  CeedScalar de_kinetic = Dot3(dY.velocity, s.Y.velocity);
  CeedScalar e_potential = -Dot3(gas->g, x);
  CeedScalar de_potential = -Dot3(gas->g, dx);
  CeedScalar e_total = s.U.E_total / s.U.density;
  CeedScalar de_total = (dU.E_total - e_total * dU.density) / s.U.density;
  CeedScalar e_internal = e_total - e_kinetic - e_potential;
  CeedScalar de_internal = de_total - de_kinetic - de_potential;
  dY.temperature = de_internal / gas->cv;
  dY.pressure = (gas->cp / gas->cv - 1)
                * (dU.density * e_internal + s.U.density * de_internal);
  return dY;
}

CEED_QFUNCTION_HELPER State StateFromU(NewtonianIdealGasContext gas,
                                       const CeedScalar U[5], const CeedScalar x[3]) {
  State s;
  s.U.density = U[0];
  s.U.momentum[0] = U[1];
  s.U.momentum[1] = U[2];
  s.U.momentum[2] = U[3];
  s.U.E_total = U[4];
  s.Y = StatePrimitiveFromConservative(gas, s.U, x);
  return s;
}

CEED_QFUNCTION_HELPER void FluxInviscid(NewtonianIdealGasContext gas, State s,
                                        StateConservative Flux[3]) {
  for (int i=0; i<3; i++) {
    Flux[i].density = s.U.momentum[i];
    for (int j=0; j<3; j++)
      Flux[i].momentum[j] = s.U.momentum[i] * s.Y.velocity[j]
                            + s.Y.pressure * (i == j);
    Flux[i].E_total = (s.U.E_total + s.Y.pressure) * s.Y.velocity[i];
  }
}

CEED_QFUNCTION_HELPER void FluxInviscid_fwd(NewtonianIdealGasContext gas,
    State s, State ds, StateConservative dFlux[3]) {
  for (int i=0; i<3; i++) {
    dFlux[i].density = ds.U.momentum[i];
    for (int j=0; j<3; j++)
      dFlux[i].momentum[j] = ds.U.momentum[i] * s.Y.velocity[j] +
                             s.U.momentum[i] * ds.Y.velocity[j] + ds.Y.pressure * (i == j);
    dFlux[i].E_total = (ds.U.E_total + ds.Y.pressure) * s.Y.velocity[i] +
                       (s.U.E_total + s.Y.pressure) * ds.Y.velocity[i];
  }
}

// Kelvin-Mandel notation
CEED_QFUNCTION_HELPER void KMStrainRate(const State grad_s[3],
                                        CeedScalar strain_rate[6]) {
  const CeedScalar weight = 1 / sqrt(2.);
  strain_rate[0] = grad_s[0].Y.velocity[0];
  strain_rate[1] = grad_s[1].Y.velocity[1];
  strain_rate[2] = grad_s[2].Y.velocity[2];
  strain_rate[3] = weight * (grad_s[2].Y.velocity[1] + grad_s[1].Y.velocity[2]);
  strain_rate[4] = weight * (grad_s[2].Y.velocity[0] + grad_s[0].Y.velocity[2]);
  strain_rate[5] = weight * (grad_s[1].Y.velocity[0] + grad_s[0].Y.velocity[1]);
}

CEED_QFUNCTION_HELPER void KMUnpack(const CeedScalar v[6], CeedScalar A[3][3]) {
  const CeedScalar weight = 1 / sqrt(2.);
  A[0][0] = v[0];
  A[1][1] = v[1];
  A[2][2] = v[2];
  A[2][1] = A[1][2] = weight * v[3];
  A[2][0] = A[0][2] = weight * v[4];
  A[1][0] = A[0][1] = weight * v[5];
}

CEED_QFUNCTION_HELPER void NewtonianStress(NewtonianIdealGasContext gas,
    const CeedScalar strain_rate[6], CeedScalar stress[6]) {
  CeedScalar div_u = strain_rate[0] + strain_rate[1] + strain_rate[2];
  for (int i=0; i<6; i++) {
    stress[i] = gas->mu * (2 * strain_rate[i] + gas->lambda * div_u * (i < 3));
  }
}

CEED_QFUNCTION_HELPER void ViscousEnergyFlux(NewtonianIdealGasContext gas,
    StatePrimitive Y, const State grad_s[3], const CeedScalar stress[3][3],
    CeedScalar Fe[3]) {
  for (int i=0; i<3; i++) {
    Fe[i] = - Y.velocity[0] * stress[0][i]
            - Y.velocity[1] * stress[1][i]
            - Y.velocity[2] * stress[2][i]
            - gas->k * grad_s[i].Y.temperature;
  }
}

// *****************************************************************************
// Helper function for computing flux Jacobian
// *****************************************************************************
CEED_QFUNCTION_HELPER void computeFluxJacobian_NS(CeedScalar dF[3][5][5],
    const CeedScalar rho, const CeedScalar u[3], const CeedScalar E,
    const CeedScalar gamma, const CeedScalar g[3], const CeedScalar x[3]) {
  CeedScalar u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]; // Velocity square
  CeedScalar e_potential = -(g[0]*x[0] + g[1]*x[1] + g[2]*x[2]);
  for (CeedInt i=0; i<3; i++) { // Jacobian matrices for 3 directions
    for (CeedInt j=0; j<3; j++) { // Rows of each Jacobian matrix
      dF[i][j+1][0] = ((i==j) ? ((gamma-1.)*(u_sq/2. - e_potential)) : 0.) -
                      u[i]*u[j];
      for (CeedInt k=0; k<3; k++) { // Columns of each Jacobian matrix
        dF[i][0][k+1]   = ((i==k) ? 1. : 0.);
        dF[i][j+1][k+1] = ((j==k) ? u[i] : 0.) +
                          ((i==k) ? u[j] : 0.) -
                          ((i==j) ? u[k] : 0.) * (gamma-1.);
        dF[i][4][k+1]   = ((i==k) ? (E*gamma/rho - (gamma-1.)*u_sq/2.) : 0.) -
                          (gamma-1.)*u[i]*u[k];
      }
      dF[i][j+1][4] = ((i==j) ? (gamma-1.) : 0.);
    }
    dF[i][4][0] = u[i] * ((gamma-1.)*u_sq - E*gamma/rho);
    dF[i][4][4] = u[i] * gamma;
  }
}

// *****************************************************************************
// Helper function for computing flux Jacobian of Primitive variables
// *****************************************************************************
CEED_QFUNCTION_HELPER void computeFluxJacobian_NSp(CeedScalar dF[3][5][5],
    const CeedScalar rho, const CeedScalar u[3], const CeedScalar E,
    const CeedScalar Rd, const CeedScalar cv) {
  CeedScalar u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]; // Velocity square
  // TODO Add in gravity's contribution

  CeedScalar T    = ( E / rho - u_sq / 2. ) / cv;
  CeedScalar drdT = -rho / T;
  CeedScalar drdP = 1. / ( Rd * T);
  CeedScalar etot =  E / rho ;
  CeedScalar e2p  = drdP * etot + 1. ;
  CeedScalar e3p  = ( E  + rho * Rd * T );
  CeedScalar e4p  = drdT * etot + rho * cv ;

  for (CeedInt i=0; i<3; i++) { // Jacobian matrices for 3 directions
    for (CeedInt j=0; j<3; j++) { // j counts F^{m_j}
//        [row][col] of A_i
      dF[i][j+1][0] = drdP * u[i] * u[j] + ((i==j) ? 1. : 0.); // F^{{m_j} wrt p
      for (CeedInt k=0; k<3; k++) { // k counts the wrt vel_k
        dF[i][0][k+1]   =  ((i==k) ? rho  : 0.);   // F^c wrt u_k
        dF[i][j+1][k+1] = (((j==k) ? u[i] : 0.) +  // F^m_j wrt u_k
                           ((i==k) ? u[j] : 0.) ) * rho;
        dF[i][4][k+1]   = rho * u[i] * u[k]
                          + ((i==k) ? e3p  : 0.) ; // F^e wrt u_k
      }
      dF[i][j+1][4] = drdT * u[i] * u[j]; // F^{m_j} wrt T
    }
    dF[i][4][0] = u[i] * e2p; // F^e wrt p
    dF[i][4][4] = u[i] * e4p; // F^e wrt T
    dF[i][0][0] = u[i] * drdP; // F^c wrt p
    dF[i][0][4] = u[i] * drdT; // F^c wrt T
  }
}

CEED_QFUNCTION_HELPER void PrimitiveToConservative_fwd(const CeedScalar rho,
    const CeedScalar u[3], const CeedScalar E, const CeedScalar Rd,
    const CeedScalar cv, const CeedScalar dY[5], CeedScalar dU[5]) {
  CeedScalar u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
  CeedScalar T    = ( E / rho - u_sq / 2. ) / cv;
  CeedScalar drdT = -rho / T;
  CeedScalar drdP = 1. / ( Rd * T);
  dU[0] = drdP * dY[0] + drdT * dY[4];
  CeedScalar de_kinetic = 0;
  for (int i=0; i<3; i++) {
    dU[1+i] = dU[0] * u[i] + rho * dY[1+i];
    de_kinetic += u[i] * dY[1+i];
  }
  dU[4] = rho * cv * dY[4] + dU[0] * cv * T // internal energy: rho * e
          + rho * de_kinetic + .5 * dU[0] * u_sq; // kinetic energy: .5 * rho * |u|^2
}

// *****************************************************************************
// Helper function for computing Tau elements (stabilization constant)
//   Model from:
//     PHASTA
//
//   Tau[i] = itau=0 which is diagonal-Shakib (3 values still but not spatial)
//
// Where NOT UPDATED YET
// *****************************************************************************
CEED_QFUNCTION_HELPER void Tau_diagPrim(CeedScalar Tau_d[3],
                                        const CeedScalar dXdx[3][3], const CeedScalar u[3],
                                        const CeedScalar cv, const NewtonianIdealGasContext newt_ctx,
                                        const CeedScalar mu, const CeedScalar dt,
                                        const CeedScalar rho) {
  // Context
  const CeedScalar Ctau_t = newt_ctx->Ctau_t;
  const CeedScalar Ctau_v = newt_ctx->Ctau_v;
  const CeedScalar Ctau_C = newt_ctx->Ctau_C;
  const CeedScalar Ctau_M = newt_ctx->Ctau_M;
  const CeedScalar Ctau_E = newt_ctx->Ctau_E;
  CeedScalar gijd[6];
  CeedScalar tau;
  CeedScalar dts;
  CeedScalar fact;

  //*INDENT-OFF*
  gijd[0] =   dXdx[0][0] * dXdx[0][0]
            + dXdx[1][0] * dXdx[1][0]
            + dXdx[2][0] * dXdx[2][0];

  gijd[1] =   dXdx[0][0] * dXdx[0][1]
            + dXdx[1][0] * dXdx[1][1]
            + dXdx[2][0] * dXdx[2][1];

  gijd[2] =   dXdx[0][1] * dXdx[0][1]
            + dXdx[1][1] * dXdx[1][1]
            + dXdx[2][1] * dXdx[2][1];

  gijd[3] =   dXdx[0][0] * dXdx[0][2]
            + dXdx[1][0] * dXdx[1][2]
            + dXdx[2][0] * dXdx[2][2];

  gijd[4] =   dXdx[0][1] * dXdx[0][2]
            + dXdx[1][1] * dXdx[1][2]
            + dXdx[2][1] * dXdx[2][2];

  gijd[5] =   dXdx[0][2] * dXdx[0][2]
            + dXdx[1][2] * dXdx[1][2]
            + dXdx[2][2] * dXdx[2][2];
  //*INDENT-ON*

  dts = Ctau_t / dt ;

  tau = rho*rho*((4. * dts * dts)
                 + u[0] * ( u[0] * gijd[0] + 2. * ( u[1] * gijd[1] + u[2] * gijd[3]))
                 + u[1] * ( u[1] * gijd[2] + 2. *   u[2] * gijd[4])
                 + u[2] *   u[2] * gijd[5])
        + Ctau_v* mu * mu *
        (gijd[0]*gijd[0] + gijd[2]*gijd[2] + gijd[5]*gijd[5] +
         + 2. * (gijd[1]*gijd[1] + gijd[3]*gijd[3] + gijd[4]*gijd[4]));

  fact=sqrt(tau);

  Tau_d[0] = Ctau_C * fact / (rho*(gijd[0] + gijd[2] + gijd[5]))*0.125;

  Tau_d[1] = Ctau_M / fact;
  Tau_d[2] = Ctau_E / ( fact * cv );

// consider putting back the way I initially had it  Ctau_E * Tau_d[1] /cv
//  to avoid a division if the compiler is smart enough to see that cv IS
// a constant that it could invert once for all elements
// but in that case energy tau is scaled by the product of Ctau_E * Ctau_M
// OR we could absorb cv into Ctau_E but this puts more burden on user to
// know how to change constants with a change of fluid or units.  Same for
// Ctau_v * mu * mu IF AND ONLY IF we don't add viscosity law =f(T)
}

// *****************************************************************************
// This QFunction sets a "still" initial condition for generic Newtonian IG problems
// *****************************************************************************
CEED_QFUNCTION(ICsNewtonianIG)(void *ctx, CeedInt Q,
                               const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Context
  const SetupContext context = (SetupContext)ctx;
  const CeedScalar theta0    = context->theta0;
  const CeedScalar P0        = context->P0;
  const CeedScalar cv        = context->cv;
  const CeedScalar cp        = context->cp;
  const CeedScalar *g        = context->g;
  const CeedScalar Rd        = cp - cv;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar q[5] = {0.};

    // Setup
    // -- Coordinates
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};
    const CeedScalar e_potential = -(g[0]*x[0] + g[1]*x[1] + g[2]*x[2]);

    // -- Density
    const CeedScalar rho = P0 / (Rd*theta0);

    // Initial Conditions
    q[0] = rho;
    q[1] = 0.0;
    q[2] = 0.0;
    q[3] = 0.0;
    q[4] = rho * (cv*theta0 + e_potential);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop
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
CEED_QFUNCTION(RHSFunction_Newtonian)(void *ctx, CeedInt Q,
                                      const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar mu     = context->mu;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar *g     = context->g;
  const CeedScalar dt     = context->dt;
  const CeedScalar gamma  = cp / cv;
  const CeedScalar Rd     = cp - cv;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar U[5];
    for (int j=0; j<5; j++) U[j] = q[j][i];
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    State s = StateFromU(context, U, x_i);

    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ      =   q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*

    State grad_s[3];
    for (int j=0; j<3; j++) {
      CeedScalar dx_i[3] = {0};
      grad_s[j].U.density = dq[0][0][i] * dXdx[0][j]
                            + dq[1][0][i] * dXdx[1][j] + dq[2][0][i] * dXdx[2][j];
      for (int k=0; k<3; k++) grad_s[j].U.momentum[k] = dq[0][k+1][i] * dXdx[0][j]
            + dq[1][k+1][i] * dXdx[1][j] + dq[2][k+1][i] * dXdx[2][j];
      grad_s[j].U.E_total = dq[0][4][i] * dXdx[0][j] + dq[1][4][i] * dXdx[1][j] +
                            dq[2][4][i] * dXdx[2][j];
      dx_i[j] = 1.;
      grad_s[j].Y = StatePrimitiveFromConservative_fwd(context, s, grad_s[j].U,
                    x_i, dx_i);
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
    for (int j=0; j<3; j++) {
      Flux[0][j] = F_inviscid[j].density;
      for (int k=0; k<3; k++)
        Flux[k+1][j] = F_inviscid[j].momentum[k] - stress[k][j];
      Flux[4][j] = F_inviscid[j].E_total + Fe[j];
    }

    for (int j=0; j<3; j++) {
      for (int k=0; k<5; k++) {
        dv[j][k][i] = wdetJ * (dXdx[j][0] * Flux[k][0] +
                               dXdx[j][1] * Flux[k][1] +
                               dXdx[j][2] * Flux[k][2]);
      }
    }

    const CeedScalar body_force[5] = {0, s.U.density *g[0], s.U.density *g[1], s.U.density *g[2], 0};
    for (int j=0; j<5; j++)
      v[j][i] = wdetJ * body_force[j];

    // jacob_F_conv[3][5][5] = dF(convective)/dq at each direction
    CeedScalar jacob_F_conv[3][5][5] = {0};
    computeFluxJacobian_NS(jacob_F_conv, s.U.density, s.Y.velocity, s.U.E_total,
                           gamma, g, x_i);
    CeedScalar grad_U[5][3];
    for (int j=0; j<3; j++) {
      grad_U[0][j] = grad_s[j].U.density;
      for (int k=0; k<3; k++) grad_U[k+1][j] = grad_s[j].U.momentum[k];
      grad_U[4][j] = grad_s[j].U.E_total;
    }

    // strong_conv = dF/dq * dq/dx    (Strong convection)
    CeedScalar strong_conv[5] = {0};
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          strong_conv[k] += jacob_F_conv[j][k][l] * grad_U[l][j];

    // -- Stabilization method: none, SU, or SUPG
    CeedScalar stab[5][3] = {{0.}};
    CeedScalar tau_strong_conv[5] = {0.}, tau_strong_conv_conservative[5] = {0};
    CeedScalar Tau_d[3] = {0.};
    switch (context->stabilization) {
    case STAB_NONE:        // Galerkin
      break;
    case STAB_SU:        // SU
      Tau_diagPrim(Tau_d, dXdx, s.Y.velocity, cv, context, mu, dt, s.U.density);
      tau_strong_conv[0] = Tau_d[0] * strong_conv[0];
      tau_strong_conv[1] = Tau_d[1] * strong_conv[1];
      tau_strong_conv[2] = Tau_d[1] * strong_conv[2];
      tau_strong_conv[3] = Tau_d[1] * strong_conv[3];
      tau_strong_conv[4] = Tau_d[2] * strong_conv[4];
      PrimitiveToConservative_fwd(s.U.density, s.Y.velocity, s.U.E_total, Rd, cv,
                                  tau_strong_conv,
                                  tau_strong_conv_conservative);
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] += jacob_F_conv[j][k][l] * tau_strong_conv_conservative[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] -= wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    case STAB_SUPG:        // SUPG is not implemented for explicit scheme
      break;
    }

  } // End Quadrature Point Loop

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
CEED_QFUNCTION(IFunction_Newtonian)(void *ctx, CeedInt Q,
                                    const CeedScalar *const *in,
                                    CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
                   (*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3],
                   (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar mu     = context->mu;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar *g     = context->g;
  const CeedScalar dt     = context->dt;
  const CeedScalar gamma  = cp / cv;
  const CeedScalar Rd     = cp-cv;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar U[5];
    for (int j=0; j<5; j++) U[j] = q[j][i];
    const CeedScalar x_i[3] = {x[0][i], x[1][i], x[2][i]};
    State s = StateFromU(context, U, x_i);

    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ      =   q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*
    State grad_s[3];
    for (int j=0; j<3; j++) {
      CeedScalar dx_i[3];
      grad_s[j].U.density = dq[0][0][i] * dXdx[0][j]
                            + dq[1][0][i] * dXdx[1][j] + dq[2][0][i] * dXdx[2][j];
      for (int k=0; k<3; k++) grad_s[j].U.momentum[k] = dq[0][k+1][i] * dXdx[0][j]
            + dq[1][k+1][i] * dXdx[1][j] + dq[2][k+1][i] * dXdx[2][j];
      grad_s[j].U.E_total = dq[0][4][i] * dXdx[0][j] + dq[1][4][i] * dXdx[1][j] +
                            dq[2][4][i] * dXdx[2][j];
      dx_i[j] = 1.;
      grad_s[j].Y = StatePrimitiveFromConservative_fwd(context, s, grad_s[j].U,
                    x_i, dx_i);
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
    for (int j=0; j<3; j++) {
      Flux[0][j] = F_inviscid[j].density;
      for (int k=0; k<3; k++)
        Flux[k+1][j] = F_inviscid[j].momentum[k] - stress[k][j];
      Flux[4][j] = F_inviscid[j].E_total + Fe[j];
    }

    for (int j=0; j<3; j++) {
      for (int k=0; k<5; k++) {
        dv[j][k][i] = -wdetJ * (dXdx[j][0] * Flux[k][0] +
                                dXdx[j][1] * Flux[k][1] +
                                dXdx[j][2] * Flux[k][2]);
      }
    }

    const CeedScalar body_force[5] = {0, s.U.density *g[0], s.U.density *g[1], s.U.density *g[2], 0};
    for (int j=0; j<5; j++)
      v[j][i] = wdetJ * (q_dot[j][i] - body_force[j]);

    // jacob_F_conv[3][5][5] = dF(convective)/dq at each direction
    CeedScalar jacob_F_conv[3][5][5] = {0};
    computeFluxJacobian_NS(jacob_F_conv, s.U.density, s.Y.velocity, s.U.E_total,
                           gamma, g, x_i);
    CeedScalar grad_U[5][3];
    for (int j=0; j<3; j++) {
      grad_U[0][j] = grad_s[j].U.density;
      for (int k=0; k<3; k++) grad_U[k+1][j] = grad_s[j].U.momentum[k];
      grad_U[4][j] = grad_s[j].U.E_total;
    }

    // strong_conv = dF/dq * dq/dx    (Strong convection)
    CeedScalar strong_conv[5] = {0};
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          strong_conv[k] += jacob_F_conv[j][k][l] * grad_U[l][j];

    // Strong residual
    CeedScalar strong_res[5];
    for (int j=0; j<5; j++)
      strong_res[j] = q_dot[j][i] + strong_conv[j] - body_force[j];

    // -- Stabilization method: none, SU, or SUPG
    CeedScalar stab[5][3] = {{0.}};
    CeedScalar tau_strong_res[5] = {0.}, tau_strong_res_conservative[5] = {0};
    CeedScalar tau_strong_conv[5] = {0.}, tau_strong_conv_conservative[5] = {0};
    CeedScalar Tau_d[3] = {0.};
    switch (context->stabilization) {
    case STAB_NONE:        // Galerkin
      break;
    case STAB_SU:        // SU
      Tau_diagPrim(Tau_d, dXdx, s.Y.velocity, cv, context, mu, dt, s.U.density);
      tau_strong_conv[0] = Tau_d[0] * strong_conv[0];
      tau_strong_conv[1] = Tau_d[1] * strong_conv[1];
      tau_strong_conv[2] = Tau_d[1] * strong_conv[2];
      tau_strong_conv[3] = Tau_d[1] * strong_conv[3];
      tau_strong_conv[4] = Tau_d[2] * strong_conv[4];
      PrimitiveToConservative_fwd(s.U.density, s.Y.velocity, s.U.E_total, Rd, cv,
                                  tau_strong_conv, tau_strong_conv_conservative);
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] += jacob_F_conv[j][k][l] * tau_strong_conv_conservative[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] += wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    case STAB_SUPG:        // SUPG
      Tau_diagPrim(Tau_d, dXdx, s.Y.velocity, cv, context, mu, dt, s.U.density);
      tau_strong_res[0] = Tau_d[0] * strong_res[0];
      tau_strong_res[1] = Tau_d[1] * strong_res[1];
      tau_strong_res[2] = Tau_d[1] * strong_res[2];
      tau_strong_res[3] = Tau_d[1] * strong_res[3];
      tau_strong_res[4] = Tau_d[2] * strong_res[4];
// Alternate route (useful later with primitive variable code)
// this function was verified against PHASTA for as IC that was as close as possible
//    computeFluxJacobian_NSp(jacob_F_conv_p, rho, u, E, Rd, cv);
// it has also been verified to compute a correct through the following
//   stab[k][j] += jacob_F_conv_p[j][k][l] * tau_strong_res[l] // flux Jacobian wrt primitive
// applied in the triple loop below
//  However, it is more flops than using the existing Jacobian wrt q after q_{,Y} viz
      PrimitiveToConservative_fwd(s.U.density, s.Y.velocity, s.U.E_total, Rd, cv,
                                  tau_strong_res, tau_strong_res_conservative);
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] += jacob_F_conv[j][k][l] * tau_strong_res_conservative[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] += wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    }

  } // End Quadrature Point Loop

  // Return
  return 0;
}
// *****************************************************************************
#endif // newtonian_h
