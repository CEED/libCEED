// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Operator for Navier-Stokes example using PETSc


#ifndef newtonian_h
#define newtonian_h

#include <math.h>
#include <ceed.h>
#include "../navierstokes.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#ifndef newtonian_context_struct
#define newtonian_context_struct
typedef struct NewtonianIdealGasContext_ *NewtonianIdealGasContext;
struct NewtonianIdealGasContext_ {
  CeedScalar lambda;
  CeedScalar mu;
  CeedScalar k;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar g;
  CeedScalar c_tau;
  StabilizationType stabilization;
};
#endif

// *****************************************************************************
// Helper function for computing flux Jacobian
// *****************************************************************************
CEED_QFUNCTION_HELPER void computeFluxJacobian_NS(CeedScalar dF[3][5][5],
    const CeedScalar rho, const CeedScalar u[3], const CeedScalar E,
    const CeedScalar gamma, const CeedScalar g, CeedScalar z) {
  CeedScalar u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]; // Velocity square
  for (CeedInt i=0; i<3; i++) { // Jacobian matrices for 3 directions
    for (CeedInt j=0; j<3; j++) { // Rows of each Jacobian matrix
      dF[i][j+1][0] = ((i==j) ? ((gamma-1.)*(u_sq/2. - g*z)) : 0.) - u[i]*u[j];
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
// Helper function for computing Tau elements (stabilization constant)
//   Model from:
//     Stabilized Methods for Compressible Flows, Hughes et al 2010
//
//   Spatial criterion #2 - Tau is a 3x3 diagonal matrix
//   Tau[i] = c_tau h[i] Xi(Pe) / rho(A[i]) (no sum)
//
// Where
//   c_tau     = stabilization constant (0.5 is reported as "optimal")
//   h[i]      = 2 length(dxdX[i])
//   Pe        = Peclet number ( Pe = sqrt(u u) / dot(dXdx,u) diffusivity )
//   Xi(Pe)    = coth Pe - 1. / Pe (1. at large local Peclet number )
//   rho(A[i]) = spectral radius of the convective flux Jacobian i,
//               wave speed in direction i
// *****************************************************************************
CEED_QFUNCTION_HELPER void Tau_spatial(CeedScalar Tau_x[3],
                                       const CeedScalar dXdx[3][3], const CeedScalar u[3],
                                       const CeedScalar sound_speed, const CeedScalar c_tau) {
  for (int i=0; i<3; i++) {
    // length of element in direction i
    CeedScalar h = 2 / sqrt(dXdx[0][i]*dXdx[0][i] + dXdx[1][i]*dXdx[1][i] +
                            dXdx[2][i]*dXdx[2][i]);
    // fastest wave in direction i
    CeedScalar fastest_wave = fabs(u[i]) + sound_speed;
    Tau_x[i] = c_tau * h / fastest_wave;
  }
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
//
// Equation of State:
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
CEED_QFUNCTION(Newtonian)(void *ctx, CeedInt Q,
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
  const CeedScalar lambda = context->lambda;
  const CeedScalar mu     = context->mu;
  const CeedScalar k      = context->k;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar g      = context->g;
  const CeedScalar c_tau  = context->c_tau;
  const CeedScalar gamma  = cp / cv;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup
    // -- Interp in
    const CeedScalar rho        =   q[0][i];
    const CeedScalar u[3]       =  {q[1][i] / rho,
                                    q[2][i] / rho,
                                    q[3][i] / rho
                                   };
    const CeedScalar E          =   q[4][i];
    // -- Grad in
    const CeedScalar drho[3]    =  {dq[0][0][i],
                                    dq[1][0][i],
                                    dq[2][0][i]
                                   };
    const CeedScalar dU[3][3]   = {{dq[0][1][i],
                                    dq[1][1][i],
                                    dq[2][1][i]},
                                   {dq[0][2][i],
                                    dq[1][2][i],
                                    dq[2][2][i]},
                                   {dq[0][3][i],
                                    dq[1][3][i],
                                    dq[2][3][i]}
                                  };
    const CeedScalar dE[3]      =  {dq[0][4][i],
                                    dq[1][4][i],
                                    dq[2][4][i]
                                   };
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
    // -- Grad-to-Grad q_data
    // dU/dx
    CeedScalar du[3][3] = {{0}};
    CeedScalar drhodx[3] = {0};
    CeedScalar dEdx[3] = {0};
    CeedScalar dUdx[3][3] = {{0}};
    CeedScalar dXdxdXdxT[3][3] = {{0}};
    for (int j=0; j<3; j++) {
      for (int k=0; k<3; k++) {
        du[j][k] = (dU[j][k] - drho[k]*u[j]) / rho;
        drhodx[j] += drho[k] * dXdx[k][j];
        dEdx[j] += dE[k] * dXdx[k][j];
        for (int l=0; l<3; l++) {
          dUdx[j][k] += dU[j][l] * dXdx[l][k];
          dXdxdXdxT[j][k] += dXdx[j][l]*dXdx[k][l];  //dXdx_j,k * dXdx_k,j
        }
      }
    }
    CeedScalar dudx[3][3] = {{0}};
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        for (int l=0; l<3; l++)
          dudx[j][k] += du[j][l] * dXdx[l][k];
    // -- grad_T
    const CeedScalar grad_T[3]  = {(dEdx[0]/rho - E*drhodx[0]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][0] + u[1]*dudx[1][0] + u[2]*dudx[2][0]))/cv,
                                   (dEdx[1]/rho - E*drhodx[1]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][1] + u[1]*dudx[1][1] + u[2]*dudx[2][1]))/cv,
                                   (dEdx[2]/rho - E*drhodx[2]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][2] + u[1]*dudx[1][2] + u[2]*dudx[2][2]) - g)/cv
                                  };

    // -- Fuvisc
    // ---- Symmetric 3x3 matrix
    const CeedScalar Fu[6]     =  {mu*(dudx[0][0] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[1][1] + dudx[2][2])),
                                   mu*(dudx[0][1] + dudx[1][0]), /* *NOPAD* */
                                   mu*(dudx[0][2] + dudx[2][0]), /* *NOPAD* */
                                   mu*(dudx[1][1] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[0][0] + dudx[2][2])),
                                   mu*(dudx[1][2] + dudx[2][1]), /* *NOPAD* */
                                   mu*(dudx[2][2] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[0][0] + dudx[1][1]))
                                  };
    // -- Fevisc
    const CeedScalar Fe[3]     =  {u[0]*Fu[0] + u[1]*Fu[1] + u[2]*Fu[2] + /* *NOPAD* */
                                   k*grad_T[0], /* *NOPAD* */
                                   u[0]*Fu[1] + u[1]*Fu[3] + u[2]*Fu[4] + /* *NOPAD* */
                                   k*grad_T[1], /* *NOPAD* */
                                   u[0]*Fu[2] + u[1]*Fu[4] + u[2]*Fu[5] + /* *NOPAD* */
                                   k*grad_T[2] /* *NOPAD* */
                                  };
    // Pressure
    const CeedScalar
    E_kinetic   = 0.5 * rho * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]),
    E_potential = rho*g*x[2][i],
    E_internal  = E - E_kinetic - E_potential,
    P           = E_internal * (gamma - 1.); // P = pressure

    // jacob_F_conv[3][5][5] = dF(convective)/dq at each direction
    CeedScalar jacob_F_conv[3][5][5] = {{{0.}}};
    computeFluxJacobian_NS(jacob_F_conv, rho, u, E, gamma, g, x[2][i]);

    // jacob_F_conv_T = jacob_F_conv^T
    CeedScalar jacob_F_conv_T[3][5][5];
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          jacob_F_conv_T[j][k][l] = jacob_F_conv[j][l][k];

    // dqdx collects drhodx, dUdx and dEdx in one vector
    CeedScalar dqdx[5][3];
    for (int j=0; j<3; j++) {
      dqdx[0][j] = drhodx[j];
      dqdx[4][j] = dEdx[j];
      for (int k=0; k<3; k++)
        dqdx[k+1][j] = dUdx[k][j];
    }

    // strong_conv = dF/dq * dq/dx    (Strong convection)
    CeedScalar strong_conv[5] = {0};
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          strong_conv[k] += jacob_F_conv[j][k][l] * dqdx[l][j];

    // Body force
    const CeedScalar body_force[5] = {0, 0, 0, -rho*g, 0};

    // The Physics
    // Zero dv so all future terms can safely sum into it
    for (int j=0; j<5; j++)
      for (int k=0; k<3; k++)
        dv[k][j][i] = 0;

    // -- Density
    // ---- u rho
    for (int j=0; j<3; j++)
      dv[j][0][i]  += wdetJ*(rho*u[0]*dXdx[j][0] + rho*u[1]*dXdx[j][1] +
                             rho*u[2]*dXdx[j][2]);
    // -- Momentum
    // ---- rho (u x u) + P I3
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        dv[k][j+1][i]  += wdetJ*((rho*u[j]*u[0] + (j==0?P:0))*dXdx[k][0] +
                                 (rho*u[j]*u[1] + (j==1?P:0))*dXdx[k][1] +
                                 (rho*u[j]*u[2] + (j==2?P:0))*dXdx[k][2]);
    // ---- Fuvisc
    const CeedInt Fuviscidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}}; // symmetric matrix indices
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        dv[k][j+1][i] -= wdetJ*(Fu[Fuviscidx[j][0]]*dXdx[k][0] +
                                Fu[Fuviscidx[j][1]]*dXdx[k][1] +
                                Fu[Fuviscidx[j][2]]*dXdx[k][2]);
    // -- Total Energy Density
    // ---- (E + P) u
    for (int j=0; j<3; j++)
      dv[j][4][i]  += wdetJ * (E + P) * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] +
                                         u[2]*dXdx[j][2]);
    // ---- Fevisc
    for (int j=0; j<3; j++)
      dv[j][4][i] -= wdetJ * (Fe[0]*dXdx[j][0] + Fe[1]*dXdx[j][1] +
                              Fe[2]*dXdx[j][2]);
    // Body Force
    for (int j=0; j<5; j++)
      v[j][i] = wdetJ * body_force[j];

    // Stabilization
    // -- Tau elements
    const CeedScalar sound_speed = sqrt(gamma * P / rho);
    CeedScalar Tau_x[3] = {0.};
    Tau_spatial(Tau_x, dXdx, u, sound_speed, c_tau);

    // -- Stabilization method: none or SU
    CeedScalar stab[5][3];
    switch (context->stabilization) {
    case STAB_NONE:        // Galerkin
      break;
    case STAB_SU:        // SU
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] = jacob_F_conv_T[j][k][l] * Tau_x[j] * strong_conv[l];

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
  const CeedScalar lambda = context->lambda;
  const CeedScalar mu     = context->mu;
  const CeedScalar k      = context->k;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar g      = context->g;
  const CeedScalar c_tau  = context->c_tau;
  const CeedScalar gamma  = cp / cv;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho        =   q[0][i];
    const CeedScalar u[3]       =  {q[1][i] / rho,
                                    q[2][i] / rho,
                                    q[3][i] / rho
                                   };
    const CeedScalar E          =   q[4][i];
    // -- Grad in
    const CeedScalar drho[3]    =  {dq[0][0][i],
                                    dq[1][0][i],
                                    dq[2][0][i]
                                   };
    // *INDENT-OFF*
    const CeedScalar dU[3][3]   = {{dq[0][1][i],
                                    dq[1][1][i],
                                    dq[2][1][i]},
                                   {dq[0][2][i],
                                    dq[1][2][i],
                                    dq[2][2][i]},
                                   {dq[0][3][i],
                                    dq[1][3][i],
                                    dq[2][3][i]}
                                  };
    // *INDENT-ON*
    const CeedScalar dE[3]      =  {dq[0][4][i],
                                    dq[1][4][i],
                                    dq[2][4][i]
                                   };
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
    // -- Grad-to-Grad q_data
    // dU/dx
    CeedScalar du[3][3] = {{0}};
    CeedScalar drhodx[3] = {0};
    CeedScalar dEdx[3] = {0};
    CeedScalar dUdx[3][3] = {{0}};
    CeedScalar dXdxdXdxT[3][3] = {{0}};
    for (int j=0; j<3; j++) {
      for (int k=0; k<3; k++) {
        du[j][k] = (dU[j][k] - drho[k]*u[j]) / rho;
        drhodx[j] += drho[k] * dXdx[k][j];
        dEdx[j] += dE[k] * dXdx[k][j];
        for (int l=0; l<3; l++) {
          dUdx[j][k] += dU[j][l] * dXdx[l][k];
          dXdxdXdxT[j][k] += dXdx[j][l]*dXdx[k][l];  //dXdx_j,k * dXdx_k,j
        }
      }
    }
    CeedScalar dudx[3][3] = {{0}};
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        for (int l=0; l<3; l++)
          dudx[j][k] += du[j][l] * dXdx[l][k];
    // -- grad_T
    const CeedScalar grad_T[3]  = {(dEdx[0]/rho - E*drhodx[0]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][0] + u[1]*dudx[1][0] + u[2]*dudx[2][0]))/cv,
                                   (dEdx[1]/rho - E*drhodx[1]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][1] + u[1]*dudx[1][1] + u[2]*dudx[2][1]))/cv,
                                   (dEdx[2]/rho - E*drhodx[2]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][2] + u[1]*dudx[1][2] + u[2]*dudx[2][2]) - g)/cv
                                  };
    // -- Fuvisc
    // ---- Symmetric 3x3 matrix
    const CeedScalar Fu[6]     =  {mu*(dudx[0][0] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[1][1] + dudx[2][2])),
                                   mu*(dudx[0][1] + dudx[1][0]), /* *NOPAD* */
                                   mu*(dudx[0][2] + dudx[2][0]), /* *NOPAD* */
                                   mu*(dudx[1][1] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[0][0] + dudx[2][2])),
                                   mu*(dudx[1][2] + dudx[2][1]), /* *NOPAD* */
                                   mu*(dudx[2][2] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[0][0] + dudx[1][1]))
                                  };
    // -- Fevisc
    const CeedScalar Fe[3]     =  {u[0]*Fu[0] + u[1]*Fu[1] + u[2]*Fu[2] + /* *NOPAD* */
                                   k*grad_T[0], /* *NOPAD* */
                                   u[0]*Fu[1] + u[1]*Fu[3] + u[2]*Fu[4] + /* *NOPAD* */
                                   k*grad_T[1], /* *NOPAD* */
                                   u[0]*Fu[2] + u[1]*Fu[4] + u[2]*Fu[5] + /* *NOPAD* */
                                   k*grad_T[2] /* *NOPAD* */
                                  };
    // Pressure
    const CeedScalar
    E_kinetic   = 0.5 * rho * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]),
    E_potential = rho*g*x[2][i],
    E_internal  = E - E_kinetic - E_potential,
    P           = E_internal * (gamma - 1.); // P = pressure

    // jacob_F_conv[3][5][5] = dF(convective)/dq at each direction
    CeedScalar jacob_F_conv[3][5][5] = {{{0.}}};
    computeFluxJacobian_NS(jacob_F_conv, rho, u, E, gamma, g, x[2][i]);

    // jacob_F_conv_T = jacob_F_conv^T
    CeedScalar jacob_F_conv_T[3][5][5];
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          jacob_F_conv_T[j][k][l] = jacob_F_conv[j][l][k];
    // dqdx collects drhodx, dUdx and dEdx in one vector
    CeedScalar dqdx[5][3];
    for (int j=0; j<3; j++) {
      dqdx[0][j] = drhodx[j];
      dqdx[4][j] = dEdx[j];
      for (int k=0; k<3; k++)
        dqdx[k+1][j] = dUdx[k][j];
    }
    // strong_conv = dF/dq * dq/dx    (Strong convection)
    CeedScalar strong_conv[5] = {0};
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          strong_conv[k] += jacob_F_conv[j][k][l] * dqdx[l][j];

    // Body force
    const CeedScalar body_force[5] = {0, 0, 0, -rho*g, 0};

    // Strong residual
    CeedScalar strong_res[5];
    for (int j=0; j<5; j++)
      strong_res[j] = q_dot[j][i] + strong_conv[j] - body_force[j];

    // The Physics
    //-----mass matrix
    for (int j=0; j<5; j++)
      v[j][i] = wdetJ*q_dot[j][i];

    // Zero dv so all future terms can safely sum into it
    for (int j=0; j<5; j++)
      for (int k=0; k<3; k++)
        dv[k][j][i] = 0;

    // -- Density
    // ---- u rho
    for (int j=0; j<3; j++)
      dv[j][0][i]  -= wdetJ*(rho*u[0]*dXdx[j][0] + rho*u[1]*dXdx[j][1] +
                             rho*u[2]*dXdx[j][2]);
    // -- Momentum
    // ---- rho (u x u) + P I3
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        dv[k][j+1][i]  -= wdetJ*((rho*u[j]*u[0] + (j==0?P:0))*dXdx[k][0] +
                                 (rho*u[j]*u[1] + (j==1?P:0))*dXdx[k][1] +
                                 (rho*u[j]*u[2] + (j==2?P:0))*dXdx[k][2]);
    // ---- Fuvisc
    const CeedInt Fuviscidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}}; // symmetric matrix indices
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        dv[k][j+1][i] += wdetJ*(Fu[Fuviscidx[j][0]]*dXdx[k][0] +
                                Fu[Fuviscidx[j][1]]*dXdx[k][1] +
                                Fu[Fuviscidx[j][2]]*dXdx[k][2]);
    // -- Total Energy Density
    // ---- (E + P) u
    for (int j=0; j<3; j++)
      dv[j][4][i]  -= wdetJ * (E + P) * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] +
                                         u[2]*dXdx[j][2]);
    // ---- Fevisc
    for (int j=0; j<3; j++)
      dv[j][4][i] += wdetJ * (Fe[0]*dXdx[j][0] + Fe[1]*dXdx[j][1] +
                              Fe[2]*dXdx[j][2]);
    // Body Force
    for (int j=0; j<5; j++)
      v[j][i] -= wdetJ*body_force[j];

    // Stabilization
    // -- Tau elements
    const CeedScalar sound_speed = sqrt(gamma * P / rho);
    CeedScalar Tau_x[3] = {0.};
    Tau_spatial(Tau_x, dXdx, u, sound_speed, c_tau);

    // -- Stabilization method: none, SU, or SUPG
    CeedScalar stab[5][3];
    switch (context->stabilization) {
    case STAB_NONE:        // Galerkin
      break;
    case STAB_SU:        // SU
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] = jacob_F_conv_T[j][k][l] * Tau_x[j] * strong_conv[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] += wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    case STAB_SUPG:        // SUPG
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] = jacob_F_conv_T[j][k][l] * Tau_x[j] * strong_res[l];

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
