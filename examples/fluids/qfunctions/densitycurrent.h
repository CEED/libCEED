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
/// Density current initial condition and operator for Navier-Stokes example using PETSc

// Model from:
//   Semi-Implicit Formulations of the Navier-Stokes Equations: Application to
//   Nonhydrostatic Atmospheric Modeling, Giraldo, Restelli, and Lauter (2010).

#ifndef densitycurrent_h
#define densitycurrent_h

#include <math.h>

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#ifndef setup_context_struct
#define setup_context_struct
typedef struct SetupContext_ *SetupContext;
struct SetupContext_ {
  CeedScalar theta0;
  CeedScalar thetaC;
  CeedScalar P0;
  CeedScalar N;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar g;
  CeedScalar rc;
  CeedScalar lx;
  CeedScalar ly;
  CeedScalar lz;
  CeedScalar center[3];
  CeedScalar dc_axis[3];
  CeedScalar wind[3];
  CeedScalar time;
  int wind_type;              // See WindType: 0=ROTATION, 1=TRANSLATION
  int bubble_type;            // See BubbleType: 0=SPHERE, 1=CYLINDER
  int bubble_continuity_type; // See BubbleContinuityType: 0=SMOOTH, 1=BACK_SHARP 2=THICK
};
#endif

#ifndef dc_context_struct
#define dc_context_struct
typedef struct DCContext_ *DCContext;
struct DCContext_ {
  CeedScalar lambda;
  CeedScalar mu;
  CeedScalar k;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar g;
  int stabilization; // See StabilizationType: 0=none, 1=SU, 2=SUPG
};
#endif

// *****************************************************************************
// This function sets the initial conditions and the boundary conditions
//
// These initial conditions are given in terms of potential temperature and
//   Exner pressure and then converted to density and total energy.
//   Initial momentum density is zero.
//
// Initial Conditions:
//   Potential Temperature:
//     theta = thetabar + delta_theta
//       thetabar   = theta0 exp( N**2 z / g )
//       delta_theta = r <= rc : thetaC(1 + cos(pi r/rc)) / 2
//                     r > rc : 0
//         r        = sqrt( (x - xc)**2 + (y - yc)**2 + (z - zc)**2 )
//         with (xc,yc,zc) center of domain, rc characteristic radius of thermal bubble
//   Exner Pressure:
//     Pi = Pibar + deltaPi
//       Pibar      = 1. + g**2 (exp( - N**2 z / g ) - 1) / (cp theta0 N**2)
//       deltaPi    = 0 (hydrostatic balance)
//   Velocity/Momentum Density:
//     Ui = ui = 0
//
// Conversion to Conserved Variables:
//   rho = P0 Pi**(cv/Rd) / (Rd theta)
//   E   = rho (cv T + (u u)/2 + g z)
//
//  Boundary Conditions:
//    Mass Density:
//      0.0 flux
//    Momentum Density:
//      0.0
//    Energy Density:
//      0.0 flux
//
// Constants:
//   theta0          ,  Potential temperature constant
//   thetaC          ,  Potential temperature perturbation
//   P0              ,  Pressure at the surface
//   N               ,  Brunt-Vaisala frequency
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   Rd     = cp - cv,  Specific heat difference
//   g               ,  Gravity
//   rc              ,  Characteristic radius of thermal bubble
//   center          ,  Location of bubble center
//   dc_axis         ,  Axis of density current cylindrical anomaly, or {0,0,0} for spherically symmetric
// *****************************************************************************

// *****************************************************************************
// This helper function provides support for the exact, time-dependent solution
//   (currently not implemented) and IC formulation for density current
// *****************************************************************************
CEED_QFUNCTION_HELPER int Exact_DC(CeedInt dim, CeedScalar time,
                                   const CeedScalar X[], CeedInt Nf, CeedScalar q[],
                                   void *ctx) {
  // Context
  const SetupContext context = (SetupContext)ctx;
  const CeedScalar theta0   = context->theta0;
  const CeedScalar thetaC   = context->thetaC;
  const CeedScalar P0       = context->P0;
  const CeedScalar N        = context->N;
  const CeedScalar cv       = context->cv;
  const CeedScalar cp       = context->cp;
  const CeedScalar g        = context->g;
  const CeedScalar rc       = context->rc;
  const CeedScalar *center  = context->center;
  const CeedScalar *dc_axis = context->dc_axis;
  const CeedScalar Rd       = cp - cv;

  // Setup
  // -- Coordinates
  const CeedScalar x = X[0];
  const CeedScalar y = X[1];
  const CeedScalar z = X[2];

  // -- Potential temperature, density current
  CeedScalar rr[3] = {x - center[0], y - center[1], z - center[2]};
  // (I - q q^T) r: distance from dc_axis (or from center if dc_axis is the zero vector)
  for (CeedInt i=0; i<3; i++)
    rr[i] -= dc_axis[i] *
             (dc_axis[0]*rr[0] + dc_axis[1]*rr[1] + dc_axis[2]*rr[2]);
  const CeedScalar r = sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]);
  const CeedScalar delta_theta = r <= rc ? thetaC*(1. + cos(M_PI*r/rc))/2. : 0.;
  const CeedScalar theta = theta0*exp(N*N*z/g) + delta_theta;

  // -- Exner pressure, hydrostatic balance
  const CeedScalar Pi = 1. + g*g*(exp(-N*N*z/g) - 1.) / (cp*theta0*N*N);
  // -- Density

  const CeedScalar rho = P0 * pow(Pi, cv/Rd) / (Rd*theta);

  // Initial Conditions
  q[0] = rho;
  q[1] = 0.0;
  q[2] = 0.0;
  q[3] = 0.0;
  q[4] = rho * (cv*theta*Pi + g*z);

  return 0;
}

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
// This QFunction sets the initial conditions for density current
// *****************************************************************************
CEED_QFUNCTION(ICsDC)(void *ctx, CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar q[5] = {0.};

    Exact_DC(3, 0., x, 5, q, ctx);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop

  // Return
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
CEED_QFUNCTION(DC)(void *ctx, CeedInt Q,
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
  DCContext context = (DCContext)ctx;
  const CeedScalar lambda = context->lambda;
  const CeedScalar mu     = context->mu;
  const CeedScalar k      = context->k;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar g      = context->g;
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

    //Stabilization
    CeedScalar uX[3];
    for (int j=0; j<3; j++)
      uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
    const CeedScalar uiujgij = uX[0]*uX[0] + uX[1]*uX[1] + uX[2]*uX[2];
    const CeedScalar Cc      = 1.;
    const CeedScalar Ce      = 1.;
    const CeedScalar f1      = rho * sqrt(uiujgij);
    const CeedScalar TauC   = (Cc * f1) /
                              (8 * (dXdxdXdxT[0][0] + dXdxdXdxT[1][1] + dXdxdXdxT[2][2]));
    const CeedScalar TauM   = 1. / (f1>1. ? f1 : 1.);
    const CeedScalar TauE   = TauM / (Ce * cv);
    const CeedScalar Tau[5]  = {TauC, TauM, TauM, TauM, TauE};
    CeedScalar stab[5][3];
    switch (context->stabilization) {
    case 0:        // Galerkin
      break;
    case 1:        // SU
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] = jacob_F_conv_T[j][k][l] * Tau[l] * strong_conv[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] -= wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    case 2:        // SUPG is not implemented for explicit scheme
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
CEED_QFUNCTION(IFunction_DC)(void *ctx, CeedInt Q,
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
  DCContext context = (DCContext)ctx;
  const CeedScalar lambda = context->lambda;
  const CeedScalar mu     = context->mu;
  const CeedScalar k      = context->k;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar g      = context->g;
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

    //Stabilization
    CeedScalar uX[3];
    for (int j=0; j<3; j++)
      uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
    const CeedScalar uiujgij = uX[0]*uX[0] + uX[1]*uX[1] + uX[2]*uX[2];
    const CeedScalar Cc      = 1.;
    const CeedScalar Ce      = 1.;
    const CeedScalar f1      = rho * sqrt(uiujgij);
    const CeedScalar TauC   = (Cc * f1) /
                              (8 * (dXdxdXdxT[0][0] + dXdxdXdxT[1][1] + dXdxdXdxT[2][2]));
    const CeedScalar TauM   = 1. / (f1>1. ? f1 : 1.);
    const CeedScalar TauE   = TauM / (Ce * cv);
    const CeedScalar Tau[5]  = {TauC, TauM, TauM, TauM, TauE};

    CeedScalar stab[5][3];
    switch (context->stabilization) {
    case 0:        // Galerkin
      break;
    case 1:        // SU
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] = jacob_F_conv_T[j][k][l] * Tau[l] * strong_conv[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] += wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    case 2:        // SUPG
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] = jacob_F_conv_T[j][k][l] * Tau[l] * strong_res[l];

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

#endif // densitycurrent_h
