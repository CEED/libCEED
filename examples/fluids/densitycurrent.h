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

#ifndef __CUDACC__
#  include <math.h>
#endif

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
  CeedScalar Rd;
  CeedScalar g;
  CeedScalar rc;
  CeedScalar lx;
  CeedScalar ly;
  CeedScalar lz;
  CeedScalar center[3];
  CeedScalar dc_axis[3];
  CeedScalar wind[3];
  CeedScalar time;
  int wind_type;
};
#endif

#ifndef advection_context_struct
#define advection_context_struct
typedef struct DCContext_ *DCContext;
struct DCContext_ {
  int stabilization; // See StabilizationType: 0=none, 1=SU, 2=SUPG
};
#endif

// *****************************************************************************
// These function sets the the initial conditions and boundary conditions
//
// These initial conditions are given in terms of potential temperature and
//   Exner pressure and then converted to density and total energy.
//   Initial momentum density is zero.
//
// Initial Conditions:
//   Potential Temperature:
//     theta = thetabar + deltatheta
//       thetabar   = theta0 exp( N**2 z / g )
//       deltatheta = r <= rc : thetaC(1 + cos(pi r/rc)) / 2
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
static inline int Exact_DC(CeedInt dim, CeedScalar time, const CeedScalar X[],
                           CeedInt Nf, CeedScalar q[], void *ctx) {
  // Context
  const SetupContext context = (SetupContext)ctx;

  const CeedScalar theta0 = context->theta0;
  const CeedScalar thetaC = context->thetaC;
  const CeedScalar P0     = context->P0;
  const CeedScalar N      = context->N;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar Rd     = context->Rd;
  const CeedScalar g      = context->g;
  const CeedScalar rc     = context->rc;
  const CeedScalar *center = context->center;
  const CeedScalar *dc_axis = context->dc_axis;

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
  const CeedScalar deltatheta = r <= rc ? thetaC*(1. + cos(M_PI*r/rc))/2. : 0.;
  const CeedScalar theta = theta0*exp(N*N*z/g) + deltatheta;
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
    CeedScalar q[5];

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
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  const CeedScalar *context = (const CeedScalar *)ctx;
  const CeedScalar lambda = context[0];
  const CeedScalar mu     = context[1];
  const CeedScalar k      = context[2];
  const CeedScalar cv     = context[3];
  const CeedScalar cp     = context[4];
  const CeedScalar g      = context[5];
  const CeedScalar Rd     = context[6];
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
    // -- Interp-to-Interp qdata
    const CeedScalar wdetJ      =   qdata[0][i];
    // -- Interp-to-Grad qdata
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {{qdata[1][i],
                                    qdata[2][i],
                                    qdata[3][i]},
                                   {qdata[4][i],
                                    qdata[5][i],
                                    qdata[6][i]},
                                   {qdata[7][i],
                                    qdata[8][i],
                                    qdata[9][i]}
                                  };
    // *INDENT-ON*
    // -- Grad-to-Grad qdata
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
    // -- gradT
    const CeedScalar gradT[3]  = {(dEdx[0]/rho - E*drhodx[0]/(rho*rho) - /* *NOPAD* */
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
                                   k*gradT[0], /* *NOPAD* */
                                   u[0]*Fu[1] + u[1]*Fu[3] + u[2]*Fu[4] + /* *NOPAD* */
                                   k*gradT[1], /* *NOPAD* */
                                   u[0]*Fu[2] + u[1]*Fu[4] + u[2]*Fu[5] + /* *NOPAD* */
                                   k*gradT[2] /* *NOPAD* */
                                  };
    // ke = kinetic energy
    const CeedScalar ke = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) / 2.;
    // P = pressure
    const CeedScalar P  = (E - ke * rho - rho*g*x[2][i]) * (gamma - 1.);
    // dFconvdq[3][5][5] = dF(convective)/dq at each direction
    CeedScalar dFconvdq[3][5][5] = {{{0}}};
    for (int j=0; j<3; j++) {
      dFconvdq[j][4][0] = u[j] * (2*Rd*ke/cv - E*gamma);
      dFconvdq[j][4][4] = u[j] * gamma;
      for (int k=0; k<3; k++) {
        dFconvdq[j][k+1][0] = -u[j]*u[k] + (j==k?(ke*Rd/cv):0);
        dFconvdq[j][k+1][4] = (j==k?(Rd/cv):0);
        dFconvdq[j][k+1][k+1] = (j!=k?u[j]:0);
        dFconvdq[j][0][k+1] = (j==k?1:0);
        dFconvdq[j][j+1][k+1] = u[k] * ((j==k?2:0) - Rd/cv);
        dFconvdq[j][4][k+1] = -(Rd/cv)*u[j]*u[k] + (j==k?(E*gamma - Rd*ke/cv):0);
      }
    }
    dFconvdq[0][2][1] = u[1];
    dFconvdq[0][3][1] = u[2];
    dFconvdq[1][1][2] = u[0];
    dFconvdq[1][3][2] = u[2];
    dFconvdq[2][1][3] = u[0];
    dFconvdq[2][2][3] = u[1];
    // dFconvdqT = dFconvdq^T
    CeedScalar dFconvdqT[3][5][5];
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          dFconvdqT[j][k][l] = dFconvdq[j][l][k];
    // dqdx collects drhodx, dUdx and dEdx in one vector
    CeedScalar dqdx[5][3];
    for (int j=0; j<3; j++) {
      dqdx[0][j] = drhodx[j];
      dqdx[4][j] = dEdx[j];
      for (int k=0; k<3; k++)
        dqdx[k+1][j] = dUdx[k][j];
    }
    // StrongConv = dF/dq * dq/dx    (Strong convection)
    CeedScalar StrongConv[5] = {0};
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          StrongConv[k] += dFconvdq[j][k][l] * dqdx[l][j];
    // Body force
    const CeedScalar BodyForce[5] = {0, 0, 0, -rho*g, 0};

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
      v[j][i] = wdetJ * BodyForce[j];

    //Stabilization
    CeedScalar uX[3];
    for (int j=0; j<3;
         j++) uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
    const CeedScalar uiujgij = uX[0]*uX[0] + uX[1]*uX[1] + uX[2]*uX[2];
    const CeedScalar Cc   = 1.;
    const CeedScalar Ce   = 1.;
    const CeedScalar f1   = rho * sqrt(uiujgij);
    const CeedScalar TauC = (Cc * f1) /
                            (8 * (dXdxdXdxT[0][0] + dXdxdXdxT[1][1] + dXdxdXdxT[2][2]));
    const CeedScalar TauM = 1. / (f1>1. ? f1 : 1.);
    const CeedScalar TauE = TauM / (Ce * cv);
    // *INDENT-ON*
    const CeedScalar Tau[5] = {TauC, TauM, TauM, TauM, TauE};
    CeedScalar stab[5][3];
    DCContext context = (DCContext)ctx;
    switch (context->stabilization) {
    case 0:        // Galerkin
      break;
    case 1:        // SU
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] = dFconvdqT[j][k][l] * Tau[l] * StrongConv[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] -= stab[j][0] * dXdx[k][0] +
                         stab[j][1] * dXdx[k][1] +
                         stab[j][2] * dXdx[k][2];
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
//  SUPG = Galerkin + grad(v) . ( Ai^T * Tau * (qdot + Aj q,j - body force) )
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
                   (*qdot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3],
                   (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  // Context
  const CeedScalar *context = (const CeedScalar *)ctx;
  const CeedScalar lambda = context[0];
  const CeedScalar mu     = context[1];
  const CeedScalar k      = context[2];
  const CeedScalar cv     = context[3];
  const CeedScalar cp     = context[4];
  const CeedScalar g      = context[5];
  const CeedScalar Rd     = context[6];
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
    // -- Interp-to-Interp qdata
    const CeedScalar wdetJ      =   qdata[0][i];
    // -- Interp-to-Grad qdata
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {{qdata[1][i],
                                    qdata[2][i],
                                    qdata[3][i]},
                                   {qdata[4][i],
                                    qdata[5][i],
                                    qdata[6][i]},
                                   {qdata[7][i],
                                    qdata[8][i],
                                    qdata[9][i]}
                                  };
    // *INDENT-ON*
    // -- Grad-to-Grad qdata
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
    // -- gradT
    const CeedScalar gradT[3]  = {(dEdx[0]/rho - E*drhodx[0]/(rho*rho) - /* *NOPAD* */
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
                                   k*gradT[0], /* *NOPAD* */
                                   u[0]*Fu[1] + u[1]*Fu[3] + u[2]*Fu[4] + /* *NOPAD* */
                                   k*gradT[1], /* *NOPAD* */
                                   u[0]*Fu[2] + u[1]*Fu[4] + u[2]*Fu[5] + /* *NOPAD* */
                                   k*gradT[2] /* *NOPAD* */
                                  };
    // ke = kinetic energy
    const CeedScalar ke = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) / 2.;
    // P = pressure
    const CeedScalar P  = (E - ke * rho - rho*g*x[2][i]) * (gamma - 1.);
    // dFconvdq[3][5][5] = dF(convective)/dq at each direction
    CeedScalar dFconvdq[3][5][5] = {{{0}}};
    for (int j=0; j<3; j++) {
      dFconvdq[j][4][0] = u[j] * (2*Rd*ke/cv - E*gamma);
      dFconvdq[j][4][4] = u[j] * gamma;
      for (int k=0; k<3; k++) {
        dFconvdq[j][k+1][0] = -u[j]*u[k] + (j==k?(ke*Rd/cv):0);
        dFconvdq[j][k+1][4] = (j==k?(Rd/cv):0);
        dFconvdq[j][k+1][k+1] = (j!=k?u[j]:0);
        dFconvdq[j][0][k+1] = (j==k?1:0);
        dFconvdq[j][j+1][k+1] = u[k] * ((j==k?2:0) - Rd/cv);
        dFconvdq[j][4][k+1] = -(Rd/cv)*u[j]*u[k] + (j==k?(E*gamma - Rd*ke/cv):0);
      }
    }
    dFconvdq[0][2][1] = u[1];
    dFconvdq[0][3][1] = u[2];
    dFconvdq[1][1][2] = u[0];
    dFconvdq[1][3][2] = u[2];
    dFconvdq[2][1][3] = u[0];
    dFconvdq[2][2][3] = u[1];
    // dFconvdqT = dFconvdq^T
    CeedScalar dFconvdqT[3][5][5];
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          dFconvdqT[j][k][l] = dFconvdq[j][l][k];
    // dqdx collects drhodx, dUdx and dEdx in one vector
    CeedScalar dqdx[5][3];
    for (int j=0; j<3; j++) {
      dqdx[0][j] = drhodx[j];
      dqdx[4][j] = dEdx[j];
      for (int k=0; k<3; k++)
        dqdx[k+1][j] = dUdx[k][j];
    }
    // StrongConv = dF/dq * dq/dx    (Strong convection)
    CeedScalar StrongConv[5] = {0};
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          StrongConv[k] += dFconvdq[j][k][l] * dqdx[l][j];
    // Body force
    const CeedScalar BodyForce[5] = {0, 0, 0, -rho*g, 0};
    // Strong residual
    CeedScalar StrongResid[5];
    for (int j=0; j<5; j++)
      StrongResid[j] = qdot[j][i] + StrongConv[j] - BodyForce[j];

    // The Physics
    //-----mass matrix
    for (int j=0; j<5; j++)
      v[j][i] = wdetJ*qdot[j][i];

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
      v[j][i] -= wdetJ*BodyForce[j];

    //Stabilization
    CeedScalar uX[3];
    for (int j=0; j<3;
         j++) uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
    const CeedScalar uiujgij = uX[0]*uX[0] + uX[1]*uX[1] + uX[2]*uX[2];
    const CeedScalar Cc   = 1.;
    const CeedScalar Ce   = 1.;
    const CeedScalar f1   = rho * sqrt(uiujgij);
    const CeedScalar TauC = (Cc * f1) /
                            (8 * (dXdxdXdxT[0][0] + dXdxdXdxT[1][1] + dXdxdXdxT[2][2]));
    const CeedScalar TauM = 1. / (f1>1. ? f1 : 1.);
    const CeedScalar TauE = TauM / (Ce * cv);
    const CeedScalar Tau[5] = {TauC, TauM, TauM, TauM, TauE};
    CeedScalar stab[5][3];
    DCContext context = (DCContext)ctx;
    switch (context->stabilization) {
    case 0:        // Galerkin
      break;
    case 1:        // SU
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] = dFconvdqT[j][k][l] * Tau[l] * StrongConv[l];

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
            stab[k][j] = dFconvdqT[j][k][l] * Tau[l] * StrongResid[l];

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
