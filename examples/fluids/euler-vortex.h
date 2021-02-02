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
/// Euler traveling vortex initial condition and operator for Navier-Stokes
/// example using PETSc

// Model from:
//   On the Order of Accuracy and Numerical Performance of Two Classes of
//   Finite Volume WENO Schemes, Zhang, Zhang, and Shu (2011).

#ifndef eulervortex_h
#define eulervortex_h

#ifndef __CUDACC__
#  include <math.h>
#endif

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#ifndef euler_context_struct
#define euler_context_struct
typedef struct EulerContext_ *EulerContext;
struct EulerContext_ {
  CeedScalar time;
  CeedScalar center[3];
  CeedScalar currentTime;
  CeedScalar vortex_strength;
  CeedScalar etv_mean_velocity[3];
  int stabilization;
  int euler_test;
  bool implicit;
};
#endif

// *****************************************************************************
// This function sets the initial conditions
//
//   Temperature:
//     T   = 1 - (gamma - 1) vortex_strength**2 exp(1 - r**2) / (8 gamma pi**2)
//   Density:
//     rho = (T/S_vortex)^(1 / (gamma - 1))
//   Pressure:
//     P   = rho * T
//   Velocity:
//     ui  = 1 + vortex_strength exp((1 - r**2)/2.) [yc - y, x - xc] / (2 pi)
//     r   = sqrt( (x - xc)**2 + (y - yc)**2 )
//   Velocity/Momentum Density:
//     Ui  = rho ui
//   Total Energy:
//     E   = P / (gamma - 1) + rho (u u)/2
//
// Constants:
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   vortex_strength ,  Strength of vortex
//   center          ,  Location of bubble center
//   gamma  = cp / cv,  Specific heat ratio
//
// *****************************************************************************

// *****************************************************************************
// This helper function provides support for the exact, time-dependent solution
//   (currently not implemented) and IC formulation for Euler traveling vortex
// *****************************************************************************
static inline int Exact_Euler(CeedInt dim, CeedScalar time,
                              const CeedScalar X[],
                              CeedInt Nf, CeedScalar q[], void *ctx) {
  // Context
  const EulerContext context = (EulerContext)ctx;
  const CeedScalar vortex_strength = context->vortex_strength;
  const CeedScalar *center = context->center; // Center of the domain
  const CeedScalar *etv_mean_velocity = context->etv_mean_velocity;

  // Setup
  const CeedScalar gamma = 1.4;
  const CeedScalar cv = 2.5;
  const CeedScalar R = 1.;
  const CeedScalar x = X[0], y = X[1], z = X[2]; // Coordinates
  // Vortex center
  const CeedScalar xc = center[0] + etv_mean_velocity[0] * time;
  const CeedScalar yc = center[1] + etv_mean_velocity[1] * time;

  const CeedScalar x0 = x - xc;
  const CeedScalar y0 = y - yc;
  const CeedScalar r = sqrt( x0*x0 + y0*y0 );
  const CeedScalar C = vortex_strength * exp((1. - r*r)/2.) / (2. * M_PI);
  const CeedScalar delta_T = - (gamma - 1) * vortex_strength * vortex_strength *
                             exp(1 - r*r) / (8 * gamma * M_PI * M_PI);
  const CeedScalar S_vortex = 1; // no perturbation in the entropy P / rho^gamma
  const CeedScalar S_bubble = (gamma - 1.) * vortex_strength * vortex_strength /
                              (8.*gamma*M_PI*M_PI);
  CeedScalar rho, P, T, E, u[3] = {0.};

  // Initial Conditions
  switch (context->euler_test) {
  case 0: // Traveling vortex
    T = 1 + delta_T;
    // P = rho * T
    // P = S * rho^gamma
    // Solve for rho, then substitute for P
    rho = pow(T/S_vortex, 1 / (gamma - 1));
    P = rho * T;
    u[0] = etv_mean_velocity[0] - C*y0;
    u[1] = etv_mean_velocity[1] + C*x0;

    q[0] = rho;
    q[1] = rho * u[0];
    q[2] = rho * u[1];
    q[3] = rho * u[2];
    q[4] = P / (gamma - 1.) + rho * (u[0]*u[0] + u[1]*u[1]) / 2.;
    break;
  case 1: // Constant zero velocity, density constant, total energy constant
    rho = 1.;
    E = 2.;

    q[0] = rho;
    q[1] = rho * u[0];
    q[2] = rho * u[1];
    q[3] = rho * u[2];
    q[4] = E;
    break;
  case 2: // Constant nonzero velocity, density constant, total energy constant
    rho = 1.;
    E = 2.;
    u[0] = etv_mean_velocity[0];
    u[1] = etv_mean_velocity[1];

    q[0] = rho;
    q[1] = rho * u[0];
    q[2] = rho * u[1];
    q[3] = rho * u[2];
    q[4] = E;
    break;
  case 3: // Velocity zero, pressure constant
    // (so density and internal energy will be non-constant),
    // but the velocity should stay zero and the bubble won't diffuse
    // (for Euler, where there is no thermal conductivity)
    P = 1.;
    T = 1. - S_bubble * exp(1. - r*r);
    rho = P / (R*T);

    q[0] = rho;
    q[1] = rho * u[0];
    q[2] = rho * u[1];
    q[3] = rho * u[2];
    q[4] = rho * (cv * T + (u[0]*u[0] + u[1]*u[1])/2.);
    break;
  case 4: // Constant nonzero velocity, pressure constant
    // (so density and internal energy will be non-constant),
    // it should be transported across the domain, but velocity stays constant
    P = 1.;
    T = 1. - S_bubble * exp(1. - r*r);
    rho = P / (R*T);
    u[0] = etv_mean_velocity[0];
    u[1] = etv_mean_velocity[1];

    q[0] = rho;
    q[1] = rho * u[0];
    q[2] = rho * u[1];
    q[3] = rho * u[2];
    q[4] = rho * (cv * T + (u[0]*u[0] + u[1]*u[1])/2.);
    break;
  }
  // Return
  return 0;
}

// *****************************************************************************
// This QFunction sets the initial conditions for Euler traveling vortex
// *****************************************************************************
CEED_QFUNCTION(ICsEuler)(void *ctx, CeedInt Q,
                         const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  const EulerContext context = (EulerContext)ctx;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar q[5];

    Exact_Euler(3, context->currentTime, x, 5, q, ctx);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of Euler equations
//   with explicit time stepping method
//
// This is 3D Euler for compressible gas dynamics in conservation
//   form with state variables of density, momentum density, and total
//   energy density.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density,      Ui = rho ui
//   E   - Total Energy Density,  E  = P / (gamma - 1) + rho (u u)/2
//
// Euler Equations:
//   drho/dt + div( U )                   = 0
//   dU/dt   + div( rho (u x u) + P I3 )  = 0
//   dE/dt   + div( (E + P) u )           = 0
//
// Equation of State:
//   P = (gamma - 1) (E - rho (u u) / 2)
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
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   g               ,  Gravity
//   gamma  = cp / cv,  Specific heat ratio
// *****************************************************************************

// *****************************************************************************
// This helper function provides forcing term for Euler traveling vortex
//   manufactured solution
// *****************************************************************************
static inline int MMSforce_Euler(CeedInt dim, CeedScalar time,
                                 const CeedScalar X[],
                                 CeedInt Nf, CeedScalar force[], void *ctx) {
  // Context
  const EulerContext context = (EulerContext)ctx;
  const CeedScalar vortex_strength = context->vortex_strength;
  const CeedScalar *center = context->center; // Center of the domain
  CeedScalar *etv_mean_velocity = context->etv_mean_velocity;
  const int euler_test = context->euler_test;

  // For test cases 1 and 3 the velocity is zero
  if (euler_test == 1 || euler_test == 3)
    for (CeedInt i=0; i<3; i++) etv_mean_velocity[i] = 0.;

  // Setup
  const CeedScalar gamma = 1.4;
  const CeedScalar cv = 2.5; // cv computed based on R = 1
  const CeedScalar x = X[0], y = X[1], z = X[2]; // Coordinates
  // Vortex center
  const CeedScalar xc = center[0] + etv_mean_velocity[0] * time;
  const CeedScalar yc = center[1] + etv_mean_velocity[1] * time;

  const CeedScalar x0 = x - xc;
  const CeedScalar y0 = y - yc;
  const CeedScalar r = sqrt( x0*x0 + y0*y0 );
  const CeedScalar C = vortex_strength * exp((1. - r*r)/2.) / (2. * M_PI);
  const CeedScalar S = (gamma - 1.) * vortex_strength * vortex_strength /
                       (8.*gamma*M_PI*M_PI);
  // Note this is not correct for test cases
  const CeedScalar u[3] = {etv_mean_velocity[0] - C*y0,
                           etv_mean_velocity[1] + C*x0,
                           0.
                          };
// TODO: Forcing terms
  for (int j=0; j<5; j++) force[j] = 0.;
  return 0;
}
// *****************************************************************************
CEED_QFUNCTION(Euler)(void *ctx, CeedInt Q,
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
  // Context
  const EulerContext context = (EulerContext)ctx;
  const CeedScalar currentTime = context->currentTime;
  const int stabilization = context->stabilization;
  const CeedScalar gamma  = 1.4;
  const CeedScalar cv = 2.5;
  const CeedScalar R = 1.;

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
    const CeedScalar
    E_kinetic = 0.5 * rho * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]),
    E_internal = E - E_kinetic,
    P = E_internal * (gamma - 1); // P = pressure
    const CeedScalar X[] = {x[0][i], x[1][i], x[2][i]};
    CeedScalar force[5];
    MMSforce_Euler(3, currentTime, X, 5, force, ctx);


    // dFconvdq[3][5][5] = dF(convective)/dq at each direction
    CeedScalar dFconvdq[3][5][5] = {{{0}}};
    for (int j=0; j<3; j++) {
      dFconvdq[j][4][0] = u[j] * (2*R*E_kinetic/cv - E*gamma);
      dFconvdq[j][4][4] = u[j] * gamma;
      for (int k=0; k<3; k++) {
        dFconvdq[j][k+1][0] = -u[j]*u[k] + (j==k?(E_kinetic*R/cv):0);
        dFconvdq[j][k+1][4] = (j==k?(R/cv):0);
        dFconvdq[j][k+1][k+1] = (j!=k?u[j]:0);
        dFconvdq[j][0][k+1] = (j==k?1:0);
        dFconvdq[j][j+1][k+1] = u[k] * ((j==k?2:0) - R/cv);
        dFconvdq[j][4][k+1] = -(R/cv)*u[j]*u[k] + (j==k?(E*gamma - R*E_kinetic/cv):0);
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

    // The Physics
    for (int j=0; j<5; j++) {
      v[j][i] = force[j]; // MMS forcing term
      for (int k=0; k<3; k++)
        dv[k][j][i] = 0; // Zero dv so all future terms can safely sum into it
    }

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
    // -- Total Energy Density
    // ---- (E + P) u
    for (int j=0; j<3; j++)
      dv[j][4][i]  += wdetJ * (E + P) * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] +
                                         u[2]*dXdx[j][2]);
    //Stabilization
    CeedScalar uX[3];
    for (int j=0; j<3; j++)
      uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
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
    switch (stabilization) {
    case 0:        // Galerkin
      break;
    case 1:        // SU
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] = dFconvdqT[j][k][l] * Tau[l] * StrongConv[l];

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
// This QFunction implements the Euler equations with (mentioned above)
//   with implicit time stepping method
//
//  SU   = Galerkin + grad(v) . ( Ai^T * Tau * (Aj q,j) )
//  SUPG = Galerkin + grad(v) . ( Ai^T * Tau * (qdot + Aj q,j - body force) )
//
// *****************************************************************************
CEED_QFUNCTION(IFunction_Euler)(void *ctx, CeedInt Q,
                                const CeedScalar *const *in, CeedScalar *const *out) {
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
  // Context
  const EulerContext context = (EulerContext)ctx;
  const CeedScalar currentTime = context->currentTime;
  const int stabilization = context->stabilization;
  const CeedScalar gamma  = 1.4;
  const CeedScalar cv = 2.5;
  const CeedScalar R = 1.;

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
    const CeedScalar
    E_kinetic = 0.5 * rho * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]),
    E_internal = E - E_kinetic,
    P = E_internal * (gamma - 1); // P = pressure
    const CeedScalar X[] = {x[0][i], x[1][i], x[2][i]};
    CeedScalar force[5];
    MMSforce_Euler(3, currentTime, X, 5, force, ctx);


    // dFconvdq[3][5][5] = dF(convective)/dq at each direction
    CeedScalar dFconvdq[3][5][5] = {{{0}}};
    for (int j=0; j<3; j++) {
      dFconvdq[j][4][0] = u[j] * (2*R*E_kinetic/cv - E*gamma);
      dFconvdq[j][4][4] = u[j] * gamma;
      for (int k=0; k<3; k++) {
        dFconvdq[j][k+1][0] = -u[j]*u[k] + (j==k?(E_kinetic*R/cv):0);
        dFconvdq[j][k+1][4] = (j==k?(R/cv):0);
        dFconvdq[j][k+1][k+1] = (j!=k?u[j]:0);
        dFconvdq[j][0][k+1] = (j==k?1:0);
        dFconvdq[j][j+1][k+1] = u[k] * ((j==k?2:0) - R/cv);
        dFconvdq[j][4][k+1] = -(R/cv)*u[j]*u[k] + (j==k?(E*gamma - R*E_kinetic/cv):0);
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

    // Strong residual
    CeedScalar StrongResid[5];
    for (int j=0; j<5; j++)
      StrongResid[j] = qdot[j][i] + StrongConv[j];

    // The Physics
    // Zero v and dv so all future terms can safely sum into it
    for (int j=0; j<5; j++) {
      v[j][i] = 0;
      for (int k=0; k<3; k++)
        dv[k][j][i] = 0;
    }
    //-----mass matrix
    for (int j=0; j<5; j++)
      v[j][i] += wdetJ*qdot[j][i];

    // Forcing
    for (int j=0; j<5; j++)
      v[j][i] -= force[j]; // MMS forcing term

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
    // -- Total Energy Density
    // ---- (E + P) u
    for (int j=0; j<3; j++)
      dv[j][4][i]  -= wdetJ * (E + P) * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] +
                                         u[2]*dXdx[j][2]);
    //Stabilization
    CeedScalar uX[3];
    for (int j=0; j<3; j++)
      uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
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
    switch (stabilization) {
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
    case 2:
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
// This QFunction sets the boundary conditions
//   In this problem, only in/outflow BCs are implemented
//
//  Inflow and outflow faces are determined based on
//    sign(dot(etv_mean_velocity, normal)):
//      sign(dot(etv_mean_velocity, normal)) > 0 : outflow BCs
//      sign(dot(etv_mean_velocity, normal)) < 0 : inflow BCs
//
//  Outflow BCs:
//    The validity of the weak form of the governing equations is
//      extended to the outflow.
//
//  Inflow BCs:
//    Prescribed T_inlet and P_inlet are converted to conservative variables
//      and applied weakly.
//
// *****************************************************************************
CEED_QFUNCTION(Euler_Sur)(void *ctx, CeedInt Q,
                          const CeedScalar *const *in,
                          CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*qdataSur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1],
                   (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  EulerContext context = (EulerContext)ctx;
  const int euler_test = context->euler_test;
  const bool implicit = context->implicit;
  CeedScalar *etv_mean_velocity = context->etv_mean_velocity;
  CeedScalar T_inlet = 1.;
  CeedScalar P_inlet = 1.;

  // For test cases 1 and 3 the background velocity is zero
  if (euler_test == 1 || euler_test == 3)
    for (CeedInt i=0; i<3; i++) etv_mean_velocity[i] = 0.;

  // For test cases 1 and 2, T_inlet = T_inlet = 0.4
  if (euler_test == 1 || euler_test == 2) T_inlet = P_inlet = .4;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho      =  q[0][i];
    const CeedScalar u[3]     = {q[1][i] / rho,
                                 q[2][i] / rho,
                                 q[3][i] / rho
                                };
    const CeedScalar E        =  q[4][i];

    // -- Interp-to-Interp qdata
    // For explicit mode, the surface integral is on the RHS of ODE qdot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(qdot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb     =   (implicit ? -1. : 1.) * qdataSur[0][i];
    // ---- Normal vectors
    const CeedScalar norm[3]    =   {qdataSur[1][i],
                                     qdataSur[2][i],
                                     qdataSur[3][i]
                                    };
    const CeedScalar X[] = {x[0][i], x[1][i], x[2][i]};
    const CeedScalar gamma = 1.4;
    const CeedScalar cv = 2.5;
    const CeedScalar R = 1.;

    // face_n = Normal vector of the face
    const CeedScalar face_n = norm[0]*etv_mean_velocity[0] +
                              norm[1]*etv_mean_velocity[1] +
                              norm[2]*etv_mean_velocity[2];
    // The Physics
    // Zero v so all future terms can safely sum into it
    for (int j=0; j<5; j++) v[j][i] = 0;

    // Implementing in/outflow BCs
    if (face_n > 0) { // outflow
      const CeedScalar E_kinetic = (u[0]*u[0] + u[1]*u[1]) / 2.;
      const CeedScalar P  = (E - E_kinetic * rho) * (gamma - 1.); // pressure
      const CeedScalar u_n = norm[0]*u[0] + norm[1]*u[1] +
                             norm[2]*u[2]; // Normal velocity
      // The Physics
      // -- Density
      v[0][i] -= wdetJb * rho * u_n;

      // -- Momentum
      for (int j=0; j<3; j++)
        v[j+1][i] -= wdetJb *(rho * u_n * u[j] + norm[j] * P);

      // -- Total Energy Density
      v[4][i] -= wdetJb * u_n * (E + P);

    } else { // inflow
      const CeedScalar rho_inlet = P_inlet/(R*T_inlet);
      const CeedScalar E_kinetic_inlet = (etv_mean_velocity[0]*etv_mean_velocity[0] +
                                          etv_mean_velocity[1]*etv_mean_velocity[1]) / 2.;
      // incoming total energy
      const CeedScalar E_inlet = rho_inlet * (cv * T_inlet + E_kinetic_inlet);

      // The Physics
      // -- Density
      v[0][i] -= wdetJb * rho_inlet * face_n;

      // -- Momentum
      for (int j=0; j<3; j++)
        v[j+1][i] -= wdetJb *(rho_inlet * face_n * etv_mean_velocity[j] +
                              norm[j] * P_inlet);

      // -- Total Energy Density
      v[4][i] -= wdetJb * face_n * (E_inlet + P_inlet);
    }

  } // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************

#endif // eulervortex_h
