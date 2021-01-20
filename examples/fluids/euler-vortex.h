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
  CeedScalar T_inlet;
  CeedScalar P_inlet;
  int euler_test;
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
//   Density     = 1
//   Pressure    = 1
//   Temperature = P / rho - (gamma - 1) vortex_strength**2
//                 exp(1 - r**2) / (8 gamma pi**2)
//   Velocity = 1 + vortex_strength exp((1 - r**2)/2.) [yc - y, x - xc] / (2 pi)
//         r  = sqrt( (x - xc)**2 + (y - yc)**2 )
//
// Conversion to Conserved Variables:
//   E   = P / (gamma - 1) + rho (u u)/2
//
// Constants:
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   vortex_strength ,  Strength of vortex
//   center          ,  Location of bubble center
//   gamma  = cp / cv,  Specific heat ratio
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
  const CeedScalar S = (gamma - 1.) * vortex_strength * vortex_strength /
                       (8.*gamma*M_PI*M_PI);
  CeedScalar rho, P, T, E, u[3] = {0.};

  // Initial Conditions
  switch (context->euler_test) {
  case 0: // Traveling vortex
    rho = 1.;
    P = 1.;
    T = P / rho - S * exp(1. - r*r);
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
    P = 1.;
    E = 2.;

    q[0] = rho;
    q[1] = rho * u[0];
    q[2] = rho * u[1];
    q[3] = rho * u[2];
    q[4] = E;
    break;
  case 2: // Constant nonzero velocity, density constant, total energy constant
    rho = 1.;
    P = 1.;
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
    T = 1. - S * exp(1. - r*r);
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
    T = 1. - S * exp(1. - r*r);
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

    Exact_Euler(3, context->time, x, 5, q, ctx);

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
//   E   - Total Energy Density,  E  = rho ( cv T + (u u) / 2 )
//
// Euler Equations:
//   drho/dt + div( U )                   = 0
//   dU/dt   + div( rho (u x u) + P I3 )  = 0
//   dE/dt   + div( (E + P) u )           = 0
//
// Equation of State:
//   P = (gamma - 1) (E - rho (u u) / 2)
//
// Constants:
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   g               ,  Gravity
//   gamma  = cp / cv,  Specific heat ratio
//
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
  const CeedScalar cv = 2.5; // cv computed based on Rd = 1
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
  const CeedScalar gamma  = 1.4;

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
    const CeedScalar P  = 1.; // P = pressure
    const CeedScalar X[] = {x[0][i], x[1][i], x[2][i]};
    CeedScalar force[5];
    MMSforce_Euler(3, currentTime, X, 5, force, ctx);

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
  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements outflow and inflow BCs for
//      3D Euler traveling vortex
//
//  Inflow and outflow faces are determined based on
//    sign(dot(etv_mean_velocity, normal)):
//      sign(dot(etv_mean_velocity, normal)) > 0 : outflow BCs
//      sign(dot(etv_mean_velocity, normal)) < 0 : inflow BCs
//
//  Outflow BCs:
//    The validity of the weak form of the governing equations is
//    extended to the outflow.
//
//  Inflow BCs:
//    Prescribed T_inlet and P_inlet are converted to conservative variables
//      and applied weakly.
//
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
  const CeedScalar T_inlet = context->T_inlet;
  const CeedScalar P_inlet = context->P_inlet;
  CeedScalar *etv_mean_velocity = context->etv_mean_velocity;
  const int euler_test = context->euler_test;

  // For test cases 1 and 3 the velocity is zero
  if (euler_test == 1 || euler_test == 3)
    for (CeedInt i=0; i<3; i++) etv_mean_velocity[i] = 0.;

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
    const CeedScalar wdetJb   =    qdataSur[0][i];
    const CeedScalar norm[3]  =   {qdataSur[1][i],
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
    if (face_n > 1E-5) { // outflow
      const CeedScalar ke = (u[0]*u[0] + u[1]*u[1]) / 2.;  // kinetic energy
      const CeedScalar P  = (E - ke * rho) * (gamma - 1.); // pressure
      const CeedScalar u_n = norm[0]*u[0] + norm[1]*u[1] +
                             norm[2]*u[2]; // Normal velocity
      // -- Density
      v[0][i] -= wdetJb * rho * u_n;
      // -- Momentum
      for (int j=0; j<3; j++)
        v[j+1][i] -= wdetJb *(rho * u_n * u[j] + norm[j] * P);
      // -- Total Energy Density
      v[4][i] -= wdetJb * u_n * (E + P);
    } else if (face_n < -1E-5) { // inflow
      const CeedScalar rho_inlet = P_inlet/(R*T_inlet);    // incoming density
      const CeedScalar ke_inlet = (etv_mean_velocity[0]*etv_mean_velocity[0] +
                                   etv_mean_velocity[1]*etv_mean_velocity[1]) / 2.; // kinetic energy
      // incoming total energy
      const CeedScalar E_inlet = rho_inlet * (cv * T_inlet + ke_inlet);
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
