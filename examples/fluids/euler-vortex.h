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
//   On the order of accuracy and numerical performance of two classes of
//   finite volume WENO, Zhang, Zhang, and Shu (2009).

#ifndef eulervortex_h
#define eulervortex_h

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
  CeedScalar vortex_strength;
  int wind_type;
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
//   Density     = 1
//   Pressure    = 1
//   Temperature = P / (Rd rho) - (gamma - 1) vortex_strength**2
//                 exp(1 - r**2) / (8 gamma pi**2)
//   Velocity = vortex_strength exp((1 - r**2)/2.) [yc - y, x - xc, 0] / (2 pi)
//         r  = sqrt( (x - xc)**2 + (y - yc)**2 + (z - zc)**2 )
//
// Conversion to Conserved Variables:
//   E   = rho (cv T + (u u)/2 )
//
// TODO: Not sure what to do about BCs
//
// Constants:
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   Rd     = cp - cv,  Specific heat difference
//   vortex_strength ,  Strength of vortex
//   center          ,  Location of bubble center
//   gamma  = cp / cv,  Specific heat ratio
// *****************************************************************************

// *****************************************************************************
// This helper function provides support for the exact, time-dependent solution
//   (currently not implemented) and IC formulation for Euler traveling vortex
// *****************************************************************************
static inline int Exact_Euler(CeedInt dim, CeedScalar time, const CeedScalar X[],
                           CeedInt Nf, CeedScalar q[], void *ctx) {
  // Context
  const SetupContext context = (SetupContext)ctx;
  const CeedScalar cv = context->cv;
  const CeedScalar cp = context->cp;
  const CeedScalar Rd = context->Rd;
  const CeedScalar vortex_strength = context->vortex_strength;
  const CeedScalar *center = context->center;
  const CeedScalar gamma = cp / cv;

  // Setup
  const CeedScalar x = X[0], y = X[1], z = X[2]; // Coordinates
  const CeedScalar x0 = x - center[0];
  const CeedScalar y0 = y - center[1];
  const CeedScalar z0 = z - center[2];
  const CeedScalar r = sqrt( x0*x0 + y0*y0 + z0*z0 );
  // Coefficient for computing perturbation in Velocity
  const CeedScalar C = vortex_strength * exp((1. - r*r)/2.)  / (2. * M_PI);

  // Exact Solutions
  const CeedScalar rho = 1.;
  const CeedScalar P = 1.;
  const CeedScalar T = P / (Rd*rho) - (gamma - 1.) * vortex_strength *
                       vortex_strength * exp(1. - r*r) / (8.*gamma*M_PI*M_PI);
  const CeedScalar u[3] = {1. - C * y0, 1. + C * x0, 0};

  // Initial Conditions
  q[0] = rho;
  q[1] = rho * u[0];
  q[2] = rho * u[1];
  q[3] = rho * u[2];
  q[4] = rho * ( cv*T + (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) / 2. );

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

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar q[5];

    Exact_Euler(3, 0., x, 5, q, ctx);

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
  // *INDENT-ON*

  // Context
  const CeedScalar *context = (const CeedScalar *)ctx;
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
    // P = pressure
    const CeedScalar P  = (E - rho * (u[0]*u[0] + u[1]*u[1] +
                           u[2]*u[2]) / 2.) * (gamma - 1.);

    // The Physics
    for (int j=0; j<5; j++) {
      v[j][i] = 0; // No body force
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
CEED_QFUNCTION(Euler_In)(void *ctx, CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*qdataSur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  return 0;
}
// *****************************************************************************
CEED_QFUNCTION(Euler_Out)(void *ctx, CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*qdataSur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  return 0;
}
// *****************************************************************************

#endif // eulervortex_h
