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
#include <ceed.h>

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

  return 0;
}

#endif // densitycurrent_h
