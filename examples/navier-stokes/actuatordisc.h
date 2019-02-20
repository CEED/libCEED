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

#include <math.h>

// *****************************************************************************
// This QFunction sets the the initial conditions and boundary conditions
//
// Initial Conditions:
//   Mass Density:
//     Constant mass density of 1.0
//   Momentum Density:
//     Translational field in x
//   Energy Density:
//      0.
//
//  Boundary Conditions:
//    Mass Density:
//      0.0 flux
//    Momentum Density:
//      0.0
//    Energy Density:
//      0.0 flux
//
// *****************************************************************************
static int ICsActuatorDisc(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *X = in[0];
  // Outputs
  CeedScalar *q0 = out[0], *coords = out[1];
  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar rc         = context[8];
  const CeedScalar lx         = context[9];
  const CeedScalar ly         = context[10];
  const CeedScalar lz         = context[11];
  // Setup
  const CeedScalar tol = 1.e-14;
  const CeedScalar x0[3] = {0.25*lx, 0.5*ly, 0.5*lz};
  const CeedScalar center[3] = {0.5*lx, 0.5*ly, 0.5*lz};

  #pragma omp simd
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Coordinates
    const CeedScalar x = X[i+0*Q];
    const CeedScalar y = X[i+1*Q];
    const CeedScalar z = X[i+2*Q];
//    // -- Energy
//    const CeedScalar r = sqrt(pow((x - x0[0]), 2) +
//                              pow((y - x0[1]), 2) +
//                              pow((z - x0[2]), 2));

    // Initial Conditions
    q0[i+0*Q] = 1.;
    q0[i+1*Q] = 0.5;
    q0[i+2*Q] = 0.0;
    q0[i+3*Q] = 0.0;
    q0[i+4*Q] = 0.0;

    // Homogeneous Dirichlet Boundary Conditions for Momentum
    if ( fabs(x - 0.0) < tol || fabs(x - lx) < tol
         || fabs(y - 0.0) < tol || fabs(y - ly) < tol
         || fabs(z - 0.0) < tol || fabs(z - lz) < tol ) {
      q0[i+1*Q] = 0.0;
      q0[i+2*Q] = 0.0;
      q0[i+3*Q] = 0.0;
    }

    // Coordinates
    coords[i+0*Q] = x;
    coords[i+1*Q] = y;
    coords[i+2*Q] = z;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}


// *****************************************************************************
// This QFunction implements the following formulation of the actuator disc model
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density    ,  Ui = rho ui
//   E   - Total Energy Density
//
// Body Force for Actuator Disc Model:
//   f = ....
//
// *****************************************************************************
static int AD(void *ctx, CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q = in[0], *dq = in[1], *qdata = in[2];
  // Outputs
  CeedScalar *v = out[0], *dv = out[1];
  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar Adisc         = context[0];
  const CeedScalar CT            = context[1];

  #pragma omp simd
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho     =   q[i+0*Q];
    const CeedScalar u[3]    = { q[i+1*Q] / rho,
                                 q[i+2*Q] / rho,
                                 q[i+3*Q] / rho
                               };
    const CeedScalar E       =   q[i+4*Q];
    // -- Grad in
    const CeedScalar drho[3] = {  dq[i+(0+5*0)*Q],
                                  dq[i+(0+5*1)*Q],
                                  dq[i+(0+5*2)*Q]
                               };
    const CeedScalar du[9]   = { (dq[i+(1+5*0)*Q] - drho[0]*u[0]) / rho,
                                 (dq[i+(1+5*1)*Q] - drho[1]*u[0]) / rho,
                                 (dq[i+(1+5*2)*Q] - drho[2]*u[0]) / rho,
                                 (dq[i+(2+5*0)*Q] - drho[0]*u[1]) / rho,
                                 (dq[i+(2+5*1)*Q] - drho[1]*u[1]) / rho,
                                 (dq[i+(2+5*2)*Q] - drho[2]*u[1]) / rho,
                                 (dq[i+(3+5*0)*Q] - drho[0]*u[2]) / rho,
                                 (dq[i+(3+5*1)*Q] - drho[1]*u[2]) / rho,
                                 (dq[i+(3+5*2)*Q] - drho[2]*u[2]) / rho
                               };
    const CeedScalar dE[3]   = {  dq[i+(4+5*0)*Q],
                                  dq[i+(4+5*1)*Q],
                                  dq[i+(4+5*2)*Q]
                               };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ       =   qdata[i+ 0*Q];

    // The Physics

    // -- Density
    // ---- No Change
    dv[i+(0+0*5)*Q] = 0;
    dv[i+(0+1*5)*Q] = 0;
    dv[i+(0+2*5)*Q] = 0;
    v[i+0*Q] = 0;

    // -- Momentum
    dv[i+(1+0*5)*Q] = 0;
    dv[i+(1+1*5)*Q] = 0;
    dv[i+(1+2*5)*Q] = 0;
    dv[i+(2+0*5)*Q] = 0;
    dv[i+(2+1*5)*Q] = 0;
    dv[i+(2+2*5)*Q] = 0;
    dv[i+(3+0*5)*Q] = 0;
    dv[i+(3+1*5)*Q] = 0;
    dv[i+(3+2*5)*Q] = 0;
    v[i+1*Q] = 0.5*rho*u[0]*u[0]*Adisc*CT*wJ; // new body force only in the x-component
    v[i+2*Q] = 0;
    v[i+3*Q] = 0;

    // -- Total Energy
    // ---- No Change
    if (1) {
      dv[i+(4+5*0)*Q]  = 0;
      dv[i+(4+5*1)*Q]  = 0;
      dv[i+(4+5*2)*Q]  = 0;
      v[i+4*Q] = 0;
    }

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
