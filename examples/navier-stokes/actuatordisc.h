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
  const CeedScalar *q = in[0], *dq = in[1], *qdata = in[2], *x = in[3];
  // Outputs
  CeedScalar *v = out[0], *dv = out[1];
  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar Adisc         = context[12];
  const CeedScalar CT            = context[13];

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
    v[i+1*Q] = .5*rho*u[0]*u[0]*Adisc*CT*wJ; // new body force only in the x-component
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
