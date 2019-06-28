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
/// Initial condition and operator for the shallow-water equations example using PETSc

#ifndef densitycurrent_h
#define densitycurrent_h

#ifndef CeedPragmaOMP
#  ifdef _OPENMP
#    define CeedPragmaOMP_(a) _Pragma(#a)
#    define CeedPragmaOMP(a) CeedPragmaOMP_(omp a)
#  else
#    define CeedPragmaOMP(a)
#  endif
#endif

#include <math.h>

// *****************************************************************************
// This QFunction sets the the initial conditions and boundary conditions
//
//  TO DO
//
// *****************************************************************************
static int ICsSW(void *ctx, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

  // Inputs
  const CeedScalar *X = in[0];
  // Outputs
  CeedScalar *q0 = out[0], *h_s = out[1], *coords = out[2];
  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar u0     = context[0];
  const CeedScalar v0     = context[1];
  const CeedScalar h0     = context[2];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Coordinates
    const CeedScalar x    = X[i+0*Q];
    const CeedScalar y    = X[i+1*Q];

    // Initial Conditions
    q0[i+0*Q]             = u0;
    q0[i+1*Q]             = v0;
    q0[i+2*Q]             = h0;
    // Terrain topography
    h_s[i+0*Q]            = sin(x) + cos(y); // put 0 for constant flat topography

    // Coordinates
    coords[i+0*Q]         = x;
    coords[i+1*Q]         = y;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the explicit terms of the shallow-water
// equations
//
// The equations represent 2D shallow-water flow on a spherical surface, where
// the state variable, h, represents the height function.
//
// State (scalar) variable: h
//
// Shallow-water Equations spatial terms of explicit function G(t,q) = (G_1(t,q), G_2(t,q)):
// G_1(t,q) = - (omega + f) * khat curl u - grad(|u|^2/2)
// G_2(t,q) = - div(h u)
// *****************************************************************************
static int SWExplicit(void *ctx, CeedInt Q, const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q = in[0], *dq = in[1], *qdata = in[2], *x = in[3];
  // Outputs
  CeedScalar *v = out[0], *dv = out[1];
  // Context
  const CeedScalar *context        =  (const CeedScalar*)ctx;
  const CeedScalar omega           =   context[0];
  const CeedScalar f               =   context[1];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // Interp in
    const CeedScalar u[2]          = { q[i+0*Q],
                                       q[i+1*Q]
                                      };
    const CeedScalar h             =   q[i+2*Q];
    // Grad in
    const CeedScalar du[4]         = { dq[i+(0+4*0)*Q],
                                       dq[i+(0+4*1)*Q],
                                       dq[i+(1+4*0)*Q],
                                       dq[i+(1+4*1)*Q]
                                      };
    const CeedScalar dh[2]         = { dq[i+(2+4*0)*Q],
                                       dq[i+(2+4*1)*Q]
                                      };
    // Interp-to-Interp qdata
    const CeedScalar wJ            =   qdata[i+ 0*Q];
    // Interp-to-Grad qdata
    const CeedScalar wBJ[4]        = { qdata[i+ 1*Q],
                                       qdata[i+ 2*Q],
                                       qdata[i+ 3*Q],
                                       qdata[i+ 4*Q]
                                     };
    // Grad-to-Grad qdata
    // Symmetric 2x2 matrix (only store 3 entries)
    const CeedScalar wBBJ[3]       = { qdata[i+5*Q],
                                       qdata[i+6*Q],
                                       qdata[i+7*Q]
                                     };
    // |u|^2
    /*const CeedScalar gradmodusq[2] = { u[0],
                                       u[1]
                                     };*/
    // curl u
    const CeedScalar curlu         =   du[2]-du[1];

    // The Physics

    // Explicit spatial equation for u
    // - |u|^2/2
    dv[i+(0+4*0)*Q]  -= u[0]*wBJ[0] + u[0]*wBJ[1];
    dv[i+(0+4*1)*Q]  -= u[0]*wBJ[2] + u[0]*wBJ[3];
    dv[i+(1+4*0)*Q]  -= u[1]*wBJ[0] + u[1]*wBJ[1];
    dv[i+(1+4*1)*Q]  -= u[1]*wBJ[2] + u[1]*wBJ[3];
    // - (omega + f) * khat curl u
    v[i+0*Q] = 0;
    v[i+1*Q] = 0;
    v[i+2*Q] = - (omega + f) * wJ * curlu;

    // Explicit spatial equation for h
    // h u
    dv[i+(1+4*0)*Q]  = h*u[0]*wBJ[0] + h*u[1]*wBJ[1];
    dv[i+(1+4*1)*Q]  = h*u[0]*wBJ[2] + h*u[1]*wBJ[3];
    // No Change
    v[i+1*Q] = 0;
    v[i+2*Q] = 0;


  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the implicit terms of the shallow-water
// equations
//
// The equations represent 2D shallow-water flow on a spherical surface, where
// the state variable, h, represents the height function.
//
// State (scalar) variable: h
//
// Shallow-water Equations spatial terms of implicit function: F(t,q) = (F_1(t,q), F_2(t,q)):
// F_1(t,q) = g * grad(h + h_s)
// F_2(t,q) = h0 * div u
// *****************************************************************************
static int SWImplicit(void *ctx, CeedInt Q, const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q = in[0], *dq = in[1], *qdata = in[2], *x = in[3];
  // Outputs
  CeedScalar *v = out[0], *dv = out[1];
  // Context
  const CeedScalar *context     = (const CeedScalar*)ctx;
  const CeedScalar h0           = context[0];
  const CeedScalar g            = context[1];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // Interp in
    const CeedScalar u[2]     = { q[i+0*Q],
                                  q[i+1*Q]
                                  };
    const CeedScalar h        =   q[i+2*Q];
    const CeedScalar h_s      =   q[i+3*Q];;
    // Grad in
    const CeedScalar du[4]    = { dq[i+(0+4*0)*Q],
                                  dq[i+(0+4*1)*Q],
                                  dq[i+(1+4*0)*Q],
                                  dq[i+(1+4*1)*Q]
                                 };
    const CeedScalar dh[2]    = { dq[i+(2+4*0)*Q],
                                  dq[i+(2+4*1)*Q]
                                 };
    const CeedScalar dh_s[2]  = { dq[i+(3+4*0)*Q],
                                  dq[i+(3+4*1)*Q]
                                 };
    // Interp-to-Interp qdata
    const CeedScalar wJ       =   qdata[i+ 0*Q];
    // Interp-to-Grad qdata
    const CeedScalar wBJ[4]   = { qdata[i+ 1*Q],
                                  qdata[i+ 2*Q],
                                  qdata[i+ 3*Q],
                                  qdata[i+ 4*Q]
                                };
    // Grad-to-Grad qdata
    // Symmetric 2x2 matrix (only store 3 entries)
    const CeedScalar wBBJ[3]  = { qdata[i+5*Q],
                                  qdata[i+6*Q],
                                  qdata[i+7*Q]
                                };

    // The Physics

    // Implicit spatial equation for u
    // g * grad(h + h_s)
    dv[i+(0+4*0)*Q]  = g*(dh[0] + dh_s[0])*wBJ[0] + g*(dh[0] + dh_s[1])*wBJ[1];
    dv[i+(0+4*1)*Q]  = g*(dh[0] + dh_s[0])*wBJ[2] + g*(dh[0] + dh_s[1])*wBJ[3];
    dv[i+(1+4*0)*Q]  = g*(dh[1] + dh_s[0])*wBJ[0] + g*(dh[1] + dh_s[1])*wBJ[1];
    dv[i+(1+4*1)*Q]  = g*(dh[1] + dh_s[0])*wBJ[2] + g*(dh[1] + dh_s[1])*wBJ[3];
    // No Change
    v[i+0*Q] = 0;
    v[i+1*Q] = 0;

    // Implicit spatial equation for h
    // h0 * div u
    dv[i+(1+4*0)*Q]  = h0*(u[0]*wBJ[0] + u[1]*wBJ[1]);
    dv[i+(1+4*1)*Q]  = h0*(u[0]*wBJ[2] + u[1]*wBJ[3]);
    // No Change
    v[i+3*Q] = 0;


  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the Jacobian of the shallow-water
// equations
//
// The equations represent 2D shallow-water flow on a spherical surface, where
// the state variable, h, represents the height function.
//
// State (scalar) variable: u, v, h
// *****************************************************************************
static int SWJacobian(void *ctx, CeedInt Q, const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q = in[0], *dq = in[1], *qdata = in[2], *x = in[3];
  // Outputs
  CeedScalar *v = out[0], *dv = out[1];
  // Context
  const CeedScalar *context        =  (const CeedScalar*)ctx;
  const CeedScalar h0           = context[0];
  const CeedScalar g            = context[1];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {

    // TO DO

  } // End Quadrature Point Loop

  // Return
  return 0;
}


// *****************************************************************************
#endif
