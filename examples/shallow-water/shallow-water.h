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
//  For now we have sinusoidal terrain and constant reference height H0
//
// *****************************************************************************
static int SWICs(void *ctx, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

  // Inputs
  const CeedScalar *X = in[0];
  // Outputs
  CeedScalar *q0 = out[0], *coords = out[1];
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
//    h_s[i+0*Q]            = sin(x) + cos(y); // put 0 for constant flat topography
    // Reference height
//    H_0[i+0*Q]            = 0; // flat

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
// the state variables, u_lambda, u_theta, represent the longitudinal and latitudinal components
// of the velocity field, and h, represents the height function.
//
// State variable vector: q = (u_lambda,u_theta,h)
//
// Shallow-water Equations spatial terms of explicit function G(t,q) = (G_1(t,q), G_2(t,q)):
// G_1(t,q) = - (omega + f) * khat curl u - grad(|u|^2/2)
// G_2(t,q) = 0
// *****************************************************************************
static int SWExplicit(void *ctx, CeedInt Q, const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q = in[0], *dq = in[1], *qdata = in[2], *x = in[3];
  // Outputs
  CeedScalar *v = out[0], *dv = out[1];
  // Context
  const CeedScalar *context        =  (const CeedScalar*)ctx;
  const CeedScalar f               =   context[0];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // Interp in
    const CeedScalar u[2]      = { q[i+0*Q],
                                   q[i+1*Q]
                                  };
    const CeedScalar h         =   q[i+2*Q];
    // Grad in
    const CeedScalar du[2][2]  = {{dq[i+(0+4*0)*Q],  // du/dx
                                   dq[i+(0+4*1)*Q]}, // du/dy
                                  {dq[i+(1+4*0)*Q],  // dv/dx
                                   dq[i+(1+4*1)*Q]}  // dv/dy
                                 };
    const CeedScalar dh[2]     = { dq[i+(2+4*0)*Q],  // dh/dx
                                   dq[i+(2+4*1)*Q]   // dh/dy
                                  };
    // Interp-to-Interp qdata
    const CeedScalar wJ        =   qdata[i+ 0*Q];

    // The Physics

    // Explicit spatial terms of G_1(t,q):
    // Explicit terms multiplying v
    // - (omega + f) * khat curl u - grad(|u|^2/2)
    v[i+0*Q] = - wJ*(u[0]*du[0][0] + u[1]*du[0][1] + f*u[1]);
    // No explicit terms multiplying dv
    dv[i+(0+3*0)*Q]  = 0;
    dv[i+(0+3*1)*Q]  = 0;

    // Explicit spatial terms of G_2(t,q):
    // Explicit terms multiplying v
    // - (omega + f) * khat curl u - grad(|u|^2/2)
    v[i+1*Q] = - wJ*(u[0]*du[1][0] + u[1]*du[1][1] - f*u[0]);
    // No explicit terms multiplying dv
    dv[i+(1+3*0)*Q]  = 0;
    dv[i+(1+3*1)*Q]  = 0;

    // Explicit spatial terms for G_3(t,q):
    // No explicit terms multiplying v
    v[i+2*Q] = 0;
    // No explicit terms multiplying dv
    dv[i+(2+3*0)*Q]  = 0;
    dv[i+(2+3*1)*Q]  = 0;


  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the implicit terms of the shallow-water
// equations
//
// The equations represent 2D shallow-water flow on a spherical surface, where
// the state variables, u_lambda, u_theta, represent the longitudinal and latitudinal components
// of the velocity field, and h, represents the height function.
//
// State variable vector: q = (u_lambda, u_theta, h)
//
// Shallow-water Equations spatial terms of implicit function: F(t,q) = (F_1(t,q), F_2(t,q)):
// F_1(t,q) = g(grad(h + h_s))
// F_2(t,q) = div((h + H_0) u)
// *****************************************************************************
static int SWImplicit(void *ctx, CeedInt Q, const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q = in[0], *dq = in[1], *qdata = in[2], *x = in[3];
  // Outputs
  CeedScalar *v = out[0], *dv = out[1];
  // Context
  const CeedScalar *context     = (const CeedScalar*)ctx;
  const CeedScalar g            = context[0];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // Interp in
    const CeedScalar u[2]      = { q[i+0*Q],
                                   q[i+1*Q]
                                 };
    const CeedScalar h         =   q[i+2*Q];
    // Grad in
    const CeedScalar du[2][2]  = {{dq[i+(0+4*0)*Q],  // du/dx
                                   dq[i+(0+4*1)*Q]}, // du/dy
                                  {dq[i+(1+4*0)*Q],  // dv/dx
                                   dq[i+(1+4*1)*Q]}  // dv/dy
                                 };
    const CeedScalar dh[2]     = { dq[i+(2+4*0)*Q],  // dh/dx
                                   dq[i+(2+4*1)*Q]   // dh/dy
                                  };

    // Interp-to-Interp qdata
    const CeedScalar wJ        =   qdata[i+ 0*Q];
    // Interp-to-Grad qdata
    const CeedScalar wBJ[2][2] = {{qdata[i+ 1*Q],
                                   qdata[i+ 2*Q]},
                                  {qdata[i+ 3*Q],
                                   qdata[i+ 4*Q]}
                                 };
    // h_s
    const CeedScalar h_s       =   qdata[i+8*Q];
    // H0
    const CeedScalar H_0       =   qdata[i+9*Q];

    // The Physics

    // Implicit spatial terms for F_1(t,q):
    // No implicit terms multiplying v
    v[i+0*Q] = 0;
    // Implicit terms multiplying dv
    // g * grad(h + h_s)
    dv[i+(0+3*0)*Q]  = - g*(h + h_s)*(wBJ[0][0] + wBJ[0][1]); // lambda component
    dv[i+(0+3*1)*Q]  = 0;                                     // theta component

    // Implicit spatial terms for F_2(t,q):
    // No implicit terms multiplying v
    v[i+1*Q] = 0;
    // Implicit terms multiplying dv
    // g * grad(h + h_s)
    dv[i+(1+3*0)*Q]  = 0;                                     // lambda component
    dv[i+(1+3*1)*Q]  = - g*(h + h_s)*(wBJ[1][0] + wBJ[1][1]); // theta component

    // Implicit spatial terms for F_3(t,q):
    // No implicit terms multiplying v
    v[i+2*Q] = 0;
    // Implicit terms multiplying dv
    // div((h + H_0) u)
    dv[i+(2+3*0)*Q]  = - (h + H_0)*(u[0]*wBJ[0][0] + u[1]*wBJ[0][1]);
    dv[i+(2+3*1)*Q]  = - (h + H_0)*(u[0]*wBJ[1][0] + u[1]*wBJ[1][1]);

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the Jacobian of the shallow-water
// equations
//
// The equations represent 2D shallow-water flow on a spherical surface, where
// the state variables, u_lambda, u_theta, represent the longitudinal and latitudinal components
// of the velocity field, and h, represents the height function.
//
// Discrete Jacobian: dF/dq^n = sigma * dF/dqdot|q^n + dF/dq|q^n ("sigma * dF/dqdot|q^n" will be added later)
// *****************************************************************************
static int SWJacobian(void *ctx, CeedInt Q, const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q = in[0], *deltaq = in[1], *qdata = in[2];
  // Outputs
  CeedScalar *dv = out[0];
  // Context
  const CeedScalar *context    =  (const CeedScalar*)ctx;
  const CeedScalar g           = context[0];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // Interp in
    const CeedScalar u[2]      = { q[i+0*Q],
                                   q[i+1*Q]
                                 };
    const CeedScalar h         =   q[i+2*Q];
    // Functional derivatives in
    const CeedScalar deltau[2] = { deltaq[i+0*Q],
                                   deltaq[i+1*Q]
                                 };
    const CeedScalar deltah    =   deltaq[i+2*Q];

    // Interp-to-Grad qdata
    const CeedScalar wBJ[2][2] = {{qdata[i+ 1*Q],
                                   qdata[i+ 2*Q]},
                                  {qdata[i+ 3*Q],
                                   qdata[i+ 4*Q]}
                                 };
    // H0
    const CeedScalar H_0       =   qdata[i+9*Q];

    // The Physics

    // Jacobian spatial terms for F_1(t,q):
    // - dv \cdot (dh u)
    dv[i+(0+3*0)*Q] = - g*wBJ[0][0]*deltah; // lambda component
    dv[i+(0+3*1)*Q] = 0;                    // theta component
    // Jacobian spatial terms for F_2(t,q):
    // - dv \cdot (dh u)
    dv[i+(1+3*0)*Q] = 0;                    // lambda component
    dv[i+(1+3*1)*Q] = - g*wBJ[1][1]*deltah; // theta component
    // Jacobian spatial terms for F_3(t,q):
    // - dv \cdot ((H_0 + h) delta u)
    dv[i+(2+3*0)*Q] = - (H_0 + h)*(deltau[0]*wBJ[0][0] + deltau[1]*wBJ[0][1]);
    dv[i+(2+3*1)*Q] = - (H_0 + h)*(deltau[0]*wBJ[1][0] + deltau[1]*wBJ[1][1]);

  } // End Quadrature Point Loop

  // Return
  return 0;
}


// *****************************************************************************
#endif
