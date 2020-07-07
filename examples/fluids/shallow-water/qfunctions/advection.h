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
/// Initial condition and operator for the shallow-water example using PETSc

#ifndef advection_h
#define advection_h

#include "../sw_headers.h"

#ifndef __CUDACC__
#  include <math.h>
#endif

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

// *****************************************************************************
// This QFunction sets the the initial condition for the advection of a cosine
// bell shaped scalar function (test case 1 in "A Standard Test Set for 
// Numerical Approximations to the Shallow Water Equations in Spherical 
// Geometry" by Williamson et al. (1992)
// *****************************************************************************
static inline int Exact_SW_Advection(CeedInt dim, CeedScalar time, 
                                     const CeedScalar X[], CeedInt Nf, 
                                     CeedScalar q[], void *ctx) {

  // Context
  const PhysicsContext context = (PhysicsContext)ctx;
  const CeedScalar u0          = context->u0;
  const CeedScalar h0          = context->h0;
  const CeedScalar R           = context->R;
  const CeedScalar gamma       = context->gamma;
  const CeedScalar rho         = R / 3.;

  // Setup
  // -- Compute latitude
  const CeedScalar theta    = asin(X[2] / R);
  // -- Compute longitude
  const CeedScalar lambda   = atan2(X[1], X[0]); 
  // -- Compute great circle distance between (lambda, theta) and the center,
  //    (lambda_c, theta_c)
  const CeedScalar lambda_c = 3. * M_PI / 2.;
  const CeedScalar theta_c  = 0.;
  const CeedScalar r        = R * acos(sin(theta_c)*sin(theta) + cos(theta_c)*cos(theta)*cos(lambda-lambda_c));  

  // Initial Conditions
  q[0] =  u0 * (cos(theta)*cos(gamma) + sin(theta)*cos(lambda)*sin(gamma));
  q[1] = -u0 * sin(lambda)*sin(gamma);
  q[2] = r < rho ? .5*h0*(1 + cos(M_PI*r/rho)) : 0.; // cosine bell
  // Return
  return 0;
}

// *****************************************************************************
// Initial conditions
// *****************************************************************************
CEED_QFUNCTION(ICsSW_Advection)(void *ctx, CeedInt Q,
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

    Exact_SW_Advection(2, 0., x, 5, q, ctx);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the explicit terms of the advection test for the 
// shallow-water equations solver.
//
// For this simple test, the equation represents the tranport of the scalar 
// field h advected by the wind u, on a spherical surface, where the state 
// variables, u_lambda, u_theta (or u_1, u_2) represent the longitudinal 
// and latitudinal components of the velocity (wind) field, and h, represents 
// the height function.
//
// State variable vector: q = (u_lambda, u_theta, h)
//
// Advection Equation:
//   dh/dt + div( (h + H0) u ) = 0
//
// Spatial term of explicit function:
// G(t,q) = - v div((h + H0) u)
//
// This QFunction has been adapted from navier-stokes/advection.h
// *****************************************************************************
CEED_QFUNCTION(SWExplicit_Advection)(void *ctx, CeedInt Q, 
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1],
                   (*dq)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context for the test
  ProblemContext context = (ProblemContext)ctx;
  const CeedScalar H0 = context->H0;
  const CeedScalar CtauS = context->CtauS;
  const CeedScalar strong_form = context->strong_form;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup
    // -- Interp in
    const CeedScalar u[2]      =  {q[0][i],
                                   q[1][i]
                                  };
    const CeedScalar h         =   q[2][i];
    // *INDENT-OFF*
    const CeedScalar du[2][2]  = {{dq[0][0][i],  // du_1/dx
                                   dq[1][0][i]}, // du_1/dy
                                  {dq[0][1][i],  // du_2/dx
                                   dq[1][1][i]}  // du_2/dy
                                 };
    // *INDENT-ON*
    const CeedScalar dh[2]     =   {dq[0][2][i],
                                    dq[1][2][i]
                                   };
    // Interp-to-Interp qdata
    const CeedScalar wdetJ    =   qdata[0][i];
    // Interp-to-Grad qdata
    // Pseudo inverse of dxdX: (x_i,j)+ = X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[2][3] = {{qdata[1][i],
                                    qdata[2][i],
                                    qdata[3][i]},
                                   {qdata[4][i],
                                    qdata[5][i],
                                    qdata[6][i]}
                                  };
    // *INDENT-ON*

    // Note with the order that du was filled and the order that dXdx was filled
    //   du[j][k]= du_j / dX_K    (note cap K to be clear this is u_{j,xi_k})
    //   dXdx[k][j] = dX_K / dx_j
    //   X_K=Kth reference element coordinate (note cap X and K instead of xi_k}
    //   x_j and u_j are jth  physical position and velocity components

    // The Physics

    // No Change in velocity
    for (CeedInt f=0; f<2; f++) {
      for (CeedInt j=0; j<2; j++)
        dv[j][f][i] = 0;
      v[f][i] = 0;
    }

    // -- Height:
    // Evaluate the strong form using 
    // div((h + H0) u) = u . grad(h) + (h + H0) div(u), with H0 constant
    // or in index notation: (u_j h)_{,j} = u_j h_j + (h + H0) u_{j,j}
    CeedScalar div_u = 0, u_dot_grad_h = 0;
    for (CeedInt j=0; j<2; j++) {
      CeedScalar dhdx_j = 0;
      for (CeedInt k=0; k<2; k++) {
        div_u += du[j][k] * dXdx[k][j]; // u_{j,j} = u_{j,K} X_{K,j}
        dhdx_j += dh[k] * dXdx[k][j];
      }
      u_dot_grad_h += u[j] * dhdx_j;
    }
    CeedScalar strongConv = (h + H0)*div_u + u_dot_grad_h;

    // Weak Galerkin convection term: dv \cdot ((h + H0) u)
    for (CeedInt j=0; j<2; j++)
      dv[j][2][i] = (1 - strong_form) * wdetJ * (h + H0) * 
                    (u[0]*dXdx[j][0] + u[1]*dXdx[j][1]);
    v[2][i] = 0;

    // Strong Galerkin convection term: - v div((h + H0) u)
    v[2][i] = -strong_form * wdetJ * strongConv;

    // Stabilization requires a measure of element transit time in the velocity
    // field u.
    CeedScalar uX[2];
    for (CeedInt j=0; j<2; j++) 
      uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1];
    const CeedScalar TauS = CtauS / sqrt(uX[0]*uX[0] + uX[1]*uX[1]);
    for (CeedInt j=0; j<2; j++)
      dv[j][2][i] -= wdetJ * TauS * strongConv * uX[j];
  } // End Quadrature Point Loop

  return 0;
}

// *****************************************************************************
// This QFunction implements the implicit terms of the advection test for the 
// shallow-water equations solver.
//
// For this simple test, the equation represents the tranport of the scalar 
// field h advected by the wind u, on a spherical surface, where the state 
// variables, u_lambda, u_theta (or u_1, u_2) represent the longitudinal 
// and latitudinal components of the velocity (wind) field, and h, represents 
// the height function.
//
// State variable vector: q = (u_lambda, u_theta, h)
//
// Advection Equation:
//   dh/dt + div( h u ) = 0
//
// Spatial term of explicit function:
// F(t,q) = - v div(h u)
//
// To the spatial term F(t,q) one needs to add qdot (time derivative) on the LHS
// This QFunction has been adapted from navier-stokes/advection.h
// *****************************************************************************
CEED_QFUNCTION(SWImplicit_Advection)(void *ctx, CeedInt Q, 
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*dq)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*qdot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  // Context for the test
  ProblemContext context = (ProblemContext)ctx;
  const CeedScalar H0 = context->H0;
  const CeedScalar CtauS = context->CtauS;
  const CeedScalar strong_form = context->strong_form;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar u[2]      =  {q[0][i],
                                   q[1][i]
                                  };
    const CeedScalar h         =   q[2][i];
    // *INDENT-OFF*
    const CeedScalar du[2][2]  = {{dq[0][0][i],  // du_1/dx
                                   dq[1][0][i]}, // du_1/dy
                                  {dq[0][1][i],  // du_2/dx
                                   dq[1][1][i]}  // du_2/dy
                                 };
    // *INDENT-ON*
    const CeedScalar dh[2]     =   {dq[0][2][i],
                                    dq[1][2][i]
                                   };
    // Interp-to-Interp qdata
    const CeedScalar wdetJ    =   qdata[0][i];
    // Interp-to-Grad qdata
    // Pseudo inverse of dxdX: (x_i,j)+ = X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[2][3] = {{qdata[1][i],
                                    qdata[2][i],
                                    qdata[3][i]},
                                   {qdata[4][i],
                                    qdata[5][i],
                                    qdata[6][i]}
                                  };
    // *INDENT-ON*

    // Note with the order that du was filled and the order that dXdx was filled
    //   du[j][k]= du_j / dX_K    (note cap K to be clear this is u_{j,xi_k} )
    //   dXdx[k][j] = dX_K / dx_j
    //   X_K=Kth reference element coordinate (note cap X and K instead of xi_k}
    //   x_j and u_j are jth  physical position and velocity components
    
     // The Physics
    
     // No Change in velocity
     for (CeedInt f=0; f<2; f++) {
       for (CeedInt j=0; j<2; j++)
         dv[j][f][i] = 0;
       v[f][i] = wdetJ * qdot[f][i]; //K Mass/transient term
     }

     // -- Height:
     // Evaluate the strong form using
     // div((h + H0) u) = u . grad(h) + (h + H0) div(u), with H0 constant
     // or in index notation: (u_j h)_{,j} = u_j h_j + (h + H0) u_{j,j}
     CeedScalar div_u = 0, u_dot_grad_h = 0;
     for (CeedInt j=0; j<2; j++) {
       CeedScalar dhdx_j = 0;
       for (CeedInt k=0; k<2; k++) {
         div_u += du[j][k] * dXdx[k][j]; // u_{j,j} = u_{j,K} X_{K,j}
         dhdx_j += dh[k] * dXdx[k][j];
       }
       u_dot_grad_h += u[j] * dhdx_j;
     }
     CeedScalar strongConv = (h + H0) *div_u + u_dot_grad_h;
     CeedScalar strongResid = qdot[4][i] + strongConv;
 
     v[2][i] = wdetJ * qdot[2][i]; // transient part (ALWAYS)
 
     // Weak Galerkin convection term: -dv \cdot ((h + H0) u)
     for (CeedInt j=0; j<2; j++)
       dv[j][2][i] = -wdetJ * (1 - strong_form) * (h + H0) * 
                     (u[0]*dXdx[j][0] + u[1]*dXdx[j][1]);
 
     // Strong Galerkin convection term: v div((h + H0) u)
     v[2][i] += wdetJ * strong_form * strongConv;
 
     // Stabilization requires a measure of element transit time in the velocity
     // field u.
     CeedScalar uX[2];
     for (CeedInt j=0; j<2; j++) 
       uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1];
     const CeedScalar TauS = CtauS / sqrt(uX[0]*uX[0] + uX[1]*uX[1]);
 
     for (CeedInt j=0; j<2; j++)
       switch (context->stabilization) {
       case 0:
         break;
       case 1: dv[j][2][i] += wdetJ * TauS * strongConv * uX[j];  //SU
         break;
       case 2: dv[j][2][i] += wdetJ * TauS * strongResid * uX[j];  //SUPG
         break;
       }
   } // End Quadrature Point Loop
 
   return 0;
 }

// *****************************************************************************
// This QFunction implements the Jacobian of of the advection test for the 
// shallow-water equations solver.
//
// For this simple test, the equation represents the tranport of the scalar 
// field h advected by the wind u, on a spherical surface, where the state 
// variables, u_lambda, u_theta (or u_1, u_2) represent the longitudinal 
// and latitudinal components of the velocity (wind) field, and h, represents 
// the height function.
//
// Discrete Jacobian: 
// dF/dq^n = sigma * dF/dqdot|q^n + dF/dq|q^n
// ("sigma * dF/dqdot|q^n" will be added later)
// *****************************************************************************
CEED_QFUNCTION(SWJacobian_Advection)(void *ctx, CeedInt Q, 
                                     const CeedScalar *const *in,
                                     CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*deltaq)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*deltadvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  // Context
  ProblemContext context = (ProblemContext)ctx;
  const CeedScalar H0           = context->H0;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // Interp in
    const CeedScalar h          =   q[2][i];
    // Functional derivatives in
    const CeedScalar deltau[2]  =  {deltaq[0][0][i],
                                    deltaq[1][0][i]
                                   };

    // Interp-to-Interp qdata
    const CeedScalar wdetJ      =   qdata[0][i];
    // Interp-to-Grad qdata
    // Pseudo inverse of dxdX: (x_i,j)+ = X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[2][3] = {{qdata[1][i],
                                    qdata[2][i],
                                    qdata[3][i]},
                                   {qdata[4][i],
                                    qdata[5][i],
                                    qdata[6][i]}
                                  };
    // *INDENT-ON*

    // The Physics
    // Jacobian spatial terms for F_1(t,q):
    // 0
    deltadvdX[0][0][i] = 0; // lambda component
    deltadvdX[1][0][i] = 0; // theta component
    // Jacobian spatial terms for F_2(t,q):
    // 0
    deltadvdX[0][1][i] = 0; // lambda component
    deltadvdX[1][1][i] = 0; // theta component
    // Jacobian spatial terms for F_3(t,q):
    // - dv \cdot ((H_0 + h) delta u)
    deltadvdX[1][2][i] = - (H0 + h)*wdetJ*(deltau[0]*dXdx[1][0] + deltau[1]*dXdx[1][1]); // lambda component
    deltadvdX[0][2][i] = - (H0 + h)*wdetJ*(deltau[0]*dXdx[0][0] + deltau[1]*dXdx[0][1]); // theta component

  } // End Quadrature Point Loop

  // Return
  return 0;
}


// *****************************************************************************
#endif // advection_h
