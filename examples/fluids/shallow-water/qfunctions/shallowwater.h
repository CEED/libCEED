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

#ifndef densitycurrent_h
#define densitycurrent_h

#ifndef __CUDACC__
#  include <math.h>
#endif

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#ifndef physics_context_struct
#define physics_context_struct
typedef struct {
  CeedScalar u0;
  CeedScalar v0;
  CeedScalar h0;
  CeedScalar Omega;
  CeedScalar R;
  CeedScalar g;
  CeedScalar H0;
  CeedScalar time;
} PhysicsContext_s;
typedef PhysicsContext_s *PhysicsContext;
#endif // physics_context_struct

// *****************************************************************************
// This QFunction sets the the initial conditions and boundary conditions
//
//  For now we have sinusoidal terrain and constant reference height H0
//
// *****************************************************************************
static inline int Exact_SW(CeedInt dim, CeedScalar time, const CeedScalar X[],
                           CeedInt Nf, CeedScalar q[], void *ctx) {

  // Context
  const PhysicsContext context = (PhysicsContext)ctx;
  const CeedScalar u0          = context->u0;
  const CeedScalar v0          = context->v0;
  const CeedScalar h0          = context->h0;

  // Setup
  // -- Coordinates
  const CeedScalar x = X[0];
  const CeedScalar y = X[1];

  // Initial Conditions
  q[0] = u0;
  q[1] = v0;
  q[2] = h0;
  // Return
  return 0;
}

// *****************************************************************************
// Initial conditions for shallow-water
// *****************************************************************************
CEED_QFUNCTION(ICsSW)(void *ctx, CeedInt Q,
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

    Exact_SW(2, 0., x, 5, q, ctx);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop

  // Return
  return 0;
}


// *****************************************************************************
// This QFunction implements the explicit terms of the shallow-water
// equations
//
// The equations represent 2D shallow-water flow on a spherical surface, where
// the state variables, u_lambda, u_theta (or u_1, u_2) represent the 
// longitudinal and latitudinal components of the velocity field, and h, 
// represents the height function.
//
// State variable vector: q = (u_lambda, u_theta, h)
//
// Shallow-water Equations spatial terms of explicit function 
// G(t,q) = (G_1(t,q), G_2(t,q)):
//   G_1(t,q) = - (omega + f) * khat curl u - grad(|u|^2/2)
//   G_2(t,q) = 0
// *****************************************************************************
CEED_QFUNCTION(SWExplicit)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                           CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1],
                   (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  const PhysicsContext context = (PhysicsContext)ctx;
  const CeedScalar Omega       = context->Omega;
  const CeedScalar R           = context->R;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup
    // -- Compute latitude
    const CeedScalar theta     =  asin(X[2][i] / R);
    // -- Compute Coriolis parameter
    const CeedScalar f         = 2*Omega*sin(theta);
    // -- Interp in
    const CeedScalar u[2]      =  {q[0][i],
                                   q[1][i]
                                  };
    // *INDENT-OFF*
    const CeedScalar du[2][2]  = {{dq[0][0][i],  // du_1/dx
                                   dq[1][0][i]}, // du_1/dy
                                  {dq[0][1][i],  // du_2/dx
                                   dq[1][1][i]}  // du_2/dy
                                 };
    // *INDENT-ON*
    // Interp-to-Interp qdata
    const CeedScalar wdetJ    =   qdata[0][i];

    // The Physics
    // Explicit spatial terms of G_1(t,q):
    // Explicit terms multiplying v
    // - (omega + f) * khat curl u - grad(|u|^2/2)
    v[0][i] = - wdetJ*(u[0]*du[0][0] + u[1]*du[0][1] + f*u[1]);
    // No explicit terms multiplying dv
    dv[0][0][i] = 0;
    dv[1][0][i] = 0;

    // Explicit spatial terms of G_2(t,q):
    // Explicit terms multiplying v
    // - (omega + f) * khat curl u - grad(|u|^2/2)
    v[1][i] = - wdetJ*(u[0]*du[1][0] + u[1]*du[1][1] - f*u[0]);
    // No explicit terms multiplying dv
    dv[0][1][i] = 0;
    dv[1][1][i] = 0;

    // Explicit spatial terms for G_3(t,q):
    // No explicit terms multiplying v
    v[2][i] = 0;
    // No explicit terms multiplying dv
    dv[0][2][i] = 0;
    dv[1][2][i] = 0;

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the implicit terms of the shallow-water
// equations
//
// The equations represent 2D shallow-water flow on a spherical surface, where
// the state variables, u_lambda, u_theta (or u_1, u_2) represent the 
// longitudinal and latitudinal components of the velocity field, and h, 
// represents the height function.
//
// State variable vector: q = (u_lambda, u_theta, h)
//
// Shallow-water Equations spatial terms of implicit function: 
// F(t,q) = (F_1(t,q), F_2(t,q)):
//   F_1(t,q) = g(grad(h + h_s))
//   F_2(t,q) = div((h + H_0) u)
//
// To the spatial term F(t,q) one needs to add qdot (time derivative) on the LHS
// *****************************************************************************
CEED_QFUNCTION(SWImplicit)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                           CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*qdot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  // Context
  const PhysicsContext context  = (PhysicsContext)ctx;
  const CeedScalar g            = context->g;
  const CeedScalar H0           = context->H0;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // Interp in
    const CeedScalar u[2]       =  {q[0][i],
                                    q[1][i]
                                   };
    const CeedScalar h          =   q[2][i];

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
    // h_s
    const CeedScalar hs        =   q[10][i];

    // The Physics
    // Mass matrix
    for (int j=0; j<3; j++)
      v[j][i] = wdetJ*qdot[j][i];

    // Implicit spatial terms for F_1(t,q):
    // No implicit terms multiplying v
    v[0][i] = 0;
    // Implicit terms multiplying dv
    // g * grad(h + h_s)
    dv[0][0][i] = - g*(h + hs)*wdetJ*(dXdx[0][0] + dXdx[0][1]); // lambda component
    dv[1][0][i] = 0;                                            // theta component

    // Implicit spatial terms for F_2(t,q):
    // No implicit terms multiplying v
    v[1][i] = 0;
    // Implicit terms multiplying dv
    // g * grad(h + h_s)
    dv[0][1][i] = 0;                                            // lambda component
    dv[1][1][i] = - g*(h + hs)*wdetJ*(dXdx[1][0] + dXdx[1][1]); // theta component

    // Implicit spatial terms for F_3(t,q):
    // No implicit terms multiplying v
    v[2][i] = 0;
    // Implicit terms multiplying dv
    // div((h + H_0) u)
    dv[0][2][i] = - (h + H0)*wdetJ*(u[0]*dXdx[0][0] + u[1]*dXdx[0][1]); // lambda component
    dv[1][2][i] = - (h + H0)*wdetJ*(u[0]*dXdx[1][0] + u[1]*dXdx[1][1]); // theta component

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the Jacobian of the shallow-water
// equations
//
// The equations represent 2D shallow-water flow on a spherical surface, where
// the state variables, u_lambda, u_theta (or u_1, u_2) represent the 
// longitudinal and latitudinal components of the velocity field, and h, 
// represents the height function.
//
// Discrete Jacobian: 
// dF/dq^n = sigma * dF/dqdot|q^n + dF/dq|q^n
// ("sigma * dF/dqdot|q^n" will be added later)
// *****************************************************************************
CEED_QFUNCTION(SWJacobian)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                           CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*deltaq)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*deltadvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  // Context
  const PhysicsContext context  = (PhysicsContext)ctx;
  const CeedScalar g            = context->g;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // Interp in
    const CeedScalar h          =   q[2][i];
    // H0
    const CeedScalar H_0        =   q[4][i];
    // Functional derivatives in
    const CeedScalar deltau[2]  =  {deltaq[0][0][i],
                                    deltaq[1][0][i]
                                   };
    const CeedScalar deltah     =   deltaq[0][1][i];

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
    // - dv \cdot (delta h u)
    deltadvdX[0][0][i] = - g*wdetJ*dXdx[0][0]*deltah; // lambda component
    deltadvdX[1][0][i] = 0;                           // theta component
    // Jacobian spatial terms for F_2(t,q):
    // - dv \cdot (delta h u)
    deltadvdX[0][1][i] = 0;                           // lambda component
    deltadvdX[1][1][i] = - g*wdetJ*dXdx[1][1]*deltah; // theta component
    // Jacobian spatial terms for F_3(t,q):
    // - dv \cdot ((H_0 + h) delta u)
    deltadvdX[1][2][i] = - (H_0 + h)*wdetJ*(deltau[0]*dXdx[1][0] + deltau[1]*dXdx[1][1]); // theta component
    deltadvdX[0][2][i] = - (H_0 + h)*wdetJ*(deltau[0]*dXdx[0][0] + deltau[1]*dXdx[0][1]); // lambda component

  } // End Quadrature Point Loop

  // Return
  return 0;
}


// *****************************************************************************
#endif
