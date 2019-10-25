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
/// Advection initial condition and operator for Navier-Stokes example using PETSc

#ifndef advection2d_h
#define advection2d_h

#include <math.h>

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

// *****************************************************************************
// This QFunction sets the the initial conditions and boundary conditions
//
// Initial Conditions:
//   Mass Density:
//     Constant mass density of 1.0
//   Momentum Density:
//     Rotational field in x,y
//   Energy Density:
//     Maximum of 1. x0 decreasing linearly to 0. as radial distance increases
//       to (1.-r/rc), then 0. everywhere else
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
CEED_QFUNCTION(ICsAdvection2d)(void *ctx, CeedInt Q,
                               const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[Q] = (CeedScalar(*)[Q])in[0];
  // Outputs
  CeedScalar (*q0)[Q] = (CeedScalar(*)[Q])out[0],
             (*coords)[Q] = (CeedScalar(*)[Q])out[1];
  // Context
  const CeedScalar *context = (const CeedScalar *)ctx;
  const CeedScalar rc = context[8];
  const CeedScalar lx = context[9];
  const CeedScalar ly = context[10];
  const CeedScalar lz = context[11];
  const CeedScalar *periodic = &context[12];
  // Setup
  const CeedScalar tol = 1.e-14;
  const CeedScalar x0[3] = {0.25*lx, 0.5*ly, 0.5*lz};
  const CeedScalar x1[3] = {0.625*lx, (0.5 + sqrt(3)/8)*ly, 0.5*lz};
  const CeedScalar x2[3] = {0.625*lx, (0.5 - sqrt(3)/8)*ly, 0.5*lz};
  const CeedScalar center[3] = {0.5*lx, 0.5*ly, 0.5*lz};

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Coordinates
    const CeedScalar x = X[0][i];
    const CeedScalar y = X[1][i];

    // Initial Conditions
    q0[0][i] = 1.;
    q0[1][i] = -(y - center[1]);
    q0[2][i] =  (x - center[0]);
    q0[3][i] = 0;
    q0[4][i] = 0;
    CeedScalar r = sqrt(pow(x - x0[0], 2) + pow(y - x0[1], 2));
    CeedScalar E = 1 - r/rc;
    if (q0[4][i] < E) q0[4][i] = E;
    r = sqrt(pow(x - x1[0], 2) + pow(y - x1[1], 2));
    if (r <= rc) q0[4][i] = 1;
    r = sqrt(pow(x - x2[0], 2) + pow(y - x2[1], 2));
    E = (r <= rc) ? .5 + .5*cos(r*M_PI/rc) : 0;
    if (q0[4][i] < E) q0[4][i] = E;

    // Coordinates
    coords[0][i] = x;
    coords[1][i] = y;
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

typedef struct Advection2dContext_ *Advection2dContext;
struct Advection2dContext_ {
  CeedScalar CtauS;
  CeedScalar strong_form;
  int stabilization; // See StabilizationType: 0=none, 1=SU, 2=SUPG
};

// *****************************************************************************
// This QFunction implements the following formulation of the advection equation
//
// This is 2D advection given in two formulations based upon the weak form.
//
// State Variables: q = ( rho, U1, U2, E )
//   rho - Mass Density
//   Ui  - Momentum Density    ,  Ui = rho ui
//   E   - Total Energy Density
//
// Advection Equation:
//   dE/dt + div( E u ) = 0
//
// *****************************************************************************
CEED_QFUNCTION(Advection2d)(void *ctx, CeedInt Q,
                            const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*q)[Q] = (CeedScalar(*)[Q])in[0],
                   (*dq)[5][Q] = (CeedScalar(*)[5][Q])in[1],
                   (*qdata)[Q] = (CeedScalar(*)[Q])in[2],
                   (*x)[Q] = (CeedScalar(*)[Q])in[3];
  // Outputs
  CeedScalar (*v)[Q] = (CeedScalar(*)[Q])out[0],
             (*dv)[5][Q] = (CeedScalar(*)[5][Q])out[1];
  Advection2dContext context = ctx;
  const CeedScalar CtauS = context->CtauS;
  const CeedScalar strong_form = context->strong_form;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho        =    q[0][i];
    const CeedScalar u[3]       =   {q[1][i] / rho,
                                     q[2][i] / rho,
                                     q[3][i] / rho
                                    };
    const CeedScalar E          =    q[4][i];
    // -- Grad in
    const CeedScalar drho[2]    =   {dq[0][0][i],
                                     dq[1][0][i],
                                    };
    const CeedScalar du[3][2]   = {{(dq[0][1][i] - drho[0]*u[0]) / rho,
                                    (dq[1][1][i] - drho[1]*u[0]) / rho},
                                   {(dq[0][2][i] - drho[0]*u[1]) / rho,
                                    (dq[1][2][i] - drho[1]*u[1]) / rho},
                                   {(dq[0][3][i] - drho[0]*u[2]) / rho,
                                    (dq[1][3][i] - drho[1]*u[2]) / rho},
                                  };
    const CeedScalar dE[3]      =   {dq[0][4][i],
                                     dq[1][4][i],
                                    };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ         =    qdata[0][i];
    // -- Interp-to-Grad qdata
    // ---- Inverse of change of coordinate matrix: X_i,j
    const CeedScalar dXdx[2][2] =  {{qdata[1][i],
                                     qdata[2][i]},
                                    {qdata[3][i],
                                     qdata[4][i]},
                                   };

    // The Physics

    // No Change in density or momentum
    for (int f=0; f<4; f++) {
      for (int j=0; j<2; j++)
        dv[j][f][i] = 0;
      v[f][i] = 0;
    }

    // -- Total Energy
    // Evaluate the strong form using div(E u) = u . grad(E) + E div(u)
    // or in index notation: (u_j E)_{,j} = u_j E_j + E u_{j,j}
    CeedScalar div_u = 0, u_dot_grad_E = 0;
    for (int j=0; j<2; j++) {
      CeedScalar dEdx_j = 0;
      for (int k=0; k<2; k++) {
        div_u += du[j][k] * dXdx[k][j]; // u_{j,j} = u_{j,K} X_{K,j}
        dEdx_j += dE[k] * dXdx[k][j];
      }
      u_dot_grad_E += u[j] * dEdx_j;
    }
    CeedScalar strongConv = E*div_u + u_dot_grad_E;

    // Weak Galerkin convection term: dv \cdot (E u)
    for (int j=0; j<2; j++)
      dv[j][4][i] = (1 - strong_form) * wJ * E * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1]);
    v[4][i] = 0;

    // Strong Galerkin convection term: - v div(E u)
    v[4][i] = -strong_form * wJ * strongConv;

    // Stabilization requires a measure of element transit time in the velocity
    // field u.
    CeedScalar uX[2];
    for (int j=0; j<2; j++) uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1];
    const CeedScalar TauS = CtauS / sqrt(uX[0]*uX[0] + uX[1]*uX[1]);
    for (int j=0; j<2; j++)
      dv[j][4][i] -= wJ * TauS * strongConv * uX[j];
  } // End Quadrature Point Loop

  return 0;
}
// *****************************************************************************
CEED_QFUNCTION(IFunction_Advection2d)(void *ctx, CeedInt Q,
    const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*q)[Q] = (CeedScalar(*)[Q])in[0],
                   (*dq)[5][Q] = (CeedScalar(*)[5][Q])in[1],
                   (*qdot)[Q] = (CeedScalar(*)[Q])in[2],
                   (*qdata)[Q] = (CeedScalar(*)[Q])in[3];
  // Outputs
  CeedScalar (*v)[Q] = (CeedScalar(*)[Q])out[0],
             (*dv)[5][Q] = (CeedScalar(*)[5][Q])out[1];
  Advection2dContext context = ctx;
  const CeedScalar CtauS = context->CtauS;
  const CeedScalar strong_form = context->strong_form;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho        =    q[0][i];
    const CeedScalar u[3]       =   {q[1][i] / rho,
                                     q[2][i] / rho,
                                     q[3][i] / rho
                                    };
    const CeedScalar E          =    q[4][i];
    // -- Grad in
    const CeedScalar drho[2]    =   {dq[0][0][i],
                                     dq[1][0][i],
                                    };
    const CeedScalar du[3][2]   = {{(dq[0][1][i] - drho[0]*u[0]) / rho,
                                    (dq[1][1][i] - drho[1]*u[0]) / rho},
                                   {(dq[0][2][i] - drho[0]*u[1]) / rho,
                                    (dq[1][2][i] - drho[1]*u[1]) / rho},
                                   {(dq[0][3][i] - drho[0]*u[2]) / rho,
                                    (dq[1][3][i] - drho[1]*u[2]) / rho},
                                  };
    const CeedScalar dE[3]      =   {dq[0][4][i],
                                     dq[1][4][i],
                                    };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ         =    qdata[0][i];
    // -- Interp-to-Grad qdata
    // ---- Inverse of change of coordinate matrix: X_i,j
    const CeedScalar dXdx[2][2] =  {{qdata[1][i],
                                     qdata[2][i]},
                                    {qdata[3][i],
                                     qdata[4][i]},
                                   };

    // The Physics

    // No Change in density or momentum
    for (int f=0; f<4; f++) {
      for (int j=0; j<2; j++)
        dv[j][f][i] = 0;
      v[f][i] = wJ * qdot[f][i];
    }

    // -- Total Energy
    // Evaluate the strong form using div(E u) = u . grad(E) + E div(u)
    // or in index notation: (u_j E)_{,j} = u_j E_j + E u_{j,j}
    CeedScalar div_u = 0, u_dot_grad_E = 0;
    for (int j=0; j<2; j++) {
      CeedScalar dEdx_j = 0;
      for (int k=0; k<2; k++) {
        div_u += du[j][k] * dXdx[k][j]; // u_{j,j} = u_{j,K} X_{K,j}
        dEdx_j += dE[k] * dXdx[k][j];
      }
      u_dot_grad_E += u[j] * dEdx_j;
    }
    CeedScalar strongConv = E*div_u + u_dot_grad_E;
    CeedScalar strongResid = qdot[4][i] + strongConv;

    v[4][i] = wJ * qdot[4][i]; // transient part

    // Weak Galerkin convection term: -dv \cdot (E u)
    for (int j=0; j<2; j++)
      dv[j][4][i] = -wJ * (1 - strong_form) * E * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1]);

    // Strong Galerkin convection term: v div(E u)
    v[4][i] += wJ * strong_form * strongConv;

    // Stabilization requires a measure of element transit time in the velocity
    // field u.
    CeedScalar uX[2];
    for (int j=0; j<2; j++) uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1];
    const CeedScalar TauS = CtauS / sqrt(uX[0]*uX[0] + uX[1]*uX[1]);

    for (int j=0; j<2; j++)
      switch (context->stabilization) {
      case 0:
        break;
      case 1: dv[j][4][i] += wJ * TauS * strongConv * uX[j];
        break;
      case 2: dv[j][4][i] += wJ * TauS * strongResid * uX[j];
        break;
      }
  } // End Quadrature Point Loop

  return 0;
}

// *****************************************************************************
#endif
