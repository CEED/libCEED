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

#ifndef advection_h
#define advection_h

#include <math.h>

// *****************************************************************************
// This QFunction sets the the initial conditions and boundary conditions
//
// Initial Conditions:
//   Mass Density:
//     Constant mass density of 1.0
//   Momentum Density:
//     Rotational field in x,y with no momentum in z
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
CEED_QFUNCTION(ICsAdvection)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*X)[Q] = (CeedScalar(*)[Q])in[0];

  // Outputs
  CeedScalar (*q0)[Q] = (CeedScalar(*)[Q])out[0],
             (*coords)[Q] = (CeedScalar(*)[Q])out[1];
  // *INDENT-ON*

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
  const CeedScalar center[3] = {0.5*lx, 0.5*ly, 0.5*lz};

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Coordinates
    const CeedScalar x = X[0][i];
    const CeedScalar y = X[1][i];
    const CeedScalar z = X[2][i];
    // -- Energy
    CeedScalar r ;
    CeedInt dimBubble=2; // 3 is a sphere, 2 is a cylinder
    switch (dimBubble) {
    //  original sphere
    case 3: {
      r = sqrt(pow((x - x0[0]), 2) +
               pow((y - x0[1]), 2) +
               pow((z - x0[2]), 2));
    } break;
    // cylinder (needs periodicity to work properly)
    case 2: {
      r = sqrt(pow((x - x0[0]), 2) +
               pow((y - x0[1]), 2) );
    } break;
    }

    // Initial Conditions
    q0[0][i] = 1.;
    q0[1][i] = -(y - center[1]);
    q0[2][i] =  (x - center[0]);
    q0[3][i] = 0.0;
    CeedInt continuityBubble=-1; // 0 is original sphere switch to -1 to challenge solver with sharp gradients in back half of bubble
    switch (continuityBubble) {
    // original continuous, smooth shape
    case 0: {
      q0[4][i] = r <= rc ? (1.-r/rc) : 0.;
    } break;
    // discontinuous, sharp back half shape
    case -1: {
      q0[4][i] = ((r <= rc) && (y<center[1])) ? (1.-r/rc) : 0.;
    } break;
    // attempt to define a finite thickness that will get resolved under grid refinement
    case 2: {
      q0[4][i] = ((r <= rc) && (y<center[1])) ? (1.-r/rc)*fmin(1.0,(center[1]-y)/1.25) : 0.;
    } break;
    }


    // Coordinates
    coords[0][i] = x;
    coords[1][i] = y;
    coords[2][i] = z;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

typedef struct AdvectionContext_ *AdvectionContext;
struct AdvectionContext_ {
  CeedScalar CtauS;
  CeedScalar strong_form;
  int stabilization; // See StabilizationType: 0=none, 1=SU, 2=SUPG
};

// *****************************************************************************
// This QFunction implements the following formulation of the advection equation
//
// This is 3D advection given in two formulations based upon the weak form.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density    ,  Ui = rho ui
//   E   - Total Energy Density
//
// Advection Equation:
//   dE/dt + div( E u ) = 0
//
// *****************************************************************************
CEED_QFUNCTION(Advection)(void *ctx, CeedInt Q,
                          const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  // *INDENT-OFF*
  const CeedScalar (*q)[Q] = (CeedScalar(*)[Q])in[0],
                   (*dq)[5][Q] = (CeedScalar(*)[5][Q])in[1],
                   (*qdata)[Q] = (CeedScalar(*)[Q])in[2],
                   (*x)[Q] = (CeedScalar(*)[Q])in[3];

  // Outputs
  CeedScalar (*v)[Q] = (CeedScalar(*)[Q])out[0],
             (*dv)[5][Q] = (CeedScalar(*)[5][Q])out[1];
  // *INDENT-ON*

  // Context
  AdvectionContext context = ctx;
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
    const CeedScalar drho[3]    =   {dq[0][0][i],
                                     dq[1][0][i],
                                     dq[2][0][i]
                                    };
    // *INDENT-OFF*
    const CeedScalar du[3][3]   = {{(dq[0][1][i] - drho[0]*u[0]) / rho,
                                    (dq[1][1][i] - drho[1]*u[0]) / rho,
                                    (dq[2][1][i] - drho[2]*u[0]) / rho},
                                   {(dq[0][2][i] - drho[0]*u[1]) / rho,
                                    (dq[1][2][i] - drho[1]*u[1]) / rho,
                                    (dq[2][2][i] - drho[2]*u[1]) / rho},
                                   {(dq[0][3][i] - drho[0]*u[2]) / rho,
                                    (dq[1][3][i] - drho[1]*u[2]) / rho,
                                    (dq[2][3][i] - drho[2]*u[2]) / rho}
                                  };
    // *INDENT-ON*
    const CeedScalar dE[3]      =   {dq[0][4][i],
                                     dq[1][4][i],
                                     dq[2][4][i]
                                    };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ         =    qdata[0][i];
    // -- Interp-to-Grad qdata
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] =  {{qdata[1][i],
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

    // Note with the order that du was filled and the order that dXdx was filled
    //   du[j][k]= du_j / dX_K    (note cap K to be clear this is u_{j,xi_k})
    //   dXdx[k][j] = dX_K / dx_j
    //   X_K=Kth reference element coordinate (note cap X and K instead of xi_k}
    //   x_j and u_j are jth  physical position and velocity components

    // The Physics

    // No Change in density or momentum
    for (int f=0; f<4; f++) {
      for (int j=0; j<3; j++)
        dv[j][f][i] = 0;
      v[f][i] = 0;
    }

    // -- Total Energy
    // Evaluate the strong form using div(E u) = u . grad(E) + E div(u)
    // or in index notation: (u_j E)_{,j} = u_j E_j + E u_{j,j}
    CeedScalar div_u = 0, u_dot_grad_E = 0;
    for (int j=0; j<3; j++) {
      CeedScalar dEdx_j = 0;
      for (int k=0; k<3; k++) {
        div_u += du[j][k] * dXdx[k][j]; // u_{j,j} = u_{j,K} X_{K,j}
        dEdx_j += dE[k] * dXdx[k][j];
      }
      u_dot_grad_E += u[j] * dEdx_j;
    }
    CeedScalar strongConv = E*div_u + u_dot_grad_E;

    // Weak Galerkin convection term: dv \cdot (E u)
    for (int j=0; j<3; j++)
      dv[j][4][i] = (1 - strong_form) * wJ * E * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] + u[2]*dXdx[j][2]);
    v[4][i] = 0;

    // Strong Galerkin convection term: - v div(E u)
    v[4][i] = -strong_form * wJ * strongConv;

    // Stabilization requires a measure of element transit time in the velocity
    // field u.
    CeedScalar uX[3];
    for (int j=0; j<3; j++) uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
    const CeedScalar TauS = CtauS / sqrt(uX[0]*uX[0] + uX[1]*uX[1] + uX[2]*uX[2]);
    for (int j=0; j<3; j++)
      dv[j][4][i] -= wJ * TauS * strongConv * uX[j];
  } // End Quadrature Point Loop

  return 0;
}
// *****************************************************************************
CEED_QFUNCTION(IFunction_Advection)(void *ctx, CeedInt Q,
                          const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*q)[Q] = (CeedScalar(*)[Q])in[0],
                   (*dq)[5][Q] = (CeedScalar(*)[5][Q])in[1],
                   (*qdot)[Q] = (CeedScalar(*)[Q])in[2],
                   (*qdata)[Q] = (CeedScalar(*)[Q])in[3];
  // Outputs
  CeedScalar (*v)[Q] = (CeedScalar(*)[Q])out[0],
             (*dv)[5][Q] = (CeedScalar(*)[5][Q])out[1];
  AdvectionContext context = ctx;
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
    const CeedScalar drho[3]    =   {dq[0][0][i],
                                     dq[1][0][i],
                                     dq[2][0][i]
                                    };
    const CeedScalar du[3][3]   = {{(dq[0][1][i] - drho[0]*u[0]) / rho,
                                    (dq[1][1][i] - drho[1]*u[0]) / rho,
                                    (dq[2][1][i] - drho[2]*u[0]) / rho},
                                   {(dq[0][2][i] - drho[0]*u[1]) / rho,
                                    (dq[1][2][i] - drho[1]*u[1]) / rho,
                                    (dq[2][2][i] - drho[2]*u[1]) / rho},
                                   {(dq[0][3][i] - drho[0]*u[2]) / rho,
                                    (dq[1][3][i] - drho[1]*u[2]) / rho,
                                    (dq[2][3][i] - drho[2]*u[2]) / rho}
                                  };
    const CeedScalar dE[3]      =   {dq[0][4][i],
                                     dq[1][4][i],
                                     dq[2][4][i]
                                    };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ         =    qdata[0][i];
    // -- Interp-to-Grad qdata
    // ---- Inverse of change of coordinate matrix: X_i,j
    const CeedScalar dXdx[3][3] =  {{qdata[1][i],
                                     qdata[2][i],
                                     qdata[3][i]},
                                    {qdata[4][i],
                                     qdata[5][i],
                                     qdata[6][i]},
                                    {qdata[7][i],
                                     qdata[8][i],
                                     qdata[9][i]}
                                   };
// Note with the order that du was filled and the order that dXdx was filled
//   du[j][k]= du_j / dX_K    (note cap K to be clear this is u_{j,xi_k} )
//   dXdx[k][j] = dX_K / dx_j
//   X_K=Kth reference element coordinate (note cap X and K instead of xi_k}
//   x_j and u_j are jth  physical position and velocity components


    // The Physics

    // No Change in density or momentum
    for (int f=0; f<4; f++) {
      for (int j=0; j<2; j++)
        dv[j][f][i] = 0;
      v[f][i] = wJ * qdot[f][i]; //K Mass/transient term
    }

    // -- Total Energy
    // Evaluate the strong form using div(E u) = u . grad(E) + E div(u)
    // or in index notation: (u_j E)_{,j} = u_j E_j + E u_{j,j}
    CeedScalar div_u = 0, u_dot_grad_E = 0;
    for (int j=0; j<3; j++) {
      CeedScalar dEdx_j = 0;
      for (int k=0; k<3; k++) {
        div_u += du[j][k] * dXdx[k][j]; // u_{j,j} = u_{j,K} X_{K,j}
        dEdx_j += dE[k] * dXdx[k][j];
      }
      u_dot_grad_E += u[j] * dEdx_j;
    }
    CeedScalar strongConv = E*div_u + u_dot_grad_E;
    CeedScalar strongResid = qdot[4][i] + strongConv;

    v[4][i] = wJ * qdot[4][i]; // transient part (ALWAYS)

    // Weak Galerkin convection term: -dv \cdot (E u)
    for (int j=0; j<3; j++)
      dv[j][4][i] = -wJ * (1 - strong_form) * E * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] + u[2]*dXdx[j][2]);

    // Strong Galerkin convection term: v div(E u)
    v[4][i] += wJ * strong_form * strongConv;

    // Stabilization requires a measure of element transit time in the velocity
    // field u.
    CeedScalar uX[3];
    for (int j=0; j<3; j++) uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
    const CeedScalar TauS = CtauS / sqrt(uX[0]*uX[0] + uX[1]*uX[1] + uX[2]*uX[2]);

    for (int j=0; j<3; j++)
      switch (context->stabilization) {
      case 0:
        break;
      case 1: dv[j][4][i] += wJ * TauS * strongConv * uX[j];  //SU
        break;
      case 2: dv[j][4][i] += wJ * TauS * strongResid * uX[j];  //SUPG
        break;
      }
  } // End Quadrature Point Loop

  return 0;
}

// *****************************************************************************
#endif
