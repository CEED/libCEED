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
/// Shock tube initial condition and Euler equation operator for Navier-Stokes
/// example using PETSc - modified from eulervortex.h

// Model from:
//   On the Order of Accuracy and Numerical Performance of Two Classes of
//   Finite Volume WENO Schemes, Zhang, Zhang, and Shu (2011).

#ifndef shocktube_h
#define shocktube_h

#ifndef __CUDACC__
#  include <math.h>
#endif

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#ifndef shocktube_context_struct
#define shocktube_context_struct
typedef struct ShockTubeContext_ *ShockTubeContext;
struct ShockTubeContext_ {
  CeedScalar cv;
  CeedScalar Cyzb;
  CeedScalar Byzb;
  bool implicit;
  bool yzb;
  int stabilization;
};
#endif

// *****************************************************************************
// This function sets the initial conditions
//
//   Temperature:
//     T   = P / (rho * R)
//   Density:
//     rho = 1.0        if x <= mid_point
//         =            if x >  mid_point
//   Pressure:
//     P   = 1.0        if x <= mid_point
//         = 0.1        if x >  mid_point
//   Velocity:
//     u   = 0
//   Velocity/Momentum Density:
//     Ui  = rho ui
//   Total Energy:
//     E   = P / (gamma - 1) + rho (u u)/2
//
// Constants:
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   mid_point       ,  Location of initial domain mid_point
//   gamma  = cp / cv,  Specific heat ratio
//
// *****************************************************************************

// *****************************************************************************
// This helper function provides support for the exact, time-dependent solution
//   (currently not implemented) and IC formulation for Euler traveling vortex
// *****************************************************************************
CEED_QFUNCTION_HELPER int Exact_ShockTube(CeedInt dim, CeedScalar time,
    const CeedScalar X[], CeedInt Nf, CeedScalar q[],
    void *ctx) {

  // Context
  const SetupContext context = (SetupContext)ctx;
  const CeedScalar mid_point = context->mid_point;      // Midpoint of the domain
  const CeedScalar P_high = context->P_high;            // Driver section pressure
  const CeedScalar rho_high = context->rho_high;        // Driver section density
  const CeedScalar P_low = context->P_low;              // Driven section pressure
  const CeedScalar rho_low = context->rho_low;          // Driven section density

  // Setup
  const CeedScalar gamma = 1.4;    // ratio of specific heats
  const CeedScalar x     = X[0];   // Coordinates

  CeedScalar rho, P, u[3] = {0.};

  // Initial Conditions
  if (x <= mid_point) {
    rho = rho_high;
    P   = P_high;
  } else {
    rho = rho_low;
    P   = P_low;
  }

  // Assign exact solution
  q[0] = rho;
  q[1] = rho * u[0];
  q[2] = rho * u[1];
  q[3] = rho * u[2];
  q[4] = P / (gamma-1.0) + rho * (u[0]*u[0]) / 2.;

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction sets the initial conditions for shock tube
// *****************************************************************************
CEED_QFUNCTION(ICsShockTube)(void *ctx, CeedInt Q,
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

    Exact_ShockTube(3, 0., x, 5, q, ctx);

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
//   E   - Total Energy Density,  E  = P / (gamma - 1) + rho (u u)/2
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
// *****************************************************************************
CEED_QFUNCTION(EulerShockTube)(void *ctx, CeedInt Q,
                               const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  const CeedScalar gamma = 1.4;

  ShockTubeContext context = (ShockTubeContext)ctx;
  const CeedScalar cv = context->cv;
  const CeedScalar Cyzb = context->Cyzb;
  const CeedScalar Byzb = context->Byzb;

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
    const CeedScalar drho[3]    =  {dq[0][0][i],
                                    dq[1][0][i],
                                    dq[2][0][i]
                                   };
    const CeedScalar dU[3][3]   = {{dq[0][1][i],
                                    dq[1][1][i],
                                    dq[2][1][i]},
                                   {dq[0][2][i],
                                    dq[1][2][i],
                                    dq[2][2][i]},
                                   {dq[0][3][i],
                                    dq[1][3][i],
                                    dq[2][3][i]}
                                  };
    const CeedScalar dE[3]      =  {dq[0][4][i],
                                    dq[1][4][i],
                                    dq[2][4][i]
                                   };
    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ      =   q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // dU/dx
    CeedScalar du[3][3] = {{0}};
    CeedScalar drhodx[3] = {0};
    CeedScalar dEdx[3] = {0};
    CeedScalar dUdx[3][3] = {{0}};
    CeedScalar dXdxdXdxT[3][3] = {{0}};
    for (int j=0; j<3; j++) {
      for (int k=0; k<3; k++) {
        du[j][k] = (dU[j][k] - drho[k]*u[j]) / rho;
        drhodx[j] += drho[k] * dXdx[k][j];
        dEdx[j] += dE[k] * dXdx[k][j];
        for (int l=0; l<3; l++) {
          dUdx[j][k] += dU[j][l] * dXdx[l][k];
          dXdxdXdxT[j][k] += dXdx[j][l]*dXdx[k][l];  //dXdx_j,k * dXdx_k,j
        }
      }
    }

    // *INDENT-ON*
    const CeedScalar
    E_kinetic  = 0.5 * rho * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]),
    E_internal = E - E_kinetic,
    P          = E_internal * (gamma - 1); // P = pressure

    // The Physics
    // Zero v and dv so all future terms can safely sum into it
    for (int j=0; j<5; j++) {
      v[j][i] = 0;
      for (int k=0; k<3; k++)
        dv[k][j][i] = 0;
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

    // -- YZB stabilization
    if (context->yzb) {
      CeedScalar drho_norm = 0.0;         // magnitude of the density gradient
      CeedScalar j_vec[3] = {0.0};        // unit vector aligned with the density gradient
      CeedScalar j_gradn[3] = {0.0};      // j * grad(N)
      CeedScalar h_shock = 0.0;           // element lengthscale
      CeedScalar acoustic_vel = 0.0;      // characteristic velocity, set to acoustic speed
      CeedScalar tau_shock = 0.0;         // timescale
      CeedScalar nu_shock = 0.0;          // artificial diffusion

      // Unit vector aligned with the density gradient
      drho_norm = sqrt(drhodx[0]*drhodx[0] + drhodx[1]*drhodx[1] +
                       drhodx[2]*drhodx[2]);
      for (int j=0; j<3; j++)
        j_vec[j] = drhodx[j] / (drho_norm + 1e-20);

      // Approximate dot(j_vec,grad(N)) using the metric tensor
      for (int j=0; j<3; j++)
        j_gradn[j] = j_vec[0] * dXdx[0][j]
                     + j_vec[1] * dXdx[1][j]
                     + j_vec[2] * dXdx[2][j];

      if (drho_norm == 0.0) {
        nu_shock = 0.0;
      } else {
        h_shock = 2.0 / (Cyzb * sqrt(j_gradn[0]*j_gradn[0] + j_gradn[1]*j_gradn[1] +
                                     j_gradn[2]*j_gradn[2]));
        acoustic_vel = sqrt(gamma*P/rho);
        tau_shock = h_shock / (2*acoustic_vel) * pow(drho_norm * h_shock / rho, Byzb);
        nu_shock = fabs(tau_shock * acoustic_vel * acoustic_vel);
      }

      for (int j=0; j<3; j++)
        dv[j][0][i] -= wdetJ * nu_shock * drhodx[j];

      for (int k=0; k<3; k++)
        for (int j=0; j<3; j++)
          dv[j][k][i] -= wdetJ * nu_shock * du[k][j];

      for (int j=0; j<3; j++)
        dv[j][4][i] -= wdetJ * nu_shock * dEdx[j];
    }

    // Stabilization

    // Need the Jacobian for the advective fluxes for stabilization
    //    indexed as: jacob_F_conv[direction][flux component][solution component]
    CeedScalar jacob_F_conv[3][5][5] = {{{0.}}};
    CeedScalar vel_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    CeedScalar rho_sq = rho*rho;
    for (int j=0; j<3; j++) {
      jacob_F_conv[j][4][0] = (gamma-1.) * (-u[j]*P/rho - u[j]/rho*(E-vel_sq/
                                            (2.*rho)) + u[j]*(vel_sq/(2.*rho)));
      jacob_F_conv[j][4][4] =  u[j] * gamma;
      for (int k=0; k<3; k++) {
        jacob_F_conv[j][k+1][0] = -u[j]*u[k] + (j==k?((gamma-1.)*vel_sq/(2*rho_sq)):0.);
        jacob_F_conv[j][k+1][4] = (j==k?(gamma-1.):0.);
        jacob_F_conv[j][0][k+1] = (j==k?1.:0.);
        jacob_F_conv[j][4][k+1] = (j==k?(E/rho + (gamma-1.)/rho*(E-vel_sq/
                                         (2*rho))):0.) - (gamma-1.)*u[j]*u[k];
        jacob_F_conv[j][j+1][k+1] = -(gamma-1.)*u[k];
        jacob_F_conv[j][k+1][k+1] += (j==k?(2*u[j]):(u[j]));
      }
    }
    jacob_F_conv[0][2][1] = u[1];
    jacob_F_conv[0][3][1] = u[2];
    jacob_F_conv[1][1][2] = u[0];
    jacob_F_conv[1][3][2] = u[2];
    jacob_F_conv[2][1][3] = u[0];
    jacob_F_conv[2][2][3] = u[1];

    // Transpose of the Jacobian
    CeedScalar jacob_F_conv_T[3][5][5];
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          jacob_F_conv_T[j][k][l] = jacob_F_conv[j][l][k];

    // dqdx collects drhodx, dUdx and dEdx in one vector
    CeedScalar dqdx[5][3];
    for (int j=0; j<3; j++) {
      dqdx[0][j] = drhodx[j];
      dqdx[4][j] = dEdx[j];
      for (int k=0; k<3; k++)
        dqdx[k+1][j] = dUdx[k][j];
    }

    // strong_conv = dF/dq * dq/dx    (Strong convection)
    CeedScalar strong_conv[5] = {0};
    for (int j=0; j<3; j++)
      for (int k=0; k<5; k++)
        for (int l=0; l<5; l++)
          strong_conv[k] += jacob_F_conv[j][k][l] * dqdx[l][j];

    // Tau elements
    CeedScalar uX[3];
    for (int j=0; j<3;
         j++) uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
    const CeedScalar uiujgij = uX[0]*uX[0] + uX[1]*uX[1] + uX[2]*uX[2];
    const CeedScalar Cc      = 1.;
    const CeedScalar Ce      = 1.;
    const CeedScalar f1      = rho * sqrt(uiujgij);
    const CeedScalar TauC    = (Cc * f1) /
                               (8. * (dXdxdXdxT[0][0] + dXdxdXdxT[1][1] + dXdxdXdxT[2][2]));
    const CeedScalar TauM    = 1. / (f1>1. ? f1 : 1.);
    const CeedScalar TauE    = TauM / (Ce * cv);

    const CeedScalar Tau[5]  = {TauC, TauM, TauM, TauM, TauE};

    CeedScalar stab[5][3];
    switch (context->stabilization) {
    case 0:        // Galerkin
      break;
    case 1:        // SU
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++) {
            stab[k][j] = jacob_F_conv_T[j][k][l] * Tau[l] * strong_conv[l];
          }
      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] -= wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    }

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the Euler equations with (mentioned above)
//   with implicit time stepping method
//
// *****************************************************************************
CEED_QFUNCTION(IFunction_EulerShockTube)(void *ctx, CeedInt Q,
    const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];

  const CeedScalar gamma     = 1.4;

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

    // -- Interp-to-Interp q_data
    const CeedScalar wdetJ      =   q_data[0][i];
    // -- Interp-to-Grad q_data
    // ---- Inverse of change of coordinate matrix: X_i,j
    // *INDENT-OFF*
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*
    const CeedScalar
    E_kinetic  = 0.5 * rho * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]),
    E_internal = E - E_kinetic,
    P          = E_internal * (gamma - 1); // P = pressure

    // The Physics
    // Zero v and dv so all future terms can safely sum into it
    for (int j=0; j<5; j++) {
      v[j][i] = 0;
      for (int k=0; k<3; k++)
        dv[k][j][i] = 0;
    }
    //-----mass matrix
    for (int j=0; j<5; j++)
      v[j][i] += wdetJ*q_dot[j][i];

    // -- Density
    // ---- u rho
    for (int j=0; j<3; j++)
      dv[j][0][i]  -= wdetJ*(rho*u[0]*dXdx[j][0] + rho*u[1]*dXdx[j][1] +
                             rho*u[2]*dXdx[j][2]);
    // -- Momentum
    // ---- rho (u x u) + P I3
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        dv[k][j+1][i]  -= wdetJ*((rho*u[j]*u[0] + (j==0?P:0))*dXdx[k][0] +
                                 (rho*u[j]*u[1] + (j==1?P:0))*dXdx[k][1] +
                                 (rho*u[j]*u[2] + (j==2?P:0))*dXdx[k][2]);
    // -- Total Energy Density
    // ---- (E + P) u
    for (int j=0; j<3; j++)
      dv[j][4][i]  -= wdetJ * (E + P) * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] +
                                         u[2]*dXdx[j][2]);
  } // End Quadrature Point Loop

  // Return
  return 0;
}
// *****************************************************************************

#endif // shocktube_h
