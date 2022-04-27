// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc


#ifndef newtonian_h
#define newtonian_h

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
  CeedScalar g[3];
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

#ifndef newtonian_context_struct
#define newtonian_context_struct
typedef enum {
  STAB_NONE = 0,
  STAB_SU   = 1, // Streamline Upwind
  STAB_SUPG = 2, // Streamline Upwind Petrov-Galerkin
} StabilizationType;

typedef struct NewtonianIdealGasContext_ *NewtonianIdealGasContext;
struct NewtonianIdealGasContext_ {
  CeedScalar lambda;
  CeedScalar mu;
  CeedScalar k;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar g[3];
  CeedScalar c_tau;
  CeedScalar Ctau_t;
  CeedScalar Ctau_v;
  CeedScalar Ctau_C;
  CeedScalar Ctau_M;
  CeedScalar Ctau_E;
  CeedScalar dt;
  StabilizationType stabilization;
};
#endif

// *****************************************************************************
// Helper function for computing flux Jacobian
// *****************************************************************************
CEED_QFUNCTION_HELPER void computeFluxJacobian_NS(CeedScalar dF[3][5][5],
    const CeedScalar rho, const CeedScalar u[3], const CeedScalar E,
    const CeedScalar gamma, const CeedScalar g[3], const CeedScalar x[3]) {
  CeedScalar u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]; // Velocity square
  CeedScalar e_potential = -(g[0]*x[0] + g[1]*x[1] + g[2]*x[2]);
  for (CeedInt i=0; i<3; i++) { // Jacobian matrices for 3 directions
    for (CeedInt j=0; j<3; j++) { // Rows of each Jacobian matrix
      dF[i][j+1][0] = ((i==j) ? ((gamma-1.)*(u_sq/2. - e_potential)) : 0.) -
                      u[i]*u[j];
      for (CeedInt k=0; k<3; k++) { // Columns of each Jacobian matrix
        dF[i][0][k+1]   = ((i==k) ? 1. : 0.);
        dF[i][j+1][k+1] = ((j==k) ? u[i] : 0.) +
                          ((i==k) ? u[j] : 0.) -
                          ((i==j) ? u[k] : 0.) * (gamma-1.);
        dF[i][4][k+1]   = ((i==k) ? (E*gamma/rho - (gamma-1.)*u_sq/2.) : 0.) -
                          (gamma-1.)*u[i]*u[k];
      }
      dF[i][j+1][4] = ((i==j) ? (gamma-1.) : 0.);
    }
    dF[i][4][0] = u[i] * ((gamma-1.)*u_sq - E*gamma/rho);
    dF[i][4][4] = u[i] * gamma;
  }
}

// *****************************************************************************
// Helper function for computing flux Jacobian of Primitive variables
// *****************************************************************************
CEED_QFUNCTION_HELPER void computeFluxJacobian_NSp(CeedScalar dF[3][5][5],
    const CeedScalar rho, const CeedScalar u[3], const CeedScalar E,
    const CeedScalar Rd, const CeedScalar cv) {
  CeedScalar u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]; // Velocity square
  // TODO Add in gravity's contribution

  CeedScalar T    = ( E / rho - u_sq / 2. ) / cv;
  CeedScalar drdT = -rho / T;
  CeedScalar drdP = 1. / ( Rd * T);
  CeedScalar etot =  E / rho ;
  CeedScalar e2p  = drdP * etot + 1. ;
  CeedScalar e3p  = ( E  + rho * Rd * T );
  CeedScalar e4p  = drdT * etot + rho * cv ;

  for (CeedInt i=0; i<3; i++) { // Jacobian matrices for 3 directions
    for (CeedInt j=0; j<3; j++) { // j counts F^{m_j}
//        [row][col] of A_i
      dF[i][j+1][0] = drdP * u[i] * u[j] + ((i==j) ? 1. : 0.); // F^{{m_j} wrt p
      for (CeedInt k=0; k<3; k++) { // k counts the wrt vel_k
        // this loop handles middle columns for all 5 rows
        dF[i][0][k+1]   =  ((i==k) ? rho  : 0.);   // F^c wrt vel_k
        dF[i][j+1][k+1] = (((j==k) ? u[i] : 0.) +  // F^m_j wrt u_k
                           ((i==k) ? u[j] : 0.) ) * rho;
        dF[i][4][k+1]   = rho * u[i] * u[k]
                          + ((i==k) ? e3p  : 0.) ; // F^e wrt u_k
      }
      dF[i][j+1][4] = drdT * u[i] * u[j]; // F^{m_j} wrt T
    }
    dF[i][4][0] = u[i] * e2p; // F^e wrt p
    dF[i][4][4] = u[i] * e4p; // F^e wrt T
    dF[i][0][0] = u[i] * drdP; // F^c wrt p
    dF[i][0][4] = u[i] * drdT; // F^c wrt T
  }
}

CEED_QFUNCTION_HELPER void PrimitiveToConservative_fwd(const CeedScalar rho,
    const CeedScalar u[3], const CeedScalar E, const CeedScalar Rd,
    const CeedScalar cv, const CeedScalar dY[5], CeedScalar dU[5]) {
  CeedScalar u_sq = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
  CeedScalar T    = ( E / rho - u_sq / 2. ) / cv;
  CeedScalar drdT = -rho / T;
  CeedScalar drdP = 1. / ( Rd * T);
  dU[0] = drdP * dY[0] + drdT * dY[4];
  CeedScalar de_kinetic = 0;
  for (int i=0; i<3; i++) {
    dU[1+i] = dU[0] * u[i] + rho * dY[1+i];
    de_kinetic += u[i] * dY[1+i];
  }
  dU[4] = rho * cv * dY[4] + dU[0] * cv * T // internal energy: rho * e
          + rho * de_kinetic + .5 * dU[0] * u_sq; // kinetic energy: .5 * rho * |u|^2
}

// *****************************************************************************
// Helper function for computing Tau elements (stabilization constant)
//   Model from:
//     PHASTA
//
//   Tau[i] = itau=0 which is diagonal-Shakib (3 values still but not spatial)
//
// Where NOT UPDATED YET
// *****************************************************************************
CEED_QFUNCTION_HELPER void Tau_diagPrim(CeedScalar Tau_d[3],
                                        const CeedScalar dXdx[3][3], const CeedScalar u[3],
                                        const CeedScalar cv, const NewtonianIdealGasContext newt_ctx,
                                        const CeedScalar mu, const CeedScalar dt,
                                        const CeedScalar rho) {
  // Context
  const CeedScalar Ctau_t = newt_ctx->Ctau_t;
  const CeedScalar Ctau_v = newt_ctx->Ctau_v;
  const CeedScalar Ctau_C = newt_ctx->Ctau_C;
  const CeedScalar Ctau_M = newt_ctx->Ctau_M;
  const CeedScalar Ctau_E = newt_ctx->Ctau_E;
  CeedScalar gijd[6];
  CeedScalar tau;
  CeedScalar dts;
  CeedScalar fact;

  //*INDENT-OFF*
  gijd[0] =   dXdx[0][0] * dXdx[0][0]
            + dXdx[1][0] * dXdx[1][0]
            + dXdx[2][0] * dXdx[2][0];

  gijd[1] =   dXdx[0][0] * dXdx[0][1]
            + dXdx[1][0] * dXdx[1][1]
            + dXdx[2][0] * dXdx[2][1];

  gijd[2] =   dXdx[0][1] * dXdx[0][1]
            + dXdx[1][1] * dXdx[1][1]
            + dXdx[2][1] * dXdx[2][1];

  gijd[3] =   dXdx[0][0] * dXdx[0][2]
            + dXdx[1][0] * dXdx[1][2]
            + dXdx[2][0] * dXdx[2][2];

  gijd[4] =   dXdx[0][1] * dXdx[0][2]
            + dXdx[1][1] * dXdx[1][2]
            + dXdx[2][1] * dXdx[2][2];

  gijd[5] =   dXdx[0][2] * dXdx[0][2]
            + dXdx[1][2] * dXdx[1][2]
            + dXdx[2][2] * dXdx[2][2];
  //*INDENT-ON*

  dts = Ctau_t / dt ;

  tau = rho*rho*((4. * dts * dts)
                 + u[0] * ( u[0] * gijd[0] + 2. * ( u[1] * gijd[1] + u[2] * gijd[3]))
                 + u[1] * ( u[1] * gijd[2] + 2. *   u[2] * gijd[4])
                 + u[2] *   u[2] * gijd[5])
        + Ctau_v* mu * mu *
        (gijd[0]*gijd[0] + gijd[2]*gijd[2] + gijd[5]*gijd[5] +
         + 2. * (gijd[1]*gijd[1] + gijd[3]*gijd[3] + gijd[4]*gijd[4]));

  fact=sqrt(tau);

  Tau_d[0] = Ctau_C * fact / (rho*(gijd[0] + gijd[2] + gijd[5]))*0.125;

  Tau_d[1] = Ctau_M / fact;
  Tau_d[2] = Ctau_E / ( fact * cv );

// consider putting back the way I initially had it  Ctau_E * Tau_d[1] /cv
//  to avoid a division if the compiler is smart enough to see that cv IS
// a constant that it could invert once for all elements
// but in that case energy tau is scaled by the product of Ctau_E * Ctau_M
// OR we could absorb cv into Ctau_E but this puts more burden on user to
// know how to change constants with a change of fluid or units.  Same for
// Ctau_v * mu * mu IF AND ONLY IF we don't add viscosity law =f(T)
}

// *****************************************************************************
// Helper function for computing Tau elements (stabilization constant)
//   Model from:
//     Stabilized Methods for Compressible Flows, Hughes et al 2010
//
//   Spatial criterion #2 - Tau is a 3x3 diagonal matrix
//   Tau[i] = c_tau h[i] Xi(Pe) / rho(A[i]) (no sum)
//
// Where
//   c_tau     = stabilization constant (0.5 is reported as "optimal")
//   h[i]      = 2 length(dxdX[i])
//   Pe        = Peclet number ( Pe = sqrt(u u) / dot(dXdx,u) diffusivity )
//   Xi(Pe)    = coth Pe - 1. / Pe (1. at large local Peclet number )
//   rho(A[i]) = spectral radius of the convective flux Jacobian i,
//               wave speed in direction i
// *****************************************************************************
CEED_QFUNCTION_HELPER void Tau_spatial(CeedScalar Tau_x[3],
                                       const CeedScalar dXdx[3][3], const CeedScalar u[3],
                                       /* const CeedScalar sound_speed, const CeedScalar c_tau) { */
                                       const CeedScalar sound_speed, const CeedScalar c_tau,
                                       const CeedScalar viscosity) {
  const CeedScalar a_i[3] = {fabs(u[0]) + sound_speed, fabs(u[1]) + sound_speed,fabs(u[2]) + sound_speed};
  /* const CeedScalar mag_a_visc = sqrt(a_i[0]*a_i[0] +a_i[1]*a_i[1] +a_i[2]*a_i[2]) / (2*viscosity); */
  const CeedScalar mag_a_visc = sqrt(u[0]*u[0] +u[1]*u[1] +u[2]*u[2]) /
                                (2*viscosity);
  for (int i=0; i<3; i++) {
    // length of element in direction i
    CeedScalar h = 2 / sqrt(dXdx[0][i]*dXdx[0][i] + dXdx[1][i]*dXdx[1][i] +
                            dXdx[2][i]*dXdx[2][i]);
    CeedScalar Pe = mag_a_visc*h;
    CeedScalar Xi = cosh(Pe)/sinh(Pe) - 1/Pe; //TODO Fix overflow issues at high Pe
    // fastest wave in direction i
    CeedScalar fastest_wave = fabs(u[i]) + sound_speed;
    Tau_x[i] = c_tau * h * Xi / fastest_wave;
  }
}

// *****************************************************************************
// This QFunction sets a "still" initial condition for generic Newtonian IG problems
// *****************************************************************************
CEED_QFUNCTION(ICsNewtonianIG)(void *ctx, CeedInt Q,
                               const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Context
  const SetupContext context = (SetupContext)ctx;
  const CeedScalar theta0    = context->theta0;
  const CeedScalar P0        = context->P0;
  const CeedScalar N         = context->N;
  const CeedScalar cv        = context->cv;
  const CeedScalar cp        = context->cp;
  const CeedScalar *g        = context->g;
  const CeedScalar Rd        = cp - cv;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar q[5] = {0.};

    // Setup
    // -- Coordinates
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};
    const CeedScalar e_potential = -(g[0]*x[0] + g[1]*x[1] + g[2]*x[2]);

    // -- Density
    const CeedScalar rho = P0 / (Rd*theta0);

    // Initial Conditions
    q[0] = rho;
    q[1] = 0.0;
    q[2] = 0.0;
    q[3] = 0.0;
    q[4] = rho * (cv*theta0 + e_potential);

    for (CeedInt j=0; j<5; j++)
      q0[j][i] = q[j];
  } // End of Quadrature Point Loop
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of Navier-Stokes with
//   explicit time stepping method
//
// This is 3D compressible Navier-Stokes in conservation form with state
//   variables of density, momentum density, and total energy density.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density,      Ui = rho ui
//   E   - Total Energy Density,  E  = rho (cv T + (u u)/2 + g z)
//
// Navier-Stokes Equations:
//   drho/dt + div( U )                               = 0
//   dU/dt   + div( rho (u x u) + P I3 ) + rho g khat = div( Fu )
//   dE/dt   + div( (E + P) u )                       = div( Fe )
//
// Viscous Stress:
//   Fu = mu (grad( u ) + grad( u )^T + lambda div ( u ) I3)
//
// Thermal Stress:
//   Fe = u Fu + k grad( T )
// Equation of State
//   P = (gamma - 1) (E - rho (u u) / 2 - rho g z)
//
// Stabilization:
//   Tau = diag(TauC, TauM, TauM, TauM, TauE)
//     f1 = rho  sqrt(ui uj gij)
//     gij = dXi/dX * dXi/dX
//     TauC = Cc f1 / (8 gii)
//     TauM = min( 1 , 1 / f1 )
//     TauE = TauM / (Ce cv)
//
//  SU   = Galerkin + grad(v) . ( Ai^T * Tau * (Aj q,j) )
//
// Constants:
//   lambda = - 2 / 3,  From Stokes hypothesis
//   mu              ,  Dynamic viscosity
//   k               ,  Thermal conductivity
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   g               ,  Gravity
//   gamma  = cp / cv,  Specific heat ratio
//
// We require the product of the inverse of the Jacobian (dXdx_j,k) and
// its transpose (dXdx_k,j) to properly compute integrals of the form:
// int( gradv gradu )
//
// *****************************************************************************
CEED_QFUNCTION(Newtonian)(void *ctx, CeedInt Q,
                          const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar lambda = context->lambda;
  const CeedScalar mu     = context->mu;
  const CeedScalar k      = context->k;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar *g     = context->g;
  const CeedScalar dt     = context->dt;
  const CeedScalar c_tau  = context->c_tau;
  const CeedScalar Ctau_t = context-> Ctau_t;
  const CeedScalar Ctau_v = context-> Ctau_v;
  const CeedScalar Ctau_C = context-> Ctau_C;
  const CeedScalar Ctau_M = context-> Ctau_M;
  const CeedScalar Ctau_E = context-> Ctau_E;
  const CeedScalar gamma  = cp / cv;
  const CeedScalar Rd     = cp - cv;

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
    // -- Grad in
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
    const CeedScalar x_i[3]       = {x[0][i], x[1][i], x[2][i]};
    // *INDENT-ON*
    // -- Grad-to-Grad q_data
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
    CeedScalar dudx[3][3] = {{0}};
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        for (int l=0; l<3; l++)
          dudx[j][k] += du[j][l] * dXdx[l][k];
    // -- grad_T
    const CeedScalar grad_T[3]  = {(dEdx[0]/rho - E*drhodx[0]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][0] + u[1]*dudx[1][0] + u[2]*dudx[2][0]) + g[0])/cv,
                                   (dEdx[1]/rho - E*drhodx[1]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][1] + u[1]*dudx[1][1] + u[2]*dudx[2][1]) + g[1])/cv,
                                   (dEdx[2]/rho - E*drhodx[2]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][2] + u[1]*dudx[1][2] + u[2]*dudx[2][2]) + g[2])/cv
                                  };

    // -- Fuvisc
    // ---- Symmetric 3x3 matrix
    const CeedScalar Fu[6]     =  {mu*(dudx[0][0] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[1][1] + dudx[2][2])),
                                   mu*(dudx[0][1] + dudx[1][0]), /* *NOPAD* */
                                   mu*(dudx[0][2] + dudx[2][0]), /* *NOPAD* */
                                   mu*(dudx[1][1] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[0][0] + dudx[2][2])),
                                   mu*(dudx[1][2] + dudx[2][1]), /* *NOPAD* */
                                   mu*(dudx[2][2] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[0][0] + dudx[1][1]))
                                  };
    // -- Fevisc
    const CeedScalar Fe[3]     =  {u[0]*Fu[0] + u[1]*Fu[1] + u[2]*Fu[2] + /* *NOPAD* */
                                   k*grad_T[0], /* *NOPAD* */
                                   u[0]*Fu[1] + u[1]*Fu[3] + u[2]*Fu[4] + /* *NOPAD* */
                                   k*grad_T[1], /* *NOPAD* */
                                   u[0]*Fu[2] + u[1]*Fu[4] + u[2]*Fu[5] + /* *NOPAD* */
                                   k*grad_T[2] /* *NOPAD* */
                                  };
    // Pressure
    const CeedScalar
    E_kinetic   = 0.5 * rho * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]),
    E_potential = -rho*(g[0]*x_i[0] + g[1]*x_i[1] + g[2]*x_i[2]),
    E_internal  = E - E_kinetic - E_potential,
    P           = E_internal * (gamma - 1.); // P = pressure

    // jacob_F_conv[3][5][5] = dF(convective)/dq at each direction
    CeedScalar jacob_F_conv[3][5][5] = {{{0.}}};
    computeFluxJacobian_NS(jacob_F_conv, rho, u, E, gamma, g, x_i);

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

    // Body force
    const CeedScalar body_force[5] = {0, rho *g[0], rho *g[1], rho *g[2], 0};

    // The Physics
    // Zero dv so all future terms can safely sum into it
    for (int j=0; j<5; j++)
      for (int k=0; k<3; k++)
        dv[k][j][i] = 0;

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
    // ---- Fuvisc
    const CeedInt Fuviscidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}}; // symmetric matrix indices
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        dv[k][j+1][i] -= wdetJ*(Fu[Fuviscidx[j][0]]*dXdx[k][0] +
                                Fu[Fuviscidx[j][1]]*dXdx[k][1] +
                                Fu[Fuviscidx[j][2]]*dXdx[k][2]);
    // -- Total Energy Density
    // ---- (E + P) u
    for (int j=0; j<3; j++)
      dv[j][4][i]  += wdetJ * (E + P) * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] +
                                         u[2]*dXdx[j][2]);
    // ---- Fevisc
    for (int j=0; j<3; j++)
      dv[j][4][i] -= wdetJ * (Fe[0]*dXdx[j][0] + Fe[1]*dXdx[j][1] +
                              Fe[2]*dXdx[j][2]);
    // Body Force
    for (int j=0; j<5; j++)
      v[j][i] = wdetJ * body_force[j];

    // Spatial Stabilization
    // -- Not used in favor of diagonal tau. Kept for future testing
    // const CeedScalar sound_speed = sqrt(gamma * P / rho);
    // CeedScalar Tau_x[3] = {0.};
    // Tau_spatial(Tau_x, dXdx, u, sound_speed, c_tau, mu);

    // -- Stabilization method: none, SU, or SUPG
    CeedScalar stab[5][3] = {{0.}};
    CeedScalar tau_strong_conv[5] = {0.}, tau_strong_conv_conservative[5] = {0};
    CeedScalar Tau_d[3] = {0.};
    switch (context->stabilization) {
    case STAB_NONE:        // Galerkin
      break;
    case STAB_SU:        // SU
      Tau_diagPrim(Tau_d, dXdx, u, cv, context, mu, dt, rho);
      tau_strong_conv[0] = Tau_d[0] * strong_conv[0];
      tau_strong_conv[1] = Tau_d[1] * strong_conv[1];
      tau_strong_conv[2] = Tau_d[1] * strong_conv[2];
      tau_strong_conv[3] = Tau_d[1] * strong_conv[3];
      tau_strong_conv[4] = Tau_d[2] * strong_conv[4];
      PrimitiveToConservative_fwd(rho, u, E, Rd, cv, tau_strong_conv,
                                  tau_strong_conv_conservative);
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] += jacob_F_conv[j][k][l] * tau_strong_conv_conservative[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] -= wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    case STAB_SUPG:        // SUPG is not implemented for explicit scheme
      break;
    }

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the Navier-Stokes equations (mentioned above) with
//   implicit time stepping method
//
//  SU   = Galerkin + grad(v) . ( Ai^T * Tau * (Aj q,j) )
//  SUPG = Galerkin + grad(v) . ( Ai^T * Tau * (q_dot + Aj q,j - body force) )
//                                       (diffussive terms will be added later)
//
// *****************************************************************************
CEED_QFUNCTION(IFunction_Newtonian)(void *ctx, CeedInt Q,
                                    const CeedScalar *const *in,
                                    CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*dq)[5][CEED_Q_VLA] = (const CeedScalar(*)[5][CEED_Q_VLA])in[1],
                   (*q_dot)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3],
                   (*x)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*dv)[5][CEED_Q_VLA] = (CeedScalar(*)[5][CEED_Q_VLA])out[1];
  // *INDENT-ON*
  // Context
  NewtonianIdealGasContext context = (NewtonianIdealGasContext)ctx;
  const CeedScalar lambda = context->lambda;
  const CeedScalar mu     = context->mu;
  const CeedScalar k      = context->k;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar *g     = context->g;
  const CeedScalar c_tau  = context->c_tau;
  const CeedScalar dt     = context->dt;
  const CeedScalar gamma  = cp / cv;
  const CeedScalar Rd     = cp-cv;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho        =   q[0][i];
    const CeedScalar u[3]       =  {q[1][i] / rho,
                                    q[2][i] / rho,
                                    q[3][i] / rho
                                   };
    const CeedScalar E          =   q[4][i];
    // -- Grad in
    const CeedScalar drho[3]    =  {dq[0][0][i],
                                    dq[1][0][i],
                                    dq[2][0][i]
                                   };
    // *INDENT-OFF*
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
    // *INDENT-ON*
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
    const CeedScalar x_i[3]     = {x[0][i], x[1][i], x[2][i]};
    // *INDENT-ON*
    // -- Grad-to-Grad q_data
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
    CeedScalar dudx[3][3] = {{0}};
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        for (int l=0; l<3; l++)
          dudx[j][k] += du[j][l] * dXdx[l][k];
    // -- grad_T
    const CeedScalar grad_T[3]  = {(dEdx[0]/rho - E*drhodx[0]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][0] + u[1]*dudx[1][0] + u[2]*dudx[2][0]) + g[0])/cv,
                                   (dEdx[1]/rho - E*drhodx[1]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][1] + u[1]*dudx[1][1] + u[2]*dudx[2][1]) + g[1])/cv,
                                   (dEdx[2]/rho - E*drhodx[2]/(rho*rho) - /* *NOPAD* */
                                    (u[0]*dudx[0][2] + u[1]*dudx[1][2] + u[2]*dudx[2][2]) + g[2])/cv
                                  };
    // -- Fuvisc
    // ---- Symmetric 3x3 matrix
    const CeedScalar Fu[6]     =  {mu*(dudx[0][0] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[1][1] + dudx[2][2])),
                                   mu*(dudx[0][1] + dudx[1][0]), /* *NOPAD* */
                                   mu*(dudx[0][2] + dudx[2][0]), /* *NOPAD* */
                                   mu*(dudx[1][1] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[0][0] + dudx[2][2])),
                                   mu*(dudx[1][2] + dudx[2][1]), /* *NOPAD* */
                                   mu*(dudx[2][2] * (2 + lambda) + /* *NOPAD* */
                                       lambda * (dudx[0][0] + dudx[1][1]))
                                  };
    // -- Fevisc
    const CeedScalar Fe[3]     =  {u[0]*Fu[0] + u[1]*Fu[1] + u[2]*Fu[2] + /* *NOPAD* */
                                   k*grad_T[0], /* *NOPAD* */
                                   u[0]*Fu[1] + u[1]*Fu[3] + u[2]*Fu[4] + /* *NOPAD* */
                                   k*grad_T[1], /* *NOPAD* */
                                   u[0]*Fu[2] + u[1]*Fu[4] + u[2]*Fu[5] + /* *NOPAD* */
                                   k*grad_T[2] /* *NOPAD* */
                                  };
    // Pressure
    const CeedScalar
    E_kinetic   = 0.5 * rho * (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]),
    E_potential = -rho*(g[0]*x_i[0] + g[1]*x_i[1] + g[2]*x_i[2]),
    E_internal  = E - E_kinetic - E_potential,
    P           = E_internal * (gamma - 1.); // P = pressure

    // jacob_F_conv[3][5][5] = dF(convective)/dq at each direction
    CeedScalar jacob_F_conv[3][5][5] = {{{0.}}};
    computeFluxJacobian_NS(jacob_F_conv, rho, u, E, gamma, g, x_i);

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

    // Body force
    const CeedScalar body_force[5] = {0, rho *g[0], rho *g[1], rho *g[2], 0};

    // Strong residual
    CeedScalar strong_res[5];
    for (int j=0; j<5; j++)
      strong_res[j] = q_dot[j][i] + strong_conv[j] - body_force[j];

    // The Physics
    //-----mass matrix
    for (int j=0; j<5; j++)
      v[j][i] = wdetJ*q_dot[j][i];

    // Zero dv so all future terms can safely sum into it
    for (int j=0; j<5; j++)
      for (int k=0; k<3; k++)
        dv[k][j][i] = 0;

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
    // ---- Fuvisc
    const CeedInt Fuviscidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}}; // symmetric matrix indices
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        dv[k][j+1][i] += wdetJ*(Fu[Fuviscidx[j][0]]*dXdx[k][0] +
                                Fu[Fuviscidx[j][1]]*dXdx[k][1] +
                                Fu[Fuviscidx[j][2]]*dXdx[k][2]);
    // -- Total Energy Density
    // ---- (E + P) u
    for (int j=0; j<3; j++)
      dv[j][4][i]  -= wdetJ * (E + P) * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] +
                                         u[2]*dXdx[j][2]);
    // ---- Fevisc
    for (int j=0; j<3; j++)
      dv[j][4][i] += wdetJ * (Fe[0]*dXdx[j][0] + Fe[1]*dXdx[j][1] +
                              Fe[2]*dXdx[j][2]);
    // Body Force
    for (int j=0; j<5; j++)
      v[j][i] -= wdetJ*body_force[j];

    // Spatial Stabilization
    // -- Not used in favor of diagonal tau. Kept for future testing
    // const CeedScalar sound_speed = sqrt(gamma * P / rho);
    // CeedScalar Tau_x[3] = {0.};
    // Tau_spatial(Tau_x, dXdx, u, sound_speed, c_tau, mu);

    // -- Stabilization method: none, SU, or SUPG
    CeedScalar stab[5][3] = {{0.}};
    CeedScalar tau_strong_res[5] = {0.}, tau_strong_res_conservative[5] = {0};
    CeedScalar tau_strong_conv[5] = {0.}, tau_strong_conv_conservative[5] = {0};
    CeedScalar Tau_d[3] = {0.};
    switch (context->stabilization) {
    case STAB_NONE:        // Galerkin
      break;
    case STAB_SU:        // SU
      Tau_diagPrim(Tau_d, dXdx, u, cv, context, mu, dt, rho);
      tau_strong_conv[0] = Tau_d[0] * strong_conv[0];
      tau_strong_conv[1] = Tau_d[1] * strong_conv[1];
      tau_strong_conv[2] = Tau_d[1] * strong_conv[2];
      tau_strong_conv[3] = Tau_d[1] * strong_conv[3];
      tau_strong_conv[4] = Tau_d[2] * strong_conv[4];
      PrimitiveToConservative_fwd(rho, u, E, Rd, cv, tau_strong_conv,
                                  tau_strong_conv_conservative);
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] += jacob_F_conv[j][k][l] * tau_strong_conv_conservative[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] += wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    case STAB_SUPG:        // SUPG
      Tau_diagPrim(Tau_d, dXdx, u, cv, context, mu, dt, rho);
      tau_strong_res[0] = Tau_d[0] * strong_res[0];
      tau_strong_res[1] = Tau_d[1] * strong_res[1];
      tau_strong_res[2] = Tau_d[1] * strong_res[2];
      tau_strong_res[3] = Tau_d[1] * strong_res[3];
      tau_strong_res[4] = Tau_d[2] * strong_res[4];
// Alternate route (useful later with primitive variable code)
// this function was verified against PHASTA for as IC that was as close as possible
//    computeFluxJacobian_NSp(jacob_F_conv_p, rho, u, E, Rd, cv);
// it has also been verified to compute a correct through the following
//   stab[k][j] += jacob_F_conv_p[j][k][l] * tau_strong_res[l] // flux Jacobian wrt primitive
// applied in the triple loop below
//  However, it is more flops than using the existing Jacobian wrt q after q_{,Y} viz
      PrimitiveToConservative_fwd(rho, u, E, Rd, cv, tau_strong_res,
                                  tau_strong_res_conservative);
      for (int j=0; j<3; j++)
        for (int k=0; k<5; k++)
          for (int l=0; l<5; l++)
            stab[k][j] += jacob_F_conv[j][k][l] * tau_strong_res_conservative[l];

      for (int j=0; j<5; j++)
        for (int k=0; k<3; k++)
          dv[k][j][i] += wdetJ*(stab[j][0] * dXdx[k][0] +
                                stab[j][1] * dXdx[k][1] +
                                stab[j][2] * dXdx[k][2]);
      break;
    }

  } // End Quadrature Point Loop

  // Return
  return 0;
}
// *****************************************************************************
#endif // newtonian_h
