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
/// Density current initial condition and operator for Navier-Stokes example using PETSc

// Model from:
//   Semi-Implicit Formulations of the Navier-Stokes Equations: Application to
//   Nonhydrostatic Atmospheric Modeling, Giraldo, Restelli, and Lauter (2010).

#ifndef density_current_mms_ad_h
#define density_current_mms_ad_h

#include <math.h>

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

#ifndef dc_context_struct
#define dc_context_struct
typedef struct DCContext_ *DCContext;
struct DCContext_ {
  CeedScalar lambda;
  CeedScalar mu;
  CeedScalar k;
  CeedScalar cv;
  CeedScalar cp;
  CeedScalar g;
  CeedScalar Rd;
  CeedScalar curr_time;
  bool implicit;
  int stabilization; // See StabilizationType: 0=none, 1=SU, 2=SUPG
};
#endif

// *****************************************************************************
// True Solutions for the Method of Manufactured Solutions
// *****************************************************************************
CEED_QFUNCTION_HELPER int ExactSolution(CeedScalar* __restrict__ q, CeedScalar time,
                                        CeedScalar* __restrict__ X) {
  // -- Time
  double t = time;
  // -- Coordinates
  double x = X[0];
  double y = X[1];
  double z = X[2];

  // Exact solutions
  q[0] = cos(2*x)*sin(3*y)*cos(4*z) + .01; // exp(6*t)*
  q[1] = sin(3*x)*cos(4*y)*cos(5*z) + .01; // exp(7*t)*
  q[2] = cos(4*x)*sin(5*y)*sin(3*z) + .01; // exp(7*t)*
  q[3] = sin(5*x)*sin(3*y)*cos(4*z) + .01; // exp(7*t)*
  q[4] = sin(4*x)*cos(5*y)*sin(6*z) + .01; // exp(8*t)*

  return 0;
}
// *****************************************************************************

// *****************************************************************************
// Enzyme-AD
// *****************************************************************************
// -- Enzyme functions and variables
void __enzyme_autodiff(void *, ...);
double __enzyme_autodiffDouble(void *, ...);
int enzyme_const;
int enzyme_dupnoneed;

// -- q_diff
CEED_QFUNCTION_HELPER void q_diff(double q[5], double q_dot[5], double dq[5][3],
                                  double *t, double x[]) {
  for (int i=0; i<5; i++) {
    double q_[5] = {0.}; q_[i] = 1.;
    q_dot[i] = __enzyme_autodiffDouble((void *)ExactSolution,
                      q, q_,
                      *t,
                      x, &dq[i]);
  }
}

void __enzyme_fwddiff(void *, ...);

CEED_QFUNCTION_HELPER void grad_u_wrt_X(double du_dx[5][3], double X[3], double t) {
  for (int i=0; i<3; i++) {
    double dX[3] = {0.}; dX[i] = 1.;
    double dQ[5];
     __enzyme_fwddiff((void *)ExactSolution,
                      enzyme_dupnoneed, (void*)0, dQ,
                      X, dX, enzyme_const, t);
    // If transpose desired, can swap
    for (int j=0; j<5; j++) {
      flux_x[j][i] = dQ[j];
    }
  }
}

/*
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
//
// Equation of State:
//   P = (gamma - 1) (E - rho (u u) / 2 - rho g z)
*/


// u (x) u ?=
// [u1 u1   u1 u2   u1 u3 ]
// [u2 u1   u2 u2   u2 u3 ]
// [u3 u1   u3 u2   u3 u3 ]

// Compute "flux" = rho (u x u) + P I3 => 3x3 Matrix
CEED_QFUNCTION_HELPER void flux(double flux_output[3][3], double q[5], double x[3], double cv, double cp, double g) {
  double rho = q[0];
  double u[3] = {q[1]/rho, q[2]/rho, q[3]/rho};
  double E = q[4];

  double gamma = cp/cv;
  double kinetic_energy = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) / 2.;
  double P = (E - kinetic_energy * rho - rho*g*x[2]) * (gamma - 1.);

  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      flux_output[i][j] = rho*u[i]*u[j] + ((i == j) ? P : 0);
    }
  }
}

// d/dx Flux => 3x3 Matrix,    \grad Flux => 3x(3x3)


// -- dFlux[0]/dx
CEED_QFUNCTION_HELPER void Fi0(double f0[5], double *t, double x[],
                               double lambda, double mu, double k, double cv, double cp, double g) {
  // Compute state variables and the derivatives
  double q[5];
  double q_dot[5] = {0.};
  double dq[5][3] = {{0.}};
  q_diff(q, q_dot, dq, t, x);

  // -- q
  double rho = q[0];
  double u[3] = {q[1]/rho, q[2]/rho, q[3]/rho};
  double E = q[4];
  // -- dq
  double drho[3]   = {dq[0][0], dq[0][1], dq[0][2]};
  // *INDENT-OFF*
  double dU[3][3] = {{dq[1][0], dq[1][1], dq[1][2]},
                     {dq[2][0], dq[2][1], dq[2][2]},
                     {dq[3][0], dq[3][1], dq[3][2]}
                    };
  // *INDENT-ON*
  double dE[3] =     {dq[4][0], dq[4][1], dq[4][2]};
  double du[3][3] = {{0.}};
  for (int j=0; j<3; j++)
    for (int k=0; k<3; k++)
      du[j][k] = (dU[j][k] - drho[k]*u[j]) / rho;

  // Density
  // -- Advective flux
  double F_adv_density[3] = {rho *u[0], rho *u[1], rho *u[2]};

  // -- No diffusive flux

  // Momentum
  // -- Advective flux
  double gamma = cp/cv;
  double kinetic_energy = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) / 2.;
  double P = (E - kinetic_energy * rho - rho*g*x[2]) * (gamma - 1.);
  // *INDENT-OFF*
  double F_adv_momentum[3][3] = {{rho *u[0] *u[0] + P, rho *u[0] *u[1],     rho *u[0] *u[2]},
                                 {rho *u[1] *u[0],     rho *u[1] *u[1] + P, rho *u[1] *u[2]},
                                 {rho *u[2] *u[0],     rho *u[2] *u[1],     rho *u[2] *u[02] + P}
                               };
  // *INDENT-ON*
  // -- Diffusive Flux
  double Fu[6] = {mu *(du[0][0] * (2 + lambda) + lambda * (du[1][1] + du[2][2])),
                  mu *(du[0][1] + du[1][0]),
                  mu *(du[0][2] + du[2][0]),
                  mu *(du[1][1] * (2 + lambda) + lambda * (du[0][0] + du[2][2])),
                  mu *(du[1][2] + du[2][1]),
                  mu *(du[2][2] * (2 + lambda) + lambda * (du[0][0] + du[1][1]))
                 };

  const int Fuviscidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}}; // symmetric matrix indices
  double F_dif_momentum[3][3];
  for (int j=0; j<3; j++)
    for (int k=0; k<3; k++)
      F_dif_momentum[j][k] = Fu[Fuviscidx[j][k]];

  // Total Energy
  // -- Advective flux
  double F_adv_energy[3] = {(E + P) *u[0], (E + P) *u[1], (E + P) *u[2]};

  // -- Diffusive Flux
  double dT[3] = {(dE[0]/rho - E *drho[0]/(rho*rho) - (u[0]*du[0][0] + u[1]*du[1][0] + u[2]*du[2][0]))    /cv,
                  (dE[1]/rho - E *drho[1]/(rho*rho) - (u[0]*du[0][1] + u[1]*du[1][1] + u[2]*du[2][1]))    /cv,
                  (dE[2]/rho - E *drho[2]/(rho*rho) - (u[0]*du[0][2] + u[1]*du[1][2] + u[2]*du[2][2]) - g)/cv
                 };
  double F_dif_energy[3] = {u[0] *Fu[0] + u[1] *Fu[1] + u[2] *Fu[2] + k *dT[0],
                            u[0] *Fu[1] + u[1] *Fu[3] + u[2] *Fu[4] + k *dT[1],
                            u[0] *Fu[2] + u[1] *Fu[4] + u[2] *Fu[5] + k *dT[2]
                           };

  // Populate Flux
  // -- Zero f0
  for (int j=0; j<5; j++) f0[j] = 0.;

  // -- Density
  f0[0] += F_adv_density[0];

  // -- Momentum
  for (int j=0; j<3; j++) {
    f0[j+1] += F_adv_momentum[j][0];
    f0[j+1] -= F_dif_momentum[j][0];
  }

  // -- Energy
  f0[4] += F_adv_energy[0];
  f0[4] -= F_dif_energy[0];

}

CEED_QFUNCTION_HELPER void dFi0_dx(double df0_dx[5], double *t, double x[],
                                   double lambda, double mu, double k, double cv, double cp, double g) {
  double f[5];
  for (int i=0; i<5; i++) {
    double f_[5] = {0.}; f_[i] = 1.;
    double df[3] = {0.};
    __enzyme_autodiff((void *)Fi0,
                      f, f_,
                      enzyme_const, t,
                      x, df,
                      enzyme_const, lambda,
                      enzyme_const, mu,
                      enzyme_const, k,
                      enzyme_const, cv,
                      enzyme_const, cp,
                      enzyme_const, g);
    df0_dx[i] = df[0];
  }
}

// -- dFlux[1]/dy
CEED_QFUNCTION_HELPER void Fi1(double f1[5], double *t, double x[],
                               double lambda, double mu, double k, double cv, double cp, double g) {
  // Compute state variables and the derivatives
  double q[5];
  double q_dot[5] = {0.};
  double dq[5][3] = {{0.}};
  q_diff(q, q_dot, dq, t, x);

  // -- q
  double rho = q[0];
  double u[3] = {q[1]/rho, q[2]/rho, q[3]/rho};
  double E = q[4];
  // -- dq
  double drho[3]   = {dq[0][0], dq[0][1], dq[0][2]};
  double dU[3][3] = {{dq[1][0], dq[1][1], dq[1][2]},
    {dq[2][0], dq[2][1], dq[2][2]},
    {dq[3][0], dq[3][1], dq[3][2]}
  };
  double dE[3] =     {dq[4][0], dq[4][1], dq[4][2]};
  double du[3][3] = {{0.}};
  for (int j=0; j<3; j++)
    for (int k=0; k<3; k++)
      du[j][k] = (dU[j][k] - drho[k]*u[j]) / rho;

  // Density
  // -- Advective flux
  double F_adv_density[3] = {rho *u[0], rho *u[1], rho *u[2]};

  // -- No diffusive flux

  // Momentum
  // -- Advective flux
  double gamma = cp/cv;
  double kinetic_energy = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) / 2.;
  double P = (E - kinetic_energy * rho - rho*g*x[2]) * (gamma - 1.);
  // *INDENT-OFF*
  double F_adv_momentum[3][3] = {{rho *u[0] *u[0] + P, rho *u[0] *u[1],     rho *u[0] *u[2]},
                                 {rho *u[1] *u[0],     rho *u[1] *u[1] + P, rho *u[1] *u[2]},
                                 {rho *u[2] *u[0],     rho *u[2] *u[1],     rho *u[2] *u[02] + P}
                               };
  // *INDENT-ON*
  // -- Diffusive Flux
  double Fu[6] = {mu *(du[0][0] * (2 + lambda) + lambda * (du[1][1] + du[2][2])),
                  mu *(du[0][1] + du[1][0]),
                  mu *(du[0][2] + du[2][0]),
                  mu *(du[1][1] * (2 + lambda) + lambda * (du[0][0] + du[2][2])),
                  mu *(du[1][2] + du[2][1]),
                  mu *(du[2][2] * (2 + lambda) + lambda * (du[0][0] + du[1][1]))
                 };

  const int Fuviscidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}}; // symmetric matrix indices
  double F_dif_momentum[3][3];
  for (int j=0; j<3; j++)
    for (int k=0; k<3; k++)
      F_dif_momentum[j][k] = Fu[Fuviscidx[j][k]];

  // Total Energy
  // -- Advective flux
  double F_adv_energy[3] = {(E + P) *u[0], (E + P) *u[1], (E + P) *u[2]};

  // -- Diffusive Flux
  double dT[3] = {(dE[0]/rho - E *drho[0]/(rho*rho) - (u[0]*du[0][0] + u[1]*du[1][0] + u[2]*du[2][0]))    /cv,
                  (dE[1]/rho - E *drho[1]/(rho*rho) - (u[0]*du[0][1] + u[1]*du[1][1] + u[2]*du[2][1]))    /cv,
                  (dE[2]/rho - E *drho[2]/(rho*rho) - (u[0]*du[0][2] + u[1]*du[1][2] + u[2]*du[2][2]) - g)/cv
                 };
  double F_dif_energy[3] = {u[0] *Fu[0] + u[1] *Fu[1] + u[2] *Fu[2] + k *dT[0],
                            u[0] *Fu[1] + u[1] *Fu[3] + u[2] *Fu[4] + k *dT[1],
                            u[0] *Fu[2] + u[1] *Fu[4] + u[2] *Fu[5] + k *dT[2]
                           };

  // Populate Flux
  // -- Zero f0
  for (int j=0; j<5; j++) f1[j] = 0.;

  // -- Density
  f1[0] += F_adv_density[1];

  // -- Momentum
  for (int j=0; j<3; j++) {
    f1[j+1] += F_adv_momentum[j][1];
    f1[j+1] -= F_dif_momentum[j][1];
  }

  // -- Energy
  f1[4] += F_adv_energy[1];
  f1[4] -= F_dif_energy[1];
}

CEED_QFUNCTION_HELPER void dFi1_dy(double df1_dy[5], double *t, double x[],
                                   double lambda, double mu, double k, double cv, double cp, double g) {
  double f[5];
  for (int i=0; i<5; i++) {
    double f_[5] = {0.}; f_[i] = 1.;
    double df[3] = {0.};
    __enzyme_autodiff((void *)Fi1,
                      f, f_,
                      enzyme_const, t,
                      x, df,
                      enzyme_const, lambda,
                      enzyme_const, mu,
                      enzyme_const, k,
                      enzyme_const, cv,
                      enzyme_const, cp,
                      enzyme_const, g);
    df1_dy[i] = df[1];
  }
}

// -- dFlux[2]/dz
CEED_QFUNCTION_HELPER void Fi2(double f2[5], double *t, double x[],
                               double lambda, double mu, double k, double cv, double cp, double g) {
  // Compute state variables and the derivatives
  double q[5];
  double q_dot[5] = {0.};
  double dq[5][3] = {{0.}};
  q_diff(q, q_dot, dq, t, x);

  // -- q
  double rho = q[0];
  double u[3] = {q[1]/rho, q[2]/rho, q[3]/rho};
  double E = q[4];
  // -- dq
  double drho[3]   = {dq[0][0], dq[0][1], dq[0][2]};
  // *INDENT-OFF*
  double dU[3][3] = {{dq[1][0], dq[1][1], dq[1][2]},
                     {dq[2][0], dq[2][1], dq[2][2]},
                     {dq[3][0], dq[3][1], dq[3][2]}
                   };
  // *INDENT-ON*
  double dE[3] =     {dq[4][0], dq[4][1], dq[4][2]};
  double du[3][3] = {{0.}};
  for (int j=0; j<3; j++)
    for (int k=0; k<3; k++)
      du[j][k] = (dU[j][k] - drho[k]*u[j]) / rho;

  // Density
  // -- Advective flux
  double F_adv_density[3] = {rho *u[0], rho *u[1], rho *u[2]};

  // -- No diffusive flux

  // Momentum
  // -- Advective flux
  double gamma = cp/cv;
  double kinetic_energy = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) / 2.;
  double P = (E - kinetic_energy * rho - rho*g*x[2]) * (gamma - 1.);
  // *INDENT-OFF*
  double F_adv_momentum[3][3] = {{rho *u[0] *u[0] + P, rho *u[0] *u[1],     rho *u[0] *u[2]},
                                 {rho *u[1] *u[0],     rho *u[1] *u[1] + P, rho *u[1] *u[2]},
                                 {rho *u[2] *u[0],     rho *u[2] *u[1],     rho *u[2] *u[02] + P}
                               };
  // *INDENT-ON*
  // -- Diffusive Flux
  double Fu[6] = {mu *(du[0][0] * (2 + lambda) + lambda * (du[1][1] + du[2][2])),
                  mu *(du[0][1] + du[1][0]),
                  mu *(du[0][2] + du[2][0]),
                  mu *(du[1][1] * (2 + lambda) + lambda * (du[0][0] + du[2][2])),
                  mu *(du[1][2] + du[2][1]),
                  mu *(du[2][2] * (2 + lambda) + lambda * (du[0][0] + du[1][1]))
                 };

  const int Fuviscidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}}; // symmetric matrix indices
  double F_dif_momentum[3][3];
  for (int j=0; j<3; j++)
    for (int k=0; k<3; k++)
      F_dif_momentum[j][k] = Fu[Fuviscidx[j][k]];

  // Total Energy
  // -- Advective flux
  double F_adv_energy[3] = {(E + P) *u[0], (E + P) *u[1], (E + P) *u[2]};

  // -- Diffusive Flux
  double dT[3] = {(dE[0]/rho - E *drho[0]/(rho*rho) - (u[0]*du[0][0] + u[1]*du[1][0] + u[2]*du[2][0]))    /cv,
                  (dE[1]/rho - E *drho[1]/(rho*rho) - (u[0]*du[0][1] + u[1]*du[1][1] + u[2]*du[2][1]))    /cv,
                  (dE[2]/rho - E *drho[2]/(rho*rho) - (u[0]*du[0][2] + u[1]*du[1][2] + u[2]*du[2][2]) - g)/cv
                 };
  double F_dif_energy[3] = {u[0] *Fu[0] + u[1] *Fu[1] + u[2] *Fu[2] + k *dT[0],
                            u[0] *Fu[1] + u[1] *Fu[3] + u[2] *Fu[4] + k *dT[1],
                            u[0] *Fu[2] + u[1] *Fu[4] + u[2] *Fu[5] + k *dT[2]
                           };

  // Populate Flux
  // -- Zero f0
  for (int j=0; j<5; j++) f2[j] = 0.;

  // -- Density
  f2[0] += F_adv_density[2];

  // -- Momentum
  for (int j=0; j<3; j++) {
    f2[j+1] += F_adv_momentum[j][2];
    f2[j+1] -= F_dif_momentum[j][2];
  }

  // -- Energy
  f2[4] += F_adv_energy[2];
  f2[4] -= F_dif_energy[2];
}

CEED_QFUNCTION_HELPER void dFi2_dz(double df2_dz[5], double *t, double x[],
                                   double lambda, double mu, double k, double cv, double cp, double g) {
  double f[5];
  for (int i=0; i<5; i++) {
    double f_[5] = {0.}; f_[i] = 1.;
    double df[3] = {0.};
    __enzyme_autodiff((void *)Fi2,
                      f, f_,
                      enzyme_const, t,
                      x, df,
                      enzyme_const, lambda,
                      enzyme_const, mu,
                      enzyme_const, k,
                      enzyme_const, cv,
                      enzyme_const, cp,
                      enzyme_const, g);
    df2_dz[i] = df[2];
  }
}

// *****************************************************************************

// *****************************************************************************
// This helper function provides support for the exact, time-dependent solution
//   and IC formulation for density current
// *****************************************************************************
CEED_QFUNCTION_HELPER int Exact_DC_MMS(CeedInt dim, CeedScalar time,
                                       const CeedScalar X[], CeedInt Nf, CeedScalar q[],
                                       void *ctx) {
  CeedScalar t[1] = {time};
  CeedScalar x[] = {X[0], X[1], X[2]};
  ExactSolution(q, t, x);
  return 0;
}
// *****************************************************************************
// This QFunction sets the initial conditions for density current
// *****************************************************************************
CEED_QFUNCTION(ICsDC_MMS)(void *ctx, CeedInt Q,
                          const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Context
  const DCContext context = (DCContext)ctx;
  const CeedScalar time   = context->curr_time;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar q[5];
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};
    Exact_DC_MMS(3, time, x, 5, q, ctx);
    for (CeedInt j=0; j<5; j++) q0[j][i] = q[j];
  } // End of Quadrature Point Loop

  return 0;
}

// *****************************************************************************
// Forcing terms for Method of Manufactured Solutions
// *****************************************************************************
CEED_QFUNCTION(DC_MMS)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                       CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // Outputs
  CeedScalar (*force)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  DCContext context = (DCContext)ctx;
  const CeedScalar lambda = context->lambda;
  const CeedScalar mu     = context->mu;
  const CeedScalar k      = context->k;
  const CeedScalar cv     = context->cv;
  const CeedScalar cp     = context->cp;
  const CeedScalar g      = context->g;
  const CeedScalar t      = context->curr_time;
  const bool implicit     = context->implicit;
  CeedScalar time[1] = {t}; // TODO: This needs a fix

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar wdetJ = (implicit ? -1. : 1.) * q_data[0][i];
    CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};

    // Zero force so all future terms can safely sum into it
    for (CeedInt j=0; j<5; j++) force[j][i] = 0.;

    // -------------------------------------------------------------------------
    // q_diff
    // -------------------------------------------------------------------------
    double q[5];
    double q_dot[5] = {0.};
    double dq[5][3] = {{0.}};
    q_diff(q, q_dot, dq, time, x);

    // Add Qdot to the forcing terms
    for (CeedInt j=0; j<5; j++) force[j][i] += wdetJ*q_dot[j];

    // -------------------------------------------------------------------------
    // div(flux) = div(flux_advective - flux_diffusive)
    // -------------------------------------------------------------------------
    // -- Compute df
    double df[3][5] = {{0.}};
    dFi0_dx(df[0], time, x, lambda, mu, k, cv, cp, g);
    dFi1_dy(df[1], time, x, lambda, mu, k, cv, cp, g);
    dFi2_dz(df[2], time, x, lambda, mu, k, cv, cp, g);

    // -- Compute div(flux)
    double div_f[5] = {0.};
    for (int j=0; j<5; j++) for (int k=0; k<3; k++) div_f[j] += df[k][j];

    // Add div(flux) to the forcing terms
    for (CeedInt j=0; j<5; j++) force[j][i] += wdetJ*div_f[j];

    // -------------------------------------------------------------------------
    // Body force
    // -------------------------------------------------------------------------
    double rho = q[0];

    // Add body force to the forcing terms
    force[3][i] += wdetJ*rho*g;

  } // End Quadrature Point Loop

  return 0; // Return
}
// *****************************************************************************

#endif // density_current_mms_ad_h
