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
CEED_QFUNCTION_HELPER int ExactSolution(CeedScalar q[], CeedScalar *time,
                                        CeedScalar *X, CeedScalar *Y, CeedScalar *Z) {
  // -- Time
  CeedScalar t = time[0];
  // -- Coordinates
  CeedScalar x = X[0];
  CeedScalar y = Y[0];
  CeedScalar z = Z[0];

  // Exact solutions
  q[0] = 10.*(t*t+t+1.) + (1.*x) + (6. *y) + (11.*z);
  q[1] = 20.*(t*t+t+1.) + (2.*x) + (7. *y) + (12.*z);
  q[2] = 30.*(t*t+t+1.) + (3.*x) + (8. *y) + (13.*z);
  q[3] = 40.*(t*t+t+1.) + (4.*x) + (9. *y) + (14.*z);
  q[4] = 50.*(t*t+t+1.) + (5.*x) + (10.*y) + (15.*z);

  //q[0] = exp(6*t)*exp(2*x)*sin(3*y)*cos(4*z)/1e5;
  //q[1] = exp(7*t)*sin(3*x)*cos(4*y)*cos(5*z)/1e5;
  //q[2] = exp(7*t)*cos(4*x)*sin(5*y)*sin(3*z)/1e5;
  //q[3] = exp(7*t)*sin(5*x)*sin(3*y)*cos(4*z)/1e5;
  //q[4] = exp(8*t)*sin(4*x)*cos(5*y)*exp(6*z)/1e5;

  //q[0] = 2*(t*t + t + 1.) + (1*x*x) + (6 *y*y) +  (20*z*z);
  //q[1] = 3*(t*t + t + 1.) + (2*x*x) + (7 *y*y) +  (30*z*z);
  //q[2] = 4*(t*t + t + 1.) + (3*x*x) + (8 *y*y) +  (40*z*z);
  //q[3] = 5*(t*t + t + 1.) + (4*x*x) + (9 *y*y) +  (50*z*z);
  //q[4] = 6*(t*t + t + 1.) + (5*x*x) + (10*y*y) +  (60*z*z);

  return 0;
}
// *****************************************************************************

// *****************************************************************************
// Enzyme-AD
// *****************************************************************************
// -- Enzyme functions and variables
int  __enzyme_autodiff(void *, ...);
int enzyme_const;

// -----------------------------------------------------------------------------
// -- Compute q_dot
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER void computeQdot(CeedScalar *q_dot, CeedScalar *t,
                                       CeedScalar *x, CeedScalar *y, CeedScalar *z) {
  CeedScalar q[5];
  for (CeedInt i=0; i<5; i++) {
    CeedScalar q_[5] = {0.}; q_[i] = 1.;
    __enzyme_autodiff((void *)ExactSolution,
                      q, q_,
                      t, &q_dot[i],
                      enzyme_const, x,
                      enzyme_const, y,
                      enzyme_const, z);
  }
}

// -----------------------------------------------------------------------------
// -- Compute grad_q
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER void computeGrad_q(CeedScalar grad_q[3][5], CeedScalar *t,
    CeedScalar *x, CeedScalar *y, CeedScalar *z) {
  CeedScalar q[5];
  // Derivative wrt x
  for (CeedInt i=0; i<5; i++) {
    CeedScalar q_[5] = {0.}; q_[i] = 1.;
    __enzyme_autodiff((void *)ExactSolution,
                      q, q_,
                      enzyme_const, t,
                      x, &grad_q[0][i],
                      enzyme_const, y,
                      enzyme_const, z);
  }
  // Derivative wrt y
  for (CeedInt i=0; i<5; i++) {
    CeedScalar q_[5] = {0.}; q_[i] = 1.;
    __enzyme_autodiff((void *)ExactSolution,
                      q, q_,
                      enzyme_const, t,
                      enzyme_const, x,
                      y, &grad_q[1][i],
                      enzyme_const, z);
  }
  // Derivative wrt z
  for (CeedInt i=0; i<5; i++) {
    CeedScalar q_[5] = {0.}; q_[i] = 1.;
    __enzyme_autodiff((void *)ExactSolution,
                      q, q_,
                      enzyme_const, t,
                      enzyme_const, x,
                      enzyme_const, y,
                      z, &grad_q[2][i]);
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
  CeedScalar x[1] = {X[0]}, y[1] = {X[1]}, z[1] = {X[2]};
  ExactSolution(q, t, x, y, z);
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
// Forcing term for Method of Manufactured Solutions
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
  //const CeedScalar lambda = context->lambda;
  //const CeedScalar mu     = context->mu;
  //const CeedScalar k      = context->k;
  //const CeedScalar cv     = context->cv;
  //const CeedScalar cp     = context->cp;
  //const CeedScalar g      = context->g;
  //const CeedScalar Rd     = context->Rd;
  //const CeedScalar gamma  = cp / cv;
  const CeedScalar t      = context->curr_time;
  const bool implicit     = context->implicit;
  CeedScalar time[1] = {t};

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    printf("\n-------------------------------------------------------------------------");
    printf("\nqpt  = %d \ntime = %f", i, t);
    printf("\n-------------------------------------------------------------------------");
    const CeedScalar wdetJ = (implicit ? -1. : 1.) * q_data[0][i];
    CeedScalar x[1] = {X[0][i]}, y[1] = {X[1][i]}, z[1] = {X[2][i]};

    // Zero force so all future terms can safely sum into it
    for (CeedInt j=0; j<5; j++) force[j][i] = 0.;

    // -------------------------------------------------------------------------
    // q_dot
    // -------------------------------------------------------------------------
    CeedScalar q_dot[5] = {0.};
    computeQdot(q_dot, time, x, y, z);

    // Print output
    printf("\nq_dot:\n");
    for (int j=0; j<5; j++) printf("%f\t", q_dot[j]);
    printf("\n");

    // Add Qdot to the forcing term
    for (CeedInt j=0; j<5; j++) force[j][i] += wdetJ*q_dot[j];

    // -------------------------------------------------------------------------
    // grad_q
    // -------------------------------------------------------------------------
    CeedScalar grad_q[3][5] = {{0.}};
    computeGrad_q(grad_q, time, x, y, z);

    // Print output
    for (int i=0; i<3; i++) {
      printf("\nDerivative in direction %d:\n", i);
      for (int j=0; j<5; j++) printf("%f\t", grad_q[i][j]);
      printf("\n");
    }
  } // End Quadrature Point Loop

  return 0; // Return
}
// *****************************************************************************

#endif // density_current_mms_ad_h
