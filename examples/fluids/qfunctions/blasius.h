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
/// Operator for Navier-Stokes example using PETSc


#ifndef blasius_h
#define blasius_h

#include <math.h>
#include <ceed.h>
#include "../navierstokes.h"

#ifndef blasius_context_struct
#define blasius_context_struct
typedef struct BlasiusContext_ *BlasiusContext;
struct BlasiusContext_ {
  bool       implicit;  // !< Using implicit timesteping or not
  CeedScalar delta0;    // !< Boundary layer height at inflow
  CeedScalar Uinf;      // !< Velocity at boundary layer edge
  CeedScalar P0;        // !< Pressure at outflow
  CeedScalar theta0;    // !< Temperature at inflow
  struct NewtonianIdealGasContext_ newtonian_ctx;
};
#endif

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

void CEED_QFUNCTION_HELPER(BlasiusSolution)(const CeedScalar y,
    const CeedScalar Uinf, const CeedScalar x0, const CeedScalar x,
    const CeedScalar rho, CeedScalar *u, CeedScalar *v,
    const NewtonianIdealGasContext newt_ctx) {

  CeedInt nprofs = 26;
  CeedScalar eta_table[] = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 8.2, 8.7, 8.8, 9, 10, 11, 12, 12.43};
  // *INDENT-OFF*
  CeedScalar f_table[] = { 0,                  0.00166028097748329, 0.00664099782185387, 0.0149414623187717, 0.0265598832055768,
                           0.0414928195150539, 0.0597346375181186,  0.0812769754437425,  0.106108220767729,  0.134213005526786,
                           0.165571725783800,  0.650024370215956,   1.39680823153785,    2.30574641937620,   3.28327366522910,
                           4.27962092307682,   5.27923881151476,    6.27921343179810,    6.47921288713369,   6.97921243151176,
                           7.07921240368407,   7.27921237111197,    8.27921234339988,    9.27921234294946,   10.2792123429452,   10.7092123429449 };
  CeedScalar fp_table[] = { 0,                 0.0332054966058561, 0.0664077801596799, 0.0995985889897647, 0.132764155997273,
                            0.165885252325028, 0.198937252436665,  0.231890235983639,  0.264709138163229,  0.297353957812383,
                            0.329780030672017, 0.629765736670949,  0.846044443888272,  0.955518229353090,  0.991541900689297,
                            0.998972872289725, 0.999921604109742,  0.999996274564183,  0.999998087480233,  0.999999668030006,
                            0.999999769481724, 0.999999890448371,  0.999999998015206,  0.999999999977930,  0.999999999998648, 1 };
  CeedScalar fpp_table[] = { 0.332057336270228,   0.332048145748033,    0.331983834255578,     0.331809346697686,      0.331469843619160,
                             0.330910954899200,   0.330079127676020,    0.328922067860142,     0.327389270302448,      0.325432629177788,
                             0.323007116916611,   0.266751545690649,    0.161360318755386,     0.0642341216112545,     0.0159067979373118,
                             0.00240204010581148, 0.000220169039772643, 0.0000122408522222333, 0.00000646792883303279, 0.00000120272733477146,
                             8.46312294375786e-7, 4.12807423557125e-7,  8.44248043699535e-9,   1.04517612148937e-10,   5.63589958345985e-12, 0 };
  // *INDENT-ON*

  CeedScalar nu = newt_ctx->mu / rho;
  CeedScalar eta = y*sqrt(Uinf/(nu*(x0+x)));
  CeedInt idx=-1;

  for(CeedInt i=0; i<nprofs; i++) {
    if (eta < eta_table[i]) {
      idx = i;
      break;
    }
  }
  CeedScalar f, fp, fpp;

  if (idx > 0) { // eta within the bounds of eta_table
    CeedScalar coeff = (eta - eta_table[idx-1]) / (eta_table[idx] - eta_table[idx
                       -1]);

    f   = f_table[idx-1]   + coeff*( f_table[idx]   - f_table[idx-1] );
    fp  = fp_table[idx-1]  + coeff*( fp_table[idx]  - fp_table[idx-1] );
    fpp = fpp_table[idx-1] + coeff*( fpp_table[idx] - fpp_table[idx-1] );
  } else { // eta outside bounds of eta_table
    f   = f_table[nprofs-1];
    fp  = fp_table[nprofs-1];
    fpp = fpp_table[nprofs-1];
    eta = eta_table[nprofs-1];
  }

  *u = Uinf*fp;
  *v = 0.5*sqrt(nu*Uinf/(x0+x))*(eta*fp - f);
}

// *****************************************************************************
// This QFunction sets a Blasius boundary layer for the initial condition
// *****************************************************************************
CEED_QFUNCTION(ICsBlasius)(void *ctx, CeedInt Q,
                           const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0];

  // Outputs
  CeedScalar (*q0)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  const BlasiusContext context = (BlasiusContext)ctx;
  const CeedScalar cv     = context->newtonian_ctx.cv;
  const CeedScalar cp     = context->newtonian_ctx.cp;
  const CeedScalar Rd     = cp - cv;
  const CeedScalar gamma  = cp/cv;
  const CeedScalar mu     = context->newtonian_ctx.mu;

  const CeedScalar theta0 = context->theta0;
  const CeedScalar P0     = context->P0;
  const CeedScalar delta0 = context->delta0;
  const CeedScalar Uinf   = context->Uinf;

  const CeedScalar e_internal = cv * theta0;
  const CeedScalar rho        = P0 / ((gamma - 1) * e_internal);
  const CeedScalar x0         = Uinf*rho / (mu*25/ (delta0*delta0) );
  CeedScalar u, v;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};

    BlasiusSolution(x[1], Uinf, x0, x[0], rho, &u, &v, &context->newtonian_ctx);

    q0[0][i] = rho;
    q0[1][i] = u * rho;
    q0[2][i] = v * rho;
    q0[3][i] = 0.;
    q0[4][i] = rho * e_internal + 0.5*(u*u + v*v)*rho;
  } // End of Quadrature Point Loop
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Blasius_Inflow)(void *ctx, CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1],
                   (*X)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*
  const BlasiusContext context = (BlasiusContext)ctx;
  const bool implicit     = context->implicit;
  const CeedScalar mu     = context->newtonian_ctx.mu;
  const CeedScalar cv     = context->newtonian_ctx.cv;
  const CeedScalar cp     = context->newtonian_ctx.cp;
  const CeedScalar Rd     = cp - cv;
  const CeedScalar gamma  = cp/cv;

  const CeedScalar theta0 = context->theta0;
  const CeedScalar P0     = context->P0;
  const CeedScalar delta0 = context->delta0;
  const CeedScalar Uinf   = context->Uinf;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // Calcualte prescribed inflow values
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};

    // Find pressure using state inside the domain
    const CeedScalar rho = q[0][i];
//    const CeedScalar u[3] = {q[1][i]/rho, q[2][i]/rho, q[3][i]/rho};
//    const CeedScalar E_internal = q[4][i] - .5 * rho * (u[0]*u[0] + u[1]*u[1] +
//                                  u[2]*u[2]);
//    const CeedScalar P = E_internal * (gamma - 1.);

    // Find inflow state using calculated P and prescribed velocity, theta0
    const CeedScalar e_internal = cv * theta0;
//    const CeedScalar rho_in = P / ((gamma - 1) * e_internal); // mixed up
    const CeedScalar rho_0 = P0 / ((gamma - 1) * e_internal); // rho exterior but what for?
    const CeedScalar P=rho*Rd*theta0; // interior rho with exterior T

    const CeedScalar x0     = Uinf*rho / (mu*25/ (delta0*delta0) );
    CeedScalar velocity[3] = {0.};
    BlasiusSolution(x[1], Uinf, x0, x[0], rho_0, &velocity[0], &velocity[1],
                    &context->newtonian_ctx);

//    const CeedScalar E_kinetic = .5 * rho_in * (velocity[0]*velocity[0] +
    const CeedScalar E_kinetic = .5 * rho * (velocity[0]*velocity[0] +
                                 velocity[1]*velocity[1] +
                                 velocity[2]*velocity[2]);
//    const CeedScalar E = rho_in * e_internal + E_kinetic;
    const CeedScalar E = rho * e_internal + E_kinetic;  // use interior rho
                           // from T       and  u exterior
    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    // The Physics
    // Zero v so all future terms can safely sum into it
    for (int j=0; j<5; j++) v[j][i] = 0.;

    const CeedScalar u_normal = norm[0]*velocity[0] +
                                norm[1]*velocity[1] +
                                norm[2]*velocity[2];

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal; // interior rho

    // -- Momentum
    for (int j=0; j<3; j++)
      v[j+1][i] -= wdetJb * (rho * u_normal * velocity[j] + // interior rho
                             norm[j] * P); // mixed P

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);

  } // End Quadrature Point Loop
  return 0;
}

// *****************************************************************************
CEED_QFUNCTION(Blasius_Outflow)(void *ctx, CeedInt Q,
                                const CeedScalar *const *in,
                                CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*q)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const BlasiusContext context = (BlasiusContext)ctx;
  const bool implicit     = context->implicit;
  const CeedScalar mu     = context->newtonian_ctx.mu;
  const CeedScalar cv     = context->newtonian_ctx.cv;
  const CeedScalar cp     = context->newtonian_ctx.cp;
  const CeedScalar Rd     = cp - cv;
  const CeedScalar gamma  = cp/cv;

  const CeedScalar theta0 = context->theta0;
  const CeedScalar P0     = context->P0;
  const CeedScalar delta0 = context->delta0;
  const CeedScalar Uinf   = context->Uinf;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho      =  q[0][i];
    const CeedScalar u[3]     = {q[1][i] / rho,
                                 q[2][i] / rho,
                                 q[3][i] / rho
                                };
    const CeedScalar E        =  q[4][i];

    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    // The Physics
    // Zero v so all future terms can safely sum into it
    for (int j=0; j<5; j++) v[j][i] = 0.;

    // Implementing outflow condition
    const CeedScalar E_kinetic = (u[0]*u[0] + u[1]*u[1]) / 2.;
    const CeedScalar P         = P0; // pressure
    const CeedScalar u_normal  = norm[0]*u[0] + norm[1]*u[1] +
                                 norm[2]*u[2]; // Normal velocity
    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal;

    // -- Momentum
    for (int j=0; j<3; j++)
      v[j+1][i] -= wdetJb *(rho * u_normal * u[j] + norm[j] * P);

    // -- Total Energy Density
    v[4][i] -= wdetJb * u_normal * (E + P);

  } // End Quadrature Point Loop
  return 0;
}
#endif // blasius_h
