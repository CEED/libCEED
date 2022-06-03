// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Operator for Navier-Stokes example using PETSc


#ifndef blasius_h
#define blasius_h

#include <math.h>
#include <ceed.h>
#include "newtonian_types.h"

typedef struct BlasiusContext_ *BlasiusContext;
struct BlasiusContext_ {
  bool       implicit;  // !< Using implicit timesteping or not
  bool       weakT;     // !< flag to set Temperature weakly at inflow
  CeedScalar delta0;    // !< Boundary layer height at inflow
  CeedScalar Uinf;      // !< Velocity at boundary layer edge
  CeedScalar P0;        // !< Pressure at outflow
  CeedScalar theta0;    // !< Temperature at inflow
  CeedScalar x_inflow;  // !< Location of inflow in x
  struct NewtonianIdealGasContext_ newtonian_ctx;
};

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

void CEED_QFUNCTION_HELPER(BlasiusSolution)(const CeedScalar y,
    const CeedScalar Uinf, const CeedScalar x0, const CeedScalar x,
    const CeedScalar rho, CeedScalar *u, CeedScalar *v, CeedScalar *t12,
    const NewtonianIdealGasContext newt_ctx) {

  CeedInt nprofs = 50;
  // *INDENT-OFF*
  CeedScalar eta_table[] = {
    0.000000000000000000e+00, 1.282051282051281937e-01, 2.564102564102563875e-01, 3.846153846153845812e-01, 5.128205128205127750e-01,
    6.410256410256409687e-01, 7.692307692307691624e-01, 8.974358974358973562e-01, 1.025641025641025550e+00, 1.153846153846153744e+00,
    1.282051282051281937e+00, 1.410256410256410131e+00, 1.538461538461538325e+00, 1.666666666666666519e+00, 1.794871794871794712e+00,
    1.923076923076922906e+00, 2.051282051282051100e+00, 2.179487179487179294e+00, 2.307692307692307487e+00, 2.435897435897435681e+00,
    2.564102564102563875e+00, 2.692307692307692069e+00, 2.820512820512820262e+00, 2.948717948717948456e+00, 3.076923076923076650e+00,
    3.205128205128204844e+00, 3.333333333333333037e+00, 3.461538461538461231e+00, 3.589743589743589425e+00, 3.717948717948717618e+00,
    3.846153846153845812e+00, 3.974358974358974006e+00, 4.102564102564102200e+00, 4.230769230769229949e+00, 4.358974358974358587e+00,
    4.487179487179487225e+00, 4.615384615384614975e+00, 4.743589743589742724e+00, 4.871794871794871362e+00, 5.000000000000000000e+00,
    5.500000000000000000e+00, 6.000000000000000000e+00, 6.500000000000000000e+00, 7.000000000000000000e+00, 7.500000000000000000e+00,
    8.000000000000000000e+00, 8.500000000000000000e+00, 9.000000000000000000e+00, 9.500000000000000000e+00, 1.000000000000000000e+01};

  CeedScalar f_table[] = {
    0.000000000000000000e+00, 2.728923405566200267e-03, 1.091524811461423369e-02, 2.455658828897525764e-02, 4.364674649279581820e-02,
    6.817382707725749835e-02, 9.811838418932711248e-02, 1.334516294237205192e-01, 1.741337304561980659e-01, 2.201122374410622862e-01,
    2.713206781625860375e-01, 3.276773654929600599e-01, 3.890844612583744255e-01, 4.554273387986328414e-01, 5.265742820946719416e-01,
    6.023765522220410062e-01, 6.826688421431770237e-01, 7.672701287583111318e-01, 8.559849171804534418e-01, 9.486048570979430661e-01,
    1.044910695686512625e+00, 1.144674516826549082e+00, 1.247662203367335465e+00, 1.353636048811749593e+00, 1.462357437868362364e+00,
    1.573589512396551759e+00, 1.687099740622293842e+00, 1.802662313062363353e+00, 1.920060297987626230e+00, 2.039087501786055245e+00,
    2.159549994377929050e+00, 2.281267275838891884e+00, 2.404073076539093190e+00, 2.527815798402052838e+00, 2.652358618452637540e+00,
    2.777579287003750341e+00, 2.903369661199559637e+00, 3.029635020019957992e+00, 3.156293209307130088e+00, 3.283273665161465349e+00,
    3.780571892998292771e+00, 4.279620922520262383e+00, 4.779322325882148448e+00, 5.279238811036782053e+00, 5.779218028455369804e+00,
    6.279213431354994768e+00, 6.779212528163703233e+00, 7.279212370655419484e+00, 7.779212346288013613e+00, 8.279212342945751146e+00};

  CeedScalar fp_table[] = {
    0.000000000000000000e+00, 4.257083277988830267e-02, 8.513297869782740501e-02, 1.276641169537044151e-01, 1.701271279078802878e-01,
    2.124702831905590783e-01, 2.546276046951935212e-01, 2.965194442747576264e-01, 3.380533304776729975e-01, 3.791251204629754179e-01,
    4.196204840172004791e-01, 4.594167322894788796e-01, 4.983849866855867838e-01, 5.363926638765821320e-01, 5.733062319885513514e-01,
    6.089941719927144392e-01, 6.433300586189647507e-01, 6.761956584341198839e-01, 7.074839307288774970e-01, 7.371018110314454530e-01,
    7.649726585225528064e-01, 7.910382579383948842e-01, 8.152602836158657773e-01, 8.376211573266827415e-01, 8.581242609418713307e-01,
    8.767934976651666767e-01, 8.936722290953328374e-01, 9.088216471306606037e-01, 9.223186672607004422e-01, 9.342534510898168332e-01,
    9.447266795705382414e-01, 9.538467037387058367e-01, 9.617266968332524035e-01, 9.684819213624265011e-01, 9.742272083384174719e-01,
    9.790747253056680810e-01, 9.831320868743089747e-01, 9.865008381344084754e-01, 9.892753192614093249e-01, 9.915419001656551323e-01,
    9.968788209317821503e-01, 9.989728724371175206e-01, 9.996990677381791812e-01, 9.999216041491896245e-01, 9.999818594083667023e-01,
    9.999962745365539307e-01, 9.999993214550036980e-01, 9.999998904550418954e-01, 9.999999843329338001e-01, 9.999999980166356384e-01};

  CeedScalar fpp_table[] = {
    3.320573362157903663e-01, 3.320379743512646420e-01, 3.319024760665882368e-01, 3.315350015070190337e-01, 3.308206767975666041e-01,
    3.296466995822193158e-01, 3.279038639411161471e-01, 3.254884713737624113e-01, 3.223045750196085746e-01, 3.182664816607024272e-01,
    3.133014118810801829e-01, 3.073521951089355775e-01, 3.003798556086043625e-01, 2.923659305537876785e-01, 2.833143548208253981e-01,
    2.732527514995234941e-01, 2.622329840371728227e-01, 2.503308560706500874e-01, 2.376448876931176457e-01, 2.242941499773744018e-01,
    2.104151994284793603e-01, 1.961582158440171031e-01, 1.816825052623964043e-01, 1.671515786102889534e-01, 1.527280512426029968e-01,
    1.385686249977987894e-01, 1.248194106805364800e-01, 1.116118251613979206e-01, 9.905925581301598670e-02, 8.725462988794610575e-02,
    7.626896310981794158e-02, 6.615089622448211415e-02, 5.692716644118058639e-02, 4.860390768479891377e-02, 4.116863313890323922e-02,
    3.459272784597366285e-02, 2.883426862493499582e-02, 2.384099224121952881e-02, 1.955324839409207718e-02, 1.590679868531958210e-02,
    6.578593141419011685e-03, 2.402039843751689954e-03, 7.741093231657678389e-04, 2.201689553063347941e-04, 5.526217815680267893e-05,
    1.224092624232004387e-05, 2.392841910090350858e-06, 4.127879363882133676e-07, 6.284244603762621373e-08, 8.442944409712819646e-09};
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
  *t12 = rho*nu*Uinf*fpp*sqrt(Uinf/(nu*(x0+x)));
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
  const CeedScalar gamma  = cp/cv;
  const CeedScalar mu     = context->newtonian_ctx.mu;

  const CeedScalar theta0 = context->theta0;
  const CeedScalar P0     = context->P0;
  const CeedScalar delta0 = context->delta0;
  const CeedScalar Uinf   = context->Uinf;
  const CeedScalar x_inflow   = context->x_inflow;

  const CeedScalar e_internal = cv * theta0;
  const CeedScalar rho        = P0 / ((gamma - 1) * e_internal);
  const CeedScalar x0         = Uinf*rho / (mu*25/ (delta0*delta0) );
  CeedScalar u, v, t12;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar x[] = {X[0][i], X[1][i], X[2][i]};

    BlasiusSolution(x[1], Uinf, x0, x[0] - x_inflow, rho, &u, &v, &t12,
                    &context->newtonian_ctx);

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
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*X)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[3];

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

  const CeedScalar theta0   = context->theta0;
  const CeedScalar P0       = context->P0;
  const CeedScalar delta0   = context->delta0;
  const CeedScalar Uinf     = context->Uinf;
  const CeedScalar x_inflow = context->x_inflow;
  const bool       weakT    = context->weakT;
  const CeedScalar rho_0    = P0 / (Rd * theta0);
  const CeedScalar x0       = Uinf*rho_0 / (mu*25/ (delta0*delta0) );

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // Calculate inflow values
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar velocity[3] = {0.};
    CeedScalar t12;
    BlasiusSolution(x[1], Uinf, x0, x[0] - x_inflow, rho_0, &velocity[0],
                    &velocity[1], &t12, &context->newtonian_ctx);

    // enabling user to choose between weak T and weak rho inflow
    CeedScalar rho,E_internal, P, E_kinetic;
    if (weakT) {
      // rho should be from the current solution
      rho = q[0][i];
      // Temperature is being set weakly (theta0) and for constant cv this sets E_internal
      E_internal = rho * cv * theta0;
      // Find pressure using
      P = rho*Rd*theta0; // interior rho with exterior T
      E_kinetic = .5 * rho * Dot3(velocity, velocity);
    } else {
      //  Fixing rho weakly on the inflow to a value  consistent with theta0 and P0
      rho =  rho_0;
      E_kinetic = .5 * rho * Dot3(velocity, velocity);
      E_internal = q[4][i] - E_kinetic; // uses set rho and u but E from solution
      P = E_internal * (gamma - 1.);
    }
    const CeedScalar E = E_internal + E_kinetic;
    // ---- Normal vect
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    // The Physics
    // Zero v so all future terms can safely sum into it
    for (CeedInt j=0; j<5; j++) v[j][i] = 0.;

    const CeedScalar u_normal = Dot3(norm, velocity);
    const CeedScalar viscous_flux[3] = {-t12 *norm[1], -t12 *norm[0], 0};

    // The Physics
    // -- Density
    v[0][i] -= wdetJb * rho * u_normal; // interior rho

    // -- Momentum
    for (CeedInt j=0; j<3; j++)
      v[j+1][i] -= wdetJb * (rho * u_normal * velocity[j] // interior rho
                             + norm[j] * P // mixed P
                             + viscous_flux[j]);

    // -- Total Energy Density
    v[4][i] -= wdetJb * (u_normal * (E + P) + Dot3(viscous_flux, velocity));

  } // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(Blasius_Inflow_Jacobian)(void *ctx, CeedInt Q,
                                        const CeedScalar *const *in,
                                        CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*dq)[CEED_Q_VLA]         = (const CeedScalar(*)[CEED_Q_VLA])in[0],
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
  const bool weakT        = context->weakT;
  const CeedScalar rho_0  = P0 / (Rd * theta0);
  const CeedScalar x0     = Uinf*rho_0 / (mu*25/ (delta0*delta0) );

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp-to-Interp q_data
    // For explicit mode, the surface integral is on the RHS of ODE q_dot = f(q).
    // For implicit mode, it gets pulled to the LHS of implicit ODE/DAE g(q_dot, q).
    // We can effect this by swapping the sign on this weight
    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];

    // Calculate inflow values
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar velocity[3] = {0};
    CeedScalar t12;
    BlasiusSolution(x[1], Uinf, x0, x[0], rho_0, &velocity[0], &velocity[1],
                    &t12, &context->newtonian_ctx);

    // enabling user to choose between weak T and weak rho inflow
    CeedScalar drho, dE, dP;
    if (weakT) {
      // rho should be from the current solution
      drho = dq[0][i];
      CeedScalar dE_internal = drho * cv * theta0;
      CeedScalar dE_kinetic = .5 * drho * Dot3(velocity, velocity);
      dE = dE_internal + dE_kinetic;
      dP = drho * Rd * theta0; // interior rho with exterior T
    } else { // rho specified, E_internal from solution
      drho = 0;
      dE = dq[4][i];
      dP = dE * (gamma - 1.);
    }
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    const CeedScalar u_normal = Dot3(norm, velocity);

    v[0][i] = - wdetJb * drho * u_normal;
    for (int j=0; j<3; j++)
      v[j+1][i] = -wdetJb * (drho * u_normal * velocity[j] + norm[j] * dP);
    v[4][i] = - wdetJb * u_normal * (dE + dP);
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
                   (*q_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2],
                   (*X)[CEED_Q_VLA]          = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA]            = (CeedScalar(*)[CEED_Q_VLA])out[0],
             (*jac_data_sur)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[1];
  // *INDENT-ON*

  const BlasiusContext context = (BlasiusContext)ctx;
  const bool implicit     = context->implicit;
  const CeedScalar mu     = context->newtonian_ctx.mu;
  const CeedScalar cv     = context->newtonian_ctx.cv;
  const CeedScalar cp     = context->newtonian_ctx.cp;
  const CeedScalar Rd     = cp - cv;

  const CeedScalar theta0   = context->theta0;
  const CeedScalar P0       = context->P0;
  const CeedScalar rho_0    = P0 / (Rd*theta0);
  const CeedScalar delta0   = context->delta0;
  const CeedScalar Uinf     = context->Uinf;
  const CeedScalar x0       = Uinf*rho_0 / (mu*25/ (delta0*delta0) );
  const CeedScalar x_inflow = context->x_inflow;

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

    // Implementing outflow condition
    const CeedScalar P         = P0; // pressure
    const CeedScalar u_normal  = Dot3(norm, u); // Normal velocity

    // Calculate prescribed outflow traction values
    const CeedScalar x[3] = {X[0][i], X[1][i], X[2][i]};
    CeedScalar velocity[3] = {0.};
    CeedScalar t12;
    BlasiusSolution(x[1], Uinf, x0, x[0] - x_inflow, rho_0, &velocity[0],
                    &velocity[1], &t12, &context->newtonian_ctx);
    const CeedScalar viscous_flux[3] = {-t12 *norm[1], -t12 *norm[0], 0};

    // -- Density
    v[0][i] = -wdetJb * rho * u_normal;

    // -- Momentum
    for (CeedInt j=0; j<3; j++)
      v[j+1][i] = -wdetJb * (rho * u_normal * u[j]
                             + norm[j] * P + viscous_flux[j]);

    // -- Total Energy Density
    v[4][i] = -wdetJb * (u_normal * (E + P)
                         + Dot3(viscous_flux, velocity));

    // Save values for Jacobian
    jac_data_sur[0][i] = rho;
    jac_data_sur[1][i] = u[0];
    jac_data_sur[2][i] = u[1];
    jac_data_sur[3][i] = u[2];
    jac_data_sur[4][i] = E;
  } // End Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(Blasius_Outflow_Jacobian)(void *ctx, CeedInt Q,
    const CeedScalar *const *in,
    CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*dq)[CEED_Q_VLA]           = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*q_data_sur)[CEED_Q_VLA]   = (const CeedScalar(*)[CEED_Q_VLA])in[1],
                   (*jac_data_sur)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const BlasiusContext context = (BlasiusContext)ctx;
  const bool implicit     = context->implicit;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar rho = jac_data_sur[0][i];
    const CeedScalar u[3] = {jac_data_sur[1][i], jac_data_sur[2][i], jac_data_sur[3][i]};
    const CeedScalar E = jac_data_sur[4][i];

    const CeedScalar drho      =  dq[0][i];
    const CeedScalar dmomentum[3] = {dq[1][i], dq[2][i], dq[3][i]};
    const CeedScalar dE        =  dq[4][i];

    const CeedScalar wdetJb  = (implicit ? -1. : 1.) * q_data_sur[0][i];
    const CeedScalar norm[3] = {q_data_sur[1][i],
                                q_data_sur[2][i],
                                q_data_sur[3][i]
                               };

    CeedScalar du[3];
    for (int j=0; j<3; j++) du[j] = (dmomentum[j] - u[j] * drho) / rho;
    const CeedScalar u_normal  = Dot3(norm, u);
    const CeedScalar du_normal = Dot3(norm, du);
    const CeedScalar dmomentum_normal = drho * u_normal + rho * du_normal;
    const CeedScalar P = context->P0;
    const CeedScalar dP = 0;

    v[0][i] = -wdetJb * dmomentum_normal;
    for (int j=0; j<3; j++)
      v[j+1][i] = -wdetJb * (dmomentum_normal * u[j] + rho * u_normal * du[j]);
    v[4][i] = -wdetJb * (du_normal * (E + P) + u_normal * (dE + dP));
  } // End Quadrature Point Loop
  return 0;
}

#endif // blasius_h
