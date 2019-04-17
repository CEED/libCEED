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

#ifndef densitycurrent_h
#define densitycurrent_h

#ifndef CeedPragmaOMP
#  ifdef _OPENMP
#    define CeedPragmaOMP_(a) _Pragma(#a)
#    define CeedPragmaOMP(a) CeedPragmaOMP_(omp a)
#  else
#    define CeedPragmaOMP(a)
#  endif
#endif

#include <math.h>

// *****************************************************************************
// This QFunction sets the the initial conditions and boundary conditions
//
// These initial conditions are given in terms of potential temperature and
//   Exner pressure and then converted to density and total energy.
//   Initial momentum density is zero.
//
// Initial Conditions:
//   Potential Temperature:
//     theta = thetabar + deltatheta
//       thetabar   = theta0 exp( N**2 z / g )
//       deltatheta = r <= rc : theta0(1 + cos(pi r/rc)) / 2
//                     r > rc : 0
//         r        = sqrt( (x - xc)**2 + (y - yc)**2 + (z - zc)**2 )
//         with (xc,yc,zc) center of domain, rc characteristic radius of thermal bubble
//   Exner Pressure:
//     Pi = Pibar + deltaPi
//       Pibar      = g**2 (exp( - N**2 z / g ) - 1) / (cp theta0 N**2)
//       deltaPi    = 0 (hydrostatic balance)
//   Velocity/Momentum Density:
//     Ui = ui = 0
//
// Conversion to Conserved Variables:
//   rho = P0 Pi**(cv/Rd) / (Rd theta)
//   E   = rho (cv theta Pi + (u u)/2 + g z)
//
//  Boundary Conditions:
//    Mass Density:
//      0.0 flux
//    Momentum Density:
//      0.0
//    Energy Density:
//      0.0 flux
//
// Constants:
//   theta0          ,  Potential temperature constant
//   thetaC          ,  Potential temperature perturbation
//   P0              ,  Pressure at the surface
//   Nf              ,  Brunt-Vaisala frequency
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   Rd     = cp - cv,  Specific heat difference
//   g               ,  Gravity
//   rc              ,  Characteristic radius of thermal bubble
//   lx              ,  Characteristic length scale of domain in x
//   ly              ,  Characteristic length scale of domain in y
//   lz              ,  Characteristic length scale of domain in z
//
// *****************************************************************************
static int ICsDC(void *ctx, CeedInt Q, CeedInt N, CeedQFunctionArguments args) {

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

  // Inputs
  const CeedScalar *X = args.in[0];
  // Outputs
  CeedScalar *q0 = args.out[0], *coords = args.out[1];
  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar theta0     = context[0];
  const CeedScalar thetaC     = context[1];
  const CeedScalar P0         = context[2];
  const CeedScalar Nf         = context[3];
  const CeedScalar cv         = context[4];
  const CeedScalar cp         = context[5];
  const CeedScalar Rd         = context[6];
  const CeedScalar g          = context[7];
  const CeedScalar rc         = context[8];
  const CeedScalar lx         = context[9];
  const CeedScalar ly         = context[10];
  const CeedScalar lz         = context[11];
  // Setup
  const CeedScalar tol = 1.e-14;
  const CeedScalar center[3] = {0.5*lx, 0.5*ly, 0.5*lz};

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Coordinates
    const CeedScalar x = X[i+0*N];
    const CeedScalar y = X[i+1*N];
    const CeedScalar z = X[i+2*N];
    // -- Potential temperature, density current
    const CeedScalar r = sqrt(pow((x - center[0]), 2) +
                              pow((y - center[1]), 2) +
                              pow((z - center[2]), 2));
    const CeedScalar deltatheta = r <= rc ? thetaC*(1. + cos(M_PI*r/rc))/2. : 0.;
    const CeedScalar theta = theta0*exp(Nf*Nf*z/g) + deltatheta;
    // -- Exner pressure, hydrostatic balance
    const CeedScalar Pi = 1. + g*g*(exp(-Nf*Nf*z/g) - 1.) / (cp*theta0*Nf*Nf);
    // -- Density
    const CeedScalar rho = P0 * pow(Pi, cv/Rd) / (Rd*theta);

    // Initial Conditions
    q0[i+0*N] = rho;
    q0[i+1*N] = 0.0;
    q0[i+2*N] = 0.0;
    q0[i+3*N] = 0.0;
    q0[i+4*N] = rho * (cv*theta*Pi + g*z);

    // Homogeneous Dirichlet Boundary Conditions for Momentum
    if ( fabs(x - 0.0) < tol || fabs(x - lx) < tol ||
         fabs(y - 0.0) < tol || fabs(y - ly) < tol ||
         fabs(z - 0.0) < tol || fabs(z - lz) < tol ) {
      q0[i+1*N] = 0.0;
      q0[i+2*N] = 0.0;
      q0[i+3*N] = 0.0;
    }

    // Coordinates
    coords[i+0*N] = x;
    coords[i+1*N] = y;
    coords[i+2*N] = z;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of Navier-Stokes
//
// This is 3D compressible Navier-Stokes in conservation form with state
//   variables of density, momentum density, and total energy density.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density   ,  Ui = rho ui
//   E   - Total Energy Density,  E  = rho cv T + rho (u u) / 2 + rho g z
//
// Navier-Stokes Equations:
//   drho/dt + div( U )                               = 0
//   dU/dt   + div( rho (u x u) + P I3 ) + rho g khat = div( Fu )
//   dE/dt   + div( (E + P) u )                       = div( Fe )
//
// Viscous Stress:
//   Fu = mu (grad( u ) + grad( u )^T + lambda div ( u ) I3)
// Thermal Stress:
//   Fe = u Fu + k grad( T )
//
// Equation of State:
//   P = (gamma - 1) (E - rho (u u) / 2 - rho g z)
//
// Temperature:
//   T = (E / rho - (u u) / 2 - g z) / cv
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
// *****************************************************************************
static int DC(void *ctx, CeedInt Q, CeedInt N, CeedQFunctionArguments args) {
  // Inputs
  const CeedScalar *q = args.in[0], *dq = args.in[1], *qdata = args.in[2],
                   *x = args.in[3];
  // Outputs
  CeedScalar *v = args.out[0], *dv = args.out[1];
  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar lambda     = context[0];
  const CeedScalar mu         = context[1];
  const CeedScalar k          = context[2];
  const CeedScalar cv         = context[3];
  const CeedScalar cp         = context[4];
  const CeedScalar g          = context[5];
  const CeedScalar gamma      = cp / cv;

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho     =   q[i+0*N];
    const CeedScalar u[3]    = { q[i+1*N] / rho,
                                 q[i+2*N] / rho,
                                 q[i+3*N] / rho
                               };
    const CeedScalar E       =   q[i+4*N];
    // -- Grad in
    const CeedScalar drho[3] = {  dq[i+(0+5*0)*N],
                                  dq[i+(0+5*1)*N],
                                  dq[i+(0+5*2)*N]
                               };
    const CeedScalar du[9]   = { (dq[i+(1+5*0)*N] - drho[0]*u[0]) / rho,
                                 (dq[i+(1+5*1)*N] - drho[1]*u[0]) / rho,
                                 (dq[i+(1+5*2)*N] - drho[2]*u[0]) / rho,
                                 (dq[i+(2+5*0)*N] - drho[0]*u[1]) / rho,
                                 (dq[i+(2+5*1)*N] - drho[1]*u[1]) / rho,
                                 (dq[i+(2+5*2)*N] - drho[2]*u[1]) / rho,
                                 (dq[i+(3+5*0)*N] - drho[0]*u[2]) / rho,
                                 (dq[i+(3+5*1)*N] - drho[1]*u[2]) / rho,
                                 (dq[i+(3+5*2)*N] - drho[2]*u[2]) / rho
                               };
    const CeedScalar dE[3]   = {  dq[i+(4+5*0)*N],
                                  dq[i+(4+5*1)*N],
                                  dq[i+(4+5*2)*N]
                               };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ       =   qdata[i+ 0*N];
    // -- Interp-to-Grad qdata
    //      Symmetric 3x3 matrix
    const CeedScalar wBJ[9]   = { qdata[i+ 1*N],
                                  qdata[i+ 2*N],
                                  qdata[i+ 3*N],
                                  qdata[i+ 4*N],
                                  qdata[i+ 5*N],
                                  qdata[i+ 6*N],
                                  qdata[i+ 7*N],
                                  qdata[i+ 8*N],
                                  qdata[i+ 9*N]
                                };
    // -- Grad-to-Grad qdata
    const CeedScalar wBBJ[6]  = { qdata[i+10*N],
                                  qdata[i+11*N],
                                  qdata[i+12*N],
                                  qdata[i+13*N],
                                  qdata[i+14*N],
                                  qdata[i+15*N]
                                };
    // -- gradT
    const CeedScalar gradT[3] = { (dE[0]/rho - E*drho[0]/(rho*rho) -
                                   (u[0]*du[0+3*0] + u[1]*du[1+3*0] +
                                    u[2]*du[2+3*0])) / cv,
                                  (dE[1]/rho - E*drho[1]/(rho*rho) -
                                   (u[0]*du[0+3*1] + u[1]*du[1+3*1] +
                                    u[2]*du[2+3*1])) / cv,
                                  (dE[2]/rho - E*drho[2]/(rho*rho) -
                                   (u[0]*du[0+3*2] + u[1]*du[1+3*2] +
                                    u[2]*du[2+3*2]) - g) / cv
                                };
    // -- Fuvisc
    //      Symmetric 3x3 matrix
    const CeedScalar Fu[6] =  { mu * (du[0+3*0] * (2 + lambda) +
                                      lambda * (du[1+3*1] + du[2+3*2])),
                                mu * (du[0+3*1] + du[1+3*0]),
                                mu * (du[0+3*2] + du[2+3*0]),
                                mu * (du[1+3*1] * (2 + lambda) +
                                      lambda * (du[0+3*0] + du[2+3*2])),
                                mu * (du[1+3*2] + du[2+3*1]),
                                mu * (du[2+3*2] * (2 + lambda) +
                                      lambda * (du[0+3*0] + du[1+3*1]))
                              };

    // -- Fevisc
    const CeedScalar Fe[3] = { u[0]*Fu[0] + u[1]*Fu[1] + u[2]*Fu[2] +
                               k * gradT[0],
                               u[0]*Fu[1] + u[1]*Fu[3] + u[2]*Fu[4] +
                               k * gradT[1],
                               u[0]*Fu[2] + u[1]*Fu[4] + u[2]*Fu[5] +
                               k * gradT[2]
                             };
    // -- P
    const CeedScalar P = (E - (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])*rho/2 -
                          rho*g*x[i+N*2]) * (gamma - 1);

    // The Physics

    // -- Density
    // ---- u rho
    dv[i+(0+5*0)*N]  = rho*u[0]*wBJ[0] + rho*u[1]*wBJ[1] + rho*u[2]*wBJ[2];
    dv[i+(0+5*1)*N]  = rho*u[0]*wBJ[3] + rho*u[1]*wBJ[4] + rho*u[2]*wBJ[5];
    dv[i+(0+5*2)*N]  = rho*u[0]*wBJ[6] + rho*u[1]*wBJ[7] + rho*u[2]*wBJ[8];
    // ---- No Change
    v[i+0*N] = 0;

    // -- Momentum
    // ---- rho (u x u) + P I3
    dv[i+(1+5*0)*N]  = (rho*u[0]*u[0]+P)*wBJ[0] + rho*u[0]*u[1]*wBJ[1] +
                       rho*u[0]*u[2]*wBJ[2];
    dv[i+(1+5*1)*N]  = (rho*u[0]*u[0]+P)*wBJ[3] + rho*u[0]*u[1]*wBJ[4] +
                       rho*u[0]*u[2]*wBJ[5];
    dv[i+(1+5*2)*N]  = (rho*u[0]*u[0]+P)*wBJ[6] + rho*u[0]*u[1]*wBJ[7] +
                       rho*u[0]*u[2]*wBJ[8];
    dv[i+(2+5*0)*N]  =  rho*u[1]*u[0]*wBJ[0] +   (rho*u[1]*u[1]+P)*wBJ[1] +
                        rho*u[1]*u[2]*wBJ[2];
    dv[i+(2+5*1)*N]  =  rho*u[1]*u[0]*wBJ[3] +   (rho*u[1]*u[1]+P)*wBJ[4] +
                        rho*u[1]*u[2]*wBJ[5];
    dv[i+(2+5*2)*N]  =  rho*u[1]*u[0]*wBJ[6] +   (rho*u[1]*u[1]+P)*wBJ[7] +
                        rho*u[1]*u[2]*wBJ[8];
    dv[i+(3+5*0)*N]  =  rho*u[2]*u[0]*wBJ[0] +    rho*u[2]*u[1]*wBJ[1] +
                        (rho*u[2]*u[2]+P)*wBJ[2];
    dv[i+(3+5*1)*N]  =  rho*u[2]*u[0]*wBJ[3] +    rho*u[2]*u[1]*wBJ[4] +
                        (rho*u[2]*u[2]+P)*wBJ[5];
    dv[i+(3+5*2)*N]  =  rho*u[2]*u[0]*wBJ[6] +    rho*u[2]*u[1]*wBJ[7] +
                        (rho*u[2]*u[2]+P)*wBJ[8];
    // ---- Fuvisc
    dv[i+(1+5*0)*N] -= Fu[0]*wBBJ[0] + Fu[1]*wBBJ[1] + Fu[2]*wBBJ[2];
    dv[i+(1+5*1)*N] -= Fu[0]*wBBJ[1] + Fu[1]*wBBJ[3] + Fu[2]*wBBJ[4];
    dv[i+(1+5*2)*N] -= Fu[0]*wBBJ[2] + Fu[1]*wBBJ[4] + Fu[2]*wBBJ[5];
    dv[i+(2+5*0)*N] -= Fu[1]*wBBJ[0] + Fu[3]*wBBJ[1] + Fu[4]*wBBJ[2];
    dv[i+(2+5*1)*N] -= Fu[1]*wBBJ[1] + Fu[3]*wBBJ[3] + Fu[4]*wBBJ[4];
    dv[i+(2+5*2)*N] -= Fu[1]*wBBJ[2] + Fu[3]*wBBJ[4] + Fu[4]*wBBJ[5];
    dv[i+(3+5*0)*N] -= Fu[2]*wBBJ[0] + Fu[4]*wBBJ[1] + Fu[5]*wBBJ[2];
    dv[i+(3+5*1)*N] -= Fu[2]*wBBJ[1] + Fu[4]*wBBJ[3] + Fu[5]*wBBJ[4];
    dv[i+(3+5*2)*N] -= Fu[2]*wBBJ[2] + Fu[4]*wBBJ[4] + Fu[5]*wBBJ[5];
    // ---- -rho g khat
    v[i+1*N] = 0;
    v[i+2*N] = 0;
    v[i+3*N] = - rho*g*wJ;

    // -- Total Energy
    // ---- (E + P) u
    dv[i+(4+5*0)*N]  = (E + P)*(u[0]*wBJ[0] + u[1]*wBJ[1] + u[2]*wBJ[2]);
    dv[i+(4+5*1)*N]  = (E + P)*(u[0]*wBJ[3] + u[1]*wBJ[4] + u[2]*wBJ[5]);
    dv[i+(4+5*2)*N]  = (E + P)*(u[0]*wBJ[6] + u[1]*wBJ[7] + u[2]*wBJ[8]);
    // ---- Fevisc
    dv[i+(4+5*0)*N] -= Fe[0]*wBBJ[0] + Fe[1]*wBBJ[1] + Fe[2]*wBBJ[2];
    dv[i+(4+5*1)*N] -= Fe[0]*wBBJ[1] + Fe[1]*wBBJ[3] + Fe[2]*wBBJ[4];
    dv[i+(4+5*2)*N] -= Fe[0]*wBBJ[2] + Fe[1]*wBBJ[4] + Fe[2]*wBBJ[5];
    // ---- No Change
    v[i+4*N] = 0;

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
#endif
