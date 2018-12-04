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

#include <math.h>

// *****************************************************************************
// This QFunction sets up the geometric factors required for integration and
//   coordinate transformations
//
// All data is stored in 16 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( v u )
//
// Determinant of Jacobian:
//   detJ = J11*A11 + J21*A12 + J31*A13
//     Jij = Jacobian entry ij
//     Aij = Adjoint ij
//
// Stored: w detJ
//   qd: 0
//
// We require the transpose of the inverse of the Jacobian to properly compute
//   integrals of the form: int( gradv u )
//
// Inverse of Jacobian:
//   Bij = Aij / detJ
//
// Stored: w B detJ = w A
//   qd: 1 2 3
//       4 5 6
//       7 8 9
//
// We require the product of the inverse of the Jacobian and its transpose to
//   properly compute integrals of the form: int( gradv gradu)
//
// Product of Inverse and Transpose:
//   BBij = sum( Bik Bkj )
//
// Stored: w B^T B detJ = w A^T A / detJ
//   Note: This matrix is symmetric
//     qd: 10 11 12
//         11 13 14
//         12 14 15
//
// *****************************************************************************
static int Setup(void *ctx, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *J = in[0], *w = in[1];
  // Outputs
  CeedScalar *qdata = out[0];

  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J31 = J[i+Q*2];
    const CeedScalar J12 = J[i+Q*3];
    const CeedScalar J22 = J[i+Q*4];
    const CeedScalar J32 = J[i+Q*5];
    const CeedScalar J13 = J[i+Q*6];
    const CeedScalar J23 = J[i+Q*7];
    const CeedScalar J33 = J[i+Q*8];
    const CeedScalar A11 = J22*J33 - J23*J32;
    const CeedScalar A12 = J13*J32 - J12*J33;
    const CeedScalar A13 = J12*J23 - J13*J22;
    const CeedScalar A21 = J23*J31 - J21*J33;
    const CeedScalar A22 = J11*J33 - J13*J31;
    const CeedScalar A23 = J13*J21 - J11*J23;
    const CeedScalar A31 = J21*J32 - J22*J31;
    const CeedScalar A32 = J12*J31 - J11*J32;
    const CeedScalar A33 = J11*J22 - J12*J21;
    const CeedScalar detJ = J11*A11 + J21*A12 + J31*A13;
    const CeedScalar qw = w[i] / detJ;

    // Qdata
    // -- Interp-to-Interp qdata
    qdata[i+ 0*Q] = w[i] * detJ;
    // -- Interp-to-Grad qdata
    qdata[i+ 1*Q] = w[i] * A11;
    qdata[i+ 2*Q] = w[i] * A12;
    qdata[i+ 3*Q] = w[i] * A13;
    qdata[i+ 4*Q] = w[i] * A21;
    qdata[i+ 5*Q] = w[i] * A22;
    qdata[i+ 6*Q] = w[i] * A23;
    qdata[i+ 7*Q] = w[i] * A31;
    qdata[i+ 8*Q] = w[i] * A32;
    qdata[i+ 9*Q] = w[i] * A33;
    // -- Grad-to-Grad qdata
    qdata[i+10*Q] = qw * (A11*A11 + A12*A12 + A13*A13);
    qdata[i+11*Q] = qw * (A11*A21 + A12*A22 + A13*A23);
    qdata[i+12*Q] = qw * (A11*A31 + A12*A32 + A13*A33);
    qdata[i+13*Q] = qw * (A21*A21 + A22*A22 + A23*A23);
    qdata[i+14*Q] = qw * (A21*A31 + A22*A32 + A23*A33);
    qdata[i+15*Q] = qw * (A31*A31 + A32*A32 + A33*A33);

//printf("%d\n:", i);
//printf("%f\n",qdata[i+1*Q]);
//printf("%f\n",qdata[i+2*Q]);
//printf("%f\n",qdata[i+3*Q]);
//printf("%f\n",qdata[i+4*Q]);
//printf("%f\n",qdata[i+5*Q]);
//printf("%f\n",qdata[i+6*Q]);
//printf("%f\n",qdata[i+7*Q]);
//printf("%f\n",qdata[i+8*Q]);
//printf("%f\n\n",qdata[i+9*Q]);

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction sets up the mass matrix, where u and w are the input arguments and
// v is the output. This is defined for u and v as 5-field (interlaced values) functions
// and w a scalar weight for every quadrature point, identical for all fields
// *****************************************************************************
static int Mass(void *ctx, CeedInt Q,
                const CeedScalar *const *in, CeedScalar *const *out) {
  (void)ctx;
  const CeedScalar *u = in[0], *w = in[1];
  CeedScalar *v = out[0];
  for (CeedInt i=0; i<Q; i++) {
    v[i+0*Q] = w[i+0*Q] * u[i+0*Q];
    v[i+1*Q] = w[i+0*Q] * u[i+1*Q];
    v[i+2*Q] = w[i+0*Q] * u[i+2*Q];
    v[i+3*Q] = w[i+0*Q] * u[i+3*Q];
    v[i+4*Q] = w[i+0*Q] * u[i+4*Q];
  }
  return 0;
}

// *****************************************************************************
// This QFunction sets the the initial conditions
//
// These initial conditions are given in terms of potential temperature and
//   Exner pressure and potential temperature and then converted to density and
//   total energy. Initial velocity and momentum is zero.
//
// Initial Conditions:
//   Potential Temperature:
//     Theta = ThetaBar + deltaTheta
//       ThetaBar   = Theta0 exp( N**2 z / g )
//       detlaTheta = r <= 1: Theta0(1 + cos(pi r)) / 2
//                     r > 1: 0
//         r        = sqrt( (x - xr)**2 + (y - yr)**2 + (z - zr)**2 )
//   Exner Pressure:
//     Pi = PiBar + deltaPi
//       PiBar      = g**2 (exp( - N**2 z / g ) - 1) / (Cp Theta0 N**2)
//       deltaPi    = 0 (hydrostatic balance)
//   Velocity/Momentum:
//     Ui = ui = 0
//
// Conversion to Conserved Variables:
//   rho = P0 Pi**(Cv/Rd) / (Rd Theta)
//   E   = rho (Cv Theta Pi + (u u)/2 + g z)
//
// Constants:
//   Theta0          ,  Potential temperature constant
//   ThetaC          ,  Potential temperature perturbation
//   P0              ,  Pressure at the surface
//   N               ,  Brunt-Vaisala frequency
//   Cv              ,  Specific heat, constant volume
//   Cp              ,  Specific heat, constant pressure
//   Rd     = Cp - Cv,  Specific heat difference
//   g               ,  Gravity
//
// *****************************************************************************
static int ICs(void *ctx, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {

  #ifndef M_PI
  #define M_PI    3.14159265358979323846
  #endif

  // Inputs
  const CeedScalar *x = in[0];
  // Outputs
  CeedScalar *q0 = out[0];
  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar Theta0     = context[0];
  const CeedScalar ThetaC     = context[1];
  const CeedScalar P0         = context[2];
  const CeedScalar N          = context[3];
  const CeedScalar Cv         = context[4];
  const CeedScalar Cp         = context[5];
  const CeedScalar Rd         = context[6];
  const CeedScalar g          = context[7];

  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Potential temperature, density current
//    const CeedScalar r = sqrt(pow((x[i+Q*0] - 0.5)/4, 2) +
//                              pow((x[i+Q*1] - 0.5)/4, 2)) +
//                              pow((x[i+Q*2] - 0.5)/4, 2));
    const CeedScalar r2 = sqrt(pow((x[i+Q*0] - 0.25), 2) +
                              pow((x[i+Q*1] - 0.5), 2) +
                              pow((x[i+Q*2] - 0.5), 2));
//    const CeedScalar deltaTheta = r<= 1. ? ThetaC*(1 + cos(M_PI*r))/2 : 0;
//    const CeedScalar Theta = Theta0*exp(N*N*x[i+Q*2]/g) + deltaTheta;
    // -- Exner pressure, hydrostatic balance
//    const CeedScalar Pi = 1. + g*g*(exp(-N*N*x[i+Q*2]/g) - 1.) / (Cp*Theta0*N*N);
    // -- Density
//    const CeedScalar rho = P0 * pow(Pi, Cv/Rd) / (Rd*Theta);

    // Initial Conditions
    q0[i+0*Q] = 1.;//rho;
    q0[i+1*Q] = 0.0;//-0.5*(x[i+Q*1] - 0.5);//0.0;
    q0[i+2*Q] = 0.01;//0.5*(x[i+Q*0] - 0.5);//.01;
    q0[i+3*Q] = 0.0;
    q0[i+4*Q] = r2 <= 1./8. ? (1.-8.*r2) : 0.;//rho * (Cv*Theta*Pi + g*x[i+Q*2]);

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction implements the following formulation of Navier-Stokes
//
// This is 3D compressible Navier-Stokes in conservation form with state
//   variables of density, momentum, and total energy.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Density
//   Ui  - Momentum    ,  Ui = rho ui
//   E   - Total Energy,  E  = rho Cv T + rho (u u) / 2 + rho g z
//
// Navier-Stokes Equations:
//   drho/dt + div( U )                               = 0
//   dU/dt   + div( rho (u x u) + P I3 ) + rho g khat = div( Fu )
//   dE/dt   + div( (E + P) u )                       = div( Fe )
//
// Viscous Fluxes:
//   Fu = mu (grad( u ) + grad( u )^T + lambda div ( u ) I3)
//   Fe = u Fu + k grad( T )
//
// Equation of State:
//   P = (gamma - 1) (E - rho (u u) / 2 - rho g z)
//
// Temperature:
//   T = (E / rho - (u u) / 2 - g z) / Cv
//
// Constants:
//   lambda = - 2 / 3,  From Stokes hypothesis
//   mu              ,  Dynamic viscosity
//   k               ,  Thermal conductivity
//   Cv              ,  Specific heat, constant volume
//   Cp              ,  Specific heat, constant pressure
//   g               ,  Gravity
//   gamma  = Cp / Cv,  Specific heat ratio
//
// *****************************************************************************
static int NS(void *ctx, CeedInt Q,
                const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *q = in[0], *dq = in[1], *qdata = in[2], *x = in[3];
  // Outputs
  CeedScalar *v = out[0], *dv = out[1];
  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar lambda     = context[0];
  const CeedScalar mu         = context[1];
  const CeedScalar k          = context[2];
  const CeedScalar Cv         = context[3];
  const CeedScalar Cp         = context[4];
  const CeedScalar g          = context[5];
  const CeedScalar gamma      = Cp / Cv;

  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho     =   q[i+0*Q];
    const CeedScalar u[3]    = { q[i+1*Q] / rho,
                                 q[i+2*Q] / rho,
                                 q[i+3*Q] / rho };
    const CeedScalar E       =   q[i+4*Q];
    // -- Grad in
    const CeedScalar drho[3] = {  dq[i+(0+5*0)*Q],
                                  dq[i+(0+5*1)*Q],
                                  dq[i+(0+5*2)*Q] };
    const CeedScalar du[9]   = { (dq[i+(1+5*0)*Q] - drho[0]*u[0]) / rho,
                                 (dq[i+(1+5*1)*Q] - drho[1]*u[0]) / rho,
                                 (dq[i+(1+5*2)*Q] - drho[2]*u[0]) / rho,
                                 (dq[i+(2+5*0)*Q] - drho[0]*u[1]) / rho,
                                 (dq[i+(2+5*1)*Q] - drho[1]*u[1]) / rho,
                                 (dq[i+(2+5*2)*Q] - drho[2]*u[1]) / rho,
                                 (dq[i+(3+5*0)*Q] - drho[0]*u[2]) / rho,
                                 (dq[i+(3+5*1)*Q] - drho[1]*u[2]) / rho,
                                 (dq[i+(3+5*2)*Q] - drho[2]*u[2]) / rho };
    const CeedScalar dE[3]   = {  dq[i+(4+5*0)*Q],
                                  dq[i+(4+5*1)*Q],
                                  dq[i+(4+5*2)*Q] };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ       =   qdata[i+ 0*Q];
    // -- Interp-to-Grad qdata
    //      Symmetric 3x3 matrix
    const CeedScalar wBJ[9]   = { qdata[i+ 1*Q],
                                  qdata[i+ 2*Q],
                                  qdata[i+ 3*Q],
                                  qdata[i+ 4*Q],
                                  qdata[i+ 5*Q],
                                  qdata[i+ 6*Q],
                                  qdata[i+ 7*Q],
                                  qdata[i+ 8*Q],
                                  qdata[i+ 9*Q] };
    // -- Grad-to-Grad qdata
    const CeedScalar wBBJ[6]  = { qdata[i+10*Q],
                                  qdata[i+11*Q],
                                  qdata[i+12*Q],
                                  qdata[i+13*Q],
                                  qdata[i+14*Q],
                                  qdata[i+15*Q] };
    // -- gradT
    const CeedScalar gradT[3] = { (dE[0]/rho - E*drho[0]/(rho*rho) -
                                    (u[0]*du[0+3*0] + u[1]*du[1+3*0] +
                                     u[2]*du[2+3*0])) / Cv,
                                  (dE[1]/rho - E*drho[1]/(rho*rho) -
                                    (u[0]*du[0+3*1] + u[1]*du[1+3*1] +
                                     u[2]*du[2+3*1])) / Cv,
                                  (dE[2]/rho - E*drho[2]/(rho*rho) -
                                    (u[0]*du[0+3*2] + u[1]*du[1+3*2] +
                                     u[2]*du[2+3*2]) - g) / Cv };
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
                                      lambda * (du[0+3*0] + du[1+3*1])) };

    // -- Fevisc
    const CeedScalar Fe[3] = { u[0]*Fu[0] + u[1]*Fu[1] + u[2]*Fu[2] +
                                 k * gradT[0],
                               u[0]*Fu[1] + u[1]*Fu[3] + u[2]*Fu[4] +
                                 k * gradT[1],
                               u[0]*Fu[2] + u[1]*Fu[4] + u[2]*Fu[5] +
                                 k * gradT[2] };
    // -- P
    const CeedScalar P = (E - (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])*rho/2 -
                               rho*g*x[i+Q*2]) * (gamma - 1);


    // The Physics

    // -- Density
    // ---- u rho
//    dv[i+(0+5*0)*Q]  = rho*u[0]*wBJ[0] + rho*u[1]*wBJ[1] + rho*u[2]*wBJ[2];
//    dv[i+(0+5*1)*Q]  = rho*u[0]*wBJ[3] + rho*u[1]*wBJ[4] + rho*u[2]*wBJ[5];
//    dv[i+(0+5*2)*Q]  = rho*u[0]*wBJ[6] + rho*u[1]*wBJ[7] + rho*u[2]*wBJ[8];

    // -- Momentum
    // ---- rho (u x u) + P I3
//    v[i+(1+5*0)*Q]  = (rho*u[0]*u[0]+P)*wBJ[0] + rho*u[0]*u[1]*wBJ[1] +
//                        rho*u[0]*u[2]*wBJ[2];
//    dv[i+(1+5*1)*Q]  = (rho*u[0]*u[0]+P)*wBJ[3] + rho*u[0]*u[1]*wBJ[4] +
//                        rho*u[0]*u[2]*wBJ[5];
//    dv[i+(1+5*2)*Q]  = (rho*u[0]*u[0]+P)*wBJ[6] + rho*u[0]*u[1]*wBJ[7] +
//                        rho*u[0]*u[2]*wBJ[8];
//    dv[i+(2+5*0)*Q]  =  rho*u[1]*u[0]*wBJ[0] +   (rho*u[1]*u[1]+P)*wBJ[1] +
//                        rho*u[1]*u[2]*wBJ[2];
//    dv[i+(2+5*1)*Q]  =  rho*u[1]*u[0]*wBJ[3] +   (rho*u[1]*u[1]+P)*wBJ[4] +
//                        rho*u[1]*u[2]*wBJ[5];
//    dv[i+(2+5*2)*Q]  =  rho*u[1]*u[0]*wBJ[6] +   (rho*u[1]*u[1]+P)*wBJ[7] +
//                        rho*u[1]*u[2]*wBJ[8];
//    dv[i+(3+5*0)*Q]  =  rho*u[2]*u[0]*wBJ[0] +    rho*u[2]*u[1]*wBJ[1] +
//                       (rho*u[2]*u[2]+P)*wBJ[2];
//    dv[i+(3+5*1)*Q]  =  rho*u[2]*u[0]*wBJ[3] +    rho*u[2]*u[1]*wBJ[4] +
//                       (rho*u[2]*u[2]+P)*wBJ[5];
//    dv[i+(3+5*2)*Q]  =  rho*u[2]*u[0]*wBJ[6] +    rho*u[2]*u[1]*wBJ[7] +
//                       (rho*u[2]*u[2]+P)*wBJ[8];
    // ---- Fuvisc
//    dv[i+(1+5*0)*Q] -= Fu[0]*wBBJ[0] + Fu[1]*wBBJ[1] + Fu[2]*wBBJ[2];
//    dv[i+(1+5*1)*Q] -= Fu[0]*wBBJ[1] + Fu[1]*wBBJ[3] + Fu[2]*wBBJ[4];
//    dv[i+(1+5*2)*Q] -= Fu[0]*wBBJ[2] + Fu[1]*wBBJ[4] + Fu[2]*wBBJ[5];
//    dv[i+(2+5*0)*Q] -= Fu[1]*wBBJ[0] + Fu[3]*wBBJ[1] + Fu[4]*wBBJ[2];
//    dv[i+(2+5*1)*Q] -= Fu[1]*wBBJ[1] + Fu[3]*wBBJ[3] + Fu[4]*wBBJ[4];
//    dv[i+(2+5*2)*Q] -= Fu[1]*wBBJ[2] + Fu[3]*wBBJ[4] + Fu[4]*wBBJ[5];
//    dv[i+(3+5*0)*Q] -= Fu[2]*wBBJ[0] + Fu[4]*wBBJ[1] + Fu[5]*wBBJ[2];
//    dv[i+(3+5*1)*Q] -= Fu[2]*wBBJ[1] + Fu[4]*wBBJ[3] + Fu[5]*wBBJ[4];
//    dv[i+(3+5*2)*Q] -= Fu[2]*wBBJ[2] + Fu[4]*wBBJ[4] + Fu[5]*wBBJ[5];
    // ---- -rho g khat
//    v[i+3*Q] = - rho*g*wJ;
    for (int c=0; c<5; c++) {
      v[c*Q+i] = 0;
      for (int d=0; d<3; d++)
        dv[(d*5+c)*Q+i] = 0;
    }

    // -- Total Energy
    // ---- (E + P) u
    if (1) {
    dv[i+(4+5*0)*Q]  = (E + 0)*(u[0]*wBJ[0] + u[1]*wBJ[1] + u[2]*wBJ[2]);
    dv[i+(4+5*1)*Q]  = (E + 0)*(u[0]*wBJ[3] + u[1]*wBJ[4] + u[2]*wBJ[5]);
    dv[i+(4+5*2)*Q]  = (E + 0)*(u[0]*wBJ[6] + u[1]*wBJ[7] + u[2]*wBJ[8]);
    }
    //CeedScalar grad_E[3];

    if (0) {
    for (int d=0; d<1; d++) {
      //grad_E[d] = (1/wJ) * wBJ[d*3+d] * dq[(d*5+4)*Q+i];
      // u . grad(E)
      //v[4*Q+i] -= u[d] * wBJ[d*3+d] * dq[(d*5+4)*Q+i];
      v[4*Q+i] -= 1. * wBJ[d*3+d] * dq[(d*5+4)*Q+i];
    }
    }
    // ---- Fevisc
//    dv[i+(4+5*0)*Q] -= Fe[0]*wBBJ[0] + Fe[1]*wBBJ[1] + Fe[2]*wBBJ[2];
//    dv[i+(4+5*1)*Q] -= Fe[0]*wBBJ[1] + Fe[1]*wBBJ[3] + Fe[2]*wBBJ[4];
//    dv[i+(4+5*2)*Q] -= Fe[0]*wBBJ[2] + Fe[1]*wBBJ[4] + Fe[2]*wBBJ[5];

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
