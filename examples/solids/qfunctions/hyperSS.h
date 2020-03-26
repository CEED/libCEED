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
/// Hyperelasticity, small strain for solid mechanics example using PETSc

#ifndef HYPER_SS_H
#define HYPER_SS_H

#ifndef __CUDACC__
#  include <math.h>
#endif

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  PetscScalar   nu;      // Poisson's ratio
  PetscScalar   E;       // Young's Modulus
};
#endif

// -----------------------------------------------------------------------------
// Series approximation of log1p()
//  log1p() is not vectorized in libc
//
//  The series expansion is accurate to 1e-7 in the range sqrt(2)/2 < J < sqrt(2),
//  with machine precision accuracy near J=1.  The initialization extends this range
//  to 0.35 ~= sqrt(2)/4 < J < sqrt(2)*2 ~= 2.83, which should be sufficient for
//  applications of the Neo-Hookean model.
// -----------------------------------------------------------------------------
static inline CeedScalar log1p_series(CeedScalar x) {
  CeedScalar sum = 0;
  CeedScalar y = x / (2. + x);
  const CeedScalar y2 = y*y;
  sum += y;
  y *= y2;
  sum += y / 3;
  y *= y2;
  sum += y / 5;
  y *= y2;
  sum += y / 7;
  return 2 * sum;
};

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, small strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(HyperSSF)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][Q] = (CeedScalar(*)[3][Q])in[0],
                   (*qdata)[Q] = (CeedScalar(*)[Q])in[1];

  // Outputs
  CeedScalar (*dvdX)[3][Q] = (CeedScalar(*)[3][Q])out[0];
  // Store gradu for HyperFSdF (Jacobian of HyperFSF)
  CeedScalar (*gradu)[3][Q] = (CeedScalar(*)[3][Q])out[1];
  // *INDENT-ON*

  // Context
  const Physics context = ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;

  // Constants
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3]   = {{ug[0][0][i],
                                    ug[1][0][i],
                                    ug[2][0][i]},
                                   {ug[0][1][i],
                                    ug[1][1][i],
                                    ug[2][1][i]},
                                   {ug[0][2][i],
                                    ug[1][2][i],
                                    ug[2][2][i]}
                                  };
    // -- Qdata
    const CeedScalar wJ         =   qdata[0][i];
    const CeedScalar dXdx[3][3] = {{qdata[1][i],
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

    // Compute gradu
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = gradu
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        gradu[j][k][i] = 0;
        for (int m = 0; m < 3; m++)
          gradu[j][k][i] += dXdx[m][k] * du[j][m];
      }

    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)
    const CeedScalar e00 = (gradu[0][0][i] + gradu[0][0][i])/2.,
                     e01 = (gradu[0][1][i] + gradu[1][0][i])/2.,
                     e02 = (gradu[0][2][i] + gradu[2][0][i])/2.,
                     e11 = (gradu[1][1][i] + gradu[1][1][i])/2.,
                     e12 = (gradu[1][2][i] + gradu[2][1][i])/2.,
                     e22 = (gradu[2][2][i] + gradu[2][2][i])/2.;
    // *INDENT-OFF*
    const CeedScalar e[3][3] =  {{e00, e01, e02},
                                 {e01, e11, e12},
                                 {e02, e12, e22}
                                };
    // *INDENT-ON*

    // strain (epsilon)
    //    and
    // stress (sigma) in Voigt notation:
    //           [e00]              [sigma00]
    //           [e11]              [sigma11]
    // epsilon = [e22]  ,   sigma = [sigma22]
    //           [e12]              [sigma12]
    //           [e02]              [sigma02]
    //           [e01]              [sigma01]
    //
    // mu = E / (2 * (1 + nu))
    // bulk modulus = E / (2 * (1 - 2 * nu))
    // lambda = (3 * bulk modulus - 2 * mu) / 3
    // e_v = volumetric strain = e00 + e11 + e22
    //
    // sigma = lambda * log(1 + e_v) + 2 * mu * epsilon
    //
    // Above Voigt Notation is placed in a 3x3 matrix:
    // Volumetric strain
    const CeedScalar strain_vol = e[0][0] + e[1][1] + e[2][2];
    const CeedScalar llv = log1p_series(strain_vol);
    const CeedScalar sigma00 = lambda*llv + TwoMu*e[0][0],
                     sigma11 = lambda*llv + TwoMu*e[1][1],
                     sigma22 = lambda*llv + TwoMu*e[2][2],
                     sigma12 = TwoMu*e[1][2],
                     sigma02 = TwoMu*e[0][2],
                     sigma01 = TwoMu*e[0][1];
    // *INDENT-OFF*
    const CeedScalar sigma[3][3] = {{sigma00, sigma01, sigma02},
                                    {sigma01, sigma11, sigma12},
                                    {sigma02, sigma12, sigma22}
                                   };
    // *INDENT-ON*

    // Apply dXdx^T and weight to sigma
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        dvdX[k][j][i] = 0;
        for (int m = 0; m < 3; m++)
          dvdX[k][j][i] += dXdx[k][m] * sigma[j][m] * wJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for hyperelasticity, small strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(HyperSSdF)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                          CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*deltaug)[3][Q] = (CeedScalar(*)[3][Q])in[0],
                   (*qdata)[Q] = (CeedScalar(*)[Q])in[1];
  // gradu is used for hyperelasticity (non-linear)
  const CeedScalar (*gradu)[3][Q] = (CeedScalar(*)[3][Q])in[2];

  // Outputs
  CeedScalar (*deltadvdX)[3][Q] = (CeedScalar(*)[3][Q])out[0];
  // *INDENT-ON*

  // Context
  const Physics context = ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;

  // Constants
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar deltadu[3][3] = {{deltaug[0][0][i],
                                       deltaug[1][0][i],
                                       deltaug[2][0][i]},
                                      {deltaug[0][1][i],
                                       deltaug[1][1][i],
                                       deltaug[2][1][i]},
                                      {deltaug[0][2][i],
                                       deltaug[1][2][i],
                                       deltaug[2][2][i]}
                                     };
    // -- Qdata
    const CeedScalar wJ         =      qdata[0][i];
    const CeedScalar dXdx[3][3] =    {{qdata[1][i],
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

    // Compute graddeltau
    // Apply dXdx^-1 to deltadu = graddeltau
    CeedScalar graddeltau[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        graddeltau[j][k] = 0;
        for (int m = 0; m < 3; m++)
          graddeltau[j][k] += dXdx[m][k] * deltadu[j][m];
      }

    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)
    const CeedScalar de00 = (graddeltau[0][0] + graddeltau[0][0])/2.,
                     de01 = (graddeltau[0][1] + graddeltau[1][0])/2.,
                     de02 = (graddeltau[0][2] + graddeltau[2][0])/2.,
                     de11 = (graddeltau[1][1] + graddeltau[1][1])/2.,
                     de12 = (graddeltau[1][2] + graddeltau[2][1])/2.,
                     de22 = (graddeltau[2][2] + graddeltau[2][2])/2.;
    // *INDENT-OFF*
    const CeedScalar de[3][3] =  {{de00, de01, de02},
                                  {de01, de11, de12},
                                  {de02, de12, de22}
                                 };
    // *INDENT-ON*

    //strain (epsilon)
    //    and
    //stress (sigma) in Voigt notation:
    //            [e00]               [sigma00]
    //            [e11]               [sigma11]
    // depsilon = [e22]  ,   dsigma = [sigma22]
    //            [e12]               [sigma12]
    //            [e02]               [sigma02]
    //            [e01]               [sigma01]
    //
    // mu = E / (2 * (1 + nu))
    // bulk modulus = E / (2 * (1 - 2 * nu))
    // lambda = (3 * bulk modulus - 2 * mu) / 3
    // e_v = volumetric strain = e00 + e11 + e22
    // lambda bar = lambda / (1 + e_v)
    //
    // dSigma = S * epsilon
    //
    // S_ijkl = lambda bar * delta_ij * delta_kl + 2 * mu * delta_ik * delta_jl
    //
    // Matrix form:
    //
    //     [2 mu + lambda bar     lambda bar         lambda bar                       ]
    //     [   lambda bar      2 mu + lambda bar     lambda bar                       ]
    //     [   lambda bar         lambda bar      2 mu + lambda bar                   ]
    // S = [                                                           mu             ]
    //     [                                                                 mu       ]
    //     [                                                                       mu ]
    //
    // Above Voigt Notation is placed in a 3x3 matrix:
    const CeedScalar strain_vol = gradu[0][0][i]+gradu[1][1][i]+gradu[2][2][i];
    const CeedScalar lambda_bar = lambda / (1 + strain_vol);
    const CeedScalar lambda_dtrace = lambda_bar*(de[0][0]+de[1][1]+de[2][2]);
    const CeedScalar dsigma00 = lambda_dtrace + TwoMu*de[0][0],
                     dsigma11 = lambda_dtrace + TwoMu*de[1][1],
                     dsigma22 = lambda_dtrace + TwoMu*de[2][2],
                     dsigma12 = TwoMu*de[1][2],
                     dsigma02 = TwoMu*de[0][2],
                     dsigma01 = TwoMu*de[0][1];
    // *INDENT-OFF*
    const CeedScalar dsigma[3][3] = {{dsigma00, dsigma01, dsigma02},
                                     {dsigma01, dsigma11, dsigma12},
                                     {dsigma02, dsigma12, dsigma22}
                                    };
    // *INDENT-ON*

    // Apply dXdx^-T and weight
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        deltadvdX[k][j][i] = 0;
        for (int m = 0; m < 3; m++)
          deltadvdX[k][j][i] += dXdx[k][m] * dsigma[j][m] * wJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------
#endif // End of HYPER_SS_H
