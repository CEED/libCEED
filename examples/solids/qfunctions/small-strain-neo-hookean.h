// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Hyperelasticity, small strain for solid mechanics example using PETSc

#ifndef ELAS_SS_NH_H
#define ELAS_SS_NH_H

#include <ceed.h>
#include <math.h>

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  CeedScalar nu;  // Poisson's ratio
  CeedScalar E;   // Young's Modulus
};
#endif

// -----------------------------------------------------------------------------
// Series approximation of log1p()
//  log1p() is not vectorized in libc
//
//  The series expansion is accurate to 1e-7 in the range sqrt(2)/2 < J < sqrt(2),
//  with machine precision accuracy near J=1.
// -----------------------------------------------------------------------------
#ifndef LOG1P_SERIES
#define LOG1P_SERIES
CEED_QFUNCTION_HELPER CeedScalar log1p_series(CeedScalar x) {
  CeedScalar       sum = 0;
  CeedScalar       y   = x / (2. + x);
  const CeedScalar y2  = y * y;
  sum += y;
  y *= y2;
  sum += y / 3;
  y *= y2;
  sum += y / 5;
  y *= y2;
  sum += y / 7;
  return 2 * sum;
};
#endif

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, small strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasSSNHF)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar(*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // Store grad_u for HyperFSdF (Jacobian of HyperFSF)
  CeedScalar(*grad_u)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  const Physics    context = (Physics)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;

  // Constants
  const CeedScalar TwoMu  = E / (1 + nu);
  const CeedScalar Kbulk  = E / (3 * (1 - 2 * nu));  // Bulk modulus
  const CeedScalar lambda = (3 * Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3] = {
        {ug[0][0][i], ug[1][0][i], ug[2][0][i]},
        {ug[0][1][i], ug[1][1][i], ug[2][1][i]},
        {ug[0][2][i], ug[1][2][i], ug[2][2][i]}
    };
    // -- Qdata
    const CeedScalar wdetJ      = q_data[0][i];
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    for (int j = 0; j < 3; j++) {    // Component
      for (int k = 0; k < 3; k++) {  // Derivative
        grad_u[j][k][i] = 0;
        for (int m = 0; m < 3; m++) grad_u[j][k][i] += dXdx[m][k] * du[j][m];
      }
    }

    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)
    const CeedScalar e00 = (grad_u[0][0][i] + grad_u[0][0][i]) / 2., e01 = (grad_u[0][1][i] + grad_u[1][0][i]) / 2.,
                     e02 = (grad_u[0][2][i] + grad_u[2][0][i]) / 2., e11 = (grad_u[1][1][i] + grad_u[1][1][i]) / 2.,
                     e12 = (grad_u[1][2][i] + grad_u[2][1][i]) / 2., e22 = (grad_u[2][2][i] + grad_u[2][2][i]) / 2.;
    // *INDENT-OFF*
    const CeedScalar e[3][3] = {
        {e00, e01, e02},
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
    const CeedScalar llv        = log1p_series(strain_vol);
    const CeedScalar sigma00 = lambda * llv + TwoMu * e[0][0], sigma11 = lambda * llv + TwoMu * e[1][1], sigma22 = lambda * llv + TwoMu * e[2][2],
                     sigma12 = TwoMu * e[1][2], sigma02 = TwoMu * e[0][2], sigma01 = TwoMu * e[0][1];
    // *INDENT-OFF*
    const CeedScalar sigma[3][3] = {
        {sigma00, sigma01, sigma02},
        {sigma01, sigma11, sigma12},
        {sigma02, sigma12, sigma22}
    };
    // *INDENT-ON*

    // Apply dXdx^T and weight to sigma
    for (int j = 0; j < 3; j++) {    // Component
      for (int k = 0; k < 3; k++) {  // Derivative
        dvdX[k][j][i] = 0;
        for (int m = 0; m < 3; m++) dvdX[k][j][i] += dXdx[k][m] * sigma[j][m] * wdetJ;
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for hyperelasticity, small strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasSSNHdF)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*deltaug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
        (*q_data)[CEED_Q_VLA]               = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // grad_u is used for hyperelasticity (non-linear)
  const CeedScalar(*grad_u)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar(*deltadvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics    context = (Physics)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;

  // Constants
  const CeedScalar TwoMu  = E / (1 + nu);
  const CeedScalar Kbulk  = E / (3 * (1 - 2 * nu));  // Bulk modulus
  const CeedScalar lambda = (3 * Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar deltadu[3][3] = {
        {deltaug[0][0][i], deltaug[1][0][i], deltaug[2][0][i]},
        {deltaug[0][1][i], deltaug[1][1][i], deltaug[2][1][i]},
        {deltaug[0][2][i], deltaug[1][2][i], deltaug[2][2][i]}
    };
    // -- Qdata
    const CeedScalar wdetJ      = q_data[0][i];
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*

    // Compute graddeltau
    // Apply dXdx^-1 to deltadu = graddeltau
    CeedScalar graddeltau[3][3];
    for (int j = 0; j < 3; j++) {    // Component
      for (int k = 0; k < 3; k++) {  // Derivative
        graddeltau[j][k] = 0;
        for (int m = 0; m < 3; m++) graddeltau[j][k] += dXdx[m][k] * deltadu[j][m];
      }
    }

    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)
    const CeedScalar de00 = (graddeltau[0][0] + graddeltau[0][0]) / 2., de01 = (graddeltau[0][1] + graddeltau[1][0]) / 2.,
                     de02 = (graddeltau[0][2] + graddeltau[2][0]) / 2., de11 = (graddeltau[1][1] + graddeltau[1][1]) / 2.,
                     de12 = (graddeltau[1][2] + graddeltau[2][1]) / 2., de22 = (graddeltau[2][2] + graddeltau[2][2]) / 2.;
    // *INDENT-OFF*
    const CeedScalar de[3][3] = {
        {de00, de01, de02},
        {de01, de11, de12},
        {de02, de12, de22}
    };
    // *INDENT-ON*

    // strain (epsilon)
    //     and
    // stress (sigma) in Voigt notation:
    //             [e00]               [sigma00]
    //             [e11]               [sigma11]
    //  depsilon = [e22]  ,   dsigma = [sigma22]
    //             [e12]               [sigma12]
    //             [e02]               [sigma02]
    //             [e01]               [sigma01]
    //
    //  mu = E / (2 * (1 + nu))
    //  bulk modulus = E / (2 * (1 - 2 * nu))
    //  lambda = (3 * bulk modulus - 2 * mu) / 3
    //  e_v = volumetric strain = e00 + e11 + e22
    //  lambda bar = lambda / (1 + e_v)
    //
    //  dSigma = S * epsilon
    //
    //  S_ijkl = lambda bar * delta_ij * delta_kl + 2 * mu * delta_ik * delta_jl
    //
    //  Matrix form:
    //
    //      [2 mu + lambda bar     lambda bar         lambda bar                       ]
    //      [   lambda bar      2 mu + lambda bar     lambda bar                       ]
    //      [   lambda bar         lambda bar      2 mu + lambda bar                   ]
    //  S = [                                                           mu             ]
    //      [                                                                 mu       ]
    //      [                                                                       mu ]
    //
    //  Above Voigt Notation is placed in a 3x3 matrix:
    const CeedScalar strain_vol    = grad_u[0][0][i] + grad_u[1][1][i] + grad_u[2][2][i];
    const CeedScalar lambda_bar    = lambda / (1 + strain_vol);
    const CeedScalar lambda_dtrace = lambda_bar * (de[0][0] + de[1][1] + de[2][2]);
    const CeedScalar dsigma00 = lambda_dtrace + TwoMu * de[0][0], dsigma11 = lambda_dtrace + TwoMu * de[1][1],
                     dsigma22 = lambda_dtrace + TwoMu * de[2][2], dsigma12 = TwoMu * de[1][2], dsigma02 = TwoMu * de[0][2],
                     dsigma01 = TwoMu * de[0][1];
    // *INDENT-OFF*
    const CeedScalar dsigma[3][3] = {
        {dsigma00, dsigma01, dsigma02},
        {dsigma01, dsigma11, dsigma12},
        {dsigma02, dsigma12, dsigma22}
    };
    // *INDENT-ON*

    // Apply dXdx^-T and weight
    for (int j = 0; j < 3; j++) {    // Component
      for (int k = 0; k < 3; k++) {  // Derivative
        deltadvdX[k][j][i] = 0;
        for (int m = 0; m < 3; m++) deltadvdX[k][j][i] += dXdx[k][m] * dsigma[j][m] * wdetJ;
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Strain energy computation for hyperelasticity, small strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasSSNHEnergy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar(*energy) = (CeedScalar(*))out[0];
  // *INDENT-ON*

  // Context
  const Physics    context = (Physics)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;

  // Constants
  const CeedScalar TwoMu  = E / (1 + nu);
  const CeedScalar mu     = TwoMu / 2;
  const CeedScalar Kbulk  = E / (3 * (1 - 2 * nu));  // Bulk Modulus
  const CeedScalar lambda = (3 * Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3] = {
        {ug[0][0][i], ug[1][0][i], ug[2][0][i]},
        {ug[0][1][i], ug[1][1][i], ug[2][1][i]},
        {ug[0][2][i], ug[1][2][i], ug[2][2][i]}
    };
    // -- Qdata
    const CeedScalar wdetJ      = q_data[0][i];
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    CeedScalar grad_u[3][3];
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        grad_u[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) grad_u[j][k] += dXdx[m][k] * du[j][m];
      }
    }

    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)

    // *INDENT-OFF*
    const CeedScalar e[3][3] = {
        {(grad_u[0][0] + grad_u[0][0]) / 2., (grad_u[0][1] + grad_u[1][0]) / 2., (grad_u[0][2] + grad_u[2][0]) / 2.},
        {(grad_u[1][0] + grad_u[0][1]) / 2., (grad_u[1][1] + grad_u[1][1]) / 2., (grad_u[1][2] + grad_u[2][1]) / 2.},
        {(grad_u[2][0] + grad_u[0][2]) / 2., (grad_u[2][1] + grad_u[1][2]) / 2., (grad_u[2][2] + grad_u[2][2]) / 2.}
    };
    // *INDENT-ON*

    // Strain Energy
    const CeedScalar strain_vol = e[0][0] + e[1][1] + e[2][2];
    const CeedScalar llv        = log1p_series(strain_vol);
    energy[i] =
        (lambda * (1 + strain_vol) * (llv - 1) + strain_vol * mu + (e[0][1] * e[0][1] + e[0][2] * e[0][2] + e[1][2] * e[1][2]) * 2 * mu) * wdetJ;

  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Nodal diagnostic quantities for hyperelasticity, small strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasSSNHDiagnostic)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
        (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar(*diagnostic)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics    context = (Physics)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;

  // Constants
  const CeedScalar TwoMu  = E / (1 + nu);
  const CeedScalar mu     = TwoMu / 2;
  const CeedScalar Kbulk  = E / (3 * (1 - 2 * nu));  // Bulk Modulus
  const CeedScalar lambda = (3 * Kbulk - TwoMu) / 3;

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Read spatial derivatives of u
    // *INDENT-OFF*
    const CeedScalar du[3][3] = {
        {ug[0][0][i], ug[1][0][i], ug[2][0][i]},
        {ug[0][1][i], ug[1][1][i], ug[2][1][i]},
        {ug[0][2][i], ug[1][2][i], ug[2][2][i]}
    };
    // -- Qdata
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    CeedScalar grad_u[3][3];
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        grad_u[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) grad_u[j][k] += dXdx[m][k] * du[j][m];
      }
    }

    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)

    // *INDENT-OFF*
    const CeedScalar e[3][3] = {
        {(grad_u[0][0] + grad_u[0][0]) / 2., (grad_u[0][1] + grad_u[1][0]) / 2., (grad_u[0][2] + grad_u[2][0]) / 2.},
        {(grad_u[1][0] + grad_u[0][1]) / 2., (grad_u[1][1] + grad_u[1][1]) / 2., (grad_u[1][2] + grad_u[2][1]) / 2.},
        {(grad_u[2][0] + grad_u[0][2]) / 2., (grad_u[2][1] + grad_u[1][2]) / 2., (grad_u[2][2] + grad_u[2][2]) / 2.}
    };
    // *INDENT-ON*

    // Displacement
    diagnostic[0][i] = u[0][i];
    diagnostic[1][i] = u[1][i];
    diagnostic[2][i] = u[2][i];

    // Pressure
    const CeedScalar strain_vol = e[0][0] + e[1][1] + e[2][2];
    const CeedScalar llv        = log1p_series(strain_vol);
    diagnostic[3][i]            = -lambda * llv;

    // Stress tensor invariants
    diagnostic[4][i] = strain_vol;
    diagnostic[5][i] = 0.;
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt m = 0; m < 3; m++) diagnostic[5][i] += e[j][m] * e[m][j];
    }
    diagnostic[6][i] = 1 + strain_vol;

    // Strain energy
    diagnostic[7][i] =
        (lambda * (1 + strain_vol) * (llv - 1) + strain_vol * mu + (e[0][1] * e[0][1] + e[0][2] * e[0][2] + e[1][2] * e[1][2]) * 2 * mu);
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // End of ELAS_SS_NH_H
