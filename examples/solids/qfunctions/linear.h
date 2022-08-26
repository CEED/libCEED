// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Linear elasticity for solid mechanics example using PETSc

#ifndef ELAS_LINEAR_H
#define ELAS_LINEAR_H

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
// Residual evaluation for linear elasticity
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasLinearF)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar(*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // grad_u not used for linear elasticity
  // (*grad_u)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  const Physics    context = (Physics)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;

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

    //
    // Formulation uses Voigt notation:
    //  stress (sigma)      strain (epsilon)
    //
    //    [sigma00]             [e00]
    //    [sigma11]             [e11]
    //    [sigma22]   =  S   *  [e22]
    //    [sigma12]             [e12]
    //    [sigma02]             [e02]
    //    [sigma01]             [e01]
    //
    //        where
    //                         [1-nu   nu    nu                                    ]
    //                         [ nu   1-nu   nu                                    ]
    //                         [ nu    nu   1-nu                                   ]
    // S = E/((1+nu)*(1-2*nu)) [                  (1-2*nu)/2                       ]
    //                         [                             (1-2*nu)/2            ]
    //                         [                                        (1-2*nu)/2 ]

    // Above Voigt Notation is placed in a 3x3 matrix:
    const CeedScalar ss      = E / ((1 + nu) * (1 - 2 * nu));
    const CeedScalar sigma00 = ss * ((1 - nu) * e[0][0] + nu * e[1][1] + nu * e[2][2]),
                     sigma11 = ss * (nu * e[0][0] + (1 - nu) * e[1][1] + nu * e[2][2]),
                     sigma22 = ss * (nu * e[0][0] + nu * e[1][1] + (1 - nu) * e[2][2]), sigma12 = ss * (1 - 2 * nu) * e[1][2] * 0.5,
                     sigma02 = ss * (1 - 2 * nu) * e[0][2] * 0.5, sigma01 = ss * (1 - 2 * nu) * e[0][1] * 0.5;
    // *INDENT-OFF*
    const CeedScalar sigma[3][3] = {
        {sigma00, sigma01, sigma02},
        {sigma01, sigma11, sigma12},
        {sigma02, sigma12, sigma22}
    };
    // *INDENT-ON*

    // Apply dXdx^T and weight to sigma
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        dvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++) dvdX[k][j][i] += dXdx[k][m] * sigma[j][m] * wdetJ;
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for linear elasticity
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasLineardF)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*deltaug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
        (*q_data)[CEED_Q_VLA]               = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // grad_u not used for linear elasticity
  // (*grad_u)[3][Q] = (CeedScalar(*)[3][Q])in[2];

  // Outputs
  CeedScalar(*deltadvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics    context = (Physics)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;

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
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to deltadu = graddeltau
    CeedScalar graddeltau[3][3];
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        graddeltau[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) graddeltau[j][k] += dXdx[m][k] * deltadu[j][m];
      }
    }

    // Compute Strain : e (epsilon)
    // e = 1/2 (grad u + (grad u)^T)
    // *INDENT-OFF*
    const CeedScalar de[3][3] = {
        {(graddeltau[0][0] + graddeltau[0][0]) / 2., (graddeltau[0][1] + graddeltau[1][0]) / 2., (graddeltau[0][2] + graddeltau[2][0]) / 2.},
        {(graddeltau[1][0] + graddeltau[0][1]) / 2., (graddeltau[1][1] + graddeltau[1][1]) / 2., (graddeltau[1][2] + graddeltau[2][1]) / 2.},
        {(graddeltau[2][0] + graddeltau[0][2]) / 2., (graddeltau[2][1] + graddeltau[1][2]) / 2., (graddeltau[2][2] + graddeltau[2][2]) / 2.}
    };

    // *INDENT-ON*
    // Formulation uses Voigt notation:
    //  stress (sigma)      strain (epsilon)
    //
    //    [dsigma00]             [de00]
    //    [dsigma11]             [de11]
    //    [dsigma22]   =  S   *  [de22]
    //    [dsigma12]             [de12]
    //    [dsigma02]             [de02]
    //    [dsigma01]             [de01]
    //
    //        where
    //                         [1-nu   nu    nu                                    ]
    //                         [ nu   1-nu   nu                                    ]
    //                         [ nu    nu   1-nu                                   ]
    // S = E/((1+nu)*(1-2*nu)) [                  (1-2*nu)/2                       ]
    //                         [                             (1-2*nu)/2            ]
    //                         [                                        (1-2*nu)/2 ]

    // Above Voigt Notation is placed in a 3x3 matrix:
    const CeedScalar ss       = E / ((1 + nu) * (1 - 2 * nu));
    const CeedScalar dsigma00 = ss * ((1 - nu) * de[0][0] + nu * de[1][1] + nu * de[2][2]),
                     dsigma11 = ss * (nu * de[0][0] + (1 - nu) * de[1][1] + nu * de[2][2]),
                     dsigma22 = ss * (nu * de[0][0] + nu * de[1][1] + (1 - nu) * de[2][2]), dsigma12 = ss * (1 - 2 * nu) * de[1][2] / 2,
                     dsigma02 = ss * (1 - 2 * nu) * de[0][2] / 2, dsigma01 = ss * (1 - 2 * nu) * de[0][1] / 2;
    // *INDENT-OFF*
    const CeedScalar dsigma[3][3] = {
        {dsigma00, dsigma01, dsigma02},
        {dsigma01, dsigma11, dsigma12},
        {dsigma02, dsigma12, dsigma22}
    };
    // *INDENT-ON*

    // Apply dXdx^T and weight
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        deltadvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++) deltadvdX[k][j][i] += dXdx[k][m] * dsigma[j][m] * wdetJ;
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Strain energy computation for linear elasticity
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasLinearEnergy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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

    // Strain energy
    const CeedScalar strain_vol = e[0][0] + e[1][1] + e[2][2];
    energy[i] =
        (lambda * strain_vol * strain_vol / 2. + strain_vol * mu + (e[0][1] * e[0][1] + e[0][2] * e[0][2] + e[1][2] * e[1][2]) * 2 * mu) * wdetJ;

  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Nodal diagnostic quantities for linear elasticity
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasLinearDiagnostic)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
    diagnostic[3][i]            = -lambda * strain_vol;

    // Stress tensor invariants
    diagnostic[4][i] = strain_vol;
    diagnostic[5][i] = 0.;
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt m = 0; m < 3; m++) diagnostic[5][i] += e[j][m] * e[m][j];
    }
    diagnostic[6][i] = 1 + strain_vol;

    // Strain energy
    diagnostic[7][i] =
        (lambda * strain_vol * strain_vol / 2. + strain_vol * mu + (e[0][1] * e[0][1] + e[0][2] * e[0][2] + e[1][2] * e[1][2]) * 2 * mu);
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // End of ELAS_LINEAR_H
