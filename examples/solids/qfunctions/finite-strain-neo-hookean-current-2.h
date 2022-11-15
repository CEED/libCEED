// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Hyperelasticity, finite strain for solid mechanics example using PETSc

#ifndef ELAS_FSCurrentNH2_H
#define ELAS_FSCurrentNH2_H

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
//  with machine precision accuracy near J=1.  The initialization extends this range
//  to 0.35 ~= sqrt(2)/4 < J < sqrt(2)*2 ~= 2.83, which should be sufficient for
//  applications of the Neo-Hookean model.
// -----------------------------------------------------------------------------
#ifndef LOG1P_SERIES_SHIFTED
#define LOG1P_SERIES_SHIFTED
CEED_QFUNCTION_HELPER CeedScalar log1p_series_shifted(CeedScalar x) {
  const CeedScalar left = sqrt(2.) / 2 - 1, right = sqrt(2.) - 1;
  CeedScalar       sum = 0;
  if (1) {           // Disable if the smaller range sqrt(2)/2 < J < sqrt(2) is sufficient
    if (x < left) {  // Replace if with while for arbitrary range (may hurt vectorization)
      sum -= log(2.) / 2;
      x = 1 + 2 * x;
    } else if (right < x) {
      sum += log(2.) / 2;
      x = (x - 1) / 2;
    }
  }
  CeedScalar       y  = x / (2. + x);
  const CeedScalar y2 = y * y;
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
// Compute det F - 1
// -----------------------------------------------------------------------------
#ifndef DETJM1
#define DETJM1
CEED_QFUNCTION_HELPER CeedScalar computeJM1(const CeedScalar grad_u[3][3]) {
  return grad_u[0][0] * (grad_u[1][1] * grad_u[2][2] - grad_u[1][2] * grad_u[2][1]) +
         grad_u[0][1] * (grad_u[1][2] * grad_u[2][0] - grad_u[1][0] * grad_u[2][2]) +
         grad_u[0][2] * (grad_u[1][0] * grad_u[2][1] - grad_u[2][0] * grad_u[1][1]) + grad_u[0][0] + grad_u[1][1] + grad_u[2][2] +
         grad_u[0][0] * grad_u[1][1] + grad_u[0][0] * grad_u[2][2] + grad_u[1][1] * grad_u[2][2] - grad_u[0][1] * grad_u[1][0] -
         grad_u[0][2] * grad_u[2][0] - grad_u[1][2] * grad_u[2][1];
};
#endif

// -----------------------------------------------------------------------------
// Compute matrix^(-1), where matrix is nonsymetric, returns array of 9
// -----------------------------------------------------------------------------
#ifndef MatinvNonSym
#define MatinvNonSym
CEED_QFUNCTION_HELPER int computeMatinvNonSym(const CeedScalar A[3][3], const CeedScalar detA, CeedScalar Ainv[9]) {
  // Compute A^(-1) : A-Inverse
  CeedScalar B[9] = {
      A[1][1] * A[2][2] - A[1][2] * A[2][1], /* *NOPAD* */
      A[0][0] * A[2][2] - A[0][2] * A[2][0], /* *NOPAD* */
      A[0][0] * A[1][1] - A[0][1] * A[1][0], /* *NOPAD* */
      A[0][2] * A[1][0] - A[0][0] * A[1][2], /* *NOPAD* */
      A[0][1] * A[1][2] - A[0][2] * A[1][1], /* *NOPAD* */
      A[0][2] * A[2][1] - A[0][1] * A[2][2], /* *NOPAD* */
      A[0][1] * A[2][0] - A[0][0] * A[2][1], /* *NOPAD* */
      A[1][0] * A[2][1] - A[1][1] * A[2][0], /* *NOPAD* */
      A[1][2] * A[2][0] - A[1][0] * A[2][2]  /* *NOPAD* */
  };
  for (CeedInt m = 0; m < 9; m++) Ainv[m] = B[m] / (detA);

  return 0;
};
#endif

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSCurrentNH2F)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar(*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // Store dXdx
  CeedScalar(*dXdx)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // Store tau
  CeedScalar(*tau)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[2];
  // Store constant lam_log_J = lambda*log(J)
  CeedScalar(*lam_log_J)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[3];
  // *INDENT-ON*

  // Context
  const Physics    context = (Physics)ctx;
  const CeedScalar E       = context->E;
  const CeedScalar nu      = context->nu;
  const CeedScalar TwoMu   = E / (1 + nu);
  const CeedScalar mu      = TwoMu / 2;
  const CeedScalar Kbulk   = E / (3 * (1 - 2 * nu));  // Bulk Modulus
  const CeedScalar lambda  = (3 * Kbulk - TwoMu) / 3;

  // Formulation Terminology:
  //  I3    : 3x3 Identity matrix
  //  b     : left Cauchy-Green tensor
  //  F     : deformation gradient
  //  tau   : Kirchhoff stress (in current config)
  // Formulation:
  //  F =  I3 + Grad_ue
  //  J = det(F)
  //  b = F*F^{T}
  //  tau = mu*b - (mu - lambda*log(J))*I3;

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
    const CeedScalar wdetJ = q_data[0][i];
    // dXdx_initial = dX/dx_initial
    // X is natural coordinate sys OR Reference [-1,1]^dim
    // x_initial is initial config coordinate system
    const CeedScalar dXdx_initial[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };

    // *INDENT-ON*
    // X is natural coordinate sys OR Reference system
    // x_initial is initial config coordinate system
    // Grad_u =du/dx_initial= du/dX * dX/dx_initial
    CeedScalar Grad_u[3][3];
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        Grad_u[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) Grad_u[j][k] += du[j][m] * dXdx_initial[m][k];
      }
    }

    // Compute The Deformation Gradient : F = I3 + Gradu
    // *INDENT-OFF*
    const CeedScalar F[3][3] = {
        {Grad_u[0][0] + 1, Grad_u[0][1],     Grad_u[0][2]    },
        {Grad_u[1][0],     Grad_u[1][1] + 1, Grad_u[1][2]    },
        {Grad_u[2][0],     Grad_u[2][1],     Grad_u[2][2] + 1}
    };

    // *INDENT-ON*
    // b - I3 = (Grad_u + Grad_u^T + Grad_u*Grad_u^T)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar    bMI3[6];
    for (CeedInt m = 0; m < 6; m++) {
      bMI3[m] = Grad_u[indj[m]][indk[m]] + Grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++) bMI3[m] += Grad_u[indj[m]][n] * Grad_u[indk[m]][n];
    }

    const CeedScalar Jm1  = computeJM1(Grad_u);
    const CeedScalar logJ = log1p_series_shifted(Jm1);

    // store lam_log_J = lambda*log(J)
    lam_log_J[0][i] = lambda * logJ;

    // tau = mu*b - Cc1*I3;
    tau[0][i] = mu * bMI3[0] + lam_log_J[0][i];
    tau[1][i] = mu * bMI3[1] + lam_log_J[0][i];
    tau[2][i] = mu * bMI3[2] + lam_log_J[0][i];
    tau[3][i] = mu * bMI3[3];
    tau[4][i] = mu * bMI3[4];
    tau[5][i] = mu * bMI3[5];

    // Computer F^{-1}
    const CeedScalar detF = Jm1 + 1.;
    CeedScalar       Finvwork[9];
    computeMatinvNonSym(F, detF, Finvwork);
    CeedScalar Finv[3][3];
    Finv[0][0] = Finvwork[0];
    Finv[0][1] = Finvwork[5];
    Finv[0][2] = Finvwork[4];
    Finv[1][0] = Finvwork[8];
    Finv[1][1] = Finvwork[1];
    Finv[1][2] = Finvwork[3];
    Finv[2][0] = Finvwork[7];
    Finv[2][1] = Finvwork[6];
    Finv[2][2] = Finvwork[2];

    // x is current config coordinate system
    // dXdx = dX/dx = dX/dx_initial * F^{-1}
    // Note that F^{-1} = dx_initial/dx
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        dXdx[j][k][i] = 0;
        for (CeedInt m = 0; m < 3; m++) dXdx[j][k][i] += dXdx_initial[j][m] * Finv[m][k];
      }
    }

    // *INDENT-OFF*
    const CeedScalar temptau[3][3] = {
        {tau[0][i], tau[5][i], tau[4][i]},
        {tau[5][i], tau[1][i], tau[3][i]},
        {tau[4][i], tau[3][i], tau[2][i]}
    };
    // *INDENT-ON*
    // Apply dXdx^T and weight to intermediate stress
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        dvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++) dvdX[k][j][i] += dXdx[k][m][i] * temptau[j][m] * wdetJ;
      }
    }

  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSCurrentNH2dF)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*deltaug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
        (*q_data)[CEED_Q_VLA]               = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // dXdx computed in residual
  const CeedScalar(*dXdx)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];
  // tau computed in residual
  const CeedScalar(*tau)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[3];
  // lam_log_J = lambda*log(J) computed in residual
  const CeedScalar(*lam_log_J)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[4];
  // Outputs
  CeedScalar(*deltadvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
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
    // Read spatial derivatives of delta_u
    // *INDENT-OFF*
    const CeedScalar deltadu[3][3] = {
        {deltaug[0][0][i], deltaug[1][0][i], deltaug[2][0][i]},
        {deltaug[0][1][i], deltaug[1][1][i], deltaug[2][1][i]},
        {deltaug[0][2][i], deltaug[1][2][i], deltaug[2][2][i]}
    };
    // -- Qdata
    const CeedScalar wdetJ = q_data[0][i];
    // *INDENT-ON*

    // Compute grad_du = \nabla_x (deltau) = deltau * dX/dx
    CeedScalar grad_du[3][3];
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        grad_du[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) grad_du[j][k] += deltadu[j][m] * dXdx[m][k][i];
      }
    }

    // *INDENT-OFF*
    const CeedScalar temptau[3][3] = {
        {tau[0][i], tau[5][i], tau[4][i]},
        {tau[5][i], tau[1][i], tau[3][i]},
        {tau[4][i], tau[3][i], tau[2][i]}
    };

    // Compute grad_du_tau = grad_du*tau
    CeedScalar grad_du_tau[3][3];
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        grad_du_tau[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) grad_du_tau[j][k] += grad_du[j][m] * temptau[m][k];
      }
    }

    // Compute depsilon = (grad_du + grad_du^T)/2
    const CeedScalar depsilon[3][3] = {
        {(grad_du[0][0] + grad_du[0][0]) / 2., (grad_du[0][1] + grad_du[1][0]) / 2., (grad_du[0][2] + grad_du[2][0]) / 2.},
        {(grad_du[1][0] + grad_du[0][1]) / 2., (grad_du[1][1] + grad_du[1][1]) / 2., (grad_du[1][2] + grad_du[2][1]) / 2.},
        {(grad_du[2][0] + grad_du[0][2]) / 2., (grad_du[2][1] + grad_du[1][2]) / 2., (grad_du[2][2] + grad_du[2][2]) / 2.}
    };
    // Compute trace(depsilon)
    CeedScalar tr_deps = depsilon[0][0] + depsilon[1][1] + depsilon[2][2];
    // Compute grad_du*tau + trace(depsilon)I3
    grad_du_tau[0][0] += lambda * tr_deps;
    grad_du_tau[1][1] += lambda * tr_deps;
    grad_du_tau[2][2] += lambda * tr_deps;
    // Compute dp = grad_du*tau + trace(depsilon)I3 +2(mu-lambda*logJ)depsilon
    CeedScalar dp[3][3];
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        dp[j][k] = grad_du_tau[j][k] + 2 * (mu - lam_log_J[0][i]) * depsilon[j][k];
      }
    }

    // Apply dXdx^T and weight
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        deltadvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++) deltadvdX[k][j][i] += dXdx[k][m][i] * dp[j][m] * wdetJ;
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Strain energy computation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSCurrentNH2Energy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
  const CeedScalar TwoMu   = E / (1 + nu);
  const CeedScalar mu      = TwoMu / 2;
  const CeedScalar Kbulk   = E / (3 * (1 - 2 * nu));  // Bulk Modulus
  const CeedScalar lambda  = (3 * Kbulk - TwoMu) / 3;

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

    // Compute Grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    CeedScalar Grad_u[3][3];
    for (int j = 0; j < 3; j++) {    // Component
      for (int k = 0; k < 3; k++) {  // Derivative
        Grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++) Grad_u[j][k] += dXdx[m][k] * du[j][m];
      }
    }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (Grad_u + Grad_u^T + Grad_u^T*Grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar    E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = Grad_u[indj[m]][indk[m]] + Grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++) E2work[m] += Grad_u[n][indj[m]] * Grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {
        {E2work[0], E2work[5], E2work[4]},
        {E2work[5], E2work[1], E2work[3]},
        {E2work[4], E2work[3], E2work[2]}
    };
    // *INDENT-ON*
    const CeedScalar Jm1  = computeJM1(Grad_u);
    const CeedScalar logJ = log1p_series_shifted(Jm1);

    // Strain energy Phi(E) for compressible Neo-Hookean
    energy[i] = (lambda * logJ * logJ / 2. - mu * logJ + mu * (E2[0][0] + E2[1][1] + E2[2][2]) / 2.) * wdetJ;

  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Nodal diagnostic quantities for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSCurrentNH2Diagnostic)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
  const CeedScalar TwoMu   = E / (1 + nu);
  const CeedScalar mu      = TwoMu / 2;
  const CeedScalar Kbulk   = E / (3 * (1 - 2 * nu));  // Bulk Modulus
  const CeedScalar lambda  = (3 * Kbulk - TwoMu) / 3;

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

    // Compute Grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    CeedScalar Grad_u[3][3];
    for (int j = 0; j < 3; j++) {    // Component
      for (int k = 0; k < 3; k++) {  // Derivative
        Grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++) Grad_u[j][k] += dXdx[m][k] * du[j][m];
      }
    }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (Grad_u + Grad_u^T + Grad_u^T*Grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar    E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = Grad_u[indj[m]][indk[m]] + Grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++) E2work[m] += Grad_u[n][indj[m]] * Grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {
        {E2work[0], E2work[5], E2work[4]},
        {E2work[5], E2work[1], E2work[3]},
        {E2work[4], E2work[3], E2work[2]}
    };
    // *INDENT-ON*

    // Displacement
    diagnostic[0][i] = u[0][i];
    diagnostic[1][i] = u[1][i];
    diagnostic[2][i] = u[2][i];

    // Pressure
    const CeedScalar Jm1  = computeJM1(Grad_u);
    const CeedScalar logJ = log1p_series_shifted(Jm1);
    diagnostic[3][i]      = -lambda * logJ;

    // Stress tensor invariants
    diagnostic[4][i] = (E2[0][0] + E2[1][1] + E2[2][2]) / 2.;
    diagnostic[5][i] = 0.;
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt m = 0; m < 3; m++) diagnostic[5][i] += E2[j][m] * E2[m][j] / 4.;
    }
    diagnostic[6][i] = Jm1 + 1.;

    // Strain energy
    diagnostic[7][i] = (lambda * logJ * logJ / 2. - mu * logJ + mu * (E2[0][0] + E2[1][1] + E2[2][2]) / 2.);
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // End of ELAS_FSCurrentNH2_H
