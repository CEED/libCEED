// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Hyperelasticity, finite strain for solid mechanics example using PETSc

#ifndef ELAS_FSInitialMR1_H
#define ELAS_FSInitialMR1_H

#include <ceed.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Mooney-Rivlin context
#ifndef PHYSICS_STRUCT_MR
#define PHYSICS_STRUCT_MR
typedef struct Physics_private_MR *Physics_MR;

struct Physics_private_MR {
  // material properties for MR
  CeedScalar mu_1;
  CeedScalar mu_2;
  CeedScalar lambda;
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
// Compute matrix^(-1), where matrix is symetric, returns array of 6
// -----------------------------------------------------------------------------
#ifndef MatinvSym
#define MatinvSym
CEED_QFUNCTION_HELPER int computeMatinvSym(const CeedScalar A[3][3], const CeedScalar detA, CeedScalar Ainv[6]) {
  // Compute A^(-1) : A-Inverse
  CeedScalar B[6] = {
      A[1][1] * A[2][2] - A[1][2] * A[2][1], /* *NOPAD* */
      A[0][0] * A[2][2] - A[0][2] * A[2][0], /* *NOPAD* */
      A[0][0] * A[1][1] - A[0][1] * A[1][0], /* *NOPAD* */
      A[0][2] * A[1][0] - A[0][0] * A[1][2], /* *NOPAD* */
      A[0][1] * A[1][2] - A[0][2] * A[1][1], /* *NOPAD* */
      A[0][2] * A[2][1] - A[0][1] * A[2][2]  /* *NOPAD* */
  };
  for (CeedInt m = 0; m < 6; m++) Ainv[m] = B[m] / (detA);

  return 0;
};
#endif
// -----------------------------------------------------------------------------
// Common computations between FS and dFS
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int commonFSMR1(const CeedScalar mu_1, const CeedScalar mu_2, const CeedScalar lambda, const CeedScalar grad_u[3][3],
                                      CeedScalar Swork[6], CeedScalar Cwork[6], CeedScalar Cinvwork[6], CeedScalar *logJ) {
  // E - Green-Lagrange strain tensor
  //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
  const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
  CeedScalar    E2work[6];
  for (CeedInt m = 0; m < 6; m++) {
    E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
    for (CeedInt n = 0; n < 3; n++) E2work[m] += grad_u[n][indj[m]] * grad_u[n][indk[m]];
  }
  // *INDENT-OFF*
  CeedScalar E2[3][3] = {
      {E2work[0], E2work[5], E2work[4]},
      {E2work[5], E2work[1], E2work[3]},
      {E2work[4], E2work[3], E2work[2]}
  };

  // C : right Cauchy-Green tensor
  // C = I + 2E
  const CeedScalar C[3][3] = {
      {1 + E2[0][0], E2[0][1],     E2[0][2]    },
      {E2[0][1],     1 + E2[1][1], E2[1][2]    },
      {E2[0][2],     E2[1][2],     1 + E2[2][2]}
  };

  Cwork[0] = C[0][0];
  Cwork[1] = C[1][1];
  Cwork[2] = C[2][2];
  Cwork[3] = C[1][2];
  Cwork[4] = C[0][2];
  Cwork[5] = C[0][1];
  // *INDENT-ON*
  // Compute invariants
  // I_1 = trace(C)
  const CeedScalar I_1 = C[0][0] + C[1][1] + C[2][2];
  // J-1
  const CeedScalar Jm1 = computeJM1(grad_u);
  // J = Jm1 + 1
  // Compute C^(-1) : C-Inverse
  const CeedScalar detC = (Jm1 + 1.) * (Jm1 + 1.);
  computeMatinvSym(C, detC, Cinvwork);

  // Compute the Second Piola-Kirchhoff (S)
  // S = (lambda*logJ - mu_1 -2*mu_2)*Cinvwork +(mu_1+mu_2*I_1)*I3-mu_2*Cwork
  // *1 for indices 0-2 for I_3

  *logJ = log1p_series_shifted(Jm1);
  // *INDENT-OFF*
  for (CeedInt i = 0; i < 6; i++) {
    Swork[i] = (lambda * *logJ - mu_1 - 2 * mu_2) * Cinvwork[i] + (mu_1 + mu_2 * I_1) * (i < 3)  // identity I_3
               - mu_2 * Cwork[i];
  }

  return 0;
};

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialMR1F)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar(*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // Store grad_u for HyperFSdF (Jacobian of HyperFSF)
  CeedScalar(*grad_u)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1    = context->mu_1;
  const CeedScalar mu_2    = context->mu_2;
  const CeedScalar lambda  = context->lambda;

  // Formulation Terminology:
  //  I3    : 3x3 Identity matrix
  //  C     : right Cauchy-Green tensor
  //  C_inv  : inverse of C
  //  F     : deformation gradient
  //  S     : 2nd Piola-Kirchhoff
  //  P     : 1st Piola-Kirchhoff

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
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        grad_u[j][k][i] = 0;
        for (CeedInt m = 0; m < 3; m++) grad_u[j][k][i] += dXdx[m][k] * du[j][m];
      }
    }

    // I3 : 3x3 Identity matrix
    // Compute The Deformation Gradient : F = I3 + grad_u
    // *INDENT-OFF*
    const CeedScalar F[3][3] = {
        {grad_u[0][0][i] + 1, grad_u[0][1][i],     grad_u[0][2][i]    },
        {grad_u[1][0][i],     grad_u[1][1][i] + 1, grad_u[1][2][i]    },
        {grad_u[2][0][i],     grad_u[2][1][i],     grad_u[2][2][i] + 1}
    };

    const CeedScalar tempgradu[3][3] = {
        {grad_u[0][0][i], grad_u[0][1][i], grad_u[0][2][i]},
        {grad_u[1][0][i], grad_u[1][1][i], grad_u[1][2][i]},
        {grad_u[2][0][i], grad_u[2][1][i], grad_u[2][2][i]}
    };

    // Common components of finite strain calculations
    CeedScalar Swork[6], Cwork[6], Cinvwork[6], logJ;
    commonFSMR1(mu_1, mu_2, lambda, tempgradu, Swork, Cwork, Cinvwork, &logJ);

    // Second Piola-Kirchhoff (S)
    const CeedScalar S[3][3] = {
        {Swork[0], Swork[5], Swork[4]},
        {Swork[5], Swork[1], Swork[3]},
        {Swork[4], Swork[3], Swork[2]}
    };
    // *INDENT-ON*

    // Compute the First Piola-Kirchhoff : P = F*S
    CeedScalar P[3][3];
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        P[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) P[j][k] += F[j][m] * S[m][k];
      }
    }

    // Apply dXdx^T and weight to P (First Piola-Kirchhoff)
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        dvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++) dvdX[k][j][i] += dXdx[k][m] * P[j][m] * wdetJ;
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialMR1dF)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1    = context->mu_1;
  const CeedScalar mu_2    = context->mu_2;
  const CeedScalar lambda  = context->lambda;

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
    const CeedScalar wdetJ      = q_data[0][i];
    const CeedScalar dXdx[3][3] = {
        {q_data[1][i], q_data[2][i], q_data[3][i]},
        {q_data[4][i], q_data[5][i], q_data[6][i]},
        {q_data[7][i], q_data[8][i], q_data[9][i]}
    };
    // *INDENT-ON*

    // Compute graddeltau
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to deltadu = graddelta
    // this is dF
    CeedScalar graddeltau[3][3];
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        graddeltau[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) graddeltau[j][k] += dXdx[m][k] * deltadu[j][m];
      }
    }

    // I3 : 3x3 Identity matrix
    // Deformation Gradient : F = I3 + grad_u
    // *INDENT-OFF*
    const CeedScalar F[3][3] = {
        {grad_u[0][0][i] + 1, grad_u[0][1][i],     grad_u[0][2][i]    },
        {grad_u[1][0][i],     grad_u[1][1][i] + 1, grad_u[1][2][i]    },
        {grad_u[2][0][i],     grad_u[2][1][i],     grad_u[2][2][i] + 1}
    };

    const CeedScalar tempgradu[3][3] = {
        {grad_u[0][0][i], grad_u[0][1][i], grad_u[0][2][i]},
        {grad_u[1][0][i], grad_u[1][1][i], grad_u[1][2][i]},
        {grad_u[2][0][i], grad_u[2][1][i], grad_u[2][2][i]}
    };

    // Common components of finite strain calculations
    CeedScalar Swork[6], Cwork[6], Cinvwork[6], logJ;
    commonFSMR1(mu_1, mu_2, lambda, tempgradu, Swork, Cwork, Cinvwork, &logJ);

    // *INDENT-ON*
    // dE - Green-Lagrange strain tensor
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar    dEwork[6];
    for (CeedInt m = 0; m < 6; m++) {
      dEwork[m] = 0;
      for (CeedInt n = 0; n < 3; n++) dEwork[m] += (graddeltau[n][indj[m]] * F[n][indk[m]] + F[n][indj[m]] * graddeltau[n][indk[m]]) / 2.;
    }
    // *INDENT-OFF*
    CeedScalar dE[3][3] = {
        {dEwork[0], dEwork[5], dEwork[4]},
        {dEwork[5], dEwork[1], dEwork[3]},
        {dEwork[4], dEwork[3], dEwork[2]}
    };
    // C : right Cauchy-Green tensor
    // C^(-1) : C-Inverse
    const CeedScalar C[3][3] = {
        {Cwork[0], Cwork[5], Cwork[4]},
        {Cwork[5], Cwork[1], Cwork[3]},
        {Cwork[4], Cwork[3], Cwork[2]}
    };
    const CeedScalar C_inv[3][3] = {
        {Cinvwork[0], Cinvwork[5], Cinvwork[4]},
        {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
        {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
    };
    // *INDENT-ON*
    // -- C_inv:dE
    CeedScalar Cinv_contract_dE = 0;
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) Cinv_contract_dE += C_inv[j][k] * dE[j][k];
    }

    // -- C:dE
    CeedScalar C_contract_dE = 0;
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) C_contract_dE += C[j][k] * dE[j][k];
    }

    // -- dE*C_inv
    CeedScalar dE_Cinv[3][3];
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        dE_Cinv[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) dE_Cinv[j][k] += dE[j][m] * C_inv[m][k];
      }
    }

    // -- C_inv*dE*C_inv
    CeedScalar Cinv_dE_Cinv[3][3];
    // This product is symmetric and we only use the upper-triangular part
    // below, but naively compute the whole thing here
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        Cinv_dE_Cinv[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) Cinv_dE_Cinv[j][k] += C_inv[j][m] * dE_Cinv[m][k];
      }
    }

    // Compute dS = (mu_2)*((2*I_3:dE)*I_3 - dE) + 2*d*Cinv_dE_Cinv + lambda*Cinv_contract_dE*Cinvwork - 2*lambda*logJ*Cinv_dE_Cinv;
    // (2*I_3:dE)*I_3 - dE = 2*trace(dE)*I_3 - dE = 2trace(dE) - dE on the diagonal
    // (2*I_3:dE)*I_3 - dE = -dE elsewhere
    // CeedScalar J = Jm1 + 1;
    CeedScalar tr_dE = dE[0][0] + dE[1][1] + dE[2][2];
    CeedScalar dSwork[6];
    for (CeedInt i = 0; i < 6; i++) {
      dSwork[i] = lambda * Cinv_contract_dE * Cinvwork[i] + 2 * (mu_1 + 2 * mu_2 - lambda * logJ) * Cinv_dE_Cinv[indj[i]][indk[i]] +
                  2 * mu_2 * (tr_dE * (i < 3) - dEwork[i]);
    }

    // *INDENT-OFF*
    CeedScalar dS[3][3] = {
        {dSwork[0], dSwork[5], dSwork[4]},
        {dSwork[5], dSwork[1], dSwork[3]},
        {dSwork[4], dSwork[3], dSwork[2]}
    };
    // Second Piola-Kirchhoff (S)
    const CeedScalar S[3][3] = {
        {Swork[0], Swork[5], Swork[4]},
        {Swork[5], Swork[1], Swork[3]},
        {Swork[4], Swork[3], Swork[2]}
    };
    // *INDENT-ON*
    // dP = dPdF:dF = dF*S + F*dS
    CeedScalar dP[3][3];
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        dP[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) dP[j][k] += graddeltau[j][m] * S[m][k] + F[j][m] * dS[m][k];
      }
    }

    // Apply dXdx^T and weight
    for (CeedInt j = 0; j < 3; j++) {    // Component
      for (CeedInt k = 0; k < 3; k++) {  // Derivative
        deltadvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++) deltadvdX[k][j][i] += dXdx[k][m] * dP[j][m] * wdetJ;
      }
    }
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------
// Strain energy computation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialMR1Energy)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0], (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar(*energy) = (CeedScalar(*))out[0];
  // *INDENT-ON*

  // Context
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1    = context->mu_1;
  const CeedScalar mu_2    = context->mu_2;
  const CeedScalar lambda  = context->lambda;

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
    for (int j = 0; j < 3; j++) {    // Component
      for (int k = 0; k < 3; k++) {  // Derivative
        grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++) grad_u[j][k] += dXdx[m][k] * du[j][m];
      }
    }

    // E - Green-Lagrange strain tensor
    // E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar    E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++) E2work[m] += grad_u[n][indj[m]] * grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {
        {E2work[0], E2work[5], E2work[4]},
        {E2work[5], E2work[1], E2work[3]},
        {E2work[4], E2work[3], E2work[2]}
    };

    // C : right Cauchy-Green tensor
    // C = I + 2E
    // *INDENT-OFF*
    const CeedScalar C[3][3] = {
        {1 + E2[0][0], E2[0][1],     E2[0][2]    },
        {E2[0][1],     1 + E2[1][1], E2[1][2]    },
        {E2[0][2],     E2[1][2],     1 + E2[2][2]}
    };
    // *INDENT-ON*
    // Compute CC = C*C = C^2
    CeedScalar CC[3][3];
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        CC[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) CC[j][k] += C[j][m] * C[m][k];
      }
    }

    const CeedScalar Jm1 = computeJM1(grad_u);
    // CeedScalar J = Jm1 + 1;
    // Compute invariants
    // I_1 = trace(C)
    const CeedScalar I_1 = C[0][0] + C[1][1] + C[2][2];
    // Trace(C^2)
    const CeedScalar tr_CC = CC[0][0] + CC[1][1] + CC[2][2];
    // I_2 = 0.5(I_1^2 - trace(C^2))
    const CeedScalar I_2 = 0.5 * (I_1 * I_1 - tr_CC);

    // *INDENT-OFF*
    const CeedScalar logJ = log1p_series_shifted(Jm1);
    // Strain energy Phi(E) for Mooney-Rivlin
    energy[i] = (0.5 * lambda * (logJ) * (logJ) - (mu_1 + 2 * mu_2) * logJ + (mu_1 / 2.) * (I_1 - 3) + (mu_2 / 2.) * (I_2 - 3)) * wdetJ;

  }  // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Nodal diagnostic quantities for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialMR1Diagnostic)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar(*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0], (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
        (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar(*diagnostic)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1    = context->mu_1;
  const CeedScalar mu_2    = context->mu_2;
  const CeedScalar lambda  = context->lambda;

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
    for (int j = 0; j < 3; j++) {    // Component
      for (int k = 0; k < 3; k++) {  // Derivative
        grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++) grad_u[j][k] += dXdx[m][k] * du[j][m];
      }
    }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar    E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++) E2work[m] += grad_u[n][indj[m]] * grad_u[n][indk[m]];
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
    const CeedScalar Jm1  = computeJM1(grad_u);
    const CeedScalar logJ = log1p_series_shifted(Jm1);
    diagnostic[3][i]      = -lambda * logJ;

    // Stress tensor invariants
    diagnostic[4][i] = (E2[0][0] + E2[1][1] + E2[2][2]) / 2.;
    diagnostic[5][i] = 0.;
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt m = 0; m < 3; m++) diagnostic[5][i] += E2[j][m] * E2[m][j] / 4.;
    }
    diagnostic[6][i] = Jm1 + 1;

    // C : right Cauchy-Green tensor
    // C = I + 2E
    // *INDENT-OFF*
    const CeedScalar C[3][3] = {
        {1 + E2[0][0], E2[0][1],     E2[0][2]    },
        {E2[0][1],     1 + E2[1][1], E2[1][2]    },
        {E2[0][2],     E2[1][2],     1 + E2[2][2]}
    };
    // *INDENT-ON*
    // Compute CC = C*C = C^2
    CeedScalar CC[3][3];
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        CC[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++) CC[j][k] += C[j][m] * C[m][k];
      }
    }

    // CeedScalar J = Jm1 + 1;
    // Compute invariants
    // I_1 = trace(C)
    const CeedScalar I_1 = C[0][0] + C[1][1] + C[2][2];
    // Trace(C^2)
    const CeedScalar tr_CC = CC[0][0] + CC[1][1] + CC[2][2];
    // I_2 = 0.5(I_1^2 - trace(C^2))
    const CeedScalar I_2 = 0.5 * (pow(I_1, 2) - tr_CC);

    // *INDENT-OFF*
    // Strain energy
    diagnostic[7][i] = (0.5 * lambda * logJ * logJ - (mu_1 + 2 * mu_2) * logJ + (mu_1 / 2.) * (I_1 - 3) + (mu_2 / 2.) * (I_2 - 3));
  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // End of ELAS_FSInitialMR1_H
