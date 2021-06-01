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
/// Hyperelasticity, finite strain for solid mechanics example using PETSc

#ifndef ELAS_FSInitialMR1_H
#define ELAS_FSInitialMR1_H

#ifndef __CUDACC__
#  include <math.h>
#endif

// -----------------------------------------------------------------------------
// Mooney-Rivlin context
#ifndef PHYSICS_STRUCT_MR
#define PHYSICS_STRUCT_MR
typedef struct Physics_private_MR *Physics_MR;

struct Physics_private_MR {
  //material properties for MR
  CeedScalar mu_1; //
  CeedScalar mu_2; //
  CeedScalar k_1; //
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
static inline CeedScalar log1p_series_shifted(CeedScalar x) {
  const CeedScalar left = sqrt(2.)/2 - 1, right = sqrt(2.) - 1;
  CeedScalar sum = 0;
  if (1) { // Disable if the smaller range sqrt(2) < J < sqrt(2) is sufficient
    if (x < left) { // Replace if with while for arbitrary range (may hurt vectorization)
      sum -= log(2.) / 2;
      x = 1 + 2 * x;
    } else if (right < x) {
      sum += log(2.) / 2;
      x = (x - 1) / 2;
    }
  }
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
#endif

// -----------------------------------------------------------------------------
// Compute det F - 1
// -----------------------------------------------------------------------------
#ifndef DETJM1
#define DETJM1
static inline CeedScalar computeJM1(const CeedScalar grad_u[3][3]) {
  return grad_u[0][0]*(grad_u[1][1]*grad_u[2][2]-grad_u[1][2]*grad_u[2][1]) +
         grad_u[0][1]*(grad_u[1][2]*grad_u[2][0]-grad_u[1][0]*grad_u[2][2]) +
         grad_u[0][2]*(grad_u[1][0]*grad_u[2][1]-grad_u[2][0]*grad_u[1][1]) +
         grad_u[0][0] + grad_u[1][1] + grad_u[2][2] +
         grad_u[0][0]*grad_u[1][1] + grad_u[0][0]*grad_u[2][2] +
         grad_u[1][1]*grad_u[2][2] - grad_u[0][1]*grad_u[1][0] -
         grad_u[0][2]*grad_u[2][0] - grad_u[1][2]*grad_u[2][1];
};
#endif

// -----------------------------------------------------------------------------
// Common computations between FS and dFS
// -----------------------------------------------------------------------------
static inline int commonFSMR(const CeedScalar mu_1, const CeedScalar mu_2,
                           const CeedScalar k_1, const CeedScalar grad_u[3][3], CeedScalar Swork[6],
                           CeedScalar Cinvwork[6], CeedScalar dI1bar_dE[6],
                           CeedScalar dI2bar_dE[6], CeedScalar *I_1,
                           CeedScalar *I_2, CeedScalar *Jm1) {
  // E - Green-Lagrange strain tensor
  //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
  const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
  CeedScalar E2work[6];
  for (CeedInt m = 0; m < 6; m++) {
    E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
    for (CeedInt n = 0; n < 3; n++)
      E2work[m] += grad_u[n][indj[m]]*grad_u[n][indk[m]];
  }
  // *INDENT-OFF*
  CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                         {E2work[5], E2work[1], E2work[3]},
                         {E2work[4], E2work[3], E2work[2]}
                        };

  // C : right Cauchy-Green tensor
  // C = I + 2E
  const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                              {E2[0][1], 1 + E2[1][1], E2[1][2]},
                              {E2[0][2], E2[1][2], 1 + E2[2][2]}
                             };
  // *INDENT-ON*
  // compute CC = C*C = C^2
  CeedScalar CC[3][3];
  for (CeedInt j = 0; j < 3; j++)
    for (CeedInt k = 0; k < 3; k++) {
      CC[j][k] = 0;
      for (CeedInt m = 0; m < 3; m++)
        CC[j][k] += C[j][m] * C[m][k];
      }

  // compute invariants
  // I_1 = trace(C)
  (*I_1) = C[0][0] + C[1][1] + C[2][2];
  // trace(C^2)
  CeedScalar tr_CC = CC[0][0] + CC[1][1] + CC[2][2];
  // I_2 = 0.5(I_1^2 - trace(C^2))
  (*I_2) = 0.5*(pow((*I_1), 2) - tr_CC);
  // J-1
  (*Jm1) = computeJM1(grad_u);
  // J = Jm1 + 1
  CeedScalar J = *Jm1 + 1;
  CeedScalar J2 = J*J;
  // Compute C^(-1) : C-Inverse
  CeedScalar A[6] = {C[1][1]*C[2][2] - C[1][2]*C[2][1], /* *NOPAD* */
                     C[0][0]*C[2][2] - C[0][2]*C[2][0], /* *NOPAD* */
                     C[0][0]*C[1][1] - C[0][1]*C[1][0], /* *NOPAD* */
                     C[0][2]*C[1][0] - C[0][0]*C[1][2], /* *NOPAD* */
                     C[0][1]*C[1][2] - C[0][2]*C[1][1], /* *NOPAD* */
                     C[0][2]*C[2][1] - C[0][1]*C[2][2] /* *NOPAD* */
                    };
  for (CeedInt m = 0; m < 6; m++)
    Cinvwork[m] = A[m] / (J2);

  // Compute the Second Piola-Kirchhoff (S) (canceled 1/2 already in S with 2 in dI1bar_dE)
  // S = mu_1*dI1bar_dE + mu_2*dI2bar_dE + k1*(J^2 - J)*C^{-1}
  // dI1bar_dE = J^{-2/3}*(I3 - 1/3*I_1*C^{-1})
  dI1bar_dE[0] = pow(J,-2/3) * ( 1 - (1/3)* (*I_1) * Cinvwork[0] );
  dI1bar_dE[1] = pow(J,-2/3) * ( 1 - (1/3)* (*I_1) * Cinvwork[1] );
  dI1bar_dE[2] = pow(J,-2/3) * ( 1 - (1/3)* (*I_1) * Cinvwork[2] );
  dI1bar_dE[3] = pow(J,-2/3) * (-(1/3)* (*I_1) * Cinvwork[3] );
  dI1bar_dE[4] = pow(J,-2/3) * (-(1/3)* (*I_1) * Cinvwork[4] );
  dI1bar_dE[5] = pow(J,-2/3) * (-(1/3)* (*I_1) * Cinvwork[5] );

  // dI2bar_dE = J^{-4/3}*(I_1*I3 - C - 2/3*I_2*C^{-1})
  dI2bar_dE[0] = pow(J,-4/3) * ( (*I_1) - C[0][0] - (2/3)* (*I_2) * Cinvwork[0] );
  dI2bar_dE[1] = pow(J,-4/3) * ( (*I_1) - C[1][1] - (2/3)* (*I_2) * Cinvwork[1] );
  dI2bar_dE[2] = pow(J,-4/3) * ( (*I_1) - C[2][2] - (2/3)* (*I_2) * Cinvwork[2] );
  dI2bar_dE[3] = pow(J,-4/3) * (-C[1][2] - (2/3)* (*I_2) * Cinvwork[3] );
  dI2bar_dE[4] = pow(J,-4/3) * (-C[0][2] - (2/3)* (*I_2) * Cinvwork[4] );
  dI2bar_dE[5] = pow(J,-4/3) * (-C[0][1] - (2/3)* (*I_2) * Cinvwork[5] );

  // (J^2 - J)*C^{-1} or J*(J - 1)*C^{-1}
  CeedScalar JJm1Cinv[6];
  JJm1Cinv[0] = J * (*Jm1) * Cinvwork[0];
  JJm1Cinv[1] = J * (*Jm1) * Cinvwork[1];
  JJm1Cinv[2] = J * (*Jm1) * Cinvwork[2];
  JJm1Cinv[3] = J * (*Jm1) * Cinvwork[3];
  JJm1Cinv[4] = J * (*Jm1) * Cinvwork[4];
  JJm1Cinv[5] = J * (*Jm1) * Cinvwork[5];

  // compute S = mu_1*dI1bar_dE + mu_2*dI2bar_dE + k1*J*(J - 1)*C^{-1}
  Swork[0] = mu_1*dI1bar_dE[0] + mu_2*dI2bar_dE[0] + k_1*JJm1Cinv[0];
  Swork[1] = mu_1*dI1bar_dE[1] + mu_2*dI2bar_dE[1] + k_1*JJm1Cinv[1];
  Swork[2] = mu_1*dI1bar_dE[2] + mu_2*dI2bar_dE[2] + k_1*JJm1Cinv[2];
  Swork[3] = mu_1*dI1bar_dE[3] + mu_2*dI2bar_dE[3] + k_1*JJm1Cinv[3];
  Swork[4] = mu_1*dI1bar_dE[4] + mu_2*dI2bar_dE[4] + k_1*JJm1Cinv[4];
  Swork[5] = mu_1*dI1bar_dE[5] + mu_2*dI2bar_dE[5] + k_1*JJm1Cinv[5];

  return 0;
};

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialMR1F)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // Store grad_u for HyperFSdF (Jacobian of HyperFSF)
  CeedScalar (*grad_u)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1  = context->mu_1;
  const CeedScalar mu_2 = context->mu_2;
  const CeedScalar k_1 = context->k_1;

  // Formulation Terminology:
  //  I3    : 3x3 Identity matrix
  //  C     : right Cauchy-Green tensor
  //  C_inv  : inverse of C
  //  F     : deformation gradient
  //  S     : 2nd Piola-Kirchhoff (in current config)
  //  P     : 1st Piola-Kirchhoff (in referential config)

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
    const CeedScalar wdetJ      =   q_data[0][i];
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        grad_u[j][k][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          grad_u[j][k][i] += dXdx[m][k] * du[j][m];
      }

    // I3 : 3x3 Identity matrix
    // Compute The Deformation Gradient : F = I3 + grad_u
    // *INDENT-OFF*
    const CeedScalar F[3][3] =  {{grad_u[0][0][i] + 1,
                                  grad_u[0][1][i],
                                  grad_u[0][2][i]},
                                 {grad_u[1][0][i],
                                  grad_u[1][1][i] + 1,
                                  grad_u[1][2][i]},
                                 {grad_u[2][0][i],
                                  grad_u[2][1][i],
                                  grad_u[2][2][i] + 1}
                                };

    const CeedScalar tempgradu[3][3] =  {{grad_u[0][0][i],
                                          grad_u[0][1][i],
                                          grad_u[0][2][i]},
                                         {grad_u[1][0][i],
                                          grad_u[1][1][i],
                                          grad_u[1][2][i]},
                                         {grad_u[2][0][i],
                                          grad_u[2][1][i],
                                          grad_u[2][2][i]}
                                        };
    // *INDENT-ON*
    // Common components of finite strain calculations
    CeedScalar Swork[6], Cinvwork[6], dI1bar_dE[6], dI2bar_dE[6], I_1, I_2, Jm1;
    commonFSMR(mu_1, mu_2, k_1, tempgradu, Swork, Cinvwork, dI1bar_dE, dI2bar_dE, &I_1, &I_2, &Jm1);

    // Second Piola-Kirchhoff (S)
    // *INDENT-OFF*
    const CeedScalar S[3][3] = {{Swork[0], Swork[5], Swork[4]},
                                {Swork[5], Swork[1], Swork[3]},
                                {Swork[4], Swork[3], Swork[2]}
                               };
    // *INDENT-ON*

    // Compute the First Piola-Kirchhoff : P = F*S
    CeedScalar P[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        P[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          P[j][k] += F[j][m] * S[m][k];
      }

    // Apply dXdx^T and weight to P (First Piola-Kirchhoff)
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        dvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          dvdX[k][j][i] += dXdx[k][m] * P[j][m] * wdetJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialMR1dF)(void *ctx, CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*deltaug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // grad_u is used for hyperelasticity (non-linear)
  const CeedScalar (*grad_u)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*deltadvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1  = context->mu_1;
  const CeedScalar mu_2 = context->mu_2;
  const CeedScalar k_1 = context->k_1;

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of delta_u
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
    const CeedScalar wdetJ      =      q_data[0][i];
    const CeedScalar dXdx[3][3] =    {{q_data[1][i],
                                       q_data[2][i],
                                       q_data[3][i]},
                                      {q_data[4][i],
                                       q_data[5][i],
                                       q_data[6][i]},
                                      {q_data[7][i],
                                       q_data[8][i],
                                       q_data[9][i]}
                                      };
    // *INDENT-ON*

    // Compute graddeltau
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to deltadu = graddelta
    // this is dF
    CeedScalar graddeltau[3][3];
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        graddeltau[j][k] = 0;
        for (CeedInt m =0 ; m < 3; m++)
          graddeltau[j][k] += dXdx[m][k] * deltadu[j][m];
      }

    // I3 : 3x3 Identity matrix
    // Deformation Gradient : F = I3 + grad_u
    // *INDENT-OFF*
    const CeedScalar F[3][3] =      {{grad_u[0][0][i] + 1,
                                      grad_u[0][1][i],
                                      grad_u[0][2][i]},
                                     {grad_u[1][0][i],
                                      grad_u[1][1][i] + 1,
                                      grad_u[1][2][i]},
                                     {grad_u[2][0][i],
                                      grad_u[2][1][i],
                                      grad_u[2][2][i] + 1}
                                    };

    const CeedScalar tempgradu[3][3] =  {{grad_u[0][0][i],
                                          grad_u[0][1][i],
                                          grad_u[0][2][i]},
                                         {grad_u[1][0][i],
                                          grad_u[1][1][i],
                                          grad_u[1][2][i]},
                                         {grad_u[2][0][i],
                                          grad_u[2][1][i],
                                          grad_u[2][2][i]}
                                        };
    // *INDENT-ON*
    // Common components of finite strain calculations
    CeedScalar Swork[6], Cinvwork[6], dI1bar_dE[6], dI2bar_dE[6], I_1, I_2, Jm1;
    commonFSMR(mu_1, mu_2, k_1, tempgradu, Swork, Cinvwork, dI1bar_dE, dI2bar_dE, &I_1, &I_2, &Jm1);

    // debugging plan below
    // Estimate dS using finite differencing
    CeedScalar epsilon = 1e-8; //
    CeedScalar Swork2[6], Cinvwork2[6], dI1bar_dE2[6], dI2bar_dE2[6], I_12, I_22, Jm12;
    CeedScalar temp_graddeltau_eps[3][3]; // epsilon * graddeltau
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++)
        temp_graddeltau_eps[j][k] =  graddeltau[j][k] * epsilon;
    
    CeedScalar temp_gradu_eps_graddeltau[3][3];
      
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        temp_gradu_eps_graddeltau[j][k] = tempgradu[j][k] + temp_graddeltau_eps[j][k];
      }
    commonFSMR(mu_1, mu_2, k_1, temp_gradu_eps_graddeltau, Swork2, Cinvwork2, dI1bar_dE2, dI2bar_dE2, &I_12, &I_22, &Jm12);
      // dS = (Swork2 - Swork) / epsilon;
      // Integrate: dF S + F dS

      // Notes
      // Start by testing convergence with finite differencing dS (above), skipping the analytic code below
      // Test case for convergence:
      // ./elasticity -mu_1 0.5 -mu_2 0 -K 10 -nu .4 -E .5 -degree 1 -dm_plex_box_faces 4,4,4 -problem FSInitial-MR1 -num_steps 1 -bc_clamp 3 -bc_traction 4 -bc_traction_4 0,0,-.1 -snes_linesearch_type cp -snes_monitor -snes_linesearch_monitor -snes_linesearch_atol 1e-30 -snes_fd -outer_ksp_converged_reason -outer_pc_type lu -outer_ksp_monitor_true_residual -outer_ksp_max_it 10 -outer_ksp_type gmres -outer_ksp_norm_type preconditioned -snes_view
      //
      // Linear solves should converge in 2 iterations and Newton converge quadratically. Swap in FSInitial-NH1 to compare.
      //
      // Once that works (you're done for purposes of PSAAP meeting), use the FD approximation here to compare with analytic result below.
      // I think we can reformulate so the code below is cleaner and more closely mirrors the math.

    // dE - Green-Lagrange strain tensor
    // const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    // CeedScalar dEwork[6];
    // for (CeedInt m = 0; m < 6; m++) {
    //   dEwork[m] = 0;
    //   for (CeedInt n = 0; n < 3; n++)
    //     dEwork[m] += (graddeltau[n][indj[m]]*F[n][indk[m]] +
    //                       F[n][indj[m]]*graddeltau[n][indk[m]])/2.;
    // }
    // // *INDENT-OFF*
    // CeedScalar dE[3][3] = {{dEwork[0], dEwork[5], dEwork[4]},
    //                        {dEwork[5], dEwork[1], dEwork[3]},
    //                        {dEwork[4], dEwork[3], dEwork[2]}
    //                       };
    // // *INDENT-ON*
    // // C : right Cauchy-Green tensor
    // // C^(-1) : C-Inverse
    // // *INDENT-OFF*
    // const CeedScalar C_inv[3][3] = {{Cinvwork[0], Cinvwork[5], Cinvwork[4]},
    //                                 {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
    //                                 {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
    //                                };
    // // *INDENT-ON*
    // // compute C_inv2 = C_inv*C_inv = C_inv^2
    // CeedScalar C_inv2[3][3];
    // for (CeedInt j = 0; j < 3; j++)     // Component
    //   for (CeedInt k = 0; k < 3; k++) { // Derivative
    //     C_inv2[j][k] = 0;
    //     for (CeedInt m = 0; m < 3; m++)
    //       C_inv2[j][k] += C_inv[j][m] * C_inv[m][k];
    //     }

    // // -- C_inv:dE
    // CeedScalar Cinv_contract_dE = 0;
    // for (CeedInt j = 0; j < 3; j++)
    //   for (CeedInt k = 0; k < 3; k++)
    //     Cinv_contract_dE += C_inv[j][k]*dE[j][k];

    // // -- C_inv2:dE
    // CeedScalar Cinv2_contract_dE = 0;
    // for (CeedInt j = 0; j < 3; j++)
    //   for (CeedInt k = 0; k < 3; k++)
    //     Cinv2_contract_dE += C_inv2[j][k]*dE[j][k];

    // // -- dE*C_inv
    // CeedScalar dE_Cinv[3][3];
    // for (CeedInt j = 0; j < 3; j++)
    //   for (CeedInt k = 0; k < 3; k++) {
    //     dE_Cinv[j][k] = 0;
    //     for (CeedInt m = 0; m < 3; m++)
    //       dE_Cinv[j][k] += dE[j][m]*C_inv[m][k];
    //   }

    // // -- C_inv*dE*C_inv
    // CeedScalar Cinv_dE_Cinv[3][3];
    // for (CeedInt j = 0; j < 3; j++)
    //   for (CeedInt k = 0; k < 3; k++) {
    //     Cinv_dE_Cinv[j][k] = 0;
    //     for (CeedInt m = 0; m < 3; m++)
    //       Cinv_dE_Cinv[j][k] += C_inv[j][m]*dE_Cinv[m][k];
    //   }

    // // compute dS = mu_1*d2I1bar_dE2:dE + mu_2*d2I2bar_dE2:dE
    // //             + k_1*[(2*J2-J)*(C_inv2:dE)I3 -2*(J2-J)*Cinv*dE*Cinv]

    // CeedScalar J = Jm1 + 1;
    // CeedScalar tr_dE = dE[0][0] + dE[1][1] + dE[2][2];
    // //...d2I1bar_dE2:dE
    // CeedScalar d2I1bar_dE2_dE[6];
    // d2I1bar_dE2_dE[0] = (2/3)*pow(J,-2/3)*I_1*Cinv_dE_Cinv[0][0] - (1/3)*Cinv_contract_dE*( dI1bar_dE[0] + 2*pow(J,-2/3) );
    // d2I1bar_dE2_dE[1] = (2/3)*pow(J,-2/3)*I_1*Cinv_dE_Cinv[1][1] - (1/3)*Cinv_contract_dE*( dI1bar_dE[1] + 2*pow(J,-2/3) );
    // d2I1bar_dE2_dE[2] = (2/3)*pow(J,-2/3)*I_1*Cinv_dE_Cinv[2][2] - (1/3)*Cinv_contract_dE*( dI1bar_dE[2] + 2*pow(J,-2/3) );
    // d2I1bar_dE2_dE[3] = (2/3)*pow(J,-2/3)*I_1*Cinv_dE_Cinv[1][2] - (1/3)*Cinv_contract_dE*( dI1bar_dE[3] );
    // d2I1bar_dE2_dE[4] = (2/3)*pow(J,-2/3)*I_1*Cinv_dE_Cinv[0][2] - (1/3)*Cinv_contract_dE*( dI1bar_dE[4] );
    // d2I1bar_dE2_dE[5] = (2/3)*pow(J,-2/3)*I_1*Cinv_dE_Cinv[0][1] - (1/3)*Cinv_contract_dE*( dI1bar_dE[5] );
    // //...d2I2bar_dE2:dE
    // CeedScalar d2I2bar_dE2_dE[6];
    // d2I2bar_dE2_dE[0] = (4/3)*pow(J,-4/3)*I_2*Cinv_dE_Cinv[0][0] - (2/3)*Cinv_contract_dE*( dI2bar_dE[0] + 2*pow(J,-4/3)*I_1 ) + 2*pow(J,-4/3)*( (5/3)*tr_dE - dE[0][0] );
    // d2I2bar_dE2_dE[1] = (4/3)*pow(J,-4/3)*I_2*Cinv_dE_Cinv[1][1] - (2/3)*Cinv_contract_dE*( dI2bar_dE[1] + 2*pow(J,-4/3)*I_1 ) + 2*pow(J,-4/3)*( (5/3)*tr_dE - dE[1][1] );
    // d2I2bar_dE2_dE[2] = (4/3)*pow(J,-4/3)*I_2*Cinv_dE_Cinv[2][2] - (2/3)*Cinv_contract_dE*( dI2bar_dE[2] + 2*pow(J,-4/3)*I_1 ) + 2*pow(J,-4/3)*( (5/3)*tr_dE - dE[2][2] );
    // d2I2bar_dE2_dE[3] = (4/3)*pow(J,-4/3)*I_2*Cinv_dE_Cinv[1][2] - (2/3)*Cinv_contract_dE*( dI2bar_dE[3] ) + 2*pow(J,-4/3)*(-dE[1][2] );
    // d2I2bar_dE2_dE[4] = (4/3)*pow(J,-4/3)*I_2*Cinv_dE_Cinv[0][2] - (2/3)*Cinv_contract_dE*( dI2bar_dE[4] ) + 2*pow(J,-4/3)*(-dE[0][2] );
    // d2I2bar_dE2_dE[5] = (4/3)*pow(J,-4/3)*I_2*Cinv_dE_Cinv[0][1] - (2/3)*Cinv_contract_dE*( dI2bar_dE[5] ) + 2*pow(J,-4/3)*(-dE[0][1] );
    // // scalar (2*J2-J)*(C_inv2:dE)= J(J + J - 1)*(C_inv2:dE)
    // CeedScalar JJm1Cinv2_contract_dE = J*(J+Jm1) * Cinv2_contract_dE;
    // //...[(2*J2-J)*(C_inv2:dE)I3 -2*(J2-J)*Cinv*dE*Cinv]
    // CeedScalar JJm1CinvdECinv[6];
    // JJm1CinvdECinv[0] = -2 * J*(Jm1)*Cinv_dE_Cinv[0][0] + JJm1Cinv2_contract_dE;
    // JJm1CinvdECinv[1] = -2 * J*(Jm1)*Cinv_dE_Cinv[1][1] + JJm1Cinv2_contract_dE;
    // JJm1CinvdECinv[2] = -2 * J*(Jm1)*Cinv_dE_Cinv[2][2] + JJm1Cinv2_contract_dE;
    // JJm1CinvdECinv[3] = -2 * J*(Jm1)*Cinv_dE_Cinv[1][2];
    // JJm1CinvdECinv[4] = -2 * J*(Jm1)*Cinv_dE_Cinv[0][2];
    // JJm1CinvdECinv[5] = -2 * J*(Jm1)*Cinv_dE_Cinv[0][1];

    // // dS...
    // CeedScalar dSwork[6];
    // dSwork[0] = mu_1*d2I1bar_dE2_dE[0] + mu_2*d2I2bar_dE2_dE[0] + k_1*JJm1CinvdECinv[0];
    // dSwork[1] = mu_1*d2I1bar_dE2_dE[1] + mu_2*d2I2bar_dE2_dE[1] + k_1*JJm1CinvdECinv[1];
    // dSwork[2] = mu_1*d2I1bar_dE2_dE[2] + mu_2*d2I2bar_dE2_dE[2] + k_1*JJm1CinvdECinv[2];
    // dSwork[3] = mu_1*d2I1bar_dE2_dE[3] + mu_2*d2I2bar_dE2_dE[3] + k_1*JJm1CinvdECinv[3];
    // dSwork[4] = mu_1*d2I1bar_dE2_dE[4] + mu_2*d2I2bar_dE2_dE[4] + k_1*JJm1CinvdECinv[4];
    // dSwork[5] = mu_1*d2I1bar_dE2_dE[5] + mu_2*d2I2bar_dE2_dE[5] + k_1*JJm1CinvdECinv[5];

    // // dS
    // // *INDENT-OFF*
    // CeedScalar dS[3][3] = {{dSwork[0], dSwork[5], dSwork[4]},
    //                        {dSwork[5], dSwork[1], dSwork[3]},
    //                        {dSwork[4], dSwork[3], dSwork[2]}
    //                       };
    // // Second Piola-Kirchhoff (S)
    // const CeedScalar S[3][3] = {{Swork[0], Swork[5], Swork[4]},
    //                             {Swork[5], Swork[1], Swork[3]},
    //                             {Swork[4], Swork[3], Swork[2]}
    //                            };
    // dS = (Swork2 - Swork) / epsilon
    CeedScalar dS2[3][3] = {{(Swork2[0] - Swork[0])/ epsilon, (Swork2[5] - Swork[5])/ epsilon, (Swork2[4] - Swork[4])/ epsilon},
                           {(Swork2[5] - Swork[5])/ epsilon, (Swork2[1] - Swork[1])/ epsilon, (Swork2[3] - Swork[3])/ epsilon},
                           {(Swork2[4] - Swork[4])/ epsilon, (Swork2[3] - Swork[3])/ epsilon, (Swork2[2] - Swork[2])/ epsilon}
                          };
    // Second Piola-Kirchhoff (S)
    const CeedScalar S2[3][3] = {{Swork2[0], Swork2[5], Swork2[4]},
                                {Swork2[5], Swork2[1], Swork2[3]},
                                {Swork2[4], Swork2[3], Swork2[2]}
                               };

    // // dP = dPdF:dF = dF*S + F*dS
    // CeedScalar dP[3][3];
    // for (CeedInt j = 0; j < 3; j++)
    //   for (CeedInt k = 0; k < 3; k++) {
    //     dP[j][k] = 0;
    //     for (CeedInt m = 0; m < 3; m++)
    //       dP[j][k] += graddeltau[j][m]*S[m][k] + F[j][m]*dS[m][k];
    //   }

    // // Apply dXdx^T and weight
    // for (CeedInt j = 0; j < 3; j++)     // Component
    //   for (CeedInt k = 0; k < 3; k++) { // Derivative
    //     deltadvdX[k][j][i] = 0;
    //     for (CeedInt m = 0; m < 3; m++)
    //       deltadvdX[k][j][i] += dXdx[k][m] * dP[j][m] * wdetJ;
    //   }
    
    // dP = dPdF:dF = dF*S + F*dS
    CeedScalar dP[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        dP[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          dP[j][k] += graddeltau[j][m]*S2[m][k] + F[j][m]*dS2[m][k];
      }

    // Apply dXdx^T and weight
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        deltadvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltadvdX[k][j][i] += dXdx[k][m] * dP[j][m] * wdetJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Strain energy computation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialMR1Energy)(void *ctx, CeedInt Q,
                                       const CeedScalar *const *in,
                                       CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*energy) = (CeedScalar(*))out[0];
  // *INDENT-ON*

  // Context
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1  = context->mu_1;
  const CeedScalar mu_2 = context->mu_2;
  const CeedScalar k_1 = context->k_1;

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
    const CeedScalar wdetJ      =   q_data[0][i];
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    CeedScalar grad_u[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++)
          grad_u[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += grad_u[n][indj[m]]*grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*
    const CeedScalar Jm1 = computeJM1(grad_u);

    // C : right Cauchy-Green tensor
    // C = I + 2E
    // *INDENT-OFF*
    const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                                {E2[0][1], 1 + E2[1][1], E2[1][2]},
                                {E2[0][2], E2[1][2], 1 + E2[2][2]}
                               };
    // *INDENT-ON*
    // compute CC = C*C = C^2
    CeedScalar CC[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        CC[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
            CC[j][k] += C[j][m] * C[m][k];
        }

    CeedScalar J = Jm1 + 1;
    // compute invariants
    // I_1 = trace(C)
    const CeedScalar I_1 = C[0][0] + C[1][1] + C[2][2];
    // trace(C^2)
    const CeedScalar tr_CC = CC[0][0] + CC[1][1] + CC[2][2];
    // I_2 = 0.5(I_1^2 - trace(C^2))
    const CeedScalar I_2 = 0.5*(pow(I_1, 2) - tr_CC);
    const CeedScalar I1_bar = pow(J,-2/3)*I_1;
    const CeedScalar I2_bar = pow(J,-4/3)*I_2;


    // Strain energy Phi(E) for Moony-Rivlin
    energy[i] = (0.5*mu_1*(I1_bar - 3) + 0.5*mu_2*(I2_bar - 3) + 0.5*k_1*(Jm1)*(Jm1)) * wdetJ;

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Nodal diagnostic quantities for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(ElasFSInitialMR1Diagnostic)(void *ctx, CeedInt Q,
    const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*diagnostic)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1  = context->mu_1;
  const CeedScalar mu_2 = context->mu_2;
  const CeedScalar k_1 = context->k_1;

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
    const CeedScalar dXdx[3][3] = {{q_data[1][i],
                                    q_data[2][i],
                                    q_data[3][i]},
                                   {q_data[4][i],
                                    q_data[5][i],
                                    q_data[6][i]},
                                   {q_data[7][i],
                                    q_data[8][i],
                                    q_data[9][i]}
                                  };
    // *INDENT-ON*

    // Compute grad_u
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to du = grad_u
    CeedScalar grad_u[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        grad_u[j][k] = 0;
        for (int m = 0; m < 3; m++)
          grad_u[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (grad_u + grad_u^T + grad_u^T*grad_u)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = grad_u[indj[m]][indk[m]] + grad_u[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += grad_u[n][indj[m]]*grad_u[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*

    // Displacement
    diagnostic[0][i] = u[0][i];
    diagnostic[1][i] = u[1][i];
    diagnostic[2][i] = u[2][i];

    // Pressure
    const CeedScalar Jm1 = computeJM1(grad_u);
    diagnostic[3][i] = -k_1*Jm1;

    // Stress tensor invariants
    diagnostic[4][i] = (E2[0][0] + E2[1][1] + E2[2][2]) / 2.;
    diagnostic[5][i] = 0.;
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt m = 0; m < 3; m++)
        diagnostic[5][i] += E2[j][m] * E2[m][j] / 4.;
    diagnostic[6][i] = Jm1 + 1.;

    // C : right Cauchy-Green tensor
    // C = I + 2E
    // *INDENT-OFF*
    const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                                {E2[0][1], 1 + E2[1][1], E2[1][2]},
                                {E2[0][2], E2[1][2], 1 + E2[2][2]}
                               };
    // *INDENT-ON*
    // compute CC = C*C = C^2
    CeedScalar CC[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        CC[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
            CC[j][k] += C[j][m] * C[m][k];
        }

    CeedScalar J = Jm1 + 1;
    // compute invariants
    // I_1 = trace(C)
    const CeedScalar I_1 = C[0][0] + C[1][1] + C[2][2];
    // trace(C^2)
    const CeedScalar tr_CC = CC[0][0] + CC[1][1] + CC[2][2];
    // I_2 = 0.5(I_1^2 - trace(C^2))
    const CeedScalar I_2 = 0.5*(pow(I_1, 2) - tr_CC);
    const CeedScalar I1_bar = pow(J, -2/3)*I_1;
    const CeedScalar I2_bar = pow(J, -4/3)*I_2;


    // Strain energy
    diagnostic[7][i] = (0.5*mu_1*(I1_bar - 3) + 0.5*mu_2*(I2_bar - 3) + 0.5*k_1*(Jm1)*(Jm1));

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End of ELAS_FSInitialMR1_H
