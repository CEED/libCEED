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

#ifndef HYPER_FS_H
#define HYPER_FS_H

#ifndef __CUDACC__
#  include <math.h>
#endif

#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;
struct Physics_private {
  CeedScalar   nu;      // Poisson's ratio
  CeedScalar   E;       // Young's Modulus
};
#endif

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(HyperFSF)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*qdata)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // Store gradu for HyperFSdF (Jacobian of HyperFSF)
  CeedScalar (*gradu)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Formulation Terminology:
  //  I3    : 3x3 Identity matrix
  //  C     : right Cauchy-Green tensor
  //  Cinv  : inverse of C
  //  F     : deformation gradient
  //  S     : 2nd Piola-Kirchhoff (in current config)
  //  P     : 1st Piola-Kirchhoff (in referential config)
  // Formulation:
  //  F =  I3 + grad_ue
  //  J = det(F)
  //  C = F(^T)*F
  //  S = mu*I3 + (lambda*log(J)-mu)*Cinv;
  //  P = F*S

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
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        gradu[j][k][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          gradu[j][k][i] += dXdx[m][k] * du[j][m];
      }

    // I3 : 3x3 Identity matrix
    // Compute The Deformation Gradient : F = I3 + gradu
    // *INDENT-OFF*
    const CeedScalar F[3][3] =  {{gradu[0][0][i] + 1,
                                  gradu[0][1][i],
                                  gradu[0][2][i]},
                                 {gradu[1][0][i],
                                  gradu[1][1][i] + 1,
                                  gradu[1][2][i]},
                                 {gradu[2][0][i],
                                  gradu[2][1][i],
                                  gradu[2][2][i] + 1}
                                };
    // *INDENT-ON*

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (gradu + gradu^T + gradu^T*gradu)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = gradu[indj[m]][indk[m]][i] + gradu[indk[m]][indj[m]][i];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += gradu[indj[m]][n][i]*gradu[indk[m]][n][i];
    }           
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*
    const CeedScalar detC_m1 = E2[0][0]*(E2[1][1]*E2[2][2]-E2[1][2]*E2[1][2]) +
                               E2[0][1]*(E2[0][2]*E2[1][2]-E2[0][1]*E2[2][2]) +
                               E2[0][2]*(E2[0][1]*E2[1][2]-E2[0][2]*E2[1][1]) +
                               E2[0][0] + E2[1][1] + E2[2][2] +
                               E2[0][0]*E2[1][1] + E2[0][0]*E2[2][2] +
                               E2[1][1]*E2[2][2] - E2[0][1]*E2[0][1] -
                               E2[0][2]*E2[0][2] - E2[1][2]*E2[1][2];

    // C : right Cauchy-Green tensor
    // C = I + 2E
    // *INDENT-OFF*
    const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                                {E2[0][1], 1 + E2[1][1], E2[1][2]},
                                {E2[0][2], E2[1][2], 1 + E2[2][2]}
                               };
    // *INDENT-ON*

    // Compute C^(-1) : C-Inverse
    const CeedScalar A00 = C[1][1]*C[2][2] - C[1][2]*C[2][1];
    const CeedScalar A01 = C[0][2]*C[2][1] - C[0][1]*C[2][2];
    const CeedScalar A02 = C[0][1]*C[1][2] - C[0][2]*C[1][1];
    const CeedScalar A11 = C[0][0]*C[2][2] - C[0][2]*C[2][0];
    const CeedScalar A12 = C[0][2]*C[1][0] - C[0][0]*C[1][2];
    const CeedScalar A22 = C[0][0]*C[1][1] - C[0][1]*C[1][0];
    const CeedScalar Cinv00 = A00/(detC_m1 + 1.), Cinv01 = A01/(detC_m1 + 1.),
                     Cinv02 = A02/(detC_m1 + 1.), Cinv11 = A11/(detC_m1 + 1.),
                     Cinv12 = A12/(detC_m1 + 1.), Cinv22 = A22/(detC_m1 + 1.);
    // *INDENT-OFF*
    const CeedScalar Cinv[3][3] = {{Cinv00, Cinv01, Cinv02},
                                   {Cinv01, Cinv11, Cinv12},
                                   {Cinv02, Cinv12, Cinv22}
                                  };
    // *INDENT-ON*

    // Compute the Second Piola-Kirchhoff (S)
    const CeedScalar llnj = lambda*log1p(detC_m1)/2.;
    const CeedScalar S00 = llnj*Cinv[0][0] +
                           mu*(Cinv[0][0]*E2[0][0] + Cinv[0][1]*E2[1][0] +
                               Cinv[0][2]*E2[2][0]),
                     S01 = llnj*Cinv[0][1] +
                           mu*(Cinv[0][0]*E2[0][1] + Cinv[0][1]*E2[1][1] +
                               Cinv[0][2]*E2[2][1]),
                     S02 = llnj*Cinv[0][2] +
                           mu*(Cinv[0][0]*E2[0][2] + Cinv[0][1]*E2[1][2] +
                               Cinv[0][2]*E2[2][2]),
                     S11 = llnj*Cinv[1][1] +
                           mu*(Cinv[1][0]*E2[0][1] + Cinv[1][1]*E2[1][1] +
                               Cinv[1][2]*E2[2][1]),
                     S12 = llnj*Cinv[1][2] +
                           mu*(Cinv[1][0]*E2[0][2] + Cinv[1][1]*E2[1][2] +
                               Cinv[1][2]*E2[2][2]),
                     S22 = llnj*Cinv[2][2] +
                           mu*(Cinv[2][0]*E2[0][2] + Cinv[2][1]*E2[1][2] +
                               Cinv[2][2]*E2[2][2]);
    // *INDENT-OFF*
    CeedScalar S[3][3] = {{S00, S01, S02},
                          {S01, S11, S12},
                          {S02, S12, S22}
                         };
    // *INDENT-ON*

    // Compute the First Piola-Kirchhoff : P = F*S
    CeedScalar P[3][3];
    for (CeedInt j = 0; j < 3; j++)
       for (CeedInt k = 0; k < 3; k++) {
          P[j][k] = 0;
           for (CeedInt m = 0; m < 3; m++)
              P[j][k] += F[m][k] * S[j][m];
       }

    // Apply dXdx^T and weight to P (First Piola-Kirchhoff)
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        dvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          dvdX[k][j][i] += dXdx[k][m] * P[j][m] * wJ;
      }

    } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Jacobian evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(HyperFSdF)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                          CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*deltaug)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*qdata)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])in[1];
  // gradu is used for hyperelasticity (non-linear)
  const CeedScalar (*gradu)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*deltadvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;

  // Constants
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

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
    //   dXdx = (dx/dX)^(-1)
    // Apply dXdx to deltadu = graddelta
    CeedScalar graddeltau[3][3];
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        graddeltau[j][k] = 0;
        for (CeedInt m =0 ; m < 3; m++)
          graddeltau[j][k] += dXdx[m][k] * deltadu[j][m];
      }

    // I3 : 3x3 Identity matrix
    // Compute The Deformation Gradient : F = I3 + gradu
    // *INDENT-OFF*
    const CeedScalar F[3][3] =      {{gradu[0][0][i] + 1,
                                      gradu[0][1][i],
                                      gradu[0][2][i]},
                                     {gradu[1][0][i],
                                      gradu[1][1][i] + 1,
                                      gradu[1][2][i]},
                                     {gradu[2][0][i],
                                      gradu[2][1][i],
                                      gradu[2][2][i] + 1}
                                    };
    // *INDENT-ON*

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (gradu + gradu^T + gradu^T*gradu)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = gradu[indj[m]][indk[m]][i] + gradu[indk[m]][indj[m]][i];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += gradu[indj[m]][n][i]*gradu[indk[m]][n][i];
    }           
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*
    const CeedScalar detC_m1 = E2[0][0]*(E2[1][1]*E2[2][2]-E2[1][2]*E2[1][2]) +
                               E2[0][1]*(E2[0][2]*E2[1][2]-E2[0][1]*E2[2][2]) +
                               E2[0][2]*(E2[0][1]*E2[1][2]-E2[0][2]*E2[1][1]) +
                               E2[0][0] + E2[1][1] + E2[2][2] +
                               E2[0][0]*E2[1][1] + E2[0][0]*E2[2][2] +
                               E2[1][1]*E2[2][2] - E2[0][1]*E2[0][1] -
                               E2[0][2]*E2[0][2] - E2[1][2]*E2[1][2];

    // deltaE - Green-Lagrange strain tensor
    CeedScalar deltaEwork[6];
    for (CeedInt m = 0; m < 6; m++) {
      deltaEwork[m] = 0;
      for (CeedInt n = 0; n < 3; n++)
        deltaEwork[m] += (graddeltau[n][indj[m]]*F[n][indk[m]] +
                          F[n][indj[m]]*graddeltau[n][indk[m]])/2.;
    }           
    // *INDENT-OFF*
    CeedScalar deltaE[3][3] = {{deltaEwork[0], deltaEwork[5], deltaEwork[4]},
                               {deltaEwork[5], deltaEwork[1], deltaEwork[3]},
                               {deltaEwork[4], deltaEwork[3], deltaEwork[2]}
                              };
    // *INDENT-ON*

    // C : right Cauchy-Green tensor
    // C = I + 2E
    // *INDENT-OFF*
    const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                                {E2[0][1], 1 + E2[1][1], E2[1][2]},
                                {E2[0][2], E2[1][2], 1 + E2[2][2]}
                               };
    // *INDENT-ON*

    // Compute C^(-1) : C-Inverse
    const CeedScalar A00 = C[1][1]*C[2][2] - C[1][2]*C[2][1];
    const CeedScalar A01 = C[0][2]*C[2][1] - C[0][1]*C[2][2];
    const CeedScalar A02 = C[0][1]*C[1][2] - C[0][2]*C[1][1];
    const CeedScalar A11 = C[0][0]*C[2][2] - C[0][2]*C[2][0];
    const CeedScalar A12 = C[0][2]*C[1][0] - C[0][0]*C[1][2];
    const CeedScalar A22 = C[0][0]*C[1][1] - C[0][1]*C[1][0];
    const CeedScalar Cinv00 = A00/(detC_m1 + 1.), Cinv01 = A01/(detC_m1 + 1.),
                     Cinv02 = A02/(detC_m1 + 1.), Cinv11 = A11/(detC_m1 + 1.),
                     Cinv12 = A12/(detC_m1 + 1.), Cinv22 = A22/(detC_m1 + 1.);
    // *INDENT-OFF*
    const CeedScalar Cinv[3][3] = {{Cinv00, Cinv01, Cinv02},
                                   {Cinv01, Cinv11, Cinv12},
                                   {Cinv02, Cinv12, Cinv22}
                                  };
    // *INDENT-ON*

    // Compute the Second Piola-Kirchhoff (S)
    const CeedScalar llnj = lambda*log1p(detC_m1)/2.;
    const CeedScalar S00 = llnj*Cinv[0][0] +
                           mu*(Cinv[0][0]*E2[0][0] + Cinv[0][1]*E2[1][0] +
                               Cinv[0][2]*E2[2][0]),
                     S01 = llnj*Cinv[0][1] +
                           mu*(Cinv[0][0]*E2[0][1] + Cinv[0][1]*E2[1][1] +
                               Cinv[0][2]*E2[2][1]),
                     S02 = llnj*Cinv[0][2] +
                           mu*(Cinv[0][0]*E2[0][2] + Cinv[0][1]*E2[1][2] +
                               Cinv[0][2]*E2[2][2]),
                     S11 = llnj*Cinv[1][1] +
                           mu*(Cinv[1][0]*E2[0][1] + Cinv[1][1]*E2[1][1] +
                               Cinv[1][2]*E2[2][1]),
                     S12 = llnj*Cinv[1][2] +
                           mu*(Cinv[1][0]*E2[0][2] + Cinv[1][1]*E2[1][2] +
                               Cinv[1][2]*E2[2][2]),
                     S22 = llnj*Cinv[2][2] +
                           mu*(Cinv[2][0]*E2[0][2] + Cinv[2][1]*E2[1][2] +
                               Cinv[2][2]*E2[2][2]);
    // *INDENT-OFF*
    CeedScalar S[3][3] = {{S00, S01, S02},
                          {S01, S11, S12},
                          {S02, S12, S22}
                         };
    // *INDENT-ON*

    // deltaS = dSdE:deltaE
    //      = lambda (Cinv:deltaE)Cinv + 2(mu-lambda*log(J))Cinv*deltaE*Cinv
    // -- Cinv:deltaE
    CeedScalar Cinv_contract_E = 0;
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++)
        Cinv_contract_E += Cinv[j][k]*deltaE[j][k];
    // -- deltaE*Cinv
    CeedScalar deltaECinv[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        deltaECinv[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
            deltaECinv[j][k] += deltaE[j][m]*Cinv[m][k];
      }
    // -- intermediate deltaS = Cinv*deltaE*Cinv
    CeedScalar deltaS[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        deltaS[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltaS[j][k] += Cinv[j][m]*deltaECinv[m][k];
      }
    // -- deltaS = lambda (Cinv:deltaE)Cinv - 2(lambda*log(J)-mu)*(intermediate)
    const CeedScalar llnj_m = llnj - mu;
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++)
        deltaS[j][k] = lambda*Cinv_contract_E*Cinv[j][k] -
                       2.*llnj_m*deltaS[j][k];
    
    // deltaP = dPdF:deltaF = deltaF*S + F*deltaS
    CeedScalar deltaP[3][3];
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt k = 0; k < 3; k++) {
        deltaP[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltaP[j][k] += graddeltau[j][m]*S[m][k] + F[j][m]*deltaS[m][k];
      }

    // Apply dXdx^T and weight
    for (CeedInt j = 0; j < 3; j++)     // Component
      for (CeedInt k = 0; k < 3; k++) { // Derivative
        deltadvdX[k][j][i] = 0;
        for (CeedInt m = 0; m < 3; m++)
          deltadvdX[k][j][i] += dXdx[k][m] * deltaP[j][m] * wJ;
      }

    } // End of Quadrature Point Loop

   return 0;
}
// -----------------------------------------------------------------------------
#endif // End of HYPER_FS_H
