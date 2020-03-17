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

#ifndef HYPER_FS_H
#define HYPER_FS_H

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
  const Physics context = ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu /2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // Formulation Terminology:
  //  I3    : 3x3 Identity matrix
  //  C     : right Cauchy-Green tensor
  //  C_inv : inverse of C
  //  F     : deformation gradient
  //  S     : 2nd Piola-Kirchhoff (in current config)
  //  P     : 1st Piola-Kirchhoff (in referential config)
  // Formulation:
  //  F =  I3 + grad_ue
  //  J = det(F)
  //  C = F(^T)*F
  //  S = mu*I3 + (lambda*log(J)-mu)*C_inv;
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
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        gradu[j][k][i] = 0;
        for (int m = 0; m < 3; m++)
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

    // Compute the determinant of F
    const CeedScalar Fa00 = F[1][1]*F[2][2] - F[1][2]*F[2][1];
    const CeedScalar Fa01 = F[0][2]*F[2][1] - F[0][1]*F[2][2];
    const CeedScalar Fa02 = F[0][1]*F[1][2] - F[0][2]*F[1][1];
    const CeedScalar detF = F[0][0]*Fa00 + F[1][0]*Fa01 + F[2][0]*Fa02;

    // C : right Cauchy-Green tensor
    // C = F^T * F (^T  means Transpose)
    CeedScalar C[3][3];
    for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
            C[j][k] = 0;
            for(int m = 0; m < 3; m++)
               C[j][k] += F[k][m] * F[j][m];
        }


    // Compute C^(-1) : C-Inverse
    const CeedScalar A00 = C[1][1]*C[2][2] - C[1][2]*C[2][1];
    const CeedScalar A01 = C[0][2]*F[2][1] - C[0][1]*C[2][2];
    const CeedScalar A02 = C[0][1]*C[1][2] - C[0][2]*C[1][1];
    const CeedScalar A10 = C[1][2]*C[2][0] - C[1][0]*C[2][2];
    const CeedScalar A11 = C[0][0]*C[2][2] - C[0][2]*C[2][0];
    const CeedScalar A12 = C[0][2]*C[1][0] - C[0][0]*C[1][2];
    const CeedScalar A20 = C[1][0]*C[2][1] - C[1][1]*C[2][0];
    const CeedScalar A21 = C[0][1]*C[2][0] - C[0][0]*C[2][1];
    const CeedScalar A22 = C[0][0]*C[1][1] - C[0][1]*C[1][0];
    const CeedScalar detC = C[0][0]*A00 + C[1][0]*A01 + C[2][0]*A02;

    // *INDENT-OFF*
    const CeedScalar C_inv[3][3] = {{A00/detC, A01/detC, A02/detC},
                                    {A10/detC, A11/detC, A12/detC},
                                    {A20/detC, A21/detC, A22/detC}
                                   };

    // Compute the Second Piola-Kirchhoff (S)
    const CeedScalar llnj_m = lambda *log(detF)-mu;
    CeedScalar S[3][3] = {{mu + llnj_m* C_inv[0][0],      llnj_m* C_inv[0][1],      llnj_m* C_inv[0][2]},
                          {     llnj_m* C_inv[1][0], mu + llnj_m* C_inv[1][1],      llnj_m* C_inv[1][2]},
                          {     llnj_m* C_inv[2][0],      llnj_m* C_inv[2][1], mu + llnj_m* C_inv[2][2]}
                         };
    // *INDENT-ON*

    // Compute the First Piola-Kirchhoff : P = F*S
    CeedScalar P[3][3];
    for(int j = 0; j < 3; j++)
       for(int k = 0; k < 3; k++) {
          P[j][k] = 0;
           for(int m = 0; m < 3; m++)
              P[j][k] += F[m][k] * S[j][m];
       }


    // Apply dXdx^T and weight to P (First Piola-Kirchhoff)
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        dvdX[k][j][i] = 0;
        for (int m = 0; m < 3; m++)
          dvdX[k][j][i] += dXdx[k][m] * P[j][m] * wJ;
      }

    } // End of Quadrature Point Loop

  return 0;
}

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
  const Physics context = ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;

  // Constants
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); //bulk Modulus
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
    // Apply dXdx to deltadu = graddeltau (9x1 vector [u1,x  u2,x  u3,x  u1,y ....]) to multiply by reshaped 9x9 dPdF
    CeedScalar graddeltau[9];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        graddeltau[3*j+k] = 0;
        for (int m =0 ; m < 3; m++)
          graddeltau[3*j+k] += dXdx[m][k] * deltadu[j][m];
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

    // Compute the F_inv componets and det(F)
    const CeedScalar Fa00 = F[1][1]*F[2][2] - F[1][2]*F[2][1];
    const CeedScalar Fa01 = F[0][2]*F[2][1] - F[0][1]*F[2][2];
    const CeedScalar Fa02 = F[0][1]*F[1][2] - F[0][2]*F[1][1];
    const CeedScalar Fa10 = F[1][2]*F[2][0] - F[1][0]*F[2][2];
    const CeedScalar Fa11 = F[0][0]*F[2][2] - F[0][2]*F[2][0];
    const CeedScalar Fa12 = F[0][2]*F[1][0] - F[0][0]*F[1][2];
    const CeedScalar Fa20 = F[1][0]*F[2][1] - F[1][1]*F[2][0];
    const CeedScalar Fa21 = F[0][1]*F[2][0] - F[0][0]*F[2][1];
    const CeedScalar Fa22 = F[0][0]*F[1][1] - F[0][1]*F[1][0];
    const CeedScalar detF = F[0][0]*Fa00 + F[1][0]*Fa01 + F[2][0]*Fa02;

    // *INDENT-OFF*
    const CeedScalar F_inv[3][3] = {{Fa00/detF, Fa01/detF, Fa02/detF},
                                    {Fa10/detF, Fa11/detF, Fa12/detF},
                                    {Fa20/detF, Fa21/detF, Fa22/detF}
                                   };
    // *INDENT-ON*

    // C : right Cauchy-Green tensor
    // C = F^T * F (^T  means Transpose)
    CeedScalar C[3][3];
    for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
            C[j][k] = 0;
            for(int m = 0; m < 3; m++)
               C[j][k] += F[k][m] * F[j][m];
        }


    // Compute C^(-1) : C-Inverse
    const CeedScalar A00 = C[1][1]*C[2][2] - C[1][2]*C[2][1];
    const CeedScalar A01 = C[0][2]*F[2][1] - C[0][1]*C[2][2];
    const CeedScalar A02 = C[0][1]*C[1][2] - C[0][2]*C[1][1];
    const CeedScalar A10 = C[1][2]*C[2][0] - C[1][0]*C[2][2];
    const CeedScalar A11 = C[0][0]*C[2][2] - C[0][2]*C[2][0];
    const CeedScalar A12 = C[0][2]*C[1][0] - C[0][0]*C[1][2];
    const CeedScalar A20 = C[1][0]*C[2][1] - C[1][1]*C[2][0];
    const CeedScalar A21 = C[0][1]*C[2][0] - C[0][0]*C[2][1];
    const CeedScalar A22 = C[0][0]*C[1][1] - C[0][1]*C[1][0];
    const CeedScalar detC = C[0][0]*A00 + C[1][0]*A01 + C[2][0]*A02;

    // *INDENT-OFF*
    const CeedScalar C_inv[3][3] = {{A00/detC, A01/detC, A02/detC},
                                    {A10/detC, A11/detC, A12/detC},
                                    {A20/detC, A21/detC, A22/detC}
                                   };

    // Compute the Second Piola-Kirchhoff (S)
    const CeedScalar llnj_m = lambda *log(detF)-mu;
    CeedScalar S[3][3] = {{mu + llnj_m*C_inv[0][0],      llnj_m*C_inv[0][1],      llnj_m*C_inv[0][2]},
                          {     llnj_m*C_inv[1][0], mu + llnj_m*C_inv[1][1],      llnj_m*C_inv[1][2]},
                          {     llnj_m*C_inv[2][0],      llnj_m*C_inv[2][1], mu + llnj_m*C_inv[2][2]}
                         };
    // *INDENT-ON*

    // dPdF is indexed as an array of 81  elements
    CeedScalar dPdF[81];
    for (int j = 0; j < 3; j++)
      for(int J = 0; J < 3; J++)
        for(int a = 0; a < 3; a++)
          for(int A = 0; A < 3; A++)
            dPdF[27*j+9*J+3*a+A] = (a == j ? 1 : 0) * (S[A][J]) +
                                   lambda*F_inv[A][a]*F_inv[J][j]
                                   - (lambda*log(detF) - mu) *
                                   (F_inv[A][j]*F_inv[J][a] +
                                   (a == j ? 1 : 0) * C_inv[A][J]);

    // dPdF is indexed as a 9x9 matrix
    CeedScalar dPdF_grad_del_u[9];
    for(int j = 0; j < 9; j++) {
       dPdF_grad_del_u[j] = 0.;
       for(int k = 0; k < 9; k++)
          dPdF_grad_del_u[j] += dPdF[9*j+k]* graddeltau[k];
    }


    // Apply dXdx^T and weight
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        deltadvdX[k][j][i] = 0;
        for (int m = 0; m < 3; m++)
          deltadvdX[k][j][i] += dXdx[k][m] * dPdF_grad_del_u[3*j+m] * wJ;
      }

    } // End of Quadrature Point Loop

   return 0;
}
// -----------------------------------------------------------------------------
#endif // End of HYPER_FS_H
