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

// -----------------------------------------------------------------------------
// Neo-Hookean context
#ifndef PHYSICS_STRUCT
#define PHYSICS_STRUCT
typedef struct Physics_private *Physics;

struct Physics_private { 
  CeedScalar   nu;      // Poisson's ratio
  CeedScalar   E;       // Young's Modulus
};
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
// Generalized Polynomial context
#ifndef PHYSICS_STRUCT_GP
#define PHYSICS_STRUCT_GP
typedef struct Physics_private_GP *Physics_GP;

struct Physics_private_GP { 
  CeedScalar   nu;      // Poisson's ratio rm
  CeedScalar   E;       // Young's Modulus rm
  //material properties for GP
  CeedScalar C_mat[6][6]; // 2D matrix
  CeedScalar K[6]; // 1D array
  CeedScalar N; // max value of the sum; usually 1 or 2
};

#endif
// -----------------------------------------------------------------------------
// end of contexts
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Series approximation of log1p()
//  log1p() is not vectorized in libc
//
//  The series expansion is accurate to 1e-7 in the range sqrt(2)/2 < J < sqrt(2),
//  with machine precision accuracy near J=1.  The initialization extends this range
//  to 0.35 ~= sqrt(2)/4 < J < sqrt(2)*2 ~= 2.83, which should be sufficient for
//  applications of the Neo-Hookean model.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Compute det C - 1
// -----------------------------------------------------------------------------
static inline CeedScalar computeDetCM1(CeedScalar E2work[6]) {
  return E2work[0]*(E2work[1]*E2work[2]-E2work[3]*E2work[3]) +
         E2work[5]*(E2work[4]*E2work[3]-E2work[5]*E2work[2]) +
         E2work[4]*(E2work[5]*E2work[3]-E2work[4]*E2work[1]) +
         E2work[0] + E2work[1] + E2work[2] +
         E2work[0]*E2work[1] + E2work[0]*E2work[2] +
         E2work[1]*E2work[2] - E2work[5]*E2work[5] -
         E2work[4]*E2work[4] - E2work[3]*E2work[3];
};


// Functions for various material models
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Neo-Hookean model

CeedScalar NH_energyModel(void *ctx, const CeedScalar detC_m1, CeedScalar E2[][3], CeedScalar wdetJ){
  //requires unpacking the ctx struct again; 
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;
  
  CeedScalar logj = log1p_series_shifted(detC_m1)/2.;

  return (lambda*logj*logj/2. - mu*logj + mu*(E2[0][0] + E2[1][1] + E2[2][2])/2.)* wdetJ;
}

// -----------------------------------------------------------------------------
// Mooney-Rivlin model
CeedScalar MR_energyModel(void *ctx, CeedScalar detC_m1, CeedScalar E2[][3], CeedScalar wdetJ){
  //requires unpacking the ctx struct again; 
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1 = context -> mu_1; // material constant mu_1
  const CeedScalar mu_2 = context -> mu_2; // material constant mu_2
  const CeedScalar k_1 = context -> k_1; // material constant k_1

  //energy model:
  const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                              {E2[0][1], 1 + E2[1][1], E2[1][2]},
                              {E2[0][2], E2[1][2], 1 + E2[2][2]}
                             };
  
  CeedScalar I_1 = C[0][0] + C[1][1] + C[2][2];
  CeedScalar I_2 = pow(I_1, 2);
  CeedScalar bar_I_1 = pow(detC_m1, -2/3)* I_1; //using detC_m1 as J
  CeedScalar bar_I_2 = pow(detC_m1, -4/3)* I_2; //using detC_m1 as J

  // phi = (mu_1/2)(bar_I_1 - 3) + (mu_2/2)(bar_I_2 - 3) + (K/2)(J-1)^2
  return (mu_1/2)*(bar_I_1 - 3) + (mu_2/2)*(bar_I_2 - 3) + (k_1/2) * pow((detC_m1-1), 2);
}
// -----------------------------------------------------------------------------
// Generalized Polynomial model
CeedScalar GP_energyModel(void *ctx, CeedScalar detC_m1, CeedScalar E2[][3], CeedScalar wdetJ){
  //requires unpacking the ctx struct again; 
  // const Physics_GP context = (Physics_GP)ctx;
  // const CeedScalar E  = context->E;
  // const CeedScalar nu = context->nu;
  // const CeedScalar TwoMu = E / (1 + nu);
  // const CeedScalar mu = TwoMu / 2;
  // const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  // const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;
  // TO-DO unpack remaining needed for GP
  // const CeedScalar *C_mat = context -> C_mat; //material constant C
  // const CeedScalar *K = context -> K; //material constant K 
  // const CeedScalar N = context -> N; //Max value to sum to
  
  //TO-DO update with correct energy model
  // phi = sum_{i+j = 1}^N C_mat_[i,j](\bar_I_1 -3)^i(\bar_I_2 -3)^j + sum_{i=1}^N(K[i]/2)(J - 1)^{2i}
  return 0;
}
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Energy derivations S for models; PASS IN FOR SWORK_FUNC IN COMMONFS_GENERIC
// -----------------------------------------------------------------------------
// Neo-Hookean model
CeedScalar NH_2nd_PK(void *ctx, CeedScalar Swork[6], CeedScalar Cinvwork[6], const CeedScalar *detC_m1,
                           CeedScalar *llnj, const CeedInt indj[6], const CeedInt indk[6], CeedScalar E2[][3]){
  // unpack ctx here
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // C : right Cauchy-Green tensor
  // C = I + 2E
  // *INDENT-OFF*
  const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                              {E2[0][1], 1 + E2[1][1], E2[1][2]},
                              {E2[0][2], E2[1][2], 1 + E2[2][2]}
                             };
  // *INDENT-ON*

  // Compute C^(-1) : C-Inverse
  CeedScalar A[6] = {C[1][1]*C[2][2] - C[1][2]*C[2][1], /* *NOPAD* */
                     C[0][0]*C[2][2] - C[0][2]*C[2][0], /* *NOPAD* */
                     C[0][0]*C[1][1] - C[0][1]*C[1][0], /* *NOPAD* */
                     C[0][2]*C[1][0] - C[0][0]*C[1][2], /* *NOPAD* */
                     C[0][1]*C[1][2] - C[0][2]*C[1][1], /* *NOPAD* */
                     C[0][2]*C[2][1] - C[0][1]*C[2][2] /* *NOPAD* */
                    };
  for (CeedInt m = 0; m < 6; m++)
    Cinvwork[m] = A[m] / (*detC_m1 + 1.);

  // *INDENT-OFF* //
  const CeedScalar Cinv[3][3] = {{Cinvwork[0], Cinvwork[5], Cinvwork[4]},
                                 {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
                                 {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
                                };

  // calculate 2nd PK 
  (*llnj) = lambda*log1p_series_shifted(*detC_m1)/2.; // lambda*logJ/2
  for (CeedInt m = 0; m < 6; m++) {
    Swork[m] = (*llnj)*Cinvwork[m];
    for (CeedInt n = 0; n < 3; n++)
      Swork[m] += mu*Cinv[indj[m]][n]*E2[n][indk[m]];
  }
  return 0;
}
// -----------------------------------------------------------------------------
// Mooney-Rivlin model
CeedScalar MR_2nd_PK(void *ctx, CeedScalar Swork[6], CeedScalar Cinvwork[6], const CeedScalar *detC_m1,
                           CeedScalar *llnj, const CeedInt indj[6], const CeedInt indk[6], CeedScalar E2[][3]){
  // unoack context
  const Physics_MR context = (Physics_MR)ctx;
  const CeedScalar mu_1 = context -> mu_1; // material constant mu_1
  const CeedScalar mu_2 = context -> mu_2; // material constant mu_2
  const CeedScalar k_1 = context -> k_1; // material constant k_1
  // C : right Cauchy-Green tensor
  // C = I + 2E
  // *INDENT-OFF*
  const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
                              {E2[0][1], 1 + E2[1][1], E2[1][2]},
                              {E2[0][2], E2[1][2], 1 + E2[2][2]}
                             };
  // *INDENT-ON*

  // Compute C^(-1) : C-Inverse
  CeedScalar A[6] = {C[1][1]*C[2][2] - C[1][2]*C[2][1], /* *NOPAD* */
                     C[0][0]*C[2][2] - C[0][2]*C[2][0], /* *NOPAD* */
                     C[0][0]*C[1][1] - C[0][1]*C[1][0], /* *NOPAD* */
                     C[0][2]*C[1][0] - C[0][0]*C[1][2], /* *NOPAD* */
                     C[0][1]*C[1][2] - C[0][2]*C[1][1], /* *NOPAD* */
                     C[0][2]*C[2][1] - C[0][1]*C[2][2] /* *NOPAD* */
                    };
  for (CeedInt m = 0; m < 6; m++)
    Cinvwork[m] = A[m] / (*detC_m1 + 1.);

  // *INDENT-OFF* //
  const CeedScalar Cinv[3][3] = {{Cinvwork[0], Cinvwork[5], Cinvwork[4]},
                                 {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
                                 {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
                                };

  // compute invariants
  CeedScalar I_1 = C[0][0] + C[1][1] + C[2][2];
  CeedScalar I_2 = pow(I_1, 2);
  // CeedScalar bar_I_1 = pow(*detC_m1, -2/3)* I_1; //using detC_m1 as J
  // CeedScalar bar_I_2 = pow(*detC_m1, -4/3)* I_2;  //using detC_m1 as J
  
  const CeedScalar I3[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; //I3 is identity matrix
  CeedScalar mu_1_J_2 = mu_1*pow(*detC_m1, (-2/3)); //mu_1*J^(-2/3)
  CeedScalar mu_2_J_4 = mu_2*pow(*detC_m1, (-4/3)); //mu_2*J^(-4/3)
  CeedScalar k1_J2_J = k_1*(pow(*detC_m1, 2) - *detC_m1); //k_1*(J^2 -J)

  //compute Swork: mu_1*J^(-2/3)*(I3-(1/3)*I_1*Cinv) + mu_2*J^(-4/3)*(I_1*I3 - C - (2/3)*I_2*Cinv) + k_1*(J^2 -J)* Cinv
  for (CeedInt m = 0; m < 6; m++){
    for(CeedInt n = 0; n < 3; n++)
    Swork[m] = mu_1_J_2*(I3[indj[m]][n] - (I_1/3)*Cinv[indj[m]][n]) + mu_2_J_4*(I_1*I3[indj[m]][n] - C[indj[m]][n] - (2/3)*I_2*Cinv[indj[m]][n]) + k1_J2_J*Cinv[indj[m]][n];
  }
  return 0;
}
// -----------------------------------------------------------------------------
// Generalized Polynomial model
CeedScalar GP_2nd_PK(void *ctx, CeedScalar Swork[6], CeedScalar Cinvwork[6], const CeedScalar *detC_m1,
                           CeedScalar *llnj, const CeedInt indj[6], const CeedInt indk[6], CeedScalar E2[][3]){
  // // C : right Cauchy-Green tensor
  // // C = I + 2E
  // // *INDENT-OFF*
  // const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
  //                             {E2[0][1], 1 + E2[1][1], E2[1][2]},
  //                             {E2[0][2], E2[1][2], 1 + E2[2][2]}
  //                            };
  // // *INDENT-ON*

  // // Compute C^(-1) : C-Inverse
  // CeedScalar A[6] = {C[1][1]*C[2][2] - C[1][2]*C[2][1], /* *NOPAD* */
  //                    C[0][0]*C[2][2] - C[0][2]*C[2][0], /* *NOPAD* */
  //                    C[0][0]*C[1][1] - C[0][1]*C[1][0], /* *NOPAD* */
  //                    C[0][2]*C[1][0] - C[0][0]*C[1][2], /* *NOPAD* */
  //                    C[0][1]*C[1][2] - C[0][2]*C[1][1], /* *NOPAD* */
  //                    C[0][2]*C[2][1] - C[0][1]*C[2][2] /* *NOPAD* */
  //                   };
  // for (CeedInt m = 0; m < 6; m++)
  //   Cinvwork[m] = A[m] / (*detC_m1 + 1.);

  // // *INDENT-OFF* //
  // const CeedScalar Cinv[3][3] = {{Cinvwork[0], Cinvwork[5], Cinvwork[4]},
  //                                {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
  //                                {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
  //                               };

  // compute invariants
  // CeedScalar I_1 = C[0][0] + C[1][1] + C[2][2];
  // CeedScalar I_2 = pow(I_1, 2);
  // CeedScalar bar_I_1 = pow(*detC_m1, -2/3)* I_1; //using detC_m1 as J
  // CeedScalar bar_I_2 = pow(*detC_m1, -4/3)* I_2;  //using detC_m1 as J
  return 0;
}
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Common computations between FS and dFS
// -----------------------------------------------------------------------------
static inline int commonFS_generic(void *ctx, const CeedScalar gradu[][3], CeedScalar Swork[6],
                           CeedScalar Cinvwork[6], CeedScalar *detC_m1,
                           CeedScalar *llnj, 
                           CeedScalar (*Swork_func)(void *ctx, CeedScalar Swork[6], CeedScalar Cinvwork[6], const CeedScalar *detC_m1,
                           CeedScalar *llnj, const CeedInt indj[6], const CeedInt indk[6], CeedScalar E2[][3])) {
  // E - Green-Lagrange strain tensor
  //     E = 1/2 (gradu + gradu^T + gradu^T*gradu)
  const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
  CeedScalar E2work[6];
  for (CeedInt m = 0; m < 6; m++) {
    E2work[m] = gradu[indj[m]][indk[m]] + gradu[indk[m]][indj[m]];
    for (CeedInt n = 0; n < 3; n++)
      E2work[m] += gradu[n][indj[m]]*gradu[n][indk[m]];
  }
  // *INDENT-OFF*
  CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                         {E2work[5], E2work[1], E2work[3]},
                         {E2work[4], E2work[3], E2work[2]}
                        };
  // *INDENT-ON*
  (*detC_m1) = computeDetCM1(E2work);
  Swork_func(ctx, Swork, Cinvwork, detC_m1, llnj, indj, indk, E2);

  // move below to Swork_func ------------------------------------
  // C : right Cauchy-Green tensor
  // C = I + 2E
  // *INDENT-OFF*
  // const CeedScalar C[3][3] = {{1 + E2[0][0], E2[0][1], E2[0][2]},
  //                             {E2[0][1], 1 + E2[1][1], E2[1][2]},
  //                             {E2[0][2], E2[1][2], 1 + E2[2][2]}
  //                            };
  // // *INDENT-ON*

  // // Compute C^(-1) : C-Inverse
  // CeedScalar A[6] = {C[1][1]*C[2][2] - C[1][2]*C[2][1], /* *NOPAD* */
  //                    C[0][0]*C[2][2] - C[0][2]*C[2][0], /* *NOPAD* */
  //                    C[0][0]*C[1][1] - C[0][1]*C[1][0], /* *NOPAD* */
  //                    C[0][2]*C[1][0] - C[0][0]*C[1][2], /* *NOPAD* */
  //                    C[0][1]*C[1][2] - C[0][2]*C[1][1], /* *NOPAD* */
  //                    C[0][2]*C[2][1] - C[0][1]*C[2][2] /* *NOPAD* */
  //                   };
  // for (CeedInt m = 0; m < 6; m++)
  //   Cinvwork[m] = A[m] / (*detC_m1 + 1.);

  // // *INDENT-OFF* //
  // const CeedScalar Cinv[3][3] = {{Cinvwork[0], Cinvwork[5], Cinvwork[4]},
  //                                {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
  //                                {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
  //                               };
  // *INDENT-ON*

  // Compute the Second Piola-Kirchhoff (S) //move this whole thing to new method. 
  // (*llnj) = lambda*log1p_series_shifted(*detC_m1)/2.;
  // for (CeedInt m = 0; m < 6; m++) {
  //   Swork[m] = (*llnj)*Cinvwork[m];
  //   for (CeedInt n = 0; n < 3; n++)
  //     Swork[m] += mu*Cinv[indj[m]][n]*E2[n][indk[m]]; // TO-DO: requires model specific derivatives; unpack ctx in method
  // }
  // ------------------------------------
  

  return 0;
};

// -----------------------------------------------------------------------------
// Residual evaluation for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
static inline int HyperFSF_Generic(void *ctx, CeedInt Q, const CeedScalar *const *in,
                         CeedScalar *const *out, 
                         CeedScalar (*Swork_func)(void *ctx, CeedScalar Swork[6], CeedScalar Cinvwork[6], const CeedScalar *detC_m1,
                         CeedScalar *llnj, const CeedInt indj[6], const CeedInt indk[6], CeedScalar E2[][3])) { // add helper to call specific model
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*dvdX)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[0];
  // Store gradu for HyperFSdF (Jacobian of HyperFSF)
  CeedScalar (*gradu)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context
  // const Physics context = (Physics)ctx;
  // const CeedScalar E  = context->E;
  // const CeedScalar nu = context->nu;
  // const CeedScalar TwoMu = E / (1 + nu);
  // const CeedScalar mu = TwoMu / 2;
  // const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  // const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

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
    const CeedScalar wdetJ      =   qdata[0][i];
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

    // Common components of finite strain calculations
    CeedScalar Swork[6], Cinvwork[6], llnj, detC_m1;
    // *INDENT-OFF*
    const CeedScalar tempgradu[3][3] =  {{gradu[0][0][i],
                                          gradu[0][1][i],
                                          gradu[0][2][i]},
                                         {gradu[1][0][i],
                                          gradu[1][1][i],
                                          gradu[1][2][i]},
                                         {gradu[2][0][i],
                                          gradu[2][1][i],
                                          gradu[2][2][i]}
                                        };
    // *INDENT-ON*
    commonFS_generic(ctx, tempgradu, Swork, Cinvwork, &detC_m1, &llnj, Swork_func);

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
static inline int HyperFSdF_Generic(void *ctx, CeedInt Q, const CeedScalar *const *in,
                          CeedScalar *const *out, 
                          CeedScalar (*Swork_func)(void *ctx, CeedScalar Swork[6], CeedScalar Cinvwork[6], const CeedScalar *detC_m1,
                          CeedScalar *llnj, const CeedInt indj[6], const CeedInt indk[6], CeedScalar E2[][3])) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*deltaug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // gradu is used for hyperelasticity (non-linear)
  const CeedScalar (*gradu)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[2];

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
    const CeedScalar wdetJ      =      qdata[0][i];
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
    // Deformation Gradient : F = I3 + gradu
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

    // Common components of finite strain calculations
    CeedScalar Swork[6], Cinvwork[6], llnj, detC_m1;
    // *INDENT-OFF*
    const CeedScalar tempgradu[3][3] =  {{gradu[0][0][i],
                                          gradu[0][1][i],
                                          gradu[0][2][i]},
                                         {gradu[1][0][i],
                                          gradu[1][1][i],
                                          gradu[1][2][i]},
                                         {gradu[2][0][i],
                                          gradu[2][1][i],
                                          gradu[2][2][i]}
                                        };
    // *INDENT-ON*
    commonFS_generic(ctx, tempgradu, Swork, Cinvwork, &detC_m1, &llnj, Swork_func);

    // deltaE - Green-Lagrange strain tensor
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
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
    // C^(-1) : C-Inverse
    // *INDENT-OFF*
    const CeedScalar Cinv[3][3] = {{Cinvwork[0], Cinvwork[5], Cinvwork[4]},
                                   {Cinvwork[5], Cinvwork[1], Cinvwork[3]},
                                   {Cinvwork[4], Cinvwork[3], Cinvwork[2]}
                                  };
    // *INDENT-ON*

    // Second Piola-Kirchhoff (S)
    // *INDENT-OFF*
    const CeedScalar S[3][3] = {{Swork[0], Swork[5], Swork[4]},
                                {Swork[5], Swork[1], Swork[3]},
                                {Swork[4], Swork[3], Swork[2]}
                               };
    // *INDENT-ON*

    // deltaS = dSdE:deltaE
    //      = lambda(Cinv:deltaE)Cinv + 2(mu-lambda*log(J))Cinv*deltaE*Cinv
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
    // -- deltaS = lambda(Cinv:deltaE)Cinv - 2(lambda*log(J)-mu)*(intermediate)
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
          deltadvdX[k][j][i] += dXdx[k][m] * deltaP[j][m] * wdetJ;
      }

  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
// Strain energy computation for hyperelasticity, finite strain                 UPDATE TO ALLOW FOR CALLING DIFFERENT MODEL TYPES
// -----------------------------------------------------------------------------
static inline int HyperFSEnergy_Generic(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out, CeedScalar (*energyFunc)(void *ctx, CeedScalar detC_m1, CeedScalar E2[][3], CeedScalar wdetJ)){//, 
                              //CeedScalar (*Swork_func)(void *ctx, CeedScalar Swork[6], CeedScalar Cinvwork[6], const CeedScalar *detC_m1,
                              //CeedScalar *llnj, const CeedInt indj[6], const CeedInt indk[6], CeedScalar E2[][3]))
                              //{ // update swork funcs
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];

  // Outputs
  CeedScalar (*energy) = (CeedScalar(*))out[0];
  // CeedScalar (*gradu)[3][CEED_Q_VLA] = (CeedScalar(*)[3][CEED_Q_VLA])out[1];
  // *INDENT-ON*

  // Context - PASS IN THE OTHER COEFFS FOR MR HERE; ADD TO STRUCT AND JUST NOT USE IF NOT DOING MR/GP?
  //remove and just unpack in energy function?
  // const Physics context = (Physics)ctx;
  // const CeedScalar E  = context->E;
  // const CeedScalar nu = context->nu;
  // const CeedScalar TwoMu = E / (1 + nu);
  // const CeedScalar mu = TwoMu / 2;
  // const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
  // const CeedScalar lambda = (3*Kbulk - TwoMu) / 3;

  // CeedScalar Swork[6], Cinvwork[6], llnj;

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
    const CeedScalar wdetJ      =   qdata[0][i];
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
    CeedScalar gradu[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        gradu[j][k] = 0;
        for (int m = 0; m < 3; m++)
          gradu[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (gradu + gradu^T + gradu^T*gradu)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = gradu[indj[m]][indk[m]] + gradu[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += gradu[n][indj[m]]*gradu[n][indk[m]];
    }
    // *INDENT-OFF*
    CeedScalar E2[3][3] = {{E2work[0], E2work[5], E2work[4]},
                           {E2work[5], E2work[1], E2work[3]},
                           {E2work[4], E2work[3], E2work[2]}
                          };
    // *INDENT-ON*
    const CeedScalar detC_m1 = computeDetCM1(E2work); 
    //commonFS_generic(ctx, gradu, &Swork, &Cinvwork, &detC_m1, &llnj, Swork_func);

    // Strain energy Phi(E) for compressible Neo-Hookean 
    // CeedScalar logj = log1p_series_shifted(detC_m1)/2.;
    // energy[i] = (lambda*logj*logj/2. - mu*logj + mu*(E2[0][0] + E2[1][1] + E2[2][2])/2.) * wdetJ; ORIGINAL 
    energy[i] = energyFunc(ctx, detC_m1, E2, wdetJ); // NEW
  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------
// method signatures for Finite Strain Energy Models
// -----------------------------------------------------------------------------
CEED_QFUNCTION(HyperFSEnergy_NH)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out) { // Neo-Hookean
    return HyperFSEnergy_Generic(ctx, Q, in, out, NH_energyModel); //, NH_2nd_PK);
}

CEED_QFUNCTION(HyperFSEnergy_MR)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out) { // Mooney-Rivlin
    return HyperFSEnergy_Generic(ctx, Q, in, out, MR_energyModel); //, MR_2nd_PK);
}

CEED_QFUNCTION(HyperFSEnergy_GP)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out) { // Generalized Polynomial
    return HyperFSEnergy_Generic(ctx, Q, in, out, GP_energyModel); //, GP_2nd_PK);
}
// update below

CEED_QFUNCTION(HyperFSF_NH)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out) { // Neo-Hookean
    return HyperFSF_Generic(ctx, Q, in, out, NH_2nd_PK);
}

CEED_QFUNCTION(HyperFSF_MR)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out) { // Mooney-Rivlin
    return HyperFSF_Generic(ctx, Q, in, out, MR_2nd_PK);
}

CEED_QFUNCTION(HyperFSF_GP)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out) { // Generalized Polynomial
    return HyperFSF_Generic(ctx, Q, in, out, GP_2nd_PK);
}

CEED_QFUNCTION(HyperFSdF_NH)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out) { // Neo-Hookean
    return HyperFSdF_Generic(ctx, Q, in, out, NH_2nd_PK);
}

CEED_QFUNCTION(HyperFSdF_MR)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out) { // Mooney-Rivlin
    return HyperFSdF_Generic(ctx, Q, in, out, MR_2nd_PK);
}

CEED_QFUNCTION(HyperFSdF_GP)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                              CeedScalar *const *out) { // Generalized Polynomial
    return HyperFSdF_Generic(ctx, Q, in, out, GP_2nd_PK);
}

// -----------------------------------------------------------------------------
// Nodal diagnostic quantities for hyperelasticity, finite strain
// -----------------------------------------------------------------------------
CEED_QFUNCTION(HyperFSDiagnostic)(void *ctx, CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*ug)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*qdata)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*diagnostic)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Context
  const Physics context = (Physics)ctx;
  const CeedScalar E  = context->E;
  const CeedScalar nu = context->nu;
  const CeedScalar TwoMu = E / (1 + nu);
  const CeedScalar mu = TwoMu / 2;
  const CeedScalar Kbulk = E / (3*(1 - 2*nu)); // Bulk Modulus
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
    CeedScalar gradu[3][3];
    for (int j = 0; j < 3; j++)     // Component
      for (int k = 0; k < 3; k++) { // Derivative
        gradu[j][k] = 0;
        for (int m = 0; m < 3; m++)
          gradu[j][k] += dXdx[m][k] * du[j][m];
      }

    // E - Green-Lagrange strain tensor
    //     E = 1/2 (gradu + gradu^T + gradu^T*gradu)
    const CeedInt indj[6] = {0, 1, 2, 1, 0, 0}, indk[6] = {0, 1, 2, 2, 2, 1};
    CeedScalar E2work[6];
    for (CeedInt m = 0; m < 6; m++) {
      E2work[m] = gradu[indj[m]][indk[m]] + gradu[indk[m]][indj[m]];
      for (CeedInt n = 0; n < 3; n++)
        E2work[m] += gradu[n][indj[m]]*gradu[n][indk[m]];
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
    const CeedScalar detC_m1 = computeDetCM1(E2work);
    CeedScalar logj = log1p_series_shifted(detC_m1)/2.;
    diagnostic[3][i] = -lambda*logj;

    // Stress tensor invariants
    diagnostic[4][i] = (E2[0][0] + E2[1][1] + E2[2][2]) / 2.;
    diagnostic[5][i] = 0.;
    for (CeedInt j = 0; j < 3; j++)
      for (CeedInt m = 0; m < 3; m++)
        diagnostic[5][i] += E2[j][m] * E2[m][j] / 4.;
    diagnostic[6][i] = sqrt(detC_m1 + 1);

    // Strain energy
    diagnostic[7][i] = (lambda*logj*logj/2. - mu*logj +
                        mu*(E2[0][0] + E2[1][1] + E2[2][2])/2.);

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End of HYPER_FS_H
