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
/// Mixed poisson 2D quad element using PETSc

#ifndef POISSON_MASS2D_H
#define POISSON_MASS2D_H

#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction applies the mass operator for a vector field of 2 components.
//
// Inputs:
//   w     - weight of quadrature
//   J     - dx/dX. x physical coordinate, X reference coordinate [-1,1]^dim
//   u     - Input basis at quadrature points
//
// Output:
//   v     - Output vector (test functions) at quadrature points
// Note we need to apply Piola map on the basis, which is J*u/detJ
// So (v,u) = \int (v^T * u detJ*w) ==> \int (v^T J^T*J*u*w/detJ)
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMass)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                          CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    //printf("u1 %f\n",u[0][i]);
    // *INDENT-OFF*
    // Setup, JJ = dx/dX
    const CeedScalar JJ[2][2] = {{J[0][0][i], J[1][0][i]},
                                 {J[0][1][i], J[1][1][i]}};
    const CeedScalar detJ = JJ[0][0]*JJ[1][1] - JJ[0][1]*JJ[1][0];

    const CeedScalar u1[2]   = {u[0][i], u[1][i]};
    // *INDENT-ON*
    // Piola map: J^T*J*u*w/detJ
    // 1) Compute J^T * J
    CeedScalar JTJ[2][2];
    for (CeedInt j = 0; j < 2; j++) {
      for (CeedInt k = 0; k < 2; k++) {
        JTJ[j][k] = 0;
        for (CeedInt m = 0; m < 2; m++)
          JTJ[j][k] += JJ[m][j] * JJ[m][k];
      }
    }
    // 2) Compute Piola map J^T*J*u * w /detJ
    for (CeedInt k = 0; k < 2; k++) {
      v[k][i] = 0;
      for (CeedInt m = 0; m < 2; m++)
        v[k][i] += JTJ[k][m] * u1[m] * w[i]/detJ;
    }
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------

#endif //End of POISSON_MASS2D_H
