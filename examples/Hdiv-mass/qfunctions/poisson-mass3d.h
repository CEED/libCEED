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
/// Mixed poisson 3D hex element using PETSc

#ifndef POISSON_MASS3D_H
#define POISSON_MASS3D_H

#include <math.h>

// -----------------------------------------------------------------------------
// Compute determinant of 3x3 matrix
// -----------------------------------------------------------------------------
#ifndef DetMat
#define DetMat
CEED_QFUNCTION_HELPER CeedScalar ComputeDetMat(const CeedScalar A[3][3]) {
  // Compute det(A)
  const CeedScalar B11 = A[1][1]*A[2][2] - A[1][2]*A[2][1];
  const CeedScalar B12 = A[0][2]*A[2][1] - A[0][1]*A[2][2];
  const CeedScalar B13 = A[0][1]*A[1][2] - A[0][2]*A[1][1];
  CeedScalar detA = A[0][0]*B11 + A[1][0]*B12 + A[2][0]*B13;

  return detA;
};
#endif

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
CEED_QFUNCTION(SetupMass3D)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                            CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar detJ = ComputeDetMat(J);
    const CeedScalar u1[3]   = {u[0][i], u[1][i], u[2][i]};
    // *INDENT-ON*
    // Piola map: J^T*J*u*w/detJ
    // 1) Compute J^T * J
    CeedScalar JTJ[3][3];
    for (CeedInt j = 0; j < 3; j++) {
      for (CeedInt k = 0; k < 3; k++) {
        JTJ[j][k] = 0;
        for (CeedInt m = 0; m < 3; m++)
          JTJ[j][k] += J[m][j] * J[m][k];
      }
    }
    // 2) Compute J^T*J*u * w /detJ
    for (CeedInt k = 0; k < 3; k++) {
      v[k][i] = 0;
      for (CeedInt m = 0; m < 3; m++)
        v[k][i] += JTJ[k][m] * u1[m] * w[i]/detJ;
    }
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------

#endif //End of POISSON_MASS3D_H
