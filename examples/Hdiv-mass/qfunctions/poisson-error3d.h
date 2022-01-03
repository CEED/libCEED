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
/// Compute pointwise error of the H(div) example using PETSc

#ifndef ERROR3D_H
#define ERROR3D_H

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
// Compuet error
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupError3D)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*dxdX)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1],
                   (*target) = in[2], (*w) = in[3];
  // Outputs
  CeedScalar (*error) = out[0];
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup, J = dx/dX
    const CeedScalar J[3][3] = {{dxdX[0][0][i], dxdX[1][0][i], dxdX[2][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i], dxdX[2][1][i]},
                                {dxdX[0][2][i], dxdX[1][2][i], dxdX[2][2][i]}};
    const CeedScalar detJ = ComputeDetMat(J);
    // Compute Piola map:uh = J*u/detJ
    CeedScalar uh[3];
    for (CeedInt k = 0; k < 3; k++) {
      uh[k] = 0;
      for (CeedInt m = 0; m < 3; m++)
        uh[k] += J[k][m] * u[m][i]/detJ;
    }
    // Error
    error[i+0*Q] = (uh[0] - target[i+0*Q])*(uh[0] - target[i+0*Q])*w[i]*detJ;
    error[i+1*Q] = (uh[1] - target[i+1*Q])*(uh[1] - target[i+1*Q])*w[i]*detJ;
    error[i+2*Q] = (uh[2] - target[i+2*Q])*(uh[2] - target[i+2*Q])*w[i]*detJ;

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End ERROR3D_H
