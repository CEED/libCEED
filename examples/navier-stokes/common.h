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
/// Geometric factors and mass operator for Navier-Stokes example using PETSc

#ifndef common_h
#define common_h

#ifndef CeedPragmaOMP
#  ifdef _OPENMP
#    define CeedPragmaOMP_(a) _Pragma(#a)
#    define CeedPragmaOMP(a) CeedPragmaOMP_(omp a)
#  else
#    define CeedPragmaOMP(a)
#  endif
#endif

#include <math.h>

// *****************************************************************************
// This QFunction sets up the geometric factors required for integration and
//   coordinate transformations
//
// Reference (parent) coordinates: X
// Physical (current) coordinates: x
// Change of coordinate matrix: dxdX_{i,j} = x_{i,j} (indicial notation)
// Inverse of change of coordinate matrix: dXdx_{i,j} = (detJ^-1) * X_{i,j}
//
// All quadrature data is stored in 10 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( v u )
//
// Determinant of Jacobian:
//   detJ = J11*A11 + J21*A12 + J31*A13
//     Jij = Jacobian entry ij
//     Aij = Adjoint ij
//
// Stored: w detJ
//   in qdata[0]
//
// We require the transpose of the inverse of the Jacobian to properly compute
//   integrals of the form: int( gradv u )
//
// Inverse of Jacobian:
//   dXdx_i,j = Aij / detJ
//
// Stored: Aij / detJ
//   in qdata[1:9] as
//   (detJ^-1) * [A11 A12 A13]
//               [A21 A22 A23]
//               [A31 A32 A33]
//
// *****************************************************************************
CEED_QFUNCTION(Setup)(void *ctx, CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*J)[3][Q] = (CeedScalar(*)[3][Q])in[0],
                   (*w) = in[1];
  // Outputs
  CeedScalar (*qdata)[Q] = (CeedScalar(*)[Q])out[0];

  CeedPragmaOMP(simd)
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    const CeedScalar J11 = J[0][0][i];
    const CeedScalar J21 = J[0][1][i];
    const CeedScalar J31 = J[0][2][i];
    const CeedScalar J12 = J[1][0][i];
    const CeedScalar J22 = J[1][1][i];
    const CeedScalar J32 = J[1][2][i];
    const CeedScalar J13 = J[2][0][i];
    const CeedScalar J23 = J[2][1][i];
    const CeedScalar J33 = J[2][2][i];
    const CeedScalar A11 = J22*J33 - J23*J32;
    const CeedScalar A12 = J13*J32 - J12*J33;
    const CeedScalar A13 = J12*J23 - J13*J22;
    const CeedScalar A21 = J23*J31 - J21*J33;
    const CeedScalar A22 = J11*J33 - J13*J31;
    const CeedScalar A23 = J13*J21 - J11*J23;
    const CeedScalar A31 = J21*J32 - J22*J31;
    const CeedScalar A32 = J12*J31 - J11*J32;
    const CeedScalar A33 = J11*J22 - J12*J21;
    const CeedScalar detJ = J11*A11 + J21*A12 + J31*A13;

    // Qdata
    // -- Interp-to-Interp qdata
    qdata[0][i] = w[i] * detJ;
    // -- Interp-to-Grad qdata
    // Inverse of change of coordinate matrix: X_i,j
    qdata[1][i] = A11 / detJ;
    qdata[2][i] = A12 / detJ;
    qdata[3][i] = A13 / detJ;
    qdata[4][i] = A21 / detJ;
    qdata[5][i] = A22 / detJ;
    qdata[6][i] = A23 / detJ;
    qdata[7][i] = A31 / detJ;
    qdata[8][i] = A32 / detJ;
    qdata[9][i] = A33 / detJ;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction applies the mass matrix to five interlaced fields.
//
// Inputs:
//   u - Input vector at quadrature points
//   w - Quadrature weights
//
// Output:
//   v - Output vector at quadrature points
//
// *****************************************************************************
CEED_QFUNCTION(Mass)(void *ctx, CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  (void)ctx;
  const CeedScalar (*u)[Q] = (CeedScalar(*)[Q])in[0],
                   (*w) = in[1];
  CeedScalar (*v)[Q] = (CeedScalar(*)[Q])out[0];

  CeedPragmaOMP(simd)
  for (CeedInt i=0; i<Q; i++) {
    v[0][i] = w[i] * u[0][i];
    v[1][i] = w[i] * u[1][i];
    v[2][i] = w[i] * u[2][i];
    v[3][i] = w[i] * u[3][i];
    v[4][i] = w[i] * u[4][i];
  }
  return 0;
}

// *****************************************************************************
#endif
