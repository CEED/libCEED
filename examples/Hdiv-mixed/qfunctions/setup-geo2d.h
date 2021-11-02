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
/// Geometric factors for 2D H(div) examples example using PETSc

#ifndef SETUP_GEO_2D_H
#define SETUP_GEO_2D_H

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
//   detJ = J11*J22 - J21*J12
//     Jij = Jacobian entry ij
//
// Stored: w detJ
//   in q_data[0]
//
// We require the transpose of the inverse of the Jacobian to properly compute
//   integrals of the form: int( gradv u )
//
// Inverse of Jacobian:
//   dXdx_i,j = Aij / detJ
//   Aij = Adjoint ij
//
// Stored: Aij / detJ
//   in q_data[1:4] as
//   (detJ^-1) * [A11 A12]
//               [A21 A22]
//
// *****************************************************************************
CEED_QFUNCTION(SetupGeo2D)(void *ctx, CeedInt Q, const CeedScalar *const *in,
                           CeedScalar *const *out) {
    // *INDENT-OFF*
     // Inputs
     const CeedScalar (*w) = in[0],
                      (*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1];

     // Outputs
     CeedScalar (*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
     // *INDENT-ON*
  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup, J = dx/dX
    const CeedScalar J11 = J[0][0][i];
    const CeedScalar J21 = J[0][1][i];
    const CeedScalar J12 = J[1][0][i];
    const CeedScalar J22 = J[1][1][i];
    const CeedScalar detJ = J11*J22 - J12*J21;

    // Qdata dX/dx
    q_data[0][i] = w[i] * detJ;

    // Inverse of change of coordinate matrix: X_i,j
    q_data[1][i] = J22 / detJ;
    q_data[2][i] = -J21 / detJ;
    q_data[3][i] = -J12 / detJ;
    q_data[4][i] = J11 / detJ;

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End of SETUP_GEO_2D_H
