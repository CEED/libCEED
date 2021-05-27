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
/// Geometric factors (3D) for Navier-Stokes example using PETSc

#ifndef setup_geo_h
#define setup_geo_h

#ifndef __CUDACC__
#  include <math.h>
#endif

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
//   in q_data[0]
//
// We require the transpose of the inverse of the Jacobian to properly compute
//   integrals of the form: int( gradv u )
//
// Inverse of Jacobian:
//   dXdx_i,j = Aij / detJ
//
// Stored: Aij / detJ
//   in q_data[1:9] as
//   (detJ^-1) * [A11 A12 A13]
//               [A21 A22 A23]
//               [A31 A32 A33]
//
// *****************************************************************************
CEED_QFUNCTION(Setup)(void *ctx, CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*w) = in[1];

  // Outputs
  CeedScalar (*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  CeedPragmaSIMD
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
    // -- Interp-to-Interp q_data
    q_data[0][i] = w[i] * detJ;
    // -- Interp-to-Grad q_data
    // Inverse of change of coordinate matrix: X_i,j
    q_data[1][i] = A11 / detJ;
    q_data[2][i] = A12 / detJ;
    q_data[3][i] = A13 / detJ;
    q_data[4][i] = A21 / detJ;
    q_data[5][i] = A22 / detJ;
    q_data[6][i] = A23 / detJ;
    q_data[7][i] = A31 / detJ;
    q_data[8][i] = A32 / detJ;
    q_data[9][i] = A33 / detJ;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction sets up the geometric factor required for integration when
//   reference coordinates are in 2D and the physical coordinates are in 3D
//
// Reference (parent) 2D coordinates: X
// Physical (current) 3D coordinates: x
// Change of coordinate matrix:
//   dxdX_{i,j} = dx_i/dX_j (indicial notation) [3 * 2]
//
// (J1,J2,J3) is given by the cross product of the columns of dxdX_{i,j}
//
// detJb is the magnitude of (J1,J2,J3)
//
// All quadrature data is stored in 4 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// Stored: w detJb
//   in q_data_sur[0]
//
// Normal vector = (J1,J2,J3) / detJb
//
// Stored: (J1,J2,J3) / detJb
//   in q_data_sur[1:3] as
//   (detJb^-1) * [ J1 ]
//                [ J2 ]
//                [ J3 ]
//
// *****************************************************************************
CEED_QFUNCTION(SetupBoundary)(void *ctx, CeedInt Q,
                              const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*w) = in[1];
  // Outputs
  CeedScalar (*q_data_sur)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    const CeedScalar dxdX[3][2] = {{J[0][0][i],
                                    J[1][0][i]},
                                   {J[0][1][i],
                                    J[1][1][i]},
                                   {J[0][2][i],
                                    J[1][2][i]}
                                   };
    // *INDENT-ON*
    // J1, J2, and J3 are given by the cross product of the columns of dxdX
    const CeedScalar J1 = dxdX[1][0]*dxdX[2][1] - dxdX[2][0]*dxdX[1][1];
    const CeedScalar J2 = dxdX[2][0]*dxdX[0][1] - dxdX[0][0]*dxdX[2][1];
    const CeedScalar J3 = dxdX[0][0]*dxdX[1][1] - dxdX[1][0]*dxdX[0][1];

    const CeedScalar detJb = sqrt(J1*J1 + J2*J2 + J3*J3);

    // q_data_sur
    // -- Interp-to-Interp q_data_sur
    q_data_sur[0][i] = w[i] * detJb;
    q_data_sur[1][i] = J1 / detJb;
    q_data_sur[2][i] = J2 / detJb;
    q_data_sur[3][i] = J3 / detJb;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************

#endif // setup_geo_h
