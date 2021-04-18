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
/// Geometric factors for boundary integral in Navier-Stokes example using PETSc

#ifndef setupboundary_h
#define setupboundary_h

#ifndef __CUDACC__
#  include <math.h>
#endif

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
// This QFunction sets up the geometric factor required for integration when
//   reference coordinates are in 1D and the physical coordinates are in 2D
//
// Reference (parent) 1D coordinates: X
// Physical (current) 2D coordinates: x
// Change of coordinate vector:
//           J1 = dx_1/dX
//           J2 = dx_2/dX
//
// detJb is the magnitude of (J1,J2)
//
// All quadrature data is stored in 3 field vector of quadrature data.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// Stored: w detJb
//   in q_data_sur[0]
//
// Normal vector is given by the cross product of (J1,J2)/detJ and ẑ
//
// Stored: (J1,J2,0) x (0,0,1) / detJb
//   in q_data_sur[1:2] as
//   (detJb^-1) * [ J2 ]
//                [-J1 ]
//
// *****************************************************************************
CEED_QFUNCTION(SetupBoundary2d)(void *ctx, CeedInt Q,
                                const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*J)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*w) = in[1];
  // Outputs
  CeedScalar (*q_data_sur)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    const CeedScalar J1 = J[0][i];
    const CeedScalar J2 = J[1][i];

    const CeedScalar detJb = sqrt(J1*J1 + J2*J2);

    q_data_sur[0][i] = w[i] * detJb;
    q_data_sur[1][i] = J2 / detJb;
    q_data_sur[2][i] = -J1 / detJb;
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************

#endif // setupboundary_h
