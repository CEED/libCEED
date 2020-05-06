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

#ifndef setup_boundary_h
#define setup_boundary_h

#ifndef __CUDACC__
#  include <math.h>
#endif

// *****************************************************************************
// TODO: Comment on this QFunction
//
// *****************************************************************************
CEED_QFUNCTION(SetupBoundary)(void *ctx, CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                   (*w) = in[1];
  // Outputs
  CeedScalar (*qdataSur)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

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

    // J1, J2, and J3 are given by the cross product of the columns of dxdX
    const CeedScalar J1 = dxdX[1][0]*dxdX[2][1] - dxdX[2][0]*dxdX[1][1];
    const CeedScalar J2 = dxdX[2][0]*dxdX[0][1] - dxdX[0][0]*dxdX[2][1];
    const CeedScalar J3 = dxdX[0][0]*dxdX[1][1] - dxdX[1][0]*dxdX[0][1];

    const CeedScalar detJb = sqrt(J1*J1 + J2*J2 + J3*J3);

    // qdataSur
    // -- Interp-to-Interp qdataSur
    qdataSur[0][i] = w[i] * detJb;
    qdataSur[1][i] = J1 / detJb;
    qdataSur[2][i] = J2 / detJb;
    qdataSur[3][i] = J3 / detJb;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// TODO: Comment on this QFunction
//
// *****************************************************************************
CEED_QFUNCTION(SetupBoundary2d)(void *ctx, CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*J)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*w) = in[1];
  // Outputs
  CeedScalar (*qdataSur)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    const CeedScalar J1 = J[0][i];
    const CeedScalar J2 = J[1][i];

    const CeedScalar detJb = sqrt(J1*J1 + J2*J2);

    qdataSur[0][i] = w[i] * detJb;
    qdataSur[1][i] = J1 / detJb;
    qdataSur[2][i] = J2 / detJb;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************

#endif // setup_boundary_h