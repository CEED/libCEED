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
/// Geometric factors (2D) for Navier-Stokes example using PETSc

#ifndef setup_geo_2d_h
#define setup_geo_2d_h

#ifndef __CUDACC__
#  include <math.h>
#endif

// *****************************************************************************
// This function provides the 2D variant of the setup in setupgeo.h
// *****************************************************************************
CEED_QFUNCTION(Setup2d)(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0],
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
    const CeedScalar J12 = J[1][0][i];
    const CeedScalar J22 = J[1][1][i];
    const CeedScalar detJ = J11*J22 - J21*J12;

    // Qdata
    // -- Interp-to-Interp q_data
    q_data[0][i] = w[i] * detJ;
    // -- Interp-to-Grad q_data
    // Inverse of change of coordinate matrix: X_i,j
    q_data[1][i] =  J22 / detJ;
    q_data[2][i] = -J21 / detJ;
    q_data[3][i] = -J12 / detJ;
    q_data[4][i] =  J11 / detJ;
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************

#endif // setup_geo_2d_h
