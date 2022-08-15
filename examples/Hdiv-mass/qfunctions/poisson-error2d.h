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

#ifndef ERROR2D_H
#define ERROR2D_H

#include <math.h>
#include "utils.h"
// -----------------------------------------------------------------------------
// Compuet error
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupError2D)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1],
                   (*target) = in[2], (*w) = in[3];
  // Outputs
  CeedScalar (*error) = out[0];
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup, J = dx/dX
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);            
    // Compute Piola map:uh = J*u/detJ
    CeedScalar u1[2]   = {u[0][i], u[1][i]}, uh[2];
    AlphaMatVecMult2x2(1/det_J, J, u1, uh);
    // Error
    error[i+0*Q] = (uh[0] - target[i+0*Q])*(uh[0] - target[i+0*Q])*w[i]*det_J;
    error[i+1*Q] = (uh[1] - target[i+1*Q])*(uh[1] - target[i+1*Q])*w[i]*det_J;

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // End ERROR2D_H
