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
/// Force of Richard problem 2D (quad element) using PETSc

#ifndef POST_PROCESSING2D_H
#define POST_PROCESSING2D_H

#include <math.h>
#include <ceed.h>
#include "ceed/ceed-f64.h"
#include "utils.h"

// -----------------------------------------------------------------------------
// We solve (v, u) = (v, uh), to project Hdiv to L2 space
// This QFunction create post_rhs = (v, uh)
// -----------------------------------------------------------------------------
CEED_QFUNCTION(PostProcessingRhs2D)(void *ctx, const CeedInt Q,
                                    const CeedScalar *const *in,
                                    CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];
  // Outputs
  CeedScalar (*post_rhs) = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};

    // 1) Compute Piola map: uh = J*u/detJ
    // 2) rhs = (v, uh) = uh*w*det_J
    // ==> rhs = J*u*w
    CeedScalar u1[2] = {u[0][i], u[1][i]}, rhs[2];
    AlphaMatVecMult2x2(w[i], J, u1, rhs);

    post_rhs[i+0*Q] = rhs[0];
    post_rhs[i+1*Q] = rhs[1];
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
// We solve (v, u) = (v, uh), to project Hdiv to L2 space
// This QFunction create mass matrix (v, u), then we solve using ksp to have 
// projected uh in L2 space and use it for post-processing
// -----------------------------------------------------------------------------
CEED_QFUNCTION(PostProcessingMass2D)(void *ctx, const CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*w) = in[0],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*u)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[2];

  // Outputs
  CeedScalar (*v)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // *INDENT-OFF*
    // Setup, J = dx/dX
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    const CeedScalar det_J = MatDet2x2(J);

    // *INDENT-ON*
    // (v, u): v = u*w*detJ
    for (CeedInt k = 0; k < 2; k++) {
      v[k][i] = u[k][i]*w[i]*det_J;
    }
  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif //End of POST_PROCESSING2D_H
