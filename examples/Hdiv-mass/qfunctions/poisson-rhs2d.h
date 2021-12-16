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
/// Mixed poisson 2D quad element using PETSc

#ifndef POISSON_RHS2D_H
#define POISSON_RHS2D_H

#include <math.h>

// -----------------------------------------------------------------------------
// This QFunction sets up the rhs for the problem
// Inputs:
//   x     - interpolation of the physical coordinate
//   w     - weight of quadrature
//   J     - dx/dX. x physical coordinate, X reference coordinate [-1,1]^dim
//
// Output:
//   rhs       - Output vector (test functions) at quadrature points
// Note we need to apply Piola map on the basis, which is J*u/detJ
// So (v,ue) = \int (v^T * ue detJ*w) ==> \int (v^T J^T* ue * w)
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupRhs)(void *ctx, const CeedInt Q,
                         const CeedScalar *const *in,
                         CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*x) = in[0],
                   (*w) = in[1],
                   (*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[2];
  // Outputs
  //CeedScalar (*rhs)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar (*rhs) = out[0];

  // Quadrature Point Loop
  //printf("--------------------\n");
  //printf("inside qfunction poisson-rhs2d.h CEED_Q_VLA: %d, Q: %d \n",CEED_Q_VLA, Q);
  //printf("--------------------\n");
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup, JJ = dx/dX
    const CeedScalar JJ[2][2] = {{J[0][0][i], J[1][0][i]},
                                 {J[0][1][i], J[1][1][i]}};
    // *INDENT-ON*
    // Compute J^T*ue
    CeedScalar ue[2] = {x[i] - x[i+1*Q], x[i] + x[i+1*Q]};
    CeedScalar rhs1[2];
    for (CeedInt k = 0; k < 2; k++) {
      rhs1[k] = 0;
      for (CeedInt m = 0; m < 2; m++)
        rhs1[k] += JJ[m][k] * ue[m];
    }
    // Component 1
    rhs[i+0*Q] = rhs1[0] * w[i];
    // Component 2
    rhs[i+1*Q] = rhs1[1] * w[i];
  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif //End of POISSON_RHS2D_H
