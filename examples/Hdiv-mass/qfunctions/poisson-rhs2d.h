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
#include "utils.h"

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif
// -----------------------------------------------------------------------------
// This QFunction sets up the rhs and true solution for the problem
// Inputs:
//   x     - interpolation of the physical coordinate
//   w     - weight of quadrature
//   J     - dx/dX. x physical coordinate, X reference coordinate [-1,1]^dim
//
// Output:
//   true_soln - True solution that we use it in poisson-error2d.h
//               to compute pointwise max error
//   rhs       - Output vector (test functions) at quadrature points
// Note we need to apply Piola map on the basis, which is J*u/detJ
// So (v,ue) = \int (v^T * ue detJ*w) ==> \int (v^T J^T* ue * w)
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupRhs2D)(void *ctx, const CeedInt Q,
                           const CeedScalar *const *in,
                           CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*coords) = in[0],
                   (*w) = in[1],
                   (*dxdX)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[2];
  // Outputs
  //CeedScalar (*rhs)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  CeedScalar (*true_soln) = out[0], (*rhs) = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Setup, (x,y) and J = dx/dX
    CeedScalar x = coords[i+0*Q], y = coords[i+1*Q];
    const CeedScalar J[2][2] = {{dxdX[0][0][i], dxdX[1][0][i]},
                                {dxdX[0][1][i], dxdX[1][1][i]}};
    // *INDENT-ON*
    // Compute J^T*ue
    CeedScalar ue[2] = {-M_PI*cos(M_PI*x) *sin(M_PI*y), -M_PI*sin(M_PI*x) *cos(M_PI*y)};
    //CeedScalar ue[2] = {x-y, x+y};
    CeedScalar rhs1[2];
    AlphaMatTransposeVecMult2x2(1, J, ue, rhs1);

    // Component 1
    true_soln[i+0*Q] = ue[0];
    rhs[i+0*Q] = rhs1[0] * w[i];
    // Component 2
    true_soln[i+1*Q] = ue[1];
    rhs[i+1*Q] = rhs1[1] * w[i];
  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

#endif //End of POISSON_RHS2D_H
