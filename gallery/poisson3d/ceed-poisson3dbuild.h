// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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

/**
  @brief Ceed QFunction for building the geometric data for the 3D poisson
           operator
**/
CEED_QFUNCTION(Poisson3DBuild)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.

  // in[0] is Jacobians with shape [3, nc=3, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *qw = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar *qd = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Compute the adjoint
    CeedScalar A[3][3];
    for (CeedInt j=0; j<3; j++)
      for (CeedInt k=0; k<3; k++)
        // Equivalent code with J as a VLA and no mod operations:
        // A[k][j] = J[j+1][k+1]*J[j+2][k+2] - J[j+1][k+2]*J[j+2][k+1]
        A[k][j] = J[i+Q*((j+1)%3+3*((k+1)%3))]*J[i+Q*((j+2)%3+3*((k+2)%3))] -
                  J[i+Q*((j+1)%3+3*((k+2)%3))]*J[i+Q*((j+2)%3+3*((k+1)%3))];

    // Compute quadrature weight / det(J)
    const CeedScalar w = qw[i] / (J[i+Q*0]*A[0][0] + J[i+Q*1]*A[1][1] +
                                  J[i+Q*2]*A[2][2]);

    // Compute geometric factors
    // Stored in Voigt convention
    // 0 5 4
    // 5 1 3
    // 4 3 2
    qd[i+Q*0] = w * (A[0][0]*A[0][0] + A[0][1]*A[0][1] + A[0][2]*A[0][2]);
    qd[i+Q*1] = w * (A[1][0]*A[1][0] + A[1][1]*A[1][1] + A[1][2]*A[1][2]);
    qd[i+Q*2] = w * (A[2][0]*A[2][0] + A[2][1]*A[2][1] + A[2][2]*A[2][2]);
    qd[i+Q*3] = w * (A[1][0]*A[2][0] + A[1][1]*A[2][1] + A[1][2]*A[2][2]);
    qd[i+Q*4] = w * (A[0][0]*A[2][0] + A[0][1]*A[2][1] + A[0][2]*A[2][2]);
    qd[i+Q*5] = w * (A[0][0]*A[1][0] + A[0][1]*A[1][1] + A[0][2]*A[1][2]);
  } // End of Quadrature Point Loop

  return 0;
}
