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
  @brief Ceed QFunction for building the geometric data for the 3D Poisson
           operator
**/

#ifndef poisson3dbuild_h
#define poisson3dbuild_h

CEED_QFUNCTION(Poisson3DBuild)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // At every quadrature point, compute w/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.
  // *INDENT-OFF*
  // in[0] is Jacobians with shape [3, nc=3, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar (*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0],
                                    *w = in[1];
  // out[0] is qdata, size (6*Q)
  CeedScalar     (*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const CeedInt dim = 3;

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Compute the adjoint
    CeedScalar A[3][3];
    for (CeedInt j=0; j<dim; j++)
      for (CeedInt k=0; k<dim; k++)
        // Equivalent code with no mod operations:
        // A[k][j] = J[k+1][j+1]*J[k+2][j+2] - J[k+2][j+1]*J[k+1][j+2]
        A[k][j] = J[(k+1)%dim][(j+1)%dim][i]*J[(k+2)%dim][(j+2)%dim][i] -
                  J[(k+2)%dim][(j+1)%dim][i]*J[(k+1)%dim][(j+2)%dim][i];

    // Compute quadrature weight / det(J)
    const CeedScalar qw = w[i] / (J[0][0][i]*A[0][0] + J[0][1][i]*A[1][1] +
                                  J[0][2][i]*A[2][2]);

    // Compute geometric factors
    // Stored in Voigt convention
    // 0 5 4
    // 5 1 3
    // 4 3 2
    q_data[0][i] = qw * (A[0][0]*A[0][0] + A[0][1]*A[0][1] + A[0][2]*A[0][2]);
    q_data[1][i] = qw * (A[1][0]*A[1][0] + A[1][1]*A[1][1] + A[1][2]*A[1][2]);
    q_data[2][i] = qw * (A[2][0]*A[2][0] + A[2][1]*A[2][1] + A[2][2]*A[2][2]);
    q_data[3][i] = qw * (A[1][0]*A[2][0] + A[1][1]*A[2][1] + A[1][2]*A[2][2]);
    q_data[4][i] = qw * (A[0][0]*A[2][0] + A[0][1]*A[2][1] + A[0][2]*A[2][2]);
    q_data[5][i] = qw * (A[0][0]*A[1][0] + A[0][1]*A[1][1] + A[0][2]*A[1][2]);
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // poisson3dbuild_h
