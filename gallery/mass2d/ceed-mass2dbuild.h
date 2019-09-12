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
  @brief Ceed QFunction for building the geometric data for the 2D mass matrix
**/
CEED_QFUNCTION(Mass2DBuild)(void *ctx, const CeedInt Q,
                            const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians with shape [2, nc=2, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *qw = in[1];
  // out[0] is quadrature data, size (Q)
  CeedScalar *qd = out[0];

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    qd[i] = (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
  } // End of Quadrature Point Loop

  return 0;
}
