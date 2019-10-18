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
/// libCEED QFunctions for mass operator example using PETSc

#ifndef __CUDACC__
#  include <math.h>
#endif

// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassGeo)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  const CeedScalar *J = in[0], *w = in[1];
  CeedScalar *qdata = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar det = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                            J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                            J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6]));
    qdata[i] = det * w[i];
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassRhs)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  const CeedScalar *x = in[0], *J = in[1], *w = in[2];
  CeedScalar *true_soln = out[0], *rhs = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar det = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                            J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                            J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6]));

    true_soln[i] = sqrt(x[i]*x[i] + x[i+Q]*x[i+Q] + x[i+2*Q]*x[i+2*Q]);

    rhs[i] = det * w[i] * true_soln[i];
  } // End of Quadrature Point Loop
  return 0;
}

// -----------------------------------------------------------------------------
CEED_QFUNCTION(Mass)(void *ctx, const CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *qdata = in[1];
  CeedScalar *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++)
    v[i] = qdata[i] * u[i];

  return 0;
}
// -----------------------------------------------------------------------------
