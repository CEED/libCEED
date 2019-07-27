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

// *****************************************************************************
CEED_QFUNCTION int SetupMass3(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  CeedScalar *rho = out[0], *true_soln = out[1], *rhs = out[2];
  const CeedScalar *x = in[0];
  const CeedScalar *J = in[1];
  const CeedScalar *w = in[2];
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar det = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                            J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                            J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6]));
    const CeedScalar r = det * w[i];
    rho[i] = r;
    const CeedScalar true_sol = sqrt(x[i]*x[i] + x[i+Q]*x[i+Q] + x[i+2*Q]*x[i+2*Q]);
    true_soln[i+0*Q] = true_sol;
    true_soln[i+1*Q] = true_sol;
    true_soln[i+2*Q] = true_sol;
    rhs[i+0*Q] = r * true_sol;
    rhs[i+1*Q] = r * true_sol;
    rhs[i+2*Q] = r * true_sol;
  }
  return 0;
}

CEED_QFUNCTION int Mass3(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *rho = in[1];
  CeedScalar *v = out[0];
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar r = rho[i];
    v[i+0*Q] = r * u[i+0*Q];
    v[i+1*Q] = r * u[i+1*Q];
    v[i+2*Q] = r * u[i+2*Q];
  }
  return 0;
}
