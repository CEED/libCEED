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

// *****************************************************************************
static int Setup(void *ctx, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {
  CeedScalar *rho = out[0], *true_soln = out[1], *rhs = out[2];
  const CeedScalar (*x)[Q] = (const CeedScalar (*)[Q])in[0];
  const CeedScalar (*J)[3][Q] = (const CeedScalar (*)[3][Q])in[1];
  const CeedScalar *w = in[2];
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar det = (+ J[0][0][i] * (J[1][1][i]*J[2][2][i] - J[1][2][i]*J[2][1][i])
                      - J[0][1][i] * (J[1][0][i]*J[2][2][i] - J[1][2][i]*J[2][0][i])
                      + J[0][2][i] * (J[1][0][i]*J[2][1][i] - J[1][1][i]*J[2][0][i]));
    rho[i] = det * w[i];
    true_soln[i] = PetscSqrtScalar(PetscSqr(x[0][i]) + PetscSqr(x[1][i]) + PetscSqr(
                                     x[2][i]));
    rhs[i] = rho[i] * true_soln[i];
  }
  return 0;
}

static int Mass(void *ctx, CeedInt Q,
                const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *rho = in[1];
  CeedScalar *v = out[0];
  for (CeedInt i=0; i<Q; i++) {
    v[i] = rho[i] * u[i];
  }
  return 0;
}

static int Error(void *ctx, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *target = in[1];
  CeedScalar *err = out[0];
  for (CeedInt i=0; i<Q; i++) {
    err[i] = u[i] - target[i];
  }
  return 0;
}
