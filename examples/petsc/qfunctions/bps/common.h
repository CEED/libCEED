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
/// libCEED QFunctions for BP examples using PETSc

// -----------------------------------------------------------------------------
CEED_QFUNCTION(Error)(void *ctx, CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *target = in[1];
  CeedScalar *err = out[0];
  for (CeedInt i=0; i<Q; i++) {
    err[i] = u[i] - target[i];
  }
  return 0;
}

// -----------------------------------------------------------------------------
CEED_QFUNCTION(Error3)(void *ctx, CeedInt Q,
                       const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *u = in[0], *target = in[1];
  CeedScalar *err = out[0];
  for (CeedInt i=0; i<3*Q; i++) {
    err[i] = u[i] - target[i];
  }
  return 0;
}
// -----------------------------------------------------------------------------
