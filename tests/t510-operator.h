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
/// QFunction definitions for t510-operator.c

// *****************************************************************************
static int setup(void *ctx, CeedInt Q, CeedInt N, CeedQFunctionArguments args) {
  const CeedScalar *weight = args.in[0], *J = args.in[1];
  CeedScalar *rho = args.out[0];

  for (CeedInt i=0; i<Q; i++) {
    rho[i] = weight[i] * (J[i+N*0]*J[i+N*3] - J[i+N*1]*J[i+N*2]);
  }
  return 0;
}

// *****************************************************************************
static int mass(void *ctx, CeedInt Q, CeedInt N, CeedQFunctionArguments args) {
  const CeedScalar *rho = args.in[0], *u = args.in[1];
  CeedScalar *v = args.out[0];

  for (CeedInt i=0; i<Q; i++) {
    v[i] = rho[i] * u[i];
  }
  return 0;
}
