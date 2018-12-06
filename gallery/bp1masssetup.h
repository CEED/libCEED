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
/// libCEED QFunctions for BP1 mass operator

// *****************************************************************************
static int Fields(Ceed ceed, CeedQFunction *qf) {
  // Create the Q-function that builds the mass operator. (i.e. computes its
  // quadrature data)
  CeedQFunctionCreateInterior(ceed, 1, Setup, __FILE__ ":Setup", &qf);
  CeedQFunctionAddInput(qf, "dx", 3, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddOutput(qf, "rho", 1, CEED_EVAL_NONE);
}
// *****************************************************************************
static int Setup(void *ctx, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *J = in[0], *w = in[1];
  CeedScalar *rho = out[0];
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar det = (+J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7])
                      -J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6])
                      +J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6]));
    rho[i] = det * w[i];
  }
  return 0;
}
// *****************************************************************************
