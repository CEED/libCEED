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
/// libCEED QFunctions for BP3 diffusion operator

// *****************************************************************************
static int Fields(Ceed ceed, CeedQFunction *qf) {
  // Create the Q-function that defines the action of the diff operator.
  CeedQFunctionCreateInterior(ceed, 1, Diff, __FILE__ ":Diff", &qf);
  CeedQFunctionAddInput(qf, "u", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddInput(qf, "rho", 6, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf, "v", 1, CEED_EVAL_GRAD);
  return 0;
}
// *****************************************************************************
static int Diff(void *ctx, CeedInt Q,
                const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *ug = in[0], *qd = in[1];
  CeedScalar *vg = out[0];
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar ug0 = ug[i+Q*0];
    const CeedScalar ug1 = ug[i+Q*1];
    const CeedScalar ug2 = ug[i+Q*2];
    vg[i+Q*0] = qd[i+Q*0]*ug0 + qd[i+Q*1]*ug1 + qd[i+Q*2]*ug2;
    vg[i+Q*1] = qd[i+Q*1]*ug0 + qd[i+Q*3]*ug1 + qd[i+Q*4]*ug2;
    vg[i+Q*2] = qd[i+Q*2]*ug0 + qd[i+Q*4]*ug1 + qd[i+Q*5]*ug2;
  }
  return 0;
}
// *****************************************************************************
