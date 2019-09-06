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
  @brief Ceed QFunction for applying the geometric data for the 3D diffusion
           operator
**/
CEED_QFUNCTION(diff3DApply)(void *ctx, const CeedInt Q,
                            const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is gradient u, shape [3, nc=1, Q]
  // in[1] is quadrature data, size (6*Q)
  const CeedScalar *du = in[0], *qd = in[1];

  // out[0] is output to multiply against gradient v, shape [3, nc=1, Q]
  CeedScalar *dv = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
      const CeedScalar du0 = du[i+Q*0];
      const CeedScalar du1 = du[i+Q*1];
      const CeedScalar du2 = du[i+Q*2];
      dv[i+Q*0] = qd[i+Q*0]*du0 + qd[i+Q*1]*du1 + qd[i+Q*2]*du2;
      dv[i+Q*1] = qd[i+Q*1]*du0 + qd[i+Q*3]*du1 + qd[i+Q*4]*du2;
      dv[i+Q*2] = qd[i+Q*2]*du0 + qd[i+Q*4]*du1 + qd[i+Q*5]*du2;
  }

  return 0;
}
