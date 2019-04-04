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

// *****************************************************************************
extern "C" __global__ void masssetupf(void *ctx, CeedInt Q,
                                      CeedQFunctionArguments args) {
  const CeedScalar *x = (const CeedScalar *)args.in[0];
  const CeedScalar *J = (const CeedScalar *)args.in[1];
  const CeedScalar *w = (const CeedScalar *)args.in[2];
  CeedScalar *rho = args.out[0], *rhs = args.out[1];

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < Q;
       i += blockDim.x * gridDim.x) {
    CeedScalar det = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
                      J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
                      J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6]));
    rho[i] = det * w[i];
    rhs[i] = rho[i] * w[i] * 
               sqrt(x[i]*x[i] + x[i+Q]*x[i+Q] + x[i+2*Q]*x[i+2*Q]);
  }
}

extern "C" __global__ void massf(void *ctx, CeedInt Q,
                                 CeedQFunctionArguments args) {
  const CeedScalar *u = (const CeedScalar *)args.in[0];
  const CeedScalar *rho = (const CeedScalar *)args.in[1];
  CeedScalar *v = args.out[0];

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < Q;
       i += blockDim.x * gridDim.x) {
    v[i] = rho[i] * u[i];
  }
}
