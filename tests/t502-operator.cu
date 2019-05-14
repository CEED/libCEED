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
extern "C" __global__ void setup(void *ctx, CeedInt Q, CeedInt N,
                                 CeedQFunctionArguments args) {
  const CeedScalar *weight = (const CeedScalar *)args.in[0];
  const CeedScalar *dxdX = (const CeedScalar *)args.in[1];
  CeedScalar *rho = args.out[0];

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < Q;
    i += blockDim.x * gridDim.x)
  {
    rho[i] = weight[i] * dxdX[i];
  }
}

// *****************************************************************************
extern "C" __global__ void mass(void *ctx, CeedInt Q, CeedInt N,
                                CeedQFunctionArguments args) {
  const CeedScalar *rho = (const CeedScalar *)args.in[0];
  const CeedScalar *u = (const CeedScalar *)args.in[1];
  CeedScalar *v = args.out[0];

  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < Q;
    i += blockDim.x * gridDim.x)
  {
    v[i] = rho[i] * u[i];
  }
}
