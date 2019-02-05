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

/// A structure used to pass additional data to f_build_mass
struct BuildContext { CeedInt dim, space_dim; };

/// libCEED Q-function for building quadrature data for a mass operator
extern "C" __global__ void f_build_mass(void *ctx, CeedInt Q,
                        Fields_Cuda fields) {
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  struct BuildContext *bc = (struct BuildContext*)ctx;
  const CeedScalar *J = fields.inputs[0], *qw = fields.inputs[1];
  CeedScalar *qd = fields.outputs[0];
  switch (bc->dim + 10*bc->space_dim) {
  case 11:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Q;
         i += blockDim.x * gridDim.x) {
      qd[i] = J[i] * qw[i];
    }
    break;
  case 22:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Q;
         i += blockDim.x * gridDim.x) {
      // 0 2
      // 1 3
      qd[i] = (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
    }
    break;
  case 33:
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < Q;
         i += blockDim.x * gridDim.x) {
      // 0 3 6
      // 1 4 7
      // 2 5 8
      qd[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
               J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
               J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * qw[i];
    }
    break;
  }
}

/// libCEED Q-function for applying a mass operator
extern "C" __global__ void f_apply_mass(void *ctx, CeedInt Q,
                        Fields_Cuda fields) {
  const CeedScalar *u = fields.inputs[0], *w = fields.inputs[1];
  CeedScalar *v = fields.outputs[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < Q;
       i += blockDim.x * gridDim.x) {
    v[i] = w[i] * u[i];
  }
}

