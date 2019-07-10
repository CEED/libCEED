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
extern "C" __global__ void SetupDiff3(void *ctx, CeedInt Q,
                                      Fields_Cuda fields) {
  #ifndef M_PI
  #define M_PI    3.14159265358979323846
  #endif
  const CeedScalar *x = (const CeedScalar *)fields.inputs[0];
  const CeedScalar *J = (const CeedScalar *)fields.inputs[1];
  const CeedScalar *w = (const CeedScalar *)fields.inputs[2];
  CeedScalar *qd = fields.outputs[0], *true_soln = fields.outputs[1], *rhs = fields.outputs[2];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < Q;
       i += blockDim.x * gridDim.x) {
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J31 = J[i+Q*2];
    const CeedScalar J12 = J[i+Q*3];
    const CeedScalar J22 = J[i+Q*4];
    const CeedScalar J32 = J[i+Q*5];
    const CeedScalar J13 = J[i+Q*6];
    const CeedScalar J23 = J[i+Q*7];
    const CeedScalar J33 = J[i+Q*8];
    const CeedScalar A11 = J22*J33 - J23*J32;
    const CeedScalar A12 = J13*J32 - J12*J33;
    const CeedScalar A13 = J12*J23 - J13*J22;
    const CeedScalar A21 = J23*J31 - J21*J33;
    const CeedScalar A22 = J11*J33 - J13*J31;
    const CeedScalar A23 = J13*J21 - J11*J23;
    const CeedScalar A31 = J21*J32 - J22*J31;
    const CeedScalar A32 = J12*J31 - J11*J32;
    const CeedScalar A33 = J11*J22 - J12*J21;
    const CeedScalar qw = w[i] / (J11*A11 + J21*A12 + J31*A13);
    qd[i+Q*0] = qw * (A11*A11 + A12*A12 + A13*A13);
    qd[i+Q*1] = qw * (A11*A21 + A12*A22 + A13*A23);
    qd[i+Q*2] = qw * (A11*A31 + A12*A32 + A13*A33);
    qd[i+Q*3] = qw * (A21*A21 + A22*A22 + A23*A23);
    qd[i+Q*4] = qw * (A21*A31 + A22*A32 + A23*A33);
    qd[i+Q*5] = qw * (A31*A31 + A32*A32 + A33*A33);

    const CeedScalar c[3] = { 0, 1., 2. };
    const CeedScalar k[3] = { 1., 2., 3. };

    true_soln[i+0*Q] = sin(M_PI*(c[0] + k[0]*x[i+Q*0])) *
                       sin(M_PI*(c[1] + k[1]*x[i+Q*1])) *
                       sin(M_PI*(c[2] + k[2]*x[i+Q*2]));
    true_soln[i+1*Q] = true_soln[i+0*Q];
    true_soln[i+2*Q] = true_soln[i+0*Q];

    const CeedScalar rho = w[i] * (J11*A11 + J21*A12 + J31*A13);
    rhs[i+0*Q] = rho * M_PI*M_PI * (k[0]*k[0] + k[1]*k[1] + k[2]*k[2]) *
                 true_soln[i+0*Q];
    rhs[i+1*Q] = rhs[i+0*Q];
    rhs[i+2*Q] = rhs[i+0*Q];
  }
}

extern "C" __global__ void Diff3(void *ctx, CeedInt Q,
                                 Fields_Cuda fields) {
  const CeedScalar *ug = (const CeedScalar *)fields.inputs[0];
  const CeedScalar *qd = (const CeedScalar *)fields.inputs[1];
  CeedScalar *vg = fields.outputs[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < Q;
       i += blockDim.x * gridDim.x) {
    const CeedScalar ug00 = ug[i+Q*(0+0*3)];
    const CeedScalar ug01 = ug[i+Q*(0+1*3)];
    const CeedScalar ug02 = ug[i+Q*(0+2*3)];
    vg[i+Q*(0+0*3)] = qd[i+Q*0]*ug00 + qd[i+Q*1]*ug01 + qd[i+Q*2]*ug02;
    vg[i+Q*(0+1*3)] = qd[i+Q*1]*ug00 + qd[i+Q*3]*ug01 + qd[i+Q*4]*ug02;
    vg[i+Q*(0+2*3)] = qd[i+Q*2]*ug00 + qd[i+Q*4]*ug01 + qd[i+Q*5]*ug02;

    const CeedScalar ug10 = ug[i+Q*(1+0*3)];
    const CeedScalar ug11 = ug[i+Q*(1+1*3)];
    const CeedScalar ug12 = ug[i+Q*(1+2*3)];
    vg[i+Q*(1+0*3)] = qd[i+Q*0]*ug10 + qd[i+Q*1]*ug11 + qd[i+Q*2]*ug12;
    vg[i+Q*(1+1*3)] = qd[i+Q*1]*ug10 + qd[i+Q*3]*ug11 + qd[i+Q*4]*ug12;
    vg[i+Q*(1+2*3)] = qd[i+Q*2]*ug10 + qd[i+Q*4]*ug11 + qd[i+Q*5]*ug12;

    const CeedScalar ug20 = ug[i+Q*(2+0*3)];
    const CeedScalar ug21 = ug[i+Q*(2+1*3)];
    const CeedScalar ug22 = ug[i+Q*(2+2*3)];
    vg[i+Q*(2+0*3)] = qd[i+Q*0]*ug20 + qd[i+Q*1]*ug21 + qd[i+Q*2]*ug22;
    vg[i+Q*(2+1*3)] = qd[i+Q*1]*ug20 + qd[i+Q*3]*ug21 + qd[i+Q*4]*ug22;
    vg[i+Q*(2+2*3)] = qd[i+Q*2]*ug20 + qd[i+Q*4]*ug21 + qd[i+Q*5]*ug22;
  }
}
