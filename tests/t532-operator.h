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

CEED_QFUNCTION(setup_mass)(void *ctx, const CeedInt Q,
                           const CeedScalar *const *in,
                           CeedScalar *const *out) {
  const CeedScalar *J = in[0], *weight = in[1];
  CeedScalar *rho = out[0];
  for (CeedInt i=0; i<Q; i++) {
    rho[i] = weight[i] * (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]);
  }
  return 0;
}

CEED_QFUNCTION(setup_diff)(void *ctx, const CeedInt Q,
                           const CeedScalar *const *in,
                           CeedScalar *const *out) {
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.

  // in[0] is Jacobians with shape [2, nc=2, Q]
  // in[1] is quadrature weights, size (Q)
  const CeedScalar *J = in[0], *qw = in[1];

  // out[0] is qdata, size (Q)
  CeedScalar *qd = out[0];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    // J: 0 2   qd: 0 2   adj(J):  J22 -J12
    //    1 3       2 1           -J21  J11
    const CeedScalar J11 = J[i+Q*0];
    const CeedScalar J21 = J[i+Q*1];
    const CeedScalar J12 = J[i+Q*2];
    const CeedScalar J22 = J[i+Q*3];
    const CeedScalar w = qw[i] / (J11*J22 - J21*J12);
    qd[i+Q*0] =   w * (J12*J12 + J22*J22);
    qd[i+Q*1] =   w * (J11*J11 + J21*J21);
    qd[i+Q*2] = - w * (J11*J12 + J21*J22);
  }

  return 0;
}

CEED_QFUNCTION(apply)(void *ctx, const CeedInt Q, const CeedScalar *const *in,
                      CeedScalar *const *out) {
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is mass quadrature data, size (Q)
  // in[2] is Poisson quadrature data, size (3*Q)
  // in[3] is u, size (Q)
  const CeedScalar *du = in[0], *qd_mass = in[1], *qd_diff = in[2], *u = in[3];

  // out[0] is output to multiply against v, size (Q)
  // out[1] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar *v = out[0], *dv = out[1];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    // Mass
    v[i] = qd_mass[i]*u[i];
    // Diff
    const CeedScalar du0 = du[i+Q*0];
    const CeedScalar du1 = du[i+Q*1];
    dv[i+Q*0] = qd_diff[i+Q*0]*du0 + qd_diff[i+Q*2]*du1;
    dv[i+Q*1] = qd_diff[i+Q*2]*du0 + qd_diff[i+Q*1]*du1;
  }

  return 0;
}

CEED_QFUNCTION(apply_lin)(void *ctx, const CeedInt Q,
                          const CeedScalar *const *in,
                          CeedScalar *const *out) {
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is assembled quadrature data, size (9*Q)
  // in[2] is u, size (Q)
  const CeedScalar *du = in[0], *qd = in[1], *u = in[2];

  // out[0] is output to multiply against v, size (Q)
  // out[1] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar *v = out[0], *dv = out[1];

  // Quadrature point loop
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar du0 = du[i+Q*0];
    const CeedScalar du1 = du[i+Q*1];
    v[i+Q*0] = qd[i+Q*0]*du0 + qd[i+Q*3]*du1 + qd[i+Q*6]*u[i];
    dv[i+Q*0] = qd[i+Q*1]*du0 + qd[i+Q*4]*du1 + qd[i+Q*7]*u[i];
    dv[i+Q*1] = qd[i+Q*2]*du0 + qd[i+Q*5]*du1 + qd[i+Q*8]*u[i];
  }

  return 0;
}
