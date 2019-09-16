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


/// A structure used to pass additional data to f_build_diff and f_apply_diff
struct BuildContext { CeedInt dim, space_dim; };

/// libCEED Q-function for building quadrature data for a diffusion operator
CEED_QFUNCTION(f_build_diff)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in, CeedScalar *const *out) {
  BuildContext *bc = (BuildContext *)ctx;
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  //
  // At every quadrature point, compute w/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.
  const CeedScalar *J = in[0], *w = in[1];
  CeedScalar *qdata = out[0];

  switch (bc->dim + 10*bc->space_dim) {
  case 11:
    // Quadrature Point Loop
    CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      qdata[i] = w[i] / J[i];
    }
    break;
  case 22:
    // Quadrature Point Loop
    CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      // J: 0 2   qdata: 0 2   adj(J):  J22 -J12
      //    1 3          2 1           -J21  J11
      const CeedScalar J11 = J[i+Q*0];
      const CeedScalar J21 = J[i+Q*1];
      const CeedScalar J12 = J[i+Q*2];
      const CeedScalar J22 = J[i+Q*3];
      const CeedScalar qw = w[i] / (J11*J22 - J21*J12);
      qdata[i+Q*0] =   qw * (J12*J12 + J22*J22);
      qdata[i+Q*1] =   qw * (J11*J11 + J21*J21);
      qdata[i+Q*2] = - qw * (J11*J12 + J21*J22);
    }
    break;
  case 33:
    // Quadrature Point Loop
    CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      // J: 0 3 6   qdata: 0 5 4
      //    1 4 7          5 1 3
      //    2 5 8          4 3 2
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
      qdata[i+Q*0] = qw * (A11*A11 + A12*A12 + A13*A13);
      qdata[i+Q*1] = qw * (A21*A21 + A22*A22 + A23*A23);
      qdata[i+Q*2] = qw * (A31*A31 + A32*A32 + A33*A33);
      qdata[i+Q*3] = qw * (A21*A31 + A22*A32 + A23*A33);
      qdata[i+Q*4] = qw * (A11*A31 + A12*A32 + A13*A33);
      qdata[i+Q*5] = qw * (A11*A21 + A12*A22 + A13*A23);
    }
    break;
  }
  return 0;
}

/// libCEED Q-function for applying a diff operator
CEED_QFUNCTION(f_apply_diff)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in, CeedScalar *const *out) {
  BuildContext *bc = (BuildContext *)ctx;
  // in[0], out[0] have shape [dim, nc=1, Q]
  const CeedScalar *ug = in[0], *qdata = in[1];
  CeedScalar *vg = out[0];

  switch (bc->dim) {
  case 1:
    // Quadrature Point Loop
    CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      vg[i] = ug[i] * qdata[i];
    }
    break;
  case 2:
    // Quadrature Point Loop
    CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      const CeedScalar ug0 = ug[i+Q*0];
      const CeedScalar ug1 = ug[i+Q*1];
      vg[i+Q*0] = qdata[i+Q*0]*ug0 + qdata[i+Q*2]*ug1;
      vg[i+Q*1] = qdata[i+Q*2]*ug0 + qdata[i+Q*1]*ug1;
    }
    break;
  case 3:
    // Quadrature Point Loop
    CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      const CeedScalar ug0 = ug[i+Q*0];
      const CeedScalar ug1 = ug[i+Q*1];
      const CeedScalar ug2 = ug[i+Q*2];
      vg[i+Q*0] = qdata[i+Q*0]*ug0 + qdata[i+Q*5]*ug1 + qdata[i+Q*4]*ug2;
      vg[i+Q*1] = qdata[i+Q*5]*ug0 + qdata[i+Q*1]*ug1 + qdata[i+Q*3]*ug2;
      vg[i+Q*2] = qdata[i+Q*4]*ug0 + qdata[i+Q*3]*ug1 + qdata[i+Q*2]*ug2;
    }
    break;
  }
  return 0;
}

