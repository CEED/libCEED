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

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q,
                      const CeedScalar *const *in,
                      CeedScalar *const *out) {
  const CeedScalar *qw = in[0], *J = in[1];
  CeedScalar *qd = out[0];

  for (CeedInt i=0; i<Q; i++) {
    // Qdata stored in Voigt convention
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
  } // End of Quadrature Point Loop
  return 0;
}

CEED_QFUNCTION(diff)(void *ctx, const CeedInt Q, const CeedScalar *const *in,
                     CeedScalar *const *out) {
  const CeedScalar *qd = in[0], *ug = in[1];
  CeedScalar *vg = out[0];

  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar du[2]        =  {ug[i+Q*0],
                                      ug[i+Q*1]
                                     };

    // Read qdata (dXdxdXdxT symmetric matrix)
    // Stored in Voigt convention
    // 0 2
    // 2 1
    const CeedScalar dXdxdXdxT[2][2] = {{qd[i+0*Q],
                                         qd[i+2*Q]},
                                        {qd[i+2*Q],
                                         qd[i+1*Q]}
                                       };
    // j = direction of vg
    for (int j=0; j<2; j++)
      vg[i+j*Q] = (du[0] * dXdxdXdxT[0][j] +
                   du[1] * dXdxdXdxT[1][j]);
  }
  return 0;
}
