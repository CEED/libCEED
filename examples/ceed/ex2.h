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

/// A structure used to pass additional data to f_build_diff
struct BuildContext { CeedInt dim, space_dim; };

/// libCEED Q-function for building quadrature data for a diffusion operator
CEED_QFUNCTION(f_build_diff)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in, CeedScalar *const *out) {
  struct BuildContext *bc = (struct BuildContext *)ctx;
  // in[0] is Jacobians with shape [dim, nc=dim, Q]
  // in[1] is quadrature weights, size (Q)
  //
  // At every quadrature point, compute w/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.
  const CeedScalar *J = in[0], *w = in[1];
  CeedScalar *qdata = out[0];

  switch (bc->dim + 10*bc->space_dim) {
  case 11:
  CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      qdata[i] = w[i] / J[i];
    } // End of Quadrature Point Loop
    break;
  case 22:
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
    } // End of Quadrature Point Loop
    break;
  case 33:
  CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      // Compute the adjoint
      CeedScalar A[3][3];
      for (CeedInt j=0; j<3; j++)
        for (CeedInt k=0; k<3; k++)
          // Equivalent code with J as a VLA and no mod operations:
          // A[k][j] = J[j+1][k+1]*J[j+2][k+2] - J[j+1][k+2]*J[j+2][k+1]
          A[k][j] = J[i+Q*((j+1)%3+3*((k+1)%3))]*J[i+Q*((j+2)%3+3*((k+2)%3))] -
                    J[i+Q*((j+1)%3+3*((k+2)%3))]*J[i+Q*((j+2)%3+3*((k+1)%3))];

      // Compute quadrature weight / det(J)
      const CeedScalar qw = w[i] / (J[i+Q*0]*A[0][0] + J[i+Q*1]*A[1][1] +
                                    J[i+Q*2]*A[2][2]);

      // Compute geometric factors
      // Stored in Voigt convention
      // 0 5 4
      // 5 1 3
      // 4 3 2
      qdata[i+Q*0] = qw * (A[0][0]*A[0][0] + A[0][1]*A[0][1] + A[0][2]*A[0][2]);
      qdata[i+Q*1] = qw * (A[1][0]*A[1][0] + A[1][1]*A[1][1] + A[1][2]*A[1][2]);
      qdata[i+Q*2] = qw * (A[2][0]*A[2][0] + A[2][1]*A[2][1] + A[2][2]*A[2][2]);
      qdata[i+Q*3] = qw * (A[1][0]*A[2][0] + A[1][1]*A[2][1] + A[1][2]*A[2][2]);
      qdata[i+Q*4] = qw * (A[0][0]*A[2][0] + A[0][1]*A[2][1] + A[0][2]*A[2][2]);
      qdata[i+Q*5] = qw * (A[0][0]*A[1][0] + A[0][1]*A[1][1] + A[0][2]*A[1][2]);
      } // End of Quadrature Point Loop
    break;
  }
  return 0;
}

/// libCEED Q-function for applying a diff operator
CEED_QFUNCTION(f_apply_diff)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in, CeedScalar *const *out) {
  struct BuildContext *bc = (struct BuildContext *)ctx;
  // in[0], out[0] have shape [dim, nc=1, Q]
  const CeedScalar *ug = in[0], *qdata = in[1];
  CeedScalar *vg = out[0];

  switch (bc->dim) {
  case 1:
    CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      vg[i] = ug[i] * qdata[i];
    } // End of Quadrature Point Loop
    break;
  case 2:
    CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      // Read spatial derivatives of u
      const CeedScalar du[2]        =  {ug[i+Q*0],
                                        ug[i+Q*1]
                                       };

      // Read qdata (dXdxdXdxT symmetric matrix)
      // Stored in Voigt convention
      // 0 2
      // 2 1
      const CeedScalar dXdxdXdxT[2][2] = {{qdata[i+0*Q],
                                           qdata[i+2*Q]},
                                          {qdata[i+2*Q],
                                           qdata[i+1*Q]}
                                         };
      // j = direction of vg
      for (int j=0; j<2; j++)
        vg[i+j*Q] = (du[0] * dXdxdXdxT[0][j] +
                     du[1] * dXdxdXdxT[1][j]);
      } // End of Quadrature Point Loop
    break;
  case 3:
    CeedPragmaSIMD
    for (CeedInt i=0; i<Q; i++) {
      // Read spatial derivatives of u
      const CeedScalar du[3]        =  {ug[i+Q*0],
                                        ug[i+Q*1],
                                        ug[i+Q*2]
                                       };

      // Read qdata (dXdxdXdxT symmetric matrix)
      // Stored in Voigt convention
      // 0 5 4
      // 5 1 3
      // 4 3 2
      const CeedScalar dXdxdXdxT[3][3] = {{qdata[i+0*Q],
                                           qdata[i+5*Q],
                                           qdata[i+4*Q]},
                                          {qdata[i+5*Q],
                                           qdata[i+1*Q],
                                           qdata[i+3*Q]},
                                          {qdata[i+4*Q],
                                           qdata[i+3*Q],
                                           qdata[i+2*Q]}
                                         };
      // j = direction of vg
      for (int j=0; j<3; j++)
        vg[i+j*Q] = (du[0] * dXdxdXdxT[0][j] +
                     du[1] * dXdxdXdxT[1][j] +
                     du[2] * dXdxdXdxT[2][j]);
    } // End of Quadrature Point Loop
    break;
  }
  return 0;
}
