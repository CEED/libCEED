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
/// libCEED QFunctions for diffusion operator example using PETSc

// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs3)(void *ctx, CeedInt Q,
                              const CeedScalar *const *in,
                              CeedScalar *const *out) {
#ifndef M_PI
#  define M_PI    3.14159265358979323846
#endif
  const CeedScalar *x = in[0], *J = in[1], *w = in[2];
  CeedScalar *true_soln = out[0], *rhs = out[1];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
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

    const CeedScalar c[3] = { 0, 1., 2. };
    const CeedScalar k[3] = { 1., 2., 3. };

    // Component 1
    true_soln[i+0*Q] = sin(M_PI*(c[0] + k[0]*x[i+Q*0])) *
                       sin(M_PI*(c[1] + k[1]*x[i+Q*1])) *
                       sin(M_PI*(c[2] + k[2]*x[i+Q*2]));
    // Component 2
    true_soln[i+1*Q] = true_soln[i+0*Q];
    // Component 3
    true_soln[i+2*Q] = true_soln[i+0*Q];

    const CeedScalar rho = w[i] * (J11*A11 + J21*A12 + J31*A13);
    // Component 1
    rhs[i+0*Q] = rho * M_PI*M_PI * (k[0]*k[0] + k[1]*k[1] + k[2]*k[2]) *
                 true_soln[i+0*Q];
    // Component 2
    rhs[i+1*Q] = rhs[i+0*Q];
    // Component 3
    rhs[i+2*Q] = rhs[i+0*Q];
  } // End of Quadrature Point Loop

  return 0;
}

// -----------------------------------------------------------------------------
CEED_QFUNCTION(Diff3)(void *ctx, CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *ug = in[0], *qd = in[1];
  CeedScalar *vg = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u components
    const CeedScalar uJ[3][3]        = {{ug[i+(0+0*3)*Q],
                                         ug[i+(0+1*3)*Q],
                                         ug[i+(0+2*3)*Q]},
                                        {ug[i+(1+0*3)*Q],
                                         ug[i+(1+1*3)*Q],
                                         ug[i+(1+2*3)*Q]},
                                        {ug[i+(2+0*3)*Q],
                                         ug[i+(2+1*3)*Q],
                                         ug[i+(2+2*3)*Q]}
                                       };
    // Read qdata (dXdxdXdxT symmetric matrix)
    const CeedScalar dXdxdXdxT[3][3] = {{qd[i+0*Q],
                                         qd[i+1*Q],
                                         qd[i+2*Q]},
                                        {qd[i+1*Q],
                                         qd[i+3*Q],
                                         qd[i+4*Q]},
                                        {qd[i+2*Q],
                                         qd[i+4*Q],
                                         qd[i+5*Q]}
                                       };

    for (int k=0; k<3; k++) // k = component
      for (int j=0; j<3; j++) // j = direction of vg
        vg[i+(k+j*3)*Q] = (uJ[k][0] * dXdxdXdxT[0][j] +
                           uJ[k][1] * dXdxdXdxT[1][j] +
                           uJ[k][2] * dXdxdXdxT[2][j]);
  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------
