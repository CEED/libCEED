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
  @brief Ceed QFunction for applying the geometric data for the 3D poisson
           operator
**/
CEED_QFUNCTION(Poisson3DApply)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // in[0] is gradient u, shape [3, nc=1, Q]
  // in[1] is quadrature data, size (6*Q)
  const CeedScalar *ug = in[0], *qd = in[1];

  // out[0] is output to multiply against gradient v, shape [3, nc=1, Q]
  CeedScalar *vg = out[0];

  // Quadrature point loop
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
    // *INDENT-OFF*
    const CeedScalar dXdxdXdxT[3][3] = {{qd[i+0*Q],
                                         qd[i+5*Q],
                                         qd[i+4*Q]},
                                        {qd[i+5*Q],
                                         qd[i+1*Q],
                                         qd[i+3*Q]},
                                        {qd[i+4*Q],
                                         qd[i+3*Q],
                                         qd[i+2*Q]}
                                       };
    // *INDENT-ON*

    // Apply Poisson Operator
    // j = direction of vg
    for (int j=0; j<3; j++)
      vg[i+j*Q] = (du[0] * dXdxdXdxT[0][j] +
                   du[1] * dXdxdXdxT[1][j] +
                   du[2] * dXdxdXdxT[2][j]);
  } // End of Quadrature Point Loop

  return 0;
}
