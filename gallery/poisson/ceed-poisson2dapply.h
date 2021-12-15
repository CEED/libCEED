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
  @brief Ceed QFunction for applying the 2D Poisson operator
**/

#ifndef poisson2dapply_h
#define poisson2dapply_h

CEED_QFUNCTION(Poisson2DApply)(void *ctx, const CeedInt Q,
                               const CeedScalar *const *in,
                               CeedScalar *const *out) {
  // *INDENT-OFF*
  // in[0] is gradient u, shape [2, nc=1, Q]
  // in[1] is quadrature data, size (3*Q)
  const CeedScalar (*ug)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
               (*q_data)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[1];
  // out[0] is output to multiply against gradient v, shape [2, nc=1, Q]
  CeedScalar       (*vg)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  const CeedInt dim = 2;

  // Quadrature point loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read qdata (dXdxdXdxT symmetric matrix)
    // Stored in Voigt convention
    // 0 2
    // 2 1
    // *INDENT-OFF*
    const CeedScalar dXdxdXdxT[2][2] = {{q_data[0][i],
                                         q_data[2][i]},
                                        {q_data[2][i],
                                         q_data[1][i]}
                                       };
    // *INDENT-ON*

    // Apply Poisson operator
    // j = direction of vg
    for (CeedInt j=0; j<dim; j++)
      vg[j][i] = (ug[0][i] * dXdxdXdxT[0][j] +
                  ug[1][i] * dXdxdXdxT[1][j]);
  } // End of Quadrature Point Loop

  return CEED_ERROR_SUCCESS;
}

#endif // poisson2dapply_h
