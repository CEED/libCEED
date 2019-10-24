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
/// libCEED QFunctions for mass operator example for a scalar field on the sphere using PETSc

#ifndef __CUDACC__
#  include <math.h>
#endif

// *****************************************************************************
// This QFunction sets up the geometric factor required for integration when
//   reference coordinates have a different dimension than the one of
//   pysical coordinates
//
// Reference (parent) 2D coordinates: X \in [-1, 1]^2
//
// Global physical coordinates given by the mesh (3D): xx \in [-l, l]^3
//
// Local physical coordinates on the manifold (2D): x \in [-l, l]^2
//
// Change of coordinates matrix computed by the library:
//   (pysical 3D coords relative to reference 2D coords)
//   dxx_j/dX_i (indicial notation) [3 x 2]
//
// Change of coordinates x (pysical 2D) relative to xx (phyisical 3D):
//   dx_i/dxx_j (indicial notation) [2 x 3]
//
// Change of coordinates x (physical 2D) relative to X (reference 2D):
//   (by chain rule)
//   dx_i/dX_j = dx_i/dxx_k * dxx_k/dX_j
//
// The quadrature data is stored in the array qdata.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// Qdata: w * det(dx_i/dX_j)
//
// *****************************************************************************

// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupMassGeo)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  // Inputs
  const CeedScalar *J = in[0], *w = in[1];
  // Outputs
  CeedScalar *qdata = out[0];


  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read dxxdX Jacobian entries, stored as
    // 0 3
    // 1 4
    // 2 5
    const CeedScalar dxxdX[3][2] = {{J[i+Q*0],
                                     J[i+Q*3]},
                                    {J[i+Q*1],
                                     J[i+Q*4]},
                                    {J[i+Q*2],
                                     J[i+Q*5]}
                                   };

    // Modulus of dxxdX column vectors
    const CeedScalar modg1 = sqrt(dxxdX[0][0]*dxxdX[0][0] +
                                  dxxdX[1][0]*dxxdX[1][0] +
                                  dxxdX[2][0]*dxxdX[2][0]);
    const CeedScalar modg2 = sqrt(dxxdX[0][1]*dxxdX[0][1] +
                                  dxxdX[1][1]*dxxdX[1][1] +
                                  dxxdX[2][1]*dxxdX[2][1]);

    // Use normalized column vectors of dxxdX as rows of dxdxx
    const CeedScalar dxdxx[2][3] = {{dxxdX[0][0] / modg1,
                                     dxxdX[1][0] / modg1,
                                     dxxdX[2][0] / modg1},
                                    {dxxdX[0][1] / modg2,
                                     dxxdX[1][1] / modg2,
                                     dxxdX[2][1] / modg2}
                                   };

    CeedScalar dxdX[2][2];
    for (int j=0; j<2; j++)
      for (int k=0; k<2; k++) {
        dxdX[j][k] = 0;
        for (int l=0; l<3; l++)
          dxdX[j][k] += dxdxx[j][l]*dxxdX[l][k];
      }

    qdata[i+Q*0] = (dxdX[0][0]*dxdX[1][1] - dxdX[1][0]*dxdX[0][1]) * w[i];

  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------

// *****************************************************************************
// This QFunction applies the mass matrix for a scalar field.
//
// Inputs:
//   u     - Input vector at quadrature points
//   qdata - Geometric factors
//
// Output:
//   v     - Output vector (test function) at quadrature points
//
// *****************************************************************************

// -----------------------------------------------------------------------------
CEED_QFUNCTION(Mass)(void *ctx, const CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *u = in[0], *qdata = in[1];
  // Outputs
  CeedScalar *v = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++)
    v[i] = qdata[i] * u[i];

  return 0;
}
// -----------------------------------------------------------------------------
