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
  const CeedScalar *X = in[0], *J = in[1], *w = in[2];
  // Outputs
  CeedScalar *qdata = out[0];

  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar R        = context[0];
  const CeedScalar l        = context[1];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read global Cartesian coordinates
    const CeedScalar xx = X[i+0*Q];
    const CeedScalar yy = X[i+1*Q];
    const CeedScalar zz = X[i+2*Q];

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

    // Convert to Latitude-Longitude (lambda, theta) geographic system
    //    const CeedScalar lambda =  asin(zz / R);
    //    const CeedScalar theta  = atan2(yy, xx);
    // Convert to Longitude-Latitude: xcirc = (theta, lambda) system (from paper)
    const CeedScalar theta =  asin(zz / R);
    const CeedScalar lambda  = atan2(yy, xx);

    // Converto to local cubed-sphere system
    // These are from lat-long online:
    //    const CeedScalar x = l * cos(lambda) * cos(theta);
    //    const CeedScalar y = l * cos(lambda) * sin(theta);
    // These are from paper
    const CeedScalar x = l * tan(lambda);
    const CeedScalar y = l * tan(theta) / cos(lambda);

    const CeedScalar delta = sqrt(l*l + x*x + y*y);

    // Setup
    const CeedScalar dxcircdxx[2][3] = {{0,
                                         0,
                                         1. / R*sqrt(1-zz*zz)},
                                        {yy / (1.+1./(xx*xx)),
                                         1. / (xx*(1.+yy*yy)),
                                         0}
                                       };

    const CeedScalar dxdxcirc[2][2] = {{0,
                                        l / (cos(lambda)*cos(lambda))},
                                       {l / (cos(lambda)*cos(theta)*cos(theta)),
                                        l*sin(theta)*sin(lambda) / (cos(theta)*cos(lambda)*cos(lambda))}
                                      };

    CeedScalar dxdxx[2][3];
    for (int j=0; j<2; j++)
      for (int k=0; k<3; k++) {
        dxdxx[j][k] = 0;
        for (int l=0; l<2; l++)
          dxdxx[j][k] += dxdxcirc[j][l]*dxcircdxx[l][k];
      }


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
