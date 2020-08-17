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
/// Geometric factors and mass operator for shallow-water example using PETSc

#ifndef setupgeo_h
#define setupgeo_h

#ifndef __CUDACC__
#  include <math.h>
#endif

// *****************************************************************************
// This QFunction sets up the geometric factors required for integration and
//   coordinate transformations. See ref: "A Discontinuous Galerkin Transport
//   Scheme on the Cubed Sphere", by Nair et al. (2004).
//
// Reference (parent) 2D coordinates: X \in [-1, 1]^2.
//
// Local 2D physical coordinates on the 2D manifold: x \in [-l, l]^2
//   with l half edge of the cube inscribed in the sphere. These coordinate
//   systems vary locally on each face (or panel) of the cube.
//
// (theta, lambda) represnt latitude and longitude coordinates.
//
// Change of coordinates from x (on the 2D manifold) to xx (phyisical 3D on
//   the sphere), i.e., "cube-to-sphere" A, with equidistant central projection:
//
//   For lateral panels (P0-P3):
//   A = R cos(theta)cos(lambda) / l * [cos(lambda)                       0]
//                                     [-sin(theta)sin(lambda)   cos(theta)]
//
//   For top panel (P4):
//   A = R sin(theta) / l * [cos(lambda)                        sin(lambda)]
//                          [-sin(theta)sin(lambda)   sin(theta)cos(lambda)]
//
//   For bottom panel(P5):
//   A = R sin(theta) / l * [-cos(lambda)                       sin(lambda)]
//                          [sin(theta)sin(lambda)    sin(theta)cos(lambda)]
//
// The inverse of A, A^{-1}, is the "sphere-to-cube" change of coordinates.
//
// The metric tensor g_{ij} = A^TA, and its inverse,
// g^{ij} = g_{ij}^{-1} = A^{-1}A^{-T}
//
// modJ represents the magnitude of the cross product of the columns of A, i.e.,
// J = det(g_{ij})
//
// The quadrature data is stored in the array qdata.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v ).
//
// *****************************************************************************
CEED_QFUNCTION(SetupGeo)(void *ctx, CeedInt Q,
                         const CeedScalar *const *in, CeedScalar *const *out) {
  // *INDENT-OFF*
  // Inputs
  const CeedScalar (*X)[CEED_Q_VLA] = (const CeedScalar(*)[CEED_Q_VLA])in[0],
                   (*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[1],
                   (*w) = in[2];
  // Outputs
  CeedScalar (*qdata)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];
  // *INDENT-ON*

  CeedInt panel = -1;

  CeedPragmaSIMD
  // Quadrature point loop to determine which panel the element belongs to
  for (CeedInt i=0; i<Q; i++) {
    const CeedScalar pidx = X[2][i];
    if (pidx != -1)
      panel = pidx;
    break;
  }

  // Check that we found nodes panel
  if (panel == -1)
    return -1;

  CeedPragmaSIMD
  // Quadrature point loop for metric factors
  for (CeedInt i=0; i<Q; i++) {
    // Read local panel coordinates and relative panel index
    CeedScalar x[2] = {X[0][i],
                       X[1][i]
                      };
    const CeedScalar pidx = X[2][i];

    if (pidx != panel) {
      const CeedScalar theta  = X[0][i];
      const CeedScalar lambda = X[1][i];

      CeedScalar T_inv[2][2];

      switch (panel) {
      case 0:
      case 1:
      case 2:
      case 3:
        // For P_0 (front), P_1 (east), P_2 (back), P_3 (west):
        T_inv[0][0] = 1./(cos(theta)*cos(lambda)) * (1./cos(lambda));
        T_inv[0][1] = 1./(cos(theta)*cos(lambda)) * 0.;
        T_inv[1][0] = 1./(cos(theta)*cos(lambda)) * tan(theta)*tan(lambda);
        T_inv[1][1] = 1./(cos(theta)*cos(lambda)) * (1./cos(theta));
        break;
      case 4:
        // For P4 (north):
        T_inv[0][0] = 1./(sin(theta)*sin(theta)) * sin(theta)*cos(lambda);
        T_inv[0][1] = 1./(sin(theta)*sin(theta)) * (-sin(lambda));
        T_inv[1][0] = 1./(sin(theta)*sin(theta)) * sin(theta)*sin(lambda);
        T_inv[1][1] = 1./(sin(theta)*sin(theta)) * cos(lambda);
        break;
      case 5:
        // For P5 (south):
        T_inv[0][0] = 1./(sin(theta)*sin(theta)) * (-sin(theta)*cos(lambda));
        T_inv[0][1] = 1./(sin(theta)*sin(theta)) * sin(lambda);
        T_inv[1][0] = 1./(sin(theta)*sin(theta)) * sin(theta)*sin(lambda);
        T_inv[1][1] = 1./(sin(theta)*sin(theta)) * cos(lambda);
        break;
      }
      x[0] = T_inv[0][0]*theta + T_inv[0][1]*lambda;
      x[1] = T_inv[1][0]*theta + T_inv[1][1]*lambda;
    }

    const CeedScalar xxT[2][2] = {{x[0]*x[0],
                                   x[0]*x[1]},
                                  {x[1]*x[0],
                                   x[1]*x[1]}
                                  };

    // Read dxdX Jacobian entries, stored in columns
    // J_00 J_10
    // J_01 J_11
    const CeedScalar dxdX[2][2] = {{J[0][0][i],
                                    J[1][0][i]},
                                   {J[0][1][i],
                                    J[1][1][i]}
                                   };

    CeedScalar dxxdX[2][2];
    for (int j=0; j<2; j++)
      for (int k=0; k<2; k++) {
        dxxdX[j][k] = 0;
        for (int l=0; l<2; l++)
          dxxdX[j][k] += xxT[j][l]*dxdX[l][k];
      }

    // Interp-to-Interp qdata
    qdata[0][i] = (dxxdX[0][0]*dxxdX[1][1] - dxxdX[0][1]*dxxdX[1][0]) * w[i];

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

#endif // setupgeo_h
