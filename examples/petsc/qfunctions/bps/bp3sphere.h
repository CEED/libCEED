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
/// libCEED QFunctions for diffusion operator example for a scalar field on the sphere using PETSc

#ifndef bp3sphere_h
#define bp3sphere_h

#ifndef __CUDACC__
#  include <math.h>
#endif

// *****************************************************************************
// This QFunction sets up the geometric factors required for integration and
//   coordinate transformations when reference coordinates have a different
//   dimension than the one of physical coordinates
//
// Reference (parent) 2D coordinates: X \in [-1, 1]^2
//
// Global 3D physical coordinates given by the mesh: xx \in [-R, R]^3
//   with R radius of the sphere
//
// Local 3D physical coordinates on the 2D manifold: x \in [-l, l]^3
//   with l half edge of the cube inscribed in the sphere
//
// Change of coordinates matrix computed by the library:
//   (physical 3D coords relative to reference 2D coords)
//   dxx_j/dX_i (indicial notation) [3 * 2]
//
// Change of coordinates x (on the 2D manifold) relative to xx (phyisical 3D):
//   dx_i/dxx_j (indicial notation) [3 * 3]
//
// Change of coordinates x (on the 2D manifold) relative to X (reference 2D):
//   (by chain rule)
//   dx_i/dX_j [3 * 2] = dx_i/dxx_k [3 * 3] * dxx_k/dX_j [3 * 2]
//
// modJ is given by the magnitude of the cross product of the columns of dx_i/dX_j
//
// The quadrature data is stored in the array qdata.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int( u v )
//
// qdata[0]: modJ * w
//
// We use the Moore–Penrose (left) pseudoinverse of dx_i/dX_j, to compute dX_i/dx_j (and its transpose),
//   needed to properly compute integrals of the form: int( gradv gradu )
//
// dX_i/dx_j [2 * 3] = (dx_i/dX_j)+ = (dxdX^T dxdX)^(-1) dxdX
//
// and the product simplifies to yield the contravariant metric tensor
//
// g^{ij} = dX_i/dx_k dX_j/dx_k = (dxdX^T dxdX)^{-1}
//
// Stored: g^{ij} (in Voigt convention) in
//
//   qdata[1:3]: [dXdxdXdxT00 dXdxdXdxT01]
//               [dXdxdXdxT01 dXdxdXdxT11]
// *****************************************************************************

// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffGeo)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  const CeedScalar *X = in[0], *J = in[1], *w = in[2];
  CeedScalar *qdata = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read global Cartesian coordinates
    const CeedScalar xx[3] = {X[i+0*Q],
                              X[i+1*Q],
                              X[i+2*Q]
                             };

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

    // Setup
    // x = xx (xx^T xx)^{-1/2}
    // dx/dxx = I (xx^T xx)^{-1/2} - xx xx^T (xx^T xx)^{-3/2}
    const CeedScalar modxxsq = xx[0]*xx[0]+xx[1]*xx[1]+xx[2]*xx[2];
    CeedScalar xxsq[3][3];
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        xxsq[j][k] = xx[j]*xx[k] / (sqrt(modxxsq) * modxxsq);

    const CeedScalar dxdxx[3][3] = {{1./sqrt(modxxsq) - xxsq[0][0],
                                     -xxsq[0][1],
                                     -xxsq[0][2]},
                                    {-xxsq[1][0],
                                     1./sqrt(modxxsq) - xxsq[1][1],
                                     -xxsq[1][2]},
                                    {-xxsq[2][0],
                                     -xxsq[2][1],
                                     1./sqrt(modxxsq) - xxsq[2][2]}
                                   };

    CeedScalar dxdX[3][2];
    for (int j=0; j<3; j++)
      for (int k=0; k<2; k++) {
        dxdX[j][k] = 0;
        for (int l=0; l<3; l++)
          dxdX[j][k] += dxdxx[j][l]*dxxdX[l][k];
      }

    // J is given by the cross product of the columns of dxdX
    const CeedScalar J[3]= {dxdX[1][0]*dxdX[2][1] - dxdX[2][0]*dxdX[1][1],
                            dxdX[2][0]*dxdX[0][1] - dxdX[0][0]*dxdX[2][1],
                            dxdX[0][0]*dxdX[1][1] - dxdX[1][0]*dxdX[0][1]
                           };

    // Use the magnitude of J as our detJ (volume scaling factor)
    const CeedScalar modJ = sqrt(J[0]*J[0]+J[1]*J[1]+J[2]*J[2]);

    // Interp-to-Interp qdata
    qdata[i+Q*0] = modJ * w[i];

    // dxdX_k,j * dxdX_j,k
    CeedScalar dxdXTdxdX[2][2];
    for (int j=0; j<2; j++)
      for (int k=0; k<2; k++) {
        dxdXTdxdX[j][k] = 0;
        for (int l=0; l<3; l++)
          dxdXTdxdX[j][k] += dxdX[l][j]*dxdX[l][k];
      }

    const CeedScalar detdxdXTdxdX =  dxdXTdxdX[0][0] * dxdXTdxdX[1][1]
                                    -dxdXTdxdX[1][0] * dxdXTdxdX[0][1];

    // Compute inverse of dxdXTdxdX, which is the 2x2 contravariant metric tensor g^{ij}
    CeedScalar dxdXTdxdXinv[2][2];
    dxdXTdxdXinv[0][0] =  dxdXTdxdX[1][1] / detdxdXTdxdX;
    dxdXTdxdXinv[0][1] = -dxdXTdxdX[0][1] / detdxdXTdxdX;
    dxdXTdxdXinv[1][0] = -dxdXTdxdX[1][0] / detdxdXTdxdX;
    dxdXTdxdXinv[1][1] =  dxdXTdxdX[0][0] / detdxdXTdxdX;

    // Stored in Voigt convention
    qdata[i+Q*1] = dxdXTdxdXinv[0][0];
    qdata[i+Q*2] = dxdXTdxdXinv[1][1];
    qdata[i+Q*3] = dxdXTdxdXinv[0][1];
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
// This QFunction sets up the rhs and true solution for the problem
// *****************************************************************************

// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  // Inputs
  const CeedScalar *X = in[0], *qdata = in[1];
  // Outputs
  CeedScalar *true_soln = out[0], *rhs = out[1];

  // Context
  const CeedScalar *context = (const CeedScalar*)ctx;
  const CeedScalar R        = context[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read global Cartesian coordinates
    CeedScalar x = X[i+Q*0], y = X[i+Q*1], z = X[i+Q*2];
    // Normalize quadrature point coordinates to sphere
    CeedScalar rad = sqrt(x*x + y*y + z*z);
    x *= R / rad;
    y *= R / rad;
    z *= R / rad;
    // Compute latitude and longitude
    const CeedScalar theta  = asin(z / R); // latitude
    const CeedScalar lambda = atan2(y, x); // longitude

    true_soln[i+Q*0] = sin(lambda) * cos(theta);

    rhs[i+Q*0] = qdata[i+Q*0] * 2 * sin(lambda)*cos(theta) / (R*R);

  } // End of Quadrature Point Loop

  return 0;
}

// *****************************************************************************
// This QFunction applies the diffusion operator for a scalar field.
//
// Inputs:
//   ug     - Input vector gradient at quadrature points
//   qdata  - Geometric factors
//
// Output:
//   vg     - Output vector (test functions) gradient at quadrature points
//
// *****************************************************************************

// -----------------------------------------------------------------------------
CEED_QFUNCTION(Diff)(void *ctx, CeedInt Q,
                     const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar *ug = in[0], *qdata = in[1];
  // Outputs
  CeedScalar *vg = out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    // Read spatial derivatives of u
    const CeedScalar du[2]           =  {ug[i+Q*0],
                                         ug[i+Q*1]
                                        };
    // Read qdata
    const CeedScalar wdetJ           =   qdata[i+Q*0];
    // -- Grad-to-Grad qdata
    // ---- dXdx_j,k * dXdx_k,j
    const CeedScalar dXdxdXdxT[2][2] = {{qdata[i+Q*1],
                                         qdata[i+Q*3]},
                                        {qdata[i+Q*3],
                                         qdata[i+Q*2]}
                                       };

    for (int j=0; j<2; j++) // j = direction of vg
      vg[i+j*Q] = wdetJ * (du[0] * dXdxdXdxT[0][j] +
                           du[1] * dXdxdXdxT[1][j]);

  } // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif // bp3sphere_h
