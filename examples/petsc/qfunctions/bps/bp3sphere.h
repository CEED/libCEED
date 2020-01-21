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
    const CeedScalar xx[3][1] = {{X[i+0*Q]},
                                 {X[i+1*Q]},
                                 {X[i+2*Q]}
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
    const CeedScalar modxxsq = xx[0][0]*xx[0][0]+xx[1][0]*xx[1][0]+xx[2][0]*xx[2][0];
    CeedScalar xxsq[3][3];
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++) {
        xxsq[j][k] = 0;
        for (int l=0; l<1; l++)
          xxsq[j][k] += xx[j][l]*xx[k][l] / (sqrt(modxxsq) * modxxsq);
      }

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

    // dxdX_j,k * dxdX_k,j
    CeedScalar dxdXdxdXT[3][3];
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++) {
        dxdXdxdXT[j][k] = 0;
        for (int l=0; l<2; l++)
          dxdXdxdXT[j][k] += dxdX[j][l]*dxdX[k][l];
      }

    // Invert dxdX_j,k * dxdX_k,j
    // -- Find cofactors
    const CeedScalar A11 = dxdXdxdXT[2][2]*dxdXdxdXT[3][3] - dxdXdxdXT[2][3]*dxdXdxdXT[3][2];
    const CeedScalar A12 = dxdXdxdXT[1][3]*dxdXdxdXT[3][2] - dxdXdxdXT[1][2]*dxdXdxdXT[3][3];
    const CeedScalar A13 = dxdXdxdXT[1][2]*dxdXdxdXT[2][3] - dxdXdxdXT[1][3]*dxdXdxdXT[2][2];
    const CeedScalar A21 = dxdXdxdXT[2][3]*dxdXdxdXT[3][1] - dxdXdxdXT[2][1]*dxdXdxdXT[3][3];
    const CeedScalar A22 = dxdXdxdXT[1][1]*dxdXdxdXT[3][3] - dxdXdxdXT[1][3]*dxdXdxdXT[3][1];
    const CeedScalar A23 = dxdXdxdXT[1][3]*dxdXdxdXT[2][1] - dxdXdxdXT[1][1]*dxdXdxdXT[2][3];
    const CeedScalar A31 = dxdXdxdXT[2][1]*dxdXdxdXT[3][2] - dxdXdxdXT[2][2]*dxdXdxdXT[3][1];
    const CeedScalar A32 = dxdXdxdXT[1][2]*dxdXdxdXT[3][1] - dxdXdxdXT[1][1]*dxdXdxdXT[3][2];
    const CeedScalar A33 = dxdXdxdXT[1][1]*dxdXdxdXT[2][2] - dxdXdxdXT[1][2]*dxdXdxdXT[2][1];
    const CeedScalar detdXdxdXdxT = dxdXdxdXT[1][1]*A11 + dxdXdxdXT[2][1]*A12 + dxdXdxdXT[3][1]*A13;


    // Interp-to-Interp qdata
    qdata[i+Q*0] = w[i] * detdXdxdXdxT;

    // Inverse of dxdX_j,k * dxdX_k,j = dXdx_j,k * dXdx_k,j
    CeedScalar dXdxdXdxT[3][3];
    dXdxdXdxT[0][0] = A11 / detdXdxdXdxT;
    dXdxdXdxT[0][1] = A12 / detdXdxdXdxT;
    dXdxdXdxT[0][2] = A13 / detdXdxdXdxT;
    dXdxdXdxT[1][0] = A21 / detdXdxdXdxT;
    dXdxdXdxT[1][1] = A22 / detdXdxdXdxT;
    dXdxdXdxT[1][2] = A23 / detdXdxdXdxT;
    dXdxdXdxT[2][0] = A31 / detdXdxdXdxT;
    dXdxdXdxT[2][1] = A32 / detdXdxdXdxT;
    dXdxdXdxT[2][2] = A33 / detdXdxdXdxT;

    // Grad-to-Grad qdata is given by (dXdx_j,k * dXdx_k,j) * (dXdx_j,k * dXdx_k,j)T
    CeedScalar dXdx[3][3];
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++) {
        dXdx[j][k] = 0;
        for (int l=0; l<2; l++)
          dXdx[j][k] += dXdxdXdxT[j][l]*dXdxdXdxT[k][l];
      }

    qdata[i+Q*1] = dXdx[0][0];
    qdata[i+Q*2] = dXdx[0][1];
    qdata[i+Q*3] = dXdx[0][2];
    qdata[i+Q*4] = dXdx[1][0];
    qdata[i+Q*5] = dXdx[1][1];
    qdata[i+Q*6] = dXdx[1][2];
    qdata[i+Q*7] = dXdx[2][0];
    qdata[i+Q*8] = dXdx[2][1];
    qdata[i+Q*9] = dXdx[2][2];
  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupDiffRhs)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
#ifndef M_PI
#  define M_PI    3.14159265358979323846
#endif
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
    const CeedScalar c     = sqrt(5 / (4 * M_PI));
    const CeedScalar theta =  asin(X[i+2*Q] / R);
    // Use spherical harmonics of degree 2, order 0 as solution of Laplacian on sphere
    const CeedScalar P2sin = .5 * c * (3 * sin(theta) * sin(theta) - 1);

    true_soln[i] = P2sin;

    rhs[i] = qdata[i] * true_soln[i];
  } // End of Quadrature Point Loop

  return 0;
}

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
    const CeedScalar du[3]       =  {ug[i+Q*0],
                                     ug[i+Q*1],
                                     ug[i+Q*2]
                                    };
    // Read qdata (dXdxT symmetric matrix)
    const CeedScalar dXdxT[3][3] = {{qdata[i+0*Q],
                                     qdata[i+1*Q],
                                     qdata[i+2*Q]},
                                    {qdata[i+1*Q],
                                     qdata[i+3*Q],
                                     qdata[i+4*Q]},
                                    {qdata[i+2*Q],
                                     qdata[i+4*Q],
                                     qdata[i+5*Q]}
                                   };

    for (int j=0; j<3; j++) // j = direction of vg
      vg[i+j*Q] = (du[0] * dXdxT[0][j] +
                   du[1] * dXdxT[1][j] +
                   du[2] * dXdxT[2][j]);

  } // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------
