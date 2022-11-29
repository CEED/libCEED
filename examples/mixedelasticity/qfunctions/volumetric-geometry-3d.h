/// @file
/// volumetric geometric factors computation QFunction source

#ifndef volumetric_geometry3d_qf_h
#define volumetric_geometry3d_qf_h

#include "utils.h"
// -----------------------------------------------------------------------------
// This QFunction setup q_data, the inverse of the Jacobian
// Inputs:
//   J          : dx/dX
//   w          : weight of quadrature
//
// Output:
//   q_data     : updated weight of quadrature and inverse of the Jacobian J; [wdetJ, dXdx]
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupVolumeGeometry3D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*J)[3][CEED_Q_VLA] = (const CeedScalar(*)[3][CEED_Q_VLA])in[0];
  const CeedScalar(*w)                = in[1];

  // Outputs
  CeedScalar(*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    const CeedScalar dxdX[3][3] = {
        {J[0][0][i], J[1][0][i], J[2][0][i]},
        {J[0][1][i], J[1][1][i], J[2][1][i]},
        {J[0][2][i], J[1][2][i], J[2][2][i]}
    };
    const CeedScalar detJ = ComputeDet3(dxdX);
    CeedScalar       dXdx_voigt[9];
    MatComputeInverseNonSymmetric3(dxdX, detJ, dXdx_voigt);
    q_data[0][i] = w[i] * detJ;
    for (CeedInt j = 0; j < 9; j++) {
      q_data[j + 1][i] = dXdx_voigt[j];
    }
    // CeedScalar dXdx[3][3];
    // VoigtUnpackNonSymmetric3(dXdx_voigt, dXdx);
    // printf("dXdx[0,:] %f, %f, %f\n", dXdx[0][0], dXdx[0][1], dXdx[0][2]);
    // printf("dXdx[1,:] %f, %f, %f\n", dXdx[1][0], dXdx[1][1], dXdx[1][2]);
    // printf("dXdx[2,:] %f, %f, %f\n", dXdx[2][0], dXdx[2][1], dXdx[2][2]);
    // printf("dxdX[0,:] %f, %f, %f\n", dxdX[0][0], dxdX[0][1], dxdX[0][2]);
    // printf("dxdX[1,:] %f, %f, %f\n", dxdX[1][0], dxdX[1][1], dxdX[1][2]);
    // printf("dxdX[2,:] %f, %f, %f\n", dxdX[2][0], dxdX[2][1], dxdX[2][2]);

  }  // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------
#endif  // volumetric_geometry3d_qf_h
