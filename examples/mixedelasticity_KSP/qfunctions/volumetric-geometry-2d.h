/// @file
/// volumetric geometric factors computation QFunction source

#ifndef volumetric_geometry2d_qf_h
#define volumetric_geometry2d_qf_h

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
CEED_QFUNCTION(SetupVolumeGeometry2D)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar(*J)[2][CEED_Q_VLA] = (const CeedScalar(*)[2][CEED_Q_VLA])in[0];
  const CeedScalar(*w)                = in[1];

  // Outputs
  CeedScalar(*q_data)[CEED_Q_VLA] = (CeedScalar(*)[CEED_Q_VLA])out[0];

  // Quadrature Point Loop
  CeedPragmaSIMD for (CeedInt i = 0; i < Q; i++) {
    // Setup
    const CeedScalar dxdX[2][2] = {
        {J[0][0][i], J[0][1][i]},
        {J[1][0][i], J[1][1][i]}
    };
    const CeedScalar det_dxdX = ComputeDet2(dxdX);
    // printf("det_dxdX %f\n", det_dxdX);
    CeedScalar dXdx_voigt[4];
    MatComputeInverseNonSymmetric2(dxdX, det_dxdX, dXdx_voigt);
    q_data[0][i] = w[i] * det_dxdX;
    for (CeedInt j = 0; j < 4; j++) {
      q_data[j + 1][i] = dXdx_voigt[j];
    }
  }  // End of Quadrature Point Loop
  return 0;
}
// -----------------------------------------------------------------------------
#endif  // volumetric_geometry2d_qf_h
