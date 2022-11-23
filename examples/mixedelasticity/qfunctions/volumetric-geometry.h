/// @file
/// volumetric geometric factors computation QFunction source

#ifndef volumetric_geometry_qf_h
#define volumetric_geometry_qf_h

#include "utils.h"
// -----------------------------------------------------------------------------
// This QFunction sets up the geometric factors required to apply the
//   diffusion operator
//
// We require the product of the inverse of the Jacobian and its transpose to
//   properly compute integrals of the form: int( gradv gradu)
//
// Determinant of Jacobian:
//   detJ = J11*A11 + J21*A12 + J31*A13
//     Jij = Jacobian entry ij
//     Aij = Adjoint ij
//
// Inverse of Jacobian:
//   Bij = Aij / detJ
//
// Product of Inverse and Transpose:
//   BBij = sum( Bik Bkj )
//
// Stored: w B^T B detJ = w A^T A / detJ
//   Note: This matrix is symmetric, so we only store 6 distinct entries
//     qd: 1 4 7
//         2 5 8
//         3 6 9
// -----------------------------------------------------------------------------
CEED_QFUNCTION(SetupVolumeGeometry)(void *ctx, CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
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
    const CeedScalar det_dxdX = ComputeDet(dxdX);
    // printf("det_dxdX %f\n", det_dxdX);
    CeedScalar dXdx_voigt[9];
    MatComputeInverseNonSymmetric(dxdX, det_dxdX, dXdx_voigt);
    q_data[0][i] = w[i] * det_dxdX;
    for (CeedInt j = 0; j < 9; j++) {
      q_data[j + 1][i] = dXdx_voigt[j];
    }

  }  // End of Quadrature Point Loop

  return 0;
}
// -----------------------------------------------------------------------------

#endif  // volumetric_geometry_qf_h
