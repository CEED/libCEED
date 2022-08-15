/// @file
/// Utility helpers QFunction source

#ifndef utils_qf_h
#define utils_qf_h

#include "ceed/ceed-f64.h"
#include <math.h>

#define PI_DOUBLE 3.14159265358979323846

// -----------------------------------------------------------------------------
// Compute alpha * A * B = C
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatMatMult3x3(const CeedScalar alpha,
    const CeedScalar A[3][3], const CeedScalar B[3][3], CeedScalar C[3][3]) {
  for (CeedInt j = 0; j < 3; j++) {
    for (CeedInt k = 0; k < 3; k++) {
      C[j][k] = 0;
      for (CeedInt m = 0; m < 3; m++) {
        C[j][k] += alpha * A[j][m] * B[m][k];
      }
    }
  }

  return 0;
}

// -----------------------------------------------------------------------------
// Compute alpha * A^T * B = C
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatTransposeMatMult3x3(const CeedScalar alpha,
    const CeedScalar A[3][3], const CeedScalar B[3][3], CeedScalar C[3][3]) {
  for (CeedInt j = 0; j < 3; j++) {
    for (CeedInt k = 0; k < 3; k++) {
      C[j][k] = 0;
      for (CeedInt m = 0; m < 3; m++) {
        C[j][k] += alpha * A[m][j] * B[m][k];
      }
    }
  }

  return 0;
}

// -----------------------------------------------------------------------------
// Compute determinant of 3x3 matrix
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER CeedScalar MatDet3x3(const CeedScalar A[3][3]) {
  // Compute det(A)
  const CeedScalar B11 = A[1][1]*A[2][2] - A[1][2]*A[2][1];
  const CeedScalar B12 = A[0][2]*A[2][1] - A[0][1]*A[2][2];
  const CeedScalar B13 = A[0][1]*A[1][2] - A[0][2]*A[1][1];
  return A[0][0]*B11 + A[1][0]*B12 + A[2][0]*B13;

};

// -----------------------------------------------------------------------------
// Compute inverse of 3x3 symmetric matrix
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int MatInverse3x3(const CeedScalar A[3][3],
                                        const CeedScalar det_A, CeedScalar A_inv[3][3]) {
  // Compute A^(-1) : A-Inverse
  CeedScalar B[6] = {
    A[1][1] * A[2][2] - A[1][2] * A[2][1], /* *NOPAD* */
    A[0][0] * A[2][2] - A[0][2] * A[2][0], /* *NOPAD* */
    A[0][0] * A[1][1] - A[0][1] * A[1][0], /* *NOPAD* */
    A[0][2] * A[1][0] - A[0][0] * A[1][2], /* *NOPAD* */
    A[0][1] * A[1][2] - A[0][2] * A[1][1], /* *NOPAD* */
    A[0][2] * A[2][1] - A[0][1] * A[2][2]  /* *NOPAD* */
  };
  CeedScalar A_inv1[6];
  for (CeedInt m = 0; m < 6; m++) {
    A_inv1[m] = B[m] / (det_A);
  }
  A_inv[0][0] = A_inv1[0];
  A_inv[0][1] = A_inv1[5];
  A_inv[0][2] = A_inv1[4];
  A_inv[1][0] = A_inv1[5];
  A_inv[1][1] = A_inv1[1];
  A_inv[1][2] = A_inv1[3];
  A_inv[2][0] = A_inv1[4];
  A_inv[2][1] = A_inv1[3];
  A_inv[2][2] = A_inv1[2];
  return 0;
};

// -----------------------------------------------------------------------------
// Compute matrix-vector product: alpha*A*u
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatVecMult3x3(const CeedScalar alpha,
    const CeedScalar A[3][3], const CeedScalar u[3], CeedScalar v[3]) {
  // Compute v = alpha*A*u
  for (CeedInt k = 0; k < 3; k++) {
    v[k] = 0;
    for (CeedInt m = 0; m < 3; m++)
      v[k] += A[k][m] * u[m] * alpha;
  }

  return 0;
};

// -----------------------------------------------------------------------------
// Compute matrix-vector product: alpha*A^T*u
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatTransposeVecMult3x3(const CeedScalar alpha,
    const CeedScalar A[3][3], const CeedScalar u[3], CeedScalar v[3]) {
  // Compute v = alpha*A^T*u
  for (CeedInt k = 0; k < 3; k++) {
    v[k] = 0;
    for (CeedInt m = 0; m < 3; m++)
      v[k] += A[m][k] * u[m] * alpha;
  }

  return 0;
};

// -----------------------------------------------------------------------------
// Compute alpha * A * B = C
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatMatMult2x2(const CeedScalar alpha,
    const CeedScalar A[2][2], const CeedScalar B[2][2], CeedScalar C[2][2]) {
  for (CeedInt j = 0; j < 2; j++) {
    for (CeedInt k = 0; k < 2; k++) {
      C[j][k] = 0;
      for (CeedInt m = 0; m < 2; m++) {
        C[j][k] += alpha * A[j][m] * B[m][k];
      }
    }
  }

  return 0;
}

// -----------------------------------------------------------------------------
// Compute alpha * A^T * B = C
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatTransposeMatMult2x2(const CeedScalar alpha,
    const CeedScalar A[2][2], const CeedScalar B[2][2], CeedScalar C[2][2]) {
  for (CeedInt j = 0; j < 2; j++) {
    for (CeedInt k = 0; k < 2; k++) {
      C[j][k] = 0;
      for (CeedInt m = 0; m < 2; m++) {
        C[j][k] += alpha * A[m][j] * B[m][k];
      }
    }
  }

  return 0;
}

// -----------------------------------------------------------------------------
// Compute determinant of 2x2 matrix
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER CeedScalar MatDet2x2(const CeedScalar A[2][2]) {
  // Compute det(A)
  return A[0][0]*A[1][1] - A[1][0]*A[0][1];

};

// -----------------------------------------------------------------------------
// Compute inverse of 2x2 symmetric matrix
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int MatInverse2x2(const CeedScalar A[2][2],
                                        const CeedScalar det_A, CeedScalar A_inv[2][2]) {
  // Compute A^(-1) : A-Inverse
  A_inv[0][0] = A[1][1]/ det_A;
  A_inv[0][1] = -A[0][1]/ det_A;
  A_inv[1][0] = -A[1][0]/ det_A;
  A_inv[1][1] = A[0][0]/ det_A;

  return 0;
};

// -----------------------------------------------------------------------------
// Compute matrix-vector product: alpha*A*u
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatVecMult2x2(const CeedScalar alpha,
    const CeedScalar A[2][2], const CeedScalar u[2], CeedScalar v[2]) {
  // Compute v = alpha*A*u
  for (CeedInt k = 0; k < 2; k++) {
    v[k] = 0;
    for (CeedInt m = 0; m < 2; m++)
      v[k] += A[k][m] * u[m] * alpha;
  }

  return 0;
};

// -----------------------------------------------------------------------------
// Compute matrix-vector product: alpha*A^T*u
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatTransposeVecMult2x2(const CeedScalar alpha,
    const CeedScalar A[2][2], const CeedScalar u[2], CeedScalar v[2]) {
  // Compute v = alpha*A^T*u
  for (CeedInt k = 0; k < 2; k++) {
    v[k] = 0;
    for (CeedInt m = 0; m < 2; m++)
      v[k] += A[m][k] * u[m] * alpha;
  }

  return 0;
};

#endif  // utils_qf_h
