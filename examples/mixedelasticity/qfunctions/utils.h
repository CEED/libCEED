/// @file
/// utility helpers QFunction source

#ifndef utils_qf_h
#define utils_qf_h

#include <math.h>

#define PI_DOUBLE 3.14159265358979323846

// -----------------------------------------------------------------------------
// Trace of a matrix
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER CeedScalar Trace2(const CeedScalar A[2][2]) { return A[0][0] + A[1][1]; }

CEED_QFUNCTION_HELPER CeedScalar Trace3(const CeedScalar A[3][3]) { return A[0][0] + A[1][1] + A[2][2]; }

// -----------------------------------------------------------------------------
// Compute alpha * A * B = C
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatMatMult2(const CeedScalar alpha, const CeedScalar A[2][2], const CeedScalar B[2][2], CeedScalar C[2][2]) {
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

CEED_QFUNCTION_HELPER int AlphaMatMatMult3(const CeedScalar alpha, const CeedScalar A[3][3], const CeedScalar B[3][3], CeedScalar C[3][3]) {
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
// Compute alpha * A * B^T = C
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatMatTransposeMult2(const CeedScalar alpha, const CeedScalar A[2][2], const CeedScalar B[2][2], CeedScalar C[2][2]) {
  for (CeedInt j = 0; j < 2; j++) {
    for (CeedInt k = 0; k < 2; k++) {
      C[j][k] = 0;
      for (CeedInt m = 0; m < 2; m++) {
        C[j][k] += alpha * A[j][m] * B[k][m];
      }
    }
  }

  return 0;
}

CEED_QFUNCTION_HELPER int AlphaMatMatTransposeMult3(const CeedScalar alpha, const CeedScalar A[3][3], const CeedScalar B[3][3], CeedScalar C[3][3]) {
  for (CeedInt j = 0; j < 3; j++) {
    for (CeedInt k = 0; k < 3; k++) {
      C[j][k] = 0;
      for (CeedInt m = 0; m < 3; m++) {
        C[j][k] += alpha * A[j][m] * B[k][m];
      }
    }
  }

  return 0;
}

// -----------------------------------------------------------------------------
// Compute alpha * A^T * B = C
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int AlphaMatTransposeMatMult2(const CeedScalar alpha, const CeedScalar A[2][2], const CeedScalar B[2][2], CeedScalar C[2][2]) {
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

CEED_QFUNCTION_HELPER int AlphaMatTransposeMatMult3(const CeedScalar alpha, const CeedScalar A[3][3], const CeedScalar B[3][3], CeedScalar C[3][3]) {
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
// Compute alpha * A : B = c
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER CeedScalar MatMatContract2(const CeedScalar alpha, const CeedScalar A[2][2], const CeedScalar B[2][2]) {
  CeedScalar c = 0;
  for (CeedInt j = 0; j < 2; j++) {
    for (CeedInt k = 0; k < 2; k++) {
      c += alpha * A[j][k] * B[j][k];
    }
  }

  return c;
}

CEED_QFUNCTION_HELPER CeedScalar MatMatContract3(const CeedScalar alpha, const CeedScalar A[3][3], const CeedScalar B[3][3]) {
  CeedScalar c = 0;
  for (CeedInt j = 0; j < 3; j++) {
    for (CeedInt k = 0; k < 3; k++) {
      c += alpha * A[j][k] * B[j][k];
    }
  }

  return c;
}

// -----------------------------------------------------------------------------
// Compute A^-1, where A is symmetric, returns array in Voigt notation
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int MatComputeInverseSymmetric2(const CeedScalar A[2][2], const CeedScalar det_A, CeedScalar A_inv[3]) {
  // Compute A^(-1) : A-Inverse
  CeedScalar B[3] = {
      A[1][1],  /* *NOPAD* */
      A[0][0],  /* *NOPAD* */
      -A[0][1], /* *NOPAD* */
  };
  for (CeedInt m = 0; m < 3; m++) {
    A_inv[m] = B[m] / (det_A);
  }

  return 0;
}

CEED_QFUNCTION_HELPER int MatComputeInverseSymmetric3(const CeedScalar A[3][3], const CeedScalar det_A, CeedScalar A_inv[6]) {
  // Compute A^(-1) : A-Inverse
  CeedScalar B[6] = {
      A[1][1] * A[2][2] - A[1][2] * A[2][1], /* *NOPAD* */
      A[0][0] * A[2][2] - A[0][2] * A[2][0], /* *NOPAD* */
      A[0][0] * A[1][1] - A[0][1] * A[1][0], /* *NOPAD* */
      A[0][2] * A[1][0] - A[0][0] * A[1][2], /* *NOPAD* */
      A[0][1] * A[1][2] - A[0][2] * A[1][1], /* *NOPAD* */
      A[0][2] * A[2][1] - A[0][1] * A[2][2]  /* *NOPAD* */
  };
  for (CeedInt m = 0; m < 6; m++) {
    A_inv[m] = B[m] / (det_A);
  }

  return 0;
}

// -----------------------------------------------------------------------------
// Compute A^-1, where A is nonsymmetric, returns array in Voigt notation
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER int MatComputeInverseNonSymmetric2(const CeedScalar A[2][2], const CeedScalar det_A, CeedScalar A_inv[4]) {
  // Compute A^(-1) : A-Inverse
  CeedScalar B[4] = {
      A[1][1],  /* *NOPAD* */
      A[0][0],  /* *NOPAD* */
      -A[0][1], /* *NOPAD* */
      -A[1][0]  /* *NOPAD* */
  };
  for (CeedInt m = 0; m < 4; m++) {
    A_inv[m] = B[m] / (det_A);
  }

  return 0;
}

CEED_QFUNCTION_HELPER int MatComputeInverseNonSymmetric3(const CeedScalar A[3][3], const CeedScalar det_A, CeedScalar A_inv[9]) {
  // Compute A^(-1) : A-Inverse
  CeedScalar B[9] = {
      A[1][1] * A[2][2] - A[1][2] * A[2][1], /* *NOPAD* */
      A[0][0] * A[2][2] - A[0][2] * A[2][0], /* *NOPAD* */
      A[0][0] * A[1][1] - A[0][1] * A[1][0], /* *NOPAD* */
      A[0][2] * A[1][0] - A[0][0] * A[1][2], /* *NOPAD* */
      A[0][1] * A[1][2] - A[0][2] * A[1][1], /* *NOPAD* */
      A[0][2] * A[2][1] - A[0][1] * A[2][2], /* *NOPAD* */
      A[0][1] * A[2][0] - A[0][0] * A[2][1], /* *NOPAD* */
      A[1][0] * A[2][1] - A[1][1] * A[2][0], /* *NOPAD* */
      A[1][2] * A[2][0] - A[1][0] * A[2][2]  /* *NOPAD* */
  };
  for (CeedInt m = 0; m < 9; m++) {
    A_inv[m] = B[m] / (det_A);
  }

  return 0;
}

// -----------------------------------------------------------------------------
//  Create symetric entries from full tensor
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER void VoigtPack2(const CeedScalar full[2][2], CeedScalar sym[3]) {
  sym[0] = full[0][0];
  sym[1] = full[1][1];
  sym[2] = full[0][1];
}

CEED_QFUNCTION_HELPER void VoigtPack3(const CeedScalar full[3][3], CeedScalar sym[6]) {
  sym[0] = full[0][0];
  sym[1] = full[1][1];
  sym[2] = full[2][2];
  sym[3] = full[1][2];
  sym[4] = full[0][2];
  sym[5] = full[0][1];
}

// -----------------------------------------------------------------------------
//  Create full tensor from symetric entries
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER void VoigtUnpackSymmetric2(const CeedScalar sym[3], CeedScalar full[2][2]) {
  full[0][0] = sym[0];
  full[0][1] = sym[2];
  full[1][0] = sym[2];
  full[1][1] = sym[1];
}

CEED_QFUNCTION_HELPER void VoigtUnpackSymmetric3(const CeedScalar sym[6], CeedScalar full[3][3]) {
  full[0][0] = sym[0];
  full[0][1] = sym[5];
  full[0][2] = sym[4];
  full[1][0] = sym[5];
  full[1][1] = sym[1];
  full[1][2] = sym[3];
  full[2][0] = sym[4];
  full[2][1] = sym[3];
  full[2][2] = sym[2];
}

// -----------------------------------------------------------------------------
//  Create full tensor from non-symmetric entries
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER void VoigtUnpackNonSymmetric2(const CeedScalar nonsym[4], CeedScalar full[2][2]) {
  full[0][0] = nonsym[0];
  full[0][1] = nonsym[2];
  full[1][0] = nonsym[3];
  full[1][1] = nonsym[1];
}

CEED_QFUNCTION_HELPER void VoigtUnpackNonSymmetric3(const CeedScalar nonsym[9], CeedScalar full[3][3]) {
  full[0][0] = nonsym[0];
  full[0][1] = nonsym[5];
  full[0][2] = nonsym[4];
  full[1][0] = nonsym[8];
  full[1][1] = nonsym[1];
  full[1][2] = nonsym[3];
  full[2][0] = nonsym[7];
  full[2][1] = nonsym[6];
  full[2][2] = nonsym[2];
}

// -----------------------------------------------------------------------------
// Compute determinant
// -----------------------------------------------------------------------------
CEED_QFUNCTION_HELPER CeedScalar ComputeDet2(const CeedScalar A[2][2]) {
  // Compute det(A)
  return A[0][0] * A[1][1] - A[1][0] * A[0][1];
};

CEED_QFUNCTION_HELPER CeedScalar ComputeDet3(const CeedScalar A[3][3]) {
  // Compute det(A)
  const CeedScalar B11 = A[1][1] * A[2][2] - A[1][2] * A[2][1];
  const CeedScalar B12 = A[0][2] * A[2][1] - A[0][1] * A[2][2];
  const CeedScalar B13 = A[0][1] * A[1][2] - A[0][2] * A[1][1];
  return A[0][0] * B11 + A[1][0] * B12 + A[2][0] * B13;
};
#endif  // ratel_utils_qf_h
