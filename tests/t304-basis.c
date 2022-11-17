/// @file
/// Test Symmetric Schur Decomposition
/// \test Test Symmetric Schur Decomposition
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedInt    P = 4;
  CeedScalar M[16], Q[16], lambda[4], Q_lambda_Qt[16];
  CeedBasis  basis;

  CeedInit(argv[1], &ceed);

  // Create mass matrix
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, P, CEED_GAUSS, &basis);
  const CeedScalar *interp, *quad_weights;
  CeedBasisGetInterp(basis, &interp);
  CeedBasisGetQWeights(basis, &quad_weights);
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < P; k++) sum += interp[P * k + i] * quad_weights[k] * interp[P * k + j];
      M[P * i + j] = sum;
      Q[P * i + j] = sum;
    }
  }

  CeedSymmetricSchurDecomposition(ceed, Q, lambda, P);

  // Check diagonalization of M
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < P; k++) sum += Q[P * i + k] * lambda[k] * Q[P * j + k];
      Q_lambda_Qt[P * i + j] = sum;
    }
  }
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      if (fabs(M[P * i + j] - Q_lambda_Qt[P * i + j]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Error in diagonalization [%" CeedInt_FMT ", %" CeedInt_FMT "]: %f != %f\n", i, j, M[P * i + j], Q_lambda_Qt[P * i + j]);
        // LCOV_EXCL_STOP
      }
    }
  }

  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
