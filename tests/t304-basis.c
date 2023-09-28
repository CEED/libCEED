/// @file
/// Test Symmetric Schur Decomposition
/// \test Test Symmetric Schur Decomposition

//TESTARGS(only="cpu") {ceed_resource}
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedInt           p = 4;
  CeedScalar        M[16], Q[16], lambda[4], Q_lambda_Qt[16];
  CeedBasis         basis;
  const CeedScalar *interpolation, *quadrature_weights;

  CeedInit(argv[1], &ceed);

  // Create mass matrix
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, p, CEED_GAUSS, &basis);
  CeedBasisGetInterp(basis, &interpolation);
  CeedBasisGetQWeights(basis, &quadrature_weights);
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < p; k++) sum += interpolation[p * k + i] * quadrature_weights[k] * interpolation[p * k + j];
      M[p * i + j] = sum;
      Q[p * i + j] = sum;
    }
  }

  CeedSymmetricSchurDecomposition(ceed, Q, lambda, p);

  // Check diagonalization of M
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < p; k++) sum += Q[p * i + k] * lambda[k] * Q[p * j + k];
      Q_lambda_Qt[p * i + j] = sum;
    }
  }
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      if (fabs(M[p * i + j] - Q_lambda_Qt[p * i + j]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Error in diagonalization [%" CeedInt_FMT ", %" CeedInt_FMT "]: %f != %f\n", i, j, M[p * i + j], Q_lambda_Qt[p * i + j]);
        // LCOV_EXCL_STOP
      }
    }
  }

  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
