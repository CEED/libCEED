/// @file
/// Test Simultaneous Diagonalization
/// \test Simultaneous Diagonalization
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedInt           p = 4, q = 4;
  CeedScalar        M[p * p], K[p * p], X[p * p], lambda[p];
  CeedBasis         basis;
  const CeedScalar *interpolation, *gradient, *quadrature_weights;

  CeedInit(argv[1], &ceed);

  // Create mass, stiffness matrix
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, q, CEED_GAUSS, &basis);
  CeedBasisGetInterp(basis, &interpolation);
  CeedBasisGetGrad(basis, &gradient);
  CeedBasisGetQWeights(basis, &quadrature_weights);
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      CeedScalar sum_m = 0, sum_k = 0;
      for (int k = 0; k < q; k++) {
        sum_m += interpolation[p * k + i] * quadrature_weights[k] * interpolation[p * k + j];
        sum_k += gradient[p * k + i] * quadrature_weights[k] * gradient[p * k + j];
      }
      M[p * i + j] = sum_m;
      K[p * i + j] = sum_k;
    }
  }

  CeedSimultaneousDiagonalization(ceed, K, M, X, lambda, p);

  // Check X^T M X = I
  CeedScalar work_array[p * p];
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < p; k++) sum += M[p * i + k] * X[p * k + j];
      work_array[p * i + j] = sum;
    }
  }
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < p; k++) sum += X[p * k + i] * work_array[p * k + j];
      M[p * i + j] = sum;
    }
  }
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      if (fabs(M[p * i + j] - (i == j ? 1.0 : 0.0)) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Error in diagonalization of M [%" CeedInt_FMT ", %" CeedInt_FMT "]: %f != %f\n", i, j, M[p * i + j], (i == j ? 1.0 : 0.0));
        // LCOV_EXCL_STOP
      }
    }
  }

  // Check X^T K X = Lambda
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < p; k++) sum += K[p * i + k] * X[p * k + j];
      work_array[p * i + j] = sum;
    }
  }
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < p; k++) sum += X[p * k + i] * work_array[p * k + j];
      K[p * i + j] = sum;
    }
  }
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      if (fabs(K[p * i + j] - (i == j ? lambda[i] : 0.0)) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Error in diagonalization of K [%" CeedInt_FMT ", %" CeedInt_FMT "]: %f != %f\n", i, j, K[p * i + j], (i == j ? lambda[i] : 0.0));
        // LCOV_EXCL_STOP
      }
    }
  }

  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
