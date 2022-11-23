/// @file
/// Test Simultaneous Diagonalization
/// \test Simultaneous Diagonalization
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedInt    P = 4, Q = 4;
  CeedScalar M[P * P], K[P * P], X[P * P], lambda[P];
  CeedBasis  basis;

  CeedInit(argv[1], &ceed);

  // Create mass, stiffness matrix
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &basis);
  const CeedScalar *interp, *grad, *quad_weights;
  CeedBasisGetInterp(basis, &interp);
  CeedBasisGetGrad(basis, &grad);
  CeedBasisGetQWeights(basis, &quad_weights);
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      CeedScalar sum_m = 0, sum_k = 0;
      for (int k = 0; k < Q; k++) {
        sum_m += interp[P * k + i] * quad_weights[k] * interp[P * k + j];
        sum_k += grad[P * k + i] * quad_weights[k] * grad[P * k + j];
      }
      M[P * i + j] = sum_m;
      K[P * i + j] = sum_k;
    }
  }

  CeedSimultaneousDiagonalization(ceed, K, M, X, lambda, P);

  // Check X^T M X = I
  CeedScalar work[P * P];
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < P; k++) sum += M[P * i + k] * X[P * k + j];
      work[P * i + j] = sum;
    }
  }
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < P; k++) sum += X[P * k + i] * work[P * k + j];
      M[P * i + j] = sum;
    }
  }
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      if (fabs(M[P * i + j] - (i == j ? 1.0 : 0.0)) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Error in diagonalization of M [%" CeedInt_FMT ", %" CeedInt_FMT "]: %f != %f\n", i, j, M[P * i + j], (i == j ? 1.0 : 0.0));
        // LCOV_EXCL_STOP
      }
    }
  }

  // Check X^T K X = Lamda
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < P; k++) sum += K[P * i + k] * X[P * k + j];
      work[P * i + j] = sum;
    }
  }
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      CeedScalar sum = 0;
      for (int k = 0; k < P; k++) sum += X[P * k + i] * work[P * k + j];
      K[P * i + j] = sum;
    }
  }
  for (int i = 0; i < P; i++) {
    for (int j = 0; j < P; j++) {
      if (fabs(K[P * i + j] - (i == j ? lambda[i] : 0.0)) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Error in diagonalization of K [%" CeedInt_FMT ", %" CeedInt_FMT "]: %f != %f\n", i, j, K[P * i + j], (i == j ? lambda[i] : 0.0));
        // LCOV_EXCL_STOP
      }
    }
  }

  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
