/// @file
/// Test Symmetric Schur Decomposition
/// \test Test Symmetric Schur Decomposition
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedScalar A[16] = {0.2, 0.0745355993, -0.0745355993, 0.0333333333,
                      0.0745355993, 1., 0.1666666667, -0.0745355993,
                      -0.0745355993, 0.1666666667, 1., 0.0745355993,
                      0.0333333333, -0.0745355993, 0.0745355993, 0.2
                     };
  CeedScalar lambda[4];

  CeedInit(argv[1], &ceed);

  CeedSymmetricSchurDecomposition(ceed, A, lambda, 4);
  fprintf(stdout, "Q:\n");
  for (int i=0; i<4; i++) {
    for (int j=0; j<4; j++) {
      if (A[j+4*i] <= 1E-14 && A[j+4*i] >= -1E-14) A[j+4*i] = 0;
      fprintf(stdout, "%12.8f\t", A[j+4*i]);
    }
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "lambda:\n");
  for (int i=0; i<4; i++) {
    if (lambda[i] <= 1E-14 && lambda[i] >= -1E-14) lambda[i] = 0;
    fprintf(stdout, "%12.8f\n", lambda[i]);
  }
  CeedDestroy(&ceed);
  return 0;
}
