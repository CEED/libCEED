/// @file
/// Test Simultaneous Diagonalization
/// \test Simultaneous Diagonalization
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedScalar M[16] = {0.19996678, 0.0745459, -0.07448852, 0.0332866,
                      0.0745459, 1., 0.16666509, -0.07448852,
                      -0.07448852, 0.16666509, 1., 0.0745459,
                      0.0332866, -0.07448852, 0.0745459, 0.19996678};
  CeedScalar K[16] = {3.03344425, -3.41501767, 0.49824435, -0.11667092,
                     -3.41501767, 5.83354662, -2.9167733, 0.49824435,
                      0.49824435, -2.9167733, 5.83354662, -3.41501767,
                      -0.11667092, 0.49824435, -3.41501767, 3.03344425};
  CeedScalar x[16], lambda[4];

  CeedInit(argv[1], &ceed);

  CeedSimultaneousDiagonalization(ceed, K, M, x, lambda, 4);
  fprintf(stdout, "x:\n");
  for (int i=0; i<4; i++) {
    for (int j=0; j<4; j++) {
      if (x[j+4*i] <= 1E-14 && x[j+4*i] >= -1E-14) x[j+4*i] = 0;
      fprintf(stdout, "%12.8f\t", x[j+4*i]);
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
