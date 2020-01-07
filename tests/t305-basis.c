/// @file
/// Test Simultaneous Diagonalization
/// \test Simultaneous Diagonalization
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedScalar M[16] = {0.2, 0.0745355993, -0.0745355993, 0.0333333333,
                      0.0745355993, 1., 0.1666666667, -0.0745355993,
                      -0.0745355993, 0.1666666667, 1., 0.0745355993,
                      0.0333333333, -0.0745355993, 0.0745355993, 0.2
                     };
  CeedScalar K[16] = {3.0333333333, -3.4148928136, 0.4982261470, -0.1166666667,
                      -3.4148928136, 5.8333333333, -2.9166666667, 0.4982261470,
                      0.4982261470, -2.9166666667, 5.8333333333, -3.4148928136,
                      -0.1166666667, 0.4982261470, -3.4148928136, 3.0333333333
                     };
  CeedScalar x[16], lambda[4], xxt[16];

  CeedInit(argv[1], &ceed);

  CeedSimultaneousDiagonalization(ceed, K, M, x, lambda, 4);

  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++) {
      xxt[j+4*i] = 0;
      for (int k=0; k<4; k++)
        xxt[j+4*i] += x[k+4*i]*x[k+4*j];
    }

  fprintf(stdout, "x x^T:\n");
  for (int i=0; i<4; i++) {
    for (int j=0; j<4; j++) {
      if (xxt[j+4*i] <= 1E-14 && xxt[j+4*i] >= -1E-14) xxt[j+4*i] = 0;
      fprintf(stdout, "%12.8f\t", xxt[j+4*i]);
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
