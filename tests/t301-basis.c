/// @file
/// Test QR Factorization
/// \test Test QR Factorization
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedScalar qr[12] = {1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0};
  CeedScalar tau[3];

  CeedInit(argv[1], &ceed);

  CeedQRFactorization(ceed, qr, tau, 4, 3);
  for (int i=0; i<12; i++) {
    if (qr[i] <= 1E-14 && qr[i] >= -1E-14) qr[i] = 0;
    fprintf(stdout, "%12.8f\n", qr[i]);
  }
  for (int i=0; i<3; i++) {
    if (tau[i] <= 1E-14 && qr[i] >= -1E-14) tau[i] = 0;
    fprintf(stdout, "%12.8f\n", tau[i]);
  }
  CeedDestroy(&ceed);
  return 0;
}
