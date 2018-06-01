// Test QR Factorization
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedScalar qr[12] = {1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0};

  CeedInit(argv[1], &ceed);
  CeedQRFactorization(qr, 4, 3);
  for (int i= 0; i<12; i++) {
    if (qr[i] <= 1E-14 && qr[i] >= -1E-14) qr[i] = 0;
    fprintf(stdout, "%12.8f\n", qr[i]);
  }
  CeedDestroy(&ceed);
  return 0;
}
