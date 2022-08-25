/// @file
/// Test QR Factorization
/// \test Test QR Factorization
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedScalar A[12]    = {1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0};
  CeedScalar qr[12]   = {1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0};
  CeedScalar A_qr[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  CeedScalar tau[4];

  CeedInit(argv[1], &ceed);

  CeedQRFactorization(ceed, qr, tau, 4, 3);
  for (CeedInt i = 0; i < 3; i++) {
    for (CeedInt j = i; j < 3; j++) A_qr[i * 3 + j] = qr[i * 3 + j];
  }
  CeedHouseholderApplyQ(A_qr, qr, tau, CEED_NOTRANSPOSE, 4, 3, 3, 3, 1);

  for (CeedInt i = 0; i < 12; i++) {
    if (fabs(A_qr[i] - A[i]) > 100. * CEED_EPSILON) {
      // LCOV_EXCL_START
      printf("Error in QR factorization A_qr[%" CeedInt_FMT "] = %f != A[%" CeedInt_FMT "] = %f\n", i, A_qr[i], i, A[i]);
      // LCOV_EXCL_STOP
    }
  }

  CeedDestroy(&ceed);
  return 0;
}
