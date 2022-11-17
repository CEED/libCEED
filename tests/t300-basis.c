/// @file
/// Test creation and distruction of a H1Lagrange basis
/// \test Test creation and distruction of a H1Lagrange basis
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed      ceed;
  CeedBasis b;

  CeedInit(argv[1], &ceed);

  // Test skipped if using single precision
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test not implemented in single precision\n");
  }

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 4, 4, CEED_GAUSS_LOBATTO, &b);
  CeedBasisView(b, stdout);
  CeedBasisDestroy(&b);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 4, 4, CEED_GAUSS, &b);
  CeedBasisView(b, stdout);
  CeedBasisDestroy(&b);

  CeedDestroy(&ceed);
  return 0;
}
