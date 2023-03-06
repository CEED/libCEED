/// @file
/// Test creation and destruction of a H1Lagrange basis
/// \test Test creation and destruction of a H1Lagrange basis
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed      ceed;
  CeedBasis basis;

  CeedInit(argv[1], &ceed);

  // Test skipped if using single precision
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) {
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test not implemented in single precision\n");
  }

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 4, 4, CEED_GAUSS_LOBATTO, &basis);
  CeedBasisView(basis, stdout);
  CeedBasisDestroy(&basis);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 4, 4, CEED_GAUSS, &basis);
  CeedBasisView(basis, stdout);
  CeedBasisDestroy(&basis);

  CeedDestroy(&ceed);
  return 0;
}
