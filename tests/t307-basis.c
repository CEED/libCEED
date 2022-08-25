/// @file
/// Test creation, copying, and distruction of a H1Lagrange basis
/// \test Test creation, copying, and distruction of a H1Lagrange basis
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed      ceed;
  CeedBasis b, b_2;
  CeedInt   P_1d = 4;

  CeedInit(argv[1], &ceed);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P_1d, 4, CEED_GAUSS_LOBATTO, &b);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P_1d + 1, 4, CEED_GAUSS_LOBATTO, &b_2);

  CeedBasisReferenceCopy(b, &b_2);  // This destroys the previous b_2
  CeedBasisDestroy(&b);

  CeedInt P_1d_2;
  CeedBasisGetNumNodes1D(b_2, &P_1d_2);
  if (P_1d != P_1d_2) printf("Error copying CeedBasis reference\n");

  CeedBasisDestroy(&b_2);
  CeedDestroy(&ceed);
  return 0;
}
