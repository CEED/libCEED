/// @file
/// Test GetNumNodes and GetNumQuadraturePoints for basis
/// \test Test GetNumNodes and GetNumQuadraturePoints for basis
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedBasis b;

  CeedInit(argv[1], &ceed);

  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, 4, 5, CEED_GAUSS_LOBATTO, &b);

  CeedInt P, Q;
  CeedBasisGetNumNodes(b, &P);
  CeedBasisGetNumQuadraturePoints(b, &Q);

  if (P != 64)
    // LCOV_EXCL_START
    printf("%d != 64\n", P);
  // LCOV_EXCL_STOP
  if (Q != 125)
    // LCOV_EXCL_START
    printf("%d != 125\n", Q);
  // LCOV_EXCL_STOP

  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
