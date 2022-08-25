/// @file
/// Test GetNumNodes and GetNumQuadraturePoints for basis
/// \test Test GetNumNodes and GetNumQuadraturePoints for basis
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed      ceed;
  CeedBasis b;

  CeedInit(argv[1], &ceed);

  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, 4, 5, CEED_GAUSS_LOBATTO, &b);

  CeedInt P, Q;
  CeedBasisGetNumNodes(b, &P);
  CeedBasisGetNumQuadraturePoints(b, &Q);

  if (P != 64) printf("%" CeedInt_FMT " != 64\n", P);
  if (Q != 125) printf("%" CeedInt_FMT " != 125\n", Q);

  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
