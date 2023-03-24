/// @file
/// Test GetNumNodes and GetNumQuadraturePoints for basis
/// \test Test GetNumNodes and GetNumQuadraturePoints for basis
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed      ceed;
  CeedBasis basis;
  CeedInt   p, q;

  CeedInit(argv[1], &ceed);

  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, 4, 5, CEED_GAUSS_LOBATTO, &basis);

  CeedBasisGetNumNodes(basis, &p);
  CeedBasisGetNumQuadraturePoints(basis, &q);

  if (p != 64) printf("%" CeedInt_FMT " != 64\n", p);
  if (q != 125) printf("%" CeedInt_FMT " != 125\n", q);

  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
