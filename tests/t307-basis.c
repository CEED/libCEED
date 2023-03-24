/// @file
/// Test creation, copying, and destruction of a H1Lagrange basis
/// \test Test creation, copying, and destruction of a H1Lagrange basis
#include <ceed.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed      ceed;
  CeedBasis basis, basis_2;
  CeedInt   p = 4;

  CeedInit(argv[1], &ceed);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p, 4, CEED_GAUSS_LOBATTO, &basis);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p + 1, 4, CEED_GAUSS_LOBATTO, &basis_2);

  CeedBasisReferenceCopy(basis, &basis_2);  // This destroys the previous basis_2
  CeedBasisDestroy(&basis);

  CeedInt p_2;
  CeedBasisGetNumNodes1D(basis_2, &p_2);
  if (p != p_2) printf("Error copying CeedBasis reference\n");

  CeedBasisDestroy(&basis_2);
  CeedDestroy(&ceed);
  return 0;
}
