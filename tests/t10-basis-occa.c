#include <ceed.h>

int main(int argc, char** argv) {
  Ceed ceed;
  CeedBasis b;

  CeedInit("/cpu/occa", &ceed);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 3, 4, CEED_GAUSS_LOBATTO, &b);
  CeedBasisView(b, stdout);
  CeedBasisDestroy(&b);
  CeedBasisCreateTensorH1Lagrange(ceed, 1,  1, 3, 4, CEED_GAUSS, &b);
  CeedBasisView(b, stdout);
  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
