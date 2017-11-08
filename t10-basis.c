#include <feme.h>

int main(int argc, char **argv) {
  Feme feme;
  FemeBasis b;

  FemeInit("/cpu/self", &feme);
  FemeBasisCreateTensorH1Lagrange(feme, 1, 2, 3, FEME_GAUSS_LOBATTO, &b);
  FemeBasisView(b, stdout);
  FemeBasisDestroy(&b);
  FemeDestroy(&feme);
  return 0;
}
