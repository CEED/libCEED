#include <feme.h>

int main(int argc, char **argv) {
  Feme feme;
  FemeBasis b;
  int i, dim = 1, P1d = 3, Q1d = 4;
  FemeScalar u[dim*Q1d],v[dim*Q1d];

  FemeInit("/cpu/self", &feme);
  for (i = 0; i < dim*Q1d; i++) {
    u[i] = 1.0;
  }
  FemeBasisCreateTensorH1Lagrange(feme, dim, P1d, Q1d, FEME_GAUSS_LOBATTO, &b);
  FemeBasisApply(b, FEME_NOTRANSPOSE, FEME_EVAL_INTERP, u, v);
  for (i = 0; i < dim*Q1d; i++) {
    printf("\t% 12.8f", v[i]);
  }
  printf("\t");

  FemeBasisDestroy(&b);
  FemeDestroy(&feme);
  return 0;
}
