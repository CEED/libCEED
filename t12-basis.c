#include <feme.h>
#include <math.h>

#define ALEN(a) (sizeof(a) / sizeof((a)[0]))

static double PolyEval(FemeScalar x, FemeInt n, const FemeScalar *p) {
  FemeScalar y = p[n-1];
  for (FemeInt i=n-2; i>=0; i--) y = y*x + p[i];
  return y;
}

int main(int argc, char **argv) {
  Feme feme;
  FemeBasis bxl, bul, bxg, bug;
  FemeInt Q = 6;
  const FemeScalar p[] = {1, 2, 3, 4, 5, 6}; // 1 + 2x + 3x^2 + ...
  const FemeScalar x[] = {-1, 1};
  FemeScalar xq[Q], uq[Q], u[Q];

  FemeInit("/cpu/self", &feme);
  FemeBasisCreateTensorH1Lagrange(feme, 1, 1, Q, FEME_GAUSS_LOBATTO, &bxl);
  FemeBasisCreateTensorH1Lagrange(feme, 1, Q-1, Q, FEME_GAUSS_LOBATTO, &bul);
  FemeBasisApply(bxl, FEME_NOTRANSPOSE, FEME_EVAL_INTERP, x, xq);
  for (FemeInt i=0; i<Q; i++) uq[i] = PolyEval(xq[i], ALEN(p), p);

  FemeBasisApply(bul, FEME_TRANSPOSE, FEME_EVAL_INTERP, uq, u); // Should be identity

  FemeBasisCreateTensorH1Lagrange(feme, 1, 1, Q, FEME_GAUSS, &bxg);
  FemeBasisCreateTensorH1Lagrange(feme, 1, Q-1, Q, FEME_GAUSS, &bug);
  FemeBasisApply(bxg, FEME_NOTRANSPOSE, FEME_EVAL_INTERP, x, xq);
  FemeBasisApply(bug, FEME_NOTRANSPOSE, FEME_EVAL_INTERP, u, uq);
  for (FemeInt i=0; i<Q; i++) {
    FemeScalar px = PolyEval(xq[i], ALEN(p), p);
    if (fabs(uq[i] - px) > 1e-14) {
      printf("%f != %f=p(%f)\n", uq[i], px, xq[i]);
    }
  }

  FemeBasisDestroy(&bxl);
  FemeBasisDestroy(&bul);
  FemeBasisDestroy(&bxg);
  FemeBasisDestroy(&bug);
  FemeDestroy(&feme);
  return 0;
}
