// Test polynomial interpolation in 1D
#include <ceed.h>
#include <math.h>

#define ALEN(a) (sizeof(a) / sizeof((a)[0]))

static CeedScalar PolyEval(CeedScalar x, CeedInt n, const CeedScalar *p) {
  CeedScalar y = p[n-1];
  for (CeedInt i=n-2; i>=0; i--) { y = y*x + p[i]; }
  return y;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedBasis bxl, bul, bxg, bug;
  CeedInt Q = 6;
  const CeedScalar p[] = {1, 2, 3, 4, 5, 6}; // 1 + 2x + 3x^2 + ...
  const CeedScalar x[] = {-1, 1};
  CeedScalar xq[Q], uq[Q], u[Q];

  CeedInit("/cpu/self", &ceed);
  CeedBasisCreateTensorH1Lagrange(ceed, 1,  1, 1, Q, CEED_GAUSS_LOBATTO, &bxl);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Q-1, Q, CEED_GAUSS_LOBATTO, &bul);
  CeedBasisApply(bxl, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, xq);
  for (CeedInt i=0; i<Q; i++) { uq[i] = PolyEval(xq[i], ALEN(p), p); }

  CeedBasisApply(bul, CEED_TRANSPOSE, CEED_EVAL_INTERP, uq,
                 u); // Should be identity

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 1, Q, CEED_GAUSS, &bxg);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Q-1, Q, CEED_GAUSS, &bug);
  CeedBasisApply(bxg, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, xq);
  CeedBasisApply(bug, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, uq);
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar px = PolyEval(xq[i], ALEN(p), p);
    if (fabs(uq[i] - px) > 1e-14) {
      printf("%f != %f=p(%f)\n", uq[i], px, xq[i]);
    }
  }

  CeedBasisDestroy(&bxl);
  CeedBasisDestroy(&bul);
  CeedBasisDestroy(&bxg);
  CeedBasisDestroy(&bug);
  CeedDestroy(&ceed);
  return 0;
}
