// Test polynomial interpolation in 1D
#include <ceed.h>
#include <math.h>

#define ALEN(a) (sizeof(a) / sizeof((a)[0]))

static CeedScalar PolyEval(CeedScalar x, CeedInt n, const CeedScalar *p) {
  CeedScalar y = p[n-1];
  for (CeedInt i=n-2; i>=0; i--) y = y*x + p[i];
  return y;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedBasis bxl, bxg, bug;
  CeedInt Q = 6;
  const CeedScalar p[] = {1, 2, 3, 4, 5, 6}; // 1 + 2x + 3x^2 + ...
  const CeedScalar x[] = {-1, 1};
  CeedScalar xq[Q], u[Q], uq[Q], w[Q], sum, error, pint[ALEN(p)+1];

  assert(argv[1]);
  CeedInit(argv[1], &ceed);
  CeedBasisCreateTensorH1Lagrange(ceed, 1,  1, 2, Q, CEED_GAUSS_LOBATTO, &bxl);
  CeedBasisApply(bxl, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, xq);
  for (CeedInt i=0; i<Q; i++) u[i] = PolyEval(xq[i], ALEN(p), p);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bxg);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Q, Q, CEED_GAUSS, &bug);
  CeedBasisApply(bxg, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, xq);
  CeedBasisApply(bug, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, uq);
  CeedBasisApply(bug, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, NULL, w);
  sum = 0;
  for (CeedInt i=0; i<Q; i++) {
    sum += w[i] * uq[i];
  }
  pint[0] = 0;
  for (CeedInt i=0; i<(CeedInt)ALEN(p); i++) pint[i+1] = p[i] / (i+1);
  error = sum - PolyEval(1, ALEN(pint), pint) + PolyEval(-1, ALEN(pint), pint);
  if (!(error < 1e-10))
    printf("Error %e  sum %g  exact %g\n", error, sum,
           PolyEval(1, ALEN(pint), pint) - PolyEval(-1, ALEN(pint), pint));

  CeedBasisDestroy(&bxl);
  CeedBasisDestroy(&bxg);
  CeedBasisDestroy(&bug);
  CeedDestroy(&ceed);
  return 0;
}
