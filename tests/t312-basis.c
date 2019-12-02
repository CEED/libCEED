/// @file
/// Test polynomial interpolation in 1D
/// \test Test polynomial interpolation in 1D
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
  CeedVector X, Xq, U, Uq, W;
  CeedBasis bxl, bxg, bug;
  CeedInt Q = 6;
  const CeedScalar p[6] = {1, 2, 3, 4, 5, 6}; // 1 + 2x + 3x^2 + ...
  const CeedScalar *xq, *uq, *w;
  CeedScalar u[Q], x[2], sum, error, pint[ALEN(p)+1];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 2, &X);
  CeedVectorCreate(ceed, Q, &Xq);
  CeedVectorSetValue(Xq, 0);
  CeedVectorCreate(ceed, Q, &U);
  CeedVectorCreate(ceed, Q, &Uq);
  CeedVectorSetValue(Uq, 0);
  CeedVectorCreate(ceed, Q, &W);
  CeedVectorSetValue(W, 0);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS_LOBATTO, &bxl);

  for (int i = 0; i < 2; i++)
    x[i] = CeedIntPow(-1, i+1);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  CeedBasisApply(bxl, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, Xq);

  CeedVectorGetArrayRead(Xq, CEED_MEM_HOST, &xq);
  for (CeedInt i=0; i<Q; i++)
    u[i] = PolyEval(xq[i], ALEN(p), p);
  CeedVectorRestoreArrayRead(Xq, &xq);
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bxg);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Q, Q, CEED_GAUSS, &bug);

  CeedBasisApply(bxg, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, Xq);
  CeedBasisApply(bug, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U, Uq);
  CeedBasisApply(bug, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT,
                 CEED_VECTOR_NONE, W);

  CeedVectorGetArrayRead(W, CEED_MEM_HOST, &w);
  CeedVectorGetArrayRead(Uq, CEED_MEM_HOST, &uq);
  sum = 0;
  for (CeedInt i=0; i<Q; i++)
    sum += w[i] * uq[i];
  CeedVectorRestoreArrayRead(W, &w);
  CeedVectorRestoreArrayRead(Uq, &uq);

  pint[0] = 0;
  for (CeedInt i=0; i<(int)ALEN(p); i++)
    pint[i+1] = p[i] / (i+1);
  error = sum - PolyEval(1, ALEN(pint), pint) + PolyEval(-1, ALEN(pint), pint);
  if (error > 1.E-10)
    // LCOV_EXCL_START
    printf("Error %e  sum %g  exact %g\n", error, sum,
           PolyEval(1, ALEN(pint), pint) - PolyEval(-1, ALEN(pint), pint));
  // LCOV_EXCL_STOP

  CeedVectorDestroy(&X);
  CeedVectorDestroy(&Xq);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&Uq);
  CeedVectorDestroy(&W);
  CeedBasisDestroy(&bxl);
  CeedBasisDestroy(&bxg);
  CeedBasisDestroy(&bug);
  CeedDestroy(&ceed);
  return 0;
}
