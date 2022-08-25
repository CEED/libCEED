/// @file
/// Test polynomial interpolation in 1D
/// \test Test polynomial interpolation in 1D
#include <ceed.h>
#include <math.h>

#define ALEN(a) (sizeof(a) / sizeof((a)[0]))

static CeedScalar PolyEval(CeedScalar x, CeedInt n, const CeedScalar *p) {
  CeedScalar y = p[n - 1];
  for (CeedInt i = n - 2; i >= 0; i--) y = y * x + p[i];
  return y;
}

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        X, X_q, U, U_q, W;
  CeedBasis         basis_x_lobatto, basis_x_gauss, basis_u_gauss;
  CeedInt           Q    = 6;
  const CeedScalar  p[6] = {1, 2, 3, 4, 5, 6};  // 1 + 2x + 3x^2 + ...
  const CeedScalar *xq, *uq, *w;
  CeedScalar        u[Q], x[2], sum, error, pint[ALEN(p) + 1];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 2, &X);
  CeedVectorCreate(ceed, Q, &X_q);
  CeedVectorSetValue(X_q, 0);
  CeedVectorCreate(ceed, Q, &U);
  CeedVectorCreate(ceed, Q, &U_q);
  CeedVectorSetValue(U_q, 0);
  CeedVectorCreate(ceed, Q, &W);
  CeedVectorSetValue(W, 0);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS_LOBATTO, &basis_x_lobatto);

  for (int i = 0; i < 2; i++) x[i] = CeedIntPow(-1, i + 1);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);

  CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &xq);
  for (CeedInt i = 0; i < Q; i++) u[i] = PolyEval(xq[i], ALEN(p), p);
  CeedVectorRestoreArrayRead(X_q, &xq);
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &basis_x_gauss);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Q, Q, CEED_GAUSS, &basis_u_gauss);

  CeedBasisApply(basis_x_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);
  CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U, U_q);
  CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, W);

  CeedVectorGetArrayRead(W, CEED_MEM_HOST, &w);
  CeedVectorGetArrayRead(U_q, CEED_MEM_HOST, &uq);
  sum = 0;
  for (CeedInt i = 0; i < Q; i++) sum += w[i] * uq[i];
  CeedVectorRestoreArrayRead(W, &w);
  CeedVectorRestoreArrayRead(U_q, &uq);

  pint[0] = 0;
  for (CeedInt i = 0; i < (int)ALEN(p); i++) pint[i + 1] = p[i] / (i + 1);
  error = sum - PolyEval(1, ALEN(pint), pint) + PolyEval(-1, ALEN(pint), pint);
  if (error > 100. * CEED_EPSILON) {
    // LCOV_EXCL_START
    printf("Error %e  sum %g  exact %g\n", error, sum, PolyEval(1, ALEN(pint), pint) - PolyEval(-1, ALEN(pint), pint));
    // LCOV_EXCL_STOP
  }

  CeedVectorDestroy(&X);
  CeedVectorDestroy(&X_q);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&U_q);
  CeedVectorDestroy(&W);
  CeedBasisDestroy(&basis_x_lobatto);
  CeedBasisDestroy(&basis_x_gauss);
  CeedBasisDestroy(&basis_u_gauss);
  CeedDestroy(&ceed);
  return 0;
}
