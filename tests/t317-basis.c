/// @file
/// Test polynomial derivative interpolation in 1D
/// \test Test polynomial derivative interpolation in 1D
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
  CeedVector        X, X_q, U, U_q;
  CeedBasis         basis_x_lobatto, basis_u_lobatto, basis_x_gauss, basis_u_gauss;
  CeedInt           Q     = 6;
  const CeedScalar  p[6]  = {1, 2, 3, 4, 5, 6};  // 1 + 2x + 3x^2 + ...
  const CeedScalar  dp[5] = {2, 6, 12, 20, 30};  // 2 + 6x + 12x^2 + ...
  const CeedScalar *xq, *uuq;
  CeedScalar        x[2], uq[Q];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 2, &X);
  CeedVectorCreate(ceed, Q, &X_q);
  CeedVectorSetValue(X_q, 0);
  CeedVectorCreate(ceed, Q, &U);
  CeedVectorSetValue(U, 0);
  CeedVectorCreate(ceed, Q, &U_q);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS_LOBATTO, &basis_x_lobatto);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Q, Q, CEED_GAUSS_LOBATTO, &basis_u_lobatto);

  for (int i = 0; i < 2; i++) x[i] = CeedIntPow(-1, i + 1);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)&x);

  CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);

  CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &xq);
  for (CeedInt i = 0; i < Q; i++) uq[i] = PolyEval(xq[i], ALEN(p), p);
  CeedVectorRestoreArrayRead(X_q, &xq);
  CeedVectorSetArray(U_q, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)&uq);

  // This operation is the identity because the quadrature is collocated
  CeedBasisApply(basis_u_lobatto, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, U_q, U);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &basis_x_gauss);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, Q, Q, CEED_GAUSS, &basis_u_gauss);

  CeedBasisApply(basis_x_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);
  CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, U, U_q);

  CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &xq);
  CeedVectorGetArrayRead(U_q, CEED_MEM_HOST, &uuq);
  for (CeedInt i = 0; i < Q; i++) {
    CeedScalar px = PolyEval(xq[i], ALEN(dp), dp);
    if (fabs(uuq[i] - px) > 1000. * CEED_EPSILON) printf("%f != %f = p(%f)\n", uuq[i], px, xq[i]);
  }
  CeedVectorRestoreArrayRead(X_q, &xq);
  CeedVectorRestoreArrayRead(U_q, &uuq);

  CeedVectorDestroy(&X);
  CeedVectorDestroy(&X_q);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&U_q);
  CeedBasisDestroy(&basis_x_lobatto);
  CeedBasisDestroy(&basis_u_lobatto);
  CeedBasisDestroy(&basis_x_gauss);
  CeedBasisDestroy(&basis_u_gauss);
  CeedDestroy(&ceed);
  return 0;
}
