/// @file
/// Test interpolation in multiple dimensions
/// \test Test interpolation in multiple dimensions
#include <ceed.h>
#include <math.h>

static CeedScalar Eval(CeedInt dim, const CeedScalar x[]) {
  CeedScalar result = 1, center = 0.1;
  for (CeedInt d = 0; d < dim; d++) {
    result *= tanh(x[d] - center);
    center += 0.1;
  }
  return result;
}

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);
  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedVector        X, X_q, U, U_q;
    CeedBasis         basis_x_lobatto, basis_u_lobatto, basis_x_gauss, basis_u_gauss;
    CeedInt           Q = 10, Q_dim = CeedIntPow(Q, dim), X_dim = CeedIntPow(2, dim);
    CeedScalar        x[X_dim * dim];
    const CeedScalar *xq, *u;
    CeedScalar        uq[Q_dim];

    for (CeedInt d = 0; d < dim; d++)
      for (CeedInt i = 0; i < X_dim; i++) x[d * X_dim + i] = (i % CeedIntPow(2, dim - d)) / CeedIntPow(2, dim - d - 1) ? 1 : -1;

    CeedVectorCreate(ceed, X_dim * dim, &X);
    CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
    CeedVectorCreate(ceed, Q_dim * dim, &X_q);
    CeedVectorSetValue(X_q, 0);
    CeedVectorCreate(ceed, Q_dim, &U);
    CeedVectorSetValue(U, 0);
    CeedVectorCreate(ceed, Q_dim, &U_q);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS_LOBATTO, &basis_x_lobatto);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, Q, Q, CEED_GAUSS_LOBATTO, &basis_u_lobatto);

    CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);

    CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &xq);
    for (CeedInt i = 0; i < Q_dim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d = 0; d < dim; d++) xx[d] = xq[d * Q_dim + i];
      uq[i] = Eval(dim, xx);
    }
    CeedVectorRestoreArrayRead(X_q, &xq);
    CeedVectorSetArray(U_q, CEED_MEM_HOST, CEED_USE_POINTER, uq);

    // This operation is the identity because the quadrature is collocated
    CeedBasisApply(basis_u_lobatto, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, U_q, U);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, 2, Q, CEED_GAUSS, &basis_x_gauss);
    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, Q, Q, CEED_GAUSS, &basis_u_gauss);

    CeedBasisApply(basis_x_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, X_q);
    CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U, U_q);

    CeedVectorGetArrayRead(X_q, CEED_MEM_HOST, &xq);
    CeedVectorGetArrayRead(U_q, CEED_MEM_HOST, &u);
    for (CeedInt i = 0; i < Q_dim; i++) {
      CeedScalar xx[dim];
      for (CeedInt d = 0; d < dim; d++) xx[d] = xq[d * Q_dim + i];
      CeedScalar fx = Eval(dim, xx);
      if (fabs(u[i] - fx) > 1E-4) {
        // LCOV_EXCL_START
        printf("[%" CeedInt_FMT "] %f != %f=f(%f", dim, u[i], fx, xx[0]);
        for (CeedInt d = 1; d < dim; d++) printf(",%f", xx[d]);
        puts(")");
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(X_q, &xq);
    CeedVectorRestoreArrayRead(U_q, &u);

    CeedVectorDestroy(&X);
    CeedVectorDestroy(&X_q);
    CeedVectorDestroy(&U);
    CeedVectorDestroy(&U_q);
    CeedBasisDestroy(&basis_x_lobatto);
    CeedBasisDestroy(&basis_u_lobatto);
    CeedBasisDestroy(&basis_x_gauss);
    CeedBasisDestroy(&basis_u_gauss);
  }
  CeedDestroy(&ceed);
  return 0;
}
