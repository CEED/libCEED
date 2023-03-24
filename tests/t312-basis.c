/// @file
/// Test polynomial interpolation in 1D
/// \test Test polynomial interpolation in 1D
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#define ALEN(a) (sizeof(a) / sizeof((a)[0]))

static CeedScalar Eval(CeedScalar x, CeedInt n, const CeedScalar *p) {
  CeedScalar y = p[n - 1];
  for (CeedInt i = n - 2; i >= 0; i--) y = y * x + p[i];
  return y;
}

int main(int argc, char **argv) {
  Ceed             ceed;
  CeedVector       x, x_q, u, u_q, w;
  CeedBasis        basis_x_lobatto, basis_x_gauss, basis_u_gauss;
  CeedInt          q    = 6;
  const CeedScalar p[6] = {1, 2, 3, 4, 5, 6};  // 1 + 2x + 3x^2 + ...
  CeedScalar       sum, error, pint[ALEN(p) + 1];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 2, &x);
  CeedVectorCreate(ceed, q, &x_q);
  CeedVectorSetValue(x_q, 0);
  CeedVectorCreate(ceed, q, &u);
  CeedVectorCreate(ceed, q, &u_q);
  CeedVectorSetValue(u_q, 0);
  CeedVectorCreate(ceed, q, &w);
  CeedVectorSetValue(w, 0);

  {
    CeedScalar x_array[2];

    for (int i = 0; i < 2; i++) x_array[i] = CeedIntPow(-1, i + 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS_LOBATTO, &basis_x_lobatto);

  CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, x_q);
  {
    const CeedScalar *x_q_array;
    CeedScalar        u_array[q];

    CeedVectorGetArrayRead(x_q, CEED_MEM_HOST, &x_q_array);
    for (CeedInt i = 0; i < q; i++) u_array[i] = Eval(x_q_array[i], ALEN(p), p);
    CeedVectorRestoreArrayRead(x_q, &x_q_array);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x_gauss);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, q, q, CEED_GAUSS, &basis_u_gauss);

  CeedBasisApply(basis_x_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, x_q);
  CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, u_q);
  CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, w);

  {
    const CeedScalar *w_array, *u_q_array;

    CeedVectorGetArrayRead(w, CEED_MEM_HOST, &w_array);
    CeedVectorGetArrayRead(u_q, CEED_MEM_HOST, &u_q_array);
    sum = 0;
    for (CeedInt i = 0; i < q; i++) sum += w_array[i] * u_q_array[i];
    CeedVectorRestoreArrayRead(w, &w_array);
    CeedVectorRestoreArrayRead(u_q, &u_q_array);
  }

  pint[0] = 0;
  for (CeedInt i = 0; i < (int)ALEN(p); i++) pint[i + 1] = p[i] / (i + 1);
  error = sum - Eval(1, ALEN(pint), pint) + Eval(-1, ALEN(pint), pint);
  if (error > 100. * CEED_EPSILON) {
    // LCOV_EXCL_START
    printf("Error %e  sum %g  exact %g\n", error, sum, Eval(1, ALEN(pint), pint) - Eval(-1, ALEN(pint), pint));
    // LCOV_EXCL_STOP
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&x_q);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&u_q);
  CeedVectorDestroy(&w);
  CeedBasisDestroy(&basis_x_lobatto);
  CeedBasisDestroy(&basis_x_gauss);
  CeedBasisDestroy(&basis_u_gauss);
  CeedDestroy(&ceed);
  return 0;
}
