/// @file
/// Test polynomial derivative interpolation in 1D
/// \test Test polynomial derivative interpolation in 1D
#include <ceed.h>
#include <math.h>

#define ALEN(a) (sizeof(a) / sizeof((a)[0]))

static CeedScalar Eval(CeedScalar x, CeedInt n, const CeedScalar *p) {
  CeedScalar y = p[n - 1];
  for (CeedInt i = n - 2; i >= 0; i--) y = y * x + p[i];
  return y;
}

int main(int argc, char **argv) {
  Ceed             ceed;
  CeedVector       x, x_q, u, u_q;
  CeedBasis        basis_x_lobatto, basis_u_lobatto, basis_x_gauss, basis_u_gauss;
  CeedInt          q     = 6;
  const CeedScalar p[6]  = {1, 2, 3, 4, 5, 6};  // 1 + 2x + 3x^2 + ...
  const CeedScalar dp[5] = {2, 6, 12, 20, 30};  // 2 + 6x + 12x^2 + ...

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, 2, &x);
  CeedVectorCreate(ceed, q, &x_q);
  CeedVectorSetValue(x_q, 0);
  CeedVectorCreate(ceed, q, &u);
  CeedVectorSetValue(u, 0);
  CeedVectorCreate(ceed, q, &u_q);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS_LOBATTO, &basis_x_lobatto);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, q, q, CEED_GAUSS_LOBATTO, &basis_u_lobatto);

  {
    CeedScalar x_array[2];

    for (int i = 0; i < 2; i++) x_array[i] = CeedIntPow(-1, i + 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedBasisApply(basis_x_lobatto, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, x_q);

  {
    const CeedScalar *x_q_array;
    CeedScalar        u_q_array[q];

    CeedVectorGetArrayRead(x_q, CEED_MEM_HOST, &x_q_array);
    for (CeedInt i = 0; i < q; i++) u_q_array[i] = Eval(x_q_array[i], ALEN(p), p);
    CeedVectorRestoreArrayRead(x_q, &x_q_array);
    CeedVectorSetArray(u_q, CEED_MEM_HOST, CEED_COPY_VALUES, u_q_array);
  }

  // This operation is the identity because the quadrature is collocated
  CeedBasisApply(basis_u_lobatto, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, u_q, u);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x_gauss);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, q, q, CEED_GAUSS, &basis_u_gauss);

  CeedBasisApply(basis_x_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, x, x_q);
  CeedBasisApply(basis_u_gauss, 1, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u, u_q);

  {
    const CeedScalar *x_q_array, *u_q_array;

    CeedVectorGetArrayRead(x_q, CEED_MEM_HOST, &x_q_array);
    CeedVectorGetArrayRead(u_q, CEED_MEM_HOST, &u_q_array);
    for (CeedInt i = 0; i < q; i++) {
      CeedScalar px = Eval(x_q_array[i], ALEN(dp), dp);
      if (fabs(u_q_array[i] - px) > 1000. * CEED_EPSILON) printf("%f != %f = p(%f)\n", u_q_array[i], px, x_q_array[i]);
    }
    CeedVectorRestoreArrayRead(x_q, &x_q_array);
    CeedVectorRestoreArrayRead(u_q, &u_q_array);
  }

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&x_q);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&u_q);
  CeedBasisDestroy(&basis_x_lobatto);
  CeedBasisDestroy(&basis_u_lobatto);
  CeedBasisDestroy(&basis_x_gauss);
  CeedBasisDestroy(&basis_u_gauss);
  CeedDestroy(&ceed);
  return 0;
}
