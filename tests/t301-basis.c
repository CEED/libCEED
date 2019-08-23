/// @file
/// Test square Gauss Lobatto interp1d is identity
/// \test Test square Gauss Lobatto interp1d is identity
#include <ceed.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedBasis b;
  CeedVector U, V;
  int i, dim = 2, P1d = 4, Q1d = 4, len = (int)(pow((double)(Q1d), dim) + 0.4);
  CeedScalar u[len];
  const CeedScalar *v;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &U);
  CeedVectorCreate(ceed, len, &V);

  for (i = 0; i < len; i++) {
    u[i] = 1.0;
  }
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, (CeedScalar *)&u);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P1d, Q1d, CEED_GAUSS_LOBATTO, &b);

  CeedBasisApply(b, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U, V);

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
  for (i = 0; i < len; i++)
    if (fabs(v[i] - 1.) > 1e-15)
      // LCOV_EXCL_START
      printf("v[%d] = %f != 1.\n", i, v[i]);
      // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(V, &v);

  CeedBasisDestroy(&b);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
