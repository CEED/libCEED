/// @file
/// Test square Gauss Lobatto interp_1d is identity
/// \test Test square Gauss Lobatto interp_1d is identity
#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedBasis         b;
  CeedVector        U, V;
  int               i, dim = 2, P_1d = 4, Q_1d = 4, len = (int)(pow((CeedScalar)(Q_1d), dim) + 0.4);
  CeedScalar        u[len];
  const CeedScalar *v;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &U);
  CeedVectorCreate(ceed, len, &V);

  for (i = 0; i < len; i++) u[i] = 1.0;
  CeedVectorSetArray(U, CEED_MEM_HOST, CEED_USE_POINTER, u);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, P_1d, Q_1d, CEED_GAUSS_LOBATTO, &b);

  CeedBasisApply(b, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U, V);

  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &v);
  for (i = 0; i < len; i++) {
    if (fabs(v[i] - 1.) > 10. * CEED_EPSILON) printf("v[%" CeedInt_FMT "] = %f != 1.\n", i, v[i]);
  }
  CeedVectorRestoreArrayRead(V, &v);

  CeedBasisDestroy(&b);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
