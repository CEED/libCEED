/// @file
/// Test square Gauss Lobatto interp_1d is identity
/// \test Test square Gauss Lobatto interp_1d is identity
#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector u, v;
  int        i, dim = 2, p = 4, q = 4, len = (int)(pow((CeedScalar)(q), dim) + 0.4);

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &u);
  CeedVectorCreate(ceed, len, &v);

  {
    CeedScalar u_array[len];

    for (i = 0; i < len; i++) u_array[i] = 1.0;
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }

  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS_LOBATTO, &basis);

  CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);

  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (i = 0; i < len; i++) {
      if (fabs(v_array[i] - 1.) > 10. * CEED_EPSILON) printf("v[%" CeedInt_FMT "] = %f != 1.\n", i, v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedBasisDestroy(&basis);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
}
