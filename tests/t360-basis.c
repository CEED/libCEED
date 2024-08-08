/// @file
/// Test interpolation ApplyAdd in multiple dimensions
/// \test Test interpolation ApplyAdd in multiple dimensions
#include <ceed.h>
#include <math.h>
#include <stdio.h>

int main(int argc, char **argv) {
  Ceed ceed;

  CeedInit(argv[1], &ceed);

  for (CeedInt dim = 1; dim <= 3; dim++) {
    CeedVector u, u_q, v, v_q, w_q;
    CeedBasis  basis;
    CeedInt    p = 4, q = 5, p_dim = CeedIntPow(p, dim), q_dim = CeedIntPow(q, dim);

    CeedVectorCreate(ceed, p_dim, &u);
    CeedVectorCreate(ceed, p_dim, &v);
    CeedVectorSetValue(u, 1.0);
    CeedVectorSetValue(v, 0.0);
    CeedVectorCreate(ceed, q_dim, &u_q);
    CeedVectorCreate(ceed, q_dim, &v_q);
    CeedVectorCreate(ceed, q_dim, &w_q);

    CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis);

    // Compute area
    CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, u_q);
    CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_WEIGHT, CEED_VECTOR_NONE, w_q);
    CeedVectorPointwiseMult(v_q, u_q, w_q);
    CeedBasisApply(basis, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, v_q, v);
    // Double area computed
    CeedBasisApplyAdd(basis, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, v_q, v);

    // Check area
    {
      const CeedScalar *v_array;
      CeedScalar        area = 0.0;

      CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
      for (CeedInt i = 0; i < p_dim; i++) area += v_array[i];
      if (fabs(area - 2.0 * CeedIntPow(2, dim)) > 5E-6) printf("Incorrect area computed %f != %f\n", area, 2.0 * CeedIntPow(2, dim));
      CeedVectorRestoreArrayRead(v, &v_array);
    }

    CeedVectorDestroy(&u);
    CeedVectorDestroy(&v);
    CeedVectorDestroy(&u_q);
    CeedVectorDestroy(&v_q);
    CeedVectorDestroy(&w_q);
    CeedBasisDestroy(&basis);
  }
  CeedDestroy(&ceed);
  return 0;
}
