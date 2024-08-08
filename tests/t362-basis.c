/// @file
/// Test integration ApplyAdd with a 2D Simplex non-tensor H^1 basis
/// \test Test integration ApplyAdd with a 2D Simplex non-tensor H^1 basis
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#include "t320-basis.h"

// main test
int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    u, v, u_q, v_q, w_q;
  const CeedInt p = 6, q = 4, dim = 2;
  CeedBasis     basis;
  CeedScalar    q_ref[dim * q], q_weight[q];
  CeedScalar    interp[p * q], grad[dim * p * q];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, p, &u);
  CeedVectorCreate(ceed, p, &v);
  CeedVectorSetValue(u, 1.0);
  CeedVectorSetValue(v, 0.0);
  CeedVectorCreate(ceed, q, &u_q);
  CeedVectorCreate(ceed, q, &v_q);
  CeedVectorCreate(ceed, q, &w_q);

  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, p, q, interp, grad, q_ref, q_weight, &basis);

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
    for (CeedInt i = 0; i < p; i++) area += v_array[i];
    if (fabs(area - 1.0) > 1E-6) printf("Incorrect area computed %f != %f\n", area, 1.0);
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&u_q);
  CeedVectorDestroy(&v_q);
  CeedVectorDestroy(&w_q);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
