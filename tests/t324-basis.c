/// @file
/// Test grad transpose with a 2D Simplex non-tensor H1 basis
/// \test Test grad transpose with a 2D Simplex non-tensor H1 basis
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#include "t320-basis.h"

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    u, v;
  const CeedInt p = 6, q = 4, dim = 2;
  CeedBasis     basis;
  CeedScalar    q_ref[dim * q], q_weight[q];
  CeedScalar    interp[p * q], grad[dim * p * q];
  CeedScalar    column_sum[p];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, q * dim, &u);
  CeedVectorSetValue(u, 1);
  CeedVectorCreate(ceed, p, &v);
  CeedVectorSetValue(v, 0);

  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, p, q, interp, grad, q_ref, q_weight, &basis);

  CeedBasisApply(basis, 1, CEED_TRANSPOSE, CEED_EVAL_GRAD, u, v);

  // Check values at quadrature points
  for (int i = 0; i < p; i++) {
    column_sum[i] = 0;
    for (int j = 0; j < q * dim; j++) {
      column_sum[i] += grad[i + j * p];
    }
  }
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (int i = 0; i < p; i++) {
      if (fabs(column_sum[i] - v_array[i]) > 100. * CEED_EPSILON) printf("[%" CeedInt_FMT "] %f != %f\n", i, v_array[i], column_sum[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
