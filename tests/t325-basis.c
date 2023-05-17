/// @file
/// Test grad transpose with a 2D Simplex non-tensor H^1 basis
/// \test Test grad transpose with a 2D Simplex non-tensor H^1 basis
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#include "t320-basis.h"

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    u, v;
  const CeedInt p = 6, q = 4, dim = 2, num_comp = 3;
  CeedBasis     basis;
  CeedScalar    q_ref[dim * q], q_weight[q];
  CeedScalar    interp[p * q], grad[dim * p * q];
  CeedScalar    column_sum[p];

  CeedInit(argv[1], &ceed);

  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, num_comp, p, q, interp, grad, q_ref, q_weight, &basis);

  CeedVectorCreate(ceed, q * dim * num_comp, &u);
  {
    CeedScalar *u_array;

    CeedVectorGetArrayWrite(u, CEED_MEM_HOST, &u_array);
    for (int d = 0; d < dim; d++) {
      for (int i = 0; i < num_comp; i++) {
        for (int j = 0; j < q; j++) u_array[j + (i + d * num_comp) * q] = i * 1.0;
      }
    }
    CeedVectorRestoreArray(u, &u_array);
  }
  CeedVectorCreate(ceed, p * num_comp, &v);
  CeedVectorSetValue(v, 0);

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
      for (int j = 0; j < num_comp; j++) {
        if (fabs(j * column_sum[i] - v_array[i + j * p]) > 100. * CEED_EPSILON) {
          // LCOV_EXCL_START
          printf("[%" CeedInt_FMT "] %f != %f\n", i, v_array[i + j * p], j * column_sum[i]);
          // LCOV_EXCL_STOP
        }
      }
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
