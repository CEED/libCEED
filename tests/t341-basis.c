/// @file
/// Test interpolation with a 2D Simplex non-tensor H(curl) basis
/// \test Test interpolation with a 2D Simplex non-tensor H(curl) basis
#include <ceed.h>
#include <math.h>

#include "t340-basis.h"

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    u, v;
  const CeedInt p = 8, q = 4, dim = 2;
  CeedBasis     basis;
  CeedScalar    q_ref[dim * q], q_weight[q];
  CeedScalar    interp[dim * p * q], curl[p * q];
  CeedScalar    row_sum[dim * q], column_sum[p];

  CeedInit(argv[1], &ceed);

  BuildHcurl2DSimplex(q_ref, q_weight, interp, curl);
  CeedBasisCreateHcurl(ceed, CEED_TOPOLOGY_TRIANGLE, 1, p, q, interp, curl, q_ref, q_weight, &basis);

  // Test interpolation for H(curl)
  {
    const CeedScalar *interp_in_basis;

    CeedBasisGetInterp(basis, &interp_in_basis);
    for (CeedInt i = 0; i < dim * p * q; i++) {
      if (fabs(interp[i] - interp_in_basis[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", interp[i], interp_in_basis[i]);
    }
  }

  for (int i = 0; i < dim * q; i++) {
    row_sum[i] = 0.0;
    for (int j = 0; j < p; j++) {
      row_sum[i] += interp[j + i * p];
    }
  }
  for (int i = 0; i < p; i++) {
    column_sum[i] = 0.0;
    for (int j = 0; j < dim * q; j++) {
      column_sum[i] += interp[i + j * p];
    }
  }

  CeedVectorCreate(ceed, p, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, dim * q, &v);
  CeedVectorSetValue(v, 0.0);

  CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);

  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < dim * q; i++) {
      if (fabs(row_sum[i] - v_array[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", row_sum[i], v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorSetValue(v, 1.0);
  CeedVectorSetValue(u, 0.0);

  CeedBasisApply(basis, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, v, u);

  {
    const CeedScalar *u_array;

    CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array);
    for (CeedInt i = 0; i < p; i++) {
      if (fabs(column_sum[i] - u_array[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", column_sum[i], u_array[i]);
    }
    CeedVectorRestoreArrayRead(u, &u_array);
  }

  CeedBasisDestroy(&basis);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
}
