/// @file
/// Test curl with a 2D Simplex non-tensor H(curl) basis
/// \test Test curl with a 2D Simplex non-tensor H(curl) basis
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

  // Test curl for H(curl)
  {
    const CeedScalar *curl_in_basis;

    CeedBasisGetCurl(basis, &curl_in_basis);
    for (CeedInt i = 0; i < p * q; i++) {
      if (fabs(curl[i] - curl_in_basis[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", curl[i], curl_in_basis[i]);
    }
  }

  for (int i = 0; i < q; i++) {
    row_sum[i] = 0;
    for (int j = 0; j < p; j++) {
      row_sum[i] += curl[j + i * p];
    }
  }
  for (int i = 0; i < p; i++) {
    column_sum[i] = 0;
    for (int j = 0; j < q; j++) {
      column_sum[i] += curl[i + j * p];
    }
  }

  CeedVectorCreate(ceed, p, &u);
  CeedVectorSetValue(u, 1.0);
  CeedVectorCreate(ceed, q, &v);
  CeedVectorSetValue(v, 0.0);

  CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_CURL, u, v);

  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (CeedInt i = 0; i < q; i++) {
      if (fabs(row_sum[i] - v_array[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", row_sum[i], v_array[i]);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorSetValue(v, 1.0);
  CeedVectorSetValue(u, 0.0);

  CeedBasisApply(basis, 1, CEED_TRANSPOSE, CEED_EVAL_CURL, v, u);

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
