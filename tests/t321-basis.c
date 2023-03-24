/// @file
/// Test interpolation with a 2D Simplex non-tensor H1 basis
/// \test Test interpolation with a 2D Simplex non-tensor H1 basis
#include <ceed.h>
#include <math.h>
#include <stdio.h>

#include "t320-basis.h"

// polynomial eval helper
static CeedScalar Eval(CeedScalar x1, CeedScalar x2) { return x1 * x1 + x2 * x2 + x1 * x2 + 1; }

// main test
int main(int argc, char **argv) {
  Ceed          ceed;
  CeedVector    u, v;
  const CeedInt p = 6, q = 4, dim = 2;
  CeedBasis     basis;
  CeedScalar    q_ref[dim * q], q_weight[q];
  CeedScalar    interp[p * q], grad[dim * p * q];
  CeedScalar    x_q[]   = {0.2, 0.6, 1. / 3., 0.2, 0.2, 0.2, 1. / 3., 0.6};
  CeedScalar    x_ref[] = {0., 0.5, 1., 0., 0.5, 0., 0., 0., 0., 0.5, 0.5, 1.};

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, p, &u);
  {
    CeedScalar u_array[p];

    // Interpolate function to quadrature points
    for (int i = 0; i < p; i++) u_array[i] = Eval(x_ref[0 * p + i], x_ref[1 * p + i]);
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_COPY_VALUES, u_array);
  }
  CeedVectorCreate(ceed, q, &v);
  CeedVectorSetValue(v, 0);

  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, p, q, interp, grad, q_ref, q_weight, &basis);

  CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);

  // Check values at quadrature points
  {
    const CeedScalar *v_array;

    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    for (int i = 0; i < q; i++) {
      CeedScalar fx = Eval(x_q[0 * q + i], x_q[1 * q + i]);
      if (fabs(v_array[i] - fx) > 100. * CEED_EPSILON) printf("[%" CeedInt_FMT "] %f != %f\n", i, v_array[i], fx);
    }
    CeedVectorRestoreArrayRead(v, &v_array);
  }

  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
