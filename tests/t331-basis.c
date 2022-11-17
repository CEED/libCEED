/// @file
/// Test GetInterp and BasisApply for a 2D Quad non-tensor H(div) basis
/// \test Test GetInterp and BasisApply for a 2D Quad non-tensor H(div) basis
#include <ceed.h>
#include <math.h>

#include "t330-basis.h"

int main(int argc, char **argv) {
  Ceed              ceed;
  const CeedInt     num_nodes = 4, Q = 3, dim = 2, num_qpts = Q * Q;
  CeedInt           num_comp = 1;                // one vector componenet
  CeedInt           P        = dim * num_nodes;  // dof per element!
  CeedBasis         b;
  CeedScalar        q_ref[dim * num_qpts], q_weights[num_qpts];
  CeedScalar        div[P * num_qpts], interp[P * dim * num_qpts];
  CeedVector        X, Y;
  const CeedScalar *y, *x;

  CeedInit(argv[1], &ceed);

  HdivBasisQuad(Q, q_ref, q_weights, interp, div, CEED_GAUSS);
  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, num_comp, P, num_qpts, interp, div, q_ref, q_weights, &b);

  // Test GetInterp for H(div)
  const CeedScalar *interp2;
  CeedBasisGetInterp(b, &interp2);
  for (CeedInt i = 0; i < P * dim * num_qpts; i++) {
    if (fabs(interp[i] - interp2[i]) > 100. * CEED_EPSILON) {
      // LCOV_EXCL_START
      printf("%f != %f\n", interp[i], interp2[i]);
      // LCOV_EXCL_STOP
    }
  }

  CeedVectorCreate(ceed, P, &X);
  CeedVectorSetValue(X, 1.0);
  CeedVectorCreate(ceed, num_qpts * dim, &Y);
  CeedVectorSetValue(Y, 0.);
  // BasisApply for H(div): CEED_EVAL_INTERP, NOTRANSPOSE case
  CeedBasisApply(b, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, X, Y);

  CeedVectorGetArrayRead(Y, CEED_MEM_HOST, &y);
  for (CeedInt i = 0; i < dim * num_qpts; i++) {
    if (fabs(q_ref[i] - y[i]) > 100. * CEED_EPSILON) {
      // LCOV_EXCL_START
      printf("%f != %f\n", q_ref[i], y[i]);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(Y, &y);

  CeedVectorSetValue(Y, 1.0);
  CeedVectorSetValue(X, 0.0);
  // BasisApply for Hdiv: CEED_EVAL_INTERP, TRANSPOSE case
  CeedBasisApply(b, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, Y, X);

  CeedVectorGetArrayRead(X, CEED_MEM_HOST, &x);
  CeedScalar sum = 0.;
  for (CeedInt i = 0; i < P; i++) {
    sum += x[i];
  }
  if (fabs(sum) > 100. * CEED_EPSILON) printf("sum of array %f != %f\n", sum, 0.0);
  CeedVectorRestoreArrayRead(X, &x);

  CeedBasisDestroy(&b);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&Y);
  CeedDestroy(&ceed);
  return 0;
}
