/// @file
/// Test GetDiv and BasisApply for a 2D Quad non-tensor H(div) basis
/// \test Test GetDiv and BasisApply for a 2D Quad non-tensor H(div) basis
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
  // Test GetDiv
  const CeedScalar *div2;
  CeedBasisGetDiv(b, &div2);
  for (CeedInt i = 0; i < P * num_qpts; i++) {
    if (fabs(div[i] - div2[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", div[i], div2[i]);
  }
  CeedVectorCreate(ceed, P, &X);
  CeedVectorSetValue(X, 1);
  CeedVectorCreate(ceed, num_qpts, &Y);
  CeedVectorSetValue(Y, 0);
  // BasisApply for H(div): CEED_EVAL_DIV, NOTRANSPOSE case
  CeedBasisApply(b, 1, CEED_NOTRANSPOSE, CEED_EVAL_DIV, X, Y);

  CeedVectorGetArrayRead(Y, CEED_MEM_HOST, &y);
  for (CeedInt i = 0; i < num_qpts; i++) {
    if (fabs(P * 0.25 - y[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", 2.0, y[i]);
  }
  CeedVectorRestoreArrayRead(Y, &y);

  CeedVectorSetValue(Y, 1.0);
  CeedVectorSetValue(X, 0.0);
  // BasisApply for H(div): CEED_EVAL_DIV, TRANSPOSE case
  CeedBasisApply(b, 1, CEED_TRANSPOSE, CEED_EVAL_DIV, Y, X);

  CeedVectorGetArrayRead(X, CEED_MEM_HOST, &x);
  for (CeedInt i = 0; i < P; i++) {
    if (fabs(num_qpts * 0.25 - x[i]) > 100. * CEED_EPSILON) printf("%f != %f\n", 2.0, x[i]);
  }
  CeedVectorRestoreArrayRead(X, &x);

  CeedBasisDestroy(&b);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&Y);
  CeedDestroy(&ceed);
  return 0;
}
