/// @file
/// Test creation and destruction of a 2D Simplex non-tensor H1 basis
/// \test Test creation and distruction of a 2D Simplex non-tensor H1 basis
#include "t320-basis.h"

#include <ceed.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  const CeedInt P = 6, Q = 4, dim = 2;
  CeedBasis     b;
  CeedScalar    q_ref[dim * Q], q_weight[Q];
  CeedScalar    interp[P * Q], grad[dim * P * Q];

  CeedInit(argv[1], &ceed);

  // Test skipped if using single precision
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test not implemented in single precision");

  buildmats(q_ref, q_weight, interp, grad);

  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, P, Q, interp, grad, q_ref, q_weight, &b);
  CeedBasisView(b, stdout);

  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
