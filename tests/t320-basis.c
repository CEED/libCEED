/// @file
/// Test creation and destruction of a 2D Simplex non-tensor H^1 basis
/// \test Test creation and destruction of a 2D Simplex non-tensor H^1 basis
#include "t320-basis.h"

#include <ceed.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  const CeedInt p = 6, q = 4, dim = 2;
  CeedBasis     basis;
  CeedScalar    q_ref[dim * q], q_weight[q];
  CeedScalar    interp[p * q], grad[dim * p * q];

  CeedInit(argv[1], &ceed);

  // Test skipped if using single precision
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test not implemented in single precision");

  Build2DSimplex(q_ref, q_weight, interp, grad);
  CeedBasisCreateH1(ceed, CEED_TOPOLOGY_TRIANGLE, 1, p, q, interp, grad, q_ref, q_weight, &basis);
  CeedBasisView(basis, stdout);

  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
