/// @file
/// Test creation and destruction of a 2D Quad non-tensor Hdiv basis
/// \test Test creation and distruction of a 2D Quad non-tensor Hdiv basis
#include "t330-basis.h"

#include <ceed.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  const CeedInt Q = 3, dim = 2, num_qpts = Q * Q, elem_nodes = 4;
  CeedInt       num_comp = 1;
  CeedInt       P        = dim * elem_nodes;  // dof per element! dof is vector in H(div)
  CeedBasis     b;
  CeedScalar    q_ref[dim * num_qpts], q_weights[num_qpts];
  CeedScalar    interp[dim * P * num_qpts], div[P * num_qpts];

  CeedInit(argv[1], &ceed);

  // Test skipped if using single precision
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test not implemented in single precision");
  // LCOV_EXCL_STOP

  HdivBasisQuad(Q, q_ref, q_weights, interp, div, CEED_GAUSS);
  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, num_comp, P, num_qpts, interp, div, q_ref, q_weights, &b);
  // interp[0]--.interp[num_qpts-1] ==> basis in x-direction
  // interp[num_qpts]--.interp[dim*num_qpts-1] ==> basis in y-direction
  CeedBasisView(b, stdout);

  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
