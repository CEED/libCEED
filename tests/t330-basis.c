/// @file
/// Test creation and destruction of a 2D Quad non-tensor Hdiv basis
/// \test Test creation and destruction of a 2D Quad non-tensor Hdiv basis
#include "t330-basis.h"

#include <ceed.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  const CeedInt q = 3, dim = 2, num_qpts = q * q, elem_nodes = 4;
  CeedInt       num_comp = 1;
  CeedInt       p        = dim * elem_nodes;  // DoF per element, DoF are vector in H(div)
  CeedBasis     basis;
  CeedScalar    q_ref[dim * num_qpts], q_weights[num_qpts];
  CeedScalar    interp[dim * p * num_qpts], div[p * num_qpts];

  CeedInit(argv[1], &ceed);

  // Test skipped if using single precision
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32)
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test not implemented in single precision");
  // LCOV_EXCL_STOP

  BuildHdivQuadrilateral(q, q_ref, q_weights, interp, div, CEED_GAUSS);
  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, num_comp, p, num_qpts, interp, div, q_ref, q_weights, &basis);
  // interp[0]--.interp[num_qpts-1] ==> basis in x-direction
  // interp[num_qpts]--.interp[dim*num_qpts-1] ==> basis in y-direction
  CeedBasisView(basis, stdout);

  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
