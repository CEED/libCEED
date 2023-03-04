/// @file
/// Test creation and destruction of a 2D Quad non-tensor H(div) basis
/// \test Test creation and destruction of a 2D Quad non-tensor H(div) basis
#include <ceed.h>
#include <stdio.h>

#include "t330-basis.h"

int main(int argc, char **argv) {
  Ceed          ceed;
  const CeedInt p = 8, q = 3, dim = 2, num_qpts = q * q;
  CeedBasis     basis;
  CeedScalar    q_ref[dim * num_qpts], q_weights[num_qpts];
  CeedScalar    interp[dim * p * num_qpts], div[p * num_qpts];

  CeedInit(argv[1], &ceed);

  // Test skipped if using single precision
  if (CEED_SCALAR_TYPE == CEED_SCALAR_FP32) return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Test not implemented in single precision");

  BuildHdivQuadrilateral(q, q_ref, q_weights, interp, div, CEED_GAUSS);
  CeedBasisCreateHdiv(ceed, CEED_TOPOLOGY_QUAD, 1, p, num_qpts, interp, div, q_ref, q_weights, &basis);
  CeedBasisView(basis, stdout);

  CeedBasisDestroy(&basis);
  CeedDestroy(&ceed);
  return 0;
}
