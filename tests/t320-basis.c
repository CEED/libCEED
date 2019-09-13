/// @file
/// Test creation and destruction of a 2D Simplex non-tensor H1 basis
/// \test Test creation and distruction of a 2D Simplex non-tensor H1 basis
#include <ceed.h>
#include "t320-basis.h"

int main(int argc, char **argv) {
  Ceed ceed;
  const CeedInt P = 6, Q = 4, dim = 2;
  CeedBasis b;
  CeedScalar qref[dim*Q], qweight[Q];
  CeedScalar interp[P*Q], grad[dim*P*Q];

  buildmats(qref, qweight, interp, grad);

  CeedInit(argv[1], &ceed);
  CeedBasisCreateH1(ceed, CEED_TRIANGLE, 1, P, Q, interp, grad, qref,
                    qweight, &b);
  CeedBasisView(b, stdout);

  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
