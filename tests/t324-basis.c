/// @file
/// Test grad transpose with a 2D Simplex non-tensor H1 basis
/// \test Test grad transposewith a 2D Simplex non-tensor H1 basis
#include <ceed.h>
#include <math.h>
#include "t320-basis.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector In, Out;
  const CeedInt P = 6, Q = 4, dim = 2;
  CeedBasis b;
  CeedScalar qref[dim*Q], qweight[Q];
  CeedScalar interp[P*Q], grad[dim*P*Q];
  const CeedScalar *out;
  CeedScalar colsum[P];

  buildmats(qref, qweight, interp, grad);

  CeedInit(argv[1], &ceed);

  for (int i=0; i<P; i++) {
    colsum[i] = 0;
    for (int j=0; j<Q*dim; j++) {
      colsum[i] += grad[i+j*P];
    }
  }

  CeedBasisCreateH1(ceed, CEED_TRIANGLE, 1, P, Q, interp, grad, qref,
                    qweight, &b);

  CeedVectorCreate(ceed, Q*dim, &In);
  CeedVectorSetValue(In, 1);
  CeedVectorCreate(ceed, P, &Out);
  CeedVectorSetValue(Out, 0);

  CeedBasisApply(b, 1, CEED_TRANSPOSE, CEED_EVAL_GRAD, In, Out);

  // Check values at quadrature points
  CeedVectorGetArrayRead(Out, CEED_MEM_HOST, &out);
  for (int i=0; i<P; i++)
    if (fabs(colsum[i] - out[i]) > 1E-14)
      // LCOV_EXCL_START
      printf("[%d] %f != %f\n", i, out[i], colsum[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(Out, &out);

  CeedVectorDestroy(&In);
  CeedVectorDestroy(&Out);
  CeedBasisDestroy(&b);
  CeedDestroy(&ceed);
  return 0;
}
