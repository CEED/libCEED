/// @file
/// Test that length of BasisApply input/output vectors is incompatible with basis dimensions
/// \test Test that topological and geometric dimensions of basis match
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  b;
  CeedVector U, V;
  CeedInt    Q = 8, P = 2, num_comp = 1, dim = 3, len = pow((CeedScalar)(Q), dim);

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &U);
  CeedVectorCreate(ceed, len + 1, &V);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, P, Q, CEED_GAUSS, &b);

  // Basis apply will error because dimensions don't agree
  CeedBasisApply(b, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, U, V);

  // LCOV_EXCL_START
  CeedBasisDestroy(&b);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
