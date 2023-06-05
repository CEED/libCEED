/// @file
/// Test that length of BasisApply input/output vectors is incompatible with basis dimensions
/// \test Test that topological and geometric dimensions of basis match
#include <ceed.h>
#include <math.h>

int main(int argc, char **argv) {
  Ceed       ceed;
  CeedBasis  basis;
  CeedVector u, v;
  CeedInt    q = 8, p = 2, num_comp = 1, dim = 3, len = pow((CeedScalar)(q), dim);

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &u);
  CeedVectorCreate(ceed, len + 1, &v);

  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, p, q, CEED_GAUSS, &basis);

  // Basis apply will error because dimensions don't agree
  CeedBasisApply(basis, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);

  // LCOV_EXCL_START
  CeedBasisDestroy(&basis);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
