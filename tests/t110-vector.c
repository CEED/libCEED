/// @file
/// Test CeedVectorGetArray state counter
/// \test Test CeedVectorGetArray state counter

//TESTARGS(only="cpu") {ceed_resource}
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  CeedVector  x;
  CeedInt     len = 10;
  CeedScalar *a, *b;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorSetValue(x, 0.0);

  // Two write accesses should generate an error
  CeedVectorGetArray(x, CEED_MEM_HOST, &a);
  CeedVectorGetArray(x, CEED_MEM_HOST, &b);

  // LCOV_EXCL_START
  CeedVectorRestoreArray(x, &a);
  CeedVectorRestoreArray(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
