/// @file
/// Test CeedVectorGetArray state counter
/// \test Test CeedVectorGetArray state counter

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  CeedVector  x;
  CeedInt     len = 10;
  CeedScalar *a, b[len];

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedVectorCreate(ceed, len, &x);
  CeedVectorSetValue(x, 0.0);

  // Two write accesses should generate an error
  CeedVectorGetArray(x, CEED_MEM_HOST, &a);
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, b);

  // LCOV_EXCL_START
  CeedVectorRestoreArray(x, &a);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
