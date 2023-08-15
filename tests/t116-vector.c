/// @file
/// Test CeedVectorDestroy state counter
/// \test Test CeedVectorDestroy state counter

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  CeedVector  x;
  CeedInt     len = 10;
  CeedScalar *a;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedVectorCreate(ceed, len, &x);
  CeedVectorSetValue(x, 0.0);
  CeedVectorGetArray(x, CEED_MEM_HOST, &a);

  // Write access not restored should generate an error
  CeedVectorDestroy(&x);

  // LCOV_EXCL_START
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
