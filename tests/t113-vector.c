/// @file
/// Test CeedVector readers counter
/// \test Test CeedVector readers counter

//TESTARGS(name="length 10") {ceed_resource} 10
//TESTARGS(name="length 0") {ceed_resource} 0
#include <ceed.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        x;
  CeedInt           len = 10;
  const CeedScalar *a;
  CeedScalar       *b;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedVectorCreate(ceed, len, &x);
  CeedVectorSetValue(x, 0.0);
  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &a);

  // Write access with read access generate an error
  CeedVectorGetArray(x, CEED_MEM_HOST, &b);

  // LCOV_EXCL_START
  CeedVectorRestoreArrayRead(x, &a);
  CeedVectorRestoreArray(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
