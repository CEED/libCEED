/// @file
/// Test CeedVector readers counter
/// \test Test CeedVector readers counter

//TESTARGS(only="cpu") {ceed_resource}
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed              ceed;
  CeedVector        x;
  CeedInt           len = 10;
  const CeedScalar *a;
  CeedScalar       *b;

  CeedInit(argv[1], &ceed);

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
