/// @file
/// Test CeedVector restore before get
/// \test Test CeedVector restore before get

//TESTARGS(only="cpu") {ceed_resource}
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed        ceed;
  CeedVector  x;
  CeedInt     len = 10;
  CeedScalar *a;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, len, &x);
  CeedVectorSetValue(x, 0.0);

  // Should error because no GetArray was not called
  CeedVectorRestoreArray(x, &a);

  // LCOV_EXCL_START
  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
