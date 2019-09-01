/// @file
/// Test CeedVector restore before get
/// \test Test CeedVector restore before get
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n = 10;
  CeedScalar *a;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, n, &x);

  // Should error because no GetArray was not called
  CeedVectorRestoreArray(x, &a);

  // LCOV_EXCL_START
  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
