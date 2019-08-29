/// @file
/// Test CeedVectorDestroy state counter
/// \test Test CeedVectorDestroy state counter
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  CeedScalar *a;

  CeedInit(argv[1], &ceed);
  n = 10;
  CeedVectorCreate(ceed, n, &x);
  CeedVectorGetArray(x, CEED_MEM_HOST, &a);

  // Write access not restored should generate an error
  CeedVectorDestroy(&x);
  // LCOV_EXCL_START
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
