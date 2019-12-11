/// @file
/// Test CeedVectorGetArray state counter
/// \test Test CeedVectorGetArray state counter
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  CeedScalar *a;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);

  // Write access followed by sync array should generate an error
  CeedVectorGetArray(x, CEED_MEM_HOST, &a);
  CeedVectorSyncArray(x, CEED_MEM_HOST);

  // LCOV_EXCL_START
  CeedVectorRestoreArray(x, &a);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
