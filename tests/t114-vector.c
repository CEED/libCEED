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
  CeedScalar        a[len];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);
  len = argc > 2 ? atoi(argv[2]) : len;

  CeedVectorCreate(ceed, len, &x);
  for (CeedInt i = 0; i < len; i++) a[i] = len + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, a);

  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  // Try to set vector again (should fail)
  for (CeedInt i = 0; i < len; i++) a[i] = 2 * len + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, a);

  // LCOV_EXCL_START
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
