/// @file
/// Test CeedVector readers counter
/// \test Test CeedVector readers counter
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n = 10;
  CeedScalar a[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i=0; i<n; i++)
    a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, a);

  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++)
    if (b[i] != 10+i)
      // LCOV_EXCL_START
      printf("Error reading array b[%d] = %f",i,(double)b[i]);
  // LCOV_EXCL_STOP

  // Try to set vector again (should fail)
  for (CeedInt i=0; i<n; i++)
    a[i] = 20 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  // LCOV_EXCL_START
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
  // LCOV_EXCL_STOP
}
