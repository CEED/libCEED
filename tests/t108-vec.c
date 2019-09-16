/// @file
/// Test CeedVectorGetArray to modify array
/// \test Test CeedVectorGetArray to modify array
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  const CeedInt n = 10;
  CeedScalar a[n];
  CeedScalar *b;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i=0; i<n; i++)
    a[i] = 0;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  CeedVectorGetArray(x, CEED_MEM_HOST, &b);
  b[3] = -3.14;
  CeedVectorRestoreArray(x, &b);

  if (a[3] != -3.14)
    // LCOV_EXCL_START
    printf("Error writing array a[3] = %f", (double)b[3]);
  // LCOV_EXCL_STOP

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
