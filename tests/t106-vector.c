/// @file
/// Test syncing device data to host pointer
/// \test Test syncing device data to host pointer
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedVector y;
  CeedInt n;
  CeedScalar a[10], b[10];
  const CeedScalar *c;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  CeedVectorCreate(ceed, n, &y);
  for (CeedInt i=0; i<n; i++)
    a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);

  for (CeedInt i=0; i<n; i++)
    b[i] = 0;
  CeedVectorSetArray(y, CEED_MEM_HOST, CEED_USE_POINTER, b);

  CeedVectorGetArrayRead(x, CEED_MEM_DEVICE, &c);
  CeedVectorSetArray(y, CEED_MEM_DEVICE, CEED_COPY_VALUES, (CeedScalar *)c);
  CeedVectorRestoreArrayRead(x, &c);

  CeedVectorSyncArray(y, CEED_MEM_HOST);
  for (CeedInt i=0; i<n; i++)
    if (b[i] != 10+i)
      // LCOV_EXCL_START
      printf("Error reading array b[%d] = %f", i, (double)b[i]);
  // LCOV_EXCL_STOP

  CeedVectorDestroy(&x);
  CeedVectorDestroy(&y);
  CeedDestroy(&ceed);
  return 0;
}
