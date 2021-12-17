/// @file
/// Test CeedVectorGetArrayWrite to modify array
/// \test Test CeedVectorGetArrayWrite to modify array
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  const CeedInt n = 10;
  CeedScalar *a;

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, n, &x);

  CeedVectorGetArrayWrite(x, CEED_MEM_HOST, &a);
  for (CeedInt i = 0; i < n; i++)
    a[i] = 3*i;
  CeedVectorRestoreArray(x, &a);

  CeedVectorGetArrayRead(x, CEED_MEM_HOST, (const CeedScalar **)&a);
  for (CeedInt i = 0; i < n; i++)
    if (a[i] != (CeedScalar)(3*i))
      // LCOV_EXCL_START
      printf("Error writing array a[%d] = %f", i, a[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(x, (const CeedScalar **)&a);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
