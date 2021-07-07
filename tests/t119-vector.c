/// @file
/// Test taking the reciprocal of a vector
/// \test Test taking the reciprocal of a vector
#include <math.h>
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;
  CeedVector x;
  CeedInt n;
  CeedScalar a[10];
  const CeedScalar *b;

  CeedInit(argv[1], &ceed);

  n = 10;
  CeedVectorCreate(ceed, n, &x);
  for (CeedInt i=0; i<n; i++)
    a[i] = 10 + i;
  CeedVectorSetArray(x, CEED_MEM_HOST, CEED_USE_POINTER, a);
  {
    // Sync memtype to device for GPU backends
    CeedMemType type = CEED_MEM_HOST;
    CeedGetPreferredMemType(ceed, &type);
    CeedVectorSyncArray(x, type);
  }
  CeedVectorReciprocal(x);

  CeedVectorGetArrayRead(x, CEED_MEM_HOST, &b);
  for (CeedInt i=0; i<n; i++)
    if (fabs(b[i] - 1./(10+i)) > 1e-15)
      // LCOV_EXCL_START
      printf("Error reading array b[%d] = %f",i,(double)b[i]);
  // LCOV_EXCL_STOP
  CeedVectorRestoreArrayRead(x, &b);

  CeedVectorDestroy(&x);
  CeedDestroy(&ceed);
  return 0;
}
